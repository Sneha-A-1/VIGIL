"""Measure per-step inference latency of every estimator and the compute cost of the analytics tasks
used in the edge-cloud case study (single CPU thread).

    python scripts/measure_costs.py   ->  results/raw/costs.json
"""
from __future__ import annotations

import json
import platform
import time

import numpy as np
import torch

from vigil import engine
from vigil.config import repo_path
from vigil.data.dataset import load
from vigil.evaluate import Row, fit_kf, load_estimator, _spec
from vigil.models import baselines, gbdt
from vigil.sim import retrieval as R


def timeit(fn, n=200, warm=10):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def vigil_step_cost(model, data, t0):
    """One hour at the edge: prior, VoI ranking, poll update, posterior (one farm, K=2)."""
    spec = _spec([Row("vigil", "voi", 2, kappa=0.25)], 0)
    spec.delta = torch.as_tensor(data.delta_norm)
    m = R.init_memory(data.x[t0 - 1][None], data.tmp[t0 - 1][None])
    h = model.init_state(1, data.N)
    sup = model.supports()
    dn = torch.as_tensor(data.delta_norm)
    hist = m.last_x[..., None].repeat(1, 1, 1, engine.RISE_WINDOW)
    state = {"t": t0, "h": h}

    def step():
        t = torch.tensor([state["t"]])
        with torch.no_grad():
            ctx = engine.context(data, t)
            R.advance(m, data.prcp_mm[t], data.vpd_kpa[t])
            r = engine.rise(m.last_x, hist, dn)
            h_m, prior = model.prior(state["h"], ctx, R.memory_features(m), m.last_x, r, sup)
            fresh = state.get("fresh", torch.zeros_like(m.last_x, dtype=torch.bool))
            m.floor_l = torch.where(fresh, prior[4] ** 2, m.floor_l)
            poll = engine.choose(spec, m, prior, data.x[t])
            obs = R.observe(m, poll, data.x[t], data.tmp[t])
            state["fresh"] = obs > 0
            state["h"], post = model.update(h_m, prior, obs, m.last_x * obs, m.last_tmp * obs, m.last_x,
                                            engine.rise(m.last_x, hist, dn), sup)
        state["t"] += 1
    return timeit(step, n=300)


def open_loop_cost(model, data, t0, hours):
    """Probabilistic outlook: run the prior for ``hours`` steps without observations."""
    def run():
        m = R.init_memory(data.x[t0 - 1][None], data.tmp[t0 - 1][None])
        h = model.init_state(1, data.N)
        sup = model.supports()
        dn = torch.as_tensor(data.delta_norm)
        hist = m.last_x[..., None].repeat(1, 1, 1, engine.RISE_WINDOW)
        with torch.no_grad():
            for s in range(hours):
                t = torch.tensor([t0 + s])
                R.advance(m, data.prcp_mm[t], data.vpd_kpa[t])
                h, prior = model.prior(h, engine.context(data, t), R.memory_features(m), m.last_x,
                                       engine.rise(m.last_x, hist, dn), sup)
    return timeit(run, n=10, warm=2)


def idw_map_cost(data):
    """Hourly 10 m maps (58 x 101 cells x 5 depths) by inverse-distance weighting of node estimates."""
    xs = np.linspace(data.xy[:, 0].min() - 100, data.xy[:, 0].max() + 100, 101)
    ys = np.linspace(data.xy[:, 1].min() - 100, data.xy[:, 1].max() + 100, 58)
    gx, gy = np.meshgrid(xs, ys)
    d = np.sqrt((gx.reshape(-1, 1) - data.xy[:, 0]) ** 2 + (gy.reshape(-1, 1) - data.xy[:, 1]) ** 2) + 1.0
    vals = np.random.default_rng(0).random((data.N, data.D))

    def run():
        w = 1.0 / d ** 2
        w = w / w.sum(1, keepdims=True)
        return w @ vals
    return timeit(run, n=50)


def retrain_cost(data, iters=25):
    from vigil.config import load_config
    from vigil.train import build_vigil, make_spec
    cfg = load_config("train")
    model = build_vigil(data, cfg["vigil"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    t0 = torch.randint(500, data.split["train"][1] - 400, (4,), generator=gen)
    tic = time.perf_counter()
    for _ in range(iters):
        spec = make_spec(cfg["train"], 4, gen, True)
        loss, _ = engine.rollout(model, data, t0, 168, 24, spec, train=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.perf_counter() - tic


def estimator_step_costs(data, t0):
    """Per-hour inference latency of each estimator for one farm (B=1, K=2, round-robin)."""
    res = {}
    spec = _spec([Row("x", "rr", 2)], 0)
    with torch.no_grad():
        sim = engine.simulate_inputs(data, torch.tensor([t0]), 64, 0, spec)
    inp, last = sim["inp"], sim["last_x"]
    for name in ["mlp", "gru", "dcrnn", "transformer", "gwn"]:
        model = load_estimator(name, 0, data)
        nparams = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            if model.kind == "step":
                f = lambda: model(inp[:, -1:], last[:, -1:])
            elif model.kind == "recurrent":
                h = model.init_state(1, data.N)
                f = lambda: model.step(inp[:, -1], last[:, -1], h)
            elif model.kind == "window":
                f = lambda: model(inp[:, -model.window:], last[:, -model.window:])
            else:
                f = lambda: model(inp[:, -model.receptive:], last[:, -model.receptive:])
            res[name] = {"latency_ms": 1000 * timeit(f), "params": int(nparams)}
    lg = load_estimator("lgbm", 0, data)
    res["lgbm"] = {"latency_ms": 1000 * timeit(lambda: gbdt.predict(lg, inp[0, -1:], last[0, -1:]), n=100),
                   "params": int(lg.num_trees())}
    kf = fit_kf(data)
    mean, P = kf.init(data.x[t0].numpy())
    node_age = np.zeros(data.N)

    def kf_step(greedy=False):
        nonlocal mean, P
        mean, P = kf.predict(mean, P, False)
        sel = kf.greedy_select(P, 2, node_age, 168.0) if greedy else np.array([0, 1])
        idx = np.concatenate([kf.channels(i) for i in sel])
        mean, P = kf.update(mean, P, idx, mean[idx])
    res["kf"] = {"latency_ms": 1000 * timeit(lambda: kf_step(False), n=100), "params": int(kf.n * kf.n * 2)}
    res["kf_greedy"] = {"latency_ms": 1000 * timeit(lambda: kf_step(True), n=50), "params": int(kf.n * kf.n * 2)}
    return res


def scaling(data, model, Ns=(42, 84, 168, 336, 672, 1344), Bs=(1, 8, 32, 128), kf_max_n=336, K=2):
    """Per-step latency versus graph size N (synthetic kNN graphs, random inputs, untrained weights:
    the cost does not depend on the weight values) and fleet throughput versus the number of farms B
    processed in one batch by the trained 42-logger model. KF-greedy for comparison (N <= kf_max_n)."""
    from vigil.data.build import knn_gaussian_adjacency
    from vigil.models.vigil_net import VIGILNet
    from vigil.models.kalman import RainKF
    rng = np.random.default_rng(0)
    D, C, M = data.D, engine.ctx_dim(data), engine.MEM_DIM
    out = {"nodes": [], "fleet": []}

    def net_step(net, B, N):
        ctx, mem = torch.randn(B, N, C), torch.randn(B, N, M)
        last, rise = torch.randn(B, N, D), torch.zeros(B, N, D)
        obs = torch.zeros(B, N, D)
        obs[:, rng.choice(N, K, replace=False)] = 1.0
        h = net.init_state(B, N)
        sup = net.supports()          # fixed at inference: computed once
        def f():
            with torch.no_grad():
                h_m, pr = net.prior(h, ctx, mem, last, rise, sup)
                net.update(h_m, pr, obs, obs * last, obs * last, last, rise, sup)
        return f

    for N in Ns:
        xy = rng.random((N, 2)) * 400.0 * np.sqrt(N / 42)
        adj, _ = knn_gaussian_adjacency(xy, 6)
        net = VIGILNet(N, C, M, D, hidden=model.hidden, adj=torch.as_tensor(adj)).eval()
        r = {"N": N, "vigil_ms": 1000 * timeit(net_step(net, 1, N), n=50 if N <= 336 else 20)}
        if N <= kf_max_n:
            T = 1500
            x = np.cumsum(rng.normal(0, 0.05, (T, N, D)), 0)
            kf = RainKF(shrink=0.5).fit(x, np.zeros(T))
            mean, P = kf.init(x[-1])
            age = np.zeros(N)
            def kf_step():
                nonlocal mean, P
                mean, P = kf.predict(mean, P, False)
                sel = kf.greedy_select(P, K, age, 168.0)
                idx = np.concatenate([kf.channels(i) for i in sel])
                mean, P = kf.update(mean, P, idx, mean[idx])
            r["kfgreedy_ms"] = 1000 * timeit(kf_step, n=10 if N <= 168 else 3, warm=1)
        out["nodes"].append(r)
        print(json.dumps(r), flush=True)
    for B in Bs:
        f = net_step(model, B, data.N)
        t = timeit(f, n=30 if B <= 32 else 10)
        out["fleet"].append({"B": B, "step_ms": 1000 * t, "per_farm_ms": 1000 * t / B})
        print(json.dumps(out["fleet"][-1]), flush=True)
    return out


def main():
    torch.set_num_threads(1)
    data = load()
    t0 = data.split["test"][0] + 24 * 30
    model = load_estimator("vigil", 0, data)
    costs = {
        "platform": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
                     "torch": torch.__version__, "threads": 1},
        "vigil_params": int(sum(p.numel() for p in model.parameters())),
        "edge_step_ms": 1000 * vigil_step_cost(model, data, t0),
        "alert_outlook_24h_s": open_loop_cost(model, data, t0, 24),
        "forecast_72h_s": open_loop_cost(model, data, t0, 72),
        "map_idw_s": idw_map_cost(data),
        "retrain_25it_s": retrain_cost(data),
        "estimators": estimator_step_costs(data, t0),
        "scaling": scaling(data, model),
    }
    out = repo_path("results", "raw", "costs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(costs, open(out, "w"), indent=1)
    print(json.dumps(costs, indent=1))


if __name__ == "__main__":
    main()
