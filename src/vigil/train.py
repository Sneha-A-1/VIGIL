"""Training of the VIGIL network and of the neural baselines.

    python -m vigil.train --model vigil --seed 0                 # configs/train.yaml
    python -m vigil.train --model vigil --variant no_graph --seed 0
    python -m vigil.train --model gru --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from vigil import engine
from vigil.config import load_config, merge, repo_path
from vigil.data.dataset import load
from vigil.models import baselines
from vigil.models.vigil_net import VIGILNet

BASELINES = list(baselines.MODELS)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def window_starts(lo: int, hi: int, n: int, gen: torch.Generator, wet_pool: torch.Tensor | None = None,
                  wet_frac: float = 0.0) -> torch.Tensor:
    """Random window starts; a fraction ``wet_frac`` is drawn from ``wet_pool`` (windows with rain)."""
    t0 = torch.randint(lo, hi, (n,), generator=gen)
    if wet_pool is not None and wet_frac > 0:
        m = torch.rand(n, generator=gen) < wet_frac
        t0[m] = wet_pool[torch.randint(len(wet_pool), (int(m.sum()),), generator=gen)]
    return t0


def wet_window_pool(data, lo: int, hi: int, window: int, min_mm: float) -> torch.Tensor:
    """Window starts whose window contains at least ``min_mm`` of rain within some 24 h."""
    import pandas as pd
    p24 = pd.Series(data.prcp_mm.numpy()).rolling(24, min_periods=1).sum().to_numpy()
    mx = pd.Series(p24).rolling(window, min_periods=1).max().shift(-(window - 1)).to_numpy()
    idx = np.arange(lo, hi)
    return torch.as_tensor(idx[np.nan_to_num(mx[idx]) >= min_mm])


def val_starts(data, cfg) -> torch.Tensor:
    a, b = data.split["val"]
    n = cfg["val_windows"]
    lo, hi = a + cfg["warm"] + 25, b - cfg["window"] - 1
    return torch.linspace(lo, hi, n).long()


def make_spec(cfg, B, gen, closed_loop: bool) -> engine.PolicySpec:
    mix = dict(cfg["policy_mix"])
    if not closed_loop:
        for k in [k for k in mix if k.startswith("voi")]:
            mix.pop(k)
    names, probs = list(mix), np.array(list(mix.values()), float)
    probs = probs / probs.sum()
    idx = torch.multinomial(torch.as_tensor(probs), B, replacement=True, generator=gen).numpy()
    budgets = torch.as_tensor(cfg["budgets"])
    k = budgets[torch.randint(len(budgets), (B,), generator=gen)]
    eta = torch.rand(B, generator=gen)
    kappa = torch.rand(B, generator=gen)
    tb = torch.rand(B, 42, generator=gen) * 1e-3
    return engine.PolicySpec(names=[names[i] for i in idx], k=k, eta=eta, kappa=kappa, max_age=cfg["max_age"],
                             tiebreak=tb, gen=gen)


def val_spec(cfg, B) -> engine.PolicySpec:
    budgets = cfg["budgets"]
    k = torch.as_tensor([budgets[i % len(budgets)] for i in range(B)])
    tb = torch.arange(42).float()[None].repeat(B, 1) * 1e-3
    return engine.PolicySpec(names=["rr"] * B, k=k, eta=torch.full((B,), 0.5), max_age=cfg["max_age"], tiebreak=tb)


def build_vigil(data, mcfg) -> VIGILNet:
    return VIGILNet(data.N, engine.ctx_dim(data), engine.MEM_DIM, data.D, hidden=mcfg["hidden"],
                    node_emb=mcfg["node_emb"], adj=data.adj, use_graph=mcfg["use_graph"],
                    use_adaptive=mcfg["use_adaptive"], predict_update=mcfg["predict_update"])


def train_vigil(cfg: dict, seed: int, out: Path, variant: str) -> dict:
    tcfg, mcfg = cfg["train"], cfg["vigil"]
    set_seed(seed)
    data = load(tcfg.get("weather", "power"))
    model = build_vigil(data, mcfg)
    model.set_input_stats(*engine.input_stats(data))
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, tcfg["iters"]) / tcfg["iters"])))
    gen = torch.Generator().manual_seed(seed)
    lo, hi = tcfg["warm"] + 26, data.split["train"][1] - tcfg["window"] - 1
    pool = wet_window_pool(data, lo, hi, tcfg["window"], tcfg.get("wet_min_mm", 10.0))
    vs = val_starts(data, tcfg)
    vspec = val_spec(tcfg, len(vs))
    best, best_state, log = float("inf"), None, []
    t_begin = time.time()
    for it in range(1, tcfg["iters"] + 1):
        model.train()
        closed = mcfg["closed_loop"] and it > tcfg["closed_loop_start"]
        spec = make_spec(tcfg, tcfg["batch"], gen, closed)
        t0 = window_starts(lo, hi, tcfg["batch"], gen, pool, tcfg.get("wet_frac", 0.0))
        loss, st = engine.rollout(model, data, t0, tcfg["window"], tcfg["warm"], spec, train=True,
                                  beta=mcfg["beta"], w_event=mcfg["w_event"], hetero=mcfg["hetero"],
                                  w_look=mcfg.get("w_look", 1.0))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip"])
        opt.step()
        sched.step()
        entry = {"it": it, "loss": float(loss.detach()), **st}
        if it % tcfg["eval_every"] == 0 or it == tcfg["iters"]:
            model.eval()
            with torch.no_grad():
                vloss, vst = engine.rollout(model, data, vs, tcfg["window"], tcfg["warm"], vspec, train=True,
                                            grad=False, beta=mcfg["beta"], w_event=mcfg["w_event"],
                                            hetero=mcfg["hetero"], w_look=mcfg.get("w_look", 1.0))
            sel = engine.selection_score(vst, mcfg)
            entry["val_loss"] = float(vloss)
            entry["val_select"] = sel
            entry.update({f"val_{k}": v for k, v in vst.items()})
            if sel < best and it >= tcfg.get("select_after", 0):
                best = sel
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in entry.items()}), flush=True)
        log.append(entry)
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    info = {"model": "vigil", "variant": variant, "seed": seed, "best_val": best,
            "train_seconds": time.time() - t_begin, "params": sum(p.numel() for p in model.parameters()),
            "config": cfg}
    torch.save({"state": best_state, "info": info}, out / "model.pt")
    with open(out / "train_log.json", "w") as fh:
        json.dump({"info": info, "log": log}, fh, default=str)
    return info


def baseline_loss(model, batch, warm):
    mu, _ = baselines.run_sequence(model, batch["inp"], batch["last_x"])
    x = batch["x"][:, warm:]
    target = torch.where(batch["obs"][:, warm:] > 0, torch.full_like(x, float("nan")), x)
    l, n = engine.mse_loss(mu[:, warm:], target)
    return l / max(int(n), 1)


def transformer_batch(batch, W, gen):
    """Cut random W-hour sub-windows for the windowed Transformer."""
    S = batch["inp"].shape[1]
    starts = torch.randint(0, S - W + 1, (batch["inp"].shape[0],), generator=gen)
    sl = lambda v: torch.stack([v[b, s:s + W] for b, s in enumerate(starts.tolist())])
    return {k: sl(v) for k, v in batch.items()}


def train_baseline(cfg: dict, name: str, seed: int, out: Path) -> dict:
    tcfg, bcfg = cfg["train"], cfg["baselines"][name]
    set_seed(seed)
    data = load(tcfg.get("weather", "power"))
    d_in = engine.ctx_dim(data) + engine.MEM_DIM
    model = baselines.build(name, d_in, data.D, data.adj, data.N, **bcfg)
    model.set_input_stats(*engine.input_stats(data))
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, tcfg["iters"]) / tcfg["iters"])))
    gen = torch.Generator().manual_seed(seed)
    lo, hi = tcfg["warm"] + 26, data.split["train"][1] - tcfg["window"] - 1
    pool = wet_window_pool(data, lo, hi, tcfg["window"], tcfg.get("wet_min_mm", 10.0))
    vs = val_starts(data, tcfg)
    with torch.no_grad():
        vbatch = engine.simulate_inputs(data, vs, tcfg["window"], tcfg["warm"], val_spec(tcfg, len(vs)))
    best, best_state, log = float("inf"), None, []
    t_begin = time.time()
    warm = tcfg["warm"]
    for it in range(1, tcfg["iters"] + 1):
        model.train()
        spec = make_spec(tcfg, tcfg["batch"], gen, closed_loop=False)
        t0 = window_starts(lo, hi, tcfg["batch"], gen, pool, tcfg.get("wet_frac", 0.0))
        with torch.no_grad():
            batch = engine.simulate_inputs(data, t0, tcfg["window"], warm, spec)
        if model.kind == "window":
            sub = transformer_batch(batch, model.window, gen)
            mu = model(sub["inp"], sub["last_x"])
            x = sub["x"][:, model.window // 2:]
            target = torch.where(sub["obs"][:, model.window // 2:] > 0, torch.full_like(x, float("nan")), x)
            l, n = engine.mse_loss(mu[:, model.window // 2:], target)
            loss = l / max(int(n), 1)
        else:
            loss = baseline_loss(model, batch, warm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip"])
        opt.step()
        sched.step()
        entry = {"it": it, "loss": float(loss.detach())}
        if it % tcfg["eval_every"] == 0 or it == tcfg["iters"]:
            model.eval()
            with torch.no_grad():
                if model.kind == "window":
                    vl = []
                    for s0 in range(warm, vbatch["inp"].shape[1], 24):
                        a = s0 - model.window + 1
                        sub = {k: v[:, a:s0 + 1] for k, v in vbatch.items()}
                        mu = model(sub["inp"], sub["last_x"])[:, -1]
                        x = sub["x"][:, -1]
                        tgt = torch.where(sub["obs"][:, -1] > 0, torch.full_like(x, float("nan")), x)
                        l, n = engine.mse_loss(mu, tgt)
                        vl.append(float(l) / max(int(n), 1))
                    vloss = float(np.mean(vl))
                else:
                    vloss = float(baseline_loss(model, vbatch, warm))
            entry["val_loss"] = vloss
            if vloss < best and it >= tcfg.get("select_after", 0):
                best = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in entry.items()}), flush=True)
        log.append(entry)
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    info = {"model": name, "variant": "default", "seed": seed, "best_val": best,
            "train_seconds": time.time() - t_begin, "params": sum(p.numel() for p in model.parameters()),
            "config": cfg}
    torch.save({"state": best_state, "info": info}, out / "model.pt")
    with open(out / "train_log.json", "w") as fh:
        json.dump({"info": info, "log": log}, fh, default=str)
    return info


def train_lgbm(cfg: dict, seed: int, out: Path, n_chunks: int = 8, per_chunk: int = 16) -> dict:
    """LightGBM baseline trained on the same simulated polling trajectories as the neural baselines."""
    import lightgbm as lgb
    from vigil.models import gbdt
    tcfg = cfg["train"]
    set_seed(seed)
    data = load(tcfg.get("weather", "power"))
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    lo, hi = tcfg["warm"] + 26, data.split["train"][1] - tcfg["window"] - 1
    pool = wet_window_pool(data, lo, hi, tcfg["window"], tcfg.get("wet_min_mm", 10.0))
    warm = tcfg["warm"]
    cap = 1_500_000 // n_chunks

    def chunk_rows(batch, cap_rows):
        X = gbdt.rows(batch["inp"][:, warm:])
        y = (batch["x"][:, warm:] - batch["last_x"][:, warm:]).reshape(-1).numpy()
        keep = np.isfinite(y) & (batch["obs"][:, warm:].reshape(-1).numpy() == 0)
        X, y = X[keep], y[keep]
        if len(y) > cap_rows:
            sel = rng.choice(len(y), cap_rows, replace=False)
            X, y = X[sel], y[sel]
        return X, y

    tic = time.time()
    Xs, ys = [], []
    for c in range(n_chunks):
        spec = make_spec(tcfg, per_chunk, gen, closed_loop=False)
        with torch.no_grad():
            batch = engine.simulate_inputs(data, window_starts(lo, hi, per_chunk, gen, pool, tcfg.get("wet_frac", 0.0)),
                                           tcfg["window"], warm, spec)
        X, y = chunk_rows(batch, cap)
        Xs.append(X)
        ys.append(y)
    vs = val_starts(data, tcfg)
    with torch.no_grad():
        vb = engine.simulate_inputs(data, vs, tcfg["window"], warm, val_spec(tcfg, len(vs)))
    Xv, yv = chunk_rows(vb, 300_000)
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "seed": seed, "num_threads": 1, "verbose": -1}
    model = lgb.train(params, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    model.save_model(str(out / "model.txt"), num_iteration=model.best_iteration)
    info = {"model": "lgbm", "variant": "default", "seed": seed, "best_iter": int(model.best_iteration),
            "best_val": float(model.best_score["valid_0"]["l2"]), "n_train_rows": int(sum(len(y) for y in ys)),
            "train_seconds": time.time() - tic, "params": params}
    with open(out / "train_log.json", "w") as fh:
        json.dump({"info": info}, fh)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vigil", choices=["vigil", "lgbm"] + BASELINES)
    ap.add_argument("--variant", default="default", help="key of configs/train.yaml:variants")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="train")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iters", type=int, default=None, help="override train.iters (smoke tests)")
    ap.add_argument("--out", default="results/models")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.wet_frac=0.5 vigil.beta=1.0")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config(a.config)
    if a.variant != "default":
        cfg = merge(cfg, cfg["variants"][a.variant])
    for kv in a.set:
        k, v = kv.split("=", 1)
        sec, key = k.split(".", 1)
        cfg[sec][key] = yaml.safe_load(v)
    if a.iters:
        cfg["train"]["iters"] = a.iters
        cfg["train"]["eval_every"] = max(1, min(cfg["train"]["eval_every"], a.iters // 2))
    out = repo_path(a.out, f"{a.model}-{a.variant}", f"seed{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    if a.model == "vigil":
        info = train_vigil(cfg, a.seed, out, a.variant)
    elif a.model == "lgbm":
        info = train_lgbm(cfg, a.seed, out)
    else:
        info = train_baseline(cfg, a.model, a.seed, out)
    print("done", json.dumps({k: v for k, v in info.items() if k != "config"}))


if __name__ == "__main__":
    main()
