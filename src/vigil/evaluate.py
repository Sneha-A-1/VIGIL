"""Closed-loop evaluation of estimator x polling-policy pipelines over the validation or test period.

Every evaluation starts from a full poll 168 h before the period and runs the budgeted policy during
that burn-in, so that ages are in steady state when scoring starts. Event-alert thresholds are chosen
on the validation period (hourly F1) and applied unchanged to the test period.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from vigil import engine, metrics
from vigil.config import load_config, merge, repo_path
from vigil.data.dataset import CAFData, load
from vigil.models import baselines, gbdt
from vigil.models.kalman import RainKF
from vigil.models.vigil_net import VIGILNet
from vigil.sim import retrieval as R

WARMUP = 168
EVENT_DELTA = 0.02


@dataclass
class Row:
    estimator: str            # vigil[-variant] | locf | kf | lgbm | mlp | gru | dcrnn | transformer | gwn
    policy: str               # rr | random | ars | voi | oracle | kfgreedy
    k: int
    eta: float = 0.5
    seed: int = 0
    kappa: float = 0.5
    weather: str = "power"
    extra: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        e = f"_eta{self.eta:g}_kap{self.kappa:g}" if self.policy.startswith(("voi", "kfvoi")) else ""
        w = "" if self.weather == "power" else f"_w{self.weather}"
        return f"{self.estimator}__{self.policy}{e}__K{self.k}__s{self.seed}{w}"


def period_bounds(data: CAFData, period: str, limit: tuple | None = None) -> tuple[int, int]:
    a, b = data.split[period]
    if limit is not None:  # restrict to a sub-period given as timestamps
        a = max(a, int(data.index.searchsorted(pd.Timestamp(limit[0]))))
        b = min(b, int(data.index.searchsorted(pd.Timestamp(limit[1]), side="right")))
    return a, b


def _spec(rows: list[Row], gen_seed: int) -> engine.PolicySpec:
    B = len(rows)
    g = torch.Generator().manual_seed(gen_seed)
    # the round-robin order depends on the seed only, so every pipeline evaluated with the same seed and
    # budget follows exactly the same round-robin schedule (paired comparisons)
    tb = (torch.rand(1, 42, generator=g) * 1e-3).expand(B, -1).clone()
    return engine.PolicySpec(names=[r.policy for r in rows], k=torch.as_tensor([r.k for r in rows]),
                             eta=torch.as_tensor([r.eta for r in rows], dtype=torch.float32),
                             kappa=torch.as_tensor([r.kappa for r in rows], dtype=torch.float32), max_age=168.0,
                             tiebreak=tb, gen=g)


# --------------------------------------------------------------------------- estimators
def load_vigil(path: Path, data: CAFData) -> VIGILNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ck["info"]["config"]["vigil"]
    from vigil.train import build_vigil
    m = build_vigil(data, mcfg)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def run_vigil(model: VIGILNet, data: CAFData, rows: list[Row], a: int, b: int) -> list[dict]:
    B = len(rows)
    spec = _spec(rows, 1000 + rows[0].seed)
    t0 = torch.full((B,), a, dtype=torch.long)
    with torch.no_grad():
        rec = engine.rollout(model, data, t0, b - a, WARMUP, spec, train=False, record=True).stack()
    out = []
    for j in range(B):
        out.append({"mu": rec["mu"][j], "sigma": rec["sigma"][j], "p_head": gaussian_event_prob(rec["mu"][j], rec["sigma"][j], data),
                    "p_net": rec["p"][j].astype(np.float32), "nu": rec["nu"][j], "poll": rec["poll"][j]})
    return out


def gaussian_event_prob(mu: np.ndarray, sigma: np.ndarray, data: CAFData) -> np.ndarray:
    """P(x_t - min_{24h} >= delta) under the probabilistic nowcast N(mu, sigma^2), with the trailing
    minimum taken over the estimator's own means (as for the Kalman filter)."""
    S, N, D = mu.shape
    dn = EVENT_DELTA / data.vwc_std
    mn = pd.DataFrame(mu.reshape(S, N * D)).rolling(25, min_periods=1).min().to_numpy().reshape(S, N, D)
    return (1 - norm.cdf((dn - (mu - mn)) / np.maximum(sigma, 1e-6))).astype(np.float32)


def run_baseline(name: str, model, data: CAFData, row: Row, a: int, b: int) -> dict:
    spec = _spec([row], 1000 + row.seed)
    t0 = torch.tensor([a])
    with torch.no_grad():
        sim = engine.simulate_inputs(data, t0, b - a, WARMUP, spec)
        inp, last = sim["inp"][0], sim["last_x"][0]           # [S,N,F], [S,N,D]
        if name == "locf":
            mu = last
        elif name == "lgbm":
            mu = gbdt.predict(model, inp, last)
        elif model.kind in ("step", "conv"):
            mu = model(inp[None], last[None])[0]
        elif model.kind == "recurrent":
            mu, _ = baselines.run_sequence(model, inp[None], last[None])
            mu = mu[0]
        elif model.kind == "window":
            W = model.window
            pad_i = torch.cat([inp[:1].expand(W - 1, -1, -1), inp], 0)
            pad_l = torch.cat([last[:1].expand(W - 1, -1, -1), last], 0)
            outs = []
            for c in range(0, inp.shape[0], 64):
                idx = torch.arange(c, min(c + 64, inp.shape[0]))
                wi = torch.stack([pad_i[i:i + W] for i in idx.tolist()])
                wl = torch.stack([pad_l[i:i + W] for i in idx.tolist()])
                outs.append(model(wi, wl)[:, -1])
            mu = torch.cat(outs, 0)
        else:
            raise ValueError(name)
        obs = sim["obs"][0] > 0
        mu = torch.where(obs, last, mu)
    S = WARMUP
    return {"mu": mu[S:].numpy(), "sigma": None, "p_head": None, "poll": sim["poll"][0, S:].numpy()}


def run_kf(kf: RainKF, data: CAFData, row: Row, a: int, b: int) -> dict:
    N, D = data.N, data.D
    rng = np.random.default_rng(1000 + row.seed)
    tieb = (torch.rand(1, N, generator=torch.Generator().manual_seed(1000 + row.seed)) * 1e-3)[0].numpy()
    prcp = data.prcp_mm.numpy()
    p24 = pd.Series(prcp).rolling(24, min_periods=1).sum().to_numpy()
    x = data.x.numpy()
    t_init = a - WARMUP - 1
    mean, P = kf.init(x[t_init])
    node_age = np.zeros(N)
    avail = np.where(np.isfinite(x[t_init].reshape(-1)), 1.0, 0.5)
    # state of the VoI index applied to the KF's predictive distributions ('kfvoi')
    floor = np.diag(P).copy()
    fresh_prev = np.isfinite(x[t_init].reshape(-1))
    last_obs = mean.copy()
    cum_g = np.zeros(N)
    dn_flat = np.tile(np.asarray(data.delta_norm, float), N)
    S = b - a + WARMUP
    mu_out = np.empty((S, N, D), np.float32)
    sd_out = np.empty((S, N, D), np.float32)
    poll_out = np.zeros((S, N), bool)
    for s in range(S):
        t = a - WARMUP + s
        mean, P = kf.predict(mean, P, p24[t] >= kf.wet_mm_24h)
        node_age += 1
        due = np.where(node_age >= 168)[0]
        if row.policy == "kfgreedy":
            sel = kf.greedy_select(P, row.k, node_age, 168.0, avail)
        elif row.policy == "kfvoi":
            # same index as VIGIL (Eqs. growth-whittle) computed from the KF state variances
            var = np.diag(P).copy()
            floor = np.where(fresh_prev, var, floor)
            gs = (avail * np.clip(var - floor, 0.0, None)).reshape(N, D).sum(1)
            a_ = np.maximum(node_age, 1.0)
            rate = gs / a_
            rate = row.kappa * rate + (1 - row.kappa) * rate.mean()
            gs = rate * a_
            sd_ = np.sqrt(var + kf.R)
            gu = (avail * (1 - norm.cdf((dn_flat - (mean - last_obs)) / sd_))).reshape(N, D).sum(1)
            g = (1 - row.eta) * gs / max(gs.mean(), 1e-12) + row.eta * gu / max(gu.mean(), 1e-12)
            cum_g += g
            sc = np.where(node_age >= 168, 1e6 + node_age, node_age * g - cum_g + tieb)
            sel = np.argsort(-sc)[:row.k]
            cum_g[sel] = 0.0
        else:
            sc = node_age + tieb if row.policy == "rr" else rng.random(N)
            sc = np.where(node_age >= 168, 1e6 + node_age, sc)
            sel = np.argsort(-sc)[:row.k]
        poll_out[s, sel] = True
        node_age[sel] = 0
        y = x[t, sel].reshape(-1)
        idx = np.concatenate([kf.channels(i) for i in sel])
        ok = np.isfinite(y)
        avail[idx] = 0.7 * avail[idx] + 0.3 * ok
        mean, P = kf.update(mean, P, idx[ok], y[ok])
        fresh_prev = np.zeros(N * D, bool)
        fresh_prev[idx[ok]] = True
        last_obs[idx[ok]] = y[ok]
        mu_out[s] = mean.reshape(N, D)
        sd_out[s] = np.sqrt(np.clip(np.diag(P), 1e-10, None)).reshape(N, D)
        # observed channels are known exactly at this hour
        o = np.zeros(N * D, bool)
        o[idx[ok]] = True
        o = o.reshape(N, D)
        mu_out[s][o] = x[t][o]
        sd_out[s][o] = engine.SIGMA_OBS
    S0 = WARMUP
    mu, sd = mu_out[S0:], sd_out[S0:]
    # Gaussian probability that the rise above the trailing 24-h minimum (of the means) exceeds delta
    dn = EVENT_DELTA / data.vwc_std
    flat = pd.DataFrame(mu_out.reshape(S, N * D))
    mn = flat.rolling(25, min_periods=1).min().to_numpy().reshape(S, N, D)[S0:]
    p = 1 - norm.cdf((dn - (mu - mn)) / np.maximum(sd, 1e-6))
    return {"mu": mu, "sigma": sd, "p_head": p.astype(np.float32), "poll": poll_out[S0:]}


# --------------------------------------------------------------------------- scoring
# 0.1..2 (31 log-spaced points) extended below 0.1 so that very sharp filters can be recalibrated too
SCALE_GRID = np.concatenate([np.exp(np.linspace(np.log(0.01), np.log(0.1), 11))[:-1],
                             np.exp(np.linspace(np.log(0.1), np.log(2.0), 31))])


def fit_sigma_scale(out: dict, data: CAFData, a: int, b: int, max_points: int = 2_000_000, seed: int = 0) -> float:
    """Post-hoc recalibration on the validation period: the scalar s that minimises the mean CRPS of the
    Gaussian predictive distributions N(mu, (s*sigma)^2) of unpolled channels."""
    x = data.x[a:b].numpy()
    polled = out["poll"][:, :, None] & np.isfinite(x)
    unobs = np.isfinite(x) & ~polled
    z = ((out["mu"] - x) / out["sigma"])[unobs]
    sg = out["sigma"][unobs]
    if len(z) > max_points:
        idx = np.random.default_rng(seed).choice(len(z), max_points, replace=False)
        z, sg = z[idx], sg[idx]
    best, best_v = 1.0, np.inf
    for sc in SCALE_GRID:
        zz = z / sc
        crps = sc * sg * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / np.sqrt(np.pi))
        v = float(np.mean(crps))
        if v < best_v:
            best, best_v = float(sc), v
    return best


def recalibrate(out: dict, data: CAFData, scale: float) -> dict:
    """Scale the published standard deviations and recompute the Gaussian event probabilities."""
    o = dict(out)
    o["sigma"] = out["sigma"] * scale
    o["p_head"] = gaussian_event_prob(o["mu"], o["sigma"], data)
    return o


def score_run(out: dict, data: CAFData, a: int, b: int, taus: dict | None) -> tuple[dict, dict, dict]:
    """Compute metrics. If ``taus`` is None (validation), thresholds are fitted and returned."""
    x = data.x[a:b].numpy()
    ev = data.ev[a:b].numpy()
    idx = data.index[a:b]
    poll = out["poll"]
    polled_ch = poll[:, :, None] & np.isfinite(x)
    res, daily = metrics.state_metrics(out["mu"], x, polled_ch, data.vwc_std, idx, out["sigma"])
    res["polls_per_hour"] = float(poll.sum(1).mean())
    dn = EVENT_DELTA / data.vwc_std
    scores = {"rule": metrics.rule_event_score(out["mu"], dn)}
    if out.get("p_head") is not None:
        scores["head"] = out["p_head"]
    fitted = {}
    extra = {"daily_rmse": daily.to_numpy().astype(np.float32),
             "days": np.asarray(daily.index.astype(str).to_numpy(), dtype="U10")}
    # error by weather regime (wet = >= 5 mm of rain in the previous 24 h) and by month
    err2 = ((out["mu"] - x) * data.vwc_std) ** 2
    p24 = pd.Series(data.prcp_mm.numpy()).rolling(24, min_periods=1).sum().to_numpy()[a:b]
    for name, msk in [("wet", p24 >= 5.0), ("dry", p24 < 1.0)]:
        e = err2[msk]
        res[f"rmse_{name}"] = float(np.sqrt(np.nanmean(e))) if np.isfinite(e).any() else float("nan")
        res[f"hours_{name}"] = int(msk.sum())
    mon = idx.to_period("M").astype(str)
    mser = pd.Series(np.nanmean(err2.reshape(len(idx), -1), axis=1), index=mon)
    extra["monthly_rmse"] = np.sqrt(mser.groupby(level=0).mean()).to_numpy().astype(np.float32)
    extra["months"] = np.array(sorted(set(mon)), dtype="U7")
    if out.get("nu") is not None:
        u = data.u[a:b].numpy()
        ok = np.isfinite(u)
        res["outlook24_rmse"] = float(np.sqrt(np.nanmean(((out["nu"] - u) * data.vwc_std) ** 2)))
        res["outlook24_rmse_persist"] = float(np.sqrt(np.nanmean(((out["mu"] - u) * data.vwc_std) ** 2)))
    if out.get("sigma") is not None:
        unobs = np.isfinite(x) & ~polled_ch
        z = np.abs((out["mu"] - x) / out["sigma"])[unobs]
        levels = np.linspace(0.1, 0.9, 9)
        from scipy.stats import norm as _n
        res["calib_levels"] = levels.tolist()
        res["calib_coverage"] = [float((z <= _n.ppf(0.5 + l / 2)).mean()) for l in levels]
    for name, sc in scores.items():
        tau = metrics.best_threshold(sc, ev) if taus is None else taus[name]
        fitted[name] = tau
        em, delays = metrics.event_metrics(sc, ev, tau)
        res.update({f"ev_{name}_{k}": v for k, v in em.items()})
        res[f"ev_{name}_tau"] = tau
        extra[f"delays_{name}"] = delays.astype(np.float32)
        # alert onsets aggregated per logger-hour (input to the edge-cloud case study)
        al = sc >= tau
        rise = al & ~np.concatenate([np.zeros_like(al[:1]), al[:-1]], 0)
        any_rise = rise.any(-1)
        tt, ii = np.where(any_rise)
        pp = np.where(rise, sc, -np.inf).max(-1)[tt, ii]
        extra[f"alerts_{name}_t"] = tt.astype(np.int32)
        extra[f"alerts_{name}_i"] = ii.astype(np.int16)
        extra[f"alerts_{name}_p"] = pp.astype(np.float32)
    return res, fitted, extra


def evaluate_rows(rows: list[Row], estimator_obj, data_cache: dict, out_dir: Path,
                  test_limit: tuple | None = None, skip_existing: bool = True) -> list[dict]:
    """Evaluate rows that share one estimator object (same estimator name + seed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (skip_existing and (out_dir / f"{r.tag}.json").exists())]
    if not todo:
        return []
    results = []
    est = todo[0].estimator
    groups: dict[str, list[Row]] = {}
    for r in todo:
        groups.setdefault(r.weather, []).append(r)
    for wsrc, grp in groups.items():
        data_val = data_cache.setdefault("power", load("power"))
        data_test = data_cache.setdefault(wsrc, load(wsrc))
        va, vb = period_bounds(data_val, "val")
        ta, tb = period_bounds(data_test, "test", test_limit)
        # rows are processed in chunks that are scored and written immediately (bounded memory)
        size = 6 if est.startswith("vigil") else 1
        for c in range(0, len(grp), size):
            chunk = grp[c:c + size]
            tic = time.time()
            if est.startswith("vigil"):
                outs_v = run_vigil(estimator_obj, data_val, chunk, va, vb)
                outs_t = run_vigil(estimator_obj, data_test, chunk, ta, tb)
            elif est == "kf":
                outs_v = [run_kf(estimator_obj, data_val, r, va, vb) for r in chunk]
                outs_t = [run_kf(estimator_obj, data_test, r, ta, tb) for r in chunk]
            else:
                outs_v = [run_baseline(est, estimator_obj, data_val, r, va, vb) for r in chunk]
                outs_t = [run_baseline(est, estimator_obj, data_test, r, ta, tb) for r in chunk]
            el = time.time() - tic
            for r, ov, ot in zip(chunk, outs_v, outs_t):
                scale = float("nan")
                if ov.get("sigma") is not None:          # probabilistic estimators: variance recalibration
                    scale = fit_sigma_scale(ov, data_val, va, vb)
                    ov, ot = recalibrate(ov, data_val, scale), recalibrate(ot, data_test, scale)
                _, taus, _ = score_run(ov, data_val, va, vb, None)
                res, _, extra = score_run(ot, data_test, ta, tb, taus)
                res["sigma_scale"] = scale
                res.update({"estimator": r.estimator, "policy": r.policy, "k": r.k, "eta": r.eta, "seed": r.seed,
                            "kappa": r.kappa, "weather": r.weather, "test_start": str(data_test.index[ta]),
                            "test_end": str(data_test.index[tb - 1]), "eval_seconds_chunk": el,
                            "chunk_rows": len(chunk)})
                with open(out_dir / f"{r.tag}.json", "w") as fh:
                    json.dump(res, fh)
                np.savez_compressed(out_dir / f"{r.tag}.npz", **extra)
                results.append(res)
                print(r.tag, json.dumps({k: round(v, 5) for k, v in res.items()
                                         if k in ("rmse", "rmse_unobs", "ev_head_auprc", "ev_rule_auprc",
                                                  "ev_head_recall_6h", "ev_rule_recall_6h", "crps_unobs")}), flush=True)
            del outs_v, outs_t
    return results


def load_estimator(name: str, seed: int, data: CAFData):
    if name.startswith("vigil"):
        variant = name.split("-", 1)[1] if "-" in name else "default"
        return load_vigil(repo_path("results", "models", f"vigil-{variant}", f"seed{seed}", "model.pt"), data)
    if name == "locf":
        return None
    if name == "kf":
        return fit_kf(data)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(repo_path("results", "models", "lgbm-default", f"seed{seed}", "model.txt")))
    ck = torch.load(repo_path("results", "models", f"{name}-default", f"seed{seed}", "model.pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["info"]["config"]
    d_in = engine.ctx_dim(data) + engine.MEM_DIM
    m = baselines.build(name, d_in, data.D, data.adj, data.N, **cfg["baselines"][name])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def fit_kf(data: CAFData, shrink_grid=(0.2, 0.5, 0.8, 1.0), r_grid=(0.001, 0.01, 0.1, 1.0)) -> RainKF:
    """Fit the rain-conditioned KF on the training period; the shrinkage of the process-noise
    covariance towards its diagonal and the scale of the measurement noise are selected by
    validation RMSE (round-robin, K=2)."""
    cache = repo_path("results", "models", "kf-default", "kf_selection.json")
    a, b = data.split["train"]
    p24 = pd.Series(data.prcp_mm.numpy()).rolling(24, min_periods=1).sum().to_numpy()
    if cache.exists():
        sel = json.load(open(cache))
        return RainKF(shrink=sel["shrink"], r_scale=sel["r_scale"]).fit(data.x[a:b].numpy(), p24[a:b])
    va, vb = period_bounds(data, "val")
    scores = {}
    for lam in shrink_grid:
        for rs in r_grid:
            kf = RainKF(shrink=lam, r_scale=rs).fit(data.x[a:b].numpy(), p24[a:b])
            out = run_kf(kf, data, Row("kf", "rr", 2), va, vb)
            err = (out["mu"] - data.x[va:vb].numpy()) * data.vwc_std
            scores[(lam, rs)] = float(np.sqrt(np.nanmean(err ** 2)))
            print("kf selection", lam, rs, round(scores[(lam, rs)], 6), flush=True)
    best = min(scores, key=scores.get)
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"shrink": best[0], "r_scale": best[1],
               "val_rmse": {f"{k[0]}|{k[1]}": v for k, v in scores.items()}}, open(cache, "w"), indent=1)
    return RainKF(shrink=best[0], r_scale=best[1]).fit(data.x[a:b].numpy(), p24[a:b])
