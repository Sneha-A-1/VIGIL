"""Example trajectories around the largest wetting episode of the test period (for Fig. 'trace').

    python scripts/example_trace.py  ->  results/raw/example/trace.npz

Runs LOCF + round-robin, VIGIL + round-robin and VIGIL + VoI (seed 0, K=2) over a 14-day window that
contains the hour with the most event onsets in the test period (48-h rolling count over all channels).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from vigil.config import load_config, repo_path
from vigil.data.dataset import load
from vigil.evaluate import Row, load_estimator, run_baseline, run_vigil


def main(k: int = 2, seed: int = 0):
    torch.set_num_threads(1)
    d = load()
    a, b = d.split["test"]
    ev = d.ev[a:b].numpy()
    prev = np.concatenate([np.zeros_like(ev[:1]), ev[:-1]], 0)
    onsets = ((np.nan_to_num(ev) > 0.5) & (np.nan_to_num(prev) < 0.5)).sum((1, 2))
    roll = pd.Series(onsets).rolling(48, min_periods=1).sum().to_numpy()
    peak = a + int(np.argmax(roll))
    wa, wb = peak - 5 * 24, peak + 9 * 24
    tun = repo_path("results", "raw", "tuning.json")
    kappa = float(json.load(open(tun))["kappa"]) if tun.exists() else 0.5
    m = load_estimator("vigil", seed, d)
    rows = [Row("vigil", "rr", k, seed=seed), Row("vigil", "voi", k, seed=seed, kappa=kappa)]
    outs = run_vigil(m, d, rows, wa, wb)
    locf = run_baseline("locf", None, d, Row("locf", "rr", k, seed=seed), wa, wb)
    res = {"x": d.x[wa:wb].numpy() * d.vwc_std + d.vwc_mean, "prcp": d.prcp_mm[wa:wb].numpy(),
           "ev": d.ev[wa:wb].numpy(), "hours": np.asarray(d.index[wa:wb].astype(str).to_numpy(), dtype="U19"),
           "locf_mu": locf["mu"] * d.vwc_std + d.vwc_mean, "locf_poll": locf["poll"], "k": k, "kappa": kappa}
    # validation-year variance recalibration of the same pipelines (see evaluate.fit_sigma_scale)
    ev_dir = repo_path("results", "raw", "eval")
    scales = {}
    for name, f in [("rr", ev_dir / "estimators" / f"vigil__rr__K{k}__s{seed}.json"),
                    ("voi", ev_dir / "policies" / f"vigil__voi_eta0.5_kap{kappa:g}__K{k}__s{seed}.json")]:
        scales[name] = float(json.load(open(f)).get("sigma_scale", 1.0)) if f.exists() else 1.0
    res["sigma_scales"] = np.array([scales["rr"], scales["voi"]])
    for name, o in zip(["rr", "voi"], outs):
        res[f"{name}_mu"] = o["mu"] * d.vwc_std + d.vwc_mean
        res[f"{name}_sigma"] = o["sigma"] * d.vwc_std * scales[name]
        res[f"{name}_poll"] = o["poll"]
        res[f"{name}_p"] = o["p_head"]
    out = repo_path("results", "raw", "example")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trace.npz", **res)
    print("window", d.index[wa], "->", d.index[wb - 1], "peak", d.index[peak])


if __name__ == "__main__":
    main()
