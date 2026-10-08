"""Run an evaluation group of configs/experiments.yaml.

    python scripts/run_experiments.py --group estimators [--shard 0 --nshards 2]

Results: results/raw/eval/<group>/<tag>.json (+ .npz with daily errors, onset delays and alerts).
"""
from __future__ import annotations

import argparse
import itertools
import json

import torch

from vigil.config import load_config, repo_path
from vigil.data.dataset import load
from vigil.evaluate import Row, evaluate_rows, load_estimator


def tuned_kappa(cfg: dict) -> float:
    if cfg.get("kappa", "tuned") != "tuned":
        return float(cfg["kappa"])
    f = repo_path("results", "raw", "tuning.json")
    return float(json.load(open(f))["kappa"]) if f.exists() else 0.5


def rows_for(group: str, cfg: dict) -> list[Row]:
    g = cfg["groups"][group]
    seeds = g.get("seeds", cfg["seeds"])
    budgets = g.get("budgets", cfg["budgets"])
    kap = tuned_kappa(cfg)
    rows = []
    for est, pol, k, s, w in itertools.product(g["estimators"], g["policies"], budgets, seeds,
                                               g.get("weather", ["power"])):
        policy = pol
        if est.endswith("no_lookahead") and pol == "voi":
            policy = "voi_myopic"        # without look-ahead heads the index uses the current-value variance
        etas = g.get("eta_grid", [cfg["eta"]]) if policy.startswith("voi") else [cfg["eta"]]
        kappa = kap
        if policy == "voi_nokappa":      # ablation: no shrinkage of logger-specific growth (kappa = 1)
            policy, kappa = "voi", 1.0
        for eta in etas:
            rows.append(Row(estimator=est, policy=policy, k=int(k), eta=float(eta), seed=int(s), weather=w,
                            kappa=float(kappa)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--estimators", nargs="*", default=None, help="restrict to these estimators")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="restrict to these seeds")
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = load_config("experiments")
    rows = rows_for(a.group, cfg)
    if a.estimators:
        rows = [r for r in rows if r.estimator in a.estimators]
    if a.seeds is not None:
        rows = [r for r in rows if r.seed in a.seeds]
    keys = sorted({(r.estimator, r.seed) for r in rows})
    keys = keys[a.shard::a.nshards]
    out_dir = repo_path("results", "raw", "eval", a.group)
    limit = cfg["groups"][a.group].get("test_limit")
    cache = {}
    data = cache.setdefault("power", load("power"))
    for est, seed in keys:
        sub = [r for r in rows if r.estimator == est and r.seed == seed]
        if all((out_dir / f"{r.tag}.json").exists() for r in sub):
            continue
        try:
            obj = load_estimator(est, seed, data)
        except Exception as ex:          # missing model file (torch: FileNotFoundError; LightGBM: LightGBMError)
            print(f"skip {est} seed {seed}: model not available ({type(ex).__name__}: {ex})", flush=True)
            continue
        evaluate_rows(sub, obj, cache, out_dir, test_limit=tuple(limit) if limit else None)


if __name__ == "__main__":
    main()
