"""Select the shrinkage weight kappa of the VIGIL polling index on the validation year (2014).

    python scripts/tune_policy.py  ->  results/raw/tuning.json

The urgent/strategic weight is fixed a priori at eta = 0.5 (both terms are normalised to unit mean);
kappa is chosen by the mean validation RMSE over budgets and seeds. Test data are not used.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

from vigil import engine, metrics
from vigil.config import load_config, repo_path
from vigil.data.dataset import load
from vigil.evaluate import WARMUP, load_vigil

KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    torch.set_num_threads(1)
    cfg = load_config("experiments")
    d = load()
    a, b = d.split["val"]
    x = d.x[a:b].numpy()
    ev = d.ev[a:b].numpy()
    rows = []
    for seed in cfg["seeds"]:
        m = load_vigil(repo_path("results", "models", "vigil-default", f"seed{seed}", "model.pt"), d)
        for k in cfg["budgets"]:                       # one rollout per budget (bounded memory)
            B = len(KAPPAS)
            g = torch.Generator().manual_seed(1000 + seed)
            tb = (torch.rand(1, 42, generator=g) * 1e-3).expand(B, -1).clone()
            spec = engine.PolicySpec(names=["voi"] * B, k=torch.full((B,), k), eta=torch.full((B,), 0.5),
                                     kappa=torch.tensor(KAPPAS), max_age=168.0, tiebreak=tb, gen=g)
            with torch.no_grad():
                rec = engine.rollout(m, d, torch.full((B,), a), b - a, WARMUP, spec, train=False, record=True).stack()
            for j, kap in enumerate(KAPPAS):
                err = (rec["mu"][j] - x) * d.vwc_std
                sc = metrics.rule_event_score(rec["mu"][j], d.delta_norm)
                em, _ = metrics.event_metrics(sc, ev, 1.0)
                rows.append({"seed": seed, "k": k, "kappa": kap, "rmse": float(np.sqrt(np.nanmean(err ** 2))),
                             "recall_6h": em["recall_6h"], "auprc": em["auprc"]})
                print(json.dumps(rows[-1]), flush=True)
            del rec
    import pandas as pd
    df = pd.DataFrame(rows)
    # relative RMSE per (seed, budget) so that every budget weighs equally
    df["rel"] = df["rmse"] / df.groupby(["seed", "k"])["rmse"].transform("min")
    score = df.groupby("kappa")["rel"].mean()
    best = float(score.idxmin())
    out = {"eta": 0.5, "kappa": best, "kappa_scores": {str(k): float(v) for k, v in score.items()},
           "runs": rows}
    json.dump(out, open(repo_path("results", "raw", "tuning.json"), "w"), indent=1)
    print("selected kappa", best, score.round(4).to_dict())


if __name__ == "__main__":
    main()
