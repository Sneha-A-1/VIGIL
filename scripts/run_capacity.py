"""Neural forecasting of co-tenant CPU load on Bitbrains VMs (used by the urgency-aware dispatcher).

    python scripts/run_capacity.py

Outputs results/raw/capacity/{split.json, results.json, gru_seed<k>.pt}.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from vigil.config import data_root, load_config, repo_path
from vigil.data.bitbrains import load_grid
from vigil.models import capacity


def main(seeds=(0, 1, 2, 3, 4)):
    torch.set_num_threads(1)
    cfg = load_config("data")
    out = repo_path("results", "raw", "capacity")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "bitbrains_grid.npz"
    if cache.exists():
        g = dict(np.load(cache))
        g["t0"] = int(g["t0"])
    else:
        g = load_grid(data_root() / cfg["paths"]["bitbrains_dir"])
        np.savez_compressed(cache, **g)
    use = g["use_pct"] / 100.0
    V = use.shape[0]
    rng = np.random.default_rng(2024)
    perm = rng.permutation(V)
    n_te = V // 5
    te, tr = np.sort(perm[:n_te]), np.sort(perm[n_te:])
    json.dump({"train_vms": g["vm_ids"][tr].tolist(), "test_vms": g["vm_ids"][te].tolist(),
               "n_vms": int(V), "steps": int(use.shape[1])}, open(out / "split.json", "w"))
    results = []
    for s in seeds:
        model = capacity.train(use[tr], g["t0"], seed=s)
        r = capacity.evaluate(model, use[te], g["t0"])
        r["seed"] = s
        results.append(r)
        torch.save(model.state_dict(), out / f"gru_seed{s}.pt")
        print(json.dumps(r), flush=True)
    json.dump(results, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
