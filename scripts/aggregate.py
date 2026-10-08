"""Aggregate evaluation JSON files into tidy CSV tables (mean over seeds + spread kept for the repo).

    python scripts/aggregate.py   ->  results/processed/<group>.csv  and  <group>_seeds.csv
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from vigil.config import repo_path

KEYS = ["estimator", "policy", "k", "eta", "weather"]


def load_group(group: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(repo_path("results", "raw", "eval", group, "*.json")))):
        r = json.load(open(f))
        r = {k: v for k, v in r.items() if not isinstance(v, list)}
        r["tag"] = Path(f).stem
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = [c for c in df.columns if c not in KEYS + ["seed", "tag", "test_start", "test_end"]
           and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(KEYS, dropna=False)
    mean = g[num].mean().reset_index()
    std = g[num].std().reset_index()
    mean["n_seeds"] = g.size().values
    return mean, std


def main():
    out = repo_path("results", "processed")
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted(glob.glob(str(repo_path("results", "raw", "eval", "*")))):
        group = Path(d).name
        df = load_group(group)
        if df.empty:
            continue
        df.to_csv(out / f"{group}_seeds.csv", index=False)
        mean, std = summarise(df)
        mean.to_csv(out / f"{group}.csv", index=False)
        std.to_csv(out / f"{group}_std.csv", index=False)
        print(group, len(df), "runs ->", len(mean), "configurations")


if __name__ == "__main__":
    main()
