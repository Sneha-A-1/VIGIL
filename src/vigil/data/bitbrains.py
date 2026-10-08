"""GWA-T-12 Bitbrains fastStorage trace (Shen et al., 2015): per-VM CPU usage on a 5-minute grid.

Each file holds one VM; rows: timestamp (s since epoch; the archive description says ms but the
values are seconds), cores, provisioned MHz, used MHz, used %, memory, disk and network throughput.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["ts", "cores", "cap_mhz", "use_mhz", "use_pct", "mem_cap_kb", "mem_use_kb",
        "disk_r_kbs", "disk_w_kbs", "net_rx_kbs", "net_tx_kbs"]
STEP = 300  # seconds


def load_grid(vm_dir: Path, min_rows: int = 8000) -> dict:
    """Load every VM with at least ``min_rows`` samples onto a common 5-min grid.

    Returns dict(vm_ids, cores, cap_mhz, use_pct [V,T], t0 (s)). Gaps are linearly interpolated;
    duplicate timestamps are averaged.
    """
    files = sorted(Path(vm_dir).glob("*.csv"), key=lambda p: int(p.stem))
    frames = {}
    t_lo, t_hi = None, None
    for f in files:
        d = pd.read_csv(f, sep=";\t", engine="python", header=0, names=COLS)
        if len(d) < min_rows:
            continue
        slot = (d["ts"] // STEP) * STEP
        g = d.assign(slot=slot).groupby("slot").agg(use_pct=("use_pct", "mean"), cores=("cores", "median"),
                                                     cap_mhz=("cap_mhz", "median"))
        frames[int(f.stem)] = g
        t_lo = g.index.min() if t_lo is None else min(t_lo, g.index.min())
        t_hi = g.index.max() if t_hi is None else max(t_hi, g.index.max())
    grid = np.arange(t_lo, t_hi + STEP, STEP)
    ids = sorted(frames)
    use = np.full((len(ids), len(grid)), np.nan, dtype=np.float32)
    cores = np.zeros(len(ids))
    cap = np.zeros(len(ids))
    for j, vid in enumerate(ids):
        g = frames[vid]
        s = g["use_pct"].reindex(grid).interpolate(limit_direction="both")
        use[j] = np.clip(s.to_numpy(), 0, 100)
        cores[j] = float(g["cores"].median())
        cap[j] = float(g["cap_mhz"].median())
    return {"vm_ids": np.array(ids), "cores": cores, "cap_mhz": cap, "use_pct": use, "t0": int(grid[0]),
            "step": STEP}
