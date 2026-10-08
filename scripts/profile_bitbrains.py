"""Profile Bitbrains GWA-T-12 fastStorage trace.

Usage: python3 -I profile_bitbrains.py <extracted_dir> <out_dir>
Checks file count, columns, timestamp unit, sampling interval, time span,
missing values and per-VM utilisation statistics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
files = sorted(root.rglob("*.csv"), key=lambda p: int(p.stem))
print("VM files:", len(files))

cols = ["ts", "cores", "cpu_cap_mhz", "cpu_use_mhz", "cpu_pct", "mem_cap_kb", "mem_use_kb",
        "disk_r_kbs", "disk_w_kbs", "net_rx_kbs", "net_tx_kbs"]
rows, agg = [], None
for f in files:
    d = pd.read_csv(f, sep=";\t", engine="python", header=0, names=cols)
    dt = np.diff(d["ts"].values)
    rows.append(dict(
        vm=int(f.stem), n=len(d), t0=d.ts.min(), t1=d.ts.max(),
        dt_median=float(np.median(dt)) if len(dt) else np.nan,
        dt_share_300=float(np.mean(dt == 300)) if len(dt) else np.nan,
        dup_ts=int(d.ts.duplicated().sum()), nan_cells=int(d.isna().sum().sum()),
        cores=d.cores.median(), cpu_pct_mean=d.cpu_pct.mean(), cpu_pct_p95=d.cpu_pct.quantile(0.95),
        mem_frac_mean=(d.mem_use_kb / d.mem_cap_kb.replace(0, np.nan)).mean(),
    ))
    # cluster-level CPU demand on a common 5-minute grid
    s = d.assign(slot=(d.ts // 300) * 300).groupby("slot")["cpu_use_mhz"].mean()
    agg = s if agg is None else agg.add(s, fill_value=0)

st = pd.DataFrame(rows)
st.to_csv(out / "per_vm_stats.csv", index=False)
print("Total rows:", st.n.sum())
print("Rows per VM:", st.n.describe().round(0).to_dict())
t0, t1 = pd.to_datetime(st.t0.min(), unit="s"), pd.to_datetime(st.t1.max(), unit="s")
print("Span (timestamps read as SECONDS):", t0, "->", t1, f"({(t1 - t0).days} days)")
print("Median sampling interval (s):", st.dt_median.median(),
      " mean share of 300-s steps:", round(st.dt_share_300.mean(), 3))
print("VMs with duplicate timestamps:", int((st.dup_ts > 0).sum()), " NaN cells total:", st.nan_cells.sum())
print("Cores per VM:", st.cores.value_counts().sort_index().to_dict())
print("Mean CPU% per VM:", st.cpu_pct_mean.describe().round(2).to_dict())
print("VMs with mean CPU% < 5:", int((st.cpu_pct_mean < 5).sum()))
agg.index = pd.to_datetime(agg.index, unit="s")
agg.rename("cluster_cpu_use_mhz").to_csv(out / "cluster_cpu_demand_5min.csv")
print("Cluster CPU demand (GHz): mean %.1f, p5 %.1f, p95 %.1f, max %.1f" % tuple(
    np.array([agg.mean(), agg.quantile(.05), agg.quantile(.95), agg.max()]) / 1000))
