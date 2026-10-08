"""Compare NASA POWER (gridded, MERRA-2/CERES) with SCAN 2198 (on-farm station).

Usage: python3 -I compare_power_scan.py <power_csv> <scan_csv> <out_dir>
Answers: is POWER a usable weather substitute for CAF months without SCAN data?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

p_csv, s_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

# ---- NASA POWER ----
lines = p_csv.read_text().splitlines()
start = lines.index("-END HEADER-") + 1
p = pd.read_csv(p_csv, skiprows=start)
p.index = pd.to_datetime(dict(year=p.YEAR, month=p.MO, day=p.DY, hour=p.HR))
p = p.drop(columns=["YEAR", "MO", "DY", "HR"]).replace(-999, np.nan)
print("POWER rows:", len(p), p.index.min(), "->", p.index.max())
print("POWER missing (-999) per column:", p.isna().sum().to_dict())
exp = pd.date_range(p.index.min(), p.index.max(), freq="h")
print("POWER absent hours:", len(exp.difference(p.index)), " duplicates:", p.index.duplicated().sum())
p = p.rename(columns={"T2M": "temp_c", "RH2M": "rh", "PRECTOTCORR": "prcp_mm",
                      "WS2M": "wind_ms", "ALLSKY_SFC_SW_DWN": "srad_wm2"})
p["srad_wm2"] = p["srad_wm2"] * 1e6 / 3600.0  # MJ/hr -> W/m2

# ---- SCAN (convert to SI) ----
s = pd.read_csv(s_csv, comment="#")
s.index = pd.to_datetime(s["Date"], format="%Y-%m-%d %H:%M")
s = pd.DataFrame({
    "temp_c": (s["Air Temperature Observed (degF)"] - 32) * 5 / 9,
    "rh": s["Relative Humidity (pct)"],
    "prcp_mm": s["Precipitation Increment (in)"] * 25.4,
    "wind_ms": s["Wind Speed Average (mph)"] * 0.44704,
    "srad_wm2": s["Solar Radiation Average (watt/m2)"],
})

# ---- time alignment check: lag maximising hourly temperature correlation ----
best = []
for lag in range(-4, 5):
    j = s[["temp_c"]].join(p[["temp_c"]].shift(lag, freq="h"), lsuffix="_s", rsuffix="_p", how="inner").dropna()
    best.append((lag, j.corr().iloc[0, 1]))
print("Temp correlation by POWER lag (h):", [(l, round(c, 3)) for l, c in best])

j = s.join(p, lsuffix="_scan", rsuffix="_power", how="inner")
rows = []
for v in ["temp_c", "rh", "prcp_mm", "wind_ms", "srad_wm2"]:
    a = j[[f"{v}_scan", f"{v}_power"]].dropna()
    d = j[[f"{v}_scan", f"{v}_power"]].resample("D").sum() if v == "prcp_mm" else \
        j[[f"{v}_scan", f"{v}_power"]].resample("D").mean()
    d = d.dropna()
    rows.append(dict(variable=v, n_hours=len(a),
                     r_hourly=a.corr().iloc[0, 1],
                     bias_power_minus_scan=(a.iloc[:, 1] - a.iloc[:, 0]).mean(),
                     rmse_hourly=np.sqrt(((a.iloc[:, 1] - a.iloc[:, 0]) ** 2).mean()),
                     r_daily=d.corr().iloc[0, 1],
                     scan_mean=a.iloc[:, 0].mean(), power_mean=a.iloc[:, 1].mean()))
res = pd.DataFrame(rows).set_index("variable").round(3)
res.to_csv(out / "power_vs_scan.csv")
print(res.to_string())

# precipitation event agreement (daily, wet day = >1 mm)
d = j[["prcp_mm_scan", "prcp_mm_power"]].resample("D").sum(min_count=20).dropna()
ws, wp = d.prcp_mm_scan > 1, d.prcp_mm_power > 1
print("Daily wet-day (>1 mm) agreement: hit rate %.2f, false alarm ratio %.2f, totals SCAN %.0f mm vs POWER %.0f mm" % (
    (ws & wp).sum() / ws.sum(), (wp & ~ws).sum() / wp.sum(), d.prcp_mm_scan.sum(), d.prcp_mm_power.sum()))
