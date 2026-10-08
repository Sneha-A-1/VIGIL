"""Profile SCAN 2198 hourly export. Usage: python3 -I profile_scan.py <csv> <out_dir>"""
import sys
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(src, comment="#")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M")
df = df.set_index("Date").sort_index()
print("Rows:", len(df), "span:", df.index.min(), "->", df.index.max())
print("Duplicate timestamps:", df.index.duplicated().sum())
full = pd.date_range(df.index.min().floor("D"), "2016-06-16 23:00", freq="h")
missing_rows = full.difference(df.index)
print("Hours expected to 2016-06-16 23:00:", len(full), " rows absent:", len(missing_rows))
# contiguous absent blocks
if len(missing_rows):
    s = pd.Series(missing_rows)
    blocks = (s.diff() != pd.Timedelta("1h")).cumsum()
    g = s.groupby(blocks).agg(["min", "max", "count"]).sort_values("count", ascending=False)
    print("Largest absent-row blocks:\n", g.head(8).to_string())

short = {c: c.split(" (")[0] for c in df.columns}
avail = df.notna().mean().rename(index=short).round(3)
print("\nFraction non-missing per column:\n", avail.to_string())

mon = df.notna().groupby(df.index.to_period("M")).mean()
mon.columns = [short[c] for c in mon.columns]
mon.round(2).to_csv(out / "scan_availability_by_month.csv")
print("\nMonthly availability (selected cols):\n",
      mon[["Air Temperature Observed", "Precipitation Increment", "Solar Radiation Average",
           "Soil Moisture Percent -8in"]].round(2).to_string())
print("\nDescribe:\n", df.describe().T[["min", "50%", "max"]].rename(index=short).round(2).to_string())
