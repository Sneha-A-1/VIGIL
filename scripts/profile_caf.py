"""Quick profile of the Cook Agronomy Farm (CAF) sensor dataset.

Usage: python3 -I profile_caf.py <extracted_dir> <out_dir>
Outputs coverage tables (per year x depth, per location) and the farm's
sensor coordinates in WGS84 so the matching weather record can be requested.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

vw_cols = [f"VW_{d}cm" for d in (30, 60, 90, 120, 150)]
t_cols = [f"T_{d}cm" for d in (30, 60, 90, 120, 150)]

# ---- hourly files -------------------------------------------------------
frames = []
for f in sorted((root / "caf_sensors" / "Hourly").glob("CAF*.txt")):
    df = pd.read_csv(f, sep="\t", na_values=["NA"])
    frames.append(df)
h = pd.concat(frames, ignore_index=True)
h["ts"] = pd.to_datetime(h["Date"] + " " + h["Time"], format="%m/%d/%Y %H:%M")
print("Hourly rows:", len(h), "locations:", h["Location"].nunique())
print("Time span:", h["ts"].min(), "->", h["ts"].max())
print("Rows per location (expected equal):", h.groupby("Location").size().describe().to_dict())

# step check
one = h[h["Location"] == h["Location"].iloc[0]].sort_values("ts")
steps = one["ts"].diff().value_counts().head(5)
print("Timestep distribution (first location):\n", steps)

# overall availability
avail = h[vw_cols + t_cols].notna().mean().round(3)
print("\nFraction non-missing overall:\n", avail)

# per year x depth (VW)
h["year"] = h["ts"].dt.year
yr = h.groupby("year")[vw_cols].apply(lambda g: g.notna().mean()).round(3)
print("\nVW availability by year:\n", yr)
yr.to_csv(out / "vw_availability_by_year.csv")

yrT = h.groupby("year")[t_cols].apply(lambda g: g.notna().mean()).round(3)
yrT.to_csv(out / "t_availability_by_year.csv")

# per location
loc = h.groupby("Location")[vw_cols].apply(lambda g: g.notna().mean()).round(3)
loc.to_csv(out / "vw_availability_by_location.csv")
print("\nVW availability by location (summary):\n", loc.mean(axis=1).describe())

# monthly fraction of sensors (210 VW channels) reporting
h["month"] = h["ts"].dt.to_period("M")
mon = h.groupby("month")[vw_cols].apply(lambda g: g.notna().mean().mean()).rename("vw_frac")
mon.to_csv(out / "vw_availability_by_month.csv")

# best contiguous window: months with >=70% of VW channels reporting
good = mon[mon >= 0.70]
print("\nMonths with >=70% of VW channels reporting:", len(good), "of", len(mon))
if len(good):
    print("First/last such month:", good.index.min(), good.index.max())

# value ranges after dropping NA
print("\nVW describe:\n", h[vw_cols].describe().round(3))
print("\nT describe:\n", h[t_cols].describe().round(2))

# ---- coordinates ----------------------------------------------------------
sf = shapefile.Reader(str(root / "CAF_sensors"))
fields = [f[0] for f in sf.fields[1:]]
recs = []
tr = Transformer.from_crs("EPSG:26911", "EPSG:4326", always_xy=True)  # NAD83 / UTM 11N -> WGS84
for sr in sf.shapeRecords():
    x, y = sr.shape.points[0]
    lon, lat = tr.transform(x, y)
    rec = dict(zip(fields, sr.record))
    rec.update(easting=x, northing=y, lon=lon, lat=lat)
    recs.append(rec)
coords = pd.DataFrame(recs)
coords.to_csv(out / "sensor_coordinates.csv", index=False)
print("\nShapefile fields:", fields)
print("Sensor coordinates (WGS84): lat %.5f..%.5f, lon %.5f..%.5f" % (
    coords.lat.min(), coords.lat.max(), coords.lon.min(), coords.lon.max()))
print("Farm centroid: lat %.5f, lon %.5f" % (coords.lat.mean(), coords.lon.mean()))
