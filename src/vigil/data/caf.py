"""Loading and quality control of the Cook Agronomy Farm (CAF) sensor network.

Gasch et al. (2017), Vadose Zone Journal; data: USDA Ag Data Commons (CAF_Sensor_Dataset.zip).
42 logger locations x 5 depths, hourly volumetric water content (VWC, m3/m3) and soil
temperature (T, degC), 2007-04-20 .. 2016-06-16.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import tifffile

DEPTHS = (30, 60, 90, 120, 150)
VW_COLS = [f"VW_{d}cm" for d in DEPTHS]
T_COLS = [f"T_{d}cm" for d in DEPTHS]


def list_locations(caf_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(caf_dir) / "caf_sensors" / "Hourly").glob("CAF*.txt"))


def load_hourly(caf_dir: Path, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Return VWC and T arrays shaped [time, location, depth] on ``index`` (hourly).

    Duplicate timestamps (present for CAF201) are resolved by averaging the duplicates.
    """
    locs = list_locations(caf_dir)
    vw = np.full((len(index), len(locs), len(DEPTHS)), np.nan, dtype=np.float32)
    tt = np.full_like(vw, np.nan)
    info = {"duplicates": {}}
    for j, loc in enumerate(locs):
        f = Path(caf_dir) / "caf_sensors" / "Hourly" / f"{loc}.txt"
        d = pd.read_csv(f, sep="\t", na_values=["NA"])
        d["ts"] = pd.to_datetime(d["Date"] + " " + d["Time"], format="%m/%d/%Y %H:%M")
        n_dup = int(d["ts"].duplicated().sum())
        if n_dup:
            info["duplicates"][loc] = n_dup
            d = d.groupby("ts", as_index=False)[VW_COLS + T_COLS].mean()
        d = d.set_index("ts").reindex(index)
        vw[:, j, :] = d[VW_COLS].to_numpy(np.float32)
        tt[:, j, :] = d[T_COLS].to_numpy(np.float32)
    return vw, tt, locs, info


def _read_flag_file(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d.columns = [c.strip('"') for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].str.strip('"')
    return d


def apply_qc(vw: np.ndarray, tt: np.ndarray, locs: list[str], index: pd.DatetimeIndex,
             caf_dir: Path, vwc_range=(0.0, 0.6), drop_negative_t=True,
             flag_kinds=("spikes", "breaks")) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the CAF quality-control rules (Dorigo et al., 2013 as implemented by the data authors).

    * VWC outside ``vwc_range`` -> missing (flag 'C' in the CAF QC files)
    * T < 0 degC -> missing (flag 'D')
    * hourly spike / break flags from the QC/Flags folder -> missing
    The 'flats' (plateau) flags are not applied: they mark 24-h windows whose variance is below
    5e-4 (m3/m3)^2, which is the normal behaviour of deep-profile sensors (see counts in ``report``).
    """
    vw, tt = vw.copy(), tt.copy()
    report: dict = {}
    before = int(np.isfinite(vw).sum())
    bad = (vw < vwc_range[0]) | (vw > vwc_range[1])
    report["vwc_out_of_range"] = int(bad.sum())
    vw[bad] = np.nan
    if drop_negative_t:
        neg = tt < 0
        report["t_negative"] = int(neg.sum())
        tt[neg] = np.nan
    loc_idx = {l: i for i, l in enumerate(locs)}
    t_pos = pd.Series(np.arange(len(index)), index=index)
    flag_dir = Path(caf_dir) / "QC" / "Flags"
    for kind in flag_kinds:
        n_kind = 0
        for k, d in enumerate(DEPTHS):
            f = flag_dir / f"{kind}VW_{d}.txt"
            if not f.exists():
                continue
            fl = _read_flag_file(f)
            if fl.empty:
                continue
            ts = pd.to_datetime(fl["Date"] + " " + fl["Time"], format="%m/%d/%Y %H:%M")
            ti = t_pos.reindex(ts).to_numpy()
            li = fl["Location"].map(loc_idx).to_numpy()
            ok = np.isfinite(ti) & pd.notna(li)
            ti, li = ti[ok].astype(int), li[ok].astype(int)
            n_kind += int(np.isfinite(vw[ti, li, k]).sum())
            vw[ti, li, k] = np.nan
        report[f"vwc_{kind}"] = n_kind
    # how many readings the (not applied) plateau rule would have removed, for transparency
    n_flat, n_flat_total = 0, 0
    for k, d in enumerate(DEPTHS):
        f = flag_dir / f"flatsVW_{d}.txt"
        if f.exists():
            n = sum(1 for _ in open(f, "r", encoding="utf-8", errors="ignore")) - 1
            n_flat_total += n
    report["flats_flagged_all_years_not_applied"] = n_flat_total
    report["vwc_valid_before_qc"] = before
    report["vwc_valid_after_qc"] = int(np.isfinite(vw).sum())
    return vw, tt, report


def read_table(path: Path) -> pd.DataFrame:
    """Read the small CAF text tables, which use classic-Mac (CR) or Unix line endings."""
    raw = Path(path).read_bytes().decode("utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    from io import StringIO
    return pd.read_csv(StringIO(raw), sep="\t")


def sensor_coordinates(caf_dir: Path, locs: list[str]) -> pd.DataFrame:
    sf = shapefile.Reader(str(Path(caf_dir) / "CAF_sensors"))
    fields = [f[0] for f in sf.fields[1:]]
    rows = []
    for sr in sf.shapeRecords():
        rec = dict(zip(fields, sr.record))
        x, y = sr.shape.points[0]
        rows.append({"Location": rec["Location"], "easting": x, "northing": y})
    df = pd.DataFrame(rows).set_index("Location").loc[locs]
    return df


class Raster:
    """Minimal reader for the CAF GeoTIFFs (north-up, NAD83 / UTM 11N, 10 m cells)."""

    def __init__(self, path: Path):
        with tifffile.TiffFile(path) as t:
            p = t.pages[0]
            self.a = p.asarray().astype(np.float64)
            tags = {k: v.value for k, v in p.tags.items()}
        sx, sy = tags[33550][:2]
        tp = tags[33922]
        self.x0, self.y0, self.sx, self.sy = tp[3], tp[4], sx, sy
        nod = tags.get(42113)
        if nod is not None:
            try:
                self.a[self.a == float(nod)] = np.nan
            except ValueError:
                pass

    def rc(self, x: float, y: float) -> tuple[int, int]:
        c = int((x - self.x0) / self.sx)
        r = int((self.y0 - y) / self.sy)
        return r, c

    def sample(self, x: float, y: float) -> float:
        r, c = self.rc(x, y)
        if 0 <= r < self.a.shape[0] and 0 <= c < self.a.shape[1]:
            return float(self.a[r, c])
        return float("nan")

    def window_mean(self, x: float, y: float, radius_cells: int) -> float:
        r, c = self.rc(x, y)
        w = self.a[max(r - radius_cells, 0): r + radius_cells + 1, max(c - radius_cells, 0): c + radius_cells + 1]
        return float(np.nanmean(w))


def terrain_features(dem: Raster, x: float, y: float) -> dict:
    """Elevation, slope (deg), aspect (sin/cos) and topographic position index (TPI, 50 m)."""
    r, c = dem.rc(x, y)
    a = dem.a
    z = a[r, c]
    win = a[r - 1: r + 2, c - 1: c + 2]
    # Horn (1981) finite differences on a 3x3 window
    dzdx = ((win[0, 2] + 2 * win[1, 2] + win[2, 2]) - (win[0, 0] + 2 * win[1, 0] + win[2, 0])) / (8 * dem.sx)
    dzdy = ((win[2, 0] + 2 * win[2, 1] + win[2, 2]) - (win[0, 0] + 2 * win[0, 1] + win[0, 2])) / (8 * dem.sy)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.arctan2(dzdx, dzdy)  # radians; downslope direction convention is irrelevant for sin/cos use
    tpi = z - dem.window_mean(x, y, 5)
    return {"elev": z, "slope": slope, "aspect_sin": np.sin(aspect), "aspect_cos": np.cos(aspect), "tpi50": tpi}


def static_features(caf_dir: Path, locs: list[str]) -> pd.DataFrame:
    """Per-logger static covariates from the CAF auxiliary data."""
    caf_dir = Path(caf_dir)
    co = sensor_coordinates(caf_dir, locs)
    dem = Raster(caf_dir / "CAF_DEM.tif")
    eca_s = Raster(caf_dir / "CAF_Spring_ECa.tif")
    eca_f = Raster(caf_dir / "CAF_Fall_ECa.tif")
    bt = {d: Raster(caf_dir / f"CAF_Bt_{d}cm.tif") for d in DEPTHS}
    ps = read_table(caf_dir / "CAF_ParticleSize.txt")
    bd = read_table(caf_dir / "CAF_BulkDensity.txt")
    rows = []
    for loc in locs:
        x, y = co.loc[loc, "easting"], co.loc[loc, "northing"]
        row = {"Location": loc, "easting": x, "northing": y}
        row.update(terrain_features(dem, x, y))
        row["eca_spring"] = eca_s.sample(x, y)
        row["eca_fall"] = eca_f.sample(x, y)
        for d in DEPTHS:
            row[f"bt_{d}"] = bt[d].sample(x, y)
            p = ps[(ps["Location"] == loc) & (ps["Depth"] == d)]
            row[f"sand_{d}"] = float(p["Sand"].iloc[0]) if len(p) else np.nan
            row[f"clay_{d}"] = float(p["Clay"].iloc[0]) if len(p) else np.nan
            b = bd[(bd["Location"] == loc) & (bd["Depth"] == d)]
            row[f"bd_{d}"] = float(b["BulkDensity"].iloc[0]) if len(b) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Location")
    return df


# Functional crop groups (water-use timing). Spring canola (2015-2016 only) is grouped with the
# spring broadleaf crops so that no group is absent from the training period (2011-2013).
CROP_GROUPS = {
    "WW": "winter", "WB": "winter", "WT": "winter", "WC": "winter", "WL": "winter", "WP": "winter",
    "SW": "spring_cereal", "SB": "spring_cereal",
    "SL": "spring_broadleaf", "SP": "spring_broadleaf", "SC": "spring_broadleaf",
    "Fallow": "spring_broadleaf",
}
CROP_GROUP_NAMES = ["winter", "spring_cereal", "spring_broadleaf"]


def crop_assignment(caf_dir: Path, locs: list[str]) -> pd.DataFrame:
    """Crop code per logger and harvest year (2007-2016).

    Loggers are located in field strips with point-in-polygon on CAF_strips.shp; crop codes are taken
    from CAF_CropID.txt (the documented table; the shapefile attributes differ for a few 2015-2016 cells).
    """
    caf_dir = Path(caf_dir)
    sf = shapefile.Reader(str(caf_dir / "CAF_strips"), encoding="cp1252")
    fields = [f[0] for f in sf.fields[1:]]
    polys = []
    for sr in sf.shapeRecords():
        rec = dict(zip(fields, sr.record))
        polys.append((str(rec["Field"]).strip(), int(rec["Strip"]), sr.shape))
    tab = read_table(caf_dir / "CAF_CropID.txt")
    tab = tab[pd.to_numeric(tab["Strip"], errors="coerce").notna()].copy()
    tab["Strip"] = tab["Strip"].astype(int)
    tab["Field"] = tab["Field"].astype(str).str.strip()
    co = sensor_coordinates(caf_dir, locs)

    def inside(pt, shape):
        # ray casting over every ring of the polygon
        x, y = pt
        parts = list(shape.parts) + [len(shape.points)]
        hit = False
        for a, b in zip(parts[:-1], parts[1:]):
            ring = shape.points[a:b]
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
                if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                    hit = not hit
        return hit

    out = []
    for loc in locs:
        pt = (co.loc[loc, "easting"], co.loc[loc, "northing"])
        cand = [(f, s) for f, s, shp in polys if inside(pt, shp)]
        if not cand:  # nearest strip centroid as fallback
            dists = []
            for f, s, shp in polys:
                px, py = np.mean(np.array(shp.points), axis=0)
                dists.append(((px - pt[0]) ** 2 + (py - pt[1]) ** 2, f, s))
            _, f, s = min(dists)
            cand = [(f, s)]
        f, s = cand[0]
        row = tab[(tab["Field"] == f) & (tab["Strip"] == s)]
        rec = {"Location": loc, "field": f, "strip": s, "located": bool(len(cand))}
        for y in range(2007, 2017):
            code = str(row[f"Y{y}"].iloc[0]).strip().rstrip("*") if len(row) else "NA"
            rec[y] = code
        out.append(rec)
    df = pd.DataFrame(out).set_index("Location")
    # footnote of CAF_CropID.txt: location 217 was cropped in spring canola in 2016
    if "CAF217" in df.index:
        df.loc["CAF217", 2016] = "SC"
    return df
