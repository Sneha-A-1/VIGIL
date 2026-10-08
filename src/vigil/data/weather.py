"""Weather covariates: NASA POWER (gridded, MERRA-2/CERES) and NRCS SCAN 2198 (on-farm station)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_VARS = ["temp_c", "rh", "prcp_mm", "wind_ms", "srad_wm2"]


def load_power(path: Path) -> pd.DataFrame:
    """NASA POWER hourly point CSV (LST) -> SI-unit frame indexed by hour."""
    lines = Path(path).read_text().splitlines()
    start = lines.index("-END HEADER-") + 1
    p = pd.read_csv(path, skiprows=start)
    p.index = pd.to_datetime(dict(year=p.YEAR, month=p.MO, day=p.DY, hour=p.HR))
    p = p.drop(columns=["YEAR", "MO", "DY", "HR"]).replace(-999, np.nan)
    out = pd.DataFrame({
        "temp_c": p["T2M"],
        "rh": p["RH2M"],
        "prcp_mm": p["PRECTOTCORR"],
        "wind_ms": p["WS2M"],
        "srad_wm2": p["ALLSKY_SFC_SW_DWN"] * 1e6 / 3600.0,  # MJ m-2 h-1 -> W m-2
    }, index=p.index)
    return out


def load_scan(path: Path) -> pd.DataFrame:
    """NRCS SCAN report-generator CSV (US units) -> SI-unit frame indexed by hour."""
    s = pd.read_csv(path, comment="#")
    idx = pd.to_datetime(s["Date"], format="%Y-%m-%d %H:%M")
    out = pd.DataFrame({
        "temp_c": (s["Air Temperature Observed (degF)"].to_numpy() - 32) * 5 / 9,
        "rh": s["Relative Humidity (pct)"].to_numpy(),
        "prcp_mm": s["Precipitation Increment (in)"].to_numpy() * 25.4,
        "wind_ms": s["Wind Speed Average (mph)"].to_numpy() * 0.44704,
        "srad_wm2": s["Solar Radiation Average (watt/m2)"].to_numpy(),
    }, index=idx)
    out = out[~out.index.duplicated()]
    return out


def vapour_pressure_deficit(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """VPD in kPa using the Tetens saturation vapour pressure (FAO-56 Eq. 11)."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return es * (1.0 - np.clip(rh, 0, 100) / 100.0)


def derive_features(w: pd.DataFrame, index: pd.DatetimeIndex, precip_windows=(6, 24, 72, 168),
                    rad_window: int = 24) -> pd.DataFrame:
    """Causal weather features on ``index``: current values, VPD, trailing precipitation sums
    (log1p) and trailing radiation mean. Only past and present hours are used."""
    w = w.reindex(index)
    # short gaps in station data are bridged by time interpolation (precipitation gaps -> 0)
    w["prcp_mm"] = w["prcp_mm"].fillna(0.0).clip(lower=0)
    for c in ["temp_c", "rh", "wind_ms", "srad_wm2"]:
        w[c] = w[c].interpolate(limit_direction="both")
    f = pd.DataFrame(index=index)
    f["temp_c"] = w["temp_c"]
    f["rh"] = w["rh"]
    f["vpd_kpa"] = vapour_pressure_deficit(w["temp_c"].to_numpy(), w["rh"].to_numpy())
    f["wind_ms"] = w["wind_ms"]
    f["srad_wm2"] = w["srad_wm2"]
    f["prcp_log1p"] = np.log1p(w["prcp_mm"])
    for h in precip_windows:
        f[f"prcp_sum{h}h_log1p"] = np.log1p(w["prcp_mm"].rolling(h, min_periods=1).sum())
    f[f"srad_mean{rad_window}h"] = w["srad_wm2"].rolling(rad_window, min_periods=1).mean()
    f["temp_mean24h"] = w["temp_c"].rolling(24, min_periods=1).mean()
    return f


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hod = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()
    return pd.DataFrame({
        "hod_sin": np.sin(2 * np.pi * hod / 24), "hod_cos": np.cos(2 * np.pi * hod / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)
