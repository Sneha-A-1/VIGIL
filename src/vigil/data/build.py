"""Build the processed, analysis-ready dataset from the raw downloads.

    python -m vigil.data.build            # uses configs/data.yaml

Produces ``data/processed/caf_processed.npz`` and ``data/processed/build_report.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vigil.config import data_root, load_config, repo_path
from vigil.data import caf, weather


def wetting_events(vwc: np.ndarray, delta: float, window: int, min_valid: int = 12) -> np.ndarray:
    """Event state e[t,i,d] = 1 if x(t) - min(x[t-window..t]) >= delta (NaN when undefined)."""
    T, N, D = vwc.shape
    flat = pd.DataFrame(vwc.reshape(T, N * D))
    rise = flat - flat.rolling(window + 1, min_periods=min_valid).min()
    ev = (rise >= delta).astype(np.float32).where(rise.notna())
    return ev.to_numpy(np.float32).reshape(T, N, D)


def knn_gaussian_adjacency(xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    N = len(xy)
    nn = np.argsort(d, axis=1)[:, 1:k + 1]
    sigma = d[np.arange(N)[:, None], nn].std()
    w = np.zeros((N, N))
    for i in range(N):
        w[i, nn[i]] = np.exp(-(d[i, nn[i]] ** 2) / (sigma ** 2))
    w = np.maximum(w, w.T)  # symmetrise
    return w.astype(np.float32), d.astype(np.float32)


def crop_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Harvest (crop) year: Sep-Dec belong to the following harvest year."""
    return np.where(index.month >= 9, index.year + 1, index.year)


def main(cfg_name: str = "data") -> None:
    cfg = load_config(cfg_name)
    root = data_root()
    caf_dir = root / cfg["paths"]["caf_dir"]
    out_dir = repo_path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    st = cfg["study"]
    index = pd.date_range(st["start"], st["end"], freq="h")
    report: dict = {"hours": len(index)}

    # ---- soil sensors ---------------------------------------------------------------
    vwc, tmp, locs, info = caf.load_hourly(caf_dir, index)
    report["caf_duplicates"] = info["duplicates"]
    vwc, tmp, qc = caf.apply_qc(vwc, tmp, locs, index, caf_dir,
                                vwc_range=tuple(cfg["qc"]["vwc_range"]),
                                drop_negative_t=cfg["qc"]["drop_negative_temperature"],
                                flag_kinds=tuple(cfg["qc"]["use_flag_files"]))
    report["qc"] = qc

    # ---- static covariates ----------------------------------------------------------
    stat = caf.static_features(caf_dir, locs)
    crops = caf.crop_assignment(caf_dir, locs)
    report["static_missing"] = {c: int(v) for c, v in stat.isna().sum().items() if v}
    stat = stat.fillna(stat.mean())
    cy = crop_year(index)
    groups = caf.CROP_GROUP_NAMES
    crop_idx = np.zeros((len(index), len(locs)), dtype=np.int8)
    for j, loc in enumerate(locs):
        for y in np.unique(cy):
            code = crops.loc[loc, int(y)] if int(y) in crops.columns else "NA"
            g = caf.CROP_GROUPS.get(code, "spring_broadleaf")
            crop_idx[cy == y, j] = groups.index(g)
    report["crop_codes"] = crops.drop(columns=["located"]).reset_index().astype(str).to_dict(orient="records")

    # ---- weather --------------------------------------------------------------------
    pw = weather.load_power(root / cfg["paths"]["power_csv"])
    sc = weather.load_scan(root / cfg["paths"]["scan_csv"])
    wcfg = cfg["weather"]
    f_power = weather.derive_features(pw, index, wcfg["precip_windows_h"], wcfg["radiation_window_h"])
    scan_cov = (index >= sc.index.min()) & (index <= sc.index.max())
    f_scan = weather.derive_features(sc, index, wcfg["precip_windows_h"], wcfg["radiation_window_h"])
    f_scan.loc[~scan_cov] = np.nan
    tfeat = weather.time_features(index)
    report["scan_coverage"] = [str(sc.index.min()), str(sc.index.max())]

    # ---- events and graph -----------------------------------------------------------
    ecfg = cfg["events"]
    events = wetting_events(vwc, ecfg["wetting_delta"], ecfg["wetting_window_h"])
    xy = stat[["easting", "northing"]].to_numpy()
    adj, dist = knn_gaussian_adjacency(xy, cfg["graph"]["knn"])

    # ---- splits and normalisation (training period only) ----------------------------
    tr_end = np.searchsorted(index, pd.Timestamp(st["train_end"]), side="right")
    va_end = np.searchsorted(index, pd.Timestamp(st["val_end"]), side="right")
    split = {"train": [0, int(tr_end)], "val": [int(tr_end), int(va_end)], "test": [int(va_end), len(index)]}
    tr = slice(0, tr_end)
    norm = {
        "vwc_mean": np.nanmean(vwc[tr], axis=(0, 1)), "vwc_std": np.nanstd(vwc[tr], axis=(0, 1)),
        "tmp_mean": np.nanmean(tmp[tr], axis=(0, 1)), "tmp_std": np.nanstd(tmp[tr], axis=(0, 1)),
        "w_mean": f_power.iloc[tr].mean().to_numpy(), "w_std": f_power.iloc[tr].std().to_numpy(),
        "s_mean": stat.mean().to_numpy(), "s_std": stat.std().replace(0, 1).to_numpy(),
    }
    for name, (a, b) in split.items():
        report[f"{name}_vwc_availability"] = float(np.isfinite(vwc[a:b]).mean())
        report[f"{name}_event_rate"] = float(np.nanmean(events[a:b]))
        e = np.nan_to_num(events[a:b])
        report[f"{name}_event_onsets"] = int(((e[1:] == 1) & (e[:-1] == 0)).sum())

    np.savez_compressed(
        out_dir / "caf_processed.npz",
        index=index.values.astype("datetime64[s]").astype(np.int64), locs=np.array(locs, dtype="U"), depths=np.array(caf.DEPTHS),
        vwc=vwc, tmp=tmp, events=events,
        w_power=f_power.to_numpy(np.float32), w_scan=f_scan.to_numpy(np.float32),
        w_names=np.array(list(f_power.columns), dtype="U"), tfeat=tfeat.to_numpy(np.float32),
        static=stat.to_numpy(np.float32), s_names=np.array(list(stat.columns), dtype="U"),
        crop_idx=crop_idx, crop_names=np.array(groups, dtype="U"),
        adj=adj, dist=dist, xy=xy.astype(np.float64),
        split=np.array([split["train"], split["val"], split["test"]]),
        **{k: np.asarray(v, dtype=np.float32) for k, v in norm.items()},
    )
    stat.to_csv(out_dir / "static_features.csv")
    crops.to_csv(out_dir / "crop_assignment.csv")
    with open(out_dir / "build_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "crop_codes"}, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data")
    main(ap.parse_args().config)
