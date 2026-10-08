"""Analysis-ready tensors built from ``data/processed/caf_processed.npz``."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from vigil.config import repo_path

N_CROP = 3


@dataclass
class CAFData:
    """Normalised tensors (float32, CPU).

    Shapes: T hours, N loggers, D depths, Fw weather features, Fs static features.
    ``x`` / ``tmp`` are standardised per depth with training-period statistics and contain NaN
    where the sensor produced no valid reading. ``ev`` holds wetting-event states (NaN = undefined).
    """

    index: pd.DatetimeIndex
    locs: list
    depths: list
    x: torch.Tensor          # [T,N,D] normalised VWC
    tmp: torch.Tensor        # [T,N,D] normalised soil temperature
    ev: torch.Tensor         # [T,N,D] event state {0,1,NaN}
    w: torch.Tensor          # [T,Fw] normalised weather + time features (active source)
    prcp_mm: torch.Tensor    # [T] hourly precipitation (mm) of the active source
    vpd_kpa: torch.Tensor    # [T] vapour-pressure deficit (kPa) of the active source
    s: torch.Tensor          # [N,Fs] normalised static features
    crop: torch.Tensor       # [T,N,3] one-hot crop group
    adj: torch.Tensor        # [N,N] Gaussian kNN adjacency
    xy: np.ndarray           # [N,2] UTM coordinates (m)
    split: dict
    u: torch.Tensor          # [T,N,D] look-ahead target: mean normalised VWC over the next L hours
    lookahead: int
    delta_norm: np.ndarray   # [D] event threshold (0.02 m3/m3) in normalised units
    vwc_mean: np.ndarray
    vwc_std: np.ndarray
    w_names: list
    s_names: list
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.x.shape[0]

    @property
    def N(self) -> int:
        return self.x.shape[1]

    @property
    def D(self) -> int:
        return self.x.shape[2]

    def denorm(self, x: np.ndarray | torch.Tensor):
        """Normalised VWC -> m3/m3 (broadcast over the last axis = depth)."""
        if isinstance(x, torch.Tensor):
            return x * torch.as_tensor(self.vwc_std) + torch.as_tensor(self.vwc_mean)
        return x * self.vwc_std + self.vwc_mean

    def raw_vwc(self) -> np.ndarray:
        return self.denorm(self.x.numpy())


def lookahead_target(x: np.ndarray, L: int, boundaries: list[int]) -> np.ndarray:
    """u[t] = mean of x[t+1 .. t+L] (needs >= L/2 valid values); NaN within L hours of a split end,
    so that no target crosses from one split into the next."""
    T = x.shape[0]
    valid = np.isfinite(x)
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(np.nan_to_num(x), 0)], 0)
    cv = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(valid, 0)], 0)
    u = np.full(x.shape, np.nan, dtype=np.float32)
    t = np.arange(T - L)
    tot = cs[t + L + 1] - cs[t + 1]
    cnt = cv[t + L + 1] - cv[t + 1]
    u[t] = np.where(cnt >= L / 2, tot / np.maximum(cnt, 1), np.nan)
    for b in boundaries:
        u[max(0, b - L):b] = np.nan
    return u


def load(weather_source: str = "power", path: str | None = None, lookahead: int = 24) -> CAFData:
    """Load the processed dataset. ``weather_source`` in {'power', 'scan'}.

    For 'scan' the on-farm station features replace the NASA POWER ones where available (normalised
    with the POWER training statistics so that a model trained on POWER can be evaluated on SCAN);
    hours without SCAN data fall back to POWER.
    """
    z = np.load(path or repo_path("data", "processed", "caf_processed.npz"), allow_pickle=False)
    index = pd.to_datetime(z["index"], unit="s")
    w_names = [str(s) for s in z["w_names"]]
    wp = z["w_power"].astype(np.float64)
    if weather_source == "scan":
        ws = z["w_scan"].astype(np.float64)
        ok = np.isfinite(ws).all(axis=1)
        w_raw = np.where(ok[:, None], ws, wp)
    elif weather_source == "power":
        w_raw = wp
        ok = np.ones(len(index), bool)
    else:
        raise ValueError(weather_source)
    w = (w_raw - z["w_mean"]) / z["w_std"]
    w = np.concatenate([w, z["tfeat"]], axis=1)
    prcp = np.expm1(w_raw[:, w_names.index("prcp_log1p")])
    vpd = w_raw[:, w_names.index("vpd_kpa")]

    vm, vs = z["vwc_mean"], z["vwc_std"]
    x = (z["vwc"] - vm) / vs
    tmp = (z["tmp"] - z["tmp_mean"]) / z["tmp_std"]
    s = (z["static"] - z["s_mean"]) / z["s_std"]
    crop = np.eye(N_CROP, dtype=np.float32)[z["crop_idx"]]
    split = {k: tuple(int(v) for v in z["split"][i]) for i, k in enumerate(["train", "val", "test"])}
    u = lookahead_target(x.astype(np.float32), lookahead, [split["train"][1], split["val"][1], split["test"][1]])
    return CAFData(
        index=index, locs=[str(l) for l in z["locs"]], depths=[int(d) for d in z["depths"]],
        x=torch.from_numpy(x.astype(np.float32)), tmp=torch.from_numpy(tmp.astype(np.float32)),
        ev=torch.from_numpy(z["events"].astype(np.float32)),
        w=torch.from_numpy(w.astype(np.float32)),
        prcp_mm=torch.from_numpy(prcp.astype(np.float32)), vpd_kpa=torch.from_numpy(vpd.astype(np.float32)),
        s=torch.from_numpy(s.astype(np.float32)), crop=torch.from_numpy(crop),
        adj=torch.from_numpy(z["adj"]), xy=z["xy"], split=split,
        u=torch.from_numpy(u), lookahead=lookahead, delta_norm=(0.02 / vs).astype(np.float32),
        vwc_mean=vm.astype(np.float32), vwc_std=vs.astype(np.float32),
        w_names=w_names + ["hod_sin", "hod_cos", "doy_sin", "doy_cos"],
        s_names=[str(n) for n in z["s_names"]],
        meta={"weather_source": weather_source, "scan_hours": int(ok.sum())},
    )
