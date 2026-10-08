"""Evaluation metrics: field-state accuracy, probabilistic calibration and urgent-event detection."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score


def state_metrics(mu: np.ndarray, x: np.ndarray, polled: np.ndarray, std: np.ndarray,
                  index: pd.DatetimeIndex, sigma: np.ndarray | None = None) -> tuple[dict, pd.Series]:
    """mu, x: [S,N,D] normalised; polled: [S,N,D] bool (channel observed this hour); std: [D]."""
    err = (mu - x) * std                         # m3/m3
    ok = np.isfinite(err)
    unobs = ok & ~polled
    e2 = np.where(ok, err ** 2, 0.0)
    res = {
        "rmse": float(np.sqrt(e2.sum() / ok.sum())),
        "mae": float(np.abs(np.where(ok, err, 0)).sum() / ok.sum()),
        "rmse_unobs": float(np.sqrt(np.where(unobs, err ** 2, 0).sum() / max(unobs.sum(), 1))),
        "frac_polled_channels": float((ok & polled).sum() / ok.sum()),
    }
    for d in range(x.shape[2]):
        okd = ok[:, :, d]
        res[f"rmse_d{d}"] = float(np.sqrt(e2[:, :, d].sum() / max(okd.sum(), 1)))
    # daily RMSE series (for paired significance tests)
    day = index.normalize()
    num = pd.Series(e2.sum(axis=(1, 2)), index=index).groupby(day).sum()
    den = pd.Series(ok.sum(axis=(1, 2)), index=index).groupby(day).sum()
    daily = np.sqrt(num / den.clip(lower=1))
    if sigma is not None:
        s = np.where(unobs, sigma, np.nan)
        z = (mu - x) / s
        crps = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
        crps = crps * std
        res["crps_unobs"] = float(np.nanmean(np.where(unobs, crps, np.nan)))
        res["cov90_unobs"] = float(np.nanmean(np.where(unobs, np.abs(z) <= 1.6449, np.nan)))
        res["cov50_unobs"] = float(np.nanmean(np.where(unobs, np.abs(z) <= 0.6745, np.nan)))
    return res, daily


def rule_event_score(mu: np.ndarray, delta_norm: np.ndarray, window: int = 24) -> np.ndarray:
    """Continuous event score from point estimates: rise above the trailing 24-h minimum divided by
    the event threshold (>= 1 means 'event'). mu [S,N,D] normalised; delta_norm [D]."""
    S, N, D = mu.shape
    flat = pd.DataFrame(mu.reshape(S, N * D))
    rise = (flat - flat.rolling(window + 1, min_periods=1).min()).to_numpy().reshape(S, N, D)
    return rise / delta_norm


def _episodes(b: np.ndarray):
    """Start/end (inclusive) indices of runs of True in a 1-D bool array."""
    if not b.any():
        return np.empty(0, int), np.empty(0, int)
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0] - 1


def event_metrics(score: np.ndarray, ev: np.ndarray, tau: float, early: int = 6, tol: int = 12) -> tuple[dict, np.ndarray]:
    """score/ev: [S,N,D]; alerts = score >= tau. Returns metrics and per-onset delays (inf = missed)."""
    valid = np.isfinite(ev)
    y = ev[valid] > 0.5
    s = score[valid]
    res = {"auprc": float(average_precision_score(y, s)) if y.any() else float("nan"),
           "event_rate": float(y.mean())}
    a = s >= tau
    tp, fp, fn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum())
    res["f1_hourly"] = 2 * tp / max(2 * tp + fp + fn, 1)
    res["precision_hourly"] = tp / max(tp + fp, 1)
    res["recall_hourly"] = tp / max(tp + fn, 1)
    S, N, D = ev.shape
    delays, n_alert_ep, n_false = [], 0, 0
    for i in range(N):
        for d in range(D):
            e = np.nan_to_num(ev[:, i, d]) > 0.5
            al = score[:, i, d] >= tau
            on, off = _episodes(e)
            prev_ok = np.concatenate([[False], np.isfinite(ev[:-1, i, d]) & (np.nan_to_num(ev[:-1, i, d]) < 0.5)])
            for t0, t1 in zip(on, off):
                if not prev_ok[t0]:
                    continue          # episode start not observed (gap) -> not a scorable onset
                lo = max(0, t0 - early)
                hit = np.where(al[lo:t1 + 1])[0]
                delays.append(max(0, lo + hit[0] - t0) if len(hit) else np.inf)
            a_on, a_off = _episodes(al)
            n_alert_ep += len(a_on)
            for u0, u1 in zip(a_on, a_off):
                overlap = ((on - tol) <= u1) & ((off + tol) >= u0)
                if not overlap.any():
                    n_false += 1
    delays = np.array(delays, float)
    det = np.isfinite(delays)
    res.update({
        "n_onsets": int(len(delays)),
        "onset_recall": float(det.mean()) if len(delays) else float("nan"),
        "recall_6h": float((delays <= 6).mean()) if len(delays) else float("nan"),
        "recall_24h": float((delays <= 24).mean()) if len(delays) else float("nan"),
        "median_delay_h": float(np.median(delays[det])) if det.any() else float("nan"),
        "mean_delay_h": float(np.mean(delays[det])) if det.any() else float("nan"),
        "alert_episodes": int(n_alert_ep),
        "alert_precision": float(1 - n_false / n_alert_ep) if n_alert_ep else float("nan"),
        "false_alerts_per_day": float(n_false / (S / 24.0)),
    })
    return res, delays


def best_threshold(score: np.ndarray, ev: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Threshold maximising the hourly F1 (used on the validation period only)."""
    valid = np.isfinite(ev)
    y = ev[valid] > 0.5
    s = score[valid]
    if grid is None:
        qs = np.unique(np.quantile(s, np.linspace(0.80, 0.9995, 120)))
        grid = qs
    best, best_f1 = float(grid[0]), -1.0
    for t in grid:
        a = s >= t
        tp = (a & y).sum()
        f1 = 2 * tp / max(a.sum() + y.sum(), 1)
        if f1 > best_f1:
            best, best_f1 = float(t), float(f1)
    return best
