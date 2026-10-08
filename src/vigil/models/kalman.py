"""Rain-conditioned spatio-temporal Kalman filter (STKF): the classical model-based counterpart of
the VIGIL predict-update network, with greedy A-optimal (trace-reduction) polling.

State x_t in R^{N*D} (normalised VWC, all loggers and depths), random-walk dynamics
    x_t = x_{t-1} + w_t,  w_t ~ N(0, Q_{r(t)}),  r(t) in {dry, wet}
observed through the polled channels with white noise of covariance R (diagonal).
Q_dry, Q_wet and R are estimated from the training period: for a random walk observed with white
noise, Cov(dx_t) = Q + 2R and Cov(dx_t, dx_{t-1}) = -R (dx_t = y_t - y_{t-1}).
"""
from __future__ import annotations

import numpy as np


def _pairwise_cov(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Covariance between columns of a and b with pairwise-complete observations (zero mean assumed)."""
    b = a if b is None else b
    ma, mb = np.isfinite(a), np.isfinite(b)
    a0, b0 = np.nan_to_num(a), np.nan_to_num(b)
    num = a0.T @ b0
    cnt = ma.T.astype(np.float64) @ mb.astype(np.float64)
    return num / np.maximum(cnt, 1.0)


def _nearest_psd(m: np.ndarray, floor: float) -> np.ndarray:
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    return (v * np.maximum(w, floor)) @ v.T


class RainKF:
    def __init__(self, shrink: float = 0.2, wet_mm_24h: float = 1.0, r_floor: float = 1e-8, r_scale: float = 1.0):
        self.shrink, self.wet_mm_24h, self.r_floor, self.r_scale = shrink, wet_mm_24h, r_floor, r_scale

    # ------------------------------------------------------------------ fitting
    def fit(self, x: np.ndarray, prcp24: np.ndarray) -> "RainKF":
        """x: [T,N,D] normalised truth over the training period (NaN gaps); prcp24: [T] mm."""
        T, N, D = x.shape
        y = x.reshape(T, N * D).astype(np.float64)
        dy = y[1:] - y[:-1]
        wet = prcp24[1:] >= self.wet_mm_24h
        lag = _pairwise_cov(dy[1:], dy[:-1])
        r = np.clip(-np.diag(lag), self.r_floor, None)
        self.R = r
        self.Q = {}
        for name, msk in [("dry", ~wet), ("wet", wet)]:
            c = _pairwise_cov(dy[msk])
            c = c - 2 * np.diag(r)
            c = (1 - self.shrink) * c + self.shrink * np.diag(np.diag(c))
            self.Q[name] = _nearest_psd(c, 1e-6)
        self.R = np.maximum(self.r_scale * r, self.r_floor)   # filter noise (moment estimate x scale)
        self.n = N * D
        self.N, self.D = N, D
        return self

    # ------------------------------------------------------------------ filtering
    def init(self, x0: np.ndarray, p_missing: float = 1.0):
        """x0: [N,D] (NaN = no reading) -> mean and covariance after a full poll."""
        v = x0.reshape(-1).astype(np.float64)
        ok = np.isfinite(v)
        mean = np.where(ok, v, 0.0)
        P = np.diag(np.where(ok, self.R, p_missing))
        return mean, P

    def predict(self, mean, P, wet: bool):
        return mean, P + self.Q["wet" if wet else "dry"]

    def update(self, mean, P, idx: np.ndarray, y: np.ndarray):
        if len(idx) == 0:
            return mean, P
        S = P[np.ix_(idx, idx)] + np.diag(self.R[idx])
        PHt = P[:, idx]
        K = np.linalg.solve(S, PHt.T).T
        mean = mean + K @ (y - mean[idx])
        P = P - K @ PHt.T
        P = 0.5 * (P + P.T)
        return mean, P

    def channels(self, logger: int) -> np.ndarray:
        return np.arange(logger * self.D, (logger + 1) * self.D)

    def greedy_select(self, P, k: int, node_age: np.ndarray, max_age: float,
                      avail: np.ndarray | None = None) -> np.ndarray:
        """Greedy A-optimal polling: repeatedly add the logger whose observation most reduces the
        trace of the posterior covariance (loggers past the freshness guard are taken first).
        ``avail`` [N*D]: estimated probability that a channel returns a valid reading; only channels
        with avail >= 0.5 are assumed to be observed when computing the gain."""
        chosen = []
        due = np.where(node_age >= max_age)[0]
        for i in due[np.argsort(-node_age[due])][:k]:
            chosen.append(int(i))
        live = np.ones(self.n, bool) if avail is None else (avail >= 0.5)
        Pc = P.copy()
        for i in chosen:
            idx = self.channels(i)
            idx = idx[live[idx]]
            if len(idx) == 0:
                continue
            S = Pc[np.ix_(idx, idx)] + np.diag(self.R[idx])
            Kt = np.linalg.solve(S, Pc[idx, :])
            Pc = Pc - Pc[:, idx] @ Kt
        while len(chosen) < k:
            best, best_gain, best_upd = None, -1.0, None
            for i in range(self.N):
                if i in chosen:
                    continue
                idx = self.channels(i)
                idx = idx[live[idx]]
                if len(idx) == 0:
                    if best is None:
                        best, best_gain, best_upd = i, 0.0, None
                    continue
                S = Pc[np.ix_(idx, idx)] + np.diag(self.R[idx])
                A = Pc[:, idx]
                Kt = np.linalg.solve(S, A.T)          # [5, n]
                gain = float(np.einsum("ij,ji->", A, Kt))  # trace(A S^-1 A^T)
                if gain > best_gain:
                    best, best_gain, best_upd = i, gain, (A, Kt)
            chosen.append(best)
            if best_upd is not None:
                A, Kt = best_upd
                Pc = Pc - A @ Kt
        return np.array(chosen, dtype=int)
