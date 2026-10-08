"""Short-horizon forecasting of co-tenant CPU load (residual host capacity) from Bitbrains traces.

A small GRU reads the last ``lookback`` 5-min CPU-usage samples of a VM (plus time of day) and predicts
the mean usage over the next ``horizon`` samples as a residual correction of the last sample
(zero-initialised, so training starts from persistence). Trained on one set of VMs, tested on held-out VMs.
Windows are drawn by index so that memory stays small.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LoadGRU(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(3, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):                 # x [B, L, 3]; returns mean usage of the next horizon (0..1)
        h, _ = self.gru(x)
        return (x[:, -1, 0] + self.out(h[:, -1])[:, 0]).clamp(0.0, 1.0)


def _tod_feats(t0: int, T: int) -> np.ndarray:
    tod = ((t0 + np.arange(T) * 300) % 86400) / 86400.0
    return np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)], -1).astype(np.float32)


def gather(use: np.ndarray, feats: np.ndarray, v: np.ndarray, s: np.ndarray, lookback: int, horizon: int):
    """Windows ending just before step s of VMs v: X [n, lookback, 3], y [n]."""
    off = np.arange(-lookback, 0)
    u = use[v[:, None], s[:, None] + off]                         # [n, L]
    f = feats[s[:, None] + off]                                    # [n, L, 2]
    X = np.concatenate([u[..., None], f], -1).astype(np.float32)
    y = use[v[:, None], s[:, None] + np.arange(horizon)].mean(1).astype(np.float32)
    return X, y


def train(use_tr: np.ndarray, t0: int, lookback=12, horizon=3, epochs=3, seed=0, n_samples=400_000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    V, T = use_tr.shape
    feats = _tod_feats(t0, T)
    v = rng.integers(0, V, n_samples)
    s = rng.integers(lookback, T - horizon, n_samples)
    X, y = gather(use_tr, feats, v, s, lookback, horizon)
    model = LoadGRU()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for ep in range(epochs):
        perm = torch.randperm(len(yt))
        for a in range(0, len(yt), 1024):
            idx = perm[a:a + 1024]
            loss = (model(Xt[idx]) - yt[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def evaluate(model, use_te: np.ndarray, t0: int, lookback=12, horizon=3, alpha=0.5, chunk=50_000) -> dict:
    """MAE (percentage points of CPU) of the GRU and of persistence, 1-h mean and EWMA baselines on all
    windows of the held-out VMs."""
    V, T = use_te.shape
    feats = _tod_feats(t0, T)
    vv, ss = np.meshgrid(np.arange(V), np.arange(lookback, T - horizon), indexing="ij")
    vv, ss = vv.ravel(), ss.ravel()
    err = {"gru": 0.0, "last": 0.0, "mean1h": 0.0, "ewma": 0.0}
    for a in range(0, len(vv), chunk):
        X, y = gather(use_te, feats, vv[a:a + chunk], ss[a:a + chunk], lookback, horizon)
        with torch.no_grad():
            pred = model(torch.from_numpy(X)).numpy()
        ew = X[:, 0, 0].copy()
        for k in range(1, lookback):
            ew = alpha * X[:, k, 0] + (1 - alpha) * ew
        err["gru"] += np.abs(pred - y).sum()
        err["last"] += np.abs(X[:, -1, 0] - y).sum()
        err["mean1h"] += np.abs(X[:, :, 0].mean(1) - y).sum()
        err["ewma"] += np.abs(ew - y).sum()
    n = len(vv)
    return {f"mae_{k}": float(v / n * 100) for k, v in err.items()} | {"n": int(n)}
