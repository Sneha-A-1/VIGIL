"""Gradient-boosted trees (LightGBM) estimator on the same per-node inputs as the neural models.

One row per (hour, logger, depth): node inputs (context + memory), a one-hot depth indicator and the
channel's own last value / age / temperature. Target: change since the last retrieved value.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import torch

D_DEFAULT = 5


def rows(inp: torch.Tensor, D: int = D_DEFAULT) -> np.ndarray:
    """inp [..., N, F] -> feature matrix [(...*N*D), F + D + 3]."""
    f = inp.reshape(-1, inp.shape[-1]).numpy()
    n = f.shape[0]
    # memory block layout (see sim.retrieval.memory_features): last_x | age | last_tmp | obs | ...
    ctx_end = f.shape[1] - (4 * D + 3)
    own = []
    for d in range(D):
        own.append(np.stack([f[:, ctx_end + d], f[:, ctx_end + D + d], f[:, ctx_end + 2 * D + d]], 1))
    own = np.stack(own, 1)                                   # [n, D, 3]
    onehot = np.broadcast_to(np.eye(D, dtype=np.float32), (n, D, D))
    base = np.broadcast_to(f[:, None, :], (n, D, f.shape[1]))
    return np.concatenate([base, onehot, own], -1).reshape(n * D, -1).astype(np.float32)


def fit(train_inp, train_last, train_x, train_obs, val_inp, val_last, val_x, val_obs, seed: int,
        max_rows: int = 1_500_000, params: dict | None = None):
    def build(inp, last, x, obs, cap, rng):
        X = rows(inp)
        y = (x - last).reshape(-1).numpy()
        keep = (np.isfinite(y)) & (obs.reshape(-1).numpy() == 0)
        X, y = X[keep], y[keep]
        if len(y) > cap:
            sel = rng.choice(len(y), cap, replace=False)
            X, y = X[sel], y[sel]
        return X, y

    rng = np.random.default_rng(seed)
    Xtr, ytr = build(train_inp, train_last, train_x, train_obs, max_rows, rng)
    Xva, yva = build(val_inp, val_last, val_x, val_obs, max_rows // 5, rng)
    p = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
         "seed": seed, "num_threads": 1, "verbose": -1}
    p.update(params or {})
    dtr, dva = lgb.Dataset(Xtr, ytr), lgb.Dataset(Xva, yva)
    model = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, {"n_train_rows": int(len(ytr)), "n_val_rows": int(len(yva)), "best_iter": int(model.best_iteration)}


def predict(model, inp: torch.Tensor, last_x: torch.Tensor, chunk: int = 2000) -> torch.Tensor:
    """inp [S,N,F], last_x [S,N,D] -> estimates [S,N,D]."""
    S, N, D = last_x.shape
    out = np.empty((S, N, D), dtype=np.float32)
    for a in range(0, S, chunk):
        b = min(S, a + chunk)
        pr = model.predict(rows(inp[a:b]), num_iteration=model.best_iteration)
        out[a:b] = pr.reshape(b - a, N, D)
    return last_x + torch.from_numpy(out)
