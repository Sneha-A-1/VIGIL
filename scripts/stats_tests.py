"""Paired significance tests between retrieval pipelines on the test period.

    python scripts/stats_tests.py   ->  results/processed/stats_tests.csv, stats_events.csv

Field-state accuracy: daily RMSE series (seed-averaged daily MSE) of two pipelines are compared with
the two-sided Wilcoxon signed-rank test and the Diebold-Mariano test on the daily MSE differential
(Newey-West long-run variance, lag 7 days, Harvey-Leybourne-Newbold small-sample correction).
Urgent detection: per-onset detection within 6 h (seed-averaged indicator); the paired difference is tested
with a sign-flip randomisation test that flips whole days (onsets of the same day are dependent because
storms trigger many onsets at once).
p-values are adjusted with Holm's procedure within each family.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy import stats

from vigil.config import repo_path

EVAL = repo_path("results", "raw", "eval")
BUDGETS = [1, 2, 4, 8]
SEEDS = [0, 1, 2, 3, 4]


def files(group: str, est: str, pol: str, k: int, eta: float | None = 0.5) -> list[str]:
    if pol.startswith(("voi", "kfvoi")):
        pat = f"{est}__{pol}_eta{eta:g}_kap*__K{k}__s*.npz"
    else:
        pat = f"{est}__{pol}__K{k}__s*.npz"
    return sorted(f for f in glob.glob(str(EVAL / group / pat)) if "_w" not in f.rsplit("__", 1)[-1])


def daily_mse(fs: list[str]) -> pd.Series | None:
    if not fs:
        return None
    ser = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        ser.append(pd.Series(z["daily_rmse"].astype(float) ** 2, index=z["days"]))
    return pd.concat(ser, axis=1).mean(axis=1)


def det6(fs: list[str]) -> np.ndarray | None:
    if not fs:
        return None
    out = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        out.append(z["delays_head"] if "delays_head" in z.files else z["delays_rule"])
    return np.mean([d <= 6 for d in out], 0)


def onset_days() -> np.ndarray:
    """Calendar day of every scorable onset of the test period, in the order of metrics.event_metrics."""
    from vigil.data.dataset import load
    from vigil.metrics import _episodes
    d = load()
    a, b = d.split["test"]
    ev = d.ev[a:b].numpy()
    idx = d.index[a:b]
    S, N, D = ev.shape
    days = []
    for i in range(N):
        for k in range(D):
            e = np.nan_to_num(ev[:, i, k]) > 0.5
            on, off = _episodes(e)
            prev_ok = np.concatenate([[False], np.isfinite(ev[:-1, i, k]) & (np.nan_to_num(ev[:-1, i, k]) < 0.5)])
            for t0 in on:
                if prev_ok[t0]:
                    days.append(idx[t0].normalize().value)
    return np.array(days)


def cluster_signflip(diff: np.ndarray, clusters: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Two-sided p-value of sum(diff) under random sign flips of whole clusters."""
    u, inv = np.unique(clusters, return_inverse=True)
    tot = np.bincount(inv, weights=diff, minlength=len(u))
    obs = abs(tot.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(u)))
    null = np.abs(signs @ tot)
    return float((1 + (null >= obs - 1e-12).sum()) / (n + 1))


def newey_west(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    T = len(d)
    v = d @ d / T
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        v += 2 * w * (d[l:] @ d[:-l]) / T
    return v


def dm_test(e_a: np.ndarray, e_b: np.ndarray, lag: int = 7) -> tuple[float, float]:
    """Diebold-Mariano on loss differential d = e_a - e_b (daily MSE); HLN-corrected, t_{T-1}."""
    d = e_a - e_b
    T = len(d)
    v = newey_west(d, lag)
    dm = d.mean() / np.sqrt(v / T)
    h = lag + 1
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * corr
    return float(stat), float(2 * stats.t.sf(abs(stat), T - 1))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def comparisons() -> list[tuple]:
    """(family label, A=(group, est, pol), B=(group, est, pol), budgets)."""
    V_VOI = ("policies", "vigil", "voi")
    V_RR = ("estimators", "vigil", "rr")
    out = []
    for k in BUDGETS:
        out += [
            ("policy", "VIGIL VoI vs VIGIL RR", V_VOI, V_RR, k),
            ("policy", "VIGIL VoI vs VIGIL ARS", V_VOI, ("policies", "vigil", "ars"), k),
            ("policy", "VIGIL VoI vs VIGIL Random", V_VOI, ("policies", "vigil", "random"), k),
            ("pipeline", "VIGIL VoI vs LOCF RR", V_VOI, ("estimators", "locf", "rr"), k),
            ("pipeline", "VIGIL VoI vs Rain-KF KF-greedy", V_VOI, ("kf_pipeline", "kf", "kfgreedy"), k),
            ("pipeline", "VIGIL VoI vs Rain-KF VoI", V_VOI, ("kf_voi", "kf", "kfvoi"), k),
            ("pipeline", "VIGIL VoI vs GWN ARS", V_VOI, ("baseline_ars", "gwn", "ars"), k),
            ("pipeline", "VIGIL VoI vs LightGBM ARS", V_VOI, ("baseline_ars", "lgbm", "ars"), k),
        ]
        for b in ["locf", "kf", "lgbm", "mlp", "gru", "transformer", "dcrnn", "gwn"]:
            out.append(("estimator", f"VIGIL RR vs {b} RR", V_RR, ("estimators", b, "rr"), k))
    return out


def main():
    rows, erows = [], []
    days = onset_days()
    for fam, label, A, B, k in comparisons():
        fa, fb = files(A[0], A[1], A[2], k), files(B[0], B[1], B[2], k)
        ma, mb = daily_mse(fa), daily_mse(fb)
        if ma is None or mb is None:
            continue
        j = pd.concat([ma, mb], axis=1, join="inner").dropna()
        a, b = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
        ra, rb = np.sqrt(a), np.sqrt(b)
        w = stats.wilcoxon(ra, rb, zero_method="wilcox", alternative="two-sided")
        dm, pdm = dm_test(a, b)
        rows.append({"family": fam, "comparison": label, "k": k, "days": len(a), "seeds_a": len(fa), "seeds_b": len(fb),
                     "rmse_a": float(np.sqrt(a.mean())), "rmse_b": float(np.sqrt(b.mean())),
                     "gain_pct": float(100 * (1 - np.sqrt(a.mean()) / np.sqrt(b.mean()))),
                     "days_a_better_pct": float(100 * (ra < rb).mean()),
                     "wilcoxon_p": float(w.pvalue), "dm_stat": dm, "dm_p": pdm})
        if fam in ("policy", "pipeline"):
            da, db = det6(fa), det6(fb)
            if da is not None and db is not None and len(da) == len(db) == len(days):
                diff = da - db
                p = cluster_signflip(diff, days) if np.any(diff != 0) else 1.0
                erows.append({"family": fam, "comparison": label, "k": k, "onsets": len(da),
                              "recall6_a": float(da.mean()), "recall6_b": float(db.mean()),
                              "onsets_a_only": float((diff > 0).sum()), "onsets_b_only": float((diff < 0).sum()),
                              "onset_days": int(len(np.unique(days))), "signflip_p": float(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no comparisons available")
        return
    for col in ["wilcoxon_p", "dm_p"]:                      # Holm within each family of comparisons
        df[col + "_holm"] = df.groupby("family")[col].transform(lambda v: holm(v.to_numpy()))
    out = repo_path("results", "processed")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stats_tests.csv", index=False)
    ef = pd.DataFrame(erows)
    if not ef.empty:
        ef["signflip_p_holm"] = ef.groupby("family")["signflip_p"].transform(lambda v: holm(v.to_numpy()))
        ef.to_csv(out / "stats_events.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())
    print(ef.round(4).to_string() if not ef.empty else "")


if __name__ == "__main__":
    main()
