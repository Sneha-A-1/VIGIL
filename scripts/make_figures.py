"""Generate every data-driven figure of the manuscript (manuscript/figures/*.pdf).

    python scripts/make_figures.py [--only name ...]

Inputs: data/processed (dataset figure), results/processed/*.csv, results/raw/eval/**, results/raw/example,
results/raw/costs.json, results/raw/edgecloud/results.json. Figures whose inputs are missing are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from vigil.config import repo_path

FIG = repo_path("manuscript", "figures")
P = repo_path("results", "processed")
RAW = repo_path("results", "raw")
SINGLE, DOUBLE = 3.5, 7.2           # inches (Elsevier single / double column)
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
})
# Okabe-Ito palette
C = {"locf": "#7f7f7f", "kf": "#56B4E9", "lgbm": "#E69F00", "gwn": "#009E73", "vigil_rr": "#0072B2",
     "ars": "#CC79A7", "voi": "#D55E00", "oracle": "#000000", "random": "#B8B8B8", "rain": "#4C72B0"}
BUDGETS = [1, 2, 4, 8]
MAC_DATA = {}


def read(name):
    f = P / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def sel(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[np.isclose(q[k], v)] if isinstance(v, float) else q[q[k] == v]
    return q.sort_values("k") if "k" in q else q


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


# --------------------------------------------------------------------------- dataset figure
def fig_data():
    from vigil.data.dataset import load
    d = load()
    st = pd.read_csv(repo_path("data", "processed", "static_features.csv"))
    x = d.x.numpy() * d.vwc_std + d.vwc_mean
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.3], height_ratios=[1.15, 1], wspace=0.28, hspace=0.42)
    # (a) map with graph
    ax = fig.add_subplot(gs[0, 0])
    xy = d.xy.numpy() if hasattr(d.xy, "numpy") else np.asarray(d.xy)
    xy = xy - xy.min(0)
    A = d.adj.numpy() if hasattr(d.adj, "numpy") else np.asarray(d.adj)
    for i, j in zip(*np.nonzero(np.triu(A + A.T))):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="0.8", lw=0.5, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=st["elev"], cmap="viridis", s=14, edgecolor="k", lw=0.3, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.32, fraction=0.06, aspect=22)
    cb.set_label("Elevation (m)")
    panel(ax, "a")
    # (b) one logger, five depths, one wet season of the test period, with rain
    t0 = int(d.index.searchsorted(pd.Timestamp("2015-09-01")))
    t1 = int(d.index.searchsorted(pd.Timestamp("2016-06-01")))
    avail = np.isfinite(x[t0:t1]).mean((0, 2))
    i = int(np.argmax(avail))
    ax = fig.add_subplot(gs[0, 1])
    depths = [30, 60, 90, 120, 150]
    cols = plt.cm.viridis(np.linspace(0.0, 0.85, 5))
    tt = d.index[t0:t1]
    ev = d.ev[t0:t1, i].numpy()
    on = np.nan_to_num(ev).max(1) > 0.5
    ax.fill_between(tt, 0, 1, where=on, transform=ax.get_xaxis_transform(), color=C["voi"], alpha=0.18, lw=0,
                    label="wetting event")
    for k in range(d.D):
        ax.plot(tt, x[t0:t1, i, k], color=cols[k], lw=0.8, label=f"{depths[k]} cm")
    ax.set_ylabel(r"VWC (m$^3$ m$^{-3}$)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=0.9)
    ax2 = ax.twinx()
    pr = pd.Series(d.prcp_mm[t0:t1].numpy(), index=tt).resample("D").sum()
    ax2.bar(pr.index, pr.values, width=1.0, color=C["rain"], alpha=0.6)
    ax2.set_ylim(pr.max() * 3.2, 0)
    ax2.set_ylabel("Rain (mm d$^{-1}$)")
    ax2.spines["right"].set_visible(True)
    panel(ax, "b")
    # (c) availability raster (fraction of valid channel-hours per logger and week)
    ax = fig.add_subplot(gs[1, :])
    valid = pd.DataFrame(np.isfinite(x).mean(2), index=d.index).resample("W").mean()
    order = np.argsort(xy[:, 0])
    im = ax.imshow(valid.to_numpy().T[order], aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                   extent=[0, len(valid), d.N, 0])
    yrs = pd.date_range(d.index[0], d.index[-1], freq="YS")
    pos = [valid.index.searchsorted(y) for y in yrs]
    ax.set_xticks(pos)
    ax.set_xticklabels([str(y.year) for y in yrs])
    for nm, (s0, s1) in d.split.items():
        p_ = valid.index.searchsorted(d.index[s0])
        ax.axvline(p_, color=C["voi"], lw=0.9)
        ax.text(p_ + 2, -1.2, {"train": "training", "val": "validation", "test": "test"}[nm], color=C["voi"],
                fontsize=6.5)
    ax.set_ylabel("Logger (west to east)")
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label("Valid fraction")
    panel(ax, "c")
    MAC_DATA["ExampleLogger"] = str(st["Location"].iloc[i])
    save(fig, "fig_data")


# --------------------------------------------------------------------------- budgets
def pipeline_series():
    est, pol, kfp, ars = read("estimators"), read("policies"), read("kf_pipeline"), read("baseline_ars")
    S = []
    for df, e, p, lab, col, ls, mk in [
        (est, "locf", "rr", "LOCF + RR", C["locf"], "-", "o"),
        (kfp, "kf", "kfgreedy", "Rain-KF + KF-greedy", C["kf"], "-", "s"),
        (read("kf_voi"), "kf", "kfvoi", "Rain-KF + VoI index", C["kf"], "--", "s"),
        (est, "gwn", "rr", "Graph WaveNet + RR", C["gwn"], "-", "^"),
        (ars, "gwn", "ars", "Graph WaveNet + ARS", C["gwn"], "--", "^"),
        (est, "vigil", "rr", "VIGIL + RR", C["vigil_rr"], "-", "D"),
        (pol, "vigil", "ars", "VIGIL + ARS", C["ars"], "--", "D"),
        (pol, "vigil", "voi", "VIGIL + VoI (ours)", C["voi"], "-", "o"),
        (pol, "vigil", "oracle", "VIGIL + Oracle (not deployable)", C["oracle"], ":", "x"),
    ]:
        if df.empty:
            continue
        q = sel(df, estimator=e, policy=p)
        if p == "voi":
            q = q[np.isclose(q["eta"], 0.5)]
        if len(q):
            S.append((q, lab, col, ls, mk, p))
    return S


def fig_budget():
    S = pipeline_series()
    if not S:
        return
    base = sel(read("estimators"), estimator="locf", policy="rr").set_index("k")["rmse"]
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 4.6))
    axs = axs.ravel()
    for q, lab, col, ls, mk, p in S:
        sc = "head" if "ev_head_recall_6h" in q and q["ev_head_recall_6h"].notna().all() else "rule"
        lw = 1.8 if p == "voi" else 1.0
        kw = dict(ls=ls, marker=mk, ms=3, color=col, lw=lw)
        axs[0].plot(q["k"], 1e3 * q["rmse"], label=lab, **kw)
        if p != "oracle":
            rel = 100 * (q.set_index("k")["rmse"] / base - 1)
            axs[1].plot(rel.index, rel.values, **kw)
        axs[2].plot(q["k"], 100 * q[f"ev_{sc}_recall_6h"], **kw)
        axs[3].plot(q["k"], q[f"ev_{sc}_median_delay_h"], **kw)
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(k) for k in BUDGETS])
        ax.set_xticks([], minor=True)
    for ax in axs[2:]:
        ax.set_xlabel("Polls per hour $K$ (of 42 loggers)")
    axs[0].set_yscale("log")
    axs[0].set_yticks([1, 2, 3, 4, 6, 8])
    axs[0].set_yticks([], minor=True)
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axs[0].set_ylabel(r"Field RMSE ($10^{-3}$ m$^3$ m$^{-3}$)")
    axs[1].axhline(0, color="0.6", lw=0.6)
    axs[1].set_ylabel("RMSE change vs LOCF + RR (%)")
    axs[2].set_ylabel("Onsets detected within 6 h (%)")
    axs[3].set_ylabel("Median detection delay (h)")
    for a_, s_ in zip(axs, "abcd"):
        panel(a_, s_)
    axs[0].legend(loc="lower center", bbox_to_anchor=(1.1, 1.06), ncol=4, columnspacing=1.0, handlelength=2.2)
    fig.subplots_adjust(wspace=0.28, hspace=0.32)
    save(fig, "fig_budget")


# --------------------------------------------------------------------------- example trace
def fig_trace():
    f = RAW / "example" / "trace.npz"
    if not f.exists():
        return
    z = np.load(f, allow_pickle=True)
    tt = pd.to_datetime(z["hours"].astype(str))
    x = z["x"]
    # channel with the largest rise in the window, weighted by how much round-robin misses it
    with np.errstate(all="ignore"):
        rise = np.nanmax(x, 0) - np.nanmin(x, 0)
        err_rr = np.nanmean(np.abs(z["rr_mu"] - x), 0)
    score = np.where(np.isfinite(rise), rise, 0) * (1 + np.nan_to_num(err_rr) / (np.nanmean(err_rr) + 1e-9))
    i, k = np.unravel_index(np.nanargmax(score), score.shape)
    MAC_DATA["TraceDepth"] = str([30, 60, 90, 120, 150][k])
    # timing of the rise and recession of the shown channel (crossing of the mid level) and poll allocation
    xs = x[:, i, k]
    mid = np.nanmin(xs) + 0.5 * (np.nanmax(xs) - np.nanmin(xs))
    up = lambda a: int(np.where(a >= mid)[0][0])
    t_up = up(xs)
    pk = int(np.nanargmax(xs))
    down = lambda a: pk + int(np.where(a[pk:] < mid)[0][0])
    t_dn = down(xs)
    for nm, key in [("rr_mu", "Rr"), ("voi_mu", "Voi"), ("locf_mu", "Locf")]:
        MAC_DATA[f"Trace{key}RiseLag"] = str(up(z[nm][:, i, k]) - t_up)
        MAC_DATA[f"Trace{key}FallLag"] = str(down(z[nm][:, i, k]) - t_dn)
        MAC_DATA[f"Trace{key}Rmse"] = f"{1e3 * float(np.sqrt(np.nanmean((z[nm] - x) ** 2))):.2f}"
    MAC_DATA["TraceLow"] = f"{np.nanmin(xs):.2f}"
    MAC_DATA["TraceHigh"] = f"{np.nanmax(xs):.2f}"
    with np.errstate(all="ignore"):
        dyn_ = np.nansum(np.nanmean(np.abs(np.diff(x, axis=0)), axis=0), axis=1)
    top = np.argsort(-dyn_)[:10]
    for nm, key in [("rr_poll", "Rr"), ("voi_poll", "Voi")]:
        P_ = z[nm]
        MAC_DATA[f"Trace{key}TopShare"] = f"{100 * P_[:, top].sum() / P_.sum():.0f}"
        MAC_DATA[f"Trace{key}LoggerPolls"] = str(int(P_[:, i].sum()))
    MAC_DATA["TraceStart"] = tt[0].strftime("%-d %B %Y")
    MAC_DATA["TraceEnd"] = tt[-1].strftime("%-d %B %Y")
    depths = [30, 60, 90, 120, 150]
    fig = plt.figure(figsize=(DOUBLE, 5.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 1.6, 0.95, 0.95], hspace=0.22)
    ax0 = fig.add_subplot(gs[0])
    ax0.bar(tt, z["prcp"], width=1 / 24, color=C["rain"])
    ax0.set_ylabel("Rain\n(mm h$^{-1}$)")
    ax0.tick_params(labelbottom=False)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(tt, x[:, i, k], color="k", lw=1.3, label="truth (hidden)")
    ax1.step(tt, z["locf_mu"][:, i, k], where="post", color=C["locf"], lw=1.0, label="LOCF + RR")
    lo = np.nanmin(x[:, i, k])
    span = np.nanmax(x[:, i, k]) - lo
    for j, (nm, col, lab) in enumerate([("rr", C["vigil_rr"], "VIGIL + RR"), ("voi", C["voi"], "VIGIL + VoI")]):
        mu, sd = z[f"{nm}_mu"][:, i, k], z[f"{nm}_sigma"][:, i, k]
        ax1.plot(tt, mu, color=col, lw=1.1, label=lab)
        ax1.fill_between(tt, mu - 1.645 * sd, mu + 1.645 * sd, color=col, alpha=0.18, lw=0)
        pt = np.where(z[f"{nm}_poll"][:, i])[0]
        ax1.plot(tt[pt], np.full(len(pt), lo - (0.10 + 0.06 * j) * span), "|", color=col, ms=6, mew=1.3)
    ax1.set_ylim(lo - 0.2 * span, None)
    ax1.set_ylabel(f"VWC at {depths[k]} cm\n(m$^3$ m$^{{-3}}$)")
    ax1.legend(ncol=1, loc="center right", handlelength=1.5)
    ax1.tick_params(labelbottom=False)
    # loggers sorted by how much their profile changed in the window (most dynamic at the top)
    dyn = np.nansum(np.nanmean(np.abs(np.diff(x, axis=0)), axis=0), axis=1)
    order = np.argsort(-dyn)
    pos = int(np.where(order == i)[0][0])
    for row, (nm, lab) in enumerate([("rr", "RR"), ("voi", "VoI")]):
        ax = fig.add_subplot(gs[2 + row], sharex=ax0)
        P_ = z[f"{nm}_poll"].T[order].astype(float)
        ax.imshow(P_, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1,
                  extent=[matplotlib.dates.date2num(tt[0]), matplotlib.dates.date2num(tt[-1]), P_.shape[0], 0])
        ax.axhline(pos + 0.5, color=C["voi"] if nm == "voi" else C["vigil_rr"], lw=0.5, alpha=0.8)
        ax.set_ylabel(f"{lab} polls\nlogger rank")
        if row == 0:
            ax.tick_params(labelbottom=False)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b %Y"))
    for a_, s_ in zip(fig.axes, "abcd"):
        a_.text(-0.075, 1.0, s_, transform=a_.transAxes, fontsize=8, fontweight="bold", va="top", ha="right")
    save(fig, "fig_trace")


# --------------------------------------------------------------------------- urgent/strategic trade-off
def fig_tradeoff():
    tr, pol, est = read("tradeoff"), read("policies"), read("estimators")
    if tr.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 2.3))
    for ax, k in zip(axs, [2, 4]):
        q = pd.concat([sel(tr, estimator="vigil", policy="voi", k=k), sel(pol, estimator="vigil", policy="voi", k=k)])
        q = q.drop_duplicates("eta").sort_values("eta")
        ax.plot(1e3 * q["rmse"], 100 * q["ev_head_recall_6h"], "-o", color=C["voi"], ms=3, label="VoI, $\\eta=0,0.25,\\dots,1$")
        d5 = q[np.isclose(q["eta"], 0.5)]
        ax.plot(1e3 * d5["rmse"], 100 * d5["ev_head_recall_6h"], "*", color=C["voi"], ms=9, label="VoI, $\\eta=0.5$ (default)")
        for e_, dx, dy in [(0.0, 4, -8), (1.0, -4, 5)]:
            r = q[np.isclose(q["eta"], e_)].iloc[0]
            ax.annotate(f"$\\eta={e_:g}$", (1e3 * r["rmse"], 100 * r["ev_head_recall_6h"]), fontsize=6,
                        xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right")
        for df, e, p, col, mk, lab in [(est, "vigil", "rr", C["vigil_rr"], "D", "RR"),
                                       (pol, "vigil", "ars", C["ars"], "D", "ARS"),
                                       (pol, "vigil", "random", C["random"], "v", "Random")]:
            r = sel(df, estimator=e, policy=p, k=k)
            if len(r):
                ax.plot(1e3 * r["rmse"], 100 * r["ev_head_recall_6h"], mk, color=col, ms=4, label=f"VIGIL + {lab}")
        ax.set_title(f"$K={k}$")
        ax.set_xlabel(r"Field RMSE ($10^{-3}$ m$^3$ m$^{-3}$)")
    axs[0].set_ylabel("Onsets detected within 6 h (%)")
    axs[0].legend(loc="lower right", fontsize=5.5)
    for a_, s in zip(axs, "ab"):
        panel(a_, s)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_tradeoff")


# --------------------------------------------------------------------------- calibration and seasonality
def calib(group, pat):
    fs = sorted(glob.glob(str(RAW / "eval" / group / pat)))
    cov = [json.load(open(f)).get("calib_coverage") for f in fs]
    cov = [c for c in cov if c]
    if not cov:
        return None, None
    lv = json.load(open(fs[0]))["calib_levels"]
    return np.array(lv), np.mean(cov, 0)


def fig_calib():
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw={"width_ratios": [1, 1.8]})
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
    any_ = False
    for group, pat, col, lab in [("estimators", "vigil__rr__K2__s*.json", C["vigil_rr"], "VIGIL + RR"),
                                 ("policies", "vigil__voi_eta0.5_*__K2__s*.json", C["voi"], "VIGIL + VoI"),
                                 ("estimators", "kf__rr__K2__s*.json", C["kf"], "Rain-KF + RR")]:
        lv, cv = calib(group, pat)
        if lv is not None:
            ax.plot(lv, cv, "-o", ms=2.5, color=col, label=lab)
            any_ = True
    ax.set_xlabel("Nominal central interval")
    ax.set_ylabel("Empirical coverage (unpolled)")
    ax.legend(loc="upper left")
    ax.set_title("$K=2$", fontsize=6.5)
    panel(ax, "a")
    ax = axs[1]
    for group, pat, col, lab in [("estimators", "locf__rr__K2__s*.npz", C["locf"], "LOCF + RR"),
                                 ("estimators", "vigil__rr__K2__s*.npz", C["vigil_rr"], "VIGIL + RR"),
                                 ("policies", "vigil__voi_eta0.5_*__K2__s*.npz", C["voi"], "VIGIL + VoI")]:
        fs = sorted(glob.glob(str(RAW / "eval" / group / pat)))
        if not fs:
            continue
        m = np.mean([np.load(f, allow_pickle=True)["monthly_rmse"] for f in fs], 0)
        months = pd.PeriodIndex(np.load(fs[0], allow_pickle=True)["months"].astype(str), freq="M").to_timestamp()
        ax.plot(months, 1e3 * m, "-o", ms=2.5, color=col, label=lab)
        any_ = True
    ax.set_ylabel(r"Monthly RMSE ($10^{-3}$ m$^3$ m$^{-3}$)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.legend(loc="upper right")
    ax.set_title("$K=2$, test period", fontsize=6.5)
    panel(ax, "b")
    fig.subplots_adjust(wspace=0.3)
    if any_:
        save(fig, "fig_calib")
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- scalability
def fig_scaling():
    f = RAW / "costs.json"
    if not f.exists():
        return
    c = json.load(open(f))
    sc = c.get("scaling")
    if not sc:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 2.2))
    n = pd.DataFrame(sc["nodes"])
    axs[0].loglog(n["N"], n["vigil_ms"], "-o", color=C["voi"], ms=3, label="VIGIL network step")
    if "kfgreedy_ms" in n:
        q = n.dropna(subset=["kfgreedy_ms"])
        axs[0].loglog(q["N"], q["kfgreedy_ms"], "-s", color=C["kf"], ms=3, label="Rain-KF + KF-greedy step")
    axs[0].set_xlabel("Loggers $N$ in one network")
    axs[0].set_ylabel("Latency per hourly step (ms)")
    axs[0].set_xticks(list(n["N"]))
    axs[0].set_xticklabels([str(v) for v in n["N"]], rotation=0)
    axs[0].set_xticks([], minor=True)
    axs[0].set_ylim(top=3000)
    axs[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    fl = pd.DataFrame(sc["fleet"])
    axs[1].loglog(fl["B"], fl["per_farm_ms"], "-o", color=C["voi"], ms=3, label="per farm")
    axs[1].loglog(fl["B"], fl["step_ms"], "--o", color="0.5", ms=3, label="whole batch")
    axs[1].set_xlabel("Farms processed in one batch $B$")
    axs[1].set_ylabel("Latency per hourly step (ms)")
    axs[1].set_xticks(list(fl["B"]))
    axs[1].set_xticklabels([str(v) for v in fl["B"]])
    axs[1].set_xticks([], minor=True)
    axs[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    for a_, s in zip(axs, "ab"):
        panel(a_, s)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig_scaling")


# --------------------------------------------------------------------------- edge-cloud
def fig_edgecloud():
    f = RAW / "edgecloud" / "results.json"
    if not f.exists():
        return
    r = pd.DataFrame(json.load(open(f))["results"])
    g = r.groupby(["policy", "utilisation_target"]).mean(numeric_only=True).reset_index()
    names = {"fcfs": "FCFS", "edf": "EDF", "mct": "MCT", "edf_p": "Preemptive EDF",
             "urgency": "Urgency-aware", "urgency_oracle": "Urgency-aware, oracle capacity$^\\dagger$"}
    cols = {"fcfs": "#7f7f7f", "edf": "#56B4E9", "mct": "#E69F00", "edf_p": "#009E73", "urgency": "#D55E00",
            "urgency_oracle": "#000000"}
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
    for p in names:
        q = g[g.policy == p].sort_values("utilisation_target")
        if q.empty:
            continue
        st = ":x" if p == "urgency_oracle" else "-o"
        kw = dict(color=cols[p], ms=3 if p != "urgency_oracle" else 4, lw=2.2 if p == "urgency" else 1.0, label=names[p])
        axs[0].plot(100 * q.utilisation_target, 100 * q.urgent_miss_rate, st, **kw)
        axs[1].plot(100 * q.utilisation_target, q.urgent_resp_p95_s / 3600, st, **kw)
        axs[2].plot(100 * q.utilisation_target, 100 * q.strategic_miss_rate, st, **kw)
    axs[0].set_ylabel("Urgent deadline misses (%)")
    axs[1].set_ylabel("Urgent 95th-pct. response (h)")
    axs[1].set_yscale("log")
    axs[2].set_ylabel("Strategic deadline misses (%)")
    for a_, s in zip(axs, "abc"):
        a_.set_xlabel("Offered load (%)")
        panel(a_, s)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.75, 1.3), ncol=3)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_edgecloud")


FIGS = {"data": fig_data, "budget": fig_budget, "trace": fig_trace, "tradeoff": fig_tradeoff, "calib": fig_calib,
        "scaling": fig_scaling, "edgecloud": fig_edgecloud}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for name, fn in FIGS.items():
        if a.only and name not in a.only:
            continue
        try:
            fn()
        except Exception:
            print("FAILED", name)
            traceback.print_exc()
    if MAC_DATA:   # macros quoted in figure captions (merged with existing ones)
        f = repo_path("manuscript", "macros", "figures.tex")
        old = {}
        if f.exists():
            for line in f.read_text().splitlines():
                if line.startswith("\\newcommand"):
                    k = line.split("{\\", 1)[1].split("}", 1)[0]
                    old[k] = line.rsplit("{", 1)[1].rstrip("}")
        old.update(MAC_DATA)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("% generated by scripts/make_figures.py\n" + "\n".join(
            f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(old.items())) + "\n")


if __name__ == "__main__":
    main()
