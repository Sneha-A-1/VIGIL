"""Additional tables and macros (ablations, weather source, significance tests, costs, edge-cloud,
capacity forecasting, validation tuning, per-depth errors). Called by scripts/make_tables.py.

All numbers are means over seeds (single numbers); spreads are in results/processed/*_std.csv.
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np
import pandas as pd

from vigil.config import repo_path


def fit_width(text: str) -> str:
    """Wrap every tabular in an adjustbox that shrinks it to the text width only if it is wider."""
    text = re.sub(r"(\\begin\{tabular\})", r"\\begin{adjustbox}{max width=\\textwidth}\n\1", text)
    text = re.sub(r"(?<=[&\s])-(?=\d)", r"$-$", text)      # typographic minus in table cells
    return text.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")

P = repo_path("results", "processed")
RAW = repo_path("results", "raw")
T = repo_path("manuscript", "tables")
W = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}


def read(name):
    f = P / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def num(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def get(df, **kw):
    q = df
    for k, v in kw.items():
        if q.empty:
            return None
        q = q[np.isclose(q[k], v)] if isinstance(v, float) else q[q[k] == v]
    return q.iloc[0] if len(q) else None


def bold(vals, lower=True, nd=None):
    arr = np.array([v if v is not None and np.isfinite(v) else (np.inf if lower else -np.inf) for v in vals], float)
    if nd is not None:
        arr = np.where(np.isfinite(arr), np.round(arr, nd), arr)
    if not np.isfinite(arr).any():
        return [False] * len(vals)
    b = np.nanmin(arr) if lower else np.nanmax(arr)
    return [abs(a - b) < 1e-12 for a in arr]


def fmt_cells(rows, nds, lowers):
    cols = list(zip(*rows))
    bs = [bold(c, lo, nd) if lo is not None else [False] * len(c) for c, lo, nd in zip(cols, lowers, nds)]
    out = []
    for i, r in enumerate(rows):
        cells = []
        for j, v in enumerate(r):
            s = num(v, nds[j])
            cells.append(rf"\textbf{{{s}}}" if bs[j][i] and s != "--" else s)
        out.append(cells)
    return out


def seed_mean(group, seeds, **kw):
    """Mean over the given seeds from the seed-level CSV of a group."""
    f = P / f"{group}_seeds.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df.seed.isin(seeds)]
    for k, v in kw.items():
        df = df[np.isclose(df[k], v)] if isinstance(v, float) else df[df[k] == v]
    if df.empty:
        return None
    return df.mean(numeric_only=True)


# --------------------------------------------------------------------------- ablations
def table_ablation(MAC):
    abl, pab = read("ablations"), read("policy_ablations")
    if abl.empty and pab.empty:
        return
    seeds3 = [0, 1, 2]
    rows, names = [], []

    def add(name, rr_src, voi_src):
        r = []
        for k in (2, 4):
            a = rr_src(k)
            r.append(None if a is None else 1e3 * a["rmse"])
        for k in (2, 4):
            b = voi_src(k)
            r += [None if b is None else 1e3 * b["rmse"], None if b is None else 100 * b["ev_head_recall_6h"]]
        a4 = rr_src(4)
        r.append(None if a4 is None or not np.isfinite(a4.get("crps_unobs", np.nan)) else 1e3 * a4["crps_unobs"])
        rows.append(r)
        names.append(name)

    full_rr = lambda k: seed_mean("estimators", seeds3, estimator="vigil", policy="rr", k=k)
    full_voi = lambda k: seed_mean("policies", seeds3, estimator="vigil", policy="voi", k=k, eta=0.5)
    add(r"\vigil{} (full)", full_rr, full_voi)
    for v, lab in [("no_graph", "w/o graph message passing"), ("no_predict_update", "w/o predict--update split"),
                   ("no_hetero", r"w/o $\beta$-NLL (MSE loss)"), ("no_closed_loop", "w/o closed-loop training"),
                   ("no_lookahead", "w/o look-ahead heads (current-value VoI)")]:
        e = f"vigil-{v}"
        pol = "voi_myopic" if v == "no_lookahead" else "voi"
        add(lab, lambda k, e=e: get(abl, estimator=e, policy="rr", k=k),
            lambda k, e=e, pol=pol: get(abl, estimator=e, policy=pol, k=k))
    na = len(rows)
    for pol, lab in [("voi_greedy", "greedy ranking instead of Whittle index"),
                     ("voi_myopic", "current-value variance instead of look-ahead growth"),
                     ("voi", r"no shrinkage ($\kappa=1$)")]:
        add(lab, lambda k: None, lambda k, pol=pol: seed_mean("policy_ablations", seeds3, estimator="vigil", policy=pol, k=k))
    cells = fmt_cells(rows, [2, 2, 2, 1, 2, 1, 2], [True, True, True, False, True, False, True])
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         r"\caption{Ablations (test period, mean of seeds 0--2). Network ablations retrain the estimator without one component and are evaluated with round-robin (RR) and VoI polling; policy ablations keep the trained \vigil{} network and change one element of the polling index. RMSE and CRPS in $10^{-3}$\,m$^3$\,m$^{-3}$; R@6h: wetting onsets detected within 6~h (\%). Best values in bold.}",
         r"\label{tab:ablation}", r"\begin{tabular}{lcc|cccc|c}", r"\toprule",
         r" & \multicolumn{2}{c|}{RR: RMSE} & \multicolumn{2}{c}{VoI, $K=2$} & \multicolumn{2}{c|}{VoI, $K=4$} & RR, $K=4$ \\",
         r"Variant & $K{=}2$ & $K{=}4$ & RMSE & R@6h & RMSE & R@6h & CRPS \\", r"\midrule"]
    for i, (n, c) in enumerate(zip(names, cells)):
        if i == 1:
            L.append(r"\multicolumn{8}{l}{\emph{Network ablations}} \\")
        if i == na:
            L.append(r"\midrule")
            L.append(r"\multicolumn{8}{l}{\emph{Polling-index ablations (full network)}} \\")
        L.append(f"{n} & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (T / "ablation.tex").write_text(fit_width("\n".join(L) + "\n"))
    # macros: relative RMSE changes under VoI at K=2 and K=4
    base = {k: rows[0][2 if k == 2 else 4] for k in (2, 4)}
    keys = ["Full", "NoGraph", "NoPU", "NoHetero", "NoClosed", "NoLook", "Greedy", "Myopic", "NoKappa"]
    for key, r in zip(keys, rows):
        for k in (2, 4):
            v = r[2 if k == 2 else 4]
            if v is not None and base[k] is not None:
                MAC[f"Abl{key}Voi{W[k]}"] = f"{v:.2f}"
                MAC[f"Abl{key}VoiDelta{W[k]}"] = f"{100 * (v / base[k] - 1):.1f}"
            rec = r[3 if k == 2 else 5]
            if rec is not None:
                MAC[f"Abl{key}Rec{W[k]}"] = f"{rec:.1f}"
    arch = []
    for key, r in zip(keys, rows):
        if key in ("NoGraph", "NoPU", "NoClosed", "NoLook"):
            for k in (2, 4):
                v = r[2 if k == 2 else 4]
                if v is not None and base[k] is not None:
                    arch.append(100 * (v / base[k] - 1))
    if arch:
        MAC["AblArchMin"] = f"{min(arch):+.1f}"
        MAC["AblArchMax"] = f"{max(arch):+.1f}"
    # seed-to-seed variation of the full model under VoI (coefficient of variation of the RMSE, seeds 0-4)
    f = P / "policies_seeds.csv"
    if f.exists():
        d = pd.read_csv(f)
        for k in (2, 4):
            q = d[(d.estimator == "vigil") & (d.policy == "voi") & (d.k == k) & np.isclose(d.eta, 0.5)]
            if len(q) > 1:
                MAC[f"VoiSeedCv{W[k]}"] = f"{100 * q.rmse.std() / q.rmse.mean():.1f}"


# --------------------------------------------------------------------------- weather source
def table_weather(MAC):
    w = read("weather")
    if w.empty:
        return
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Robustness to the weather source at test time (1~January~2015 -- 5~January~2016, when the on-farm station reports). The model is trained with NASA POWER inputs in both cases. RMSE in $10^{-3}$\,m$^3$\,m$^{-3}$; R@6h in \%; mean of five seeds.}",
         r"\label{tab:weather}", r"\begin{tabular}{llcccc}", r"\toprule",
         r"Policy & $K$ & RMSE (POWER) & RMSE (SCAN) & R@6h (POWER) & R@6h (SCAN) \\", r"\midrule"]
    for pol, lab in [("rr", "Round-robin"), ("voi", r"\vigil{} VoI")]:
        for k in (2, 4):
            a = get(w, estimator="vigil", policy=pol, k=k, weather="power")
            b = get(w, estimator="vigil", policy=pol, k=k, weather="scan")
            if a is None or b is None:
                continue
            L.append(f"{lab} & {k} & {1e3 * a['rmse']:.2f} & {1e3 * b['rmse']:.2f} & {100 * a['ev_head_recall_6h']:.1f} & {100 * b['ev_head_recall_6h']:.1f} \\\\")
            MAC[f"Wx{pol.capitalize()}{W[k]}Delta"] = f"{100 * (b['rmse'] / a['rmse'] - 1):+.1f}"
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (T / "weather.tex").write_text(fit_width("\n".join(L) + "\n"))


# --------------------------------------------------------------------------- significance
def fmt_pm(p):
    """p-value as math-mode text for the manuscript: 'p<10^{-4}' or 'p=0.013'."""
    if p < 1e-4:
        return r"p<10^{-4}"
    return f"p={p:.4f}" if p < 0.01 else f"p={p:.3f}"


def fmt_p(p):
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def table_stats(MAC):
    s = read("stats_tests")
    e = read("stats_events")
    if s.empty:
        return
    keep = ["VIGIL VoI vs LOCF RR", "VIGIL VoI vs VIGIL RR", "VIGIL VoI vs VIGIL ARS", "VIGIL VoI vs Rain-KF KF-greedy",
            "VIGIL VoI vs Rain-KF VoI", "VIGIL VoI vs LightGBM ARS", "VIGIL VoI vs GWN ARS",
            "VIGIL RR vs locf RR", "VIGIL RR vs gwn RR", "VIGIL RR vs lgbm RR"]
    pretty = {"VIGIL VoI vs LOCF RR": "VoI vs LOCF+RR", "VIGIL VoI vs VIGIL RR": r"VoI vs \vigil{}+RR",
              "VIGIL VoI vs VIGIL ARS": r"VoI vs \vigil{}+ARS", "VIGIL VoI vs Rain-KF KF-greedy": "VoI vs KF+greedy",
              "VIGIL VoI vs Rain-KF VoI": "VoI vs KF+VoI index", "VIGIL VoI vs LightGBM ARS": "VoI vs LightGBM+ARS",
              "VIGIL VoI vs GWN ARS": "VoI vs GWN+ARS", "VIGIL RR vs locf RR": r"\vigil{}+RR vs LOCF+RR",
              "VIGIL RR vs gwn RR": r"\vigil{}+RR vs GWN+RR", "VIGIL RR vs lgbm RR": r"\vigil{}+RR vs LightGBM+RR"}
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         r"\caption{Paired significance tests on the test period (VoI: \vigil{} with VoI polling; first pipeline A, second B). Gain: relative reduction of the field RMSE by A; days: share of test days on which A has the lower daily RMSE; Wilcoxon signed-rank and Diebold--Mariano (DM; Newey--West variance with lag 7~d and small-sample correction) tests on daily errors, DM $<0$ favouring A; detection: wetting onsets detected within 6~h (\%) by A and B, with a paired randomisation test that flips the labels of all onsets of a day together. $p$-values are Holm-adjusted within each family of comparisons.}",
         r"\label{tab:stats}", r"\begin{tabular}{lc|cccccc|ccc}", r"\toprule",
         r" & & \multicolumn{6}{c|}{Field-state accuracy (daily errors)} & \multicolumn{3}{c}{Onset detection} \\",
         r"Comparison (A vs B) & $K$ & Gain (\%) & Days (\%) & Wilcoxon $p$ & DM & DM $p$ & & R@6h A & R@6h B & $p$ \\",
         r"\midrule"]
    for c in keep:
        for k in (2, 4):
            r = get(s, comparison=c, k=k)
            if r is None:
                continue
            ev = get(e, comparison=c, k=k) if not e.empty else None
            evc = (f"{100 * ev['recall6_a']:.1f} & {100 * ev['recall6_b']:.1f} & {fmt_p(ev['signflip_p_holm'])}"
                   if ev is not None else "-- & -- & --")
            L.append(f"{pretty[c]} & {k} & {r['gain_pct']:.1f} & {r['days_a_better_pct']:.1f} & {fmt_p(r['wilcoxon_p_holm'])} & "
                     f"{r['dm_stat']:.2f} & {fmt_p(r['dm_p_holm'])} & & {evc} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    L = [l if "multicolumn" in l else l.replace(" & & ", " & ") for l in L]
    L = [l.replace(r"\begin{tabular}{lc|cccccc|ccc}", r"\begin{tabular}{lc|ccccc|ccc}")
          .replace(r"\multicolumn{6}{c|}{Field-state", r"\multicolumn{5}{c|}{Field-state")
          .replace(r"DM $p$ & & R@6h", r"DM $p$ & R@6h") for l in L]
    (T / "stats.tex").write_text(fit_width("\n".join(L) + "\n"))
    MAC["NStatTests"] = str(len(s))
    MAC["NStatSig"] = str(int((s["dm_p_holm"] < 0.05).sum()))
    MAC["MaxPVoiLocf"] = fmt_p(s[s.comparison == "VIGIL VoI vs LOCF RR"]["wilcoxon_p_holm"].max())
    MAC["MaxPVoiRr"] = fmt_p(s[s.comparison == "VIGIL VoI vs VIGIL RR"]["wilcoxon_p_holm"].max())
    for c, nm in [("VIGIL VoI vs LOCF RR", "VoiLocf"), ("VIGIL VoI vs VIGIL RR", "VoiRr"),
                  ("VIGIL VoI vs LightGBM ARS", "VoiLgbmArs"), ("VIGIL VoI vs Rain-KF VoI", "VoiKfVoi")]:
        for k in (1, 2, 4, 8):
            r = get(s, comparison=c, k=k)
            if r is None:
                continue
            MAC[f"{nm}Days{W[k]}"] = f"{r['days_a_better_pct']:.0f}"
            MAC[f"{nm}DmP{W[k]}"] = fmt_pm(r["dm_p_holm"])
            MAC[f"{nm}WilP{W[k]}"] = fmt_pm(r["wilcoxon_p_holm"])
            MAC[f"{nm}DmPraw{W[k]}"] = fmt_pm(r["dm_p"])
            if not e.empty:
                ev = get(e, comparison=c, k=k)
                if ev is not None:
                    MAC[f"{nm}DetP{W[k]}"] = fmt_pm(ev["signflip_p_holm"])
    MAC["NStatTests"] = str(len(s))
    MAC["NStatSig"] = str(int((s["dm_p_holm"] < 0.05).sum()))
    MAC["MaxPVoiLocf"] = fmt_p(s[s.comparison == "VIGIL VoI vs LOCF RR"]["wilcoxon_p_holm"].max())
    MAC["MaxPVoiRr"] = fmt_p(s[s.comparison == "VIGIL VoI vs VIGIL RR"]["wilcoxon_p_holm"].max())


# --------------------------------------------------------------------------- costs and scalability
def table_costs(MAC):
    f = RAW / "costs.json"
    if not f.exists():
        return
    c = json.load(open(f))
    names = {"mlp": "MLP", "gru": "GRU", "transformer": "Transformer", "dcrnn": "DCRNN", "gwn": "Graph WaveNet",
             "lgbm": "LightGBM", "kf": "Rain-KF", "kf_greedy": "Rain-KF + KF-greedy"}

    def train_s(model):
        fs = glob.glob(str(repo_path("results", "models", f"{model}-default", "seed*", "train_log.json")))
        v = [json.load(open(x))["info"].get("train_seconds") for x in fs]
        v = [x for x in v if x]
        return np.mean(v) / 60 if v else None

    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Model size, training time and inference latency per hourly step for one 42-logger farm on one CPU thread. Training times are wall-clock means over seeds, measured while two jobs shared the two virtual CPUs of the machine; latencies were measured on an otherwise idle machine. The \vigil{} edge step includes the predict step, the VoI index, polling and the update step.}",
         r"\label{tab:costs}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Estimator & Parameters & Training (min) & Latency (ms) \\", r"\midrule"]
    for k in ["lgbm", "mlp", "gru", "transformer", "dcrnn", "gwn", "kf", "kf_greedy"]:
        if k not in c["estimators"]:
            continue
        e = c["estimators"][k]
        par = f"{e['params']:,} trees" if k == "lgbm" else ("--" if k.startswith("kf") else f"{e['params']:,}")
        tr = train_s(k) if not k.startswith("kf") else None
        L.append(f"{names[k]} & {par} & {num(tr, 1)} & {e['latency_ms']:.2f} \\\\")
    L.append(f"\\vigil{{}} edge step & {c['vigil_params']:,} & {num(train_s('vigil'), 1)} & {c['edge_step_ms']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (T / "costs.tex").write_text(fit_width("\n".join(L) + "\n"))
    MAC["EdgeStepMs"] = f"{c['edge_step_ms']:.1f}"
    MAC["VigilParams"] = f"{c['vigil_params']:,}"
    MAC["OutlookTwentyFourS"] = f"{1000 * c['alert_outlook_24h_s']:.0f}"
    MAC["OutlookSeventyTwoS"] = f"{1000 * c['forecast_72h_s']:.0f}"
    MAC["MapIdwMs"] = f"{1000 * c['map_idw_s']:.1f}"
    MAC["RetrainS"] = f"{c['retrain_25it_s'] * 300 / 25 / 60:.1f}"
    tv = train_s("vigil")
    if tv:
        MAC["VigilTrainMin"] = f"{tv:.1f}"
    if "kf_greedy" in c["estimators"]:
        MAC["KfGreedyMs"] = f"{c['estimators']['kf_greedy']['latency_ms']:.1f}"
    sc = c.get("scaling")
    if sc:
        n = pd.DataFrame(sc["nodes"])
        big = n.iloc[-1]
        MAC["ScaleNmax"] = f"{int(big['N']):,}"
        MAC["ScaleNmaxMs"] = f"{big['vigil_ms']:.1f}"
        if "kfgreedy_ms" in n:
            q = n.dropna(subset=["kfgreedy_ms"]).iloc[-1]
            MAC["ScaleKfN"] = f"{int(q['N'])}"
            MAC["ScaleKfMs"] = f"{q['kfgreedy_ms']:.0f}"
            MAC["ScaleVigilAtKfN"] = f"{float(n[n.N == q['N']]['vigil_ms'].iloc[0]):.1f}"
        # log-log slope of the VIGIL latency
        sl = np.polyfit(np.log(n["N"]), np.log(n["vigil_ms"]), 1)[0]
        MAC["ScaleSlope"] = f"{sl:.2f}"
        fl = pd.DataFrame(sc["fleet"])
        MAC["FleetBmax"] = f"{int(fl.B.iloc[-1])}"
        MAC["FleetPerFarmMs"] = f"{fl.per_farm_ms.iloc[-1]:.2f}"
        MAC["FleetFarmsPerHour"] = f"{3.6e6 / fl.per_farm_ms.iloc[-1]:,.0f}"


# --------------------------------------------------------------------------- edge-cloud and capacity
def table_edgecloud(MAC):
    f = RAW / "edgecloud" / "results.json"
    cap = RAW / "capacity" / "results.json"
    if cap.exists():
        cr = pd.DataFrame(json.load(open(cap)))
        m = cr.mean(numeric_only=True)
        for k, nm in [("gru", "Gru"), ("last", "Last"), ("mean1h", "MeanHour"), ("ewma", "Ewma")]:
            MAC[f"CapMae{nm}"] = f"{m[f'mae_{k}']:.3f}"
        MAC["CapGainVsLast"] = f"{100 * (1 - m['mae_gru'] / m['mae_last']):.1f}"
        MAC["CapGainVsEwma"] = f"{100 * (1 - m['mae_gru'] / m['mae_ewma']):.1f}"
    if not f.exists():
        return
    J = json.load(open(f))
    r = pd.DataFrame(J["results"])
    g = r.groupby(["policy", "utilisation_target"]).mean(numeric_only=True).reset_index()
    names = {"fcfs": "FCFS", "edf": "EDF", "mct": "MCT (min.\\ completion time)", "edf_p": "Preemptive EDF",
             "urgency": r"Urgency-aware (\vigil{} dispatch)", "urgency_oracle": r"Urgency-aware, oracle capacity$^\dagger$"}
    allu = sorted(g.utilisation_target.unique())
    us = [allu[0], allu[-1]]
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         r"\caption{Edge--cloud case study: processing of \vigil{} analytics for regional fleets of farms on six highly variable co-tenant hosts whose residual capacity follows Bitbrains traces (December 2015 -- February 2016, mean of three workload seeds), at the lowest and highest offered load (mean task work divided by mean residual capacity). Urgent jobs (alert advisories) have a 120-s deadline; strategic tasks (maps, outlooks, weekly model refresh) have deadlines of 1~h to 1~day. Urg.\ p95: 95th percentile of the urgent response time. Best values in bold; $^\dagger$not deployable.}",
         r"\label{tab:edgecloud}", r"\begin{tabular}{l" + "|ccc" * len(us) + "}", r"\toprule",
         " & " + " & ".join([rf"\multicolumn{{3}}{{c{'|' if i < len(us) - 1 else ''}}}{{Offered load {int(100 * u)}\%}}" for i, u in enumerate(us)]) + r" \\",
         "Scheduler & " + " & ".join([r"Urg.\ miss (\%) & Urg.\ p95 (h) & Strat.\ miss (\%)"] * len(us)) + r" \\",
         r"\midrule"]
    rows, nm = [], []
    for p in names:
        row = []
        for u in us:
            q = get(g, policy=p, utilisation_target=float(u))
            row += [None if q is None else 100 * q["urgent_miss_rate"], None if q is None else q["urgent_resp_p95_s"] / 3600,
                    None if q is None else 100 * q["strategic_miss_rate"]]
        rows.append(row)
        nm.append(names[p])
    cells = fmt_cells(rows, [1, 1, 1] * len(us), [True] * 3 * len(us))
    for n_, c in zip(nm, cells):
        L.append(f"{n_} & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (T / "edgecloud.tex").write_text(fit_width("\n".join(L) + "\n"))
    MAC["EcHosts"] = str(len(J["hosts"]))
    MAC["EcResidualCores"] = f"{J['mean_residual_cores']:.1f}"
    hi = us[-1]
    for p, key in [("urgency", "Urg"), ("fcfs", "Fcfs"), ("edf", "Edf"), ("edf_p", "Edfp"), ("urgency_oracle", "UrgOrc"),
                   ("mct", "Mct")]:
        q = get(g, policy=p, utilisation_target=float(hi))
        if q is not None:
            MAC[f"Ec{key}UrgMiss"] = f"{100 * q['urgent_miss_rate']:.1f}"
            MAC[f"Ec{key}StratMiss"] = f"{100 * q['strategic_miss_rate']:.1f}"
            MAC[f"Ec{key}UrgPninetyfive"] = f"{q['urgent_resp_p95_s'] / 60:.0f}"
    MAC["EcLoadHi"] = f"{int(100 * hi)}"
    lo = us[0]
    MAC["EcLoadLo"] = f"{int(100 * lo)}"
    for p_, key in [("urgency", "Urg"), ("fcfs", "Fcfs"), ("edf", "Edf"), ("edf_p", "Edfp"), ("mct", "Mct")]:
        q = get(g, policy=p_, utilisation_target=float(lo))
        if q is not None:
            MAC[f"Ec{key}UrgMissLo"] = f"{100 * q['urgent_miss_rate']:.1f}"
            MAC[f"Ec{key}StratMissLo"] = f"{100 * q['strategic_miss_rate']:.1f}"
    MAC["EcFarmsLo"] = f"{int(r[r.utilisation_target == lo].farms.mean()):,}"
    MAC["EcFarmsHi"] = f"{int(r[r.utilisation_target == hi].farms.mean()):,}"


# --------------------------------------------------------------------------- tuning (appendix)
def table_tuning(MAC):
    f = RAW / "tuning.json"
    kf = repo_path("results", "models", "kf-default", "kf_selection.json")
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Validation-year (2014) selection of the shrinkage weight $\kappa$ of the polling index: validation RMSE relative to the best $\kappa$ for each seed and budget, averaged over five seeds and $K\in\{1,2,4,8\}$ (lower is better).}",
         r"\label{tab:tuning}", r"\begin{tabular}{lccccc}", r"\toprule"]
    if f.exists():
        t = json.load(open(f))
        ks = sorted(t["kappa_scores"], key=float)
        L.append(r"$\kappa$ & " + " & ".join(f"{float(k):g}" for k in ks) + r" \\")
        L.append(r"\midrule")
        L.append(r"Relative RMSE & " + " & ".join(f"{t['kappa_scores'][k]:.4f}" for k in ks) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (T / "tuning.tex").write_text(fit_width("\n".join(L) + "\n"))
    if kf.exists():
        k = json.load(open(kf))
        MAC["KfShrink"] = f"{k['shrink']:g}"
        MAC["KfRscale"] = f"{k['r_scale']:g}"


# --------------------------------------------------------------------------- per-depth and regime errors
def table_depth(MAC):
    est, pol, kfp = read("estimators"), read("policies"), read("kf_pipeline")
    if est.empty:
        return
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Test RMSE ($10^{-3}$\,m$^3$\,m$^{-3}$) by sensor depth and by weather regime at $K=2$ (wet: $\ge5$~mm of rain in the previous 24~h; dry: $<1$~mm), and RMSE of the 24-h look-ahead outlook. Mean of five seeds.}",
         r"\label{tab:depth}", r"\begin{tabular}{l" + "c" * 5 + "|cc|c}", r"\toprule",
         r"Pipeline & 30 cm & 60 cm & 90 cm & 120 cm & 150 cm & Wet & Dry & Outlook \\", r"\midrule"]
    rows, nm = [], []
    for df, e, p, lab in [(est, "locf", "rr", "LOCF + RR"), (kfp, "kf", "kfgreedy", "Rain-KF + KF-greedy"),
                          (est, "gwn", "rr", "Graph WaveNet + RR"), (est, "vigil", "rr", r"\vigil{} + RR"),
                          (pol, "vigil", "voi", r"\vigil{} + VoI")]:
        r = get(df, estimator=e, policy=p, k=2) if not df.empty else None
        if r is None:
            continue
        row = [1e3 * r[f"rmse_d{d}"] for d in range(5)] + [1e3 * r["rmse_wet"], 1e3 * r["rmse_dry"]]
        row.append(1e3 * r["outlook24_rmse"] if "outlook24_rmse" in r and pd.notna(r["outlook24_rmse"]) else None)
        rows.append(row)
        nm.append(lab)
    cells = fmt_cells(rows, [2] * 8, [True] * 8)
    for n_, c in zip(nm, cells):
        L.append(f"{n_} & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (T / "depth.tex").write_text(fit_width("\n".join(L) + "\n"))
    v = get(pol, estimator="vigil", policy="voi", k=2) if not pol.empty else None
    if v is not None and "outlook24_rmse" in v:
        MAC["OutlookRmseTwo"] = f"{1e3 * v['outlook24_rmse']:.2f}"
        MAC["OutlookPersistTwo"] = f"{1e3 * v['outlook24_rmse_persist']:.2f}"


def macros_tradeoff(MAC):
    tr, pol = read("tradeoff"), read("policies")
    if tr.empty:
        return
    q = pd.concat([tr, pol[pol.policy == "voi"]]) if not pol.empty else tr
    names = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQuarter", 1.0: "Full"}
    for k in (2, 4):
        base = get(q, estimator="vigil", policy="voi", k=k, eta=0.0)
        for eta, nm in names.items():
            r = get(q, estimator="vigil", policy="voi", k=k, eta=float(eta))
            if r is None:
                continue
            MAC[f"Tr{W[k]}{nm}Rmse"] = f"{1e3 * r['rmse']:.2f}"
            MAC[f"Tr{W[k]}{nm}Rec"] = f"{100 * r['ev_head_recall_6h']:.1f}"
            MAC[f"Tr{W[k]}{nm}Delay"] = f"{r['ev_head_median_delay_h']:.1f}"
            if base is not None:
                MAC[f"Tr{W[k]}{nm}RmseVsZero"] = f"{100 * (r['rmse'] / base['rmse'] - 1):.1f}"
                MAC[f"Tr{W[k]}{nm}RmseVsZeroAbs"] = f"{abs(100 * (r['rmse'] / base['rmse'] - 1)):.1f}"


def macros_calib(MAC):
    est, pol = read("estimators"), read("policies")
    for nm, df, e, p_ in [("Voi", pol, "vigil", "voi"), ("VigilRr", est, "vigil", "rr"), ("Kf", est, "kf", "rr")]:
        for k in (2, 4):
            r = get(df, estimator=e, policy=p_, k=k) if not df.empty else None
            if r is None or not np.isfinite(r.get("cov90_unobs", np.nan)):
                continue
            MAC[f"{nm}CovNinety{W[k]}"] = f"{100 * r['cov90_unobs']:.0f}"
            MAC[f"{nm}CovFifty{W[k]}"] = f"{100 * r['cov50_unobs']:.0f}"
            MAC[f"{nm}CrpsVal{W[k]}"] = f"{1e3 * r['crps_unobs']:.3f}"
            MAC[f"{nm}SigmaScale{W[k]}"] = f"{r['sigma_scale']:.2f}"


def macros_regime(MAC):
    est, pol = read("estimators"), read("policies")
    for k in (2, 4):
        a = get(pol, estimator="vigil", policy="voi", k=k, eta=0.5)
        b = get(est, estimator="locf", policy="rr", k=k)
        c = get(est, estimator="vigil", policy="rr", k=k)
        if a is None or b is None:
            continue
        for reg, nm in [("wet", "Wet"), ("dry", "Dry")]:
            MAC[f"Voi{nm}GainLocf{W[k]}"] = f"{100 * (1 - a[f'rmse_{reg}'] / b[f'rmse_{reg}']):.1f}"
            if c is not None:
                MAC[f"Voi{nm}GainRr{W[k]}"] = f"{100 * (1 - a[f'rmse_{reg}'] / c[f'rmse_{reg}']):.1f}"
        MAC["WetHoursShare"] = f"{100 * a['hours_wet'] / (a['hours_wet'] + a['hours_dry']):.0f}"


def run(MAC):
    T.mkdir(parents=True, exist_ok=True)
    for fn in [table_ablation, table_weather, table_stats, table_costs, table_edgecloud, table_tuning, table_depth,
               macros_tradeoff, macros_calib, macros_regime]:
        try:
            fn(MAC)
        except Exception as ex:  # keep going; report
            import traceback
            print("FAILED", fn.__name__)
            traceback.print_exc()
