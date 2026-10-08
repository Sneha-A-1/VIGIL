"""Generate every results table (manuscript/tables/*.tex) and the results macros
(manuscript/macros/results.tex) from results/processed/*.csv and the raw JSON files.

    python scripts/aggregate.py && python scripts/stats_tests.py && python scripts/make_tables.py

All reported numbers are means over seeds (single numbers, no +/-); spreads are in *_std.csv.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import yaml

from vigil.config import load_config, repo_path


def fit_width(text: str) -> str:
    """Wrap every tabular in an adjustbox that shrinks it to the text width only if it is wider."""
    text = re.sub(r"(\\begin\{tabular\})", r"\\begin{adjustbox}{max width=\\textwidth}\n\1", text)
    text = re.sub(r"(?<=[&\s])-(?=\d)", r"$-$", text)      # typographic minus in table cells
    return text.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")

P = repo_path("results", "processed")
T = repo_path("manuscript", "tables")
MAC = {}
EST_NAMES = {"locf": "LOCF", "kf": "Rain-KF", "lgbm": "LightGBM", "mlp": "MLP", "gru": "GRU",
             "transformer": "Transformer", "dcrnn": "DCRNN", "gwn": "Graph WaveNet", "vigil": r"\vigil{}"}
POL_NAMES = {"rr": "Round-robin", "random": "Random", "ars": "ARS", "kfgreedy": "KF-greedy",
             "kfvoi": r"VoI index (KF variances)",
             "voi": r"\vigil{} VoI", "oracle": r"Oracle$^\dagger$"}
BUDGETS = [1, 2, 4, 8]


def read(name):
    f = P / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def num(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def bold_best(vals, lower=True, nd=None):
    """Mark the best value(s); with ``nd`` the comparison uses the displayed (rounded) values, so ties at
    the displayed precision are all marked."""
    arr = np.array([v if v is not None and np.isfinite(v) else (np.inf if lower else -np.inf) for v in vals], float)
    if nd is not None:
        arr = np.where(np.isfinite(arr), np.round(arr, nd), arr)
    best = arr.min() if lower else arr.max()
    return [abs(a - best) < 1e-12 for a in arr]


def cell(v, nd, b):
    s = num(v, nd)
    return rf"\textbf{{{s}}}" if b and s != "--" else s


def get(df, **kw):
    q = df
    for k, v in kw.items():
        if isinstance(v, float):
            q = q[np.isclose(q[k], v)]
        else:
            q = q[q[k] == v]
    return q.iloc[0] if len(q) else None


def mac(name, value):
    MAC[name] = value


# --------------------------------------------------------------------------- hyperparameters
def table_hyper():
    tr = load_config("train")
    t, v = tr["train"], tr["vigil"]
    params = {}
    for name in ["vigil", "mlp", "gru", "transformer", "dcrnn", "gwn"]:
        f = repo_path("results", "models", f"{name}-default", "seed0", "train_log.json")
        if f.exists():
            params[name] = json.load(open(f))["info"]["params"]
    rows = [
        ("Hidden size $H$ (\\vigil{}, GRU, DCRNN, Transformer)", f"{v['hidden']}"),
        ("Graph", f"Gaussian kNN ($k={load_config('data')['graph']['knn']}$) + adaptive adjacency (dim 8)"),
        ("Look-ahead horizon $L$", "24 h"),
        ("$\\beta$ of the $\\beta$-NLL, look-ahead weight $\\lambda_L$", f"{v['beta']}, {v['w_look']}"),
        ("Optimiser, learning rate, weight decay", f"Adam, {t['lr']}, $10^{{{int(np.log10(t['weight_decay']))}}}$ (cosine decay)"),
        ("Iterations $I$, batch $B$", f"{t['iters']}, {t['batch']} windows"),
        ("Window $W$, burn-in $W_0$", f"{t['window']} h, {t['warm']} h"),
        ("Gradient clipping", f"{t['clip']}"),
        ("Training budgets $K$", ", ".join(map(str, t['budgets']))),
        ("Training policy mix (random, RR, VoI)", f"{t['policy_mix']['random']}, {t['policy_mix']['rr']}, {t['policy_mix']['voi']} (VoI after {t['closed_loop_start']} it.)"),
        ("Checkpoint selection", f"validation MSE (32 windows of 2014), from iteration {t['select_after']}"),
        ("Freshness guarantee $A_{\\max}$", f"{t['max_age']} h"),
        ("Event threshold $\\delta$", "0.02 m$^3$\\,m$^{-3}$ within 24 h"),
    ]
    if params:
        rows.append(("Trainable parameters", ", ".join(f"{EST_NAMES.get(k, k)} {v:,}" for k, v in params.items())))
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\caption{Training and model settings (identical for \vigil{} and the neural baselines unless stated).}",
             r"\label{tab:hyper}", r"\begin{tabular}{p{5.2cm}p{7.2cm}}", r"\toprule", r"Setting & Value \\", r"\midrule"]
    lines += [rf"{a} & {b} \\" for a, b in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (T / "hyperparams.tex").write_text(fit_width("\n".join(lines) + "\n"))


# --------------------------------------------------------------------------- E1 estimators
def table_estimators():
    df = read("estimators")
    if df.empty:
        return
    ests = [e for e in EST_NAMES if e in set(df.estimator)]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Estimators under the same round-robin polling schedule (test period, mean of five seeds). Field-state RMSE in $10^{-3}$\,m$^3$\,m$^{-3}$ over all channel-hours for each budget $K$ (polls per hour); MAE, unpolled-channel RMSE, CRPS (probabilistic estimators) and urgent detection at $K=\KMain$: AUPRC of the hourly event scores, share of onsets detected within 6~h (R@6h) and false alerts per day (FA/d) at the validation-selected threshold. Best values in bold.}",
             r"\label{tab:estimators}", r"\begin{tabular}{l" + "c" * 4 + "|cccccc}", r"\toprule",
             r" & \multicolumn{4}{c|}{RMSE ($10^{-3}$)} & \multicolumn{6}{c}{$K=\KMain$} \\",
             r"Estimator & $K{=}1$ & $K{=}2$ & $K{=}4$ & $K{=}8$ & MAE & RMSE$_{\text{unpol}}$ & CRPS & AUPRC & R@6h (\%) & FA/d \\",
             r"\midrule"]
    km = int(MAC.get("KMain", 4))
    vals = {}
    for e in ests:
        row = []
        for k in BUDGETS:
            r = get(df, estimator=e, policy="rr", k=k)
            row.append(None if r is None else 1e3 * r["rmse"])
        r = get(df, estimator=e, policy="rr", k=km)
        score = "head" if (r is not None and "ev_head_auprc" in r and pd.notna(r.get("ev_head_auprc"))) else "rule"
        row += [None if r is None else 1e3 * r["mae"], None if r is None else 1e3 * r["rmse_unobs"],
                None if r is None or pd.isna(r.get("crps_unobs", np.nan)) else 1e3 * r["crps_unobs"],
                None if r is None else r[f"ev_{score}_auprc"], None if r is None else 100 * r[f"ev_{score}_recall_6h"],
                None if r is None else r[f"ev_{score}_false_alerts_per_day"]]
        vals[e] = row
    cols = list(zip(*vals.values()))
    lower = [True] * 7 + [False, False, True]
    nds = [2] * 6 + [3, 3, 1, 2]
    bests = [bold_best(c, lo, nd) for c, lo, nd in zip(cols, lower, nds)]
    for i, e in enumerate(vals):
        cells = [cell(v, nds[j], bests[j][i]) for j, v in enumerate(vals[e])]
        lines.append(f"{EST_NAMES[e]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (T / "estimators.tex").write_text(fit_width("\n".join(lines) + "\n"))
    for k in BUDGETS:
        r_l = get(df, estimator="locf", policy="rr", k=k)
        r_v = get(df, estimator="vigil", policy="rr", k=k)
        if r_l is not None and r_v is not None:
            mac(f"RrVigilVsLocf{['One','Two','Four','Eight'][BUDGETS.index(k)]}", f"{100 * (1 - r_v['rmse'] / r_l['rmse']):.1f}")
    learned = [e for e in ["lgbm", "mlp", "gru", "transformer", "dcrnn", "gwn", "vigil"] if e in ests]
    words = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
    for k in BUDGETS:
        r_l = get(df, estimator="locf", policy="rr", k=k)
        g = []
        for e in learned:
            r = get(df, estimator=e, policy="rr", k=k)
            if r is not None and r_l is not None:
                g.append(100 * (1 - r["rmse"] / r_l["rmse"]))
        if g:
            mac(f"EstGainMin{words[k]}", f"{min(g):.1f}")
            mac(f"EstGainMax{words[k]}", f"{max(g):.1f}")
            mac(f"EstGainMin{words[k]}Abs", f"{abs(min(g)):.1f}")
            mac(f"EstLossMax{words[k]}", f"{max(0.0, -min(g)):.1f}")
            mac(f"EstLossMin{words[k]}", f"{max(0.0, -max(g)):.1f}")
    r_l = get(df, estimator="locf", policy="rr", k=km)
    ratios = [get(df, estimator=e, policy="rr", k=km)["mae"] / r_l["mae"] for e in learned
              if get(df, estimator=e, policy="rr", k=km) is not None]
    if ratios:
        mac("NeuralMaeRatioMin", f"{min(ratios):.1f}")
        mac("NeuralMaeRatioMax", f"{max(ratios):.1f}")
    best_base = None
    for e in ests:
        if e == "vigil":
            continue
        r = get(df, estimator=e, policy="rr", k=km)
        if r is not None and (best_base is None or r["rmse"] < best_base[1]):
            best_base = (e, r["rmse"])
    if best_base:
        mac("BestBaselineRr", EST_NAMES[best_base[0]])
        mac("BestBaselineRrRmse", f"{1e3 * best_base[1]:.2f}")


# --------------------------------------------------------------------------- E2 policies / pipelines
def pipelines():
    out = []
    est = read("estimators")
    pol = read("policies")
    kfp = read("kf_pipeline")
    kfv = read("kf_voi")
    ars = read("baseline_ars")
    for e, p, df in [("locf", "rr", est), ("lgbm", "rr", est), ("gwn", "rr", est), ("vigil", "rr", est),
                     ("vigil", "random", pol), ("vigil", "ars", pol), ("lgbm", "ars", ars), ("gwn", "ars", ars),
                     ("kf", "kfgreedy", kfp), ("kf", "kfvoi", kfv), ("vigil", "voi", pol), ("vigil", "oracle", pol)]:
        if df is not None and not df.empty:
            out.append((e, p, df))
    return out


def table_policies():
    pl = pipelines()
    if not pl:
        return
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Complete retrieval pipelines (estimator + polling policy) on the test period (mean of five seeds; KF-greedy is deterministic). Field-state RMSE in $10^{-3}$\,m$^3$\,m$^{-3}$ for each budget; urgent wetting-onset detection at $K=2$ and $K=4$: share of onsets detected within 6~h (R@6h, \%), median delay of detected onsets (h) and false alerts per day (FA/d; alert episodes that overlap no event within $\pm12$~h, over all 210 channels), each at the pipeline's own validation-selected alert threshold. $^\dagger$The oracle uses hidden ground truth and is not deployable. Best deployable values in bold.}",
             r"\label{tab:policies}", r"\begin{tabular}{ll" + "c" * 4 + "|ccc|ccc}", r"\toprule",
             r" & & \multicolumn{4}{c|}{RMSE ($10^{-3}$)} & \multicolumn{3}{c|}{$K=2$} & \multicolumn{3}{c}{$K=4$} \\",
             r"Estimator & Policy & $K{=}1$ & $K{=}2$ & $K{=}4$ & $K{=}8$ & R@6h & Delay & FA/d & R@6h & Delay & FA/d \\",
             r"\midrule"]
    vals, names = [], []
    for e, p, df in pl:
        row = []
        for k in BUDGETS:
            r = get(df, estimator=e, policy=p, k=k)
            row.append(None if r is None else 1e3 * r["rmse"])
        for k in (2, 4):
            r = get(df, estimator=e, policy=p, k=k)
            sc = "head" if (r is not None and pd.notna(r.get("ev_head_recall_6h", np.nan))) else "rule"
            row += [None if r is None else 100 * r[f"ev_{sc}_recall_6h"], None if r is None else r[f"ev_{sc}_median_delay_h"],
                    None if r is None else r[f"ev_{sc}_false_alerts_per_day"]]
        vals.append(row)
        names.append((e, p))
    deploy = [i for i, (e, p) in enumerate(names) if p != "oracle"]
    lower = [True] * 4 + [False, True, True, False, True, True]
    nds = [2] * 4 + [1, 1, 2, 1, 1, 2]
    bests = []
    for j in range(len(lower)):
        col = [vals[i][j] if i in deploy else None for i in range(len(vals))]
        bests.append(bold_best(col, lower[j], nds[j]) if any(v is not None for v in col) else [False] * len(vals))
    for i, ((e, p), row) in enumerate(zip(names, vals)):
        if p == "oracle":
            lines.append(r"\midrule")
        cells = [cell(v, nds[j], bests[j][i] and i in deploy) for j, v in enumerate(row)]
        lines.append(f"{EST_NAMES[e]} & {POL_NAMES[p]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (T / "policies.tex").write_text(fit_width("\n".join(lines) + "\n"))


# --------------------------------------------------------------------------- headline macros
def equiv_budget(curve: dict, target: float) -> float | None:
    """Budget at which a policy with RMSE curve {K: rmse} reaches ``target`` (log-log interpolation)."""
    ks = sorted(curve)
    for k0, k1 in zip(ks[:-1], ks[1:]):
        r0, r1 = curve[k0], curve[k1]
        if (r0 - target) * (r1 - target) <= 0 and r0 != r1:
            w = (np.log(target) - np.log(r0)) / (np.log(r1) - np.log(r0))
            return float(np.exp(np.log(k0) + w * (np.log(k1) - np.log(k0))))
    return None


def headline_macros():
    est, pol = read("estimators"), read("policies")
    kfp, ars = read("kf_pipeline"), read("baseline_ars")
    if est.empty or pol.empty:
        return
    km = int(MAC.get("KMain", 4))
    words = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
    rr_curve = {}
    for k in BUDGETS:
        r = get(est, estimator="vigil", policy="rr", k=k)
        if r is not None:
            rr_curve[k] = r["rmse"]
    locf_curve = {k: get(est, estimator="locf", policy="rr", k=k)["rmse"] for k in BUDGETS
                  if get(est, estimator="locf", policy="rr", k=k) is not None}
    for k in BUDGETS:
        w = words[k]
        rr = get(est, estimator="vigil", policy="rr", k=k)
        rl = get(est, estimator="locf", policy="rr", k=k)
        vo = get(pol, estimator="vigil", policy="voi", k=k, eta=0.5)
        if vo is None or rl is None:
            continue
        mac(f"VoiRmse{w}", f"{1e3 * vo['rmse']:.2f}")
        mac(f"LocfRrRmse{w}", f"{1e3 * rl['rmse']:.2f}")
        mac(f"VoiVsLocfGain{w}", f"{100 * (1 - vo['rmse'] / rl['rmse']):.1f}")
        mac(f"VoiCrps{w}", f"{1e3 * vo['crps_unobs']:.3f}")
        if rr is not None:
            mac(f"VigilRrRmse{w}", f"{1e3 * rr['rmse']:.2f}")
            mac(f"VoiVsRrGain{w}", f"{100 * (1 - vo['rmse'] / rr['rmse']):.1f}")
            mac(f"VigilRrVsLocfGain{w}", f"{100 * (1 - rr['rmse'] / rl['rmse']):.1f}")
            mac(f"VigilRrRecallSix{w}", f"{100 * rr['ev_head_recall_6h']:.1f}")
            mac(f"VigilRrMae{w}", f"{1e3 * rr['mae']:.2f}")
            mac(f"VigilRrFalseAlerts{w}", f"{rr['ev_head_false_alerts_per_day']:.2f}")
        mac(f"LocfRrMae{w}", f"{1e3 * rl['mae']:.2f}")
        mac(f"VoiMae{w}", f"{1e3 * vo['mae']:.2f}")
        for nm, df, e, p_ in [("Oracle", pol, "vigil", "oracle"), ("Ars", pol, "vigil", "ars"),
                              ("Random", pol, "vigil", "random"), ("Kfg", kfp, "kf", "kfgreedy"),
                              ("Kfvoi", read("kf_voi"), "kf", "kfvoi"),
                              ("GwnArs", ars, "gwn", "ars"), ("LgbmArs", ars, "lgbm", "ars")]:
            r = get(df, estimator=e, policy=p_, k=k) if not df.empty else None
            if r is not None:
                sc_ = "head" if pd.notna(r.get("ev_head_recall_6h", np.nan)) else "rule"
                mac(f"{nm}FalseAlerts{w}", f"{r[f'ev_{sc_}_false_alerts_per_day']:.2f}")
                mac(f"{nm}Rmse{w}", f"{1e3 * r['rmse']:.2f}")
                mac(f"{nm}VsRrPct{w}", f"{100 * (r['rmse'] / rl['rmse'] - 1):.1f}")
                sc = "head" if pd.notna(r.get("ev_head_recall_6h", np.nan)) else "rule"
                mac(f"{nm}RecallSix{w}", f"{100 * r[f'ev_{sc}_recall_6h']:.1f}")
        kv = read("kf_voi")
        rkv = get(kv, estimator="kf", policy="kfvoi", k=k) if not kv.empty else None
        if rkv is not None:
            mac(f"VoiVsKfvoiGain{w}", f"{100 * (1 - vo['rmse'] / rkv['rmse']):.1f}")
            mac(f"VoiVsKfvoiGain{w}Abs", f"{abs(100 * (1 - vo['rmse'] / rkv['rmse'])):.1f}")
            mac(f"KfvoiVsLocfGain{w}", f"{100 * (1 - rkv['rmse'] / rl['rmse']):.1f}")
            share = (rl["rmse"] - rkv["rmse"]) / (rl["rmse"] - vo["rmse"]) if rl["rmse"] > vo["rmse"] else float("nan")
            mac(f"KfvoiShareOfGain{w}", f"{100 * share:.0f}")
        orc = get(pol, estimator="vigil", policy="oracle", k=k)
        if orc is not None and rr is not None:
            gap = (rr["rmse"] - orc["rmse"])
            if gap > 0:
                mac(f"VoiOracleShare{w}", f"{100 * (rr['rmse'] - vo['rmse']) / gap:.0f}")
        for nm, rr_, sc in [("Voi", vo, "head"), ("Rr", rl, "rule")]:
            mac(f"{nm}RecallSix{w}", f"{100 * rr_[f'ev_{sc}_recall_6h']:.1f}")
            mac(f"{nm}RecallDay{w}", f"{100 * rr_[f'ev_{sc}_recall_24h']:.1f}")
            mac(f"{nm}Delay{w}", f"{rr_[f'ev_{sc}_median_delay_h']:.1f}")
            mac(f"{nm}Auprc{w}", f"{rr_[f'ev_{sc}_auprc']:.3f}")
            mac(f"{nm}AlertPrec{w}", f"{100 * rr_[f'ev_{sc}_alert_precision']:.1f}")
            mac(f"{nm}FalseAlerts{w}", f"{rr_[f'ev_{sc}_false_alerts_per_day']:.2f}")
        eq = equiv_budget(locf_curve, vo["rmse"])
        if eq is not None:
            mac(f"VoiEquivK{w}", f"{eq:.1f}")
            mac(f"VoiBudgetSaving{w}", f"{100 * (1 - k / eq):.0f}")
    w = words[km]
    for a_, b_ in [("VoiVsLocfGainKmain", f"VoiVsLocfGain{w}"), ("VoiVsRrGainKmain", f"VoiVsRrGain{w}"),
                   ("VoiRecallSixKmain", f"VoiRecallSix{w}"), ("RrRecallSixKmain", f"RrRecallSix{w}"),
                   ("VoiEquivKKmain", f"VoiEquivK{w}"), ("VoiBudgetSavingKmain", f"VoiBudgetSaving{w}")]:
        if b_ in MAC:
            MAC[a_] = MAC[b_]
    gains = [float(MAC[f"VoiVsLocfGain{words[k]}"]) for k in BUDGETS if f"VoiVsLocfGain{words[k]}" in MAC]
    if gains:
        mac("VoiVsLocfGainMin", f"{min(gains):.1f}")
        mac("VoiVsLocfGainMax", f"{max(gains):.1f}")
    gains = [float(MAC[f"VoiVsRrGain{words[k]}"]) for k in BUDGETS if f"VoiVsRrGain{words[k]}" in MAC]
    if gains:
        mac("VoiVsRrGainMin", f"{min(gains):.1f}")
        mac("VoiVsRrGainMax", f"{max(gains):.1f}")


def write_macros():
    import re
    bad = [k for k in MAC if not re.fullmatch(r"[A-Za-z]+", k)]
    assert not bad, f"invalid LaTeX macro names: {bad}"
    out = repo_path("manuscript", "macros")
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())]
    (out / "results.tex").write_text("% generated by scripts/make_tables.py\n" + "\n".join(lines) + "\n")


def main():
    T.mkdir(parents=True, exist_ok=True)
    tun = repo_path("results", "raw", "tuning.json")
    mac("KappaStar", f"{json.load(open(tun))['kappa']:g}" if tun.exists() else "0.5")
    mac("KMain", "4")
    table_hyper()
    table_estimators()
    table_policies()
    headline_macros()
    import importlib.util, sys
    extra = repo_path("scripts", "make_tables_extra.py")
    if extra.exists():
        spec = importlib.util.spec_from_file_location("mte", extra)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run(MAC)
    write_macros()
    print(json.dumps(MAC, indent=1))


if __name__ == "__main__":
    main()
