"""Edge-cloud case study: urgency-aware processing of VIGIL analytics for a regional fleet of farms on
co-tenant cloud VMs whose residual capacity follows the GWA-T-12 Bitbrains traces.

    python scripts/run_edgecloud.py

Inputs : results/raw/eval/policies/*.npz   (alert onsets of the VIGIL test runs)
         results/raw/costs.json            (measured task costs)
         results/raw/capacity/*            (Bitbrains grid and the VM split of run_capacity.py)
Output : results/raw/edgecloud/results.json
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from vigil.config import load_config, repo_path
from vigil.data.dataset import load
from vigil.sim.edgecloud import Hosts, Tasks, simulate

REF_MHZ = 2100.0   # clock of the CPU on which the task costs were measured (scripts/measure_costs.py)
CFG = {
    "window": ["2015-12-01 00:00", "2016-02-29 23:00"],  # core of the wet season of the test period
    "farms_per_region": 500,
    "advisory_batch": 50,          # farms per urgent advisory job
    "hosts": 6,
    "urgent_deadline_s": 120.0,
    "map_deadline_s": 3600.0,
    "outlook_deadline_s": 7200.0,
    "refresh_deadline_s": 86400.0,
    "utilisation": [0.3, 0.5, 0.7, 0.85],
    # urgency: placement on the idle host with the largest *current* residual capacity (a GRU forecaster
    # did not beat persistence at the 15-min horizon, see run_capacity.py); urgency_oracle: placement with
    # the true mean capacity of the next 15 min (not deployable, upper reference)
    "policies": ["fcfs", "edf", "mct", "edf_p", "urgency", "urgency_oracle"],
    "seeds": [0, 1, 2],
}


def pick_hosts(grid: dict, test_vms: list, n: int) -> np.ndarray:
    """Busy and variable co-tenant VMs (>= 4 cores) among the forecaster's held-out VMs."""
    ids = list(grid["vm_ids"])
    idx = np.array([ids.index(v) for v in test_vms])
    use = grid["use_pct"][idx]
    ok = (grid["cores"][idx] >= 4) & (use.mean(1) > 5)
    cand = idx[ok]
    score = grid["use_pct"][cand].std(1)
    return cand[np.argsort(-score)[:n]]


def residual_cores(grid: dict, hosts: np.ndarray, use_pct: np.ndarray | None = None) -> np.ndarray:
    use = grid["use_pct"][hosts] if use_pct is None else use_pct
    speed = grid["cap_mhz"][hosts] / np.maximum(grid["cores"][hosts], 1) / REF_MHZ
    return np.clip(grid["cores"][hosts][:, None] * (1 - use / 100.0), 0, None) * speed[:, None]


def future_capacity(grid: dict, hosts: np.ndarray, horizon=3) -> np.ndarray:
    """True mean residual capacity of the next ``horizon`` 5-min slots (oracle placement reference)."""
    use = grid["use_pct"][hosts]
    fut = np.stack([pd.Series(u).rolling(horizon, min_periods=1).mean().shift(-(horizon - 1)).bfill().to_numpy()
                    for u in use])
    return residual_cores(grid, hosts, fut)


def alert_streams(window, data) -> list[np.ndarray]:
    """Per VIGIL test run (policy 'voi', K=2): alert onset times (s from window start) and probabilities."""
    out = []
    files = sorted(glob.glob(str(repo_path("results", "raw", "eval", "policies", "vigil__voi_*__K2__s*.npz"))))
    a = data.split["test"][0]
    w0 = data.index.searchsorted(pd.Timestamp(window[0])) - a
    w1 = data.index.searchsorted(pd.Timestamp(window[1])) - a
    for f in files:
        z = np.load(f, allow_pickle=True)
        t, p = z["alerts_head_t"], z["alerts_head_p"]
        m = (t >= w0) & (t < w1)
        out.append(np.stack([(t[m] - w0) * 3600.0, p[m]], 1))
    return out


def build_tasks(costs, streams, n_regions, hours, rng) -> Tasks:
    fpr = CFG["farms_per_region"]
    arr, work, dl, urg, pri, kind = [], [], [], [], [], []
    for g in range(n_regions):
        # urgent: alert advisories. An advisory is a 24-h probabilistic outlook of an alerting farm (one per farm
        # and hour, whatever the number of alerting channels). Storms hit the farms of a region together, so the
        # advisories of one region and hour are processed as jobs of at most ``advisory_batch`` farms whose
        # priority is the largest alert probability among them.
        recs = []
        for f in range(fpr):
            st = streams[rng.integers(len(streams))]
            if not len(st):
                continue
            jitter = rng.integers(0, 4) * 3600.0
            t = st[:, 0] + jitter + rng.uniform(0, 300, len(st))
            h = np.floor(t / 3600.0).astype(np.int64)
            for hh in np.unique(h):
                m = h == hh
                recs.append((int(hh), float(t[m].min()), float(st[m, 1].max())))
        if recs:
            R_ = np.array(recs)
            for hh in np.unique(R_[:, 0]):
                rows = R_[R_[:, 0] == hh]
                rows = rows[np.argsort(-rows[:, 2])]          # most probable alerts first
                for c in range(0, len(rows), CFG["advisory_batch"]):
                    chunk = rows[c:c + CFG["advisory_batch"]]
                    arr.append(float(chunk[:, 1].max()))
                    work.append(len(chunk) * costs["alert_outlook_24h_s"])
                    dl.append(CFG["urgent_deadline_s"])
                    urg.append(True)
                    pri.append(float(chunk[:, 2].max()))
                    kind.append(0)
        # strategic: hourly maps and daily 72-h outlooks per region
        for h in range(hours):
            arr.append(h * 3600.0 + rng.uniform(0, 300))
            work.append(fpr * costs["map_idw_s"])
            dl.append(CFG["map_deadline_s"])
            urg.append(False)
            pri.append(0.0)
            kind.append(1)
            if h % 24 == 5:
                arr.append(h * 3600.0 + rng.uniform(0, 300))
                work.append(fpr * costs["forecast_72h_s"])
                dl.append(CFG["outlook_deadline_s"])
                urg.append(False)
                pri.append(0.0)
                kind.append(2)
        # weekly model refresh (fine-tuning) per farm at a random time of each week
        for f in range(fpr):
            for wk in range(hours // 168):
                arr.append((wk * 168 + rng.uniform(0, 168)) * 3600.0)
                work.append(costs["refresh_s"])
                dl.append(CFG["refresh_deadline_s"])
                urg.append(False)
                pri.append(0.0)
                kind.append(3)
    arr = np.array(arr)
    return Tasks(arrival=arr, work=np.array(work), deadline=arr + np.array(dl), urgent=np.array(urg),
                 prio=np.array(pri), kind=np.array(kind))


def main():
    cap_dir = repo_path("results", "raw", "capacity")
    g = dict(np.load(cap_dir / "bitbrains_grid.npz"))
    split = json.load(open(cap_dir / "split.json"))
    hosts = pick_hosts(g, split["test_vms"], CFG["hosts"])
    R = residual_cores(g, hosts)
    fc = {"oracle": future_capacity(g, hosts)}
    costs = json.load(open(repo_path("results", "raw", "costs.json")))
    # weekly refresh = 300-iteration fine-tuning (cost measured for 25 iterations, scaled linearly)
    costs["refresh_s"] = costs["retrain_25it_s"] * 300 / 25
    data = load()
    streams = alert_streams(CFG["window"], data)
    hours = int((pd.Timestamp(CFG["window"][1]) - pd.Timestamp(CFG["window"][0])) / pd.Timedelta("1h")) + 1
    per_region = CFG["farms_per_region"] * (costs["refresh_s"] / 168 + costs["map_idw_s"]
                                            + costs["forecast_72h_s"] / 24) \
        + CFG["farms_per_region"] * np.mean([len(np.unique(np.floor(s[:, 0] / 3600))) for s in streams]) / hours \
        * costs["alert_outlook_24h_s"]
    cap_mean = R.mean(1).sum()
    results = []
    for rho in CFG["utilisation"]:
        n_regions = max(1, int(round(rho * cap_mean * 3600 / per_region)))
        for seed in CFG["seeds"]:
            rng = np.random.default_rng(seed)
            tasks = build_tasks(costs, streams, n_regions, hours, rng)
            horizon = hours * 3600.0 + 7 * 86400
            Hs = Hosts(R, horizon)
            slot = lambda t: min(int(t // 300), Hs.n - 1)
            for pol in CFG["policies"]:
                cf = None
                name = pol
                if pol == "urgency_oracle":
                    Rf = Hosts(fc["oracle"], horizon).R
                    cf = lambda h, t, Rf=Rf: Rf[h, slot(t)]
                    name = "urgency"
                r = simulate(tasks, Hs, name, cap_forecast=cf)
                r.update({"policy": pol, "utilisation_target": rho, "seed": seed, "regions": n_regions,
                          "farms": n_regions * CFG["farms_per_region"], "tasks": int(len(tasks.arrival))})
                results.append(r)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    out = repo_path("results", "raw", "edgecloud")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"config": CFG, "hosts": g["vm_ids"][hosts].tolist(), "mean_residual_cores": float(cap_mean),
               "costs": costs, "results": results}, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
