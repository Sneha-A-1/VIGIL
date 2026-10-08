"""Discrete-event simulation of urgency-aware processing of VIGIL analytics on co-tenant cloud hosts.

Hosts are virtual machines whose *residual* CPU capacity (provisioned minus used, in reference cores)
follows real GWA-T-12 Bitbrains traces. A host runs one task at a time and a task may use all of the
residual capacity (moldable data-parallel analytics); the completion time of a task with work d
(reference core-seconds) started at s solves  W_h(t) - W_h(s) = d,  W_h(t) = int_0^t R_h(u) du.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

STEP = 300.0


@dataclass
class Tasks:
    arrival: np.ndarray    # [n] s
    work: np.ndarray       # [n] reference core-seconds
    deadline: np.ndarray   # [n] absolute s
    urgent: np.ndarray     # [n] bool
    prio: np.ndarray       # [n] event probability (urgent) or 0
    kind: np.ndarray       # [n] int task type id


class Hosts:
    def __init__(self, residual: np.ndarray, horizon_s: float):
        """residual: [H, T5] residual cores per 5-min slot (tiled cyclically to the horizon)."""
        H, T5 = residual.shape
        n = int(np.ceil(horizon_s / STEP)) + 2
        self.R = residual[:, np.arange(n) % T5].astype(np.float64)
        self.cumW = np.concatenate([np.zeros((H, 1)), np.cumsum(self.R * STEP, axis=1)], axis=1)
        self.H, self.n = H, n

    def work_at(self, h: int, t: float) -> float:
        k = min(int(t // STEP), self.n - 1)
        return self.cumW[h, k] + self.R[h, k] * (t - k * STEP)

    def finish(self, h: int, start: float, work: float) -> float:
        target = self.work_at(h, start) + work
        j = int(np.searchsorted(self.cumW[h], target, side="right")) - 1
        if j >= self.n:
            return float("inf")
        r = self.R[h, j]
        if r <= 0:
            return float("inf")
        return j * STEP + (target - self.cumW[h, j]) / r

    def rate(self, h: int, t: float) -> float:
        return self.R[h, min(int(t // STEP), self.n - 1)]


def simulate(tasks: Tasks, hosts: Hosts, policy: str, cap_forecast=None, horizon_slots: int = 3) -> dict:
    """policy in {'fcfs', 'edf', 'mct', 'edf_p', 'urgency'}.

    cap_forecast(h, t) -> predicted mean residual cores of host h over the next ``horizon_slots``
    slots (used by 'urgency' for placement; default: current capacity).
    """
    n = len(tasks.arrival)
    order = np.argsort(tasks.arrival, kind="stable")
    remaining = tasks.work.astype(np.float64).copy()
    start_t = np.full(n, np.nan)
    finish_t = np.full(n, np.nan)
    first_start = np.full(n, np.nan)
    busy = np.full(hosts.H, -1, dtype=np.int64)      # running task per host
    run_start = np.zeros(hosts.H)
    version = np.zeros(hosts.H, dtype=np.int64)
    events = []                                       # (time, seq, type, payload)
    seq = 0
    for i in order:
        heapq.heappush(events, (float(tasks.arrival[i]), seq, 0, int(i)))
        seq += 1
    queue = []                                        # global priority queue (list policies)
    host_q = [[] for _ in range(hosts.H)]             # per-host FIFO queues (mct)
    host_free_est = np.zeros(hosts.H)
    preemptive = policy in ("edf_p", "urgency")

    def key(i):
        if policy == "fcfs":
            return (tasks.arrival[i], i)
        if policy in ("edf", "edf_p"):
            return (tasks.deadline[i], i)
        if policy == "urgency":
            return (0 if tasks.urgent[i] else 1, -tasks.prio[i] if tasks.urgent[i] else tasks.deadline[i], i)
        raise ValueError(policy)

    def pick_host(t, idle):
        if policy == "urgency" and cap_forecast is not None:
            return max(idle, key=lambda h: cap_forecast(h, t))
        return max(idle, key=lambda h: hosts.rate(h, t))

    def start(h, i, t):
        nonlocal seq
        busy[h] = i
        run_start[h] = t
        if np.isnan(first_start[i]):
            first_start[i] = t
        version[h] += 1
        f = hosts.finish(h, t, remaining[i])
        heapq.heappush(events, (f, seq, 1, (h, int(version[h]))))
        seq += 1

    def dispatch(t):
        if policy == "mct":
            for h in range(hosts.H):
                if busy[h] < 0 and host_q[h]:
                    start(h, host_q[h].pop(0), t)
            return
        idle = [h for h in range(hosts.H) if busy[h] < 0]
        while idle and queue:
            _, i = heapq.heappop(queue)
            h = pick_host(t, idle)
            idle.remove(h)
            start(h, i, t)

    while events:
        t, _, typ, payload = heapq.heappop(events)
        if t == float("inf"):
            break
        if typ == 0:                                  # arrival
            i = payload
            if policy == "mct":
                est = [max(host_free_est[h], t) + remaining[i] / max(hosts.rate(h, t), 1e-3) for h in range(hosts.H)]
                h = int(np.argmin(est))
                host_free_est[h] = est[h]
                host_q[h].append(i)
            else:
                heapq.heappush(queue, (key(i), i))
                if preemptive and tasks.urgent[i] and all(busy >= 0):
                    # preempt the running non-urgent task with the latest deadline
                    cand = [h for h in range(hosts.H) if not tasks.urgent[busy[h]]]
                    if cand:
                        h = max(cand, key=lambda hh: tasks.deadline[busy[hh]])
                        j = busy[h]
                        done = hosts.work_at(h, t) - hosts.work_at(h, run_start[h])
                        remaining[j] = max(remaining[j] - done, 0.0)
                        busy[h] = -1
                        version[h] += 1
                        heapq.heappush(queue, (key(j), j))
            dispatch(t)
        else:                                         # completion
            h, v = payload
            if v != version[h]:
                continue
            i = busy[h]
            finish_t[i] = t
            remaining[i] = 0.0
            busy[h] = -1
            dispatch(t)

    resp = finish_t - tasks.arrival
    miss = ~(finish_t <= tasks.deadline)
    out = {}
    for name, m in [("urgent", tasks.urgent), ("strategic", ~tasks.urgent)]:
        if m.any():
            r = resp[m]
            fin = np.isfinite(r)
            out[f"{name}_n"] = int(m.sum())
            out[f"{name}_miss_rate"] = float(miss[m].mean())
            out[f"{name}_resp_mean_s"] = float(np.nanmean(np.where(fin, r, np.nan)))
            out[f"{name}_resp_p95_s"] = float(np.nanpercentile(np.where(fin, r, np.nan), 95))
            out[f"{name}_wait_mean_s"] = float(np.nanmean(first_start[m] - tasks.arrival[m]))
    hi = tasks.urgent & (tasks.prio >= np.median(tasks.prio[tasks.urgent])) if tasks.urgent.any() else tasks.urgent
    if hi.any():
        out["urgent_hi_miss_rate"] = float(miss[hi].mean())
        out["urgent_hi_resp_p95_s"] = float(np.nanpercentile(np.where(np.isfinite(resp[hi]), resp[hi], np.nan), 95))
    out["unfinished"] = int(np.isnan(finish_t).sum())
    return out
