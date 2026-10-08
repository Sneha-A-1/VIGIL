"""Retrieval (polling) policies. Every policy returns a score per logger; the top-K are polled,
subject to the freshness guard (loggers not polled for ``max_age`` hours are served first)."""
from __future__ import annotations

import torch

from vigil.sim.retrieval import Memory


def voi_score(sigma: torch.Tensor, p_event: torch.Tensor, eta: torch.Tensor | float,
              avail: torch.Tensor | None = None, floor_l: torch.Tensor | None = None,
              floor_e: torch.Tensor | None = None) -> torch.Tensor:

    q = torch.ones_like(sigma) if avail is None else avail
    var_l = sigma ** 2
    var_e = p_event * (1.0 - p_event)
    if floor_l is not None:
        var_l = (var_l - floor_l).clamp(min=0.0)
    if floor_e is not None:
        var_e = (var_e - floor_e).clamp(min=0.0)
    v_state = (q * var_l).mean(-1)
    v_event = (q * var_e).mean(-1)
    v_state = v_state / v_state.mean(dim=1, keepdim=True).clamp(min=1e-12)
    v_event = v_event / v_event.mean(dim=1, keepdim=True).clamp(min=1e-12)
    eta = torch.as_tensor(eta, dtype=sigma.dtype)
    if eta.dim() == 1:
        eta = eta[:, None]
    return (1.0 - eta) * v_state + eta * v_event


def urgent_mass(mu: torch.Tensor, sigma: torch.Tensor, last_x: torch.Tensor, delta: torch.Tensor,
                avail: torch.Tensor | None = None) -> torch.Tensor:
    """Expected number of *undetected* wetting increments at each logger: sum over depths of
    P(x_t - x_last >= delta) under the network's Gaussian nowcast N(mu, sigma^2) [B,N]."""
    z = (delta - (mu - last_x)) / sigma.clamp(min=1e-6)
    p = 0.5 * torch.erfc(z / 2 ** 0.5)
    if avail is not None:
        p = p * avail
    return p.sum(-1)


def state_growth(sigma_l: torch.Tensor, floor_l: torch.Tensor, avail: torch.Tensor, kappa, age: torch.Tensor):
    """Look-ahead variance accumulated since the last poll [B,N]. Its average rate per hour of poll age
    is shrunk towards the field-mean rate with weight kappa in [0,1] (empirical-Bayes style): kappa = 0
    gives every logger the same growth rate (staleness only, i.e. round-robin behaviour), kappa = 1
    trusts the network's logger-specific uncertainty fully."""
    g = (avail * (sigma_l ** 2 - floor_l).clamp(min=0.0)).sum(-1)
    a = age.clamp(min=1.0)
    rate = g / a
    k = torch.as_tensor(kappa, dtype=g.dtype)
    if k.dim() == 1:
        k = k[:, None]
    rate = k * rate + (1 - k) * rate.mean(dim=1, keepdim=True)
    return rate * a


def whittle_index(g: torch.Tensor, m: Memory) -> torch.Tensor:
    """Whittle-type index for restless polling: polling a logger resets its accumulated uncertainty
    growth g. For a cost that grows as g(a) with the poll age a, the index a*g(a) - sum_{tau<=a} g(tau)
    equals r*a^2/2 when g = r*a (random-walk error), so revisit intervals scale as 1/sqrt(r) -- the
    optimal periodic schedule -- instead of the 1/r produced by a greedy rule. ``m.cum_g`` must already
    contain the current g."""
    return m.node_age * g - m.cum_g


def rr_score(m: Memory, tiebreak: torch.Tensor) -> torch.Tensor:
    """Round-robin == oldest-poll-first with a fixed per-row tie-break order."""
    return m.node_age + tiebreak


def random_score(m: Memory, gen: torch.Generator | None = None) -> torch.Tensor:
    return torch.rand(m.node_age.shape, generator=gen)


def ars_score(m: Memory, eps: float = 1e-3) -> torch.Tensor:
    """Adaptive-rate sampling: poll age x recently observed rate of change (non-neural heuristic)."""
    rate = ((m.last_x - m.prev_x).abs() / m.prev_gap.clamp(min=1.0)).mean(-1)
    return m.node_age * (rate + eps)


def oracle_score(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Upper reference: poll where the current (pre-poll) estimate is truly worst."""
    err = (estimate - torch.nan_to_num(truth, nan=0.0)) ** 2 * torch.isfinite(truth)
    return err.mean(-1)
