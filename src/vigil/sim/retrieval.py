"""Gateway-side retrieval memory for budgeted polling of soil-sensor loggers (batched, torch).

Each batch row b is an independent simulated deployment that starts at hour ``t0[b]``.
A *poll* of logger i at hour t returns every valid channel (5 depths of VWC and temperature);
channels whose sensor produced no valid reading stay stale.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

AGE_CAP = 720.0  # hours used to normalise ages (30 days)
AVAIL_DECAY = 0.7  # EMA factor of the per-channel availability estimate


@dataclass
class Memory:
    last_x: torch.Tensor     # [B,N,D] last retrieved (normalised) VWC, 0 if never observed
    last_tmp: torch.Tensor   # [B,N,D] last retrieved (normalised) temperature
    age: torch.Tensor        # [B,N,D] hours since the channel was last observed
    node_age: torch.Tensor   # [B,N] hours since the logger was last polled
    cum_p: torch.Tensor      # [B,N] precipitation (mm) since the logger was last polled
    cum_vpd: torch.Tensor    # [B,N] summed VPD (kPa h) since the logger was last polled
    prev_x: torch.Tensor     # [B,N,D] value retrieved at the poll before the last one (ARS policy)
    prev_gap: torch.Tensor   # [B,N,D] hours between the last two observations (ARS policy)
    obs: torch.Tensor        # [B,N,D] 1 if observed at the current hour
    avail: torch.Tensor      # [B,N,D] EMA of "a poll returned a valid reading" (sensor availability)
    floor_l: torch.Tensor    # [B,N,D] look-ahead variance right after the channel's last poll (residual floor)
    floor_e: torch.Tensor    # [B,N,D] event Bernoulli variance right after the channel's last poll
    cum_g: torch.Tensor      # [B,N] accumulated uncertainty growth since the last poll (Whittle index)

    def clone(self) -> "Memory":
        return Memory(**{k: v.clone() for k, v in self.__dict__.items()})


def init_memory(x0: torch.Tensor, tmp0: torch.Tensor) -> Memory:
    """Memory after a full poll of every logger (x0/tmp0: [B,N,D] with NaN for missing)."""
    ok = torch.isfinite(x0)
    B, N, D = x0.shape
    age = torch.where(ok, torch.zeros_like(x0), torch.full_like(x0, AGE_CAP))
    lx = torch.nan_to_num(x0, nan=0.0)
    lt = torch.nan_to_num(tmp0, nan=0.0)
    z = torch.zeros(B, N)
    return Memory(last_x=lx, last_tmp=lt, age=age, node_age=z.clone(), cum_p=z.clone(), cum_vpd=z.clone(),
                  prev_x=lx.clone(), prev_gap=torch.full_like(x0, 24.0), obs=ok.float(),
                  avail=0.5 + 0.5 * ok.float(), floor_l=torch.zeros_like(x0), floor_e=torch.zeros_like(x0),
                  cum_g=z.clone())


def advance(m: Memory, prcp: torch.Tensor, vpd: torch.Tensor) -> None:
    """One hour passes. prcp/vpd: [B] hourly values at the new hour."""
    # out-of-place updates: earlier memory tensors may still be referenced by the autograd graph
    m.age = (m.age + 1.0).clamp(max=AGE_CAP)
    m.node_age = (m.node_age + 1.0).clamp(max=AGE_CAP)
    m.cum_p = m.cum_p + prcp[:, None]
    m.cum_vpd = m.cum_vpd + vpd[:, None]
    m.obs = torch.zeros_like(m.obs)


def observe(m: Memory, poll: torch.Tensor, x: torch.Tensor, tmp: torch.Tensor) -> torch.Tensor:
    """Apply a poll decision. poll: [B,N] bool; x/tmp: [B,N,D] true values (NaN = sensor gap).

    Returns the channel-level observation mask [B,N,D] (float)."""
    o = poll[:, :, None] & torch.isfinite(x)
    of = o.float()
    m.prev_gap = torch.where(o, m.age.clamp(min=1.0), m.prev_gap)
    m.prev_x = torch.where(o, m.last_x, m.prev_x)
    m.last_x = torch.where(o, torch.nan_to_num(x), m.last_x)
    tok = poll[:, :, None] & torch.isfinite(tmp)
    m.last_tmp = torch.where(tok, torch.nan_to_num(tmp), m.last_tmp)
    m.age = torch.where(o, torch.zeros_like(m.age), m.age)
    m.node_age = torch.where(poll, torch.zeros_like(m.node_age), m.node_age)
    m.cum_p = torch.where(poll, torch.zeros_like(m.cum_p), m.cum_p)
    m.cum_vpd = torch.where(poll, torch.zeros_like(m.cum_vpd), m.cum_vpd)
    m.obs = of
    # availability estimate, updated only for polled loggers
    m.avail = torch.where(poll[:, :, None], AVAIL_DECAY * m.avail + (1 - AVAIL_DECAY) * of, m.avail)
    return of


def memory_features(m: Memory) -> torch.Tensor:
    """Per-node memory features [B,N,4D+3]."""
    agef = torch.log1p(m.age) / torch.log1p(torch.tensor(AGE_CAP))
    nagef = (torch.log1p(m.node_age) / torch.log1p(torch.tensor(AGE_CAP)))[..., None]
    return torch.cat([m.last_x, agef, m.last_tmp, m.obs,
                      torch.log1p(m.cum_p)[..., None], (m.cum_vpd / 100.0)[..., None], nagef], dim=-1)


def select_top_k(score: torch.Tensor, k: torch.Tensor, node_age: torch.Tensor | None = None,
                 max_age: float | None = None) -> torch.Tensor:
    """Pick the ``k[b]`` highest-scoring loggers of each row. Loggers whose poll age reached
    ``max_age`` are served first (freshness guard), oldest first."""
    s = score.clone()
    if node_age is not None and max_age is not None:
        due = node_age >= max_age
        s = torch.where(due, 1e6 + node_age, s)
    order = torch.argsort(s, dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(s.shape[1]).expand_as(order))
    return ranks < k[:, None]
