"""VIGIL network: a neural predict-update graph filter with probabilistic, look-ahead and event heads.

At every hour t the network
  1. *predicts* (prior) the field state from its recurrent state, the gateway memory (last
     retrieved values and their ages) and exogenous context (weather, soil, terrain, crop):
         h_t^- = GGRU_pred(h_{t-1}, phi(c_t, m_{t-1}))
         (mu^-, sigma^-, p^-, nu^-, sigma_L^-) = heads^-(h_t^-, m_{t-1}, r_t^-)
     where (mu, sigma) describe the current value, p the probability of a wetting event, and
     (nu, sigma_L) the mean level of the next L hours (look-ahead);
  2. the acquisition rule ranks loggers by the value of information of polling them, computed from
     the look-ahead variance sigma_L^2 (persistent, reducible uncertainty) and p(1-p), and polls K;
  3. *updates* (posterior) with the innovations of the polled channels, which graph message passing
     propagates to un-polled loggers:
         h_t^+ = GGRU_upd(h_t^-, psi(o_t, o_t*(y_t - mu_t^-), o_t*y_t, o_t*tau_t, h_t^-))
         (mu^+, sigma^+, p^+, nu^+, sigma_L^+) = heads^+(h_t^+, m_t, r_t^+)
Means are residuals of the last retrieved value; r is the rise of that value above the trailing
24-h minimum of the network's own estimates (in units of the event threshold).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vigil.models.layers import AdaptiveAdjacency, GraphGRUCell, row_normalise

SIGMA_MIN = 1e-3
N_OUT = 5  # mu, sigma, event logit, look-ahead mean, look-ahead sigma


class Heads(nn.Module):
    def __init__(self, d_hid: int, D: int, sigma0: float = 0.1, p0: float = 0.01):
        super().__init__()
        self.D = D
        self.mlp = nn.Sequential(nn.Linear(d_hid + 2 * D, d_hid), nn.GELU(), nn.Linear(d_hid, N_OUT * D))
        # start from persistence (means = last retrieved value), small spreads and a rare-event prior
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.zero_()
            s0 = math.log(math.expm1(sigma0))
            last.bias[D:2 * D] = s0
            last.bias[2 * D:3 * D] = math.log(p0 / (1 - p0))
            last.bias[4 * D:5 * D] = s0

    def forward(self, h: torch.Tensor, last_x: torch.Tensor, rise: torch.Tensor):
        out = self.mlp(torch.cat([h, last_x, rise], dim=-1))
        dmu, s, logit, dnu, sl = out.split(self.D, dim=-1)
        return (last_x + dmu, F.softplus(s) + SIGMA_MIN, logit, last_x + dnu, F.softplus(sl) + SIGMA_MIN)


class VIGILNet(nn.Module):
    def __init__(self, n_nodes: int, d_ctx: int, d_mem: int, D: int = 5, hidden: int = 48,
                 node_emb: int = 0, adj: torch.Tensor | None = None, use_graph: bool = True,
                 use_adaptive: bool = True, predict_update: bool = True):
        super().__init__()
        self.D, self.hidden, self.predict_update = D, hidden, predict_update
        self.use_graph = use_graph
        self.node_emb = nn.Parameter(torch.randn(n_nodes, node_emb) * 0.1) if node_emb else None
        self.adaptive = AdaptiveAdjacency(n_nodes) if (use_graph and use_adaptive) else None
        if adj is not None:
            self.register_buffer("a_dist", row_normalise(adj.float()))
        else:
            self.a_dist = None
        n_sup = (1 + int(self.adaptive is not None)) if use_graph else 0
        d_e = node_emb or 0
        self.register_buffer("in_mean", torch.zeros(d_ctx + d_mem))
        self.register_buffer("in_std", torch.ones(d_ctx + d_mem))
        self.inp = nn.Sequential(nn.Linear(d_ctx + d_mem + d_e, hidden), nn.GELU())
        self.pred_cell = GraphGRUCell(hidden, hidden, n_sup)
        self.head_prior = Heads(hidden, D)
        if predict_update:
            self.obs_in = nn.Sequential(nn.Linear(4 * D + hidden, hidden), nn.GELU())
            self.upd_cell = GraphGRUCell(hidden, hidden, n_sup)
            self.head_post = Heads(hidden, D)

    def supports(self) -> list[torch.Tensor]:
        if not self.use_graph:
            return []
        s = [self.a_dist]
        if self.adaptive is not None:
            s.append(self.adaptive())
        return s

    def init_state(self, B: int, N: int) -> torch.Tensor:
        return torch.zeros(B, N, self.hidden)

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.in_mean.copy_(mean)
        self.in_std.copy_(std)

    def _embed(self, ctx: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        z = (torch.cat([ctx, mem], dim=-1) - self.in_mean) / self.in_std
        parts = [z]
        if self.node_emb is not None:
            parts.append(self.node_emb.expand(ctx.shape[0], -1, -1))
        return self.inp(torch.cat(parts, dim=-1))

    def prior(self, h, ctx, mem, last_x, rise, sup):
        z = self._embed(ctx, mem)
        h_m = self.pred_cell(z, h, sup)
        return h_m, self.head_prior(h_m, last_x, rise)

    def update(self, h_m, prior_out, obs, y, tau, last_x_post, rise_post, sup):
        """obs: [B,N,D] observation mask; y/tau: observed (normalised) VWC/temperature (0 where unobserved)."""
        if not self.predict_update:
            return h_m, prior_out
        innov = obs * (y - prior_out[0])
        z = self.obs_in(torch.cat([obs, innov, obs * y, obs * tau, h_m], dim=-1))
        h_p = self.upd_cell(z, h_m, sup)
        return h_p, self.head_post(h_p, last_x_post, rise_post)
