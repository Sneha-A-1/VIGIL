"""Neural baseline estimators. All of them receive the same per-node inputs as the VIGIL network
(exogenous context + post-poll gateway memory) and predict the current VWC of every channel as a
residual of the last retrieved value. They are trained with MSE on un-polled channels."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vigil.models.layers import AdaptiveAdjacency, GraphConv, GraphGRUCell, row_normalise


class MLPEst(nn.Module):
    kind = "step"

    def __init__(self, d_in: int, D: int = 5, hidden: int = 128, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, D))

    def forward(self, inp: torch.Tensor, last_x: torch.Tensor) -> torch.Tensor:
        return last_x + self.net(inp)


class GRUEst(nn.Module):
    """Per-node recurrent estimator (no spatial message passing)."""
    kind = "recurrent"

    def __init__(self, d_in: int, D: int = 5, hidden: int = 48, **_):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU())
        self.cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden + D, hidden), nn.GELU(), nn.Linear(hidden, D))

    def init_state(self, B, N):
        return torch.zeros(B, N, self.hidden)

    def step(self, inp, last_x, h):
        B, N, _ = inp.shape
        z = self.inp(inp).reshape(B * N, -1)
        h = self.cell(z, h.reshape(B * N, -1)).reshape(B, N, -1)
        return last_x + self.out(torch.cat([h, last_x], -1)), h


class DCRNNEst(nn.Module):
    """Diffusion-convolutional GRU (Li et al., 2018) on the fixed distance graph (1- and 2-hop)."""
    kind = "recurrent"

    def __init__(self, d_in: int, D: int = 5, hidden: int = 48, adj: torch.Tensor | None = None, **_):
        super().__init__()
        self.hidden = hidden
        a = row_normalise(adj.float())
        self.register_buffer("a1", a)
        self.register_buffer("a2", a @ a)
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU())
        self.cell = GraphGRUCell(hidden, hidden, 2)
        self.out = nn.Sequential(nn.Linear(hidden + D, hidden), nn.GELU(), nn.Linear(hidden, D))

    def init_state(self, B, N):
        return torch.zeros(B, N, self.hidden)

    def step(self, inp, last_x, h):
        h = self.cell(self.inp(inp), h, [self.a1, self.a2])
        return last_x + self.out(torch.cat([h, last_x], -1)), h


class TransformerEst(nn.Module):
    """Per-node causal Transformer over the last ``window`` hours of inputs."""
    kind = "window"

    def __init__(self, d_in: int, D: int = 5, hidden: int = 48, window: int = 48, layers: int = 2,
                 heads: int = 4, **_):
        super().__init__()
        self.window = window
        self.inp = nn.Linear(d_in, hidden)
        self.pos = nn.Parameter(torch.randn(window, hidden) * 0.02)
        enc = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.Linear(hidden + D, hidden), nn.GELU(), nn.Linear(hidden, D))
        self.register_buffer("mask", torch.triu(torch.full((window, window), float("-inf")), diagonal=1))

    def forward(self, inp: torch.Tensor, last_x: torch.Tensor) -> torch.Tensor:
        """inp [B,W,N,F] (W == window), returns estimates for every position [B,W,N,D]."""
        B, W, N, Fd = inp.shape
        z = self.inp(inp).permute(0, 2, 1, 3).reshape(B * N, W, -1) + self.pos[:W]
        z = self.enc(z, mask=self.mask[:W, :W])
        z = z.reshape(B, N, W, -1).permute(0, 2, 1, 3)
        return last_x + self.out(torch.cat([z, last_x], -1))


class GWNEst(nn.Module):
    """Graph WaveNet-style estimator (Wu et al., 2019): gated dilated causal convolutions in time and
    graph convolutions on the distance graph plus a learned adaptive adjacency."""
    kind = "conv"

    def __init__(self, d_in: int, D: int = 5, hidden: int = 32, adj: torch.Tensor | None = None,
                 n_nodes: int = 42, dilations=(1, 2, 4, 8, 16), **_):
        super().__init__()
        self.dilations = list(dilations)
        self.receptive = 1 + sum(self.dilations)
        self.register_buffer("a_dist", row_normalise(adj.float()))
        self.adaptive = AdaptiveAdjacency(n_nodes)
        self.inp = nn.Linear(d_in, hidden)
        self.filt = nn.ModuleList([nn.Conv1d(hidden, hidden, 2, dilation=d) for d in self.dilations])
        self.gate = nn.ModuleList([nn.Conv1d(hidden, hidden, 2, dilation=d) for d in self.dilations])
        self.gconv = nn.ModuleList([GraphConv(hidden, hidden, 2) for _ in self.dilations])
        self.skip = nn.ModuleList([nn.Linear(hidden, hidden) for _ in self.dilations])
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden + D, hidden), nn.GELU(), nn.Linear(hidden, D))

    def forward(self, inp: torch.Tensor, last_x: torch.Tensor) -> torch.Tensor:
        """inp [B,S,N,F] -> [B,S,N,D] (causal: output at s depends on inputs <= s)."""
        B, S, N, _ = inp.shape
        sup = [self.a_dist, self.adaptive()]
        x = self.inp(inp)                                   # [B,S,N,H]
        skip = 0
        for d, fc, gc, gconv, sk in zip(self.dilations, self.filt, self.gate, self.gconv, self.skip):
            r = x
            z = x.permute(0, 2, 3, 1).reshape(B * N, -1, S)    # [BN,H,S]
            z = F.pad(z, (d, 0))
            z = torch.tanh(fc(z)) * torch.sigmoid(gc(z))
            z = z.reshape(B, N, -1, S).permute(0, 3, 1, 2)     # [B,S,N,H]
            skip = skip + sk(z)
            x = gconv(z, sup) + r
        return last_x + self.out(torch.cat([skip, last_x], -1))


class InputNorm(nn.Module):
    """Fixed standardisation of the per-node inputs (statistics from simulated training windows)."""

    def __init__(self, d_in: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d_in))
        self.register_buffer("std", torch.ones(d_in))

    def forward(self, x):
        return (x - self.mean) / self.std


class Normalised(nn.Module):
    """Wrap an estimator so that its inputs are standardised; exposes the wrapped API."""

    def __init__(self, model: nn.Module, d_in: int):
        super().__init__()
        self.model = model
        self.norm = InputNorm(d_in)
        self.kind = model.kind
        for a in ("window", "receptive", "hidden"):
            if hasattr(model, a):
                setattr(self, a, getattr(model, a))

    def set_input_stats(self, mean, std):
        self.norm.mean.copy_(mean)
        self.norm.std.copy_(std)

    def forward(self, inp, last_x):
        return self.model(self.norm(inp), last_x)

    def init_state(self, B, N):
        return self.model.init_state(B, N)

    def step(self, inp, last_x, h):
        return self.model.step(self.norm(inp), last_x, h)


def _zero_last(module: nn.Module) -> None:
    """Zero-initialise the last Linear layer so that a residual estimator starts at persistence."""
    last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


MODELS = {"mlp": MLPEst, "gru": GRUEst, "dcrnn": DCRNNEst, "transformer": TransformerEst, "gwn": GWNEst}


def build(name: str, d_in: int, D: int, adj: torch.Tensor, n_nodes: int, **kw) -> nn.Module:
    m = MODELS[name](d_in=d_in, D=D, adj=adj, n_nodes=n_nodes, **kw)
    _zero_last(m.out if hasattr(m, "out") else m.net)
    return Normalised(m, d_in)


def run_sequence(model: nn.Module, inp: torch.Tensor, last_x: torch.Tensor, h=None):
    """Apply a baseline to a full sequence inp [B,S,N,F]; returns (mu [B,S,N,D], final state)."""
    B, S, N, _ = inp.shape
    if model.kind == "step":
        return model(inp, last_x), None
    if model.kind == "recurrent":
        h = model.init_state(B, N) if h is None else h
        outs = []
        for s in range(S):
            mu, h = model.step(inp[:, s], last_x[:, s], h)
            outs.append(mu)
        return torch.stack(outs, 1), h
    if model.kind == "conv":
        return model(inp, last_x), None
    if model.kind == "window":
        W = model.window
        outs = []
        for s in range(S):
            a = max(0, s - W + 1)
            mu = model(inp[:, a:s + 1], last_x[:, a:s + 1])[:, -1]
            outs.append(mu)
        return torch.stack(outs, 1), None
    raise ValueError(model.kind)
