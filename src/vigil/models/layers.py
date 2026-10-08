"""Graph building blocks shared by the VIGIL network and the graph baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalise(a: torch.Tensor) -> torch.Tensor:
    """Random-walk normalisation D^-1 A (rows sum to one)."""
    return a / a.sum(dim=1, keepdim=True).clamp(min=1e-8)


class AdaptiveAdjacency(nn.Module):
    """Learned dense adjacency softmax(ReLU(E1 E2^T)) (Wu et al., 2019)."""

    def __init__(self, n_nodes: int, dim: int = 8):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n_nodes, dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, dim) * 0.1)

    def forward(self) -> torch.Tensor:
        return F.softmax(F.relu(self.e1 @ self.e2.T), dim=1)


class GraphConv(nn.Module):
    """y = x W_0 + sum_k (A_k x) W_k  for a list of [N,N] supports."""

    def __init__(self, d_in: int, d_out: int, n_supports: int, bias: bool = True):
        super().__init__()
        self.n_supports = n_supports
        self.lin = nn.Linear(d_in * (1 + n_supports), d_out, bias=bias)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        if self.n_supports == 0:
            return self.lin(x)
        feats = [x] + [torch.matmul(a, x) for a in supports]
        return self.lin(torch.cat(feats, dim=-1))


class GraphGRUCell(nn.Module):
    """GRU cell whose gates are graph convolutions (DCRNN-style recurrence, Li et al., 2018)."""

    def __init__(self, d_in: int, d_hid: int, n_supports: int):
        super().__init__()
        self.d_hid = d_hid
        self.gc_ru = GraphConv(d_in + d_hid, 2 * d_hid, n_supports)
        self.gc_c = GraphConv(d_in + d_hid, d_hid, n_supports)

    def forward(self, x: torch.Tensor, h: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        ru = torch.sigmoid(self.gc_ru(torch.cat([x, h], dim=-1), supports))
        r, u = ru.chunk(2, dim=-1)
        c = torch.tanh(self.gc_c(torch.cat([x, r * h], dim=-1), supports))
        return u * h + (1.0 - u) * c
