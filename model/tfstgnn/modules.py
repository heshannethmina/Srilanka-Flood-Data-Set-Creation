"""Neural building blocks (PROJECT_PROPOSAL.md §7.7).

The relational GATv2 is written directly against torch scatter primitives. With
51 nodes and 239 edges, PyTorch Geometric would add a heavyweight, hard-to-build
dependency for no speed benefit — and this way the exact same file runs on
Windows/Python 3.13 locally and on Kaggle unmodified.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import N_RELATIONS


# ------------------------------------------------------- stream 1: temporal

class TemporalEncoder(nn.Module):
    """2-layer GRU over the lookback window + learned attention pooling.

    The attention weights are returned as well: averaged per basin they show the
    rainfall→discharge lag, which is the interpretability figure promised in the
    proposal.
    """

    def __init__(self, n_in: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2, pool: str = "attention"):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.pool = pool
        if pool == "attention":
            self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: [M, L, F] → ([M, H], attention [M, L] or None)."""
        h, _ = self.gru(x)
        if self.pool == "last":
            return self.drop(h[:, -1]), None
        if self.pool == "mean":
            return self.drop(h.mean(1)), None
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)      # [M, L]
        return self.drop((h * a.unsqueeze(-1)).sum(1)), a


# --------------------------------------------------------- stream 2: terrain

class FiLM(nn.Module):
    """Static terrain modulates the temporal state: h ← γ ⊙ h + β, then LayerNorm.

    γ is parameterised as 1 + Δγ so the module starts at identity, which keeps
    M1 a strict superset of M0 at initialisation.
    """

    def __init__(self, n_static: int, hidden: int, feat: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * feat))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.norm = nn.LayerNorm(feat)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """h: [B, N, D], s: [N, n_static] → [B, N, D]."""
        gamma, beta = self.net(s).chunk(2, dim=-1)             # [N, D] each
        return self.norm(h * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0))


# ---------------------------------------------------------- stream 3: vision

class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (nn.Sequential() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + self.short(x), inplace=True)


class SarCNN(nn.Module):
    """ResNet-18 with a 2-channel stem (VV, VH), trained from scratch.

    Pretrained ImageNet weights are deliberately not used: SAR backscatter in dB
    has nothing in common with RGB natural-image statistics, and the 2-channel
    stem would have to be re-initialised anyway.
    """

    def __init__(self, in_ch: int = 2, out_dim: int = 64, width: int = 64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 7, 2, 3, bias=False), nn.BatchNorm2d(w),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        cfg = [(w, w, 1), (w, w, 1), (w, 2 * w, 2), (2 * w, 2 * w, 1),
               (2 * w, 4 * w, 2), (4 * w, 4 * w, 1), (4 * w, 8 * w, 2), (8 * w, 8 * w, 1)]
        self.blocks = nn.Sequential(*[BasicBlock(a, b, s) for a, b, s in cfg])
        self.head = nn.Linear(8 * w, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [M, 2, H, W] → [M, out_dim]."""
        z = self.blocks(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(z, 1).flatten(1))


# ----------------------------------------------------------- stream 4: graph

def _scatter_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over edges sharing a destination node. logits: [E, H]."""
    idx = index.unsqueeze(-1).expand_as(logits)
    m = torch.zeros(n, logits.size(-1), device=logits.device, dtype=logits.dtype)
    m = m.index_reduce(0, index, logits, "amax", include_self=False)
    ex = torch.exp(logits - m.gather(0, idx))
    den = torch.zeros_like(m).index_add(0, index, ex)
    return ex / (den.gather(0, idx) + 1e-16)


class RelationalGATv2(nn.Module):
    """GATv2 with **separate parameters per relation** (flow / spatial / self).

    Attention is normalised jointly across a node's incoming edges regardless of
    relation, so the layer learns how much to weight its upstream parent against
    its spatial neighbours rather than having that ratio fixed by architecture.
    """

    def __init__(self, dim_in: int, heads: int = 4, head_dim: int = 32,
                 edge_dim: int = 4, dropout: float = 0.2,
                 n_relations: int = N_RELATIONS):
        super().__init__()
        self.h, self.d, self.R = heads, head_dim, n_relations
        out = heads * head_dim
        self.lin_src = nn.ModuleList(nn.Linear(dim_in, out) for _ in range(n_relations))
        self.lin_dst = nn.ModuleList(nn.Linear(dim_in, out) for _ in range(n_relations))
        self.lin_edge = nn.ModuleList(nn.Linear(edge_dim, out) for _ in range(n_relations))
        self.att = nn.ParameterList(
            nn.Parameter(torch.empty(heads, head_dim)) for _ in range(n_relations))
        for a in self.att:
            nn.init.xavier_uniform_(a)
        self.proj = nn.Linear(out, dim_in)
        self.norm = nn.LayerNorm(dim_in)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_rel: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, N, D] (residual + LayerNorm)."""
        B, N, _ = x.shape
        E = edge_index.size(1)
        if E == 0:
            return self.norm(x)
        src, dst = edge_index[0], edge_index[1]

        logits = x.new_zeros(B, E, self.h)
        msg = x.new_zeros(B, E, self.h, self.d)
        for r in range(self.R):
            sel = edge_rel == r
            if not bool(sel.any()):
                continue
            s_r, d_r = src[sel], dst[sel]
            a_r = edge_attr[sel]
            m = self.lin_src[r](x)[:, s_r]                       # [B, E_r, out]
            g = self.lin_dst[r](x)[:, d_r] + self.lin_edge[r](a_r).unsqueeze(0)
            pre = F.leaky_relu(m + g, 0.2).view(B, -1, self.h, self.d)
            logits[:, sel] = (pre * self.att[r]).sum(-1)
            msg[:, sel] = m.view(B, -1, self.h, self.d)

        alpha = torch.stack([_scatter_softmax(logits[b], dst, N) for b in range(B)])
        alpha = self.drop(alpha)                                  # [B, E, H]

        agg = x.new_zeros(B, N, self.h, self.d)
        agg.index_add_(1, dst, msg * alpha.unsqueeze(-1))
        return self.norm(x + self.drop(self.proj(agg.reshape(B, N, self.h * self.d))))
