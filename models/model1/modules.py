"""Model 1's own layer: the relational GATv2 (PROJECT_PROPOSAL.md §7.7 stream 4).

Written directly against torch scatter primitives. With 51 nodes and 239 edges,
PyTorch Geometric would add a heavyweight, hard-to-build dependency for no speed
benefit — and this way the exact same file runs on Windows/Python 3.13 locally
and on Kaggle unmodified.

The layers shared with model 2 (GRU encoder, FiLM, SAR CNN) are in
`floodlib.blocks`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floodlib.graph import N_RELATIONS


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

    Note on parameter counts: this allocates one parameter set per relation
    whether or not that relation has any edges. Under `graph_mode="spatial"` the
    flow relation's parameters — 67,584 of M2's reported 581,319, 11.6% — are
    therefore never used and never receive a gradient (likewise the spatial set
    under `"flow"`). They do not affect the forward pass or training in any way,
    but the `n_params` column overstates those two modes by that much and a
    like-for-like efficiency comparison should subtract it. Left in place
    deliberately: removing the unused sets would shift the RNG stream and so
    change the already-reported M2 numbers, for no change in what the model
    computes.
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
