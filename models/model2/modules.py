"""Model 2's layers: numerical embeddings, tokenising transformers, gated fusion.

The design answers a specific weakness in the model 1 results. On the temporal
protocol the gradient-boosted-tree baseline reaches PR-AUC 0.850 while the best
network reaches 0.742 — the familiar result that trees beat neural networks on
tabular data. The published fix for that is not a bigger network but a better
input layer: give every scalar feature its own learned *embedding* instead of
feeding it as one number into a linear layer, and the gap largely closes
(Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep
Learning", NeurIPS 2022). That is what `NumericEmbedding` is.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------- representation: features

class NumericEmbedding(nn.Module):
    """Per-feature embedding of a scalar: [..., F] → [..., F, d_emb].

    Two variants, selected by `kind`:

    ``linear``
        the plain baseline — feature j is mapped by its own affine map followed
        by a ReLU. Already stronger than sharing one projection across features,
        because a millimetre of rain and a soil-wetness fraction get separate
        parameters.
    ``plr``
        periodic-linear-ReLU. Feature j is first expanded into
        ``[sin(2πc_j x), cos(2πc_j x)]`` over `n_freq` learned frequencies, then
        projected and rectified. The periodic expansion is what lets a network
        represent a sharp decision boundary in a scalar — the thing a decision
        tree gets for free from a split point and a plain MLP struggles with.

    Frequencies are initialised from N(0, σ²) with a small σ: large initial
    frequencies alias the input and the model never recovers.
    """

    def __init__(self, n_features: int, d_emb: int = 8, kind: str = "plr",
                 n_freq: int = 8, sigma: float = 0.05):
        super().__init__()
        self.kind = kind
        d_in = 2 * n_freq if kind == "plr" else 1
        if kind == "plr":
            self.coef = nn.Parameter(torch.randn(n_features, n_freq) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, d_in, d_emb))
        self.bias = nn.Parameter(torch.zeros(n_features, d_emb))
        nn.init.normal_(self.weight, std=d_in ** -0.5)
        self.d_emb = d_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "plr":
            v = 2 * math.pi * x.unsqueeze(-1) * self.coef      # [..., F, n_freq]
            z = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        else:
            z = x.unsqueeze(-1)                                 # [..., F, 1]
        return F.relu(torch.einsum("...fi,fio->...fo", z, self.weight) + self.bias)


# --------------------------------------------------- architecture: transformers

def _encoder(d_model: int, n_heads: int, ff_mult: int, dropout: float,
             layers: int) -> nn.TransformerEncoder:
    """Pre-LN transformer stack. Pre-LN trains without a warmup-sensitive phase,
    which matters here because the ladder shares one optimiser schedule."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ff_mult * d_model,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers,
                                 norm=nn.LayerNorm(d_model))


class TemporalTransformer(nn.Module):
    """Self-attention over the `lookback` window, one token per day.

    Each day's 33 channels are embedded feature-wise and mixed into a single
    d_model token; a CLS token then pools the window. Compared with the GRU it
    replaces, attention can look straight back at the day the rain fell rather
    than carrying it forward through 14 recurrent steps — and the attention row
    of the CLS token is directly readable as the rainfall→discharge lag.
    """

    def __init__(self, n_features: int, lookback: int, d_model: int = 128,
                 d_emb: int = 8, kind: str = "plr", n_freq: int = 8,
                 sigma: float = 0.05, n_heads: int = 4, ff_mult: int = 2,
                 layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.embed = NumericEmbedding(n_features, d_emb, kind, n_freq, sigma)
        self.to_token = nn.Linear(n_features * d_emb, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, lookback + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.enc = _encoder(d_model, n_heads, ff_mult, dropout, layers)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [M, L, F] → (pooled [M, d_model], per-feature embeddings [M, L, F, d])."""
        z = self.embed(x)                                       # [M, L, F, d_emb]
        tok = self.to_token(z.flatten(-2))                      # [M, L, d_model]
        tok = torch.cat([self.cls.expand(tok.size(0), -1, -1), tok], dim=1)
        h = self.enc(self.drop(tok + self.pos[:, : tok.size(1)]))
        return h[:, 0], z


class FeatureAttention(nn.Module):
    """Self-attention *across the 33 channels* of a time-averaged window.

    The temporal transformer mixes features through a single linear layer, which
    cannot express "this much rain matters only when the soil is already wet".
    Attending over feature tokens can. It is affordable here only because the
    time axis is averaged out first: 33 tokens per node-day rather than 33 x 14.
    """

    def __init__(self, n_features: int, d_emb: int, d_model: int,
                 n_heads: int = 4, ff_mult: int = 2, layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(d_emb, d_model)
        self.feat_tok = nn.Parameter(torch.zeros(1, n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.feat_tok, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.enc = _encoder(d_model, n_heads, ff_mult, dropout, layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [M, L, F, d_emb] → [M, d_model]."""
        t = self.proj(z.mean(dim=1)) + self.feat_tok            # [M, F, d_model]
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1)
        return self.enc(t)[:, 0]


# ------------------------------------------------------------- gated fusion

class GatedSarFusion(nn.Module):
    """Fuse a SAR embedding into the tabular state through a learned gate.

    Model 1 concatenated the SAR embedding onto the fusion input and the branch
    returned nothing (RQ5). Concatenation is the wrong operator for this signal:
    Sentinel-1 revisits every ~12 days, so the overwhelming majority of node-days
    carry no observation at all, and a fixed slot in a concatenation forces the
    network to spend capacity distinguishing "no water" from "no picture" on
    every single sample.

    A gate makes that conditional instead. `g` is computed from the tabular
    state, the image embedding, the presence flag and the frame's age, and the
    image contributes `g ⊙ W v` — nothing at all when no frame exists. The gate's
    output layer is zero-initialised with a negative bias, so the module starts
    shut and the SAR rung of the ladder is a strict superset of the rung below it
    at initialisation.
    """

    def __init__(self, d_model: int, image_dim: int, init_bias: float = -3.0):
        super().__init__()
        self.proj = nn.Linear(image_dim, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model + image_dim + 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_bias)
        self.norm = nn.LayerNorm(d_model)
        #: Mean gate opening on node-days that actually have a frame, from the
        #: last forward pass. Read by the diagnostics: a branch whose gate never
        #: opens contributes nothing, and without this the only evidence would
        #: be a metric difference too small to attribute to anything.
        self.last_gate_mean: float = float("nan")

    def forward(self, h: torch.Tensor, v: torch.Tensor, pres: torch.Tensor,
                age: torch.Tensor) -> torch.Tensor:
        """h: [B, N, D], v: [B, N, image_dim], pres/age: [B, N]."""
        pres = pres.unsqueeze(-1)
        ctx = torch.cat([h, v, pres, (age / 30.0).unsqueeze(-1)], dim=-1)
        g = torch.sigmoid(self.gate(ctx))
        with torch.no_grad():
            denom = pres.sum().clamp_min(1.0) * g.size(-1)
            self.last_gate_mean = float((g * pres).sum() / denom)
        return self.norm(h + g * self.proj(v) * pres)
