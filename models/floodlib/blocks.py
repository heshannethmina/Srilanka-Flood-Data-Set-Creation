"""Neural building blocks used by more than one model family.

Anything architecture-defining lives in the model package that defines it — the
relational GATv2 in `model1`, the tokenising transformer in `model2`. What is
here is shared machinery, so that a difference between the two families is a
difference of architecture and never an accident of two slightly different
implementations of the same layer.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------ temporal (GRU)

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


# ------------------------------------------------------------------- terrain

class FiLM(nn.Module):
    """Static terrain modulates the temporal state: h ← γ ⊙ h + β, then LayerNorm.

    γ is parameterised as 1 + Δγ so the module starts at identity, which keeps
    the conditioned model a strict superset of the unconditioned one at
    initialisation.
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


# -------------------------------------------------------------------- vision

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
    stem would have to be re-initialised anyway. `model2` instead pretrains this
    same encoder on the labelled flood/dry chips — see `model2/pretrain_sar.py`.
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
        self.frozen_bn = False

    def train(self, mode: bool = True):
        """Keep BatchNorm in eval mode when the encoder is frozen.

        Freezing the weights is not enough on its own: BatchNorm would still
        update its running statistics on every forward pass, so a "frozen"
        feature extractor would quietly drift away from the representation it
        was pretrained to produce.
        """
        super().train(mode)
        if self.frozen_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [M, 2, H, W] → [M, out_dim]."""
        z = self.blocks(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(z, 1).flatten(1))


def load_pretrained_encoder(cnn: SarCNN, path: str, freeze: bool = True,
                            verbose: bool = True) -> SarCNN:
    """Load `pretrain_sar.py` weights into a `SarCNN`, tolerating a head mismatch.

    The pretraining task has a 2-way classifier on top; the projection head that
    feeds the fusion layer is a different shape, so only stem+blocks transfer.
    """
    state = torch.load(path, map_location="cpu")
    state = state.get("encoder", state)
    keep = {k: v for k, v in state.items()
            if k in cnn.state_dict() and cnn.state_dict()[k].shape == v.shape}
    missing = [k for k in cnn.state_dict() if k not in keep]
    cnn.load_state_dict(keep, strict=False)
    if freeze:
        for name, p in cnn.named_parameters():
            p.requires_grad = name.startswith("head")
        cnn.frozen_bn = True
        cnn.eval()
    if verbose:
        print(f"[sar-pretrain] loaded {len(keep)}/{len(cnn.state_dict())} tensors "
              f"from {path} (not loaded: {len(missing)}) | "
              f"{'frozen' if freeze else 'fine-tuning'}")
    return cnn
