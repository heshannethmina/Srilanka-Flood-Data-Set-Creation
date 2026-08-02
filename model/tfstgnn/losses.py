"""Loss functions (PROJECT_PROPOSAL.md §7.7 "Loss")."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .config import TrainConfig


def focal_bce(logits: torch.Tensor, target: torch.Tensor,
              alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Element-wise focal loss on logits (no reduction)."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return a_t * (1 - p_t).pow(gamma) * bce


class MultiHeadLoss(torch.nn.Module):
    """Weighted sum over the four classification heads plus the regression heads.

    `mask` zeroes invalid node-days and nodes outside the current split;
    `conf` is `label_confidence`, which down-weights days sitting within ±15% of
    the discharge threshold — i.e. the days where the label itself is arguable.
    """

    def __init__(self, cfg: TrainConfig, cls_heads: List[str]):
        super().__init__()
        self.cfg = cfg
        self.cls_heads = cls_heads
        self.w = torch.tensor([cfg.head_weights.get(h, 0.0) for h in cls_heads],
                              dtype=torch.float32)

    def forward(self, out: Dict[str, torch.Tensor], y: torch.Tensor, r: torch.Tensor,
                mask: torch.Tensor, conf: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = out["logits"]                       # [B, N, H]
        w = self.w.to(logits.device)
        m = mask.unsqueeze(-1)                       # [B, N, 1]

        if self.cfg.loss == "bce":
            per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        elif self.cfg.loss == "wbce":
            pw = torch.as_tensor(self.cfg.pos_weight or 1.0, device=logits.device)
            per = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                                     pos_weight=pw)
        else:
            per = focal_bce(logits, y, self.cfg.focal_alpha, self.cfg.focal_gamma)

        weight = m
        if self.cfg.loss == "focal_conf":
            weight = weight * conf.unsqueeze(-1)

        denom = weight.sum().clamp_min(1.0)
        cls = ((per * weight).sum(dim=(0, 1)) / denom * w).sum()

        reg = F.huber_loss(out["reg"], r, reduction="none", delta=1.0)
        reg = (reg * m).sum() / (m.sum() * out["reg"].shape[-1]).clamp_min(1.0)

        return {"loss": cls + self.cfg.reg_weight * reg, "cls": cls.detach(),
                "reg": reg.detach()}
