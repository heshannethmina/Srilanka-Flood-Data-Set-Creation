"""Training configuration and the preset container.

Shared by every model family: the optimiser, the loss, the split protocol and
the calibration policy are properties of *the experiment*, not of the network.
Holding them here is what makes an M-row and an N-row in the results table an
apples-to-apples comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from .schema import BaseModelConfig, LossName, Protocol


@dataclass
class TrainConfig:
    protocol: Protocol = "temporal"
    loss: LossName = "focal_conf"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    pos_weight: Optional[float] = None       # used by `wbce` only

    head_weights: Dict[str, float] = field(default_factory=lambda: {
        "target_flood_1d": 1.0,
        "target_flood_2d": 0.3,
        "target_flood_3d": 0.3,
        "target_onset_1d": 0.5,
    })
    reg_weight: float = 0.2

    epochs: int = 60
    warmup_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32          # full-graph day snapshots
    grad_clip: float = 1.0
    patience: int = 10            # early stopping on val event-level PR-AUC
    select_metric: str = "pr_auc"

    seed: int = 0
    n_seeds: int = 1              # >1 → deep ensemble
    calibration: Literal["none", "temperature", "isotonic"] = "none"

    device: str = "auto"
    num_workers: int = 0
    truncate_after: Optional[str] = "2024-12-31"   # §11.5 ragged-2025 policy
    log_every: int = 1


@dataclass
class Preset:
    """One rung of a build ladder: a name, what it answers, and two configs."""
    name: str
    answers: str
    model: BaseModelConfig
    train: TrainConfig
