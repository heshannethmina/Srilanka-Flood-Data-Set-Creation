"""Configuration dataclasses for the STGCN baseline architecture."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from ..tfstgnn.config import (
    CLS_HEADS, DYNAMIC_FEATURES, REG_HEADS, STATIC_DIM,
    LossName, Protocol, TrainConfig,
)


@dataclass
class STGCNModelConfig:
    """Architecture hyperparameters for Spatio-Temporal Graph Convolutional Network (STGCN)."""

    lookback: int = 14
    n_dynamic: int = len(DYNAMIC_FEATURES)
    n_static: int = STATIC_DIM

    # ST-Conv Block setup
    st_conv_channels: list[int] = field(default_factory=lambda: [64, 64, 128])
    temporal_kernel_size: int = 3
    gcn_kernel_size: int = 1               # 1st-order Chebyshev / Normalized Adjacency
    dropout: float = 0.2

    # Head configuration
    head_hidden: int = 256
    head_hidden2: int = 128
    n_cls_heads: int = len(CLS_HEADS)
    n_reg_heads: int = len(REG_HEADS)


@dataclass
class STGCNTrainConfig:
    """Training configuration for STGCN model."""

    protocol: Protocol = "temporal"
    loss: LossName = "focal_conf"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    pos_weight: Optional[float] = None

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
    batch_size: int = 32
    grad_clip: float = 1.0
    patience: int = 10
    select_metric: str = "pr_auc"

    seed: int = 0
    n_seeds: int = 1
    calibration: Literal["none", "temperature", "isotonic"] = "none"

    device: str = "auto"
    num_workers: int = 0
    truncate_after: Optional[str] = "2024-12-31"
    log_every: int = 1
