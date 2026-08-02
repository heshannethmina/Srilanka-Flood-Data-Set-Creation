"""Configuration objects and the M0 → M6 preset ladder (PROJECT_PROPOSAL.md §8)."""
from dataclasses import dataclass, field, replace
from typing import Dict, Literal, Optional

# ---------------------------------------------------------------- feature sets

DYNAMIC_FEATURES = [
    "precipitation_sum", "precip_sum_2d", "precip_sum_3d",
    "precip_sum_5d", "precip_sum_7d", "precip_sum_15d", "precip_sum_30d",
    "precip_max_3d", "precip_max_7d", "api_k090", "wetdays_7d",
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m", "windspeed_10m_mean", "windspeed_10m_max",
    "shortwave_radiation",
    "soil_wet_top", "soil_wet_root", "soil_wet_profile",
    "soil_wet_top_anom", "soil_wet_root_anom", "soil_wet_profile_anom",
    "discharge", "log_discharge", "discharge_rise_1d", "discharge_rise_3d",
    "discharge_mean_3d", "discharge_mean_7d", "discharge_anom",
    "discharge_zscore", "discharge_pctl",
]  # 33 dynamic channels

# Heavy-tailed columns get log1p before z-scoring (§7.7 "Normalisation").
LOG1P_FEATURES = {
    "precipitation_sum", "precip_sum_2d", "precip_sum_3d", "precip_sum_5d",
    "precip_sum_7d", "precip_sum_15d", "precip_sum_30d",
    "precip_max_3d", "precip_max_7d", "api_k090",
    "discharge", "discharge_mean_3d", "discharge_mean_7d",
}

# Static terrain: 2 continuous + zone one-hot (3) + position one-hot (4) = 9.
STATIC_CONTINUOUS = ["elevation_m", "log_drainage_proxy"]
ZONES = ["wet", "intermediate", "dry"]
POSITIONS = ["upstream", "mid", "downstream", "outlet"]
STATIC_DIM = len(STATIC_CONTINUOUS) + len(ZONES) + len(POSITIONS)

# Classification heads, in fixed order. Index 0 is the primary target.
CLS_HEADS = ["target_flood_1d", "target_flood_2d", "target_flood_3d", "target_onset_1d"]
# Regression heads: next-day discharge percentile, forward 3-day max discharge z-score.
REG_HEADS = ["reg_discharge_pctl_1d", "reg_discharge_z_max_3d"]

# Per-frame SAR scalars used by the `scalars` vision mode (§7.7 stream 3).
SAR_SCALARS = ["water_fraction", "vv_mean", "vh_mean", "valid_fraction"]


# ---------------------------------------------------------------- model config

GraphMode = Literal["none", "spatial", "flow", "both"]
StaticMode = Literal["none", "concat", "film"]
VisionMode = Literal["none", "scalars", "cnn"]


@dataclass
class ModelConfig:
    """Architecture. Defaults are the full TF-STGNN of §7.7."""
    lookback: int = 14
    n_dynamic: int = len(DYNAMIC_FEATURES)
    n_static: int = STATIC_DIM

    # stream 1 — temporal
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.2
    temporal_pool: Literal["attention", "last", "mean"] = "attention"

    # stream 2 — static terrain
    static_mode: StaticMode = "film"
    static_hidden: int = 64

    # stream 3 — vision
    vision_mode: VisionMode = "none"
    image_dim: int = 64
    image_px: int = 256          # 512 for the full-res Kaggle frames
    image_channels: int = 2      # VV, VH

    # fusion
    fusion_hidden: int = 192
    fusion_out: int = 128

    # stream 4 — graph
    graph_mode: GraphMode = "both"
    gat_layers: int = 2
    gat_heads: int = 4
    gat_head_dim: int = 32
    edge_attr_dim: int = 4

    # head
    head_hidden: int = 256
    head_hidden2: int = 128
    n_cls_heads: int = len(CLS_HEADS)
    n_reg_heads: int = len(REG_HEADS)

    @property
    def gat_dim(self) -> int:
        return self.gat_heads * self.gat_head_dim


# ------------------------------------------------------------- training config

LossName = Literal["bce", "wbce", "focal", "focal_conf"]
Protocol = Literal["temporal", "basin", "event", "random"]


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
    n_seeds: int = 1              # >1 → deep ensemble (M5)
    calibration: Literal["none", "temperature", "isotonic"] = "none"

    device: str = "auto"
    num_workers: int = 0
    truncate_after: Optional[str] = "2024-12-31"   # §11.5 ragged-2025 policy
    log_every: int = 1


@dataclass
class Preset:
    name: str
    answers: str
    model: ModelConfig
    train: TrainConfig


def _m(**kw) -> ModelConfig:
    return replace(ModelConfig(), **kw)


def _t(**kw) -> TrainConfig:
    return replace(TrainConfig(), **kw)


#: The incremental build ladder. Each step changes exactly one thing.
PRESETS: Dict[str, Preset] = {
    "M0": Preset(
        "M0", "establishes the floor",
        _m(static_mode="none", graph_mode="none", vision_mode="none"),
        _t(loss="bce"),
    ),
    "M1": Preset(
        "M1", "RQ4 — FiLM static conditioning",
        _m(static_mode="film", graph_mode="none", vision_mode="none"),
        _t(loss="bce"),
    ),
    "M2": Preset(
        "M2", "spatial edges only",
        _m(static_mode="film", graph_mode="spatial", vision_mode="none"),
        _t(loss="bce"),
    ),
    "M3": Preset(
        "M3", "RQ1 — directed flow edges, held as a separate relation",
        _m(static_mode="film", graph_mode="both", vision_mode="none"),
        _t(loss="bce"),
    ),
    "M4": Preset(
        "M4", "class imbalance — focal x label_confidence",
        _m(static_mode="film", graph_mode="both", vision_mode="none"),
        _t(loss="focal_conf"),
    ),
    "M5": Preset(
        "M5", "RQ3 — 5-seed deep ensemble + temperature scaling",
        _m(static_mode="film", graph_mode="both", vision_mode="none"),
        _t(loss="focal_conf", n_seeds=5, calibration="temperature"),
    ),
    "M6_scalars": Preset(
        "M6_scalars", "RQ5 — SAR scalar features appended to the dynamic stream",
        _m(static_mode="film", graph_mode="both", vision_mode="scalars"),
        _t(loss="focal_conf", n_seeds=5, calibration="temperature"),
    ),
    "M6_cnn": Preset(
        "M6_cnn", "RQ5 — SAR CNN branch over the 2,578 Sentinel-1 frames",
        _m(static_mode="film", graph_mode="both", vision_mode="cnn", image_px=512),
        _t(loss="focal_conf", n_seeds=5, calibration="temperature", batch_size=8),
    ),
}
