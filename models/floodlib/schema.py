"""Dataset schema — the contract every model family shares.

This module holds the things that describe *the data*, not *a model*: which
columns are read, which are log-scaled, which targets are predicted, and the
core configuration fields the training engine needs regardless of architecture.

It lives in `floodlib` rather than in a model package on purpose. If `model1`
and `model2` disagreed about which 33 channels are the input or which column is
the primary target, their result tables would not be comparable, and comparing
them is the whole point of having two.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


# ------------------------------------------------------------------ modes

GraphMode = Literal["none", "spatial", "flow", "both"]
StaticMode = Literal["none", "concat", "film"]
VisionMode = Literal["none", "scalars", "cnn"]
LossName = Literal["bce", "wbce", "focal", "focal_conf"]
Protocol = Literal["temporal", "basin", "event", "random"]


# ------------------------------------------------------------ base model config

@dataclass
class BaseModelConfig:
    """Fields the training engine reads out of *any* architecture's config.

    A model family subclasses this and adds its own hyperparameters. The engine
    never touches those — it only needs to know how long a window is, which
    modalities to prepare, and how many heads to expect.
    """
    lookback: int = 14
    n_dynamic: int = len(DYNAMIC_FEATURES)
    n_static: int = STATIC_DIM

    static_mode: StaticMode = "film"
    static_hidden: int = 64

    vision_mode: VisionMode = "none"
    image_dim: int = 64
    image_px: int = 256          # 512 for the full-res Kaggle frames
    image_channels: int = 2      # VV, VH

    graph_mode: GraphMode = "none"

    n_cls_heads: int = len(CLS_HEADS)
    n_reg_heads: int = len(REG_HEADS)
