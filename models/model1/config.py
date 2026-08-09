"""Model 1 architecture config and the M0 → M6 preset ladder (PROJECT_PROPOSAL.md §8).

The data-level schema (which columns, which targets) lives in `floodlib.schema`;
what is here is the shape of the network alone.
"""
from dataclasses import dataclass, replace
from typing import Dict, Literal

from floodlib.schema import BaseModelConfig, GraphMode
from floodlib.traincfg import Preset, TrainConfig

FAMILY = "TF-STGNN"


@dataclass
class ModelConfig(BaseModelConfig):
    """Terrain-Fused Spatiotemporal GNN. Defaults are the full network of §7.7."""

    # stream 1 — temporal
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.2
    temporal_pool: Literal["attention", "last", "mean"] = "attention"

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

    @property
    def gat_dim(self) -> int:
        return self.gat_heads * self.gat_head_dim


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
