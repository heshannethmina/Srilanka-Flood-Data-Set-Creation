"""Model 2 architecture config and the N0 → N6 preset ladder.

Model 2 is deliberately **graph-free**. Model 1 already answers "does relational
message passing help?", but only against a weak per-node floor (M0, a plain
GRU). Putting a properly built per-node network on the other side of that
comparison is what makes the answer mean something.
"""
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional

from floodlib.schema import BaseModelConfig, GraphMode
from floodlib.traincfg import Preset, TrainConfig

FAMILY = "MMF-Net"


@dataclass
class MMFConfig(BaseModelConfig):
    """Multimodal Flood Net: tokenising temporal transformer + gated SAR fusion."""

    graph_mode: GraphMode = "none"      # no message passing, by design

    # stream 1 — tabular tokenisation
    num_embed: Literal["linear", "plr"] = "plr"
    d_emb: int = 8
    n_freq: int = 8
    freq_sigma: float = 0.05

    # stream 1 — temporal transformer
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.2

    # stream 2 — cross-feature attention
    feature_attn: bool = True
    feature_layers: int = 1

    # stream 4 — SAR
    sar_pretrained: Optional[str] = None   # path from model2/pretrain_sar.py
    sar_freeze: bool = True

    # fusion and head — same widths as model 1, so a parameter-count comparison
    # is about the encoder and not about an arbitrarily wider classifier.
    fusion_hidden: int = 192
    fusion_out: int = 128
    head_hidden: int = 256
    head_hidden2: int = 128


def _m(**kw) -> MMFConfig:
    return replace(MMFConfig(), **kw)


def _t(**kw) -> TrainConfig:
    return replace(TrainConfig(), **kw)


#: One change per rung, exactly as the M-ladder. N0 mirrors M0 (no terrain, no
#: graph, plain BCE) so the two families start from a comparable floor.
PRESETS: Dict[str, Preset] = {
    "N0": Preset(
        "N0", "floor — transformer with plain linear feature embeddings",
        _m(num_embed="linear", feature_attn=False, static_mode="none",
           vision_mode="none"),
        _t(loss="bce"),
    ),
    "N1": Preset(
        "N1", "RQ6 — periodic (PLR) numerical embeddings, the tree-gap fix",
        _m(num_embed="plr", feature_attn=False, static_mode="none",
           vision_mode="none"),
        _t(loss="bce"),
    ),
    "N2": Preset(
        "N2", "RQ7 — cross-feature attention over the 33 channels",
        _m(num_embed="plr", feature_attn=True, static_mode="none",
           vision_mode="none"),
        _t(loss="bce"),
    ),
    "N3": Preset(
        "N3", "RQ4 again — FiLM terrain conditioning, without a graph",
        _m(num_embed="plr", feature_attn=True, static_mode="film",
           vision_mode="none"),
        _t(loss="bce"),
    ),
    "N4": Preset(
        "N4", "class imbalance — focal x label_confidence",
        _m(num_embed="plr", feature_attn=True, static_mode="film",
           vision_mode="none"),
        _t(loss="focal_conf"),
    ),
    "N5": Preset(
        "N5", "headline — 5-seed deep ensemble + temperature scaling",
        _m(num_embed="plr", feature_attn=True, static_mode="film",
           vision_mode="none"),
        _t(loss="focal_conf", n_seeds=5, calibration="temperature"),
    ),
    "N6_scalars": Preset(
        "N6_scalars", "RQ5 again — SAR scalars appended to the dynamic stream",
        _m(num_embed="plr", feature_attn=True, static_mode="film",
           vision_mode="scalars"),
        _t(loss="focal_conf", n_seeds=5, calibration="temperature"),
    ),
    "N6_gated": Preset(
        "N6_gated", "RQ8 — pretrained SAR encoder fused through a learned gate",
        _m(num_embed="plr", feature_attn=True, static_mode="film",
           vision_mode="cnn", image_px=512, sar_freeze=True),
        _t(loss="focal_conf", n_seeds=5, calibration="temperature", batch_size=8),
    ),
    "N6_gated_ft": Preset(
        "N6_gated_ft", "RQ8 — as N6_gated but the encoder is fine-tuned, not frozen",
        _m(num_embed="plr", feature_attn=True, static_mode="film",
           vision_mode="cnn", image_px=512, sar_freeze=False),
        _t(loss="focal_conf", n_seeds=5, calibration="temperature", batch_size=8),
    ),
}
