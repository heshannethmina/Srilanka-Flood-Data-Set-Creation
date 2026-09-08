"""Model 3 architecture config and the P0 → P4 preset ladder.

Model 3 is the missing cell of a 2x2 that the first two families left open:

    ================  =========================  ==========================
                      weak input layer           PLR numerical embeddings
    ================  =========================  ==========================
    no graph          N0  (0.6083)               N5_bce  (0.8355)
    relational graph  M5  (0.7421, focal loss)   **model 3**
    ================  =========================  ==========================

Nobody has run the bottom-right cell, and it is the only one that can answer
RQ1. Reading "N5_bce beats M5, therefore the graph is useless" off that table is
invalid three times over: the encoder differs, the loss differs (M5 sits on the
focal rung that model 2 later showed costs +0.0509 PR-AUC), and the data extent
differs (model 1's 2026-08-02 run predates `truncate_after="2024-12-31"`).

This ladder removes all three confounds. Every rung shares one encoder, one
loss, one panel, one seed policy — and differs only in whether messages pass
along the river.

Deliberately *unchanged* from model 2
-------------------------------------
`grad_clip` stays 1.0 and `epochs` stays 60 even though the model 2 diagnostics
argue for 2.0 and 80. Changing them here would break the control: `P0` and
`P0_x5` exist to reproduce `N3` and `N5_bce`, and they cannot do that under a
different optimiser schedule. Sweep them separately once the ladder has landed
(`--grad-clip 2.0 --epochs 80` writes its own tagged output).
"""
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional

from floodlib.schema import BaseModelConfig, GraphMode
from floodlib.traincfg import Preset, TrainConfig

FAMILY = "STG-Former"

#: Index of `target_onset_1d` in `floodlib.schema.CLS_HEADS`.
ONSET_HEAD = 3

#: Head weighting that makes onset the target being optimised rather than an
#: auxiliary. The 1-day flood head is kept at 0.3 rather than dropped: it is
#: dense where onset is sparse, so it still supplies a usable gradient early in
#: training, which onset alone (~0.4% positive) does not.
ONSET_WEIGHTS = {
    "target_flood_1d": 0.3,
    "target_flood_2d": 0.1,
    "target_flood_3d": 0.1,
    "target_onset_1d": 1.0,
}


@dataclass
class STGFConfig(BaseModelConfig):
    """Spatio-Temporal Graph-Former.

    The tabular and temporal fields are copied verbatim from model 2's
    `MMFConfig` and the graph fields verbatim from model 1's `ModelConfig`,
    including their defaults. That is what lets `P0` reproduce `N3` exactly
    rather than approximately.
    """

    # stream 1 — tabular tokenisation (model 2's, unchanged)
    num_embed: Literal["linear", "plr"] = "plr"
    d_emb: int = 8
    n_freq: int = 8
    freq_sigma: float = 0.05

    # stream 1 — temporal transformer (model 2's, unchanged)
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.2

    # stream 2 — cross-feature attention (model 2's, unchanged)
    feature_attn: bool = True
    feature_layers: int = 1

    # stream 3 — SAR. Supported so model 3 is a strict superset of both
    # families, but off every rung of the default ladder: imagery has now lost
    # on the merits twice (M6_cnn null; N6_gated needed 17x the parameters to
    # score below N5_bce) and reaches only 9 of 51 nodes.
    sar_pretrained: Optional[str] = None
    sar_freeze: bool = True

    # stream 4 — graph (model 1's, unchanged)
    graph_mode: GraphMode = "both"
    gat_layers: int = 2
    gat_heads: int = 4
    gat_head_dim: int = 32
    edge_attr_dim: int = 4

    # fusion and head — same widths as both other families, so a parameter
    # count comparison is about the encoder and the graph, not the classifier.
    fusion_hidden: int = 192
    fusion_out: int = 128
    head_hidden: int = 256
    head_hidden2: int = 128

    @property
    def gat_dim(self) -> int:
        return self.gat_heads * self.gat_head_dim


def _m(**kw) -> STGFConfig:
    return replace(STGFConfig(), **kw)


def _t(**kw) -> TrainConfig:
    return replace(TrainConfig(), **kw)


#: The encoder every rung shares. Written once so no rung can drift from it.
_ENCODER = dict(num_embed="plr", feature_attn=True, static_mode="film",
                vision_mode="none")

#: The ladder. Rungs P0 → P3 change exactly one thing each; the two `_ctrl`
#: rungs are graph-free twins that exist to be *paired* against a graph rung at
#: identical seed counts, because an unpaired comparison is what made model 1's
#: M2-vs-M5 result unreadable.
PRESETS: Dict[str, Preset] = {
    "P0": Preset(
        # Byte-identical to model 2's N3 by construction. If this does not
        # reproduce 0.8269, the assembly is wrong and nothing below it is
        # interpretable — run it first and check before spending GPU on P1-P3.
        "P0", "control — graph off; must reproduce model 2's N3 (0.8269)",
        _m(**_ENCODER, graph_mode="none"),
        _t(loss="bce"),
    ),
    "P1": Preset(
        "P1", "+ spatial k-NN edges (204), single relation",
        _m(**_ENCODER, graph_mode="spatial"),
        _t(loss="bce"),
    ),
    "P2": Preset(
        # The rung the whole family exists for.
        "P2", "RQ1 — + 35 directed flow edges, held as a separate relation",
        _m(**_ENCODER, graph_mode="both"),
        _t(loss="bce"),
    ),
    "P0_x5": Preset(
        # Reproduces N5_bce (0.8355). The paired control for P3: same encoder,
        # same loss, same 5 seeds, same calibrator, no graph.
        "P0_x5", "paired control — graph off, 5-seed ensemble + temperature",
        _m(**_ENCODER, graph_mode="none"),
        _t(loss="bce", n_seeds=5, calibration="temperature"),
    ),
    "P3": Preset(
        # P3 - P0_x5 is the graph effect, with every other variable pinned. This
        # difference, with its per-seed spread, is the headline of RQ1.
        "P3", "headline — full graph, 5-seed ensemble + temperature",
        _m(**_ENCODER, graph_mode="both"),
        _t(loss="bce", n_seeds=5, calibration="temperature"),
    ),
    "P4_onset": Preset(
        # PR-AUC on target_flood_1d is a ceiling, not a contest: the
        # discharge_pctl baseline alone reaches 0.816 because the target is
        # tomorrow's discharge over the 98th percentile and discharge is
        # autocorrelated. Onset is the head where every baseline actually fails
        # (ev.det 0.16-0.22). This rung optimises and scores it.
        "P4_onset", "early-warning headline — onset as the optimised and scored head",
        _m(**_ENCODER, graph_mode="both"),
        _t(loss="bce", n_seeds=5, calibration="temperature",
           head_weights=dict(ONSET_WEIGHTS), eval_head=ONSET_HEAD),
    ),
    "P4_onset_ctrl": Preset(
        "P4_onset_ctrl", "paired control for P4_onset — same, graph off",
        _m(**_ENCODER, graph_mode="none"),
        _t(loss="bce", n_seeds=5, calibration="temperature",
           head_weights=dict(ONSET_WEIGHTS), eval_head=ONSET_HEAD),
    ),
    "P5_sar": Preset(
        # Off the default ladder. Only worth the ~3 h if P3 wins: it asks
        # whether imagery helps the *best* backbone, which is the one form of
        # the SAR question that has never been asked. Needs --sar-pretrained.
        "P5_sar", "off-ladder — gated pretrained SAR on the best backbone",
        _m(**{**_ENCODER, "vision_mode": "cnn"}, graph_mode="both",
           image_px=512, sar_freeze=True),
        _t(loss="bce", n_seeds=5, calibration="temperature", batch_size=8),
    ),
}

#: What `--stage ladder3` runs, in order. P0 first because it is the assembly
#: check and it is cheap; if it misses N3 by more than seed noise, stop.
LADDER = ["P0", "P1", "P2", "P0_x5", "P3"]

#: The early-warning pair, run after the ladder lands.
ONSET_RUNGS = ["P4_onset_ctrl", "P4_onset"]
