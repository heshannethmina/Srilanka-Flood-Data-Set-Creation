"""Model 3 — STG-Former: model 2's input layer, model 1's river graph.

The two existing families each answered half of RQ1 and confounded the other
half. Model 1 put a relational GATv2 on top of a *weak* per-node encoder (a GRU
fed raw scalars); model 2 built a *strong* per-node encoder (PLR numerical
embeddings + a temporal transformer) and deliberately had no graph at all. So
the standing comparison — N5_bce 0.8355 without a graph against M5 0.7421 with
one — varies the encoder, the loss and the data extent all at once, and cannot
attribute anything to message passing.

Model 3 holds the encoder fixed at model 2's and turns the graph on. Its `P0`
rung is byte-identical to model 2's `N3` by construction (see `model.py`), so
the graph is the only moving part in the ladder.
"""

from .config import FAMILY, PRESETS, STGFConfig
from .model import STGFormer

__all__ = ["FAMILY", "PRESETS", "STGFConfig", "STGFormer"]
