"""TF-STGNN — Terrain-aware multimodal spatiotemporal GNN for flood early warning.

Implements the architecture in docs/PROJECT_PROPOSAL.md §7.7 and the incremental
build ladder M0 → M6 (§8).

Deliberately has **no PyTorch Geometric dependency**: the graph is 51 nodes and
239 edges, so the relational GATv2 is implemented directly with scatter ops in
`modules.py`. That keeps the package installable anywhere torch runs (Windows +
Python 3.13 included) and identical on Kaggle.
"""

__version__ = "0.1.0"
