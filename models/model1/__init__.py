"""Model 1 — TF-STGNN, a terrain-fused relational spatiotemporal graph network.

GRU + attention pooling → FiLM terrain conditioning → optional SAR CNN →
relational GATv2 over the 51-node river graph → multitask heads.

Shared data, metrics and training loop come from `floodlib`.
"""
