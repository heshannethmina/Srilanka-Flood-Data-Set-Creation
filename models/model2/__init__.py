"""Model 2 — MMF-Net, a graph-free multimodal tokenising transformer.

Periodic numerical embeddings → temporal transformer over the lookback window
(+ optional cross-feature attention) → FiLM terrain conditioning → gated fusion
of a pretrained SAR encoder → multitask heads.

Built to answer two things model 1 left open: whether a properly constructed
tabular deep network can close the gap to gradient-boosted trees, and whether
the SAR branch was useless or merely fused badly.
"""
