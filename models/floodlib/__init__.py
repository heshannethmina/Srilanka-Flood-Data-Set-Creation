"""Shared library behind every model family in this repo.

`floodlib` owns the data and the evaluation; a model package owns only its
architecture. The split is what keeps `model1` (TF-STGNN, a relational graph
network) and `model2` (MMF-Net, a tokenising transformer) comparable: they read
the same 33 channels, use the same splits, and are scored by the same metrics
code, so a difference in their result rows is a difference in architecture.

    floodlib.schema     column contract, modes, BaseModelConfig
    floodlib.traincfg   TrainConfig, Preset
    floodlib.data       parquet → dense [T, N, F] panel + SnapshotBatcher
    floodlib.graph      river graph (used by model1 only)
    floodlib.sar        Sentinel-1 join, frame store, scalar channels
    floodlib.blocks     layers shared by both families (GRU, FiLM, SarCNN)
    floodlib.losses     multi-head focal / BCE loss
    floodlib.metrics    PR-AUC, Brier decomposition, ECE, event detection
    floodlib.calibrate  temperature and isotonic calibration
    floodlib.engine     the training loop both families run through
    floodlib.baselines  the four non-neural reference models
"""

__all__ = [
    "schema", "traincfg", "data", "graph", "sar", "blocks",
    "losses", "metrics", "calibrate", "engine", "baselines",
]
