"""STGCN (Spatio-Temporal Graph Convolutional Network) baseline model package."""

from .config import STGCNModelConfig, STGCNTrainConfig
from .stgcn_model import STGCNModel

__all__ = ["STGCNModelConfig", "STGCNTrainConfig", "STGCNModel"]
