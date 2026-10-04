"""Model 4 v2: retain the summary encoder and add ordered sequence attention.

This is a candidate ablation, not an asserted improvement. Image pixels and
their discharge-derived labels are not used as extra independent examples.
"""
import torch
from torch import nn

from floodlib.blocks import FiLM
from model2.modules import TemporalTransformer, FeatureAttention
from model4.workflow import HydroTEM


class HydroTemporalV2(HydroTEM):
    def __init__(self, data, cfg):
        super().__init__(data,cfg,temporal=True)
        if cfg.width % 4:
            raise ValueError('Sequence attention width must be divisible by four.')
        n = data['x'].shape[-1]
        # Every observed day remains a token; positional embeddings distinguish
        # windows that have the same averages but different rainfall timing.
        self.sequence = TemporalTransformer(n,cfg.lookback,d_model=cfg.width,
                d_emb=cfg.embedding,n_heads=4,layers=3,dropout=cfg.dropout)
        self.feature_attention = FeatureAttention(n,cfg.embedding,cfg.width,
                n_heads=4,layers=1,dropout=cfg.dropout)
        self.sequence_norm = nn.LayerNorm(cfg.width)
        self.static_film = FiLM(data['s'].shape[-1],cfg.width,cfg.width)
        self.sequence_gate = nn.Linear(2*cfg.width+data['s'].shape[-1],cfg.width)
        nn.init.zeros_(self.sequence_gate.weight)
        nn.init.constant_(self.sequence_gate.bias,-2.)
        self.fused_norm = nn.LayerNorm(cfg.width)

    def encode(self, x, s):
        current = super().encode(x,s)
        sequence, embeddings = self.sequence(x)
        sequence = self.sequence_norm(sequence+self.feature_attention(embeddings))
        sequence = self.static_film(sequence.unsqueeze(0),s).squeeze(0)
        gate = self.sequence_gate(torch.cat([current,sequence,s],-1)).sigmoid()
        return self.fused_norm(current+gate*sequence)
