"""STG-Former — model 3, the assembled network.

    x     [B, L, N, F]     dynamic window (F includes SAR scalars in `scalars` mode)
    s     [N, S]           static terrain
    img   [K, C, P, P]     packed SAR frames, only the node-days that have one
    img_m [B, N]           1 where a frame is present

Output: logits [B, N, n_cls_heads] and regressions [B, N, n_reg_heads].

Structure
---------
Model 2's encoder, unchanged, up to the fusion layer; model 1's relational
GATv2, unchanged, after it::

    PLR embeddings -> temporal transformer -> cross-feature attention
        -> FiLM(terrain) -> [gated SAR] -> fuse -> RelationalGATv2 x L
        -> concat(skip, graph) -> head

Both halves are *imported*, not reimplemented. That is the point: if model 3
carried its own copy of `TemporalTransformer` the two could silently drift, and
"model 3 minus model 2 equals the graph" would stop being true the first time
one of them was touched.

Why P0 reproduces N3 exactly
----------------------------
Under `graph_mode="none"` the GATv2 stack is an empty `ModuleList`, `head_in`
collapses to `fusion_out`, and the forward pass reduces to MMFNet's line for
line. Submodules are also *constructed in MMFNet's order*, so the parameter
initialisation draws from the RNG in the same sequence and `P0` at seed 0 is
byte-identical to `N3` at seed 0 — not merely within seed noise of it. Keep that
property when editing: adding a module before `self.head`, or reordering the
constructor, silently breaks the control that the rest of the ladder is read
against.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from floodlib.blocks import FiLM, SarCNN, load_pretrained_encoder
from floodlib.graph import Graph
from model1.modules import RelationalGATv2
from model2.modules import FeatureAttention, GatedSarFusion, TemporalTransformer

from .config import STGFConfig


class STGFormer(nn.Module):
    def __init__(self, cfg: STGFConfig, graph: Optional[Graph] = None):
        super().__init__()
        self.cfg = cfg

        # ---- model 2's encoder, in model 2's construction order -------------
        self.temporal = TemporalTransformer(
            n_features=cfg.n_dynamic, lookback=cfg.lookback, d_model=cfg.d_model,
            d_emb=cfg.d_emb, kind=cfg.num_embed, n_freq=cfg.n_freq,
            sigma=cfg.freq_sigma, n_heads=cfg.n_heads, ff_mult=cfg.ff_mult,
            layers=cfg.n_layers, dropout=cfg.dropout)

        self.feat_attn = None
        if cfg.feature_attn:
            self.feat_attn = FeatureAttention(
                cfg.n_dynamic, cfg.d_emb, cfg.d_model, cfg.n_heads,
                cfg.ff_mult, cfg.feature_layers, cfg.dropout)
            self.merge = nn.LayerNorm(cfg.d_model)

        fusion_in = cfg.d_model
        if cfg.static_mode == "film":
            self.film = FiLM(cfg.n_static, cfg.static_hidden, cfg.d_model)
        elif cfg.static_mode == "concat":
            fusion_in += cfg.n_static

        self.sar = None
        if cfg.vision_mode == "cnn":
            self.cnn = SarCNN(cfg.image_channels, cfg.image_dim)
            if cfg.sar_pretrained:
                load_pretrained_encoder(self.cnn, cfg.sar_pretrained,
                                        freeze=cfg.sar_freeze)
            self.sar = GatedSarFusion(cfg.d_model, cfg.image_dim)

        self.fuse = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden), nn.GELU(),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_out), nn.GELU(),
            nn.LayerNorm(cfg.fusion_out),
        )

        # ---- model 1's graph stage ------------------------------------------
        # An empty ModuleList draws nothing from the RNG, which is what keeps
        # the graph-free rungs identical to model 2.
        self.gats = nn.ModuleList()
        if cfg.graph_mode != "none":
            self.gats = nn.ModuleList(
                RelationalGATv2(cfg.fusion_out, cfg.gat_heads, cfg.gat_head_dim,
                                cfg.edge_attr_dim, cfg.dropout)
                for _ in range(cfg.gat_layers))

        # The skip path lets a node bypass message passing entirely, so the
        # graph rungs can only add information, never destroy the pre-graph
        # representation. The Gin basin holdout depends on it.
        head_in = cfg.fusion_out * 2 if len(self.gats) else cfg.fusion_out
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden2), nn.GELU())
        self.cls = nn.Linear(cfg.head_hidden2, cfg.n_cls_heads)
        self.reg = nn.Linear(cfg.head_hidden2, cfg.n_reg_heads)

        self.register_buffer("temperature", torch.ones(1), persistent=True)
        self._set_graph(graph)

    # -------------------------------------------------------------- graph bufs
    def _set_graph(self, graph: Optional[Graph]) -> None:
        if graph is None or self.cfg.graph_mode == "none":
            e_i = torch.zeros(2, 0, dtype=torch.long)
            e_a = torch.zeros(0, self.cfg.edge_attr_dim)
            e_r = torch.zeros(0, dtype=torch.long)
        else:
            e_i = torch.as_tensor(graph.edge_index, dtype=torch.long)
            e_a = torch.as_tensor(graph.edge_attr, dtype=torch.float32)
            e_r = torch.as_tensor(graph.edge_rel, dtype=torch.long)
        self.register_buffer("edge_index", e_i, persistent=False)
        self.register_buffer("edge_attr", e_a, persistent=False)
        self.register_buffer("edge_rel", e_r, persistent=False)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, s: torch.Tensor,
                img: Optional[torch.Tensor] = None,
                img_pos: Optional[torch.Tensor] = None,
                img_mask: Optional[torch.Tensor] = None,
                img_age: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        B, L, N, _ = x.shape

        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        h, z = self.temporal(seq)
        if self.feat_attn is not None:
            h = self.merge(h + self.feat_attn(z))
        h = h.view(B, N, -1)

        if self.cfg.static_mode == "film":
            h = self.film(h, s)
        elif self.cfg.static_mode == "concat":
            h = torch.cat([h, s.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        if self.sar is not None:
            flat = h.new_zeros(B * N, self.cfg.image_dim)
            if img is not None and img.numel() and img_pos is not None:
                flat.index_copy_(0, img_pos, self.cnn(img))
            pres = img_mask if img_mask is not None else h.new_zeros(B, N)
            age = img_age if img_age is not None else h.new_zeros(B, N)
            h = self.sar(h, flat.view(B, N, -1), pres, age)

        z_in = self.fuse(h)
        zg = z_in
        for gat in self.gats:
            zg = gat(zg, self.edge_index, self.edge_attr, self.edge_rel)

        feat = torch.cat([z_in, zg], dim=-1) if len(self.gats) else z_in
        feat = self.head(feat)
        out = {"logits": self.cls(feat), "reg": self.reg(feat)}
        if return_attention:
            out["temporal_attention"] = None
        return out

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_proba(self, *args, **kw) -> torch.Tensor:
        """Calibrated probabilities for the primary head (temperature applied)."""
        logits = self.forward(*args, **kw)["logits"]
        return torch.sigmoid(logits / self.temperature.clamp_min(1e-3))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
