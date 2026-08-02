"""TF-STGNN — the assembled network (PROJECT_PROPOSAL.md §7.7)."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .graph import Graph
from .modules import FiLM, RelationalGATv2, SarCNN, TemporalEncoder


class TFSTGNN(nn.Module):
    """Inputs per batch of day-snapshots:

        x     [B, L, N, F]  dynamic window   (F includes SAR scalars in `scalars` mode)
        s     [N, S]        static terrain
        img   [B, N, 2, P, P] optional SAR frames   (cnn mode)
        img_m [B, N]        1 where a frame is present

    Output: logits [B, N, n_cls_heads] and regressions [B, N, n_reg_heads].
    """

    def __init__(self, cfg: ModelConfig, graph: Optional[Graph] = None):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalEncoder(cfg.n_dynamic, cfg.gru_hidden, cfg.gru_layers,
                                        cfg.dropout, cfg.temporal_pool)

        fusion_in = cfg.gru_hidden
        if cfg.static_mode == "film":
            self.film = FiLM(cfg.n_static, cfg.static_hidden, cfg.gru_hidden)
        elif cfg.static_mode == "concat":
            fusion_in += cfg.n_static

        if cfg.vision_mode == "cnn":
            self.cnn = SarCNN(cfg.image_channels, cfg.image_dim)
            # +2: presence flag and age-in-days (§11.2 — SAR is 12-day, not daily)
            fusion_in += cfg.image_dim + 2

        self.fuse = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden), nn.GELU(),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_out), nn.GELU(),
            nn.LayerNorm(cfg.fusion_out),
        )

        self.gats = nn.ModuleList()
        if cfg.graph_mode != "none":
            self.gats = nn.ModuleList(
                RelationalGATv2(cfg.fusion_out, cfg.gat_heads, cfg.gat_head_dim,
                                cfg.edge_attr_dim, cfg.dropout)
                for _ in range(cfg.gat_layers))

        # The skip path lets an out-of-distribution basin bypass the graph — the
        # Gin holdout depends on it.
        head_in = cfg.fusion_out * 2 if len(self.gats) else cfg.fusion_out
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
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
        """`img` is **packed**: [K, C, P, P] holding only the node-days that have
        a frame, with `img_pos` [K] their flat index into B*N (see sar.FrameStore)."""
        B, L, N, _ = x.shape

        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        h, att = self.temporal(seq)
        h = h.view(B, N, -1)

        if self.cfg.static_mode == "film":
            h = self.film(h, s)
        elif self.cfg.static_mode == "concat":
            h = torch.cat([h, s.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        if self.cfg.vision_mode == "cnn":
            flat = h.new_zeros(B * N, self.cfg.image_dim)
            if img is not None and img.numel() and img_pos is not None:
                flat.index_copy_(0, img_pos, self.cnn(img))
            v = flat.view(B, N, -1)
            # Nodes with no frame keep a zero embedding; the presence flag tells
            # the fusion layer that the zero is "no observation", not "no water".
            pres = (img_mask if img_mask is not None else h.new_zeros(B, N)).unsqueeze(-1)
            age = ((img_age if img_age is not None
                    else h.new_zeros(B, N)) / 30.0).unsqueeze(-1)
            h = torch.cat([h, v * pres, pres, age], dim=-1)

        z_in = self.fuse(h)
        z = z_in
        for gat in self.gats:
            z = gat(z, self.edge_index, self.edge_attr, self.edge_rel)

        feat = torch.cat([z_in, z], dim=-1) if len(self.gats) else z_in
        feat = self.head(feat)
        out = {"logits": self.cls(feat), "reg": self.reg(feat)}
        if return_attention:
            out["temporal_attention"] = None if att is None else att.view(B, N, L)
        return out

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_proba(self, *args, **kw) -> torch.Tensor:
        """Calibrated probabilities for the primary head (temperature applied)."""
        logits = self.forward(*args, **kw)["logits"]
        return torch.sigmoid(logits / self.temperature.clamp_min(1e-3))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
