"""MMF-Net — model 2, the assembled network.

Same inputs, same targets and same evaluation as model 1; a different answer to
the question of how a node's own history should be encoded.

    x     [B, L, N, F]   dynamic window (F includes SAR scalars in `scalars` mode)
    s     [N, S]         static terrain
    img   [K, C, P, P]   packed SAR frames, only the node-days that have one
    img_m [B, N]         1 where a frame is present

Output: logits [B, N, n_cls_heads] and regressions [B, N, n_reg_heads].
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from floodlib.blocks import FiLM, SarCNN, load_pretrained_encoder

from .config import MMFConfig
from .modules import FeatureAttention, GatedSarFusion, TemporalTransformer


class MMFNet(nn.Module):
    def __init__(self, cfg: MMFConfig, graph=None):
        super().__init__()
        self.cfg = cfg
        if graph is not None and cfg.graph_mode != "none":
            raise ValueError("model2 is graph-free; set graph_mode='none'")

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

        # Unlike model 1, the SAR embedding is gated into the state rather than
        # concatenated onto it, so it adds no width to the fusion input.
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
        self.head = nn.Sequential(
            nn.Linear(cfg.fusion_out, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden2), nn.GELU())
        self.cls = nn.Linear(cfg.head_hidden2, cfg.n_cls_heads)
        self.reg = nn.Linear(cfg.head_hidden2, cfg.n_reg_heads)

        self.register_buffer("temperature", torch.ones(1), persistent=True)

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

        feat = self.head(self.fuse(h))
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
