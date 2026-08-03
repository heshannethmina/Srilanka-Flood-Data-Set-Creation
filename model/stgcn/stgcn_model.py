"""Spatio-Temporal Graph Convolutional Network (STGCN) baseline model implementation."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import STGCNModelConfig


class TemporalConvBlock(nn.Module):
    """1D Temporal Convolution with Gated Linear Unit (GLU) activation.

    Applies causal 1D convolution along time for each node.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation
        # Double out_channels for GLU split
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=(kernel_size, 1),
            padding=(self.padding, 0),
            dilation=(dilation, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N]
        out = self.conv(x)  # [B, 2*C_out, T_padded, N]
        if self.padding > 0:
            out = out[:, :, :-self.padding, :]  # causal trim along time
        p, q = out.chunk(2, dim=1)
        return p * torch.sigmoid(q)


class SpatialGCNLayer(nn.Module):
    """Graph Convolution using normalized adjacency matrix A_hat = D^-1/2 (A + I) D^-1/2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, N]
        # norm_adj: [N, N]
        B, C, T, N = x.shape
        # Permute for graph matrix product: [B, T, N, C]
        x_perm = x.permute(0, 2, 3, 1)
        # Graph convolution: A_hat @ X @ W
        ax = torch.einsum("ij,btjc->btic", norm_adj, x_perm)  # [B, T, N, C_in]
        out = torch.matmul(ax, self.weight) + self.bias       # [B, T, N, C_out]
        return out.permute(0, 3, 1, 2)                         # [B, C_out, T, N]


class STConvBlock(nn.Module):
    """Sandwich Spatio-Temporal Convolutional Block.

    Structure: Temporal Conv (GLU) -> Spatial GCN (ReLU) -> Temporal Conv (GLU) -> LayerNorm + Residual
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.t_conv1 = TemporalConvBlock(in_channels, hidden_channels, kernel_size=kernel_size)
        self.s_gcn = SpatialGCNLayer(hidden_channels, hidden_channels)
        self.t_conv2 = TemporalConvBlock(hidden_channels, out_channels, kernel_size=kernel_size)
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, N]
        res = self.residual(x)
        x = self.t_conv1(x)
        x = self.s_gcn(x, norm_adj)
        x = F.relu(x)
        x = self.t_conv2(x)
        x = self.dropout(x)

        # LayerNorm over (C, N) dimension
        # x shape: [B, C, T, N] -> permute to [B, T, N, C] for LayerNorm
        x_perm = (x + res).permute(0, 2, 3, 1)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2)


class STGCNModel(nn.Module):
    """Spatio-Temporal Graph Convolutional Network (STGCN) Baseline Model.

    Inputs:
        x: [B, T, N, F_dynamic] dynamic panel features
        s: [N, F_static] static terrain features
        edge_index: [2, E] edge tensor (used to compute normalized adjacency matrix)
    Outputs:
        Dict with "logits" [B, N, 4] and "reg" [B, N, 2]
    """

    def __init__(self, cfg: STGCNModelConfig, n_nodes: int = 51):
        super().__init__()
        self.cfg = cfg
        self.n_nodes = n_nodes

        in_dim = cfg.n_dynamic + cfg.n_static
        ch = cfg.st_conv_channels  # e.g., [64, 64, 128]

        self.input_projection = nn.Linear(in_dim, ch[0])

        # Stacked ST-Conv Blocks
        self.st_block1 = STConvBlock(
            ch[0], ch[1], ch[1], kernel_size=cfg.temporal_kernel_size, dropout=cfg.dropout
        )
        self.st_block2 = STConvBlock(
            ch[1], ch[1], ch[2], kernel_size=cfg.temporal_kernel_size, dropout=cfg.dropout
        )

        # Temporal Pooling Attention
        self.temporal_att = nn.Linear(ch[2], 1)

        # Multi-task Heads
        head_in = ch[2]
        self.cls_head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden2),
            nn.GELU(),
            nn.Linear(cfg.head_hidden2, cfg.n_cls_heads),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden2),
            nn.GELU(),
            nn.Linear(cfg.head_hidden2, cfg.n_reg_heads),
        )

    def _compute_norm_adj(self, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute normalized adjacency matrix A_hat = D^-1/2 (A + I) D^-1/2."""
        N = self.n_nodes
        adj = torch.zeros((N, N), device=device)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
            adj[edge_index[1], edge_index[0]] = 1.0  # Symmetrize for STGCN GCN layer

        # Add self-loops (A_tilde = A + I)
        adj = adj + torch.eye(N, device=device)

        # Degree matrix D_tilde
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_mat = torch.diag(deg_inv_sqrt)

        # A_hat = D^-1/2 @ A_tilde @ D^-1/2
        norm_adj = d_mat @ adj @ d_mat
        return norm_adj

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # x: [B, T, N, F_d]
        # s: [N, F_s]
        B, T, N, F_d = x.shape

        # Expand static features across B and T: [B, T, N, F_s]
        s_exp = s.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)

        # Concatenate dynamic + static: [B, T, N, F_d + F_s]
        feat = torch.cat([x, s_exp], dim=-1)

        # Project input features: [B, T, N, C_0]
        h = self.input_projection(feat)

        # Reshape to [B, C_0, T, N] for Conv2d
        h = h.permute(0, 3, 1, 2)

        # Compute normalized adjacency matrix
        norm_adj = self._compute_norm_adj(edge_index, device=x.device)

        # Forward ST-Conv blocks
        h = self.st_block1(h, norm_adj)  # [B, C_1, T, N]
        h = self.st_block2(h, norm_adj)  # [B, C_2, T, N]

        # Reshape back to [B, T, N, C_2] for temporal attention pooling
        h_flat = h.permute(0, 2, 3, 1)

        # Learned temporal attention weights across T
        att_weights = torch.softmax(self.temporal_att(h_flat), dim=1)  # [B, T, N, 1]
        z = (h_flat * att_weights).sum(dim=1)                          # [B, N, C_2]

        # Compute multi-head predictions
        logits = self.cls_head(z)  # [B, N, 4]
        reg = self.reg_head(z)     # [B, N, 2]

        return {"logits": logits, "reg": reg}
