"""Cross-spatiotemporal interaction network for RUL prediction.

The model treats every (temporal patch, sensor-variable) pair as one node. A
directed graph-attention layer only sends messages from patch p-1 to patch p,
jointly modeling temporal degradation and cross-variable interactions while
preserving the forward direction of time.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

class SensorPatchEmbedding(nn.Module):
    """Convert [batch, time, sensor] windows into patch-sensor node tokens."""

    def __init__(
        self,
        window_size: int,
        in_features: int,
        patch_size: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if window_size <= 0 or in_features <= 0 or patch_size <= 0 or d_model <= 0:
            raise ValueError("window_size, in_features, patch_size and d_model must be positive.")
        if window_size % patch_size != 0:
            raise ValueError(
                f"window_size ({window_size}) must be divisible by patch_size ({patch_size})."
            )

        self.window_size = window_size
        self.in_features = in_features
        self.patch_size = patch_size
        self.num_patches = window_size // patch_size

        self.patch_projection = nn.Linear(patch_size, d_model)
        self.sensor_embedding = nn.Parameter(torch.empty(1, 1, in_features, d_model))
        self.patch_embedding = nn.Parameter(torch.empty(1, self.num_patches, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.sensor_embedding, std=0.02)
        nn.init.trunc_normal_(self.patch_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input [batch, time, sensor], got shape {tuple(x.shape)}.")
        if x.shape[1] != self.window_size or x.shape[2] != self.in_features:
            raise ValueError(
                "Unexpected input shape: "
                f"expected [batch, {self.window_size}, {self.in_features}], got {tuple(x.shape)}."
            )

        # [B, W, C] -> [B, C, P, L] -> [B, P, C, L]
        patches = x.transpose(1, 2).unfold(-1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        nodes = self.patch_projection(patches)
        return self.dropout(nodes + self.sensor_embedding + self.patch_embedding)


class AdjacentDirectedGraphAttention(nn.Module):
    """Attend from all sensor variables in patch p-1 to patch p."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, sensor_count, _ = x.shape
        x = x.reshape(batch, patches, sensor_count, self.n_heads, self.head_dim)
        return x.permute(0, 3, 1, 2, 4)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 4:
            raise ValueError(f"Expected node tensor [B, P, C, D], got {tuple(x.shape)}.")
        if x.shape[1] < 2:
            raise ValueError("At least two patches are required for adjacent-patch attention.")

        query = self._split_heads(self.q_projection(x))[:, :, 1:, :, :]
        key = self._split_heads(self.k_projection(x))[:, :, :-1, :, :]
        value = self._split_heads(self.v_projection(x))[:, :, :-1, :, :]

        # A[b,h,p,t,s] is the edge weight from source sensor s in patch p
        # to target sensor t in patch p+1. Softmax is over source sensors.
        scores = torch.einsum("bhptd,bhpsd->bhpts", query, key) * self.scale
        attention = torch.softmax(scores, dim=-1)
        messages = torch.einsum(
            "bhpts,bhpsd->bhptd", self.attention_dropout(attention), value
        )

        batch, _, transitions, sensor_count, _ = messages.shape
        messages = messages.permute(0, 2, 3, 1, 4).contiguous()
        messages = messages.reshape(batch, transitions, sensor_count, self.d_model)
        messages = self.out_projection(messages)

        # Patch 0 has no incoming temporal edge. Concatenation avoids an in-place
        # update and makes the causal boundary explicit.
        first_patch = torch.zeros_like(x[:, :1, :, :])
        update = torch.cat([first_patch, messages], dim=1)
        return update, attention if return_attention else None


class CSINetEncoderLayer(nn.Module):
    """Pre-normalized directed graph-attention block followed by an FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.graph_norm = nn.LayerNorm(d_model)
        self.graph_attention = AdjacentDirectedGraphAttention(d_model, n_heads, dropout)
        self.graph_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        graph_update, attention = self.graph_attention(
            self.graph_norm(x), return_attention=return_attention
        )
        x = x + self.graph_dropout(graph_update)
        x = x + self.ffn(self.ffn_norm(x))
        return x, attention


class PostGraphGlobalAttention(nn.Module):
    """Let final-patch nodes read all graph-encoded nodes causally."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, nodes: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if nodes.ndim != 4:
            raise ValueError(f"Expected node tensor [B, P, C, D], got {tuple(nodes.shape)}.")

        batch, patches, sensor_count, d_model = nodes.shape
        final_nodes = nodes[:, -1, :, :]
        memory = nodes.reshape(batch, patches * sensor_count, d_model)
        normalized_memory = self.memory_norm(memory)
        attention_update, attention = self.attention(
            self.query_norm(final_nodes),
            normalized_memory,
            normalized_memory,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        final_nodes = final_nodes + self.attention_dropout(attention_update)
        final_nodes = final_nodes + self.ffn(self.ffn_norm(final_nodes))

        if attention is not None:
            # [B, H, target sensor, source patch, source sensor]
            attention = attention.reshape(
                batch, attention.shape[1], sensor_count, patches, sensor_count
            )
        return final_nodes, attention


class SensorHealthReadout(nn.Module):
    """Pool the final patch's sensor-variable nodes into a health vector."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(d_model // 2, 1)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, final_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sensor_weights = torch.softmax(self.score(final_patch), dim=1)
        health_vector = torch.sum(sensor_weights * final_patch, dim=1)
        return health_vector, sensor_weights.squeeze(-1)


class CSINet(nn.Module):
    """Cross-Spatiotemporal Interaction Network for scalar RUL regression."""

    def __init__(
        self,
        window_size: int = 30,
        in_features: int = 14,
        patch_size: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_post_graph_attention: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")

        self.window_size = window_size
        self.in_features = in_features
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.use_post_graph_attention = use_post_graph_attention

        self.embedding = SensorPatchEmbedding(
            window_size, in_features, patch_size, d_model, dropout
        )
        self.num_patches = self.embedding.num_patches
        self.encoder_layers = nn.ModuleList(
            [
                CSINetEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.post_graph_attention = (
            PostGraphGlobalAttention(d_model, n_heads, d_ff, dropout)
            if use_post_graph_attention
            else None
        )
        self.readout = SensorHealthReadout(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def encode_nodes(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        nodes = self.embedding(x)
        attentions: List[torch.Tensor] = []
        for layer in self.encoder_layers:
            nodes, attention = layer(nodes, return_attention=return_attention)
            if attention is not None:
                attentions.append(attention)
        return self.final_norm(nodes), attentions

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        nodes, _ = self.encode_nodes(x, return_attention=False)
        if self.post_graph_attention is None:
            final_nodes = nodes[:, -1, :, :]
        else:
            final_nodes, _ = self.post_graph_attention(nodes, return_attention=False)
        health_vector, _ = self.readout(final_nodes)
        return health_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.feature_extractor(x))

    def forward_with_attention(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor | None, torch.Tensor]:
        """Return prediction and graph, post-graph and readout attention weights."""
        nodes, attentions = self.encode_nodes(x, return_attention=True)
        if self.post_graph_attention is None:
            final_nodes = nodes[:, -1, :, :]
            post_graph_attention = None
        else:
            final_nodes, post_graph_attention = self.post_graph_attention(
                nodes, return_attention=True
            )
            if post_graph_attention is None:
                raise RuntimeError("Post-graph attention weights were not returned.")
        health_vector, sensor_weights = self.readout(final_nodes)
        prediction = self.regression_head(health_vector)
        return prediction, attentions, post_graph_attention, sensor_weights


__all__ = [
    "AdjacentDirectedGraphAttention",
    "CSINet",
    "CSINetEncoderLayer",
    "PostGraphGlobalAttention",
    "SensorHealthReadout",
    "SensorPatchEmbedding",
]
