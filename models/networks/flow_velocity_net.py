"""
Velocity network for Conditional Flow Matching.

Predicts a 2D velocity v_θ(z_t, t, c) where:
  - z_t: interpolated 2D coordinates at time t
  - t: scalar time in [0, 1]
  - c: conditioning context (fused DINO features)

The network regresses the constant velocity u_t = z_1 - z_0 for
the OT-path interpolation z_t = (1-t)*z_0 + t*z_1.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1) time values in [0, 1].
        Returns:
            Embedding of shape (B, dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (B, 1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t * freqs.unsqueeze(0)  # (B, half_dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block with optional conditioning injection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class FlowVelocityNet(nn.Module):
    """
    Conditional velocity network for flow matching point tracking.

    Architecture:
        [z_t (2) || time_emb (T) || context (C)] → Linear → ResMLPBlock × N → Linear → v (2)

    The context c is a fused representation from DINO features:
        - Source point embedding (sampled from source frame DINO features)
        - Correlation-based features (similarity of source embedding with target frame)
    """

    def __init__(
        self,
        context_dim: int = 384,
        hidden_dim: int = 512,
        time_embed_dim: int = 64,
        num_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input: z_t (2D coords) + time_embed + context
        input_dim = 2 + time_embed_dim + context_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # 2D velocity output
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer to near-zero for stable training start."""
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (B, 2) interpolated coordinates at time t.
            t: (B,) time values in [0, 1].
            context: (B, context_dim) conditioning context from DINO features.

        Returns:
            v: (B, 2) predicted velocity vector.
        """
        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        x = torch.cat([z_t, t_emb, context], dim=-1)  # (B, 2 + T + C)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        v = self.output_proj(x)
        return v
