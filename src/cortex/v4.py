"""
System Classification: src.cortex.v4
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
License: Apache-2.0

Theoretical Framework:
    - SubstanceNet theoretical framework (Onasenko, 2025-2026)
    - Visual Cortex V4 (Zeki, 1983)

Visual Cortex V4 — Object Feature Extraction
===========================================================
V4 creates position-invariant object-level representations.
Multi-scale attention pooling selects important features,
then Hebbian compression learns which feature combinations
co-occur (without backprop).

Biological basis: V4 neurons develop selectivity through
experience — unlike V1/V2 which are innate. V4 learns
complex shapes, curvature, object parts through exposure.

Mathematical Basis:
    V4(x) = HebbianCompress(MultiScalePool(x))
    P_hat = V4 . V3 . V2 . V1  (complete projection)

Key References:
    - Zeki S. (1983) Neuroscience 9:741-765
    - Pasupathy A., Connor C.E. (2001) J. Neurophysiol. 86:2505-2519

Changelog:
    2026-03-15 v1.0.0 — Initial (random Linear)
    2026-03-18 v2.0.0 — Hebbian compression (no backprop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cortex.hebbian import HebbianLinear


class ObjectFeaturesV4(nn.Module):
    """
    V4 cortex: object-level feature extraction with Hebbian learning.

    Multi-scale attention pooling + Hebbian feature compression.
    Compression weights evolve through co-activation patterns,
    not gradient descent.

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    num_scales : int
        Number of attention scales.
    compression : float
        Bottleneck ratio (0-1).

    Notes
    -----
    Without Hebbian maturation, V4 compression may degrade features
    (exp03: after_v4=0.628 < after_v2=0.684). This is expected —
    randomly initialized HebbianLinear acts as noise. Maturation on
    relevant data (exp05) resolves this (+2.4% recognition).
    """

    def __init__(self, dim: int, num_scales: int = 3,
                 compression: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        # Multi-scale attention (small networks, keep as nn.Linear)
        self.scale_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // num_scales),
                nn.Tanh(),
                nn.Linear(dim // num_scales, 1),
            )
            for _ in range(num_scales)
        ])

        # Hebbian compression: learns which feature combinations
        # co-occur across scales. No backprop needed.
        bottleneck_dim = max(int(dim * compression), 1)
        self.compress_in = HebbianLinear(
            dim * num_scales, bottleneck_dim,
            learning_rate=0.0001,
            oja_alpha=1.0,
        )
        self.compress_out = HebbianLinear(
            bottleneck_dim, dim,
            learning_rate=0.0001,
            oja_alpha=1.0,
        )
        self.compress_norm = nn.LayerNorm(dim)

        # Residual gate
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale attention pooling + Hebbian compression.

        Parameters
        ----------
        x : torch.Tensor
            V3 output [B, seq_len, dim].

        Returns
        -------
        torch.Tensor
            V4 features [B, seq_len, dim].
        """
        batch, seq_len, dim = x.shape

        # Multi-scale attention pooling
        scale_outputs = []
        attention_weights = []
        for scale_attn in self.scale_weights:
            attn = F.softmax(scale_attn(x), dim=1)  # [B, seq, 1]
            pooled = (attn * x).sum(dim=1, keepdim=True)
            scale_outputs.append(pooled.expand_as(x))
            attention_weights.append(attn)

        # Concatenate scales
        multi_scale = torch.cat(scale_outputs, dim=-1)  # [B, seq, dim*scales]

        # Hebbian compression (learns co-activation patterns)
        # Phase signal: mean attention as coherence indicator
        attn_coherence = torch.cat(attention_weights, dim=-1)  # [B, seq, scales]

        compressed = F.relu(self.compress_in(
            multi_scale.reshape(-1, dim * self.num_scales),
            phase=attn_coherence.reshape(-1, self.num_scales)
        ))
        compressed = self.compress_out(compressed)
        compressed = self.compress_norm(
            compressed.reshape(batch, seq_len, dim))

        # Residual
        g = torch.sigmoid(self.residual_gate)
        return g * compressed + (1 - g) * x
