"""
System Classification: src.model.layers
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - Nonlocal Potential V_ij (Chapter 4)
    - Visual Cortex V2 Stripes (Hubel & Wiesel, 1968)

Model Assembly Layers
===========================================================
Helper layers that connect core modules into a working network.
These are NOT standalone tools like core modules, but glue
components for the SubstanceNet model template.

Layers:
    - OrientationSelectivity: V1 -> oriented features (per-channel)
    - NonLocalInteraction: attention-based V_ij analogue
    - AbstractionLayer: spatial -> abstract representation
    - PhaseCoherenceLoss: phase alignment regularization
    - TopologicalLoss: topological number n_i regularization

Changelog:
    2026-02-11 v0.1.0 — Extracted from v3.2 models_v2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrientationSelectivity(nn.Module):
    """
    Orientation selectivity via grouped 1D convolution.

    Expands features into multiple orientation channels,
    analogous to V1 orientation columns processing.

    Parameters
    ----------
    in_channels : int
        Input feature dimensionality.
    num_orientations : int
        Number of orientation channels per feature.
    """

    def __init__(self, in_channels: int, num_orientations: int = 8):
        super().__init__()
        self.num_orientations = num_orientations
        self.conv = nn.Conv1d(
            in_channels, num_orientations * in_channels,
            kernel_size=3, padding=1, bias=False,
            groups=in_channels,
        )

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Features [B, seq_len, in_channels].

        Returns
        -------
        torch.Tensor
            Oriented features [B, seq_len, in_channels * num_orientations].
        """
        return F.relu(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))


class NonLocalInteraction(nn.Module):
    """
    Attention-based nonlocal interaction, analogue of V_ij.

    Implements scaled dot-product attention with gated residual,
    modelling the nonlocal potential between wave functions on Sigma.

    Parameters
    ----------
    dim : int
        Feature dimensionality.
    num_heads : int
        Number of attention heads.
    gate_init : float
        Initial gate value (controls attention vs residual balance).
    """

    def __init__(self, dim: int, num_heads: int = 2,
                 gate_init: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Features [B, seq_len, dim].

        Returns
        -------
        torch.Tensor
            Processed features [B, seq_len, dim].
        """
        g = torch.sigmoid(self.gate)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + g * attn_out)


class AbstractionLayer(nn.Module):
    """
    Reduces spatial features to abstract representation.

    Pools across spatial/sequential dimension then projects
    to abstract_dim, creating the input for consciousness module.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    abstract_dim : int
        Output abstract representation dimensionality.
    dropout : float
        Dropout rate.
    """

    def __init__(self, in_dim: int, abstract_dim: int,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, abstract_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Spatial features [B, seq_len, in_dim].

        Returns
        -------
        torch.Tensor
            Abstract representation [B, abstract_dim].
        """
        return self.net(x.mean(dim=1))


class PhaseCoherenceLoss(nn.Module):
    """
    Phase coherence regularization.

    Encourages alignment of phases across spatial positions,
    corresponding to coherent wave function on Sigma.

    Parameters
    ----------
    weight : float
        Loss weight.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        phases : torch.Tensor
            Phase values [B, seq_len, channels].

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if phases.ndim != 3:
            return torch.tensor(0.0, device=phases.device)
        diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        return self.weight * torch.mean(diff ** 2)


class TopologicalLoss(nn.Module):
    """
    Topological number regularization.

    Encourages phase field to have target winding number n_i,
    corresponding to topological invariant from Def 1.5.

    Parameters
    ----------
    weight : float
        Loss weight.
    target_n : float
        Target topological number.
    """

    def __init__(self, weight: float = 0.01, target_n: float = 0.5):
        super().__init__()
        self.weight = weight
        self.target_n = target_n

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        phases : torch.Tensor
            Phase values [B, H, W] (2D field).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if phases.ndim != 3 or phases.shape[1] < 2 or phases.shape[2] < 2:
            return torch.tensor(0.0, device=phases.device)
        gy, gx = torch.gradient(phases, dim=(1, 2))
        curl = torch.mean(torch.abs(
            gx[:, :, 1:] - gy[:, :, :-1]))
        n_approx = curl / (2 * math.pi)
        return self.weight * (n_approx - self.target_n) ** 2
