"""
System Classification: src.hippocampus.cells
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - Hippocampal Spatial Representation (O'Keefe & Dostrovsky, 1971)
    - Grid Cells (Hafting et al., 2005)

Hippocampal Cell Types
===========================================================
Implements biologically-inspired cell types found in hippocampus:
    - Grid cells: hexagonal spatial coding (entorhinal cortex)
    - Place cells: location-specific firing (CA1/CA3)
    - Time cells: temporal sequence coding (CA1)

These provide the spatial-temporal scaffolding for episodic memory
formation and retrieval, analogous to the coordinate system on
manifold Sigma in 2D-Substance Theory.

Mathematical Basis:
    Grid cells: cos(2 pi <pos, v_k> / scale_i) for 6 hex directions
    Place cells: exp(-||x - c_j||^2 / (2 sigma^2))  Gaussian fields
    Time cells: exp(-t / tau_k)  logarithmic time constants

Key References:
    - O'Keefe J., Dostrovsky J. (1971) Brain Res. 34:171-175
    - Hafting T. et al. (2005) Nature 436:801-806
    - MacDonald C.J. et al. (2011) Neuron 71:737-749
    - Moser E.I. et al. (2008) Annu. Rev. Neurosci. 31:69-89

Changelog:
    2026-02-11 v0.1.0 — Ported from SubstanceNet v3.2 hippocampus_module_v2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GridCells(nn.Module):
    """
    Grid cells for spatial coding via hexagonal lattices.

    Models the multi-scale periodic spatial representations found in
    medial entorhinal cortex. Each module operates at a different
    spatial scale, providing a hierarchical coordinate system.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    num_modules : int
        Number of grid modules at different scales.
    scale_ratio : float
        Ratio between consecutive grid scales.

    Notes
    -----
    Output dimensionality: num_modules * 6 (6 hexagonal directions).
    """

    def __init__(self, input_dim: int, num_modules: int = 4,
                 scale_ratio: float = 1.4):
        super().__init__()
        self.num_modules = num_modules
        self.scales = [scale_ratio ** i for i in range(num_modules)]

        # Project input to 2D position for each module
        self.projections = nn.ModuleList([
            nn.Linear(input_dim, 2) for _ in range(num_modules)
        ])

        # 6 hexagonal directions (60 degree spacing)
        angles = [2 * math.pi * k / 6 for k in range(6)]
        hex_vectors = torch.tensor(
            [[math.cos(a), math.sin(a)] for a in angles],
            dtype=torch.float32,
        )
        self.register_buffer('hex_vectors', hex_vectors)

    @property
    def output_dim(self) -> int:
        return self.num_modules * 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute grid cell activations.

        Parameters
        ----------
        x : torch.Tensor
            Input features [B, input_dim].

        Returns
        -------
        torch.Tensor
            Grid cell activations [B, num_modules * 6].
        """
        activations = []

        for projection, scale in zip(self.projections, self.scales):
            pos_2d = projection(x)  # [B, 2]

            for vec in self.hex_vectors:
                proj = torch.sum(pos_2d * vec.unsqueeze(0), dim=1)
                activations.append(torch.cos(2 * math.pi * proj / scale))

        return torch.stack(activations, dim=1)


class PlaceCells(nn.Module):
    """
    Place cells for location-specific coding.

    Models CA1/CA3 place cells with Gaussian receptive fields
    centered at learned positions in a projected feature space.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    num_places : int
        Number of place cell units.
    field_std : float
        Standard deviation of Gaussian place fields.
    projection_dim : int
        Dimensionality of intermediate projection space.
    """

    def __init__(self, input_dim: int, num_places: int = 100,
                 field_std: float = 0.1, projection_dim: int = 64):
        super().__init__()
        self.num_places = num_places
        self.field_std = field_std

        self.projection = nn.Linear(input_dim, projection_dim)
        self.centers = nn.Parameter(torch.randn(num_places, projection_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute place cell activations.

        Parameters
        ----------
        x : torch.Tensor
            Input features [B, input_dim].

        Returns
        -------
        torch.Tensor
            Place cell activations [B, num_places].
        """
        projected = self.projection(x)
        distances = torch.cdist(projected.unsqueeze(1),
                                self.centers.unsqueeze(0)).squeeze(1)
        return torch.exp(-(distances ** 2) / (2 * self.field_std ** 2))


class TimeCells(nn.Module):
    """
    Time cells for temporal coding.

    Models CA1 time cells with logarithmically-spaced time constants,
    providing a multi-scale temporal representation.

    Parameters
    ----------
    num_cells : int
        Number of time cell units.
    max_tau : float
        Maximum time constant.
    """

    def __init__(self, num_cells: int = 50, max_tau: float = 10.0):
        super().__init__()
        self.num_cells = num_cells

        tau = torch.logspace(-1, math.log10(max_tau), num_cells)
        self.register_buffer('tau', tau)

    def forward(self, elapsed_time: float) -> torch.Tensor:
        """
        Compute time cell activations.

        Parameters
        ----------
        elapsed_time : float
            Time elapsed since episode start (seconds).

        Returns
        -------
        torch.Tensor
            Time cell activations [num_cells].
        """
        return torch.exp(-elapsed_time / self.tau)
