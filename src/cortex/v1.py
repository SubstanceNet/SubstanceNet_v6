"""
System Classification: src.cortex.v1
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - Visual Cortex Hierarchy (Hubel & Wiesel, 1962, 1968)

Visual Cortex V1 — Orientation Selectivity
===========================================================
Implements V1 simple and complex cells using Gabor-like filters
for edge and orientation detection. This is the first stage of
the projection operator P_hat.

Architecture:
    GaborFilterBank → SimpleCells → ComplexCells → HyperColumns

Mathematical Basis:
    G(x,y) = exp(-(x'^2 + gamma^2 y'^2)/(2 sigma^2)) * cos(2 pi x'/lambda + phi)
    where x' = x*cos(theta) + y*sin(theta), y' = -x*sin(theta) + y*cos(theta)

    Simple cells: half-wave rectification (ON/OFF channels)
    Complex cells: energy model sqrt(even^2 + odd^2)
    Hypercolumns: integration across orientations

Key References:
    - Onasenko O. (2026) Monograph "2D-Substance", Chapter 6
    - Hubel D.H., Wiesel T.N. (1962) J. Physiol. 160:106-154
    - Hubel D.H., Wiesel T.N. (1968) J. Physiol. 195:215-243
    - Daugman J.G. (1985) J. Opt. Soc. Am. A 2:1160-1169

Changelog:
    2026-02-11 v0.1.0 — Ported from SubstanceNet v3.2 biological_v1_encoder.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gabor_filter(size: int = 11, sigma: float = 3.0,
                        theta: float = 0.0, lambda_: float = 10.0,
                        gamma: float = 0.5, psi: float = 0.0) -> torch.Tensor:
    """
    Create a Gabor filter modeling V1 receptive fields.

    Parameters
    ----------
    size : int
        Filter size in pixels.
    sigma : float
        Gaussian envelope standard deviation.
    theta : float
        Orientation angle in radians.
    lambda_ : float
        Sinusoidal wavelength.
    gamma : float
        Spatial aspect ratio.
    psi : float
        Phase offset.

    Returns
    -------
    torch.Tensor
        Normalized Gabor filter [size, size].
    """
    coords = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(coords, coords)

    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    gaussian = np.exp(-(X_rot**2 + gamma**2 * Y_rot**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * X_rot / lambda_ + psi)
    gabor = gaussian * sinusoid

    gabor -= gabor.mean()
    gabor /= np.abs(gabor).max() + 1e-8

    return torch.FloatTensor(gabor)


class GaborFilterBank(nn.Module):
    """
    Bank of Gabor filters for multi-scale, multi-orientation edge detection.
    Models V1 simple cell receptive fields.

    Parameters
    ----------
    n_orientations : int
        Number of orientation channels.
    n_scales : int
        Number of spatial frequency scales.
    size : int
        Filter kernel size.
    """

    def __init__(self, n_orientations: int = 8, n_scales: int = 4,
                 size: int = 11):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.size = size

        filters = []
        for scale_idx in range(n_scales):
            sigma = 2.0 + scale_idx * 0.5
            lambda_ = 8.0 + scale_idx * 2.0

            for orient_idx in range(n_orientations):
                theta = (math.pi * orient_idx) / n_orientations

                # Even (cosine) and odd (sine) phase pair
                filters.append(create_gabor_filter(
                    size, sigma, theta, lambda_, 0.5, 0))
                filters.append(create_gabor_filter(
                    size, sigma, theta, lambda_, 0.5, math.pi / 2))

        # Shape: [n_scales * n_orientations * 2, 1, size, size]
        filter_tensor = torch.stack(filters).unsqueeze(1).float()
        self.register_buffer('gabor_filters', filter_tensor)

    @property
    def num_filters(self) -> int:
        return self.n_orientations * self.n_scales * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gabor filters to grayscale input.

        Parameters
        ----------
        x : torch.Tensor
            Input image [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Filter responses [B, num_filters, H, W].
        """
        return F.conv2d(x, self.gabor_filters.type_as(x),
                        padding=self.size // 2)


class SimpleCells(nn.Module):
    """
    V1 simple cells: orientation-selective with ON/OFF channels.

    Applies half-wave rectification (biological analog) and
    lateral inhibition via layer normalization.
    """

    def __init__(self, gabor_bank: GaborFilterBank):
        super().__init__()
        self.gabor_bank = gabor_bank
        self.norm = nn.LayerNorm([gabor_bank.num_filters])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Grayscale image [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Normalized responses [B, num_filters, H, W].
        """
        responses = self.gabor_bank(x)

        # Half-wave rectification (ON + OFF channels)
        on = F.relu(responses)
        off = F.relu(-responses)
        combined = on + off

        # Lateral inhibition (layer norm across channels)
        b, c, h, w = combined.shape
        combined = combined.permute(0, 2, 3, 1)
        combined = self.norm(combined)
        return combined.permute(0, 3, 1, 2)


class ComplexCells(nn.Module):
    """
    V1 complex cells: phase-invariant via energy model.

    Combines even/odd phase responses: E = sqrt(even^2 + odd^2).
    Applies spatial pooling for position invariance.
    """

    def __init__(self, n_orientations: int = 8, n_scales: int = 4):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.spatial_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, simple_responses: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        simple_responses : torch.Tensor
            Simple cell output [B, n_scales*n_orient*2, H, W].

        Returns
        -------
        torch.Tensor
            Energy responses [B, n_scales*n_orient, H/2, W/2].
        """
        b, c, h, w = simple_responses.shape
        x = simple_responses.view(
            b, self.n_scales, self.n_orientations, 2, h, w)

        even = x[..., 0, :, :]
        odd = x[..., 1, :, :]
        energy = torch.sqrt(even**2 + odd**2 + 1e-8)

        energy = energy.view(b, self.n_scales * self.n_orientations, h, w)
        return self.spatial_pool(energy)


class HyperColumns(nn.Module):
    """
    V1 hypercolumns: integration across orientations with
    center-surround antagonism.
    """

    def __init__(self, n_orientations: int = 8, n_scales: int = 4,
                 output_dim: int = 64):
        super().__init__()
        input_dim = n_orientations * n_scales

        self.integrate = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.center_surround = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, complex_responses: torch.Tensor) -> torch.Tensor:
        integrated = self.integrate(complex_responses)
        return self.center_surround(integrated)


class BiologicalV1(nn.Module):
    """
    Complete primary visual cortex (V1) model.

    Pipeline: GaborFilterBank → SimpleCells → ComplexCells → HyperColumns

    Parameters
    ----------
    n_orientations : int
        Number of orientation channels.
    n_scales : int
        Number of spatial frequency scales.
    output_dim : int
        Output feature dimensionality.

    Returns
    -------
    sequence : torch.Tensor
        [B, seq_len, output_dim] for downstream processing.
    activations : dict
        Intermediate layer activations for analysis.
    """

    def __init__(self, n_orientations: int = 8, n_scales: int = 4,
                 output_dim: int = 64):
        super().__init__()

        self.gabor_bank = GaborFilterBank(n_orientations, n_scales)
        self.simple_cells = SimpleCells(self.gabor_bank)
        self.complex_cells = ComplexCells(n_orientations, n_scales)
        self.hypercolumns = HyperColumns(n_orientations, n_scales, output_dim)

        # Convert spatial features to sequence (receptive field patches)
        self.to_sequence = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Grayscale image [B, 1, H, W].

        Returns
        -------
        sequence : torch.Tensor
            Feature sequence [B, 9, output_dim].
        activations : dict
            Keys: 'simple', 'complex', 'hypercolumns'.
        """
        simple = self.simple_cells(x)
        complex_ = self.complex_cells(simple)
        hypercolumns = self.hypercolumns(complex_)

        # Spatial → sequence (3x3 = 9 patches)
        patches = self.to_sequence(hypercolumns)
        b, c, h, w = patches.shape
        sequence = patches.view(b, c, h * w).transpose(1, 2).contiguous()

        activations = {
            'simple': simple,
            'complex': complex_,
            'hypercolumns': hypercolumns,
        }

        return sequence, activations
