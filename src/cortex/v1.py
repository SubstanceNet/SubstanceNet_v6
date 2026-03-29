"""
System Classification: src.cortex.v1
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

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


class RetinalLayer(nn.Module):
    """
    Retinal preprocessing: converts RGB to 4 photoreceptor channels.

    Models the four types of photoreceptors in the human retina:
        - Rods:    Luminance (achromatic, scotopic vision)
        - L-cones: Long wavelength (~564nm, "red")
        - M-cones: Medium wavelength (~534nm, "green")
        - S-cones: Short wavelength (~420nm, "blue")

    Uses fixed spectral sensitivity functions based on
    CIE physiological data (Stockman & Sharpe, 2000).

    For grayscale input (1 channel), duplicates to rods only.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for grayscale, 3 for RGB).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        if in_channels == 3:
            # Spectral sensitivity matrix: RGB → [Rods, L, M, S]
            # Based on CIE cone fundamentals (approximate)
            sensitivity = torch.tensor([
                [0.2126, 0.7152, 0.0722],  # Rods (luminance, ITU-R BT.709)
                [0.7000, 0.3000, 0.0000],  # L-cones (red-sensitive)
                [0.2000, 0.7000, 0.1000],  # M-cones (green-sensitive)
                [0.0000, 0.1000, 0.9000],  # S-cones (blue-sensitive)
            ], dtype=torch.float32)
            # Shape for 1x1 conv: [out_channels=4, in_channels=3, 1, 1]
            self.register_buffer(
                'spectral_weights',
                sensitivity.unsqueeze(-1).unsqueeze(-1))

            # Center-surround antagonism (retinal ganglion cells)
            self.center_surround = nn.Sequential(
                nn.Conv2d(4, 4, kernel_size=5, padding=2, groups=4, bias=False),
                nn.BatchNorm2d(4),
            )
        else:
            # Grayscale: single rod channel
            self.spectral_weights = None
            self.center_surround = None

    @property
    def out_channels(self) -> int:
        return 4 if self.in_channels == 3 else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image [B, 1, H, W] or [B, 3, H, W].

        Returns
        -------
        torch.Tensor
            Retinal activations [B, 4, H, W] or [B, 1, H, W].
        """
        if self.in_channels == 3 and x.shape[1] == 3:
            # RGB → 4 photoreceptor channels
            retinal = F.conv2d(x, self.spectral_weights.type_as(x))
            # Center-surround antagonism
            retinal = retinal + self.center_surround(retinal)
            return retinal
        else:
            # Grayscale passthrough
            return x


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
                 size: int = 11, in_channels: int = 1):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.size = size
        self.in_channels = in_channels

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
        filter_single = torch.stack(filters).unsqueeze(1).float()
        # Replicate Gabor across input channels (each V1 cell sees all retinal channels)
        filter_tensor = filter_single.repeat(1, self.in_channels, 1, 1) / self.in_channels
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
                 output_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels

        # Retinal preprocessing (RGB -> 4 photoreceptor channels)
        if in_channels == 3:
            self.retina = RetinalLayer(in_channels=3)
            gabor_in = self.retina.out_channels  # 4
        else:
            self.retina = None
            gabor_in = 1

        self.gabor_bank = GaborFilterBank(n_orientations, n_scales,
                                          in_channels=gabor_in)
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
        # Retinal preprocessing (if RGB input)
        if self.retina is not None and x.shape[1] == 3:
            x = self.retina(x)
        elif x.shape[1] == 3 and self.retina is None:
            # Fallback: convert RGB to grayscale if no retina
            x = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]

        # Retinal preprocessing (if RGB input)
        if self.retina is not None and x.shape[1] == 3:
            x = self.retina(x)
        elif x.shape[1] == 3 and self.retina is None:
            # Fallback: convert RGB to grayscale
            x = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]

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
