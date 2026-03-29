"""
System Classification: src.wave.quantum_wave
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - The Emergence Parameter kappa ~ 1 (Onasenko, 2025)

Quantum Wave Function Module
===========================================================
Implements psi = A * exp(i * phi) wave function generation for neural
network feature processing. Maps oriented features to complex-valued
representations on a computational analogue of manifold Sigma.

Mathematical Basis:
    psi_i(xi, eta) = A_i(xi, eta) * exp(i * phi_i(xi, eta))  — Def 1.5
    n_i = (1/2pi) oint nabla phi_i . dl  — topological number
    Theta_delta(|psi| - epsilon) = 0.5 * (1 + tanh((|psi| - eps) / delta))
    L_0(p) = delta_eps(|psi|) * |nabla psi|^2  — zero minimization

Key References:
    - Onasenko O. (2026) Monograph "2D-Substance", Chapter 1
    - Onasenko O. (2026) Monograph "2D-Substance", Chapter 6, Appendix 6.A.2

Changelog:
    2026-02-11 v0.1.0 — Ported from SubstanceNet v3.2 quantum_wavefunction.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumWaveFunction(nn.Module):
    """
    Quantum-inspired wave function: psi = A * exp(i * phi).

    Generates complex-valued wave functions from oriented features,
    implementing the computational analogue of wave functions on
    manifold Sigma from 2D-Substance Theory.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input oriented features.
    out_channels : int
        Total output channels (split equally into amplitude and phase).
        Must be even.
    grid_size : int
        Spatial grid size for wave function computation.

    Attributes
    ----------
    epsilon : nn.Parameter
        Projection threshold (Def 6.2: epsilon ~ 10^-3 MeV).
    delta : nn.Parameter
        Regularization parameter for Heaviside function (delta << epsilon).
    gamma_0 : nn.Parameter
        Weight for zero-minimization loss L_0.
    """

    def __init__(self, in_channels: int, out_channels: int, grid_size: int = 256):
        super().__init__()

        if out_channels % 2 != 0:
            raise ValueError("out_channels must be even.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.half_out = out_channels // 2

        # Amplitude and phase projections
        self.amplitude_fc = nn.Linear(in_channels, self.half_out)
        self.phase_fc = nn.Linear(in_channels, self.half_out)

        # Theory-derived parameters (Chapter 6, Appendix 6.A.2)
        self.gamma_0 = nn.Parameter(torch.tensor(1e-3))
        self.epsilon = nn.Parameter(torch.tensor(1e-3))
        self.delta = nn.Parameter(torch.tensor(1e-6))

    def forward(self, oriented_features: torch.Tensor):
        """
        Compute complex wave function from oriented features.

        Parameters
        ----------
        oriented_features : torch.Tensor
            Shape [B, grid_size, in_channels].

        Returns
        -------
        psi_complex : torch.Tensor (complex)
            Shape [B, grid_size, half_out]. Complex wave function.
        amplitude : torch.Tensor
            Shape [B, grid_size, half_out]. Non-negative amplitude A.
        phase : torch.Tensor
            Shape [B, grid_size, half_out]. Phase phi (unbounded).
        """
        amplitude = F.softplus(self.amplitude_fc(oriented_features))
        phase = self.phase_fc(oriented_features)

        real_part = amplitude * torch.cos(phase)
        imag_part = amplitude * torch.sin(phase)
        psi_complex = torch.complex(real_part, imag_part)

        return psi_complex, amplitude, phase

    def theta_delta(self, amplitude: torch.Tensor) -> torch.Tensor:
        """
        Regularized Heaviside: Theta_delta(|psi| - epsilon).

        From Appendix 6.A.2:
            Theta_delta(x) = 0.5 * (1 + tanh(x / delta))
        """
        return 0.5 * (1.0 + torch.tanh((amplitude - self.epsilon) / self.delta))

    def zero_loss(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Approximate zero-minimization loss L_0.

        L_0(p) = delta_eps(|psi|) * |nabla psi|^2

        Approximated via phase gradient penalty weighted by amplitude
        and projection mask Theta_delta.

        Parameters
        ----------
        amplitude : torch.Tensor
            Shape [B, grid_size, channels].
        phase : torch.Tensor
            Shape [B, grid_size, channels].

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        """
        if amplitude.shape[1] <= 1:
            return torch.tensor(0.0, device=amplitude.device, dtype=amplitude.dtype)

        theta_mask = self.theta_delta(amplitude)
        phase_grad = phase[:, 1:, :] - phase[:, :-1, :]
        avg_amp = 0.5 * (amplitude[:, 1:, :] + amplitude[:, :-1, :])
        penalty = theta_mask[:, :-1, :] * avg_amp * (phase_grad ** 2)

        return self.gamma_0 * torch.mean(penalty)
