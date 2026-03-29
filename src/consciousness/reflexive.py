"""
System Classification: src.consciousness.reflexive
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - The Emergence Parameter kappa ~ 1 (Onasenko, 2025)
    - Reflexive Consciousness Theorem (Th 6.22)

Reflexive Consciousness Module
===========================================================
Implements the reflexive consciousness mechanism:
    psi_C = F[P_hat[psi_C]]

The system evaluates its own processing output through iterative
refinement. Reflexivity level R in [0,1] represents the depth
of self-observation.

Empirical finding: optimal R in [0.35, 0.47] (SubstanceNet v3.1.1).
This corresponds to kappa ~ 1 critical regime.

Mathematical Basis:
    psi_C = F[P_hat[psi_C]]  — reflexive projection (Th 6.22)
    P_hat = P3 . P2 . P1  — projection operator (Chapter 6)
    Theta_delta(|psi| - eps)  — projection threshold

Key References:
    - Onasenko O. (2026) Monograph "2D-Substance", Chapter 6, Th 6.22
    - Yerkes R.M., Dodson J.D. (1908) J. Comp. Neurol. Psychol. 18:459-482
    - Beggs J.M., Plenz D. (2003) J. Neurosci. 23:11167-11177

Changelog:
    2026-02-11 v0.1.0 — Ported from SubstanceNet v3.2 reflexive_consciousness_v2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflexiveConsciousness(nn.Module):
    """
    Reflexive consciousness: psi_C = F[P_hat[psi_C]].

    Implements iterative self-referential processing where the system
    projects its consciousness state and then transforms it back,
    converging toward a fixed point.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the abstract network state.
    consciousness_dim : int
        Dimensionality of consciousness state (must be even).
    num_iterations : int
        Number of reflexive iterations (depth of self-observation).

    Notes
    -----
    The consciousness_dim is split into amplitude (first half) and
    phase (second half), following psi = A * exp(i * phi).
    """

    def __init__(self, input_dim: int, consciousness_dim: int = 32,
                 num_iterations: int = 3):
        super().__init__()

        if consciousness_dim % 2 != 0:
            raise ValueError("consciousness_dim must be even.")

        self.input_dim = input_dim
        self.consciousness_dim = consciousness_dim
        self.half_dim = consciousness_dim // 2
        self.num_iterations = num_iterations

        # Projection threshold from theory (Chapter 6)
        self.epsilon = nn.Parameter(torch.tensor(1e-3))
        self.delta = nn.Parameter(torch.tensor(1e-6))

        # F: Reflexive transformation F[P_hat[psi_C], network_state]
        self.F_transform = nn.Sequential(
            nn.Linear(consciousness_dim + input_dim, consciousness_dim * 2),
            nn.Tanh(),
            nn.Linear(consciousness_dim * 2, consciousness_dim),
        )

        # P_hat: Simplified projection operator
        self.projection = nn.Sequential(
            nn.Linear(consciousness_dim, consciousness_dim),
            nn.LayerNorm(consciousness_dim),
        )

        # Learnable initial consciousness state
        self.psi_c_init = nn.Parameter(torch.randn(1, consciousness_dim) * 0.1)

        # Stability coefficient (prevents chaotic dynamics)
        self.register_buffer('stability_alpha', torch.tensor(0.8))

        # Phase coupling strength
        self.phase_coupling = nn.Parameter(torch.tensor(1.0))

    def theta_delta(self, amplitude: torch.Tensor) -> torch.Tensor:
        """Regularized Heaviside: Theta_delta(|psi| - epsilon)."""
        return 0.5 * (1.0 + torch.tanh((amplitude - self.epsilon) / self.delta))

    def project(self, psi_c: torch.Tensor) -> torch.Tensor:
        """
        Simplified projection operator P_hat.

        Applies threshold filtering and normalization, analogous to
        P_hat = P3 . P2 . P1 from Chapter 6.
        """
        amp, ph = psi_c.chunk(2, dim=-1)

        # Threshold filtering (P1): Theta_delta(|psi| - epsilon)
        amp_pos = F.softplus(amp)
        amp_projected = amp_pos * self.theta_delta(amp_pos)

        # Phase normalization to [-pi, pi]
        ph_normalized = torch.tanh(ph) * math.pi

        projected = torch.cat([amp_projected, ph_normalized], dim=-1)
        return self.projection(projected)

    def forward(self, network_state: torch.Tensor):
        """
        Iterative reflexive evolution: psi_C = F[P_hat[psi_C]].

        Parameters
        ----------
        network_state : torch.Tensor
            Abstract network state [B, input_dim].

        Returns
        -------
        psi_c_complex : torch.Tensor (complex)
            Complex consciousness wave function [B, half_dim].
        amplitude : torch.Tensor
            Consciousness amplitude [B, half_dim].
        phase : torch.Tensor
            Consciousness phase [B, half_dim].
        """
        batch_size = network_state.shape[0]
        psi_c = self.psi_c_init.expand(batch_size, -1)

        # Iterative reflexive evolution
        for _ in range(self.num_iterations):
            projected = self.project(psi_c)
            combined = torch.cat([projected, network_state], dim=-1)
            psi_c_new = self.F_transform(combined)

            # Stability mixing (prevents chaotic divergence)
            psi_c = self.stability_alpha * psi_c_new + \
                    (1 - self.stability_alpha) * psi_c

        # Final amplitude/phase decomposition
        amplitude = F.softplus(psi_c[:, :self.half_dim])
        phase_raw = psi_c[:, self.half_dim:]
        phase = torch.tanh(phase_raw * self.phase_coupling) * math.pi

        # Complex wave function
        real_part = amplitude * torch.cos(phase)
        imag_part = amplitude * torch.sin(phase)
        psi_c_complex = torch.complex(real_part, imag_part)

        return psi_c_complex, amplitude, phase

    def consciousness_loss(self, amplitude: torch.Tensor,
                           phase: torch.Tensor) -> torch.Tensor:
        """
        Consciousness loss combining reflexivity, coherence,
        stability, and entropy terms.
        """
        device = amplitude.device
        batch_size = amplitude.shape[0]

        # 1. Reflexivity loss: target OPTIMAL R, not R -> 1.0
        # R = 1/(1+MSE), optimal R ~ 0.41 -> target MSE ~ 1.44
        # Instead of minimizing MSE (which pushes R -> 1.0),
        # we penalize deviation from the optimal MSE value.
        # This implements kappa ~ 1 critical regime.
        psi_c = torch.cat([amplitude, phase], dim=-1)
        psi_c_proj = self.project(psi_c)
        mse = F.mse_loss(psi_c, psi_c_proj)
        target_mse = torch.tensor(1.44, device=device)  # R ~ 0.41
        reflexivity_loss = (mse - target_mse) ** 2

        # 2. Phase coherence loss (batch-level)
        if batch_size > 1:
            psi_complex = torch.complex(
                amplitude * torch.cos(phase),
                amplitude * torch.sin(phase),
            )
            psi_mean = torch.mean(psi_complex, dim=0, keepdim=True)
            coherence = torch.abs(psi_mean) / (torch.mean(amplitude) + 1e-8)
            coherence_loss = -torch.mean(coherence)
        else:
            coherence_loss = torch.tensor(0.0, device=device)

        # 3. Stability loss (penalize near-zero amplitudes)
        stability_loss = torch.mean(torch.exp(-amplitude / self.epsilon))

        # 4. Entropy loss (maintain information complexity)
        amp_norm = amplitude / (amplitude.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -torch.sum(amp_norm * torch.log(amp_norm + 1e-8), dim=-1)
        entropy_loss = -torch.mean(entropy) * 0.1

        return (0.3 * reflexivity_loss + 0.3 * coherence_loss +
                0.3 * stability_loss + 0.1 * entropy_loss)

    def get_metrics(self, amplitude: torch.Tensor,
                    phase: torch.Tensor) -> dict:
        """
        Compute consciousness metrics for monitoring.

        Returns dict with: reflexivity_score, phase_coherence,
        mean_amplitude, amplitude_entropy, complexity, stability_ratio.
        """
        with torch.no_grad():
            # Reflexivity score
            psi_c = torch.cat([amplitude, phase], dim=-1)
            psi_c_proj = self.project(psi_c)
            ref_error = F.mse_loss(psi_c, psi_c_proj).item()
            reflexivity_score = 1.0 / (1.0 + ref_error)

            # Phase coherence
            if amplitude.shape[0] > 1:
                psi = torch.complex(
                    amplitude * torch.cos(phase),
                    amplitude * torch.sin(phase),
                )
                psi_mean = torch.mean(psi, dim=0)
                coh_mag = torch.mean(torch.abs(psi_mean)).item()
                phase_coherence = coh_mag / (torch.mean(amplitude).item() + 1e-8)
            else:
                phase_coherence = 1.0

            # Entropy
            amp_norm = amplitude / (amplitude.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(amp_norm * torch.log(amp_norm + 1e-8), dim=-1)

            # Complexity (significant components above epsilon)
            complexity = torch.sum(
                amplitude > self.epsilon.item(), dim=-1
            ).float().mean().item()

            # Stability ratio
            max_a = amplitude.max(dim=-1)[0]
            min_a = amplitude.min(dim=-1)[0] + 1e-8

        return {
            'reflexivity_score': reflexivity_score,
            'phase_coherence': phase_coherence,
            'mean_amplitude': torch.mean(amplitude).item(),
            'amplitude_entropy': torch.mean(entropy).item(),
            'complexity': complexity,
            'stability_ratio': torch.mean(max_a / min_a).item(),
        }
