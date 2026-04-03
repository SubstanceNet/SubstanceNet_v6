"""
System Classification: src.cortex.hebbian
Author: Oleksii Onasenko
Developer: SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Theoretical Framework:
    - SubstanceNet theoretical framework (Onasenko, 2025-2026)
    - Spike-Timing-Dependent Plasticity (Bi & Poo, 2001)

Hebbian Linear Layer — Phase-Coherence Learning
===========================================================
Replaces nn.Linear with a layer whose weights update based on
phase coherence between input and output:

    dW_ij = eta * cos(phi_i - phi_j) * x_i * y_j - alpha * W_ij * y_j^2

The first term is Hebbian: strengthen connections where phases
are coherent. The second term is Oja's normalization: prevents
unbounded weight growth by decorrelating.

No backpropagation needed for weight updates. The layer can
still participate in a backprop graph if needed (for downstream
modules), but its own weights evolve through local Hebbian rule.

Key References:
    - Hebb D.O. (1949) The Organization of Behavior
    - Oja E. (1982) J. Math. Biol. 15:267-273
    - Bi G., Poo M. (2001) Annu Rev Neurosci 24:139-166

Changelog:
    2026-03-17 v1.0.0 — Initial implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianLinear(nn.Module):
    """
    Linear layer with Hebbian weight updates based on phase coherence.

    Weights update during forward pass (not backward):
        dW = eta * (coherence * x^T @ y - alpha * W * y^2)

    Where coherence = cos(phi_input - phi_output) measures
    phase alignment between pre and post-synaptic signals.

    Parameters
    ----------
    in_features : int
        Input dimensionality.
    out_features : int
        Output dimensionality.
    learning_rate : float
        Hebbian learning rate (eta). Small values for stability.
    oja_alpha : float
        Oja normalization strength. Prevents unbounded growth.
    momentum : float
        Exponential moving average for weight updates.
    """

    def __init__(self, in_features: int, out_features: int,
                 learning_rate: float = 0.001,
                 oja_alpha: float = 0.01,
                 momentum: float = 0.9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eta = learning_rate
        self.oja_alpha = oja_alpha
        self.momentum = momentum

        # Weight matrix (initialized with small random values)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01,
            requires_grad=False  # NO backprop gradients
        )

        # Running average of weight updates (momentum)
        self.register_buffer(
            'weight_update_ema',
            torch.zeros(out_features, in_features))

        # Track whether we are in learning mode
        self.learning = True

    def forward(self, x: torch.Tensor,
                phase: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional Hebbian update.

        Parameters
        ----------
        x : torch.Tensor
            Input features [..., in_features].
        phase : torch.Tensor, optional
            Phase information [..., phase_dim] for coherence.
            If None, uses standard Hebbian (no phase modulation).

        Returns
        -------
        torch.Tensor
            Output [..., out_features].
        """
        # Standard linear transform
        y = F.linear(x, self.weight)

        # Hebbian update (only during training/learning, no grad)
        if self.learning and self.training:
            with torch.no_grad():
                self._hebbian_update(x, y, phase)

        return y

    def _hebbian_update(self, x: torch.Tensor, y: torch.Tensor,
                        phase: torch.Tensor = None):
        """
        Update weights via phase-coherence Hebbian rule.

        dW = eta * (coherence * x^T @ y - alpha * W * y^2)

        Oja term (-alpha * W * y^2) keeps weights bounded.
        """
        # Flatten batch dimensions: [*, in] -> [N, in]
        x_flat = x.reshape(-1, self.in_features)
        y_flat = y.reshape(-1, self.out_features)

        # Phase coherence modulation
        if phase is not None:
            # Compute coherence between input and output phases
            # Phase is [..., phase_dim], may differ from in/out dims
            phase_flat = phase.reshape(-1, phase.shape[-1])
            # Simple coherence: mean cos(phase) as scalar per sample
            coherence = torch.cos(phase_flat).mean(dim=-1, keepdim=True)
            # Scale: [N, 1] — modulates each sample contribution
        else:
            coherence = torch.ones(x_flat.shape[0], 1,
                                   device=x.device, dtype=x.dtype)

        # Hebbian term: x^T @ y (outer product, averaged over batch)
        # Weight by coherence: stronger update when phases align
        x_weighted = x_flat * coherence  # [N, in]
        hebbian = (x_weighted.t() @ y_flat) / x_flat.shape[0]  # [in, out]
        hebbian = hebbian.t()  # [out, in]

        # Oja normalization: -alpha * W * mean(y^2)
        y_sq = (y_flat ** 2).mean(dim=0)  # [out]
        oja = self.oja_alpha * self.weight * y_sq.unsqueeze(1)  # [out, in]

        # Combined update
        dW = self.eta * (hebbian - oja)

        # Momentum
        self.weight_update_ema = (
            self.momentum * self.weight_update_ema +
            (1 - self.momentum) * dW
        )

        # Apply
        self.weight.add_(self.weight_update_ema)

    def set_learning(self, enabled: bool):
        """Enable/disable Hebbian learning."""
        self.learning = enabled

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"eta={self.eta}, oja_alpha={self.oja_alpha}, "
                f"hebbian=True")
