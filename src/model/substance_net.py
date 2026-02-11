"""
System Classification: src.model.substance_net
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - Reflexive Consciousness Theorem (Th 6.22)
    - The Emergence Parameter kappa ~ 1 (Onasenko, 2025)
    - Visual Cortex Hierarchy (Hubel & Wiesel, 1962, 1968)

SubstanceNet Model Template
===========================================================
Working template that assembles core modules into a complete
bio-inspired neural network with reflexive consciousness.

This is a CONSTRUCTOR, not a monolithic model. Researchers can:
    1. Use it directly for image classification
    2. Subclass and override components
    3. Copy and modify for custom architectures
    4. Import core modules directly for full control

Data Flow:
    Image [B, 1, H, W]
        -> BiologicalV1        (Gabor -> Simple -> Complex -> Hyper)
        -> OrientationSelectivity  (per-channel orientation expansion)
        -> QuantumWaveFunction     (psi = A * exp(i * phi))
        -> NonLocalInteraction     (attention-based V_ij)
        -> AbstractionLayer        (spatial pooling + projection)
        -> ReflexiveConsciousness  (psi_C = F[P_hat[psi_C]])
        -> Classifier              (logits)

Reference Results (SubstanceNet v3.1.1 / v3.2):
    MNIST:    93.74% accuracy (Stream mode, R = 0.382)
    CIFAR-10: 74.23% accuracy (5.64M params)

Key References:
    - Onasenko O. (2026) Monograph "2D-Substance", Chapters 1, 6
    - Hubel D.H., Wiesel T.N. (1962, 1968)
    - Yerkes R.M., Dodson J.D. (1908) — optimal arousal
    - Beggs J.M., Plenz D. (2003) — criticality

Changelog:
    2026-02-11 v0.1.0 — Clean redesign from v3.2 models_v2.py
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cortex import BiologicalV1
from src.wave import QuantumWaveFunction
from src.consciousness import ReflexiveConsciousness
from src.model.layers import (
    OrientationSelectivity,
    NonLocalInteraction,
    AbstractionLayer,
    PhaseCoherenceLoss,
    TopologicalLoss,
)


class SubstanceNet(nn.Module):
    """
    Bio-inspired neural network with reflexive consciousness.

    Assembles core modules (V1, wave function, consciousness) into
    a working image classifier. Serves as a template for research.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    v1_orientations : int
        Number of V1 orientation channels.
    v1_scales : int
        Number of V1 spatial frequency scales.
    v1_dim : int
        V1 output feature dimensionality.
    wave_channels : int
        Number of wave function output channels (must be even).
    orient_expand : int
        Orientation expansion factor.
    consciousness_dim : int
        Consciousness state dimensionality (must be even).
    abstract_dim : int
        Abstract representation dimensionality.
    num_iterations : int
        Reflexive consciousness iterations.
    num_attn_heads : int
        Number of attention heads in nonlocal interaction.

    Examples
    --------
    >>> model = SubstanceNet(num_classes=10)
    >>> x = torch.randn(4, 1, 28, 28)
    >>> output = model(x)
    >>> output['logits'].shape
    torch.Size([4, 10])
    """

    def __init__(
        self,
        num_classes: int = 10,
        v1_orientations: int = 8,
        v1_scales: int = 3,
        v1_dim: int = 64,
        wave_channels: int = 128,
        orient_expand: int = 8,
        consciousness_dim: int = 32,
        abstract_dim: int = 16,
        num_iterations: int = 3,
        num_attn_heads: int = 2,
    ):
        super().__init__()

        # Store config for reproducibility
        self.config = {k: v for k, v in locals().items()
                       if k != 'self' and k != '__class__'}

        # === Core Modules ===

        # V1: image -> sequence [B, 9, v1_dim]
        self.v1 = BiologicalV1(v1_orientations, v1_scales, v1_dim)

        # Orientation expansion: [B, 9, v1_dim] -> [B, 9, v1_dim * orient_expand]
        self.orientation = OrientationSelectivity(v1_dim, orient_expand)
        oriented_dim = v1_dim * orient_expand

        # Wave function: [B, 9, oriented_dim] -> psi, amp, phase
        self.wave = QuantumWaveFunction(oriented_dim, wave_channels)
        feature_dim = wave_channels  # amp + phase concatenated

        # Nonlocal interaction (V_ij analogue)
        self.nonlocal_interaction = NonLocalInteraction(
            feature_dim, num_attn_heads)

        # Abstraction: [B, 9, feature_dim] -> [B, abstract_dim]
        self.abstraction = AbstractionLayer(feature_dim, abstract_dim)

        # Reflexive consciousness: [B, abstract_dim] -> psi_C
        self.consciousness = ReflexiveConsciousness(
            abstract_dim, consciousness_dim, num_iterations)

        # === Classification Head ===
        # Uses features before consciousness (consciousness monitors,
        # does not directly produce classification features)
        seq_len = 9  # V1 output: 3x3 patches
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # === Loss Functions ===
        self.phase_loss = PhaseCoherenceLoss(weight=0.1)
        self.topo_loss = TopologicalLoss(weight=0.01)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Grayscale image [B, 1, H, W].

        Returns
        -------
        dict
            Keys:
            - logits: classification logits [B, num_classes]
            - abstract: abstract representation [B, abstract_dim]
            - psi_c: complex consciousness wave function [B, c_dim/2]
            - amplitude_c: consciousness amplitude [B, c_dim/2]
            - phase_c: consciousness phase [B, c_dim/2]
            - amplitude: wave function amplitude [B, 9, wave_ch/2]
            - phase: wave function phase [B, 9, wave_ch/2]
            - v1_activations: dict of V1 layer activations
        """
        batch_size = x.shape[0]

        # V1: biological visual processing
        v1_seq, v1_act = self.v1(x)

        # Orientation expansion
        oriented = self.orientation(v1_seq)

        # Wave function generation
        psi, amplitude, phase = self.wave(oriented)

        # Combine amplitude and phase as real-valued features
        features = torch.cat([amplitude, phase], dim=-1)

        # Nonlocal interaction (attention-based V_ij)
        features = self.nonlocal_interaction(features)

        # Abstract representation (for consciousness)
        abstract = self.abstraction(features)

        # Reflexive consciousness
        psi_c, amplitude_c, phase_c = self.consciousness(abstract)

        # Classification (from processed features)
        logits = self.classifier(features.reshape(batch_size, -1))

        return {
            'logits': logits,
            'abstract': abstract,
            'psi_c': psi_c,
            'amplitude_c': amplitude_c,
            'phase_c': phase_c,
            'amplitude': amplitude,
            'phase': phase,
            'v1_activations': v1_act,
        }

    def compute_loss(self, output: Dict[str, torch.Tensor],
                     target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Parameters
        ----------
        output : dict
            Output from forward().
        target : torch.Tensor
            Class labels [B].

        Returns
        -------
        dict
            Keys: total, classification, consciousness, zero_loss,
            phase_coherence, topological.
        """
        # Classification loss
        cls_loss = F.cross_entropy(output['logits'], target)

        # Consciousness loss (reflexivity + coherence + stability)
        cons_loss = self.consciousness.consciousness_loss(
            output['amplitude_c'], output['phase_c'])

        # Wave function zero-minimization loss
        zero_loss = self.wave.zero_loss(
            output['amplitude'], output['phase'])

        # Phase coherence loss
        phase_loss = self.phase_loss(output['phase'])

        # Topological loss (reshape phase to 2D if possible)
        phase_2d = output['phase'].mean(dim=-1)  # [B, seq_len]
        sqrt_seq = int(math.sqrt(phase_2d.shape[1]))
        if sqrt_seq * sqrt_seq == phase_2d.shape[1]:
            phase_2d = phase_2d.view(-1, sqrt_seq, sqrt_seq)
            topo_loss = self.topo_loss(phase_2d)
        else:
            topo_loss = torch.tensor(0.0, device=target.device)

        # Total loss
        total = (cls_loss +
                 0.1 * cons_loss +
                 0.01 * zero_loss +
                 phase_loss +
                 topo_loss)

        return {
            'total': total,
            'classification': cls_loss,
            'consciousness': cons_loss,
            'zero_loss': zero_loss,
            'phase_coherence': phase_loss,
            'topological': topo_loss,
        }

    def get_consciousness_metrics(self,
                                  output: Dict[str, torch.Tensor]) -> dict:
        """
        Extract consciousness metrics from forward output.

        Parameters
        ----------
        output : dict
            Output from forward().

        Returns
        -------
        dict
            Consciousness metrics for monitoring.
        """
        return self.consciousness.get_metrics(
            output['amplitude_c'], output['phase_c'])

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per module."""
        counts = {}
        for name, module in [
            ('v1', self.v1),
            ('orientation', self.orientation),
            ('wave', self.wave),
            ('nonlocal', self.nonlocal_interaction),
            ('abstraction', self.abstraction),
            ('consciousness', self.consciousness),
            ('classifier', self.classifier),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
