"""
System Classification: src.model.substance_net
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

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

from src.cortex import BiologicalV1, MosaicField18, DynamicFormV3, ObjectFeaturesV4
from src.hippocampus import Hippocampus
from src.constants import OPTIMAL_REFLEXIVITY_MIN, OPTIMAL_REFLEXIVITY_MAX
from src.wave import QuantumWaveFunction, WaveFunctionOnT, NonlocalWaveInteraction
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
        in_channels: int = 1,
        v1_orientations: int = 8,
        v1_scales: int = 3,
        v1_dim: int = 64,
        wave_channels: int = 128,
        orient_expand: int = 8,
        consciousness_dim: int = 32,
        abstract_dim: int = 3,
        num_iterations: int = 3,
        num_attn_heads: int = 2,
        use_wave_on_t: bool = False,
    ):
        super().__init__()

        # Store config for reproducibility
        self.config = {k: v for k, v in locals().items()
                       if k != 'self' and k != '__class__'}

        # === Core Modules ===

        # V1: image -> sequence [B, 9, v1_dim]
        self.v1 = BiologicalV1(v1_orientations, v1_scales, v1_dim, in_channels)

        # Orientation expansion: [B, 9, v1_dim] -> [B, 9, v1_dim * orient_expand]
        self.orientation = OrientationSelectivity(v1_dim, orient_expand)
        oriented_dim = v1_dim * orient_expand

        # Wave function: [B, 9, oriented_dim] -> psi, amp, phase
        self.wave = QuantumWaveFunction(oriented_dim, wave_channels)
        feature_dim = wave_channels  # amp + phase concatenated

        # Nonlocal interaction (V_ij analogue)
        self.nonlocal_interaction = NonLocalInteraction(
            feature_dim, num_attn_heads)

        # V2: three-stripe contour/texture processing (Hubel & Wiesel V2 cortex)
        # CRITICAL: prevents abstract collapse and consciousness saturation
        self.v2 = MosaicField18(feature_dim, feature_dim)

        # Wave function on configuration space T (Onasenko/Dubovikov/Tsien)
        self.use_wave_on_t = use_wave_on_t
        if use_wave_on_t:
            v2_stream_dim = feature_dim // 3  # V2 has 3 streams
            self.wave_on_t = WaveFunctionOnT(
                n_streams=3, stream_dim=v2_stream_dim)
            self.wave_on_t_interaction = NonlocalWaveInteraction(
                self.wave_on_t, output_dim=feature_dim)

        # V3: cross-stream gating for dynamic form (Felleman & Van Essen)
        self.v3 = DynamicFormV3(feature_dim)

        # V4: object-level feature extraction (Zeki)
        self.v4 = ObjectFeaturesV4(feature_dim)

        # Coherence compression: V4 output -> reduced dim (matches v3.1.1)
        self.coherent_dim = feature_dim // 2  # 128 -> 64
        self.coherence_fc = nn.Linear(feature_dim, self.coherent_dim)

        # Stability pathway for classifier (matches v3.1.1)
        self.stability_fc = nn.Linear(self.coherent_dim, self.coherent_dim)

        # Abstraction: [B, 9, coherent_dim] -> [B, abstract_dim]
        self.abstraction = AbstractionLayer(self.coherent_dim, abstract_dim)

        # Reflexive consciousness: [B, abstract_dim] -> psi_C
        self.consciousness = ReflexiveConsciousness(
            abstract_dim, consciousness_dim, num_iterations)

        # === Classification Head ===
        # Uses features before consciousness (consciousness monitors,
        # does not directly produce classification features)
        seq_len = 9  # V1 output: 3x3 patches
        self.classifier = nn.Sequential(
            nn.Linear(self.coherent_dim * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # === Abstract Classification Head ===
        # Forces abstract representations to be class-discriminative.
        # Without this, abstraction layer collapses (all samples -> same output)
        # and consciousness trivially converges to R -> 1.0.
        # In v3.1.1 this was abstract_loss = CE(abstract_pred, abstract_target).
        self.abstract_classifier = nn.Linear(abstract_dim, num_classes)

        # === Cognitive Input Path ===
        # Bypasses V1 for non-image cognitive tasks (logic, memory, etc.)
        # Maps flat cognitive data to same shape as V1 output: [B, 9, feature_dim]
        self.cognitive_input = nn.Sequential(
            nn.Linear(64, 64 * 9),  # max input=64, output matches V1 (64 per patch)
            nn.ReLU(),
        )
        self._v1_dim = 64  # V1 output dim
        self._seq_len = 9  # V1 output sequence length

        # === Hippocampus (episodic memory) ===
        # Receives abstract representations modulated by consciousness.
        # Not part of forward pass — controlled by training loop via
        # store_episode(), recall(), and consolidate_memory() methods.
        self.hippocampus = Hippocampus(
            input_dim=abstract_dim,
            hidden_dim=256,
            consciousness_dim=consciousness_dim,
            buffer_size=1000,
            num_prototypes=num_classes,
        )

        # === Loss Functions ===
        self.phase_loss = PhaseCoherenceLoss(weight=0.1)
        self.topo_loss = TopologicalLoss(weight=0.01)

    def forward(self, x: torch.Tensor,
                mode: str = "auto") -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Image [B, 1, H, W] or flat cognitive data [B, *].
        mode : str
            "image" — force V1 path.
            "cognitive" — force cognitive input path (bypass V1).
            "auto" — detect from tensor shape (4D = image, else cognitive).
        """
        batch_size = x.shape[0]

        # Determine input mode
        if mode == "auto":
            if x.dim() == 5:
                mode = "video"
            elif x.dim() == 4:
                mode = "image"
            else:
                mode = "cognitive"

        if mode == "video":
            # Video path: process each frame through V1->V2, collect for V3 temporal
            # Input: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            v2_sequence = []
            amp_sequence = []
            phase_sequence = []
            for t in range(T):
                frame = x[:, t]  # [B, C, H, W]
                v1_seq_t, _ = self.v1(frame)
                oriented_t = self.orientation(v1_seq_t)
                _, amp_t, phase_t = self.wave(oriented_t)
                feat_t = torch.cat([amp_t, phase_t], dim=-1)
                feat_t = self.nonlocal_interaction(feat_t)
                v2_t = self.v2(feat_t)
                v2_sequence.append(v2_t)
                amp_sequence.append(amp_t)
                phase_sequence.append(phase_t)
            # Stack: [B, T, seq_len, dim]
            v2_temporal = torch.stack(v2_sequence, dim=1)
            amp_temporal = torch.stack(amp_sequence, dim=1)
            phase_temporal = torch.stack(phase_sequence, dim=1)
            # Wave function on T (if enabled) — per frame
            if self.use_wave_on_t:
                v2_processed = []
                for t in range(T):
                    v2_t = v2_temporal[:, t]  # [B, seq, dim]
                    ss = v2_t.shape[-1] // 3
                    streams_t = [
                        v2_t[..., :ss],
                        v2_t[..., ss:2*ss],
                        v2_t[..., 2*ss:2*ss+ss],
                    ]
                    v2_processed.append(
                        self.wave_on_t_interaction(streams_t))
                v2_temporal = torch.stack(v2_processed, dim=1)

            # V3 temporal integration with phase interference
            v3_features = self.v3(v2_temporal, amp_temporal, phase_temporal)
            # V4 onward uses the temporally integrated features
            v4_features = self.v4(v3_features)
            coherent = F.relu(self.coherence_fc(v4_features))
            abstract = self.abstraction(coherent)
            psi_c, amplitude_c, phase_c = self.consciousness(abstract)
            stable = F.relu(self.stability_fc(coherent))
            logits = self.classifier(stable.reshape(B, -1))
            # Use last frame for amplitude/phase output
            _, amplitude, phase = self.wave(self.orientation(self.v1(x[:, -1])[0]))
            psi = amplitude * torch.exp(1j * phase)
            return {
                'logits': logits,
                'abstract': abstract,
                'psi_c': psi_c,
                'amplitude_c': amplitude_c,
                'phase_c': phase_c,
                'amplitude': amplitude,
                'phase': phase,
                'psi': psi,
                'v2_temporal': v2_temporal,
            }
        elif mode == "image":
            # V1: biological visual processing
            v1_seq, v1_act = self.v1(x)
        else:
            # Cognitive path: flat tensor -> [B, 9, feature_dim]
            x_flat = x.reshape(batch_size, -1)
            if x_flat.shape[1] < 64:
                pad = torch.zeros(batch_size, 64 - x_flat.shape[1],
                                  device=x.device, dtype=x.dtype)
                x_flat = torch.cat([x_flat, pad], dim=1)
            elif x_flat.shape[1] > 64:
                x_flat = x_flat[:, :64]
            v1_seq = self.cognitive_input(x_flat).reshape(
                batch_size, self._seq_len, self._v1_dim)
            v1_act = {}

        # Orientation expansion
        oriented = self.orientation(v1_seq)

        # Wave function generation
        psi, amplitude, phase = self.wave(oriented)

        # Combine amplitude and phase as real-valued features
        features = torch.cat([amplitude, phase], dim=-1)

        # Nonlocal interaction (attention-based V_ij)
        features = self.nonlocal_interaction(features)

        # V2: three-stripe processing (Hubel V2 cortex)
        v2_features = self.v2(features)

        # Wave function on T (if enabled)
        if self.use_wave_on_t:
            ss = v2_features.shape[-1] // 3
            v2_streams = [
                v2_features[..., :ss],
                v2_features[..., ss:2*ss],
                v2_features[..., 2*ss:2*ss+ss],
            ]
            v2_features = self.wave_on_t_interaction(v2_streams)

        # V3: dynamic form (cross-stream gating)
        v3_features = self.v3(v2_features)

        # V4: object features (multi-scale attention)
        v4_features = self.v4(v3_features)

        # Coherence compression (V4 -> reduced representation)
        coherent = F.relu(self.coherence_fc(v4_features))

        # Abstract representation (for consciousness)
        abstract = self.abstraction(coherent)

        # Reflexive consciousness
        psi_c, amplitude_c, phase_c = self.consciousness(abstract)

        # Stability pathway (for classifier)
        stable = F.relu(self.stability_fc(coherent))

        # Classification (from stability-processed features)
        logits = self.classifier(stable.reshape(batch_size, -1))

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
            Keys: total, classification, abstract, consciousness,
            zero_loss, phase_coherence, topological.
        """
        # Classification loss
        cls_loss = F.cross_entropy(output['logits'], target)

        # Abstract classification loss — CRITICAL for preventing
        # abstract collapse. Without this, abstraction layer produces
        # identical output for all samples, consciousness trivially
        # converges to R -> 1.0 (SubstanceNet v4 diagnostic finding).
        abstract_logits = self.abstract_classifier(output['abstract'])
        abstract_loss = F.cross_entropy(abstract_logits, target)

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

        # R-regulation: penalize reflexivity outside optimal range [0.35, 0.47]
        # This prevents consciousness from saturating to trivial fixed point
        with torch.no_grad():
            psi_c_cat = torch.cat([output['amplitude_c'], output['phase_c']], dim=-1)
            psi_c_proj = self.consciousness.project(psi_c_cat)
            ref_error = F.mse_loss(psi_c_cat, psi_c_proj)
            current_r = 1.0 / (1.0 + ref_error.item())

        # Soft penalty: quadratic cost for R outside optimal band
        r_target = (OPTIMAL_REFLEXIVITY_MIN + OPTIMAL_REFLEXIVITY_MAX) / 2  # 0.41
        r_penalty = torch.tensor((current_r - r_target) ** 2,
                                 device=target.device, dtype=cls_loss.dtype)

        # Total loss (v3.1.1 compatible weighting + R-regulation)
        total = (cls_loss +
                 abstract_loss +
                 0.1 * cons_loss +
                 0.01 * zero_loss +
                 phase_loss +
                 topo_loss +
                 0.5 * r_penalty)

        return {
            'total': total,
            'classification': cls_loss,
            'abstract': abstract_loss,
            'consciousness': cons_loss,
            'zero_loss': zero_loss,
            'phase_coherence': phase_loss,
            'topological': topo_loss,
            'r_penalty': r_penalty,
            'current_r': current_r,
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

    def store_episode(self, output: Dict[str, torch.Tensor],
                       task_type: str = 'unknown',
                       metrics: Optional[dict] = None) -> dict:
        """
        Store current abstract state as episodic memory.

        Call from training loop after forward() and compute_loss().
        Consciousness state modulates importance scoring.

        Parameters
        ----------
        output : dict
            Output from forward().
        task_type : str
            Task identifier.
        metrics : dict, optional
            Performance metrics (accuracy, loss, etc).

        Returns
        -------
        dict
            Episode info: id, importance, buffer_size.
        """
        abstract = output['abstract'].detach()
        # Use consciousness amplitude as importance signal
        consciousness_state = output['amplitude_c'].detach()
        return self.hippocampus.encode_and_store(
            abstract, consciousness_state, task_type, metrics)

    def recall(self, output: Dict[str, torch.Tensor],
               top_k: int = 5) -> list:
        """
        Retrieve similar episodes from memory.

        Parameters
        ----------
        output : dict
            Output from forward().
        top_k : int
            Number of episodes to retrieve.

        Returns
        -------
        list
            Similar episodes with context and importance.
        """
        abstract = output['abstract'].detach()
        consciousness_state = output['amplitude_c'].detach()
        return self.hippocampus.retrieve_similar(
            abstract, consciousness_state, top_k)

    def consolidate_memory(self):
        """Consolidate short-term episodic memory into long-term prototypes."""
        self.hippocampus.consolidate()

    def get_memory_state(self) -> dict:
        """Return current hippocampus state summary."""
        return self.hippocampus.get_state()

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per module."""
        counts = {}
        for name, module in [
            ('v1', self.v1),
            ('orientation', self.orientation),
            ('wave', self.wave),
            ('v2', self.v2),
            ('v3', self.v3),
            ('v4', self.v4),
            ('nonlocal', self.nonlocal_interaction),
            ('coherence_fc', self.coherence_fc),
            ('stability_fc', self.stability_fc),
            ('abstraction', self.abstraction),
            ('consciousness', self.consciousness),
            ('classifier', self.classifier),
            ('abstract_classifier', self.abstract_classifier),
            ('cognitive_input', self.cognitive_input),
            ('hippocampus', self.hippocampus),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
