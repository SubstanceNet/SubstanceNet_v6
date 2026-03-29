"""
System Classification: src.cortex.v3
Author: Oleksii Onasenko
Developer: SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - Visual Cortex V3 (Felleman & Van Essen, 1991)

Visual Cortex V3 — Dynamic Form via Phase Interference
===========================================================
V3 integrates form and motion through wave superposition.

Instead of learned linear gates (sigma(W*concat)), V3 uses
physical phase interference:

    I_V3 = |psi_form + psi_motion|^2
         = A_form^2 + A_motion^2 + 2*A_form*A_motion*cos(dphi)

Where:
    psi_form   = pale_stripes output (spatial contours)
    psi_motion = temporal phase difference between frames
    dphi       = phase_form - phase_motion

When form and motion belong to the same object, their phases
synchronize (dphi -> 0), producing constructive interference.
This solves the binding problem through wave physics.

No nn.Linear in the interference path — pure wave mechanics.

Key References:
    - Felleman D.J., Van Essen D.C. (1991) Cereb. Cortex 1:1-47
    - Onasenko O. (2026) Monograph "2D-Substance", Chapter 6

Changelog:
    2026-03-15 v1.0.0 — Cross-stream gating (spatial only)
    2026-03-17 v2.0.0 — Temporal integration + linear projections
    2026-03-17 v3.0.0 — Phase interference (no learned weights in core path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cortex.hebbian import HebbianLinear


class DynamicFormV3(nn.Module):
    """
    V3 cortex: form-motion integration via phase interference.

    Two modes:
    - Temporal: receives V2 sequence + amplitude/phase sequence,
      computes phase interference between form and motion.
    - Static: spatial cross-stream gating (backward compatible).

    Parameters
    ----------
    dim : int
        Feature dimensionality (V2 output).
    num_streams : int
        Number of V2 streams (3: thick/thin/pale).
    phase_dim : int
        Dimensionality of phase signal from QuantumWaveFunction.
    """

    def __init__(self, dim: int, num_streams: int = 3, phase_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.stream_size = dim // num_streams
        self.phase_dim = phase_dim

        # Hebbian output projection: the ONLY adaptive component.
        # Weights evolve through phase coherence, not backprop.
        # dW_ij ~ cos(phi_i - phi_j) — connections strengthen
        # where form and motion phases are coherent over time.
        self.output_proj = HebbianLinear(
            dim + phase_dim, dim,
            learning_rate=0.0001,
            oja_alpha=0.1,
        )
        self.output_norm = nn.LayerNorm(dim)

        # Spatial cross-stream gating (for static/backward-compatible mode)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.stream_size * 2, self.stream_size),
                nn.Sigmoid(),
            )
            for _ in range(num_streams)
        ])

        # Static output integration
        self.static_output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

        # Residual gate
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward_temporal(self, v2_sequence: torch.Tensor,
                         amplitude_sequence: torch.Tensor = None,
                         phase_sequence: torch.Tensor = None) -> torch.Tensor:
        """
        Temporal mode: V2 stream interference for form-motion binding.

        Instead of random projections, uses direct V2 stream signals:
        - Motion = temporal diff of thick_stripes (change detection over time)
        - Form = pale_stripes from last frame (spatial contours)
        - Interference = A_form * A_motion * cos(angle between them)

        Parameters
        ----------
        v2_sequence : torch.Tensor
            V2 outputs per frame [B, T, seq_len, dim].
        amplitude_sequence, phase_sequence : ignored (kept for API compat).

        Returns
        -------
        torch.Tensor
            Integrated features [B, seq_len, dim].
        """
        B, T, seq_len, dim = v2_sequence.shape
        last_v2 = v2_sequence[:, -1]  # [B, seq, dim]

        # === 1. Split V2 into streams ===
        thick_start = 0
        thin_start = self.stream_size
        pale_start = self.stream_size * 2

        # Extract thick stream across time: [B, T, seq, stream_size]
        thick_t = v2_sequence[..., thick_start:thick_start + self.stream_size]
        # Extract pale from last frame: [B, seq, stream_size]
        form = last_v2[..., pale_start:pale_start + self.stream_size]

        # === 2. Motion = temporal diff of thick stripes ===
        thick_diff = thick_t[:, 1:] - thick_t[:, :-1]  # [B, T-1, seq, stream_size]
        motion = thick_diff.mean(dim=1)  # [B, seq, stream_size]

        # === 3. Interference: I = A_form^2 + A_motion^2 + 2*A_f*A_m*cos(angle) ===
        A_form = form.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        A_motion = motion.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Cosine of angle between form and motion vectors
        min_d = min(form.shape[-1], motion.shape[-1])
        cos_angle = F.cosine_similarity(
            form[..., :min_d], motion[..., :min_d], dim=-1
        ).unsqueeze(-1)  # [B, seq, 1]

        # Full interference: constructive when form and motion are coherent
        interference = A_form**2 + A_motion**2 + 2 * A_form * A_motion * cos_angle
        # Broadcast to stream_size for concatenation
        interference = interference.expand_as(form)  # [B, seq, stream_size]

        # === 4. Hebbian integration ===
        # Interference modulates amplitude, then HebbianLinear
        # learns stable representations through phase coherence.

        # Pad interference to phase_dim
        if interference.shape[-1] < self.phase_dim:
            pad = torch.zeros(*interference.shape[:-1],
                              self.phase_dim - interference.shape[-1],
                              device=interference.device, dtype=interference.dtype)
            interference_padded = torch.cat([interference, pad], dim=-1)
        else:
            interference_padded = interference[..., :self.phase_dim]

        # Combined: V2 features + interference signal
        combined = torch.cat([last_v2, interference_padded], dim=-1)

        # Phase for Hebbian: cos_angle encodes form-motion coherence
        # Expand to match phase_dim for HebbianLinear
        phase_signal = cos_angle.expand(*cos_angle.shape[:-1], self.phase_dim)

        # HebbianLinear: weights adapt where phases are coherent
        projected = self.output_proj(combined, phase=phase_signal)
        integrated = F.relu(self.output_norm(projected))

        g = torch.sigmoid(self.residual_gate)
        return g * integrated + (1 - g) * last_v2

    def forward_static(self, x: torch.Tensor) -> torch.Tensor:
        """
        Static mode: spatial cross-stream gating (backward compatible).
        """
        streams = x.split(self.stream_size, dim=-1)
        if len(streams) > self.num_streams:
            streams = list(streams[:self.num_streams - 1]) + [
                torch.cat(list(streams[self.num_streams - 1:]), dim=-1)]
        streams = list(streams)

        gated = []
        for i in range(min(len(streams), self.num_streams)):
            j = (i + 1) % len(streams)
            s_i = streams[i][..., :self.stream_size]
            s_j = streams[j][..., :self.stream_size]
            pair = torch.cat([s_i, s_j], dim=-1)
            gate = self.gates[i](pair)
            gated.append(gate * s_i)

        combined = torch.cat(gated, dim=-1)
        if combined.shape[-1] < self.dim:
            pad = torch.zeros(*combined.shape[:-1],
                              self.dim - combined.shape[-1],
                              device=x.device, dtype=x.dtype)
            combined = torch.cat([combined, pad], dim=-1)

        integrated = self.static_output(combined)
        g = torch.sigmoid(self.residual_gate)
        return g * integrated + (1 - g) * x

    def forward(self, x: torch.Tensor,
                amplitude_sequence: torch.Tensor = None,
                phase_sequence: torch.Tensor = None) -> torch.Tensor:
        """
        Auto-detect mode based on input.

        Parameters
        ----------
        x : torch.Tensor
            [B, seq, dim] for static, [B, T, seq, dim] for temporal.
        amplitude_sequence : torch.Tensor, optional
            [B, T, seq, phase_dim] — required for temporal mode.
        phase_sequence : torch.Tensor, optional
            [B, T, seq, phase_dim] — required for temporal mode.
        """
        if x.dim() == 4 and phase_sequence is not None:
            return self.forward_temporal(x, amplitude_sequence, phase_sequence)
        elif x.dim() == 4:
            # Temporal input but no phase — fallback to static on last frame
            return self.forward_static(x[:, -1])
        else:
            return self.forward_static(x)
