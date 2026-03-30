"""
System Classification: src.wave.wave_on_t
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Wave Function on Configuration Space T
=======================================
Implements ψ(x, τ) on the configuration space T = 2^i − 1 (Tsien cliques).

Three independent frameworks converge to one structure:
- Onasenko (2025): ψ on manifold Σ with nonlocal potential V_ij
- Dubovikov (2013, 2016): Tensor of ostensive definitions T = 2^n
- Tsien (2015-2016): Neural cliques N = 2^i − 1 (FCM)

Identification:
    Σ ≡ T ≡ FCM
    d_Σ(p,q) ≡ δ(τ,τ') (weighted Hamming)
    V_ij = ∬ K(d)·ψ_i*·ψ_j·e^(i·Δφ) dV ≡ Σ_q K(δ)·A_p·A_q·cos(φ_q−φ_p)

References:
    - Onasenko O. (2025) Emergence Parameter κ ≈ 1. DOI: 10.5281/zenodo.17844282
    - Dubovikov M.M. (2013) Mathematical Formalisation... LAP Lambert.
    - Dubovikov M.M. (2026) Tensors of Ostensive Definitions. Preprint v0.4.
    - Tsien J.Z. (2016) Principles of Intelligence. Front. Syst. Neurosci. 9:186.
    - Xie K. et al. (2016) Power-of-Two Logic. Front. Syst. Neurosci. 10:95.
    - Hubel D.H., Wiesel T.N. (1962) J. Physiol. 160:106-154.

Changelog:
    2026-03-30 v0.1.0 — Initial implementation for V2 level (i=3, N=7)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import Optional, Tuple, Dict


class WaveFunctionOnT(nn.Module):
    """
    Wave function ψ(x, τ) defined on configuration space T = 2^i − 1.

    For i input streams, builds all non-empty binary configurations
    (Tsien cliques). Each clique τ represents a combination of active
    streams. ψ(τ) = A(τ)·e^(iφ(τ)) describes activation and phase
    synchronization of that clique.

    Parameters
    ----------
    n_streams : int
        Number of input streams (i in Tsien's formula).
    stream_dim : int
        Dimensionality of each input stream.
    max_depth : int, optional
        Maximum combination depth. None = full 2^i − 1.
    """

    def __init__(self, n_streams: int, stream_dim: int,
                 max_depth: Optional[int] = None):
        super().__init__()
        self.n_streams = n_streams
        self.stream_dim = stream_dim

        # Build configurations (Tsien cliques)
        if max_depth is None:
            max_depth = n_streams
        self.max_depth = max_depth

        configs = []
        config_names = []
        for k in range(1, max_depth + 1):
            for combo in combinations(range(n_streams), k):
                config = [0] * n_streams
                for idx in combo:
                    config[idx] = 1
                configs.append(config)
                config_names.append('+'.join(str(c) for c in combo))

        self.N = len(configs)  # Number of cliques
        self.config_names = config_names

        # [N, n_streams] — binary mask for each configuration
        self.register_buffer('configs',
                             torch.tensor(configs, dtype=torch.float32))

        # Hamming distance matrix [N, N]
        self.register_buffer('hamming_dist',
                             self._compute_hamming_matrix())

        # Positional phase from configuration structure
        # Analogous to φ = n·arctan(η/ξ) in 2d_substance_v2
        self.register_buffer('positional_phase',
                             self._compute_positional_phase())

        # Amplitude projection: stream features → activation per clique
        # Each clique combines its active streams
        self.amplitude_combine = nn.ModuleList([
            nn.Linear(int(self.configs[i].sum().item()) * stream_dim, stream_dim)
            for i in range(self.N)
        ])
        self.amplitude_activation = nn.Softplus()

        # Phase projection: learned contextual phase per clique
        self.phase_combine = nn.ModuleList([
            nn.Linear(int(self.configs[i].sum().item()) * stream_dim, stream_dim)
            for i in range(self.N)
        ])

        # Interaction kernel length scale (learnable)
        self.ell = nn.Parameter(torch.tensor(1.0))

        # Gradient energy weight
        self.gamma_grad = nn.Parameter(torch.tensor(1e-3))

    def _compute_hamming_matrix(self) -> torch.Tensor:
        """Weighted Hamming distance δ(τ,τ') between all configuration pairs."""
        configs = self.configs  # [N, n_streams]
        # δ(τ,τ') = (1/n) · Σ |τ_i − τ'_i|
        diff = (configs.unsqueeze(0) - configs.unsqueeze(1)).abs()
        return diff.sum(dim=-1) / self.n_streams  # [N, N]

    def _compute_positional_phase(self) -> torch.Tensor:
        """Phase encoding from configuration structure.

        Each configuration maps to a unique angle based on which
        streams are active. Analogous to arctan(η/ξ) on manifold.
        """
        configs = self.configs  # [N, n_streams]
        # Weight each stream by golden-ratio-based angles for maximal separation
        angles = torch.tensor([
            2 * math.pi * i / (1 + math.sqrt(5)) * 2
            for i in range(self.n_streams)
        ])
        # Phase = sum of angles for active streams
        phase = (configs * angles.unsqueeze(0)).sum(dim=-1)  # [N]
        return phase

    def compute_kernel(self) -> torch.Tensor:
        """Gaussian kernel K(δ) = exp(−δ²/ℓ²) on configuration space.

        This is the discrete analogue of Onasenko's nonlocal kernel
        K(p,q) = (g·C_norm)/(κ·ℓ⁴) · exp(−d_Σ²(p,q)/ℓ²)
        with Dubovikov's Hamming metric as distance.
        """
        return torch.exp(
            -self.hamming_dist ** 2 / (self.ell ** 2 + 1e-8))  # [N, N]

    def forward(self, streams: list) -> Tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor]:
        """
        Compute ψ(τ) for each configuration from input streams.

        Parameters
        ----------
        streams : list of torch.Tensor
            List of i tensors, each [B, seq_len, stream_dim].

        Returns
        -------
        psi : torch.Tensor (complex)
            [B, seq_len, N, stream_dim] — wave function on T.
        amplitude : torch.Tensor
            [B, seq_len, N, stream_dim] — activation per clique.
        phase : torch.Tensor
            [B, seq_len, N, stream_dim] — phase per clique.
        """
        B, S, D = streams[0].shape

        amplitudes = []
        phases = []

        for i in range(self.N):
            # Select active streams for this configuration
            active_mask = self.configs[i]  # [n_streams]
            active_streams = [
                streams[j] for j in range(self.n_streams)
                if active_mask[j] > 0.5
            ]

            # Concatenate active streams
            combined = torch.cat(active_streams, dim=-1)  # [B, S, k*D]

            # Amplitude: how strongly this clique is activated
            a = self.amplitude_activation(
                self.amplitude_combine[i](combined))  # [B, S, D]
            amplitudes.append(a)

            # Phase: positional + contextual
            phi_learned = self.phase_combine[i](combined)  # [B, S, D]
            phi = self.positional_phase[i] + phi_learned
            phases.append(phi)

        # Stack: [B, S, N, D]
        amplitude = torch.stack(amplitudes, dim=2)
        phase = torch.stack(phases, dim=2)

        # Complex wave function
        psi = amplitude * torch.exp(1j * phase)

        return psi, amplitude, phase

    def compute_nonlocal_potential(self, amplitude: torch.Tensor,
                                   phase: torch.Tensor) -> torch.Tensor:
        """
        Nonlocal potential V(p) = Σ_q K(δ(p,q)) · A(p)·A(q) · cos(φ(q)−φ(p))

        This is the discrete version of Onasenko's V_ij on manifold Σ,
        using Dubovikov's Hamming metric for the kernel distance.

        Constructive interference: coherent phases amplify.
        Destructive interference: incoherent phases cancel.

        Parameters
        ----------
        amplitude : [B, seq_len, N, dim]
        phase : [B, seq_len, N, dim]

        Returns
        -------
        V : [B, seq_len, N, dim] — potential at each clique
        """
        K = self.compute_kernel()  # [N, N]

        # Phase difference: φ(q) − φ(p) for all pairs
        phase_diff = phase.unsqueeze(3) - phase.unsqueeze(2)  # [B, S, N, N, D]
        interference = torch.cos(phase_diff)  # constructive/destructive

        # Amplitude product: A(p)·A(q)
        A_pq = amplitude.unsqueeze(3) * amplitude.unsqueeze(2)  # [B, S, N, N, D]

        # Kernel-weighted interaction
        K_expanded = K.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, N, 1]
        V = (K_expanded * A_pq * interference).sum(dim=3)  # sum over q → [B, S, N, D]

        return V

    def compute_gradient_energy(self, amplitude: torch.Tensor,
                                phase: torch.Tensor) -> torch.Tensor:
        """
        Gradient energy |∇ψ|² on configuration space.

        For neighboring configurations (Hamming distance = 1/n):
        |∇ψ|² ≈ Σ_neighbors |ψ(τ) − ψ(τ')|² · n²

        Parameters
        ----------
        amplitude, phase : [B, seq_len, N, dim]

        Returns
        -------
        energy : scalar
        """
        psi = amplitude * torch.exp(1j * phase)

        # Find neighbors (Hamming distance = 1/n_streams)
        threshold = 1.0 / self.n_streams + 0.01
        neighbors = (self.hamming_dist > 0) & (self.hamming_dist < threshold)

        energy = torch.tensor(0.0, device=amplitude.device)
        count = 0

        for i in range(self.N):
            nbr_idx = neighbors[i].nonzero(as_tuple=True)[0]
            if len(nbr_idx) > 0:
                diff = psi[:, :, i:i + 1, :] - psi[:, :, nbr_idx, :]
                energy = energy + (diff.abs() ** 2).mean()
                count += 1

        if count > 0:
            energy = energy / count

        return self.gamma_grad * energy

    def get_diagnostics(self, amplitude: torch.Tensor,
                        phase: torch.Tensor) -> Dict[str, float]:
        """Diagnostic metrics for monitoring wave function state."""
        K = self.compute_kernel()

        return {
            'n_cliques': self.N,
            'n_streams': self.n_streams,
            'mean_amplitude': amplitude.mean().item(),
            'std_amplitude': amplitude.std().item(),
            'phase_coherence': torch.cos(
                phase.unsqueeze(3) - phase.unsqueeze(2)
            ).mean().item(),
            'kernel_mean': K.mean().item(),
            'kernel_min': K.min().item(),
            'ell': self.ell.item(),
        }


class NonlocalWaveInteraction(nn.Module):
    """
    Nonlocal interaction via wave function on configuration space T.

    Replaces MultiheadAttention with physics-based potential:
    V(p) = Σ_q K(δ(p,q)) · A(p)·A(q) · cos(Δφ)

    Output is projected back to feature space with gated residual.

    Parameters
    ----------
    wave_function : WaveFunctionOnT
    output_dim : int
        Output feature dimensionality.
    """

    def __init__(self, wave_function: WaveFunctionOnT, output_dim: int):
        super().__init__()
        self.wave_fn = wave_function
        N = wave_function.N
        D = wave_function.stream_dim

        # Project from clique space back to feature space
        self.output_proj = nn.Linear(N * D, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, streams: list,
                return_wave: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        streams : list of [B, seq_len, stream_dim]
        return_wave : bool
            If True, also return (psi, amplitude, phase).

        Returns
        -------
        output : [B, seq_len, output_dim]
        """
        B, S, D = streams[0].shape

        # Compute wave function on T
        psi, A, phi = self.wave_fn(streams)  # [B, S, N, D]

        # Compute nonlocal potential
        V = self.wave_fn.compute_nonlocal_potential(A, phi)  # [B, S, N, D]

        # Modulate amplitude by potential
        modulated = A * (1 + V)  # V acts as modulation

        # Flatten clique dimension and project
        flat = modulated.reshape(B, S, -1)  # [B, S, N*D]
        output = self.output_proj(flat)  # [B, S, output_dim]
        output = F.relu(self.output_norm(output))

        # Gated residual with input (projected to same dim)
        input_cat = torch.cat(streams, dim=-1)  # [B, S, n*D]
        if not hasattr(self, 'input_proj') or self.input_proj is None:
            self.input_proj = nn.Linear(
                input_cat.shape[-1], output.shape[-1]
            ).to(output.device)
        residual = self.input_proj(input_cat)
        g = torch.sigmoid(self.gate)
        result = g * output + (1 - g) * residual

        if return_wave:
            return result, (psi, A, phi)
        return result


if __name__ == '__main__':
    """Quick test."""
    print('=== WaveFunctionOnT test ===')

    # V2 level: i=3 streams
    n_streams = 3
    stream_dim = 42
    wave = WaveFunctionOnT(n_streams=n_streams, stream_dim=stream_dim)

    print(f'Streams: {n_streams}')
    print(f'Cliques (N = 2^{n_streams} - 1): {wave.N}')
    print(f'Configurations:')
    for i, (cfg, name) in enumerate(zip(wave.configs, wave.config_names)):
        print(f'  τ_{i}: [{cfg.tolist()}] = stream {name}')

    print(f'\nHamming distance matrix:')
    print(wave.hamming_dist.numpy().round(3))

    print(f'\nPositional phases:')
    print(wave.positional_phase.numpy().round(3))

    print(f'\nKernel K(δ):')
    print(wave.compute_kernel().numpy().round(3))

    # Forward pass
    B, S = 2, 9
    streams = [torch.randn(B, S, stream_dim) for _ in range(n_streams)]
    psi, A, phi = wave(streams)
    print(f'\nForward: psi={psi.shape}, A={A.shape}, phi={phi.shape}')

    # Nonlocal potential
    V = wave.compute_nonlocal_potential(A, phi)
    print(f'Potential: {V.shape}')

    # Gradient energy
    E = wave.compute_gradient_energy(A, phi)
    print(f'Gradient energy: {E.item():.6f}')

    # Diagnostics
    diag = wave.get_diagnostics(A, phi)
    print(f'Diagnostics: {diag}')

    # NonlocalWaveInteraction
    print(f'\n=== NonlocalWaveInteraction test ===')
    interaction = NonlocalWaveInteraction(wave, output_dim=128)
    output = interaction(streams)
    print(f'Output: {output.shape}')

    params = sum(p.numel() for p in wave.parameters())
    params_nl = sum(p.numel() for p in interaction.parameters())
    print(f'\nParameters: WaveOnT={params}, NonlocalInteraction={params_nl}')
