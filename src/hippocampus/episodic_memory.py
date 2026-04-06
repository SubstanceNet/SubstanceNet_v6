"""
System Classification: src.hippocampus.episodic_memory
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
License: Apache-2.0

Theoretical Framework:
    - SubstanceNet theoretical framework (Onasenko, 2025-2026)
    - Episodic Memory (Tulving, 1972)
    - Memory Consolidation (Frankland & Bontempi, 2005)

Episodic Memory System
===========================================================
Implements encoding, storage, retrieval, and consolidation of
episodic memories with spatial-temporal context integration.

Architecture:
    Encode: input + GridCells + PlaceCells + TimeCells -> context
    Store:  context + importance -> short-term buffer
    Retrieve: query + consciousness -> attention-weighted recall
    Consolidate: short-term buffer -> long-term prototypes

Key References:
    - Tulving E. (1972) "Episodic and semantic memory"
    - Frankland P.W., Bontempi B. (2005) Nat. Rev. Neurosci. 6:119-130
    - Kumaran D. et al. (2016) Trends Cogn. Sci. 20:512-534

Changelog:
    2026-02-11 v0.1.0 — Ported from SubstanceNet v3.2 hippocampus_module_v2.py
"""
import math

import time
import hashlib
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import GridCells, PlaceCells, TimeCells


class EpisodicEncoder(nn.Module):
    """
    Encodes episodes with spatial-temporal context.

    Combines input features with grid cell, place cell, and time cell
    activations into a unified context representation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Dimensionality of encoded context.
    consciousness_dim : int
        Dimensionality of consciousness projection.
    num_places : int
        Number of place cell units.
    num_grid_modules : int
        Number of grid cell modules.
    num_time_cells : int
        Number of time cell units.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 consciousness_dim: int = 32, num_places: int = 100,
                 num_grid_modules: int = 4, num_time_cells: int = 50):
        super().__init__()

        self.grid_cells = GridCells(input_dim, num_grid_modules)
        self.place_cells = PlaceCells(input_dim, num_places)
        self.time_cells = TimeCells(num_time_cells)

        context_input_dim = (input_dim + self.grid_cells.output_dim +
                             num_places + num_time_cells)

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.consciousness_projection = nn.Linear(hidden_dim, consciousness_dim)

        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim + consciousness_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + consciousness_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, elapsed_time: float = 0.0,
               consciousness_state: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Encode an episode with spatial-temporal context.

        Parameters
        ----------
        x : torch.Tensor
            Input features [B, input_dim].
        elapsed_time : float
            Time since episode start.
        consciousness_state : torch.Tensor, optional
            Consciousness state [B, consciousness_dim].

        Returns
        -------
        dict
            Keys: context, grid_code, place_code, time_code,
            consciousness_proj, importance.
        """
        batch_size = x.shape[0]
        device = x.device

        grid_code = self.grid_cells(x)
        place_code = self.place_cells(x)
        time_code = self.time_cells(elapsed_time).unsqueeze(0).expand(
            batch_size, -1).to(device)

        combined = torch.cat([x, grid_code, place_code, time_code], dim=1)
        context = self.context_encoder(combined)
        consciousness_proj = self.consciousness_projection(context)

        if consciousness_state is not None:
            if consciousness_state.shape[1] != consciousness_proj.shape[1]:
                consciousness_state = F.adaptive_avg_pool1d(
                    consciousness_state.unsqueeze(1),
                    consciousness_proj.shape[1],
                ).squeeze(1)
            integrated = torch.cat([context, consciousness_state], dim=1)
            importance = self.importance_scorer(integrated)
        else:
            importance = torch.full((batch_size, 1), 0.5, device=device)

        return {
            'context': context,
            'grid_code': grid_code,
            'place_code': place_code,
            'time_code': time_code,
            'consciousness_proj': consciousness_proj,
            'importance': importance,
        }

    def decode(self, context: torch.Tensor,
               consciousness_state: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """Decode context back to input space."""
        if consciousness_state is None:
            c_dim = self.consciousness_projection.out_features
            consciousness_state = torch.zeros(
                context.shape[0], c_dim, device=context.device)
        return self.decoder(torch.cat([context, consciousness_state], dim=1))


class ConsciousRetrieval(nn.Module):
    """
    Consciousness-modulated memory retrieval via attention.

    Queries are matched against stored memories using scaled
    dot-product attention, with consciousness state modulating
    the attention weights.

    Parameters
    ----------
    memory_dim : int
        Dimensionality of memory representations.
    """

    def __init__(self, memory_dim: int):
        super().__init__()

        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)

        self.consciousness_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid(),
        )

    def forward(self, query: torch.Tensor,
                memory_bank: List[torch.Tensor],
                psi_c: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memories modulated by consciousness.

        Parameters
        ----------
        query : torch.Tensor
            Query vector [B, memory_dim].
        memory_bank : list of torch.Tensor
            Stored memory contexts.
        psi_c : torch.Tensor, optional
            Complex consciousness wave function [B, c_dim].

        Returns
        -------
        retrieved : torch.Tensor
            Retrieved memory [B, memory_dim].
        weights : torch.Tensor
            Attention weights [B, num_memories].
        """
        if len(memory_bank) == 0:
            return torch.zeros_like(query), torch.zeros(query.shape[0], 1,
                                                        device=query.device)

        q = self.query_proj(query)
        memories = torch.stack(memory_bank, dim=1)
        k = self.key_proj(memories)
        v = self.value_proj(memories)

        scale = math.sqrt(q.shape[-1])
        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / scale

        # Consciousness modulation of attention
        if psi_c is not None:
            psi_amp = torch.abs(psi_c)
            if psi_amp.shape[1] != scores.shape[2]:
                psi_amp = F.adaptive_avg_pool1d(
                    psi_amp.unsqueeze(1), scores.shape[2]).squeeze(1)
            scores = scores * psi_amp.unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, v).squeeze(1)

        # Consciousness gating
        if psi_c is not None:
            gate = self.consciousness_gate(
                torch.cat([query, retrieved], dim=1))
            retrieved = retrieved * gate

        return retrieved, weights.squeeze(1)


class MemoryConsolidation(nn.Module):
    """
    Adaptive consolidation from short-term to long-term memory.

    Uses learned prototypes that are updated via weighted
    aggregation of episodic memories, modulated by importance
    and reflexivity scores.

    Parameters
    ----------
    memory_dim : int
        Dimensionality of memory representations.
    num_prototypes : int
        Number of long-term memory prototypes.
    """

    def __init__(self, memory_dim: int, num_prototypes: int = 10):
        super().__init__()
        self.num_prototypes = num_prototypes

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, memory_dim))

        self.consolidation_net = nn.Sequential(
            nn.Linear(memory_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.prototype_updater = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh(),
        )

    def consolidate(self, episodes: List[Dict[str, torch.Tensor]],
                    reflexivity_scores: Optional[torch.Tensor] = None
                    ) -> torch.Tensor:
        """
        Consolidate episodes into long-term prototypes.

        Parameters
        ----------
        episodes : list of dict
            Episode data with 'context' and 'importance' keys.
        reflexivity_scores : torch.Tensor, optional
            Reflexivity weight per episode [num_episodes].

        Returns
        -------
        torch.Tensor
            Updated prototypes [num_prototypes, memory_dim].
        """
        if len(episodes) == 0:
            return self.prototypes

        contexts = torch.stack([ep['context'] for ep in episodes])
        avg_importances = torch.stack(
            [ep['importance'].mean() for ep in episodes])

        if reflexivity_scores is not None:
            avg_importances = avg_importances * reflexivity_scores.to(
                avg_importances.device)

        # Flatten batch dimension if present
        if contexts.dim() == 3:
            contexts = contexts.mean(dim=1)

        new_prototypes = []
        for i in range(self.num_prototypes):
            proto = self.prototypes[i].unsqueeze(0).expand(len(episodes), -1)
            consolidation_input = torch.cat(
                [contexts, proto, avg_importances.unsqueeze(1)], dim=1)

            weights = self.consolidation_net(consolidation_input)
            aggregated = (contexts * weights).sum(dim=0) / (
                weights.sum() + 1e-8)

            update_input = torch.cat([self.prototypes[i], aggregated])
            new_prototypes.append(self.prototype_updater(update_input))

        return torch.stack(new_prototypes)
