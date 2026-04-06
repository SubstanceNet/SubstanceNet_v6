"""
System Classification: src.hippocampus.hippocampus
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
License: Apache-2.0

Theoretical Framework:
    - SubstanceNet theoretical framework (Onasenko, 2025-2026)
    - Hippocampal Memory System (Squire, 1992)
    - Complementary Learning Systems (McClelland et al., 1995)

Hippocampus Module
===========================================================
Complete hippocampal memory system integrating spatial coding
(grid/place cells), temporal coding (time cells), episodic
encoding, consciousness-modulated retrieval, and adaptive
consolidation from short-term to long-term memory.

Architecture:
    GridCells + PlaceCells + TimeCells  (spatial-temporal scaffold)
           |
    EpisodicEncoder  (context formation)
           |
    ShortTermBuffer  (working memory, deque)
           |
    ConsciousRetrieval  (attention + psi_C modulation)
           |
    MemoryConsolidation  (prototype-based long-term)

Key References:
    - Squire L.R. (1992) Psychol. Rev. 99:195-231
    - McClelland J.L. et al. (1995) Psychol. Rev. 102:419-457
    - Kumaran D. et al. (2016) Trends Cogn. Sci. 20:512-534

Changelog:
    2026-02-11 v0.1.0 — Ported from SubstanceNet v3.2 hippocampus_module_v2.py
"""

import time
import json
import os
import hashlib
from collections import deque
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodic_memory import EpisodicEncoder, ConsciousRetrieval, MemoryConsolidation


class Hippocampus(nn.Module):
    """
    Complete hippocampal memory system.

    Integrates episodic encoding with spatial-temporal context,
    consciousness-modulated retrieval, and adaptive consolidation
    into long-term prototypical memories.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Dimensionality of memory context representations.
    consciousness_dim : int
        Dimensionality of consciousness state.
    buffer_size : int
        Maximum capacity of short-term memory buffer.
    num_prototypes : int
        Number of long-term memory prototypes.
    forgetting_rate : float
        Threshold below which episodes are forgotten.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 consciousness_dim: int = 32, buffer_size: int = 1000,
                 num_prototypes: int = 10, forgetting_rate: float = 0.01,
                 feature_dim: int = 128):
        super().__init__()

        self.encoder = EpisodicEncoder(
            input_dim, hidden_dim, consciousness_dim)
        self.retrieval = ConsciousRetrieval(hidden_dim)
        self.consolidation = MemoryConsolidation(hidden_dim, num_prototypes)

        self.short_term_buffer: deque = deque(maxlen=buffer_size)
        self.episode_start_time = time.time()
        self.total_episodes = 0
        self.forgetting_rate = forgetting_rate
        self.memory_decay = nn.Parameter(torch.tensor(0.99))

        # Feature encoder for recognition (128-dim amp+phase → hidden_dim)
        self.feature_dim = feature_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feature_memory = []
        self.max_feature_memory = buffer_size

    def encode_and_store(self, x: torch.Tensor,
                         consciousness_state: Optional[torch.Tensor] = None,
                         task_type: str = 'unknown',
                         metrics: Optional[Dict[str, float]] = None
                         ) -> Dict[str, Any]:
        """
        Encode input and store as episodic memory.

        Parameters
        ----------
        x : torch.Tensor
            Input features [B, input_dim].
        consciousness_state : torch.Tensor, optional
            Consciousness state for importance scoring.
        task_type : str
            Task identifier for the episode.
        metrics : dict, optional
            Performance metrics to store with episode.

        Returns
        -------
        dict
            Episode info: id, importance, buffer_size.
        """
        current_time = time.time()
        elapsed = current_time - self.episode_start_time

        episode_data = self.encoder.encode(x, elapsed, consciousness_state)

        episode = {
            'id': hashlib.md5(
                f"{task_type}_{current_time}_{self.total_episodes}"
                .encode()).hexdigest(),
            'timestamp': current_time,
            'task_type': task_type,
            'data': episode_data,
            'metrics': metrics or {},
            'decay_factor': 1.0,
        }

        self.short_term_buffer.append(episode)
        self.total_episodes += 1

        return {
            'episode_id': episode['id'],
            'importance': episode_data['importance'].mean().item(),
            'buffer_size': len(self.short_term_buffer),
        }

    def retrieve_similar(self, query: torch.Tensor,
                         consciousness_state: Optional[torch.Tensor] = None,
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most similar episodes from short-term buffer.

        Parameters
        ----------
        query : torch.Tensor
            Query features [B, input_dim].
        consciousness_state : torch.Tensor, optional
            Consciousness state for retrieval modulation.
        top_k : int
            Number of episodes to retrieve.

        Returns
        -------
        list of dict
            Top-k most similar episodes.
        """
        if len(self.short_term_buffer) == 0:
            return []

        query_enc = self.encoder.encode(query, 0.0, consciousness_state)
        query_ctx = query_enc['context']

        similarities = []
        for episode in self.short_term_buffer:
            ep_ctx = episode['data']['context']
            sim = F.cosine_similarity(
                query_ctx.unsqueeze(1), ep_ctx.unsqueeze(1)
            ).mean().item()
            sim *= episode['decay_factor']
            similarities.append((sim, episode))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in similarities[:top_k]]

    def consolidate(self, reflexivity_scores: Optional[torch.Tensor] = None,
                    min_episodes: int = 10, window: int = 50):
        """
        Consolidate short-term memories into long-term prototypes.

        Parameters
        ----------
        reflexivity_scores : torch.Tensor, optional
            Reflexivity weight per recent episode.
        min_episodes : int
            Minimum buffer size before consolidation triggers.
        window : int
            Number of recent episodes to consolidate.
        """
        if len(self.short_term_buffer) < min_episodes:
            return

        recent = list(self.short_term_buffer)[-window:]
        episode_data = [ep['data'] for ep in recent]

        if reflexivity_scores is None:
            reflexivity_scores = torch.tensor(
                [ep['data']['importance'].mean().item() for ep in recent])

        self.consolidation.prototypes.data = self.consolidation.consolidate(
            episode_data, reflexivity_scores)

        self._apply_forgetting()

    def _apply_forgetting(self):
        """Apply exponential decay and remove forgotten episodes."""
        to_remove = []
        for episode in self.short_term_buffer:
            episode['decay_factor'] *= self.memory_decay.item()
            if episode['decay_factor'] < self.forgetting_rate:
                to_remove.append(episode)

        for episode in to_remove:
            self.short_term_buffer.remove(episode)

    def store_feature(self, features: torch.Tensor, label: int):
        """
        Store recognition features (128-dim) in feature memory.

        Parameters
        ----------
        features : torch.Tensor
            [feature_dim] — amplitude+phase features.
        label : int
            Class label.
        """
        with torch.no_grad():
            encoded = self.feature_encoder(features.unsqueeze(0)).squeeze(0)
        self.feature_memory.append((encoded.detach().cpu(), label))
        if len(self.feature_memory) > self.max_feature_memory:
            self.feature_memory.pop(0)

    def recognize(self, features: torch.Tensor, top_k: int = 5) -> int:
        """
        Recognize by comparing features with stored memory.

        Parameters
        ----------
        features : torch.Tensor
            [feature_dim] — query features.
        top_k : int
            Number of neighbors for voting.

        Returns
        -------
        int
            Predicted class label (-1 if memory empty).
        """
        if not self.feature_memory:
            return -1

        with torch.no_grad():
            query = self.feature_encoder(features.unsqueeze(0)).squeeze(0)
            query = query.cpu()

            mem_feats = torch.stack([m[0] for m in self.feature_memory])
            sims = torch.nn.functional.cosine_similarity(
                query.unsqueeze(0), mem_feats, dim=1)
            topk_vals, topk_idx = sims.topk(min(top_k, len(self.feature_memory)))

            votes = {}
            for j in range(topk_vals.shape[0]):
                lbl = self.feature_memory[topk_idx[j].item()][1]
                votes[lbl] = votes.get(lbl, 0) + topk_vals[j].item()

        return max(votes, key=votes.get) if votes else -1

    def clear_feature_memory(self):
        """Clear feature memory."""
        self.feature_memory = []

    def get_feature_memory_size(self) -> int:
        """Return number of stored features."""
        return len(self.feature_memory)
    def get_state(self) -> Dict[str, Any]:
        """Return current memory system state summary."""
        recent = list(self.short_term_buffer)[-10:]
        avg_importance = (
            np.mean([ep['data']['importance'].mean().item()
                     for ep in recent])
            if recent else 0.0
        )
        return {
            'short_term_size': len(self.short_term_buffer),
            'total_episodes': self.total_episodes,
            'num_prototypes': self.consolidation.num_prototypes,
            'avg_importance': avg_importance,
            'memory_decay': self.memory_decay.item(),
        }

    def save_state(self, filepath: str):
        """Save memory state to JSON file."""
        state = {
            'prototypes': self.consolidation.prototypes.detach().cpu().tolist(),
            'total_episodes': self.total_episodes,
            'memory_decay': self.memory_decay.item(),
            'buffer_meta': [
                {
                    'id': ep['id'],
                    'timestamp': ep['timestamp'],
                    'task_type': ep['task_type'],
                    'metrics': ep['metrics'],
                    'decay_factor': ep['decay_factor'],
                }
                for ep in list(self.short_term_buffer)[-100:]
            ],
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load memory state from JSON file."""
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.consolidation.prototypes.data = torch.tensor(
            state['prototypes'], dtype=torch.float32)
        self.total_episodes = state['total_episodes']
        self.memory_decay.data = torch.tensor(state['memory_decay'])
