"""
System Classification: tests.test_hippocampus
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Tests for Hippocampus module.
"""

import torch
import pytest


def test_grid_cells_output_shape():
    """Grid cells produce correct output dimensions."""
    from src.hippocampus.cells import GridCells

    gc = GridCells(input_dim=32, num_modules=4)
    x = torch.randn(4, 32)
    out = gc(x)

    assert out.shape == (4, 24), f"Expected (4, 24), got {out.shape}"
    assert gc.output_dim == 24


def test_grid_cells_periodic():
    """Grid cell activations are bounded in [-1, 1] (cosine)."""
    from src.hippocampus.cells import GridCells

    gc = GridCells(input_dim=16, num_modules=3)
    x = torch.randn(8, 16) * 10  # large inputs
    out = gc(x)

    assert out.min() >= -1.0 and out.max() <= 1.0


def test_place_cells_output_shape():
    """Place cells produce correct output dimensions."""
    from src.hippocampus.cells import PlaceCells

    pc = PlaceCells(input_dim=32, num_places=50)
    x = torch.randn(4, 32)
    out = pc(x)

    assert out.shape == (4, 50)
    assert (out >= 0).all(), "Place cell activations must be non-negative"


def test_time_cells_decay():
    """Time cells decay monotonically with elapsed time."""
    from src.hippocampus.cells import TimeCells

    tc = TimeCells(num_cells=20)
    early = tc(0.1)
    late = tc(10.0)

    assert (early >= late).all(), "Time cells must decay with time"


def test_episodic_encoder():
    """Episodic encoder produces all expected keys."""
    from src.hippocampus.episodic_memory import EpisodicEncoder

    enc = EpisodicEncoder(input_dim=32, hidden_dim=64, consciousness_dim=16)
    x = torch.randn(4, 32)
    result = enc.encode(x)

    expected_keys = {'context', 'grid_code', 'place_code',
                     'time_code', 'consciousness_proj', 'importance'}
    assert expected_keys == set(result.keys())


def test_hippocampus_encode_and_retrieve():
    """Full hippocampus encode-store-retrieve cycle."""
    from src.hippocampus import Hippocampus

    hippo = Hippocampus(input_dim=32, hidden_dim=64,
                        consciousness_dim=16, buffer_size=100)

    # Store 5 episodes
    for i in range(5):
        x = torch.randn(2, 32)
        info = hippo.encode_and_store(x, task_type=f"task_{i}")
        assert 'episode_id' in info
        assert info['buffer_size'] == i + 1

    # Retrieve similar
    query = torch.randn(2, 32)
    results = hippo.retrieve_similar(query, top_k=3)
    assert len(results) <= 3
    assert len(results) > 0


def test_hippocampus_consolidation():
    """Consolidation runs without errors when buffer is large enough."""
    from src.hippocampus import Hippocampus

    hippo = Hippocampus(input_dim=32, hidden_dim=64,
                        consciousness_dim=16, buffer_size=100)

    # Store 15 episodes (above min_episodes=10)
    for i in range(15):
        x = torch.randn(2, 32)
        hippo.encode_and_store(x, task_type="consolidation_test")

    # Should run without error
    hippo.consolidate(min_episodes=10)
    state = hippo.get_state()
    assert state['total_episodes'] == 15


def test_hippocampus_state_summary():
    """Memory state summary contains expected keys."""
    from src.hippocampus import Hippocampus

    hippo = Hippocampus(input_dim=16, hidden_dim=32, consciousness_dim=8)
    state = hippo.get_state()

    expected_keys = {'short_term_size', 'total_episodes',
                     'num_prototypes', 'avg_importance', 'memory_decay'}
    assert expected_keys == set(state.keys())
