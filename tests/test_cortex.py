"""
System Classification: tests.test_cortex
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
License: Apache-2.0

Tests for BiologicalV1 cortex module.
"""

import torch
import pytest


def test_v1_output_shape_mnist():
    """V1 produces correct output for MNIST-sized input."""
    from src.cortex import BiologicalV1

    v1 = BiologicalV1(n_orientations=8, n_scales=3, output_dim=64)
    x = torch.randn(2, 1, 28, 28)
    sequence, activations = v1(x)

    assert sequence.shape == (2, 9, 64), f"sequence shape: {sequence.shape}"
    assert 'simple' in activations
    assert 'complex' in activations
    assert 'hypercolumns' in activations


def test_v1_output_shape_cifar():
    """V1 produces correct output for CIFAR-sized input."""
    from src.cortex import BiologicalV1

    v1 = BiologicalV1(n_orientations=8, n_scales=4, output_dim=128)
    # CIFAR grayscale (single channel)
    x = torch.randn(2, 1, 32, 32)
    sequence, activations = v1(x)

    assert sequence.shape == (2, 9, 128)


def test_gabor_filter_bank_count():
    """Gabor bank has correct number of filters."""
    from src.cortex.v1 import GaborFilterBank

    bank = GaborFilterBank(n_orientations=8, n_scales=4)
    assert bank.num_filters == 8 * 4 * 2  # orientations * scales * (even+odd)
    assert bank.gabor_filters.shape[0] == 64


def test_gabor_filter_normalized():
    """Gabor filters are zero-mean and unit-max."""
    from src.cortex.v1 import create_gabor_filter

    g = create_gabor_filter(size=11, sigma=3, theta=0.5)
    assert abs(g.mean().item()) < 0.1, "Filter should be approximately zero-mean"
    assert abs(g.abs().max().item() - 1.0) < 0.1, "Filter max should be ~1"
