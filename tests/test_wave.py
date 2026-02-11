"""
System Classification: tests.test_wave
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Tests for QuantumWaveFunction module.
"""

import torch
import pytest


def test_wave_function_output_shape():
    """Wave function produces correct output shapes."""
    from src.wave import QuantumWaveFunction

    wf = QuantumWaveFunction(in_channels=48, out_channels=32, grid_size=256)
    x = torch.randn(2, 256, 48)
    psi, amp, phase = wf(x)

    assert psi.shape == (2, 256, 16), f"psi shape: {psi.shape}"
    assert amp.shape == (2, 256, 16), f"amp shape: {amp.shape}"
    assert phase.shape == (2, 256, 16), f"phase shape: {phase.shape}"
    assert psi.is_complex(), "psi must be complex"


def test_amplitude_non_negative():
    """Amplitude is always non-negative (softplus guarantee)."""
    from src.wave import QuantumWaveFunction

    wf = QuantumWaveFunction(in_channels=16, out_channels=8)
    x = torch.randn(4, 64, 16)
    _, amp, _ = wf(x)

    assert (amp >= 0).all(), "Amplitude must be non-negative"


def test_zero_loss_computable():
    """Zero-minimization loss computes without errors."""
    from src.wave import QuantumWaveFunction

    wf = QuantumWaveFunction(in_channels=16, out_channels=8)
    x = torch.randn(2, 64, 16)
    _, amp, phase = wf(x)
    loss = wf.zero_loss(amp, phase)

    assert loss.shape == (), "Loss must be scalar"
    assert not torch.isnan(loss), "Loss must not be NaN"


def test_even_channels_required():
    """Odd out_channels raises ValueError."""
    from src.wave import QuantumWaveFunction

    with pytest.raises(ValueError):
        QuantumWaveFunction(in_channels=16, out_channels=7)
