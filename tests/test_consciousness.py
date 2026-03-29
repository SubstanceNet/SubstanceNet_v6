"""
System Classification: tests.test_consciousness
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Tests for ReflexiveConsciousness and TemporalConsciousnessController.
"""

import torch
import pytest


def test_reflexive_output_shape():
    """Reflexive consciousness produces correct output shapes."""
    from src.consciousness import ReflexiveConsciousness

    rc = ReflexiveConsciousness(input_dim=8, consciousness_dim=16)
    x = torch.randn(4, 8)
    psi_c, amp, phase = rc(x)

    assert psi_c.shape == (4, 8), f"psi_c shape: {psi_c.shape}"
    assert amp.shape == (4, 8), f"amp shape: {amp.shape}"
    assert phase.shape == (4, 8), f"phase shape: {phase.shape}"
    assert psi_c.is_complex(), "psi_c must be complex"


def test_reflexivity_score_bounded():
    """Reflexivity score is in (0, 1]."""
    from src.consciousness import ReflexiveConsciousness

    rc = ReflexiveConsciousness(input_dim=8, consciousness_dim=16)
    x = torch.randn(4, 8)
    _, amp, phase = rc(x)
    metrics = rc.get_metrics(amp, phase)

    assert 0 < metrics['reflexivity_score'] <= 1.0


def test_consciousness_loss_computable():
    """Consciousness loss computes without errors."""
    from src.consciousness import ReflexiveConsciousness

    rc = ReflexiveConsciousness(input_dim=8, consciousness_dim=16)
    x = torch.randn(4, 8)
    _, amp, phase = rc(x)
    loss = rc.consciousness_loss(amp, phase)

    assert loss.shape == (), "Loss must be scalar"
    assert not torch.isnan(loss), "Loss must not be NaN"


def test_controller_modes():
    """All consciousness modes work correctly."""
    from src.consciousness import TemporalConsciousnessController

    for mode in ['stream', 'balanced', 'focused', 'energy_save']:
        ctrl = TemporalConsciousnessController(mode=mode)
        assert ctrl.current_level == ctrl.base_level

        # Simulate 50 batches
        for _ in range(50):
            level = ctrl.update({'reflexivity_score': 0.8})

        assert level <= ctrl.saturation_cap, \
            f"Mode '{mode}': {level} > cap {ctrl.saturation_cap}"


def test_controller_reset():
    """Controller resets to base level on task change."""
    from src.consciousness import TemporalConsciousnessController

    ctrl = TemporalConsciousnessController(mode='balanced')
    for _ in range(20):
        ctrl.update({'reflexivity_score': 0.9})

    level_before = ctrl.current_level
    ctrl.update({'reflexivity_score': 0.9}, task_changed=True)
    assert ctrl.current_level < level_before


def test_controller_analyze():
    """Controller analysis returns expected keys."""
    from src.consciousness import TemporalConsciousnessController

    ctrl = TemporalConsciousnessController()
    for _ in range(10):
        ctrl.update({'reflexivity_score': 0.5})

    analysis = ctrl.analyze()
    expected_keys = {'current_level', 'average', 'phase', 'mode'}
    assert expected_keys.issubset(analysis.keys())
