"""
System Classification: tests.test_constants
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Tests for constants validation and consistency.
"""

import pytest


def test_constants_import():
    """Constants module imports without errors (validation runs at import)."""
    from src import constants
    assert constants.__version__ == "0.1.0"


def test_reflexivity_bounds():
    """Optimal reflexivity range is valid."""
    from src.constants import OPTIMAL_REFLEXIVITY_MIN, OPTIMAL_REFLEXIVITY_MAX
    assert 0.0 < OPTIMAL_REFLEXIVITY_MIN < OPTIMAL_REFLEXIVITY_MAX < 1.0


def test_consciousness_modes():
    """All consciousness modes have required parameters."""
    from src.constants import CONSCIOUSNESS_MODES
    required_keys = {'inertia', 'base_reflexivity', 'saturation_cap', 'description'}
    for mode_name, params in CONSCIOUSNESS_MODES.items():
        assert required_keys.issubset(params.keys()), \
            f"Mode '{mode_name}' missing keys: {required_keys - params.keys()}"


def test_reference_results():
    """Reference results contain expected experiments."""
    from src.constants import REFERENCE_RESULTS
    assert 'mnist_stream_1epoch' in REFERENCE_RESULTS
    assert REFERENCE_RESULTS['mnist_stream_1epoch']['accuracy'] > 0.93
