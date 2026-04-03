"""
System Classification: src.constants
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

Theoretical Framework:
    - SubstanceNet theoretical framework (Onasenko, 2025-2026)
    - The Emergence Parameter κ ≈ 1 (Onasenko, 2025)

Constants and Configuration — Single Source of Truth
===========================================================
WARNING: All other modules MUST import constants FROM HERE.
Do NOT duplicate values in other modules.

Calibration Sources:
    - Reflexivity optimum: SubstanceNet v3.1.1 experiments (2025-08)
    - Architecture params: SubstanceNet v3/v3.2 experiments (2025)
    - Physical constants: 2d_substance_v2 constants.py v2.3.0

Version: 0.1.0
Date: 2026-02-11

Changelog:
    2026-02-11 v0.1.0 — Initial constants from v3.1.1 and v3.2 experiments
"""

# ============================================================================
# ARCHITECTURE
# ============================================================================

WAVE_DIM = 128                    # Quantum wave function dimensionality
CORTEX_CHANNELS = [64, 128, 256]  # V1 → V2 → V4 channel progression
GABOR_ORIENTATIONS = 8            # V1 Gabor filter orientations

# ============================================================================
# REFLEXIVE CONSCIOUSNESS (from SubstanceNet v3.1.1 experiments)
# ============================================================================

# Empirical optimum: R ∈ [0.35, 0.47] corresponds to κ ≈ 1
OPTIMAL_REFLEXIVITY_MIN = 0.35
OPTIMAL_REFLEXIVITY_MAX = 0.47

# TemporalConsciousnessController modes
# Format: {mode: {inertia, base_reflexivity, saturation_cap}}
CONSCIOUSNESS_MODES = {
    'stream': {
        'inertia': 0.99,
        'base_reflexivity': 0.30,
        'saturation_cap': 1.0,
        'description': 'Gradual adaptation, highest accuracy (93.74% MNIST)',
    },
    'balanced': {
        'inertia': 0.92,
        'base_reflexivity': 0.50,
        'saturation_cap': 0.9,
        'description': 'Balanced adaptation (93.47% MNIST)',
    },
    'focused': {
        'inertia': 0.80,
        'base_reflexivity': 0.40,
        'saturation_cap': 0.85,
        'description': 'Fast adaptation, focused processing',
    },
    'energy_save': {
        'inertia': 0.50,
        'base_reflexivity': 0.30,
        'saturation_cap': 0.7,
        'description': 'Minimal resources, low reflexivity',
    },
}

DEFAULT_CONSCIOUSNESS_MODE = 'stream'

# ============================================================================
# CRITICALITY (κ ≈ 1)
# ============================================================================

KAPPA_TARGET = 1.0                # Emergence parameter target
EWC_LAMBDA = 100.0                # Elastic Weight Consolidation strength

# ============================================================================
# EXPERIMENTAL RESULTS (reference values for validation)
# ============================================================================

REFERENCE_RESULTS = {
    'mnist_stream_1epoch': {
        'accuracy': 0.9374,
        'reflexivity': 0.382,
        'mode': 'stream',
    },
    'mnist_balanced_1epoch': {
        'accuracy': 0.9347,
        'reflexivity': 0.468,
        'mode': 'balanced',
    },
    'cifar10_best': {
        'accuracy': 0.7423,
        'reflexivity': 0.379,
        'params': 5_640_000,
    },
    'cifar10_efficient': {
        'accuracy': 0.6704,
        'reflexivity': 0.348,
        'params': 915_000,
        'efficiency_score': 7.34,
    },
}

# ============================================================================
# TRAINING DEFAULTS
# ============================================================================

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = 'cuda'
RANDOM_SEED = 42

# ============================================================================
# VALIDATION
# ============================================================================

def _validate_constants():
    """Validate consistency of constants at import time."""
    assert 0.0 < OPTIMAL_REFLEXIVITY_MIN < OPTIMAL_REFLEXIVITY_MAX < 1.0, \
        "Reflexivity bounds must be 0 < min < max < 1"
    assert KAPPA_TARGET > 0, "κ target must be positive"
    assert len(CONSCIOUSNESS_MODES) == 4, \
        f"Expected 4 consciousness modes, got {len(CONSCIOUSNESS_MODES)}"
    assert DEFAULT_CONSCIOUSNESS_MODE in CONSCIOUSNESS_MODES, \
        f"Default mode '{DEFAULT_CONSCIOUSNESS_MODE}' not in CONSCIOUSNESS_MODES"
    for mode, params in CONSCIOUSNESS_MODES.items():
        assert 0 <= params['base_reflexivity'] <= 1, \
            f"Mode '{mode}': base_reflexivity must be in [0, 1]"
        assert 0 <= params['saturation_cap'] <= 1, \
            f"Mode '{mode}': saturation_cap must be in [0, 1]"

_validate_constants()

# ============================================================================
# VERSION
# ============================================================================

__version__ = "0.1.0"
__date__ = "2026-02-11"
