"""
System Classification: src.consciousness.controller
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - The Emergence Parameter kappa ~ 1 (Onasenko, 2025)

Temporal Consciousness Controller
===========================================================
Prevents reflexivity saturation (R -> 1.0) by implementing
biologically-inspired temporal dynamics with inertia and
saturation caps. Maintains R in optimal range [0.35, 0.47].

Mathematical Basis:
    R(t) = inertia * R(t-1) + (1 - inertia) * R_raw(t)
    R(t) = min(R(t), saturation_cap)
    kappa = (A/A_c) * tau * (Lambda/Lambda_c) ~ 1  — target regime

Key References:
    - Onasenko O. (2026) Monograph "2D-Substance", Chapter 6
    - SubstanceNet v3.1.1: discovery of accumulative integration effect

Changelog:
    2026-02-11 v0.1.0 — Ported from SubstanceNet v3.1.1 adaptive_consciousness_v2.py
"""

import numpy as np
from src.constants import CONSCIOUSNESS_MODES, DEFAULT_CONSCIOUSNESS_MODE


class TemporalConsciousnessController:
    """
    Temporal dynamics controller for reflexive consciousness.

    Manages accumulation and reset of reflexivity level, preventing
    saturation through inertia-based smoothing and hard caps.

    Parameters
    ----------
    mode : str
        Consciousness mode: 'stream', 'balanced', 'focused', 'energy_save'.

    Attributes
    ----------
    current_level : float
        Current reflexivity level R in [0, 1].
    reflexivity_history : list
        History of R values for analysis.
    """

    def __init__(self, mode: str = DEFAULT_CONSCIOUSNESS_MODE):
        self.reflexivity_history = []
        self.coherence_history = []
        self.batch_counter = 0
        self.task_switches = []

        self.set_mode(mode)

    def set_mode(self, mode: str):
        """Set consciousness operating mode."""
        if mode not in CONSCIOUSNESS_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Available: {list(CONSCIOUSNESS_MODES.keys())}"
            )

        self.mode_name = mode
        self.mode = CONSCIOUSNESS_MODES[mode]
        self.current_level = self.mode['base_reflexivity']

    @property
    def inertia(self) -> float:
        return self.mode['inertia']

    @property
    def saturation_cap(self) -> float:
        return self.mode['saturation_cap']

    @property
    def base_level(self) -> float:
        return self.mode['base_reflexivity']

    def update(self, metrics: dict, task_changed: bool = False) -> float:
        """
        Update reflexivity level with temporal dynamics.

        Parameters
        ----------
        metrics : dict
            Must contain 'reflexivity_score' key.
        task_changed : bool
            If True, reset to base level.

        Returns
        -------
        float
            Updated reflexivity level.
        """
        self.batch_counter += 1

        if task_changed:
            self.reset()
            self.task_switches.append(self.batch_counter)

        reflexivity = metrics.get('reflexivity_score', 0.5)

        # Inertia-based smoothing
        self.current_level = (
            self.current_level * self.inertia +
            reflexivity * (1 - self.inertia)
        )

        # Saturation cap
        self.current_level = min(self.current_level, self.saturation_cap)

        # Record history
        self.reflexivity_history.append(self.current_level)
        self.coherence_history.append(
            metrics.get('phase_coherence', 1.0)
        )

        return self.current_level

    def reset(self):
        """Reset reflexivity to base level."""
        self.current_level = self.base_level

    def get_phase(self) -> str:
        """Determine current consciousness phase."""
        if self.current_level < 0.5:
            return 'initialization'
        elif self.current_level < 0.8:
            return 'accumulation'
        elif self.current_level < 0.95:
            return 'convergence'
        return 'saturation'

    def analyze(self) -> dict:
        """
        Analyze temporal dynamics.

        Returns
        -------
        dict
            Analysis with current_level, average, std, min, max,
            growth_rate, saturation_ratio, phase, total_batches.
        """
        if len(self.reflexivity_history) < 2:
            return {'current_level': self.current_level, 'phase': 'startup'}

        history = np.array(self.reflexivity_history)

        return {
            'current_level': self.current_level,
            'average': float(np.mean(history)),
            'std': float(np.std(history)),
            'min': float(np.min(history)),
            'max': float(np.max(history)),
            'growth_rate': float((history[-1] - history[0]) / len(history)),
            'saturation_ratio': self.current_level / self.saturation_cap,
            'phase': self.get_phase(),
            'total_batches': self.batch_counter,
            'task_switches': len(self.task_switches),
            'mode': self.mode_name,
        }
