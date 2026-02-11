"""
System Classification: src.__init__
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

SubstanceNet v4 — Bio-Inspired Neural Network
with Reflexive Consciousness
===========================================================
Neural network architecture implementing reflexive consciousness
based on 2D-Substance Theory.

    from src import BiologicalV1, QuantumWaveFunction
    from src import ReflexiveConsciousness, TemporalConsciousnessController

Theoretical Foundations:
    - 2D-Substance Theory: psi_i on Sigma -> P_hat -> R^(3+1)
    - Reflexive Consciousness: psi_C = F[P_hat[psi_C]] (Th 6.22)
    - Emergence Parameter: kappa ~ 1 at criticality
    - Visual Cortex: V1->V2->V3->V4 projection (Hubel & Wiesel)
"""

__version__ = "0.1.0"

from .wave import QuantumWaveFunction
from .cortex import BiologicalV1
from .consciousness import ReflexiveConsciousness, TemporalConsciousnessController

__all__ = [
    '__version__',
    'QuantumWaveFunction',
    'BiologicalV1',
    'ReflexiveConsciousness',
    'TemporalConsciousnessController',
]
