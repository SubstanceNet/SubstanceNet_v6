"""
System Classification: src.__init__
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: Apache-2.0

SubstanceNet v4 — Bio-Inspired Neural Network
with Reflexive Consciousness
===========================================================
Usage:
    from src import SubstanceNet                  # Full model template
    from src import BiologicalV1    # Core modules
    from src import ReflexiveConsciousness, TemporalConsciousnessController
    from src import Hippocampus

Theoretical Foundations:
    - 2D-Substance Theory: psi_i on Sigma -> P_hat -> R^(3+1)
    - Reflexive Consciousness: psi_C = F[P_hat[psi_C]] (Th 6.22)
    - Emergence Parameter: kappa ~ 1 at criticality
    - Visual Cortex: V1->V2->V3->V4 projection (Hubel & Wiesel)
    - Hippocampal Memory: grid/place/time cells (O'Keefe, Moser)
"""

__version__ = "0.1.0"

from .cortex import BiologicalV1
from .consciousness import ReflexiveConsciousness, TemporalConsciousnessController
from .hippocampus import Hippocampus
from .model import SubstanceNet

__all__ = [
    '__version__',
    'SubstanceNet',
    'BiologicalV1',
    'ReflexiveConsciousness',
    'TemporalConsciousnessController',
    'Hippocampus',
]
