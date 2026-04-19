"""
Wave module — v6: core wave modules moved to research/wave_dynamics/

History:
    v5: contained QuantumWaveFunction, pairwise phase coherence,
        V3 phase interference primitives.
    v6.0: wave formalism replaced by FeatureProjection
        (see src/model/substance_net.py). Wave code archived
        in research/wave_dynamics/.

Retained in v6: NonlocalInteraction (attention-based, used in pipeline)
                moved to src/model/layers.py.

Tests: the replacement is tested by tests/test_feature_proj.py
       (formerly tests/test_wave.py).

Candidate for v7 reactivation: pairwise cos_matrix phase coherence
    in ReflexiveConsciousness v2 (research/reflexive_v2.py).
"""
