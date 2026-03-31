# Changelog

## v0.6.0 (2026-04-01)

### Architecture
- **Plain vectors replace wave formalism** in V1→V4 pipeline. FeatureProjection (Linear+ReLU) instead of QuantumWaveFunction (softplus+cos/sin). Confirmed by ablation: plain vectors outperform wave on all metrics.
- **Top-down gate infrastructure** (ψ_C → V4 modulation, inactive). Biological basis: prefrontal → V4 feedback. Ready for v7 activation.
- **Hippocampus feature_dim=128** for recognition. store_feature/recognize API alongside existing episodic memory (abstract_dim=3).

### Experiments
- 6 clean experiments (01-06) with methodology documents
- Exp06: κ ≈ 1 emergence analysis — **κ = 0.993 ± 0.010** (He-II ref: 0.989 ± 0.007)
- All results improved vs v5: MNIST +2.0%, Recognition +6.3%, κ closer to He-II

### Key Results
| Metric | v6 | v5 |
|--------|-----|-----|
| MNIST backprop | 97.4% | 95.4% |
| Recognition 100-shot | 73.2% ± 2.1% | 66.9% |
| Cognitive R | 0.409 ± 0.001 | 0.410 ± 0.001 |
| κ ≈ 1 | 0.993 ± 0.010 | 0.986 ± 0.013 |

### Cleanup
- Wave modules moved to research/wave_dynamics/ (quantum_wave, wave_on_t, reflexive_v2)
- Old docs moved to research/docs_v4_v5/
- Deprecated experiments (08, 09, 11) moved to research/experiments_v5/
- 38/38 tests pass

---

## v0.5.0 (2026-03-31)

### New Modules
- WaveFunctionOnT: wave function on configuration space T = 2^i − 1
- ReflexiveConsciousnessV2: energy-based evolution (experimental, unstable)
- Hippocampus feature_dim=128 fix

### Experiments
- Exp09: Wave ablation (3-way: Classic/WaveOnT/Plain)
- Exp10: κ ≈ 1 emergence analysis with He-II analogy
- Exp11: Statistical analysis with 95% CIs

### Theoretical Insights
- Three frameworks converge: Onasenko (Σ) ≡ Dubovikov (T) ≡ Tsien (FCM)
- R-targeting = model of physiological constraints
- Wave dynamics belongs in consciousness, not feature extraction

---

## v0.4.0 (2026-03-29)

### Infrastructure
- 6 reproducible experiments with publication-quality figures
- Full demo suite (4 scripts)
- run_all_experiments.py (~74s)
- CITATION.cff, CHANGELOG, ARCHITECTURE.md

### Experiments
- Exp01-05, Exp08: MNIST, cognitive battery, recognition, velocity, Hebbian, moving MNIST
- External review: 8.7/10

---

## v0.3.0 — v0.1.0 (2026-02 — 2026-03)

Historical development: V1-V4 cortex, consciousness module, Hebbian learning, hippocampus, MosaicField18 restoration, recognition paradigm. See research/docs_v4_v5/ for session logs.
