# Changelog

## v0.6.1 (2026-04-04)

### Release Preparation
- **ARCHITECTURE.md rewritten** for publication: Key Concepts glossary, neuroscience motivation (4 findings → 4 decisions), CNN/Transformer comparison table, biological context for all modules
- **All 6 methodology documents rewritten** to academic standard: Objective → Method → Results → Discussion → Conclusions → References
- **README.md rewritten** for release: cognitive architecture positioning, two key figures, Quick Start, Citation

### Code Cleanup
- Remove duplicated retinal preprocessing in BiologicalV1.forward()
- Remove duplicated methods in Hippocampus (store_feature, recognize)
- Add Notes on V4 feature degradation without Hebbian maturation
- Document two Hippocampus memory interfaces (episodic dim=3, recognition dim=128)
- Fix version label v4 → v6 in config.py and run_all_experiments.py

### Reference Cleanup
- Replace unpublished '2D-Substance Theory' with 'SubstanceNet theoretical framework' (all source files)
- Remove all references to unpublished monograph
- Replace 'Th 6.22' with inline equation, 'Chapter N' with descriptive references
- Remove 'Code: Claude (Anthropic)' from all source files
- Standardize attribution: Author + Developer + License in all 24 files

### Project Restructuring
- Rename experiments/ → experiments_v6/ (versioned experiment sets for future v7+)
- Move figures/ into experiments_v6/figures/
- Remove empty directories (scripts/, outputs/, checkpoints/)
- Update CITATION.cff to v6 (v4 → v6, wave-based → plain vectors)

### Figure Fixes
- Fix annotation overlaps in consolidation.png, hebbian_maturation.png, hebbian_recognition.png, innate_vs_acquired.png

### Verification
- 38/38 tests passed
- 6/6 experiments passed, all results identical to v0.6.0
- 10 figures regenerated in new location

---
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
