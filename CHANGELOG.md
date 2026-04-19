# Changelog

## v0.6.2 (2026-04-19)

### Context
Patch release addressing formulation clarifications and code hygiene items
identified during external L1-factual code audit (95.1% audit confirmation rate).
**No published numerical results have changed.** All modifications are to
documentation, comments, labels, and dead-code annotations.

### Fixed (code ↔ documentation)
- **Recognition feature stage** (preprint §5.3, §6.1, methodology 03): features for kNN
  are extracted at the FeatureProjection stage (V1 + Orientation + projection),
  not from the full V1→V4 pipeline. V2/V3/V4 run in parallel for classification
  logits but are bypassed for recognition feature extraction.
- **exp06 Test 3 label** (`'No consciousness'` → `'Consciousness frozen'`): the
  ablation freezes consciousness parameters (requires_grad=False) but keeps the
  module active in forward pass and loss. This is a gradient ablation, not a
  structural ablation.
- **exp06 methodology**: Test 1 uses 50 gradient steps per task (was documented
  as 100); Test 2 uses 100 as documented. Terminology clarified from 'epochs' to
  'gradient steps' (fresh synthetic batch per step, no fixed dataset).
- **exp04/exp05 methodology**: `image_size=32` in generate_sequence calls, not
  28×28 as previously documented. MNIST (exp01) remains on 28×28.

### Documentation (preprint SubstanceNet_v6_preprint_v4.md)
- §5.4 Loss table: r_penalty annotated as monitoring-only (torch.no_grad(),
  requires_grad=False, no gradient contribution). Effective R-targeting is
  through consciousness_loss.reflexivity_loss (weight 0.1 × 0.3 = 0.03).
- §3 Finding 2: HebbianLinear formula aligned with code. Code implements
  global phase coherence `<cos(φ)>` (simplified STDP variant), not pairwise
  `cos(φ_i − φ_j)`. Bi & Poo 2001 reference reformulated accordingly.
- §6.4 Test 2: running normalization `Λ_c = max(Λ, 1e-4) ≈ Λ` makes κ
  trivially equal to τ per checkpoint. Test 2 does not provide independent
  evidence of compensating mechanism; Test 1 (10 tasks) remains the primary
  κ-variation measurement.
- §6.4 Table 9: 'No consciousness' → 'Consciousness frozen' with explanatory
  note on the distinction between gradient and structural ablation.
- §3 Finding 1: 'Innate' defined as 'functional without gradient training',
  not 'lacking trainable parameters'. V1 contains ~50K backprop-learnable
  parameters in addition to fixed Gabor filter bank.
- §5.3: Hippocampus architectural status explicitly noted — complete and
  tested (8 unit tests) but not invoked in v6 publication experiments.
  Full activation planned for v7.
- §8 Limitations: two new points added — (7) hippocampus architectural
  underuse, (8) recognition feature stage.

### Cleanup (code annotations, no behavioral change)
- `tests/test_wave.py` → `tests/test_feature_proj.py` (rename; contents
  already tested FeatureProjection, name was legacy from v5).
- `src/constants.py`: __version__ synced from "0.1.0" to "0.6.2";
  EWC_LAMBDA constant removed (unused legacy from v3 continual learning).
- `src/utils.py`: `compute_integration_info`, `compute_phi_approx` annotated
  as UNUSED IN v6 (retained from v5 wave-formalism; candidates for v7).
- `src/wave/__init__.py`: docstring expanded with v5→v6 migration history
  and pointers to replacement (FeatureProjection) and archive
  (research/wave_dynamics/).
- `src/model/substance_net.py::compute_loss`: r_penalty computation annotated
  with monitoring-only semantics and pointer to effective R-regularization.
- `experiments_v6/06_kappa_analysis.py::measure_kappa_components`: fallback
  branch (pairwise cos_matrix for v2 wave-consciousness) annotated as
  always-taken in v6; retained for research/reflexive_v2.py compatibility.
- `experiments_v6/06_kappa_analysis.py`: unused `use_v2` tuple parameter
  removed from Test 1, Test 2, and Test 3 loops.

### Added
- `experiments_v6/06_kappa_analysis.py`: Test 3 results now persisted to
  JSON under key `test3_consciousness_ablation`. Previously only printed
  to console, making Preprint Table 9 non-auto-reproducible.
  Note: existing results JSON remains from original v0.6.0 run; new key
  will appear on next exp06 execution.

### Not changed (explicitly verified)
- All 6 experiments produce identical numerical results to v0.6.1
- Published central results unchanged:
  - MNIST 97.4% (1-epoch backprop, exp01)
  - Recognition 73.2% ± 2.1% (100-shot kNN, exp03)
  - κ = 0.993 ± 0.010 across 10 cognitive tasks (exp06 Test 1)
  - R = 0.4090 ± 0.0009 κ-plateau (exp02, exp06)
  - V3 motion 1.6× amplification after Hebbian maturation (exp05)
  - 720× V3:abstract compression ratio (exp04)
- Architectural design unchanged
- Training protocols unchanged

### Verification
- 38/38 tests passed (post-rename, post-version-sync)
- Python syntax validated on all modified files
- No experiments re-run; published JSONs from v0.6.0 remain reference

---

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
