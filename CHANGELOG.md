# Changelog

All notable changes to SubstanceNet v4 are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/)

## [0.4.0] — 2026-03-18

### Added
- **HebbianLinear** (`src/cortex/hebbian.py`) — phase-coherence plasticity without backpropagation (Oja stabilization)
- **V4 Hebbian compression** — `ObjectFeaturesV4` with HebbianLinear replacing nn.Linear
- **Moving MNIST** experiments — cross-modal recognition (static→moving, moving→static)
- **Hebbian maturation** — unsupervised V3/V4 weight adaptation through observation
- Video mode tests, cognitive mode tests, reflexivity range test in test_model.py

### Changed
- V3 `output_proj` replaced with HebbianLinear (phase-coherence learning)
- V4 `compress` replaced with two HebbianLinear layers
- V4 `oja_alpha` tuned to 1.0 for stability
- `count_parameters()` now includes v2, coherence_fc, stability_fc, cognitive_input
- Tests updated for current architecture (38 tests, all passing)

### Results
- Hebbian V3: 11× motion signal amplification after 50 unsupervised steps
- Hebbian maturation on MNIST: V4 accuracy +5.0% (0.510→0.561)
- Moving MNIST recognition: 36.0% at speed=2.0 (3.6× random)
- Cross-modal: moving protos → static test = 34.3%
- CIFAR-10 RGB: 24.4% (100-shot, matured)

## [0.3.0] — 2026-03-17

### Added
- **DynamicFormV3** (`src/cortex/v3.py`) — phase interference for form-motion binding
- **Dynamic primitives generator** (`src/data/dynamic_primitives.py`) — moving shapes [B,T,1,H,W]
- **Video mode** in SubstanceNet — `mode='video'` for frame sequences
- V2 streams interface — `return_streams=True` for thick/thin/pale separation

### Results
- Velocity tuning curve: 0.0→1.19, logarithmic saturation matching primate MT/V3
- V3 raw diff (moving vs static): 1.17 (V4 abstract diff: 0.001 — correct invariance)

## [0.2.0] — 2026-03-16

### Added
- **RetinalLayer** (`src/cortex/v1.py`) — RGB→4 photoreceptor channels (rods + L/M/S cones)
- **Recognition paradigm** (`research/recognition_paradigm.py`) — encode→store→recognize without backprop
- GaborFilterBank `in_channels` parameter for color support

### Results
- CIFAR-10 RGB: 40.56% vs grayscale 36.36% (+4.2%)
- Recognition 100-shot: 71.9% without backpropagation
- Consolidation: 10 prototypes > 200 raw episodes (46.3% vs 44.7%, 20× compression)
- Innate V1+V2 features: 53.7% without any training

## [0.1.1] — 2026-03-15

### Added
- **MosaicField18** (`src/cortex/v2.py`) — V2 cortex (thick/thin/pale stripes)
- **DynamicFormV3** (`src/cortex/v3.py`) — V3 cortex (cross-stream gating)
- **ObjectFeaturesV4** (`src/cortex/v4.py`) — V4 cortex (multi-scale attention)
- **R-targeting** in consciousness loss (target MSE=1.44 → R≈0.41)
- **TemporalConsciousnessController** integration with κ≈1 phase calibration
- **Hippocampus** integration (store_episode, recall, consolidate_memory)
- `src/utils.py` — cognitive data generators (transferred from v3.1.1)
- coherence_fc, stability_fc in SubstanceNet

### Fixed
- Consciousness saturation R→0.999 resolved (root cause: missing V2)
- Abstract representation collapse (variance 2.5e-11 → 10–259)
- Zero gradients in abstraction and consciousness modules
- External dependency on v3.1.1 sys.path hack removed

### Results
- MNIST 1-epoch: 95.94% (vs v3.1.1: 93.74%), R=0.41
- Cognitive tasks: 99.61% accuracy, R=0.41 (κ-plateau)
- V2 ablation: -46.9% accuracy drop (critical module)

## [0.1.0] — 2026-02-11

### Added
- Initial project structure with modular architecture
- `src/constants.py` — single source of truth for parameters
- BiologicalV1 with Gabor filter bank
- QuantumWaveFunction (ψ = A·e^(iφ))
- ReflexiveConsciousness (ported from v3.1.1)
- Hippocampus modules (ported from v3.1.1, not connected)
- 6 test files, research templates

### Known Issues
- Consciousness saturated (R→0.999) — resolved in v0.1.1
- V2/V3/V4 cortex were empty stubs — implemented in v0.1.1
