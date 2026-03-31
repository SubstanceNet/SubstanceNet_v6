# SubstanceNet v6

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Bio-inspired neural architecture for numerical verification of neuroscience hypotheses**

SubstanceNet integrates empirically established results from visual neuroscience, memory systems, and criticality theory into a single modular architecture. Each module corresponds to a brain structure; results can be compared with experimental data.

---

## Key Results
```
MNIST (1 epoch backprop):       97.4%    R = 0.410 (critical regime)
Cognitive battery (10 tasks):   99.8%    R = 0.409 ± 0.001
Recognition (100-shot, no BP):  73.2% ± 2.1%  (7.3× random baseline)
Innate features (V1+V2):       68.4%    without any training
Velocity tuning peak:           4.64     logarithmic saturation
Hebbian maturation:             1.6×     +2.4% recognition
κ ≈ 1 emergence:               0.993 ± 0.010  (He-II ref: 0.989 ± 0.007)
```

All results reproducible: `python experiments/run_all_experiments.py` (~80 seconds on GPU)

---

## Architecture
```
Input [B, C, H, W]
  → RetinalLayer (RGB → rods + L/M/S cones)           Retina
  → BiologicalV1 (Gabor → Simple → Complex → Hyper)   V1 [Hubel & Wiesel]
  → OrientationSelectivity (×8 orientations)
  → FeatureProjection (Linear + ReLU)                  Plain vectors
  → NonLocalInteraction (attention)
  → MosaicField18 (thick/thin/pale stripes)            V2 [Livingstone & Hubel]
  → DynamicFormV3 (cross-stream gating + temporal)     V3 [Felleman & Van Essen]
  → ObjectFeaturesV4 (multi-scale + Hebbian)           V4 [Zeki]
  → AbstractionLayer → ReflexiveConsciousness           ψ_C = F[P̂[ψ_C]]
  → Classifier → logits
  + Hippocampus (episodic + feature memory, 128-dim)   [Tulving, O'Keefe]
  + Top-down gate (ψ_C → V4, inactive — v7)           [prefrontal → V4]
```

Three modes: `image` (static), `video` (temporal V3), `cognitive` (bypasses V1).

Parameters: **1.46M**

---

## Theoretical Foundation

### κ ≈ 1 Emergence Parameter

The system operates in a critical regime analogous to He-II λ-transition (superfluid helium):

| System | κ | Mechanism |
|--------|---|-----------|
| He-II (Lipa et al. 2003) | 0.989 ± 0.007 | ζ ≈ ν → τ↑ × Λ↓ = const |
| **SubstanceNet** | **0.993 ± 0.010** | accuracy↑ × coherence↓ = const |

R-targeting models physiological constraints (GABA/glutamate balance, metabolic budget), not artificial forcing — same role as equation of state in helium.

### Three Frameworks Converge

| Author | Framework | Structure |
|--------|-----------|-----------|
| Onasenko (2025-2026) | ψ on manifold Σ | Wave function, V_ij potential |
| Dubovikov (2013, 2026) | T = 2^n configuration space | Hamming metric |
| Tsien (2015-2016) | N = 2^i − 1 neural cliques | FCM theory |

**Identification:** Σ ≡ T ≡ FCM — three independent approaches describe the same structure.

### What This Project Verifies

**Established neuroscience (implemented):**

| Fact | Source | Module |
|------|--------|--------|
| V1→V4 hierarchy | Hubel & Wiesel (1962) | BiologicalV1 → ObjectFeaturesV4 |
| V1 Gabor receptive fields | Hubel & Wiesel (1962) | GaborFilterBank (fixed) |
| V2 parallel streams | Livingstone & Hubel (1987) | MosaicField18 |
| Hebbian plasticity | Hebb (1949), Oja (1982) | HebbianLinear |
| Episodic memory | Tulving (1972) | Hippocampus |

**Hypotheses under investigation:**

| Hypothesis | Source | Status |
|------------|--------|--------|
| Critical brain | Beggs & Plenz (2003) | κ-plateau observed |
| κ ≈ 1 mechanism | Onasenko (2025) | **Confirmed** (exp06) |
| Reflexive consciousness | Onasenko (2026), Th 6.22 | Implemented, regularizer role |

---

## Experiments

| # | Experiment | Key Result |
|---|-----------|------------|
| 01 | MNIST Backpropagation | 97.4%, R = 0.410 stable |
| 02 | Cognitive Battery (10 tasks) | 99.8%, R = 0.409 ± 0.001 |
| 03 | Recognition Paradigm | 73.2% ± 2.1% without backprop |
| 04 | Velocity Tuning | Peak 4.64, log saturation, 720× invariance |
| 05 | Hebbian Maturation | 1.6× amplification, +2.4% recognition |
| 06 | κ ≈ 1 Analysis | 0.993 ± 0.010 (He-II: 0.989 ± 0.007) |

Each experiment has a methodology document in `experiments/methodology/`.

---

## Quick Start

### Install
```bash
git clone https://github.com/SubstanceNet/SubstanceNet.git
cd SubstanceNet
pip install -r requirements.txt
```

### Demo
```bash
python demo/demo_quick.py          # Model health check
python demo/demo_velocity.py       # V3 velocity tuning
python demo/demo_recognition.py    # See → remember → recognize
python demo/demo_consciousness.py  # κ-plateau across tasks
```

### Reproduce all results
```bash
python experiments/run_all_experiments.py
# → 6 experiments, ~80s on GPU
# → JSON results + publication-quality figures
```

---

## Repository Structure
```
SubstanceNet/
├── src/                            Core architecture
│   ├── cortex/                     V1→V4 visual hierarchy + Hebbian
│   ├── consciousness/              Reflexive consciousness + controller
│   ├── hippocampus/                Episodic + feature memory
│   ├── wave/                       (reserved for v7 binding)
│   ├── model/                      SubstanceNet assembler
│   └── data/                       Dynamic primitives generator
├── experiments/                    6 reproducible experiments
│   ├── 01-06_*.py                  Scripts
│   ├── results/                    JSON outputs
│   └── methodology/                6 methodology documents
├── demo/                           4 quick demonstrations
├── figures/                        Publication-quality plots (PNG+PDF)
├── tests/                          38 tests (pytest)
├── docs/                           Architecture documentation
│   ├── ARCHITECTURE.md             Module reference
│   └── references/                 Key papers
└── research/                       Experimental modules (not in pipeline)
    ├── wave_dynamics/              WaveFunctionOnT, ConsciousnessV2
    ├── experiments_v5/             Deprecated experiments
    └── docs_v4_v5/                 Historical documentation
```

---

## References

1. Onasenko O. (2025) *Emergence Parameter κ ≈ 1.* [DOI: 10.5281/zenodo.17844282](https://doi.org/10.5281/zenodo.17844282)
2. Onasenko O. (2026) *Monograph "2D-Substance"*, Chapter 6, Theorem 6.22.
3. Hubel D.H., Wiesel T.N. (1962) *J. Physiol.* 160:106-154.
4. Tsien J.Z. (2016) *Front. Syst. Neurosci.* 9:186.
5. Dubovikov M.M. (2013, 2026) Tensors of Ostensive Definitions.
6. Lipa J.A. et al. (2003) *Phys. Rev. B* 68:174518.
7. Beggs J.M., Plenz D. (2003) *J. Neurosci.* 23:11167-11177.
8. McClelland J.L. et al. (1995) *Psychol. Rev.* 102:419-457.

---

## Citation
```bibtex
@software{onasenko2026substancenet,
  author = {Onasenko, Oleksii},
  title = {SubstanceNet: Bio-inspired Neural Architecture for
           Numerical Verification of Neuroscience Hypotheses},
  year = {2026},
  version = {0.6.0},
  url = {https://github.com/SubstanceNet/SubstanceNet}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Author

**Oleksii Onasenko** — [ORCID: 0009-0007-7017-8161](https://orcid.org/0009-0007-7017-8161)
