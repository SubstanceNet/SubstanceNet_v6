# SubstanceNet v6 — Experiments

6 reproducible experiments validating the bio-inspired architecture.

## Quick Start
```bash
# Run all experiments (~65 seconds on GPU)
python experiments_v6/run_all_experiments.py
```

## Experiments

| # | Script | Description | Methodology |
|---|--------|-------------|-------------|
| 01 | 01_mnist_backprop.py | Supervised MNIST baseline (97.4%, 1 epoch) | [methodology](methodology/01_mnist_backprop_methodology.md) |
| 02 | 02_cognitive_battery.py | 10 cognitive tasks, R = 0.409 ± 0.001 | [methodology](methodology/02_cognitive_battery_methodology.md) |
| 03 | 03_recognition_paradigm.py | 73.2% recognition without backprop | [methodology](methodology/03_recognition_paradigm_methodology.md) |
| 04 | 04_velocity_tuning.py | V3 velocity tuning matches MT/V3 electrophysiology | [methodology](methodology/04_velocity_tuning_methodology.md) |
| 05 | 05_hebbian_maturation.py | Unsupervised Hebbian learning, sensitive period | [methodology](methodology/05_hebbian_maturation_methodology.md) |
| 06 | 06_kappa_analysis.py | κ = 0.993 ± 0.010 (He-II ref: 0.989 ± 0.007) | [methodology](methodology/06_kappa_analysis_methodology.md) |

## Structure

experiments_v6/
├── 01–06_*.py           Experiment scripts
├── config.py            Shared configuration (seed, paths, plot style)
├── run_all_experiments.py
├── methodology/         Academic methodology documents (per experiment)
├── results/             JSON results (reproducible, seed=42)
└── figures/             Publication-quality figures (PNG 300dpi + PDF)

## Reproducibility

All experiments use seed=42, deterministic CUDA, and fixed random states. Results are identical across runs on the same hardware. JSON results and figures are regenerated on each run.
