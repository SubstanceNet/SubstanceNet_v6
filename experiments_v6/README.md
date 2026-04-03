# SubstanceNet v4 — Experiments

Reproducible experiment suite. Each script generates JSON results and publication-quality figures.

## Quick Start
```bash
# Run all experiments (~25 min on GPU)
python experiments/run_all_experiments.py

# Run individual experiment
python experiments/01_mnist_backprop.py
```

## Experiments

| # | Script | Key Result | Figures |
|---|--------|------------|---------|
| 01 | `01_mnist_backprop.py` | 98.0% MNIST (1 epoch), R=0.41 | mnist_training |
| 02 | `02_cognitive_battery.py` | 99.8% accuracy, κ-plateau R=0.409±0.001 | kappa_plateau, kappa_convergence |
| 03 | `03_recognition_paradigm.py` | 73.7%±2.0% kNN recognition (no backprop) | recognition_scaling, innate_vs_acquired, consolidation |
| 04 | `04_velocity_tuning.py` | Logarithmic velocity tuning from wave math | velocity_tuning_curve |
| 05 | `05_hebbian_maturation.py` | 8× motion amplification, +5.2% from relevant maturation | hebbian_maturation, hebbian_recognition |
| 08 | `08_moving_mnist.py` | 52.6% moving recognition, cross-modal limits | model_boundaries |

## Output

- `results/` — JSON files with all numerical data
- `methodology/` — Detailed methodology documents per experiment
- `../figures/` — PNG (300dpi) + PDF figures

## Configuration

All experiments use `config.py`:
- Seed: 42 (fixed for reproducibility)
- Device: CUDA if available
- Unified plot style and color scheme
- Helper functions: `set_seed()`, `save_results()`, `save_figure()`, `create_model()`

## Reproducibility

Every result in the README and documentation can be reproduced by running the corresponding script. JSON output includes metadata (timestamp, seed, device, torch version).
