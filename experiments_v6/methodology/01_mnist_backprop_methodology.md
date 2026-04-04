# Experiment 01: MNIST Supervised Training Baseline

**Version:** 0.6.1  
**Date:** 2026-04-04  
**Script:** `experiments_v6/01_mnist_backprop.py`  
**Results:** `experiments_v6/results/01_mnist_backprop.json`  
**Figure:** `experiments_v6/figures/mnist_training.png`

---

## 1. Objective

Establish a supervised training baseline for SubstanceNet v6 on the MNIST handwritten digit benchmark and verify that the reflexive consciousness module maintains the critical operating regime (R ≈ 0.41, κ ≈ 1) throughout the entire training process.

This experiment answers two questions:
1. Does the V1→V4 biological hierarchy with consciousness regularization achieve competitive classification accuracy?
2. Does R remain stable in the optimal range [0.35, 0.47] during learning, or does training destabilize the critical regime?

---

## 2. Method

### 2.1. Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | SubstanceNet v6 | Full V1→V2→V3→V4 pipeline |
| Feature projection | Plain vectors (Linear + ReLU) | Wave formalism removed after ablation (exp09, v5) |
| Hebbian learning | Disabled | Isolates backprop contribution; Hebbian tested in exp05 |
| Consciousness mode | Stream | Highest accuracy in v3.1.1 (inertia=0.99) |
| Random seed | 42 | Fixed for reproducibility |
| Total parameters | 1,458,775 | See ARCHITECTURE.md §7 |

### 2.2. Training Protocol

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (60,000 train / 10,000 test) |
| Epochs | 1 |
| Batch size | 64 |
| Optimizer | Adam (lr=0.001) |
| Loss | Multi-objective (see ARCHITECTURE.md §6) |
| Input mode | Image (`mode='image'`) |
| Monitoring | Metrics logged every 50 batches |

### 2.3. Evaluation

- **Accuracy:** percentage of correctly classified digits on the full test set (10,000 images)
- **Per-class accuracy:** accuracy per digit (0–9) to identify systematic weaknesses
- **Reflexivity R:** mean R across test batches, computed as R = 1/(1 + MSE(ψ_C, P̂[ψ_C]))
- **Consciousness phase:** categorical assessment (subcritical / critical / supercritical / saturated)
- **Reference baseline:** SubstanceNet v3.1.1 (stream mode, 1 epoch): 93.74% accuracy, R = 0.382

### 2.4. Reproducibility

Single command: `python experiments_v6/01_mnist_backprop.py`  
Environment: Python 3.10, PyTorch 2.8.0+cu128, CUDA GPU, seed=42.  
All random seeds fixed (torch, numpy, CUDA deterministic mode).

---

## 3. Results

### 3.1. Summary

| Metric | SubstanceNet v6 | v3.1.1 reference | Δ |
|--------|----------------|-------------------|---|
| Test accuracy | **97.43%** | 93.74% | +3.69% |
| Reflexivity R | 0.4096 | 0.382 | +0.028 |
| Phase | critical | critical | — |

### 3.2. Training Dynamics

Over 938 batches (1 epoch):
- **Loss:** 2.13 → 0.04, monotonic decrease with expected stochastic noise
- **Batch accuracy:** 87.5% → 98.4%, rapid learning within a single epoch
- **Reflexivity R:** remained in [0.406, 0.412] throughout training (range < 0.007)

The consciousness module showed no transient instability during the learning process — R converged to the critical regime within the first 50 batches and maintained it for the remainder of training.

### 3.3. Per-Class Accuracy

| Digit | Accuracy | Digit | Accuracy |
|-------|----------|-------|----------|
| 0 | 98.5% | 5 | 98.1% |
| 1 | 99.3% | 6 | 98.5% |
| 2 | 99.0% | 7 | 96.0% |
| 3 | 98.4% | 8 | 96.4% |
| 4 | 93.3% | 9 | 96.5% |

Weakest classes: digit 4 (93.3%) and digit 7 (96.0%) — consistent with known MNIST difficulty (stylistic variation in handwriting of 4s and 7s).

---

## 4. Discussion

**Accuracy improvement over v3.1.1 (+3.69%).** The gain is attributed to two v6 changes: (a) replacement of the wave formalism with plain vector projection, which removed a computationally trivial transformation (confirmed by ablation in exp09, v5); and (b) addition of MosaicField18 (V2), which provides multi-stream features (thick/thin/pale) that prevent abstract collapse and maintain diversity in representations reaching the consciousness module.

**Stable R throughout training.** The reflexivity R remained within a 0.007 range around the target value of 0.41 during all 938 training batches. This confirms that the R-targeting mechanism (target_mse = 1.44) and TemporalConsciousnessController (stream mode, inertia = 0.99) jointly maintain the critical regime even as classification weights change rapidly. The multi-objective loss balances classification learning against consciousness constraints without destabilizing either.

**Single-epoch protocol.** One epoch is intentionally limited — the goal is not maximum accuracy but verification that the biological pipeline functions correctly. Extended training would improve accuracy further but is not the focus of this experiment.

---

## 5. Conclusions

1. SubstanceNet v6 achieves 97.43% MNIST accuracy in a single epoch — a +3.69% improvement over v3.1.1, attributable to architectural refinements (plain vectors, V2 integration).
2. Reflexivity R remains stable at 0.410 ± 0.003 throughout training, confirming that the critical regime (κ ≈ 1) is robust to gradient-based weight updates.
3. The consciousness module operates as designed: it regularizes the network toward the critical operating point without interfering with classification learning.

---

## References

- Onasenko O. (2025) Emergence Parameter κ ≈ 1. DOI: 10.5281/zenodo.17844282
- LeCun Y. et al. (1998) Gradient-based learning applied to document recognition. Proc. IEEE 86:2278-2324
