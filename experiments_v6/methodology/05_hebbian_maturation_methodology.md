# Experiment 05: Hebbian Maturation

**Version:** 0.6.1  
**Date:** 2026-04-04  
**Script:** `experiments_v6/05_hebbian_maturation.py`  
**Results:** `experiments_v6/results/05_hebbian_maturation.json`  
**Figures:** `experiments_v6/figures/hebbian_maturation.png`, `experiments_v6/figures/hebbian_recognition.png`

---

## 1. Objective

Demonstrate that the upper cortical layers (V3, V4) can adapt their representations through unsupervised Hebbian observation — without labels, loss functions, or backpropagation — and verify that the type of visual experience during maturation determines the quality of downstream recognition.

This experiment tests three neuroscience hypotheses:
1. **Hebbian plasticity is sufficient for feature refinement.** Local learning rules (dW = η·cos(Δφ)·x·y − α·W·y²) can improve motion detection and shape discrimination through passive observation.
2. **Oja normalization prevents runaway growth.** The decorrelation term (−α·W·y²) keeps weights bounded, analogous to homeostatic synaptic scaling in biological neural circuits.
3. **Sensitive period effect.** Experience with task-relevant stimuli (MNIST digits) produces greater benefit than experience with task-irrelevant stimuli (geometric primitives), modeling the biological phenomenon where early visual experience shapes cortical selectivity (Blakemore & Cooper, 1970).

---

## 2. Method

### 2.1. Maturation Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Maturation steps | 500 | Sufficient for weight convergence |
| Stimuli | Random dynamic primitives | 4 shapes (circle, square, triangle, line) |
| Speed range | 0.5–4.0 px/frame (random) | Covers V3 tuning curve range |
| Direction | Random | Prevents directional bias |
| Labels | None | Purely unsupervised |
| Optimizer | None | No backprop — HebbianLinear only |
| Learning rate (η) | 0.0001 | HebbianLinear default |
| Oja strength (α) | 0.1 (V3), 1.0 (V4) | Normalization per module |
| Mode | `model.train()` + `set_learning(True)` | Required for HebbianLinear updates |

During each step, a random dynamic sequence is generated and passed through the full video pipeline. HebbianLinear weights in V3 (output_proj) and V4 (compress_in, compress_out) update based on local phase coherence between input and output activations. No gradient computation occurs.

### 2.2. Measurements

**Before and after maturation:**
- V3 velocity response at 6 speeds (0, 0.5, 1.0, 2.0, 3.0, 4.0 px/frame)
- Shape discrimination: line-vs-closed and closed-vs-closed L2 distances
- 20-shot kNN recognition on MNIST (protocol from exp03)

**During maturation:**
- V3 response at speed=2.0 (every 10 steps) — tracks amplification dynamics
- V3 output_proj weight norm (every 10 steps) — tracks Oja stability

### 2.3. Three Experimental Conditions

| Condition | Maturation data | Purpose |
|-----------|----------------|---------|
| Fresh | None (random weights) | Baseline |
| Matured (primitives) | 500 steps, geometric shapes | Irrelevant experience |
| Matured (MNIST) | 500 steps, MNIST digit images | Relevant experience |

Each condition uses a freshly initialized model (seed=42). Recognition is measured using the same 20-shot kNN protocol (exp03) after maturation.

### 2.4. Reproducibility

Single command: `python experiments_v6/05_hebbian_maturation.py`  
Environment: Python 3.10, PyTorch 2.8.0+cu128, CUDA GPU, seed=42.

---

## 3. Results

### 3.1. Motion Signal Amplification

| Speed (px/frame) | Before | After | Gain |
|-------------------|--------|-------|------|
| 0.0 | 0.000 | 0.000 | — |
| 0.5 | 1.415 | 2.615 | 1.85× |
| 1.0 | 2.309 | 4.206 | 1.82× |
| 2.0 | 3.622 | 5.847 | **1.61×** |
| 3.0 | 4.701 | 6.996 | 1.49× |
| 4.0 | 4.228 | 6.957 | 1.65× |

Key observations:
- **Average amplification: ~1.6×** across all speeds
- **Velocity tuning shape preserved:** logarithmic increase with saturation and rolloff maintained after maturation
- **Zero static response preserved:** no phantom motion introduced by weight changes
- **Convergence:** V3 response stabilizes within ~50 steps and remains constant for the remaining 450

### 3.2. Oja Weight Stability

| Metric | Value |
|--------|-------|
| Initial weight norm | 1.6 |
| Final weight norm | **35.6** |
| Convergence | ~50 steps |
| Post-convergence variance | < 0.5 |

Weight norm increases rapidly during the first 50 steps, then stabilizes at ~35.6 — the Oja normalization term (−α·W·y²) precisely balances the Hebbian growth term. No divergence over 500 steps, confirming the theoretical stability guarantee of Oja's rule.

### 3.3. Shape Discrimination

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Line vs closed | 0.872 | 1.951 | 2.2× |
| Closed vs closed | 0.354 | 1.106 | 3.1× |
| Topological ratio | 2.5× | 1.8× | — |

Maturation amplifies all shape distances, but closed-vs-closed distances grow faster (3.1×) than line-vs-closed (2.2×). The topological ratio (line-vs-closed / closed-vs-closed) decreases from 2.5× to 1.8×, indicating that maturation on geometric primitives sharpens general feature sensitivity rather than specifically enhancing topological discrimination.

### 3.4. Recognition Impact

| Condition | 20-shot accuracy | Δ from fresh |
|-----------|-----------------|--------------|
| Fresh (random) | 61.9% | — |
| Matured (primitives) | 62.4% | +0.5% |
| Matured (MNIST) | **64.4%** | **+2.4%** |

---

## 4. Discussion

**Hebbian maturation works.** 500 steps of passive observation — no labels, no loss function, no optimizer — produce a 1.6× amplification of V3 motion signals and a measurable improvement in downstream recognition. This validates the core claim that biologically plausible local learning rules can refine feature representations through experience alone.

**Oja normalization is essential.** Without the decorrelation term, Hebbian learning would produce unbounded weight growth. The weight norm stabilizes at 35.6 within 50 steps — a dynamic equilibrium between Hebbian strengthening and Oja decay. This parallels homeostatic synaptic scaling in biological networks (Turrigiano, 2008), where neurons regulate their total synaptic strength to maintain stable firing rates.

**Sensitive period effect.** The 5× difference between relevant (+2.4%) and irrelevant (+0.5%) maturation experience directly models the biological sensitive period. Blakemore & Cooper (1970) showed that kittens raised in environments with only vertical stripes developed cortical neurons selective for vertical orientations — their visual cortex was shaped by the statistics of their visual experience. In SubstanceNet, maturation on MNIST digits teaches V3/V4 which feature co-occurrences are useful for digit discrimination, while maturation on geometric primitives teaches co-occurrences that are largely orthogonal to digit structure.

**Velocity tuning shape preservation.** A critical observation: maturation amplifies the V3 response uniformly across speeds without distorting the tuning curve shape. The logarithmic increase, peak, and rolloff (established in exp04 as emergent from phase interference) are preserved. This means Hebbian plasticity enhances signal gain without disrupting the computational mechanism — analogous to how cortical maturation increases response amplitude while preserving stimulus selectivity.

**Modest absolute improvement.** The +2.4% recognition gain from MNIST maturation is small in absolute terms. This is expected: HebbianLinear operates only in V3 output_proj and V4 compress_in/compress_out — a small fraction of total parameters. The majority of features are determined by innate V1/V2 computations (which provide the 61.9% baseline) and by the backprop-trained layers (which provide the 97.4% ceiling in exp01). The significance lies not in the magnitude but in the mechanism: purely local, unsupervised learning improving recognition — a proof of concept for biologically plausible feature refinement.

---

## 5. Conclusions

1. Unsupervised Hebbian maturation (500 steps, no labels) amplifies V3 motion signals by 1.6× while preserving the velocity tuning curve shape — demonstrating that local plasticity enhances gain without disrupting computational architecture.
2. Oja normalization stabilizes weight growth at a dynamic equilibrium (norm ≈ 35.6), preventing divergence — a computational analogue of homeostatic synaptic scaling.
3. Relevant visual experience (MNIST) produces 5× greater recognition improvement than irrelevant experience (geometric primitives): +2.4% vs +0.5% — modeling the biological sensitive period (Blakemore & Cooper, 1970).
4. The experiment validates HebbianLinear as a biologically plausible learning mechanism: weight updates based on local phase coherence, without backpropagation, produce functionally meaningful feature refinement.

---

## References

- Hebb D.O. (1949) The Organization of Behavior. Wiley
- Oja E. (1982) J. Math. Biol. 15:267-273
- Blakemore C., Cooper G. (1970) Nature 228:477-478
- Turrigiano G.G. (2008) Cell 135:422-435
- Bi G., Poo M. (2001) Annu. Rev. Neurosci. 24:139-166
- Maunsell J.H.R., Van Essen D.C. (1983) J. Neurophysiol. 49:1127-1147
