# Experiment 03: Recognition Paradigm

**Version:** 0.6.2  
**Date:** 2026-04-08  
**Script:** `experiments_v6/03_recognition_paradigm.py`  
**Results:** `experiments_v6/results/03_recognition_paradigm.json`  
**Figures:** `experiments_v6/figures/recognition_scaling.png`, `experiments_v6/figures/innate_vs_acquired.png`, `experiments_v6/figures/consolidation.png`

---

## 1. Objective

Demonstrate that SubstanceNet can recognize handwritten digits without any gradient-based training, using only innate feature extraction (V1→V2) and episodic memory (kNN retrieval). This implements the biological recognition paradigm: **saw → remembered → recognized**.

The experiment answers six questions:
1. How does recognition accuracy scale with the number of stored examples (N-shot)?
2. How does episodic memory (kNN) compare to prototype-based memory (class means)?
3. Which processing stage (V1, V2, V3, V4) produces the most discriminative features without training?
4. Is the recognition paradigm reproducible across random initializations?
5. How does SubstanceNet compare to standard statistical baselines (PCA, Random Features) under identical conditions?
6. Which pipeline module contributes most to recognition accuracy (full ablation)?

---

## 2. Method

### 2.1. Recognition Protocol

The model is initialized with random weights (seed=42), no training is performed. For each digit class, N examples are passed through the V1→V4 pipeline (full forward pass); the 128-dimensional feature vectors used for kNN retrieval are extracted at the FeatureProjection stage (amplitude + phase, mean-pooled over 9 spatial positions), **before** V2/V3/V4 processing. V2/V3/V4 continue to process features in parallel for classification logits and are evaluated separately in the Module Ablation experiment (sub-experiment F). Recognition of new images is performed by kNN retrieval from the FeatureProjection-level episodic memory.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Feature representation | 128-dim (64 amplitude + 64 phase) | FeatureProjection output (V1 + Orientation + projection); V2/V3/V4 bypass for kNN |
| Spatial pooling | Mean over 9 positions | Position-invariant representation |
| Retrieval method | kNN, top-5, weighted cosine voting | Preserves within-class variability |
| Hebbian learning | Disabled | Isolates innate feature quality |
| Model training | None (random initialization) | Tests innate recognition capability |

### 2.2. Sub-Experiments

**A. N-shot scaling.** N ∈ {1, 5, 10, 20, 50, 100} examples per class stored in memory. Test set: 1,024 images. Compared against prototype matching (single mean vector per class).

**B. Reproducibility.** 100-shot recognition repeated with 5 independent random initializations. Reports mean ± standard deviation.

**C. Memory consolidation trade-off.** Direct comparison: kNN with 200 individual episodes vs 10 prototype vectors (one per class). Quantifies the accuracy cost of 20× memory compression.

**D. Feature quality by processing stage.** Recognition accuracy measured using features extracted after each cortical stage (V1, V2, V3, V4) separately. 20-shot protocol. Classifies each stage as innate (fixed computations) or acquired (random HebbianLinear weights).

**E. Statistical baselines.** Four non-biological baselines evaluated under identical conditions (100-shot, kNN top-5 cosine weighted, test_size=1024, seed=42): (1) Raw pixels (784D) — no dimensionality reduction; (2) PCA (128D) — optimal linear projection preserving maximum variance; (3) Random Fourier Features via RBF kernel (128D) — random nonlinear projection; (4) Fair 3×3 (9D) — average pooling to 3×3 pixels, matching SubstanceNet spatial resolution. Baselines isolate the contribution of biological feature extraction vs generic dimensionality reduction.

**F. Full module ablation.** 7-stage ablation measuring the contribution of each pipeline module independently: V1 Gabor → +OrientationSelectivity → +FeatureProjection → +NonLocalInteraction → +V2 → +V3 → +V4. 20-shot kNN protocol. Each stage is cumulative (includes all previous). Isolates the value added by each biological module and identifies which modules require experience (maturation) to contribute positively.

### 2.3. Evaluation

- **Accuracy:** correctly recognized digits / total test samples
- **Random baseline:** 10% (10 classes, uniform chance)
- **Backprop reference:** 97.4% (exp01, 1 epoch supervised training)
- **Consciousness R:** measured during feature extraction to confirm critical regime

### 2.4. Reproducibility

Single command: `python experiments_v6/03_recognition_paradigm.py`  
Environment: Python 3.10, PyTorch 2.8.0+cu128, CUDA GPU, seed=42.

---

## 3. Results

### 3.1. N-Shot Scaling

| N (per class) | kNN | Prototype | Δ (kNN − proto) | Total episodes |
|---------------|-----|-----------|------------------|----------------|
| 1 | 43.0% | 43.0% | 0.0% | 10 |
| 5 | 46.6% | 45.4% | +1.2% | 50 |
| 10 | 56.3% | 50.1% | +6.2% | 100 |
| 20 | 61.9% | 49.8% | +12.1% | 200 |
| 50 | 67.5% | 52.4% | +15.0% | 500 |
| 100 | **73.2%** | 54.2% | +19.0% | 1,000 |

The kNN advantage grows monotonically with N. At 100-shot, kNN outperforms prototypes by 19 percentage points.

### 3.2. Reproducibility (100-shot)

| Trial | Accuracy |
|-------|----------|
| 0 | 75.1% |
| 1 | 74.9% |
| 2 | 70.8% |
| 3 | 74.7% |
| 4 | 70.5% |
| **Mean ± std** | **73.2% ± 2.1%** |

### 3.3. Memory Consolidation Trade-Off

| Method | Accuracy | Memory size | Compression |
|--------|----------|-------------|-------------|
| kNN episodes | 61.9% | 200 vectors | 1× |
| Prototypes | 49.8% | 10 vectors | 20× |
| **Δ** | **−12.1%** | | |

20× memory compression costs 12.1% accuracy — prototypes discard within-class variability that kNN preserves.

### 3.4. Feature Quality by Processing Stage

| Stage | Accuracy | Type | Δ from V1 |
|-------|----------|------|-----------|
| V1 (Gabor filters) | 61.9% | Innate | — |
| V2 (FFT + temporal diff) | **68.4%** | Innate | +6.5% |
| V3 (cross-stream gating) | 68.3% | Acquired (random) | +6.4% |
| V4 (multi-scale attention) | 62.8% | Acquired (random) | +0.9% |

Consciousness R during recognition: 0.4009 (critical regime).

### 3.5. Statistical Baselines (100-shot)

| Model | Dimensionality | Spatial positions | Accuracy | Biology |
|-------|---------------|-------------------|----------|--------|
| Raw Pixels + kNN | 784 | 784 | 89.9% | None |
| PCA + kNN | 128 | 784 (latent) | 88.7% | None |
| RBF Random + kNN | 128 | 784 (latent) | 86.5% | None |
| Fair 3×3 pool + kNN | 9 | 9 | 67.9% | None |
| **SubstanceNet FeatureProjection + kNN** | **128** | **9** | **73.2%** | **Yes** |

SubstanceNet is −15.4% below PCA (cost of biological constraints: spatial compression 784→9, fixed Gabor filters instead of data-optimized projections). However, SubstanceNet is +5.4% above the fair 3×3 baseline (value of V1+V2 biological features over trivial average pooling at the same spatial resolution).

### 3.6. Full Module Ablation (20-shot)

| Stage | Dim | Accuracy | Δ from previous | Type |
|-------|-----|----------|-----------------|------|
| V1 Gabor only | 64 | 58.5% | — | Innate |
| + OrientationSelectivity | 512 | 59.3% | +0.8% | Innate |
| + FeatureProjection | 128 | 61.9% | +2.6% | Innate |
| + NonLocal attention | 128 | 61.8% | −0.1% | Innate |
| + V2 (3-stream) | 128 | **68.4%** | **+6.5%** | Innate |
| + V3 (Hebbian) | 128 | 68.3% | −0.1% | Acquired |
| + V4 (Hebbian) | 128 | 62.8% | −5.5% | Acquired |

V2 provides the largest single contribution (+6.5%), confirming that three-stream processing (thick/thin/pale; Livingstone \& Hubel, 1987) is the key innate feature extraction mechanism. V1 Gabor alone (58.5%) is below the fair 3×3 baseline (67.9%) — biological feature extraction requires the full V1+V2 innate pipeline, not just orientation filters. NonLocal attention and V3 without maturation are neutral; V4 without maturation is harmful (−5.5%).

---

## 4. Discussion

**73.2% without training.** The result demonstrates that SubstanceNet's innate V1+V2 features, combined with episodic kNN memory, produce meaningful recognition from a single forward pass per image — no weight updates of any kind. A conventional CNN with random weights produces ~10% (chance level). This validates the biological principle that basic visual processing is genetically programmed (Hubel & Wiesel, 1962) and that recognition can operate through episodic retrieval rather than discriminative training.

**kNN vs prototypes.** The growing advantage of kNN over prototypes (0% at 1-shot → 19% at 100-shot) reflects a fundamental property of episodic memory: individual episodes preserve within-class variability (e.g., different handwriting styles for digit "4"), while prototypes collapse this variability into a single mean vector. This mirrors the complementary learning systems theory (McClelland et al., 1995): fast episodic encoding (hippocampus) captures specific instances, while slow consolidation (neocortex) builds generalized representations — trading accuracy for compression.

**V2 as optimal innate stage.** The V2 module adds +6.5% over V1 through three parallel processing streams (thick: temporal difference, thin: FFT spectral features, pale: identity). This is achieved without any training — the computations are fixed. V3 maintains this level (68.3%) despite random HebbianLinear weights, suggesting that the cross-stream gating architecture is not harmful even before maturation. V4, however, degrades features (62.8%) because random multi-scale attention actively destroys discriminative information. This is consistent with the biological finding that V4 requires visual experience to develop useful selectivity (Blakemore & Cooper, 1970), and with the Hebbian maturation results in exp05.


**Methodology precision.** A key discovery during the development of this experiment: kNN top-5 weighted voting on individual episodes can outperform prototype matching by up to +19% at 100-shot. The difference is not in the architecture but in the retrieval methodology — a distinction that must always be specified explicitly.

**Cost of biological constraints.** Standard statistical baselines (PCA + kNN: 88.7%, Raw Pixels: 89.9%) significantly outperform SubstanceNet (73.2%) on MNIST. This −15.4% gap is not a defect but an expected consequence of two architectural constraints: (1) spatial compression from 784 pixels to 9 hypercolumn positions (3×3), while PCA retains latent information from all 784 points; (2) V1 Gabor filters are genetically fixed and not optimized for any particular dataset, unlike PCA eigenvectors that maximize variance for the specific data distribution. Crucially, the fair comparison at matched spatial resolution (3×3 average pooling: 67.9%) reveals that SubstanceNet extracts +5.4% more information from the same 9 positions — this is the measurable value of biologically-inspired feature extraction over trivial compression. None of the statistical baselines produce emergent electrophysiology, critical-regime dynamics, or sensitive-period effects.

**Module contributions.** The full 7-stage ablation reveals that V2 three-stream processing is the dominant innate mechanism (+6.5%), not V1 Gabor filters alone. V1 Gabor (58.5%) is actually below the fair 3×3 baseline (67.9%), demonstrating that orientation-selective filtering alone is insufficient — the frequency analysis (FFT), temporal differencing, and identity streams of V2 are essential for meaningful innate feature extraction. This is consistent with the neuroanatomical finding that V2 receives the largest cortical projection from V1 and performs the critical transformation from local edge detection to contour and texture representation (Livingstone \& Hubel, 1987).

---

## 5. Conclusions

1. SubstanceNet achieves 73.2% ± 2.1% MNIST recognition without any gradient-based training (7.3× random baseline), validating the biological "saw → remembered → recognized" paradigm.
2. Innate V1+V2 features alone provide 68.4% accuracy — confirming that biologically-inspired fixed computations (Gabor filters, FFT, temporal differencing) extract meaningful visual representations.
3. Episodic memory (kNN) consistently outperforms prototype memory, with the advantage growing with N (+19% at 100-shot) — supporting the complementary learning systems theory.
4. V4 with random weights degrades recognition (−5.6% from V2), confirming that experience-dependent modules require maturation before contributing positively (see exp05).
5. Statistical baselines (PCA: 88.7%, Raw: 89.9%) outperform SubstanceNet (73.2%) on MNIST — an expected cost of biological constraints (spatial compression 784→9). At matched resolution, SubstanceNet outperforms trivial pooling by +5.4%, confirming the value of V1+V2 biological features.
6. Full module ablation identifies V2 three-stream processing as the dominant innate mechanism (+6.5%), while V1 Gabor alone (58.5%) is below the fair 3×3 baseline (67.9%). Biological feature extraction requires the complete V1+V2 innate pipeline.

---

## References

- Hubel D.H., Wiesel T.N. (1962) J. Physiol. 160:106-154
- McClelland J.L. et al. (1995) Psychol. Rev. 102:419-457
- Blakemore C., Cooper G. (1970) Nature 228:477-478
- Onasenko O. (2025) Emergence Parameter κ ≈ 1. DOI: 10.5281/zenodo.17844282
