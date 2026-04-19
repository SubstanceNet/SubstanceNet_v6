# SubstanceNet: A Bio-Inspired Cognitive Architecture for Numerical Verification of Neuroscience Hypotheses

**Oleksii Onasenko, MD**

Ophthalmologist; independent researcher in complex systems physics and emergence theory  
Research Center for Quantum Medicine "Vidguk", Ministry of Health of Ukraine (Scientific School of Prof. S. P. Sitko). Kyiv, Ukraine

ORCID: 0009-0007-7017-8161 · Email: olexxa62@gmail.com  
Repository: https://github.com/SubstanceNet/SubstanceNet_v6

---

## Abstract

SubstanceNet is a modular bio-inspired cognitive architecture where each module corresponds to a specific brain structure (V1→V4 visual cortex, hippocampus, reflexive consciousness). Unlike standard neural networks optimized for benchmarks, SubstanceNet serves as a computational platform for numerical verification of neuroscience hypotheses. According to the author's prior work, the emergence parameter κ ≈ 1 is a universal signature of criticality in physical and biological systems (Onasenko, 2025; meta-analysis of seven systems, p = 0.518). Here we show that an architecture built on neurobiological principles naturally enters this regime. Results of seven experiments: 73.2% MNIST recognition without network parameter optimization (at matched spatial resolution of 3×3, standard average pooling yields 67.9% — a +5.4% gain from innate V1 features); an emergent velocity tuning curve reproducing primate electrophysiology; Hebbian maturation with a sensitive period effect; emergence parameter κ = 0.993 ± 0.010 (He-II (quantum liquid) reference: 0.989 ± 0.007), confirmed by an independent neuronal avalanche metric (α = −1.498 vs cortical −1.5). All results reproducible in ~65 seconds.

**Keywords:** bio-inspired architecture, cognitive neural network, visual cortex, Hebbian learning, reflexive consciousness, emergence parameter, criticality

---

## 1. What SubstanceNet Is

SubstanceNet belongs to the class of cognitive neural architectures — alongside HTM (Hawkins & Blakeslee, 2004), NEF (Eliasmith & Anderson, 2003), and LIDA (Franklin et al., 2014). Each module corresponds to a specific brain structure; the system integrates Hebbian plasticity, episodic memory, and recurrent self-monitoring. Unlike standard cognitive architectures focused on behavioral modeling, SubstanceNet serves as a computational platform for numerical verification of neuroscience hypotheses — its results are designed for comparison with electrophysiology data and phase transition physics.

One example: the system achieves 73.2% MNIST recognition without network parameter optimization — using only innate visual features and episodic memory. The standard PCA+kNN baseline at the same dimensionality (128D) yields 88.7%, but at a fair comparison with matched spatial resolution (3×3 = 9 positions) — only 67.9%. SubstanceNet achieves 73.2% through the biological paradigm "saw → remembered → recognized," extracting 5.4% more information from the same 9 positions than trivial compression.

This paper explicitly separates empirically established facts (V1→V4 hierarchy, place cells, Hebbian learning) from hypotheses under active verification (κ ≈ 1 criticality, reflexive consciousness, the critical brain).

| Aspect | CNN / Transformer | SubstanceNet |
|--------|-------------------|-------------|
| Learning | End-to-end backprop | Lower layers fixed (innate), upper layers Hebbian |
| Structure | Uniform layers | Each module = specific brain structure |
| Objective | Minimize classification loss | Multi-objective: classification + κ ≈ 1 |
| Memory | Stateless | Hippocampus (grid/place/time cells) |
| Self-monitoring | None | Reflexive consciousness: ψ_C = F[P̂[ψ_C]] |
| Recognition without optimization | ~10% (random chance) | 73.2% via innate features + kNN |

| Aspect | HMAX (Riesenhuber & Poggio, 1999) | HTM | SubstanceNet |
|--------|------|-----|-------------|
| Visual hierarchy | V1→IT (2 layers) | Cortical columns | V1→V4 (4 layers, each = brain area) |
| Learning | No learning / SVM | Spatial pooling | Hebbian + backprop (hybrid) |
| Consciousness | None | None | ψ_C = F[P̂[ψ_C]], R-targeting |
| Episodic memory | None | None | Hippocampus (grid/place/time cells) |
| Critical regime | Not modeled | Not modeled | κ ≈ 1, verified |
| Electrophysiology | Partial (V1) | No | V3 velocity tuning = MT recordings |

---

## 2. Key Concepts

**Criticality (κ ≈ 1).** Many complex systems function best near the transition point between order and chaos. Too much order — the system freezes; too much chaos — it cannot maintain stable behavior. The emergence parameter κ (Onasenko, 2025) measures how close a system is to this point. When κ ≈ 1 — the optimal regime with maximum dynamic range and computational capacity (Shew & Plenz, 2013).

**Reflexivity (R ≈ 0.41).** R measures how well the system can predict its own internal state. R = 1.0 — collapse into a trivial fixed point (like a brain during an epileptic seizure: perfectly synchronized, but non-functional). R = 0 — the system cannot maintain coherence. The optimal value R ≈ 0.41 was discovered empirically: the system performed best when R stabilized in the range [0.35, 0.47]. Subsequent analysis revealed this corresponds to κ ≈ 1 — the same critical regime observed in superfluid helium (a quantum liquid) and biological flocks.

**Hebbian learning.** "Neurons that fire together, wire together" (Hebb, 1949). Instead of backpropagating error signals (for which no known biological mechanism exists — Crick, 1989), connection strengths change based on local activity. Oja normalization (1982) prevents unbounded growth.

**Power-law distributions.** In systems near criticality, events of all sizes follow P(size) ~ size^(−α). Like earthquakes: small ones every day, large ones once a century. Neuronal avalanches in the brain follow this pattern with α ≈ 1.5 (Beggs & Plenz, 2003).

**Fixed-point iteration.** A process where a value is transformed and fed back into the same transformation. If it converges — a fixed point is found. Consciousness in SubstanceNet is modeled exactly this way.

---

## 3. Why This Architecture

Three neuroscience findings and one hypothesis motivate four architectural decisions:

**Finding 1: The visual cortex is a hierarchy, not a single network.**
Hubel and Wiesel (1962, 1968; Nobel Prize 1981) showed that visual processing passes through sequential stages: V1 detects edges and orientations, V2 processes contours and textures, V3 integrates form and motion, V4 extracts object-level features.
→ **Decision:** Four separate modules (V1→V4), each implementing the operations of its biological counterpart. A note on terminology: "innate" throughout this work means "functional without gradient training." V1 contains ~50K trainable parameters (retinal preprocessing, normalization) that remain at random initialization (seed=42) during recognition testing. Gabor filters are the only components with structured initialization.

**Finding 2: The brain does not do backpropagation.**
Crick (1989) argued that error backpropagation is biologically implausible. Upper cortical areas learn through local Hebbian rules (Hebb, 1949) with Oja normalization (1982).
→ **Decision:** V1 and V2 are fixed (innate). V3 and V4 use HebbianLinear for feature learning (weight updates through global phase-coherence gating ⟨cos(φ)⟩, without backprop gradients on these specific layers), alongside auxiliary components trained via backprop (attention, normalization, gates). Hebbian learning is disabled by default; it is activated only in experiment 05 (maturation), where 500 steps of passive observation amplify V3 by 1.6× and improve recognition by +2.4%.

**Finding 3: The brain operates near a critical point.**
Beggs and Plenz (2003) demonstrated neuronal avalanches with power-law distributions. Shew et al. (2009) showed that criticality maximizes the dynamic range of cortical networks. Hengen and Shew (2025) proposed that criticality serves as a homeostatic setpoint of the brain. The Yerkes-Dodson law (1908) describes the same principle from the behavioral side. The parameter κ ≈ 1 (Onasenko, 2025) quantifies this regime.
→ **Decision:** Reflexivity R is targeted to ≈ 0.41 — the empirically discovered center of the optimal range [0.35, 0.47], corresponding to κ ≈ 1.

**Hypothesis 4: Consciousness involves recurrent self-monitoring.**
The Global Neuronal Workspace theory (Dehaene & Naccache, 2001; Mashour et al., 2020) — one of the leading but debated hypotheses of consciousness — describes it as recurrent amplification and global broadcast of information.
→ **Decision:** The ReflexiveConsciousness module implements the fixed-point equation ψ_C = F[P̂[ψ_C]]. In the current version, consciousness acts as a regularizer — constraining V1→V4 weights toward the critical regime through the loss function. The top-down pathway (ψ_C → V4) is architecturally complete; its activation is planned as the next development step.

---

## 4. Data Flow

```
Input [B, C, H, W]
  |
  v
RetinalLayer ---- RGB -> 4 channels: rods + L/M/S cones
  |                Retinal photoreceptor model (Stockman & Sharpe, 2000)
  v
BiologicalV1 ---- Gabor filters -> Simple cells -> Complex cells -> Hypercolumns
  |                V1 orientation selectivity (Hubel & Wiesel, 1962)
  |                INNATE -- fixed filters, no learning
  v
OrientationSelectivity -- Conv1d x8 orientations -> [B, 9, 512]
  v
FeatureProjection ---- Linear(512->128) + ReLU -> [B, 9, 128]
  |                     Amplitude [B, 9, 64] + phase [B, 9, 64]
  v
NonLocalInteraction -- MultiheadAttention + sigmoid gate -> [B, 9, 128]
  |                     Nonlocal potential V_ij model between features
  v
MosaicField18 (V2) -- thick (temporal diff) / thin (FFT) / pale (identity)
  |                     V2 three-stripe architecture (Livingstone & Hubel, 1987)
  |                     INNATE -- without V2, consciousness saturates to R=1.0
  v
DynamicFormV3 (V3) -- Cross-stream gating + phase interference
  |                     Form-motion integration (Felleman & Van Essen, 1991)
  |                     HebbianLinear -- learns through observation
  v
ObjectFeaturesV4 (V4) -- Multi-scale attention + Hebbian compression
  |                        Experience-dependent (Zeki, 1983)
  |
  +--> Classifier -> logits
  +--> AbstractionLayer -> ReflexiveConsciousness (3 iterations)
        -> consciousness_loss (R-targeting -> kappa ~ 1)
```

Total parameters: **1,458,775**.

---

## 5. Key Modules

### 5.1. Visual Cortex

| Module | Brain Area | Innate? | Key Operation |
|--------|-----------|---------|---------------|
| BiologicalV1 | Primary visual cortex | Yes | Gabor filters -> energy model -> hypercolumns |
| MosaicField18 | V2 | Yes | Three parallel streams: motion/texture/form |
| DynamicFormV3 | V3 | Partially | Cross-stream gating + HebbianLinear |
| ObjectFeaturesV4 | V4 | No | Multi-scale attention + Hebbian compression |

The innate/acquired boundary is the key architectural line. Innate V1+V2 achieve 68.4% MNIST recognition without any training (confirming Hubel & Wiesel). V3/V4 require experience — without it, V4 degrades results, modeling the biological sensitive period (Blakemore & Cooper, 1970).

HebbianLinear implements a phase-dependent rule with Oja normalization:
ΔW = η·(⟨cos(φ)⟩·x·y − α·W·y²),
where η is the learning rate, α is the normalization strength, x and y are input/output activations, and ⟨cos(φ)⟩ is the batch-averaged phase coherence (a simplified variant of the pairwise STDP rule of Bi & Poo, 2001, reducing computational complexity from O(n²) to O(n)). The Oja term is decorrelating (analogous to homeostatic scaling; Turrigiano, 2008).

### 5.2. Consciousness

The ReflexiveConsciousness module implements fixed-point iteration:

```
abstract [B, 3] -> iterate 3x {
    P^: amplitude threshold -> phase normalization -> LayerNorm
    F: concatenate(projection, abstraction) -> Linear+Tanh+Linear
    mixing: 0.8*new + 0.2*old
} -> amplitude_c [B, 16], phase_c [B, 16]
```

The number of iterations (3) and the mixing coefficient (0.8/0.2) are empirically chosen constants: 3 iterations ensure convergence across all tested tasks; the 0.8 coefficient balances convergence speed and stability.

R-targeting -- the key mechanism:

```
       chaos                    optimum                  collapse
R = 0 ---------------- R ~ 0.41 ---------------- R = 1.0
       no convergence      k ~ 1     trivial fixed point

R = 1 / (1 + MSE(psi_C, P^[psi_C]))
target_mse = 1/(R_target) - 1 ~ 1.44
```

The brain does not freely choose its operating point — physics constrains it, just as the equation of state constrains superfluid helium.

### 5.3. Hippocampus

Two memory interfaces:

1. **Episodic memory** (dim=3, Tulving, 1972): contextual representations with grid cell (Hafting et al., 2005), place cell (O'Keefe, 1976), and time cell (MacDonald et al., 2011) codes. Consolidation into prototypes (McClelland et al., 1995).

2. **Recognition memory** (dim=128): discriminative features from FeatureProjection (V1 + orientation selectivity + projection; V2/V3/V4 are bypassed for kNN — see §6.1). Recognition via kNN top-5 weighted cosine voting — the "saw → remembered → recognized" paradigm.

**Hippocampus integration status in v6:** the module (~50% of model parameters) is architecturally complete and validated through 8 unit tests. In experiments 1–6, the hippocampus is instantiated but not activated: episodic memory (Path A) is not called during training, and recognition (experiment 3) uses an inline kNN implementation that bypasses the feature_encoder/feature_memory API (Path B). This is a deliberate staged design; full hippocampal activation with spatially-indexed episodic storage is planned for v7.

### 5.4. Loss Function

A multi-objective function where each component models a biological constraint:

| Component | Biological Analogue |
|-----------|-------------------|
| L_classification | What to recognize (cross-entropy) |
| L_abstract | Prevent abstraction collapse -> R saturation |
| 0.1 x L_consciousness | Critical regime (R-targeting + coherence) |
| 0.01 x L_zero | Metabolic budget (~20 W for 86B neurons) |
| L_phase_coherence | Neural population synchronization |
| L_topological | Phase field structural integrity |
| 0.5 x L_R_penalty | Homeostatic monitoring* (R in [0.35, 0.47]) |

*L_R_penalty is a diagnostic metric; computed in torch.no_grad(), it does not contribute to gradients. Effective R-targeting is through L_consciousness (reflexivity_loss term, effective weight 0.03).

The brain does not simply optimize "recognize this object." It simultaneously maintains metabolic balance, regulates excitation/inhibition (GABA/glutamate), preserves spatial coherence, and keeps itself near criticality.

---

## 6. Results

All experiments are reproducible with a single command (~65 seconds on GPU, seed=42). Full methodology is available in the repository (Onasenko, 2025a).

| Experiment | Result | Significance |
|------------|--------|-------------|
| MNIST backprop (1 epoch) | 97.4% | V1->V4 + consciousness reg. |
| Cognitive battery (10 tasks) | R = 0.409 +/- 0.001 | Stable critical regime |
| Recognition (100-shot) | 73.2% +/- 2.1% | Innate features + episodic memory |
| Innate features (V1+V2) | 68.4% | Genetic programmability of vision |
| V3 velocity tuning | peak at 3.0 | Matches MT/V3 electrophysiology |
| Hebbian maturation | 1.6x, +2.4% | Sensitive period |
| kappa ~ 1 | 0.993 +/- 0.010 | He-II: 0.989 +/- 0.007 |
| Neural criticality* | k_Shew = 1.022 | Cortical: alpha = -1.5 |

*Preliminary results from the next version, included as independent confirmation.

### 6.1. Recognition Without Network Parameter Optimization

**Figure 1.** recognition_scaling.png

The model is initialized with random weights; N examples per class pass through V1->V4, and their 128-dimensional features are extracted at the FeatureProjection stage (after V1 orientation selectivity, before V2/V3/V4) and stored in episodic memory. Recognition is performed via kNN top-5 weighted cosine voting. At 100 examples per class: 73.2% +/- 2.1% (5 initializations). The kNN advantage over prototypes grows with N (+19% at 100-shot), consistent with complementary learning systems theory (McClelland et al., 1995).

**Statistical baselines** (identical conditions: 100-shot, kNN top-5 cosine, seed=42):

| Model | Dim | Spatial pos. | Accuracy | Biology |
|-------|-----|-------------|----------|---------|
| Raw Pixels + kNN | 784 | 784 | 89.9% | None |
| PCA + kNN | 128 | 784 (latent) | 88.7% | None |
| RBF Random + kNN | 128 | 784 (latent) | 86.5% | None |
| Fair 3x3 pool + kNN | 9 | 9 | 67.9% | None |
| **SubstanceNet + kNN** | **128** | **9** | **73.2%** | **Yes** |

Fair 3x3 pool denotes adaptive average pooling from 28x28 to 3x3 = 9 pixels, matching SubstanceNet's spatial resolution after V1 hypercolumn pooling.

Standard baselines (PCA: 88.7%, Raw: 89.9%) outperform SubstanceNet on MNIST. This gap (-15.4% vs PCA) is the expected cost of biological constraints: spatial compression from 784 pixels to 9 hypercolumn positions, and genetically fixed Gabor filters instead of data-optimized PCA projections. The key comparison is at matched spatial resolution (3x3 average pooling: 67.9%): SubstanceNet extracts +5.4% more information from the same 9 positions. None of the statistical baselines produce emergent electrophysiology, critical-regime dynamics, or sensitive-period effects.

**Full module ablation** (20-shot, each level cumulative):

| Stage | Dim | Accuracy | Delta from previous |
|-------|-----|----------|---------------------|
| V1 Gabor | 64 | 58.5% | -- |
| + OrientationSelectivity | 512 | 59.3% | +0.8% |
| + FeatureProjection | 128 | 61.9% | +2.6% |
| + NonLocal attention | 128 | 61.8% | -0.1% |
| **+ V2 (3-stream)** | **128** | **68.4%** | **+6.5%** |
| + V3 (Hebbian) | 128 | 68.3% | -0.1% |
| + V4 (Hebbian) | 128 | 62.8% | -5.5% |

V2 provides the largest single contribution (+6.5%), confirming that three-stream processing (Livingstone & Hubel, 1987) is the key innate mechanism. V1 Gabor alone (58.5%) falls below the fair 3x3 baseline (67.9%) — biological feature extraction requires the full innate V1+V2 pipeline, not just orientation filters.

### 6.2. Emergent Velocity Tuning Curve

The V3 module operates through phase interference: I = A^2_form + A^2_motion + 2*A_form*A_motion*cos(dphi). Dynamic primitives at 9 speeds in 6-frame sequences pass through the untrained model.

| Speed (pix/frame) | V3 response (a.u.) | Characteristic |
|-------------------|---------------------|----------------|
| 0.0 | 0.000 | Zero static response |
| 0.5 | 1.245 | Rise |
| 1.0 | 2.012 | Rise |
| 2.0 | 3.634 | Rise |
| **3.0** | **4.643** | **Peak** |
| 4.0 | 4.275 | Decline (-8%) |
| 5.0 | 3.184 | Decline (-31%) |

Three properties of primate MT/V3 neurons (Maunsell & Van Essen, 1983) are reproduced (V3 response is the L2-norm of the difference between outputs for moving and static stimuli, arbitrary units):

1. **Zero static response.** V3(0) = 0.000. Biological MT neurons likewise do not respond to stationary stimuli.
2. **Sublinear growth.** From 0.5 to 3.0. In MT, the sublinear speed dependence enables encoding a wide range — from slow eye movements to fast saccades.
3. **Saturation and decline.** Peak at the optimal speed, decrease at higher values. Biologically: the stimulus shifts more than one receptive field width between frames. In SubstanceNet: destruction of phase coherence between frames.

No parameters were tuned. The curve arises from the mathematics of phase interference: saturation from the bounded cosine term; decline from destructive interference at large displacements.

Additionally: V3 exhibits topological sensitivity (line vs closed shapes: 2.5x), and V4 provides 720x translational invariance.

### 6.3. Hebbian Maturation and the Sensitive Period

500 steps of passive observation — no labels, no loss function, no optimizer — amplify the V3 motion signal by 1.6x, preserving the velocity tuning curve shape. Oja normalization stabilizes weights within ~50 steps, modeling homeostatic synaptic scaling (Turrigiano, 2008).

Sensitive period effect: maturation on relevant stimuli (MNIST) yields +2.4% recognition; on irrelevant stimuli (geometric primitives) — only +0.5%. This reproduces the classic result of Blakemore and Cooper (1970): the type of early visual experience determines cortical selectivity.

### 6.4. Emergence Parameter kappa ~ 1

**Figure 2.** kappa_analysis.png

According to (Onasenko, 2025), kappa ~ 1 is a universal criticality signature, verified across seven physical and biological systems (kappa = 0.997 +/- 0.004, p = 0.518). Here we test this existing theory on a new object — a neural architecture built on biological principles.

Result: kappa = 0.993 +/- 0.010 across 10 cognitive tasks — comparable precision to the He-II lambda-transition (quantum liquid; 0.989 +/- 0.007; Lipa et al., 2003).

The components of kappa are identified through a formal analogy with He-II: tau = topological order parameter, identified as task accuracy; Lambda = phase coherence of psi_C; A = mean amplitude of psi_C. We use this analogy as a heuristic tool: the mathematical behavior of the tau(step) curves exhibits a compensating mechanism that topologically resembles the rho_s-xi relationship in He-II — not as a claim of physical identity between the processes.

During training, kappa remains stable at each checkpoint, although individual components change. Caveat: the normalization Lambda_c = max(Lambda, 1e-4) ~ Lambda per checkpoint makes kappa trivially equal to tau (accuracy). Test 2 therefore does not provide independent confirmation of a compensating mechanism; meaningful kappa variation is captured in Test 1 (across 10 tasks, where A_c = max(A) and Lambda_c = max(Lambda) are normalized across a set with real variation). Independent criticality validation is provided by exp07 (Shew protocol, alpha = -1.498, kappa_Shew = 1.022), which does not depend on internal normalization.

With consciousness frozen (parameters fixed, forward active), accuracy is preserved (over 97% on simple tasks), but phase coherence drops from 0.999 to 0.782.

**Independent confirmation: neural criticality (preliminary results).**

**Figure 3.** neural_criticality.png — *Shew protocol applied to SubstanceNet. Top row: (a) cascade size distribution P(s) ~ s^(-3/2), kappa_Shew = 1.022; (b) controls — trained (alpha = -1.50) vs untrained (-1.45) vs Gaussian null (-2.05); (c) threshold sensitivity — alpha = -1.5 stable at 2sigma. Bottom row: (d) kappa_Shew and alpha across target_mse sweep (E/I pharmacology analogue) — normal regime (1.44) marked with star; (e) accuracy vs kappa_Shew (cf. Shew Fig. 4); (f) Yerkes-Dodson curve — accuracy vs R with critical regime [0.35, 0.47].*

The Shew protocol (Shew et al., 2009) was applied to the V1->V4 cascade: activations exceeding a 2sigma threshold are tracked as "neuronal avalanches," for which the power-law exponent alpha and kappa_Shew are computed — a measure of the empirical distribution's deviation from the theoretical power law (CDF comparison; kappa_Shew = 1 corresponds to ideal criticality).

| Metric | SubstanceNet (R ~ 0.41) | Cortical data (Beggs & Plenz, 2003) |
|--------|------------------------|------------------------------------|
| alpha | **-1.498** | -1.5 |
| kappa_Shew | **1.022** | ~ 1.0 |
| R^2 (power law) | 0.649 | — |

Three controls:
- **Untrained model:** kappa_Shew = 1.000, but alpha = -1.455 and R^2 = 0.475 (weaker power law). Criticality is an architectural property; distribution quality improves after training.
- **Gaussian null:** alpha = -2.047. Criticality is not a threshold artifact.
- **target_mse sweep:** the closest alpha = -1.498 occurs precisely at R ~ 0.41. R-targeting optimizes not only kappa but also power-law quality.

The critical regime is confirmed by three independent metrics: kappa-analysis (0.993 +/- 0.010), kappa_Shew (1.022), and alpha (-1.498 vs biological -1.5).

### 6.5. Supervised Learning and Cognitive Battery

MNIST with 1 epoch of backprop: 97.4%. R = 0.410 throughout training — the critical regime is stable.

Cognitive battery (10 synthetic tasks: logical transitivity, working memory, categorization, analogy, spatial search, Raven matrices, numerical sequences, verbal encoding, emotional valence, insight): R = 0.409 +/- 0.001. This battery was designed to test the stability of R-targeting across different input types, not to evaluate cognitive abilities — all tasks converge to the same operating regime, confirming the robustness of the criticality maintenance mechanism.

---

## 7. Discussion

### 7.1. Consciousness Maintains Coherence, Not Accuracy

Freezing consciousness parameters (requires_grad=False; the module continues to compute psi_C on every forward pass but does not adapt through gradients) does not reduce accuracy on the tested tasks (both variants — with adaptive and with frozen consciousness — achieve over 97% on MNIST). The difference lies in the operating regime: with adaptive consciousness Lambda = 0.999, with frozen consciousness Lambda = 0.782. This is a gradient ablation of consciousness learning, not a structural ablation of the module. The result forms a testable hypothesis: on more complex tasks where coherent representations are critical, the accuracy gap should emerge. Consciousness in this model is not a decision-making algorithm but a homeostatic process maintaining criticality (analogous to GABA/glutamate balance).

### 7.2. Compensating Mechanism: Coincidence or Pattern?

The pattern tau-up x Lambda-down = const was not designed — it was discovered during analysis (Test 1, across 10 tasks). It is formally analogous to He-II, where rho_s-up x xi-down = const due to zeta ~ nu. Possible interpretations: (a) R-targeting creates a constraint analogous to zeta ~ nu, forcing components to compensate; (b) this is a general property of systems near criticality, potentially indicating universality of critical exponents; (c) a partial normalization artifact (max-normalization across the same set of tasks is meaningful but self-consistent; in Test 2 it is a complete artifact — see §6.4). Independent criticality validation is provided by exp07 (Shew protocol, alpha = -1.498), which does not depend on internal normalization. Distinguishing (a) from (b) requires further research with independent determination of critical thresholds.

### 7.3. V4 and the Biology of Visual Development

V4 with random weights degrades results (-5.5%), despite adding computational power. This is not an architectural defect — it is a model of biological development. After Hebbian maturation on relevant stimuli, V4 contributes +2.4%. The analogy: children with congenital cataracts, operated on late in life, have V1 processing but impaired higher-area processing — they "see" but do not "recognize."

### 7.4. R-Targeting: Artificial Forcing or a Model of Constraints?

R-targeting constrains a single component (R ~ 0.41). That this one local constraint produces kappa ~ 1 — a relationship among three independent components — is non-trivial. In biological terms: the brain constrains GABA/glutamate balance and metabolic budget; these local constraints ensure global criticality. R-targeting is a computational model of this mechanism.

---

## 8. Limitations

**Synthetic tasks.** The cognitive battery uses synthetic data. The tasks may not require genuine reasoning. Validation on real benchmarks is a priority.

**Scale.** 1.46M parameters — orders of magnitude smaller than modern networks. Scalability of Hebbian learning is an open question.

**MNIST.** A test bed, not the end goal. Grayscale 28x28 only. Spatial resolution limited to 9 positions (3x3).

**kappa normalization.** A_c and Lambda_c are determined as the maximum of the respective component: across tasks (test 1) or across training checkpoints (test 2). In test 2, the normalization Lambda_c = max(Lambda, 1e-4) ~ Lambda makes kappa trivially equal to tau — test 2 does not provide independent confirmation of kappa stability. In test 1, normalization across 10 tasks is meaningful but still relies on data from the same set. Independent threshold determination is a necessary next step. Independent validation is provided by exp07 (Shew protocol), which does not use internal normalization.

**Recognition feature stage.** The 73.2% recognition result is obtained from features at the FeatureProjection stage (V1 + orientation selectivity + projection), before V2/V3/V4 processing. V2/V3/V4 are active for classification logits but are bypassed for kNN feature extraction. This preserves the strict "innate" interpretation (only Gabor filters have structured initialization), but it also means that the recognition result does not benefit from V2's FFT+difference streams or V3's temporal integration.

**Hippocampus architectural underuse.** ~50% of model parameters belong to the hippocampal module, which is not activated in publication experiments (see §5.3). Full activation is planned for v7.

**Consciousness test.** Test 3 in §7.1 implements a gradient ablation (parameter freezing), not a structural ablation of the consciousness module. The module continues to compute psi_C and contribute cons_loss. Full structural ablation is technically non-trivial due to architectural dependencies and is planned for v7.

**Power-law quality.** R^2 = 0.649 for the fitted power law means 65% of variance explained. The untrained model yields R^2 = 0.475 — the difference is notable but not dramatic.

**Absence of direct comparisons.** No comparison with HMAX, HTM, or other bio-inspired architectures on the same benchmark under the same conditions.

---

## 9. Outlook

Nearest directions: top-down psi_C -> V4 activation; neural criticality validation via the Shew protocol; auditory pathway A1->A3; real cognitive tasks.

Architectural vision:

```
Layer 1 (sensory):      V1->V4 (vision) | A1->A3 (audition) | M1->M2 (speech)
Layer 2 (memory):       Hippocampus (multimodal)
Layer 3 (binding):      Wave resonance between analyzers
Layer 4 (consciousness): psi_C = F[P^[psi_C]] with top-down to all modules
```

Wave formalism is not needed within a single analyzer — individual neurons transmit spikes (vectors). Waves emerge at the ensemble level (EEG). Wave formalism belongs to binding between analyzers, not feature extraction within them.

---

## Acknowledgments

The author pays deep tribute to the memory of Prof. S. P. Sitko (1936-2020), whose pioneering work laid the foundation for integrating physical and informational approaches to complex systems. Thanks to Tetyana Dyakonova for editorial review and Mykola Dubovykov for critical analysis of mathematical formulations. This work was carried out independently, without external funding.

---

## References

[1] SubstanceNet v6 repository. https://github.com/SubstanceNet/SubstanceNet_v6
[2] Onasenko O. (2025) Emergence Parameter kappa ~ 1. Zenodo. DOI: 10.5281/zenodo.17844282
[3] Hubel D.H., Wiesel T.N. (1962) Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. J. Physiol. 160:106-154. DOI: 10.1113/jphysiol.1962.sp006837
[4] Hubel D.H., Wiesel T.N. (1968) Receptive fields and functional architecture of monkey striate cortex. J. Physiol. 195:215-243. DOI: 10.1113/jphysiol.1968.sp008455
[5] Livingstone M.S., Hubel D.H. (1987) Psychophysical evidence for separate channels for the perception of form, color, movement, and depth. J. Neurosci. 7:3416-3468. DOI: 10.1523/JNEUROSCI.07-11-03416.1987
[6] Felleman D.J., Van Essen D.C. (1991) Distributed hierarchical processing in the primate cerebral cortex. Cereb. Cortex 1:1-47. DOI: 10.1093/cercor/1.1.1-a
[7] Zeki S. (1983) Colour coding in the cerebral cortex. Neuroscience 9:741-765. DOI: 10.1016/0306-4522(83)90197-X
[8] Hebb D.O. (1949) The Organization of Behavior. New York: Wiley.
[9] Oja E. (1982) A simplified neuron model as a principal component analyzer. J. Math. Biol. 15:267-273. DOI: 10.1007/BF00275687
[10] Crick F. (1989) The recent excitement about neural networks. Nature 337:129-132. DOI: 10.1038/337129a0
[11] Beggs J.M., Plenz D. (2003) Neuronal avalanches in neocortical circuits. J. Neurosci. 23:11167-11177. DOI: 10.1523/JNEUROSCI.23-35-11167.2003
[12] Shew W.L. et al. (2009) Neuronal avalanches imply maximum dynamic range in cortical networks at criticality. J. Neurosci. 29:15595-15600. DOI: 10.1523/JNEUROSCI.3864-09.2009
[13] Shew W.L., Plenz D. (2013) The functional benefits of criticality in the cortex. The Neuroscientist 19:88-100. DOI: 10.1177/1073858412445487
[14] Hengen K.B., Shew W.L. (2025) Criticality as a unified setpoint of brain function. Neuron 113:2582-2598. DOI: 10.1016/j.neuron.2025.05.020
[15] Dehaene S., Naccache L. (2001) Towards a cognitive neuroscience of consciousness: basic evidence and a workspace framework. Cognition 79:1-37. DOI: 10.1016/S0010-0277(00)00123-2
[16] Mashour G.A., Roelfsema P., Changeux J.-P., Dehaene S. (2020) Conscious processing and the global neuronal workspace hypothesis. Neuron 105:776-798. DOI: 10.1016/j.neuron.2020.01.028
[17] McClelland J.L., McNaughton B.L., O'Reilly R.C. (1995) Why there are complementary learning systems in the hippocampus and neocortex. Psychol. Rev. 102:419-457. DOI: 10.1037/0033-295X.102.3.419
[18] O'Keefe J. (1976) Place units in the hippocampus of the freely moving rat. Exp. Neurol. 51:78-109. DOI: 10.1016/0014-4886(76)90055-8
[19] Hafting T., Fyhn M., Molden S., Moser M.-B., Moser E.I. (2005) Microstructure of a spatial map in the entorhinal cortex. Nature 436:801-806. DOI: 10.1038/nature03721
[20] MacDonald C.J., Lepage K.Q., Eden U.T., Eichenbaum H. (2011) Hippocampal "time cells" bridge the gap in memory for discontiguous events. Neuron 71:737-749. DOI: 10.1016/j.neuron.2011.07.012
[21] Tulving E. (1972) Episodic and semantic memory. In: Tulving E., Donaldson W. (eds.) Organization of Memory. New York: Academic Press, pp. 381-403.
[22] Blakemore C., Cooper G. (1970) Development of the brain depends on the visual environment. Nature 228:477-478. DOI: 10.1038/228477a0
[23] Yerkes R.M., Dodson J.D. (1908) The relation of strength of stimulus to rapidity of habit-formation. J. Comp. Neurol. Psychol. 18:459-482. DOI: 10.1002/cne.920180503
[24] Bi G., Poo M. (2001) Synaptic modification by correlated activity: Hebb's postulate revisited. Annu. Rev. Neurosci. 24:139-166. DOI: 10.1146/annurev.neuro.24.1.139
[25] Turrigiano G.G. (2008) The self-tuning neuron: synaptic scaling of excitatory synapses. Cell 135:422-435. DOI: 10.1016/j.cell.2008.10.008
[26] Maunsell J.H.R., Van Essen D.C. (1983) Functional properties of neurons in middle temporal visual area of the macaque monkey. I. Selectivity for stimulus direction, speed, and orientation. J. Neurophysiol. 49:1127-1147. DOI: 10.1152/jn.1983.49.5.1127
[27] Lipa J.A., Nissen J.A., Stricker D.A., Swanson D.R., Chui T.C.P. (2003) Specific heat of liquid helium in zero gravity very near the lambda point. Phys. Rev. B 68:174518. DOI: 10.1103/PhysRevB.68.174518
[28] Stockman A., Sharpe L.T. (2000) The spectral sensitivities of the middle- and long-wavelength-sensitive cones derived from measurements in observers of known genotype. Vision Res. 40:1711-1737. DOI: 10.1016/S0042-6989(00)00021-3
[29] Hawkins J., Blakeslee S. (2004) On Intelligence. New York: Times Books. ISBN: 978-0-8050-7456-7
[30] Eliasmith C., Anderson C.H. (2003) Neural Engineering: Computation, Representation, and Dynamics in Neurobiological Systems. Cambridge, MA: MIT Press. ISBN: 978-0-262-05071-5
[31] Franklin S., Madl T., D'Mello S., Snaider J. (2014) LIDA: A systems-level architecture for cognition, emotion, and learning. IEEE Trans. Auton. Ment. Dev. 6:19-41. DOI: 10.1109/TAMD.2013.2294399
[32] Riesenhuber M., Poggio T. (1999) Hierarchical models of object recognition in cortex. Nat. Neurosci. 2:1019-1025. DOI: 10.1038/14819
