# SubstanceNet v4 — Project Goal

**Author:** Oleksii Onasenko
**Date:** March 2026
**Status:** Active development

---

## Mission

SubstanceNet is an experimental bio-inspired neural architecture designed as a computational platform for numerical verification of empirical and theoretical results from neuroscience and cognitive science.

The system integrates findings from multiple scientific domains into a unified modular architecture where each component corresponds to a specific biological structure or computational principle. The purpose is not to compete with conventional neural networks on benchmark accuracy, but to provide a testbed where neuroscientific hypotheses can be implemented, tested, and compared with biological data.

---

## What We Verify

### Empirically established facts

These are results confirmed through decades of experimental research, often recognized with Nobel Prizes:

| Fact | Source | SubstanceNet module | Status |
|------|--------|-------------------|--------|
| Hierarchical visual cortex organization V1→V2→V3→V4 | Hubel & Wiesel (1962, 1968); Nobel 1981 | BiologicalV1, MosaicField18, DynamicFormV3, ObjectFeaturesV4 | Implemented |
| V1 orientation selectivity via Gabor-like receptive fields | Hubel & Wiesel (1962) | GaborFilterBank (fixed, no training) | Verified |
| V2 parallel processing streams (motion/texture/form) | Hubel & Livingstone (1987) | MosaicField18 thick/thin/pale stripes | Verified |
| Place cells and grid cells in hippocampus | O'Keefe (1971); Moser & Moser (2005); Nobel 2014 | PlaceCells, GridCells | Implemented |
| Time cells encoding temporal context | MacDonald et al. (2011); Eichenbaum (2014) | TimeCells | Implemented |
| Episodic vs semantic memory distinction | Tulving (1972); Scoville & Milner (1957) | EpisodicMemory, MemoryConsolidation | Implemented |
| Spike-timing-dependent plasticity (STDP) | Bi & Poo (2001) | HebbianLinear (phase-coherence variant) | Verified |
| Hebbian learning with Oja normalization | Hebb (1949); Oja (1982) | HebbianLinear | Verified |
| Optimal performance at moderate arousal (inverted U-curve) | Yerkes & Dodson (1908) | R-targeting, TemporalController | Verified |
| Retinal photoreceptor types (rods + 3 cone types) | CIE cone fundamentals | RetinalLayer | Implemented |
| Memory consolidation through replay during sleep | Wilson & McNaughton (1994) | consolidate_memory() | Verified |

### Theoretical hypotheses under verification

These are influential hypotheses from neuroscience and cognitive science that have substantial supporting evidence but remain active areas of research:

| Hypothesis | Source | SubstanceNet module | Verification status |
|------------|--------|-------------------|-------------------|
| Critical brain hypothesis: neural systems operate near phase transition | Beggs & Plenz (2003); Shew & Plenz (2013) | TemporalController, R-targeting | Partial: κ-plateau observed across tasks |
| Global Neuronal Workspace: consciousness as recurrent amplification and global broadcast | Dehaene et al. (2001); Mashour et al. (2020) | ReflexiveConsciousness (fixed-point iteration) | Partial: self-monitoring loop implemented |
| Complementary Learning Systems: fast hippocampal + slow neocortical learning | McClelland et al. (1995); Kumaran et al. (2016) | Hippocampus + Hebbian V3/V4 | Partial: 10 prototypes > 200 raw episodes |
| Quantum-like models of cognition: wave formalism for interference effects | Busemeyer & Bruza (2012); Khrennikov (2010) | QuantumWaveFunction, V3 phase interference | Partial: velocity tuning curve emerged |
| Neural oscillation phase-amplitude coupling | Buzsáki (2006); Canolty & Knight (2010) | ψ = A·e^(iφ) representation | Implemented, under investigation |
| Emergence parameter κ ≈ 1 at criticality | Onasenko (2025); DOI: 10.5281/zenodo.17844282 | R ≈ 0.41 operating point | Partial: stable κ-plateau, needs ablation |
| Innate vs experience-dependent visual processing | Hubel & Wiesel (1968) | V1/V2 fixed, V3/V4 Hebbian | Verified: maturation on irrelevant data hurts |
| Biological implausibility of backpropagation | Crick (1989) | HebbianLinear replacing nn.Linear in V3/V4 | Partial: 56.1% recognition without backprop |

---

## What We Have Demonstrated

Numerical experiments on the SubstanceNet platform have produced the following results:

**Visual processing:**
- V1+V2 innate features achieve 53.7% MNIST recognition without any training, confirming that basic visual processing is genetically programmed (Hubel & Wiesel, 1962)
- V3 phase interference produces velocity tuning curves matching primate MT/V3 electrophysiology (Maunsell & Van Essen, 1983), emerging from wave mathematics without parameter tuning
- V4 with random weights degrades recognition; V4 after Hebbian maturation on relevant stimuli improves it — matching biological sensitive periods

**Memory and learning:**
- 10 consolidated prototypes outperform 200 raw episodes (46.3% vs 44.7%) with 20× compression, consistent with CLS theory predictions
- Hebbian plasticity produces 11× motion signal amplification after 50 unsupervised observation steps
- Maturation on irrelevant stimuli (geometric primitives) degrades MNIST recognition; maturation on relevant stimuli (digits) improves it — biological domain-specificity

**Self-monitoring:**
- Reflexivity R ≈ 0.41 maintains stable κ-plateau across all cognitive task types
- R → 1.0 (saturation, pre-fix state) corresponds to representation collapse — computational analogue of excessive neural synchronization
- Inverted U-curve: optimal accuracy at moderate R, degraded at extremes — consistent with Yerkes-Dodson law

**Cross-modal perception:**
- System recognizes objects across modalities: static → moving (36.0%), moving → static (34.3%)
- V3 detects motion on real MNIST digits placed on moving canvas
- Consciousness (R) remains stable across static, video, and cognitive modalities

---

## What We Do Not Claim

- SubstanceNet is not a model of phenomenal consciousness or subjective experience
- The wave formalism ψ = A·e^(iφ) is a mathematical tool, not a claim about quantum processes in the brain
- Results on MNIST/CIFAR-10 are not intended to compete with state-of-the-art classifiers
- The emergence parameter κ ≈ 1 in SubstanceNet is an analogy with physical criticality, not a direct measurement of the physical quantity
- Hebbian plasticity in V3/V4 is demonstrated at small scale; scalability to complex tasks is an open question

---

## Known Boundaries

| Domain | Current performance | Limitation |
|--------|-------------------|------------|
| MNIST (backprop, 1 epoch) | 95.94% | Competitive |
| MNIST (recognition, 100-shot) | 71.9% | No backprop, proof of concept |
| MNIST (innate V1+V2 only) | 53.7% | No training at all |
| CIFAR-10 (recognition, matured) | 24.4% | Needs higher abstractions, saccadic scanning |
| Moving MNIST (speed=2.0) | 36.0% | Partial motion invariance |
| Cross-modal (moving→static) | 34.3% | Feature space mismatch |

---

## Open Research Questions

1. **Does κ ≈ 1 emerge naturally?** Current R-targeting stabilizes R ≈ 0.41 via loss function. Experiment needed: remove R-targeting and verify whether the architecture naturally converges to the critical regime.

2. **Is wave formalism more than notation?** Ablation study needed: replace ψ = A·e^(iφ) with plain real-valued vectors and compare. If velocity tuning curve disappears — formalism has predictive power beyond notation.

3. **Can Hebbian learning scale?** Currently demonstrated on MNIST (28×28 grayscale). Testing on CIFAR-10, natural images, and more complex visual tasks will determine the practical limits of backprop-free learning.

4. **Active perception (saccades).** Current V1 processes entire image at once. Biological vision uses sequential fixations with foveal attention. This is a fundamental architectural extension for complex image recognition.

5. **Biological validation depth.** Does V1 reproduce surround suppression, size tuning, contrast invariance? Quantitative comparison with electrophysiology data would strengthen biological validity claims.

---

## References

1. Beggs, J. M., & Plenz, D. (2003). J. Neurosci. 23:11167–11177.
2. Bi, G., & Poo, M. (2001). Annu. Rev. Neurosci. 24:139–166.
3. Busemeyer, J. R., & Bruza, P. D. (2012). Cambridge University Press.
4. Buzsáki, G. (2006). Oxford University Press.
5. Canolty, R. T., & Knight, R. T. (2010). Trends Cogn. Sci. 14:506–515.
6. Crick, F. (1989). Nature 337:129–132.
7. Dehaene, S., & Naccache, L. (2001). Cognition 79:1–37.
8. Eichenbaum, H. (2014). Nature Rev. Neurosci. 15:732–744.
9. Hafting, T., et al. (2005). Nature 436:801–806.
10. Hebb, D. O. (1949). Wiley.
11. Hubel, D. H., & Wiesel, T. N. (1962). J. Physiol. 160:106–154.
12. Hubel, D. H., & Wiesel, T. N. (1968). J. Physiol. 195:215–243.
13. Hubel, D. H., & Livingstone, M. S. (1987). J. Neurosci. 7:3378–3415.
14. Khrennikov, A. (2010). Springer.
15. Kumaran, D., et al. (2016). Trends Cogn. Sci. 20:512–534.
16. MacDonald, C. J., et al. (2011). Neuron 71:737–749.
17. Mashour, G. A., et al. (2020). Neuron 105:776–798.
18. Maunsell, J. H. R., & Van Essen, D. C. (1983). J. Neurophysiol. 49:1127–1147.
19. McClelland, J. L., et al. (1995). Psychol. Rev. 102:419–457.
20. O'Keefe, J., & Dostrovsky, J. (1971). Brain Res. 34:171–175.
21. Oja, E. (1982). J. Math. Biol. 15:267–273.
22. Onasenko, O. (2025). Zenodo. DOI: 10.5281/zenodo.17844282.
23. Scoville, W. B., & Milner, B. (1957). J. Neurol. Neurosurg. Psychiatry 20:11–21.
24. Shew, W. L., & Plenz, D. (2013). The Neuroscientist 19:88–100.
25. Tulving, E. (1972). In: Organization of Memory, pp. 381–403.
26. Wilson, M. A., & McNaughton, B. L. (1994). Science 265:676–679.
27. Yerkes, R. M., & Dodson, J. D. (1908). J. Comp. Neurol. Psychol. 18:459–482.
