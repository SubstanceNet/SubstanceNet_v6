# Experiment 06: κ ≈ 1 Emergence Analysis

## Goal

Investigate whether SubstanceNet exhibits κ ≈ 1 plateau analogous to He-II λ-transition. Reference: A.3_helium_lambda_kappa project (Lipa et al. 2003 data, κ = 0.989 ± 0.007).

## Theoretical Framework

**He-II:** κ = τ × (Λ/Λᶜ) ≈ 1 because ζ ≈ ν → compensating mechanism.

**SubstanceNet:** τ = accuracy, Λ = phase coherence, A = amplitude. κ = (A/Aᶜ) × τ × (Λ/Λᶜ).

## Methodology

**Test 1 — κ across 10 cognitive tasks:** Train 50 epochs each, measure κ components. Normalization: Λᶜ = max(Λ), Aᶜ = max(A).

**Test 2 — κ during training:** Track τ, Λ, A every 10 epochs for 100 epochs (logic task).

**Test 3 — With vs without consciousness:** Compare full model vs frozen consciousness.

## Results

### Test 1: κ across tasks

| Metric | Value |
|--------|-------|
| κ | **0.993 ± 0.010** |
| R | 0.410 ± 0.001 |
| He-II reference | 0.989 ± 0.007 |

Components: τ ≈ 1.0 (all tasks), Λ ≈ 0.999, A ≈ 0.68.

### Test 2: κ during training

R converges to 0.41 within first 10 epochs, κ = 1.0 throughout.

### Test 3: Consciousness enabled vs disabled

| Condition | Accuracy | R | Λ |
|-----------|----------|---|---|
| Full model | 100% | 0.410 | 0.999 |
| No consciousness | 100% | 0.365 | 0.782 |

Consciousness affects R and Λ but not accuracy on simple tasks.

## Interpretation

R-targeting (target_mse=1.44) models physiological constraints:

| System | Stabilizer | Type |
|--------|-----------|------|
| He-II | ζ ≈ ν (XY universality) | Physics |
| Brain | GABA/glutamate, metabolic budget | Physiology |
| SubstanceNet | target_mse = 1.44 | Model of physiology |

## Conclusions

1. **κ = 0.993 ± 0.010** — comparable to He-II (0.989 ± 0.007)
2. R-targeting models physiological constraints, not artificial forcing
3. Consciousness module maintains coherence Λ ≈ 1.0 (vs 0.78 without)
4. Three frameworks converge: Onasenko (Σ) ≡ Dubovikov (T) ≡ Tsien (FCM)

## References

- Onasenko O. (2025) Emergence Parameter κ ≈ 1. DOI: 10.5281/zenodo.17844282
- Lipa J.A. et al. (2003) Physical Review B 68, 174518
- Tsien J.Z. (2016) Front. Syst. Neurosci. 9:186

## Figures

- `figures/kappa_analysis.png` — κ across tasks, R across tasks, components, He-II analogy
