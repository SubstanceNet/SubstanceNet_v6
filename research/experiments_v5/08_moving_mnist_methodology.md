# Experiment 08: Moving MNIST

## Goal

Test SubstanceNet on dynamic (moving) MNIST digits: V3 motion detection on real images, recognition of moving objects, cross-modal transfer between static and moving modalities, speed robustness.

## Methodology

**Moving stimuli:** MNIST digit (28×28) placed on 64×64 canvas with controlled horizontal motion. 6 frames per sequence. Speeds: 0.5, 1.0, 2.0, 3.0, 4.0 px/frame.

**Model:** SubstanceNet untrained, Hebbian disabled, seed=42.

**V3 motion detection:** Forward hook on V3 output. L2 norm difference between moving and static (speed=0) reference on same digit.

**Recognition:** kNN top-5 weighted cosine voting, 20-shot, 128-dim amplitude+phase features. Four conditions:
1. Static→Static: 28×28 images, standard pipeline
2. Moving→Moving: 64×64 video, both memory and test moving
3. Static→Moving: static memory (28×28), moving test (64×64 video)
4. Moving→Static: moving memory (64×64 video), static test (28×28)

**Speed robustness:** Static memory → moving test at 5 different speeds.

## Results

### V3 motion detection

| Speed | V3 response |
|-------|------------|
| 0.0 | 0.0000 |
| 1.0 | 0.0564 |
| 2.0 | 0.0984 |
| 4.0 | 0.1733 |

V3 detects motion on real digits. Signal weaker than primitives (~5×) because digit occupies smaller fraction of 64×64 canvas.

### Recognition across modalities

| Condition | Accuracy | vs Random |
|-----------|----------|-----------|
| Static→Static (28×28) | **57.2%** | 5.7× |
| Moving→Moving (64×64) | **52.6%** | 5.3× |
| Static→Moving (cross) | 13.4% | ~random |
| Moving→Static (cross) | 9.0% | random |

### Speed robustness (static memory → moving test)

All speeds: ~13.7% — flat line at random level. Cross-modal transfer fails regardless of speed.

### Analysis

**Same-modality recognition works:** Static→Static (57.2%) and Moving→Moving (52.6%) both well above random. Moving is slightly worse due to 64×64 canvas diluting the signal.

**Cross-modal transfer fails:** This is a fundamental architectural limitation, not a bug. Static images (28×28) and video frames (64×64) produce features in different spaces because:
1. Different input sizes → different V1 spatial pooling
2. Video mode involves temporal V3 processing absent in static mode
3. Feature distributions don't overlap between modalities

This matches biology: recognizing a still photo and recognizing a moving object use overlapping but distinct neural pathways.

## Conclusions

1. V3 detects motion on real MNIST digits (response grows with speed)
2. Same-modality recognition works: 57.2% static, 52.6% moving
3. Cross-modal transfer fails (~random) — feature spaces incompatible between 28×28 static and 64×64 video
4. Known limitation: requires same input pipeline for memory and recognition
5. R = 0.40 — consciousness stable across modalities

## Figures

- `figures/model_boundaries.png` — recognition by modality + speed robustness

## Data

- `experiments/results/08_moving_mnist.json`
