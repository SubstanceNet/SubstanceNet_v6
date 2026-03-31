# SubstanceNet v4 — План: Відтворюваність, Візуалізація, Демонстрація
## Ціль: 10/10 по кожному напрямку

**Дата:** 2026-03-18
**Принцип:** Кожен результат з наших звітів має відтворюватись однією командою,
візуалізуватись публікаційним графіком, і бути доступним для демонстрації за 30 секунд.

---

## Структура нових файлів

```
SubstanceNet_v4/
├── experiments/                          # Відтворюваність
│   ├── README.md                         # Опис всіх експериментів
│   ├── config.py                         # Спільні налаштування (seeds, paths)
│   ├── 01_mnist_backprop.py              # MNIST з backprop (95.94%)
│   ├── 02_cognitive_battery.py           # 10 когнітивних задач (99.61%)
│   ├── 03_recognition_paradigm.py        # N-shot recognition (71.9%)
│   ├── 04_velocity_tuning.py             # Velocity tuning curve
│   ├── 05_hebbian_maturation.py          # V3/V4 Hebbian дозрівання
│   ├── 06_innate_vs_acquired.py          # V1+V2 vs V3+V4 якість ознак
│   ├── 07_consolidation.py              # 10 prototypes > 200 episodes
│   ├── 08_moving_mnist.py               # Cross-modal recognition
│   └── results/                          # JSON результати кожного запуску
│
├── figures/                              # Візуалізація
│   ├── architecture_diagram.png          # Діаграма архітектури
│   ├── velocity_tuning_curve.png         # V3 швидкісна крива
│   ├── kappa_plateau.png                 # R по когнітивних задачах
│   ├── recognition_scaling.png           # N-shot accuracy curve
│   ├── innate_vs_acquired.png            # Accuracy по стадіях V1→V4
│   ├── hebbian_maturation.png            # V3_diff та weight_norm по кроках
│   ├── consolidation.png                 # Prototypes vs raw episodes
│   └── model_boundaries.png             # Зведена карта меж
│
├── demo/                                 # Демонстрація
│   ├── demo_quick.py                     # 30-секундна демо (створити модель, forward, R)
│   ├── demo_recognition.py              # Впізнавання: побачив→запам'ятав→впізнав
│   ├── demo_velocity.py                 # Velocity tuning curve в реальному часі
│   └── demo_consciousness.py            # κ-plateau: здоровий vs сатурований стан
```

---

## Частина 1: ВІДТВОРЮВАНІСТЬ (10/10)

### 1.0. config.py — єдині налаштування

```python
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = 'experiments/results/'
FIGURES_DIR = 'figures/'
```

Всі експерименти використовують фіксований seed, зберігають результати в JSON,
генерують графіки в figures/.

### 1.1. 01_mnist_backprop.py
**Відтворює:** MNIST 95.94%, R=0.41 (1 epoch)
**Команда:** `python experiments/01_mnist_backprop.py`
**Вихід:**
- `experiments/results/01_mnist_backprop.json` (accuracy, R, coherence, per-class)
- `figures/mnist_training.png` (loss curve, accuracy, R по батчах)
**Час:** ~2 хвилини

### 1.2. 02_cognitive_battery.py
**Відтворює:** 10 задач, 99.61% accuracy, R=0.41, κ-plateau
**Команда:** `python experiments/02_cognitive_battery.py`
**Вихід:**
- `experiments/results/02_cognitive_battery.json`
- `figures/kappa_plateau.png` (R та abstract_var по задачах)
**Час:** ~5 хвилин

### 1.3. 03_recognition_paradigm.py
**Відтворює:** N-shot recognition (5→100 shot), consolidation, innate features
**Команда:** `python experiments/03_recognition_paradigm.py`
**Вихід:**
- `experiments/results/03_recognition.json`
- `figures/recognition_scaling.png` (accuracy vs N-shot, log scale)
- `figures/consolidation.png` (prototypes vs raw)
- `figures/innate_vs_acquired.png` (accuracy по стадіях V1→V4)
**Час:** ~5 хвилин

### 1.4. 04_velocity_tuning.py
**Відтворює:** Velocity tuning curve (0.0→4.0), shape discrimination
**Команда:** `python experiments/04_velocity_tuning.py`
**Вихід:**
- `experiments/results/04_velocity_tuning.json`
- `figures/velocity_tuning_curve.png` (V3_diff vs speed + біологічний референс)
**Час:** ~1 хвилина

### 1.5. 05_hebbian_maturation.py
**Відтворює:** V3/V4 Hebbian 500 кроків, velocity improvement, MNIST maturation
**Команда:** `python experiments/05_hebbian_maturation.py`
**Вихід:**
- `experiments/results/05_hebbian_maturation.json`
- `figures/hebbian_maturation.png` (V3_diff та weight_norm по кроках)
**Час:** ~3 хвилини

### 1.6. 06_innate_vs_acquired.py
**Відтворює:** Порівняння вродженого vs набутого зору (fresh vs matured, per stage)
**Команда:** `python experiments/06_innate_vs_acquired.py`
**Вихід:**
- `experiments/results/06_innate_vs_acquired.json`
- `figures/innate_vs_acquired.png` (grouped bar chart: fresh vs matured per stage)
**Час:** ~3 хвилини

### 1.7. 07_consolidation.py
**Відтворює:** 10 prototypes > 200 raw episodes (CLS verification)
**Команда:** `python experiments/07_consolidation.py`
**Вихід:**
- `experiments/results/07_consolidation.json`
- `figures/consolidation.png` (bar chart + memory compression ratio)
**Час:** ~1 хвилина

### 1.8. 08_moving_mnist.py
**Відтворює:** Moving MNIST recognition, cross-modal, V3 motion detection
**Команда:** `python experiments/08_moving_mnist.py`
**Вихід:**
- `experiments/results/08_moving_mnist.json`
- `figures/model_boundaries.png` (зведена карта меж по всіх модальностях)
**Час:** ~3 хвилини

### Мастер-скрипт
**Команда:** `python experiments/run_all_experiments.py`
- Запускає всі 8 експериментів послідовно
- Генерує всі графіки
- Зберігає зведений звіт `experiments/results/summary.json`
- Виводить підсумкову таблицю в консоль
**Загальний час:** ~25 хвилин

---

## Частина 2: ВІЗУАЛІЗАЦІЯ (10/10)

### Стиль графіків
- Єдиний стиль для всіх: matplotlib з кастомною палітрою
- Шрифт: sans-serif, розмір 12pt для тексту, 14pt для заголовків
- Колірна схема: blues для SubstanceNet, grays для baselines
- Формат: PNG 300dpi + PDF (vector) для публікацій
- Кожен графік має заголовок, підписи осей, легенду, grid

### Перелік графіків (8 штук)

**Fig 1. Architecture Diagram** (`architecture_diagram.png`)
- Блок-схема: Input → V1 → V2 → V3 → V4 → Consciousness → Output
- Кольорове кодування: зелений=вроджене, помаранчевий=набуте, синій=wave
- Бічна гілка: Hippocampus (store/recall/consolidate)
- Формат: SVG → PNG

**Fig 2. Velocity Tuning Curve** (`velocity_tuning_curve.png`)
- X: stimulus speed (0-4), Y: V3 response (norm diff)
- Дві криві: before maturation (dashed) та after maturation (solid)
- Shaded area: біологічний діапазон з MT/V3 electrophysiology
- Inset: raw V3 response formula

**Fig 3. κ-Plateau** (`kappa_plateau.png`)
- 10 когнітивних задач на X, R на Y (bar chart)
- Горизонтальна лінія R=0.41 (target)
- Shaded zone [0.35, 0.47] = critical regime
- Друга вісь Y: abstract variance (показує диференціацію задач)

**Fig 4. Recognition Scaling** (`recognition_scaling.png`)
- X: N examples (log scale), Y: accuracy
- Крива SubstanceNet recognition
- Horizontal lines: random baseline (10%), backprop reference (95.9%)
- Annotation: "71.9% with 1000 examples, no backprop"

**Fig 5. Innate vs Acquired** (`innate_vs_acquired.png`)
- Grouped bar chart: 4 стадії (V1, V2, V3, V4)
- 3 бари на групу: fresh / matured (irrelevant) / matured (relevant)
- Кольори: зелений (innate V1/V2) vs помаранчевий (acquired V3/V4)
- Key finding highlighted: "V4 random hurts, V4 matured helps"

**Fig 6. Hebbian Maturation** (`hebbian_maturation.png`)
- Subplots: (a) V3_diff vs steps, (b) weight_norm vs steps
- Before/after annotations
- Stabilization point marked

**Fig 7. Consolidation** (`consolidation.png`)
- Bar chart: raw episodes (200) vs prototypes (10)
- Y: accuracy, annotated with memory size
- Key message: "20× compression, +1.6% accuracy"

**Fig 8. Model Boundaries** (`model_boundaries.png`)
- Horizontal bar chart: all test conditions
- Color coded by modality (static/dynamic/color/cross-modal)
- Random baseline line
- Backprop references on top

---

## Частина 3: ДЕМОНСТРАЦІЯ (10/10)

### demo_quick.py — "30 секунд до результату"
```
$ python demo/demo_quick.py

SubstanceNet v4 — Quick Demo
=============================
Creating model (1.36M parameters)... done
Forward pass (MNIST-sized input)... done

Results:
  Reflexivity R = 0.4186 (optimal range: 0.35-0.47) ✓
  Phase: critical (κ ≈ 1)
  Output shape: [4, 10]

System status: HEALTHY (κ-plateau)
```

### demo_recognition.py — "побачив → запам'ятав → впізнав"
```
$ python demo/demo_recognition.py

SubstanceNet v4 — Recognition Demo
====================================
Phase 1: Encoding (20 examples per digit)...
Phase 2: Consolidation (200 episodes → 10 prototypes)...
Phase 3: Recognition test (1000 images)...

Results:
  Raw episodes: 44.7% (200 memories)
  Prototypes:   46.3% (10 memories, 20× compression) ✓
  Random:       10.0%

[Figure saved: figures/demo_recognition.png]
```
Графік: grid 10×10 з прикладами — зліва прототип, справа топ-3 впізнані.

### demo_velocity.py — "V3 бачить рух"
```
$ python demo/demo_velocity.py

SubstanceNet v4 — Velocity Tuning Demo
========================================
Testing V3 phase interference on moving primitives...

Speed 0.0: V3_diff = 0.0000 (static — zero response) ✓
Speed 0.5: V3_diff = 0.4824 (micro-motion)
Speed 1.0: V3_diff = 0.8440 (confident signal)
Speed 2.0: V3_diff = 1.1086 (logarithmic phase)
Speed 4.0: V3_diff = 1.1868 (saturation) ✓

Biological match: logarithmic saturation curve
consistent with primate MT/V3 electrophysiology

[Figure saved: figures/demo_velocity.png]
```

### demo_consciousness.py — "здоровий vs хворий стан"
```
$ python demo/demo_consciousness.py

SubstanceNet v4 — Consciousness States Demo
=============================================
Training model in THREE regimes:

1. HEALTHY (R-targeting ON):
   R = 0.4097 | Phase: critical | Accuracy: 95.94% ✓

2. SATURATED (R-targeting OFF, as in pre-fix v4):
   R = 0.9989 | Phase: saturated | Accuracy: 97.2%
   ⚠ High accuracy but consciousness collapsed!

3. κ-plateau across 10 cognitive tasks:
   All tasks in 'critical' phase (R ∈ [0.40, 0.42])

[Figure saved: figures/demo_consciousness.png]
```

---

## Порядок реалізації

### Етап 1: Інфраструктура (1 сесія)
1. Створити `experiments/config.py` (seeds, paths, plot style)
2. Створити `experiments/__init__.py`
3. Створити шаблон plot стилю (єдиний для всіх графіків)

### Етап 2: Ключові експерименти + графіки (2-3 сесії)
4. `04_velocity_tuning.py` + Fig 2 (найсильніший результат — першим)
5. `02_cognitive_battery.py` + Fig 3 (κ-plateau)
6. `03_recognition_paradigm.py` + Fig 4 + Fig 5 + Fig 7
7. `01_mnist_backprop.py`
8. `05_hebbian_maturation.py` + Fig 6
9. `08_moving_mnist.py` + Fig 8

### Етап 3: Демо скрипти (1 сесія)
10. `demo_quick.py`
11. `demo_velocity.py`
12. `demo_recognition.py`
13. `demo_consciousness.py`

### Етап 4: Збірка (1 сесія)
14. `run_all_experiments.py` (мастер-скрипт)
15. `experiments/README.md`
16. Оновити головний README.md з графіками
17. Фінальна перевірка: `python experiments/run_all_experiments.py` від початку до кінця

---

## Критерій готовності

Проєкт готовий до публікації коли:

```bash
git clone https://github.com/SubstanceNet/SubstanceNet_v4.git
cd SubstanceNet_v4
pip install -r requirements.txt
python experiments/run_all_experiments.py
# → 8 JSON files in experiments/results/
# → 8 PNG figures in figures/
# → Summary table in terminal matches README numbers
python demo/demo_quick.py
# → R ∈ [0.35, 0.47], phase = critical
```

Якщо це працює на чистій машині з одним git clone — 10/10.
