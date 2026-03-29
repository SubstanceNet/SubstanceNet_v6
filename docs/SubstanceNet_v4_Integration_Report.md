# SubstanceNet v4 — Звіт: Інтеграція Hippocampus та зорової кори V3/V4

**Дата:** 2026-03-15
**Автор:** Claude (Anthropic) + Олексій Онасенко

---

## Контекст

Після відновлення consciousness (Фаза 1) залишались три неінтегрованих компоненти: Hippocampus (портований але не підключений), V3 cortex (заглушка), V4 cortex (заглушка). Ця робота завершує архітектуру повної зорової ієрархії P̂ = V4 ∘ V3 ∘ V2 ∘ V1 та підключає епізодичну пам'ять.

---

## 1. Зорова кора V3 — DynamicFormV3

**Файл:** `src/cortex/v3.py`

**Біологічна основа:** Зона V3 зорової кори інтегрує інформацію про форму та рух. Нейрони V3 реагують одночасно на орієнтацію та напрямок руху — вони об'єднують шляхи "що" та "де", які розходяться після V2 (Felleman & Van Essen, 1991).

**Реалізація:** Механізм крос-потокового гейтингу (cross-stream gating). V2 створює три потоки (thick=рух, thin=текстура, pale=форма). V3 рекомбінує їх через навчальні вентилі:

```
gate_ij = σ(W_g · [stream_i ‖ stream_j])
V3(x) = Σ(gate_ij · stream_j) + residual
```

Кожен потік модулюється сусіднім: рух інформує форму, текстура модулює контур. Residual gate (навчальний параметр) балансує між новими та вхідними ознаками.

**Параметри:** 27,479

---

## 2. Зорова кора V4 — ObjectFeaturesV4

**Файл:** `src/cortex/v4.py`

**Біологічна основа:** Зона V4 — фінальна стадія зорової проекції P̂. Нейрони V4 реагують на складні форми, кривизну та частини об'єктів. Вони показують інваріантність до позиції та розміру — необхідну для розпізнавання об'єктів (Zeki, 1983; Pasupathy & Connor, 2001).

**Реалізація:** Багатомасштабне пулінг уваги (multi-scale attention pooling) з компресійним bottleneck:

```
attn_k = softmax(W_k · x)           — увага на масштабі k
pooled_k = Σ(attn_k · x)             — зважене пулінг
V4(x) = Compress(cat(pooled_1..k)) + residual
```

Три масштаби уваги дивляться на різні "рівні деталізації" одночасно. Компресія через bottleneck (50% стиснення) примушує мережу виділяти найважливіші ознаки.

**Параметри:** 49,600

---

## 3. Повний ланцюг зорової проекції P̂

```
Input [B, 1, H, W]
  → V1: BiologicalV1 (Gabor → Simple → Complex → Hyper)     P̂₁ — фільтрація
  → OrientationSelectivity (×8 орієнтацій)
  → QuantumWaveFunction (ψ = A·e^(iφ))
  → NonLocalInteraction (V_ij потенціал)
  → V2: MosaicField18 (thick/thin/pale stripes)              P̂₂ — контури/текстури
  → V3: DynamicFormV3 (cross-stream gating)                   P̂₂' — динамічна форма
  → V4: ObjectFeaturesV4 (multi-scale attention)              P̂₃ — об'єкти
  → coherence_fc → stability_fc → Classifier
                 → AbstractionLayer → Consciousness (ψ_C)
```

---

## 4. Інтеграція Hippocampus

**Файли:** `src/hippocampus/` (cells.py, episodic_memory.py, hippocampus.py) — раніше портовані, тепер підключені до SubstanceNet.

**Принцип підключення:** Hippocampus не є частиною forward pass. Він працює паралельно, керований тренувальним циклом через три методи:

| Метод | Призначення | Коли викликати |
|---|---|---|
| `model.store_episode(output, task_type, metrics)` | Зберегти abstract + consciousness як епізод | Після forward + compute_loss |
| `model.recall(output, top_k)` | Знайти схожі епізоди з пам'яті | Перед або після класифікації |
| `model.consolidate_memory()` | Перевести короткострокову → довгострокову пам'ять | Періодично (кожні N епох) |

**Зв'язок з consciousness:** Hippocampus приймає `amplitude_c` (амплітуда ψ_C) як сигнал важливості. Consciousness модулює що запам'ятовувати — епізоди з високою амплітудою свідомості зберігаються як важливіші. Це рефлексивний зв'язок: свідомість спостерігає → оцінює важливість → пам'ять зберігає.

**Компоненти:**

| Модуль | Роль | Аналог |
|---|---|---|
| GridCells | Просторове кодування (гексагональні решітки) | Координати на Σ |
| PlaceCells | Локальне кодування (гаусові поля) | Позиція на Σ |
| TimeCells | Часове кодування (логарифмічні τ) | Темпоральний контекст |
| EpisodicEncoder | Об'єднання контексту | Формування епізоду |
| ConsciousRetrieval | Пошук з модуляцією ψ_C | Усвідомлене згадування |
| MemoryConsolidation | Прототипізація | Довгострокова пам'ять |

**Параметри:** 754,758

---

## 5. Рішення щодо Meta-agent

Meta-agent (PPO) **виключений** з архітектури v4. Обґрунтування:

- Ніде не задіяний ні в v4, ні в v3.1.1 (ізольований зовнішній скрипт)
- Є стандартним RL-алгоритмом без зв'язку з теорією 2D-Субстанції
- Його функцію (утримання R в оптимумі) виконують R-targeting + Controller
- Не відповідає біологічній моделі — це "зовнішній оператор", а не частина мозку

---

## 6. Результати верифікації

### Когнітивні задачі (з повною ієрархією V1→V2→V3→V4)

| Метрика | Без V3/V4 | З V3/V4 |
|---|---|---|
| Mean Accuracy | 0.9992 | **0.9961** |
| Mean R | 0.4093 | **0.4089** |
| Phase | critical | **critical** |
| κ-плато | так | **так** |

### Hippocampus тест

```
Accuracy: 1.0000
R: 0.4092 (оптимум)
Memory: 5 episodes stored, consolidation passed
R in optimal range: True
```

### CIFAR-10 (1 epoch, grayscale)

```
Test accuracy: 0.3636 (random = 0.10, v3.1.1 ref = 0.7423 full training)
R: 0.4098 (оптимум)
```

CIFAR-10 потребує кольорового входу та більше епох. Grayscale втрачає критичну інформацію для цього dataset.

---

## 7. Повна архітектура v4

```
SubstanceNet v4 (1.33M параметрів)
├── V1: BiologicalV1              141,344 params   ✅ P̂₁
├── OrientationSelectivity          1,536 params   ✅
├── QuantumWaveFunction            65,667 params   ✅ ψ = A·e^(iφ)
├── NonLocalInteraction            66,305 params   ✅ V_ij
├── V2: MosaicField18                  — params    ✅ P̂₂ (thick/thin/pale)
├── V3: DynamicFormV3              27,479 params   ✅ P̂₂' (cross-stream)
├── V4: ObjectFeaturesV4           49,600 params   ✅ P̂₃ (multi-scale)
├── coherence_fc + stability_fc         — params   ✅
├── AbstractionLayer                2,179 params   ✅
├── ReflexiveConsciousness          5,539 params   ✅ ψ_C = F[P̂[ψ_C]]
├── TemporalController             (зовнішній)     ✅ κ ≈ 1 фази
├── Hippocampus                   754,758 params   ✅ Епізодична пам'ять
├── Classifier                    148,226 params   ✅
└── abstract_classifier                 8 params   ✅
```

---

## 8. Наступні кроки

1. **Кольоровий V1** — додати RGB/retinal preprocessing (палички + 3 типи колбочок) для повноцінної роботи з кольоровими зображеннями.

2. **Парадигма впізнавання** — переосмислити тренувальний цикл з "train→classify" на "encode→store→recognize" через Hippocampus. Це відповідає біологічній моделі та ідеям Дубовикова про ідентифікацію через тензори остенсивних визначень (порівняння з базою, а не статистичне навчання).

3. **CIFAR-10 повний тест** — після кольорового V1, більше епох тренування.

4. **Підготовка до GitHub** — README, тести, CI.

---

## Змінені файли (ця сесія)

| Файл | Зміна |
|---|---|
| `src/cortex/v2.py` | **НОВИЙ** — MosaicField18 (V2 Hubel) |
| `src/cortex/v3.py` | **ПЕРЕПИСАНИЙ** — DynamicFormV3 |
| `src/cortex/v4.py` | **ПЕРЕПИСАНИЙ** — ObjectFeaturesV4 |
| `src/cortex/__init__.py` | Додано V2, V3, V4 імпорти |
| `src/model/substance_net.py` | V2→V3→V4 chain + Hippocampus + R-penalty |
| `src/consciousness/reflexive.py` | R-targeting (MSE ≈ 1.44) |
| `src/consciousness/controller.py` | Фази перекалібровані для κ ≈ 1 |
| `src/utils.py` | **НОВИЙ** — перенесено з v3.1.1 |
| `research/cognitive_tasks/scripts/run_all.py` | Локальний імпорт + Controller |
| `docs/SubstanceNet_v4_AUDIT.md` | Оновлено (v1.1) |

---

*Архітектура SubstanceNet v4 завершена. Повна зорова ієрархія V1→V2→V3→V4 реалізована. Hippocampus інтегровано. Consciousness працює в режимі κ ≈ 1.*
