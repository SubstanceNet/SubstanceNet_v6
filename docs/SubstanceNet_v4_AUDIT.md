# SubstanceNet v4 — Технічний аудит

**Автор аудиту:** Claude (Anthropic)
**Замовник:** Олексій Онасенко
**Дата:** 2026-03-15
**Версія документу:** 1.1 (оновлено: MosaicField18 = V2 Hubel)

---

## 1. Мета документу

Зафіксувати повний стан проєкту SubstanceNet v4, ідентифікувати проблеми, що виникли при переході з v3.1.1, та створити план їх виправлення. Цей документ є єдиним джерелом істини для подальшої роботи.

---

## 2. Контекст проєкту

SubstanceNet — обчислювальна реалізація теорії 2D-Субстанції (Онасенко, 2025–2026). Нейромережа моделює рефлексивну свідомість за Теоремою 6.22: **ψ_C = F[P̂[ψ_C]]**. Система не є класичною CNN — це архітектура свідомості, де ключовий показник ефективності — не лише accuracy, а поведінка рефлексивного модуля (R-score) і його зв'язок з параметром емерджентності κ ≈ 1.

Емпіричний результат v3.1.1: оптимальна рефлексивність R ∈ [0.35, 0.47] відповідає найвищій accuracy. Надмірна рефлексивність (R → 1.0) означає сатурацію — тривіальну фіксовану точку, де свідомість не виконує корисної роботи.

---

## 3. Історія версій

### 3.1 SubstanceNet v3 (липень 2025)

Початкова робоча версія. Монолітний файл `models_v2.py` з класом `HubelInspired2DSubstanceNet`. Один src-каталог з усіма модулями. Когнітивні задачі: 10 типів, accuracy ~99.34%, R ~0.956. Включає environment_v2.py (Gymnasium), meta_agent_final.py (PPO), hippocampus_module_v2.py.

### 3.2 SubstanceNet v3.1.1 (серпень 2025)

Стабільна версія з додатковими дослідженнями свідомості. Додані: adaptive_consciousness.py (кілька варіантів), biological_v1_encoder.py, ewc_regularizer.py, MNIST-тести з різними режимами свідомості. Ключовий результат: MNIST 93.74% за 1 епоху, R = 0.382 (stream mode).

### 3.3 SubstanceNet v3.2 (дата невідома)

Проміжна версія, деталі не досліджені.

### 3.4 SubstanceNet v4 (лютий 2026)

Спроба "чистого аркуша": модуляризація коду, підготовка до GitHub. Код розбитий на модулі (cortex/, wave/, consciousness/, model/, hippocampus/). Додано біологічний V1 з Gabor-фільтрами. Діагностовано критичну проблему: R → 1.0 (сатурація), consciousness не працює.

---

## 4. Архітектура v4 — поточний стан

### 4.1 Структура каталогів

```
SubstanceNet_v4/
├── src/
│   ├── constants.py              — єдине джерело констант
│   ├── wave/quantum_wave.py      — ψ = A·exp(iφ)
│   ├── cortex/
│   │   ├── v1.py                 — Gabor → Simple → Complex → Hyper [ПРАЦЮЄ]
│   │   ├── v2.py                 — TODO заглушка
│   │   ├── v3.py                 — TODO заглушка
│   │   └── v4.py                 — TODO заглушка
│   ├── consciousness/
│   │   ├── reflexive.py          — ReflexiveConsciousness [САТУРАЦІЯ]
│   │   └── controller.py         — TemporalConsciousnessController [НЕ ІНТЕГРОВАНИЙ]
│   ├── hippocampus/
│   │   ├── cells.py              — Grid/Place/Time cells [НЕ ПІДКЛЮЧЕНИЙ]
│   │   ├── episodic_memory.py    — Encoder/Retrieval/Consolidation [НЕ ПІДКЛЮЧЕНИЙ]
│   │   └── hippocampus.py        — Головний модуль [НЕ ПІДКЛЮЧЕНИЙ]
│   ├── model/
│   │   ├── substance_net.py      — SubstanceNet (assembler)
│   │   └── layers.py             — OrientationSelectivity, NonLocal, Abstraction, losses
│   └── visualization/dynamics.py
├── research/
│   ├── mnist_v4_baseline/        — MNIST результати + діагностика
│   └── cognitive_tasks/          — Когнітивні задачі (10 задач)
├── tests/                        — 6 тест-файлів
├── docs/ARCHITECTURE.md
└── data/MNIST/
```

### 4.2 Потік даних (image path)

```
Image [B, 1, 28, 28]
  → BiologicalV1 (Gabor→Simple→Complex→Hyper→AdaptiveAvgPool(3,3))
  → sequence [B, 9, 64]
  → OrientationSelectivity (Conv1d groups, ×8 orientations)
  → oriented [B, 9, 512]
  → QuantumWaveFunction (amplitude_fc + phase_fc)
  → amplitude [B, 9, 64], phase [B, 9, 64]
  → cat → features [B, 9, 128]
  → NonLocalInteraction (MultiheadAttention + gate)
  → features [B, 9, 128]
  ├── → AbstractionLayer (mean→Linear→ReLU→Linear) → abstract [B, 3]
  │     → ReflexiveConsciousness (3 ітерації) → psi_c [ТУПИКОВА ГІЛКА]
  └── → Classifier (reshape→Linear(1152,256)→ReLU→Linear(256,10)) → logits
```

### 4.3 Потік даних (cognitive path)

```
Flat tensor [B, *]
  → pad/truncate до 64
  → cognitive_input (Linear(64, 576) + ReLU)
  → reshape [B, 9, 64]
  → (далі як image path від OrientationSelectivity)
```

---

## 5. Порівняння v3.1.1 ↔ v4

### 5.1 Ключові числові відмінності

| Параметр | v3.1.1 | v4 | Вплив |
|---|---|---|---|
| grid_size (seq_len) | 256 | 9 | Класифікатор: 16384 vs 1152 входів |
| dtype | float64 | float32 | Менша числова точність |
| abstract_dim | 3 | 3 (default) | Однаково |
| consciousness_dim | 32 | 32 | Однаково |
| hidden_dim | 64 | 64 | Однаково |
| wave_channels | (implicit) | 128 (64+64) | Більше |
| Вхідна обробка | Transformer encoder | BiologicalV1 (Gabor) | Принципово інший шлях |
| Між nonlocal та abstraction | MosaicField18 (V2) + coherence_fc + stability_fc | Нічого | ❌ Втрачена зорова ієрархія V2 |
| Classifier input | hidden_dim × grid_size = 16384 | feature_dim × seq_len = 1152 | 14× менше |

### 5.2 Компоненти: що є, що втрачено

| Компонент | v3.1.1 | v4 | Статус |
|---|---|---|---|
| Когнітивні задачі (10 типів) | utils.py | Імпортуються з v3.1.1 | ⚠️ Залежність |
| Transformer encoder | Так | Ні (замінено на V1) | Замінено |
| MosaicField18 (V2 cortex) | Так — ключовий модуль | Ні | ❌ КРИТИЧНО ВТРАЧЕНО |
| coherence_fc | Так | Ні | ❌ Втрачено |
| stability_fc | Так | Ні | ❌ Втрачено |
| BiologicalV1 (Gabor) | Окремий файл, не інтегрований | Інтегрований | ✅ Нове |
| OrientationSelectivity | Так | Так | ✅ Портовано |
| QuantumWaveFunction | Так | Так | ✅ Портовано |
| NonLocal | TransformerNonLocalInteraction | MultiheadAttention | ≈ Аналог |
| AbstractionLayer | Так | Так (ідентичний код) | ✅ Портовано |
| ReflexiveConsciousness | Так (V2) | Так (ідентичний код) | ✅ Портовано, але САТУРУЄ |
| abstract_classifier | Ні (abstract_loss через target) | Так (доданий) | ✅ Нове |
| TemporalController | adaptive_consciousness_v2.py | controller.py | ⚠️ Не інтегрований |
| Hippocampus | hippocampus_module_v2.py | hippocampus/*.py | ⚠️ Не підключений |
| Environment (Gymnasium) | environment_v2.py | Відсутній | ❌ Не портований |
| Meta-agent (PPO) | meta_agent_final.py | Відсутній | ❌ Не портований |
| PositionalEncoding | Так | Ні | ❌ Втрачено |
| simple_fc (grid projection) | Так (64 → 64×256) | Ні (V1 замість цього) | Замінено |

### 5.3 MosaicField18 — реалізація зорової кори V2 (Hubel & Wiesel)

**MosaicField18 — це не допоміжний шар, а реалізація зони V2 зорової кори** за дослідженнями Девіда Хьюбела та Торстена Візеля. Без нього архітектура принципово неповна.

V2 складається з трьох типів смуг (stripes), кожна з яких виконує окрему обробку:

```python
# v3.1.1: src/models_v2.py, рядок 62
class MosaicField18(nn.Module):
    thick_stripes  = Linear(in, out//3)   # Різниці: roll(x,1) - x → детекція руху/змін
    thin_stripes   = Linear(in, out//3)   # FFT.real через ReLU → частотний аналіз (текстури)
    pale_stripes   = Linear(in, out-2*(out//3))  # Пряме пропускання → форма/контури
    # forward: cat([thick, thin, pale], dim=2)
```

**Чому це критично для consciousness:**

Три паралельних шляхи обробки (рух + частоти + форма) створюють репрезентації, які не можуть тривіально колапсувати. Навіть однаковий вхід дає різні виходи через FFT та різниці. Після MosaicField18 дані проходять через coherence_fc (стиснення) і потім в AbstractionLayer. Ця "напруга" між різними представленнями забезпечує різноманітність abstract, що запобігає сатурації consciousness.

**Повний ланцюг V1→V2→Abstraction в v3.1.1:**

```
NonLocal [B, 256, 128]
  → MosaicField18(128→128)     ← V2: thick/thin/pale stripes
  → coherence_fc(128→64)       ← стиснення з когерентністю
  → AbstractionLayer(64→3)     ← вхід для consciousness
  → stability_fc(64→64)        ← стабілізація для classifier
  → output_fc(64×256→2)        ← класифікація
```

**В v4 цей ланцюг повністю відсутній.** Features йдуть напряму з NonLocal в Abstraction і Classifier. Саме це є головною причиною колапсу abstract та сатурації R → 1.0.

**Відповідність теорії монографії:**

| Зона кори | Реалізація в v3.1.1 | Монографія | v4 |
|---|---|---|---|
| V1 | OrientationSelectivity | P̂₁ — фільтрація | ✅ BiologicalV1 (покращено) |
| V2 | MosaicField18 | P̂₂ — контури/текстури | ❌ ВІДСУТНІЙ |
| V4 | (не реалізовано окремо) | P̂₃ — об'єкти | ❌ заглушка |

---

## 6. Діагностовані проблеми

### 6.1 КРИТИЧНА: Сатурація рефлексивності (R → 1.0)

**Симптом:** R = 0.998–0.999 по всіх 10 когнітивних задачах. В v3.1.1 було R = 0.382 (MNIST, stream mode).

**Кореневі причини:**

1. **Abstract collapse.** Діагностика MNIST показує sample_variance abstract = 2.5e-11. Всі зразки дають однаковий abstract → consciousness отримує ідентичний вхід → тривіально конвергує до фіксованої точки → MSE(psi_c, project(psi_c)) → 0 → R → 1.0.

2. **Нульові градієнти.** Діагностика підтверджує: abstraction mean_grad = 0.0, consciousness.project mean_grad = 0.0, consciousness.evolve mean_grad = 0.0. Ці модулі не навчаються.

3. **Головна причина: відсутність V2 (MosaicField18).** В v3.1.1 ланцюг MosaicField18 → coherence_fc → stability_fc створював три паралельних шляхи обробки (рух/частоти/форма), що робило колапс abstract математично неможливим. Без V2 дані йдуть напряму з NonLocal в Abstraction без достатньої трансформації. Додатково: grid_size 256 vs 9 (більша розмірність), float64 (вища точність), Transformer encoder (багатші представлення).

### 6.2 КРИТИЧНА: Consciousness — тупикова гілка без градієнтів

**Архітектурний факт (однаковий в v3.1.1 і v4):** Класифікатор отримує features ДО abstraction. Consciousness отримує abstract і обробляє його, але результат не повертається в класифікатор. Це відповідає Th 6.22: свідомість спостерігає, а не генерує.

**Проблема v4:** Єдиний шлях градієнтів до consciousness — через consciousness_loss (0.1 × cons_loss) та abstract_loss. Але якщо abstract collapsed, обидва loss стають тривіальними → нульові градієнти.

**В v3.1.1 це працювало** тому що abstract НЕ колапсував (більше вхідних шарів, більша розмірність), і abstract_loss через `abstract_target` з генератора даних (utils.py) створював реальний градієнтний сигнал.

### 6.3 ЗНАЧНА: Ablation показує відключені модулі

Діагностика MNIST (v4):

| Модуль | Accuracy drop при зашумленні | Статус |
|---|---|---|
| V1 | -85% | ✅ Підключений |
| Orientation | -90% | ✅ Підключений |
| NonLocal | -90.5% | ✅ Підключений |
| Wave | 0% | ❌ Відключений |
| Abstraction | 0% | ❌ Відключений |

Wave та Abstraction не впливають на accuracy — класифікатор їх обходить.

### 6.4 ПОМІРНА: Втрачені компоненти

- Environment (Gymnasium) та Meta-agent (PPO) не портовані. Без них неможливе мета-навчання.
- V3/V4 cortex — порожні заглушки (V2 = MosaicField18 — критичний, див. 5.3 та 6.1).
- TemporalConsciousnessController не інтегрований в тренувальний цикл.
- Hippocampus не підключений до SubstanceNet.

### 6.5 ПОМІРНА: Залежність від v3.1.1

Когнітивні задачі в `run_all.py` імпортують `generate_synthetic_cognitive_data` напряму з `/media/ssd2/ai_projects/SubstanceNet_v3.1.1/src/utils.py` через маніпуляцію sys.path. Це крихке і не відтворюване.

---

## 7. Representation quality — цифри

### 7.1 MNIST (v4, діагностика)

| Метрика | abstract | amplitude | phase |
|---|---|---|---|
| Inter-class distance | 0.0009 | 3.284 | 3.817 |
| Intra-class variance | 2.85e-7 | 0.004 | 0.009 |
| Separation ratio | 1.6 | 52.5 | 40.9 |

Abstract практично не розрізняє класи (separation 1.6). Amplitude та phase — добре (52.5 і 40.9).

### 7.2 Когнітивні задачі (v4)

| Task | Accuracy | R | Coherence | Abstract var |
|---|---|---|---|---|
| logic | 1.0000 | 0.9988 | 0.9997 | 27.79 |
| memory | 1.0000 | 0.9990 | 0.9997 | 38.51 |
| categorization | 1.0000 | 0.9990 | 0.9998 | 39.36 |
| analogy | 0.9961 | 0.9988 | 0.9999 | 43.92 |
| spatial | 1.0000 | 0.9989 | 0.9997 | 28.44 |
| raven | 0.9883 | 0.9974 | 0.9996 | 21.01 |
| numerical | 1.0000 | 0.9989 | 0.9999 | 55.88 |
| verbal | 1.0000 | 0.9986 | 0.9998 | 38.85 |
| emotional | 1.0000 | 0.9992 | 0.9999 | 41.88 |
| insight | 1.0000 | 0.9989 | 0.9998 | 29.98 |
| **MEAN** | **0.9984** | **0.9987** | **0.9998** | **36.56** |

Accuracy висока, але R однаково ~0.999 для всіх задач — consciousness не диференціює.

### 7.3 Референс v3.1.1

| Метрика | Значення |
|---|---|
| MNIST accuracy (1 epoch, stream) | 93.74% |
| MNIST reflexivity (stream) | 0.382 |
| MNIST accuracy (1 epoch, balanced) | 93.47% |
| MNIST reflexivity (balanced) | 0.468 |
| Cognitive mean accuracy | 99.34% |
| Cognitive mean reflexivity | 0.9563 |

---

## 8. Що працює коректно у v4

1. **BiologicalV1** — повноцінний Gabor-фільтр банк з Simple/Complex/HyperColumns. Це покращення відносно v3.1.1.
2. **Модульна структура** — чистий, читабельний код з документацією. Готовий для GitHub.
3. **QuantumWaveFunction** — коректний порт з v3.1.1.
4. **ReflexiveConsciousness** — код ідентичний v3.1.1, проблема не в модулі а в його вхідних даних.
5. **Hippocampus** — повноцінний порт з v3.1.1 (grid/place/time cells, episodic memory, consolidation).
6. **TemporalConsciousnessController** — повноцінний порт з v3.1.1.
7. **constants.py** — єдине джерело істини для констант.
8. **research/ структура** — організована з README, data, scripts, figures, reports.
9. **Cognitive input path** — дозволяє тестувати на когнітивних задачах без V1.

---

## 9. План виправлення

### Фаза 1: Відновлення зорової ієрархії V2 та градієнтного потоку (КРИТИЧНА)

**Мета:** Відновити ланцюг V2 (MosaicField18) → coherence → stability, щоб abstract не колапсував і consciousness отримувала реальні градієнти.

**Крок 1.1: Перенести generate_synthetic_cognitive_data в v4.**
Скопіювати відповідні функції з v3.1.1/src/utils.py в v4/src/utils.py. Прибрати sys.path хак.

**Крок 1.2: Портувати MosaicField18 в v4 як модуль V2.**
Створити src/cortex/v2.py на основі MosaicField18 з v3.1.1 (thick/thin/pale stripes). Додати coherence_fc та stability_fc в layers.py або substance_net.py. Інтегрувати між NonLocal та Abstraction у SubstanceNet.forward().

**Крок 1.3: Верифікувати gradient flow.**
Запустити тренування з моніторингом градієнтів на abstraction і consciousness. Підтвердити що abstraction mean_grad > 0 та abstract sample_variance >> 0.

**Крок 1.4: Розглянути збільшення grid_size.**
Поточний 9 (3×3 від V1) може бути замалим. Варіант: збільшити AdaptiveAvgPool до (4,4)=16 або (5,5)=25.

### Фаза 2: Інтеграція Controller (ЗНАЧНА)

**Мета:** TemporalConsciousnessController повинен модулювати R під час тренування.

**Крок 2.1:** Інтегрувати controller в тренувальний цикл (run_all.py або новий train.py).
**Крок 2.2:** Додати виклик controller.update() після кожного batch.
**Крок 2.3:** Використовувати controller.current_level для модуляції consciousness_loss.

### Фаза 3: Верифікація на когнітивних задачах

**Мета:** Підтвердити що R диференціюється між задачами.

**Крок 3.1:** Запустити повну батарею з фіксованими виправленнями.
**Крок 3.2:** Порівняти R-профіль по задачах з v3.1.1 референсом.
**Крок 3.3:** Побудувати криву "R vs accuracy" для підтвердження оптимуму ~0.4.

### Фаза 4: Підключення Hippocampus та інших модулів

**Крок 4.1:** Інтегрувати Hippocampus в SubstanceNet.forward().
**Крок 4.2:** Портувати Environment та Meta-agent.
**Крок 4.3:** Реалізувати V3/V4 cortex (V2 відновлено у Фазі 1) або видалити заглушки.

---

## 10. Відкриті питання

1. **Чи потрібен V1 для когнітивних задач?** Когнітивний шлях обходить V1 — тобто V1 релевантний лише для image tasks. Для когнітивних задач архітектура фактично відрізняється від MNIST/CIFAR.

2. **abstract_dim = 3 vs 16.** В деяких запусках використовувалось abstract_dim=16, що полегшує знаходження фіксованої точки. Потрібно визначити оптимальне значення.

3. **Роль consciousness у класифікації.** За дизайном (Th 6.22) consciousness — спостерігач. Але якщо вона ніяк не впливає на вихід, то оптимізатор не має стимулу тримати R в оптимальному діапазоні. Потрібен механізм зворотного зв'язку.

4. **Метрика успіху.** Що вважати "робочою" consciousness? Варіанти: (а) R в діапазоні [0.35, 0.47]; (б) R диференціюється між задачами; (в) Ablation consciousness впливає на accuracy; (г) κ ≈ 1.

---

## 11. Дерево залежностей файлів

```
substance_net.py
├── cortex/v1.py (BiologicalV1)
│   └── [Gabor filters, SimpleCells, ComplexCells, HyperColumns]
├── wave/quantum_wave.py (QuantumWaveFunction)
├── consciousness/reflexive.py (ReflexiveConsciousness)
├── model/layers.py
│   ├── OrientationSelectivity
│   ├── NonLocalInteraction
│   ├── AbstractionLayer
│   ├── PhaseCoherenceLoss
│   └── TopologicalLoss
└── constants.py

controller.py ← constants.py [НЕ ПІДКЛЮЧЕНИЙ до substance_net.py]

hippocampus.py ← episodic_memory.py ← cells.py [НЕ ПІДКЛЮЧЕНИЙ до substance_net.py]

research/cognitive_tasks/scripts/run_all.py
├── v3.1.1/src/utils.py (generate_synthetic_cognitive_data) [ЗОВНІШНЯ ЗАЛЕЖНІСТЬ]
└── substance_net.py
```

---

## 12. Рекомендована послідовність дій

1. ✅ Аудит (цей документ)
2. → Перенести utils.py з v3.1.1 в v4
3. → **Портувати MosaicField18 як V2 модуль** + coherence_fc + stability_fc
4. → Верифікувати gradient flow (abstraction, consciousness)
5. → Інтегрувати Controller
6. → Повний тест когнітивних задач з порівнянням R-профілів
7. → Прийняти рішення щодо Hippocampus, Environment, Meta-agent
8. → Підготовка до GitHub

---

*Документ створений на основі повного аудиту коду v4 та аналізу старого чату v3 (липень–серпень 2025).*
