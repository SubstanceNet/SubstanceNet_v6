# SubstanceNet v4 — Зміни до теоретичного фреймворку та план експериментів

**Дата:** 2026-03-18
**На основі:** рецензія колег + зовнішня критична оцінка

---

## Частина 1: Зміни до Theoretical Framework (docx)

### 1.1. Додати підрозділ "Limitations and Open Questions" (перед References)

Зміст:

**a) κ ≈ 1 як атрактор, а не hard constraint.**
Пояснити що R-targeting стабілізує природний атрактор, а не створює штучний. Факт: в v3.1.1 R = 0.382 (stream) та R = 0.468 (balanced) виникли БЕЗ таргетування, суто з архітектурних взаємодій V2 → Abstraction → Consciousness. R-targeting у v4 було додано після втрати MosaicField18 при портуванні, коли abstract колапсував. Після відновлення V2 система природно тяжіє до R ∈ [0.35, 0.47]. Таргетування — стабілізатор існуючого атрактора, а не генератор штучного.

Визнання обмеження: необхідний експеримент з вимкненим R-targeting для демонстрації що κ ≈ 1 виникає природно (див. Частина 2, експ. 1).

**b) Онтологічний статус κ в нейромережі.**
Чесно визнати що κ = (A/Aᶜ)·τ·(Λ/Λᶜ) з мета-аналізу фізичних систем не вимірюється безпосередньо в SubstanceNet. Зв'язок R ↔ κ — це аналогія через критичні експоненти (як в He-II: κ ∝ |t|^(ζ−ν) ≈ const), а не пряме обчислення. R ≈ 0.41 відповідає режиму де рефлексивний модуль не сатурований (κ > 1) і не відключений (κ < 1).

**c) Масштабованість Hebbian learning.**
Визнати що Hebbian plasticity в SubstanceNet продемонстрована на MNIST (простий dataset) і не претендує на заміну backpropagation для задач масштабу ImageNet або language modeling. SubstanceNet пропонує іншу парадигму — розпізнавання через порівняння з епізодичною пам'яттю, а не класифікацію через навчені ваги. Масштабованість цієї парадигми — відкрите дослідницьке питання.

**d) Демаркація від існуючих архітектур.**
Визнати необхідність систематичного ablation study що порівнює wave formalism з еквівалентними real-valued операціями. Попередні дані (velocity tuning curve з wave interference) вказують на нетривіальні емерджентні властивості формалізму, але формальне порівняння ще не проведено.

### 1.2. Уточнити термінологію (мінорні правки по тексту)

- Розділ 3: замінити "Self-Referential Consciousness" → "Self-Referential Processing and Metacognitive Monitoring"
- Розділ 3.3: додати речення що ψ_C = F[P̂[ψ_C]] формально еквівалентна attractor network (Hopfield, 1982), але з ключовою відмінністю — projection P̂ виконує dimensionality reduction (128 → 3), а не pattern completion
- Розділ 8 (CLS): додати що 71.9% на MNIST — proof of concept альтернативної парадигми, не конкуренція з CNN по benchmark accuracy

### 1.3. Покращити Table 1

Переробити з текстового формату на повноцінну Word-таблицю з трьома колонками: Theoretical Framework | Key References | SubstanceNet Module. Зараз таблиця може бути непомітною для читача.

---

## Частина 2: Нотатки для майбутніх експериментів

### Експеримент 1: κ без таргетування (КРИТИЧНИЙ)
**Мета:** Довести що R ≈ 0.41 — природний атрактор, а не артефакт loss function.
**Протокол:**
- Вимкнути consciousness_loss (R-targeting) повністю
- Залишити тільки classification_loss + abstract_loss
- Тренувати на MNIST та когнітивних задачах (100 epochs)
- Відстежувати R по епохах: чи конвергує до [0.35, 0.47] без примусу?
- Порівняти з v3.1.1 де R = 0.382 виникав природно

**Очікування:** R має стабілізуватися в діапазоні 0.30–0.50 завдяки архітектурним обмеженням (V2 MosaicField18 запобігає abstract collapse). Якщо ні — це фальсифікує гіпотезу природного атрактора.

### Експеримент 2: Wave formalism ablation (КРИТИЧНИЙ)
**Мета:** Кількісно показати що wave formalism дає більше ніж cosine similarity.
**Протокол:**
- Модель A (baseline): поточний SubstanceNet з ψ = A·e^(iφ)
- Модель B (ablation): замінити QuantumWaveFunction на два незалежних Linear(dim, dim) без complex-valued інтерпретації. V3 interference замінити на concat + Linear
- Порівняти на:
  - MNIST accuracy (backprop, 1 epoch)
  - Recognition paradigm (20-shot, 100-shot)
  - Velocity tuning curve (dynamic primitives)
  - Shape discrimination (dynamic primitives)

**Ключове питання:** Чи velocity tuning curve виникає в моделі B? Якщо ні — wave formalism has predictive power beyond notation.

### Експеримент 3: Biological validation — V1 response properties
**Мета:** Перевірити чи BiologicalV1 відтворює відомі властивості нейронів V1.
**Тести:**
- Size tuning: відповідь на стимули різного розміру (очікується Gaussian tuning)
- Surround suppression: стимул + контекст (очікується зниження відповіді)
- Contrast invariance: orientation tuning при різних контрастах (форма кривої має зберігатися)
- Порівняти з даними Hubel & Wiesel (1962) та Sceniak et al. (1999)

### Експеримент 4: Порівняння з predictive coding
**Мета:** Демаркація від існуючих bio-plausible моделей.
**Протокол:**
- Реалізувати мінімальну predictive coding модель (Rao & Ballard, 1999) з тими ж Gabor фільтрами
- Порівняти recognition accuracy, velocity sensitivity, та representation quality
- Визначити які саме властивості унікальні для SubstanceNet

### Експеримент 5: Тривале Hebbian дозрівання
**Мета:** Чи покращується recognition з більшим досвідом?
**Протокол:**
- Дозрівання V3/V4 на 10K, 50K, 100K MNIST зображень (unsupervised)
- Вимірювати recognition accuracy після кожного етапу
- Будувати learning curve: accuracy vs кількість спостережень
- Порівняти з few-shot learning baseline (Prototypical Networks, Matching Networks)

### Експеримент 6: CIFAR-10 з активним сприйняттям (сакади)
**Мета:** Перевірити гіпотезу Олексія про сакадичне сканування.
**Протокол:**
- Реалізувати foveal attention: високе розділення в центрі, низьке на периферії
- Послідовне сканування (3-5 фіксацій) з переміщенням fovea
- Hippocampus інтегрує інформацію з послідовних фіксацій
- Порівняти з поточним single-pass V1 на CIFAR-10

### Експеримент 7: Cross-architecture comparison
**Мета:** Жорсткий тест: що може SubstanceNet чого НЕ може transformer?
**Протокол:**
- Реалізувати ViT (Vision Transformer) з тими ж Gabor фільтрами як tokenizer
- Порівняти на задачах:
  - Few-shot recognition (без backprop) — SubstanceNet має перевагу (Hippocampus)
  - Velocity discrimination (video) — SubstanceNet має перевагу (V3 interference)
  - Standard classification (backprop) — ViT має перевагу
- Визначити niche де SubstanceNet перевершує стандартні архітектури

---

## Пріоритети

| # | Експеримент | Пріоритет | Складність | Час |
|---|---|---|---|---|
| 1 | κ без таргетування | КРИТИЧНИЙ | Низька | 1 день |
| 2 | Wave ablation | КРИТИЧНИЙ | Середня | 2-3 дні |
| 3 | V1 biological validation | Високий | Середня | 2 дні |
| 5 | Тривале Hebbian | Високий | Низька | 1 день |
| 4 | Predictive coding | Середній | Висока | 3-5 днів |
| 6 | Сакади | Середній | Висока | 5+ днів |
| 7 | Cross-architecture | Середній | Висока | 5+ днів |

Експерименти 1 та 2 необхідні ПЕРЕД публікацією. Решта — roadmap для наступних версій.
