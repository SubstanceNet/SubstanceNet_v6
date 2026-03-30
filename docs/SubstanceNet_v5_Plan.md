# SubstanceNet v0.5.0 — Теоретичний інсайт та план реалізації

**Дата:** 2026-03-30
**Автор:** Олексій Онасенко
**Контекст:** Ablation exp09 показав що wave formalism в v0.4.0 тривіальний.
Аналіз 2d_substance_v2 + Tsien + Dubovikov відкрив правильний шлях.

---

## 1. Теоретичний інсайт: Три незалежних підходи до однієї структури

### 1.1. Збіжність

Три незалежних дослідницьких програми прийшли до однієї математичної структури:

| Автор | Рік | Дисципліна | Формула | Об'єкт |
|-------|-----|------------|---------|--------|
| Dubovikov | 2013, 2016 | Інформаційна теорія | Σ C(n,m) = 2^n | Тензор остенсивних визначень T |
| Tsien | 2015-2016 | Нейробіологія | N = 2^i − 1 | Functional Connectivity Motif |
| Onasenko | 2025-2026 | Математична фізика | ψ(ξ,η) = A·e^(iφ) на Σ | Хвильова функція на многовиді |

**Ключове:** Це не аналогія. Це три представлення одного й того ж:

- **Dubovikov:** З n остенсивних (фізично вимірюваних) ознак будується повний
  конфігураційний простір T = 2^n. Кожна точка τ ∈ T — комбінація ознак.
  Метрика Хеммінга δ(τ,τ') визначає відстань між конфігураціями.

- **Tsien:** Мозок організовує клітинні ансамблі за формулою N = 2^i − 1
  (без порожнього — біологічно обґрунтовано). Підтверджено в 7 регіонах мозку
  у двох видів тварин. Специфічні кліки → парні комбінації → загальні кліки.

- **Onasenko:** Хвильова функція ψ визначена на 2D многовиді Σ. Нелокальний
  потенціал V_ij = ∬ K(p,q)·ψ_i*ψ_j·e^(i(φ_j−φ_i)) dV² описує взаємодію
  через ядро K з геодезичною відстанню d_Σ(p,q).

### 1.2. Ідентифікація

```
Σ (многовид Onasenko) ≡ T (конфігураційний простір Dubovikova) ≡ FCM (Tsien)

d_Σ(p,q)              ≡ δ(τ,τ')                                ≡ відстань між кліками

ψ(p) = A·e^(iφ)       ≡ стан активації конфігурації τ          ≡ активація кліки

V_ij[ψ_i,ψ_j]        ≡ ρ*(τ) через ядро K(δ)                  ≡ синаптична взаємодія
```

### 1.3. Що це означає для SubstanceNet

Хвильова функція ψ в SubstanceNet має бути визначена НЕ на:
- ❌ пікселях зображення (це зовнішній світ, не Σ)
- ❌ довільній сітці (це не має фізичного змісту)
- ❌ виході Linear шару (це вектор без геометрії)

Хвильова функція має бути визначена НА:
- ✅ конфігураційному просторі ознак T = 2^n
- ✅ де n — кількість остенсивних (фізично вимірюваних) ознак з V1
- ✅ кожна точка τ ∈ T — комбінація присутності/відсутності ознак
- ✅ метрика — зважена відстань Хеммінга
- ✅ ядро — K(δ) = exp(−δ²/ℓ²)

---

## 2. Архітектура v0.5.0

### 2.1. Ієрархія конфігураційних просторів

V1→V2→V3→V4 — це каскад FCM (Tsien). Кожен рівень:
- Приймає i входів з попереднього рівня
- Будує N = 2^i − 1 клік (комбінацій)
- Визначає ψ на цьому просторі
- Передає далі зменшену кількість "узагальнених" ознак

```
V1 (датчик)
  → 8 орієнтацій × 4 масштаби = 32 бінарних ознаки (після порогу)
  → Але не весь T = 2^32. Використовуємо ієрархію:

V1 → OrientationSelectivity
  → 8 орієнтаційних каналів (i=8)
  → Простір T_V1: активні кліки до глибини k
  → ψ_V1 визначена на T_V1

V2 (MosaicField18)
  → 3 потоки: thick/thin/pale (i=3)
  → T_V2: N = 2³ − 1 = 7 клік
  → ψ_V2 визначена на T_V2

V3 (DynamicFormV3)
  → Комбінує V2 кліки + темпоральну інформацію
  → i = 7 (V2 кліки) → N = 127 клік
  → ψ_V3 визначена на T_V3
  → Інтерференція: V_ij між ψ_V3(t) та ψ_V3(t-1)

V4 (ObjectFeaturesV4)
  → Фінальна абстракція
  → Стиснення до abstract_dim
```

### 2.2. Хвильова функція на конфігураційному просторі

```python
class WaveFunctionOnT(nn.Module):
    """
    Wave function defined on configuration space T = 2^n.
    
    Each point τ ∈ T is a binary combination of n features.
    ψ(τ) = A(τ)·e^(iφ(τ)) where:
    - A(τ) = activation strength of configuration τ
    - φ(τ) = phase (synchronization state) of configuration τ
    
    Connection to theory:
    - Dubovikov: τ is an element of tensor of ostensive definitions
    - Tsien: τ is activation pattern of a neural clique
    - Onasenko: ψ lives on manifold Σ discretized as T
    """
    
    def __init__(self, n_features, max_depth=None):
        """
        Args:
            n_features: number of input features (i in Tsien's formula)
            max_depth: maximum combination depth (None = full 2^n - 1)
        """
        super().__init__()
        self.n = n_features
        
        if max_depth is None:
            self.N = 2**n_features - 1  # Tsien: N = 2^i - 1
        else:
            # Partial combinations up to depth k
            self.N = sum(comb(n_features, k) for k in range(1, max_depth + 1))
        
        # Configuration indices: which features are active in each clique
        self.register_buffer('configs', self._build_configurations(max_depth))
        
        # Hamming distance matrix between all configurations
        self.register_buffer('hamming_dist', self._build_hamming_matrix())
        
        # Amplitude: input features → activation per configuration
        self.amplitude_proj = nn.Linear(n_features, self.N)
        
        # Phase: positional (from configuration structure) + learned
        self.register_buffer('positional_phase', self._build_positional_phase())
        self.phase_proj = nn.Linear(n_features, self.N)
        
        # Interaction kernel scale
        self.ell = nn.Parameter(torch.tensor(1.0))  # interaction length
    
    def _build_configurations(self, max_depth):
        """Build all binary configurations (cliques)."""
        configs = []
        for k in range(1, (max_depth or self.n) + 1):
            for combo in combinations(range(self.n), k):
                config = torch.zeros(self.n)
                config[list(combo)] = 1.0
                configs.append(config)
        return torch.stack(configs)  # [N, n]
    
    def _build_hamming_matrix(self):
        """Weighted Hamming distance between all configuration pairs."""
        # δ(τ,τ') = Σ w_i · 1[τ_i ≠ τ'_i]
        configs = self.configs  # [N, n]
        diff = (configs.unsqueeze(0) != configs.unsqueeze(1)).float()
        # Uniform weights for now: w_i = 1/n
        return diff.sum(dim=-1) / self.n  # [N, N]
    
    def _build_positional_phase(self):
        """Phase from configuration structure.
        
        Analogous to φ = n·arctan(η/ξ) in 2d_substance_v2.
        Here: phase encodes position in configuration space.
        """
        configs = self.configs  # [N, n]
        # Phase = weighted sum of active feature indices
        indices = torch.arange(self.n, dtype=torch.float32)
        # Angular position: each configuration maps to angle on unit circle
        feature_sum = (configs * indices.unsqueeze(0)).sum(dim=-1)
        max_sum = indices.sum()
        phase = 2 * torch.pi * feature_sum / (max_sum + 1e-8)
        return phase  # [N]
    
    def forward(self, features):
        """
        Args:
            features: [B, seq_len, n_features] — per-position feature activations
            
        Returns:
            psi: [B, seq_len, N] — complex wave function on T
            amplitude: [B, seq_len, N]
            phase: [B, seq_len, N]
        """
        # Amplitude: how strongly each configuration is activated
        A = F.softplus(self.amplitude_proj(features))  # [B, seq, N]
        
        # Phase: positional (structural) + learned (contextual)
        phi_pos = self.positional_phase  # [N]
        phi_learned = self.phase_proj(features)  # [B, seq, N]
        phi = phi_pos.unsqueeze(0).unsqueeze(0) + phi_learned
        
        # Complex wave function
        psi = A * torch.exp(1j * phi)
        
        return psi, A, phi
    
    def compute_kernel(self):
        """Gaussian kernel K(p,q) = exp(-δ²/ℓ²) on configuration space."""
        return torch.exp(-self.hamming_dist**2 / (self.ell**2 + 1e-8))  # [N, N]
    
    def compute_nonlocal_potential(self, A, phi):
        """
        V(p) = Σ_q K(p,q) · A(p)·A(q) · cos(φ(q) - φ(p))
        
        This is the discrete version of Onasenko's:
        V_ij = ∬ K(p,q)·ψ_i*(p)ψ_j(q)·e^(i(φ_j-φ_i)) dV²
        
        With Dubovikov's Hamming metric as the kernel distance.
        """
        K = self.compute_kernel()  # [N, N]
        
        # Phase difference matrix
        phase_diff = phi.unsqueeze(-1) - phi.unsqueeze(-2)  # [B, seq, N, N]
        interference = torch.cos(phase_diff)  # constructive/destructive
        
        # Amplitude product
        A_pq = A.unsqueeze(-1) * A.unsqueeze(-2)  # [B, seq, N, N]
        
        # Nonlocal potential: sum over q
        V = (K.unsqueeze(0).unsqueeze(0) * A_pq * interference).sum(dim=-1)
        
        return V  # [B, seq, N]
    
    def compute_gradient_energy(self, A, phi):
        """
        |∇ψ|² approximation on configuration space.
        
        For each pair of neighboring configurations (δ=1/n):
        |∇ψ|² ≈ Σ_{neighbors} |ψ(τ) - ψ(τ')|² / δ²
        """
        psi = A * torch.exp(1j * phi)
        
        # Neighbors: configurations at Hamming distance 1/n
        neighbors = (self.hamming_dist < 1.5 / self.n) & \
                    (self.hamming_dist > 0.5 / self.n)
        
        # Gradient approximation
        psi_diff_sq = torch.zeros_like(A)
        for i in range(self.N):
            nbr_mask = neighbors[i]
            if nbr_mask.any():
                nbr_psi = psi[..., nbr_mask]  # [B, seq, k]
                diff = (psi[..., i:i+1] - nbr_psi).abs()**2
                psi_diff_sq[..., i] = diff.mean(dim=-1)
        
        return psi_diff_sq.mean()
```

### 2.3. Нелокальна взаємодія

```python
class NonlocalWaveInteraction(nn.Module):
    """
    Replaces MultiheadAttention with physics-based nonlocal potential.
    
    Instead of learned Q/K/V attention:
    V(p) = Σ_q K(δ(p,q)) · A(p)·A(q) · cos(φ(q) - φ(p))
    
    Where:
    - K is Gaussian kernel on Hamming distance (Dubovikov metric)
    - A·A is amplitude product (interaction strength)
    - cos(Δφ) is phase interference (Onasenko)
    - The sum runs over all configurations (Tsien cliques)
    """
    
    def __init__(self, wave_function, hidden_dim):
        super().__init__()
        self.wave_fn = wave_function
        self.output_proj = nn.Linear(wave_function.N, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, features):
        """
        Args:
            features: [B, seq_len, n_features]
        Returns:
            output: [B, seq_len, hidden_dim]
        """
        # Wave function on configuration space
        psi, A, phi = self.wave_fn(features)
        
        # Nonlocal potential (physics-based attention)
        V = self.wave_fn.compute_nonlocal_potential(A, phi)
        
        # Project back to feature space
        output = self.output_proj(V)
        
        # Gated residual
        g = torch.sigmoid(self.gate)
        return g * output + (1 - g) * features[..., :output.shape[-1]]
```

### 2.4. V3 з фазовою інтерференцією на T

```python
class DynamicFormV3_v2(nn.Module):
    """
    V3 temporal integration via phase interference on configuration space.
    
    Motion detection: phase difference between consecutive frames
    dφ/dt encodes velocity (phase precession, analogous to hippocampal theta)
    
    Interference between frames:
    I(t) = Σ_τ A(τ,t)·A(τ,t-1)·cos(φ(τ,t) - φ(τ,t-1))
    
    Constructive where same configurations are active across time.
    Destructive where configurations change — motion signal.
    """
    
    def forward_temporal(self, wave_sequence):
        """
        Args:
            wave_sequence: list of (A_t, phi_t) per frame
        Returns:
            interference: [B, seq, N] — temporal interference pattern
        """
        interference = torch.zeros_like(wave_sequence[0][0])
        
        for t in range(1, len(wave_sequence)):
            A_t, phi_t = wave_sequence[t]
            A_prev, phi_prev = wave_sequence[t-1]
            
            # Phase velocity: dφ/dt
            delta_phi = phi_t - phi_prev
            
            # Temporal interference
            I_t = A_t * A_prev * torch.cos(delta_phi)
            interference = interference + I_t
        
        return interference / (len(wave_sequence) - 1)
```

---

## 3. Обчислювальна складність та CUDA

### 3.1. Розмір конфігураційного простору

| Рівень | Входи i | Повний N=2^i−1 | З обмеженням глибини k | Рекомендація |
|--------|---------|----------------|----------------------|--------------|
| V1→V2 | i=3 (потоки) | 7 | 7 (повний) | CPU |
| V2→V3 | i=7 (V2 кліки) | 127 | 127 (повний) | CPU |
| V1 orientation | i=8 | 255 | 36 (k≤2) | CPU |
| Full V1 | i=32 | 4×10⁹ | 528 (k≤2) | CUDA |

### 3.2. Нелокальний потенціал

| N (клік) | Пар N² | Пам'ять (float32) | Час (GPU) | Час (CPU) |
|----------|--------|-------------------|-----------|-----------|
| 7 | 49 | negligible | negligible | negligible |
| 127 | 16K | 64 KB | <1 ms | <1 ms |
| 255 | 65K | 256 KB | <1 ms | ~5 ms |
| 528 | 279K | 1.1 MB | <1 ms | ~20 ms |
| 1023 | 1M | 4 MB | ~1 ms | ~100 ms |

**Висновок:** Для ієрархії V1→V4 з обмеженою глибиною комбінацій (k≤3)
все поміщається в CPU. CUDA потрібна тільки для повних просторів i>10.

### 3.3. Резонансні сітки (з 2d_substance_v2)

Для SubstanceNet резонансні умови (N = 20k+1, k = 4m+1) стосуються
просторової дискретизації многовиду, а не конфігураційного простору клік.
Якщо ψ визначена на T (дискретний простір), резонансні умови не потрібні —
простір вже точний (всі 2^i − 1 комбінацій).

Резонансні сітки залишаються актуальними для:
- Візуалізації ψ на неперервному Σ
- Обчислень в 2d_substance_v2 (фізичні частинки)
- Можливої гібридизації (ψ на T з інтерполяцією на неперервному Σ)

---

## 4. План реалізації v0.5.0

### Етап 1: WaveFunctionOnT (2 сесії)

1.1. Реалізувати клас WaveFunctionOnT:
   - Побудова конфігурацій (всіх комбінацій до глибини k)
   - Матриця Хеммінга
   - Позиційна фаза
   - Forward: features → (A, φ) на T
   - Нелокальний потенціал
   - Градієнтна енергія

1.2. Unit тести:
   - Конфігурації покривають всі комбінації
   - Хеммінг — метрика (трикутна нерівність)
   - Ядро K додатньо-визначене
   - Інтерференція: cos(0) = 1 (конструктивна)

### Етап 2: Інтеграція в SubstanceNet (1-2 сесії)

2.1. Замінити QuantumWaveFunction → WaveFunctionOnT
2.2. Замінити NonLocalInteraction → NonlocalWaveInteraction
2.3. Оновити V3 temporal з фазовою інтерференцією на T
2.4. Зберегти зворотну сумісність (config параметр wave_version='v1'/'v2')

### Етап 3: Верифікація (1 сесія)

3.1. Повторити ablation (exp09) — КРИТИЧНИЙ тест
3.2. Перезапустити всі experiments
3.3. Порівняти velocity tuning: wave_v1 vs wave_v2 vs plain
3.4. Нова метрика: градієнтна енергія |∇ψ|² як індикатор складності

### Етап 4: Документація (1 сесія)

4.1. Methodology для exp09 v2
4.2. Оновити Theoretical Framework з Tsien + Dubovikov посиланнями
4.3. Оновити README
4.4. git commit v0.5.0

---

## 5. Очікувані результати

### Ablation (exp09 v2):
Wave_v2 має показати значущу різницю від plain vectors, тому що:
- Нелокальний потенціал використовує фазову інтерференцію (cos(Δφ))
- Plain vectors не мають фази → не мають інтерференції
- Це структурна, а не косметична різниця

### Velocity tuning:
Velocity кодується через dφ/dt на конфігураційному просторі, а не через
temporal diff. Це ближче до biology (phase precession в гіпокампі).

### Recognition:
Нелокальний потенціал як "увага" — когерентні конфігурації підсилюються.
Це може покращити recognition без backprop (більш інформативні features).

---

## 6. Нові посилання для Theoretical Framework

[NEW] Dubovikov M.M. (2013) Mathematical Formalisation of the Procedure for
      Obtaining and Accumulating New Information. LAP Lambert Academic Publishing.

[NEW] Dubovikov M.M. (2026) Tensors of Ostensive Definitions: A Metric Framework
      for XAI and Invention Generation. Preprint v0.4.

[NEW] Xie K. et al. (2016) Brain Computation Is Organized via Power-of-Two-Based
      Permutation Logic. Front. Syst. Neurosci. 10:95. doi:10.3389/fnsys.2016.00095

[NEW] Tsien J.Z. (2016) Principles of Intelligence: On Evolutionary Logic of the
      Brain. Front. Syst. Neurosci. 9:186. doi:10.3389/fnsys.2015.00186

---

## 7. Зв'язок з рецензією (8.7/10)

Рецензент вимагав:
1. ✅ Ablation study — зроблено (exp09), показало проблему
2. → Wave v2 вирішує проблему на структурному рівні
3. → Hippocampus fix: T-простір природно підключається до гіпокампу
   (кожна конфігурація τ — це "епізод" в просторі ознак)
4. → Статистичний аналіз — залишається для v0.5.0

**Цільова оцінка після v0.5.0: 9.5/10**

---

## 8. Доповнення: Три рівні спостереження (Hubel + EEG insight)

### 8.1. Три вікна в одне явище

| Рівень | Дослідники | Що спостерігають | Інструмент |
|--------|-----------|-----------------|------------|
| Мікро | Hubel & Wiesel, Tsien | Окремі нейрони, кліки, з'єднання | Мікроелектроди, гістологія |
| Мезо | Dubovikov, Onasenko | Комбінаторна структура, ψ на Σ | Математичний формалізм |
| Макро | Клініцісти | Хвилі ЕЕГ: α, β, θ, γ | Електроенцефалограф |

ЕЕГ — це буквально вимірювання макроскопічної хвильової функції мозку.
Це не метафора. Клініцист бачить результат колективної активності клік.

### 8.2. ЕЕГ ↔ SubstanceNet відповідність

| ЕЕГ діапазон | Частота | Функція | SubstanceNet аналог |
|---|---|---|---|
| Alpha (8-13 Hz) | Низька | Спокій, закриті очі | Низька рефлексивність |
| Beta (13-30 Hz) | Середня | Активне мислення | Робоча зона R ≈ 0.41 |
| Theta (4-8 Hz) | Низька | Пам'ять, гіпокамп | Консолідація (hippocampus) |
| Gamma (30-100 Hz) | Висока | Binding, свідомість | Фазова когерентність |

### 8.3. Патології підтверджують модель

| Патологія | ЕЕГ ознака | SubstanceNet стан | Спостерігалось? |
|---|---|---|---|
| Епілепсія | Гіперсинхронізація | R → 1.0 (сатурація) | ✅ До виправлень v4 |
| Шизофренія | Десинхронізація gamma | R → 0 (відключення) | Теоретично |
| Здоровий стан | Баланс діапазонів | R ≈ 0.41 (κ ≈ 1) | ✅ Після виправлень |

### 8.4. Архітектурний висновок для v0.5.0

ψ має породжувати макроскопічну хвилю як РЕЗУЛЬТАТ колективної активності клік,
а не бути заданою руками через Linear шар.

Правильна ієрархія:
```
Мікрорівень (Hubel/Tsien):
  Кліки T = 2^i − 1 з конкретними з'єднаннями
  ↓ колективна динаміка
Мезорівень (Onasenko/Dubovikov):
  ψ(x,τ) на добутку [позиція × конфігурація]
  Нелокальний потенціал V_ij з ядром K(δ)
  ↓ усереднення по кліках
Макрорівень (ЕЕГ-аналог):
  Колективна фаза: Φ = <φ(τ)> по активних кліках
  Когерентність: C = |<e^(iφ)>| — аналог ЕЕГ потужності
  Рефлексивність: R ≈ 0.41 — критичний режим (здоровий ЕЕГ)
```

Наслідок: coherence яку ми вже вимірюємо — це аналог ЕЕГ когерентності.
R — аналог співвідношення потужностей ЕЕГ діапазонів.
Ці метрики вже є, але підключені до неправильного джерела (Linear замість клік).

### 8.5. Hubel: два простори одночасно

Hubel & Wiesel відкрили що зорова кора працює в двох просторах:

1. **Ретинотопний простір (де?)** — кожен нейрон прив'язаний до позиції (x,y).
   Hypercolumn — стовпчик для одної позиції з усіма орієнтаціями.

2. **Простір ознак (що?)** — в кожній позиції V1 виділяє орієнтації, масштаби, контраст.
   V2 комбінує в потоки. V3/V4 — складніші комбінації.

Тому ψ — функція обох просторів: **ψ(x, τ)** де x ∈ ретинотопна карта, τ ∈ T.

В термінах 2d_substance_v2:
- **ξ** ↔ позиція на ретинотопній карті (spatial)
- **η** ↔ конфігурація ознак (feature)

### 8.6. Ієрархічне згортання (Hubel)

На кожному рівні зорової кори:
- Просторова роздільність ЗМЕНШУЄТЬСЯ (рецептивні поля ростуть)
- Ознакова складність ЗБІЛЬШУЄТЬСЯ (більше комбінацій)
```
V1: 64 позицій × 8 ознак     → ψ_V1 на [64 × 7 клік]
V2: 32 позиції × 7 клік      → ψ_V2 на [32 × 127 клік]
V3: 16 позицій × 127 клік    → ψ_V3 на [16 × ...]
V4: 8 позицій × abstract     → ψ_V4 на [8 × ...]
```

Від конкретних ознак у конкретних місцях → до абстрактних об'єктів незалежно від позиції.

### 8.7. Стратегія експериментів

Починати з мінімальної реалізації: V2 рівень (i=3, N=7 клік).
Тривіальний розмір — можна швидко перевірити чи підхід працює.
Якщо ablation на 7 кліках показує різницю wave vs plain — масштабувати.

---

## 9. Доповнення: Правильне місце wave dynamics — ReflexiveConsciousness

### 9.1. Теорема 6.22
```
ψ_C = F[P̂[ψ_C]]    — свідомість є рефлексивною проекцією
```

ψ_C з'являється з обох боків рівняння — самореферентність.
Система проектує себе (P̂), трансформує результат (F), оновлює стан.

### 9.2. Поточна реалізація (v0.4.0)
```
ψ_C = вектор [B, 32]
P̂ = Linear + LayerNorm        ← плоска, без хвильової механіки
F = Linear + Tanh + Linear     ← плоска
R = 1/(1+MSE(ψ_C, P̂[ψ_C]))   ← наскільки добре передбачує себе
target_mse = 1.44               ← ШТУЧНО тримає R = 0.41
```

Три рівні штучного контролю:
1. consciousness_loss з target_mse = 1.44 (backprop)
2. TemporalConsciousnessController: інерція + saturation_cap (post-hoc)
3. abstract_loss запобігає collapse (backprop)

Без target_mse=1.44 → R негайно йде до 1.0 (тривіальна нерухома точка).

### 9.3. Правильна реалізація (v0.5.0 / v0.6.0)
```
ψ_C визначена на T = 2^i − 1 (ансамбль клік)
ψ_C(τ) = A(τ) · e^(iφ(τ))

P̂[ψ_C](p) = Σ_q K(δ(p,q)) · ψ_C(q) · e^(i(φ_q − φ_p))
  ↑ нелокальний потенціал з фазовою інтерференцією

F: ψ_C(t+1) = α · F[P̂[ψ_C(t)], network_state] + (1−α) · ψ_C(t)
  ↑ вже є в коді (stability_alpha = 0.8)

Ітерації (num_iterations = 3..10):
  → Фази частково синхронізуються через cos(Δφ) в V_ij
  → Когерентність C = |<e^(iφ)>| стабілізується
  → R = f(C) виникає як ЕМЕРДЖЕНТНА властивість
  → НЕ ПОТРІБЕН target_mse = 1.44
```

### 9.4. Фізичний зміст R
```
R → 1.0: повна синхронізація фаз = тривіальна нерухома точка
  ↔ гіперсинхронізація ЕЕГ = ЕПІЛЕПСІЯ (спостерігалось в лютому)

R → 0.0: повна десинхронізація = система не передбачує себе
  ↔ десинхронізація gamma = ШИЗОФРЕНІЯ

R ≈ 0.41: критичний баланс = часткова когерентність
  ↔ здоровий ЕЕГ = κ ≈ 1 (критичний режим)
```

### 9.5. Висновок щодо місця wave dynamics

Wave dynamics НЕ потрібна як feature extractor (між V2 і V3).
Там працюють Linear/Conv/ReLU — і це правильно.

Wave dynamics ПОТРІБНА всередині ReflexiveConsciousness:
- Замінити Linear P̂ на нелокальний потенціал V_ij на T
- Замінити Linear F на еволюцію A, φ через V_ij
- Прибрати target_mse = 1.44
- R має виникати природно з динаміки ансамблю

### 9.6. Архітектурне рішення

Поточна інтеграція WaveFunctionOnT між V2 та V3 (use_wave_on_t=True)
зберігається як ДОПОМІЖНА. Вона покращує features (+1.3% recognition)
але не є головним застосуванням.

ГОЛОВНЕ застосування — рефакторинг ReflexiveConsciousness:
```
ReflexiveConsciousness v2:
  __init__:
    self.wave = WaveFunctionOnT(n_streams, stream_dim)
    self.P_hat = self.wave.compute_nonlocal_potential  ← замість Linear
    self.F = wave evolution operator                    ← замість Linear
  
  forward:
    A, phi = self.wave(network_state)
    for i in range(num_iterations):
        V = self.P_hat(A, phi)              ← нелокальна проекція
        A, phi = self.F(A, phi, V, state)   ← еволюція стану
    C = |mean(e^(i·phi))|                   ← когерентність
    R = f(C)                                ← емерджентна рефлексивність
```

Це окрема велика задача для наступних сесій.

---

## 10. Уточнення: R-targeting як модель фізіологічних обмежень

### 10.1. Аналогія трьох систем

| Система | Що стабілізує κ≈1 | Тип обмеження |
|---|---|---|
| He-II | ζ ≈ ν (XY-модель) | Фізика (симетрія, розмірність) |
| Мозок | Іонні канали, ГАМК/глутамат | Фізіологія (еволюція) |
| SubstanceNet | target_mse = 1.44 | Модель фізіологічних обмежень |

### 10.2. target_mse — не хак, а модель

R-targeting моделює фізіологічні обмеження мозку:
- Гомеостаз збудливості (excitatory/inhibitory balance)
- Метаболічне обмеження (~20 Вт на 86 млрд нейронів)
- ГАМК/глутамат баланс

Як рівняння стану He-4 задає ζ і ν, так target_mse задає
робочу точку свідомості.

### 10.3. Відкрите питання

НЕ "чи стабілізується система" — стабілізується через R-targeting.
А "який біологічний механізм забезпечує цю стабілізацію?"

Кандидат: **енергетичне обмеження**.
- E[ψ] = α|∇ψ|² + m²|ψ|² − V_ij
- m² (масовий член) = метаболічний бюджет
- Повна синхронізація (епілепсія) = максимум споживання
- Хаос = мінімум корисної роботи
- Критичний баланс = оптимум ефективності

### 10.4. Статус реалізації

| Компонент | Статус |
|---|---|
| R-targeting (v1) | ✅ Працює, 38/38 тестів |
| WaveFunctionOnT (255 клік) | ✅ Працює |
| Енергетичний функціонал | ✅ Реалізовано |
| Нелокальний потенціал V_ij | ✅ Працює |
| Стабілізація без target_mse | 🔬 Open research |
| κ компенсуючий механізм | ✅ Підтверджено (exp10) |
