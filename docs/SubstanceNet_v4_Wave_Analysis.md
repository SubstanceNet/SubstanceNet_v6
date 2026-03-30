# Wave Formalism: Діагноз та план виправлення

**Дата:** 2026-03-30
**Контекст:** Ablation (exp09) показав що поточна реалізація wave formalism тривіальна.
2d_substance_v2 має працюючу реалізацію. Аналіз різниці та план переносу.

---

## 1. Діагноз: чому поточна реалізація тривіальна

### Що робить QuantumWaveFunction в v4:

```python
amplitude = softplus(Linear(x))      # [B, 9, 64] — просто ReLU-подібна проекція
phase = Linear(x)                     # [B, 9, 64] — просто ще один вектор
psi = A*cos(φ) + i*A*sin(φ)          # polar→cartesian перетворення
```

Це математично еквівалентне двом незалежним Linear шарам з нелінійностями.
Ablation підтвердив: заміна на `ReLU(Linear(x))` + `Linear(x)` дає **ідентичні або кращі** результати.

### Чого НЕ МАЄ в v4 (але є в 2d_substance_v2):

| Властивість | 2d_substance_v2 | SubstanceNet v4 |
|---|---|---|
| Просторова структура | 2D поле на многовиді Σ з метрикою g_μν | 1D вектор (seq × dim) |
| Амплітуда | A(ξ,η) = r^|n| · exp(-r²/2l²) — топологічний профіль | softplus(Linear(x)) — довільна |
| Фаза | φ = n·arctan(η/ξ) — обмотування з winding number | Linear(x) — довільний вектор |
| Градієнт | |∇ψ|² = |∇A|² + A²|∇φ|² з метрикою | Відсутній |
| Інтерференція | V_ij = ∬ K(p,q)·ψ_i*ψ_j·e^(i(φ_j-φ_i)) dV² | cosine_similarity |
| Топологія | Winding number n ∈ ℤ, зберігається | Відсутня |
| Лагранжіан | T_kinetic + T_mass + T_self_interaction | Відсутній |
| Нормування | ∫|ψ|²dV = 1 на многовиді | Немає |

### Кореневе питання:

В 2d_substance_v2 хвильова функція — це **просторовий об'єкт** з внутрішньою геометрією.
В v4 — це **вектор ознак** з косметичною обгорткою cos/sin.

Різниця не в нотації. Різниця в тому що **фаза не має геометричного змісту**.
Фаза в 2d_substance_v2 кодує просторове положення через arctan(η/ξ).
Фаза в v4 — довільний вихід Linear шару без жодного обмеження.

---

## 2. Що з 2d_substance_v2 можна перенести

### 2.1. Просторова фаза (КРИТИЧНО)

Зараз V1 output має просторову структуру [B, seq_len=9, dim=64] — це сітка 3×3.
Фаза має кодувати **позицію на цій сітці**, а не бути довільною.

**Принцип з 2d_substance_v2:**
```python
# Фаза = n * arctan(η/ξ) — кодує позицію відносно центру
phi = n * torch.atan2(ETA, XI)
```

**Трансляція в SubstanceNet:**
```python
# Фаза включає позиційну компоненту + навчальну
positions = grid_positions(seq_len)  # [seq_len, 2] — (x, y) на сітці
positional_phase = torch.atan2(positions[:, 1], positions[:, 0])  # [seq_len]
learned_phase = phase_fc(features)                                 # [B, seq_len, dim]
total_phase = positional_phase.unsqueeze(-1) + learned_phase       # позиція + контент
```

Тепер фаза **має геометричний зміст**: позиція на рецептивному полі + відносна фаза ознак.

### 2.2. Просторовий градієнт (ВАЖЛИВО)

В 2d_substance_v2 |∇ψ|² обчислюється через метричний тензор.
В SubstanceNet на сітці 3×3 це можна апроксимувати:

```python
# Градієнт амплітуди на сітці
grad_A_x = A[:, 1:, :] - A[:, :-1, :]  # горизонтальний
grad_A_y = A.reshape(B, 3, 3, dim)
grad_A_y = grad_A_y[:, :, 1:, :] - grad_A_y[:, :, :-1, :]

# Градієнт фази
grad_phi_x = phi[:, 1:, :] - phi[:, :-1, :]

# Повна щільність |∇ψ|²
nabla_psi_sq = grad_A**2 + A**2 * grad_phi**2
```

Це дає **просторову інформацію** — де амплітуда змінюється швидко, де фаза обертається.
Саме це створює velocity tuning (не cosine similarity).

### 2.3. Нелокальний потенціал з ядром (ВАЖЛИВО)

Поточний NonLocalInteraction — це MultiheadAttention. Це не інтерференція.

Справжній нелокальний потенціал з 2d_substance_v2:
```
V_ij = ∬ K(p,q) · ψ_i*(p) · ψ_j(q) · e^(i(φ_j(q) - φ_i(p))) dV(p) dV(q)
```

Ключове: `e^(i(φ_j - φ_i))` — це **різниця фаз між позиціями p та q**.
Якщо фази однакові → конструктивна інтерференція.
Якщо фази протилежні → деструктивна.

Для SubstanceNet (сітка 3×3):
```python
# Ядро K(p,q) = exp(-|p-q|²/l²) — гауссіан на сітці
K = gaussian_kernel(grid_positions, l=interaction_scale)  # [9, 9]

# Потенціал з фазовою інтерференцією
phase_diff = phi.unsqueeze(2) - phi.unsqueeze(1)  # [B, 9, 9, dim]
interference = torch.cos(phase_diff)               # [B, 9, 9, dim]

# Зважена взаємодія
A_outer = A.unsqueeze(2) * A.unsqueeze(1)  # [B, 9, 9, dim]
V = (K.unsqueeze(0).unsqueeze(-1) * A_outer * interference).sum(dim=2)
```

Це **справжня інтерференція**: позиції з когерентними фазами підсилюють одна одну,
некогерентні — гасять. Це принципово відрізняється від attention.

### 2.4. Топологічний loss (КОРИСНО)

В 2d_substance_v2 zero_loss = δ_ε(|ψ|) · |∇ψ|² — мінімізує градієнт де амплітуда мала.
Це забезпечує локалізованість хвильової функції.

Поточний zero_loss в v4 — формальний і не впливає на нічого (0.01 вага).
З правильним градієнтом він стане фізично осмисленим.

---

## 3. Архітектурний план

### Етап 1: WaveFunction v2 (замінити QuantumWaveFunction)

```python
class WaveFunctionV2(nn.Module):
    """Wave function with spatial phase and proper gradients."""
    
    def __init__(self, in_channels, out_channels, grid_size=9):
        # grid_size = seq_len (3×3 = 9 позицій)
        self.grid_h = int(math.sqrt(grid_size))
        self.grid_w = self.grid_h
        
        # Amplitude: features → positive field
        self.amplitude_fc = nn.Linear(in_channels, out_channels // 2)
        
        # Phase: positional + learned
        self.phase_fc = nn.Linear(in_channels, out_channels // 2)
        self.register_buffer('positional_phase', 
                             self._create_positional_phase())
    
    def _create_positional_phase(self):
        """Phase encoding based on grid position (like arctan(η/ξ))."""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.grid_h),
            torch.linspace(-1, 1, self.grid_w), indexing='ij')
        # Polar angle — direct analogue of n*arctan(η/ξ)
        theta = torch.atan2(y, x).reshape(-1)  # [grid_size]
        return theta
    
    def forward(self, features):
        A = F.softplus(self.amplitude_fc(features))  # [B, 9, dim]
        phi_learned = self.phase_fc(features)          # [B, 9, dim]
        
        # Phase = positional + learned
        phi = self.positional_phase.unsqueeze(0).unsqueeze(-1) + phi_learned
        
        # Complex representation
        psi = A * torch.exp(1j * phi)
        
        return psi, A, phi
    
    def compute_gradient_norm(self, A, phi):
        """Compute |∇ψ|² on the spatial grid."""
        B, S, D = A.shape
        H, W = self.grid_h, self.grid_w
        
        A_2d = A.reshape(B, H, W, D)
        phi_2d = phi.reshape(B, H, W, D)
        
        # Spatial gradients
        dA_dx = A_2d[:, :, 1:, :] - A_2d[:, :, :-1, :]
        dA_dy = A_2d[:, 1:, :, :] - A_2d[:, :-1, :, :]
        dphi_dx = phi_2d[:, :, 1:, :] - phi_2d[:, :, :-1, :]
        dphi_dy = phi_2d[:, 1:, :, :] - phi_2d[:, :-1, :, :]
        
        # |∇ψ|² = |∇A|² + A²|∇φ|²
        A_mid_x = (A_2d[:, :, 1:, :] + A_2d[:, :, :-1, :]) / 2
        A_mid_y = (A_2d[:, 1:, :, :] + A_2d[:, :-1, :, :]) / 2
        
        grad_psi_sq_x = dA_dx**2 + A_mid_x**2 * dphi_dx**2
        grad_psi_sq_y = dA_dy**2 + A_mid_y**2 * dphi_dy**2
        
        return grad_psi_sq_x.mean() + grad_psi_sq_y.mean()
```

### Етап 2: NonlocalWaveInteraction (замінити NonLocalInteraction)

```python
class NonlocalWaveInteraction(nn.Module):
    """Nonlocal potential V_ij with kernel and phase interference."""
    
    def __init__(self, dim, grid_size=9, interaction_scale=1.0):
        self.register_buffer('kernel', 
                             self._create_gaussian_kernel(grid_size, interaction_scale))
        self.gate = nn.Parameter(torch.tensor(0.5))
    
    def _create_gaussian_kernel(self, grid_size, scale):
        H = W = int(math.sqrt(grid_size))
        positions = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        ), dim=-1).reshape(-1, 2)
        dist_sq = ((positions.unsqueeze(0) - positions.unsqueeze(1))**2).sum(-1)
        return torch.exp(-dist_sq / (2 * scale**2))  # [grid_size, grid_size]
    
    def forward(self, features, amplitude, phase):
        """
        V_ij = Σ_q K(p,q) · A(p)·A(q) · cos(φ(q) - φ(p))
        """
        # Phase difference matrix
        phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1)  # [B, S, S, D]
        interference = torch.cos(phase_diff)                    # constructive/destructive
        
        # Amplitude outer product
        A_pq = amplitude.unsqueeze(2) * amplitude.unsqueeze(1)  # [B, S, S, D]
        
        # Kernel-weighted interaction
        K = self.kernel.unsqueeze(0).unsqueeze(-1)  # [1, S, S, 1]
        V = (K * A_pq * interference).sum(dim=2)     # [B, S, D] — sum over q
        
        # Gate: balance nonlocal interaction with original features
        g = torch.sigmoid(self.gate)
        return g * V + (1 - g) * features
```

### Етап 3: V3 з реальною інтерференцією

V3 temporal interference замість cosine_similarity використовує
різницю фаз між кадрами як справжню фазову інтерференцію:

```python
# Фазова різниця між послідовними кадрами
delta_phi = phase_t[:, 1:] - phase_t[:, :-1]  # [B, T-1, seq, dim]

# Інтерференція: конструктивна де фази когерентні
interference = A_t[:, 1:] * A_t[:, :-1] * torch.cos(delta_phi)

# Це вже НЕ cosine similarity — це фізична інтерференція
# з урахуванням АМПЛІТУД на кожній позиції
```

---

## 4. Що це дає (очікувані результати)

### Velocity tuning:
Зараз velocity tuning виникає з temporal diff в V3 (thick stripes).
З правильною хвильовою механікою velocity буде кодуватись через
**швидкість обертання фази** dφ/dt — це ближче до біології
(phase precession в hippocampus, theta oscillations).

### Інтерференція:
Зараз NonLocal — це attention. З хвильовою інтерференцією
"увага" виникає природно: позиції з когерентними фазами
підсилюють одна одну, некогерентні — гасять.

### Градієнтний loss:
|∇ψ|² на просторовій сітці забезпечує фізично осмислену
регуляризацію — хвильова функція не може бути довільною,
вона має бути гладкою з обмеженим градієнтом.

### Ablation:
З правильною реалізацією ablation покаже різницю: прибрання
фазової інтерференції зруйнує nonlocal потенціал.

---

## 5. Ризики та обмеження

1. **Сітка 3×3 = 9 точок** — замало для повноцінної хвильової механіки.
   В 2d_substance_v2 використовується 101×101. Можливо потрібно збільшити
   AdaptiveAvgPool до (8,8) = 64 позицій.

2. **Продуктивність** — nonlocal потенціал O(S²) за кількістю позицій.
   Для S=64 це 4096 пар — прийнятно.

3. **Зворотна сумісність** — зміна WaveFunction та NonLocal зламає
   існуючі checkpoint та результати. Потрібен v0.5.0.

4. **Час реалізації** — ~2-3 сесії для повної інтеграції + тестування.

---

## 6. План дій

| Крок | Опис | Складність | Пріоритет |
|------|------|------------|-----------|
| 1 | WaveFunctionV2 з позиційною фазою та градієнтом | Середня | КРИТИЧНИЙ |
| 2 | NonlocalWaveInteraction з ядром | Середня | КРИТИЧНИЙ |
| 3 | V3 temporal з фазовою інтерференцією | Низька | Високий |
| 4 | Збільшити grid до 8×8 (64 позицій) | Низька | Високий |
| 5 | Інтеграція в SubstanceNet.forward() | Середня | КРИТИЧНИЙ |
| 6 | Повторити ablation (exp09) | Низька | КРИТИЧНИЙ |
| 7 | Перезапустити всі experiments | Низька | Високий |

**Очікуваний результат:** Ablation показує значущу різницю між wave та plain.
Velocity tuning кодується через фазову динаміку, а не через temporal diff.

---

## 7. Що робити з результатом ablation зараз

Результат exp09 — **чесний і важливий**. Він показує що поточна
реалізація wave formalism тривіальна. Це не провал — це наукова знахідка.

Рекомендація для документації:
- Зберегти exp09 та його methodology як є
- Додати в Limitations: "Current wave implementation (v0.4.0) is
  computationally equivalent to plain vectors. v0.5.0 will implement
  proper spatial phase and nonlocal potential."
- Velocity tuning curve залишається валідним результатом — він
  виникає з V3 архітектури (thick stripe temporal diff), не з wave formalism.
