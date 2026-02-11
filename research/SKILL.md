<!--
System Classification: research.skill
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)

Тип документу: ДОВІДНИК
Статус: Чернетка
Версія: 0.1.0
Дата: 2026-02-11
Мова: Українська
-->

# Навігація по дослідженнях

## Активні експерименти

| Директорія | Мета | Статус |
|---|---|---|
| `reflexivity_optimum/` | Крива "рефлексивність vs accuracy" | Заплановано |
| `kappa_criticality/` | Зв'язок κ ≈ 1 з performance | Заплановано |
| `consciousness_modes/` | Порівняння 4 режимів | Заплановано |
| `cifar10/` | CIFAR-10 бенчмарк | Заплановано |

## Завершені

| Директорія | Результат |
|---|---|
| `completed/` | Архів завершених експериментів |

## Методологія

1. Кожен експеримент — окрема директорія
2. Фіксований seed (42) для відтворюваності
3. Результати в JSON + візуалізація в PNG
4. По завершенню — переміщення в `completed/`
