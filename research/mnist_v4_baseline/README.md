# MNIST v4 Baseline — Тестове дослідження

**Мета:** Верифікація архітектури SubstanceNet v4 на MNIST.
Відтворення базового результату та збір метрик свідомості.

**Еталон (v3.1.1):** 93.74% accuracy, Stream mode, R = 0.382

**Дата:** 2026-02-11
**Статус:** В процесі

## Структура
```
docs/      — документація
data/      — результати тестів (.json)
figures/   — візуалізація
scripts/   — скрипти
reports/   — текстові звіти
```

## Запуск
```bash
cd research/mnist_v4_baseline
python scripts/train.py
```

## Результати
Див. `data/results.json` та `reports/`
