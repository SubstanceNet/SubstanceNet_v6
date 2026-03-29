# Шаблон дослідницького проєкту — SubstanceNet v4

## Структура (обов'язкова для всіх проєктів)
```
research/<назва_проєкту>/
├── README.md        — мета, еталон, дата, статус, структура, запуск, результати
├── docs/            — документація
├── data/            — результати тестів (.json), іменування: results_YYYYMMDD_HHMMSS.json
├── figures/         — візуалізація (.png)
├── scripts/         — скрипти (train.py, analyze.py, diagnose.py)
├── reports/         — текстові звіти
└── checkpoints/     — збережені моделі (.pt)
```

## README.md шаблон
```
# <Назва> — <Тип дослідження>
**Мета:** <одне речення>
**Еталон:** <reference метрики>
**Дата:** <YYYY-MM-DD>
**Статус:** <В процесі / Завершено / Заблоковано>
## Структура
<дерево вище>
## Запуск
<команда>
## Результати
Див. `data/` та `reports/`
```

## Обов'язкові метрики
- Accuracy (test)
- Reflexivity (R)
- Phase Coherence
- Mean Amplitude
- Complexity
- Abstract Variance
- Порівняння з попередніми версіями

## Створення нового проєкту
```bash
NAME=my_experiment
mkdir -p research/$NAME/{docs,data,figures,scripts,reports,checkpoints}
# Скопіювати та адаптувати README з цього шаблону
```
