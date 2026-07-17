---
title: "CS336 — Scaling Laws"
type: course-note
course: "Stanford CS336"
---

# CS336 — Scaling Laws

> Лекции 9 и 11 курса Stanford CS336. Как предсказать производительность модели до тренировки.

**Курс:** [[Stanford CS336/_index|Stanford CS336]]
**Лектор:** Tatsunori Hashimoto
**Связанные концепты:** [[Scaling Laws]], [[Pre-training]], [[Causal Language Modeling]]

---

## Что такое Scaling Laws

Scaling laws — эмпирические зависимости между тремя ключевыми факторами и качеством языковой модели:

- **N** — количество параметров модели
- **D** — количество токенов в обучающих данных
- **C** — compute-бюджет (FLOPs)

Зная эти зависимости, можно **до начала тренировки** предсказать:
- Какой loss получит модель заданного размера
- Оптимальное соотношение N и D для фиксированного бюджета C
- Стоит ли увеличивать модель или данные

---

## Kaplan Scaling Laws (OpenAI, 2020)

Первая систематическая работа (Kaplan et al., "Scaling Laws for Neural Language Models").

### Ключевые находки

**Power law зависимости:**

```
L(N) ∝ N^(-0.076)    # loss от размера модели
L(D) ∝ D^(-0.095)    # loss от объёма данных
L(C) ∝ C^(-0.050)    # loss от compute
```

Loss падает как степенная функция — **предсказуемо и гладко** на протяжении многих порядков.

**Вывод Kaplan:** при фиксированном бюджете **выгоднее масштабировать модель**, чем данные. Рекомендуемое соотношение: увеличивать N быстрее, чем D.

Это привело к тренду "больших недотренированных моделей" (GPT-3: 175B параметров, 300B токенов).

---

## Chinchilla Scaling Laws (DeepMind, 2022)

Hoffmann et al., "Training Compute-Optimal Large Language Models" — переломная работа.

### Методология

DeepMind обучил **400+ моделей** (от 70M до 16B параметров) на разных объёмах данных и исследовал, как loss зависит от (N, D) при фиксированном C.

### Ключевой результат

**Compute-optimal training:** модель и данные должны масштабироваться **одинаково**.

```
N_opt ∝ C^0.50    # оптимальный размер модели
D_opt ∝ C^0.50    # оптимальный объём данных
```

**Правило 20:1** — на каждый параметр нужно ~20 токенов обучающих данных.

### Chinchilla vs GPT-3

| | GPT-3 | Chinchilla |
|--|-------|------------|
| Параметры | 175B | 70B |
| Токены | 300B | 1.4T |
| Compute | ~3.5e23 FLOPs | ~5.0e23 FLOPs |
| Quality | Baseline | **Лучше** GPT-3 |

Chinchilla с 70B параметрами **побеждает** GPT-3 с 175B — потому что обучена на правильном количестве данных.

### Формула loss

```
L(N, D) = E + A/N^α + B/D^β
```

где:
- E — irreducible loss (энтропия языка)
- A/N^α — ошибка из-за ограниченного размера модели
- B/D^β — ошибка из-за ограниченного объёма данных
- α ≈ 0.34, β ≈ 0.28

---

## Post-Chinchilla: новые реальности

### Overtrained модели (LLaMA, 2023)

Meta сознательно нарушила Chinchilla-оптимум:

```
LLaMA 7B:  1T токенов  (оптимум ~140B)  → 7x overtrained
LLaMA 13B: 1T токенов  (оптимум ~260B)  → 4x overtrained
```

**Зачем?** Chinchilla оптимизирует **training compute**. Но для deployment важен **inference compute** — и маленькая модель дешевле в inference, даже если training был избыточным.

### Inference-Optimal Scaling

Новый фреймворк учитывает полную стоимость жизненного цикла модели:

```
Total Cost = C_train + N_queries × C_inference(N)
```

При большом N_queries (миллиарды запросов) оптимум сдвигается к **маленьким, хорошо обученным моделям**.

### Extreme Overtraining (2025)

Qwen3-0.6B: 600M параметров, 36T токенов = 60,000 токенов/параметр (vs 20 у Chinchilla).

```
Chinchilla:  20 tokens/param
LLaMA:       ~150 tokens/param
Qwen3-0.6B:  60,000 tokens/param
```

---

## Практическое применение

### Как использовать scaling laws

**Шаг 1:** Обучить серию маленьких моделей (10M-1B параметров)

**Шаг 2:** Фитировать power law на полученных loss

**Шаг 3:** Экстраполировать на целевой размер

```python
# Упрощённый пример
import numpy as np
from scipy.optimize import curve_fit

def scaling_law(C, a, b, alpha):
    return a + b * C ** (-alpha)

# Данные из маленьких экспериментов
compute_budgets = [1e17, 1e18, 1e19, 1e20]
losses = [3.5, 3.1, 2.8, 2.6]

params, _ = curve_fit(scaling_law, compute_budgets, losses)
# Предсказание для 1e23 FLOPs
predicted_loss = scaling_law(1e23, *params)
```

### Ограничения

1. **Экстраполяция** может быть неточной на 1-2 порядка вперёд
2. **Emergent abilities** не предсказываются scaling laws — некоторые способности появляются "скачком"
3. **Data quality** не учитывается — 1T токенов Wikipedia ≠ 1T токенов Reddit
4. **Downstream performance** коррелирует с loss, но не определяется им однозначно

---

## Compute-бюджетирование

### Формула FLOPs

Для Transformer с N параметрами, обученного на D токенов:

```
C ≈ 6 × N × D    # FLOPs (forward + backward)
```

Множитель 6: ~2 для forward pass, ~4 для backward pass (2 для gradients + 2 для activation recomputation).

### Примеры бюджетов

| Модель | N | D | C (FLOPs) | GPU-hours (A100) |
|--------|---|---|-----------|-----------------|
| GPT-2 | 1.5B | 40B | ~3.6e20 | ~800 |
| LLaMA 7B | 7B | 1T | ~4.2e22 | ~80,000 |
| LLaMA 70B | 70B | 2T | ~8.4e23 | ~1,700,000 |
| GPT-4 (est.) | ~1.8T MoE | ~13T | ~2e25 | ~25,000,000 |

---

## Scaling Laws для MoE

MoE-модели имеют свои scaling laws. Ключевое отличие:

```
Dense:  C ≈ 6 × N_total × D
MoE:    C ≈ 6 × N_active × D    # только активные параметры
```

Поэтому MoE с 400B total / 50B active параметрами тренируется как 50B dense-модель, но может иметь quality ближе к 200-300B dense.

---

## Ключевые выводы

1. **Scaling laws — power law зависимости** loss от N, D, C
2. **Chinchilla:** оптимальное соотношение ~20 токенов/параметр для training-compute
3. **Inference-optimal:** при массовом deployment выгоднее overtrained маленькие модели
4. **C ≈ 6ND** — базовая формула для оценки compute
5. Scaling laws позволяют **предсказать результат до тренировки** и сэкономить миллионы долларов

---

## Источники

- Stanford CS336, Lectures 9 & 11 — https://cs336.stanford.edu/
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020) — https://arxiv.org/abs/2001.08361
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022) — https://arxiv.org/abs/2203.15556
- Cameron R. Wolfe, "Scaling Laws for LLMs: From GPT-3 to o3" — https://cameronrwolfe.substack.com/p/llm-scaling-laws

---

**См. также:** [[Scaling Laws]], [[Pre-training]], [[CS336 — Mixture of Experts]], [[CS336 — Inference]]
