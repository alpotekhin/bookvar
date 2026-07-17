---
title: "Fast Inference from Transformers via Speculative Decoding"
url: https://arxiv.org/abs/2211.17192
authors: [Yaniv Leviathan, Matan Kalman, Yossi Matias]
year: 2022
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - inference
  - efficiency
  - decoding
  - sampling
Organization: Google Research
concepts:
  - "[[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
raw: "[[02 Areas/ML & DL/raw/papers/speculative-decoding/paper.txt]]"
---

# Fast Inference from Transformers via Speculative Decoding

**Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
**Published:** 2022 (arXiv:2211.17192, ICML 2023)
**URL:** https://arxiv.org/abs/2211.17192

## TL;DR

**Speculative Decoding** — метод ускорения авторегрессивного инференса без изменения выходного распределения. Маленькая модель-аппроксиматор Mq генерирует gamma draft-токенов, большая target-модель Mp верифицирует их все параллельно за один forward pass. Принятые токены сохраняются, отвергнутый корректируется из adjusted distribution. Гарантия: выходное распределение идентично Mp. На T5-XXL 11B: **2-3x ускорение** vs стандартный декодинг, без изменения архитектуры, без переобучения.

## Problem

Авторегрессивный инференс из больших Transformer-моделей — медленный:
- Генерация K токенов требует K последовательных forward pass через модель
- Каждый forward pass — memory-bandwidth-bound (веса модели читаются из HBM каждый раз)
- Compute resources underutilized: на inference bottleneck — memory, а не arithmetic

Существующие подходы (distillation, sparsification, quantization, early exit) требуют изменения архитектуры, переобучения, и/или меняют выходное распределение.

## Method

### Speculative Sampling (Section 2.3)

Для сэмплирования x ~ p(x) (целевое распределение):
1. Сэмплируем x ~ q(x) (из аппроксиматора)
2. Если q(x) <= p(x): принимаем x
3. Если q(x) > p(x): отвергаем с вероятностью 1 - p(x)/q(x), сэмплируем из adjusted distribution p'(x) = norm(max(0, p(x) - q(x)))

**Гарантия:** при любых p и q, результирующее распределение = p(x). Это обобщение speculative execution из процессоров на стохастический setting.

### Algorithm 1: SpeculativeDecodingStep

1. Mq авторегрессивно генерирует gamma draft-токенов x1, ..., x_gamma
2. Mp параллельно вычисляет p1, ..., p_{gamma+1} для всех позиций (один batched forward pass)
3. Для каждого i от 1 до gamma: принять xi если ri <= p_i(xi)/q_i(xi), иначе отвергнуть
4. Пусть n — число принятых токенов. Сэмплировать дополнительный токен t из скорректированного распределения
5. Вернуть x1, ..., xn, t (от 1 до gamma+1 токенов)

### Acceptance Rate alpha

alpha = E[beta], где beta = 1 - D_LK(p, q) — вероятность принятия одного токена.

D_LK(p, q) = 1 - sum_x min(p(x), q(x)) — симметричная дивергенция в [0, 1].

При i.i.d. assumption, ожидаемое число токенов за итерацию:
```
E(# tokens) = (1 - alpha^{gamma+1}) / (1 - alpha)
```

### Wall-time Improvement

Пусть c = cost(Mq) / cost(Mp) — cost coefficient. Тогда ожидаемый фактор ускорения:
```
speedup = (1 - alpha^{gamma+1}) / ((1 - alpha) * (gamma * c + 1))
```

Для negligible-cost approximation models (c -> 0): speedup -> (1 - alpha^{gamma+1}) / (1 - alpha), bounded by 1/(1-alpha).

### Выбор gamma

Оптимальный gamma находится численно как argmax формулы ускорения. При alpha=0.8, c=0.01: optimal gamma ~ 5-7.

### Модели-аппроксиматоры

- Маленькие модели той же архитектуры (~100x меньше target): alpha ~ 0.5-0.9
- N-gram модели (c = 0, table lookup): alpha ~ 0.05-0.2 — даже это дает ускорение
- Copy-from-context (для summarization): потенциально высокий alpha при c = 0
- Non-autoregressive модели

## Key Results

### T5-XXL 11B (Table 2)

| Задача | Mq | Temp | gamma | alpha | Speedup |
|--------|-----|------|-------|-------|---------|
| EnDe translation | T5-small 77M | 0 | 7 | 0.75 | **3.4x** |
| EnDe translation | T5-base 250M | 0 | 7 | 0.80 | 2.8x |
| EnDe translation | T5-small 77M | 1 | 7 | 0.62 | **2.6x** |
| CNN/DM summarization | T5-small 77M | 0 | 5 | 0.65 | **3.1x** |
| CNN/DM summarization | T5-base 250M | 0 | 5 | 0.73 | 3.0x |
| CNN/DM summarization | T5-small 77M | 1 | 5 | 0.53 | 2.3x |

T5-small (77M) дает лучший баланс c vs alpha → максимальный speedup.

### Empirical alpha values (Table 3)

| Target | Approx | Sampling | alpha |
|--------|--------|----------|-------|
| GPT-like 97M | GPT-like 6M | T=0 | 0.88 |
| GPT-like 97M | GPT-like 6M | T=1 | 0.89 |
| T5-XXL 11B (EnDe) | T5-small | T=0 | 0.75 |
| T5-XXL 11B (EnDe) | Bigram | T=0 | 0.20 |
| LaMDA 137B | LaMDA 8B | T=0 | 0.75 |
| LaMDA 137B | LaMDA 100M | T=1 | 0.57 |

alpha выше при argmax (temp=0) vs standard sampling (temp=1). Модели ~100x меньше target дают alpha 0.5-0.9.

### Arithmetic Operations

При низком alpha растет количество arithmetic operations (wasted compute при rejection). При alpha=0.9, gamma=10: 1.6x больше ops, но 6.86x speedup по latency.

## My notes

- Speculative decoding стал стандартным методом ускорения inference в 2023-2024. Используется в vLLM, TGI, llama.cpp, Medusa, EAGLE.
- Главная сила: zero-change guarantee — выходное распределение математически идентично target model. Это критично для production, где нужна воспроизводимость.
- На практике alpha зависит от задачи и пары моделей. Для chat/instruction-following alpha обычно 0.6-0.8 с хорошо подобранным draft model.
- Trade-off: метод полезен только когда inference memory-bandwidth-bound и есть свободный compute (batch=1). При больших batch sizes дополнительный compute от verification начинает конкурировать с основной нагрузкой.
- Независимая работа Chen et al. (2023) — "Accelerating LLM Decoding with Speculative Sampling" — показала аналогичные 2-2.5x ускорения на Chinchilla 70B.
- Развитие идеи: Medusa (multiple draft heads), EAGLE (feature-level draft), SpecInfer (tree-based verification) — все построены на фундаменте speculative decoding.
