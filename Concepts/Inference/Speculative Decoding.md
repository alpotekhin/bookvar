---
title: Speculative Decoding
type: concept
status: legacy
tags:
  - inference
  - optimization
  - decoding
  - LLM
date_created: 2026-04-24
date_updated: 2026-05-31
---

# Speculative Decoding

## Суть

Используем маленькую быструю модель (drafter) чтобы угадывать несколько токенов вперёд, а большую (verifier) — чтобы проверять всё за один forward pass.

```
Drafter (7B): t1, t2, t3, t4  ← быстро
Verifier (70B): [t1✅, t2✅, t3✅, t4❌] ← один forward pass
→ принять t1..t3, сгенерировать правильный вместо t4
```

Verifier проверяет N токенов за тот же cost что и 1 токен. Это бесплатно.

## Математика: rejection sampling

```
q(x) = вероятность по drafter
p(x) = вероятность по verifier

Принять с вероятностью: min(1, p(x) / q(x))
Если отклонить → семплировать из (p - q)+ / Z
```

**Гарантия:** финальное распределение идентично p. Качество не деградирует совсем.

## Ускорение

Зависит от acceptance rate (как часто drafter угадывает):

| Acceptance rate | Ускорение |
|----------------|-----------|
| 80–85% | 3–4x |
| 60–70% | 2–2.5x |
| <50% | <1.5x |

## Варианты реализации

**Draft + Verify (классика)**
Отдельная маленькая модель того же семейства. Llama-7B → drafter для Llama-70B.
Нужна совместимая архитектура и словарь.

**Self-speculative decoding**
Одна модель сама себе drafter — early exit через часть слоёв для черновика, полный проход для верификации. Не нужна вторая модель.

**Medusa**
Дополнительные "головы" на финальном слое — каждая предсказывает токен на N шагов вперёд. Дообучаются, хранятся с основной моделью. Не нужна вторая загрузка весов.

**Block Verification / Draft Tree (DDTree)**
Drafter строит дерево вариантов, verifier проверяет всё дерево за один проход через ancestor-only attention mask:
```
         t1a ─ t2a ─ t3a
        ╱
    t0
        ╲
         t1b ─ t2b ─ t3b
```
Выбирается наилучший принятый путь. Лучший acceptance length в классе.
→ [[02 Areas/ML & DL/Papers/DDTree]] (arXiv 2604.12989)

**NGram matching**
Ищет повторяющиеся паттерны прямо в промпте — работает хорошо для RAG и длинных контекстов где модель часто копирует из источника. Без второй модели, бесплатно.

**MLP Speculator**
Лёгкий MLP поверх hidden states основной модели. IBM выпустил для Granite/Llama. Быстрее отдельной draft-модели, не требует второй загрузки весов.

**FR-Spec**
Обрезают словарь drafter до топ-K токенов по частоте → drafter значительно быстрее, acceptance rate почти не падает (редкие токены drafter всё равно угадывает плохо).

**GRIFFIN**
Решает training-inference misalignment при обучении draft-моделей — ключевая проблема когда drafter учится на данных из verifier, а не из собственного распределения.
→ arxiv.org/abs/2502.11018

## vLLM: как реализовано

Поддерживает несколько режимов, выбираешь при запуске:

```bash
# Draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5

# NGram (без второй модели, хорошо для RAG)
vllm serve ... \
  --speculative-model "[ngram]" \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 4

# MLP Speculator (IBM, для Granite/Llama)
vllm serve ibm-granite/granite-3.3-8b-instruct \
  --speculative-model ibm-granite/granite-3.3-8b-instruct-spec
```

**Pipeline внутри vLLM:**
```
Request → Scheduler
           │
           ├─ Draft phase: drafter генерирует K токенов
           │   (маленький forward pass)
           │
           └─ Verify phase: verifier обрабатывает K+1 позиций
               (один forward pass, параллельно по batch)
               │
               ├─ rejection sampling по каждому токену
               └─ принимает prefix, корректирует последний
```

**Метрики в vLLM:**
```
speculative_draft_acceptance_rate  # % принятых токенов
speculative_efficiency             # реальное vs теоретическое ускорение
```

**Batch size и speculative decoding:**
- Batch = 1 → максимальный выигрыш, latency падает в 2–3x
- Batch > 1 → разные запросы имеют разный acceptance rate, сложнее батчить
- Большой batch → overhead drafter'а ест throughput, часто выгоднее отключить

**Когда включать в проде:**
- Latency-критичные приложения (чат, real-time)
- Coding-агенты (код предсказуем → высокий acceptance rate)
- RAG (ngram speculator бесплатно даёт 1.5–2x)

**Когда не включать:**
- Batch > 32 и важен throughput, а не latency
- Мультимодальные задачи (vLLM speculative decoding пока text-only)

## Где работает хорошо / плохо

✅ Batch size = 1, длинные последовательности, предсказуемый текст (код, шаблоны), RAG
❌ Большой batch size, творческий текст, сильно разные drafter и verifier

## Связанные концепты

- [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]]
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]]
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]]

## Связанные статьи

- Leviathan et al. 2023 — оригинальная статья (Google)
- Chen et al. 2023 — Draft & Verify (DeepMind)
- DDTree (arXiv 2604.12989) — Draft Tree с ancestor-only attention mask
- GRIFFIN (arXiv 2502.11018) — training-inference misalignment для draft-моделей
- [[02 Areas/ML & DL/Papers/XGrammar]] — похожая идея precheck в structured generation
- Accelerating RL Post-Training via Speculative Decoding (NVIDIA, arXiv 2604.26779) — ускорение роллаутов при RL через speculative decoding
