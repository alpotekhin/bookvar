---
title: "CS336 — Mixture of Experts"
type: course-note
course: "Stanford CS336"
---

# CS336 — Mixture of Experts

> Лекция 4 курса Stanford CS336. MoE — архитектурный паттерн, ставший стандартом для frontier-моделей.

**Курс:** [[Stanford CS336/_index|Stanford CS336]]
**Лектор:** Tatsunori Hashimoto
**Связанные концепты:** [[Feed-Forward Network]], [[Scaling Laws]], [[Attention Mechanism]]

---

## Мотивация: зачем нужен MoE

Dense-модели (GPT-3, LLaMA) используют **все параметры** для каждого токена. При масштабировании это создаёт проблему: удвоение параметров = удвоение compute на inference.

**Вопрос:** можно ли увеличить ёмкость модели (capacity) без пропорционального роста compute?

**Ответ:** Mixture of Experts. Вместо одного большого FFN-слоя ставим несколько "экспертов" и роутер, который направляет каждый токен только к части экспертов.

```
Dense model:    каждый токен → все параметры
MoE model:     каждый токен → 2 из 64 экспертов
```

**Результат:** модель с 400B параметрами может иметь compute-cost как у 70B dense-модели.

---

## Архитектура MoE

### Базовая структура

В стандартном Transformer каждый блок содержит:
1. Multi-Head [[Self-Attention]]
2. [[Feed-Forward Network]] (FFN)

В MoE-варианте **FFN заменяется** на MoE-слой:

```
Input
  │
  ▼
[Self-Attention] ← остаётся без изменений
  │
  ▼
[Router/Gate] → выбирает Top-K экспертов
  │
  ├──▶ Expert 1 (FFN)
  ├──▶ Expert 2 (FFN)   ← активны только выбранные
  ├──▶ ...
  └──▶ Expert N (FFN)
  │
  ▼
Weighted sum выходов активных экспертов
```

### Router (Gating Network)

Router — это линейный слой + softmax, который для каждого токена определяет вероятность отправки к каждому эксперту:

```
g(x) = softmax(W_g · x)          # вероятности для N экспертов
Top-K(g(x))                       # выбираем K экспертов с max вероятностью
output = Σ g_i(x) · Expert_i(x)  # взвешенная сумма
```

Типичные значения: N = 8-128 экспертов, K = 1-2 активных на токен.

### Эксперты

Каждый эксперт — стандартный FFN (два линейных слоя + активация), идентичный по структуре, но с разными весами:

```
Expert_i(x) = W_2^i · σ(W_1^i · x + b_1^i) + b_2^i
```

---

## Ключевые проблемы MoE

### 1. Load Balancing

**Проблема:** router может "коллапсировать" — отправлять все токены к 1-2 экспертам, игнорируя остальные. Это называется **expert collapse**.

**Решение — Auxiliary Load Balancing Loss:**

```
L_balance = α · N · Σ_i (f_i · p_i)
```

где:
- f_i — доля токенов, отправленных к эксперту i
- p_i — средняя вероятность роутинга к эксперту i
- α — коэффициент (обычно 0.01)

Эта loss штрафует неравномерное распределение токенов.

### 2. Communication Overhead

В распределённом тренинге разные эксперты живут на разных GPU. Роутинг токенов между GPU создаёт **all-to-all communication** — одну из самых дорогих коллективных операций.

```
GPU 0: Experts 1-8     ← токены с любого GPU могут прийти сюда
GPU 1: Experts 9-16    ← и сюда тоже
GPU 2: Experts 17-24
...
```

### 3. Expert Capacity

Для эффективности на GPU каждый эксперт обрабатывает фиксированное число токенов (capacity). Токены сверх лимита **дропаются** (token dropping). Это создаёт trade-off:

- Маленький capacity → потеря токенов
- Большой capacity → неэффективное использование GPU

---

## Модели-примеры

### Mixtral 8x7B (Mistral AI, 2023)

- 8 экспертов, Top-2 роутинг
- Всего ~47B параметров, активных ~13B на токен
- Каждый эксперт — полный FFN слой
- Побеждает LLaMA 2 70B при 5x меньшем compute

### DeepSeek-V2/V3 (DeepSeek, 2024-2025)

Инновации DeepSeek в MoE:

- **Fine-grained experts:** 256 маленьких экспертов вместо 8 больших
- **Shared experts:** часть экспертов активна для **всех** токенов (shared knowledge)
- **Top-6 из 160** (DeepSeek-V3): больше гранулярность при том же compute
- **Auxiliary-loss-free load balancing:** вместо явной loss используется bias-коррекция

```
DeepSeek-V3:
  671B total params
  37B active params per token
  256 routed experts + 1 shared expert
```

### GShard и Switch Transformer (Google)

- **Switch Transformer:** Top-1 роутинг (один эксперт на токен) — максимальная простота
- **GShard:** масштабирование MoE до 600B параметров с expert parallelism

---

## Expert Parallelism

MoE требует специального подхода к параллелизму. В дополнение к стандартным:

- **Data Parallelism** — копии модели на разных GPU
- **Tensor Parallelism** — разрезание матриц внутри слоя
- **Pipeline Parallelism** — разные слои на разных GPU

MoE добавляет:

- **Expert Parallelism** — разные эксперты на разных GPU

```
GPU 0: Layers 1-4, Experts 1-32
GPU 1: Layers 1-4, Experts 33-64
GPU 2: Layers 5-8, Experts 1-32
GPU 3: Layers 5-8, Experts 33-64
```

Комбинация EP + TP + DP — стандарт для тренировки frontier MoE-моделей.

---

## MoE vs Dense: когда что использовать

| Аспект | Dense | MoE |
|--------|-------|-----|
| Training FLOPS на токен | Высокие | Низкие (при том же quality) |
| Memory | Пропорц. параметрам | Все эксперты в памяти |
| Inference latency | Предсказуемый | Может варьироваться |
| Сложность инфраструктуры | Простая | Высокая (all-to-all) |
| Scaling efficiency | Хорошая | Отличная |

**Правило:** MoE выгоден при большом compute-бюджете. Для маленьких моделей (<7B) dense обычно лучше.

---

## Тренды 2025-2026

1. **MoE — архитектура по умолчанию** для frontier-моделей (GPT-5, Gemini, DeepSeek-V3)
2. **Fine-grained experts** (DeepSeek) вытесняют coarse-grained (Mixtral)
3. **Shared experts** становятся стандартом
4. **MoE для inference** оптимизируется через expert offloading на CPU/SSD
5. **Sparse Upcycling** — инициализация MoE из обученной dense-модели

---

## Ключевые выводы

1. MoE увеличивает ёмкость модели **без пропорционального роста compute**
2. Роутер выбирает Top-K экспертов для каждого токена
3. Load balancing — критическая проблема, решается auxiliary loss или bias-коррекцией
4. Expert parallelism — необходимый вид параллелизма для MoE
5. DeepSeek показал, что fine-grained experts + shared experts = state-of-the-art

---

## Источники

- Stanford CS336, Lecture 4 — https://cs336.stanford.edu/
- Maarten Grootendorst, "A Visual Guide to MoE" — https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts
- NVIDIA, "Applying MoE in LLM Architectures" — https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/
- DeepSeek-V3 Technical Report (2025) — https://arxiv.org/abs/2401.06066

---

**См. также:** [[Feed-Forward Network]], [[Scaling Laws]], [[CS336 — Scaling Laws]], [[CS336 — Inference]]
