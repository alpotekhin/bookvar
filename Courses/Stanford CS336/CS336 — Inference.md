---
title: "CS336 — Inference"
type: course-note
course: "Stanford CS336"
---

# CS336 — Inference

> Лекция 10 курса Stanford CS336. Как сделать генерацию быстрой и дешёвой.

**Курс:** [[Stanford CS336/_index|Stanford CS336]]
**Лектор:** Percy Liang
**Связанные концепты:** [[KV-Cache]], [[Speculative Decoding]], [[Flash Attention]], [[Attention Mechanism]]

---

## Почему inference — это проблема

Тренировка LLM — разовые затраты. Inference — **постоянные**. Для модели, обслуживающей миллионы пользователей, стоимость inference может превышать стоимость тренировки в 10-100x за время жизни модели.

**Две фазы inference:**

1. **Prefill** — обработка входного промпта (параллельно, compute-bound)
2. **Decode** — генерация токенов один за одним (последовательно, memory-bound)

```
Промпт: "Объясни теорию относительности"  ← prefill: все токены сразу
                                            ↓
Генерация: "Теория" → "относительности" → "Эйнштейна" → ...  ← decode: по одному
```

Decode-фаза — bottleneck. Каждый новый токен требует прохода через всю модель, но использует GPU на ~1-5% от пиковой производительности (memory bandwidth limited).

---

## KV-Cache

### Проблема без кэша

При генерации токена t модель вычисляет attention ко всем предыдущим токенам. Без оптимизации нужно **пересчитывать** Key и Value для всех прошлых токенов — O(t) лишней работы на каждом шаге.

### Решение

Key и Value для уже обработанных позиций **не меняются** (в causal attention). Кэшируем их:

```
Шаг 1: вычисляем K₁, V₁ → сохраняем
Шаг 2: вычисляем K₂, V₂ → сохраняем, используем K₁V₁ из кэша
Шаг 3: вычисляем K₃, V₃ → сохраняем, используем K₁V₁, K₂V₂ из кэша
...
```

**Экономия:** с O(t^2) compute на позицию t → O(t) (только attention нового токена ко всем предыдущим).

### Размер KV-Cache

```
KV_size = 2 × n_layers × n_heads × d_head × seq_len × batch_size × bytes_per_element
```

Для LLaMA 70B, seq_len=4096, batch=1, fp16:
```
2 × 80 × 64 × 128 × 4096 × 1 × 2 bytes ≈ 10.7 GB
```

Это **огромное** потребление памяти, растущее линейно с длиной контекста.

### Оптимизации KV-Cache

**Multi-Query Attention (MQA):**
Все attention heads используют **одну и ту же** пару K, V. Размер кэша уменьшается в n_heads раз.

**Grouped-Query Attention (GQA):**
Компромисс: группы голов делят K, V. LLaMA 2 70B использует 8 KV-голов вместо 64.

```
MHA: 64 Q-heads, 64 KV-heads  → 100% KV-cache
GQA: 64 Q-heads, 8 KV-heads   → 12.5% KV-cache
MQA: 64 Q-heads, 1 KV-head    → 1.6% KV-cache
```

**KV-Cache Quantization:**
Хранение KV в FP8 или INT4 вместо FP16. NVIDIA показала, что NVFP4 KV-cache сохраняет качество при 4x сжатии.

---

## Speculative Decoding

### Идея

Autoregressive decoding — последовательный: один токен за шаг. GPU простаивает. Speculative decoding разрушает этот bottleneck:

1. **Draft model** (маленькая, быстрая) генерирует K токенов-кандидатов
2. **Target model** (большая, точная) **верифицирует** все K токенов за один forward pass
3. Принятые токены сохраняются, с точки отклонения генерация перезапускается

```
Draft model (1B):    "The" → "cat" → "sat" → "on" → "the"   (5 токенов быстро)
Target model (70B):  проверяет все 5 за 1 pass
                     ✓ "The" ✓ "cat" ✓ "sat" ✗ "on" → заменяет на "upon"
Результат: 4 токена за ~1 шаг target model
```

### Математическая корректность

Speculative decoding **не меняет** распределение генерируемых токенов — output идентичен обычному autoregressive decoding. Это достигается через rejection sampling:

```
Если p_target(token) ≥ p_draft(token):
    accept с вероятностью 1
Если p_target(token) < p_draft(token):
    accept с вероятностью p_target(token) / p_draft(token)
    reject → sample из (p_target - p_draft) / Z
```

### Практические результаты

- **2-3x speedup** на типичных задачах
- Лучше работает, когда draft и target модели "согласованы"
- vLLM и TensorRT-LLM имеют встроенную поддержку
- NVIDIA показала 3.6x throughput improvement на H200

---

## Batching и Scheduling

### Continuous Batching

Классический подход: ждём, пока соберётся batch из N запросов, обрабатываем. Проблема: запросы разной длины, короткие ждут длинных.

**Continuous batching** (vLLM, TGI): новые запросы добавляются в batch на лету, завершённые — удаляются. GPU всегда загружен.

```
Время →
Batch slot 1: [req1 ████████]  [req4 ████]  [req7 ██████]
Batch slot 2: [req2 ████]  [req5 ██████████]  [req8 ██]
Batch slot 3: [req3 ██████]  [req6 ████]  [req9 ████████]
```

### PagedAttention (vLLM)

KV-cache управляется как виртуальная память в OS:

- KV-cache разбит на **блоки** (pages) фиксированного размера
- Блоки выделяются по мере генерации (не заранее на max_seq_len)
- Общие prefix (system prompt) могут **шариться** между запросами

**Результат:** ~2-4x больше throughput за счёт эффективного использования GPU memory.

---

## Quantization для Inference

### Идея

Тренировка в FP32/BF16 → inference в INT8/INT4. Меньше памяти, быстрее compute.

### Методы

**Post-Training Quantization (PTQ):**
- GPTQ — weight-only quantization, калибровка на малом dataset
- AWQ — activation-aware, защищает "важные" каналы от квантизации

**Weight-Only vs Weight+Activation:**
- INT4 weights + FP16 activations — стандарт для on-device deployment
- INT8 weights + INT8 activations — для серверного inference (TensorRT-LLM)

### Результаты

```
LLaMA 70B FP16:    ~140 GB VRAM, 2x A100 80GB
LLaMA 70B INT4:    ~35 GB VRAM, 1x A100 80GB
LLaMA 70B INT4:    ~35 GB VRAM → помещается даже на consumer GPU
```

Деградация качества при INT4 обычно <1% на бенчмарках.

---

## Оптимизация Attention

### [[Flash Attention]]

Стандартная реализация attention:
1. Вычисляем QK^T → O(n^2) памяти для матрицы attention scores
2. Softmax
3. Умножаем на V

**Flash Attention** — I/O-aware алгоритм:
- Разбивает вычисление на тайлы (tiles), помещающиеся в SRAM
- Никогда не материализует полную attention-матрицу в HBM
- Тот же результат, но ~2-4x быстрее и O(n) памяти

### Prefix Caching

Многие запросы начинаются с одного system prompt. Prefix caching сохраняет KV-cache для общего prefix и переиспользует между запросами.

```
System prompt (2000 tokens) → KV-cache вычисляется один раз
Запрос 1: system + user_1 → переиспользуем prefix KV
Запрос 2: system + user_2 → переиспользуем prefix KV
...
```

---

## Inference Stack: собираем всё вместе

Современный production inference стек:

```
Request Queue
    │
    ▼
Continuous Batching Scheduler
    │
    ▼
PagedAttention + KV-Cache (GQA, quantized)
    │
    ▼
Flash Attention kernels
    │
    ▼
Speculative Decoding (draft + target)
    │
    ▼
Quantized Weights (INT4/INT8)
    │
    ▼
Output tokens
```

Всё это реализовано в системах: **vLLM**, **TensorRT-LLM**, **SGLang**.

---

## Ключевые выводы

1. **KV-Cache** — обязательная оптимизация, GQA/MQA уменьшают его размер
2. **Speculative Decoding** даёт 2-3x ускорение без потери качества
3. **Continuous Batching** + **PagedAttention** = максимальный throughput
4. **Quantization** (INT4) позволяет запускать 70B модель на одном GPU
5. Inference-стоимость определяет экономику LLM-продуктов

---

## Источники

- Stanford CS336, Lecture 10 — https://cs336.stanford.edu/
- Introl, "Speculative Decoding Guide" — https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025
- NVIDIA, "Mastering LLM Techniques: Inference Optimization" — https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- Clarifai, "LLM Inference Optimization Techniques" — https://www.clarifai.com/blog/llm-inference-optimization/

---

**См. также:** [[KV-Cache]], [[Speculative Decoding]], [[Flash Attention]], [[CS336 — Scaling Laws]], [[CS336 — Tokenization]]
