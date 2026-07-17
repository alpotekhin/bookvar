---
title: "Mistral 7B"
aliases: [Mistral, Mistral-7B, Mistral 7B]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Mistral 7B|Mistral 7B]]"
courses: []
sources:
  - "[Mistral AI — Announcing Mistral 7B](https://mistral.ai/news/announcing-mistral-7b)"
  - "[Bradney Smith — Mistral 7B Explained (TDS)](https://medium.com/data-science/mistral-7b-explained-towards-more-efficient-language-models-7f9c6e6b7251)"
  - "[Michael Brenndoerfer — Mistral Architecture: Sliding Window Attention](https://mbrenndoerfer.com/writing/mistral-architecture-sliding-window-attention)"
---

# Mistral 7B

## Зачем это нужно: LLM — трёхмерная задача

До Mistral 7B (октябрь 2023) масштабирование LLM воспринималось как **двумерная задача**: capabilities vs. training cost. Хочешь лучше — делай больше. LLaMA 2 13B лучше LLaMA 2 7B, 70B лучше 13B.

Mistral AI показали **третье измерение**: **inference cost**. Правильный архитектурный дизайн позволяет 7B модели *бить* 13B модель **на всех бенчмарках** и приближаться к 34B. Не за счёт лучших данных или больших compute при обучении — а за счёт **эффективных архитектурных решений** при инференсе.

Три ключевых компонента — **SWA + GQA + Rolling Buffer Cache** — стали «тройкой эффективности», которую позже переняли практически все production LLM.

## Sliding Window Attention (SWA): O(n) вместо O(n^2)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mistral-7b/sliding-window-attention.png]]
*Sliding Window Attention: каждый токен attend только к W=3 предыдущим токенам в одном слое. Через стек слоёв информация распространяется на расстояние k*W (источник: оригинальная статья)*

### Проблема vanilla attention
В стандартном Transformer каждый токен attend ко **всем предыдущим** — матрица attention размером $n \times n$. При 16K токенов это 256M элементов, при 32K — 1B. Это дорого по compute и по памяти (KV-cache растёт линейно).

### Решение: ограниченное окно
Каждый токен attend только к $W = 4096$ **предыдущим токенам в одном слое**:

$$\text{Attention}(h_i^k) = \text{softmax}\left(\frac{q_i^k \cdot [k_{i-W}^{k-1}, \ldots, k_i^{k-1}]^T}{\sqrt{d_k}}\right) \cdot [v_{i-W}^{k-1}, \ldots, v_i^{k-1}]$$

### Почему это работает: рекурсивное распространение

Ключевая идея: **информация распространяется через стек слоёв**. Скрытое состояние на позиции $i$ в слое $k$ содержит информацию от токенов в окне $[i-W, i]$ предыдущего слоя. Рекурсивно, после $k$ слоёв информация покрывает до $k \times W$ токенов.

При 32 слоях и $W = 4096$:

$$\text{Теоретический attention span} = 32 \times 4096 = 131{,}072 \text{ токенов}$$

Хотя формально каждый слой видит только 4K, **эффективный** контекст — ~131K. Это работает аналогично тому, как receptive field в CNN растёт с глубиной.

### Speedup
Модификации FlashAttention и xFormers для SWA дают **2x ускорение** по сравнению с vanilla attention на длине 16K.

## Rolling Buffer Cache: фиксированная память

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mistral-7b/rolling-buffer-cache.png]]
*Rolling Buffer Cache: KV-cache фиксированного размера W. Позиция i хранится в ячейке i mod W. При переполнении старые значения перезаписываются (источник: оригинальная статья)*

### Проблема стандартного KV-cache
В vanilla Transformer KV-cache растёт **линейно** с длиной последовательности. При 32K токенов на sequence — 8x overhead по памяти по сравнению с 4K.

### Решение: кольцевой буфер

Фиксированный размер cache = $W$ (размер окна). Ключи и значения для позиции $i$ хранятся в позиции $i \bmod W$:

```
Position:  0  1  2  3  4  5  6  7  ...
Cache idx: 0  1  2  3  0  1  2  3  ...  (при W=4)
```

Когда $i > W$, старые значения **перезаписываются**. Размер cache не растёт — он фиксирован.

При последовательности 32K токенов и $W = 4096$:
- Vanilla cache: 32K entries
- Rolling buffer: 4K entries
- Экономия: **8x по памяти** без потери качества

## Grouped-Query Attention (GQA): баланс скорости и качества

### Спектр KV-sharing

| Метод | KV heads | Query heads | Ratio | Память |
|-------|----------|-------------|-------|--------|
| Multi-Head Attention (MHA) | 32 | 32 | 1:1 | Максимум |
| **Grouped-Query Attention (GQA)** | **8** | **32** | **1:4** | **4x меньше** |
| Multi-Query Attention (MQA) | 1 | 32 | 1:32 | 32x меньше |

Mistral 7B: **8 KV-heads на 32 query-heads** (ratio 1:4). Каждая группа из 4 query-голов разделяет одну пару Key-Value.

### Преимущества GQA
1. **Ускорение inference** — меньше KV-cache = меньше memory reads = быстрее
2. **Больший batch size** — экономия памяти позволяет обрабатывать больше запросов параллельно
3. **Минимальная потеря качества** — в отличие от MQA, GQA сохраняет достаточно разнообразия KV-представлений

## Pre-fill и Chunking: эффективная работа с промптами

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mistral-7b/prefill-chunking.png]]
*Pre-fill with chunking: длинный промпт разбивается на chunk'и размером W. Для каждого chunk'а attention считается по cache (sliding window) + внутри chunk'а (causal mask) (источник: оригинальная статья)*

При генерации промпт известен заранее — можно **предзаполнить** KV-cache. Если промпт длиннее $W$, он разбивается на chunk'и размера $W$. Для каждого chunk'а:
- **Causal attention** внутри chunk'а (правый блок)
- **Sliding window attention** к cache (центральный блок)
- **Нет attention** к старым токенам за пределами окна (левый блок)

## Тройка эффективности: как компоненты работают вместе

```
SWA  →  O(n*W) compute вместо O(n^2)  →  быстрее на длинных seq
 ↓
Rolling Buffer  →  O(W) memory вместо O(n)  →  фиксированный cache
 ↓
GQA  →  4x меньше KV-heads  →  ещё меньше memory, больший batch
```

Все три компонента **мультипликативны**: SWA уменьшает compute, Rolling Buffer фиксирует cache size, GQA ещё дополнительно сжимает cache. Вместе — радикальное ускорение без потери качества.

## Архитектура

| Параметр | Значение |
|----------|----------|
| dim | 4096 |
| n_layers | 32 |
| head_dim | 128 |
| hidden_dim | 14336 |
| n_heads | 32 |
| n_kv_heads | 8 (GQA) |
| window_size | 4096 |
| context_len | 8192 |
| vocab_size | 32000 |
| **Total params** | **~7B** |

## Бенчмарки: 7B бьёт 13B

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mistral-7b/benchmark-comparison.png]]
*Mistral 7B vs. LLaMA 2 (7B/13B) на различных бенчмарках. Mistral 7B превосходит LLaMA 2 13B на всех метриках (источник: оригинальная статья)*

| Модель | MMLU | HumanEval | GSM8K | MATH | MBPP | HellaSwag |
|--------|------|-----------|-------|------|------|-----------|
| LLaMA 2 7B | 44.4% | 11.6% | 16.0% | 3.9% | 26.1% | 77.1% |
| LLaMA 2 13B | 55.6% | 18.9% | 34.3% | 6.0% | 35.4% | 80.7% |
| Code-LLaMA 7B | 36.9% | 31.1% | 20.8% | 5.2% | 52.5% | 62.9% |
| **Mistral 7B** | **60.1%** | **30.5%** | **52.2%** | **13.1%** | **47.5%** | **81.3%** |

### Ключевые сравнения

**Mistral 7B vs. LLaMA 2 13B** — превосходит на **всех** метриках, при вдвое меньшем количестве параметров:
- MMLU: +4.5% (60.1 vs. 55.6)
- GSM8K: +17.9% (52.2 vs. 34.3) — драматический разрыв на математике
- HumanEval: +11.6% (30.5 vs. 18.9) — код

**Mistral 7B vs. LLaMA 1 34B** — превосходит на reasoning, математике, коде. Уступает только на knowledge-intensive задачах (ограничение 7B по ёмкости для хранения фактов).

### Equivalent Model Size

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mistral-7b/effective-model-sizes.png]]
*Effective model sizes: Mistral 7B показывает производительность, ожидаемую от LLaMA 2 модели >3x его размера на reasoning и STEM (источник: оригинальная статья)*

На reasoning и STEM (MMLU) Mistral 7B эквивалентен LLaMA 2 модели **>3x его размера** (~21B+). На knowledge benchmarks — ~1.9x (ограничение по объёму хранимых фактов).

## Mistral 7B Instruct

Fine-tuned на публичных instruction datasets (без проприетарных данных):
- **MT-Bench: 6.84** — превосходит все 7B chat модели
- Сопоставим с LLaMA 2 13B Chat (6.65) и Vicuna 13B (6.57)
- **Chatbot Arena Elo: 1031** — между WizardLM 13B (1047) и LLaMA 2 13B Chat (1012)

Это **baseline** instruct — без сложных рецептов обучения. Демонстрирует потенциал базовой модели для fine-tuning.

## Почему 7B может бить 13B: analysis

1. **Архитектурная эффективность** — SWA + GQA не просто ускоряют инференс. При одинаковом compute бюджете можно тренировать **дольше** на **большем количестве данных**, потому что каждый training step дешевле.

2. **Лучшее использование параметров** — hidden_dim = 14336 (больше, чем у LLaMA 2 13B: 13824). Более широкие FFN позволяют эффективнее использовать каждый параметр.

3. **Modern training recipe** — хотя авторы не раскрывают детали обучения, модель демонстрирует паттерн, характерный для хорошо подобранных training data: непропорционально хорошие результаты на коде и математике при относительно скромном размере.

## Наследие

Mistral 7B стал одной из самых влиятельных open-source моделей:

- **Основа для [[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral 8x7B]]** — тот же decoder, но с MoE
- **SWA + GQA** стали стандартом для всех последующих efficient LLM
- **Apache 2.0** — полностью открытая лицензия
- Массовый fine-tuning community (Zephyr, OpenHermes, Dolphin)

## Сравнение attention-модификаций

| Механизм | Compute/токен | KV-cache memory | Effective context | Используется в |
|----------|-------------|-----------------|-------------------|---------------|
| Full Attention | $O(n \cdot d)$ | $O(n \cdot d)$ | Весь контекст | GPT, LLaMA |
| SWA | $O(W \cdot d)$ | $O(W \cdot d)$ | $W \times k$ layers | **Mistral**, Mixtral |
| SWA + Rolling Buffer | $O(W \cdot d)$ | **$O(W \cdot d)$ fixed** | $W \times k$ layers | **Mistral** |
| SWA + GQA + Rolling | $O(W \cdot d)$ | **$O(W \cdot d / 4)$ fixed** | $W \times k$ layers | **Mistral** |

Каждый компонент делает следующий ещё эффективнее. SWA ограничивает compute и размер cache. Rolling Buffer фиксирует cache. GQA сжимает cache в 4 раза. Результат: при 32K контексте Mistral использует **в 32 раза меньше cache-памяти**, чем vanilla Transformer с MHA.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — базовая архитектура
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA 2|LLaMA 2]] — главный конкурент для сравнения
- [[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral of Experts]] — MoE расширение Mistral
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — Rolling Buffer Cache оптимизирует KV-cache
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — SWA реализован через модификации FlashAttention
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — SWA и GQA — модификации стандартного attention

## Дополнительные ресурсы

- [Mistral AI — Announcing Mistral 7B](https://mistral.ai/news/announcing-mistral-7b) — официальный анонс
- [Bradney Smith — Mistral 7B Explained](https://medium.com/data-science/mistral-7b-explained-towards-more-efficient-language-models-7f9c6e6b7251) — подробный разбор архитектуры
- [Michael Brenndoerfer — Mistral Architecture: SWA](https://mbrenndoerfer.com/writing/mistral-architecture-sliding-window-attention) — интерактивные визуализации SWA
- [GitHub — mistralai/mistral-src](https://github.com/mistralai/mistral-src) — референсная реализация
