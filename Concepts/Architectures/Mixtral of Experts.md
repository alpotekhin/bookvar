---
title: "Mixtral of Experts"
aliases: [Mixtral, Mixtral 8x7B, Sparse MoE, Mixture of Experts]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Mixtral of Experts|Mixtral of Experts]]"
courses: []
sources:
  - "[Hugging Face — Mixture of Experts Explained](https://huggingface.co/blog/moe)"
  - "[Maarten Grootendorst — A Visual Guide to Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)"
  - "[Shazeer et al. — Outrageously Large Neural Networks (2017)](https://arxiv.org/abs/1701.06538)"
  - "[Fedus et al. — Switch Transformers (2022)](https://arxiv.org/abs/2101.03961)"
---

# Mixtral of Experts

## Зачем это нужно: проблема масштабирования dense моделей

В dense моделях (GPT, LLaMA) **каждый параметр активируется при каждом токене**. Хочешь больше знаний — увеличивай параметры — растёт и compute. Это создаёт прямую связь: model capacity = inference cost.

Mixture of Experts (MoE) разрывает эту связь: модель может иметь **47B параметров** (огромная ёмкость), но активировать только **13B** при обработке каждого токена (низкая стоимость инференса). Mixtral 8x7B от Mistral AI (январь 2024) доказал, что этот подход работает на практике: при 5x меньшем active compute он превосходит или равен LLaMA 2 70B и GPT-3.5.

## История MoE: от 1991 до 2024

### 1991 — Jacobs et al.: Adaptive Mixture of Local Experts
Первая MoE архитектура. Несколько «экспертов» (отдельных сетей) обрабатывают разные подмножества данных. Gating network определяет веса экспертов. Обучение совместное.

### 2017 — Shazeer et al.: Sparsely-Gated MoE
**Прорыв масштаба.** «Outrageously Large Neural Networks» (соавторы: Hinton, Dean) масштабировали MoE до **137B параметров** в LSTM. Ключевые инновации:
- **Sparsity** — активируются только top-k экспертов (обычно 2 из тысяч)
- **Noisy Top-K Gating** — добавление гауссового шума к логитам для балансировки нагрузки
- **Auxiliary loss** — штраф за неравномерное распределение токенов между экспертами

### 2020 — GShard
Масштабирование до **600B+ параметров**. Top-2 routing, expert capacity (порог токенов на эксперта), overflow через residual connections. MoE слои разделяются между GPU, остальные — реплицируются.

### 2022 — Switch Transformers
**1.6 триллиона параметров** (2048 экспертов). Упрощённый routing: **один эксперт** на токен (vs. top-2). 4x speedup над T5-XXL. Expert Capacity = (tokens_per_batch / num_experts) * capacity_factor.

### 2022 — ST-MoE
Стабилизация обучения: **Router Z-loss** — штраф за большие логиты, предотвращающий числовую нестабильность. Обнаружена специализация экспертов в encoder (пунктуация, имена собственные).

### 2024 — Mixtral 8x7B
**Open-source MoE**, превзошедший GPT-3.5. Apache 2.0 лицензия. Стал стандартом для масштабирования: за ним последовали DeepSeek-MoE, Qwen2-MoE, Grok.

## Архитектура: как работает Sparse MoE

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mixtral-of-experts/smoe-layer.png]]
*Mixture of Experts Layer: каждый токен направляется к 2 из 8 экспертов. Выход — взвешенная сумма outputs выбранных экспертов (источник: оригинальная статья)*

### Базовая формула

Для входного токена $x$ с $n$ экспертами $\{E_0, E_1, \ldots, E_{n-1}\}$:

$$y = \sum_{i=0}^{n-1} G(x)_i \cdot E_i(x)$$

где $G(x)$ — gating function (router). Если $G(x)$ разрежена, выходы «выключенных» экспертов не вычисляются.

### Top-K Routing в Mixtral

$$G(x) = \text{Softmax}(\text{Top-K}(x \cdot W_g))$$

где $\text{Top-K}(\ell)_i = \ell_i$ если $\ell_i$ среди top-K наибольших, иначе $-\infty$.

В Mixtral: $K = 2$, $n = 8$. Каждый эксперт — полноценный SwiGLU FFN (hidden_dim = 14336):

$$y = \sum_{i=0}^{7} \text{Softmax}(\text{Top-2}(x \cdot W_g))_i \cdot \text{SwiGLU}_i(x)$$

Softmax нормализует **только по двум выбранным экспертам**, а не по всем восьми.

### Архитектурные параметры

| Параметр | Значение |
|----------|----------|
| dim | 4096 |
| n_layers | 32 |
| head_dim | 128 |
| hidden_dim | 14336 |
| n_heads | 32 |
| n_kv_heads | 8 (GQA) |
| context_len | 32768 |
| num_experts | 8 |
| top_k_experts | 2 |
| **Total params** | **~47B** |
| **Active params/token** | **~13B** |

Базовая архитектура — [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] (SWA, GQA), но с **FFN → MoE** заменой и расширением контекста до 32K.

## 47B total vs. 13B active: экономика MoE

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mixtral-of-experts/benchmark-comparison.png]]
*Mixtral 8x7B vs. LLaMA 2 (7B/13B/70B): при 13B active params Mixtral превосходит LLaMA 2 70B на большинстве бенчмарков (источник: оригинальная статья)*

Ключевой insight: **compute пропорционален active params, не total params**.

- Mixtral 8x7B: 47B total, 13B active → compute как у 13B dense модели
- LLaMA 2 70B: 70B total = 70B active → 5.4x больше compute

При этом Mixtral **превосходит** LLaMA 2 70B на математике, коде и мультилингвальных задачах. Память при inference пропорциональна 47B (нужно хранить всех экспертов), но это всё равно меньше 70B.

## Бенчмарки: Dense vs. Sparse

| Модель | Active Params | MMLU | HumanEval | GSM8K | MATH | MBPP |
|--------|-------------|------|-----------|-------|------|------|
| LLaMA 2 7B | 7B | 44.4% | 11.6% | 16.0% | 3.9% | 26.1% |
| LLaMA 2 13B | 13B | 55.6% | 18.9% | 34.3% | 6.0% | 35.4% |
| Mistral 7B | 7B | 62.5% | 26.2% | 50.0% | 12.7% | 50.2% |
| LLaMA 2 70B | 70B | 69.9% | 29.3% | 69.6% | 13.8% | 49.8% |
| **Mixtral 8x7B** | **13B** | **70.6%** | **40.2%** | **74.4%** | **28.4%** | **60.7%** |

Mixtral 8x7B превосходит LLaMA 2 70B при **5x меньшем active compute**. Особенно впечатляет разрыв на математике (+14.6% MATH) и коде (+10.9% HumanEval).

### Сравнение с GPT-3.5

| Бенчмарк | LLaMA 2 70B | GPT-3.5 | Mixtral 8x7B |
|----------|-------------|---------|-------------|
| MMLU | 69.9% | 70.0% | **70.6%** |
| HellaSwag | 87.1% | 85.5% | 86.7% |
| ARC-C | 85.1% | 85.2% | **85.8%** |
| GSM-8K | 53.6% | 57.1% | **58.4%** |
| MT Bench (Instruct) | 6.86 | 8.32 | **8.30** |

Mixtral 8x7B Instruct (SFT + [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]]) на Arena Elo 1121 — превосходит Claude-2.1, GPT-3.5-Turbo, Gemini Pro.

## Routing Analysis: что выучивают эксперты?

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mixtral-of-experts/routing-sample.png]]
*Визуализация routing: каждый токен подсвечен цветом выбранного эксперта. Одинаковые синтаксические конструкции (self в Python, Question в English) маршрутизируются к одним экспертам (источник: оригинальная статья)*

Удивительный результат: **эксперты НЕ специализируются по доменам**. Распределение экспертов практически идентично для ArXiv, PubMed, Philosophy и Wikipedia.

### Что вместо доменной специализации?

1. **Синтаксическая специализация** — одинаковые токены (ключевые слова, пунктуация) маршрутизируются к одним экспертам независимо от домена
2. **Temporal locality** — ~60-67% consecutive tokens идут к тому же эксперту в средних/последних слоях (при рандомном ожидании ~46%)
3. **DM Mathematics** — единственный домен с заметно другим распределением (синтетические данные)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mixtral-of-experts/expert-routing-analysis.png]]
*Распределение экспертов по доменам из The Pile. Пунктирная линия — равномерное распределение (1/8). Эксперты не специализируются по доменам (источник: оригинальная статья)*

### Implications для оптимизации
- Высокая locality → Expert Parallelism может создавать oversubscription
- Высокая locality → можно кэшировать экспертов для ускорения inference
- Синтаксическая (а не семантическая) специализация → каждый эксперт — не «эксперт по биологии», а скорее «эксперт по определённым паттернам токенов»

## Load Balancing: критическая проблема MoE

Без регуляризации router склонен направлять все токены к нескольким «любимым» экспертам. Остальные не обучаются → модель деградирует. Эволюция решений:

| Метод | Год | Идея |
|-------|-----|------|
| **Noisy Top-K** | 2017 (Shazeer) | Гауссов шум к логитам router |
| **Auxiliary loss** | 2017+ | Штраф за неравномерность |
| **Expert capacity** | 2020 (GShard) | Фиксированный порог токенов/эксперт |
| **Router Z-loss** | 2022 (ST-MoE) | Штраф за большие логиты |

Mixtral не раскрывает деталей load balancing, но использует top-2 routing, что естественно балансирует лучше, чем top-1 (каждый токен «голосует» за двух экспертов).

## Efficient Inference

- **Megablocks CUDA kernels** — MoE как sparse matrix multiplications. Эффективно обрабатывает переменное число токенов на эксперта
- **Expert Parallelism (EP)** — каждый эксперт на отдельном GPU. Токены маршрутизируются к нужному GPU, результат возвращается обратно
- **vLLM integration** — production-ready serving с полностью open-source стеком

## Long Range: 32K контекст

100% accuracy на Passkey Retrieval при 32K контекст — модель извлекает ключ из любой позиции в длинном промпте. Perplexity монотонно уменьшается с ростом контекста на proof-pile dataset.

## Хронология Sparse MoE

| Год | Milestone | Масштаб |
|-----|-----------|---------|
| 1991 | Jacobs — Adaptive MoE | Малые сети |
| 2017 | Shazeer — Sparsely-Gated MoE | 137B LSTM |
| 2020 | GShard | 600B+ |
| 2022 | Switch Transformer | 1.6T (2048 экспертов) |
| 2024 | **Mixtral 8x7B** | **47B, open-source** |
| 2024 | DeepSeek-MoE, Qwen2-MoE, Grok | Массовое adoption |

## Мультилингвальность

Mixtral значительно расширил долю мультилингвальных данных при pretrain по сравнению с Mistral 7B. Результаты на французском, немецком, испанском и итальянском **драматически превосходят** LLaMA 2 70B:

| Язык | LLaMA 2 70B (MMLU) | Mixtral 8x7B (MMLU) | Разрыв |
|------|-------------------|--------------------|----|
| French | 64.3% | **70.9%** | +6.6% |
| German | 64.2% | **71.5%** | +7.3% |
| Spanish | 66.0% | **72.5%** | +6.5% |
| Italian | 65.1% | **70.9%** | +5.8% |

Дополнительная ёмкость MoE позволяет модели хранить мультилингвальные знания без degradation на English.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — базовая архитектура, на которой построен Mixtral
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — фундамент, к которому добавляется MoE
- [[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]] — компонент, заменяемый на MoE layer
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — метод alignment для Mixtral Instruct
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — остаётся неизменным в MoE, заменяется только FFN

## Дополнительные ресурсы

- [Hugging Face — Mixture of Experts Explained](https://huggingface.co/blog/moe) — лучший обзор истории и техник MoE
- [Maarten Grootendorst — A Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — визуализации routing и архитектуры
- [Shazeer et al. — Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) — оригинальная статья Sparsely-Gated MoE
- [Fedus et al. — Switch Transformers](https://arxiv.org/abs/2101.03961) — масштабирование до 1.6T параметров
