---
title: "PEFT"
aliases: [PEFT, Parameter-Efficient Fine-Tuning, параметрически-эффективная настройка]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/LoRA|LoRA]]"
  - "[[02 Areas/ML & DL/Papers/QLoRA|QLoRA]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[HuggingFace PEFT Library](https://github.com/huggingface/peft)"
  - "[Lialin et al. — Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning (2023)](https://arxiv.org/abs/2303.15647)"
  - "[Houlsby et al. — Parameter-Efficient Transfer Learning for NLP (2019)](https://arxiv.org/abs/1902.00751)"
  - "[Li & Liang — Prefix-Tuning (2021)](https://arxiv.org/abs/2101.00190)"
  - "[Lester et al. — Prompt Tuning (2021)](https://arxiv.org/abs/2104.08691)"
---

# PEFT (Parameter-Efficient Fine-Tuning)

## Зачем это нужно: проблема полного fine-tuning

Представь, что у тебя GPT-3 на 175 миллиардов параметров. Full fine-tuning означает:
- **350 GB** только на хранение весов (FP16)
- **>780 GB** VRAM для обучения (веса + оптимизатор + градиенты + активации)
- Для каждой новой задачи — **отдельная копия** всех 175B параметров

Если у тебя 10 клиентов с разными задачами — это 3.5 TB на диске только для моделей. На практике это невозможно для большинства организаций.

**PEFT** — семейство методов, которые решают эту проблему радикально: замораживаем основную модель и обучаем лишь **малую долю параметров** (от 0.001% до 1%). Одна базовая модель, множество лёгких адаптеров — переключение между задачами на лету.

## Таксономия PEFT-методов

![[02 Areas/ML & DL/raw/papers/peft/images/peft-taxonomy.png]]
*Таксономия PEFT-методов: additive (adapters, soft prompts), reparametrization-based (LoRA, KronA), selective (BitFit, Fish-Mask). Многие методы комбинируют подходы (источник: HuggingFace PEFT Blog)*

## LoRA: стандарт де-факто

### Идея

Hu et al. (2021) заметили, что **изменение весов при fine-tuning имеет низкий ранг**. Вместо обновления полной матрицы $W_0 \in \mathbb{R}^{d \times k}$, LoRA параметризует обновление как произведение двух маленьких матриц:

$$W = W_0 + \Delta W = W_0 + BA$$

где $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, и $r \ll \min(d, k)$.

![[02 Areas/ML & DL/raw/papers/peft/images/lora-architecture.gif]]
*LoRA: замороженные pretrained веса $W \in \mathbb{R}^{d \times d}$ + обучаемые low-rank матрицы $B$ (инициализация нулями) и $A$ (Gaussian). При inference $BA$ сливается с $W_0$ — нулевой overhead (источник: HuggingFace PEFT Blog)*

### Почему это работает

Инициализация: $A$ — случайная Gaussian, $B$ — нули. В начале обучения $\Delta W = BA = 0$, модель стартует ровно с pretrained весов. Масштабирование: $\Delta W$ домножается на $\alpha / r$, где $\alpha$ — гиперпараметр (обычно $\alpha = r$ или $2r$).

### Какие слои адаптировать

Из оригинальной статьи (Table 5): адаптация **$W_q$ и $W_v$** даёт лучший результат при фиксированном бюджете параметров. Однако QLoRA показала, что адаптация **всех linear layers** (Q, K, V, O, FFN up/down) работает ещё лучше при достаточном ранге.

### Числа

На GPT-3 175B:
- **Обучаемых параметров**: 4.7M (0.003%)
- **Размер чекпоинта**: ~35 MB (вместо 350 GB)
- **Inference latency**: = baseline (после merge BA в $W_0$)

## QLoRA: LoRA для одной GPU

Dettmers et al. (2023) сделали fine-tuning LLaMA 65B возможным на **одной 48 GB GPU**. Три инновации:

1. **4-bit NormalFloat (NF4)** квантизация замороженных весов — оптимальный data type для normally distributed weights
2. **Double quantization** — квантизация самих quantization констант (экономит ~0.37 бит/параметр)
3. **Paged optimizers** — автоматический offload optimizer states в CPU RAM при нехватке GPU памяти

Ключевое отличие от оригинальной LoRA: QLoRA применяет адаптеры ко **всем** linear layers, а не только к $W_q$/$W_v$. Это компенсирует потери от квантизации.

**Результат**: QLoRA 65B на одной GPU достигает **99.3% качества** full 16-bit fine-tuning на всех бенчмарках.

## Adapter Layers: bottleneck-модули

Houlsby et al. (2019) вставляют маленькие модули внутрь каждого Transformer блока:

![[02 Areas/ML & DL/raw/papers/peft/images/adapter-bottleneck.png]]
*Adapter layers внутри Transformer блока: bottleneck-модули (down-projection → nonlinearity → up-projection + residual) добавляются после Multi-Head Attention и FFN подслоёв (источник: AdapterHub)*

Down-projection: $\mathbb{R}^d \to \mathbb{R}^m$ (где $m \ll d$, обычно $m = 64$).
Up-projection: $\mathbb{R}^m \to \mathbb{R}^d$.

**Плюс**: хорошее качество, стабильное обучение.
**Минус**: добавляет **inference latency** (+20-30% при batch=1), потому что адаптеры не merge-ятся в основные веса — они последовательны.

## Prefix Tuning: virtual tokens в attention

Li & Liang (2021): вместо дополнительных слоёв, добавляем обучаемые continuous vectors (virtual tokens) **напрямую в Key и Value** каждого attention layer:

$$\text{head}_i = \text{Attention}(x W_i^Q, [P_K; x W_i^K], [P_V; x W_i^V])$$

где $P_K, P_V \in \mathbb{R}^{l \times d}$ — обучаемые prefix-матрицы длины $l$.

**Проблема**: нестабильное обучение. Авторы вводят reparametrization через MLP ($P = \text{MLP}(\theta)$), которую убирают после обучения.
**Ещё проблема**: prefix **сокращает доступную длину контекста** на $l$ токенов.

## Prompt Tuning: простейший PEFT

Lester et al. (2021): ещё проще — обучаемые soft prompts добавляются **только к input embeddings** (а не к каждому слою, как в Prefix Tuning):

$$\text{input} = [\underbrace{p_1, p_2, \ldots, p_k}_{\text{soft prompt}}; \underbrace{x_1, x_2, \ldots, x_n}_{\text{реальный вход}}]$$

Менее выразителен на маленьких моделях, но при масштабировании **сходится к качеству full fine-tuning**: на T5-XXL (11B) Prompt Tuning с 20 soft tokens достигает паритета с full FT.

## Сводная таблица

| Метод | Обучаемых параметров (GPT-3 175B) | VRAM | Inference latency | Merge в веса | Стабильность |
|-------|-----------------------------------|------|-------------------|-------------|-------------|
| Full FT | 175B (100%) | >780 GB | baseline | N/A | высокая |
| **LoRA** | **4.7M (0.003%)** | **~350 GB** | **= baseline** | **да** | **высокая** |
| **QLoRA** | **4.7M (0.003%)** | **<48 GB** | **= baseline** | **да** | **высокая** |
| Adapters | ~40M (~0.02%) | ~350 GB | +20-30% | нет | высокая |
| Prefix Tuning | ~10M (~0.006%) | ~350 GB | = baseline | нет | низкая |
| Prompt Tuning | ~0.1M | ~350 GB | = baseline | нет | средняя |
| BitFit | ~0.1M | ~350 GB | = baseline | нет | средняя |

## Когда использовать какой метод

```
Нужен fine-tuning LLM?
│
├── Бюджет GPU ≥ 4× размер модели (FP16)?
│   └── LoRA (rank 16-64, все linear layers)
│
├── Одна consumer GPU (24-48 GB)?
│   └── QLoRA (4-bit base + LoRA rank 64)
│
├── Нужно переключать адаптеры на лету без merge?
│   └── Adapter Layers (если latency не критична)
│   └── LoRA без merge (если latency критична)
│
├── Модель > 10B и нельзя трогать веса?
│   └── Prompt Tuning (если модель достаточно большая)
│
└── Быстрый эксперимент, baseline?
    └── LoRA rank 8, только Wq/Wv
```

## Production: multi-tenant serving

Ключевое преимущество LoRA для production — **один базовый набор весов + множество адаптеров**:

```
Base Model (LLaMA 70B, ~140 GB)
├── Adapter: Customer A (sentiment, ~35 MB)
├── Adapter: Customer B (NER, ~35 MB)
├── Adapter: Customer C (summarization, ~35 MB)
└── Adapter: Customer D (translation, ~35 MB)

Итого: 140 GB + 140 MB вместо 560 GB
```

Frameworks типа [LoRAX](https://github.com/predibase/lorax) и [S-LoRA](https://arxiv.org/abs/2311.03285) позволяют serving нескольких LoRA-адаптеров одновременно с минимальным overhead.

## Хронология

| Год | Milestone | Метод |
|-----|-----------|-------|
| 2019 | Adapter layers для NLP | Houlsby et al. |
| 2021 | Prefix Tuning | Li & Liang |
| 2021 | Prompt Tuning | Lester et al. |
| **2021** | **LoRA** | **Hu et al.** |
| 2023 | QLoRA — 65B на одной GPU | Dettmers et al. |
| 2023 | S-LoRA — multi-tenant serving | Sheng et al. |
| 2024 | DoRA, LoRA+ — улучшения LoRA | Liu et al., Hayou et al. |

## Key papers

- [[02 Areas/ML & DL/Papers/LoRA|LoRA]] — low-rank adaptation с нулевым inference overhead
- [[02 Areas/ML & DL/Papers/QLoRA|QLoRA]] — 4-bit квантизация + LoRA, fine-tuning 65B на одной GPU

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] — основной метод PEFT
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — полный fine-tuning, от которого PEFT отличается
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — часто использует PEFT
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, к которой применяются PEFT методы

## Дополнительные ресурсы

- [HuggingFace PEFT Library](https://github.com/huggingface/peft) — стандартная библиотека для LoRA/Prefix/Prompt Tuning
- [Lialin et al. — Scaling Down to Scale Up (2023)](https://arxiv.org/abs/2303.15647) — лучший обзор всех PEFT методов
- [Sebastian Raschka — Finetuning LLMs with LoRA](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) — практический tutorial
