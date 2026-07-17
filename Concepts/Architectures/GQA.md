---
title: "GQA"
aliases: [GQA, Grouped Query Attention, Grouped-Query Attention]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/LLaMA 2|LLaMA 2]]"
  - "[[02 Areas/ML & DL/Papers/Mistral 7B|Mistral 7B]]"
  - "[[02 Areas/ML & DL/Papers/Gemma|Gemma]]"
courses: []
sources:
  - "[GQA: Training Generalized Multi-Query Transformer Models (2023)](https://arxiv.org/abs/2305.13245)"
---

# GQA — Grouped Query Attention

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png]]
*Сравнение Multi-Head, Grouped-Query и Multi-Query Attention: GQA группирует query heads, разделяющие общие KV проекции (GQA paper, 2305.13245)*

## Проблема: KV-cache как узкое место inference

При авторегрессивной генерации [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] на каждом шаге вычисляет [[02 Areas/ML & DL/Concepts/Training/Attention Mechanism|Attention]] между новым токеном и всеми предыдущими. Чтобы не пересчитывать Key и Value проекции для всех предыдущих токенов, используется **KV-cache** — кеш матриц K и V, растущий линейно с длиной контекста.

Для стандартного Multi-Head Attention (MHA) размер KV-cache:

$$\text{KV-cache} = 2 \times L \times H \times d_k \times S$$

где $L$ — число слоёв, $H$ — число голов, $d_k$ — размерность головы, $S$ — длина последовательности. Для модели с 32 слоями, 32 головами, $d_k = 128$ и контекстом 8192 — это **4 GB на один запрос** (FP16). При batch serving для 32 запросов одновременно — **128 GB** только на KV-cache. Это часто превышает доступную GPU memory и становится bottleneck inference.

## Эволюция: MHA → MQA → GQA

### Multi-Head Attention (MHA)

Стандартный подход из оригинального Transformer. Каждая голова внимания имеет **собственные** проекции Query, Key и Value:

- **H query heads** + **H KV heads**
- Каждая голова независимо вычисляет attention
- Максимальная выразительность, но максимальный KV-cache

### Multi-Query Attention (MQA)

Предложена Noam Shazeer (2019). Все query heads **разделяют одну пару** KV проекций:

- **H query heads** + **1 KV head**
- KV-cache уменьшается в $H$ раз (например, 32x для 32-head модели)
- Существенный speedup при inference (меньше memory bandwidth)
- **Проблема:** заметная деградация качества — одна KV пара должна обслуживать все query heads, что ограничивает attention patterns

### Grouped Query Attention (GQA)

GQA (Ainslie et al., 2023) — **промежуточный вариант** между MHA и MQA:

- **H query heads** + **G KV heads**, где $1 < G < H$
- Query heads группируются: каждая группа из $H/G$ query heads разделяет одну пару KV проекций
- KV-cache уменьшается в $H/G$ раз

```
MHA (G = H):     Q₁K₁V₁  Q₂K₂V₂  Q₃K₃V₃  Q₄K₄V₄  Q₅K₅V₅  Q₆K₆V₆  Q₇K₇V₇  Q₈K₈V₈
                 (каждая голова — свои K, V)

GQA (G = 2):     Q₁Q₂Q₃Q₄ → K₁V₁    Q₅Q₆Q₇Q₈ → K₂V₂
                 (4 query heads делят одну KV пару)

MQA (G = 1):     Q₁Q₂Q₃Q₄Q₅Q₆Q₇Q₈ → K₁V₁
                 (все query heads делят одну KV пару)
```

GQA является **обобщением**: при $G = H$ это MHA, при $G = 1$ это MQA. На практике используют $G = 8$ (LLaMA 2 70B при $H = 64$) или $G = 4$ (Gemma).

## KV-cache: экономия памяти

Размер KV-cache при GQA:

$$\text{KV-cache}_{GQA} = 2 \times L \times G \times d_k \times S$$

Экономия пропорциональна $H/G$:

$$\text{Reduction factor} = \frac{H}{G}$$

Пример для LLaMA 2 70B ($L = 80$, $H = 64$, $G = 8$, $d_k = 128$, $S = 4096$, FP16):

| Вариант | KV heads | KV-cache (1 запрос) | KV-cache (32 batch) |
|---------|----------|---------------------|---------------------|
| MHA ($G = 64$) | 64 | 5.24 GB | 167.8 GB |
| GQA ($G = 8$) | 8 | 0.66 GB | 21.0 GB |
| MQA ($G = 1$) | 1 | 0.08 GB | 2.6 GB |

GQA с $G = 8$ даёт **8x экономию** по сравнению с MHA — разница между «не влезает в GPU» и «комфортно обслуживает batch».

## Uptraining: конвертация MHA checkpoint в GQA

Ключевой вопрос: нужно ли обучать GQA-модель с нуля? Нет — авторы GQA предлагают **uptrain** из существующего MHA checkpoint.

### Процедура конвертации

1. **Группировка KV heads.** $H$ KV голов MHA разбиваются на $G$ групп по $H/G$ голов.
2. **Mean pooling.** Для каждой группы веса KV голов усредняются:

$$W_{K_g} = \frac{1}{|G_g|} \sum_{h \in G_g} W_{K_h}, \quad W_{V_g} = \frac{1}{|G_g|} \sum_{h \in G_g} W_{V_h}$$

3. **Uptrain.** Модель дообучается на небольшой доле от original pre-training compute (~5%). Этого достаточно для восстановления quality.

Авторы показали, что mean pooling значительно лучше случайного выбора одной KV головы из группы, поскольку среднее сохраняет больше информации из всех исходных голов.

Для LLaMA 2 70B uptraining занял ~5% от оригинального compute — это дни вместо месяцев обучения с нуля.

## Сравнение MHA vs MQA vs GQA

| Свойство | MHA | MQA | GQA |
|----------|-----|-----|-----|
| KV heads | $H$ | 1 | $G$ (обычно $H/8$) |
| KV-cache memory | Baseline | $H\times$ меньше | $H/G\times$ меньше |
| Inference speed | Baseline | Быстрее (memory-bound) | Быстрее (ближе к MQA) |
| Качество (perplexity) | Baseline | Заметное падение | Близко к MHA |
| Training compute | Baseline | Такой же | Такой же (или +5% uptrain) |
| Параметров модели | Baseline | Меньше (KV params) | Меньше (KV params) |

Главный результат из paper: **GQA достигает качества, близкого к MHA, при inference speed, близкой к MQA.** Это best-of-both-worlds.

На бенчмарках GQA-G8 (8 KV групп) при uptraining T5 XXL:
- Потеря quality vs MHA: **минимальна** (~0.1-0.3% на downstream tasks)
- Speedup inference vs MHA: **близок к MQA** (особенно при больших batch sizes)
- MQA quality vs MHA: **заметная деградация** (до 1-2% на некоторых задачах)

## Где используется

GQA стал **де-факто стандартом** для modern LLM:

- **LLaMA 2 70B** — $H = 64$, $G = 8$ (8 KV heads). Меньшие модели (7B, 13B) остались на MHA.
- **Mistral 7B** — $H = 32$, $G = 8$ (8 KV heads). GQA даже для 7B модели — показатель того, что overhead MHA не оправдан даже на малых масштабах.
- **Gemma** — $H = 16$, $G = 4$ (первая модель Gemma, 7B вариант использует GQA).
- **LLaMA 3** — все размеры используют GQA ($G = 8$).
- **Qwen2** — GQA во всех вариантах.

Тренд очевиден: с 2023 года **все major LLM** используют GQA. MHA остался только в legacy моделях и research baselines.

## Почему GQA работает: интуиция

Почему можно разделять KV heads без существенной потери качества? Эмпирические исследования attention patterns показывают, что **KV heads часто redundant**: многие головы внимания выучивают похожие Key/Value проекции. В MHA с 64 головами часть голов фокусируется на синтаксисе, часть — на позиционных паттернах, часть — на семантике. Внутри каждой «группы по функции» KV проекции сильно коррелируют.

GQA формализует эту интуицию: вместо того, чтобы каждая голова учила свои KV проекции, группа query heads **разделяет** KV — и query heads внутри группы специализируются на разных аспектах одного и того же «контекста» (K, V), задавая разные «вопросы» (Q) к одному «ключу».

Это объясняет, почему mean pooling при uptraining работает: усреднение KV весов похожих голов сохраняет основной паттерн, а query heads быстро адаптируются к новым shared KV через short uptraining.

## Связь с другими оптимизациями inference

GQA часто комбинируется с другими техниками:
- **Sliding Window Attention** (Mistral 7B) — ограничивает attention window, дополнительно уменьшая KV-cache
- **PagedAttention** (vLLM) — эффективное управление памятью KV-cache через paging
- **Flash Attention** — оптимизация compute, ортогональна к GQA (уменьшает memory access, а не размер KV-cache)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Attention Mechanism|Attention Mechanism]] — базовый механизм, который GQA оптимизирует
- [[02 Areas/ML & DL/Concepts/Training/Self-Attention|Self-Attention]] — тип attention в decoder моделях
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, в которой применяется GQA
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA 2|LLaMA 2]] — первая крупная модель с GQA
- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — GQA + Sliding Window Attention
