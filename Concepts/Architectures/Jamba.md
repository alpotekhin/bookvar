---
title: "Jamba"
aliases: [Jamba, Jamba 1.5, AI21 Jamba, Hybrid SSM-Transformer]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Jamba SSM-Transformer Hybrid|Jamba]]"
courses: []
sources:
  - "[AI21 — Jamba: A Hybrid Transformer-Mamba Language Model (2024)](https://arxiv.org/abs/2403.19887)"
  - "[AI21 Blog — Introducing Jamba](https://www.ai21.com/blog/announcing-jamba/)"
  - "[AI21 — Rise of Hybrid LLMs](https://www.ai21.com/blog/rise-of-hybrid-llms/)"
  - "[Greg Robison — Architectural Deep Dive into Jamba](https://gregrobison.medium.com/architectural-evolution-in-large-language-models-a-deep-dive-into-jambas-hybrid-transformer-mamba-c3efa8ca8cae)"
  - "[AI21 Blog — Jamba 1.5 Model Family](https://www.ai21.com/blog/announcing-jamba-model-family/)"
---

# Jamba

## Зачем эта модель появилась

К 2024 году стало ясно, что и Transformer, и SSM (Mamba) имеют фундаментальные ограничения:

**Transformer:**
- Мощный reasoning благодаря global attention (любой токен видит любой другой)
- Но $O(n^2)$ сложность по длине последовательности
- KV-cache растёт линейно — bottleneck при длинных контекстах

**Mamba (SSM):**
- $O(n)$ сложность — идеально для длинных последовательностей
- Постоянный размер state (не зависит от длины)
- Но хуже на задачах, требующих точного recall (in-context learning, копирование)

**Jamba** (AI21, март 2024) — **первая production-grade гибридная модель**, объединяющая Transformer attention и Mamba SSM. Название: **J**oint **A**ttention and **M**am**BA**.

## Архитектура: чередование слоёв

### Базовый блок

Jamba использует **блочную структуру**, где каждый блок содержит несколько слоёв:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/jamba/jamba-architecture.png]]
*Jamba block architecture: (a) один блок с чередованием Mamba и Attention слоёв (ratio 1:7), MoE каждые 2 слоя; (b) типы слоёв — Transformer, Mamba, Attention+MoE (источник: AI21 Labs, 2024)*

**Ratio 1:7:** на каждые 7 слоёв Mamba приходится 1 слой Transformer attention. Это не случайное число — AI21 экспериментально определили, что минимальное количество attention-слоёв достаточно для сохранения recall capability, при этом основная вычислительная нагрузка приходится на эффективные Mamba-слои.

### Экспериментальный поиск оптимального ratio

AI21 тестировали несколько конфигураций:

| Ratio (Attn:Mamba) | Throughput | Recall quality | Overall quality |
|---------------------|-----------|----------------|-----------------|
| 1:1 (pure hybrid) | Средний | Отличный | Отличный |
| 1:3 | Хороший | Хороший | Хороший |
| **1:7** | **Высокий** | **Хороший** | **Хороший** |
| 0:1 (pure Mamba) | Максимальный | Плохой | Средний |
| 1:0 (pure Transformer) | Низкий | Отличный | Хороший |

**Sweet spot 1:7:** при дальнейшем уменьшении доли attention (1:15) recall capability начинает деградировать. При увеличении (1:3) throughput страдает без заметного улучшения качества.

### Почему именно такое соотношение

**Mamba-слои** обрабатывают большинство «рутинной» работы: языковое моделирование, локальные паттерны, стилистика. Они делают это за $O(n)$ — линейно по длине.

**Attention-слои** (1 из 8) нужны для задач, где SSM слаба:
- **In-context learning** — модель должна «запомнить» и использовать информацию из начала контекста
- **Precise recall** — точное воспроизведение фактов из промпта (needle-in-a-haystack)
- **Global reasoning** — связывание информации из далёких частей текста
- **Copying patterns** — SSM теоретически не может идеально копировать arbitrary sequences

Без attention-слоёв модель «забывает» далёкую информацию (SSM state — lossy compression). С полным attention — теряет эффективность. Ratio 1:7 — sweet spot.

### Почему SSM плоха для recall: теоретический аргумент

Mamba (и любой SSM) сжимает всю историю последовательности в **фиксированный state** размера $d_{state}$. Это lossy compression: при длинных последовательностях неизбежна потеря информации. Attention хранит **все** Key/Value — это lossless, но дорого.

Analogy: SSM как человек, который слушает лекцию и делает конспект (фиксированный размер). Attention — как запись на видео (всё сохранено, но нужно место). Гибрид — конспект + несколько ключевых видеозаписей.

### MoE: увеличение capacity без cost

В некоторых слоях FFN заменяется на MoE:
- **16 экспертов**, из которых **top-2** активны на токен
- MoE добавляется **каждые 2 блока** (не в каждый)
- Это увеличивает total parameters без увеличения compute per token
- Expert balancing через standard auxiliary loss

### Итоговая конфигурация

| Параметр | Jamba (original) | Jamba 1.5 Mini | Jamba 1.5 Large |
|----------|-----------------|----------------|-----------------|
| Total Params | 52B | 52B | 398B |
| Active Params | 12B | 12B | 94B |
| Context | 256K | 256K | 256K |
| Attention:Mamba ratio | 1:7 | 1:7 | 1:7 |
| MoE Experts | 16 (top-2) | 16 (top-2) | 16 (top-2) |

## Throughput benchmarks: конкретные числа

### Jamba vs Pure Transformer

| Метрика | Jamba 52B | Mixtral 8x7B | LLaMA-2 70B |
|---------|----------|--------------|-------------|
| Throughput (256K ctx) | **~3x** | 1x (baseline) | <1x |
| KV-cache size (256K) | ~10GB | ~100GB+ | ~120GB+ |
| Single GPU fit (ctx) | **140K** | ~32K | ~16K |

### Jamba 1.5 vs Competition

| Метрика | Jamba 1.5 Large | LLaMA 3.1 70B | Mixtral 8x22B |
|---------|-----------------|---------------|---------------|
| Throughput (long ctx) | **2.5x faster** | 1x | — |
| Context window | **256K** | 128K | 64K |
| RULER (256K) | **Passes** | Degrades | N/A |
| Throughput (short ctx) | **Fastest in class** | — | — |

**RULER benchmark:** Jamba 1.5 — **единственная open model**, которая сохраняет performance на всём диапазоне 256K контекста. Большинство конкурентов деградируют после 64-128K.

### KV-cache savings

Детальная разбивка экономии памяти:

| Компонент | Pure Transformer | Jamba (1:7) |
|-----------|-----------------|-------------|
| Attention KV-cache | 100% (все слои) | **12.5%** (1/8 слоёв) |
| Mamba state | 0% | Fixed size (~tiny) |
| **Total memory** | **Baseline** | **~8-10x reduction** |

Это позволяет Jamba вместить **256K контекст** там, где pure Transformer не может обработать даже 32K.

## Архитектурные trade-offs: честный анализ

### Где Jamba сильнее pure Transformer

1. **Long context:** при >64K токенов Jamba значительно быстрее и экономичнее
2. **Throughput:** высокий tokens/second благодаря Mamba-доминированной архитектуре
3. **Memory:** KV-cache ~8x меньше — можно обслуживать больше concurrent users
4. **Latency-sensitive:** время до первого токена (TTFT) ниже

### Где Jamba уступает

1. **Reasoning quality:** на complex reasoning задачах (GPQA, hard math) pure Transformer того же active size обычно лучше
2. **In-context learning:** при задачах, требующих точного recall из середины контекста, Jamba может уступать
3. **Training complexity:** обучение гибрида сложнее — нужно балансировать SSM и attention components
4. **Ecosystem:** меньше инструментов и оптимизаций по сравнению с pure Transformer

### Когда использовать Jamba

| Сценарий | Рекомендация |
|----------|-------------|
| Long document analysis | Jamba (efficiency) |
| Streaming/real-time | Jamba (low latency) |
| Complex reasoning | Pure Transformer |
| High-throughput API | Jamba (more users/GPU) |
| Fine-tuning ecosystem | Transformer (more tools) |

## Влияние на архитектурный landscape

Jamba доказала жизнеспособность гибридного подхода, вдохновив:

- **Falcon Mamba / Falcon H1** (TII) — гибридные модели, 1.4x training и 4-8x inference speedup
- **Zamba** (Zyphra) — похожий подход с другим ratio
- **NVIDIA Hybrid Models** — исследования SSM-Transformer гибридов
- **Множество academic papers** — hybrid architectures стали active area of research

Это сдвинуло дискуссию от «Transformer vs SSM» к «Transformer **+** SSM» — идея, что разные типы слоёв хороши для разных подзадач внутри одной модели. Аналогия: в CPU есть и arithmetic units, и cache — разные компоненты для разных задач.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — SSM-компонент Jamba
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — attention-компонент Jamba
- [[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral of Experts]] — MoE подход
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — используется в attention-слоях
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — проблема, которую гибрид смягчает
- [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]] — другой линейный подход
- [[02 Areas/ML & DL/Concepts/Architectures/Falcon|Falcon]] — Falcon H1 использует аналогичный гибридный подход

## Дополнительные ресурсы

- [Jamba Paper (ICLR 2025)](https://arxiv.org/abs/2403.19887) — оригинальная статья
- [AI21 Blog — Rise of Hybrid LLMs](https://www.ai21.com/blog/rise-of-hybrid-llms/) — обзор гибридного подхода
- [Jamba 1.5 Announcement](https://www.ai21.com/blog/announcing-jamba-model-family/) — обновлённая серия
- [Greg Robison — Deep Dive](https://gregrobison.medium.com/architectural-evolution-in-large-language-models-a-deep-dive-into-jambas-hybrid-transformer-mamba-c3efa8ca8cae) — детальный архитектурный разбор
