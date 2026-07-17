---
title: "Falcon"
aliases: [Falcon, Falcon-40B, Falcon-180B, Falcon LLM, Falcon 3, TII Falcon]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Falcon|RefinedWeb / Falcon]]"
courses: []
sources:
  - "[Falcon LLM Official Page](https://falconllm.tii.ae/)"
  - "[Cameron R. Wolfe — Falcon: The Pinnacle of Open-Source LLMs](https://cameronrwolfe.substack.com/p/falcon-the-pinnacle-of-open-source)"
  - "[Sam Solutions — Falcon LLM Architecture Deep Dive](https://sam-solutions.com/blog/falcon-llm-architecture/)"
  - "[The Falcon Series of Open Language Models (2023)](https://arxiv.org/abs/2311.16867)"
  - "[GitHub — tiiuae/Falcon-H1](https://github.com/tiiuae/Falcon-H1)"
---

# Falcon

## Зачем эта модель появилась

К середине 2023 года open-source LLM ландшафт был поляризован: LLaMA (Meta) показала, что открытые модели могут быть сильными, но с лицензионными ограничениями. Большинство исследователей считали, что для обучения конкурентоспособных LLM нужны **курированные датасеты** с книгами, научными статьями, кодом — дорого и ограниченно по масштабу.

**Falcon** (Technology Innovation Institute, ОАЭ) доказал обратное: **web-only данные, правильно отфильтрованные, достаточны для frontier quality**. Falcon-40B занял #1 на HuggingFace Open LLM Leaderboard, опередив LLaMA-65B, при обучении исключительно на данных из веба.

## RefinedWeb: главная инновация

### Идея

Falcon построен на **RefinedWeb** — массивном датасете, созданном из Common Crawl через тщательную фильтрацию.

### Pipeline (пошагово)

**Этап 1: Text Extraction**
- HTML → plain text через trafilatura
- Удаление boilerplate (навигация, реклама, footer)
- Сохранение только основного контента страницы

**Этап 2: URL Filtering**
- Blocklists: adult content, malware, low-quality сайты
- Domain reputation scoring
- Результат: RW-Raw — raw extracted text

**Этап 3: Heuristic Quality Filtering**
- Длина документа (слишком короткие → удалить)
- Повторы: n-gram repetition ratio (повторяющиеся фразы → спам)
- Punctuation ratio (нет пунктуации → не natural text)
- Word length distribution (аномально длинные «слова» → мусор)
- Capital letter ratio (ALL CAPS → вероятно спам)
- Результат: RW-Filtered — первый раунд фильтрации

**Этап 4: Language Identification**
- fastText classifier для определения языка
- Фильтрация по целевому языку (EN для основного корпуса)

**Этап 5: Deduplication (двухэтапная)**
- **Exact substring deduplication:** удаление документов с идентичными подстроками длиной > threshold. Использует suffix arrays для эффективности.
- **MinHash fuzzy deduplication:** на уровне документов. MinHash signature → LSH (Locality-Sensitive Hashing) → кластеризация похожих документов → удаление дубликатов.

Результат: **RefinedWeb** — ~5T токенов, из которых ~1T отфильтрованных high-quality токенов.

### Почему это работает

Ключевой результат из paper: **properly filtered web data matches or exceeds curated corpora** (Wikipedia, Books, ArXiv) по quality per token. Предыдущие работы (GPT-3, PaLM) использовали web данные как «наполнитель», добавляя курированные источники для качества. TII показала, что если фильтрация достаточно агрессивна, web-only подход не уступает.

**Топ-15 доменов RefinedWeb по объёму:** Google, Archive.org, Blogspot, GitHub, NYTimes, WordPress, WashingtonPost, Wikia, BBC, The Guardian, eBay, Pastebin, CNN, Yahoo!, HuffPost.

**Масштаб:** RefinedWeb содержит ~5T токенов — на момент публикации это был **крупнейший публично документированный** web-корпус.

## Архитектура

### Decoder-only Transformer с оптимизациями

| Параметр | Falcon-7B | Falcon-40B | Falcon-180B |
|----------|-----------|------------|-------------|
| Params | 7B | 40B | 180B |
| Layers | 32 | 60 | 80 |
| Hidden Size | 4,544 | 8,192 | 14,848 |
| Attention | MQA | GQA (8 groups) | GQA (8 groups) |
| Context | 2,048 | 2,048 | 2,048 |
| Training Tokens | 1.5T | 1T | 3.5T |

### Multi-Query Attention (Falcon-7B)

Falcon-7B использует **Multi-Query Attention (MQA)**: все query-головы разделяют одну пару Key-Value. Это сокращает KV-cache на порядок, ускоряя inference.

$$\text{MQA: } n_h \text{ query heads} \times 1 \text{ shared KV head}$$

### Grouped-Query Attention (Falcon-40B/180B)

Falcon-40B и 180B перешли на **GQA** — компромисс между MHA и MQA: 8 групп KV-heads. Это сохраняет большую часть speed-up MQA, но с лучшим quality.

### FlashAttention

Все модели используют **FlashAttention** для IO-efficient attention computation. В сочетании с MQA/GQA это обеспечивает высокий throughput на inference.

### Parallel Attention + FFN

Falcon использует **параллельные** attention и FFN слои (вместо последовательных). Оба вычисляются одновременно и результаты складываются:

$$h = h_{prev} + \text{Attn}(h_{prev}) + \text{FFN}(h_{prev})$$

**Зачем:** стандартный sequential подход: $h = h_{prev} + \text{FFN}(h_{prev} + \text{Attn}(h_{prev}))$ — FFN ждёт завершения Attention. Parallel подход запускает оба одновременно, что лучше утилизирует GPU (два потока вычислений параллельны). Компромисс: minimal quality degradation при значительном speedup.

**Кто ещё использует:** PaLM (Google) применил ту же идею. Позднее было показано, что при достаточном масштабе разница в качестве между parallel и sequential минимальна.

## Эволюция семейства: подробности

### Falcon 1 (2023)

| Модель | Ключевое |
|--------|----------|
| Falcon-7B | MQA, RefinedWeb, Apache 2.0 |
| Falcon-40B | GQA, #1 Open LLM Leaderboard (июнь-август 2023) |
| Falcon-180B | 3.5T tokens, крупнейшая open-weight модель на тот момент, уровень GPT-3.5 |

### Falcon 2 (2024)

**Falcon-2-11B:** causal decoder-only, 11B параметров, обучен на **5.5T tokens** — значительно больше, чем Falcon 1. Четырёхэтапная стратегия обучения:
1. Pre-training с контекстом 2048
2. Расширение контекста до 4096
3. Расширение до 8192
4. Quality annealing на high-quality данных

**Falcon-2-11B-VLM:** первая мультимодальная модель в серии. Vision encoder для обработки изображений. Конкурирует с LLaVA и подобными VLM того же масштаба.

### Falcon 3 (конец 2024)

Расширенная серия с акцентом на efficiency:
- Модели от 1B до 10B
- Улучшенные pre-training данные
- **Мультимодальные возможности:** текст, изображения, и впервые в серии — **видео и аудио**
- Доступность: оптимизация для edge deployment

### Falcon Mamba (2024)

**Pure SSM модель** — Falcon Mamba использует Mamba architecture без attention-слоёв. Это экспериментальная модель для исследования SSM-подхода.

### Falcon H1 (2025)

**Hybrid SSM-Transformer** — самая продвинутая линейка:

| Параметр | Falcon H1 |
|----------|-----------|
| Architecture | Hybrid Mamba-2 + Attention |
| Params | 500M — 34B |
| Training speedup | **1.4x** vs pure Transformer |
| Inference speedup (long ctx) | **4-8x** vs pure Transformer |
| Approach | Each block = SSM + Attention |

**Falcon-H1R:** reasoning-вариант, обученный с RL. Показал, что маленькие гибридные модели могут конкурировать с **значительно большими** transformer-only альтернативами на complex reasoning.

**Отличие от Jamba:** Jamba использует ratio 1:7 (чередование слоёв). Falcon H1 интегрирует SSM и Attention **внутри каждого блока** — более tight coupling.

## Архитектурные решения: почему именно так

| Решение | Мотивация | Альтернатива |
|---------|-----------|-------------|
| MQA/GQA | Быстрый inference, малый KV-cache | MHA (больше quality, больше memory) |
| Parallel Attn+FFN | Лучшая GPU utilization | Sequential (чуть лучше quality) |
| FlashAttention | IO-efficiency | Standard attention (медленнее) |
| RefinedWeb (web-only) | Масштабируемость, нет licensing issues | Curated data (дорого, ограничено) |
| Apache 2.0 | Максимальная открытость | LLaMA license (ограничения) |

## Open-source impact

Falcon оказал значительное влияние на open-source AI:

1. **Лицензия Apache 2.0** — полностью открыт для коммерческого использования (в отличие от LLaMA 1, у которой была ограниченная лицензия)
2. **#1 на Open LLM Leaderboard** — Falcon-40B доминировал два месяца, доказав, что non-US lab может конкурировать
3. **RefinedWeb methodology** — подход к фильтрации данных стал стандартом индустрии. Dolma, FineWeb, RedPajama используют аналогичные pipelines
4. **Non-Western AI hub** — TII (Абу-Даби) показал, что frontier AI research возможен вне Silicon Valley
5. **Architectural exploration:** от pure Transformer (Falcon 1) → VLM (Falcon 2) → pure SSM (Falcon Mamba) → Hybrid (Falcon H1) — TII систематически исследует весь спектр архитектур

**Значение для Data-Centric AI:** Falcon — poster child движения «данные важнее архитектуры». Стандартная архитектура Transformer + выдающиеся данные = выдающиеся результаты.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — главный конкурент в open-source
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — SSM-архитектура, использованная в Falcon Mamba и H1
- [[02 Areas/ML & DL/Concepts/Architectures/Jamba|Jamba]] — гибридный подход, похожий на Falcon H1
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — MQA/GQA оптимизации
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — используется в Falcon
- [[02 Areas/ML & DL/Concepts/Training/Synthetic Data|Synthetic Data]] — контраст: Falcon = real data, Phi = synthetic

## Дополнительные ресурсы

- [Falcon LLM Official](https://falconllm.tii.ae/) — все модели и документация
- [Cameron Wolfe — Falcon Deep Dive](https://cameronrwolfe.substack.com/p/falcon-the-pinnacle-of-open-source) — лучший обзор архитектуры и данных
- [The Falcon Series Paper](https://arxiv.org/abs/2311.16867) — полный отчёт по серии
- [GitHub — Falcon-H1](https://github.com/tiiuae/Falcon-H1) — гибридная модель 2025
