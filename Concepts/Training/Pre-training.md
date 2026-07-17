---
title: "Pre-training"
aliases: [pre-train, self-supervised pre-training]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/T5]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[Together AI — RedPajama-Data-v2](https://www.together.ai/blog/redpajama-data-v2)"
  - "[Glenn Lockwood — LLM Training Datasets](https://www.glennklockwood.com/garden/LLM-training-datasets)"
---

# Pre-training

## Зачем это нужно: от tabula rasa к «пониманию» языка

До 2018 года каждую NLP модель обучали *с нуля* на каждый новый датасет: новая задача — новая модель, новые данные, новый бюджет. Это неэффективно: NER, classification, QA — все требуют «понимания» языка, но каждый раз это понимание выучивалось заново.

Pre-training — **первая фаза обучения**: крупная модель обучается на огромном объёме неразмеченных текстов через self-supervised objectives. Цель — получить **general-purpose представления языка**, которые затем адаптируются к downstream задачам через [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|fine-tuning]] или prompting.

Аналогия: pre-training — это как общее образование (школа + университет), fine-tuning — специализация (аспирантура или работа). Чем лучше base education, тем быстрее и эффективнее специализация.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/instructgpt/pretraining-hf.png]]
*Pre-training: модель обучается на огромном корпусе текстов через self-supervised objectives без человеческой разметки (источник: Hugging Face)*

## Три парадигмы pre-training objectives

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/pre-training/clm-vs-mlm.png]]
*Schematic comparison of pre-training objectives: Causal Language Modeling (CLM), Masked Language Modeling (MLM), и их варианты (источник: arXiv 2412.03275)*

Выбор pre-training objective определяет **архитектуру модели** и **для чего она лучше подходит**. Три основных подхода:

### 1. Masked Language Modeling (MLM) → Encoder-only

**BERT** (Devlin et al., 2019): модель «видит» текст с замаскированными токенами и предсказывает, что было скрыто. Используется **bidirectional** контекст — и левый, и правый.

Процедура маскирования (15% токенов):
- 80% → заменяются на `[MASK]`
- 10% → заменяются на случайный токен
- 10% → остаются без изменений

**Почему такое распределение?** Если всегда `[MASK]`, модель не увидит `[MASK]` при fine-tuning → mismatch. Случайные замены + неизменённые токены учат модель не доверять никаким позициям и использовать полный контекст.

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i | x_{\backslash i})$$

**Плюсы:** bidirectional контекст → отличные representations для NLU (classification, NER, QA).
**Минусы:** не умеет генерировать текст (не авторегрессивна).

**Модели:** BERT, RoBERTa, ALBERT, DeBERTa, ELECTRA.

### 2. Causal Language Modeling (CLM) → Decoder-only

**GPT** серия (Radford et al., 2018-2023): классический language modeling — предсказание следующего токена:

$$P(x) = \prod_{t=1}^{T} P(x_t | x_1, \ldots, x_{t-1})$$

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

**Однонаправленный** контекст — модель «видит» только предшествующие токены. Это позволяет **авторегрессивную генерацию**: на inference каждый новый токен генерируется на основе всех предыдущих.

**Плюсы:** нативная генерация текста, масштабируется до GPT-4 и далее.
**Минусы:** однонаправленный контекст — хуже для задач, требующих bidirectional «понимания» (NER, span extraction).

**Модели:** GPT, GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Qwen, Phi.

### 3. Span Corruption → Encoder-Decoder

**T5** (Raffel et al., 2020): заменяем случайные spans (непрерывные последовательности токенов) на sentinel tokens, задача декодера — реконструировать spans.

```
Вход:   "The <X> sat on the <Y> and purred."
Выход:  "<X> cat <Y> mat"
```

**Почему это лучше MLM?** Targets короче (только spans, не весь текст) → более эффективное обучение. Кроме того, модель учится генерировать, а не только предсказывать отдельные токены.

**Модели:** T5, FLAN-T5, UL2, mT5.

### Сравнение objectives

| Objective | Контекст | Генерация | Лучше для | Scaling |
|-----------|---------|-----------|-----------|---------|
| **MLM** | Bidirectional | Нет | NLU (NER, QA, classification) | До ~1B |
| **CLM** | Unidirectional | Да | Generation, instruction following | До 1T+ |
| **Span corruption** | Encoder: bi, Decoder: uni | Да | Seq2Seq (перевод, суммаризация) | До ~100B |

**Исторический тренд:** CLM (decoder-only) победил для scaled-up моделей. GPT-3, LLaMA, Mistral — все decoder-only. Причина: CLM проще масштабировать, и при достаточном масштабе unidirectional контекст перестаёт быть проблемой.

## Датасеты pre-training: от 16 GB до 15T+ токенов

Качество и масштаб данных — **главный фактор** quality pre-trained модели. Эволюция:

| Модель (год) | Данные | Объём | Источники |
|-------------|--------|-------|-----------|
| **BERT** (2019) | BooksCorpus + Wikipedia | 16 GB / ~3.3B tokens | 2 источника |
| **GPT-2** (2019) | WebText | ~40 GB / ~10B tokens | Reddit-filtered web |
| **RoBERTa** (2019) | BERT data + CC-News + OpenWebText + Stories | 160 GB / ~33B tokens | 5 источников |
| **GPT-3** (2020) | CommonCrawl + WebText2 + Books + Wiki | ~300B tokens | 4 источника, weighted |
| **T5** (2020) | C4 (Colossal Clean Crawled Corpus) | 750 GB / ~156B tokens | CommonCrawl filtered |
| **LLaMA** (2023) | CC + C4 + GitHub + Wiki + Books + ArXiv + StackExchange | **1.4T tokens** | 7 источников |
| **LLaMA 2** (2023) | Public mix | **2.0T tokens** | Undisclosed mix |
| **Mistral** (2023) | Undisclosed | Undisclosed | — |
| **LLaMA 3** (2024) | Curated web data | **15T+ tokens** | Extensive filtering |

### Ключевые датасеты

**C4 (Colossal Clean Crawled Corpus)**

Создан Google для T5. Основан на CommonCrawl с агрессивной фильтрацией:
- Удаление дубликатов строк
- Фильтрация «bad words»
- Удаление кода
- Только предложения, заканчивающиеся пунктуацией
- Результат: ~750 GB чистого английского текста

**The Pile (EleutherAI)**

825 GB / ~300B tokens из 22 разнообразных источников: академические статьи (PubMed, ArXiv), код (GitHub), книги (Bibliotik), энциклопедии (Wikipedia), юридические тексты (FreeLaw). Designed для diversity.

**RedPajama**

Открытая реплика LLaMA training data от Together AI:
- **RedPajama v1:** 1.2T tokens, 7 источников (CC, C4, GitHub, ArXiv, Books, Wiki, StackExchange) — повторяет LLaMA distribution
- **RedPajama v2:** 30T raw tokens (100T до дедупликации) из 84 CommonCrawl snapshots + 40+ pre-computed quality annotations

**ROOTS / BLOOM**

1.6T tokens на 46 языках — для мультиязычной модели BLOOM. Первый крупный **документированный** мультиязычный dataset.

### Data Mixing: пропорции источников

Не все данные одинаково полезны. LLaMA использует weighted sampling:

| Источник | Доля в mix | Epochs |
|---------|-----------|--------|
| CommonCrawl | 67% | 1.0 |
| C4 | 15% | 1.0 |
| GitHub | 4.5% | 0.64 |
| Wikipedia | 4.5% | 2.45 |
| Books | 4.5% | 2.23 |
| ArXiv | 2.5% | 1.06 |
| StackExchange | 2.0% | 1.03 |

**Замечание:** Wikipedia и Books имеют >1 epoch — высококачественные данные используются повторно. CommonCrawl — только 1 epoch, но доминирует по объёму.

## Архитектура и compute

### Стандартная конфигурация (decoder-only, 2023-2024)

| Компонент | Выбор | Почему |
|-----------|-------|--------|
| Architecture | Decoder-only Transformer | Масштабируется лучше всех |
| Attention | Multi-Head / GQA | GQA для inference efficiency |
| Normalization | RMSNorm (pre-norm) | Стабильнее LayerNorm, быстрее |
| Activation | SwiGLU | +1-3% quality vs ReLU/GELU |
| Positional | RoPE | Экстраполяция длины контекста |
| Vocabulary | BPE (32K-128K tokens) | Баланс compression/granularity |
| Precision | bf16 / mixed precision | Стабильнее fp16, достаточная точность |

### Compute requirements

Приблизительная формула для compute:

$$C \approx 6 \cdot N \cdot D \text{ (FLOP)}$$

где $N$ — параметры, $D$ — токены. Для LLaMA 65B на 1.4T tokens:

$$C \approx 6 \times 6.5 \times 10^{10} \times 1.4 \times 10^{12} \approx 5.5 \times 10^{23} \text{ FLOP}$$

| Модель | N | D | Compute (FLOP) | GPU-hours (A100) | Стоимость |
|--------|---|---|---------------|-----------------|-----------|
| LLaMA 7B | 7B | 1T | ~$4.2 \times 10^{22}$ | ~82K | ~$80K |
| LLaMA 13B | 13B | 1T | ~$7.8 \times 10^{22}$ | ~135K | ~$135K |
| LLaMA 65B | 65B | 1.4T | ~$5.5 \times 10^{23}$ | ~1M | ~$1-2M |
| GPT-3 | 175B | 300B | ~$3.1 \times 10^{23}$ | ~800K | ~$5-12M |
| GPT-4 | ~1.8T (MoE)? | ~13T? | ~$2 \times 10^{25}$? | — | ~$100M? |

*Стоимости приблизительные, зависят от hardware и pricing.*

## Tricks для стабильности обучения

Pre-training на масштабе сотен миллиардов параметров — инженерно сложная задача. Ключевые проблемы и решения:

### Loss Spikes

Training loss может резко возрасти (spike) из-за плохого batch данных или численной нестабильности. Решения:
- **Gradient clipping:** ограничение нормы градиента (max\_grad\_norm = 1.0)
- **Skipping bad batches:** если loss > threshold, пропускаем batch
- **PaLM approach:** откат на предыдущий checkpoint при spike

### Learning Rate Schedule

Стандартный рецепт (LLaMA, GPT-3):
- **Warm-up:** линейный рост LR от 0 до max за первые ~2000 steps
- **Cosine decay:** плавное уменьшение LR до ~10% от max
- **Max LR:** ~3e-4 для 7B, ~1.5e-4 для 65B (обратно пропорционально размеру)

### Numerical Stability

- **bf16 > fp16:** bf16 имеет больший dynamic range (8-bit exponent vs 5-bit), что предотвращает overflow/underflow
- **Loss scaling:** не нужен при bf16 (в отличие от fp16)
- **Attention softmax в fp32:** даже при bf16 training softmax вычисляется в fp32 для численной стабильности

### Distributed Training

| Метод | Что делает | Когда нужен |
|-------|-----------|-------------|
| **Data Parallel (DDP)** | Копия модели на каждый GPU, split data | Модель помещается в 1 GPU |
| **FSDP / ZeRO** | Шардирование параметров + gradients + optimizer | Модель > 1 GPU |
| **Tensor Parallel** | Разрезание матриц внутри layer | Очень большие модели |
| **Pipeline Parallel** | Разрезание по layers | Очень глубокие модели |
| **Sequence Parallel** | Разрезание по длине последовательности | Длинные контексты |

LLaMA 65B: обучение на **2048 A100 80GB** в течение ~21 дня. Потребление: ~1M GPU-hours.

## Факторы качества pre-training

Из [[02 Areas/ML & DL/Papers/RoBERTa]] — систематический ablation «что важно в pre-training»:

### 1. Dynamic vs Static Masking

BERT использует static masking — маски генерируются один раз. RoBERTa: **dynamic masking** (новые маски на каждой эпохе) → лучше при длительном обучении.

### 2. Больше данных = лучше

Увеличение данных с 16 GB (BERT) до 160 GB (RoBERTa):
- MNLI: 89.0% → **90.2%**

### 3. Дольше обучение = лучше

Увеличение training steps с 100K до 500K:
- SQuAD F1: 87.3% → **89.4%**

### 4. Большие batch sizes = лучше (при правильном LR)

BERT: batch 256. RoBERTa: batch **8K** → стабильнее обучение, лучше convergence.

### 5. NSP не нужен

BERT использует Next Sentence Prediction (NSP) как второй objective. RoBERTa показала: **удаление NSP не ухудшает** (и часто улучшает) результаты.

## Парадигмы использования pre-trained моделей

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/pre-training/nlp-paradigms.png]]
*Парадигмы использования pre-trained моделей: от feature extraction (ELMo) через fine-tuning (BERT, LoRA) к prompting (GPT-3+) и instruction tuning (ChatGPT). Каждая парадигма снижает объём task-specific адаптации (источник: Cameron Wolfe)*

| Парадигма | Эпоха | Подход | Пример |
|-----------|-------|--------|--------|
| **Feature extraction** | BERT early | Заморозить модель, обучить classifier поверх | ELMo, early BERT usage |
| **Fine-tuning** | BERT-T5 | Обновить все/часть параметров | BERT FT, LoRA, RLHF |
| **Prompting** | GPT-3+ | Zero/few-shot через промпты, без обучения | GPT-4, Claude, Gemini |
| **Instruction tuning** | 2022+ | SFT на инструкциях + alignment | InstructGPT, ChatGPT |

## Хронология

| Год | Milestone | Значение |
|-----|-----------|----------|
| 2013 | Word2Vec | Первые pre-trained word embeddings |
| 2018 | ELMo, ULMFiT | Contextualized embeddings, transfer learning для NLP |
| 2018 | **GPT** | CLM pre-training + fine-tuning |
| 2019 | **BERT** | MLM + bidirectional → революция в NLU |
| 2019 | **RoBERTa** | Ablation: что важно в BERT pre-training |
| 2020 | **GPT-3** | Scale + few-shot prompting (175B, 300B tokens) |
| 2020 | **T5** | Systematic study of pre-training objectives |
| 2022 | **Chinchilla** | Compute-optimal training (20 tok/param) |
| 2023 | **LLaMA** | Inference-optimal training (1T+ tokens на 7-65B) |
| 2024 | **LLaMA 3** | 15T+ tokens, data curation at scale |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]] — BERT pre-training objective
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]] — GPT pre-training objective
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — вторая фаза адаптации
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — сколько данных и параметров нужно
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — базовая архитектура
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — inference-optimal pre-training
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — пионер pre-training для NLU

## Дополнительные ресурсы

- [Together AI — RedPajama-Data-v2](https://www.together.ai/blog/redpajama-data-v2) — крупнейший открытый training dataset
- [Glenn Lockwood — LLM Training Datasets](https://www.glennklockwood.com/garden/LLM-training-datasets) — обзор датасетов
- [Sebastian Raschka — LLM Pre-training Insights](https://magazine.sebastianraschka.com/p/ahead-of-ai-8-the-latest-open-source) — практические insights
