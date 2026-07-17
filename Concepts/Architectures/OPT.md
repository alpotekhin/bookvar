---
title: "OPT"
aliases: [Open Pre-trained Transformer, OPT-175B]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/OPT]]"
courses: []
sources:
  - "[Zhang et al. — OPT: Open Pre-trained Transformer Language Models (2022)](https://arxiv.org/abs/2205.01068)"
  - "[Meta AI — metaseq GitHub](https://github.com/facebookresearch/metaseq)"
---

# OPT — Open Pre-trained Transformer Language Models

## Зачем эта модель появилась

К началу 2022 года крупнейшие языковые модели (GPT-3, PaLM, Chinchilla) были **полностью закрыты** — ни весов, ни кода, ни деталей обучения. Это создавало серьёзную проблему для науки: исследования interpretability, bias, safety, и fine-tuning на scale 175B+ были доступны только крупным лабораториям с ресурсами для собственного обучения.

OPT (Zhang et al., май 2022, Meta AI) — **первый по-настоящему открытый конкурент GPT-3**. Восемь моделей от 125M до 175B параметров, сопоставимые с GPT-3, но с открытыми весами, кодом, и — беспрецедентно — **дневником обучения (logbook)**, описывающим все проблемы, с которыми столкнулись инженеры.

## Архитектура: следуя GPT-3

OPT намеренно повторяет архитектуру GPT-3 (Brown et al., 2020) для fair comparison:

| Модель | Layers | Heads | d_model | LR | Batch |
|--------|--------|-------|---------|-----|-------|
| OPT-125M | 12 | 12 | 768 | 6.0e-4 | 0.5M |
| OPT-350M | 24 | 16 | 1024 | 3.0e-4 | 0.5M |
| OPT-1.3B | 24 | 32 | 2048 | 2.0e-4 | 1M |
| OPT-2.7B | 32 | 32 | 2560 | 1.6e-4 | 1M |
| OPT-6.7B | 32 | 32 | 4096 | 1.2e-4 | 2M |
| OPT-13B | 40 | 40 | 5120 | 1.0e-4 | 4M |
| OPT-66B | 64 | 72 | 9216 | 0.8e-4 | 2M |
| **OPT-175B** | **96** | **96** | **12288** | **1.2e-4** | **2M** |

### Отличия от GPT-3
- **ReLU activation** (не SwiGLU, как в LLaMA)
- **Context length**: 2048 токенов
- **Weight initialization**: из Megatron-LM, $\sigma = 0.006$, output layers масштабированы на $1/\sqrt{2L}$
- **Tokenizer**: GPT-2 BPE (50,272 tokens)
- **Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$), weight decay 0.1

## Training Corpus: ~180B токенов

| Источник | Описание |
|----------|----------|
| **RoBERTa corpus** | BookCorpus + Stories + CCNews v2 |
| **The Pile** (subset) | CommonCrawl, Project Gutenberg, HackerNews, OpenWebText2, Wikipedia, ArXiv, DM Mathematics |
| **PushShift.io Reddit** | Самая длинная цепочка комментариев в каждом треде (~33% от полного корпуса) |

**Дедупликация**: MinHashLSH с Jaccard similarity ≥ 0.95. Авторы отмечают, что The Pile «особенно полон дубликатов» — рекомендация для будущих исследователей.

Некоторые подмножества The Pile были **исключены** из-за нестабильности обучения (spike'и gradient norm на 1.3B scale).

## Logbook: беспрецедентная прозрачность

**Главная ценность OPT paper** — не модель, а **дневник обучения**. Обычно обучение 175B модели описывается как «мы обучили модель на N GPU за M дней». OPT показал реальность:

### Hardware failures

- **35+ ручных рестартов** за 2 месяца обучения
- **100+ серверов** были заменены из-за аппаратных сбоев
- **70+ автоматических рестартов** (оценка по разнице между заменёнными серверами и ручными рестартами)
- Каждый рестарт требовал диагностики, изоляции проблемного узла, и возобновления с последнего checkpoint

### Training instabilities

- **Gradient norm spikes**: непредсказуемые всплески gradient norms, вызывающие divergence
- **Решение**: снижение gradient clipping threshold с 1.0 до 0.3 (mid-flight!)
- **Loss scalar crashes**: dynamic loss scalar падал до 0, causing underflows
- **Learning rate interventions**: многократные ручные изменения LR schedule (Figure 1 из paper показывает хаотичный LR schedule)

### Другие mid-flight изменения

- Переключение между vanilla SGD и Adam
- Изменение gradient predivide factor для предотвращения over/underflows при distributed training
- Сброс dynamic loss scalar

Эта прозрачность бесценна: она показала **реальную стоимость** обучения больших моделей и помогла другим командам (BLOOM, LLaMA) избежать тех же ошибок.

## Infrastructure: 992 A100s

- **992 NVIDIA A100-80GB GPU**
- **FSDP** (Fully Sharded Data Parallel) + **Megatron-LM Tensor Parallelism**
- **147 TFLOP/s per GPU** utilization (высокая для distributed training)
- **Adam state в FP32** (sharded across hosts), модель в FP16
- **Dynamic loss scaling** для предотвращения FP16 underflows

## Ключевые результаты

### OPT-175B vs GPT-3 (zero-shot / few-shot)

| Benchmark | GPT-3 175B | OPT-175B |
|-----------|-----------|----------|
| HellaSwag | 78.9 | 78.5 |
| WinoGrande | 77.7 | 78.0 |
| StoryCloze | 87.7 | 87.2 |
| PIQA | 81.0 | 79.7 |
| BoolQ | 60.5 | 65.2 |

**Вывод**: OPT-175B **сопоставим с GPT-3** — незначительные отличия в обе стороны. Но при этом OPT полностью открыт.

### Carbon footprint

| Модель | Carbon (tCO2eq) |
|--------|-----------------|
| GPT-3 175B | ~500 (оценка) |
| **OPT-175B** | **~70** |
| Разница | **~1/7** |

Снижение на порядок достигнуто за счёт:
- Более новое оборудование (A100 vs V100)
- Более эффективный distributed training
- Оптимизированный code (metaseq)

## Почему OPT важна

### 1. Демократизация research

OPT дала академическим исследователям **первый доступ к 175B-scale модели**. До OPT исследования bias, toxicity, interpretability на этом масштабе были невозможны вне Google/OpenAI.

### 2. Logbook как новый стандарт

Публикация дневника обучения создала прецедент прозрачности. Это помогло:
- **BLOOM** (BigScience): использовали опыт OPT для более стабильного обучения
- **LLaMA**: Meta улучшила training pipeline на основе уроков OPT
- **Индустрия**: компании стали реалистичнее оценивать стоимость и сложность обучения

### 3. Carbon footprint discourse

OPT ввела **явное сопоставление carbon footprint** как стандарт для ML papers. Аргумент: если модель открыта, другим не нужно тратить compute и CO2 на собственное обучение.

### 4. Историческая роль

OPT (май 2022) → BLOOM (июль 2022) → LLaMA (февраль 2023) — эта тройка **создала open-source LLM экосистему**, которая сейчас доминирует в research. OPT была первой.

## Ограничения OPT

- **Устаревшая архитектура**: ReLU вместо SwiGLU, нет RoPE, нет GQA → менее эффективна, чем LLaMA
- **Нестабильное обучение**: многочисленные рестарты повлияли на качество
- **Данные**: ~180B токенов — значительно меньше, чем Chinchilla-оптимальное количество для 175B
- **Без alignment**: base model only, нет instruction-tuned или chat-версии

## Сравнение с другими открытыми моделями

| | OPT (May 2022) | BLOOM (Jul 2022) | LLaMA (Feb 2023) |
|--|---|---|---|
| Макс. размер | 175B | 176B | 65B |
| Activation | ReLU | GELU | SwiGLU |
| Position | Learned absolute | ALiBi | RoPE |
| Данные | ~180B tokens | 341B tokens | 1.0-1.4T tokens |
| Multilingual | Нет | Да (46 languages) | Ограниченно |
| Качество | ≈ GPT-3 | < GPT-3 | > GPT-3 |
| Logbook | Да | Да | Нет |

## Key papers

- [[02 Areas/ML & DL/Papers/OPT]] — оригинал (Zhang et al., 2022)
- [[02 Areas/ML & DL/Papers/LLaMA]] — более производительная открытая альтернатива

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — модель-target для OPT
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — successor в open-source LLM
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]] — training objective

## Дополнительные ресурсы

- [OPT paper (arXiv)](https://arxiv.org/abs/2205.01068) — оригинальная статья
- [metaseq GitHub](https://github.com/facebookresearch/metaseq) — code для обучения
- [OPT Logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) — дневник обучения 175B модели
