---
title: "Language Models are Few-Shot Learners (GPT-3)"
url: https://arxiv.org/abs/2005.14165
authors: [Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei]
year: 2020
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - GPT
  - LLM
  - arch
Date: 2020-01-05
Organization: OpenAI
Parent item:
  - "[[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/gpt-30/paper.txt]]"
---

# Language Models are Few-Shot Learners (GPT-3)

**Authors:** Tom B. Brown et al. (OpenAI), 45 авторов
**Published:** 2020 (arXiv:2005.14165v4, Jul 2020)
**URL:** https://arxiv.org/abs/2005.14165

## TL;DR

GPT-3 — автогрессивная языковая модель с **175 миллиардами параметров** (в 10× больше любой предыдущей non-sparse LM), которая впервые убедительно демонстрирует few-shot обучение без обновления весов. Ключевой инсайт: при достаточном масштабе модель способна выполнять новые задачи, увидев лишь несколько примеров в контексте (in-context learning), иногда достигая конкурентоспособного уровня с дообученными моделями — без единого шага градиентного спуска.

## Problem

Парадигма pre-train + fine-tune (BERT, RoBERTa, T5) требует тысяч-десятков тысяч размеченных примеров для каждой новой задачи. Это ограничивает применимость LM:
1. Разметка дорога и недоступна для большинства нишевых задач
2. Fine-tuning может захватывать spurious correlations в узком train-распределении → плохая out-of-distribution генерализация
3. Человек способен выполнять новые задачи из нескольких примеров или простых инструкций — LM так не умеют

Гипотеза авторов: **масштаб** решает проблему. Если сделать модель достаточно большой, она разовьёт широкие навыки в ходе pre-training, которые затем можно использовать через in-context learning во время inference.

## Method

### Архитектура

GPT-3 использует ту же архитектуру, что GPT-2: [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] Transformer с модификациями:
- Modified initialization и pre-normalization (layer norm перед attention, не после)
- Reversible tokenization (byte-level BPE)
- **Alternating dense и locally banded sparse attention** (аналог Sparse Transformer) в слоях

**8 размеров модели** от 125M до 175B параметров (Table 2.1):

| Модель | Параметры | Слои | d_model | Головы |
|--------|-----------|------|---------|--------|
| GPT-3 Small | 125M | 12 | 768 | 12 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 |
| GPT-3 Large | 760M | 24 | 1536 | 16 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 |
| **GPT-3 175B** | **175.0B** | **96** | **12288** | **96** |

Все модели: context window n_ctx = 2048, d_ff = 4 × d_model.

### Training Dataset

| Датасет | Объём (токены) | Доля в обучении |
|---------|----------------|-----------------|
| Common Crawl (filtered) | 410B | 60% |
| WebText2 | 19B | 22% |
| Books1 | 12B | 8% |
| Books2 | 55B | 8% |
| Wikipedia | 3B | 3% |

Common Crawl фильтровался по качеству (similarity к high-quality reference corpora) и дедуплицировался. Датасеты с высоким качеством семплируются чаще, чем пропорционально их размеру. Всего обучение: **300 миллиардов токенов**.

### Training Process

- Batch size растёт с размером модели (от 0.5M до 3.2M токенов)
- Learning rate уменьшается с размером модели (от 6.0×10⁻⁴ до 0.6×10⁻⁴)
- Модель партиционируется по глубине и ширине через GPU для минимизации data-transfer
- Обучение на кластере V100 GPU от Microsoft

**Scaling law:** performance (cross-entropy loss) следует степенному закону как функция compute — тренд продолжается на 2 дополнительных порядка величины без отклонений (Figure 3.1).

### In-context Learning

Авторы вводят чёткую терминологию (Figure 2.1):
- **Zero-Shot (0S):** только описание задачи на естественном языке, без примеров
- **One-Shot (1S):** одно демо + описание задачи
- **Few-Shot (FS):** K демо (типично 10–100, сколько влезает в 2048 токенов)
- **Fine-Tuning (FT):** обновление весов — в этой работе **не используется**

Во всех случаях — только forward pass, никаких gradient updates.

## Key Results

### Language Modeling

**PTB (Penn Tree Bank), zero-shot:** GPT-3 устанавливает новый SOTA — perplexity **20.50** (предыдущий: 35.8), улучшение на 15 ppl (Table 3.1).

**LAMBADA (long-range dependency):**
- Zero-shot: 76.2% (предыдущий SOTA: 68.0%) → +8.2%
- Few-shot: **86.4%** (+18.4% над SOTA, +10% над zero-shot)
Пример использования few-shot форматирования как fill-in-the-blank: это позволяет модели «понять» формат задачи.

### Closed Book QA (Table 3.3)

| Задача | Zero-Shot | One-Shot | Few-Shot | Fine-tuned SOTA |
|--------|-----------|----------|----------|-----------------|
| TriviaQA | 64.3% | 68.0% | **71.2%** | 68.0% (RAG) |
| WebQS | 14.4% | 25.3% | 41.5% | 44.7% (T5+SSM) |
| NaturalQS | 14.6% | 23.0% | 29.9% | 36.6% (T5+SSM) |

На TriviaQA few-shot GPT-3 **превышает** fine-tuned SOTA (включая RAG с 15.3B-parameter dense index). Zero-shot уже бьёт fine-tuned T5-11B на 14.2%.

### Translation (Table 3.4)

Few-shot GPT-3 превосходит unsupervised NMT при переводе **на английский**:
- Fr→En: 39.2 BLEU (SOTA unsupervised XLM: 33.3)
- De→En: 40.6 BLEU (SOTA: 34.3)
- Ro→En: 39.5 BLEU (SOTA: 31.8)

Слабее в направлении **с английского** — объясняется байт-уровневым BPE, оптимизированным под английский.

### SuperGLUE (Table 3.8)

Few-shot GPT-3 (32 примера): **71.8** (fine-tuned BERT-Large: 69.0, SOTA: 89.0).
- Сильно: COPA 92.0%, ReCoRD 91.1 F1 — near-SOTA
- Слабо: WiC 49.4% (random chance!) — задачи на сравнение двух предложений

### Commonsense Reasoning (Table 3.6)

PIQA few-shot: **82.8%** — превышает fine-tuned SOTA (79.4%).

### Reading Comprehension (Table 3.7)

CoQA few-shot: **85.0 F1** (только ~3 пункта ниже human baseline и fine-tuned SOTA).
SQuAD v2.0 few-shot: **69.8 F1** (улучшение на ~10 F1 против zero-shot).

### Arithmetic (Figure 3.10)

Few-shot GPT-3 175B:
- 2-digit addition: **100% accuracy**
- 2-digit subtraction: 98.9%
- 3-digit addition: 80.2%
- 4-digit operations: ~25%
- 2-digit multiplication: 29.2%

GPT-3 13B (второй по размеру): 2-digit — только ~50%, всё остальное < 10%. Резкий скачок именно на 175B.

## Limitations

Авторы честно документируют слабости:
- **NLI / ANLI:** Практически случайное угадывание на маленьких моделях, GPT-3 175B лишь "shows signs of life"
- **WiC:** 49.4% few-shot — задачи типа "одно и то же ли слово в двух предложениях" GPT-3 не даются
- **RACE:** На 45% ниже SOTA
- **Неизвестно, учится ли модель in-context** или просто распознаёт паттерны из pre-training
- Data contamination: некоторые бенчмарки могут присутствовать в Common Crawl

## My notes

- GPT-3 — поворотная точка: впервые задачи задаются **промптом**, а не архитектурой или fine-tuning. Это открывает эпоху prompt engineering.
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — ключевой концепт статьи. Авторы ссылаются на собственный предыдущий результат (Kaplan et al., 2020) и подтверждают: тренд степенного закона продолжается на ещё 2 порядка.
- Параллелизм по глубине и ширине через GPU — первый публичный намёк на сложность инфраструктуры для очень больших моделей.
- "Few-shot" в контексте LM — это не то же самое, что few-shot learning в CV. Никаких gradient updates. Это meta-learning через in-context conditioning.
- Дальнейшее развитие: [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] + InstructGPT → ChatGPT сделает GPT-3 значительно более полезным, решив проблему выравнивания с инструкциями.
