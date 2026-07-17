---
title: "ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools"
url: "https://arxiv.org/abs/2406.12793"
authors: [Team GLM, Zhipu AI, Tsinghua University]
year: 2024
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/glm-4/paper.pdf|PDF]]"
concepts:
  - LLM Alignment
  - Tool Use
  - Agent
  - Long Context
---

# ChatGLM: From GLM-130B to GLM-4 All Tools

## TL;DR

ChatGLM — серия LLM от Zhipu AI и Tsinghua University, прошедшая путь от GLM-130B (2022) до GLM-4 All Tools (2024). GLM-4 pre-trained на ~10T токенов (Chinese + English), поддерживает 128K контекст, close to GPT-4 по academic benchmarks (83.3 MMLU). GLM-4 All Tools автономно выбирает и использует инструменты (web browser, Python, text-to-image). Open-source GLM-4-9B — 10M+ downloads на HuggingFace в 2023.

## Проблема

Создание конкурентоспособной Chinese-English LLM, которая:
1. Достигает уровня GPT-4 на стандартных benchmarks
2. Поддерживает длинный контекст (128K-1M)
3. Умеет автономно использовать инструменты для решения сложных задач
4. Имеет open-source версии для community

## Метод

### Эволюция архитектуры (4 поколения)

**GLM-130B (2022):** DeepNorm + RoPE + Gated Linear Unit (GeLU). MMLU 44.8%.

**ChatGLM2 (2023-06):** Better architecture, 32K context, Multi-Query Attention (+42% inference speed). MMLU 66.6%.

**ChatGLM3 (2023-10):** Native function call и code interpreter. Agent capabilities. MMLU 71.0%.

**GLM-4 (2024-01):** Финальная архитектура:
- **No Bias Except QKV** — ускорение + лучшая length extrapolation
- **RMSNorm + SwiGLU** вместо LayerNorm + ReLU
- **2D RoPE** — адаптированный для GLM архитектуры
- **Group Query Attention (GQA)** — сокращение KV cache, увеличение FFN dim до 10/3 hidden size
- MMLU 83.3%

### Pre-training Data

~10T токенов, mostly Chinese и English + 24 языка. Pipeline:
1. **Deduplication** (exact + fuzzy)
2. **Filtering** — offensive language, placeholder, source code removal
3. **Tokenization** — byte-level BPE, vocabulary 150K (merged Chinese + multilingual + cl100k_base)

Re-weighting: увеличенный вес высококачественных источников (books, Wikipedia).

### Alignment

Multi-stage post-training:
1. **SFT** на authentic human prompts (не template-based, не model-generated)
2. **RLHF** для mitigation response rejection, safety, bilingual mixing, multi-turn coherence
3. **Safety alignment**

Ключевой инсайт: **authentic human interaction data >> template-based/synthetic data** для SFT.

### Context Extension: от 2K до 128K/1M

- **ChatGLM:** 2K
- **ChatGLM2:** 32K (FlashAttention)
- **GLM-4:** 128K (position encoding extension + continual training on long text + long context alignment)
- **GLM-4-9B-Chat-1M:** экспериментальная версия с 1M context (~2M Chinese characters)

### GLM-4 All Tools

Модель aligned для автономного tool use:
- **Web Browser** — multi-round search и browsing
- **Python Interpreter** — math solving, data analysis
- **CogView** — text-to-image generation
- **User-defined functions** — custom API calls

Pipeline: анализ задачи -> plan -> sequential tool calls -> intermediate feedback -> решение.

Платформа **GLMs** позволяет пользователям создавать custom agents с комбинацией инструментов.

### Developed Techniques

Серия опубликованных техник:
- **LongAlign** — recipe для long context alignment (128K, comparable с Claude 2 и GPT-4 Turbo)
- **ChatGLM-Math** — self-critique для math problem solving
- **ChatGLM-RLHF** — PPO и DPO практики
- **Self-Contrast** — feedback-free alignment без human preference data
- **AgentTuning** — agent capabilities через AgentInstruct dataset
- **APAR** — auto-parallel auto-regressive generation для ускорения inference

## Ключевые результаты

### Academic Benchmarks (GLM-4 0520)

| Benchmark | GLM-4 | GPT-4 (0314) | GPT-4 Turbo (Apr) | Claude 3 Opus |
|-----------|-------|-------------|-------------------|---------------|
| MMLU | 83.3 | 86.4 | 86.7 | 86.8 |
| GSM8K | 93.3 | 92.0 | 95.6 | 95.0 |
| MATH | 61.3 | 52.9 | 73.4 | 60.1 |
| BBH | 84.7 | 83.1 | 88.2 | 86.8 |
| GPQA | 39.9 | 35.7 | 49.3 | 50.4 |
| HumanEval | 78.5 | 67.0 | 88.2 | 84.9 |

### Instruction Following (IFEval)

GLM-4 (0520) на уровне GPT-4 Turbo: 83.7 Loose-Prompt EN, 79.7 Loose-Prompt CN.

### Chinese Alignment (AlignBench)

GLM-4 **outperforms GPT-4** и matches GPT-4-Turbo по 8 dimensions на китайском.

### Long Context (LongBench-Chat)

GLM-4 (128K): 87.3 vs GPT-4 Turbo 87.2 vs Claude 3 Opus 87.7.

### Open-source GLM-4-9B

Прогрессия по поколениям:
| | ChatGLM-6B | ChatGLM2-6B | ChatGLM3-6B | GLM-4-9B |
|---|---|---|---|---|
| GSM8K | 1.5 | 25.9 | 72.3 | **84.0** |
| MMLU | 25.2 | 45.2 | 61.4 | **74.7** |
| HumanEval | 0.0 | 9.8 | 58.5 | **70.1** |

## Мои заметки

**Уникальность GLM:** Это одна из немногих non-decoder-only архитектур, которая выжила в эпоху LLM. Оригинальный GLM использовал autoregressive blank infilling — отличие от GPT-style left-to-right generation. К GLM-4 они адаптировали стандартные best practices (GQA, SwiGLU, RMSNorm), но сохранили 2D RoPE.

**All Tools — ранний agent-first подход:** GLM-4 All Tools вышел примерно одновременно с GPT-4 All Tools, но open-source. Подход "модель сама решает, какой инструмент использовать" — в 2024 стал стандартом для agent frameworks.

**10M+ downloads ChatGLM-6B** — показатель реального demand на lightweight Chinese LLM. INT4 quantization на consumer GPU — правильный product decision.

**Authentic human data >> synthetic** — повторяющийся finding в alignment research. Template-based и model-generated data создают distribution mismatch с реальными user intent.

**GLM-4-Air** — интересный продуктовый ход: comparable quality с GLM-4 (0116) при lower latency и inference cost. Distillation или architecture optimization — не раскрыто.

**Прогресс 6B серии:** от MMLU 25.2 до 74.7 за три поколения за 1.5 года — впечатляющая демонстрация того, как pre-training и alignment recipes улучшаются со временем.

**Self-Contrast** — feedback-free alignment через self-generated negative samples. Модель сама генерирует "плохие" ответы для собственного RLHF, устраняя need для дорогих human preference annotations. Если работает robustly, это scalable alternative к human feedback.

**Emergent abilities переосмыслены:** Команда GLM предложила redefine emergent abilities не через model size, а через pre-training loss. При одном и том же loss, модели разного размера показывают одинаковую downstream performance. Некоторые задачи (MMLU, GSM8K) улучшаются выше random только когда loss падает ниже определённого порога. Это более principled framework для reasoning about model capabilities.

**APAR (Auto-Parallel Auto-Regressive)** — intriguing technique. Модель обучается планировать параллельную генерацию structured responses (lists, hierarchies). Позволяет генерировать несколько branches одновременно. Practical speedup для длинных structured outputs.

**Vocabulary 150K** — один из крупнейших. Merge Chinese + multilingual + cl100k_base (OpenAI). Большой vocab = лучшая compression = быстрее тренировка и инференс на tokenized data, но potentially хуже coverage на rare tokens.

**No Bias Except QKV** — specific choice: bias удалён отовсюду кроме QKV attention matrices. Slight improvement в length extrapolation — возможно, bias в non-attention layers creates position-dependent artifacts.

**FFN size = 10/3 hidden** — необычная пропорция. При переходе от MHA к GQA экономятся параметры на KV heads, и эти параметры перераспределяются в FFN. Это maintains total model size при better balance между attention и feed-forward compute.
