---
title: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
url: https://arxiv.org/abs/2307.09288
authors: [Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, et al.]
year: 2023
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - LLM
  - LLaMA
  - arch
Date: 2023-01-07
Organization: Meta
Parent item:
  - "[[LLaMA]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/LLaMA 2|LLaMA 2]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
raw: "[[02 Areas/ML & DL/raw/papers/llama-2/paper.txt]]"
---

# Llama 2: Open Foundation and Fine-Tuned Chat Models

**Authors:** Hugo Touvron, Louis Martin, Kevin Stone et al. (GenAI, Meta)
**Published:** 2023 (arXiv:2307.09288v2, Jul 2023)
**URL:** https://arxiv.org/abs/2307.09288

## TL;DR

Llama 2 — серия pretrained + instruction-tuned LLM от Meta (7B–70B), открыто доступных для исследовательского и коммерческого использования. Ключевые улучшения над LLaMA 1: **40% больше токенов** (2T), **вдвое больший контекст** (4k), **Grouped-Query Attention (GQA)** для крупных моделей. Главный вклад — детальное описание pipeline для **Llama 2-Chat**: SFT → итеративный RLHF с двумя отдельными reward моделями (helpfulness + safety) через Rejection Sampling + PPO + Ghost Attention для multi-turn консистентности.

## Problem

Существующие open-source LLM (LLaMA 1, Falcon, BLOOM) конкурентоспособны с закрытыми pretrained моделями (GPT-3, Chinchilla), но не с закрытыми **product** моделями (ChatGPT, BARD, Claude). Закрытые product LLM сильно fine-tuned для alignment с человеческими предпочтениями — этот процесс требует значительного compute и human annotation, и непрозрачен.

Llama 2 решает это: выпускает полный pipeline с детальным описанием fine-tuning и safety методологии, позволяя сообществу воспроизводить и улучшать alignment.

## Method

### Pretraining

Архитектура та же, что у LLaMA 1: [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] Transformer с **RMSNorm pre-norm, SwiGLU, RoPE**.

**Изменения относительно LLaMA 1 (Table 1):**

| Параметр | LLaMA 1 | LLaMA 2 |
|---------|---------|---------|
| Токены | 1.0T / 1.4T | **2.0T** |
| Context length | 2k | **4k** |
| GQA | ✗ | ✓ (34B, 70B) |
| Доступные размеры | 7B, 13B, 33B, 65B | 7B, 13B, 34B, 70B |

**Grouped-Query Attention (GQA):** В моделях 34B и 70B — несколько query головок совместно используют одну key/value головку, что снижает memory bandwidth при inference при сохранении качества.

**Данные:** 2T токенов из публично доступных источников (без данных Meta-продуктов). Up-sampling фактических источников для снижения галлюцинаций. Более строгая очистка данных.

**CO₂:** 3.3M GPU-часов на A100. Суммарные эмиссии: **539 tCO₂ eq** — 100% компенсированы программой устойчивости Meta.

### Fine-tuning Pipeline (Llama 2-Chat)

Training pipeline (Figure 4): **Pretrained LLaMA 2 → SFT → Iterative RLHF (Rejection Sampling + PPO)**

**Шаг 1: Supervised Fine-Tuning (SFT)**
- Начало с публично доступных instruction tuning данных
- Ключевой вывод: **Quality is All You Need** — несколько тысяч высококачественных примеров лучше, чем миллионы низкого качества
- Собрано **27,540 высококачественных SFT аннотаций** от вендоров (остановились на этом числе)
- SFT hyperparams: cosine LR 2×10⁻⁵, weight decay 0.1, batch size 64, sequence length 4096, **2 эпохи**
- Оригинальный факт: выходы SFT-модели уже конкурентоспособны с handwritten SFT-аннотациями → переориентировали усилия на preference annotation

**Шаг 2: Reward Modeling**
- **Два отдельных reward model**: Helpfulness RM и Safety RM (tension between objectives!)
- Инициализируются от pretrained chat model checkpoint — знают, что "знает" chat model
- Архитектура идентична pretrained LLM, но classification head заменена на regression head
- Loss function с margin (уравнение 2): `L = -log(σ(r_θ(x,y_c) - r_θ(x,y_r) - m(r)))`, где `m(r)` — дискретная функция степени предпочтения
- **Данные**: >1M бинарных сравнений от Meta (Table 6), плюс open-source датасеты
- Helpfulness RM: Avg accuracy 70.6%. Safety RM: 64.3%. Оба превосходят GPT-4 (58.6%) и Open Assistant.

**Шаг 3: Iterative RLHF (RLHF-V1...V5)**

Два алгоритма:
- **Rejection Sampling**: семплируется K ответов → выбирается лучший по reward → fine-tuning на нём. Использовался до RLHF-V4; rejection sampling только для 70B, меньшие модели дистиллируются из 70B.
- **PPO (Proximal Policy Optimization)**: стандарт RLHF. Применяется поверх Rejection Sampling checkpoint с V4+.

**PPO objective** (уравнение 4):
`R(g|p) = R̃_c(g|p) - β * KL(π_θ(g|p) || π_0(g|p))`

Составной reward `R_c`: если prompt потенциально unsafe → Safety RM; иначе → Helpfulness RM (threshold 0.15 safety score). Финальные scores whitened через logit для стабильности.

PPO гиперпараметры: AdamW (β₁=0.9, β₂=0.95), batch size 512, mini-batch 64, 1 gradient step per mini-batch, KL penalty β=0.01 (7B/13B) / 0.005 (34B/70B), 200–400 итераций.

**Шаг 4: Ghost Attention (GAtt)**
Техника для поддержания консистентности system message через несколько диалоговых ходов. Без GAtt модель "забывает" начальную инструкцию к середине диалога.

## Key Results

### Pretrained Model Benchmarks (Table 3)

Llama 2 70B vs всех open-source моделей:

| Категория | LLaMA 1 65B | LLaMA 2 70B | MPT 30B | Falcon 40B |
|-----------|-------------|-------------|---------|------------|
| Code | 30.7 | **37.5** | 28.9 | 15.2 |
| Commonsense | 70.7 | **71.9** | 64.9 | 69.2 |
| World Knowledge | 60.5 | **63.6** | 50.0 | 56.7 |
| Math | 30.8 | **35.2** | 9.1 | 12.6 |
| MMLU | 63.4 | **68.9** | 46.9 | 55.4 |
| BBH | 43.5 | **51.2** | 38.0 | 37.1 |

### vs Closed-Source Models (Table 4)

| Benchmark | GPT-3.5 | GPT-4 | PaLM-2-L | LLaMA 2 70B |
|-----------|---------|-------|----------|------------|
| MMLU | 70.0 | 86.4 | 78.3 | **68.9** |
| TriviaQA | – | – | 86.1 | **85.0** |
| GSM8K | 57.1 | 92.0 | 80.7 | **56.8** |
| HumanEval | 48.1 | 67.0 | – | **29.9** |

LLaMA 2 70B близка к GPT-3.5 на MMLU и GSM8K, но значительно уступает на coding и GPT-4/PaLM-2-L в целом.

### Helpfulness & Safety (Human Eval)

- **Helpfulness (Figure 1):** Llama 2-Chat 70B — лучшая среди open-source моделей, приближается к ChatGPT
- **Safety (Figure 3):** Llama 2-Chat превосходит все open-source по safety, сравнима с закрытыми

## Limitations

- Значительный gap с GPT-4, особенно на coding (HumanEval: 29.9% vs 67%)
- Тестирование только на английском
- Калибровка reward model деградирует без постоянного пополнения данных
- RLHF tension helpfulness/safety требует двух отдельных RM

## My notes

- **Iterative RLHF** — ключевой практический инсайт. Одного прохода недостаточно; каждая итерация RLHF сдвигает distribution модели, и reward model нужно постоянно обновлять на новых данных.
- **Quality over quantity для SFT** — 27K высококачественных примеров > миллионов средних. Это стало стандартной мудростью после LLaMA 2.
- **Два RM вместо одного** — важный архитектурный выбор. Helpfulness vs Safety — конкурирующие objectives, единая модель теряет на обоих. Впоследствии OpenAI тоже разделила.
- **Rejection Sampling как upstream для PPO** — практичный подход: сначала собери хорошие данные через RS, потом дообучай через PPO. Гибрид работает лучше каждого по отдельности.
- **Ghost Attention** — простая но эффективная техника для multi-turn; предшественник современных system prompt injection методов.
- LLaMA 2 — первая полностью открытая модель с **детально описанным RLHF pipeline**, это сделало её основой для огромного количества академических работ по alignment.
