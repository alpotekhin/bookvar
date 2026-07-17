---
title: "GPT-4 Technical Report"
url: https://arxiv.org/abs/2303.08774
authors: [OpenAI]
year: 2023
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - GPT
  - LLM
Date: 2023-01-03
Organization: OpenAI
Parent item:
  - "[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/GPT-4|GPT-4]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/gpt-40/paper.txt]]"
---

# GPT-4 Technical Report

**Authors:** OpenAI
**Published:** 2023 (arXiv:2303.08774v6, Mar 2024)
**URL:** https://arxiv.org/abs/2303.08774

## TL;DR

GPT-4 — крупномасштабная **мультимодальная** (текст + изображения) модель, которая достигает человеческого уровня на широком диапазоне профессиональных и академических экзаменов, включая top 10% на Uniform Bar Exam. Техотчёт намеренно не раскрывает архитектуру, размер, данные обучения — из-за конкурентных и safety-соображений. Ключевые технические вклады: **predictable scaling** (возможность предсказывать поведение большой модели по маленьким) и улучшенный alignment через RLHF + rule-based reward models.

## Problem

Предыдущие GPT-модели:
- Нередко галлюцинируют (ненадёжны в фактах)
- Однонаправленные: только текст, без визуального ввода
- Непредсказуемо ведут себя при масштабировании: исследователи не могут заранее оценить capabilities большой модели без её полного обучения
- Небезопасны: легко вынуждаются генерировать вредоносный контент

GPT-4 решает эти проблемы системно: predictable scaling позволяет «бронировать» safety-меры до деплоя, RLHF + RBRM фильтрует нежелательное поведение.

## Method

### Архитектура

**Намеренно не раскрыта** (раздел 2: "This report contains no further details about the architecture, hardware, training compute, dataset construction, training method"). Известно:
- Transformer-based, предобучен на предсказание следующего токена (autoregressive)
- Post-training alignment через [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]
- Принимает **текст и изображения** на вход (Vision-capable вариант)

### Predictable Scaling (Section 3)

Ключевое техническое достижение: разработана инфраструктура и методы оптимизации с **предсказуемым поведением** на разных масштабах.

**Loss Prediction (Figure 1):** Используется степенной закон с irreducible loss term:
`L(C) = aC^b + c`
Модель обучена на моделях, использующих до 10,000× меньше compute, чем GPT-4. Предсказание сделано до окончания обучения GPT-4 — и оказалось точным.

**HumanEval Prediction (Figure 2):** Предсказан pass rate на подмножестве HumanEval по формуле:
`-E_P[log(pass_rate(C))] = α * C^{-k}`
Predictions для 6 difficulty buckets совпали с реальными результатами GPT-4.

Важное наблюдение: некоторые capabilities **непредсказуемы** — GPT-4 меняет тренд на Inverse Scaling Prize задачах (Hindsight Neglect), где меньшие модели деградируют с ростом размера, а GPT-4 резко улучшается (Figure 3).

### Safety Pipeline (Section 6)

Два компонента:
1. **Safety-relevant RLHF prompts** — дополнительные данные для reward model по нежелательным категориям
2. **Rule-Based Reward Models (RBRMs)** — zero-shot GPT-4 classifiers, дающие дополнительный reward signal во время RLHF fine-tuning

**Red-teaming:** 50+ экспертов из областей long-term AI alignment, cybersecurity, biorisk, international security тестировали модель.

**Calibration:** Pre-trained GPT-4 хорошо калиброван (ECE ≈ 0.007 на MMLU). После RLHF post-training калибровка заметно ухудшается (ECE ≈ 0.074) — типичная проблема alignment (Figure 8).

## Key Results

### Academic and Professional Exams (Table 1)

| Экзамен | GPT-4 | GPT-3.5 |
|---------|-------|---------|
| Uniform Bar Exam | **298/400 (~90th percentile)** | 213/400 (~10th) |
| LSAT | 163 (~88th) | 149 (~40th) |
| SAT Math | 700/800 (~89th) | 590/800 (~70th) |
| GRE Verbal | **169/170 (~99th)** | 154/170 (~63rd) |
| GRE Quantitative | 163/170 (~80th) | 147/170 (~25th) |
| USABO Semifinal 2020 | 87/150 (**99th–100th**) | 43/150 (31st–33rd) |
| Leetcode Easy | 31/41 | 12/41 |
| Leetcode Hard | 3/45 | 0/45 |

GPT-4 достигает human-level performance на большинстве экзаменов. Сравнение bar exam: GPT-4 top 10% vs GPT-3.5 bottom 10% — принципиальный скачок.

### NLP Benchmarks (Table 2)

| Бенчмарк | GPT-4 | GPT-3.5 | LM SOTA (few-shot) | Best overall SOTA |
|----------|-------|---------|---------------------|------------------|
| MMLU (57 subjects) | **86.4%** | 70.0% | 70.7% | 75.2% |
| HellaSwag | **95.3%** | 85.5% | 84.2% | 85.6% |
| ARC Challenge | **96.3%** | 85.2% | 85.2% | 86.5% |
| WinoGrande | **87.5%** | 81.6% | 85.1% | 85.1% |
| HumanEval (Python) | **67.0%** | 48.1% | 26.2% | 65.8% |
| GSM-8K (math) | **92.0%** | 57.1% | 58.8% | 87.3% |
| DROP (F1) | 80.9 | 64.1 | 70.8 | 88.4 (SOTA) |

GPT-4 бьёт все LM few-shot бенчмарки. На HumanEval (Python coding) — почти вдвое лучше предыдущего LM SOTA.

### Multilingual MMLU (Figure 5)

GPT-4 превосходит GPT-3.5 English (70.1%) на 24 из 26 языков:
- Итальянский: 84.1%, Немецкий: 83.7%, Русский: 82.7%
- Суахили (low-resource): 78.5%, Валлийский: 77.5%, Бенгальский: 73.2%

### Factuality & Safety

**TruthfulQA (Figure 7):** GPT-4 RLHF значительно превосходит GPT-3.5 и Anthropic-LM. Base GPT-4 без RLHF — только немного лучше GPT-3.5.

**Internal factuality eval (Figure 6):** GPT-4 улучшает GPT-3.5 на **19 percentage points** на adversarial factuality evaluations по 9 категориям (learning, technology, writing, history, math, science, recommendation, code, business).

**User preference:** GPT-4 responses предпочтены GPT-3.5 на **70.2%** из 5,214 промптов от реальных пользователей ChatGPT.

## Limitations

Честно задокументированы авторами:
- **Hallucinations:** Всё ещё галлюцинирует, но реже чем GPT-3.5 (-19 ppts)
- **Cutoff:** Нет знаний о событиях после Sep 2021 (большая часть pre-training данных)
- **No learning from experience:** Не обновляется на основе диалога
- **Calibration degrades post-RLHF:** ECE растёт с 0.007 до 0.074
- **Reasoning errors:** Может делать простые логические ошибки
- **Социальные bias:** Продолжаются работы по характеризации и mitigation

## My notes

- GPT-4 Technical Report — намеренно минималистичен технически. Это само по себе стало поворотным моментом: era of "closed" LLMs. OpenAI перестала делиться деталями архитектуры/размера.
- **Predictable scaling** — ключевой практический инсайт: можно тестировать safety и capabilities до окончания обучения большой модели. Это меняет подход к alignment.
- Разрыв GPT-4 vs GPT-3.5 на Bar Exam (90th vs 10th percentile) — один из самых цитируемых примеров [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]].
- RLHF post-training ухудшает calibration — фундаментальное противоречие между alignment и честностью предсказаний. Это открытая проблема.
- RBRMs (rule-based reward models) = GPT-4 evaluating GPT-4. LLM-as-judge pattern — появился здесь как safety-инструмент, потом стал mainstream в eval.
