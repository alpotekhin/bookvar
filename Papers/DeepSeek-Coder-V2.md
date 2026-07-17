---
title: "DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence"
url: "https://arxiv.org/abs/2406.11931"
authors: [Qihao Zhu, Daya Guo, Zhihong Shao, DeepSeek-AI]
year: 2024
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/deepseek-coder-v2/paper.pdf|PDF]]"
concepts:
  - Mixture-of-Experts
  - Code Generation
  - Fill-In-Middle
  - Reinforcement Learning
---

# DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence

## TL;DR

DeepSeek-Coder-V2 — первая open-source code-модель на 236B параметров (21B active), которая обходит GPT-4 Turbo, Claude 3 Opus и Gemini 1.5 Pro на code и math бенчмарках. Строится через continued pre-training DeepSeek-V2 на дополнительных 6T токенах (60% код, 10% математика, 30% NL). Поддержка 338 языков программирования, 128K контекст.

## Проблема

К середине 2024 года open-source code-модели (StarCoder, CodeLlama, DeepSeek-Coder 33B) значительно уступали closed-source моделям (GPT-4 Turbo, Claude 3 Opus) в задачах code intelligence. Разрыв был особенно заметен на сложных задачах: SWE-Bench, LiveCodeBench, competition-level math. Ни одна open-source модель не достигала 10% на SWE-Bench — задаче решения реальных GitHub issues end-to-end.

Дополнительная проблема: code-specific модели (StarCoder, CodeLlama) обычно теряют general language capabilities. Нужна модель, которая сильна в коде **и** сохраняет general knowledge.

## Метод

### Стратегия обучения

Вместо тренировки с нуля — **continued pre-training** с промежуточного checkpoint DeepSeek-V2. Ключевое решение: начать не с финального checkpoint, а с промежуточного, чтобы модель лучше адаптировалась к code-heavy distribution.

Данные для дообучения:
- **60% source code** — 1170B code-related токенов с GitHub и CommonCrawl (338 языков вместо 86 у DeepSeek-Coder)
- **10% math** — 221B math-related токенов (2x больше DeepSeekMath)
- **30% natural language** — из training corpus DeepSeek-V2

Итого: 10.2T токенов exposure (4.2T от DeepSeek-V2 + 6T новых).

### Сбор code данных

Пайплайн на базе fastText для recall code/math страниц из CommonCrawl:
1. Seed corpus: StackOverflow, PyTorch docs, StackExchange
2. Тренировка fastText классификатора (с BPE токенизацией для китайского)
3. 3 итерации расширения: домен -> URLs -> новые страницы
4. Результат: 70B code-related + 221B math-related из web + 94B high-quality code из GitHub

### Архитектура

Идентична DeepSeek-V2 (MLA + DeepSeekMoE). Две версии:
- **236B** (21B active) — только Next-Token-Prediction
- **16B** (2.4B active) — Next-Token-Prediction + Fill-In-Middle (FIM)

FIM для 16B модели использует PSM (Prefix-Suffix-Middle) формат с rate 0.5.

### Context extension

128K через YARN: scale=40, alpha=1, beta=32. Двухэтапный continue training:
1. 32K sequence length, batch 1152, 1000 steps
2. 128K sequence length, batch 288, 1000 steps

### Alignment

**SFT:** 300M токенов из смеси code (20K), math (30K), general instruction данных.

**RL (GRPO):** Ключевой инсайт — **reward model лучше raw compiler signal**. На тестах LeetCode, использование reward model вместо прямого 0/1 feedback от компилятора давало заметно лучшие результаты, особенно когда test cases не покрывают все edge cases.

## Ключевые результаты

### Code benchmarks

| Benchmark | DeepSeek-Coder-V2 | GPT-4 Turbo | Claude 3 Opus | Gemini 1.5 Pro |
|-----------|-------------------|-------------|---------------|----------------|
| HumanEval | **90.2** | 88.2 | 84.9 | 81.7 |
| MBPP+ | **76.2** | 72.2 | 57.1 | 68.4 |
| LiveCodeBench | **43.4** | 45.7 | 34.1 | 28.7 |
| SWE-Bench | 12.7 | 18.3 | 2.7 | 11.7 |

### Math benchmarks

| Benchmark | DeepSeek-Coder-V2 | GPT-4 Turbo | Claude 3 Opus |
|-----------|-------------------|-------------|---------------|
| MATH | **75.7** | 73.4 | 60.1 |
| GSM8K | **94.9** | 93.7 | 95.0 |

### Сохранение general capabilities

MMLU: 79.2%, MT-Bench: 8.77, AlignBench: 7.84 — comparable с general-purpose моделями, значительно лучше других code-specific моделей.

### Needle In A Haystack (128K)

Perfect performance across всех context lengths и document depths на NIAH test. Long context extension через YARN работает robustly для code use cases.

### Подробные code benchmarks (instruct)

| Benchmark | DeepSeek-Coder-V2 | DeepSeek-Coder 33B | CodeLlama 70B | StarCoder2 15B |
|-----------|-------------------|--------------------|---------------|----------------|
| HumanEval | **90.2** | 56.1 | 53.0 | 46.3 |
| MBPP+ | **76.2** | 56.4 | 47.5 | 42.8 |

Leap в 30%+ по HumanEval over предыдущий DeepSeek-Coder — результат scale + data quality + alignment.

## Мои заметки

**Continued pre-training vs training from scratch:** Подход DeepSeek-Coder-V2 показывает, что можно взять сильную general-purpose MoE модель и дообучить на code-heavy данных без потери general capabilities. Это экономически выгоднее и позволяет наследовать все архитектурные преимущества (MLA, DeepSeekMoE).

**338 языков программирования** — амбициозное расширение с 86. Но нужно понимать, что для большинства из этих языков данных мало. Реальная сила — в мажорных языках, где есть достаточный объём training data.

**Reward model > compiler signal** — контринтуитивный результат. Казалось бы, binary "pass/fail" от компилятора — идеальный reward. Но на практике ограниченные test cases создают шум, и learned reward model генерализирует лучше. Этот инсайт был позже развит в DeepSeek-V3 и R1.

**Ablation на 1B модели:** Новый code corpus даёт +6.7% HumanEval и +9.4% MBPP по сравнению со старым corpus DeepSeek-Coder при тех же токенах — data quality matters.

**Первая open-source модель с 10%+ на SWE-Bench** — milestone для code intelligence. SWE-Bench требует end-to-end решения реальных GitHub issues, это гораздо сложнее синтетических бенчмарков.

**Data collection pipeline** — iterative fastText approach с BPE tokenization заслуживает внимания. Стандартный fastText на whitespace-separated tokens плохо работает для Chinese. Использование BPE tokenizer от DeepSeek-V2 значительно улучшает recall accuracy, что позволяет собрать больше quality data из CommonCrawl.

**3 итерации расширения данных** — domain -> URL annotation -> seed expansion. Каждая итерация добавляет новые страницы в seed corpus. Финальный результат: 1170B code + 221B math из web — масштабный dataset, построенный автоматически.

**FIM для 16B, но не для 236B** — pragmatic choice. Fill-In-Middle важен для code completion (IDE use case), но для 236B модели основной use case — chat/instruction, где FIM менее критичен. 16B модель — deployment target для code completion plugins.

**Gradient instability** — авторы столкнулись с training spikes из-за exponential normalization technique и откатились к conventional normalization. Прозрачность в описании таких проблем ценна для community.

**GRPO для code RL** — тот же алгоритм что в DeepSeek-V2. Consistency подхода across модели показывает, что GRPO — robust RL method, работающий для разных доменов (general, code, math).

**Общий объём exposure 10.2T** — больше, чем у большинства code-specific моделей на тот момент. Комбинация general knowledge (4.2T от DeepSeek-V2) + domain expertise (6T code/math) даёт модель, которая не только кодирует, но и рассуждает.

**Live Code Bench** — один из наиболее challenging бенчмарков, потому что обновляется регулярно (contamination resistance). 43.4% — сильный результат, хотя GPT-4 Turbo лидировал с 45.7%. На AIME 2024 competition DeepSeek-Coder-V2 превосходит все closed-source модели — math reasoning сильно усилен code training data.

**Code + Math synergy** — ключевое наблюдение. Модель, обученная на 60% code + 10% math, показывает significant improvements на обоих доменах. Это подтверждает гипотезу о transfer между code reasoning и mathematical reasoning — оба требуют structured, step-by-step problem solving.

**Две версии для разных use cases** — 236B (full capability, API serving) и 16B (code completion, local IDE). 16B с FIM support — practical choice для VS Code extensions, GitHub Copilot-like products. 2.4B activated params позволяют быстрый inference.

**Intermediate checkpoint strategy** — начало continued pre-training с промежуточного, а не финального checkpoint DeepSeek-V2. Это позволяет модели лучше адаптировать remaining training к code-heavy distribution без catastrophic forgetting general knowledge, которое уже частично закреплено.

**SFT dataset composition:** 300M токенов — compact, but targeted. 20K code + 30K math + general instruction. Меньше, чем у многих конкурентов, но focused quality. Training: cosine schedule, LR 5e-6, batch 1M tokens, total 1B tokens.

**AIME 2024 surpasses closed-source** — на competition-level math benchmark DeepSeek-Coder-V2 обходит GPT-4 Turbo, Claude 3 Opus, и Gemini 1.5 Pro. Это counterintuitive: code model beating general models на math competition. Explanation: code training develops structured reasoning и problem decomposition skills, which transfer to math.

**DeepSeek-Coder-V2-Lite (16B/2.4B active)** — practical deployment model. Enable FIM = code completion plugin. Disable FIM = instruction following. Two training objectives для different use cases из одного checkpoint.

**Open source under permissive license** — both research и unrestricted commercial use. Это accelerated adoption и community development (fine-tuning, adaptation, tool integration).

**Aider benchmark: 73.7%** — Aider оценивает способность модели выполнять реальные edit instructions на codebase. 73.7% — best among all models including GPT-4 Turbo (68.4%). Это practical metric: не синтетические задачи, а реальный code editing workflow.

**Training instability с exponential normalization** — авторы прозрачно описали revert к conventional normalization. Exponential normalization может создавать gradient spikes при large scale, и decision to revert — responsible engineering. Не все "innovative" techniques работают при scale.

**DeepSeekMath pipeline для math data** — reuse проверенного pipeline для сбора math-related web data из CommonCrawl. 221B math tokens = 2x больше оригинального DeepSeekMath corpus. Pipeline transferability между проектами — важный engineering asset.

**338 programming languages** — полный список в appendix. Включает экзотические языки (XSLT, COBOL, Fortran). Для minority languages данных мало, но model capacity позволяет хотя бы basic understanding. Для major languages (Python, JS, Java, C++) — production-grade quality.
