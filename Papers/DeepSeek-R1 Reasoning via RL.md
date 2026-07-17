---
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
url: https://arxiv.org/abs/2501.12948
authors: "DeepSeek-AI"
year: 2025
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/deepseek-r1/source]]"
concepts: [Reinforcement Learning, Chain of Thought, GRPO, Reasoning, Knowledge Distillation, Reward Hacking]
---
# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## TL;DR

DeepSeek-R1 показывает, что **чистое RL без SFT** (DeepSeek-R1-Zero) может вызвать emergent reasoning: self-reflection, verification, dynamic strategy. Финальная модель DeepSeek-R1 через multi-stage pipeline (cold-start SFT → RL → rejection sampling → RL) достигает уровня OpenAI o1 на math/code reasoning. Дистилляция в маленькие модели сохраняет reasoning capability.

## Problem

Reasoning в LLM традиционно строился на human-annotated CoT demonstrations. Это ограничивает масштабируемость, вносит cognitive biases и cap-ает модель уровнем человеческих exemplars. Вопрос: можно ли получить superior reasoning через pure RL, без человеческих reasoning traces?

## Method

### DeepSeek-R1-Zero (Pure RL)
- Base model: DeepSeek-V3-Base (без SFT!)
- RL algorithm: **GRPO** (Group Relative Policy Optimization) -- упрощённый PPO без critic model. Для каждого вопроса семплируется G=16 ответов, advantage считается через нормализацию групповых rewards.
- **Rule-based rewards only**: accuracy reward (проверка ответа) + format reward (наличие `<think>` и `<answer>` тегов). Никаких neural reward models.
- lr=3e-6, KL coeff=0.001, temperature=1, max length 32K→65K tokens
- 10,400 steps = 1.6 epochs

### Emergent Behaviors
- Модель самостоятельно развивает: self-verification, reflection, exploration of alternative approaches
- **"Aha moment"**: внезапное увеличение использования слова "wait" в reasoning -- модель учится перепроверять свои шаги
- Response length растёт с 2.5K до 17K+ токенов за тренинг

### DeepSeek-R1 (Multi-Stage Pipeline)
1. **Cold-start SFT**: тысячи примеров с human-aligned thinking process
2. **RL Stage 1**: reasoning prompts + rule-based rewards + language consistency reward
3. **Rejection Sampling + SFT**: sampling из checkpoint, фильтрация, SFT на reasoning + non-reasoning data
4. **RL Stage 2**: diverse prompts, rule-based + preference reward model (helpfulness + safety), 1700 steps

### Reward Design
- Reasoning: rule-based (accuracy + format + language consistency)
- General: model-based (helpfulness RM на 66K пар, safety RM на 106K примеров)
- Language consistency: доля целевого языка в CoT

### Distillation
- R1 → Qwen/Llama маленьких размеров (1.5B-70B)
- Distilled модели превосходят оригинальные instruct-версии на reasoning

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/deepseek-r1/fig1.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/deepseek-r1/fig2.png]]

## Key Results

| Benchmark | DeepSeek-R1 | OpenAI o1-0120 | Claude-3.5-Sonnet |
|---|---|---|---|
| AIME 2024 (Pass@1) | 79.8 | 79.2 | 16.0 |
| MATH-500 | 97.3 | 96.4 | 78.3 |
| Codeforces (Rating) | 2029 | 1891 | -- |
| GPQA Diamond | 71.5 | 75.7 | 65.0 |
| MMLU | 90.8 | 91.8 | 88.3 |
| LiveCodeBench (Pass@1) | 65.9 | 63.4 | -- |
| SWE-Bench Verified | 49.2 | 48.9 | 50.8 |

- R1-Zero: AIME 15.6% → 77.9% (pass@1), 86.7% с self-consistency
- AlpacaEval 2.0: 87.6% LC-winrate, ArenaHard: 92.3%
- Distilled Qwen-32B: AIME 72.6%, MATH 94.3%

## My notes

- Главный результат -- не бенчмарки, а **proof of concept**: pure RL без SFT порождает emergent reasoning. Это сильный аргумент в пользу того, что reasoning -- emergent свойство, которое можно "разбудить" правильным reward signal.
- "Aha moment" -- красивая демонстрация, но по сути это просто модель, которая научилась, что self-correction увеличивает reward. Не стоит антропоморфизировать.
- Практически, pure RL (R1-Zero) имеет проблемы: language mixing, poor readability, limited non-reasoning capability. Multi-stage pipeline R1 решает это, но ценой сложности.
- GRPO вместо PPO -- значимое упрощение: нет critic model, advantage через group normalization. Это делает масштабирование RL дешевле.
- Reward hacking остаётся реальной проблемой -- preference RM используется только 400 steps в финальном RL, иначе модель начинает хакать.
- Token efficiency -- слабое место: overthinking на простых задачах. R1 тратит тысячи токенов на задачи, которые V3 решает за сотни.
- Distillation preserves reasoning -- это практически самый ценный результат для production: можно получить reasoning в 7B модели.
