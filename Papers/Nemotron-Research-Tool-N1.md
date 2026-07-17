---
title: "Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models via RL"
url: https://arxiv.org/abs/2505.00024
authors: [NVIDIA Research]
year: 2025
date_reviewed: 2026-04-12
type: source-note
status: legacy
category: paper
tags:
  - RL
  - tool-use
  - GRPO
  - RLVR
Organization: NVIDIA
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]]"
  - "[[02 Areas/ML & DL/Concepts/Reasoning/Toolformer|Toolformer]]"
---

# Nemotron-Research-Tool-N1

**Authors:** NVIDIA Research
**Published:** May 2025 (arXiv:2505.00024)
**URL:** https://arxiv.org/abs/2505.00024

## TL;DR

Серия tool-calling reasoning моделей (7B, 14B), обученных через rule-based RL с **binary reward** (format validity + functional correctness). Tool-N1-7B/14B превосходят GPT-4o на major benchmarks. Главный вывод: **SFT→RL pipeline не обязательно лучше pure RL** — совпадает с [[02 Areas/ML & DL/Papers/ToolRL|ToolRL]], противоречит [[02 Areas/ML & DL/Papers/ASTRA|ASTRA]].

## Method

- **Модели:** Tool-N1-7B, Tool-N1-14B
- **RL:** [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] с binary reward
- **Данные:** 5,518 distilled reasoning trajectories (от сильной модели)
- **Reward:** бинарный — format validity + functional correctness. Никакого granular reward.

Три сравниваемых подхода:
1. SFT only
2. Pure RL
3. SFT → RL

### Ключевое отличие от ToolRL

ToolRL использует granular step-level reward (-3..+3). Tool-N1 использует **binary** reward. Тем не менее, binary reward достаточен — модель сама развивает reasoning стратегии без навязывания конкретных промежуточных паттернов.

## Key Results

- Tool-N1-7B и Tool-N1-14B **превосходят GPT-4o** на major benchmarks
- "The widely adopted SFT-then-RL paradigm does not necessarily outperform pure RL"
- Binary reward = достаточный сигнал для tool-use learning

## My notes

### Позиция в ландшафте

Три работы формируют спектр:

| Paper | SFT нужен? | Reward | Horizon |
|---|---|---|---|
| [[02 Areas/ML & DL/Papers/ASTRA|ASTRA]] | Да (claim) | Trajectory F1 | Multi-turn |
| [[02 Areas/ML & DL/Papers/ToolRL|ToolRL]] | Нет (вредит) | Step-level granular | Single-turn |
| **Tool-N1** | Нет (не обязательно) | Binary | Multi-turn |

Tool-N1 особенно интересен тем, что работает на multi-turn (как ASTRA) но без SFT (как ToolRL). Это ослабляет аргумент ASTRA что SFT необходим именно для multi-turn.

### Минимализм reward design

5,518 trajectories + binary reward → beats GPT-4o. Контраст с ASTRA (19K тулов, 7-dimensional scoring, F1 reward, similarity-band mixing). Возможно, complexity ASTRA pipeline не оправдана.
