---
title: "ToolRL: Reward is All Tool Learning Needs"
url: https://arxiv.org/abs/2504.13958
authors: [Cheng Qian, Emanuele La Malfa, Suyu Ge, Michael Backes, Yang Zhang, Huan Sun]
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
  - reward-design
Organization: OSU / CISPA / Oxford
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]]"
  - "[[02 Areas/ML & DL/Concepts/Training/DPO|DPO]]"
  - "[[02 Areas/ML & DL/Concepts/Reasoning/Toolformer|Toolformer]]"
---

# ToolRL: Reward is All Tool Learning Needs

**Authors:** Cheng Qian et al. (Ohio State, CISPA, Oxford)
**Published:** April 2025 (arXiv:2504.13958)
**URL:** https://arxiv.org/abs/2504.13958

## TL;DR

Систематическое исследование reward design для tool-use RL. Главный результат: **cold-start [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] без SFT превосходит SFT→RL** на 10% на BFCL (58.38% vs 39.25% для Qwen-2.5-7B). Step-level granular rewards (decomposed по tool name, param names, param values) лучше binary rewards. SFT вызывает memorization, которая мешает последующему RL.

## Problem

Существующие подходы к tool-use RL (Search-R1, TORL) ограничены узкими доменами (search, code). Нет систематического анализа: какой reward design оптимален? Нужен ли SFT warm-start? Как масштабируется RL на general tool-use?

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/toolrl/introduction.png]]
*ToolRL overview: GRPO с granular reward для tool-use. Cold-start RL consistently outperforms SFT+RL (источник: Qian et al., 2025)*

## Method

### Модели и алгоритмы
- **Base:** Qwen-2.5-Instruct (1.5B, 3B, 7B) + Llama-3.2-Instruct (3B)
- **RL:** GRPO (primary) + PPO (baseline comparison)
- **Бенчмарки:** BFCL-v3, API-Bank, Bamboogle
- **Single-turn** tool calling (не multi-turn как в [[02 Areas/ML & DL/Papers/ASTRA|ASTRA]])

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/toolrl/reward-design.png]]
*Reward design: format reward (binary) + correctness reward (granular по tool name, param names, param values). Step-level decomposition даёт fine-grained gradient signal (источник: Qian et al., 2025)*

### Reward Design (ключевой вклад)

Двухкомпонентный reward:

**Format Reward (binary):**
- Правильный порядок `<think>`, `<tool_call>`, `<response>` → 1, иначе 0

**Correctness Reward (granular, -3..+3):**
- **Tool name matching** — Jaccard similarity между predicted и ground-truth
- **Parameter name matching** — Jaccard по именам параметров
- **Parameter value matching** — Jaccard по значениям

Каждый компонент ∈ {-1, 0, +1}. Итого: -3..+3 на шаг.

Это **step-level** reward — каждый tool call получает свой score. В отличие от [[02 Areas/ML & DL/Papers/ASTRA|ASTRA]], где reward trajectory-level (один F1 на всю цепочку).

### Три инсайта по reward design

1. **Length ≠ quality.** Длинные reasoning traces не обязательно лучше. Length rewards деградируют performance, особенно на маленьких моделях.
2. **Dynamic scaling.** Постепенный сдвиг акцента: format adherence → correctness. Лучше чем фиксированные веса.
3. **Granularity matters.** Fine-grained decomposition (name + params + values) > binary correctness. Больше gradient signal → лучше обучение.

## Key Results

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/toolrl/cold-start-comparison.png]]
*Cold-start GRPO vs SFT+GRPO: cold-start consistently wins. SFT causes memorization that undermines subsequent RL exploration (источник: Qian et al., 2025)*

### Cold-Start RL vs SFT+RL

| Setup | BFCL (Qwen-2.5-7B) |
|---|---|
| Base (no training) | ~35% |
| SFT only | ~42% |
| SFT → GRPO | 39.25% |
| **GRPO cold-start** | **58.38%** |

**SFT вредит последующему RL:** "SFT initialization leads to memorization and overfitting, which reduces the impact of GRPO's effectiveness." SFT-модель получает выше training reward (distributional alignment), но хуже generализирует.

### Общий gain

ToolRL достигает +17% над base и +15% над SFT на BFCL.

## My notes

### Противоречие с ASTRA

ASTRA утверждает SFT необходим. ToolRL утверждает обратное. Возможные объяснения:
- **Scale:** ToolRL ≤7B, ASTRA 14B-32B. На больших моделях SFT может быть полезнее.
- **Horizon:** ToolRL = single-turn, ASTRA = multi-turn (до 32 ходов). Для длинных горизонтов cold-start RL может быть нестабилен — модель должна сначала выучить формат остановки.
- **Reward type:** ToolRL step-level, ASTRA trajectory-level. Step-level даёт достаточно сигнала для cold-start. Trajectory-level sparse reward может быть недостаточен без SFT warm-start.

Без прямого сравнения на одних моделях/бенчмарках — вопрос открыт.

### Step-level vs Trajectory-level reward

ToolRL (step-level): каждый tool call знает свой score. Точный credit assignment, но может быть greedy — оптимизирует шаги по отдельности, теряет глобальный план.

[[02 Areas/ML & DL/Papers/ASTRA|ASTRA]] (trajectory-level F1): один reward на всю цепочку. Размытый credit assignment, но учит стратегию целиком.

Идеально: комбинация обоих. Ни одна из работ этого не делает.

### Ограничения

- Single-turn only — не показано как работает на multi-turn
- Маленькие модели (≤7B) — scaling behavior неизвестен
- Нет сравнения с ASTRA, Tool-N1, APIGen-MT в одной таблице
