---
title: "ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas"
url: https://arxiv.org/abs/2601.21558
authors: [Xiaoyu Tian, Haotian Wang, Shuaiting Chen, Hao Zhou, Kaichi Yu, Yudian Zhang, Jade Ouyang, Junxi Yin, Jiong Chen, Baoyan Guo, Lei Zhang, Junjie Tao, Yuansheng Song, Ming Cui, Chengwei Liu]
year: 2026
date_reviewed: 2026-04-12
type: paper-review
category: paper
tags:
  - RL
  - tool-use
  - agents
  - GRPO
  - RLVR
  - SFT
Organization: Beike (LianjiaTech)
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Reasoning/Toolformer|Toolformer]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
---

# ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas

**Authors:** Xiaoyu Tian et al. (Beike Language and Intelligence, LianjiaTech)
**Published:** January 2026 (arXiv:2601.21558)
**URL:** https://arxiv.org/abs/2601.21558
**Code:** https://github.com/LianjiaTech/astra

## TL;DR

End-to-end фреймворк для тренировки tool-use агентов через двухэтапный pipeline: (1) автоматический синтез multi-turn траекторий из MCP-серверов для SFT, (2) автоматический синтез code-executable, rule-verifiable сред для online RL ([[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]]). ASTRA-32B на Qwen3-32B набирает 64.25 на BFCL-v3 Multi-Turn (base = 49.63), приближаясь к Claude Opus 4.5 (68.38), при этом math reasoning (AIME) не деградирует.

## Problem

Три ограничения существующих подходов к обучению tool-use агентов:
1. **Неверифицируемые среды.** RL в LLM-симулированных окружениях — reward шумный, недетерминированный, галлюцинирует
2. **Декомпозиция в single-step.** Многие методы бьют multi-turn траектории на независимые single-step примеры → теряется temporal coherence и planning signal
3. **SFT-only или RL-only.** Каждый по отдельности субоптимален: SFT не даёт exploration, RL from scratch нестабилен на длинных горизонтах

## Method

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/astra/sft-pipeline.png]]
*ASTRA SFT pipeline: 5 стадий от сбора тулов до верифицированных траекторий. Stage 5 включает 7-мерную reward system (Global + Step-wise) (источник: Tian et al., 2026)*

### Компонент A: Trajectory Synthesis (для SFT)

**Шаг 1. Сбор тулов.** 1,585 MCP-серверов, 19,036 тулов, 41 домен. Схемы нормализуются в OpenAI function-calling формат. Фильтр: ≥3 тулов на сервер, ясные описания.

**Шаг 2. Tool-chain через граф переходов.** Для каждого MCP-сервера:
- LLM генерирует плausible задачи + цепочки tool calls `(f₁ → f₂ → ... → fₙ)`
- Все цепочки агрегируются в направленный взвешенный граф `G^(s)`
- Новые цепочки сэмплируются random walk по графу (в коде — exhaustive DFS, длина 2-5)
- Граф кодирует "топологию композиции тулов", random walk даёт структурно валидные цепочки

**Шаг 3. Конструирование задач.** Два режима:
- **T_chain** — дана цепочка → LLM генерирует задачу, для которой именно эта цепочка = решение
- **T_server** — только описание сервера → LLM свободно генерирует задачи
- Аугментация по 3 осям: diversity (перефраз), complexity (ограничения), persona (стиль)
- Quality scoring → фильтрация

**Шаг 4. Сбор траекторий.** Сильная LLM (GLM-4.6-FP8) исполняет задачи:
- Real MCP-серверы или doc-based emulators
- **20% deliberate failure injection** в эмуляторах — учит agent обрабатывать ошибки
- Формат: `m₀` (system) → `m₁` (user query, **один**) → `m₂..mₖ` (agent↔tool turns) → final answer

**Шаг 5. Reward modeling (для фильтрации SFT данных).** 7 измерений (0-1 каждое):
- QU (Query Understanding), QP (Query Planning), TCU (Tool-response Understanding), TCP (Tool-response Planning), TCS (Tool Call Status), TC (Tool Conciseness), FA (Final Answer)
- Итог = среднее. Порог → только лучшие траектории идут в SFT.

### Компонент B: Reinforcement Arenas (для RL)

Ключевая новизна — автоматически синтезированные **code-executable** среды:

**Шаг 1. Q-A декомпозиция.** Сложный вопрос `q₀` с ответом `a₀` декомпозируется в атомарные подзадачи `{(qᵢ, aᵢ)}` с графом зависимостей. Финальный ответ: `a₀ = Φ({aᵢ}, G)`.

**Шаг 2. Валидация.** 4 бинарных критерия: Dependency Consistency, Sub-Question Atomicity, Sequential Rationality, Task Completeness.

**Шаг 3. Синтез среды.** Для каждой подзадачи `(qᵢ, aᵢ)`:
1. LLM генерирует спецификацию Python-функции (тула)
2. LLM генерирует реализацию
3. Sandbox execution → проверка что output содержит `aᵢ`
4. Retry при failure

**Результат:** каждая подзадача = реальный executable тул с ground-truth ответом. Reward **детерминированный** — код либо возвращает правильный ответ, либо нет. Это [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]].

**Шаг 4. Мерж подсред.** Функционально эквивалентные подзадачи объединяются, данные расширяются.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/astra/env-synthesis.png]]
*Environment Synthesis: Q-A декомпозиция → Python-функции → executable arenas. Каждая подзадача становится реальным тулом с ground-truth ответом (источник: Tian et al., 2026)*

### Training Pipeline

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/astra/rl-pipeline.png]]
*RL pipeline: multi-turn interaction в Code Sandbox. Batch rollouts → tool execution → deterministic reward. Одна траектория показана для одного промпта (источник: Tian et al., 2026)*

**Stage 1: SFT** на синтезированных траекториях. 2 эпохи, batch 32, max 20K tokens, LR 5e-6 (14B) / 2e-6 (32B).

**Stage 2: RL** — [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] в синтезированных средах:
- Без KL-penalty, без entropy bonus, без reference model
- Online multi-turn: agent реально взаимодействует с code environment
- До 32 ходов на траекторию, batch 256, LR 2e-6
- Max prompt 25.6K, max response 49.2K tokens

**F1 Reward:**
```
r = solved_subtasks / total_subtasks       (recall)
p = solved_subtasks / total_tool_calls     (precision)
reward = 2rp / (r + p)
```
- Recall-only → agent спамит тулами бесконечно → training collapse
- Precision-only → agent слишком консервативен → collapse
- F1 → баланс

**Adaptive Batch Filling.** Если все rollouts в GRPO-группе получают одинаковый reward → `std(R) ≈ 0` → нулевой gradient. Решение: буфер, сэмплируй пока не наберёшь batch с `std(R) > δ`.

**Irrelevant Tool Mixing.** Во время RL подмешиваются "лишние" тулы через embedding similarity bands (Qwen3-Embedding-8B):
- High similarity (>0.85), Medium (0.4-0.85), Low (<0.4)
- Учит модель discriminate — лучше чем random distractors

## Key Results

### Agentic Benchmarks

| Model | BFCL-MT | τ²-Bench | ACEBench |
|---|---|---|---|
| Claude Opus 4.5 | **68.38** | **85.79** | **82.09** |
| Gemini 3 Pro | 63.13 | 83.69 | 76.05 |
| GLM-4.6 | 68.00 | 69.63 | 80.00 |
| Qwen3-32B (base) | 49.63 | 49.70 | 59.79 |
| **ASTRA-32B** | **64.25** | **63.70** | **71.88** |
| Qwen3-14B (base) | 44.50 | 46.05 | 51.67 |
| **ASTRA-14B** | **58.13** | **57.69** | **68.96** |

ASTRA-14B бьёт base Qwen3-32B на всех трёх бенчмарках.

### Stage-wise Breakdown (BFCL-MT)

| Stage | 14B | 32B |
|---|---|---|
| Base | 44.50 | 49.63 |
| +SFT | +4.00 | +4.25 |
| +RL | **+9.63** | **+12.12** |
| **Total** | **+13.63** | **+16.37** |

**RL даёт в 2-3x больше чем SFT.** Но SFT — необходимый фундамент (claim авторов, оспаривается [[02 Areas/ML & DL/Papers/ToolRL|ToolRL]] и [[02 Areas/ML & DL/Papers/Nemotron-Research-Tool-N1|Tool-N1]]).

### General Capabilities (AIME)

| Model | AIME 2024 | AIME 2025 |
|---|---|---|
| Qwen3-32B | 83.00 | 66.80 |
| ASTRA-32B | 81.40 | 68.30 |

Деградации нет. Но AIME — единственный proxy (см. критику ниже).

## Ablation Studies

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/astra/reward-ablation.png]]
*Reward ablation: F1 vs recall-only vs precision-only. Recall → turns explode, precision → turns collapse, F1 → stable (источник: Tian et al., 2026)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/astra/tool-mixing-ablation.png]]
*Tool mixing ablation: similarity-band > random > none. Structured distractors дают balanced discrimination pressure (источник: Tian et al., 2026)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/astra/turn-distribution.png]]
*Turn distribution по стадиям обучения. SFT сжимает output, RL возвращает к средней длине (источник: Tian et al., 2026)*

**Reward design:** F1 > recall-only (turns explode) > precision-only (turns collapse). Только графики, без точных чисел.

**Tool mixing:** Similarity-band > random > none. Balanced discrimination pressure по всем уровням сходства.

**Поведение по стадиям:**

| Stage | Avg Steps | Tokens/Step |
|---|---|---|
| Qwen3-32B base | 3.7 | 361.7 |
| +SFT | 3.1 | 192.0 (сжатие — имитация) |
| +RL | 3.1 | 317.8 (средний — нашёл что thinking полезен) |

## My notes

### Критика

**Не раскрыт масштаб.** Нет данных о количестве SFT-траекторий, RL-сред, GPU-часов. Публичные данные — только ASTRA-SFT-1k и ASTRA-RL-1k (subset). Вторичные упоминания ~55K SFT + ~6.6K RL, но не подтверждено.

**Discrepancy в числах.** Qwen3-32B base = 49.63 (Table 1) vs 47.88 (Table 2). Не объяснена. Вероятно: 49.63 = macro-average по 4 BFCL-MT подмножествам (59.0, 51.5, 47.5, 40.5), 47.88 — другой subset/metric.

**Нет сравнения с конкурентами.** APIGen-MT, ToolRL, Tool Zero упомянуты в Related Work, но нет head-to-head таблицы. Единственное: LoopTool-32B = 57.75 vs ASTRA-32B = 64.25.

**Generator = GLM-4.6.** Нет аблации по генератору. Почему не Qwen (на котором тренируют)? Потенциальный style/capability bias.

**Противоречие с ToolRL и Tool-N1.** Оба утверждают что cold-start RL **лучше** SFT→RL. ASTRA утверждает обратное. Возможное объяснение: ToolRL работает на single-turn / маленьких моделях (≤7B), ASTRA — на multi-turn / 14B-32B. Для multi-turn с 32 ходами cold-start может быть нестабилен.

**AIME — слабый proxy.** Только math. Не проверены: IFEval, coding, creative writing, safety. "Не деградировал на AIME" ≠ "не деградировал в целом".

**Нет KL-penalty в GRPO.** Убрали для "simplicity and stability", но KL — основной механизм антифоргеттинга. T-Bank (T-Pro 2.1) прямо пишет: без сильной KL GRPO вызывает "significant degradation on other domains".

**Нет dedicated Limitations section.** Для arXiv preprint формально допустимо, но red flag.

**Рецензия (consensus с GPT-5.4 анализом): Borderline.** Сильная инженерная/system paper с хорошей интуицией, но экспериментально недозакрыт. Главный вклад — не новый алгоритм, а связная pipeline для verifiable multi-turn RL. Для Weak Accept нужно: закрыть discrepancy, дать compute/scale, добавить comparison с ToolRL/APIGen-MT/Tool Zero.

### Практическая ценность

Что стоит взять:
- F1 trajectory reward (2rp/(r+p)) — балансирует completion и efficiency
- Similarity-band tool mixing — несложно реализовать, доказанно лучше random
- Executable environments для RL reward — правильное направление для RLVR
- 20% failure injection — учит robustness
- Tool-graph construction через random walk — масштабируемее ручного написания задач

Что под вопросом:
- Необходимость SFT warm-start (оспаривается 2 из 3 конкурирующих работ)
- Воспроизводимость (нет scale/cost, часть тулов internal)
