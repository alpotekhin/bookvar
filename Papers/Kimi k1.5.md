---
title: "Kimi k1.5: Scaling Reinforcement Learning with LLMs"
url: "https://arxiv.org/abs/2501.12599"
authors: [Kimi Team, Moonshot AI]
year: 2025
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/kimi-k15/paper.pdf|PDF]]"
concepts:
  - Reinforcement Learning
  - Chain-of-Thought
  - Test-Time Compute
  - Multimodal Reasoning
---

# Kimi k1.5: Scaling Reinforcement Learning with LLMs

## TL;DR

Kimi k1.5 — мультимодальная LLM от Moonshot AI, обученная через RL, которая матчит OpenAI o1 на reasoning задачах: 77.5 AIME, 96.2 MATH 500, 94-й перцентиль Codeforces. Ключевые инновации: **long context scaling RL** (до 128K), **partial rollouts** для эффективного обучения, **simplistic framework** без Monte Carlo tree search, value functions и process reward models. Также представлены **long2short methods** для компрессии reasoning в short-CoT: 60.8 AIME, outperforming GPT-4o на +550%.

## Проблема

Pre-training с next-token prediction ограничен объёмом доступных training data (data wall). RL открывает новую ось для scaling — модель генерирует собственные данные через exploration с rewards. Но до k1.5 ни одна опубликованная работа не показала **конкурентных** результатов с RL scaling для LLM reasoning.

Дополнительно: сложные planning algorithms (MCTS, value functions, process reward models) добавляют complexity при deployment. Можно ли обойтись без них?

## Метод

### Pipeline обучения

1. **Pretraining** — multimodal (text + vision), 3 стадии (VL pretraining -> cooldown -> long-context activation до 131K)
2. **Vanilla SFT** — ~1M text + 1M vision examples
3. **Long-CoT SFT** — warmup dataset с reasoning paths (planning, evaluation, reflection, exploration)
4. **Reinforcement Learning** — основной этап

### Policy Optimization

Вариант **online policy mirror descent**. На итерации $i$ оптимизируем:

$$\max_\theta \mathbb{E}_{(x,y^*) \sim D} \left[ \mathbb{E}_{(y,z) \sim \pi_\theta} [r(x, y, y^*)] - \tau \text{KL}(\pi_\theta(x) \| \pi_{\theta_i}(x)) \right]$$

Closed-form solution существует, из него выводится surrogate loss. Gradient для каждого problem $x$ с $k$ samples:

$$\frac{1}{k} \sum_{j=1}^{k} \nabla_\theta \log \pi_\theta(y_j, z_j | x)(r(x, y_j, y^*) - \bar{r}) - \frac{\tau}{2} \nabla_\theta \left( \log \frac{\pi_\theta(y_j, z_j | x)}{\pi_{\theta_i}(y_j, z_j | x)} \right)^2$$

Ключевой design choice: **без value network**. Авторы аргументируют: classical credit assignment через value function может быть вреден для long-CoT. Если модель сделала ошибку на шаге $z_{t+1}'$, но потом исправилась и нашла правильный ответ — value function бы штрафовала это exploration. А для long-CoT именно trial-and-error паттерн критически важен.

### Long Context Scaling

RL context window скалируется до **128K токенов**. Наблюдение: performance продолжает расти с увеличением context length.

**Partial Rollouts** — ключевая техника для эффективности:
- Фиксированный token budget на rollout
- Если trajectory длиннее budget — сохраняется в replay buffer, продолжается в следующей итерации
- Только текущая итерация требует on-policy computation, предыдущие сегменты reused
- Repeat detection с early termination

### Length Penalty

Для борьбы с overthinking (модель генерирует избыточно длинные reasoning):

$$\text{len\_reward}(i) = \begin{cases} \lambda & \text{if correct} \\ \min(0, \lambda) & \text{if incorrect} \end{cases}$$

где $\lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}}$

Логика: среди правильных ответов — поощряем короткие, штрафуем длинные. Среди неправильных — явно штрафуем длинные. Warm-up: сначала без penalty, потом constant.

### Sampling Strategies

- **Curriculum Sampling:** от лёгких задач к сложным (ранний RL имеет limited performance, hard задачи неэффективны)
- **Prioritized Sampling:** sample пропорционально $1 - s_i$ (success rate), фокус на слабых местах

### Reward Modeling

**Math:** Chain-of-Thought RM (98.5% accuracy) vs Classic RM (84.4%). CoT RM генерирует step-by-step reasoning перед вердиктом.

**Code:** Автоматическая генерация test cases через CYaRon + filtering (7/10 submissions должны давать matching results).

**Vision:** 3 типа данных — real-world (science questions, charts), synthetic (geometric patterns), text-rendered (screenshots of text/code).

### Long2short Methods

4 подхода для компрессии long-CoT в short-CoT:
1. **Model Merging** — простое усреднение весов long-CoT и short-CoT моделей
2. **Shortest Rejection Sampling** — из 8 samples выбираем shortest correct для SFT
3. **DPO** — shortest correct = positive, longer = negative (включая correct but 1.5x longer)
4. **Long2short RL** — separate RL phase с length penalty и reduced max rollout length

### Infrastructure

**Hybrid Deployment:** Megatron (training) и vLLM (inference) в одном pod через Kubernetes Sidecar. < 1 min training->inference, ~10 sec inference->training.

## Ключевые результаты

### Long-CoT (matches o1)

| Benchmark | Kimi k1.5 | OpenAI o1 | o1-mini |
|-----------|-----------|-----------|---------|
| AIME 2024 | **77.5** | 74.4 | 63.6 |
| MATH 500 | **96.2** | 94.8 | 90.0 |
| Codeforces | **94th %ile** | 94th %ile | 88th %ile |
| MathVista | **74.9** | 71.4 | 53.1 |

### Short-CoT (outperforms GPT-4o by up to +550%)

| Benchmark | Kimi k1.5 | GPT-4o | Claude 3.5 Sonnet |
|-----------|-----------|--------|-------------------|
| AIME 2024 | **60.8** | 9.3 | 16.0 |
| MATH 500 | **94.6** | 74.6 | 78.3 |
| LiveCodeBench | **47.3** | 33.4 | 36.3 |
| MathVista | **70.1** | 63.8 | 65.3 |

## Мои заметки

**Simplistic framework — главный тезис статьи.** Без MCTS, без value function, без process reward model — только policy optimization + long context. Авторы аргументируют: при достаточном context length модель сама учится делать implicit search, backtracking и correction. Context length = computational budget для planning. Это радикально простой подход по сравнению с AlphaGo-style инфраструктурой.

**Без value network — контринтуитивно, но логично.** В classical RL credit assignment через advantages важен. Но для long-CoT reasoning модель должна учиться на full trajectories: ошибка -> распознавание -> correction -> правильный ответ. Value function бы penalized exploration, что подавляет развитие planning skills.

**Partial rollouts** — elegant решение проблемы длинных trajectory в RL. При 128K context rollouts могут быть очень длинными. Разбиение на сегменты + replay buffer + reuse предыдущих сегментов кардинально снижает compute cost.

**Long2short — practical value.** Long-CoT даёт лучшее качество, но дорого при inference. Компрессия знаний из long-CoT модели в short-CoT (через model merging, DPO, или dedicated RL) — путь к production deployment. Model merging (просто усреднение весов!) работает — удивительно простой baseline.

**+550% на AIME vs GPT-4o для short-CoT** — 60.8 vs 9.3. Это показывает, что RL scaling fundamentally меняет reasoning capabilities, даже в short-CoT режиме.

**CoT RM vs Classic RM: 98.5% vs 84.4%** — chain-of-thought в reward model — мощный инсайт. RM с reasoning процессом гораздо точнее определяет correctness, особенно для math где разные формы одного ответа (a^2-4 vs (a+2)(a-2)) должны считаться equivalent.

**RL prompt set curation** — недооценённый аспект RL training. Три свойства quality prompt set: diverse coverage, balanced difficulty, accurate evaluability. Конкретная деталь: prompts с easily guessable answers (multiple-choice, true/false) удаляются, чтобы избежать reward hacking. Если модель угадывает ответ без CoT за 8 attempts — prompt считается too easy-to-hack.

**Curriculum + Prioritized Sampling** — комбинированная стратегия. Curriculum: start easy -> progress to hard (основан на natural difficulty labels). Prioritized: track success rate per problem, sample пропорционально failure rate. Простые методы, значительный эффект на training efficiency.

**Vision RL data** — три типа (real-world, synthetic, text-rendered). Третий тип (text-rendered) особенно интересен: текст конвертируется в изображения (screenshots, photos of documents). Это обеспечивает consistency: модель должна давать одинаковые ответы на text query и его screenshot. Practical use case: OCR-style задачи, document understanding.

**Hybrid deployment на Kubernetes** — системная инновация. Megatron и vLLM в одном pod через Sidecar containers. vLLM terminate и restart между итерациями (вместо memory offload) из-за CUDA graph / NCCL buffer leaks. Pragmatic решение production проблем.

**Model merging for long2short** — простейший baseline (average weights of long-CoT and short-CoT models) работает! Это перекликается с findings в model merging literature (TIES, DARE, etc.) — linear combination часто surprisingly effective.

**Контекст для индустрии:** k1.5 вышел практически одновременно с DeepSeek-R1. Оба показывают, что RL scaling для reasoning — viable path. Но подходы разные: DeepSeek-R1 фокусируется на GRPO + emergent CoT, Kimi k1.5 — на long context scaling + partial rollouts. Convergence результатов при разных методах говорит о robustness самой идеи RL scaling.

**Vanilla SFT data composition** — хорошо задокументирована: 500K general QA, 200K coding, 200K math/science, 5K creative writing, 20K long-context + 1M vision examples. Asymmetric distribution: general QA доминирует, creative writing минимален. Это pragmatic — reasoning capabilities важнее стилистики.

**3-stage pretraining (vision-language -> cooldown -> long-context)** — cooldown стадия с curated + synthetic data для consolidation. Этот подход помогает стабилизировать capabilities перед context extension.

**Test case generation для coding RL** — from 1000 problems, 614 не требуют special judge, 323 попали в training set после filtering. Это realistic yield: ~1/3 проблем проходят quality bar. CYaRon library + Kimi k1.5 для генерации test cases = bootstrapping: модель помогает создавать свои training data.

**Online mirror descent** — математически grounded подход. В отличие от "просто PPO", авторы выводят gradient из surrogate loss, получая natural extension policy gradient к off-policy case с l2-regularization. Reset optimizer каждую итерацию из-за changing reference policy — теоретически motivated design choice.

**Multimodal joint training** — text + vision в одном RL loop. Модель рассуждает over both modalities simultaneously. Результаты на MathVista (74.9) и MMMU (77.3) показывают, что visual reasoning benefits от того же RL framework что и text reasoning.

**DPO для long2short** — shortest correct = positive, longer responses = negative (включая correct but 1.5x longer). Интересное использование DPO: обычно positive/negative определяются quality, здесь — length при одинаковом quality. Модель учится быть concise.

**Replay buffer** — disrupts temporal correlations в training data. Classic RL technique, applied to LLM RL. Важно для stability: без replay buffer model может overfit на recent rollouts.
