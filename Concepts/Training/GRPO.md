---
title: "GRPO"
aliases: [GRPO, Group Relative Policy Optimization]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]]"
  - "[[02 Areas/ML & DL/Papers/ASTRA|ASTRA]]"
  - "[[02 Areas/ML & DL/Papers/ToolRL|ToolRL]]"
  - "[[02 Areas/ML & DL/Papers/Nemotron-Research-Tool-N1|Tool-N1]]"
courses: []
sources:
  - "[DeepSeekMath: Pushing the Limits of Mathematical Reasoning (2024)](https://arxiv.org/abs/2402.03300)"
  - "[Cameron R. Wolfe — Group Relative Policy Optimization (GRPO)](https://cameronrwolfe.substack.com/p/grpo)"
  - "[Oxen AI — Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)"
  - "[Yuge Shi — A vision researcher's guide: PPO & GRPO](https://yugeten.github.io/posts/2025/01/ppogrpo/)"
---

# GRPO — Group Relative Policy Optimization

## Зачем это нужно: проблема PPO для LLM

[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] с PPO (Proximal Policy Optimization) — стандартный подход к alignment LLM. Но у PPO фундаментальная проблема масштабирования:

**PPO требует 4 модели одновременно в памяти:**
1. **Policy model** (LLM, которую обучаем) — ~70B params
2. **Reference model** (замороженная копия для KL-регуляризации) — ~70B params
3. **Value model (critic)** — ~70B params (обычно того же размера)
4. **Reward model** — ~70B params

Для 70B модели это **~280B параметров** в GPU memory. При FP16 — ~560 GB, что требует десятки GPU только для одного training step.

**GRPO** (DeepSeek, 2024) решает эту проблему, **убирая value model** — экономя ~25% memory и compute.

## PPO vs GRPO: обзор архитектуры

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo/ppo-vs-grpo.png]]
*PPO требует 4 модели (policy, reference, reward, value/critic). GRPO убирает value model — advantage считается через нормализацию rewards внутри группы из G ответов (источник: DeepSeekMath, 2024)*

## Как работает PPO (baseline)

В PPO для каждого промпта $q$ генерируется ответ $o$, и вычисляется advantage:

$$A_t = R_t - V_\phi(s_t)$$

где $R_t$ — cumulative reward, $V_\phi$ — **value function** (critic), предсказывающая expected reward для состояния $s_t$.

**Проблема value function для LLM:**
- Нужно обучать отдельную модель того же размера
- Value prediction для language generation крайне нестабильна
- Critic часто ошибается, давая шумные advantage estimates

## Как работает GRPO

### Ключевая идея: группа вместо critic

Вместо critic model, GRPO оценивает advantage **относительно группы ответов на один промпт:**

**Шаг 1.** Для каждого промпта $q$ генерируется **группа из $G$ ответов** $\{o_1, o_2, \ldots, o_G\}$.

**Шаг 2.** Каждый ответ оценивается reward function: $\{r_1, r_2, \ldots, r_G\}$.

**Шаг 3.** Advantage вычисляется как **нормализованное отклонение от среднего по группе:**

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

**Шаг 4.** Policy update:

$$\mathcal{L}_{GRPO} = -\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(\rho_i A_i, \; \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right)\right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

где $\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}$ — importance ratio (как в PPO).

### Визуализация: PPO vs GRPO

```
PPO:
  prompt → policy → 1 ответ → reward → critic → advantage → update
  (нужен critic model ~70B params)

GRPO:
  prompt → policy → G ответов → G rewards → normalize → G advantages → update
  (critic не нужен!)
```

## Почему группа заменяет critic

### Интуиция

Critic пытается ответить на вопрос: «какой reward **ожидается** для этого промпта?» Если reward выше ожидания — ответ хороший, ниже — плохой.

GRPO отвечает на тот же вопрос **эмпирически**: генерирует несколько ответов и сравнивает их друг с другом. Если ответ $o_i$ получил reward выше среднего по группе — он «хороший» для этого промпта.

### Почему это работает лучше для reasoning

**Reward models обучаются на comparisons:** «ответ A лучше ответа B для промпта Q». Это **comparative** по природе — reward model хорошо ранжирует ответы, но может быть плохим в абсолютных оценках.

GRPO использует rewards **сравнительно** (через нормализацию по группе), что идеально совпадает с тем, как rewards обучены. PPO с critic пытается использовать rewards **абсолютно** — менее надёжно.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo/training-methods-comparison.png]]
*Сравнение методов обучения (SFT, RFT, GRPO, PPO) на DeepSeekMath-Instruct 1.3B: GRPO конкурентоспособен с PPO при меньших ресурсах (источник: DeepSeekMath, 2024)*

## Сравнение PPO и GRPO

| Свойство | PPO | GRPO |
|----------|-----|------|
| Модели в памяти | 4 (policy, ref, critic, reward) | 2 (policy, ref) + reward fn |
| Memory overhead | ~4x policy size | ~2x policy size |
| Advantage estimation | Через value function | Через группу ответов |
| Policy updates per batch | 2-4 (переиспользуют данные) | 1 |
| Стабильность | Чувствителен к critic quality | Более стабилен |
| Compute на generation | 1 ответ/промпт | G ответов/промпт |

**Trade-off:** GRPO требует больше generation compute (G ответов вместо 1), но экономит memory (нет critic) и даёт более стабильный training signal.

## Reward Functions в GRPO

GRPO позволяет использовать **разные типы reward** в зависимости от задачи:

### Outcome-based rewards (для math/code)

$$r_i = \begin{cases} 1 & \text{если ответ правильный} \\ 0 & \text{если неправильный} \end{cases}$$

Простейший вариант: бинарная правильность. Именно это использует DeepSeek-R1-Zero — **никакого reward model**, только проверка ответа.

### Format rewards

$$r_{format} = \begin{cases} 1 & \text{если формат } \texttt{<think>...<answer>...} \\ 0 & \text{иначе} \end{cases}$$

Поощрение структурированного вывода.

### Learned reward model

Для open-ended задач (writing, QA) используется обученная reward model, как в стандартном RLHF.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo/iterative-rl-curves.png]]
*Iterative RL: кривые обучения DeepSeekMath-Instruct 7B на GSM8K и MATH через GRPO. Каждая итерация улучшает результат (источник: DeepSeekMath, 2024)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo/majk-passk.png]]
*Maj@K и Pass@K: SFT vs RL (GRPO). RL-модели генерируют более разнообразные решения — Pass@K растёт быстрее (источник: DeepSeekMath, 2024)*

## GRPO в DeepSeek-R1: что произошло

При обучении R1-Zero через GRPO с outcome-based reward на math задачах:

1. **Длина ответов росла** — модель «поняла», что длинные рассуждения → больше шансов прийти к правильному ответу
2. **Self-reflection emergence** — модель начала проверять свои шаги
3. **"Aha moment"** — спонтанное появление фраз «подождите, я ошибся, давайте пересмотрим»

Всё это — **без explicit supervision** на reasoning. Только reward за правильный ответ + GRPO оптимизация.

## GRPO для tool-use агентов

В 2025-2026 GRPO стал основным алгоритмом для обучения tool-use агентов через [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]]:

### ASTRA: GRPO без KL, без reference model

[[02 Areas/ML & DL/Papers/ASTRA|ASTRA]] (Beike, 2026) убрал KL-penalty и reference model "for simplicity and empirical stability". Reward — trajectory-level F1: `2rp/(r+p)` где `r = solved/total`, `p = solved/calls`. Проблема: без KL нет "якоря" к исходному поведению → risk деградации general capabilities.

### Adaptive Batch Filling

Проблема GRPO: если все G rollouts получают одинаковый reward → `std(R) ≈ 0` → advantage ≈ 0 → **нулевой gradient**. ASTRA решает: буфер, сэмплируй пока не наберёшь batch с `std(R) > δ`. Но ~15-25% промптов выбрасываются — модель не учится на задачах где правильный ответ = отказ.

### ToolRL: cold-start GRPO лучше SFT→GRPO

[[02 Areas/ML & DL/Papers/ToolRL|ToolRL]] (2025) показал: GRPO cold-start (без SFT warm-start) даёт **58.38%** на BFCL vs **39.25%** при SFT→GRPO (Qwen-2.5-7B). SFT вызывает memorization, мешающую exploration. Но это single-turn, ≤7B — на multi-turn и больших моделях ([[02 Areas/ML & DL/Papers/ASTRA|ASTRA]]) SFT может быть полезнее.

## Варианты и расширения

- **GRPO + online mirror descent** — используется в Kimi k1.5
- **Hybrid GRPO** — комбинация с value function для сложных задач
- **Dr. GRPO** — вариант с дополнительными стабилизирующими техниками
- **DAPO** — вариант без KL-penalty (как в ASTRA)
- **GRPO для vision** — применение к мультимодальным задачам

GRPO стал **стандартом de facto** для RL-обучения reasoning и tool-use моделей в 2025-2026 году, заменив PPO в большинстве открытых проектов.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — парадигма, в которой работает GRPO
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — альтернативный RL-free подход к alignment
- [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] — paradigm verifiable rewards, естественно сочетается с GRPO
- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-R1|DeepSeek-R1]] — модель, обученная через GRPO
- [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute|Test-time Compute]] — результат GRPO-обучения
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — паттерн, emergence через GRPO
