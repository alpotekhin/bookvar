---
title: "DeepSeek-R1"
aliases: [DeepSeek-R1, DeepSeek R1, R1-Zero, DeepSeek-R1-Zero]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]]"
courses: []
sources:
  - "[DeepSeek-R1: Incentivizing Reasoning via RL (2025)](https://arxiv.org/abs/2501.12948)"
  - "[HuggingFace Blog — From Zero to Reasoning Hero](https://huggingface.co/blog/NormalUhr/deepseek-r1-explained)"
  - "[PIA Blog — DeepSeek R1 Theory: Architecture, GRPO, KL Divergence](https://blog.piax.org/deepseek-r1-theory-tutorial-architecture-grpo-kl-divergence/)"
  - "[Nature — DeepSeek-R1 incentivizes reasoning through RL](https://www.nature.com/articles/s41586-025-09422-z)"
  - "[HuggingFace LLM Course — Understanding DeepSeek R1](https://huggingface.co/learn/llm-course/chapter12/3)"
---

# DeepSeek-R1

## Зачем эта модель появилась

В сентябре 2024 OpenAI выпустила o1 — первую модель, которая «думает» перед ответом, показав прорыв в математике и коде. Но o1 полностью закрыта: ни весов, ни деталей обучения. DeepSeek в январе 2025 ответил открытой моделью **DeepSeek-R1**, доказав радикальный тезис:

> **Reasoning можно incentivize через чистый RL, без human-annotated reasoning traces.**

Модель сравнилась с o1 на математике (AIME 2024: 79.8% vs 79.2%) и коде (Codeforces: 2029 vs 2061 рейтинг), при полной открытости весов и метода.

## Два этапа: R1-Zero и R1

### DeepSeek-R1-Zero: RL без SFT

Ключевой эксперимент — обучить reasoning **только через RL**, без предварительного SFT на reasoning-данных. Гипотеза: человеческие паттерны рассуждений могут **ограничивать** exploration модели.

**Процесс:**
1. Берётся DeepSeek-V3-Base (pre-trained, без SFT)
2. Применяется [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] с reward-сигналом **только от правильности ответа**
3. Никаких ограничений на формат рассуждения

**Reward-система R1-Zero:**
- **Accuracy reward:** бинарный (1 если ответ правильный, 0 иначе). Для math: проверка числового результата. Для code: прогон тестов.
- **Format reward:** мягкий штраф за отсутствие `<think>...</think>` тегов (подталкивает модель к структурированному рассуждению, но не ограничивает содержание).
- **Без process reward model (PRM):** в отличие от гипотез о o1, R1-Zero не использует пошаговые награды — только outcome-based.

**Что произошло (emergence):**

В процессе RL-обучения модель **самостоятельно** развила:
- **Self-verification** — «подожди, давай проверю этот шаг»
- **Reflection** — «нет, я ошибся, попробую другой подход»
- **Extended chain-of-thought** — рассуждения удлиняются по мере обучения
- **Backtracking** — возврат к предыдущим шагам при обнаружении ошибки
- **Multi-strategy exploration** — попробовать несколько подходов к задаче

### «Aha moment»: подробности

В **промежуточной версии** модели (не финальной) зафиксирован конкретный момент — модель впервые написала:

> "Wait, wait. Wait. That's an aha moment I can flag here."

А затем:

> "Hmm, wait. Let me reconsider. I think I made an error in step 3. Let me redo this calculation..."

Это **не было запрограммировано**. Модель получала reward только за правильный финальный ответ. Self-reflection возникла как **emergent strategy**: модель «обнаружила», что перепроверка увеличивает вероятность правильного ответа, а значит — получения reward.

**Количественно:** по мере RL-обучения средняя длина рассуждений R1-Zero **монотонно растёт** — от ~200 токенов в начале до ~5000-10000 к концу. Модель буквально «учится думать дольше».

**Дискуссия:** исследователи из Sea AI Lab (SAIL) провели reproduction study и поставили под сомнение, является ли "aha moment" устойчивым феноменом или стохастическим артефактом. Тем не менее, сам факт emergence self-verification из чистого RL подтверждён множеством independent reproduction.

**Проблемы R1-Zero:**
- Плохая читаемость (смешение языков, хаотичная структура)
- Бесконечные повторения (repetition loops)
- Хорошо в math/code, плохо в open-ended задачах
- Иногда чрезмерно длинные рассуждения без продвижения к ответу

### DeepSeek-R1: multi-stage pipeline

Для production-модели DeepSeek добавил структуру:

**Stage 1: Cold Start SFT**
Собирается небольшой датасет (~thousands) высококачественных CoT-примеров. Источники:
- Лучшие ответы R1-Zero (отфильтрованные по quality)
- Human-annotated reasoning traces
- Few-shot prompted CoT от DeepSeek-V3

Модель файнтюнится, чтобы задать базовый формат: `<think>...reasoning...</think><answer>...answer...</answer>`.

**Stage 2: RL с reasoning rewards**
GRPO на задачах с верифицируемыми ответами (math, code, logic). Reward = правильность + бонус за формат. Этот этап — основной source of reasoning capability.

**Stage 3: Rejection Sampling + SFT**
Из RL-модели сэмплируются **~800K** ответов, фильтруются лучшие. Добавляются open-ended задачи (writing, QA, general helpfulness). SFT на этом mixed датасете делает модель пригодной для широкого спектра задач.

**Stage 4: Финальный RL**
Второй раунд RL для alignment (helpfulness + safety). Здесь используются и rule-based rewards (для reasoning-задач), и reward model (для open-ended задач).

## GRPO: почему не PPO

DeepSeek использует **Group Relative Policy Optimization** вместо стандартного PPO:

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)} A_i, \; \text{clip}(\ldots) A_i\right) - \beta \cdot D_{KL}\right]$$

Главное отличие: **нет critic model** (value function). Advantage $A_i$ оценивается относительно группы — для каждого промпта генерируется $G$ ответов (обычно $G = 64$), вычисляются rewards, и advantage = нормализованное отклонение от среднего по группе:

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

**Почему это лучше для reasoning:**
- PPO требует 4 модели (policy, reference, critic, reward) — огромный расход памяти
- GRPO требует только 2 (policy, reference) + reward function
- Для 671B MoE модели экономия памяти от отсутствия critic — десятки GPU
- Групповая нормализация хорошо сочетается с comparative nature reasoning-задач: «лучше среднего по группе» — более стабильный сигнал, чем absolute value estimate

Подробнее: [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]]

## Distillation: reasoning для маленьких моделей

Ключевой вклад R1 — **дистилляция reasoning в компактные модели**:

| Модель | AIME 2024 | MATH-500 | LiveCodeBench | Codeforces |
|--------|-----------|----------|---------------|------------|
| DeepSeek-R1 (671B MoE) | 79.8% | 97.3% | 65.9% | 2029 |
| OpenAI o1 | 79.2% | 96.4% | — | 2061 |
| OpenAI o1-mini | 63.6% | 90.0% | — | 1820 |
| R1-Distill-Qwen-32B | 72.6% | 94.3% | 57.2% | 1691 |
| R1-Distill-Qwen-14B | 69.7% | 93.9% | 53.1% | 1481 |
| R1-Distill-Llama-70B | 70.0% | 94.5% | 57.5% | 1633 |
| R1-Distill-Qwen-7B | 55.5% | 92.8% | 37.6% | 1189 |
| R1-Distill-Qwen-1.5B | 28.9% | 83.9% | 16.9% | 954 |

**R1-Distill-Qwen-32B превосходит o1-mini** — 32B open-weight модель бьёт закрытую модель OpenAI. Метод: SFT на 800K reasoning samples, сгенерированных R1.

### Дистилляция vs чистый RL на маленьких моделях

Важный вывод из paper: **чистый RL на маленьких моделях работает значительно хуже дистилляции**:

| Подход | Qwen-32B AIME | Qwen-7B AIME |
|--------|---------------|--------------|
| Distill из R1 | **72.6%** | **55.5%** |
| Чистый RL (как R1-Zero) | 47.0% | 29.0% |

Причина: маленькие модели не имеют достаточной base capability, чтобы RL мог «открыть» сложные reasoning паттерны. Дистилляция передаёт **готовые паттерны** от большой модели, обходя ограничения exploration.

## Сравнение R1 с o1: детальный разбор

| Аспект | DeepSeek-R1 | OpenAI o1 |
|--------|-------------|-----------|
| Архитектура | 671B MoE (37B active) | Неизвестно |
| RL алгоритм | GRPO (no critic) | Неизвестно (вероятно PPO + PRM) |
| Process rewards | Нет | Вероятно да |
| MCTS / Tree search | Нет | Вероятно нет |
| Мультимодальность | Нет | Нет (o1-pro может) |
| Open weights | Да | Нет |
| Дистилляция | 6 open моделей | Нет |
| AIME 2024 | 79.8% | 79.2% |
| MATH-500 | 97.3% | 96.4% |
| Codeforces | 2029 | 2061 |
| GPQA Diamond | 71.5% | 78.3% |
| MMLU | 90.8% | 91.8% |

R1 сильнее в math, сопоставим в code, слабее в GPQA и MMLU. Ключевое отличие: R1 полностью открыт, что позволяет сообществу строить на его основе.

## Test-time Compute: новая парадигма

R1 — часть сдвига от «train bigger» к «think longer»:

| Парадигма | Scaling | Пример |
|-----------|---------|--------|
| Train-time compute | Больше параметров, больше данных | GPT-4 |
| **Test-time compute** | Больше рассуждений при inference | o1, R1 |

R1 тратит 10-100x больше токенов на ответ, чем обычная модель, но качество reasoning растёт пропорционально. Это открывает новую ось масштабирования AI — можно улучшать качество без переобучения модели, просто давая ей больше «времени на размышление».

Подробнее: [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute|Test-time Compute]]

## Влияние на экосистему

1. **Open reasoning:** первая полностью открытая reasoning-модель уровня o1
2. **GRPO как стандарт:** алгоритм стал де-факто стандартом для reasoning RL (Qwen3, Kimi k1.5 используют варианты)
3. **Distillation pipeline:** 800K reasoning samples стали public dataset, ускорив развитие open reasoning
4. **Emergence thesis:** доказательство, что complex cognitive patterns emergence из simple reward signals

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-V3|DeepSeek-V3]] — базовая модель, на которой строится R1
- [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] — алгоритм RL-обучения
- [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute|Test-time Compute]] — парадигма inference-time scaling
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — классический подход к alignment
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — техника рассуждений

## Дополнительные ресурсы

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) — оригинальная статья
- [HuggingFace — Understanding DeepSeek R1](https://huggingface.co/learn/llm-course/chapter12/3) — разбор в рамках LLM Course
- [Nature — DeepSeek-R1 Analysis](https://www.nature.com/articles/s41586-025-09422-z) — научный обзор в Nature
- [SAIL Blog — Aha Moment Pilot Study](https://sail.sea.com/blog/articles/62) — критический анализ "aha moment"
