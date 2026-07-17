---
title: "Alignment"
aliases: [Alignment, Выравнивание, LLM Alignment, AI Alignment]
type: concept
status: legacy
category: Training
papers: []
courses: []
sources:
  - "[Ouyang et al. — Training language models to follow instructions with human feedback (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155)"
  - "[Askell et al. — A General Language Assistant as a Laboratory for Alignment (2021)](https://arxiv.org/abs/2112.00861)"
  - "[Bai et al. — Constitutional AI (2022)](https://arxiv.org/abs/2212.08073)"
  - "[Rafailov et al. — Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290)"
---

# Alignment — выравнивание LLM

## Что такое alignment

**Alignment** — совокупность техник, превращающих сырую предобученную языковую модель в полезного ассистента, следующего человеческим намерениям и нормам. Это **post-training фаза** в пайплайне:

```
Pre-training  →  Alignment (SFT + RLHF/DPO + safety)  →  Deployed model
(next-token     (instruction following,
 prediction на   helpfulness, honesty,
 триллионах      harmlessness)
 токенов)
```

После [[Pre-training]] модель умеет только предсказывать следующий токен на произвольном тексте. Она не знает, что на вопрос следует **отвечать**, а не дописывать ещё вопросы. Не понимает, что нужно отказывать на вредные запросы. Не следует инструкциям пользователя, а продолжает шаблон. Alignment исправляет всё это.

## Цель: HHH (Helpful, Honest, Harmless)

Askell et al. (Anthropic, 2021) формулируют три принципа alignment, ставшие стандартом индустрии:

1. **Helpful** — модель отвечает по существу, следует инструкциям, решает задачу пользователя.
2. **Honest** — модель не галлюцинирует, признаёт незнание, не обманывает намеренно, не создаёт ложные впечатления.
3. **Harmless** — модель отказывается помогать в создании оружия, вредоносного ПО, instructions to harm, не генерирует токсичный контент.

Эти цели **конфликтуют**: helpful подталкивает всегда отвечать, harmless — иногда отказывать. Honest конфликтует с helpful, когда правильный ответ — «я не знаю». Настройка баланса между HHH — центральная задача alignment.

## Методы

### 1. [[SFT]] — Supervised Fine-Tuning

Первый этап: дообучение на датасете `(prompt, good_response)`. Демонстрации собираются аннотаторами или собираются из диалогов с уже существующими моделями (distillation).

**Что делает:** учит формату диалога, стилю ассистента, базовому следованию инструкциям. Без SFT все последующие методы работают плохо — модели нужен initial instruction-following прайор.

**Ограничение:** SFT учит имитировать, но не учит **ранжировать**. Модель не знает, насколько ответ А лучше ответа В — только копирует reference.

### 2. [[RLHF]] — Reinforcement Learning from Human Feedback

Стандартный метод с InstructGPT (2022). Три этапа:

**a) Reward Model.** Аннотаторы сравнивают пары ответов — какой лучше. Обучается [[Reward Model]] $r_\phi(x, y)$, предсказывающая preference score.

**b) RL fine-tuning.** Policy (LLM) оптимизируется через [[PPO]] или [[GRPO]] против reward model, с KL-регуляризацией к reference модели:

$$\max_\theta \mathbb{E}[r_\phi(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

**Плюсы:** работает на масштабе, делает модели значительно более helpful и harmless.
**Минусы:** сложный пайплайн (4 модели в памяти для PPO), нестабильность, reward hacking.

### 3. [[DPO]] — Direct Preference Optimization

Rafailov et al. (2023) показали, что RLHF objective можно переписать как **прямой supervised loss** на preference pairs, без reward model и RL:

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

где $y_w$ — preferred, $y_l$ — rejected. **Проще, стабильнее, дешевле PPO.** Стал стандартом в open-source alignment (Zephyr, Llama-3-Instruct, Qwen2.5).

Варианты: **IPO, KTO, SimPO, ORPO** — разные функции потерь над тем же типом данных.

### 4. [[Constitutional AI]]

Anthropic-ский подход: вместо дорогих human labels используется **модель, критикующая себя** по списку принципов («constitution»).

Пайплайн:
1. Модель генерирует ответ.
2. Другая (или та же) модель критикует его по принципам и переписывает.
3. Получаем пары `(original, revised)` для обучения без human labels.

Используется в Claude. Снижает зависимость от ассессоров и даёт более прозрачные критерии.

### 5. RLAIF — RL from AI Feedback

Reward приходит от LLM-as-a-judge, а не от людей. Масштабируемее RLHF, ценой возможных systematic biases модели-судьи.

### 6. [[RLVR]] — RL with Verifiable Rewards

Для reasoning задач (math, code) reward — это **автоматическая проверка** ответа (тесты проходят / нет, answer correct / нет). Это основа DeepSeek-R1 и современных reasoning моделей. См. [[GRPO]] — основной алгоритм для RLVR.

## Типичный современный пайплайн (2025-2026)

```
1. Pre-training        → base model (next-token)
2. Continued pretraining → domain adaptation, instruction-like data
3. SFT                 → instruction following, chat format
4. DPO / IPO           → preference tuning (helpfulness, style)
5. RLVR (GRPO)         → reasoning, tool-use
6. Safety / red-teaming → адверсариальные примеры, отказы
7. Evaluation          → MT-Bench, Arena, HarmBench, etc.
```

Открытые модели (Llama-3, Qwen, DeepSeek) публикуют детали каждого этапа. Проприетарные (GPT, Claude, Gemini) — нет.

## Alignment Tax

**Alignment tax** — регрессия возможностей модели при alignment. После SFT+RLHF модель может:

- Хуже решать reasoning-задачи (сверх-безопасные отказы, короткие ответы).
- Хуже писать код (забывает длинные форматы).
- Терять знания pre-training (catastrophic forgetting).
- Становиться слишком «вежливой» и sycophantic (соглашается со всем).
- Отказываться от невредных запросов (false refusals).

Явный пример: GPT-4 base vs GPT-4 aligned — base модель лучше на некоторых reasoning benchmarks. Alignment делает модель **полезнее в среднем**, но теряет экстремальные способности.

**Способы минимизации tax:**
- Данные высокого качества в SFT/DPO (меньше, но лучше).
- KL-регуляризация к reference (не уходить далеко от pre-trained).
- Смешивание pre-training данных в SFT (replay).
- Модульный подход: `thinking mode` отдельно от `chat mode` (DeepSeek-R1, Qwen-thinking).
- RLVR вместо RLHF где возможно — verifiable reward не страдает от reward hacking.

## Sycophancy и reward hacking

**Sycophancy:** модель соглашается с пользователем даже когда он неправ, потому что reward model предпочитает ответы, подтверждающие точку зрения пользователя. Системная проблема RLHF, решается лучшими принципами аннотации и калибровкой.

**Reward hacking:** модель находит способы получать высокий reward без настоящего решения задачи — длинные пустые ответы, подмазывание, упомянуть ключевые слова. Противодействие: constrained generation, penalty за длину, adversarial training.

## Jailbreaking

Адверсариальные промпты, обходящие alignment: `DAN`, `grandma exploit`, `prompt injection`. Alignment — не абсолютная защита, а distribution shift: новые атаки постоянно находятся.

Защита — постоянное добавление adversarial examples в SFT/DPO (red-teaming), безопасность на уровне системы (фильтры на входе/выходе), а не только модель.

## Related concepts

- [[SFT]] — первый этап alignment
- [[RLHF]] — классический alignment через preference feedback
- [[DPO]] — RL-free альтернатива
- [[Constitutional AI]] — AI-assisted alignment
- [[RLVR]] — verifiable rewards (reasoning, tool-use)
- [[GRPO]] — современный RL-алгоритм для alignment
- [[Reward Model]] — центральный компонент RLHF
- [[Instruction Tuning]] — близкое понятие, часть alignment
