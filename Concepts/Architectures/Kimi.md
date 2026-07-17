---
title: "Kimi"
aliases: [Kimi, Kimi k1.5, Moonshot AI, Kimi-K1.5]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Kimi k1.5|Kimi k1.5]]"
courses: []
sources:
  - "[Kimi k1.5: Scaling Reinforcement Learning with LLMs (2025)](https://arxiv.org/abs/2501.12599)"
  - "[MoBA: Mixture of Block Attention for Long-Context LLMs (2025)](https://arxiv.org/abs/2502.13189)"
  - "[GitHub — MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA)"
  - "[Moonshot AI — Wikipedia](https://en.wikipedia.org/wiki/Moonshot_AI)"
  - "[Medium — Kimi k1.5: Revolutionizing Multimodal Reasoning](https://medium.com/towards-agi/kimi-k1-5-how-this-next-gen-ai-model-is-revolutionizing-multimodal-reasoning-with-reinforcement-e06fbd64c12c)"
---

# Kimi

## Зачем эта модель появилась

**Moonshot AI** (Китай, 2023) появилась с амбициозной идеей: **длинный контекст — это ключ к полезности LLM**. Когда GPT-4 работал с 8K-32K контекстом, Kimi (март 2024) предложила **200K токенов** — можно загрузить целую книгу или кодовую базу.

В январе 2025 Moonshot выпустила **Kimi k1.5** — reasoning-модель, конкурирующую с o1 и DeepSeek-R1, с уникальным подходом: **мультимодальное reasoning через RL** без сложных техник вроде MCTS или process reward models.

## Kimi: пионер длинного контекста

### Почему длинный контекст важен

Большинство LLM к 2023 году имели контекст 4K-8K токенов. Это критически мало для:
- Анализа документов (контракт ~50K токенов)
- Code review (репозиторий ~100K+ токенов)
- Research synthesis (несколько статей ~200K+ токенов)

Kimi Chat (200K токенов, позже 1M+) показала, что **длинный контекст кардинально меняет user experience**: вместо «перескажи суть документа, затем задай вопрос» можно просто загрузить всё и спрашивать.

## MoBA: Mixture of Block Attention (детальный разбор)

Для эффективной обработки длинных последовательностей Moonshot разработала **MoBA** — механизм, применяющий принципы Mixture-of-Experts к attention.

### Архитектура MoBA

**Шаг 1: Разбиение на блоки.** Входная последовательность длины $n$ делится на $B$ блоков фиксированной длины $b$: $B = n / b$. Каждый блок содержит $b$ токенов с их Key и Value.

**Шаг 2: Routing.** Для каждого query-токена $q_t$ маршрутизатор вычисляет **score для каждого блока**:

$$s_j = f(q_t, \text{Block}_j) \quad \text{для } j = 1, \ldots, B$$

Функция scoring — безпараметрическая (parameter-free): используется средний или максимальный dot-product Query с Keys в блоке. Это **ключевой design choice** — отсутствие дополнительных параметров позволяет применять MoBA к любой существующей модели без дообучения.

**Шаг 3: Top-k selection.** Выбираются $k$ блоков с наибольшими scores. Токен обращает attention **только на выбранные блоки**, а не на всю последовательность.

**Шаг 4: Causal constraint.** Будущие блоки автоматически исключаются (causal masking). Текущий блок (содержащий сам query-токен) **всегда включается** — это гарантирует local context.

### Сложность MoBA

| Метод | Complexity | При 200K |
|-------|-----------|----------|
| Full Attention | $O(n^2)$ | $4 \times 10^{10}$ |
| MoBA (k=4, b=4096) | $O(n \cdot k \cdot b)$ | $3.3 \times 10^9$ |
| Speedup | | **~12x** |

При 1M токенов Moonshot сообщает **6.5x speedup** с MoBA (production deployment на Kimi.ai).

### Философия: "Less Structure"

MoBA следует принципу **"less structure"**: вместо hardcoded attention patterns (как в Longformer — фиксированное окно + global tokens) маршрутизатор **сам решает**, на какие блоки обращать внимание. Модель может адаптивно выбирать:
- Локальные блоки (для языкового моделирования)
- Далёкие блоки (для long-range dependencies)
- Блоки с ключевой информацией (для retrieval-like задач)

### MoBA в production

MoBA **уже deployed** на Kimi.ai для обработки long-context запросов. Преимущества:
- Seamless transition между full и sparse attention (можно включить MoBA только для длинных запросов)
- Совместимость с FlashAttention (block structure совпадает с FA tile structure)
- Не требует дополнительного обучения — можно применить к любой pretrained модели

## Kimi k1.5: reasoning через RL

### Философия: простота побеждает

В отличие от конкурентов, Kimi k1.5 **намеренно избегает сложных техник**:

| Техника | o1 | DeepSeek-R1 | Kimi k1.5 |
|---------|-----|-------------|-----------|
| MCTS (tree search) | Вероятно да | Нет | **Нет** |
| Value function (critic) | Вероятно да | Нет (GRPO) | **Нет** |
| Process Reward Model | Вероятно да | Нет | **Нет** |
| Мультимодальный reasoning | Нет | Нет | **Да** |

### Online Mirror Descent: policy optimization

Kimi k1.5 использует **online mirror descent** для policy optimization — алгоритм из convex optimization, адаптированный для RL:

$$\pi_{t+1} = \arg\min_\pi \left[ \langle \pi, -\hat{A}_t \rangle + \frac{1}{\eta} D(\pi \| \pi_t) \right]$$

где $D$ — KL-дивергенция, $\hat{A}_t$ — estimated advantage, $\eta$ — learning rate.

**Почему не PPO/GRPO:** mirror descent математически проще и имеет **теоретические гарантии сходимости** в online setting. На практике это упрощает hyperparameter tuning и стабилизирует обучение.

### Длинные Chain-of-Thought через Partial Rollouts

Reasoning-модели генерируют длинные CoT (10K-100K токенов). Проблема: RL-обучение требует генерировать полные trajectories, что при длинных CoT занимает огромное время.

**Partial Rollouts:** вместо генерации с нуля, переиспользуется начало предыдущей траектории, и дописывается только оставшаяся часть:

1. **Replay Buffer:** хранит ранее сгенерированные траектории
2. **Prefix reuse:** для нового rollout берётся prefix из буфера (первые N токенов reasoning)
3. **Continuation:** модель генерирует продолжение от prefix до конца
4. **Reward:** вычисляется для полной траектории (prefix + continuation)

**Результат:**
- Ускорение training в **3-5x** (не нужно генерировать полные 100K-токенные trajectories каждый раз)
- Позволяет работать с CoT длиной **128K+ токенов** (impossible без partial rollouts из-за GPU memory)
- Replay buffer обеспечивает exploration diversity

### RL Framework: полная картина

Архитектура обучения:
1. **Rollout Workers** — генерируют trajectories (ответы с reasoning). Распределены по кластеру GPU
2. **Reward Models** — специализированные для каждого домена:
   - Math: symbolic verification (проверка числового ответа)
   - Code: execution-based (прогон тестов)
   - Vision: VQA-specific metrics
   - K-12: rule-based grading
3. **Policy Optimizer** — online mirror descent с clipping
4. **Replay Buffer** — хранение и переиспользование partial rollouts

Reward-сигнал — **outcome-based** (правильность финального ответа), как и в DeepSeek-R1. Без process rewards, без step-level supervision.

### Мультимодальность: reasoning по картинкам

Уникальная особенность k1.5 — **joint text+vision reasoning**. Модель обучена рассуждать над визуальными задачами (геометрия, графики, диаграммы) с такой же CoT, как для текстовых.

**Как это работает:**
1. Vision encoder (ViT) обрабатывает изображение → visual tokens
2. Visual tokens конкатенируются с text tokens
3. Модель генерирует CoT, переключаясь между описанием визуальных элементов и текстовыми рассуждениями
4. RL-обучение проводится **совместно** на текстовых и визуальных задачах

**Пример:** для задачи по геометрии (найти площадь фигуры на картинке) модель:
1. Описывает фигуру: «Вижу трапецию с основаниями...»
2. Извлекает числа: «Основание a = 5, основание b = 8, высота = 3»
3. Применяет формулу: «S = (a+b)/2 × h = (5+8)/2 × 3 = 19.5»

## Результаты

| Бенчмарк | Kimi k1.5 | o1 | DeepSeek-R1 | GPT-4o |
|-----------|-----------|-----|-------------|--------|
| AIME 2024 | 77.5% | 79.2% | 79.8% | 9.3% |
| MATH-500 | 96.2% | 96.4% | 97.3% | 74.6% |
| MathVista (vision) | **74.9%** | 73.9% | — | 63.8% |
| LiveCodeBench | 47.3% | 49.3% | 65.9% | — |
| MMLU | 89.5% | 91.8% | 90.8% | 87.2% |

Kimi k1.5 конкурентоспособна с o1 при значительно более простом pipeline — без MCTS, без process rewards, без value function. На **vision reasoning** (MathVista) k1.5 **превосходит o1**.

## Влияние на экосистему

1. **Long context pioneer** — Kimi доказала, что пользователи хотят и могут использовать 200K+ контекст. Это ускорило развитие длинного контекста у GPT-4 Turbo (128K), Claude (200K), Gemini (1M+)
2. **MoBA:** первый production-deployed sparse attention для длинных контекстов. Open-source (GitHub), легко интегрируется с существующими моделями
3. **Simplicity thesis** — k1.5 показала, что для reasoning не нужны сложные MCTS/PRM — простой RL достаточен
4. **Multimodal reasoning** — первая открытая мультимодальная reasoning-модель с CoT по изображениям

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-R1|DeepSeek-R1]] — аналогичная reasoning-модель через RL
- [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute|Test-time Compute]] — парадигма extended thinking
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — MoBA как эффективная модификация
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — другой подход к эффективному attention
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — базовый RL для LLM

## Дополнительные ресурсы

- [Kimi k1.5 Paper](https://arxiv.org/abs/2501.12599) — полный отчёт по reasoning
- [MoBA Paper](https://arxiv.org/abs/2502.13189) — архитектура block attention
- [GitHub — MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) — open-source реализация
