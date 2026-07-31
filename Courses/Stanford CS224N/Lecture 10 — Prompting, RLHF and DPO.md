---
title: "CS224N — Lecture 10: Prompting, Instruction Finetuning, and DPO/RLHF"
course: "Stanford CS224N"
lecture: 10
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture10-prompting-rlhf]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-Context Learning]]", "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]", "[[02 Areas/ML & DL/Concepts/Training/DPO|DPO]]", "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]", "[[02 Areas/ML & DL/Concepts/NLP/Few-Shot Learning|Few-Shot Learning]]"]
---

# Lecture 10: Prompting, Instruction Finetuning, and DPO/RLHF

## Emergent Abilities при масштабировании

### От GPT к GPT-3: прорыв в каждом порядке величин

| Модель | Год | Params | Новые способности |
|--------|-----|--------|-------------------|
| GPT-1 | 2018 | 117M | Fine-tuning pretrained Transformer на downstream tasks |
| GPT-2 | 2019 | 1.5B | **Zero-shot** learning ("TL;DR" для суммаризации без обучения) |
| GPT-3 | 2020 | 175B | **Few-shot in-context learning** без обновления весов |

GPT-2 показал zero-shot transfer: модель, обученная только на language modeling, может выполнять задачи **без специального обучения**, если правильно сформулировать промпт.

GPT-3 (Brown et al., 2020) -- прорыв: **few-shot in-context learning**. Добавляем несколько пар (input, output) прямо в промпт, и модель "понимает" задачу без gradient updates.

## Zero-shot и Few-shot Learning

### Zero-shot

Формулируем задачу как текст:

```
Translate English to French: cheese =>
```

Модель генерирует "fromage". Никаких примеров -- только инструкция и задание.

### Few-shot (In-Context Learning)

Добавляем примеры прямо в промпт:

```
Translate English to French:
sea otter => loutre de mer
cheese => fromage
peppermint =>
```

Модель "обучается" на примерах **в контексте** (in context) без изменения весов. Это не обучение в классическом смысле -- скорее pattern recognition.

### Scaling in-context learning

Больше примеров -> лучше результат (до ~32 примеров). Но:
- Ограничено длиной контекстного окна
- Порядок примеров влияет на результат (prompt sensitivity)
- Модели < 1B параметров не показывают in-context learning

## Instruction Finetuning (SFT)

### Проблема pretrained LM

Pretrained LM -- "продолжатель текста". Если спросить "What is the capital of France?", модель может продолжить: "...is a common trivia question" вместо "Paris". Модель оптимизирована на **вероятность текста**, а не на **полезность ответа**.

### Supervised Fine-Tuning (SFT)

Обучение на dataset парах (instruction, response):

```
Instruction: Summarize the following article...
Response: The article discusses...
```

Datasets: FLAN (Google), InstructGPT data (OpenAI), Alpaca (Stanford), ShareGPT.

**Результат**: модель превращается из "продолжателя текста" в "помощника" -- следует инструкциям, отвечает на вопросы, отказывается от опасных запросов.

### FLAN (Finetuned Language Net)

Wei et al. (2022): fine-tuning на **1,836 задач** с инструкциями. Key finding: instruction-tuned модель обобщается на **новые задачи** (zero-shot transfer улучшается).

## RLHF: Reinforcement Learning from Human Feedback

### Мотивация: что не так с SFT?

SFT оптимизирует log-likelihood ответа из dataset. Но:
1. **Один "правильный" ответ** на каждый вопрос -- а хороших ответов может быть много
2. **Нет контроля стиля**: модель может быть verbose, unhelpful, unsafe
3. **Hallucinations**: SFT не учит отказываться от ответа, когда модель не уверена

Человеческие предпочтения сложно формализовать через loss function. RLHF позволяет оптимизировать по **неявным** человеческим критериям.

### Pipeline (InstructGPT, Ouyang et al., 2022)

**Шаг 1: SFT** -- fine-tune на демонстрациях (human-written responses to prompts)

**Шаг 2: Reward Model (RM)**
- Для каждого промпта модель генерирует несколько ответов
- Человек **ранжирует** ответы попарно: $y_w \succ y_l$ (winner vs loser)
- Reward model обучается предсказывать человеческие предпочтения:

$$L_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} [\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]$$

Это Bradley-Terry model для pairwise comparisons.

**Шаг 3: PPO (Proximal Policy Optimization)**
- Policy (LLM) оптимизируется по reward model:

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi} [r_\theta(x, y)] - \beta \cdot D_{\text{KL}}[\pi \| \pi_{\text{SFT}}]$$

- KL-penalty от SFT модели: не уходить слишком далеко от "базовых" знаний
- Без KL penalty модель **game-ит** reward model (reward hacking)

### Проблемы RLHF

1. **Сложность**: три отдельных модели (SFT, RM, policy), PPO нестабилен
2. **Reward hacking**: модель находит exploits в reward model
3. **Дорогие human labels**: нужны тысячи пар сравнений
4. **Reward model drift**: RM может не обобщаться на out-of-distribution outputs

## [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]]: Direct Preference Optimization

### Ключевая идея

Rafailov et al. (2023) показали, что optimal policy RLHF имеет **closed-form solution**:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r^*(x, y)\right)$$

Из этого можно выразить reward через policy:

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \text{const}$$

### DPO Loss

Подставляя в Bradley-Terry model:

$$L_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### Преимущества DPO

1. **Не нужна reward model**: оптимизируем policy напрямую
2. **Не нужен PPO**: стандартный cross-entropy-like loss
3. **Стабильный**: нет RL, нет reward hacking
4. **Простой**: один forward/backward pass через LLM

### DPO на практике

DPO стал де-факто стандартом alignment (2023-2024): проще имплементировать, стабильнее обучать, сравнимые результаты с RLHF. Используется в LLaMA 2, Zephyr, Mistral и многих других.

Варианты: IPO, KTO, ORPO, SimPO -- разные функции loss для preference optimization.

## Полный pipeline обучения LLM (2024)

```
Web Crawl (трлн токенов)
    |
[Pretraining: autoregressive LM]     -- base model (GPT-4 base, LLaMA base)
    |
[SFT: instruction following]         -- chat model (follows instructions)
    |
[DPO/RLHF: alignment]                -- aligned model (helpful, harmless, honest)
    |
Production model (ChatGPT, Claude)
```

Каждый этап использует **менее данных, но более качественных**:
- Pretraining: ~10T tokens, unsupervised
- SFT: ~100K examples, curated
- DPO/RLHF: ~10-100K preference pairs, human-annotated

## LM как World Models

Модели демонстрируют рудиментарное моделирование мира:
- **Othello-GPT** (Li et al., 2023): LM, обученная на последовательностях ходов, выучивает **внутреннее представление доски** (linear probe находит состояние доски в hidden states)
- **Медицинские тексты**: LLM показывают performance уровня врачей на USMLE
- **Код**: GPT-4 решает ~90%+ на [[02 Areas/ML & DL/Concepts/Evaluation/HumanEval|HumanEval]]
- **Математика**: chain-of-thought reasoning, tool use для сложных вычислений

Открытый вопрос: являются ли LLM "stochastic parrots" или формируют genuine understanding?

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-Context Learning]] -- задача задаётся примерами в промпте, без gradient updates
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] -- SFT -> Reward Model -> PPO, alignment through human preferences
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] -- Direct Preference Optimization, closed-form solution для RLHF
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] -- SFT на парах (instruction, response), FLAN
- [[02 Areas/ML & DL/Concepts/NLP/Few-Shot Learning|Few-Shot Learning]] -- обучение на нескольких примерах в контексте
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] -- цепочка рассуждений при prompting
