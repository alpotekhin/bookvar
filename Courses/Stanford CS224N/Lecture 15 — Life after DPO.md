---
title: "CS224N — Lecture 15: Life after DPO"
course: "Stanford CS224N"
lecture: 15
type: course-note
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture15-life-after-dpo-lambert]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Training/DPO|DPO]]", "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]", "[[02 Areas/ML & DL/Concepts/NLP/Alignment|Alignment]]", "[[02 Areas/ML & DL/Concepts/Training/Reward Model|Reward Model]]"]
---

# Lecture 15: Life after DPO (Guest: Nathan Lambert, Allen AI)

> *"Can ChatGPT exist without RLHF? RLHF seems to be necessary, but not sufficient."* -- Nathan Lambert

Лектор: Nathan Lambert (Allen Institute for AI, @natolambert).

## Краткая история Language Models

| Год | Событие |
|-----|---------|
| 1948 | Claude Shannon моделирует английский язык |
| 2017 | [[02 Areas/ML & DL/Concepts/Architectures/Transformer\|Transformer]] -- "Attention Is All You Need" |
| 2018 | GPT-1 (117M), ELMo, BERT |
| 2019 | GPT-2 (1.5B), scaling laws |
| 2020 | GPT-3 (175B) -- surprising capabilities, many harms |
| 2021 | "Stochastic Parrots" debate |
| 2022 | **ChatGPT** -- RLHF-aligned GPT-3.5, момент перелома |

## Определения alignment

| Термин | Значение |
|--------|----------|
| **IFT** (Instruction Fine-Tuning) | Обучение модели следовать инструкциям (autoregressive LM loss) |
| **SFT** (Supervised Fine-Tuning) | Обучение task-specific capabilities |
| **Alignment** | Обучение модели соответствовать человеческим ценностям (любой loss) |
| **[[02 Areas/ML & DL/Concepts/Training/RLHF\|RLHF]]** | Конкретный технический инструмент: RL от human feedback |
| **Preference Fine-Tuning** | Обучение на labeled preference data (RLHF, DPO, или другой loss) |

## Post-training Pipeline

```
1. Pretraining:  next-token prediction на огромном корпусе
        ↓
2. SFT/IFT:     instruction following на curated data
        ↓                    ┌─ Chat templates
        ↓                    ├─ System prompts
        ↓                    └─ Multi-turn dialogues
3. Preference Optimization: RLHF или DPO
```

### Instruction Fine-Tuning (IFT)

Адаптация base model к формату interaction: chat templates с special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`). Продолжение обучения на парах question:answer.

## [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]: объектив и компоненты

### Objective

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}[\pi \| \pi_\theta]$$

Два слагаемых:
- **Maximize reward** $r(x, y)$: ответы должны нравиться людям
- **KL penalty**: модель не должна слишком отклоняться от base policy $\pi_\theta$ (потому что **preferences сложно моделировать**, и reward model может быть hacked)

### [[02 Areas/ML & DL/Concepts/Training/Reward Model|Reward Model]]

Скалярные оценки "насколько хорош ответ" **не работают** при сборе от аннотаторов. **Pairwise preferences** -- проще и надёжнее.

**Bradley-Terry model**: оценить вероятность, что данное preference верно:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Обучение: maximize log-likelihood на парах (chosen, rejected).

**Практические детали**:
- Обучение за **1 epoch** (иначе overfitting!)
- Evaluation agreement часто лишь 65-75%
- Можно добавить margin между choices в loss

## [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] (Direct Preference Optimization)

### Ключевая идея

Rafailov, Sharma, Mitchell et al. (29 May 2023): если принять оптимальную policy RLHF и подставить обратно, можно **обойтись без reward model и RL** -- обучать напрямую на preference data.

### Характеристики DPO

1. **Extremely simple** to implement (несколько строк кода)
2. **Масштабируется** с существующими distributed training библиотеками
3. Обучает **implicit reward function** (DPO-модель можно использовать как reward model)

### DPO vs PPO (RL)

DPO и PPO -- **фундаментально разные** оптимизаторы:
- DPO: обучение **напрямую** на preferences (offline)
- PPO: RL update rules с exploration (online)
- Не совсем online vs offline RL, но различие существенно

## Хронология открытых моделей

### Первые instruction-tuned модели (март-апрель 2023)

| Модель | Дата | Данные | MT-Bench |
|--------|------|--------|----------|
| **Alpaca** | 13 Mar | 52K self-instruct от text-davinci-003 | 4.53 (13B) |
| **Vicuna** | 30 Mar | ChatGPT conversations (ShareGPT) | 6.69 (7B) |
| **Koala** | 3 Apr | Diverse (Alpaca + Anthropic HH + ShareGPT) | 6.08 (13B) |
| **Dolly** | 12 Apr | 15K human-written | 3.28 (12B) |

### Ключевой ресурс: ShareGPT

Данные из инструмента для sharing ChatGPT-разговоров. **Юридически серая зона** (unlicensed, without consent). Постепенно заменяются на carefully collected:
- **LMSYS-Chat-1M**: cleaned conversations из Chatbot Arena
- **WildChat**: free ChatGPT usage в обмен на данные

### OpenAssistant (15 April 2023)

**Первый открытый human instruction dataset**: 161K messages, 35 языков, 461K quality ratings, 13500+ волонтёров. До сих пор единственный human dataset такого масштаба.

### StableVicuna (28 April 2023)

**Первая открытая RLHF-модель**: PPO на OAsst1 + Anthropic HH + Stanford Human Preferences.

### Transition period (Jun-Oct 2023)

WizardLM, UltraLM, OpenChat, XwinLM (первая модель, побившая GPT-4 на AlpacaEval!), OpenHermes -- серия сильных моделей, но ни одна не сдвинула narrative радикально.

### DPO works: Zephyr Beta (Oct 2023)

**Первая модель**, сделавшая splash с DPO:
- Fine-tune Mistral 7B + UltraFeedback dataset
- Открытие: **очень низкие learning rates** (~5e-7) -- теперь стандарт для DPO
- MT-Bench: **7.34** (7B модель!)

### DPO scales: Tulu 2 (Allen AI)

**Первая модель**, масштабировавшая DPO до **70B** параметров. MT-Bench 70B: **7.89**. Запустила серьёзный "DPO vs PPO" debate.

### PPO strikes back: SteerLM, Starling

SteerLM (NVIDIA): attribute-conditioned fine-tuning. Starling (Berkeley): новый preference dataset **Nectar**, k-wise reward model loss (beyond pairwise). MT-Bench 7B: **8.09** -- лучше всех кроме GPT-4.

## RewardBench (Lambert et al. 2024)

### Мотивация

Reward models -- критический компонент RLHF, но **нет стандартной evaluation**. RewardBench -- первый бенчмарк для reward models.

### Структура

Четыре категории:
- **Chat**: общее качество ответов
- **Chat Hard**: тонкие различия, trick questions -- **единственная meaningful eval** по мнению автора
- **Safety**: различение safe/unsafe ответов
- **Reasoning**: математика, код, логика

### Результаты (май 2024)

- DPO-модели как reward models -- работают, но **замедляются** в прогрессе
- LLM-as-judge -- **не SOTA** как reward model
- Closed lab models (Cohere) опережают best open models на 2-3 пункта
- Chat Hard -- наиболее информативная и сложная категория

### Safety patterns

Три паттерна поведения моделей:
1. **Handles safety well**: корректно различает safe/unsafe
2. **Refuses everything**: отказывает даже на безобидные запросы
3. **Responds to everything**: не различает harmful content

## PPO vs DPO: Unpacking (Ivison et al. 2024, Allen AI)

### Эксперимент на Tulu 2 (13B)

| Шаг | Результат |
|-----|-----------|
| SFT only | Baseline |
| + DPO (Anthropic HH) | Небольшой bump в Chat, Safety, Truthfulness |
| + DPO (UltraFeedback) | Бо'льший jump -- данные важнее метода |
| **Switch to PPO** (UltraFeedback) | Bump на **больше** метрик (Factuality!), biggest jump на AlpacaEval 2 |
| Scale up RM | Reasoning улучшается, другие метрики могут **упасть** |
| Add more prompts | Messy improvements, не всегда помогает |

### Выводы

- **"Always one more thing to ablate"** -- training pipeline сложно
- **"PPO gets the best model, but we don't know why"**
- Generation в PPO **очень медленная** без accelerated inference (vLLM)
- Data quality важнее, чем DPO vs PPO

## Online DPO и открытые проблемы

### Текущий landscape

- Ресурсов для full RLHF (human data + PPO infrastructure) у **большинства labs нет**
- DPO **намного проще** для входа в alignment research
- Активные направления:
  1. **Better evaluation** для alignment (RewardBench, ArenaHard)
  2. **Улучшение DPO**: online DPO variants, better preference data
  3. **Reward model overoptimization**: как не "сломать" reward model
  4. **Scalable oversight**: как контролировать модели, превосходящие человека

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] -- Direct Preference Optimization, обучение без reward model
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] -- reward maximize + KL constraint, PPO optimization
- [[02 Areas/ML & DL/Concepts/NLP/Alignment|Alignment]] -- выравнивание моделей с человеческими ценностями
- [[02 Areas/ML & DL/Concepts/Training/Reward Model|Reward Model]] -- Bradley-Terry, pairwise preferences, RewardBench
