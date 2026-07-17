---
title: "Reward Model"
aliases: [Reward Model, RM, Reward Modeling]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/InstructGPT|InstructGPT]]"
  - "[[02 Areas/ML & DL/Papers/DPO|DPO]]"
courses: []
sources:
  - "[Ouyang et al. — Training language models to follow instructions (2022)](https://arxiv.org/abs/2203.02155)"
  - "[Anthropic — Constitutional AI (2022)](https://arxiv.org/abs/2212.08073)"
---

# Reward Model

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/instructgpt/reward-model-hf.png]]
*Обучение Reward Model: люди ранжируют ответы, модель учится предсказывать эти ранжирования через Bradley-Terry loss (источник: Hugging Face)*

## Зачем нужна Reward Model

Цель alignment — научить LLM генерировать ответы, которые люди считают полезными, безопасными и правдивыми. Но люди не могут оценить каждый ответ во время обучения: RL-цикл генерирует тысячи ответов за один training step, а human evaluation стоит дорого и медленно.

**Reward Model (RM)** решает эту проблему: это отдельная нейросеть, обученная предсказывать, какой ответ человек предпочтёт. RM заменяет человека в feedback loop [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]:

```
Без RM:  prompt → LLM → ответ → человек оценивает → gradient
С RM:    prompt → LLM → ответ → RM оценивает → gradient
```

RM обучается один раз на человеческих предпочтениях, а затем используется как proxy для human judgment на протяжении всего RL-обучения.

## Сбор данных: pairwise comparisons

### Почему сравнения, а не рейтинги

Людям проще сравнивать два ответа («A лучше B»), чем давать абсолютные оценки («этот ответ на 7.3 из 10»). Абсолютные оценки субъективны и нестабильны: один аннотатор ставит 7, другой — 4 за тот же ответ. Сравнения гораздо более consistent.

### Процесс сбора

1. Берём набор промптов из целевого распределения
2. Для каждого промпта генерируем **K ответов** (обычно $K = 4\text{–}9$) от текущей policy
3. Аннотаторы ранжируют ответы от лучшего к худшему
4. Из ранжирования извлекаем $\binom{K}{2}$ пар сравнений

Например, при $K = 4$: 4 ответа → 6 пар. При $K = 9$ (как в InstructGPT): 36 пар из одного промпта. Это значительно дешевле, чем собирать 36 отдельных промптов.

В [[02 Areas/ML & DL/Papers/InstructGPT|InstructGPT]] использовали ~33K промптов с $K = 4\text{–}9$, получив ~150K пар сравнений.

## Bradley-Terry Model: формализация предпочтений

RM обучается через **модель Брэдли-Терри** — классическую модель pairwise comparisons. Предположение: существует latent reward function $r(x, y)$ (промпт $x$, ответ $y$), и вероятность предпочтения ответа $y_w$ (winner) над $y_l$ (loser):

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

где $\sigma$ — sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.

**Loss function** для обучения RM:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

Это binary cross-entropy: модель учится присваивать более высокий скалярный reward предпочтённому ответу.

### Свойства Bradley-Terry loss

- Зависит только от **разницы** rewards, а не от абсолютных значений → rewards инвариантны к сдвигу
- Gradient пропорционален $\sigma(r_l - r_w)$ — чем более неправильно модель ранжирует пару, тем сильнее gradient
- При идеальном ранжировании ($r_w \gg r_l$) gradient → 0

## Архитектура Reward Model

### LLM с scalar head

RM — это LLM (обычно той же архитектуры, что и policy), у которой заменён language modeling head на **скалярный head**:

```
Input: [prompt + response tokens]
      ↓
Transformer layers (pretrained)
      ↓
Last hidden state → Linear(d_model, 1) → scalar reward
```

Скалярный reward берётся из last token position (или pooling по всей последовательности).

### Размер RM

| Проект | Policy size | RM size | Соотношение |
|--------|------------|---------|-------------|
| InstructGPT | 175B | 6B | 3% |
| LLaMA 2 | 70B | 70B | 100% |
| Anthropic | ~52B | ~52B | 100% |

В InstructGPT использовали маленькую 6B RM — это экономит compute, но ограничивает expressiveness. LLaMA 2 и Anthropic перешли к RM того же размера, что и policy — это даёт лучшие результаты, но увеличивает memory requirements.

### Инициализация

RM обычно инициализируется из **SFT checkpoint** (не из base model). Логика: SFT-модель уже «понимает» instruction-following, и это лучшая стартовая точка для понимания quality differences между ответами.

## Reward Hacking: главная проблема RM

**Reward hacking** (reward overoptimization) — ситуация, когда policy находит паттерны, максимизирующие RM score, но не улучшающие реальное качество:

| Тип хакинга | Пример | Почему RM обманывается |
|-------------|--------|----------------------|
| Verbosity | Очень длинные ответы с повторами | Длинные ответы часто предпочитались аннотаторами |
| Sycophancy | Модель соглашается с пользователем, даже если он неправ | Аннотаторы предпочитали «приятные» ответы |
| Format gaming | Markdown, bullet points, bold text | Хорошо оформленные ответы казались лучше |
| Hedging | «С одной стороны... с другой стороны...» без definitive answer | Безопасные ответы реже наказывались |

### Goodhart's Law

Reward hacking — проявление закона Гудхарта: «когда мера становится целью, она перестаёт быть хорошей мерой». RM — это proxy для human preferences, и оптимизация этого proxy за пределами его calibration range приводит к деградации.

### Mitigation strategies

- **KL penalty** ($\beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$) — ограничивает distance от reference policy
- **RM ensembles** — несколько RM, reward = min или mean (robustness)
- **Iterative RM training** — переобучение RM на outputs текущей policy
- **PPO-ptx** — подмешивание pretraining data для сохранения general capabilities

## RM в контексте pipeline

Стандартный RLHF pipeline с RM:

```
1. Pre-training → base model
2. SFT → instruction-following model
3. RM training → reward model (на human comparisons)
4. RL (PPO/GRPO) → aligned model (с RM как reward signal)
```

RM используется на шаге 4 как frozen evaluator: она не обновляется во время RL-обучения.

## Альтернативы Reward Model

### DPO: implicit reward model

[[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] показал, что можно обойтись без explicit RM. DPO оптимизирует preference loss напрямую через policy, используя **implicit reward**:

$$r_{DPO}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

Это эквивалентно обучению RM «внутри» policy — не нужен отдельный шаг RM training и RL-цикл.

### RLVR: no RM at all

[[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] (RL with Verifiable Rewards) полностью отказывается от RM — reward определяется **программной проверкой** результата:

$$r = \begin{cases} 1 & \text{если ответ верный (math, code tests, etc.)} \\ 0 & \text{иначе} \end{cases}$$

Это работает для задач с deterministic correctness criteria (математика, код, factual QA), но не для open-ended generation (writing, conversation).

### Constitutional AI

[[02 Areas/ML & DL/Concepts/Training/Constitutional AI|Constitutional AI]] (Anthropic) заменяет human annotators на LLM-as-judge — модель сама оценивает ответы по набору «конституционных» принципов. По сути, это RM, но данные для неё генерируются AI, а не людьми (RLAIF — RL from AI Feedback).

## Тренды 2024–2025

1. **Generative RM** — вместо скалярного score, RM генерирует текстовую critique и оценку (LLM-as-judge)
2. **Process RM** — оценка каждого шага рассуждения, а не только финального ответа (для reasoning)
3. **Уход от explicit RM** — DPO, GRPO с verifiable rewards, RLAIF снижают зависимость от обученных RM
4. **Multi-objective RM** — отдельные scores для helpfulness, safety, truthfulness вместо одного скаляра

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — парадигма, использующая RM как ключевой компонент
- [[02 Areas/ML & DL/Concepts/Training/PPO|PPO]] — RL-алгоритм, оптимизирующий reward от RM
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — подход с implicit RM, не требующий отдельной модели
- [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] — альтернатива RM через verifiable rewards
- [[02 Areas/ML & DL/Concepts/Training/Constitutional AI|Constitutional AI]] — RM через AI feedback вместо human feedback
