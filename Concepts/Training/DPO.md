---
title: "DPO"
aliases: [DPO, Direct Preference Optimization]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/DPO|DPO]]"
courses: []
sources:
  - "[Cameron Wolfe — Direct Preference Optimization](https://cameronrwolfe.substack.com/p/direct-preference-optimization)"
  - "[Hugging Face — From RLHF to DPO](https://huggingface.co/blog/ariG23498/rlhf-to-dpo)"
  - "[Together AI — DPO Technical Deep Dive](https://www.together.ai/blog/direct-preference-optimization)"
---

# DPO — Direct Preference Optimization

## Зачем это нужно: сложность RLHF pipeline

[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] работает, но его pipeline мучительно сложен:

1. Обучи SFT модель
2. Собери preference data, обучи Reward Model
3. Запусти PPO — нужно держать в памяти **4 модели** одновременно: policy, reference policy, reward model, value function
4. PPO чувствителен к гиперпараметрам ($\beta$, clip range, learning rate, batch size), нестабилен, склонен к reward hacking

DPO (Rafailov et al., 2023, Stanford, NeurIPS 2023) устраняет шаги 2 и 3 целиком. Ключевой инсайт: **языковая модель *неявно* является reward model** — оптимальная политика для KL-constrained reward maximization выражается аналитически, и можно оптимизировать её напрямую через binary cross-entropy loss.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/fig1.png]]
*RLHF (слева) vs DPO (справа): RLHF требует отдельного обучения reward model и RL loop. DPO оптимизирует policy напрямую из preference data через classification loss (источник: Rafailov et al., 2023)*

## Математическая деривация: от RLHF к DPO

DPO — не эвристика, а **точное математическое преобразование** RLHF objective. Разберём деривацию шаг за шагом.

### Шаг 1: RLHF objective

Стандартная цель RLHF — найти политику $\pi$, которая максимизирует reward, не уходя далеко от reference policy $\pi_{\text{ref}}$:

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(y|x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}(\pi(y|x) \| \pi_{\text{ref}}(y|x))$$

Этот KL-constrained objective имеет **closed-form решение** (через вариационное исчисление):

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x, y)}{\beta}\right)$$

где $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \cdot \exp(r(x,y)/\beta)$ — partition function (зависит только от промпта $x$).

**Почему это важно:** оптимальная политика выражена *аналитически* через reward function. Обычно в RLHF мы ищем её итеративно через PPO — но она имеет точную формулу.

### Шаг 2: reparametrization reward через policy

Перевернём формулу — выразим reward через policy:

$$r(x, y) = \beta \cdot \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \cdot \log Z(x)$$

Reward для любой пары $(x, y)$ полностью определяется **отношением** оптимальной policy к reference policy (плюс constant, зависящий только от $x$).

### Шаг 3: подстановка в Bradley-Terry model

Модель человеческих предпочтений Bradley-Terry:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Подставляем reparametrized reward:

$$P(y_w \succ y_l | x) = \sigma\left(\beta \cdot \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**$Z(x)$ сокращается!** Partition function зависит только от $x$ и одинакова для $y_w$ и $y_l$, поэтому в разности она исчезает. Это ключевой момент деривации — именно поэтому DPO работает без вычисления $Z(x)$, которое intractable для LLM.

### Шаг 4: DPO loss

Заменяем $\pi^*$ на обучаемую $\pi_\theta$ и применяем maximum likelihood:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left(\beta \cdot \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]$$

Это **binary cross-entropy loss**. Входные данные: тройки $(x, y_w, y_l)$. Один forward pass через $\pi_\theta$ и $\pi_{\text{ref}}$ → log ratios → BCE loss. Без RL, без sampling из policy, без reward model.

### Анализ градиента: почему DPO не деградирует

Градиент DPO loss:

$$\nabla_\theta \mathcal{L} = -\beta \cdot \sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)) \cdot \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]$$

где $\hat{r}_\theta(y) = \beta \log(\pi_\theta(y|x) / \pi_{\text{ref}}(y|x))$ — implicit reward.

Три компонента:
1. **$\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$** — вес примера: тем больше, чем хуже модель *сейчас* ранжирует пару. Если модель уже правильно ранжирует — градиент мал. **Adaptive weighting.**
2. **$\nabla_\theta \log \pi_\theta(y_w|x)$** — увеличивает вероятность preferred completion
3. **$-\nabla_\theta \log \pi_\theta(y_l|x)$** — уменьшает вероятность dispreferred completion

**Почему это не naive unlikelihood training?** Без weighting coefficient (просто «увеличь $y_w$, уменьши $y_l$») модель быстро деградирует — уменьшает вероятность *всех* ответов. Adaptive weight $\sigma(...)$ предотвращает это: модель фокусируется на примерах, где она ещё ошибается.

## Pipeline DPO на практике

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/dpo-pipeline.png]]
*DPO training pipeline: SFT model → reference policy (frozen) → preference dataset → DPO optimization через binary cross-entropy loss. Вместо 4 моделей RLHF нужны только 2: policy + reference (источник: Cameron Wolfe)*

**Гиперпараметры:**
- $\beta = 0.1$ — стандартное значение. Контролирует «жёсткость» KL constraint. Больше $\beta$ → модель ближе к reference (безопаснее, но менее выразительно)
- Learning rate: $10^{-6}$ — очень маленький, типичный для alignment
- Эпохи: 1-3 — больше → overfitting к preference data

## Результаты: DPO vs PPO

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/fig3.png]]
*Frontier analysis: DPO vs PPO на TL;DR summarization. DPO доминирует PPO по win rate при comparable KL divergence (источник: Rafailov et al., 2023)*

| Задача | Метрика | DPO | PPO | Best-of-N |
|--------|---------|-----|-----|-----------|
| **TL;DR** | Win rate vs human | **61%** | 57% | 60% |
| **CNN/DailyMail** (OOD) | Win rate | **36%** | 26% | — |
| **Anthropic-HH** | Win rate | **58%** | 52% | — |
| **Sentiment** (IMDb) | Control accuracy | **Best** | Good | — |

**Ключевое:** DPO не просто сопоставим с PPO — он **превосходит** по нескольким метрикам, при этом радикально проще.

**OOD generalization:** На CNN/DailyMail (модель не видела эти данные) DPO win rate 36% vs PPO 26%. DPO лучше обобщается за пределы training distribution.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/fig4.png]]
*Controlled sentiment generation: DPO vs PPO. DPO точнее контролирует sentiment при меньшем KL divergence (источник: Rafailov et al., 2023)*

## Теоретические гарантии

**Эквивалентность reward:** Два reward function $r$ и $r'$ эквивалентны, если $r(x,y) - r'(x,y) = f(x)$ для некоторой функции только от промпта. Implicit reward из DPO удовлетворяет этому отношению с «настоящим» RLHF reward. Это означает: **DPO оптимизирует ту же цель, что и RLHF**, без аппроксимаций.

**Полнота выразительности:** Класс implicit reward models DPO может представить **любой** класс эквивалентности reward functions — нет потери выразительности по сравнению с explicit RM.

## Сравнение pipeline: DPO vs RLHF

| Аспект | DPO | RLHF (PPO) |
|--------|-----|-----------|
| **Этапы** | SFT → DPO | SFT → RM → PPO |
| **Моделей в памяти** | 2 (policy + reference) | 4 (policy, ref, RM, value) |
| **Тип обучения** | Offline (gradient descent) | Online (RL loop) |
| **Explicit RM** | Нет | Да |
| **Sampling из policy** | Не нужен | Нужен на каждом шаге |
| **Сложность кода** | ~50 строк core logic | Тысячи строк |
| **Стабильность** | Высокая (BCE loss) | Чувствителен к гиперпараметрам |
| **Compute** | Ниже | Выше |
| **Online adaptation** | Ограничена | Возможна |

**Когда DPO хуже PPO:** DPO работает на *offline* preference data. Если quality модели существенно меняется в процессе обучения, данные устаревают. PPO (online) генерирует свежие данные на каждом шаге. Для iterative alignment PPO может быть предпочтительнее.

## Adoption и влияние

DPO стал стандартным методом alignment для open-source LLM:

- **Mixtral Instruct** — DPO alignment
- **Qwen2 / Qwen2.5** — DPO
- **Zephyr** — первая крупная open-source модель с DPO
- **HuggingFace TRL** — DPOTrainer из коробки
- **Intel Neural Chat** — DPO alignment
- **Многие модели на HF Hub** — тысячи моделей обучены через DPO

## Варианты и расширения DPO

| Метод | Идея | Когда полезен |
|-------|------|---------------|
| **IPO** (Identity PO) | Убирает Bradley-Terry assumption | Шумные preference data |
| **KTO** (Kahneman-Tversky) | Не нужны *пары* — только «хорошо/плохо» | Нет парных preference |
| **ORPO** | Объединяет SFT + DPO в один шаг | Упрощение pipeline |
| **SimPO** | Reference-free — не нужна $\pi_{\text{ref}}$ | Экономия памяти |
| **Online DPO** | Генерирует новые пары в процессе | Лучшая exploration |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — предшественник, который DPO заменяет
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — DPO как метод fine-tuning
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — SFT шаг перед DPO
- [[02 Areas/ML & DL/Concepts/Training/Constitutional AI|Constitutional AI]] — альтернативный подход к alignment
- [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] — DPO часто обучается через LoRA

## Дополнительные ресурсы

- [Cameron Wolfe — Direct Preference Optimization](https://cameronrwolfe.substack.com/p/direct-preference-optimization) — лучший deep dive с математикой
- [Hugging Face — From RLHF to DPO](https://huggingface.co/blog/ariG23498/rlhf-to-dpo) — пошаговая деривация
- [ICLR Blog — RLHF without RL](https://iclr-blogposts.github.io/2024/blog/rlhf-without-rl/) — теоретический анализ
- [Together AI — DPO Deep Dive](https://www.together.ai/blog/direct-preference-optimization) — практические аспекты
