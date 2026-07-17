---
title: "PPO"
aliases: [PPO, Proximal Policy Optimization]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/InstructGPT|InstructGPT]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2|LLaMA 2]]"
courses: []
sources:
  - "[Schulman et al. — Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)"
  - "[Hugging Face — Illustrating RLHF](https://huggingface.co/blog/rlhf)"
---

# PPO — Proximal Policy Optimization

![[02 Areas/ML & DL/raw/papers/instructgpt/images/rlhf-pipeline-hf.png]]
*PPO в RLHF pipeline: модель генерирует ответы → Reward Model оценивает → PPO обновляет веса политики с KL-штрафом относительно reference model (источник: Hugging Face)*

## Зачем PPO: проблемы vanilla policy gradient

Vanilla policy gradient (REINFORCE) обновляет параметры политики в направлении градиента:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)\right]$$

где $A(s, a)$ — advantage function. Проблема: **размер шага** ничем не ограничен. Один слишком большой update может необратимо испортить политику — модель «улетает» в область плохих параметров, и восстановление крайне затруднено.

Ранее эту проблему решал **TRPO** (Trust Region Policy Optimization, Schulman 2015), добавляя hard constraint на KL-дивергенцию между старой и новой политиками. Но TRPO требует вычисления матрицы Гессиана и conjugate gradient — это сложно и дорого.

**PPO** (Schulman, 2017) заменяет hard constraint на простой **clipping** в objective — получается почти такая же стабильность, но с простотой SGD.

## Clipped Surrogate Objective

Ключевая формула PPO — clipped surrogate objective:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t A_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

где:
- $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ — importance sampling ratio (вероятность действия под новой политикой, делённая на вероятность под старой)
- $A_t$ — estimated advantage
- $\epsilon$ — hyperparameter клиппинга (обычно 0.1–0.2)

### Как работает клиппинг

| Ситуация | $A_t > 0$ (хорошее действие) | $A_t < 0$ (плохое действие) |
|----------|------|------|
| $\rho_t > 1 + \epsilon$ | Клиппируется: не даём слишком сильно увеличить вероятность | Свободно: разрешаем уменьшить вероятность |
| $\rho_t < 1 - \epsilon$ | Свободно: разрешаем увеличить вероятность | Клиппируется: не даём слишком сильно уменьшить вероятность |

**Интуиция:** $\min$ выбирает «пессимистичный» вариант. Если действие хорошее ($A > 0$), мы хотим увеличить его вероятность, но не более чем до $1+\epsilon$ от текущей. Если действие плохое ($A < 0$), мы хотим уменьшить его вероятность, но не более чем до $1-\epsilon$. Это создаёт **trust region** — неявное ограничение размера шага.

## Advantage Estimation: GAE

PPO использует **Generalized Advantage Estimation** (GAE, Schulman 2016) для оценки advantage:

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l}$$

где $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — TD-error (temporal difference).

Параметр $\lambda \in [0, 1]$ контролирует bias-variance trade-off:
- $\lambda = 0$ — одношаговый TD (low variance, high bias)
- $\lambda = 1$ — Monte Carlo return (high variance, low bias)
- Типичное значение: $\lambda = 0.95$

Для вычисления GAE нужна **value function** $V(s)$ — это отдельная нейросеть (critic), которая предсказывает expected cumulative reward из состояния $s$.

## 4 модели в памяти: проблема масштабирования

При применении PPO к LLM в контексте [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] одновременно нужны **4 модели**:

| Модель | Назначение | Обновляется? |
|--------|-----------|-------------|
| **Policy** $\pi_\theta$ | LLM, которую обучаем | Да |
| **Reference** $\pi_{ref}$ | Замороженная копия policy для KL-penalty | Нет |
| **Value (Critic)** $V_\phi$ | Предсказывает expected reward для GAE | Да |
| **[[02 Areas/ML & DL/Concepts/Training/Reward Model\|Reward Model]]** $R_\psi$ | Оценивает качество ответа | Нет |

Для модели размером 70B это **~280B параметров** в GPU memory. При FP16 — около 560 GB VRAM. На практике это десятки GPU только для inference, не считая optimizer states.

**Reference model** нужна для KL-penalty:

$$\mathcal{L}_{total} = \mathcal{L}^{CLIP} - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

KL-penalty предотвращает «уход» обученной модели слишком далеко от исходной — без неё модель быстро деградирует в general capabilities ради максимизации reward.

## PPO-ptx: борьба с забыванием

OpenAI в [[02 Areas/ML & DL/Papers/InstructGPT|InstructGPT]] обнаружили, что PPO-обучение вызывает **catastrophic forgetting** — модель теряет знания из pretraining. Решение — **PPO-ptx** (pretraining mix):

$$\mathcal{L}_{PPO-ptx} = \mathcal{L}^{CLIP} + \alpha \cdot \mathcal{L}_{pretrain}$$

где $\mathcal{L}_{pretrain}$ — стандартный language modeling loss на данных из pretraining корпуса. Коэффициент $\alpha$ обычно подбирается так, чтобы pretraining loss составлял ~10% от общего.

Это позволяет модели сохранять language capabilities, одновременно обучаясь следовать инструкциям.

## Алгоритм PPO для LLM: пошагово

```
1. Сэмплируем batch промптов из dataset
2. Генерируем ответы текущей policy π_θ
3. Получаем rewards от Reward Model
4. Вычисляем values V_ϕ(s) для каждого токена
5. Считаем GAE advantages
6. PPO update (K эпох по mini-batches):
   a. Вычисляем ρ_t = π_θ(a|s) / π_old(a|s)
   b. Clipped objective
   c. Value function loss: L_V = (V_ϕ(s) - R_t)²
   d. KL penalty от reference model
   e. SGD step
7. Повторяем
```

Типичные гиперпараметры для LLM:
- $\epsilon = 0.2$ (clip range)
- $\lambda = 0.95$ (GAE)
- $\gamma = 1.0$ (no discounting for finite-length episodes)
- KL coefficient $\beta = 0.02$
- PPO epochs $K = 4$

## PPO vs GRPO: почему PPO уступает

[[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] (DeepSeek, 2024) решает ту же задачу оптимизации политики, но **убирает critic**:

| Свойство | PPO | GRPO |
|----------|-----|------|
| Модели в памяти | 4 | 2 + reward fn |
| Advantage estimation | Через critic $V_\phi$ | Нормализация rewards по группе |
| Стабильность critic | Проблема: critic часто ошибается | Нет critic — нет проблемы |
| Compute на generation | 1 ответ/промпт | G ответов/промпт |
| Memory overhead | ~4x policy size | ~2x policy size |

Critic для LLM — модель того же масштаба, что и policy. Её нужно обучать одновременно, и quality advantage estimates напрямую зависит от quality critic. На практике critic часто даёт шумные estimates, особенно для reasoning задач с длинными цепочками рассуждений.

GRPO заменяет critic эмпирической оценкой: генерирует G ответов на один промпт и сравнивает их rewards. Это проще, стабильнее и требует меньше памяти.

## Почему PPO вытесняется в 2024–2025

Три основных тренда привели к снижению роли PPO:

**1. GRPO для reasoning.** DeepSeek-R1 показал, что GRPO с outcome-based rewards (без reward model вовсе) обучает reasoning не хуже PPO, но значительно дешевле. Это направление развивается в [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]].

**2. [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] для alignment.** DPO (Direct Preference Optimization) вообще не требует RL-цикла — оптимизирует implicit reward через contrastive loss на парах предпочтений. Гораздо проще в реализации.

**3. Сложность инженерии.** PPO для LLM — один из самых сложных training pipeline в ML. Четыре модели, GAE, clipping, KL-penalty, PPO-ptx — каждый компонент требует тщательной настройки. На практике это означает месяцы инженерной работы для стабильного training.

PPO остаётся важным для понимания RLHF и как baseline, но в production alignment pipeline 2025 года его чаще заменяют GRPO (для reasoning) или DPO (для general alignment).

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — парадигма, в которой PPO используется как RL-алгоритм
- [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] — замена PPO без critic, стандарт 2025 года
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — RL-free альтернатива PPO для alignment
- [[02 Areas/ML & DL/Concepts/Training/Reward Model|Reward Model]] — модель оценки качества ответов, используемая в PPO
- [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] — verifiable rewards, часто сочетается с GRPO вместо PPO
