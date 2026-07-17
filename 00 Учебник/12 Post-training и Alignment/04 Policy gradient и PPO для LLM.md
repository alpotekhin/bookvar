---
title: Policy gradient и PPO для LLM
type: textbook-chapter
status: canonical
last_updated: 2026-07-17
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling]]"
next:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO]]"
primary_sources:
  - https://arxiv.org/abs/1707.06347
  - https://arxiv.org/abs/2009.01325
  - https://arxiv.org/abs/2203.02155
---

# Policy gradient и PPO для LLM

> [!abstract] После этой главы
> Можно будет самостоятельно вывести REINFORCE из вероятности ответа, объяснить
> назначение baseline и advantage, вручную посчитать PPO clipping и проследить
> один rollout через policy, reward model, value model, old policy и reference
> policy. Это глава об алгоритме обучения, а не перечень компонентов RLHF.

## 1. Где именно в генерации появляется reinforcement learning

Пусть prompt $x$ фиксирован, а модель породила ответ
$y=(y_1,\ldots,y_T)$. Авторегрессионная вероятность **всего** ответа равна

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}).
$$

В SFT нам известен target $y^*$, и мы максимизируем его likelihood. В RL target
не дан. Модель сама семплирует ответ, а среда возвращает число $R(x,y)$.
Например, для запроса «напиши функцию и тесты» reward может состоять из доли
пройденных тестов, штрафа за неверный формат и оценки полезности.

Цель policy:

$$
J(\theta)=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)].
$$

Сложность в том, что токены дискретны: через выбор `argmax` или sampling нельзя
обычным backprop провести производную reward. Выход даёт log-derivative trick.

## 2. Вывод REINFORCE без пропущенного шага

Для простоты перечислим все возможные ответы:

$$
J(\theta)=\sum_y \pi_\theta(y\mid x)R(x,y).
$$

Считаем reward внешним по отношению к параметрам policy:

$$
\begin{aligned}
\nabla_\theta J
&=\sum_y R(x,y)\nabla_\theta\pi_\theta(y\mid x)\\
&=\sum_y R(x,y)\pi_\theta(y\mid x)
  \nabla_\theta\log\pi_\theta(y\mid x)\\
&=\mathbb E_{y\sim\pi_\theta}
  [R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)].
\end{aligned}
$$

Вторая строка использует тождество
$\nabla p=p\nabla\log p$. Для последовательности:

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}),
$$

поэтому один sequence reward создаёт gradient на всех выбранных токенах:

$$
\nabla J
=\mathbb E\left[
R(x,y)\sum_t\nabla\log\pi_\theta(y_t\mid x,y_{<t})
\right].
$$

Важно: модель не получает метку «какой токен был ошибочным». Она лишь делает
всю sampled trajectory вероятнее при положительном signal и менее вероятной при
отрицательном. Это и есть проблема credit assignment.

### Ручной пример REINFORCE

Модель выбирает один из двух однотокенных ответов:

| ответ | probability | reward |
|---|---:|---:|
| `A` | 0.60 | 1 |
| `B` | 0.40 | 0 |

Ожидаемый reward равен $0.6$. Если sampled `A`, Monte Carlo estimator gradient:

$$
\hat g=1\cdot\nabla\log 0.6.
$$

Если sampled `B`, estimator равен нулю. В среднем направление верно, но отдельные
updates шумны. Для длинного текста variance гораздо выше: два почти одинаковых
ответа могут получить разные оценки, а один terminal reward умножает gradients
тысяч токенов.

## 3. Baseline и advantage: сравнение с ожиданием

Из reward можно вычесть baseline $b(x)$, не зависящий от sampled action:

$$
\mathbb E[(R-b)\nabla\log\pi]=
\mathbb E[R\nabla\log\pi],
$$

поскольку $\mathbb E[\nabla\log\pi]=0$. Bias не появляется, а variance может
сильно уменьшиться.

Value function оценивает expected future return:

$$
V_\psi(s_t)\approx
\mathbb E[R_t\mid s_t],\qquad
A_t=Q(s_t,a_t)-V(s_t).
$$

Advantage отвечает не на вопрос «хорош ли ответ вообще», а на вопрос «насколько
выбранное действие лучше ожидаемого в этом состоянии».

### Числовой пример

Для prompt модель обычно получает reward около $0.7$. Два rollout:

| rollout | $R$ | baseline $V$ | advantage |
|---|---:|---:|---:|
| хороший | 1.0 | 0.7 | +0.3 |
| плохой | 0.2 | 0.7 | −0.5 |

Оба rewards неотрицательны, но второй ответ надо сделать менее вероятным. Raw
reward этого явно не показывает, advantage показывает.

При token-level rewards используют TD residual и generalized advantage estimate:

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\qquad
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda$ задаёт компромисс bias–variance: малое значение сильнее опирается на
learned value, большое приближается к Monte Carlo return.

## 4. LLM как episodic environment

Для генерации текста удобно сопоставление:

| RL | language model |
|---|---|
| initial state | prompt $x$ |
| state $s_t$ | prompt и prefix $y_{<t}$ |
| action $a_t$ | следующий token $y_t$ |
| policy | softmax LM |
| episode | ответ до EOS/лимита |
| terminal reward | RM, verifier или tests |

В классическом RLHF reward model обычно ставит scalar всему ответу. Дополнительно
на каждом токене можно начислять KL penalty. Тогда shaped reward имеет вид

$$
r_t^{\mathrm{KL}}=-\beta
\left(\log\pi_\theta(y_t\mid s_t)
-\log\pi_{\mathrm{ref}}(y_t\mid s_t)\right),
$$

а на последнем шаге добавляется $r_\phi(x,y)$.

## 5. Пять ролей, которые нельзя называть одной «моделью PPO»

![[02 Areas/ML & DL/raw/papers/instructgpt/images/rlhf-pipeline-hf.png]]

*Схема показывает current policy, frozen reference/base model, reward model и
RL update. В полном PPO pipeline дополнительно нужен value model, а rollout
snapshot $\pi_{old}$ логически отделён от reference. Поэтому изображение удобно
как карта потока, но не как исчерпывающий список объектов. Иллюстрация:
Hugging Face, визуальный разбор RLHF.*

| роль | обучается? | назначение |
|---|---|---|
| current policy $\pi_\theta$ | да | policy, которую улучшаем |
| old policy $\pi_{\mathrm{old}}$ | frozen на batch | породила текущие rollouts; denominator PPO ratio |
| reference $\pi_{\mathrm{ref}}$ | обычно frozen дольше | якорь поведения и KL penalty |
| reward model $r_\phi$ | нет во время PPO | оценивает полный prompt-response |
| value model $V_\psi$ | да | предсказывает return и уменьшает variance |

Old и reference иногда численно совпадают в самом начале, но это не делает их
одной ролью. После optimizer update current отходит от old; после многих batches
old обновляется, тогда как reference может всё ещё быть исходной SFT policy.

> [!warning] Два разных ограничителя
> PPO clipping ограничивает один update относительно **old policy**. KL penalty
> ограничивает накопленный drift относительно **reference policy**. Один не
> является приближённым названием другого.

## 6. Почему vanilla policy gradient недостаточен

Rollout дорог: нужно авторегрессионно породить ответы и прогнать их через reward,
reference и value networks. Хотелось бы сделать несколько minibatch epochs по
одним samples. Но после первого update данные уже не on-policy.

Importance ratio корректирует likelihood выбранного action:

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
     {\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

Unclipped surrogate $\rho_tA_t$ может породить слишком большой update: optimizer
увидит удачный sample и многократно увеличит его probability. PPO-Clip ограничивает
выгоду от ухода ratio за доверительный интервал:

$$
L^{\mathrm{clip}}=
\mathbb E_t\left[
\min\left(
\rho_tA_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)\right].
$$

### PPO clipping руками

Пусть $\epsilon=0.2$.

**Хорошее действие:** $A=2$, old probability $0.20$, current $0.30$.

$$
\rho=0.30/0.20=1.5,
$$

$$
\rho A=3.0,\qquad
\operatorname{clip}(1.5,0.8,1.2)A=2.4.
$$

Берём minimum: $2.4$. Дальнейшее увеличение probability не улучшает surrogate.

**Плохое действие:** $A=-2$, old probability $0.20$, current $0.10$.

$$
\rho=0.5,quad \rho A=-1.0,quad 0.8A=-1.6.
$$

Minimum равен $-1.6$: objective не награждает чрезмерное уменьшение probability.
Знак advantage принципиален; нельзя объяснять clip только верхней границей.

### Piecewise-интуиция

| advantage | полезное направление | где возникает plateau |
|---|---|---|
| $A>0$ | повышать probability | $\rho>1+\epsilon$ |
| $A<0$ | понижать probability | $\rho<1-\epsilon$ |

PPO не доказывает строгий trust region. Это practically useful surrogate,
который обычно сочетают с monitoring approximate KL, early stopping и gradient
clipping.

## 7. Полный objective

Типичный loss реализации минимизируется:

$$
\mathcal L=
-L^{\mathrm{clip}}
+c_v\mathcal L_V
-c_H\mathcal H(\pi_\theta),
$$

где

$$
\mathcal L_V=(V_\psi(s_t)-\hat R_t)^2
$$

обучает critic, а entropy bonus препятствует слишком раннему collapse. KL к
reference может быть частью shaped reward или отдельным penalty.

InstructGPT дополнительно смешивал pretraining gradient:

$$
J_{\mathrm{PPO\text{-}ptx}}
=\mathbb E[r_\phi-\beta\log(\pi_\theta/\pi_{\mathrm{SFT}})]
+\gamma\mathbb E_{z\sim D_{\mathrm{pretrain}}}\log\pi_\theta(z).
$$

Это было отдельным средством против regressions. В их ablations простое
увеличение KL не полностью восстанавливало public-NLP capabilities.

## 8. Tensor и data flow одного batch

Пусть batch содержит $B$ prompts, максимальная длина response $T$, vocabulary
$V$.

```text
prompts [B, P]
   │ generate with π_old
   ▼
responses [B, T] + response_mask [B, T]
   ├─ π_old       → old_logprobs [B, T]       (сохранены при rollout)
   ├─ π_ref       → ref_logprobs [B, T]       (без gradient)
   ├─ reward RM   → terminal_scores [B]
   └─ value model → values [B, T]
                         │
KL-shaped rewards [B,T] + terminal score
                         │ GAE
returns [B,T], advantages [B,T]
                         │ minibatch epochs
π_current → new_logprobs [B,T]
ratio = exp(new_logprobs-old_logprobs)
                         │
policy loss + value loss + metrics
```

`logits [B,T,V]` обычно не хранят для всех сетей одновременно: из них сразу
gather'ят log-probability фактически sampled tokens. Prompt и padding positions
маскируются. Нормализация loss по token count, sequence count или prompt group
даёт разные веса длинным ответам и должна быть явной.

## 9. Один PPO iteration

```python
with no_grad():
    response, old_logp = generate(policy_old, prompts)
    ref_logp = token_logprobs(reference, prompts, response)
    score = reward_model(prompts, response)       # [B]
    old_value = value_model(prompts, response)    # [B, T]

reward = -beta * (old_logp - ref_logp)
reward[:, last_token] += score
advantage, returns = gae(reward, old_value, mask)

for minibatch in repeat_k_epochs(batch):
    new_logp = token_logprobs(policy, ...)
    ratio = exp(new_logp - old_logp)
    unclipped = ratio * advantage
    clipped = clamp(ratio, 1-eps, 1+eps) * advantage
    policy_loss = -masked_mean(min(unclipped, clipped))

    new_value = value_model(...)
    value_loss = masked_mean((new_value - returns) ** 2)
    step(policy_loss + c_v * value_loss)

policy_old.load_state_dict(policy.state_dict())
```

Это pedagogical pseudocode: production systems отдельно решают rollout
parallelism, sequence packing, value clipping, whitened advantages, adaptive KL,
distributed inference/training и stale rollouts.

## 10. Где PPO ломается

### Reward hacking

Policy ищет не человеческое намерение, а ответы с высоким score RM. Длина,
уверенный тон, повторение rubric-слов или странный formatting могут стать
shortcut. Главная защита — не только KL: нужны held-out human evaluation,
adversarial probes и обновление reward data около текущей policy.

### Distribution shift reward model

RM обучен на ответах прежней policy. PPO намеренно ищет новые ответы, поэтому
может выйти в область, где reward extrapolates плохо. Reference KL уменьшает,
но не устраняет эту проблему.

### Value collapse и noisy advantages

Плохой critic даёт неправильные signs advantage. Следят за explained variance,
value loss, scale returns и долей clipped value updates.

### Mode collapse и entropy loss

Слишком сильная оптимизация уменьшает diversity. Высокий reward при падающей
entropy и ухудшении human win-rate — тревожный признак.

### Length bias

Sequence reward и сумма per-token KL масштабируются с длиной по-разному. Нужно
проверять reward/quality conditional on length и явно задавать truncation penalty.

### Off-policy/staleness

Если generation workers сильно отстают от trainer, denominator уже не отражает
актуальную rollout policy. Ratio и approximate KL обнаруживают проблему, но
архитектура должна ограничивать staleness.

## 11. Что измерять

- task reward и отдельно каждый reward component;
- human/strong-judge win-rate на held-out prompts;
- KL к reference и KL current–old;
- clip fraction;
- entropy и response length distribution;
- mean/std advantages и explained variance critic;
- reward-model accuracy на свежих policy samples;
- capability/safety regressions вне training distribution;
- reward–quality divergence: растёт proxy, но не независимая оценка.

## 12. Практика

1. Для probabilities `A=0.6`, `B=0.4` и rewards `1`, `−1` вычислите $J$ и
   REINFORCE estimator для каждого sampled action.
2. Докажите, что вычитание baseline, зависящего только от state, не меняет
   expected gradient. Почему action-dependent baseline уже опасен?
3. При $A=-1.5$, $\epsilon=0.1$ посчитайте clipped objective для ratios
   `0.7`, `0.95`, `1.2`.
4. Нарисуйте отдельно timeline обновления current, old и reference policies.
5. Реализуйте двухдействийный bandit: сравните variance REINFORCE с baseline и
   без него.
6. Возьмите десять ответов разной длины. Проверьте, меняется ли RM score после
   добавления корректного, но ненужного абзаца.

## 13. Источники и хорошие объяснения

### Первоисточники

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Stiennon et al., [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325).
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).

### Курсы и реализации

- OpenAI, [Spinning Up: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — лучший короткий разбор clipped objective.
- Nathan Lambert, [RLHF Book](https://rlhfbook.com/) — reward modeling и RLHF именно для language models.
- Stanford CS224N, *Life after DPO* — локальные slides:
  `raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture15-life-after-dpo-lambert.pdf`.
- Hugging Face, [TRL PPO Trainer](https://huggingface.co/docs/trl/ppo_trainer) — реализация; API следует сверять с текущей версией.

> [!summary] Главное
> Policy gradient делает sampled tokens вероятнее или менее вероятными согласно
> advantage. PPO позволяет несколько осторожных updates по дорогому rollout
> batch. Old policy нужна для локального importance ratio; reference policy —
> для долговременного KL-якоря; reward и value models решают разные задачи.
