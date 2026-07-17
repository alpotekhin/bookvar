---
title: Direct Preference Optimization
type: textbook-chapter
status: canonical
last_updated: 2026-07-17
aliases:
  - DPO
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM]]"
next:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers]]"
primary_sources:
  - https://arxiv.org/abs/2305.18290
---

# Direct Preference Optimization

> [!abstract] После этой главы
> Можно будет вывести DPO из KL-регуляризованного RLHF objective, объяснить,
> почему исчезает partition function, вручную посчитать DPO logit и gradient
> weight, собрать tensor pipeline и ясно назвать то, чего DPO **не** делает:
> стандартный DPO не генерирует новые rollouts во время обучения и не отменяет
> reference policy.

## 1. Какую часть RLHF упрощает DPO

Предположим, для prompt $x$ есть два ответа:

$$
y_w \succ y_l,
$$

где $y_w$ выбран человеком или judge, а $y_l$ отвергнут. Классический pipeline:

1. обучить scalar reward model на таких comparisons;
2. генерировать новые ответы current policy;
3. оценивать их reward model;
4. оптимизировать policy через PPO с KL к reference.

DPO спрашивает: если reward нужен только для получения оптимальной policy,
можно ли выразить preference probability сразу через policy? При определённых
предпосылках — да. Получается supervised-looking binary classification loss,
который обучает policy на fixed preference pairs.

![[02 Areas/ML & DL/raw/papers/dpo/images/dpo-pipeline.png]]

*DPO paper, Figure 1. Слева explicit reward model и RL; справа preference loss
непосредственно над policy. Упрощается training loop, но данные предпочтений и
предположения о latent reward никуда не исчезают.*

## 2. Сначала — вероятность целого ответа

Для autoregressive LM:

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T_y}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

DPO сравнивает **sequence log-probabilities**, а не probability последнего
токена и не среднюю уверенность отдельных слов. Это важно для реализации:
после forward pass `logits [B,L,V]` выбираются log-probs target tokens,
prompt/padding маскируются, оставшиеся значения суммируются по response.

У суммы есть следствие: длина влияет на scale. Более длинная последовательность
обычно имеет более отрицательный log-likelihood. Reference correction многое
компенсирует, но length bias preference data и aggregation всё равно требуют
отдельной диагностики.

## 3. Reward modeling как отправная точка

Bradley–Terry model задаёт:

$$
P(y_w\succ y_l\mid x)
=\sigma(r(x,y_w)-r(x,y_l)),
$$

где $\sigma(z)=1/(1+e^{-z})$. Обычный RM loss:

$$
\mathcal L_{RM}
=-\mathbb E\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
$$

Preferences идентифицируют лишь разность rewards. Если к обоим rewards одного
prompt прибавить $C(x)$, probability не изменится. Это свойство позволит
сократить неизвестный partition function.

## 4. KL-регуляризованная задача RLHF

Рассмотрим policy objective для каждого prompt:

$$
\max_\pi
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{KL}(\pi(\cdot\mid x)\Vert\pi_{ref}(\cdot\mid x)).
$$

$\pi_{ref}$ — обычно SFT policy. $\beta>0$ задаёт цену отклонения. Распишем:

$$
J(\pi)=\sum_y\pi(y\mid x)
\left[r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}\right]
$$

при ограничении $\sum_y\pi(y\mid x)=1$. Добавим Lagrange multiplier $\lambda$:

$$
\mathcal F=J(\pi)+\lambda\left(\sum_y\pi(y\mid x)-1\right).
$$

Производная по $\pi(y\mid x)$:

$$
r(x,y)-\beta\left(
\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}+1
\right)+\lambda=0.
$$

Перегруппировка и нормировка дают Boltzmann policy:

$$
\pi_r(y\mid x)=
\frac{1}{Z(x)}\pi_{ref}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
$$

$$
Z(x)=\sum_y\pi_{ref}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

Это ключевой мост: оптимальная policy — reference, экспоненциально
перевзвешенная reward.

## 5. Выражаем implicit reward через policy

Берём логарифм:

$$
\log\pi_r(y\mid x)
=\log\pi_{ref}(y\mid x)+\frac{r(x,y)}{\beta}-\log Z(x).
$$

Отсюда

$$
r(x,y)=
\beta\log\frac{\pi_r(y\mid x)}{\pi_{ref}(y\mid x)}
+\beta\log Z(x).
$$

Для пары одного prompt:

$$
\begin{aligned}
r(x,y_w)-r(x,y_l)
=\beta\bigg[
&\log\frac{\pi_r(y_w\mid x)}{\pi_{ref}(y_w\mid x)}\\
-&\log\frac{\pi_r(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\bigg].
\end{aligned}
$$

$\beta\log Z(x)$ одинаков для обоих completions и **сокращается**. DPO не
оценивает огромную сумму по всем возможным текстам; pairwise model делает её
ненужной.

Подставляем параметризованную policy $\pi_\theta$ в Bradley–Terry likelihood:

$$
\boxed{
\mathcal L_{DPO}
=-\mathbb E_{(x,y_w,y_l)}
\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\right)}.
$$

## 6. Что именно сравнивает DPO

Введём relative log-probability:

$$
\Delta_\theta(x,y)=
\log\pi_\theta(y\mid x)-\log\pi_{ref}(y\mid x).
$$

Тогда DPO logit:

$$
z=\beta[\Delta_\theta(x,y_w)-\Delta_\theta(x,y_l)].
$$

Loss требует не просто $\pi_\theta(y_w)>\pi_\theta(y_l)$, а чтобы current policy
предпочитала chosen **сильнее, чем reference**.

### Ручной пример

Пусть sequence log-probabilities:

| | chosen $y_w$ | rejected $y_l$ |
|---|---:|---:|
| current $\log\pi_\theta$ | −3.0 | −4.0 |
| reference $\log\pi_{ref}$ | −3.4 | −3.6 |

Тогда

$$
\Delta_w=-3.0-(-3.4)=0.4,
\qquad
\Delta_l=-4.0-(-3.6)=-0.4.
$$

При $\beta=0.5$:

$$
z=0.5(0.4-(-0.4))=0.4,
$$

$$
P_\theta(y_w\succ y_l)=\sigma(0.4)\approx0.599,
$$

$$
L=-\log0.599\approx0.513.
$$

Если current повысит log-probability обоих ответов относительно reference на
одинаковые `+0.4`, margin будет нулевым: DPO не увидит улучшения preference.

### Что делает gradient

Implicit reward:

$$
\hat r_\theta(x,y)=
\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

Gradient DPO paper:

$$
\nabla_\theta L_{DPO}
=-\beta\,\mathbb E\left[
\sigma(\hat r_l-\hat r_w)
\left(
\nabla\log\pi_\theta(y_w\mid x)
-\nabla\log\pi_\theta(y_l\mid x)
\right)
\right].
$$

Обновление:

- повышает log-likelihood chosen;
- понижает log-likelihood rejected;
- сильнее весит pair, который implicit reward пока ранжирует неправильно.

В ручном примере weight $\sigma(-0.4)\approx0.401$. Если модель перепутает
ответы и получит $z=-2$, weight станет $\sigma(2)\approx0.881$. Это не
декоративная деталь: авторы показывают, что naïve unweighted variant может
дегенерировать.

## 7. Что означает $\beta$

В исходном RL objective $\beta$ — сила KL regularization. В DPO он масштабирует
preference logit. Полезная operational-интуиция:

- при малом $\beta$ один и тот же log-ratio margin даёт менее насыщенный sigmoid
  и gradients дольше остаются крупными;
- при большом $\beta$ moderate margins быстрее насыщают classifier;
- численное значение нельзя сравнивать без учёта aggregation, длины, dataset и
  normalization реализации.

Называть $\beta$ learning rate неверно: learning rate меняет размер optimizer
step, $\beta$ меняет сам statistical objective.

## 8. Data и tensor flow

Preference record:

```json
{
  "prompt": "Объясни TCP и UDP...",
  "chosen": "TCP обеспечивает...",
  "rejected": "UDP всегда быстрее..."
}
```

После chat template обычно формируются две последовательности с общим prompt.

```text
prompt + chosen   → policy logits [B,Lw,V] → gather → sum response → π_chosen [B]
prompt + rejected → policy logits [B,Ll,V] → gather → sum response → π_reject [B]
                                           │
те же sequences → frozen reference → ref_chosen [B], ref_reject [B]
                                           │
chosen_logratio  = π_chosen - ref_chosen
reject_logratio  = π_reject - ref_reject
logit             = β(chosen_logratio - reject_logratio)
loss              = -logsigmoid(logit)
```

Практически chosen/rejected часто объединяют в один concatenated forward для
policy и один для reference. Это экономит padding/communication overhead.

```python
def sequence_logp(logits, labels, response_mask):
    token_logp = log_softmax(logits[:, :-1], -1).gather(
        -1, labels[:, 1:, None]
    ).squeeze(-1)
    return (token_logp * response_mask[:, 1:]).sum(-1)

pi_w = sequence_logp(policy(chosen), chosen_ids, chosen_mask)
pi_l = sequence_logp(policy(rejected), rejected_ids, rejected_mask)

with no_grad():
    ref_w = sequence_logp(reference(chosen), chosen_ids, chosen_mask)
    ref_l = sequence_logp(reference(rejected), rejected_ids, rejected_mask)

logit = beta * ((pi_w - ref_w) - (pi_l - ref_l))
loss = -logsigmoid(logit).mean()
```

Критические детали:

- prompt tokens не входят в response sequence log-probability;
- special assistant prefix должен обрабатываться одинаково;
- padding и truncation не должны менять target незаметно;
- reference log-probs можно предварительно вычислить для fixed dataset;
- sum и mean token log-probs — разные objectives; paper использует sequence
  log-probability, то есть сумму.

## 9. Откуда берётся reference policy

В стандартном recipe:

1. pretrained/base model проходит SFT;
2. SFT checkpoint копируется;
3. одна копия становится trainable DPO policy;
4. другая frozen reference.

Если preference dataset сгенерирован другой policy, возникает distribution
shift. В DPO paper при отсутствии исходной SFT policy сначала делают preferred
fine-tuning на $(x,y_w)$ и используют полученную policy как reference/init.

Reference нужна и в loss, и в интерпретации implicit reward. Фразы «DPO без
reference model» относятся к отдельным поздним variants, но не к standard DPO.

## 10. DPO и PPO: одинаковая отправная точка, разный training regime

| | PPO RLHF | DPO |
|---|---|---|
| входной feedback | RM reward для rollout | fixed chosen/rejected pairs |
| generation во время training | да | нет |
| отдельный reward model | да | нет |
| reference policy | да | да |
| value/critic | обычно да | нет |
| exploration current policy | да | нет |
| основной риск | reward hacking/on-policy instability | coverage/noise/overfitting pairs |

DPO paper выводит loss из того же KL-regularized reward maximization. Но
практические алгоритмы не становятся эквивалентны во всём: offline data не
следует за меняющейся policy, assumptions Bradley–Terry могут быть нарушены,
а PPO способен исследовать ответы, отсутствующие в pairs.

## 11. Какие данные нужны

Качество DPO определяется не только правильностью label:

- prompts должны соответствовать deployment distribution;
- candidates должны быть достаточно сильными и различимыми;
- trivial pairs быстро насыщают loss и мало учат;
- near-ties нельзя насильно превращать в уверенный binary label;
- positional, verbosity и style biases надо измерять;
- пары от далёкой generator policy создают support mismatch;
- дубликаты одного prompt нельзя бездумно считать независимыми;
- train/test leakage через judge или benchmark answers остаётся leakage.

Полезно хранить provenance, generator checkpoint, decoding parameters,
annotator/judge, confidence и rubric. Binary JSON без этих полей плохо
поддерживает последующий аудит.

## 12. Failure modes

### Вероятность rejected обрушивается быстрее, чем растёт chosen

Preference accuracy может улучшаться за счёт сильного снижения rejected, тогда
как chosen likelihood тоже падает. Поэтому следят отдельно за
`chosen_rewards`, `rejected_rewards`, margins и raw sequence log-probabilities.

### Length bias

Annotators и judges могут предпочитать подробность. Кроме того, sequence sums
зависят от числа токенов. Нужны length-stratified evaluation, length-matched
pairs и проверка качества после контролируемого сокращения/расширения ответа.

### Label noise и non-transitivity

Bradley–Terry предполагает latent scalar utility. Реальные preferences могут
быть контекстны и цикличны: A яснее B, B точнее C, C безопаснее A. Один scalar
не обязан согласовать все критерии.

### Saturation

Когда margins велики, sigmoid weight близок к нулю и pair почти перестаёт
обучать. Это ожидаемое поведение classifier, а не доказательство достижения
абсолютно хорошей policy.

### Overoptimization fixed dataset

Offline policy может запомнить superficial separators chosen/rejected. Проверка
на held-out prompts и свежих samples обязательна.

### Chat-template mismatch

Если reference и policy видят разные role tokens, log-ratio теряет смысл.
Tokenizer, template, truncation и EOS policy должны совпадать.

### Evaluation judge повторяет bias training judge

Высокий win-rate у того же judge не является независимой проверкой. Нужны human
sample, другой judge family, task verifiers и style-controlled comparisons.

## 13. Диагностика обучения

Минимальный dashboard:

- DPO loss и preference accuracy;
- mean $\hat r_w$, $\hat r_l$ и margin;
- raw current/reference log-probs отдельно;
- chosen/rejected response lengths;
- KL current–reference на свежих generations;
- held-out pair accuracy;
- independent win-rate и task correctness;
- diversity, verbosity, refusals и format compliance;
- stratification по source, domain, language, length и annotator confidence.

> [!warning]
> Training pair accuracy может приблизиться к 100%, а модель стать хуже для
> пользователей. Это всего лишь способность воспроизвести ordering fixed pairs.

## 14. Варианты — после понимания standard DPO

- **IPO** меняет statistical treatment preferences и стремится избежать части
  overfitting behavior DPO.
- **ORPO** совмещает SFT-like likelihood и odds-ratio preference objective без
  отдельной reference model.
- **SimPO** использует length-normalized reward и reference-free margin.
- **KTO** работает с desirable/undesirable examples без явных pairs.
- **Online DPO** обновляет preference data samples текущей policy.

Их не стоит изучать как список аббревиатур. Для каждого надо задать четыре
вопроса: какой data schema, какой implicit preference model, есть ли reference,
и откуда берутся candidates во время training.

## 15. Практика

1. Воспроизведите ручной пример при $\beta=0.1$, $1$ и $5$. Постройте sigmoid
   weight как функцию margin.
2. Докажите сокращение $Z(x)$; объясните, почему нельзя сравнить два ответа от
   разных prompts тем же шагом.
3. Для пары посчитайте DPO loss, если current и reference совпадают. Какова
   preference probability и почему gradient всё равно ненулевой?
4. Реализуйте `sequence_logp` и тесты: padding не влияет на сумму; prompt mask
   равен нулю; EOS учитывается ровно один раз.
5. Создайте synthetic dataset, где chosen всегда длиннее. Обучите маленькую
   model/classifier и измерьте, стал ли length shortcut.
6. Возьмите один chosen и три rejected: фактическая ошибка, плохой стиль и
   нарушение формата. Проверьте, переносится ли learned preference между типами.
7. Сравните DPO и SFT только на chosen: какие gradients получает rejected и
   почему результаты различаются?

## 16. Проверка понимания

Вопросы к этой главе вынесены в единый список
[[02 Areas/ML & DL/04 Вопросы/Вопросы по LLM#Post-training и alignment|«Вопросы по LLM»]].
Там каждый вопрос ведёт обратно к конкретному объяснению, а не дублирует текст.

## 17. Источники и объяснения

### Первоисточник

- Rafailov et al., [Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model](https://arxiv.org/abs/2305.18290). Особенно Figure 1,
  Equations 1–7, gradient после Eq. 7, Figure 2 и Appendix A.

### Курсы и код

- Stanford CS224N, Nathan Lambert, *Life after DPO* — локальные slides:
  `raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture15-life-after-dpo-lambert.pdf`.
- Nathan Lambert, [RLHF Book](https://rlhfbook.com/) — preference data,
  reward models и direct alignment algorithms.
- Hugging Face, [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer).
- Hugging Face, [Alignment Handbook](https://github.com/huggingface/alignment-handbook) — воспроизводимые SFT/DPO recipes.

> [!summary] Главное
> DPO заменяет explicit reward-model-plus-PPO loop на likelihood preference
> pairs через аналитическую связь reward и optimal KL-regularized policy.
> Partition function исчезает только потому, что сравниваются два ответа одного
> prompt. Метод проще operationally, но остаётся зависимым от reference,
> Bradley–Terry assumptions и coverage offline preference data.
