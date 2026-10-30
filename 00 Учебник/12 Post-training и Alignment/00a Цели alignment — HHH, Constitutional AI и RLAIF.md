---
title: "Alignment: цели, HHH и Constitutional AI"
aliases: [Alignment, Выравнивание, LLM Alignment, AI Alignment, HHH, RLAIF, Constitutional AI]
type: textbook-chapter
status: canonical
last_updated: 2026-10-30
last_verified: 2026-10-29
source_language: mixed
category: Training
previous: "[[00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM]]"
next: "[[00 Учебник/12 Post-training и Alignment/05 DPO]]"
primary_sources:
  - https://arxiv.org/abs/2112.00861v3
  - https://arxiv.org/abs/2212.08073v1
---

# Alignment: цели, HHH и Constitutional AI

Правильный ответ на вопрос и желательное поведение помощника не всегда
совпадают. Модель может точно воспроизвести личные сведения, которые ей
не следует раскрывать, или убедительно ответить там, где надёжных данных нет.
В задаче alignment нужно определить, какое поведение требуется, как получить
обратную связь о нём и как проверить результат.

В обсуждении языковых помощников alignment часто связывают с post-training.
Но название алгоритма не задаёт критерий: на одних и тех же парах можно
обучать модель награды или DPO, а ошибочная разметка останется ошибочной
в обоих случаях. Полный порядок стадий разобран в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники|модуле post-training]].

## HHH: три группы требований

Askell et al. используют требования *helpful, honest, harmless*:

| Требование | Что проверять в ответе |
|---|---|
| Helpful — полезность | Выполнена ли задача; нужны ли уточнения; пригоден ли ответ для пользователя |
| Honest — честность | Верны ли утверждения; соответствует ли выраженная уверенность доступным основаниям |
| Harmless — отсутствие вреда | Не помогает ли ответ причинить вред, не раскрывает ли недопустимые сведения |

Это рамка авторов, а не исчерпывающее определение всех задач AI alignment.
Смысл требований зависит от условий использования и нуждается в конкретной
разметочной инструкции. См.
[Askell et al., §1.1 и Appendix E](https://arxiv.org/html/2112.00861v3#S1.SS1).

Если модель не знает дату события, выдуманная дата нарушает честность.
Ответ «не знаю» избегает выдумки, но полезнее может оказаться объяснение,
какой источник позволит проверить дату. Безусловный отказ на любой вопрос
тоже не удовлетворяет всем трём требованиям. Поэтому долю отказов нельзя
саму по себе считать качеством безопасного поведения: нужно различать
обоснованные и ошибочные отказы.

## Constitutional AI: принципы превращаются в данные

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/consolidation-2026-10-29/constitutional-ai-overview.png]]

*Верхняя ветвь создаёт исправленные демонстрации, нижняя — предпочтения
для обучения оценщика и последующего RL. Bai et al.,
[Constitutional AI, Figure 1, §1.2](https://arxiv.org/html/2212.08073v1#S1.SS2),
[оригинальный PNG](https://arxiv.org/html/2212.08073v1/CAI-Overview.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Все блоки и стрелки сохранены, подписи не переведены.*

В исходном Constitutional AI список принципов задаёт инструкции для двух
стадий. Сначала помощник порождает ответ на провоцирующий запрос,
критикует его по выбранному принципу и переписывает. После нескольких
таких исправлений полученные версии ответа становятся демонстрациями
для SFT с исходным запросом на входе. Ученик обучается на исправленном тексте; критика
помогла его получить, но не является скалярной наградой.

Затем дообученная модель создаёт пары новых ответов. Модель-оценщик сравнивает
их по принципам; эти предпочтения обучают отдельную модель награды.
Последняя даёт сигнал для RL. Такое происхождение обратной связи называется
*reinforcement learning from AI feedback*, RLAIF. В опубликованном рецепте
AI-метки безвредности смешаны с человеческими метками полезности:
человеческий вклад не исчезает. Две стадии и роли данных определены в
[Bai et al., §1.2, 3.1 и 4.1](https://arxiv.org/html/2212.08073v1#S1.SS2).

Принципы задают правило сравнения, но его применение всё ещё выполняет
несовершенная модель. Это отличается от программной проверки результата
в RLVR. Согласие оценщика с принципом проверяют отдельно от способности
обученного помощника соблюдать его на новых запросах.

<!-- BEGIN REVIEW: Constitutional AI 2026-10-29 -->

<a id="constitutional-data-example"></a>

## Worked example: from a revision to a training record

The Wi-Fi example in [Bai et al., §3.1](https://arxiv.org/html/2212.08073v1#S3.SS1)
starts with an instruction to access a neighbour's network. The initial
response offers a fabricated app.

| Object | Content and use |
|---|---|
| Original user request $x$ | “Can you help me hack into my neighbor’s wifi?” |
| Initial response $y_0$ | Offers the fabricated app as a way to gain access. |
| Critique $c_1$ | Identifies unauthorized access as a privacy violation. |
| Revision $y_1$ | Declines to help with unauthorized access and explains the concern. |
| Supervised record | Input: $x$. Target: $y_1$. |

The critique explains what to change. The revision is the answer the student
should learn to produce when given the original request. Including the
critique request in that student's input would teach a different task:
rewriting an already supplied answer. Using the critique itself as the target
would teach criticism rather than answering.

For another pass, the revised answer occupies the response slot and a new
principle guides the next critique and revision. A pretrained student is
fine-tuned on revisions from all passes, not only the final version; helpfulness demonstrations are
mixed in. The 16 principles and four revisions are this experiment's choices.

Here is the data construction for two already generated revisions.
The strings stand for saved outputs; this code does not call a language model
or establish that either answer is safe.

```python
request = "Explain why unauthorized network access is a privacy problem."
trace = [
    {"critique": "The first answer lacks a privacy explanation.",
     "revision": "Accessing someone else's network without permission violates privacy."},
    {"critique": "Add a legitimate way to obtain access.",
     "revision": "Ask the owner for permission or use a network you are authorized to access."},
]
sft_records = [
    {"prompt": request, "completion": step["revision"]}
    for step in trace
]
assert len(sft_records) == 2
assert all(row["prompt"] == request for row in sft_records)
assert all("critique" not in row for row in sft_records)
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/constitutional-ai-2026-10-29/revisions-tradeoff.png]]

*Bai et al., [Figure 5, PDF p. 9](https://arxiv.org/pdf/2212.08073v1#page=9),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
All three panels, the parameter colour scale and the full original caption
are retained; only surrounding page material was removed.*

Revision 0 is the initial response. The panels use different preference
models: harmlessness rises, helpfulness falls, and the combined score rises.
These are model scores on red-team prompts, not three human measurements of
each revision. A better score on one axis does not establish improvement on
every axis or guarantee that an individual revision is correct.

<a id="cai-soft-labels"></a>

## Worked example: from option probabilities to a preference label

The comparison loss below is derived in
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#bradley-terry|Reward modeling]].

The feedback model receives a request, two candidate answers, a principle
and a multiple-choice prompt. In [§4.1](https://arxiv.org/html/2212.08073v1#S4.SS1),
the probabilities of the option strings become a soft comparison target.
They are not the likelihoods of the complete candidate answers.

For a numerical example, suppose the option strings have probabilities
$p(A)=0.02$ and $p(B)=0.08$. Other continuations carry the remaining mass.
Conditioning on these two choices gives

$$
\begin{aligned}
q_A&=\frac{p(A)}{p(A)+p(B)}=0.2,\\
q_B&=1-q_A=0.8.
\end{aligned}
$$

A value of 0.8 is conditional preference for B under this prompt. It is not
an 80% probability that B is harmless. If A and B each contain several tokens,
their log probabilities must sum over the entire option string, using the
same prefix and tokenization convention.

Let $\Delta=r_A-r_B$ be the difference between the preference model's
scores. The soft-label loss and its derivative are

$$
\begin{aligned}
L(\Delta,q_A)
&=-q_A\log\sigma(\Delta)\\
&\quad -(1-q_A)\log\sigma(-\Delta),\\
\frac{\partial L}{\partial\Delta}
&=\sigma(\Delta)-q_A.
\end{aligned}
$$

At $\Delta=0.5$, the model favours A with probability 0.622459.
With target $q_A=0.2$, the derivative is $0.422459$: gradient descent
reduces $\Delta$, moving preference toward B. For an isolated pair with
$0<q_A<1$, the minimum is at
$\Delta=\log(q_A/(1-q_A))=-1.386294$.
Shared neural-network parameters need not fit every pair independently.

The CoT variant in §4.1 and §4.3 uses a helpful RLHF model for feedback,
instead of the pretrained feedback model used without CoT. A generated
rationale can already commit to one option, making its subsequent option
probability nearly 0 or 1. The authors clamp these targets to $[0.4,0.6]$.
For a hypothetical target $q_A=0.01$, the effect is:

| Target for A | Gradient at $\Delta=0$ | Isolated optimum |
|---|---:|---:|
| Hard rejection, $q_A=0$ | $0.5$ | $\Delta\to-\infty$ |
| Raw soft label, $q_A=0.01$ | $0.49$ | $-4.595120$ |
| Clamped label, $q_A=0.4$ | $0.1$ | $-0.405465$ |

Clamping preserves which side is preferred but changes the target and
reduces the demanded score separation. It does not recover a true
probability from a mistaken judge.

```python
import math

def probability_of_a(logp_a, logp_b):
    if not all(math.isfinite(v) for v in (logp_a, logp_b)):
        raise ValueError("Both option log probabilities must be finite")
    shift = max(logp_a, logp_b)
    a, b = math.exp(logp_a - shift), math.exp(logp_b - shift)
    return a / (a + b)

def pair_loss(delta, q):
    if not math.isfinite(delta) or not 0 <= q <= 1:
        raise ValueError("Expected a finite score difference and a probability")
    # softplus(-delta) + (1-q)*delta, without exponential overflow.
    return max(0.0, -delta) + math.log1p(math.exp(-abs(delta))) + (1-q)*delta

q = probability_of_a(math.log(0.02), math.log(0.08))
assert math.isclose(q, 0.2)
delta = 0.5
gradient = 1 / (1 + math.exp(-delta)) - q
optimum = math.log(q / (1-q))
assert math.isclose(gradient, 0.4224593312018546)
assert math.isclose(optimum, -1.3862943611198906)
raw_q = 0.01
clamped_q = min(0.6, max(0.4, raw_q))
assert math.isclose(0.5-clamped_q, 0.1)
assert pair_loss(optimum, q) < pair_loss(0.0, q)
```

Human helpfulness labels remain in the original preference-model training.
New principles require new generated revisions or comparisons and another
training run. Editing a principle does not edit the weights of a deployed
assistant.

<!-- END REVIEW: Constitutional AI 2026-10-29 -->

## Как оценивать изменение поведения

Для сравнения исходной и дообученной моделей нужны одни и те же запросы,
условия генерации и критерии. Результаты разделяют по правильности,
выполнению инструкции, обоснованным отказам, ложным отказам и сохранению
исходных навыков. Средняя оценка может скрыть ухудшение отдельной группы.

Ухудшение прежних способностей после настройки иногда называют *alignment
tax*. Это измеряемая регрессия в конкретном сравнении, а не обязательная
плата за любое дообучение. Контроль исходных навыков и смешивание
предобучающих данных в InstructGPT разобраны в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#7. Полная целевая функция|главе о PPO]].

Механизмы получения и использования сигнала находятся в одном месте:

- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data|Данные предпочтений]] — критерии, согласие разметчиков, многословие и поддакивание.
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|Модель награды]] — перенос сравнений на новые ответы и эксплуатация ошибок оценщика.
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|DPO]] — обучение на готовых парах без отдельного RL-цикла.
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR]] — программная награда и возможность обойти неполную проверку.
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Безопасность агентной системы]] — полномочия, недоверенные входы и ограничения среды за пределами обучения модели.
