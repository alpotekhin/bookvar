---
title: "HHH, Constitutional AI и RLAIF"
aliases: [Alignment, Выравнивание, LLM Alignment, AI Alignment, HHH, RLAIF, Constitutional AI]
type: textbook-chapter
status: canonical
last_updated: 2026-11-22
last_verified: 2026-11-22
source_language: mixed
category: Training
previous: "[[00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM]]"
next: "[[00 Учебник/12 Post-training и Alignment/05 DPO]]"
primary_sources:
  - https://arxiv.org/abs/2112.00861v3
  - https://arxiv.org/abs/2212.08073v1
  - https://arxiv.org/abs/2309.00267v2
---

# HHH, Constitutional AI и RLAIF

Помощника просят сообщить домашний адрес сотрудника. По правилам сервиса
нельзя выдавать частные адреса без разрешения. Ответ «Не знаю» тоже
может ввести в заблуждение: отсутствие разрешения и отсутствие сведений
не одно и то же. Помощник может объяснить, почему не выдаёт адрес, и
предложить публичный способ связаться с сотрудником.

В [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|главе о PPO]]
модель награды оценивает ответы по тому, чему научилась на сравнениях.
Если в обучающих парах короткие отказы всегда выигрывали у развёрнутых
ответов, помощник может научиться отказывать даже тогда, когда полезный
ответ разрешён. Чтобы избежать этого, нужно указать, в каких случаях
отказ уместен, и включить в данные примеры по обе стороны этой границы.

<span id="hhh-три-группы-требований"></span>

## Критерии HHH

[Askell et al., §1.1 и приложение E](https://arxiv.org/html/2112.00861v3#A5)
различают три требования к языковому помощнику: *helpful, honest, harmless*.
Они помогают описать желательное поведение, но сами по себе не задают
функцию потерь или универсальный порядок приоритетов.

| Критерий | Что значит для запроса об адресе |
|---|---|
| Helpful: полезность | Помочь связаться с сотрудником, например указать публичный рабочий канал, если он известен. |
| Honest: честность | Не выдумывать адрес или разрешение и не утверждать, что адреса нет, если это неизвестно. |
| Harmless: безвредность | Не раскрывать непубличный адрес третьему лицу без разрешения. |

Для разметки этих ответов «будь полезен» слишком неопределённо: интерес
пользователя, который хочет связаться с сотрудником, не отменяет права
сотрудника на приватность. В инструкции разметчику можно сперва
запретить раскрытие непубличного адреса, а между двумя ответами без
разглашения предпочесть тот, который действительно помогает связаться
с человеком. На просьбу дать опубликованный адрес организации та же
инструкция допускает прямой ответ. Такое правило даёт разметчику способ
выбрать между ответами в обоих случаях.

<span id="constitutional-ai-принципы-превращаются-в-данные"></span>

## Constitutional AI

В [Constitutional AI, §1.2](https://arxiv.org/html/2212.08073v1#S1.SS2)
Bai et al. дают модели принципы, записанные обычным языком. Принцип
используют дважды, но с разными результатами. Сначала модель составляет
исправленный ответ, на котором можно обучать помощника методом SFT.
Затем оценщик сравнивает два новых ответа и даёт метку для обучения
модели награды.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/consolidation-2026-10-29/constitutional-ai-overview.png|Constitutional AI: на верхней ветви исправленные ответы идут в SFT, на нижней сравнения ответов обучают модель награды для RL]]

*Верхняя ветвь создаёт исправленные демонстрации, нижняя размечает
пары ответов для модели награды. Её оценки поступают в RL. Bai et al.,
[Constitutional AI, Figure 1, §1.2](https://arxiv.org/html/2212.08073v1#S1.SS2),
[оригинальный PNG](https://arxiv.org/html/2212.08073v1/CAI-Overview.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Все блоки и стрелки сохранены, подписи не переведены.*

### Исправленные ответы для SFT

Помощник, обученный на полезность, пишет первый ответ. По одному из
принципов та же модель формулирует замечание, затем получает исходный
запрос, ответ и замечание и пишет исправленную версию. Следующий проход
начинается с последней версии. В опыте Bai et al. принцип на каждом
проходе выбирали случайно, а в SFT сохраняли ответы после всех четырёх
исправлений. Вместе с ними использовали демонстрации от того же помощника
на обычных запросах, чтобы в данных оставались примеры полезных ответов.
Дообучали предобученную модель, а не модель, которая сгенерировала
демонстрации ([Bai et al., §3.1–3.2](https://arxiv.org/html/2212.08073v1#S3.SS1)).

Во время SFT на вход подают исходный диалог. Если включить в него
замечание и просьбу переписать ответ, модель будет учиться другой задаче:
исправлению уже написанного текста. В таблице ниже это различие видно
по тому, какие поля попадают в обучающую запись.

<!-- BEGIN REVIEW: Constitutional AI 2026-10-29 -->

<a id="constitutional-data-example"></a>

### Worked example: from a revision to a training record

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

<details id="cai-code-sft">
<summary>Код: собрать записи SFT из исправленных ответов</summary>

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

</details>

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

<a id="cai-rlaif"></a>

## RLAIF: оценки для обучения с подкреплением

Модель после SFT генерирует два ответа на один запрос. Оценщик видит
диалог, оба ответа и выбранный принцип. Например, на запрос о частном
адресе вариант A раскрывает сведения, а B объясняет отказ. Оценщику
предлагают продолжить фразу с выбором A или B. Из вероятностей этих
двух вариантов получают метку предпочтения. На таких метках обучают
модель награды, чтобы затем оценивать новые ответы
([Bai et al., §4.1](https://arxiv.org/html/2212.08073v1#S4.SS1)).

В примере ниже оценщик отводит A и B суммарно только $0{,}10$
вероятности. Остальные продолжения отбрасывают и делят обе вероятности
на $0{,}10$. Получается $(0{,}2,\ 0{,}8)$. При обучении модели
награды это цель для вероятности парной победы, полученной из разности
оценок $r_A-r_B$. Во время RL обученную модель награды можно вызвать для
одного нового ответа. Она вернёт его скалярную оценку; заново подбирать
к нему второй ответ и спрашивать оценщика не требуется.

<a id="cai-soft-labels"></a>

### Worked example: from option probabilities to a preference label

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

<details id="cai-code-preferences">
<summary>Код: нормировать вероятности и проверить функцию потерь</summary>

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

</details>

Human helpfulness labels remain in the original preference-model training.
New principles require new generated revisions or comparisons and another
training run. Editing a principle does not edit the weights of a deployed
assistant.

<!-- END REVIEW: Constitutional AI 2026-10-29 -->

### RLAIF и RLHF

Обучение с подкреплением по обратной связи от модели называют
RLAIF (*reinforcement learning from AI feedback*).
В схеме [Lee et al., Figure 2](https://arxiv.org/html/2309.00267v2#S1.F2)
обе ветви стартуют с одной SFT-модели. Она генерирует пары ответов.
Люди размечают предпочтения в нижней ветви, а готовая языковая модель
в верхней. В каждой ветви на этих данных обучают свою модель награды,
а уже её оценки используют при обучении с подкреплением.
Стадия исправления ответов по принципам для RLAIF не обязательна.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlaif-lee-2026-11-22/rlhf-vs-rlaif.png|Одна SFT-модель порождает пары ответов; сверху их сравнивает языковая модель, снизу человек, после чего в каждой ветви отдельно обучаются модель награды и стратегия]]

*Harrison Lee et al., [RLAIF, Figure 2](https://arxiv.org/html/2309.00267v2#S1.F2),
[оригинальный PNG](https://arxiv.org/html/2309.00267v2/rlaif.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Рисунок без изменений.*

В Constitutional AI модель награды обучалась на смеси: модель
размечала безвредность, люди размечали полезность. В опытах
[Lee et al., §4.1](https://arxiv.org/html/2309.00267v2#S4.SS1)
разметку для двух ветвей собирали отдельно. При оценке резюме люди
выбрали RLAIF в 50% сравнений с RLHF. В опыте с
безвредностью разметчики оценивали ответы по одному. Безвредными они
сочли 88% ответов RLAIF и 76% ответов RLHF. Эти числа относятся к
разным задачам и процедурам оценки. Доля побед в сравнении двух резюме
не измеряет частоту вредных ответов.

В варианте [direct RLAIF, §2.2.2](https://arxiv.org/html/2309.00267v2#S2.SS2.SSS2)
отдельную модель награды не обучают. Для каждого нового ответа во время
RL оценщик получает, например, исходный текст и его резюме. Из вероятностей
оценок от 1 до 10 считают средний балл и переводят его в интервал
$[-1,1]$. Если после нормировки вероятностей ненулевую массу получили
только 7 и 9 с весами $0{,}75$ и $0{,}25$, средний балл равен $7{,}5$,
а награда после линейного преобразования равна
$2(7{,}5-1)/9-1\approx 0{,}44$. Не нужно сначала собирать пары для
модели награды, но оценщику придётся обработать каждый новый ответ,
сгенерированный для RL. Оценка по рубрике остаётся суждением модели; в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR]]
наградой может служить результат программной проверки.

<span id="как-оценивать-изменение-поведения"></span>

## Оценка поведения после обучения

Верность машинной разметки проверяют на парах с человеческими оценками
([Lee et al., §2.3](https://arxiv.org/html/2309.00267v2#S2.SS3)).
Это проверка оценщика, а не помощника. Для проверки помощника нужны
его новые ответы, которых не было ни при обучении, ни при подборе
правил. Bai et al. использовали и сравнения ответов,
и оценку отдельного диалога.

<span id="человеческие-сравнения-полезность-и-безвредность"></span>

<a id="cai-human-elo"></a>

### Парные сравнения и Elo

В опыте Bai et al. разметчик сам вёл диалог. На каждом ходе он получал
два ответа от разных моделей и выбирал предпочтительный. Для оценки
полезности и безвредности собирали отдельные сравнения. Эти диалоги
были новыми: их не использовали для обучения модели награды или
помощника. По исходам сравнений авторы оценили два рейтинга Elo
([§3.3](https://arxiv.org/html/2212.08073v1#S3.SS3)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/cai-evaluation-2026-11-14/human-elo-frontier.png|Человеческие сравнения моделей 52B: при близкой полезности поздние версии RL-CAI получили более высокий рейтинг безвредности, чем версии обычного RLHF]]

*Bai et al., [Figure 2, Constitutional AI](https://arxiv.org/html/2212.08073v1#S1.F2).
[Оригинальный PNG](https://arxiv.org/html/2212.08073v1/Constitutional.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), без изменений.
Каждая точка — модель на одном из этапов обучения. Для RL-CAI
зелёный крест обозначает старт после обучения на исправленных ответах
(SL-CAI), а красный — предобученную модель, с которой начинались
показанные здесь запуски RLHF. Интервалы неопределённости показаны
отдельно на [Figure 3](https://arxiv.org/html/2212.08073v1#S1.F3).*

Около 80 Elo по горизонтали отметки RL-CAI с CoT лежат выше
ближайшей точки обычного RLHF. При близком рейтинге полезности
разметчики чаще предпочитали ответы RL-CAI по критерию безвредности.
Если первая модель опережает вторую на 100 Elo, ожидаемая доля побед
первой равна $1/(1+10^{-100/400})\approx 0{,}64$
([HH-RLHF, приложение B.5](https://arxiv.org/html/2204.05862v1#A2.SS5)).
Обе модели при этом могут давать безвредные ответы. Кроме того,
RL-CAI начинался с SL-CAI, а сравниваемые запуски RLHF начинались
с предобученной модели. Разницу между точками нельзя приписать
одной только замене человека на оценщика.

По инструкции разметчикам, при одинаковой безвредности нужно было
предпочесть менее уклончивый ответ. Вернёмся к запросу о домашнем
адресе: «Не могу помочь» и объяснение отказа с публичным способом
связи не разглашают адрес, но второй ответ может выиграть сравнение.
Авторы связывают с этой формулировкой часть различий с прежним
экспериментом ([Bai et al., §4.4](https://arxiv.org/html/2212.08073v1#S4.SS4)).

<span id="отдельная-проверка-на-вредные-ответы"></span>

<a id="cai-harmfulness-score"></a>

### Оценка отдельного диалога

В [§4.5](https://arxiv.org/html/2212.08073v1#S4.SS5) разметчик пытался
спровоцировать вредный ответ в разговоре с одной моделью. После диалога
он оценивал, насколько это удалось, по шкале от 0 до 4. На этих
оценках обучили отдельную модель, которая предсказывает балл по всему
диалогу. На графике усреднены её прогнозы для 64 специально отобранных
отложенных запросов: по 256 ответов на каждый.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/cai-evaluation-2026-11-14/absolute-harmfulness.svg|Модельная оценка вреда Helpful RLHF растёт, у HH RLHF и вариантов RL-CAI снижается; сплошные линии T=1, пунктир T=0]]

*Bai et al., [Figure 10, Constitutional AI](https://arxiv.org/html/2212.08073v1#S4.F10).
[Оригинальный SVG](https://arxiv.org/html/2212.08073v1/HScore.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), без изменений.
Здесь большее значение соответствует более вредным ответам.
Сплошные линии — температура 1, пунктирные — температура 0.*

У HH RLHF в конце обучения человеческий рейтинг безвредности падает
([Figure 8, правая панель](https://arxiv.org/html/2212.08073v1#S4.F8)),
а предсказанный балл вреда на Figure 10 снижается. Разметчики в парных
сравнениях учитывают уклончивость; во втором опыте отдельная модель
оценивает, удалось ли собеседнику спровоцировать вредный ответ.
Два результата совместимы: помощник может реже выдавать вредное
содержание, но чаще уклоняться от разговора. По этим графикам нельзя
отделить вклад уклончивости. Кроме того, разметчики могли по-разному
использовать шкалу от 0 до 4, а 64 отобранных запроса не представляют
все случаи использования модели.

<span id="что-измерять-у-своего-помощника"></span>

### Проверка на разрешённых и запрещённых запросах

Пусть тест помощника по контактам содержит 40 запросов на адрес
организации, 40 запросов на непубличные адреса без разрешения и
20 запросов, для которых нужно уточнить полномочия. Обе версии получают
одни и те же запросы и настройки генерации. Допустим, после обучения
разглашений стало 2 вместо 10 из 40, но ложных отказов на запросы
об адресах организаций стало 8 вместо 3 из 40. Улучшение на закрытых
данных не отменяет регрессии на открытых. Третью группу оценивают отдельно:
помощник должен уточнить полномочия или указать способ их подтвердить,
а не угадать. Для обычных запросов проверяют также, что публичный адрес
верен, а не только что модель его выдала.

Регрессию на прежних задачах после настройки называют *alignment tax*.
В InstructGPT её проверяли на отложенных NLP-задачах: часть ухудшений
удалось уменьшить, добавляя в целевую функцию PPO градиент на
предобучающих данных
([Ouyang et al., §4.2](https://arxiv.org/html/2203.02155#S4.SS2)).
В системе с инструментами исполнение ограничений не стоит поручать
одним весам модели: например, доступ к закрытым контактам может
проверять сам сервис.
См. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|безопасность агентных систем]].

Если в данных уже есть пары с выбранным и отклонённым ответами, на них
можно дообучить самого помощника методом
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|DPO]],
без отдельной модели награды и нового цикла RL.
