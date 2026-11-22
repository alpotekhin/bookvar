---
title: RLVR — обучение по проверяемой награде
type: textbook-chapter
status: canonical
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - anchor: rlvr-contract
    label: "Проверяемая награда и исходная способность решать задачи"
    label_en: "Verifiable rewards and initial task competence"
  - anchor: verifiable-reward
    label: "Источник награды и алгоритм обновления"
    label_en: "Reward sources versus update algorithms"
  - anchor: rlvr-equation-example
    label: "Четыре ответа на уравнение и компоненты награды"
    label_en: "Four equation answers and reward components"
  - anchor: rlvr-reward-components
    label: "Награда за правильность и за формат"
    label_en: "Correctness and format rewards"
  - anchor: rlvr-task-records
    label: "Запись задачи, закрытые тесты и разбиение данных"
    label_en: "Task records, hidden tests and data splits"
  - anchor: rlvr-rollout-records
    label: "Версия стратегии и вероятности сгенерированного ответа"
    label_en: "Policy revisions and rollout log probabilities"
  - anchor: rlvr-objective
    label: "Ожидаемая награда и KL-регуляризация"
    label_en: "Expected rewards and KL regularization"
  - anchor: rlvr-score-function
    label: "Градиент при недифференцируемой проверке"
    label_en: "Learning from a nondifferentiable verifier"
  - anchor: rlvr-reward-scale
    label: "Масштаб награды относительно KL"
    label_en: "Reward scaling relative to KL"
  - anchor: parser-vs-verifier
    label: "Извлечение ответа и проверка результата"
    label_en: "Answer extraction versus result verification"
  - anchor: rlvr-reward-gradient
    label: "От строк и наград к обновлению вероятностей"
    label_en: "From response strings to reward gradients"
  - anchor: rlvr-symbolic-verification
    label: "Эквивалентность выражений, допуски и Math-Verify"
    label_en: "Expression equivalence, tolerances and Math-Verify"
  - anchor: rlvr-program-tests
    label: "Покрытие тестов, частичная награда и изоляция кода"
    label_en: "Test coverage, partial credit and code isolation"
  - anchor: rlvr-agent-final-state
    label: "Конечное состояние и допустимость действий агента"
    label_en: "Agent final states and permitted actions"
  - anchor: rlvr-formal-proofs
    label: "Формальное утверждение, аксиомы и доказательство"
    label_en: "Formal statements, axioms and proofs"
  - anchor: agent-reward-design
    label: "Совпадение вызова и выполнение задачи"
    label_en: "Tool-call matching versus task completion"
  - anchor: rlvr-toolrl
    label: "ToolRL: частичные оценки имени и аргументов"
    label_en: "ToolRL partial function and argument scores"
  - anchor: rlvr-tooln1
    label: "Tool-N1: бинарная проверка и single-step данные"
    label_en: "Tool-N1 binary checks and single-step data"
  - anchor: rlvr-completion-efficiency
    label: "ASTRA: полнота, число вызовов и F1"
    label_en: "ASTRA completion, call counts and F1"
  - anchor: rlvr-refusal-filtering
    label: "Отказ, недостающая информация и фильтрация групп"
    label_en: "Refusal, missing information and group filtering"
  - anchor: rlvr-process-vs-outcome
    label: "Награда за результат и за промежуточные шаги"
    label_en: "Outcome rewards versus step feedback"
  - anchor: rlvr-curriculum
    label: "Смешанные группы и оценка доступной сложности"
    label_en: "Mixed-reward groups and task difficulty"
  - anchor: verifier-audit
    label: "Аудит оценщика, шаблоны и остановка генерации"
    label_en: "Grader audits, prompt formats and stopping"
  - anchor: rlvr-false-acceptance
    label: "Редкие ложные принятия и загрязнение награды"
    label_en: "False accepts and contaminated reward signals"
  - anchor: rlvr-metrics
    label: "Метрики разбора, проверки, обучения и обобщения"
    label_en: "Parsing, verification, learning and generalization metrics"
  - anchor: rlvr-pass-at-k
    label: "Pass@k, неоднородность задач и несмещённая оценка"
    label_en: "Pass@k, task heterogeneity and unbiased estimation"
  - anchor: rlvr-tulu-experiment
    label: "Tülu 3: награда, KL, длина и перенос"
    label_en: "Tülu 3 rewards, KL, length and transfer"
  - anchor: rlvr-tulu-value-model
    label: "Функция ценности, награда RM и параметры PPO"
    label_en: "Value models, RM scores and PPO settings"
  - anchor: rlvr-rollout-learner-workers
    label: "Rollout workers, learner и устаревшие данные"
    label_en: "Rollout workers, learners and stale data"
  - anchor: rlvr-failures
    label: "Open-R1 и контракты открытых реализаций"
    label_en: "Open-R1 and implementation contracts"
source_language: mixed
aliases:
  - Reinforcement learning with verifiable rewards
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO]]"
previous: "[[00 Учебник/12 Post-training и Alignment/05 DPO]]"
next:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1]]"
primary_sources:
  - https://arxiv.org/abs/2505.00024
  - https://arxiv.org/abs/2504.13958
  - https://arxiv.org/abs/2601.21558
  - https://arxiv.org/abs/2110.14168
  - https://arxiv.org/abs/2305.20050
  - https://arxiv.org/abs/2411.15124
  - https://arxiv.org/abs/2504.13837v5
  - https://arxiv.org/abs/2505.24864v1
  - https://arxiv.org/abs/2107.03374
  - https://github.com/huggingface/open-r1
source_unit_id:
  - assignment-05-prompting-and-grading
  - meeting-04-slides-rlvr-method
  - meeting-04-slides-rlvr-results
---

# RLVR: обучение по проверяемой награде

<a id="rlvr-contract"></a>

Для уравнения $3x+6=18$ проверка корня короче вывода. Подставим
предложенное $x=4$: $3\cdot4+6=18$. Подстановка $x=12$ даёт 42,
поэтому второй ответ не проходит. Модель может писать собственные
решения, а проверяющая программа сравнит результаты с эталоном.

При обучении с подкреплением по проверяемой награде (*reinforcement
learning with verifiable rewards*, RLVR) результат проверки входит в
награду. В записи задачи нужен критерий правильности; готовый текст
решения, как в SFT, не требуется. Если модель ни разу не нашла
правильный ответ, бинарная проверка даст одни нули. Она сообщает
об успехе после генерации, но не подсказывает, с какого шага начать.

<a id="главное-различие"></a>

<span id="награда-и-алгоритм-обновления"></span>

## Проверка ответа и обновление весов

<!-- source_unit_id: meeting-04-slides-rlvr-method -->

<a id="verifiable-reward"></a>

В [[00 Учебник/12 Post-training и Alignment/05 DPO|DPO]] обновление опиралось
на два готовых ответа и метку предпочтения. В RLVR обучаемая модель
сама порождает ответы. Для каждого из них запускают проверку: сравнивают
число с эталоном, выполняют тесты или проверяют состояние среды. Результат
идёт в расчёт обновления; одну и ту же программу проверки можно использовать
с PPO, GRPO, RLOO и REINFORCE.

В распространённой схеме RLHF оценку предсказывает модель награды. В примере
RLVR ниже вердикт зависит от правил извлечения числа и от сравнения с
эталоном. Эти операции можно проверить независимо друг от друга. Если
парсер не распознаёт допустимую запись числа или тесты покрывают не все
условия задачи, ошибка попадёт и в обучающую награду.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlvr-review-2026-09/tulu-rlvr-loop.png|Цикл Tülu 3: генерация, проверяемая награда и обновление PPO]]

*В верхней части модель генерирует ответ, проверка возвращает награду
$\gamma$ или 0. Внизу этот результат входит в обновление весов; $\alpha$
на рисунке обозначает размер шага. В уравнении 8 статьи величину
положительной награды обозначают $\alpha$; дальше используем обозначения
из уравнения.
Nathan Lambert et al.,
[Tülu 3, рисунок 18](https://arxiv.org/html/2411.15124v5#S6.F18), 2024,
версия v5 от 14 апреля 2025; CC BY 4.0. Оригинал без изменений.*

Если сгенерировать четыре решения и выбрать прошедшее проверку, можно
ответить лучше и без изменения весов. Чтобы измерить эффект обучения,
надо сравнить исходную и обновлённую модели на новых задачах при
одинаковом числе попыток и одинаковых настройках генерации.

<a id="rlvr-equation-example"></a>

<span id="сквозной-пример-линейное-уравнение"></span>

## Четыре ответа на уравнение

> Реши $3x+6=18$. Последняя строка должна иметь вид `FINAL: число`.

В обучающую запись входят запрос и эталон. При генерации модель
видит только запрос: поле `verifier` читает программа проверки.

```json
{
  "id": "linear-0042",
  "prompt": "Реши 3x + 6 = 18. Последняя строка: FINAL: число",
  "task": "linear_equation",
  "verifier": {
    "type": "numeric_exact",
    "answer": "4",
    "tolerance": 0.0
  },
  "split": "train",
  "source": "generated-v1"
}
```

Парсер читает последнюю строку. Из `FINAL: 4.0` он извлекает число
`4.0`. Числовая проверка принимает его, потому что оно равно эталону
`4`. Из `FINAL: 12` число извлекается тем же способом, но сравнение
даёт отказ. Строка `Ответ 4` не доходит до сравнения: в ней нет
маркера `FINAL:`. В таблице $r_c$ обозначает результат сравнения, а
$r_f$ — соблюдение формата. Четыре ответа заданы для примера.

| Последняя строка | Что извлёк парсер | $(r_c,r_f)$ |
|---|---:|---:|
| $y_1$: `FINAL: 4` | 4 | (1,1) |
| $y_2$: `Ответ 4` | — | (0,0) |
| $y_3$: `FINAL: 4.0` | 4.0 | (1,1) |
| $y_4$: `FINAL: 12` | 12 | (0,1) |

<span id="разбор-ответа-и-проверка--разные-задачи"></span>

### Извлечение числа и проверка

<a id="parser-vs-verifier"></a>
<a id="grader-contract"></a>
<!-- source_unit_id: assignment-05-prompting-and-grading -->

Парсер возвращает число или `None`. Сравнение с эталоном выполняется
только после успешного разбора. В журнале их результаты хранят
отдельно: иначе `Ответ 4` и `FINAL: 12` выглядели бы одинаково при
одной бинарной награде.

<details id="rlvr-code-parser">
<summary>Код: извлечение числа и расчёт награды</summary>

```python
import re
from decimal import Decimal, InvalidOperation

FINAL = re.compile(r"FINAL:[ \t]*([-+]?[0-9]{1,64}(?:\.[0-9]{1,64})?)[ \t]*")
MAX_COMPLETION_CHARS = 20_000

def parse_final(text: str) -> Decimal | None:
    if len(text) > MAX_COMPLETION_CHARS or text.count("FINAL:") != 1:
        return None           # слишком длинный или неоднозначный ответ
    lines = text.splitlines()
    if not lines:
        return None
    match = FINAL.fullmatch(lines[-1])
    if match is None:          # FINAL допускается только в последней строке
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None

def numeric_reward(completion: str, answer: Decimal) -> dict[str, float]:
    if not isinstance(answer, Decimal) or not answer.is_finite():
        raise ValueError("answer must be a finite Decimal")
    predicted = parse_final(completion)
    format_ok = predicted is not None
    correct = format_ok and predicted == answer
    return {
        "correct": float(correct),
        "format": float(format_ok),
        "total": float(correct) + 0.1 * float(format_ok),
    }
```

</details>

Здесь число записывается ASCII-цифрами с необязательным знаком и дробной частью.
Один завершающий перевод строки допустим; пустая строка или текст после `FINAL`
нарушают контракт. Маркер `FINAL:` должен встретиться ровно один раз во всём
ответе, в том числе без повторов в цитатах или пояснениях. Это намеренно строгий
учебный формат, который заранее сообщается модели. Длина ограничена 20 000
символами, а целая и дробная части — 64 цифрами каждая. Разбор не вызывает
`eval` и не исполняет выражения, предложенные моделью.

Если эталон не является конечным числом, функция поднимает исключение.
Такую задачу исключают из расчёта и исправляют запись; нулевую награду
за ошибку данных модели не начисляют.

<details id="rlvr-code-parser-checks">
<summary>Код: тесты формата и правильности</summary>

```python
assert parse_final("FINAL: 4\nтекст после результата") is None
assert parse_final("3x = 12\nFINAL: 4") == Decimal("4")
assert parse_final("FINAL: +4.0\n") == Decimal("4")
assert parse_final("FINAL: -2.5") == Decimal("-2.5")
assert parse_final("") is None
assert parse_final("FINAL: 4\nFINAL: 4") is None
assert parse_final("FINAL: ошибка\nFINAL: 4") is None
assert parse_final(" FINAL: ошибка\nFINAL: 4") is None
assert parse_final("FINAL:\n4") is None
assert parse_final("FINAL: 4\n\n") is None
assert parse_final("FINAL: NaN") is None
assert parse_final("FINAL: Infinity") is None
assert parse_final("FINAL: 4e0") is None
assert parse_final("FINAL: −4") is None  # Unicode-минус вне контракта
assert parse_final("FINAL: " + "4" * 65) is None
assert parse_final("x" * 20_001 + "\nFINAL: 4") is None
assert numeric_reward("FINAL: 4", Decimal("4")) == {
    "correct": 1.0, "format": 1.0, "total": 1.1,
}
assert numeric_reward("FINAL: 12", Decimal("4")) == {
    "correct": 0.0, "format": 1.0, "total": 0.1,
}
```

</details>

`Decimal("4.0")` сравнивается с `Decimal("4")` точно, без погрешности
двоичного представления `float`. Но `1/2`, научная запись и числа с единицами
измерения здесь сознательно не поддерживаются. Добавлять их следует только
вместе с новой спецификацией формата и тестами. Иначе одна и та же строка может
получать разную награду до и после незаметного изменения программы.

<a id="rlvr-reward-components"></a>

<span id="ручной-расчёт-награды"></span>

### Награда за число и за формат

Пусть правильность служит основным сигналом, а соблюдение формата даёт лишь
небольшую дополнительную награду:

$$
r(y)=r_{correct}(y)+0.1r_{format}(y).
$$

Тогда

$$
(r_1,r_2,r_3,r_4)=(1.1,0,1.1,0.1).
$$

Если увеличить вес формата до 1, получим $(2,0,2,1)$. Ответ
$y_4$ с числом 12 всё ещё уступает верным ответам, но приносит единицу
лишь за соблюдение формата. Модель может поднять среднюю награду,
научившись ставить маркер `FINAL:` даже в неверных решениях.
В журнале поэтому отдельно считают разобранные ответы и успешные
сравнения. Чтобы узнать, сколько верных ответов потерял парсер,
вручную проверяют также выборку неразобранных строк.

Если модель написала `3x=18−6=10; x=4` и закончила `FINAL: 4`,
сравнение тоже пройдёт. Равенство в середине решения при этом
неверно. По результату такой проверки можно судить о конечном числе,
но нельзя размечать правильность каждого шага.

<span id="от-записи-данных-к-функции-оптимизации"></span>

## Данные и цель обучения

<a id="rlvr-task-records"></a>

<span id="1-формат-набора-данных"></span>

### Запись задачи

При генерации по записи `linear-0042` модели передают `prompt`.
Проверка получает также `verifier`: по типу задачи выбирает процедуру
и считывает эталон. Идентификатор и версия источника позволяют связать
новые ответы с этой задачей; `split` фиксируют до сбора генераций,
чтобы не переносить одни и те же задачи между обучением и оценкой.

Для задач с кодом примеры из условия и скрытые тесты хранят отдельно.
Модель видит примеры, проверяющая программа запускает тесты.
В её конфигурации также фиксируют время, память и разрешённые действия.

<a id="rlvr-rollout-records"></a>

<span id="2-запись-о-сгенерированном-ответе"></span>

### Журнал генераций

Ответ `FINAL: 4` можно связать с задачей по `prompt_id`. Журнал
хранит исходный текст, число, которое извлёк парсер, и вердикт
проверки. ID и логарифмы вероятностей в примере условные.

```json
{
  "prompt_id": "linear-0042",
  "policy_revision": "step-1200",
  "seed": 17,
  "temperature": 0.8,
  "completion": "... FINAL: 4",
  "token_ids": [101, 913, 42],
  "old_logprobs": [-0.8, -0.2, -0.1],
  "parsed": "4",
  "verdict": "pass",
  "reward_components": {"correct": 1.0, "format": 1.0},
  "verifier_revision": "numeric-v3"
}
```

Если парсер изменится, по исходному тексту и версии проверки можно
воспроизвести старый вердикт и сравнить его с новым. Для PPO нужны ещё
логарифмы вероятностей у стратегии, которая породила ответ. В примере
это `step-1200`. Если обучающий процесс уже дошёл до `step-1202`,
его расчёт даст числитель отношения PPO, но не знаменатель.
Знаменатель берут из сохранённых оценок или пересчитывают со старыми
весами и теми же настройками.

<a id="rlvr-objective"></a>

<span id="3-целевая-функция-обучения"></span>

### Целевая функция

Пусть $x$ обозначает запрос целиком, а $y$ — сгенерированный ответ.
Для нашей задачи $R(x,y)$ равно 1,1 за верное число, 0,1 за неверное
число в нужном формате и 0, если число извлечь не удалось. Стратегия должна чаще порождать
ответы с высокой наградой. Если ограничиваем сдвиг относительно
опорной модели, получаем цель с KL-штрафом:

$$
\begin{aligned}
J(\theta)
&=\mathbb E_{x\sim D,\ y\sim\pi_\theta(\cdot|x)}[R(x,y)]\\
&-\beta\,\mathbb E_{x\sim D}
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x)
\right).
\end{aligned}
$$

Здесь $D$ — распределение задач, $\pi_\theta$ — стратегия обучаемой
модели, а $\pi_{\mathrm{ref}}$ — фиксированная опорная стратегия.
Tülu 3 использует такую постановку с PPO
([Lambert et al., уравнения 7–8](https://arxiv.org/html/2411.15124v5#S6)).
Проверяемая награда применима и без KL-штрафа.

<a id="rlvr-score-function"></a>

<span id="почему-можно-учиться-на-недифференцируемой-проверке"></span>

### Градиент по вероятности ответа

Сравнение с эталоном возвращает число. Чтобы обновить веса, ищем
производную не от этого сравнения, а от вероятности текста, которому
досталась награда. Для одного запроса, пока без KL-штрафа:

$$
\begin{aligned}
&\nabla_\theta\mathbb E_{y\sim\pi_\theta}[R(x,y)]\\
&\quad=\mathbb E_{y\sim\pi_\theta}
\left[R(x,y)\nabla_\theta\log\pi_\theta(y|x)\right].
\end{aligned}
$$

Логарифм вероятности полного текста складывается из логарифмов
вероятностей токенов, $\sum_t\log\pi_\theta(y_t|x,y_{<t})$. Итоговая
награда действует на всю сумму: в принятом ответе обновление касается
и верных преобразований, и случайных фраз. Сравнение числа с эталоном
не показывает, какие токены привели к успеху. В
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|главе о PPO]]
разобрано, как из награды строят преимущество и ограничивают обновление.

<a id="rlvr-reward-scale"></a>

<span id="масштаб-награды-меняет-задачу"></span>

### Масштаб награды и KL

Пусть бинарный результат проверки $v$ умножается на положительное число
$\alpha$. Тогда

$$
\alpha\,\mathbb E[v]-\beta\,\mathrm{KL}
=\alpha\left(\mathbb E[v]-\frac{\beta}{\alpha}\mathrm{KL}\right).
$$

Множитель $\alpha$ снаружи скобок не сдвигает максимум. Внутри
скобок видно, что при $\alpha=10$ прежний $\beta$ даёт в десять раз
меньший относительный вес KL. Размер шага PPO из этого не следует:
он зависит и от нормировки преимущества, и от того, как обучена
функция ценности.

<a id="rlvr-reward-gradient"></a>

<span id="как-награда-меняет-вероятности-ответов"></span>

### Обновление на четырёх ответах

Вернёмся к четырём ответам на уравнение. Пусть пока каждый из них выбирается
с вероятностью $1/4$. Средняя награда равна
$(1.1+0+1.1+0.1)/4=0.575$. Для этого конечного примера можно перебрать все
ответы и вычислить градиент без шума выборки. Если $z_i$ — логит ответа,
$p_i=\operatorname{softmax}(z)_i$, то

$$
\frac{\partial\mathbb E[R]}{\partial z_i}
=p_i(R_i-\mathbb E[R]).
$$

Для двух верных ответов производная равна $0.25(1.1-0.575)=0.13125$.
Для нераспознанного ответа она равна $-0.14375$, для форматного неверного —
$-0.11875$. Последний имеет положительную награду 0.1, но уступает текущему
среднему, поэтому его вероятность при таком обновлении уменьшается.

<details id="rlvr-code-gradient">
<summary>Код: один шаг на четырёх ответах</summary>

```python
import torch

completions = [
    "3x = 12\nFINAL: 4",
    "3x = 12\nОтвет 4",
    "x = (18 - 6) / 3\nFINAL: 4.0",
    "x = 18 / 3 + 6\nFINAL: 12",
]
rewards = torch.tensor([
    numeric_reward(y, Decimal("4"))["total"] for y in completions
], dtype=torch.float64)
logits = torch.zeros(4, dtype=torch.float64, requires_grad=True)
probabilities = logits.softmax(-1)
mean_reward = (probabilities.detach() * rewards).sum()
advantages = rewards - mean_reward
# Точное ожидание REINFORCE для четырёх исходов, не случайная выборка.
rlvr_loss = -(probabilities.detach() * advantages * logits.log_softmax(-1)).sum()
rlvr_loss.backward()
ascent_gradient = -logits.grad.detach().clone()
torch.testing.assert_close(
    ascent_gradient,
    torch.tensor([0.13125, -0.14375, 0.13125, -0.11875], dtype=torch.float64),
)
with torch.no_grad():
    logits.add_(0.1 * ascent_gradient)
    updated_probabilities = logits.softmax(-1)
    updated_reward = (updated_probabilities * rewards).sum()
print([round(x, 6) for x in updated_probabilities.tolist()])
print(f"expected reward: {mean_reward.item():.6f} -> {updated_reward.item():.6f}")
```

</details>

После шага вероятности примерно равны $(0.253281,0.246411,0.253281,0.247027)$,
а ожидаемая награда растёт с 0,575 до 0,581921. Модель чаще выбирает
два ответа с числом 4; ни одного нового ответа в этом расчёте не появилось.

У языковой модели нет отдельного логита для каждого текста.
Вероятность ответа — произведение условных вероятностей его токенов.
Перебрать все тексты нельзя, поэтому градиент оценивают по выборке
генераций. Здесь перебраны четыре исхода: вычитание точного среднего
не меняет градиент. В случайном пакете среднее само зависит от того,
какие ответы выбраны. Такой базовый уровень надо анализировать
отдельно, как в главе о PPO.

## Виды проверяющих программ

<a id="rlvr-symbolic-verification"></a>

### Точная и символьная проверка математики

Строки `1/2` и `0.5` различаются, хотя обозначают одно число.
Чтобы принять обе записи, их нужно сначала привести к общему
числовому представлению.
Для выражения с переменными требуется также зафиксировать область допустимых
значений: $x/x$ и $1$ совпадают при $x\ne0$, но не задают одну и ту же функцию
на всей вещественной прямой. Аналогично $\sqrt{x^2}$ равно $|x|$, а не $x$
для произвольного вещественного $x$. Без предположений о переменных сравнение
символьных выражений может решать не ту задачу, которая была в условии.

В [Math-Verify](https://github.com/huggingface/Math-Verify#architecture)
проверка разделена на извлечение фрагмента, преобразование в представление
SymPy и сравнение с эталоном. Библиотека поддерживает не только числа, но и
множества, интервалы, матрицы и отношения. Конкретное поведение зависит от
настроек нормализации: например, удаления единиц измерения и исправления
неполной записи LaTeX. Такая терпимость полезна при оценке ответов, но в
обучении каждое послабление меняет множество строк, за которые выдаётся
положительная награда.

Порядок аргументов также входит в контракт. В Math-Verify вызывают
`verify(gold, prediction)`, а не наоборот. Например, эталон $1<x<2$ и ответ
$(1,2)$ принимаются по умолчанию, но перестановка этих представлений может
дать отказ. Такая асимметрия не позволяет получить награду простым повторением
неразрешённого неравенства из условия. Направление сравнения и флаг
`allow_set_relation_comp` описаны в
[FAQ Math-Verify](https://github.com/huggingface/Math-Verify#why-is-verify-function-not-symmetric).

Для приближённого числового ответа заранее задают условие, например
$|\hat a-a^*|\le\varepsilon_{\mathrm{abs}}+
\varepsilon_{\mathrm{rel}}|a^*|$. Абсолютный допуск работает около нуля,
относительный учитывает масштаб числа. Оба значения определяются смыслом
задачи, а не подбираются после просмотра результатов модели. Символьное
упрощение выполняют с ограничениями времени и ресурсов: сложное выражение
может надолго занять проверяющий процесс даже без исполнения произвольного кода.

<a id="rlvr-program-tests"></a>

### Программный код

Для задачи сортировки одного теста на неубывание недостаточно:
пустой список проходит его для любого входа. Проверка должна также
убедиться, что результат содержит все исходные элементы с теми же
кратностями. Программа может без ошибок завершаться и всё равно
нарушать спецификацию.

Награда может быть бинарной — пройдены все тесты — или частичной, равной доле
пройденных тестов. Во втором случае модель получает больше информации о
неполных решениях. Но и целевая функция меняется: вместо вероятности полностью
верной программы максимизируется средний успех на выбранном наборе примеров.
Если девять из десяти тестов относятся к одному простому случаю, решение
только этого случая уже получает 0.9. Поэтому важны не только число тестов, но
и их покрытие, веса и устойчивость к специально подобранным неверным программам.

Открытые примеры в условии помогают понять формат, обучающие закрытые тесты
дают награду, а отдельный комплект оценочных тестов измеряет обобщение.
«Закрытые» не означает «невозможно узнать»: многократная обратная связь сама
раскрывает информацию о проверке. Даже при отсутствии прямого доступа к файлам
оптимизация может подстроиться под ограниченный набор случаев.

Сгенерированную программу исполняют в отдельной изолированной среде с
ограничениями времени, памяти и числа процессов. В ней нет ключей, файлов
обучения, доступа к соседним задачам и произвольного сетевого доступа;
необходимая временная файловая система изолирована. Родительский процесс
должен различать неверный ответ, превышение лимита программой и сбой самой
инфраструктуры. Сбой сервиса проверки не является математическим
доказательством, что ответ неверен. Об опасности исполнения модельного кода
прямо предупреждает [официальный репозиторий HumanEval](https://github.com/openai/human-eval#usage).

<a id="rlvr-agent-final-state"></a>

### Инструменты и агенты

Синтаксически правильный вызов инструмента — лишь промежуточное действие.
Если агент должен перенести встречу, подтверждение успешного запроса API ещё
не гарантирует, что изменена нужная встреча и сохранены её участники. Награда
проверяет конечное состояние календаря относительно исходной задачи, а при
наличии ограничений — также журнал действий. Восстановление правильного
состояния в конце не отменяет запрещённую операцию, совершённую по пути.

Если вычитать штраф за каждый вызов, бездействие может получать
больше награды, чем перенос встречи. Сначала нужно определить,
считается ли задача выполненной, и лишь затем сравнивать стоимость
допустимых траекторий.

<a id="rlvr-formal-proofs"></a>

### Формальные доказательства

Проверяющее ядро подтверждает, что вывод следует из заданных правил
и аксиом. Оно не проверяет, что формальная теорема выражает исходную
задачу. Например, если в условии требуется результат для всех целых,
доказательство только для положительных чисел не отвечает на исходный вопрос.
Формулировку теоремы, окружение и разрешённые аксиомы фиксируют вне
сгенерированного ответа. Проверка не должна принимать заглушки,
новые допущения и отключение проверки как доказательство.

У вызова инструмента можно отдельно оценить совпадение аргументов
и результат всей операции. Ниже ToolRL и Tool-N1 по-разному оценивают
один и тот же ошибочный вызов, а ASTRA считает выполненные части задачи.

<!-- BEGIN REVIEW: tool rewards 2026-10-29 -->

<a id="agent-reward-design"></a>

## Tool-call matching and task completion

A correct API name, a correct argument and a completed task provide different
feedback. Suppose the requested action is to check a fare from ORD to SFO.
The call `Get_Price(loc_1="ORD", loc_2="LAX")` uses the right function and the
right departure airport, but asks about the wrong destination. An exact-match
reward treats it like a completely unrelated call. A decomposed reward can
credit the fields that match while still withholding full credit.

<a id="rlvr-toolrl"></a>

### ToolRL: compare the function and its arguments

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/toolrl/reward-design.png|ToolRL: формат, имена функций, ключи и значения аргументов]]

*Cheng Qian et al., ToolRL.
[Original repository illustration](https://github.com/qiancheng0/ToolRL/blob/8cee13ec0ca72f0461da372a93a6fd8140dbb840/assets/reward.png),
Apache 2.0; the full 6696 × 2365 image is retained without changes.
The upper part separates format from correctness. The lower part compares
function names, parameter names and values for two candidate outputs.
The [license is included with the site](/licenses/toolrl-8cee13ec-Apache-2.0.txt).
Open the original to read the individual fields.*

The released
[reward implementation](https://github.com/qiancheng0/ToolRL/blob/8cee13ec0ca72f0461da372a93a6fd8140dbb840/verl/utils/reward_score/rlla.py)
first compares the lists of function names. Its `match_score` counts repeated
names: two calls to `Get_Price` and one call to `Get_Price` have intersection
size 1 and union size 2, not two identical sets of names.

For matching calls, the implementation adds the Jaccard score of their
parameter names and the number of equal values under the same keys. The raw
score is then rescaled to the configured interval. In the default
correctness range, a raw score `S` becomes

$$
\begin{aligned}
S_{\max}&=1+|G|+\sum_{g\in G}|\operatorname{keys}(g)|,\\
R_{\mathrm{correct}}&=6\,\frac{S}{S_{\max}}-3.
\end{aligned}
$$

Here `G` is the list of ground-truth calls, not a GRPO group size.
For one expected call with two parameters, the maximum is `1 + 1 + 2 = 4`.
The incorrect-destination call earns 1 for its function name, 1 for its
parameter-name set and 1 for the departure value: `S=3` and
`R_correct=1.5`. An exact call earns 3. A completely different function earns
−3, even if its arguments happen to contain the same strings.

The following code handles only one expected call and one predicted call,
with ordinary JSON scalar values. It exposes the arithmetic without
implementing the multi-call assignment problem.

<details id="rlvr-code-toolrl">
<summary>Code: correctness score for one tool call</summary>

```python
def single_tool_reward(gold, predicted):
    gold_args, predicted_args = gold["parameters"], predicted["parameters"]
    if gold["name"] != predicted["name"]:
        return -3.0
    gkeys, pkeys = set(gold_args), set(predicted_args)
    union = gkeys | pkeys
    key_score = len(gkeys & pkeys) / len(union) if union else 1.0
    value_score = sum(
        k in predicted_args and predicted_args[k] == v
        for k, v in gold_args.items()
    )
    raw = 1.0 + key_score + value_score
    maximum = 2 + len(gold_args)
    return 6 * raw / maximum - 3

fare_gold = {
    "name": "Get_Price",
    "parameters": {"loc_1": "ORD", "loc_2": "SFO"},
}
fare_wrong_destination = {
    "name": "Get_Price",
    "parameters": {"loc_1": "ORD", "loc_2": "LAX"},
}
assert single_tool_reward(fare_gold, fare_gold) == 3.0
assert single_tool_reward(fare_gold, fare_wrong_destination) == 1.5
assert single_tool_reward(fare_gold, {"name": "Delete", "parameters": {}}) == -3.0
print("correctness:", single_tool_reward(fare_gold, fare_wrong_destination))
```

</details>

Format contributes a separate 0 or 1 in the default code; the optional length
reward is off unless enabled. A correctly formatted wrong-destination call
therefore receives `1 + 1.5 = 2.5`, versus 4 for a correctly formatted exact
call. This is partial feedback, not a certificate that the requested task
succeeded. Tracking the success rate separately prevents a rising average
reward from concealing errors in destinations or other critical arguments.

For several calls, matching itself needs verification.
[ToolRL §3.3](https://arxiv.org/html/2504.13958v1#S3.SS3) describes an optimal
assignment, whereas the pinned code greedily consumes one unused prediction
per ground-truth call. With repeated function names, the first local choice
can prevent a better overall assignment. Reordering candidates can then
change the implemented reward. An experiment must identify which procedure
it ran; the paper's formula and a repository function are not interchangeable.

Matching against one annotated call sequence also penalizes valid alternatives
unless the evaluator explicitly allows them. A different route through an API
can reach the same final state. Conversely, matching every annotated field
cannot establish compliance with a requirement that the annotation omitted.
For state-changing tasks, final-state checks and action-policy checks remain
necessary.

<a id="rlvr-tooln1"></a>

Binary matching is another choice. In
[Tool-N1 §4.1–4.3](https://arxiv.org/html/2505.00024v2#S4.SS3), an answer earns
1 only when the required tags and the function's name and argument-value
dictionary are correct. Reordering dictionary keys is allowed. Our
wrong-destination example therefore earns 0, not partial credit.
Multi-turn source trajectories are split into single-step prediction records
with the preceding interaction supplied as context; this is not full online
optimization of a newly executed trajectory.

The choice of initialization is empirical too. In
[Tool-N1 §5.3.2, Table 3](https://arxiv.org/html/2505.00024v2#S5.SS3.SSS2),
the 5,518-instance comparison reports averages of 83.17 for SFT followed by
RL and 83.24 for RL alone under its data allocations. This small reported
difference does not establish that SFT generally harms tool learning.
Both recipes start from an already instruction-tuned model.

<a id="rlvr-completion-efficiency"></a>

### Partial completion and unnecessary calls

ASTRA uses verified subtask completion and a trajectory-level F1-style reward.
For `n` required subtasks, `s` completed subtasks and `c` calls,

$$
\begin{aligned}
r&=s/n,\qquad p=s/(c+\epsilon),\\
F_1&=\frac{2pr}{p+r}.
\end{aligned}
$$

Its reported recall-only and precision-only ablations exhibit different
over-calling and under-calling failures. These are results of that experiment,
not guarantees that F1 stabilizes every agent. The authors also omit KL and
entropy bonuses, filter groups with degenerate reward variance, and buffer
surplus usable samples for subsequent batches. See
[ASTRA §3.1, §3.2.2 and §4.2](https://arxiv.org/html/2601.21558v1#S3.SS1).

Consider a constructed task with four independent checks, each obtainable by
one call. Completing two checks with two calls has recall 0.5, precision 1
and F1 ≈ 0.667. Completing all four with four calls earns 1. Completing the
same four with eight calls leaves recall at 1 but lowers precision to 0.5
and F1 to about 0.667.

For `s>0`, substitution reduces the expression to
`2s / (n + c + epsilon)`. When `s=0`, define the reward as zero rather than
evaluate `0/0`. The code below enforces the example's one-subtask-per-call
assumption. It is not a general implementation of the ASTRA environment.

<details id="rlvr-code-completion">
<summary>Code: subtask completion and call counts</summary>

```python
def completion_efficiency_reward(required, solved, calls, eps=1e-8):
    if any(type(x) is not int for x in (required, solved, calls)):
        raise ValueError("counts must be integers")
    if not (required > 0 and 0 <= solved <= required and calls >= solved):
        raise ValueError("require positive task size and solved <= calls")
    if not 0 < eps < float("inf"):
        raise ValueError("eps must be finite and positive")
    return 2 * solved / (required + calls + eps) if solved else 0.0

efficient_partial = completion_efficiency_reward(4, 2, 2)
efficient_complete = completion_efficiency_reward(4, 4, 4)
redundant_complete = completion_efficiency_reward(4, 4, 8)
assert efficient_complete > efficient_partial
assert efficient_complete > redundant_complete
assert completion_efficiency_reward(4, 0, 0) == 0.0
print([round(x, 6) for x in
       (efficient_partial, efficient_complete, redundant_complete)])
```

</details>

The denominator measures calls, not wall time, tokens, money or risk.
Replacing one cheap call with one expensive call leaves this reward
unchanged. A call that solves several subtasks would violate the example's
assumption and could make the quantity called precision exceed 1. The
subtask accounting and permitted tool semantics must therefore be specified
before using the formula.

<a id="rlvr-refusal-filtering"></a>

### Refusal, missing information and group filtering

A task need not have one unique accepted answer. A verifier can accept
several valid programs, routes or final states. It can also reward stopping
without an action when an externally specified condition makes execution
impermissible or impossible.

For example, a synthetic calendar task can expose an explicit
`authorized=false` flag to the environment. The evaluator checks that no
write occurred and that the agent requested authorization. A successful
unauthorized write receives zero; preserving the state and requesting
permission receives one. This constructed rule is verifiable because
authorization and state changes have specified representations. It does
not establish that a program can decide arbitrary natural-language safety
questions.

If both actions instead receive zero, the reward supplies no distinction
between them. A group-centred algorithm cannot recover that missing
distinction. Filtering identical-reward groups removes their relative
reward contribution regardless of whether the task was easy, impossible or
poorly specified; it does not intrinsically prohibit learning refusals.
The filtered fraction and the evaluation of unsolvable and unauthorized
tasks must be reported separately.

Finally, an executable training environment can model only some production
failures. Evaluation should separately cover timeouts, rate limits,
pagination, permissions, changing state and irreversible side effects.
The availability of a training-time verifier also says nothing about which
checks will be available when a deployed agent answers a new request.

<!-- END REVIEW: tool rewards 2026-10-29 -->

<a id="rlvr-process-vs-outcome"></a>

<span id="проверка-результата-и-проверка-хода-решения"></span>

## Результат и шаги решения

В примере с уравнением оба ответа, которые заканчиваются на `FINAL: 4`,
получают одну награду, даже если в одном написано `18−6=10`.
Для обратной связи о ходе решения нужно проверять или оценивать
конкретный шаг. Это делают двумя способами:

- **проверяемая награда за шаг:** шаг проверяет интерпретатор или система
  формальных доказательств;
- **модель награды за процесс:** обученная модель предсказывает качество шага.

Второй вариант уже не является полностью проверяемым: он наследует смещения
модели награды и страдает от сдвига распределения. Работа
[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) полезна для
сравнения надзора за результатом и за процессом, однако использованная в ней
обученная модель награды не служит формальным доказательством правильности.

Cobbe et al. в
[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
называют *verifier* обученную модель, которая ранжирует решения GSM8K.
Она предсказывает корректность текста; в нашем примере программа
сравнивает число с эталоном. Оба механизма могут использоваться для
отбора решений, но у них разные ошибки. Обучение моделей результата
и отдельных шагов разобрано в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|главе о моделях награды]].

Даже точная проверка каждого локального шага не всегда измеряет продвижение
к решению. Цепочка из ста верных преобразований, возвращающаяся к исходному
выражению, может ничего не доказывать. Если награждать каждый допустимый шаг
одинаково, появится стимул удлинять такие цепочки. Локальная корректность,
полезность шага и успех всего решения — разные свойства; их нельзя заменить
одной оценкой без дополнительных предположений.

<a id="rlvr-curriculum"></a>

<span id="учебная-программа-и-доступная-сложность"></span>

## Группы ответов и сложность задачи

Пусть модель выдала четыре ответа с нулевой наградой за правильность.
В REINFORCE без базового уровня все четыре вклада в градиент награды
равны нулю. В GRPO после вычитания среднего по группе равные награды
тоже дают нулевое преимущество. Остальные слагаемые, например KL,
могут продолжать менять параметры.

Для фиксированной задачи пусть вероятность успеха текущей модели равна $p$.
Если $G$ ответов генерируются независимо из одного распределения, то все они
верны с вероятностью $p^G$, а все неверны — с вероятностью $(1-p)^G$. Значит,
вероятность смешанной группы равна

$$
P_{\mathrm{mix}}=1-p^G-(1-p)^G.
$$

При $p=0.5,G=4$ получаем 0.875. При $p=0.01$ — лишь 0.0394:
около 96% групп не дают сравнительного сигнала по правильности. На почти
наверняка решаемых задачах возникает симметричная проблема. Это объясняет
интерес к задачам промежуточной сложности, но не доказывает, что выбор
$p=0.5$ оптимален по затратам или переносу на другие задачи. Время решения,
число токенов и тематическое разнообразие в эту простую модель не входят.

Размер группы тоже имеет цену. Для $p=0.01,G=32$ вероятность смешанной
группы повышается примерно до 0.275. Но для этого требуется в восемь раз
больше ответов, чем при $G=4$. Можно увеличить группу, подобрать более
доступные задачи, изменить исходную модель или предварительно обучить её на
нескольких примерах решения. Эти варианты решают разные причины недостатка
сигнала, поэтому сравнивать их нужно при сопоставимом бюджете генерации.

В Open-R1 есть
[инструменты отбора по доле успешных решений](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/scripts/pass_rate_filtering/README.md).
Это поддерживаемая возможность репозитория, а не обязательный шаг всех его
конфигураций. Доля успеха зависит от версии модели и параметров генерации;
её оценка на восьми ответах весьма неточна. Если истинное $p=0.1$, вероятность
не увидеть ни одного успеха в восьми попытках равна $0.9^8\approx0.43$.
Наблюдение «0 из 8» поэтому не доказывает, что задача недоступна модели.
По мере обучения отбор пересчитывают, а неизменённую оценочную выборку
сохраняют отдельно: иначе улучшение можно получить просто удалением трудных
задач из проверки.

<span id="проверяющая-программа-как-граница-безопасности"></span>

## Аудит проверяющей программы

<a id="verifier-audit"></a>

<span id="проверка-оценщика-до-первого-обновления-модели"></span>

### Проверка формата в CS336

В Stanford CS336 Assignment 5 ответ должен завершаться размеченным фрагментом
вроде `<answer>...</answer>`. Для диагностики нужны две отдельные величины.
В формуле $f=r_{format}$ — флаг успешного разбора:

$$
\begin{aligned}
r_{format}&=\mathbf 1[\operatorname{parse}(y)\ \text{успешен}],\\
r_{answer}&=\begin{cases}
\mathbf 1[\operatorname{verify}(\hat a,a^*)],&f=1,\\
0,&f=0.
\end{cases}
\end{aligned}
$$

В функции `r1_zero_reward_fn` итоговое поле `reward` равно `answer_reward`,
а `format_reward` возвращается отдельно для диагностики. При правильных тегах
и неверном ответе общая награда остаётся нулевой. Это можно проверить в
[исходном коде задания](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_alignment/drgrpo_grader.py).
Такой выбор отличается от учебной суммы `correct + 0.1 format`, приведённой
выше: в задании нет самостоятельного поощрения за одни лишь теги. Обе
постановки допустимы, но отвечают разным целям и не должны описываться как
одна функция награды.

В [Assignment 5, §3.2–3.4](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf)
до первого обновления сравнивают три запроса: только вопрос с требованием
`\boxed{}`, zero-shot `r1_zero` и его вариант с тремя разобранными примерами.
Это меняет исходную вероятность успеха и соблюдения формата при тех же весах.
Несколько показанных решений не превращают такую оценку в SFT: они остаются
частью входного текста, градиент по ним не вычисляется.

Запрос `r1_zero` уже заканчивается открывающим `<think>`. Поэтому сгенерированная
часть должна закрыть его и выдать `<answer>...</answer>`. В сохранённой версии
`r1_zero_reward_fn` ищет буквальную подстроку `</think> <answer>` с одним
пробелом: замена пробела переводом строки даёт нулевой `format_reward`, даже
если теги выглядят разумно. Это особенность данной реализации, не правило
XML и не свойство RLVR. Для `question_only` используется другая функция,
которая извлекает `\boxed{...}`. Применить первый оценщик ко второму формату
означало бы измерять несоответствие шаблонов вместо способности решать задачу.

В упражнении используется температура 1.0, top-p=1.0 и предел 512 токенов.
Для `r1_zero` генерация останавливается на `</answer>`; настройка
`include_stop_str_in_output=True` сохраняет закрывающий тег в строке,
передаваемой оценщику. Если удалить тег на стороне сервера, корректный ответ
получит нулевую награду за формат. Для `question_only` этот stop не задают.
У GSM8K исходная запись содержит решение и `#### answer`: оценщику передают
часть после `####`, а не весь текст решения. Эти условия нужно зафиксировать
до сравнения результатов, включая версию самой программы проверки.

До обучения собирают таблицу ручной проверки как минимум с четырьмя исходами:

| Ответ | Разбор, проверка и причина |
|---|---|
| Корректный и извлечённый | Разбор 1, проверка 1. Обычный положительный пример. |
| Корректный, но не извлечённый | Разбор 0, проверка не вызвана. Правильный ответ потерян при разборе. |
| Неверный, но принятый | Разбор 1, проверка 1. Ложное принятие проверяющей программой. |
| Извлечённый и отвергнутый | Разбор 1, проверка 0. Неверный ответ или ошибка сравнения эквивалентных записей. |

В журнале сохраняют исходную строку, извлечённый ответ, эталон, версии
программ разбора и проверки, а также обе составляющие награды. Пограничные случаи
включают лишние
числа, единицы измерения, научную запись, эквивалентные дроби, Unicode-минус,
несколько тегов, `NaN/Inf` и текст после закрывающего тега. Небольшая ручная
выборка ложных принятий и ложных отклонений обязательна: модульные тесты подтверждают
известные случаи, но не измеряют ошибки на распределении генераций модели.

При проверке на известных примерах пройдите весь путь от вывода модели
до записи награды. Разбор должен отклонять лишний результат, тесты
должны проверять полную спецификацию, а изолированный запуск не должен
видеть скрытую разметку и соседние задачи. Отдельно проверяйте, что
технический сбой или тайм-аут не записывается как вердикт о правильности
ответа. При оценке после обучения оставьте независимые задачи, на
которых модель не получала обратной связи.

<a id="rlvr-false-acceptance"></a>

<span id="почему-редкие-ложные-принятия-опасны"></span>

### Ложные принятия

Пусть модель действительно решает 10% задач. Проверяющая программа принимает
95% правильных ответов и по ошибке принимает 2% неправильных. На 1000 ответов
в среднем придётся 95 истинных принятий и 18 ложных. Среди всех награждённых
ответов доля ложных составит $18/(95+18)\approx15.9\%$, хотя частота ошибки
проверки на неправильных ответах равна всего 2%.

В общем виде, если $p$ — доля истинно правильных ответов, $t$ — вероятность
принятия правильного ответа, а $f$ — вероятность принятия неправильного, то

$$
P(\text{ошибка}\mid\text{принят})
=\frac{(1-p)f}{pt+(1-p)f}.
$$

Даже при той же чувствительности и той же частоте ложных принятий
доля ошибочных наград зависит от того, насколько хорошо модель решает
задачу. Если правильным оказывается лишь 1% ответов, при тех же $t=0.95$ и
$f=0.02$ доля ошибок среди принятых поднимется примерно до 68%.
По мере обучения меняются и сами ответы: модель может чаще выдавать
тот вид ошибок, который пропускает проверка. Поэтому ручную проверку
повторяют на новых генерациях, а не переносят на них прежние оценки
$t$ и $f$.

<a id="rlvr-metrics"></a>

## Метрики

Средняя награда может вырасти без улучшения решения, например если
модель чаще пишет `FINAL:`. Журнал проверки помогает отличить
изменение формата от изменения правильности. На новых задачах
считают отдельные метрики:

| слой | обязательные показатели |
|---|---|
| разбор ответа | доля успешно и неоднозначно разобранных ответов, ошибки по шаблонам |
| проверка | ложные принятия и отказы на ручной разметке, доля превышений времени |
| генерации | pass@1, pass@$k$, длина, достижение токена конца, разнообразие |
| обучающий сигнал | доля групп с разными исходами и ненулевой дисперсией |
| модель | KL к опорной модели, энтропия, доля ограниченных обновлений |
| обобщение | новые генераторы задач, сложность вне распределения, ухудшения других способностей |
| система | токены в секунду, задержка проверки, стоимость одного подтверждённого успеха |

При сравнении pass@1 до и после обучения нужно зафиксировать шаблон
запроса, температуру, предел длины и версию проверяющей программы.

<a id="rlvr-pass-at-k"></a>

<span id="одна-попытка-и-несколько-попыток"></span>

### Pass@1 и pass@k

Pass@1 измеряет успех одной генерации при заданных настройках.
Для pass@$k$ достаточно, чтобы из $k$ попыток была верной хотя бы одна.
Если пользователю показывают один ответ, нужно ещё измерить, сумел
ли способ отбора выбрать верный из этих попыток.

Для фиксированной задачи и независимых попыток pass@k равен $1-(1-p)^k$.
Но подставлять в эту формулу средний pass@1 по всему набору некорректно:
вероятности успеха на разных задачах различаются, а преобразование нелинейно.
Например, если половина задач всегда решается, а половина никогда, средний
pass@1 равен 0.5 и pass@4 тоже равен 0.5, а не $1-0.5^4=0.9375$.

При наличии $n$ независимых генераций, из которых $c$ правильны, в HumanEval
используется оценка

$$
\widehat{\operatorname{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom nk},\qquad 1\le k\le n.
$$

Числитель считает способы выбрать $k$ ответов, не включив ни одного правильного,
а знаменатель — все способы выбора. Если $n=10$, $c=2$, $k=3$, получаем
$1-\binom83/\binom{10}3=1-56/120\approx0.5333$. Сначала эту величину
вычисляют для каждой задачи и только затем усредняют. Если $n-c<k$,
числитель принимается равным нулю. Вывод и реализация приведены у Chen et al.
в [Evaluating Large Language Models Trained on Code, раздел 2.1](https://arxiv.org/abs/2107.03374).

Отношение биномиальных коэффициентов равно вероятности не выбрать
ни одного правильного ответа. Его можно вычислить произведением
$\prod_{i=n-c+1}^{n}(1-k/i)$, не вычисляя факториалы. В листинге
складываются логарифмы множителей. `expm1` избегает потери точности
при вычитании из единицы, когда вероятность успеха мала.

<details id="rlvr-code-pass-k">
<summary>Код: несмещённая оценка pass@k</summary>

```python
import math

def pass_at_k(n: int, c: int, k: int) -> float:
    if any(type(v) is not int for v in (n, c, k)):
        raise ValueError("n, c and k must be integers")
    if not (n >= 1 and 0 <= c <= n and 1 <= k <= n):
        raise ValueError("require n >= 1, 0 <= c <= n and 1 <= k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    log_failure = math.fsum(math.log1p(-k / i) for i in range(n - c + 1, n + 1))
    return -math.expm1(log_failure)

estimated_pass3 = pass_at_k(10, 2, 3)
assert math.isclose(estimated_pass3, 8 / 15)
assert pass_at_k(10, 0, 3) == 0.0
assert pass_at_k(10, 8, 3) == 1.0
print(f"pass@3: {estimated_pass3:.6f}")
```

</details>

Усреднение по задачам и ошибка подстановки эмпирического $\hat p=c/n$ — две
разные проблемы. Даже для одной задачи при конечном $n$ выражение
$1-(1-c/n)^k$ смещено. В нашем примере оно даёт $1-0.8^3=0.488$ вместо
0.5333. Несмещённость комбинаторной оценки означает совпадение её ожидания
с истинным pass@$k$ при повторном независимом сборе $n$ ответов; она не
устраняет статистическую неопределённость одного набора генераций.

В отчёте также нужна доля ответов, обрезанных по пределу длины. Увеличение
лимита может улучшить pass@1 без изменения весов, а уменьшение — уничтожить
итоговую строку с ответом в уже почти законченном решении. Такой эффект
отличается и от ошибки рассуждения, и от ошибки программы извлечения.

<a id="rlvr-pass-k-crossover"></a>

<span id="почему-кривые-passk-могут-пересечься"></span>

### Пересечение кривых pass@k

Пусть на двух задачах базовая модель даёт верный ответ с вероятностями 0,4
и 0,1. После RL первая вероятность выросла до 0,8, а вторая упала до 0,005.
В среднем за одну попытку дообученная модель выигрывает:
$(0.8+0.005)/2=0.4025$ против $(0.4+0.1)/2=0.25$. Теперь дадим обеим моделям по 16 независимых
попыток на каждую задачу. На первой они обе почти наверняка найдут ответ.
На второй вероятность успеха базовой модели вырастет до
$1-0.9^{16}\approx0.815$, а дообученной лишь до $1-0.995^{16}\approx0.077$.
После усреднения по двум задачам получается 90,7% у базовой и 53,9% у
дообученной. Преимущество на первой задаче практически исчезло, когда обе
вероятности приблизились к единице, а разница на второй сохранилась.
Это расчёт с заданными вероятностями, а не результаты запуска модели.

В [эксперименте Yue et al., §3.1](https://arxiv.org/html/2504.13837v5#S3.SS1)
сравнивали базовые модели с их версиями после GRPO. Дообучение начиналось
с весов базовой модели, без промежуточного SFT. При оценке обе версии
получали одинаковый запрос без демонстраций; температура 0,6, top-p 0,95
и предел в 16 384 новых токена тоже совпадали. Для MATH500 и Minerva
на каждую задачу собрали по 128 ответов от каждой версии. Точки левее 128
вычисляли из тех же ответов по комбинаторной формуле выше, а не запускали
новый опыт для каждого $k$
([приложение A.2](https://arxiv.org/html/2504.13837v5#A1.SS2)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlvr-pass-k-2026-11-14/yue-figure-2.svg|Четыре ряда математических тестов и четыре столбца моделей. В панели Minerva для Qwen2.5-32B линия RL выше при одной попытке, а линия Base выше при 128.]]

*Yang Yue et al., [Does Reinforcement Learning Really Incentivize Reasoning
Capacity in LLMs Beyond the Base Model?, рис. 2](https://arxiv.org/html/2504.13837v5#S2.F2),
[оригинальный SVG](https://arxiv.org/html/2504.13837v5/simpleRL_coverage_main.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Все 16 панелей
сохранены без изменений.*

В третьем ряду, Minerva, и третьем столбце, Qwen2.5-32B, при $k=1$
оценка RL составляет около 0,43 против 0,27 у Base. К правой границе,
$k=128$, порядок обратный: около 0,58 против 0,67. Это приближённое
чтение графика. Обе кривые растут: если разрешить ещё одну попытку,
найденный ответ не исчезнет. Но у базовой модели оценка растёт быстрее.

По одним кривым нельзя узнать, на каких задачах совпали успехи. В
[таблице 2](https://arxiv.org/html/2504.13837v5#S4.T2) авторы сравнили
ответы Qwen2.5-7B до и после RL на одних и тех же задачах. На MATH500
при 128 попытках обе версии хотя бы раз решили 462 задачи из 500.
Ещё 18 решила только базовая версия, 5 только дообученная, 15 не решила ни
одна. Итого 480 против 467. Формулировка «в испытании не встретилось
правильного ответа» здесь точнее, чем «модель не умеет решать задачу»:
при вероятности успеха 1% шанс увидеть 128 неудач подряд всё ещё около 28%.
И наоборот, совпавший конечный ответ сам по себе не доказывает правильность
хода решения; [ручная проверка в §3.1](https://arxiv.org/html/2504.13837v5#S3.SS1)
охватывала отдельные задачи GSM8K и AIME24, а не все ответы на графике.

Пересечение не обязательно. В [ProRL, рис. 4–5](https://arxiv.org/html/2505.24864v1#S4.F5)
отправной точкой служила уже дистиллированная модель 1,5B. При длительном
RL на смеси задач у некоторых тестов pass@128 падал, а на задаче
координации перемещений `boxnet`, которой не было в обучающей смеси,
конечная версия опережала промежуточную на всём показанном диапазоне $k$.
Это другой старт, другие данные и другой режим обучения. По двум работам
нельзя выделить вклад длительности отдельно от остальных изменений.

Для сравнения версий имеет смысл фиксировать и число попыток, и расход
токенов: 128 коротких ответов и 128 длинных стоят по-разному даже при
одинаковом пределе длины. Если система показывает пользователю один ответ,
надо отдельно измерить, как часто её собственное правило выбора находит
верный вариант. Методы такого выбора разобраны в
[[00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|главе о вычислениях во время вывода]].

<a id="rlvr-tulu-experiment"></a>

<span id="что-показывает-кривая-rlvr-а-чего-она-не-доказывает"></span>

## Результаты Tülu 3

<!-- source_unit_id: meeting-04-slides-rlvr-results -->

В Tülu 3 проверяемая награда шла после SFT и DPO. На этом этапе
модель училась на смеси из 29 946 запросов: 7 473 задачи GSM8K, 7 500 задач MATH и
14 973 инструкции с ограничениями, которые можно проверить программно. Числовой
ответ в GSM8K сравнивали точно, для MATH применяли извлечение и сравнение
ответа по процедуре MATH Flex, допускающей эквивалентные математические
записи, а в задачах следования инструкциям отдельно проверяли каждое требование —
например, число пунктов или наличие заданной фразы. Модель обновлялась с PPO;
за успешный ответ использовалась положительная награда, масштабированная
коэффициентом $\alpha=10$.

Состав смеси и выбор награды описаны в разделе 6
[Tülu 3 v5 от 14 апреля 2025 года](https://arxiv.org/abs/2411.15124v5).
Эти настройки относятся к экспериментам Tülu 3, а не к RLVR вообще.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlvr-2026-10-24/tulu-rlvr-full-figure19.png|Tülu 3: все двенадцать панелей награды, KL, длины и тестовых оценок]]

*Три независимых семейства запусков: GSM8K, MATH и инструкции с проверяемыми
ограничениями. В каждом ряду слева показана обучающая награда, в центре — KL к
исходной модели, справа — длина ответа; цвета соответствуют четырём значениям
коэффициента KL. Нижний ряд связывает KL с результатами на отложенных задачах
и средним качеством по тестам. Nathan Lambert et al.,
[Tülu 3, рисунок 19 целиком](https://arxiv.org/pdf/2411.15124v5#page=33).
CC BY 4.0. Все двенадцать панелей, легенда и английская подпись сохранены;
кадрирование вокруг рисунка из PDF при 400 dpi, без перерисовки.*

В верхнем ряду награда на GSM8K повышается при всех четырёх значениях $\beta$,
а величина KL заметно различается. Поэтому больший отход от исходной модели
сам по себе не объясняет прирост правильности. В третьем ряду слабый
KL-штраф ($\beta=0.01$) сопровождается самой высокой наградой за ограничения
и самым сильным сокращением ответов. Это уже повод проверить, не достигнут
ли успех за счёт потери содержательности.

Затем модель оценивали на отложенной части
той же целевой задачи и на более широком наборе тестов. Рост обучающей награды означает, что модель чаще проходит проверку
на запросах, по которым обновляют веса. Из неё нельзя узнать ни
о переносе на отложенные задачи, ни о правильности промежуточных
выкладок.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlvr-review-2026-09/tulu-ifeval-kl-and-generalization.png|Целевая IFEval-оценка и среднее качество при разном KL]]

*Для инструкций с ограничениями целевая оценка IFEval остаётся приблизительно
на одном уровне, тогда как средняя оценка по задачам ухудшается при большем KL.
Квадраты и левая шкала относятся к IFEval; ромбы и правая шкала — к среднему
качеству. Горизонтальные линии обозначают исходные оценки. Nathan Lambert
et al., [Tülu 3, рисунок 19, нижняя правая панель](https://arxiv.org/pdf/2411.15124v5#page=33).
CC BY 4.0. Исходный файл рисунка без изменений.*

По одной обучающей награде можно было бы предпочесть запуск с самым слабым
ограничением, но средняя оценка на тестах у него ниже. В приложении B.4
авторы также показывают ответы, которые
выполняют формальные ограничения, но теряют осмысленное содержание. Такие
примеры демонстрируют неполноту выбранного критерия: проверка числа пунктов
не обязана оценивать полезность написанного.

<a id="rlvr-tulu-value-model"></a>

<span id="модель-награды-может-исчезнуть-а-функция-ценности--остаться"></span>

### Функция ценности в PPO

В основном запуске Tülu 3 число для награды возвращает проверяющая
программа. Критик PPO всё равно предсказывает будущую награду, чтобы
посчитать преимущество, а опорная модель нужна для KL.
Начальные веса критика брали из ранее обученной модели награды.
Это способ инициализации: выходы той модели не прибавляли к награде
проверки.

В Tülu 3 отключают dropout, чтобы он не вносил новые случайные
различия в логарифмы вероятностей при генерации и повторном проходе
PPO. Для равенства до первого обновления нужны также одинаковые
веса, контекст и способ расчёта этих вероятностей.
Преимущества центрируют и нормируют, запросы перемешивают между повторными
проходами. Награда $+10$ за подтверждённую правильность дополняется штрафом
$-10$ за отсутствие EOS. Поэтому средняя обучающая награда зависит также от
того, завершает ли модель ответы, а не только от сравнения конечных чисел.
Это настройки [Tülu 3, §6.2 и Table 21](https://arxiv.org/html/2411.15124v5#S6.SS2),
а не обязательные компоненты любого RLVR.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlvr-review-2026-09/tulu-value-initialization.png|Инициализация функции ценности из RM и SFT]]

*Сравнение инициализации функции ценности из SFT-модели и из модели награды.
Сплошные кривые показывают качество GSM8K по левой шкале, пунктирные — среднее
качество по правой. Треугольники относятся к инициализации из RM, круги и
кресты — из SFT. Nathan Lambert et al.,
[Tülu 3, рисунок 21](https://arxiv.org/pdf/2411.15124v5#page=35).
CC BY 4.0. Оригинал без изменений.*

В этой абляции инициализация из модели награды давала более сильные результаты,
но и у неё рост KL не обеспечивал монотонного улучшения. В легенде исходного
графика альтернатива RM называется SFT; в описании списка экспериментов §6.2
сказано anchored DPO. Подпись здесь следует самому графику; по этой
несогласованности нельзя установить происхождение каждой сохранённой версии.

В другой абляции к проверяемой награде добавляли оценки RM. Это отдельное
изменение: начальные веса функции ценности из RM использовали в обоих случаях.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rlvr-2026-10-24/tulu-rlvr-reward-mixing.png|Проверяемая награда отдельно и вместе с оценками RM]]

*Треугольники — только проверяемая награда; квадраты и ромбы — добавление
оценки RM. Сплошные линии и левая шкала показывают GSM8K, пунктир и правая
шкала — среднее качество. В обеих группах функция ценности инициализирована
из RM. Nathan Lambert et al.,
[Tülu 3, Figure 22](https://arxiv.org/html/2411.15124v5#S6.F22), CC BY 4.0.
Полный исходный PNG без изменений.*

Лучший показанный результат GSM8K получен без добавления оценок RM. Но среднее
качество ведёт себя иначе: у варианта с RM крайняя правая точка выше остальных
своей серии. Поэтому рисунок не подтверждает, что чистая награда лучше по всем
метрикам и при любом KL. Здесь проверены конкретные данные, модель и набор
настроек; выбирать вариант нужно по заранее заданным отложенным оценкам.

Отдельно меняли начальную **генерирующую** модель. Старт после SFT иногда
позволял достигнуть той же обучающей награды, что и старт после DPO, но ценой
большего KL и с более слабым результатом на тесте. Это ещё один случай, когда
одинаковая высота кривой награды не означает одинакового качества модели.

Масштаб модели также менял картину. Итоговая таблица 23 даёт следующие оценки:

| Модель, этап | GSM8K | MATH | IFEval |
|---|---:|---:|---:|
| 8B, DPO | 84.3 | 42.0 | 81.1 |
| 8B, RLVR | 87.6 | 43.7 | 82.4 |
| 70B, DPO | 93.5 | 62.3 | 82.6 |
| 70B, RLVR | 93.5 | 63.0 | 83.2 |

*Числа в процентах перенесены из
[Tülu 3, таблица 23](https://arxiv.org/html/2411.15124v5#S6.T23);
настройки оценки соответствуют этой таблице, а не произвольному запуску
бенчмарка.*

У 8B-модели улучшились все три направления, у 70B-модели GSM8K не изменился.
Авторы связывают это с близостью к насыщению данного теста; эксперимент сам по
себе не устанавливает универсальную причину отсутствия прироста. Это результат
конкретного сочетания базовой модели, данных, PPO и трёх проверяющих программ, а
не общее обещание, что RLVR неизбежно улучшает любую математическую или
инструктивную способность.

Эксперимент и его абляции разобраны в
[Berkeley Advanced LLM Agents, meeting 4, слайды 90–120](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf).
Состав данных, функции награды и конфигурации приведены в разделе 6
и приложении E [статьи Tülu 3](https://arxiv.org/abs/2411.15124v5).

<a id="rlvr-rollout-learner-workers"></a>
## Rollout workers and learner workers

Tülu 3 separates generation from parameter updates. Rollout workers run a
policy copy with vLLM and PagedAttention; learner workers update the policy
and value models. The reference policy provides the KL comparison and is
not trained. ZeRO Stage 3 partitions training states so that the models fit
across devices.

With a synchronous schedule, generation waits while the learner updates,
then the learner waits for fresh responses. Dedicated worker pools can
overlap these phases: while one batch is being consumed, rollout workers
prepare another. This improves utilization without making the two phases
equally fast. Queued responses may come from older policy weights.

The paper uses the second-latest inference batch to control this lag.
A bounded lag does not guarantee on-policy data. Their sampling-policy
revision and log probabilities must remain attached to them; replacing
those probabilities with scores from the updated learner changes the
PPO ratio. The [[00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1#offpolicy-ratios|next chapter’s ratio analysis]]
explains this distinction.

These are the implementation choices in
[Tülu 3, §6.3](https://arxiv.org/html/2411.15124v5#S6.SS3), not requirements
of every RLVR algorithm.

<a id="rlvr-failures"></a>

## Открытые реализации

Для разбора проверки удобен
[модуль наград Open-R1](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py).
В `accuracy_reward` извлечение математического ответа и его сравнение
выполняются раздельно. Если не удалось разобрать сам эталон, возвращается
`None`, чтобы пропустить пример, а не наказать модель за повреждённые данные.
В том же модуле есть отдельные функции для формата и исполнения кода.
Наличие функции с названием `reasoning_steps_reward` не означает проверку
логики: она считает текстовые признаки шагов, а не доказывает их правильность.
Здесь зафиксирована версия репозитория `1416fa0` от 2 апреля 2026 года.

[Open Instruct](https://github.com/allenai/open-instruct) полезен для изучения
всего процесса Tülu 3: происхождения задач, генерации, вычисления наград,
оптимизации и оценки. Важно различать реализацию из статьи и текущую ветку
репозитория: новые методы могут менять параметры по умолчанию и устройство
обучения. Для воспроизведения результатов отправной точкой служит статья с
указанными в ней конфигурациями, а не только последняя версия кода.

Распределение работы между генерацией, проверкой и обучением отдельно
показывает [описание архитектуры PPO в verl](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html).
В большом запуске эти этапы имеют разную стоимость. Быстрый числовой
верификатор почти не задерживает обучение, а исполнение кода или проверка
доказательства могут стать основным ограничением пропускной способности.
Поэтому время ожидания проверки и долю её технических сбоев измеряют вместе
с числом генерируемых токенов.

<span id="практика-и-первоисточники"></span>

## Источники

### Объяснения и реализации

- Nathan Lambert — [RLHF & Post-Training, лекция 5](https://rlhfbook.com/teach/course/lec5-chap7/):
  различие обученной и проверяемой награды, цикл RLVR и его ограничения.
- Hugging Face — [Reasoning Course](https://huggingface.co/reasoning-course):
  упражнения по GRPO, TRL и DeepSeek-R1.
- DeepLearning.AI — [Reinforcement Learning from Human Feedback](https://www.deeplearning.ai/courses/reinforcement-learning-from-human-feedback):
  основы награды, стратегии и KL-дивергенции; RLVR меняет прежде всего источник награды.

### Первоисточники

- Cobbe et al., 2021 — [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).
- Lightman et al., 2023 — [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050).
- Lambert et al., 2024 — [Tülu 3](https://arxiv.org/abs/2411.15124), открытый
  многоэтапный процесс дообучения, закрепивший термин RLVR.
- DeepSeek-AI, 2025 — [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
- Chen et al., 2021 — [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374):
  оценка правильности программ и вывод pass@k.

Проверка даёт число для каждого ответа. В
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|главе о GRPO и DeepSeek-R1]]
разберём, как из наград нескольких ответов на один запрос получают
преимущество без отдельного критика.
