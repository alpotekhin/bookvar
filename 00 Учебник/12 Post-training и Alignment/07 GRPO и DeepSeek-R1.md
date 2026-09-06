---
title: GRPO и DeepSeek-R1
type: textbook-chapter
status: canonical
last_updated: 2026-09-07
aliases:
  - Group Relative Policy Optimization
  - R1-Zero и DeepSeek-R1
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers]]"
next:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation]]"
primary_sources:
  - https://arxiv.org/abs/2402.03300
  - https://arxiv.org/abs/2501.12948
  - https://arxiv.org/abs/2503.20783
  - https://arxiv.org/abs/2602.02710
  - https://arxiv.org/abs/2507.18071
  - https://arxiv.org/abs/2501.12599
  - https://arxiv.org/abs/2505.09388
source_unit_id:
  - assignment-05-reasoning-rl-setting-and-notation
  - assignment-05-deliverable-baseline-calcs-003
  - assignment-05-deliverable-baseline-calcs-004
  - assignment-05-deliverable-derive-difficulty-reweightings-023
  - assignment-05-deliverable-derive-difficulty-reweightings-024
  - assignment-05-deliverable-derive-difficulty-reweightings-025
  - assignment-05-deliverable-derive-surrogate-objectives-030
  - assignment-05-deliverable-think-about-advantage-normalization-026
  - assignment-05-deliverable-think-about-importance-reweighting-032
  - assignment-05-deliverable-think-about-length-normalization-019
  - assignment-05-deliverable-think-about-rft-022
  - assignment-05-on-policy-grpo-derivation
  - assignment-05-task-aggregate-loss-across-microbatch-constant
  - assignment-05-task-aggregate-loss-across-microbatch-sequence
  - assignment-05-task-baseline-calcs
  - assignment-05-task-compute-group-normalized-rewards-grpo
  - assignment-05-task-compute-group-normalized-rewards-drgrpo
  - assignment-05-task-compute-group-normalized-rewards-maxrl
  - assignment-05-task-derive-difficulty-reweightings
  - assignment-05-task-derive-surrogate-objectives
  - assignment-05-task-think-about-advantage-normalization
  - assignment-05-task-think-about-importance-reweighting
  - assignment-05-task-think-about-length-normalization
  - assignment-05-task-think-about-rft
  - lecture-16-grpo
  - lecture-16-grpo-bias-failures
  - lecture-16-kimi-case
  - lecture-16-qwen-case
  - lecture-16-r1-case
  - lecture-16-rlvr-experiments
---

# GRPO и DeepSeek-R1

<a id="rlvr-notation"></a>
<!-- source_unit_id: assignment-05-reasoning-rl-setting-and-notation -->

> [!abstract] После главы
> Вы сможете вручную вычислить относительные преимущества внутри группы и
> ограниченную функцию потерь, а также объяснить, какие расходы GRPO устраняет,
> а какие сохраняет. Во второй части мы восстановим полную схему обучения
> DeepSeek-R1 и увидим, почему её нельзя свести к фразе «GRPO на математических
> задачах».

## Где находится GRPO

RLVR определяет источник награды. **GRPO определяет способ оценки преимущества
и обновления стратегии.** Награда при этом может поступать от проверяющей
программы, обученной модели или их сочетания.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo/ppo-vs-grpo.png]]

*В PPO базовый уровень предсказывает отдельная модель ценности; в GRPO он
вычисляется по нескольким ответам на один запрос. Shao et al.,
[DeepSeekMath, Figure 4, с. 13](https://arxiv.org/pdf/2402.03300#page=13).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/post-training-courses/grpo-group-relative.png]]

*В GRPO несколько ответов на один запрос образуют группу: награды центрируются
относительно среднего группы и становятся преимуществами без отдельной функции
ценности. Источник: Nathan Lambert,
[RLHF & Post-Training, лекция 4, слайд 37](https://rlhfbook.com/teach/course/lec4-chap6-p2/#/36),
по Shao et al., 2024.*

## Сквозной численный пример

Для задачи $3x+6=18$ в предыдущей главе были получены четыре награды:

$$r=(1.1,0,1.1,0.1).$$

Среднее:

$$\bar r=\frac{1.1+0+1.1+0.1}{4}=0.575.$$

Используем стандартное отклонение по всей группе:

$$
s=\sqrt{\frac{(0.525)^2+(-0.575)^2+(0.525)^2+(-0.475)^2}{4}}
\approx0.5262.
$$

Преимущества:

$$
\hat A\approx(0.998,-1.093,0.998,-0.903).
$$

Если все четыре ответа получили 1.1, то $s=0$, а после вычитания среднего все
преимущества равны нулю. Даже полностью успешная, но однородная группа не даёт
относительного обучающего сигнала.

## От награды за ответ к обновлению отдельных токенов

<a id="grpo-estimator"></a>
<!-- source_unit_id: assignment-05-task-baseline-calcs -->
<!-- source_unit_id: assignment-05-deliverable-baseline-calcs-003 -->
<!-- source_unit_id: assignment-05-deliverable-baseline-calcs-004 -->
<!-- source_unit_id: lecture-16-grpo -->
<!-- source_unit_id: assignment-05-on-policy-grpo-derivation -->

Ответы породила сохранённая копия модели $\pi_{old}$. Для каждого токена ответа

$$
\rho_{i,t}(\theta)=
\exp(\log\pi_\theta(o_{i,t}|q,o_{i,<t})-
\log\pi_{old}(o_{i,t}|q,o_{i,<t})).
$$

Положительное преимущество требует увеличить вероятность всей
последовательности, а отрицательное — уменьшить. Ограничение, заимствованное из
PPO, сдерживает величину одного обновления:

$$
L^{clip}_{i,t}=\min\left(
\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right).
$$

### Ручной расчёт ограничения

Пусть $\epsilon=0.2$.

- Для хорошего $y_1$, $\hat A_1=0.998$, отношение вероятностей токена выросло до
  $\rho=1.35$. Неограниченное слагаемое равно $1.35\cdot0.998=1.347$, ограниченное —
  $1.2\cdot0.998=1.198$. Берём 1.198: чрезмерный рост не поощряется.
- Для плохого $y_2$, $\hat A_2=-1.093$, отношение упало до $0.70$.
  Неограниченное слагаемое равно $-0.765$, ограниченное — $0.8\cdot(-1.093)=-0.874$. Минимум равен
  $-0.874$: чрезмерное уменьшение также не даёт дополнительной выгоды.

Функция потерь равна отрицательному среднему $L^{clip}$, поэтому градиентный
спуск максимизирует ограниченную целевую функцию.

## Полная целевая функция

Один из распространённых вариантов имеет вид

$$
\mathcal L_{GRPO}=-\frac1G\sum_i\frac1{|o_i|}\sum_t
\left[L^{clip}_{i,t}-\beta\widehat D_{KL}(\pi_\theta||\pi_{ref})_{i,t}\right].
$$

В DeepSeekMath используется не полный перебор словаря, а положительная оценка
KL на фактически выбранном токене. Если
$u_{i,t}=\pi_{ref}(o_{i,t}\mid s_{i,t})/\pi_\theta(o_{i,t}\mid s_{i,t})$, то

$$
\widehat D_{KL,i,t}=u_{i,t}-\log u_{i,t}-1.
$$

Она равна нулю при $u=1$ и неотрицательна для любого $u>0$. В конкретной
реализации способ оценки KL нужно проверять отдельно: современные trainers
могут использовать другую оценку или вообще отключить это слагаемое.

Здесь участвуют **три разные версии модели**:

- $\pi_{old}$ — сохранённая копия, породившая ответы; она нужна в отношении вероятностей;
- $\pi_\theta$ — обновляемая стратегия;
- $\pi_{ref}$ — долговременная опорная модель, обычно исходная контрольная точка после SFT.

Сохранённая и опорная модели выполняют разные функции. Ограничение сдерживает
один цикл изменения относительно модели, породившей ответы, а KL-дивергенция —
накопленное отклонение от опорной модели.

## Минимальная реализация математики

```python
import torch

def group_advantages(rewards, eps=1e-4, scale=True):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if not scale:
        return centered
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return centered / (std + eps)

def clipped_token_loss(new_lp, old_lp, advantages, mask, clip_eps=0.2):
    # new_lp/old_lp: [batch, response_tokens]
    ratio = (new_lp - old_lp).exp()
    a = advantages[:, None]
    surrogate = torch.minimum(
        ratio * a,
        ratio.clamp(1 - clip_eps, 1 + clip_eps) * a,
    )
    per_sequence = (surrogate * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return -per_sequence.mean()
```

В рабочей реализации необходимо также исключать маской запрос и заполнители,
учитывать обрезанные ответы, добавлять слагаемое KL, синхронизировать параметры
между обучением и генерацией и разбивать пакет на мини-пакеты.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/post-training-courses/grpo-vs-ppo-implementation.png]]

*Какие компоненты исчезают при переходе от PPO к GRPO. Таблица показывает
точную экономию — модель ценности, её функция потерь и GAE, — но не говорит, что
генерация групп или вычисление вероятностей становятся бесплатными. Источник:
Nathan Lambert,
[RLHF & Post-Training, лекция 4, слайд 40](https://rlhfbook.com/teach/course/lec4-chap6-p2/#/39).*

## Почему нормировка по стандартному отклонению спорна

<a id="grpo-normalizations"></a>
<!-- source_unit_id: lecture-16-grpo-bias-failures -->
<!-- source_unit_id: assignment-05-task-compute-group-normalized-rewards-grpo -->
<!-- source_unit_id: assignment-05-task-aggregate-loss-across-microbatch-sequence -->

Деление на стандартное отклонение внутри группы делает масштабы разных запросов
похожими, но усиливает случайные различия в почти однородной группе. Кроме того,
возникает смещение, связанное со сложностью задачи.
Текущий [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/grpo_trainer)
позволяет `scale_rewards=False` или batch-level scaling.

Например, $r=(1,1,1,0)$ и $r=(0.51,0.50,0.50,0.50)$ после собственной
нормировки внутри группы дают преимущества сопоставимого масштаба, хотя во
втором случае абсолютная разница оценок ничтожна. Выбор способа нормировки
должен учитывать смысл награды.

## Нормировка по длине

Если сначала усреднить потери по токенам каждого ответа, короткая и длинная
последовательности получат одинаковый вес. Если сложить потери всех токенов и
разделить на их общее число, длинные ответы дадут больше наблюдений. Это разные
оценки одной целевой величины. Оригинальный DeepSeekMath и современные
реализации используют не всегда одинаковые правила; документация TRL отдельно
обсуждает отказ от исходной нормировки из-за смещения на уровне длины ответа.

<a id="drgrpo-rft-maxrl"></a>
<!-- source_unit_id: assignment-05-deliverable-think-about-length-normalization-019 -->
<!-- source_unit_id: assignment-05-deliverable-think-about-rft-022 -->
<!-- source_unit_id: assignment-05-deliverable-derive-difficulty-reweightings-023 -->
<!-- source_unit_id: assignment-05-deliverable-derive-difficulty-reweightings-024 -->
<!-- source_unit_id: assignment-05-deliverable-derive-difficulty-reweightings-025 -->
<!-- source_unit_id: assignment-05-deliverable-think-about-advantage-normalization-026 -->
<!-- source_unit_id: assignment-05-task-think-about-length-normalization -->
<!-- source_unit_id: assignment-05-task-compute-group-normalized-rewards-drgrpo -->
<!-- source_unit_id: assignment-05-task-aggregate-loss-across-microbatch-constant -->
<!-- source_unit_id: assignment-05-task-think-about-rft -->
<!-- source_unit_id: assignment-05-task-derive-difficulty-reweightings -->
<!-- source_unit_id: assignment-05-task-think-about-advantage-normalization -->
<!-- source_unit_id: assignment-05-task-compute-group-normalized-rewards-maxrl -->

## GRPO, Dr. GRPO, RFT и MaxRL: четыре разных взвешивания данных

Пусть пакет содержит $B$ запросов, для каждого сгенерировано $G$ ответов, а
$L_{ij}$ — число токенов ответа. В бинарной задаче $r_{ij}\in\{0,1\}$,

$$\mu_i=\frac1G\sum_{j=1}^{G}r_{ij},\qquad
\sigma_i=\sqrt{\frac1G\sum_j(r_{ij}-\mu_i)^2}.$$

У standard GRPO одновременно присутствуют **две нормировки**:

$$
\hat g_{GRPO}=\frac1{BG}\sum_{i,j}\frac1{L_{ij}}
\sum_{t=1}^{L_{ij}}
\frac{r_{ij}-\mu_i}{\sigma_i+\epsilon}
\nabla_\theta\log\pi_\theta(y_{ijt}\mid x_i,y_{ij,<t}).
$$

Деление на $\sigma_i$ меняет вес запроса: почти решённая или почти нерешаемая
задача может получить большой множитель из-за малого разброса. Деление на
$L_{ij}$ меняет вес токена: каждый ответ получает одинаковую массу независимо
от длины, поэтому токен короткого ответа весит больше токена длинного. Эти
эффекты нельзя обсуждать как одну «стабилизацию».

Есть и более тонкая деталь. Среднее группы $\mu_i$ вычислено с участием того же
ответа, градиент которого оно центрирует. Поэтому среднее группы не является
независимой от действия базовой оценкой. Если ответы сэмплированы независимо и
одинаково распределены, математическое ожидание оценки сохраняет направление
истинного градиента, но уменьшает его масштаб в $(G-1)/G$ раза. Базовая оценка,
посчитанная по всем остальным ответам группы (*leave-one-out*), устраняет этот
эффект конечного размера группы.

### Dr. GRPO

Dr. GRPO не делит преимущество на стандартное отклонение наград и заменяет
нормировку каждого ответа по его длине одним постоянным знаменателем:

$$
\hat g_{Dr}=\frac1Z\sum_{i,j}\sum_{t=1}^{L_{ij}}
(r_{ij}-\mu_i)\nabla_\theta\log\pi_\theta(y_{ijt}\mid x_i,y_{ij,<t}),
\qquad Z=BGL_{max}.
$$

Именно $Z=BGL_{max}$ использует Assignment 5 ($L_{max}=512$ в полном
референсном запуске). Это не оценка фактического числа токенов пакета, а
фиксированный масштаб, благодаря которому длина сгенерированных ответов не
меняет знаменатель от шага к шагу.

### Rejection fine-tuning, или обучение на отобранных решениях

RFT сохраняет только успешные сгенерированные решения и делает на них обычный
шаг SFT:

$$
\hat g_{RFT}=\frac1Z\sum_{i,j}\mathbf1[r_{ij}=1]
\sum_t\nabla_\theta\log\pi_\theta(y_{ijt}\mid x_i,y_{ij,<t}).
$$

В коде это соответствует `baseline="none"`,
`advantage_normalizer="none"`, `loss_normalization="constant"`. Неверные
ответы имеют нулевой вес и могут быть исключены ещё до прямого прохода. RFT
проще и часто дешевле, но после отбора сводится к обучению по сохранённым
траекториям: метод не понижает вероятность конкретных неверных ответов явным
отрицательным членом и наследует ошибки фильтра.

### MaxRL

MaxRL делит центрированную награду на среднюю успешность группы:

$$
\hat g_{MaxRL}=\frac1Z\sum_{i,j,t}
\frac{r_{ij}-\mu_i}{\mu_i+\epsilon}
\nabla_\theta\log\pi_\theta(y_{ijt}\mid x_i,y_{ij,<t}).
$$

Приближённо это повышает вес запросов, которые текущая модель решает редко. При
малой, но ненулевой $\mu_i$ множитель становится большим, поэтому в отчёте
нужны квантили весов и нормы градиента, а $\epsilon$ является частью
спецификации. Если вся группа получила ноль, числитель также равен нулю и
обучающего сигнала нет: $\epsilon$ предотвращает деление на ноль, но не создаёт
информацию. В статье
В статье о MaxRL функция потерь нормируется фактическим числом токенов пакета.
В задании 5 намеренно оставлен постоянный $Z$, чтобы в абляции менялся только
способ нормировки преимущества;
результат такого варианта нельзя подписывать просто «MaxRL» без оговорки.

### Что сравнивает честная абляция

Stanford фиксирует запросы, бюджет генерации, проверяющую программу,
инициализацию, число обновлений и процедуру оценивания, а затем сравнивает:

| вариант | базовая оценка | нормировка преимущества | нормировка функции потерь |
|---|---|---|---|
| `GRPO_constant` | mean | std | constant |
| `Dr_GRPO` | mean | none | constant |
| `RFT` | none | none | constant |
| `MaxRL_course` | mean | mean | constant |

Темп обучения предварительно настраивается для базового варианта. Поэтому победа
нового метода при этих настройках является содержательным результатом, а
проигрыш не доказывает его принципиальную слабость: выбранный темп обучения мог
оказаться для него неподходящим. Полный опыт курса требует четыре запуска с
разными начальными значениями генератора случайных чисел;
сокращённый опыт обязан показывать каждый запуск и разброс, а не одну лучшую
кривую.

## Штраф по KL не определяется одним названием GRPO

DeepSeekMath включает reference KL. В актуальном TRL `beta=0.0` по умолчанию:
в нескольких последующих схемах обучения рассуждению удалось обойтись без
явного штрафа. Поэтому параметры эксперимента следует записывать явно:

```yaml
algorithm: grpo
group_size: 8
scale_rewards: false
beta: 0.0
loss_aggregation: token_mean
```

Одного названия GRPO недостаточно, чтобы воспроизвести эксперимент.

## Минимальный пример запуска в TRL

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

data = load_dataset("trl-lib/DeepMath-103K", split="train")
args = GRPOConfig(
    output_dir="qwen-grpo-demo",
    num_generations=8,
    max_completion_length=512,
    temperature=0.8,
    learning_rate=1e-6,
    beta=0.0,
    scale_rewards=False,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=args,
    train_dataset=data,
    reward_funcs=accuracy_reward,
)
trainer.train()
```

Это учебный запуск, а не способ воспроизвести R1. Даже небольшой пример на
модели 0,5B требует значительных вычислений; основная часть расходов приходится
на генерацию $G$ ответов для каждого запроса.

## Несовпадение моделей генерации и обучения

Ответы часто генерирует vLLM, а логарифмы вероятностей для вычисления градиента
получает Transformers. Из-за различий в точности вычислений и ядрах
$\pi_{inference}\ne\pi_{train}$. Формально обновление перестаёт использовать
данные строго из текущей стратегии:

$$y\sim\pi_{inference},\qquad \nabla\log\pi_{train}(y)R(y).$$

[TRL](https://huggingface.co/docs/trl/main/grpo_trainer) предлагает усечённое
или маскированное перевзвешивание вероятностей. При раздельном запуске генератор
и процесс обучения размещают на разных ускорителях; при совместном размещении
память делят параметры модели, кеш ключей и значений и состояние оптимизатора.

<a id="offpolicy-ratios"></a>
<!-- source_unit_id: assignment-05-deliverable-derive-surrogate-objectives-030 -->
<!-- source_unit_id: assignment-05-deliverable-think-about-importance-reweighting-032 -->
<!-- source_unit_id: assignment-05-task-derive-surrogate-objectives -->
<!-- source_unit_id: assignment-05-task-think-about-importance-reweighting -->

## Когда один пакет генераций используют несколько раз

On-policy означает, что ответы сэмплированы той же policy, для которой
оценивается градиент. Если после генерации разбить 256 ответов на minibatches по
8 и сделать 32 последовательных обновления, то только первое обновление видит
текущие данные. Для остальных стратегия, породившая ответы, $\pi_0$, уже
устарела.

Точное importance correction для целого ответа равно

$$
w(y)=\frac{\pi_\theta(y\mid x)}{\pi_0(y\mid x)}
=\prod_{t=1}^{L}\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_0(y_t\mid x,y_{<t})}.
$$

Произведение корректирует всё распределение последовательности, но его
дисперсия может стремительно расти с длиной. Token-level GRPO заменяет его
$w_t=\pi_\theta(y_t\mid s_t)/\pi_0(y_t\mid s_t)$ и применяет PPO clipping к
каждому токену. Это уменьшает variance, но меняет objective: в слагаемом для
позиции $t$ prefix и suffix всё ещё распределены как $\pi_0$, а не
$\pi_\theta$. Смещение растёт вместе с policy lag.

Для $A\ge0$ clipping прекращает усиливать действие после $w_t>1+\epsilon$; для
$A<0$ — прекращает ослаблять его после $w_t<1-\epsilon$. Поэтому логируют не
только средний ratio, но и долю clipped positive/negative токенов, хвосты ratio,
возраст стратегии и effective sample size. Без этих данных фраза «32× off-policy
быстрее» смешивает экономию генерации с потерей полезного градиента.

<a id="gspo"></a>

## GSPO: одно отношение вероятностей на весь ответ

GSPO использует геометрическое среднее отношений вероятностей токенов:

$$
s(y)=\exp\left[\frac1L\sum_{t=1}^{L}
\bigl(\log\pi_\theta(y_t\mid s_t)-\log\pi_0(y_t\mid s_t)\bigr)\right].
$$

Один $s(y)$ затем входит в clipped surrogate для всего ответа:

$$
J_{GSPO}=\frac1{BG}\sum_{i,j}
\min\left(A_{ij}s_{ij},
A_{ij}\operatorname{clip}(s_{ij},1-\epsilon,1+\epsilon)\right).
$$

Вычислять произведение вероятностей напрямую нельзя: log-ratios суммируют в
log-space, причём `response_mask` исключает prompt и padding как из суммы, так
и из $L$. Геометрическое среднее не восстанавливает точный sequence importance
ratio — корень степени $1/L$ намеренно добавляет bias ради меньшей variance.
Кроме того, его градиент естественно содержит $1/L$; чтобы совместить GSPO с
constant-normalized Dr. GRPO, степень и нормировку пришлось бы определить
заново. Assignment 5 проверяет стандартный sequence-normalized вариант.

Практический вывод не сводится к рейтингу методов. Без перевзвешивания
дисперсия мала, но смещение быстро растёт по мере устаревания данных; точное
отношение вероятностей последовательности формально корректно, но часто имеет
неприемлемую дисперсию. Token clipping и GSPO занимают разные промежуточные
точки. Выбор подтверждают опытом с одинаковыми seed, числом сгенерированных
токенов и обновлений.

## R1-Zero и R1 — два разных эксперимента

<a id="reasoning-case-studies"></a>
<!-- source_unit_id: lecture-16-r1-case -->
<!-- source_unit_id: lecture-16-rlvr-experiments -->
<!-- source_unit_id: lecture-16-kimi-case -->
<!-- source_unit_id: lecture-16-qwen-case -->

### DeepSeek-R1-Zero

R1-Zero начинает обучение с **DeepSeek-V3-Base**, не проводя непосредственно
перед этапом RL обучения на размеченных цепочках рассуждений. Проверяемые награды
улучшили решение математических задач и сделали более частыми длинные ответы,
самопроверку и перебор вариантов. Это показывает, что написанные человеком
цепочки рассуждений не являются обязательной непосредственной целью для
возникновения полезных траекторий.

Однако слово «Zero» не означает отсутствия человеческих данных: базовая модель
уже была обучена на огромном корпусе. Одно обучение с подкреплением также не
обеспечило удобного для пользователя ответа: тексты страдали от плохой
читаемости и смешения языков.

### Полный DeepSeek-R1

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/deepseek-r1-figure1-hq.png]]

*Официальная схема показывает две разные ветви эксперимента. Верхняя ветвь
ведёт от DeepSeek-V3-Base непосредственно к R1-Zero через обучение с
подкреплением. Нижняя ветвь R1 начинается с небольшого набора начальных
примеров, проходит отдельный этап RL для рассуждений, создаёт новые данные
посредством отбора успешных траекторий, снова выполняет SFT на смеси и лишь
затем переходит к заключительному RL. Источник: DeepSeek-AI,
[DeepSeek-R1, Figure 2, с. 6](https://arxiv.org/pdf/2501.12948#page=6).*

1. **Начальный SFT** задаёт читаемый формат и устойчивое исходное поведение.
2. **RL на задачах рассуждения** исследует решения проверяемых задач.
3. **Отбор успешных генераций** превращает найденные решения в новый набор данных.
4. **SFT на смеси** восстанавливает общие диалоговые способности.
5. **Заключительный RL** сочетает задачи рассуждения, человеческие предпочтения
   и требования безопасности.

Следовательно, формула «R1 = GRPO» игнорирует цикл пополнения данных и два этапа
SFT, то есть значительную часть практической схемы обучения.

### Kimi k1.5: RL требует не только objective, но и производственного контура

Kimi k1.5 дополняет картину R1 деталями, которые легко потерять при чтении одной
формулы. Для математики авторы строят проверяемую награду, для программирования
генерируют тесты, а для математических ответов обучают модель проверки
эквивалентности. Генерации
длинны и неодинаковы, поэтому производительность определяется не только
backward pass. Организация генерации, переключение между inference- и
training-системами, балансировка длин и передача новых весов определяют, можно
ли вообще выполнить выбранный алгоритм с приемлемой загрузкой ускорителей.

Отдельный механизм управляет длиной reasoning. Если вознаграждать только
правильность, более длинный поиск иногда получает преимущество просто потому,
что у него больше попыток. Kimi сначала развивает long-CoT policy, затем
применяет length control и short-CoT стадии. Это не опровержение RLVR, а смена
целевой функции: качество рассматривается вместе с вычислительной ценой ответа.
Числа из отчёта относятся к конкретным моделям и инфраструктуре; переносимым
является разделение correctness reward, length objective и rollout system.

### Qwen3: несколько стадий вместо одной кнопки «reasoning»

В Qwen3 способность к рассуждению также строится последовательностью стадий:
cold-start/long-CoT data, reasoning RL, объединение thinking и non-thinking
режимов, затем общий post-training. Agentic и general RL следуют за узкой
математической стадией. Поэтому результат нельзя приписывать одному GRPO loss
или объёму одного SFT-набора.

Эти примеры нужны не как каталог семейств. Они показывают повторяющуюся
структуру: начальная стратегия определяет доступные траектории; проверяющая
программа задаёт границу достижимого сигнала; online RL исследует; отбор и
дистилляция закрепляют найденное; заключительный alignment возвращает общие способности.
Каждый переход меняет распределение данных, поэтому после него повторяют
evaluation на reasoning, general capabilities, style и safety.

## Что означает наблюдаемый момент самокоррекции

В отчёте приведён пример, где модель останавливается и пересматривает решение.
Это интересная наблюдаемая особенность текста, но она не доказывает сознание,
наличие универсального внутреннего алгоритма поиска или истинность каждого
токена рассуждения. Из опыта можно сделать более узкий вывод: оптимизация по
конечному результату повысила вероятность последовательностей, содержащих
полезные проверки и исправления на данном распределении задач.

## Характерные сбои и наблюдаемые показатели

| симптом | возможная причина | что проверить |
|---|---|---|
| награда растёт, проверочное качество — нет | обход проверки или утечка данных | новый генератор задач, намеренные атаки |
| длина быстро растёт | корреляция награды с длиной, слабое завершение | правильность при фиксированной длине |
| энтропия падает | слишком большой шаг или малое разнообразие | энтропия, число уникальных ответов |
| у 90% групп нулевая дисперсия | задачи слишком лёгкие или слишком трудные | программа сложности по доле успеха |
| KL резко меняется | устаревшие ответы или чрезмерное обновление | версии модели, отношения вероятностей |
| функция потерь становится NaN | почти нулевое отклонение или ошибочные вероятности | $\epsilon$, проверки конечности, маски |
| формат улучшается без правильности | слишком большой вес вспомогательной награды | отдельные составляющие награды |

Минимальный набор наблюдений включает составляющие обучающей награды,
pass@1 и pass@$k$ на отложенных задачах, долю неоднородных групп, квантили длины
ответов, энтропию, KL-дивергенцию, долю ограниченных обновлений, хвосты отношений
вероятностей, достижение токена конца и число сгенерированных токенов на одно
успешное решение.

## Открытые реализации

- [Open-R1](https://github.com/huggingface/open-r1): наиболее читаемый путь
  от SFT до GRPO; есть схемы для одного узла и распределённого генератора vLLM.
- [Open Instruct](https://github.com/allenai/open-instruct): Tülu 3 RLVR и
  открытые промежуточные контрольные точки.
- [verl](https://github.com/verl-project/verl): Ray workers, FSDP/Megatron,
  vLLM/SGLang и конфигурации GRPO/DAPO для крупных запусков.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): Ray + vLLM + DeepSpeed,
  динамический отбор задач и несколько алгоритмов обучения с подкреплением.

В Open-R1 следует учитывать важную деталь: шаблон диалога дистиллированных
моделей DeepSeek заранее добавляет `<think>` и может скрывать начало блока
рассуждений. Для проверки формата шаблон нужно переопределить, иначе награда
будет оптимизировать ошибку сериализации.

## Практика и первоисточники

Полное сопоставление standard GRPO, Dr. GRPO, RFT, учебного MaxRL и GSPO
вынесено в [[02 Areas/ML & DL/06 Практика/24 Post-training и RLVR для математического reasoning]].

### Задания

1. Воспроизведите ручной пример с помощью тензорных операций и получите преимущества с точностью
   $10^{-3}$.
2. Постройте график $P(\text{mixed group})$ для $G=2,4,8,16$ и $p\in[0,1]$.
3. Сравните `scale_rewards=True/False` на искусственных непрерывных наградах.
4. Напишите отдельные тесты ограничения для положительного и отрицательного
   преимущества: знак меняет активную границу.
5. Запустите небольшой опыт в TRL и сохраните исходные ответы до и после обучения.
6. Повторите сравнение с SFT на успешных генерациях при одинаковом числе обучающих токенов.
7. Измените только способ учёта длины и сравните длину ответа с pass@1.
8. В отчёте укажите текущую, сохранённую и опорную модели, версию проверяющей
   программы и механизм генерации. Без этих сведений опыт невоспроизводим.

### Объяснения и реализации

- Hugging Face — [Reasoning Course](https://huggingface.co/reasoning-course) и
  [GRPOTrainer guide](https://huggingface.co/docs/trl/main/grpo_trainer).
- Nathan Lambert — [RLHF Book course](https://rlhfbook.com/course), лекции по
  градиентам стратегии, RLVR и современному дообучению; доступны слайды.
- [Open-R1 blog](https://huggingface.co/blog/open-r1) — реконструкция
  неопубликованных деталей данных и кода R1, а также схема проекта.

### Первоисточники

- Shao et al., 2024 — [DeepSeekMath](https://arxiv.org/abs/2402.03300):
  исходная публикация GRPO.
- DeepSeek-AI, 2025 — [DeepSeek-R1](https://arxiv.org/abs/2501.12948):
  R1-Zero, multi-stage R1, результаты и distillation.
- Liu et al., 2025 — [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783):
  Dr. GRPO, анализ standard-deviation и length normalization.
- Tajwar et al., 2026 — [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710):
  исходная постановка MaxRL; её normalizer отличается от учебной абляции A5.
- Zheng et al., 2025 — [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071):
  sequence-level geometric-mean importance ratio.
- Kimi Team, 2025 — [Kimi k1.5](https://arxiv.org/abs/2501.12599), и Yang et
  al., 2025 — [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388):
  многостадийные reasoning recipes и связь RL с inference cost.
- Ahmadian et al., 2024 — [Back to Basics: REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740):
  полезный контекст о critic-free estimators.

> [!summary]
> GRPO устраняет отдельную модель ценности, но не расходы на генерацию.
> DeepSeek-R1 — не название оптимизатора, а многоэтапная схема создания данных
> и обучения модели.
