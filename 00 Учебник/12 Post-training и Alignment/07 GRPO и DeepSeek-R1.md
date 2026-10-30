---
title: GRPO и DeepSeek-R1
type: textbook-chapter
status: canonical
last_updated: 2026-10-30
last_verified: 2026-10-25
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

Одна и та же модель может решить задачу с первой попытки, ошибиться во второй
и получить правильный ответ другим способом в третьей. Если все три ответа
проверены, их можно сравнить между собой: какие решения оказались лучше
обычного результата этой модели на **этом же запросе**? GRPO использует такое
сравнение вместо предсказания функции ценности, которое мы разбирали в главе
о PPO.

## Где находится GRPO

RLVR определяет источник награды, а GRPO — способ оценить преимущество ответа
и обновить параметры модели. Это разные части обучения: GRPO может работать
и с проверяющей программой, и с обученной моделью награды. Метод впервые
описан в [DeepSeekMath, §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1);
в DeepSeek-R1 он стал частью значительно более широкой схемы подготовки данных
и дообучения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ppo-2026-10-24/ppo-framework.png]]

*Nathan Lambert, [RLHF Book — PPO](https://rlhfbook.com/c/06-policy-gradients).
GAE combines value predictions with rewards that include the KL penalty.
The value model and policy are trained; the reference and reward models
are frozen in this schematic. The GRPO diagram below replaces the value
model and GAE with a comparison among responses to the same prompt.
[Complete original PNG](https://github.com/natolambert/rlhf-book/blob/3f573487c58548c16a548cd5fb8979a0189384c7/book/images/ppo_tikz.png),
unchanged; [MIT license notice](/licenses/rlhf-book-MIT.txt).
The original paper comparison is linked as
[DeepSeekMath, Figure 4](https://arxiv.org/pdf/2402.03300#page=13).*

В GRPO группу составляют ответы на один запрос, а не случайные ответы из всего пакета.
Иначе простое уравнение пришлось бы сравнивать с олимпиадной задачей: низкая
награда могла бы означать высокую сложность запроса, а не плохой выбор ответа.
Сравнение внутри группы частично отделяет эти причины.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo-review-2026-09/group-advantages.png]]

*Модель порождает несколько ответов, оценщик присваивает им награды, а среднее
и стандартное отклонение вычисляются отдельно для этой группы. Опорная модель
нужна для штрафа по KL, а не для вычисления средней награды. Nathan Lambert,
[RLHF Book, Reinforcement Learning](https://rlhfbook.com/c/06-policy-gradients),
[исходный рисунок](https://github.com/natolambert/rlhf-book/blob/3f573487c58548c16a548cd5fb8979a0189384c7/book/images/grpo_tikz.png), MIT.*

Пусть $G$ ответов получили награды $r_1,\ldots,r_G$. Сначала вычисляется
среднее $\bar r$, затем из каждой награды вычитается это среднее. В исходном
варианте разность дополнительно делят на стандартное отклонение $s$:

$$
\hat A_i=\frac{r_i-\bar r}{s+\varepsilon_{norm}}.
$$

Положительное $\hat A_i$ означает «лучше среднего в этой группе», а не
«объективно правильный ответ». Например, если все решения неверны, но один
ответ соблюдает вознаграждаемый формат, его преимущество может оказаться
положительным. Поэтому содержание функции награды из предыдущей главы остаётся
существенным и после перехода к групповому сравнению.

## Сквозной численный пример

Для задачи $3x+6=18$ в предыдущей главе были получены четыре награды:

$$r=(1.1,0,1.1,0.1).$$

Среднее:

$$\bar r=\frac{1.1+0+1.1+0.1}{4}=0.575.$$

Используем стандартное отклонение по всей группе:

$$
s=\sqrt{\frac{1.1075}{4}}\approx0.5262.
$$

Здесь 1.1075 — сумма квадратов четырёх отклонений от среднего:
$0.525$, $-0.575$, $0.525$ и $-0.475$.

Преимущества:

$$
\hat A\approx(0.998,-1.093,0.998,-0.903).
$$

Если все четыре ответа получили 1.1, то $s=0$, а после вычитания среднего все
преимущества равны нулю. Даже полностью успешная, но однородная группа не даёт
относительного обучающего сигнала **от награды**. Штраф по KL при этом может
по-прежнему менять параметры. При $G=1$ центрирование также всегда даёт ноль:
для сравнения нужны как минимум два ответа.

Здесь знаменатель дисперсии равен $G$, а не $G-1$. Обе конвенции встречаются
в коде; они дают разные числа при маленькой группе. В примере $\varepsilon_{norm}$
пренебрежимо мало, а в коде ниже равно $10^{-4}$. Небольшое различие в последнем
знаке результата связано именно с этим слагаемым.

## От награды за ответ к обновлению отдельных токенов

<a id="grpo-estimator"></a>
<!-- source_unit_id: assignment-05-task-baseline-calcs -->
<!-- source_unit_id: assignment-05-deliverable-baseline-calcs-003 -->
<!-- source_unit_id: assignment-05-deliverable-baseline-calcs-004 -->
<!-- source_unit_id: lecture-16-grpo -->
<!-- source_unit_id: assignment-05-on-policy-grpo-derivation -->

Ответы породила сохранённая копия модели $\pi_{old}$. Обозначим префикс
перед токеном через $s_{i,t}=(q,o_{i,<t})$, а его логарифм вероятности — через
$l_{i,t}(\theta)=\log\pi_\theta(o_{i,t}\mid s_{i,t})$;
$l_{i,t}(old)$ обозначает тот же логарифм для модели, породившей ответы.
Для каждого токена ответа вычисляется отношение

$$
\rho_{i,t}(\theta)=\exp\bigl(l_{i,t}(\theta)-l_{i,t}(old)\bigr).
$$

Почему награда за целый ответ участвует в обновлении каждого токена? Вероятность
ответа раскладывается в произведение условных вероятностей, а её логарифм —
в сумму:

$$
\nabla_\theta\log\pi_\theta(o_i\mid q)
=\sum_t\nabla_\theta l_{i,t}(\theta).
$$

В оценке REINFORCE вся эта сумма умножается на одно преимущество ответа.
Это даёт направленный сигнал для последовательности, но **не** помечает
каждый её шаг как правильный или неправильный. Полезный промежуточный шаг
в неудачном ответе получает тот же знак, что и ошибка; локализация ошибок
остаётся отдельной задачей.

GRPO сохраняет эту общую оценку $\hat A_i$, но применяет отношение вероятностей
и ограничение PPO к каждому токену. Это вспомогательная целевая функция
для обновления по уже собранным ответам:

$$
\begin{aligned}
\rho^c_{i,t}&=\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon),\\
L^{clip}_{i,t}&=\min(\rho_{i,t}\hat A_i,\rho^c_{i,t}\hat A_i).
\end{aligned}
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
спуск максимизирует ограниченную целевую функцию. Ограничение одностороннее:
если вероятность токена из успешного ответа, напротив, упала до $0.7$ от прежней,
слагаемое $0.7\hat A$ остаётся активным и подталкивает её вверх. Нельзя
просто обрезать все отношения и считать, что за пределами интервала градиент
всегда равен нулю. Ограничение функции потерь также не гарантирует, что
фактические отношения после шага останутся внутри этого интервала.

## Полная целевая функция

Обозначим оценку KL для выбранного токена через $\hat k_{i,t}$.
В исходном GRPO сначала усредняют слагаемые по токенам ответа, затем по
ответам группы:

$$
\begin{aligned}
\mathcal L_{GRPO}&=\frac1G\sum_i\mathcal L_i,\\
\mathcal L_i&=-\frac1{|o_i|}\sum_t
\left[L^{clip}_{i,t}-\beta\hat k_{i,t}\right].
\end{aligned}
$$

В DeepSeekMath используется не полный перебор словаря, а неотрицательная оценка
KL на фактически выбранном токене. Если
$u_{i,t}=\pi_{ref}(o_{i,t}\mid s_{i,t})/\pi_\theta(o_{i,t}\mid s_{i,t})$, то

$$
\hat k_{i,t}=u_{i,t}-\log u_{i,t}-1.
$$

Она равна нулю при $u=1$ и неотрицательна для любого $u>0$. Например, при
$u=2$ получаем $2-\log2-1\approx0.3069$. Неотрицательность следует из
неравенства $\log u\le u-1$.

Равенство её математического ожидания KL-дивергенции требует отдельного
условия. Для фиксированного префикса токен должен быть выбран из **текущей**
стратегии $\pi_\theta$:

$$
\begin{aligned}
\mathbb E_{o\sim\pi_\theta}[u-1]&=0,\\
\mathbb E_{o\sim\pi_\theta}[-\log u]
&=D_{KL}(\pi_\theta\|\pi_{ref}).
\end{aligned}
$$

После нескольких обновлений сохранённые токены всё ещё распределены по
$\pi_{old}$, поэтому их среднее без поправки уже не обязано совпадать с этим
KL. Кроме того, равенство ожиданий не делает автоматически равными градиенты
разных реализаций штрафа. Название «оценка KL» не заменяет проверки того,
откуда взята выборка и через какие величины проходит градиент.

Например, пусть текущие вероятности двух токенов равны $(0.6,0.4)$, а
опорные — $(0.5,0.5)$. Если точно усреднить указанную оценку по текущим
вероятностям, получится $D_{KL}=0.020136$. Но при фиксированных весах выборки
производная одной только формулы $u-\log u-1$ по логиту первого токена равна
0.1; производная самого KL равна примерно 0.097312. Вторая учитывает также
изменение вероятностей, с которыми взяты токены. Это различие между
производной ожидания и ожиданием частной производной, а не численная ошибка.

```python
import math
import torch

kl_logit = torch.tensor(math.log(0.6 / 0.4), dtype=torch.float64, requires_grad=True)
kl_policy = torch.stack((kl_logit.sigmoid(), 1 - kl_logit.sigmoid()))
kl_reference = torch.tensor([0.5, 0.5], dtype=torch.float64)
kl_ratio = kl_reference / kl_policy
kl_estimate = kl_ratio - kl_ratio.log() - 1
exact_kl = (kl_policy * (kl_policy / kl_reference).log()).sum()
sampled_kl = (kl_policy.detach() * kl_estimate).sum()
exact_kl_gradient, = torch.autograd.grad(exact_kl, kl_logit, retain_graph=True)
sampled_kl_gradient, = torch.autograd.grad(sampled_kl, kl_logit)
print(f"KL: {exact_kl.item():.6f}; estimator: {sampled_kl.item():.6f}")
print(f"gradients: {exact_kl_gradient.item():.6f}, {sampled_kl_gradient.item():.6f}")
```

Здесь участвуют **три разные версии модели**:

- $\pi_{old}$ — сохранённая копия, породившая ответы; она нужна в отношении вероятностей;
- $\pi_\theta$ — обновляемая стратегия;
- $\pi_{ref}$ — опорная модель для штрафа по KL; её обновляют по отдельному правилу или оставляют фиксированной.

Сохранённая и опорная модели выполняют разные функции. Ограничение сдерживает
один цикл изменения относительно модели, породившей ответы, а KL-дивергенция —
накопленное отклонение от опорной модели. Нельзя автоматически считать её
неизменной на всём протяжении обучения: в расширенном отчёте R1-Zero
описана замена опорной модели каждые 400 шагов. Это не то же самое,
что сохранение $\pi_{old}$ для очередной группы генераций.

## Минимальная реализация математики

```python
import torch

@torch.no_grad()
def group_advantages(rewards, eps=1e-4, scale=True):
    rewards = torch.as_tensor(rewards)
    if rewards.is_complex():
        raise ValueError("rewards must be real-valued")
    if rewards.dtype not in (torch.float32, torch.float64):
        rewards = rewards.float()
    if rewards.ndim not in (1, 2) or rewards.shape[-1] < 2 or rewards.numel() == 0:
        raise ValueError("rewards must have shape [G] or [B, G], with G >= 2")
    if not torch.isfinite(rewards).all() or not 0 < eps < float("inf"):
        raise ValueError("finite rewards and a positive finite eps are required")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if not scale:
        return centered
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return centered / (std + eps)

def clipped_token_loss(new_lp, old_lp, advantages, mask, clip_eps=0.2):
    # Входы: [B, T], [B, T], [B], [B, T]. Маска отмечает только токены ответа.
    if new_lp.ndim != 2 or old_lp.shape != new_lp.shape or mask.shape != new_lp.shape:
        raise ValueError("new_lp, old_lp and mask must have the same [B, T] shape")
    if new_lp.dtype not in (torch.float32, torch.float64):
        raise ValueError("use float32 or float64 log probabilities")
    if mask.dtype != torch.bool or mask.device != new_lp.device:
        raise ValueError("mask must be boolean and on the same device as new_lp")
    lengths = mask.sum(-1)
    if new_lp.shape[0] == 0 or (lengths == 0).any() or not 0 < clip_eps < 1:
        raise ValueError("nonempty responses and 0 < clip_eps < 1 are required")
    advantages = torch.as_tensor(advantages, device=new_lp.device)
    if old_lp.is_complex() or advantages.is_complex():
        raise ValueError("log probabilities and advantages must be real-valued")
    old_lp = old_lp.detach().to(new_lp)
    advantages = advantages.to(new_lp).detach()
    if advantages.shape != (new_lp.shape[0],):
        raise ValueError("advantages must have shape [B]")
    if not (torch.isfinite(new_lp[mask]).all()
            and torch.isfinite(old_lp[mask]).all()
            and torch.isfinite(advantages).all()):
        raise ValueError("valid-token log probabilities and advantages must be finite")
    # Убираем заполнители до exp: произведение NaN на ноль всё ещё равно NaN.
    log_ratio = new_lp.masked_fill(~mask, 0) - old_lp.masked_fill(~mask, 0)
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("probability ratio overflow: check the policy update")
    a = advantages[:, None]
    surrogate = torch.minimum(
        ratio * a,
        ratio.clamp(1 - clip_eps, 1 + clip_eps) * a,
    )
    per_sequence = surrogate.masked_fill(~mask, 0).sum(-1) / lengths
    return -per_sequence.mean()
```

Здесь `B` — число **ответов**, а группы уже преобразованы в отдельные
преимущества. Для наград формы `[число запросов, G]` перед вызовом второй
функции нужен `advantages.reshape(-1)` с тем же порядком ответов. Маска
включает токены ответа, в том числе сгенерированный EOS, но исключает запрос
и заполнители. Логарифмы вероятностей должны относиться к одним и тем же
выбранным токенам после сдвига предсказаний на одну позицию.

`old_lp`, награды и преимущества здесь — фиксированные данные. Градиент
проходит только через новые логарифмы вероятностей. Код реализует исходное
усреднение по ответам без KL; он не является полным тренером. Обработка
обрезанных ответов, штраф, синхронизация весов с генератором и накопление
градиента остаются отдельными частями системы.

В Assignment 5 CS336 текстовая формула также записана с делением дисперсии
на $G$, но для реализации задания указано стандартное поведение `torch.std`
с поправкой Бесселя. В приведённом выше учебном коде намеренно выбрано
`unbiased=False`, чтобы результат совпадал с ручным расчётом. Для тестов
задания следует использовать именно его соглашение, а не менять числа
до случайного совпадения.

Экономия GRPO заключается в отсутствии отдельной модели ценности, её
оптимизатора и вычисления GAE. Расходы на генерацию не исчезают: для каждого
запроса нужны $G$ ответов, их оценки и логарифмы вероятностей. Если ответы
длинные, эта часть может стоить дороже сэкономленного прохода модели ценности.

Проверим функции на двух группах. В первой используются награды за уравнение,
во второй все ответы одинаково успешны. Во втором вызове один токен успешного
ответа уже превысил верхнюю границу, а один токен неуспешного — опустился ниже
нижней. Их градиенты обнуляются; два оставшихся продолжают участвовать.

```python
group_rewards = torch.tensor([[1.1, 0., 1.1, .1], [1., 1., 1., 1.]], dtype=torch.float64)
group_a = group_advantages(group_rewards)
assert torch.equal(group_a[1], torch.zeros(4, dtype=torch.float64))
old_token_lp = torch.full((2, 2), -2., dtype=torch.float64)
new_token_lp = (old_token_lp + torch.tensor([[1.35, .7], [.7, 1.35]], dtype=torch.float64).log()).requires_grad_()
token_mask = torch.ones((2, 2), dtype=torch.bool)
clip_example_loss = clipped_token_loss(new_token_lp, old_token_lp, [1., -1.], token_mask)
clip_example_loss.backward()
print([round(v, 6) for v in group_a[0].tolist()])
print(f"clipped loss: {clip_example_loss.item():.6f}")
print(new_token_lp.grad.tolist())
```

### Когда награда дана за отдельные шаги

В [DeepSeekMath, §4.1.3](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS3)
разобран и другой вариант GRPO. Модель награды оценивает завершение каждого
шага решения. Оценки всех шагов из группы ответов нормируют совместно,
а преимущество токена получают суммированием нормированных оценок шагов,
которые заканчиваются не раньше этого токена. По мере продвижения по ответу
уже пройденные шаги выходят из этой суммы.

Пусть два ответа содержат по два шага с оценками $(1,0)$ и $(1,-1)$.
Общее среднее четырёх оценок равно 0.25, стандартное отклонение — около 0.829156.
После нормировки получаются $(0.904534,-0.301511)$ и
$(0.904534,-1.507557)$. Токены первого шага первого ответа получают сумму
0.603023, второго шага — только -0.301511. Во втором ответе соответствующие
величины равны -0.603023 и -1.507557. Награды здесь условные, а не результаты
запуска обученного оценщика.

```python
step_rewards = torch.tensor([[1., 0.], [1., -1.]], dtype=torch.float64)
normalized_steps = group_advantages(step_rewards.flatten(), eps=1e-8).reshape(2, 2)
step_advantages = normalized_steps.flip(-1).cumsum(-1).flip(-1)
# Для примера каждый шаг занимает два токена; оценка приходит на его конце.
process_token_advantages = step_advantages.repeat_interleave(2, dim=-1)
print(normalized_steps.tolist())
print(process_token_advantages.tolist())
```

Функция `clipped_token_loss` выше принимает одно преимущество на ответ и
реализует только вариант с итоговой наградой. Для процессного варианта
понадобится матрица преимуществ по токенам. При этом предсказанная оценка PRM
остаётся мнением модели награды, а не формальной проверкой шага. В исходном
Algorithm 1 также есть итеративное обновление самой RM с 10% исторических
примеров и сменой reference; этот вариант нельзя отождествлять с RLVR,
где функцию проверки держат фиксированной.

## Почему нормировка по стандартному отклонению спорна

<a id="grpo-normalizations"></a>
<!-- source_unit_id: lecture-16-grpo-bias-failures -->
<!-- source_unit_id: assignment-05-task-compute-group-normalized-rewards-grpo -->
<!-- source_unit_id: assignment-05-task-aggregate-loss-across-microbatch-sequence -->

Деление на стандартное отклонение внутри группы делает масштабы разных запросов
похожими, но усиливает случайные различия в почти однородной группе. Кроме того,
возникает смещение, связанное со сложностью задачи.
В [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) можно
отключить это деление или использовать общий масштаб по пакету. Центрирование
и выбор масштаба при этом остаются разными операциями.

Например, $r=(1,1,1,0)$ и $r=(0.51,0.50,0.50,0.50)$ после собственной
нормировки внутри группы дают преимущества сопоставимого масштаба, хотя во
втором случае абсолютная разница оценок ничтожна. В первой группе преимущества
равны примерно $(0.577,0.577,0.577,-1.732)$, во второй —
$(1.732,-0.577,-0.577,-0.577)$, если опустить малое $\varepsilon_{norm}$.
Нормировка не умеет различать существенную разницу качества и шум оценщика.

Для бинарной награды с успешностью $p$ стандартное отклонение равно
$\sqrt{p(1-p)}$. В приближении большой группы преимущество успешного ответа
равно $\sqrt{(1-p)/p}$, а неуспешного — $-\sqrt{p/(1-p)}$. При $p=0.01$
редкий успех имеет вес около $9.95$, но он редко попадает в выборку; группа
из одних неудач по-прежнему даёт ноль. Большой вес наблюдаемого редкого успеха
нельзя путать с гарантированно большим ожидаемым прогрессом на трудной задаче.

## Нормировка по длине

Если сначала усреднить потери по токенам каждого ответа, короткая и длинная
последовательности получат одинаковый вес. Если сложить потери всех токенов и
разделить на их общее число, длинные ответы получат больший вес. В общем случае
это разные взвешенные целевые функции, а не две оценки одной величины.
Например, для ответов длиной 2 и 20 одинаковая потеря 1 на каждом токене даёт
итог 1 в обоих случаях, но веса ответов различны: $1/2,1/2$ против
$2/22,20/22$. Если потери первого и второго ответов станут 1 и 2, среднее по
ответам равно $1.5$, а по токенам — $(2+40)/22\approx1.909$.
Эти веса особенно важны для неудачных ответов: деление на длину ослабляет
отрицательный сигнал для каждого токена длинного ошибочного решения.

При накоплении градиента знаменатель должен соответствовать полному пакету.
Нельзя усреднить два микропакета разного размера поровну, если требовалось
среднее по всем токенам. На тех же 2 и 20 токенах получается следующее:

```python
per_token_losses = torch.cat((torch.ones(2), torch.full((20,), 2.))).double().requires_grad_()
sequence_mean = (per_token_losses[:2].mean() + per_token_losses[2:].mean()) / 2
token_mean = per_token_losses.sum() / 22
constant_mean = per_token_losses.sum() / (2 * 20)
microbatch_loss = per_token_losses[:2].sum() / 22 + per_token_losses[2:].sum() / 22
full_gradient, = torch.autograd.grad(token_mean, per_token_losses, retain_graph=True)
micro_gradient, = torch.autograd.grad(microbatch_loss, per_token_losses, retain_graph=True)
wrong_gradient, = torch.autograd.grad(sequence_mean, per_token_losses)
torch.testing.assert_close(full_gradient, micro_gradient)
print(f"sequence={sequence_mean.item():.6f}, token={token_mean.item():.6f}, constant={constant_mean.item():.6f}")
print(f"token weights: {full_gradient[0].item():.6f}, {full_gradient[2].item():.6f}")
print(f"equal microbatch weights: {wrong_gradient[0].item():.6f}, {wrong_gradient[2].item():.6f}")
```

Средние равны 1.5, 1.909091 и 1.05. Последнее использует условный предел
20 токенов для каждого из двух ответов. При среднем по токенам каждый имеет
вес $1/22$; при равном усреднении микропакетов первые два имеют вес $1/4$,
остальные — $1/40$. Поэтому разбиение само меняет обновление. В примере
разделены только суммы; дополнительное автоматическое деление фреймворка на
число шагов накопления тоже нужно учитывать, чтобы не применить его дважды.
Эта проверка соответствует вопросам о накоплении в CS336 Assignment 5, §4.2.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo-review-2026-09/length-difficulty-bias.png]]

*Два запроса, по четыре ответа на каждый. Квадраты обозначают токены,
направление стрелки — знак обновления, её длина — вес после деления на длину
ответа. Размер синего круга показывает дополнительный множитель от разброса
наград. Zichen Liu et al.,
[Understanding R1-Zero-Like Training, Figure 5, с. 7](https://arxiv.org/pdf/2503.20783v1#page=7),
CC BY 4.0; исходный рисунок без перерисовки.*

Таким образом, усреднение определяет, чему обучается модель, а не только
величину числа в журнале. Его нужно выбрать до сравнения методов и одинаково
реализовать на полном пакете и при накоплении градиента.

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

$$
\begin{aligned}
\mu_i&=\frac1G\sum_{j=1}^{G}r_{ij},\\
\sigma_i&=\sqrt{\frac1G\sum_j(r_{ij}-\mu_i)^2}.
\end{aligned}
$$

Чтобы отделить вес ответа от его градиента, введём $S_{ij}$ — сумму
градиентов логарифмов вероятностей его токенов. Здесь
$s_{ijt}=(x_i,y_{ij,<t})$ — запрос с уже написанной частью ответа:

$$
S_{ij}=\sum_{t=1}^{L_{ij}}
\nabla_\theta\log\pi_\theta(y_{ijt}\mid s_{ijt}).
$$

У исходного GRPO одновременно присутствуют **две нормировки**. Ниже записан
его градиент в начале обновления: новая и старая стратегии совпадают,
ограничение не активно, штраф по KL опущен.

$$
\hat g_{GRPO}=\frac1{BG}\sum_{i,j}
\frac{r_{ij}-\mu_i}{(\sigma_i+\epsilon)L_{ij}}S_{ij}.
$$

Деление на $\sigma_i$ меняет вес запроса: почти решённая или почти нерешаемая
задача может получить большой множитель из-за малого разброса. Деление на
$L_{ij}$ меняет вес токена: каждый ответ получает одинаковую массу независимо
от длины, поэтому токен короткого ответа весит больше токена длинного. Эти
эффекты нельзя обсуждать как одну «стабилизацию».

Есть и более тонкая деталь. Среднее группы $\mu_i$ вычислено с участием того же
ответа, градиент которого оно центрирует. Поэтому среднее группы не является
независимой от действия базовой оценкой. Утверждение о простом масштабе
относится к отдельной, **ненормированной** оценке для фиксированного запроса:

$$
\begin{aligned}
\hat g&=\frac1G\sum_j(r_j-\bar r)\nabla_\theta\log\pi_\theta(y_j\mid x),\\
\mathbb E[\hat g]&=\frac{G-1}{G}\nabla_\theta\mathbb E[r].
\end{aligned}
$$

Здесь ответы независимы и одинаково распределены, награда внешняя, а
$\nabla\log\pi(y_j\mid x)$ включает сумму по всем токенам **без** деления
на длину. Вклад собственного $r_j/G$ убирает $1/G$ ожидаемого градиента;
остальные награды независимы от градиента логарифма вероятности данного ответа
и дают ноль в ожидании.
Базовая оценка по остальным ответам (*leave-one-out*) устраняет именно этот
эффект: $b_{-j}=\sum_{k\ne j}r_k/(G-1)$ и
$r_j-b_{-j}=G(r_j-\bar r)/(G-1)$. Деление на зависящее от ответов стандартное
отклонение, случайную длину, а также ограничение PPO или перевзвешивание
старых генераций в это доказательство не входят:
для полной GRPO-оценки выше одного множителя $(G-1)/G$ недостаточно.

На двух независимых действиях это можно проверить полным перебором.
Пусть действие 1 приносит награду 1 и выбирается с вероятностью $p=0.2$.
Производная логарифма вероятности действия по логиту равна $a-p$.
Всего возможны четыре группы; их вероятности известны точно:

```python
from itertools import product

binary_p = 0.2
baseline_expectations = {"reinforce": 0., "group_mean": 0., "leave_one_out": 0.}
for actions in product((0., 1.), repeat=2):
    chance = math.prod(binary_p if a else 1-binary_p for a in actions)
    reward_mean = sum(actions) / 2
    score = [a-binary_p for a in actions]
    baseline_expectations["reinforce"] += chance * sum(a*s for a, s in zip(actions, score)) / 2
    baseline_expectations["group_mean"] += chance * sum((a-reward_mean)*s for a, s in zip(actions, score)) / 2
    baseline_expectations["leave_one_out"] += chance * sum((actions[j]-actions[1-j])*score[j] for j in range(2)) / 2
print({name: round(value, 6) for name, value in baseline_expectations.items()})
```

Получаются 0.16, 0.08 и 0.16. Среднее собственной группы уменьшило ожидаемый
градиент вдвое, а базовый уровень по другому ответу восстановил 0.16.
Здесь один токен на ответ, нет нормировки по длине, стандартного отклонения
или ограничения; переносить эти три числа на полную функцию потерь нельзя.

### Dr. GRPO

Dr. GRPO не делит преимущество на стандартное отклонение наград и заменяет
нормировку каждого ответа по его длине одним постоянным знаменателем:

$$
\begin{aligned}
\hat g_{Dr}&=\frac1Z\sum_{i,j}(r_{ij}-\mu_i)S_{ij},\\
Z&=BGL_{max}.
\end{aligned}
$$

Именно $Z=BGL_{max}$ использует Assignment 5 ($L_{max}=512$ в полном
референсном запуске). Это не оценка фактического числа токенов пакета, а
фиксированный масштаб, благодаря которому длина сгенерированных ответов не
меняет знаменатель от шага к шагу.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo-review-2026-09/length-and-accuracy-ablation.png]]

*В этом эксперименте на Qwen2.5-1.5B награда и среднее качество близки у двух
методов, но у GRPO растёт длина прежде всего ошибочных ответов. Это пример
того, почему одного графика награды недостаточно. Zichen Liu et al.,
[Figure 6 и условия эксперимента, с. 8](https://arxiv.org/pdf/2503.20783v1#page=8),
CC BY 4.0. Результат относится к описанным модели, шаблону и задачам MATH;
он не устанавливает универсального превосходства одной нормировки.*

### Rejection fine-tuning, или обучение на отобранных решениях

RFT сохраняет только успешные сгенерированные решения и делает на них обычный
шаг SFT:

$$
\hat g_{RFT}=\frac1Z\sum_{i,j}\mathbf1[r_{ij}=1]S_{ij}.
$$

В коде это соответствует `baseline="none"`,
`advantage_normalizer="none"`, `loss_normalization="constant"`. Неверные
ответы имеют нулевой вес и могут быть исключены ещё до прямого прохода. RFT
проще и часто дешевле, но после отбора сводится к обучению по сохранённым
траекториям: метод не понижает вероятность конкретных неверных ответов явным
отрицательным членом и наследует ошибки фильтра.

### MaxRL

В учебном варианте MaxRL из CS336 центрированная награда делится на среднюю
успешность группы:

$$
\hat g_{MaxRL}=\frac1Z\sum_{i,j}
\frac{r_{ij}-\mu_i}{\mu_i+\epsilon}S_{ij}.
$$

Приближённо это повышает вес запросов, которые текущая модель решает редко. При
малой, но ненулевой $\mu_i$ множитель становится большим, поэтому в отчёте
нужны квантили весов и нормы градиента, а $\epsilon$ является частью
спецификации. Если вся группа получила ноль, числитель также равен нулю и
обучающего сигнала нет: $\epsilon$ предотвращает деление на ноль, но не создаёт
информацию. В статье о MaxRL функция потерь нормируется фактическим числом токенов пакета.
В задании 5 намеренно оставлен постоянный $Z$, чтобы в абляции менялся только
способ нормировки преимущества;
результат такого варианта нельзя подписывать просто «MaxRL» без оговорки.

Мотивация такого веса видна из $\nabla\log p=\nabla p/p$: одинаковое
абсолютное улучшение вероятности успеха сильнее меняет её логарифм, когда
исходная вероятность мала. При $p=0.1$ и $p=0.9$ множители $1/p$ равны
10 и примерно 1.11. Однако подстановка эмпирической доли успехов вместо
неизвестного $p$ добавляет шум; формула для бесконечной выборки не является
доказательством несмещённости оценки по маленькой группе.

У статьи MaxRL есть также различие между теоретической оценкой и практическим
Algorithm 1. При $N$ попытках, $K$ успехах и
$S_j=\nabla\log\pi_\theta(y_j|x)$ теоретическая оценка с контрольной переменной
из §4.2 равна

$$
\widetilde g_N=
\frac{\sum_j r_jS_j}{K}-\frac1N\sum_jS_j.
$$

Когда $K=0$, по соглашению обнуляется только первая дробь. Во второй сумме
могут остаться ненулевые члены. Её математическое ожидание равно нулю, поэтому
вычитание не меняет ожидаемый градиент. В практическом §4.3 авторы обнуляют
обе части при $K=0$; именно с этим поведением согласуется формула учебного
варианта выше. У них совпадает обработка неуспешной группы, но разные
знаменатели по токенам и малые слагаемые в нормировке всё равно нужно указывать.

Для бинарной задачи с $p=0.2,N=2$ теоретическая оценка имеет ожидание 0.288,
а вариант с пропуском полностью неуспешной группы — 0.16. У первой остаётся
вклад 0.128 от группы двух неудач: её вероятность 0.64, а контрольный член
равен 0.2. Отбрасывание этого вклада — практическое изменение оценки,
а не следствие теоремы о несмещённости. См.
[MaxRL, §4.1–4.3 и Appendix F](https://arxiv.org/html/2602.02710v1#S4).

```python
maxrl_theoretical = 0.
maxrl_practical = 0.
for actions in product((0., 1.), repeat=2):
    chance = math.prod(binary_p if a else 1-binary_p for a in actions)
    successes = sum(actions)
    average_score = sum(a-binary_p for a in actions) / 2
    successful_score = (
        sum(a*(a-binary_p) for a in actions) / successes if successes else 0.
    )
    theoretical_update = successful_score - average_score
    maxrl_theoretical += chance * theoretical_update
    maxrl_practical += chance * theoretical_update if successes else 0.
print(f"MaxRL theoretical={maxrl_theoretical:.6f}, skip-failures={maxrl_practical:.6f}")
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo-2026-10-24/maxrl-weight-functions.svg]]

*Веса задач в зависимости от вероятности успеха. Для конечного
порядка $T$ вес MaxRL равен $[1-(1-p)^T]/p$, а $1/p$ получается только в
пределе. Это популяционные величины, а не веса одного шумного пакета.
Fahim Tajwar et al., [MaxRL, Figure 2](https://arxiv.org/html/2602.02710v1#S5.F2),
CC BY 4.0. Полный исходный SVG без изменений.*

Полезно сопоставить именно множители при центрированной награде, временно
исключив длину, ограничение и конечность группы:

| $p$ | Dr. GRPO | $1/\sqrt{p(1-p)}$ | $1/p$ |
|---:|---:|---:|---:|
| 0.01 | 1 | 10.05 | 100 |
| 0.50 | 1 | 2 | 2 |
| 0.99 | 1 | 10.05 | 1.01 |

Здесь $p$ — истинная вероятность успеха, а не доля успехов в конкретном пакете.
У группы размера $G$ с $k$ успехами стандартная нормировка даёт успешному
ответу $\sqrt{(G-k)/k}$ при $0<k<G$. Поэтому при фиксированном $G$
его максимальный вес без малого слагаемого равен $\sqrt{G-1}$, а не бесконечности.

<a id="что-сравнивает-честная-абляция"></a>

### Контролируемое сравнение методов

Stanford фиксирует запросы, бюджет генерации, проверяющую программу,
инициализацию, число обновлений и процедуру оценивания. Во всех четырёх
вариантах ниже потери делят на постоянный $Z=BGL_{max}$;
меняются базовый уровень и делитель преимущества:

| вариант | база | делитель |
|---|---|---|
| `GRPO_constant` | mean | std |
| `Dr_GRPO` | mean | none |
| `RFT` | none | none |
| `MaxRL_course` | mean | mean |

Темп обучения предварительно настраивается для базового варианта. Преимущество
нового метода при этих настройках показывает пользу изменения в данных условиях.
Но более низкий результат ещё не доказывает недостаток самого алгоритма:
выбранный темп обучения мог оказаться для него неподходящим. Полный опыт курса требует четыре запуска с
разными начальными значениями генератора случайных чисел;
сокращённый опыт обязан показывать каждый запуск и разброс, а не одну лучшую
кривую.

## Штраф по KL не определяется одним названием GRPO

DeepSeekMath включает штраф относительно опорной модели. В проверенной
[реализации TRL](https://github.com/huggingface/trl/blob/ccdaa0065cee4174d2b4eb60f1856c51e51232ed/trl/trainer/grpo_config.py)
по умолчанию стоят `beta=0.0` и `loss_type="dapo"`. Следовательно, вызов
класса `GRPOTrainer` без явных настроек не воспроизводит ни исходную формулу
DeepSeekMath, ни обучение R1. В конфигурации опыта нужно записывать сами
решения, а не только название класса:

```yaml
algorithm: grpo
group_size: 8
scale_rewards: false
beta: 0.0
loss_aggregation: token_mean
```

<a id="минимальный-пример-запуска-в-trl"></a>

## Как задать вариант алгоритма в TRL

Следующая конфигурация выбирает усреднение по последовательностям, групповую
нормировку наград и ненулевой штраф. Это пример настройки разобранной целевой
функции, **не воспроизведение гиперпараметров DeepSeek-R1**:

```python
from trl import GRPOConfig

args = GRPOConfig(
    output_dir="qwen-grpo-demo",
    num_generations=8,
    max_completion_length=512,
    temperature=0.8,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    beta=0.001,
    scale_rewards="group",
    loss_type="grpo",
    epsilon=0.2,
    bf16=False,
    fp16=False,
    use_cpu=True,             # создаём и проверяем конфигурацию, не запускаем обучение
    report_to="none",
)
```

Для запуска нужны ещё модель, токенизатор, набор запросов с эталонами и
проверяющая функция, контракт которой совпадает с форматом ответа. Например,
проверка `FINAL: число` из предыдущей главы не подходит без изменения к
датасету, где ответ оформлен как `\boxed{...}`. До обучения полезно
сгенерировать один пакет и вручную сверить каждый вердикт с текстом ответа.

На одном процессе произведение размера микропакета и числа шагов накопления
здесь равно восьми, что согласуется с `num_generations=8`. При распределённом
запуске нужно учитывать число процессов. Ссылка выше фиксирует проверенную
реализацию параметров; полный GPU-запуск этим фрагментом не проверяется.
CPU-пример выше отдельно проверяет математику потерь.

Этот блок создаёт настоящий `GRPOConfig` из TRL 0.24.0; предобученные веса и
данные не загружаются. `use_cpu=True` и отключённая смешанная точность позволяют
проверить параметры на CPU. Для большого обучения размещение и точность
задают заново под оборудование. Названия полей сверены с
[исходником v0.24.0](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py).
Из создания конфигурации не следует, что проверен полный `GRPOTrainer`.

## Несовпадение моделей генерации и обучения

Ответы часто генерирует vLLM, а логарифмы вероятностей для вычисления градиента
получает Transformers. Из-за различий в точности вычислений и ядрах
$\pi_{inference}\ne\pi_{train}$. Формально обновление перестаёт использовать
данные строго из текущей стратегии:

$$y\sim\pi_{inference},\qquad \nabla\log\pi_{train}(y)R(y).$$

Это различие может появиться даже при одинаковых ядрах: температура,
top-p и top-k меняют распределение выборки относительно исходного softmax.
Сохранённые вероятности должны соответствовать явно выбранному соглашению
о распределении генерации и обучаемой стратегии. Если усечение запретило
некоторые токены, простое перевзвешивание не восстановит отсутствующие
примеры. Ошибку численной реализации и намеренное изменение способа
генерации следует измерять раздельно.

[TRL](https://huggingface.co/docs/trl/grpo_trainer) предлагает усечённое
или маскированное перевзвешивание вероятностей. При раздельном запуске генератор
и процесс обучения размещают на разных ускорителях; при совместном размещении
память делят параметры модели, кеш ключей и значений и состояние оптимизатора.

<a id="offpolicy-ratios"></a>
<!-- source_unit_id: assignment-05-deliverable-derive-surrogate-objectives-030 -->
<!-- source_unit_id: assignment-05-deliverable-think-about-importance-reweighting-032 -->
<!-- source_unit_id: assignment-05-task-derive-surrogate-objectives -->
<!-- source_unit_id: assignment-05-task-think-about-importance-reweighting -->

## Когда один пакет генераций используют несколько раз

Обучение на данных текущей стратегии (*on-policy*) предполагает, что ответы
порождены той же моделью, для которой оценивается градиент. Если после генерации
разбить 256 ответов на пакеты по восемь и сделать 32 **шага оптимизатора**,
то только первый шаг использует данные текущей стратегии. Для остальных
параметры модели, породившей ответы, $\pi_0$, уже отличаются от обучаемых.
Это не то же самое, что 32 микропакета с накоплением градиента и одним общим
шагом: при накоплении параметры между микропакетами не меняются.

Точное перевзвешивание распределения целых ответов использует отношение

$$
w(y)=\frac{\pi_\theta(y\mid x)}{\pi_0(y\mid x)}
=\prod_{t=1}^{L}\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_0(y_t\mid x,y_{<t})}.
$$

При условии достаточного носителя $\pi_0$ это произведение корректирует
распределение всей последовательности, но может иметь очень большую дисперсию.
GRPO вместо него применяет отношение
$w_t=\pi_\theta(y_t\mid s_t)/\pi_0(y_t\mid s_t)$ и ограничение PPO отдельно
к токенам. Такая замена не корректирует распределение предшествующего текста
и последующего продолжения: они по-прежнему получены от $\pi_0$. Получается
другая, приближённая цель, а не точное перевзвешивание целой траектории.

Эту цель можно записать явно. Пусть длина ответа фиксирована и равна $L$.
Для каждой позиции $t$ определим распределение $\widetilde\pi_t$: один
токен выбирается текущей моделью, остальные — старой. При этом каждая модель
видит уже полученный префикс. Вероятность такой последовательности равна
$\pi_0(y\mid x)w_t$. Поэтому без clipping и базового уровня токеновое
перевзвешивание даёт градиент цели

$$
J_{token}(\theta)=\sum_{t=1}^{L}
\mathbb E_{y\sim\widetilde\pi_t}[R(y)].
$$

Действительно, заменив каждое ожидание на
$\mathbb E_{y\sim\pi_0}[w_tR(y)]$ и продифференцировав только $w_t$,
получим $\mathbb E_{\pi_0}[R(y)\sum_t w_t\nabla\log\pi_\theta(y_t\mid s_t)]$.
Будущие токены здесь по-прежнему выбирает старая модель. Обновление оценивает
изменение одного решения при прежнем способе продолжить ответ, а не изменение
всей последовательности текущей моделью. В этой записи награда фиксирована,
старая модель не дифференцируется, а длина не зависит от выбора токенов.

Рассмотрим два независимых двоичных токена. Награда равна единице только
для ответа `11`. Старая модель выбирает единицу с вероятностью $p_0=0.2$,
текущая — с вероятностью $p=0.6$. Пусть общий обучаемый параметр — логит
$\theta$, так что $\partial p/\partial\theta=p(1-p)$.
Текущее ожидаемое вознаграждение равно $p^2$, а его производная —
$2p^2(1-p)=0.288$. Из старой модели ответ `11` поступает с вероятностью
$0.04$. Полное отношение вероятностей $(p/p_0)^2=9$ восстанавливает эту
производную. Токеновое отношение равно только трём: оно даёт $0.096$,
производную $J_{token}=2p_0p$. Без перевзвешивания получилось бы $0.032$.

```python
offpolicy_p0 = .2
offpolicy_logit = torch.tensor(math.log(.6/.4), dtype=torch.float64, requires_grad=True)
offpolicy_p = offpolicy_logit.sigmoid()
offpolicy_old_success = offpolicy_p0**2
# Только ответ 11 имеет ненулевую награду; его log-prob равен 2*log(p).
offpolicy_score, = torch.autograd.grad(2*offpolicy_p.log(), offpolicy_logit, retain_graph=True)
offpolicy_ratio = offpolicy_p/offpolicy_p0
offpolicy_gradients = {
    "unweighted": offpolicy_old_success*offpolicy_score,
    "token": offpolicy_old_success*2*offpolicy_ratio*(1-offpolicy_p),
    "sequence": offpolicy_old_success*offpolicy_ratio.square()*offpolicy_score,
}
offpolicy_exact, = torch.autograd.grad(offpolicy_p.square(), offpolicy_logit, retain_graph=True)
offpolicy_surrogate, = torch.autograd.grad(2*offpolicy_p0*offpolicy_p, offpolicy_logit)
print({name: round(value.item(), 6) for name, value in offpolicy_gradients.items()})
print(f"true gradient={offpolicy_exact.item():.6f}, token surrogate={offpolicy_surrogate.item():.6f}")
```

Упражнение CS336 предлагает следующий промежуточный вариант: перевзвешивать
пары соседних токенов произведением их двух отношений. Тогда каждому слагаемому
соответствует распределение, в котором текущая модель выбирает два токена пары,
а остальные выбирает старая. Цель — сумма ожидаемых наград для этих
распределений. Для ответа из двух токенов единственная пара покрывает весь
ответ, и оценка совпадает с полным перевзвешиванием. Для более длинного ответа
вне пары остаются старые токены, поэтому полного исправления распределения
по-прежнему нет. Разбиение на блоки меняет и смещение, и разброс весов.
Вывод и упражнение: [CS336 A5, §6.1–6.2.1, с. 32–34](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_alignment.pdf).

Для $A\ge0$ ограничение прекращает усиливать действие после $w_t>1+\epsilon$; для
$A<0$ — прекращает ослаблять его после $w_t<1-\epsilon$. Поэтому логируют не
только среднее отношение, но и долю ограниченных слагаемых отдельно
для двух знаков преимущества, крайние значения отношений и число обновлений
со времени генерации. Для положительных весов также полезен эффективный
размер выборки $(\sum_i w_i)^2/\sum_i w_i^2$: он уменьшается,
если почти весь вес приходится на несколько ответов.

<a id="gspo"></a>

## GSPO: одно отношение вероятностей на весь ответ

GSPO использует геометрическое среднее отношений вероятностей токенов:

$$
\begin{aligned}
d_t&=\log\pi_\theta(y_t\mid s_t)\\
&\quad-\log\pi_0(y_t\mid s_t),\\
s(y)&=\exp\left[\frac1L\sum_{t=1}^{L}d_t\right].
\end{aligned}
$$

Один $s(y)$ затем входит в ограниченную целевую функцию для всего ответа:

$$
\begin{aligned}
s^c_{ij}&=\operatorname{clip}(s_{ij},1-\epsilon,1+\epsilon),\\
J_{GSPO}&=\frac1{BG}\sum_{i,j}
\min(A_{ij}s_{ij},A_{ij}s^c_{ij}).
\end{aligned}
$$

На практике суммируют разности логарифмов, а не перемножают вероятности.
`response_mask` исключает запрос и заполнители как из суммы, так и из $L$.
Геометрическое среднее не является точным отношением вероятностей
последовательности: извлечение корня уменьшает резкие изменения веса,
но вносит смещение. Его градиент естественно содержит $1/L$; поэтому
нельзя одновременно назвать формулу GSPO и приписать ей постоянную
нормировку Dr. GRPO без дополнительного определения.

Рассмотрим ответ из двух токенов с отношениями $1.5$ и $0.6$.
Точное отношение для последовательности равно $0.9$, а вес GSPO —
$\sqrt{0.9}\approx0.9487$. При положительном преимуществе и
$\epsilon=0.2$ GRPO ограничит слагаемое первого токена, но не второго;
в GSPO общий вес останется внутри интервала, и оба токена получат
градиент через одно и то же слагаемое. Это не только иной масштаб:
меняется то, какие части ответа продолжают участвовать в обновлении.
См. [GSPO, §4](https://arxiv.org/html/2507.18071v2#S4).

Практический вывод не сводится к рейтингу методов. Без перевзвешивания
дисперсия мала, но смещение быстро растёт по мере устаревания данных; точное
отношение вероятностей последовательности формально корректно, но часто имеет
неприемлемую дисперсию. Ограничение по токенам и GSPO дают разные
приближения. Выбор подтверждают опытом с одинаковыми начальными условиями,
числом сгенерированных токенов и обновлений.

При условном преимуществе $A=1$ разницу между GRPO и GSPO на том же
двухтокеновом ответе можно увидеть непосредственно в производных:

```python
gspo_old_lp = torch.full((1, 2), -2., dtype=torch.float64)
gspo_new_lp = (gspo_old_lp + torch.tensor([[1.5, .6]], dtype=torch.float64).log()).requires_grad_()
gspo_mask = torch.ones((1, 2), dtype=torch.bool)
grpo_comparison_loss = clipped_token_loss(gspo_new_lp, gspo_old_lp, [1.], gspo_mask)
grpo_comparison_gradient, = torch.autograd.grad(grpo_comparison_loss, gspo_new_lp, retain_graph=True)
sequence_log_ratio = (gspo_new_lp - gspo_old_lp).mean(-1)
gspo_ratio = sequence_log_ratio.exp()
gspo_loss = -torch.minimum(gspo_ratio, gspo_ratio.clamp(.8, 1.2)).mean()
gspo_gradient, = torch.autograd.grad(gspo_loss, gspo_new_lp)
print(f"trajectory ratio={sequence_log_ratio.exp().square().item():.6f}, GSPO ratio={gspo_ratio.item():.6f}")
print(grpo_comparison_gradient.tolist())
print(gspo_gradient.tolist())
```

GRPO даёт градиенты потери $(0,-0.3)$, GSPO — примерно
$(-0.474342,-0.474342)$. В первом случае токен с отношением 1.5 уже ограничен,
во втором оба токена получают один общий коэффициент. Интервал 0.8–1.2 здесь
выбран для сравнения арифметики. В экспериментах статьи GSPO используют
существенно более узкие границы; переносить 0.2 из GRPO как стандартную
настройку GSPO нельзя.

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo-review-2026-09/r1-zero-training-dynamics.png]]

*Слева — качество R1-Zero на AIME 2024: один ответ и выбор по согласованности
16 ответов — разные протоколы оценивания. Справа — средняя длина ответа во
время обучения. Совместный рост двух кривых сам по себе не доказывает, что
увеличение длины является единственной причиной улучшения. DeepSeek-AI,
[расширенный отчёт, Figure 1, с. 4](https://arxiv.org/pdf/2501.12948v2#page=4);
тот же авторский рисунок опубликован в [Nature, Figure 1](https://www.nature.com/articles/s41586-025-09422-z/figures/1), CC BY 4.0.*

В расширенной версии отчёта указаны $G=16$, температура генерации 1 и
коэффициент KL 0.001. Примерно после 8 200 шагов предел длины увеличили с
32 768 до 65 536 токенов. Поэтому скачок на позднем участке графика нельзя
объяснять только изменением параметров модели: изменился и разрешённый бюджет
генерации. См. [R1 v2, §2.1](https://arxiv.org/html/2501.12948v2#S2.SS1).

В этом же описании один сбор даёт 8192 ответа, которые делят на 16
минипакетов по 512 и используют в одном внутреннем проходе. Это 16 обновлений
по уже полученным генерациям, а не 16 повторных генераций свежих ответов.
Смена reference раз в 400 шагов выполняет другую роль. Формула GRPO в
расширенном R1 v2 записана через отношение вероятностей целого ответа;
токеновая формула выше следует исходному DeepSeekMath. Одно название метода
не устраняет это различие в записи и не задаёт точного кода обновления R1.

### Полный DeepSeek-R1

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/deepseek-r1/figure2-pipeline.png]]

*Схема читается слева направо. Фиолетовые блоки — модели, серо-синие — данные.
Стрелка `Sampling` означает передачу сгенерированных примеров, а не весов.
Справа сверху снова стоит DeepSeek-V3-Base: смешанный SFT начинается с базовой
модели. DeepSeek-AI,
[DeepSeek-R1 v2, Figure 2, с. 6](https://arxiv.org/pdf/2501.12948v2#page=6),
[публикация рисунка в Nature](https://www.nature.com/articles/s41586-025-09422-z/figures/2), CC BY 4.0.*

**Начальное обучение на примерах.** R1-Zero даёт решения, но их не всегда
удобно читать. Отобранные и отредактированные длинные решения используют
для начального SFT базовой модели. Этот этап задаёт подходящее для дальнейшего
обучения поведение; он не должен сам обеспечить всё итоговое качество.

**Первый этап RL.** Модель порождает новые решения и получает награды за
проверяемый результат. Дополнительно вводится оценка языковой согласованности:
доля слов требуемого языка в рассуждении. Это не проверка математики, а
отдельное требование к читаемости. Его влияние на правильность нужно оценивать
отдельно от улучшения языка: в абляции авторов эта добавка немного снижает
качество решения задач. В §3.2.1 расширенного отчёта для этого этапа указано
$\epsilon=10$, а не 0.2 из учебного примера выше. Это ещё одна причина
не принимать пример настройки GRPO за воспроизведение обучения R1.

**Создание обучающей смеси.** Из генераций промежуточной модели отбирают
решения; к ним добавляют данные для письма, ответов на общие вопросы и других
задач. На этой смеси снова обучают DeepSeek-V3-Base. Здесь результат первого
RL-этапа используется через сгенерированные данные: его веса не служат
начальной точкой второго SFT. Состав смеси, фильтры и инициализация определяют,
что именно получит следующая модель. Процедура отбора подробно разобрана в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation|следующей главе]].

**Заключительный этап RL.** Для задач с проверяемым ответом сохраняется
программная награда; для общих запросов подключаются модели предпочтений.
Полезность оценивается по заключительному ответу, а безопасность — по всему
сгенерированному тексту. Это разделение важно: внутренний длинный разбор и
краткий ответ пользователю выполняют разные функции и не обязаны оцениваться
одинаково. Подробности стадий и данных:
[R1 v2, §3 и Appendix B.3](https://arxiv.org/html/2501.12948v2#S3).

Эта схема связывает уже изученные методы. SFT задаёт поведение по примерам,
RL меняет распределение новых ответов по награде, а отбор превращает часть
этих ответов в данные для очередного SFT. Заменить все стадии словом GRPO
означало бы потерять и происхождение данных, и назначение каждого перехода.

<a id="kimi-k15-rl-требует-не-только-objective-но-и-производственного-контура"></a>

### Kimi k1.5: стоимость длинных решений

Длинное решение занимает ускорители не только при обратном проходе, но и
при последовательной генерации. Пока один запрос завершён за тысячу токенов,
другой может продолжаться десятки тысяч. В Kimi k1.5 частичные генерации
позволяют повторно использовать уже полученную часть траектории, не начиная
всё с запроса. При этом нужно учитывать, какой версией модели были получены
разные участки ответа: это тот же вопрос об устаревании данных, что выше.
См. [Kimi k1.5, §2.6.2](https://arxiv.org/html/2501.12599v1#S2.SS6.SSS2).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/grpo-2026-10-24/kimi-partial-rollouts.svg]]

*Прошлые сегменты ответа поступают из replay buffer, а текущий сегмент
генерируется на очередной итерации. Это повторное использование уже написанных
токенов, а не независимая новая попытка от исходного запроса. Kimi Team,
[Kimi k1.5, Figure 3(b)](https://arxiv.org/html/2501.12599v1#S2.F3),
CC BY-NC-ND 4.0. Полный исходный SVG без изменений.*

Для траектории задают лимит новых токенов за итерацию. Если до завершения
ответа лимит исчерпан, префикс сохраняют и продолжают позже. Рабочие процессы
не обязаны ждать одну самую длинную генерацию, прежде чем взять короткие
задачи. В отчёте также допускают исключение некоторых прошлых сегментов из
функции потерь и раннюю остановку повторяющихся фрагментов. Поэтому журнал
должен различать уже сохранённый контекст, вновь сгенерированные токены и
токены, по которым действительно вычислялся градиент. Экономия повторной
генерации сама по себе не означает точного on-policy обновления всей цепочки.

Алгоритм Kimi k1.5 тоже не совпадает с формулой GRPO выше. В §2.3.2 авторы
получают из регуляризованной задачи квадратичную потерю по логарифму отношения
вероятностей ответа. В практической оценке участвуют центрированная награда
и квадрат отклонения log-ratio; опорная модель меняется между итерациями,
а состояние оптимизатора сбрасывается. Отсутствие отдельного критика — общая
черта нескольких методов, а не достаточный признак GRPO.

Обозначим логарифм отношения вероятностей целого ответа через $\ell_j$,
а центрированную награду — через $a_j=r_j-\bar r$:

$$
\begin{aligned}
\ell_j&=\log\pi_\theta(y_j,z_j\mid x)\\
&\quad-\log\pi_{ref}(y_j,z_j\mid x).
\end{aligned}
$$

Направление обновления
из уравнения (3) отчёта можно записать как

$$
g=\frac1k\sum_j(a_j-\tau\ell_j)
\nabla_\theta\log\pi_\theta(y_j,z_j\mid x).
$$

Слагаемое $a_j$ усиливает ответы с наградой выше средней. Член
$-\tau\ell_j$ противодействует отклонению вероятности от reference:
если вероятность выросла, он отрицателен; если упала, положителен.
При $a=(0.5,-0.5)$, $\ell=(0.2,-0.2)$ и $\tau=0.5$ коэффициенты равны
$(0.4,-0.4)$. Здесь нет плато clipping, как в PPO: поправка меняется
непрерывно с $\ell$. Это сравнение формул обновления, а не доказательство,
что один способ лучше другого.

```python
kimi_old_lp = torch.tensor([-2., -2.], dtype=torch.float64)
kimi_new_lp = torch.tensor([-1.8, -2.2], dtype=torch.float64, requires_grad=True)
kimi_a = torch.tensor([.5, -.5], dtype=torch.float64)
kimi_tau = .5
kimi_log_ratio = kimi_new_lp - kimi_old_lp
# Отрицательный градиент этой потери совпадает с g при фиксированных данных.
kimi_loss = -(kimi_a * kimi_new_lp - .5 * kimi_tau * kimi_log_ratio.square()).mean()
kimi_gradient, = torch.autograd.grad(kimi_loss, kimi_new_lp)
print(f"Kimi ascent coefficients: {(kimi_a-kimi_tau*kimi_log_ratio).tolist()}")
print(f"loss gradient, including 1/k: {kimi_gradient.tolist()}")
```

Градиенты потери по двум логарифмам равны $(-0.2,0.2)$ после усреднения
по двум ответам. В модели они далее умножаются на производные логарифмов
вероятностей по общим параметрам. Средняя награда здесь — практическая
замена $\tau\log Z$ из исходного вывода; это приближение, а не тождество
для произвольного $\tau$.

Управление длиной влияет и на награду. В Kimi k1.5 короткое правильное решение
может получить бонус, длинное — штраф; неправильному решению бонус за краткость
не дают. При длинах 100 и 300 токенов нормированная величина из отчёта
$0.5-(L-L_{min})/(L_{max}-L_{min})$ равна соответственно $0.5$ и $-0.5$.
Для правильных решений сохраняется этот знак; для неправильных положительную
часть заменяют нулём. Так «ответить неверно, но быстро» не становится
самостоятельно вознаграждаемой стратегией. Если все длины равны, добавка нулевая.
Авторы вводят штраф не с начала обучения, чтобы сначала дать модели освоить
решение задач. Это явное изменение предпочтения между качеством и стоимостью,
а не техническая нормировка функции потерь.
[Kimi k1.5, §2.3.3](https://arxiv.org/html/2501.12599v1#S2.SS3.SSS3).

<a id="qwen3-несколько-стадий-вместо-одной-кнопки-reasoning"></a>

### Qwen3: обучение двух режимов ответа

В отчёте Qwen3 мая 2025 года крупные модели проходят четыре стадии:
начальный SFT на длинных решениях, RL для рассуждений, объединение режимов
с рассуждением и без него, затем RL на общих задачах. Третья стадия нужна
потому, что длинное решение полезно не для любого запроса. Модель должна
уметь как решать сложную задачу, так и отвечать на простой вопрос без
избыточного разбора.

Небольшие модели семейства получают способности крупных через дистилляцию;
не каждая модель независимо проходит все четыре стадии. Это важно при
сопоставлении вычислительных затрат и при попытке объяснить результат
конкретной модели. Нельзя также автоматически приписывать этому выпуску
GSPO из более поздней статьи. Схема относится к указанной версии семейства,
а не ко всем будущим Qwen. См.
[Qwen3 Technical Report, §4.1–4.5](https://arxiv.org/html/2505.09388v1#S4).

Для cold start сначала отбрасывают непроверяемые запросы и слишком простые
вопросы, решаемые без цепочки рассуждений. Кандидатов генерирует QwQ-32B;
фильтры проверяют ответ, повторы, смешение языков и согласованность рассуждения
с итогом. Следующий RL-этап использует отдельные 3995 пар «запрос — проверка»,
не использованных для cold start. Это не означает, что всего данных во всех
четырёх стадиях было 3995.

На стадии слияния режимов SFT продолжают именно с модели после reasoning RL.
Это отличается от второго SFT в схеме R1, где снова берут базовые веса.
Данные с рассуждением получают от текущей модели посредством отбора решений,
а для коротких ответов собирают отдельную смесь. Флаги `/think` и `/no_think`
в запросе или системном сообщении определяют выученный формат ответа.
В non-thinking режиме у ответа остаётся пустой блок `<think>...</think>`;
в многошаговом диалоге действует последний флаг. Поэтому режим задаётся
обучением и сериализацией диалога, а не заменой архитектуры Transformer.

В заключительном General RL используют три источника награды: правила,
оценивание ответа относительно эталона моделью Qwen2.5-72B-Instruct и
отдельную модель предпочтений для запросов без эталона. Поэтому этот этап
нельзя описывать как одну проверку правильного числа. Для агентных задач
генерация включает многошаговое взаимодействие с реальной средой и её
обратную связь. Какую часть ответа оценивают и какие действия разрешает
среда, определяет конкретная задача. См.
[Qwen3, §4.4](https://arxiv.org/html/2505.09388v1#S4.SS4).

## Что означает наблюдаемый момент самокоррекции

В отчёте приведён пример, где модель останавливается и пересматривает решение.
По одному такому примеру нельзя установить, какой внутренний алгоритм
использовала модель и какие из записанных шагов действительно помогли найти
ответ. Из опыта можно сделать более узкий вывод: оптимизация по
конечному результату повысила вероятность последовательностей, содержащих
полезные проверки и исправления на данном распределении задач. При этом
контрольное исследование Dr. GRPO обнаруживает похожие фразы самопроверки
у базовых моделей ещё до RL. Слово `wait` само по себе не измеряет новый
навык: нужен разбор правильности решения, а не только частот слов.
[Understanding R1-Zero-Like Training, §2.3 и Appendix D](https://arxiv.org/html/2503.20783v1#S2.SS3).

## Характерные сбои и наблюдаемые показатели

| симптом | проверка и возможная причина |
|---|---|
| награда растёт, проверочное качество — нет | проверить награду на новых задачах и намеренных атаках: возможны обход проверки или утечка данных |
| длина быстро растёт | сравнить правильность при фиксированном лимите длины; проверить награду за длину и завершение ответа |
| энтропия падает | проверить размер шага и число уникальных ответов: модель могла потерять разнообразие |
| у 90% групп нулевая дисперсия | проверить долю успехов: задачи могут быть слишком лёгкими или слишком трудными |
| KL резко меняется | проверить версии модели и отношения вероятностей: ответы могли устареть, а обновление — оказаться чрезмерным |
| функция потерь становится NaN | проверить $\epsilon$, конечность вероятностей и маски; малый разброс наград усиливает ошибки |
| формат улучшается без правильности | разделить составляющие награды: вес требования к формату может подавлять правильность |

Минимальный набор наблюдений включает составляющие обучающей награды,
pass@1 и pass@$k$ на отложенных задачах, долю неоднородных групп, квантили длины
ответов, энтропию, KL-дивергенцию, долю ограниченных обновлений, хвосты отношений
вероятностей, достижение токена конца и число сгенерированных токенов на одно
успешное решение.

## Открытые реализации

- [Open-R1](https://github.com/huggingface/open-r1): примеры перехода
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

### Проверка расчёта

В группе из четырёх ответов с наградами $(1,0,0,0)$ получаем
$\bar r=0.25$ и $s=\sqrt{3}/4$. Успешный ответ имеет преимущество
$\sqrt3\approx1.732$, каждый неуспешный — $-1/\sqrt3\approx-0.577$.
Сумма равна нулю, но это не означает нулевого градиента: преимущества
умножаются на градиенты **разных** ответов. Обнуление всего обновления
произошло бы лишь при специальных совпадениях этих градиентов.

Если убрать деление на $s$, преимущества станут $(0.75,-0.25,-0.25,-0.25)$.
Если также убрать базовый уровень и оставить только успешные решения,
веса будут $(1,0,0,0)$ — вариант RFT. Так три метода можно сопоставить
на одних и тех же ответах, не смешивая различия алгоритма и генерации данных.

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
  R1-Zero, стадии R1, результаты и дистилляция; расширенная версия v2
  уточняет данные, гиперпараметры и протоколы оценки.
- Liu et al., 2025 — [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783):
  Dr. GRPO, анализ нормировок по разбросу наград и длине ответа.
- Tajwar et al., 2026 — [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710):
  исходная постановка MaxRL; её знаменатель отличается от учебной абляции A5.
- Zheng et al., 2025 — [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071):
  геометрическое среднее отношений вероятностей и ограничение на уровне ответа.
- Kimi Team, 2025 — [Kimi k1.5](https://arxiv.org/abs/2501.12599), и Yang et
  al., 2025 — [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388):
  многостадийное обучение рассуждениям и стоимость генерации.
- Ahmadian et al., 2024 — [Back to Basics: REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740):
  оценки градиента без отдельной функции ценности.
