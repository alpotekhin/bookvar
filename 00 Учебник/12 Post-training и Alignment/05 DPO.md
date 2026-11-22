---
title: "DPO: обучение на парах ответов"
type: textbook-chapter
status: canonical
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - anchor: dpo-stages
    label: "Какие этапы RLHF заменяет DPO"
    label_en: "Which RLHF stages DPO replaces"
  - anchor: dpo-sequence-logp
    label: "Логарифм вероятности полного ответа"
    label_en: "Full-response log probability"
  - anchor: dpo-preference-model
    label: "Вероятность сравнения в модели Брэдли—Терри"
    label_en: "Bradley–Terry comparison probabilities"
  - anchor: dpo-rhf-objective
    label: "KL-регуляризованная задача и распределение Больцмана"
    label_en: "KL-regularized objective and Boltzmann policy"
  - anchor: dpo-derivation
    label: "Награда через отношение вероятностей моделей"
    label_en: "Rewards from policy probability ratios"
  - anchor: dpo-reward-normalization
    label: "Нормировка и восстановление награды: два ответа"
    label_en: "Two-response normalization and reward recovery"
  - anchor: dpo-partition-cancellation
    label: "Сокращение нормировочной константы"
    label_en: "Partition-function cancellation"
  - anchor: dpo-margin
    label: "Относительный разрыв и числовой расчёт DPO"
    label_en: "Relative margins and a numeric DPO calculation"
  - anchor: dpo-gradient
    label: "Логистический вес и градиент DPO"
    label_en: "Logistic weighting and DPO gradients"
  - anchor: dpo-beta
    label: "Коэффициент β, оптимум и размер градиента"
    label_en: "Beta, optimum and gradient magnitude"
  - anchor: dpo-dataflow
    label: "Маски, causal shift и сумма logp"
    label_en: "Masks, causal shift and summed log probabilities"
  - anchor: dpo-token-batch
    label: "Исполняемый шаг DPO с Llama"
    label_en: "An executable Llama DPO update"
  - anchor: dpo-reference
    label: "Замороженная опорная модель и кэш"
    label_en: "Frozen reference models and cached log probabilities"
  - anchor: dpo-offline-boundary
    label: "Задание CS336 и границы offline DPO"
    label_en: "The CS336 task and offline DPO boundaries"
  - anchor: dpo-controlled-comparison
    label: "Контролируемое сравнение PPO и DPO"
    label_en: "Controlled PPO versus DPO comparisons"
  - anchor: dpo-irpo
    label: "IRPO: новые пары и дополнительная NLL"
    label_en: "IRPO resampling and the additional NLL"
  - anchor: dpo-tulu-length-normalization
    label: "Tülu 3 и DPO с нормировкой по длине"
    label_en: "Tülu 3 and length-normalized DPO"
  - anchor: dpo-preference-data
    label: "Происхождение пар, критерии и шум меток"
    label_en: "Pair provenance, criteria and label noise"
  - anchor: dpo-failures
    label: "Ограничения DPO и падение обеих вероятностей"
    label_en: "DPO limitations and decreasing response probabilities"
  - anchor: dpo-diagnostics
    label: "Метрики пар и качество новых генераций"
    label_en: "Pair metrics versus new-generation quality"
  - anchor: dpo-normalized-mass-kl
    label: "Перераспределение вероятности и точный KL"
    label_en: "Probability redistribution and exact KL"
  - anchor: dpo-ipo
    label: "IPO: конечный целевой разрыв"
    label_en: "IPO finite target margins"
  - anchor: dpo-simpo
    label: "SimPO: длина и отступ без reference"
    label_en: "SimPO length normalization and reference-free margins"
  - anchor: dpo-orpo
    label: "ORPO: NLL и отношение шансов"
    label_en: "ORPO NLL and odds ratios"
  - anchor: dpo-kto
    label: "KTO: положительные и отрицательные метки без пары"
    label_en: "KTO unpaired desirable and undesirable labels"
  - anchor: dpo-online
    label: "Online DPO и обновление набора пар"
    label_en: "Online DPO and refreshed preference pairs"
aliases:
  - DPO
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM]]"
previous: "[[00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF]]"
next:
  - "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers]]"
primary_sources:
  - https://arxiv.org/abs/2305.18290
  - https://arxiv.org/abs/2404.19733
  - https://arxiv.org/abs/2406.09279
  - https://arxiv.org/abs/2411.15124
source_unit_id:
  - assignment-05-safety-supplement-dpo-objective-and-training
  - assignment-05-safety-supplement-task-look-at-hh
  - lecture-15-dpo-derivation
  - meeting-02-slides-irpo
  - meeting-02-reading-01-catalogue-record
  - meeting-02-reading-02-catalogue-record
  - meeting-04-slides-preference-optimization
  - meeting-04-slides-tulu3-preference-recipe
  - meeting-04-reading-02-catalogue-record
---

# DPO: обучение на парах ответов

«Гарантирует ли UDP доставку пакетов?» На этот вопрос модель дала два
ответа: «Нет, потерянные пакеты должно обрабатывать приложение» и
«Да, UDP повторно передаёт потерянные пакеты». Для SFT можно сохранить
запрос и правильный ответ. Для Direct Preference Optimization (DPO)
сохраняют оба ответа и отмечают, какой предпочтительнее.

Чтобы вычислить потерю, нужны четыре числа: вероятности двух ответов
у обучаемой и у замороженной опорной модели. Допустим, опорная модель
порождает верный ответ в 1,22 раза чаще ошибочного, а обучаемая в
2,72 раза чаще. Потеря зависит от того, насколько различаются эти отношения.
Если бы обе модели давали одно и то же отношение, функция потерь
не видела бы усиления предпочтения, даже если бы обе чаще давали
правильный ответ. Ниже получим эти отношения из логарифмов вероятностей
и вычислим потерю для пары.

<a id="dpo-stages"></a>

<span id="1-какую-часть-rlhf-упрощает-dpo"></span>

## DPO и RLHF

Запись содержит запрос $x$ и два ответа:

$$
y_w \succ y_l,
$$

где $y_w$ — выбранный ответ, $y_l$ — отвергнутый. Метку может дать человек
или модель, как в [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF|главе о Constitutional AI и RLAIF]].
В обычном RLHF сначала на таких парах обучают модель награды. Затем
текущий помощник генерирует новые ответы, модель награды оценивает их,
а PPO обновляет веса помощника. DPO сразу обновляет помощника по готовым
парам. Для этого в формуле парного сравнения оценку отдельной сети
заменяют выражением через вероятности языковой модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/fig1.png|RLHF с моделью награды и DPO на готовых парах]]

*Слева сравнения обучают модель награды, а по её оценкам обновляют
помощника с помощью RL. Справа сравнения сразу задают потерю для помощника.
Rafailov et al., [Figure 1](https://arxiv.org/html/2305.18290v3#S1.F1),
[оригинальный PNG](https://arxiv.org/html/2305.18290v3/figures/diagrams/teaser.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Попарная модель Брэдли — Терри нужна в обоих случаях. В левой ветви её
оценки выдаёт отдельная сеть. В правой их вычисляют из вероятностей
помощника и опорной модели.

<a id="dpo-sequence-logp"></a>

<span id="2-вероятность-целого-ответа"></span>

## Вероятность целого ответа

Для авторегрессионной языковой модели

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T_y}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

Если модель дала трём токенам ответа условные вероятности $0.5$,
$0.2$ и $0.8$, вероятность последовательности равна
$0.5\cdot0.2\cdot0.8=0.08$. Её логарифм, около $-2.526$, получается
сложением трёх логарифмов. Среднее по токенам, $-0.842$, соответствует
уже другой величине. Суммирование в логарифмической шкале позволяет
работать с длинными ответами, для которых произведение вероятностей
может округлиться до нуля.

Если в шаблоне есть токен конца ответа, его вероятность включают в
сумму. Токены запроса из суммы исключают: они задают условие для
каждого предсказания. Ниже на пакете из двух пар отдельно проверим
маски запроса, заполнителя и настоящего EOS.

При добавлении токена к сумме прибавляется неположительное число.
Однако каждый ответ DPO оценивает относительно опорной модели.
Для влияния нового токена важно, какая из двух моделей придала ему
большую условную вероятность. По одной только длине знак изменения
определить нельзя. В Tülu 3 вместо суммы брали среднее: это отдельная
разновидность функции потерь.

<a id="dpo-preference-model"></a>

<span id="3-модель-предпочтений-как-отправная-точка"></span>

## Вероятность победы в сравнении

В модели Брэдли — Терри два ответа сравнивают по разности их наград:

$$
P(y_w\succ y_l\mid x)
=\sigma(r(x,y_w)-r(x,y_l)),
$$

где $\sigma(z)=1/(1+e^{-z})$. Это вероятность **результата сравнения**, а не
вероятность сгенерировать выбранный ответ. Если разность наград равна нулю,
модель предпочтений считает оба исхода равновероятными. Разность $\log3$
означает вероятность предпочтения $0.75$; это не утверждение, что ответ верен
с вероятностью $75\%$. Обычная функция потерь модели награды:

$$
\mathcal L_{RM}
=-\mathbb E\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
$$

Если к обеим наградам прибавить $C(x)$, разность не изменится.
Поэтому парные сравнения не определяют общий сдвиг награды для данного
запроса. Например, оценки $2$ и $1$ дают ту же вероятность победы,
что $1$ и $0$.

<span id="4-задача-rlhf-с-ограничением-по-kl-дивергенции"></span>

## Целевая функция с KL-штрафом

<a id="dpo-rhf-objective"></a>

Для фиксированного запроса $x$ выберем распределение ответов $\pi$: оно
должно давать большую среднюю награду и не слишком отличаться от опорного.
Отклонение штрафуем через KL-дивергенцию ([Rafailov et al., уравнение 3](https://arxiv.org/html/2305.18290v3#S3)):

$$
\begin{gathered}
\max_\pi J(\pi),\\
J(\pi)=\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]\\
-\beta D_{KL}(\pi\Vert\pi_{ref}).
\end{gathered}
$$

$\pi_{ref}$ — обычно модель после SFT, а $\beta>0$ определяет цену отклонения от
неё. Рассматриваются ответы с положительной вероятностью у опорной модели;
предполагается, что определённая ниже нормировочная сумма $Z(x)$ конечна.
Раскроем математическое ожидание при ограничении $\sum_y\pi(y\mid x)=1$:

$$
\begin{aligned}
J(\pi)&=\sum_y\pi(y\mid x)r(x,y)\\
&\quad-\beta\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}.
\end{aligned}
$$

Добавим множитель Лагранжа $\lambda$:

$$
\mathcal F=J(\pi)+\lambda\left(\sum_y\pi(y\mid x)-1\right).
$$

Условие стационарности по $\pi(y\mid x)$:

$$
r(x,y)+\lambda=\beta\left(
\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}+1\right).
$$

После перегруппировки и нормировки получаем распределение Больцмана:

$$
\pi_r(y\mid x)=
\frac{\pi_{ref}(y\mid x)}{Z(x)}
\exp\left(\frac{r(x,y)}{\beta}\right),
$$

$$
Z(x)=\sum_y\pi_{ref}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

Опорная модель задаёт исходную массу каждого ответа. Умножаем её
на $\exp(r/\beta)$, так что более высокая награда даёт больший множитель.
Деление на $Z(x)$ возвращает сумму вероятностей к единице. Так выглядит
оптимум в задаче, где можно выбрать любое распределение на носителе
опорной модели; у конкретной сети выбор ограничен её параметрами.

Целевую функцию также можно записать как

$$
J(\pi)=\beta\log Z(x)-\beta D_{KL}(\pi\Vert\pi_r).
$$

Первое слагаемое не зависит от $\pi$, а KL-дивергенция неотрицательна. Поэтому
максимум достигается при $\pi=\pi_r$. Это также показывает, почему важна
опорная модель: если она приписывает ответу нулевую вероятность, конечная
награда не добавит этому ответу массу в таком решении.

<span id="5-выражаем-скрытую-награду-через-языковую-модель"></span>

## Награда через вероятности модели

<a id="dpo-derivation"></a>
<!-- source_unit_id: lecture-15-dpo-derivation -->
<!-- source_unit_id: assignment-05-safety-supplement-dpo-objective-and-training -->

Логарифм оптимальной вероятности разделяет вклад опорной модели,
награды и нормировки:

$$
\begin{aligned}
\log\pi_r(y\mid x)&=\log\pi_{ref}(y\mid x)\\
&\quad+\frac{r(x,y)}{\beta}-\log Z(x).
\end{aligned}
$$

Отсюда

$$
r(x,y)=
\beta\log\frac{\pi_r(y\mid x)}{\pi_{ref}(y\mid x)}
+\beta\log Z(x).
$$

<a id="dpo-reward-normalization"></a>

### Два ответа: нормировка и восстановление награды

Возьмём опорное распределение $(0.5,0.5)$, награды $(1,0)$ и $\beta=0.5$.
До нормировки веса ответов равны $0.5e^2$ и $0.5$; их сумма
$Z=4.194528$. После деления на неё получаем $(0.880797,0.119203)$.
Первый ответ стал вероятнее, но второй сохранил ненулевую вероятность.
Если теперь подставить это распределение обратно в выражение для награды,
то $\beta\log(\pi_r/\pi_{ref})$ даст $(0.283110,-0.716890)$.
Разность по-прежнему равна 1; добавление общей константы
$\beta\log Z=0.716890$ восстановит исходные $(1,0)$.

<details id="dpo-code-boltzmann">
<summary>Код: распределение Больцмана для двух ответов</summary>

```python
import math
import torch

reference_probability = torch.tensor([0.5, 0.5], dtype=torch.float64)
explicit_reward = torch.tensor([1.0, 0.0], dtype=torch.float64)
reward_beta = 0.5
log_weight = reference_probability.log() + explicit_reward / reward_beta
log_partition = torch.logsumexp(log_weight, dim=0)
optimal_probability = (log_weight - log_partition).exp()
implicit_reward = reward_beta * (
    optimal_probability.log() - reference_probability.log()
)
recovered_reward = implicit_reward + reward_beta * log_partition
torch.testing.assert_close(recovered_reward, explicit_reward)
print(optimal_probability.tolist())
print(implicit_reward.tolist())
```

</details>

Здесь сумма состоит из двух слагаемых. У языковой модели пришлось бы
просуммировать веса всех допустимых текстов.

<a id="dpo-partition-cancellation"></a>

<span id="почему-нормировка-сокращается"></span>

### Сокращение нормировочной константы

Для двух ответов на один запрос обозначим $r_w=r(x,y_w)$ и $r_l=r(x,y_l)$:

$$
\begin{aligned}
r_w-r_l&=\beta\biggl[
\log\frac{\pi_r(y_w\mid x)}{\pi_{ref}(y_w\mid x)}\\
&\qquad-\log\frac{\pi_r(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\biggr].
\end{aligned}
$$

Оба ответа даны на один запрос, поэтому добавка $\beta\log Z(x)$ у них
одинакова и исчезает при вычитании. Все остальные тексты исключились
из формулы сравнения.

Теперь возьмём распределение обучаемой модели $\pi_\theta$ на том же
носителе и определим $\hat r_\theta=\beta\log(\pi_\theta/\pi_{ref})$.
Если подставить эту награду обратно в распределение Больцмана, числитель будет равен
$\pi_\theta$, а сумма числителей — единице. Значит, $\pi_\theta$ и есть
оптимальная стратегия для такой награды. Выбирая веса $\theta$, мы
одновременно меняем стратегию и награду. Подстановка в сравнение
Брэдли — Терри даёт функцию потерь [Rafailov et al., §4](https://arxiv.org/html/2305.18290v3#S4):

$$
\boxed{\begin{aligned}
\mathcal L_{DPO}&=-\mathbb E_{\mathcal D}\log\sigma(z_\theta),\\
z_\theta&=\beta\biggl[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}\\
&\qquad-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\biggr].
\end{aligned}}
$$

Усреднение идёт по готовым записям $(x,y_w,y_l)$ набора $\mathcal D$.
Для каждой записи достаточно посчитать четыре логарифма вероятностей.
Ни отдельная сеть награды, ни генерация новых ответов в этом шаге не
нужны. Качество результата по-прежнему зависит от того, согласуются ли
метки с моделью Брэдли — Терри и какие ответы попали в набор.

<a id="dpo-margin"></a>

<span id="6-что-именно-сравнивает-функция-потерь-dpo"></span>

## Функция потерь DPO

Введём изменение логарифма вероятности относительно опорной модели:

$$
\begin{aligned}
\Delta_\theta(x,y)&=\log\pi_\theta(y\mid x)\\
&\quad-\log\pi_{ref}(y\mid x).
\end{aligned}
$$

Тогда аргумент сигмоиды в DPO равен

$$
z=\beta[\Delta_\theta(x,y_w)-\Delta_\theta(x,y_l)].
$$

При $z>0$ отношение вероятностей выбранного и отвергнутого ответов у
обучаемой модели выше, чем у опорной. Даже если
$\pi_\theta(y_w)>\pi_\theta(y_l)$, логит может быть отрицательным:
например, если это отношение было 10 у опорной модели и стало 3
у обучаемой.

<span id="ручной-пример"></span>

### Потеря для пары про UDP

Подставим логарифмы вероятностей двух ответов про UDP. $y_w$ — верный
ответ о том, что доставку UDP не гарантирует:

| Модель | $y_w$ | $y_l$ |
|---|---:|---:|
| текущая | −3.0 | −4.0 |
| опорная | −3.4 | −3.6 |

Тогда

$$
\begin{aligned}
\Delta_w&=-3.0-(-3.4)=0.4,\\
\Delta_l&=-4.0-(-3.6)=-0.4.
\end{aligned}
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

Два отношения из начала главы равны $e^{1.0}\approx2.72$ и
$e^{0.2}\approx1.22$. Их логарифмы отличаются на $0.8$. После умножения
на $\beta$ и сигмоиды получаем $0.599$ для парного сравнения.
Вероятность, что помощник сгенерирует выбранный текст, равна
$e^{-3}\approx0.050$.

Если текущая модель повысит логарифмы вероятностей обоих ответов относительно
опорной на одинаковые `+0.4`, разность останется нулевой: DPO не обнаружит
усиления предпочтения выбранного ответа.

<a id="dpo-gradient"></a>

<span id="как-действует-градиент"></span>

### Градиент и вес пары

Неявная награда, соответствующая текущей модели:

$$
\hat r_\theta(x,y)=
\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

Обозначим градиенты логарифмов вероятностей как
$g_w=\nabla_\theta\log\pi_\theta(y_w\mid x)$ и
$g_l=\nabla_\theta\log\pi_\theta(y_l\mid x)$. Градиент функции потерь DPO равен

$$
\begin{aligned}
w&=\sigma(\hat r_l-\hat r_w),\\
\nabla_\theta L_{DPO}&=-\beta\mathbb E[w(g_w-g_l)].
\end{aligned}
$$

Если временно считать $\ell_w=\log\pi_\theta(y_w\mid x)$ и
$\ell_l=\log\pi_\theta(y_l\mid x)$ независимыми координатами, то
$\partial L/\partial\ell_w=-\beta\sigma(-z)$ и
$\partial L/\partial\ell_l=+\beta\sigma(-z)$. В этих координатах спуск
направлен к росту выбранного и снижению отвергнутого ответа, особенно для
неправильно ранжированных пар. Но обе величины зависят от общих параметров
Transformer: реальное обновление может уменьшить обе вероятности, если
вероятность отвергнутого падает сильнее. Поэтому далее в IRPO отдельно
появляется NLL-компонента, непосредственно поддерживающая выбранную цепочку.

В примере про UDP вес равен $\sigma(-0.4)\approx0.401$. Если логит пары
равен $-2$, вес возрастает до $\sigma(2)\approx0.881$: такой пример
сильнее влияет на обновление при одинаковых градиентах ответов.
В абляции [Rafailov et al., Appendix C.3](https://arxiv.org/html/2305.18290v3#A3.SS3)
вариант без этого веса приводил к вырождению модели в опыте с суммаризацией.

При инициализации $\pi_\theta=\pi_{ref}$ обе неявные награды равны нулю:
$z=0$, $P_\theta(y_w\succ y_l)=1/2$, $L=\log2$. При этом производная
по $z$ равна $-1/2$, а не нулю: совпадение моделей не мешает начать обучение.
Градиент по параметрам может обнулиться для вырожденной пары — например,
когда выбранный и отвергнутый ответы совпадают и их градиенты вычитаются.

<a id="dpo-beta"></a>

<a id="7-что-означает-βbetaβ"></a>
<span id="7-что-означает-коэффициент-β"></span>

## Коэффициент β

В исходной задаче $\beta$ — цена отклонения от опорного распределения.
Рассмотрим модель с двумя возможными ответами и вероятностями $0.5$ и $0.5$.
Пусть ответы $y_1$ и $y_2$ имеют награды $1$ и $0$. В аналитическом оптимуме вероятность
первого ответа равна $\sigma(1/\beta)$:

| $\beta$ | $\pi_r(y_1)$ | $\pi_r(y_2)$ |
|---:|---:|---:|
| 0.5 | 0.881 | 0.119 |
| 1 | 0.731 | 0.269 |
| 2 | 0.622 | 0.378 |

При большей цене отклонения распределение остаётся ближе к исходным $0.5/0.5$.
В DPO награды из таблицы не заданы. Допустим, из повторных сравнений
одной и той же пары известно, что доля побед выбранного ответа равна
$q$, где $0<q<1$. Логистическая потеря достигает минимума при
$\sigma(\beta m)=q$. Для такой пары оптимальный разрыв равен

$$
m=\Delta_w-\Delta_l=\frac{1}{\beta}\log\frac{q}{1-q}.
$$

Для $q=0.75$ при $\beta=0.5$ нужен разрыв $m\approx2.197$, а при $\beta=2$
достаточно $m\approx0.549$. Меньший коэффициент может требовать более сильного
изменения отношений вероятностей, чтобы объяснить те же предпочтения.

Отдельно можно посчитать производную для одной размеченной победы.
При фиксированном $m$ её модуль равен $\beta\sigma(-\beta m)$.
Например, для $m=0.8$ он составляет $0.048$, $0.201$ и $0.090$ при
$\beta=0.1$, $0.5$ и $5$ соответственно: зависимости «чем меньше, тем сильнее»
здесь нет. При $m=0$ коэффициент равен $\beta/2$.

$\beta$ нельзя называть скоростью обучения: скорость обучения меняет величину
шага оптимизатора, тогда как $\beta$ меняет саму статистическую задачу.
Кроме того, он не задаёт жёсткий верхний предел KL: расхождение нужно измерять
на новых генерациях. Его значение зависит также от данных, длительности
обучения и того, суммирует ли реализация логарифмы вероятностей или усредняет
их по токенам.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo-review-2026-09/reward-kl-frontier.png|Награда и KL в опыте DPO на тональности IMDb]]

*Награда и отклонение от опорной модели измеряются отдельно. По горизонтали —
KL, по вертикали — оценка положительной тональности продолжения отзыва IMDb.
Точки получены при разных настройках и на разных шагах обучения: это не график
наград как функции одного только $\beta$. В этом опыте DPO даёт более высокую
награду при сопоставимом KL. Rafailov et al.,
[Direct Preference Optimization, Figure 2, левая панель](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf#page=7).
Из оригинала выделена одна панель; подписи не изменены. Иллюстрация также
опубликована в [версии статьи с CC BY 4.0](https://arxiv.org/html/2305.18290v3).*

Если выбрать наибольшую награду на каждой кривой, KL в точках может
отличаться. Для сравнения выберите один уровень KL и посмотрите, какая
кривая выше. Здесь DPO даёт преимущество для GPT-2 и классификатора
тональности IMDb. Другие датасеты и показатели ниже дают иную картину.

<span id="8-данные-и-порядок-вычислений"></span>

## Расчёт потерь в одном пакете

<a id="dpo-dataflow"></a>
<!-- source_unit_id: assignment-05-safety-supplement-task-look-at-hh -->

В наборе предпочтений пара про UDP занимает одну запись:

```json
{
  "prompt": "Гарантирует ли UDP доставку пакетов?",
  "chosen": "Нет, потерянные пакеты должно обрабатывать приложение.",
  "rejected": "Да, UDP повторно передаёт потерянные пакеты."
}
```

К общему запросу по очереди приписывают два ответа. Каждую
последовательность оценивают обе модели. Даже при объединении ответов
в один пакет из него должны вернуться четыре суммы на пару:

```text
prompt + chosen   → logits [B,Lw,V] → log_softmax → gather → sum → logp_w [B]
prompt + rejected → logits [B,Ll,V] → log_softmax → gather → sum → logp_l [B]

Те же последовательности → опорная модель → ref_logp_w [B], ref_logp_l [B]

chosen_logratio = logp_w - ref_logp_w
reject_logratio = logp_l - ref_logp_l
logit          = β(chosen_logratio - reject_logratio)
loss           = -logsigmoid(logit)
```

Выбранные и отвергнутые ответы часто объединяют в один пакет на модель.
Тогда два отдельных прохода заменяются одним. Если ответы сильно
различаются по длине, в объединённом пакете может оказаться больше
заполнителя. Эту проблему решают группировкой по длине.

Для суммирования логарифмов нужны логиты и массив `labels`. В нём
сохранены ID токенов ответа, а запрос и заполнитель отмечены `-100`,
как при SFT. В паре `logits[:, t]` и `labels[:, t + 1]` первый тензор
описывает предсказание следующего токена, а второй указывает, какой
из логарифмов выбрать. Назначение маски внимания другое: она исключает
заполнитель из контекста модели.

<details id="dpo-code-loss">
<summary>Код: сумма логарифмов вероятностей и DPO loss</summary>

```python
import math
import torch
import torch.nn.functional as F


def sequence_logp(logits, labels):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share batch and time axes")
    if logits.size(0) == 0 or logits.size(1) < 2 or logits.size(2) == 0:
        raise ValueError("empty batch, missing prediction position or empty vocabulary")
    if labels.dtype != torch.long:
        raise ValueError("labels must contain integer token ids")
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    if not mask.any(dim=-1).all():
        raise ValueError("each response must contain at least one target token")
    safe_targets = targets.masked_fill(~mask, 0)
    if torch.any((safe_targets < 0) | (safe_targets >= logits.size(-1))):
        raise ValueError("target token outside vocabulary")
    # Position t predicts target token t + 1.
    prediction_logits = logits[:, :-1]
    if prediction_logits.dtype in (torch.float16, torch.bfloat16):
        prediction_logits = prediction_logits.float()
    token_logp = prediction_logits.log_softmax(-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    return token_logp.masked_fill(~mask, 0.0).sum(-1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.5):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    if pi_w.ndim != 1 or pi_w.numel() == 0 or any(
        x.shape != pi_w.shape for x in (pi_l, ref_w, ref_l)
    ):
        raise ValueError("four log-probability vectors must have the same nonempty shape")
    margin = (pi_w - pi_l) - (ref_w.detach() - ref_l.detach())
    return -F.logsigmoid(beta * margin).mean()
```

</details>

Значение `-100` нельзя передать в `gather` как индекс словаря. Поэтому сначала
оно временно заменяется нулём, а затем выбранные таким способом значения
исключаются маской. Для каждого ответа остаётся одна сумма, и только после
попарного сравнения потери усредняются по записям. Это не усреднение токеновых
потерь по всему пакету.

Перед обучением параметры опорной модели замораживают и переводят её в режим
`eval()`, чтобы dropout не менял оценку одного и того же ответа. Её проходы
выполняют внутри `torch.no_grad()`; четыре вектора, передаваемые в `dpo_loss`,
имеют форму `[B]`. Например, `pi_w` получается вызовом
`sequence_logp(policy(**chosen_inputs).logits, chosen_labels)`. Маска внимания
в `chosen_inputs` исключает заполнение из контекста; маска в `chosen_labels`
выбирает предсказываемые токены. Эти маски решают разные задачи.

Служебные токены, обрезка и окончание ответа должны совпадать у обеих моделей.
Если ID заполнителя совпадает с EOS, маску строят по фактическим позициям
заполнения, а не по равенству ID: иначе из суммы пропадёт настоящий конец
ответа. Для фиксированных пар опорные суммы можно вычислить заранее, но такой
кэш пригоден только при неизменных весах, токенизации и обрезке.

Эта реализация следует порядку операций в
[официальном коде DPO, `preference_loss` и `_get_batch_logps`](https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py).
Дополнительно она отвергает записи, где после сдвига не осталось ни одного
целевого токена: такой пример обычно означает ошибку маски или обрезки.

<a id="dpo-token-batch"></a>

<span id="от-токенов-до-одного-шага-обучения"></span>

### Один шаг на маленькой Llama

Ниже два запроса и по два ответа к каждому. Числа — условные ID, а не
токенизация фраз об UDP. Первая пара имеет общий запрос `[1,3,4]`, вторая —
`[1,6]`. ID 2 обозначает и настоящий конец ответа, и заполнитель.
Во второй выбранной последовательности два последних ID 2 — заполнение;
предыдущий ID 2 — EOS, и он остаётся целевым токеном.
В первой строке целевые позиции 3, 4 и 5 (нумерация с нуля).
Для ID 8, 9 и 2 берём логарифмы вероятностей из логитов на позициях
2, 3 и 4 и складываем их.

Маска внимания указывает действительные позиции контекста. В `labels`
дополнительно закрыт запрос: на нём DPO не суммирует логарифмы вероятностей,
хотя последующие предсказания от него зависят.

<details id="dpo-code-batch">
<summary>Код: две пары, токены и маски</summary>

```python
import copy
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(23)
torch.set_num_threads(1)
dpo_config = LlamaConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=32, attention_dropout=0.0,
    bos_token_id=1, eos_token_id=2, pad_token_id=2,
)
dpo_config._attn_implementation = "eager"
dpo_policy = LlamaForCausalLM(dpo_config)
dpo_reference = copy.deepcopy(dpo_policy).eval().requires_grad_(False)
dpo_optimizer = torch.optim.AdamW(dpo_policy.parameters(), lr=1e-4)

# Сначала chosen для обоих запросов, затем rejected в том же порядке.
pair_ids = torch.tensor([
    [1, 3, 4,  8,  9, 2], [1, 6, 10,  2,  2, 2],
    [1, 3, 4, 11,  2, 2], [1, 6, 12, 13, 14, 2],
])
pair_attention = torch.tensor([
    [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1],
])
pair_labels = torch.tensor([
    [-100, -100, -100,  8,    9,    2],
    [-100, -100,   10,  2, -100, -100],
    [-100, -100, -100, 11,    2, -100],
    [-100, -100,   12, 13,   14,    2],
])

def model_response_logp(model, ids, attention, labels):
    # Логические номера позиций сохраняют результат и при левом padding.
    positions = attention.cumsum(-1) - 1
    positions = positions.masked_fill(attention == 0, 0)
    logits = model(input_ids=ids, attention_mask=attention,
                   position_ids=positions, use_cache=False).logits
    return sequence_logp(logits, labels)
```

</details>

Опорные суммы считаются один раз и остаются константами. Обучаемая модель
обрабатывает тот же пакет, после чего первая половина вектора сравнивается
со второй. Поскольку исходные веса совпадают, потеря до первого шага равна
$\log2$. Градиент при этом ненулевой.

<details id="dpo-code-update">
<summary>Код: один шаг обновления Llama</summary>

```python
with torch.no_grad():
    reference_pair_logp = model_response_logp(
        dpo_reference, pair_ids, pair_attention, pair_labels
    )
policy_pair_logp = model_response_logp(
    dpo_policy, pair_ids, pair_attention, pair_labels
)
batch_chosen, batch_rejected = policy_pair_logp.chunk(2)
reference_chosen, reference_rejected = reference_pair_logp.chunk(2)
batch_dpo_loss = dpo_loss(batch_chosen, batch_rejected,
                         reference_chosen, reference_rejected, beta=0.5)
torch.testing.assert_close(batch_dpo_loss.detach(), torch.tensor(math.log(2)))
dpo_optimizer.zero_grad(set_to_none=True)
batch_dpo_loss.backward()
dpo_optimizer.step()
with torch.no_grad():
    updated_pair_logp = model_response_logp(
        dpo_policy, pair_ids, pair_attention, pair_labels
    )
    updated_loss = dpo_loss(*updated_pair_logp.chunk(2),
                            reference_chosen, reference_rejected, beta=0.5)
print("response token counts:", pair_labels[:, 1:].ne(-100).sum(-1).tolist())
print(f"DPO loss: {batch_dpo_loss.item():.6f} -> {updated_loss.item():.6f}")
```

</details>

В пакете осталось три, два, два и четыре целевых токена соответственно.
На случайной маленькой Llama проверяются сдвиг, маски и один шаг
обновления. Модель не учится вести диалог, веса и данные из сети
не загружаются. В тесте те же ответы оцениваются без заполнения,
с заполнением слева и справа; настоящий EOS остаётся в сумме.
Листинги выполняются по порядку в одной среде. Проверенная CPU-среда:
PyTorch 2.14.1 и Transformers 4.57.1.

<a id="dpo-reference"></a>

<span id="9-откуда-берётся-опорная-модель"></span>

## Опорная модель

После SFT сохраняют одну контрольную точку и берут из неё обе модели.
Обучаемая копия начинает менять вероятности ответов; замороженная копия
измеряет те же ответы в исходном распределении. Поэтому сравнивать
можно изменения от одного и того же состояния.

Иногда неизвестно, какая модель породила ответы в наборе, или её веса
недоступны. В [Rafailov et al., §4](https://arxiv.org/html/2305.18290v3#S4)
в таком случае рекомендуют сначала провести SFT на выбранных ответах
$(x,y_w)$ и взять результат за опорную модель и начальную точку DPO.

Опорная модель участвует и в функции потерь, и в определении неявной награды.
Для её работы не обязательно постоянно держать вторую полную копию в памяти:
можно сохранить заранее вычисленные суммы или, в некоторых схемах обучения
адаптеров, получать опорную оценку из замороженной базовой модели. Это способы
организации вычислений; математически сравнение с опорным распределением
остаётся тем же.
Фраза «DPO без опорной модели» относится к более поздним вариантам метода, но
не к стандартному DPO.

<span id="10-dpo-и-ppo-общая-постановка-разные-способы-обучения"></span>

## Сравнение DPO и PPO

<a id="dpo-offline-boundary"></a>
<!-- source_unit_id: meeting-02-reading-01-catalogue-record -->
<!-- source_unit_id: meeting-04-slides-preference-optimization -->
<!-- source_unit_id: meeting-04-reading-02-catalogue-record -->

<a id="дополнительная-ветка-assignment-5-от-hh-пары-до-одного-loss"></a>
<a id="дополнительная-ветка-assignment-5-функция-потерь-для-пары-hh"></a>
### Функция потерь для пары HH в CS336

Stanford отделяет этот опыт от основного задания по обучению рассуждений
с подкреплением. Сначала студент загружает пары Helpful–Harmless (HH) и вручную
проверяет, согласен ли с метками. В §6.2 дополнения объединяются четыре
обучающих раздела: `harmless-base`, `helpful-base`, `helpful-online`
и `helpful-rejection-sampled`. Источник каждой строки сохраняют для анализа.
Многоходовые диалоги, где пользователь писал больше одного сообщения,
в этом задании исключаются: после первого ответа дальнейшие сообщения
в выбранной и отвергнутой беседах могут различаться. Это уже не два ответа
при одном и том же условии $x$. Правило относится к выбранной подготовке HH;
DPO в принципе допускает общий многоходовый контекст, если он совпадает.

Затем одна и та же последовательность
`prompt + response` проходит через обучаемую и замороженную опорную модели.
Для каждого ответа суммируются логарифмы вероятностей
**только его токенов**:

$$
\ell_\theta(y\mid x)=\sum_{t\in response}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

Интерфейс `run_compute_per_instance_dpo_loss` получает обе модели, общий
токенизатор, $\beta$, запрос, выбранный и отвергнутый ответы и возвращает одно
значение функции потерь. Проверка должна обнаруживать две частые ошибки:
усреднение по длине вместо суммы и несовпадающие шаблоны диалога у моделей.
Выше используются условные вероятности ответов, поэтому запрос в сумму не
входит. Если вместо них вычислять вероятности всей последовательности,
слагаемые общего запроса сократятся между ответами только при одинаковой
токенизации и контексте; различная обрезка может нарушить это условие.

После DPO сравнение проводят с исходной SFT-моделью сразу по нескольким осям:
предпочтения (AlpacaEval), безопасность (SimpleSafetyTests), математика (GSM8K)
и знания (MMLU). Рост одной метрики не доказывает общее улучшение. Эта ветка
использует заранее собранные данные: в ходе оптимизации новые ответы не
генерируются. Она не заменяет RLVR с ответами текущей обучаемой модели.

В учебной реализации §6.4 обе модели помещают на разные GPU и используют
накопление градиентов вместо большого физического пакета. Начальные настройки
задания: эффективный пакет 64, $\beta=0.1$, скорость обучения $10^{-6}$,
одна эпоха и небольшой validation-набор примерно из 200 записей.
Из-за памяти состояний оптимизатора предлагается RMSprop, а не AdamW.
Эти настройки описывают конкретное задание с большой моделью; CPU-пример
выше проверяет ту же потерю на случайной маленькой сети и не воспроизводит
его бюджет или результаты.

В формулировке задания точность описана через сравнение сырых вероятностей
выбранного и отвергнутого ответов. Для точности именно **неявной модели награды**
нужно сравнивать $\Delta_w$ и $\Delta_l$, то есть учитывать и опорные
вероятности. Это разные метрики: при совпадающих моделях неявные награды
равны нулю, хотя сырые вероятности двух разных текстов обычно не равны.

| | PPO RLHF | DPO |
|---|---|---|
| Данные | ответы с наградой | пары ответов |
| Генерация | да | нет |
| Отдельный оценщик | да | нет |
| Опорная модель | да | да |
| Критик | обычно да | нет |
| Новые ответы | исследует | не видит |

Функция потерь DPO выводится из той же задачи максимизации награды с ограничением
по KL-дивергенции. Однако практические алгоритмы не становятся полностью
эквивалентными. Заранее собранные данные не меняются вслед за моделью,
предпосылки модели Брэдли — Терри могут нарушаться, а PPO способен исследовать
ответы, которых не было в исходных парах.

<a id="dpo-controlled-comparison"></a>

<span id="что-показало-контролируемое-сравнение"></span>

### Эксперимент Ivison et al.

В [эксперименте Ivison et al., §3.2](https://arxiv.org/html/2406.09279v2#S3.SS2)
обе стратегии начинали с Tülu 2 13B после SFT. Для каждого набора пар
авторы обучали DPO непосредственно на этих парах, а для PPO сначала
обучали на них модель награды. Запросы из того же набора подавали
обучаемой PPO-модели, которая генерировала новые ответы. Размеченных
данных у обоих методов поровну; вычисления и сами ответы, по которым
обновлялись веса, различаются.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo-ivison-2026-11-14/paired-results.png|Таблица Ivison et al.: пары строк DPO и PPO для пяти наборов предпочтений; средняя разность плюс 0,7, а для truthfulness минус 2,5 пункта]]

*Ivison et al., [Table 2, Unpacking DPO and PPO](https://arxiv.org/html/2406.09279v2#S3.T2),
[PDF, с. 5](https://arxiv.org/pdf/2406.09279v2#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Таблица вырезана целиком, без изменений в ячейках. Цвет отмечает
изменение относительно SFT; для сравнения PPO с DPO нужны две строки
одного и того же набора.*

На UltraFeedback (FG) средняя оценка Tülu 2 выросла с 61,0 у DPO до
62,2 у PPO. Строки Nectar показывают, что скрывается за таким средним:
переход на PPO улучшил результат по программированию с 39,0 до 45,6,
но снизил Truthfulness с 68,1 до 60,1. В среднем по пяти наборам
выигрыш PPO составил 0,7 пункта, а Truthfulness, здесь измеряемая
с помощью TruthfulQA, снизилась на 2,5 пункта. Factuality в соседней
колонке измеряли на MMLU: это другой тест.

При смене данных эффект был больше. Если оставить DPO и заменить
ChatArena на UltraFeedback (FG), средняя оценка меняется с 54,7 до
61,0. Сравнение одного алгоритма на разных наборах не показывает,
какие свойства данных дали выигрыш. В этой таблице у ChatArena около
20,5 тыс. примеров, у четырёх других наборов — примерно по 61 тыс.;
влияние способа разметки рассмотрено отдельно в
[§3.1](https://arxiv.org/html/2406.09279v2#S3.SS1) и
[приложении E](https://arxiv.org/html/2406.09279v2#A5).

<a id="dpo-ppo-rollout-prompts"></a>

<span id="что-меняется-если-заменить-запросы-для-ppo"></span>

#### Запросы для генерации в PPO

В PPO для очередного обновления достаточно нового запроса: стратегия
сама пишет ответ, а готовая модель награды его оценивает.
Ivison et al. воспользовались этим, чтобы заменить запросы для генерации
на задачи по математике, оставив веса оценщика прежними. В отдельных
запусках брали около 20 тыс. запросов из разных источников: UltraFeedback, сборные
математические задачи или обучающая часть GSM8K. Проверяли на тестовой
части GSM8K.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo-ivison-2026-11-14/math-prompts.svg|Figure 3 Ivison et al.: при переходе на обучающие задачи GSM8K оценщики 70B дают более высокий результат, а 13B UltraFeedback не выигрывает]]

*Ivison et al., [Figure 3, §3.4.1](https://arxiv.org/html/2406.09279v2#S3.F3),
[оригинальный SVG](https://arxiv.org/html/2406.09279v2/math_prompts.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), без изменений.
По горизонтали три источника запросов, не шаги обучения. Каждая точка
получена в отдельном запуске; линия соединяет запуски с одним оценщиком.
Пунктир — исходная SFT-модель (46%).*

У 13B UltraFeedback переход к запросам GSM8K не даёт выигрыша.
С оценщиком 70B Mix правая точка достигает примерно 62%. Разница
в 16 процентных пунктов с пунктиром сравнивает полученную PPO-модель
с исходной SFT. Эффект смены запросов виден по точкам одной линии:
они получены с тем же оценщиком.

Улучшение на математике не обязательно повышает средний результат.
В отдельном опыте на ~61 тыс. запросов смешивание математики и кода
с UltraFeedback при оценщике 13B UltraFeedback увеличило точность GSM
с 53,0 до 54,5, но снизило средний результат с 62,2 до 61,9
([Table 4](https://arxiv.org/html/2406.09279v2#S3.T4)).
Сам оценщик тоже надо проверять в роли, для которой он выбран:
у 13B Mix результат на RewardBench был 79,8 против 61,0 у 13B
UltraFeedback, а после PPO средний результат генератора оказался
ниже (61,6 против 62,2; [Table 3](https://arxiv.org/html/2406.09279v2#S3.T3)).
Эти сравнения также разобраны на
[слайдах 79–85 Berkeley](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf#page=79).

<a id="dpo-irpo"></a>

<span id="11-iterative-reasoning-preference-optimization"></span>

## IRPO: пары из новых решений

<!-- source_unit_id: meeting-02-slides-irpo -->
<!-- source_unit_id: meeting-02-reading-02-catalogue-record -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-posttraining-memory/m02-p74-irpo-loop.png|Итерации IRPO с генерацией, проверкой и DPO плюс NLL]]

*На каждой итерации модель генерирует новые рассуждения. Проверка итогового
ответа разделяет их на выбранные и отвергнутые, после чего модель обучается
по совместной функции потерь DPO+NLL. Поэтому IRPO отличается от однократного
DPO на неизменных парах. Источник: Jason
Weston, Berkeley Advanced LLM Agents, meeting 2,
[слайд 74](https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf#page=74).*

В [Iterative Reasoning Preference Optimization (IRPO), §2](https://arxiv.org/html/2404.19733v3#S2)
пары фиксированы только внутри раунда обучения. После раунда обновлённая
модель снова решает те же задачи. Из новых решений составляют следующий
набор пар.

Пусть обучающий набор содержит фиксированные задачи $(x_i,a_i^*)$ с точными
ответами, а текущая модель — $M_t$. Один раунд состоит из пяти операций.

1. Для каждого $x_i$ семплируют $N$ последовательностей
   $(c_i^n,a_i^n)\sim M_t(\cdot\mid x_i)$: рассуждение и явно отделённый
   итоговый ответ.
2. Парсер извлекает $a_i^n$, проверяющая программа сравнивает его с $a_i^*$ и присваивает
   бинарную метку $r_i^n$. Промежуточные шаги при этом не размечаются.
3. Из ответов на один запрос образуют пары: выбранное рассуждение заканчивается
   правильным результатом, отвергнутое — неправильным. Если все $N$ попыток имеют
   одну метку, этот запрос сам по себе не даёт сравнительной пары. В экспериментах
   IRPO на GSM8K авторы дополнительно включали эталонное решение в множество
   выбранных: это позволяло учиться и на задачах, где все генерации ошибочны.
   В ARC-Challenge эталонных рассуждений не было, поэтому такие запросы исключали.
4. $M_t$ служит опорной моделью, а новая модель $M_\theta$ инициализируется её
   весами и обучается по сумме DPO и NLL на выбранном рассуждении:

$$
\begin{aligned}
\mathcal L_{\mathrm{IRPO}}
&=\mathcal L_{\mathrm{DPO}}(c_w,a_w,c_l,a_l\mid x)\\
&\quad-\alpha\frac{\log M_\theta(c_w,a_w\mid x)}{|c_w|+|a_w|}.
\end{aligned}
$$

5. Полученная $M_{t+1}$ снова генерирует рассуждения для тех же обучающих
   запросов. Старые пары уже могут не отражать характерные ошибки новой модели.

Вероятность правильной цепочки может упасть, если вероятность
неправильной падает ещё быстрее. Добавка NLL штрафует падение
вероятности выбранной цепочки. В опыте Pang et al. на GSM8K модель
после SFT на выбранных ответах близко оценивала выбранные и отвергнутые
цепочки; сочетание NLL с DPO разделяло их и давало большую точность.
Это результат сравнения на GSM8K.

<span id="один-раунд-на-конкретной-задаче"></span>

### Пары для одного уравнения

Для уравнения `3x+6=18` четыре генерации дали итоги `4, 4, 8, parse_error`.
Первые два рассуждения попадают в $G_i^w$, последние два — в $G_i^l$.
Например, пара

```text
chosen:  3x=12, x=4. FINAL: 4
rejected: 3x=24, x=8. FINAL: 8
```

влияет не только на финальные токены `4` и `8`: логарифм вероятности
последовательности включает все токены ответа, поэтому обновление затрагивает
и предшествующие шаги. Верификатор, однако, подтвердил лишь итог. Правильный
результат с ошибочной цепочкой всё ещё может попасть в выбранные примеры,
если не проверять рассуждения отдельно.

В следующем фрагменте заданные четыре текста только размечаются и превращаются
в пары; модель здесь ничего не генерирует. Парсер принимает целое число после
`FINAL:`. Это ограниченный формат примера, не универсальная проверка математики.
Пары строятся одновременным обходом двух списков; короткий список начинается
заново, как описано в §2 статьи. Получаются две пары, а не все четыре
элемента декартова произведения.

<details id="dpo-code-irpo">
<summary>Код: составление пар IRPO</summary>

```python
import re

def reasoning_pairs(candidates, expected_answer, count):
    if count < 0:
        raise ValueError("count must be nonnegative")
    chosen, rejected = [], []
    for text in candidates:
        match = re.search(r"FINAL:\s*(-?\d+)\s*$", text)
        correct = match is not None and int(match.group(1)) == expected_answer
        (chosen if correct else rejected).append(text)
    if not chosen or not rejected:
        return []
    return [(chosen[i % len(chosen)], rejected[i % len(rejected)])
            for i in range(count)]

reasoning_candidates = [
    "3x=12, x=4. FINAL: 4", "x=(18-6)/3=4. FINAL: 4",
    "3x=24, x=8. FINAL: 8", "Не удалось вычислить ответ.",
]
irpo_pairs = reasoning_pairs(reasoning_candidates, expected_answer=4, count=2)
assert len(irpo_pairs) == 2
assert reasoning_pairs(reasoning_candidates[2:], 4, 2) == []

# Отдельный числовой расчёт DPO+NLL для одной пары, не обучение IRPO.
irpo_pi_w, irpo_pi_l = torch.tensor([-3.0]), torch.tensor([-4.0])
irpo_ref_w, irpo_ref_l = torch.tensor([-3.4]), torch.tensor([-3.6])
irpo_nll = -irpo_pi_w.mean() / 3  # три целевых токена выбранного ответа
irpo_pair_loss = dpo_loss(irpo_pi_w, irpo_pi_l, irpo_ref_w, irpo_ref_l) + 0.2 * irpo_nll
print(f"DPO+NLL={irpo_pair_loss.item():.6f}")  # 0.713015
```

</details>

Пустой результат при всех ошибочных генерациях означает отсутствие сигнала
для попарной потери. Добавление эталонной цепочки в выбранный список —
отдельное решение о данных, а не свойство DPO. Сам цикл повторной генерации
из $M_{t+1}$ этим фрагментом не заменяется.

Полный цикл `generate → verify → pair → DPO+NLL → regenerate` показан на
[слайдах 74–78](https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf#page=74).
В версии v3 первичной статьи (26 июня 2024 года) результаты ограничены Llama-2-70B-Chat и
тремя наборами GSM8K, MATH и ARC-Challenge; их нельзя переносить на другой
верификатор или базовую модель без нового опыта.

<a id="dpo-tulu-length-normalization"></a>

<a id="12-tülu-3-данные-length-normalization-и-место-dpo-в-recipe"></a>
<span id="12-tülu-3-данные-нормировка-по-длине-и-место-dpo-в-обучении"></span>

## Данные и нормировка по длине в Tülu 3

<!-- source_unit_id: meeting-04-slides-tulu3-preference-recipe -->

В Tülu 3 состав данных предпочтений выбирали в отдельных экспериментах.
Три стадии построения пары показаны на исходном рисунке отчёта:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo-2026-10-24/tulu-preference-pipeline.png|Три стадии построения пар предпочтений Tülu 3]]

*Выбор запросов, получение альтернативных ответов и разметка предпочтений.
Включение Tülu SFT в пул генераторов позволяет получить сравнения с её
собственными ответами. Nathan Lambert et al.,
[Tülu 3 v5, Figure 7, с. 22](https://arxiv.org/pdf/2411.15124v5#page=22).
Полная фигура и английская подпись из PDF; CC BY 4.0. Выполнено только
кадрирование вокруг фигуры, без перерисовки и перевода.*

Для одного запроса авторы случайно выбирали четыре модели из набора генераторов
и получали по одному ответу от каждой. Отдельную часть сравнений строили так,
чтобы один ответ принадлежал текущей Tülu-3-SFT, а другой — внешней модели.
GPT-4o-2024-08-06 оценивал полезность, следование инструкции,
честность и достоверность каждого ответа. Вариант с наибольшей средней оценкой
становился выбранным; отвергнутый случайно выбирали среди вариантов с меньшей
оценкой.

В итоговой смеси соединились:

- ответы текущей SFT-модели (*on-policy*), показывающие именно её ошибки;
- ответы других моделей (*off-policy*), расширяющие разнообразие альтернатив;
- запросы, уже использованные в SFT, и новые запросы из тех же источников;
- общие данные предпочтений и отдельные примеры точного следования инструкциям.

Здесь *on-policy* характеризует происхождение ответов при сборе набора: они
получены от исходной Tülu-3-SFT. Это не означает генерацию на каждом шаге DPO.
После обновления весов модель уже отличается от той, которая собрала пары,
но сами пары в этом этапе остаются прежними.

Добавление ответов текущей модели улучшало агрегированный результат по сравнению
с данными только от других моделей. Новые запросы слегка превосходили простое
повторное использование запросов SFT; несколько крупных моделей-оценщиков
давали близкие результаты последующего обучения. Из трёх наборов предпочтений,
созданных по описаниям персонажей, в финальную смесь вошёл только набор
для следования инструкциям: варианты для математики и кода не улучшили
соответствующие показатели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo-2026-10-24/tulu-prompt-policy-ablations.png|Абляции запросов и генераторов ответов в Tülu 3]]

*Слева сравниваются повторно использованные и новые запросы; справа —
происхождение ответов. Ось ординат начинается с 55, а не с нуля.
Комбинированная смесь содержит 120 тысяч пар, две соседние — по 100 тысяч:
разность столбцов нельзя целиком приписать одному только происхождению ответов.
Это показатели конкретного 8B-эксперимента, не прирост в процентах.
Nathan Lambert et al., [Tülu 3 v5, Figures 10–11, с. 25](https://arxiv.org/pdf/2411.15124v5#page=25).
Обе полные панели с английскими подписями; CC BY 4.0. Только crop из PDF.*

Для оптимизации выбрали DPO с нормировкой по длине ответа:

$$
\begin{aligned}
\mathcal L_{\mathrm{LN\text{-}DPO}}&=-\mathbb E\log\sigma(z_{\mathrm{LN}}),\\
z_{\mathrm{LN}}&=\beta\left[
\frac{\Delta_\theta(x,y_w)}{|y_w|}
-\frac{\Delta_\theta(x,y_l)}{|y_l|}\right].
\end{aligned}
$$

После деления каждый ответ оценивают по среднему изменению
логарифма вероятности на токен. Разметка при этом остаётся прежней:
если оценщик награждал ответ за многословие, среднее этого не исправит.
В Tülu 3 такой вариант выбрали после сравнения нескольких функций
потерь на UltraFeedback и применили перед RLVR.

Например, у выбранного ответа из 100 токенов отношение к опорной модели дало
$\Delta_w=2$, а у отвергнутого из 10 токенов — $\Delta_l=1$. Обычный DPO
видит положительный разрыв $2-1=1$. После нормировки он отрицателен:
$2/100-1/10=-0.08$. Классификатор теперь считает пару упорядоченной неправильно.
Это существенная замена функции потерь, а не численная оптимизация, сохраняющая
исходную задачу. Средняя токеновая логарифмическая вероятность также не задаёт
нормированное распределение по полным текстам без дополнительной нормировки.

Сравнение вариантов обучения находится на
[слайдах 86–89](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf#page=86),
а состав наборов, запросы и абляции — в разделах 5.2–5.4 отчёта
[Tülu 3](https://arxiv.org/abs/2411.15124).

<a id="dpo-preference-data"></a>

<span id="13-какие-данные-нужны"></span>

## Состав набора предпочтений

Вернёмся к примеру с UDP. Пара с одним правильным и одним заведомо неверным
ответом полезна, пока модель путает их. Если она уже уверенно разделяет такие
примеры, повторение той же ошибки мало помогает научиться выбирать между двумя
технически верными ответами: кратким, но неполным и подробным, но перегруженным.
Последняя пара требует другого критерия — соответствует ли полнота объяснения
запросу пользователя. Состав пар определяет, чему именно учится модель.

Если выбранные ответы всегда длиннее, начинаются с одной вежливой фразы или
приходят от одного генератора, эти признаки могут оказаться проще фактической
проверки. Поэтому в отложенной выборке нужны случаи, где короткий ответ лучше
длинного, привычная формулировка содержит ошибку, а выбранный и отвергнутый
ответы принадлежат одному генератору. Такая проверка отделяет освоение критерия
от запоминания происхождения текста.

Ещё одна проблема — неоднозначность разметки. Если независимые разметчики
выбирают первый ответ в 6 случаях из 10, это иная информация, чем десять
единогласных сравнений. Превращение обоих случаев в одну уверенную метку
стирает различие. Повторные оценки позволяют оценить согласие; неоднозначные
пары можно размечать заново по более точному критерию или учитывать шум
в функции потерь. Универсального порога исключения здесь нет: он зависит
от задачи и способа получения оценок.

Чтобы проверить, не выучила ли модель признаки одного генератора,
сохраняйте с каждой парой версию источника ответов и настройки
декодирования. Критерии, автора оценки и степень согласия сохраняют
отдельно. В главе о
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data|данных предпочтений]]
есть пример такой записи.

<span id="14-характерные-ошибки-и-ограничения"></span>

## Ошибки и ограничения

<a id="dpo-failures"></a>

<a id="вероятность-отвергнутого-ответа-падает-быстрее-чем-растёт-вероятность-выбранного"></a>
### Обе вероятности могут уменьшаться

Точность на парах может расти только за счёт резкого снижения вероятности
отвергнутых ответов, хотя вероятность выбранных ответов тоже падает. Поэтому
оценки выбранных и отвергнутых ответов, разность между ними и исходные логарифмы
вероятностей следует наблюдать раздельно. Например, переход
$\Delta_w:0\to-1$ и $\Delta_l:0\to-3$ увеличивает разрыв с нуля до двух,
хотя оба ответа стали менее вероятными. Освободившаяся вероятность может
перейти к третьим ответам, которых вообще не было в сравнении. Только генерация
новых текстов покажет, улучшились ли эти ответы.

<span id="смещение-в-пользу-длины"></span>
<span id="шумные-метки-и-нетранзитивность"></span>
<span id="переобучение-на-неизменном-наборе-пар"></span>
<span id="оценивание-повторяет-смещения-обучающего-оценщика"></span>

### Ошибки разметки и проверки

Если выбранный ответ почти всегда длиннее, модель может выучить
длину вместо критерия, по которому ответы должны различаться. В
отложенных парах сравнивайте ответы сходной длины и отдельно проверяйте
примеры, где короткий ответ верен, а длинный содержит ошибку. Для
проверки новых генераций используйте запросы вне обучения и оценку,
независимую от обучающей: например, проверку результата или человеческие
сравнения. Судья, который ставил обучающие метки, может повторить те же
предпочтения при проверке.

Если разметчики меняют критерий от пары к паре, возможен цикл:
A яснее B, B точнее C, C безопаснее A. Одна скалярная награда
не может воспроизвести три таких предпочтения одновременно. Повторная
разметка с заранее указанным критерием отделяет неоднозначность
от ошибки модели.

<span id="насыщение-сигмоиды"></span>
<span id="несовпадение-шаблонов-диалога"></span>

### Ошибки расчёта и насыщение

Проверьте токенизацию, шаблон, обрезку и EOS у обеих моделей. Если
опорная модель видит другой префикс, вычитание уже сравнивает разные
условия. На правильно подготовленной паре очень положительный логит
тоже может дать почти нулевой градиент: вес сигмоиды мал, потому что
модель уже разделила эти два ответа. Это не проверка их качества на
других запросах.

<a id="dpo-diagnostics"></a>

<span id="15-диагностика-обучения"></span>

## Диагностика обучения

При проверке DPO нужно различать три результата: функция потерь вычисляется
правильно; модель научилась разделять пары; новые ответы действительно стали
лучше. Первый проверяется численными примерами, второй — отложенными парами,
третий — отдельным набором запросов с генерацией.

| Наблюдение | Что можно заключить |
|---|---|
| DPO loss уменьшается | Средняя логистическая ошибка на обучающих парах стала меньше. Разрыв мог вырасти не у всех пар. |
| Точность на отложенных парах растёт | Предпочтение переносится на готовые ответы. Их самостоятельная генерация не проверена. |
| Обе неявные награды падают | Оба ответа теряют вероятность относительно опорной модели. Качество остальных ответов пока неизвестно. |
| Доля побед на новых генерациях растёт | Ответы лучше по критерию оценщика. Его реакцию на длину и стиль проверяют отдельно. |

KL оценивают на ответах, сгенерированных именно текущей моделью. Среднее
$\log\pi_\theta(y\mid x)-\log\pi_{ref}(y\mid x)$ на таких выборках оценивает
$D_{KL}(\pi_\theta\Vert\pi_{ref})$. Та же средняя разность на фиксированных
выбранных ответах имеет другое распределение усреднения и не является этой
KL-дивергенцией. Поэтому метрику `rewards/chosen` нельзя переименовать в KL,
просто разделив её на $\beta$.

<a id="dpo-normalized-mass-kl"></a>
<span id="три-ответа-куда-уходит-вероятность-и-как-измерить-kl"></span>

### Перераспределение вероятности и KL

Зададим условную модель с тремя возможными ответами. В наборе предпочтений
есть только сравнение $y_w\succ y_l$; третий ответ в нём отсутствует.
Опорное распределение и два состояния обучаемой модели:

| Ответ | Опорная и исходная модель | Изменённая модель |
|---|---:|---:|
| $y_w$ | 0.4 | 0.2 |
| $y_l$ | 0.4 | 0.1 |
| $y_3$ | 0.2 | 0.7 |

Оба столбца нормированы. Вероятность выбранного ответа уменьшилась вдвое,
отвергнутого — вчетверо. Отношение их вероятностей выросло с 1 до 2.
При $\beta=0.5$ логит DPO изменился с нуля до $0.5\log2$,
а потеря уменьшилась с $\log2\approx0.693147$ до $0.534800$.
Недостающая масса перешла к $y_3$, о качестве которого сравнение ничего
не сообщает. Здесь заданы два распределения для проверки функции потерь;
это не утверждение, что конкретный шаг оптимизатора перевёл одно в другое.

Для изменённой модели точный KL равен сумме по всем трём ответам:
$0.2\log(0.2/0.4)+0.1\log(0.1/0.4)+0.7\log(0.7/0.2)
\approx0.599675$. Если вместо этого усреднить логарифмические отношения
только по выбранным ответам из набора, получится $\log(0.2/0.4)\approx-0.693147$.
Среднее по двум членам пары тоже отрицательно, примерно $-1.039721$.
Ни одно из этих средних не использует распределение изменённой модели.
В точном KL вклад третьего ответа имеет вес 0.7; при генерации он также
появлялся бы примерно в 70% случаев.

<details id="dpo-code-probability-mass">
<summary>Код: потеря и KL для трёх ответов</summary>

```python
mass_reference = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float64)
mass_before = mass_reference.clone()
mass_after = torch.tensor([0.2, 0.1, 0.7], dtype=torch.float64)

def mass_pair_loss(probability):
    logp, ref_logp = probability.log(), mass_reference.log()
    return dpo_loss(logp[:1], logp[1:2],
                    ref_logp[:1], ref_logp[1:2], beta=0.5)

mass_loss_before = mass_pair_loss(mass_before)
mass_loss_after = mass_pair_loss(mass_after)
mass_logratio = (mass_after / mass_reference).log()
mass_exact_kl = (mass_after * mass_logratio).sum()
mass_chosen_mean = mass_logratio[0]
mass_pair_mean = mass_logratio[:2].mean()

torch.testing.assert_close(mass_after.sum(), torch.tensor(1., dtype=torch.float64))
assert (mass_after[:2] < mass_before[:2]).all()
assert mass_after[2] > mass_before[2]
assert mass_loss_after < mass_loss_before
assert mass_exact_kl > 0 > mass_chosen_mean
print(f"DPO: {mass_loss_before:.6f} -> {mass_loss_after:.6f}")
print(f"KL={mass_exact_kl:.6f}, chosen={mass_chosen_mean:.6f}, "
      f"pair={mass_pair_mean:.6f}")
```

</details>

Различие следует из распределений усреднения в исходной постановке:
в [Eq. 3 статьи DPO](https://arxiv.org/html/2305.18290v3#S3)
ответы для KL взяты из текущей стратегии, а в Eq. 7 попарная потеря усредняется
по готовому набору сравнений. В маленьком примере доступна полная сумма;
у языковой модели KL обычно оценивают по новым генерациям. Одиночная
выборочная оценка может быть отрицательной из-за случайности, хотя истинный
KL неотрицателен. Систематическое усреднение по другому распределению —
отдельная ошибка, которую увеличение числа таких пар не исправляет.

Если точность высока только на обучающих парах, сравните её с
точностью на отложенных парах и с оценкой новых генераций. На
фиксированном наборе модель может запомнить порядок ответов, не начав
чаще порождать выбранные тексты.

<span id="16-варианты-dpo"></span>

## IPO, SimPO, ORPO, KTO и Online DPO

В IPO меняется сама потеря. В SimPO и ORPO исчезает опорная модель,
но для сравнения нужны два ответа. KTO обходится одиночными оценками.
По тому, какие метки удалось собрать и можно ли хранить опорную
модель, видно, какие из этих методов вообще применимы.

<a id="dpo-ipo"></a>

### Целевой разрыв в IPO

Для безошибочно разделимых пар логистические потери DPO продолжают
уменьшаться при $m\to\infty$. У одной уверенной метки
нет информации о том, насколько большим должен стать разрыв.
[IPO](https://arxiv.org/html/2310.12036v2#S5) заменяет логистическую потерю
квадратичной: $(m-1/(2\tau))^2$, где $\tau$ — его коэффициент регуляризации.
Если разрыв уже превысил целевое значение, градиент направлен обратно к нему,
а не продолжает разделять ответы. Это иной способ учитывать предпочтения;
он не гарантирует лучшего качества на любом наборе. В языковых реализациях
дополнительно проверяют, как определён $m$: сумма и нормировка по длине
не взаимозаменяемы.

Например, при $\tau=0.5$ целевой разрыв IPO равен 1. Для $m=0,1,2$
квадратичные потери равны 1, 0, 1, а производные — −2, 0, 2.
При $m=2$ IPO уже уменьшает разрыв. Логистическая потеря DPO для той же
уверенной метки всё ещё поощряла бы его рост, хотя и со всё меньшим весом.

<a id="dpo-simpo"></a>

### SimPO без опорной модели

SimPO заменяет изменение относительно $\pi_{ref}$ на оценку
$s_\theta(x,y)=\beta\log\pi_\theta(y\mid x)/|y|$ и потерю
$-\log\sigma(s_w-s_l-\gamma)$, где $\gamma>0$ — требуемый отступ.
Даже при $s_w=s_l$ выбранная пара получает отрицательный логит $-\gamma$:
для малого значения потерь необходимо не только правильно упорядочить ответы,
но и увеличить разницу их оценок. [SimPO](https://arxiv.org/abs/2405.14734)
не требует опорных проходов; это преимущество в вычислениях не является
эквивалентностью KL-регуляризованной задаче исходного DPO.

Если средние логарифмы вероятностей двух ответов равны −1 и −1.2,
$\beta=2$, а $\gamma=0.5$, то их оценки равны −2 и −2.4.
Порядок выбран верно, но логит SimPO равен $0.4-0.5=-0.1$:
предпочтение ещё не достигло заданного отступа, и потеря равна 0.744397.
Без отступа она составила бы 0.513015. Поэтому перенос прежнего $\beta$
из суммирующего DPO не сохраняет ни масштаб, ни смысл целевой функции
([SimPO v2, §2.2–2.3](https://arxiv.org/html/2405.14734v2#S2)).

<a id="dpo-orpo"></a>

### Отношение шансов в ORPO

[ORPO](https://arxiv.org/abs/2403.07691) добавляет к потере SFT
попарный штраф. Обе части вычисляют в одном обучении, без отдельной
стадии DPO.
Он сравнивает отношения шансов $p/(1-p)$, где $p$ строится из нормированной
по длине логарифмической вероятности ответа. Компонента SFT поддерживает
правдоподобие выбранного текста, а штраф учитывает отвергнутый. Опорной модели
нет. Здесь также важно не путать геометрическое среднее вероятностей токенов
с вероятностью полной последовательности.

Обозначим эту среднюю вероятность как $p_y$ и введём $o_y=p_y/(1-p_y)$.
Тогда попарная часть и вся потеря имеют вид

$$
\begin{aligned}
L_{OR}&=-\log\sigma(\log o_w-\log o_l),\\
L_{ORPO}&=-\log p_w+\lambda L_{OR}.
\end{aligned}
$$

Для $p_w=0.2$, $p_l=0.1$ получаем шансы $1/4$ и $1/9$.
Их отношение равно $9/4$; после сигмоиды логарифма отношения получается
$9/13$. Попарная потеря равна $-\log(9/13)=0.367725$, а NLL выбранного
ответа — $-\log0.2=1.609438$. При $\lambda=0.1$ общая потеря равна
1.646210. Две части дают разные сигналы: NLL зависит только от выбранного
ответа, вторая — от их сопоставления.
Точная постановка дана в [ORPO v2, §4.1–4.2](https://arxiv.org/html/2403.07691v2#S4).

<a id="dpo-kto"></a>

### Одиночные оценки в KTO

Кнопка «ответ полезен» даёт метку для одного ответа, без альтернативного
текста. [KTO](https://arxiv.org/abs/2402.01306)
может обучаться на таких положительных и отрицательных примерах. Он оценивает
изменение вероятности относительно опорной модели и сравнивает его с базовым
уровнем, связанным с KL-расхождением; для желательных и нежелательных ответов
используются разные стороны функции ценности. Пары для одного запроса не нужны,
но остаются важными баланс меток и процедура оценки базового уровня.

В записи [KTO v5, §4.1](https://arxiv.org/html/2402.01306v5#S4.SS1)
$r=\log(\pi_\theta(y\mid x)/\pi_{ref}(y\mid x))$, а $z_0$ — опорный уровень,
связанный с KL. Для одного примера положим $s=\beta(r-z_0)$.
Обозначим желательную метку как D, нежелательную — как U. Тогда

$$
L_{KTO}=\begin{cases}
\lambda_D[1-\sigma(s)],&D,\\
\lambda_U[1-\sigma(-s)],&U.
\end{cases}
$$

При $r=0.4$, $z_0=0.1$, $\beta=1$ и единичных весах желательная метка
даёт потерю 0.425557, нежелательная — 0.574443. Речь идёт о двух возможных
метках одного примера, а не о паре альтернатив для одного запроса.
Через $z_0$ градиент не проводят. В практической оценке авторы циклически
переставляют ответы между запросами мини-пакета, считают среднее логарифмическое
отношение для несовпадающих пар и ограничивают его снизу нулём.
Это смещённая замена KL, а не точное вычисление расхождения по всем текстам.
Веса $\lambda_D$ и $\lambda_U$ позволяют отдельно учитывать частоту и цену
двух типов меток. В отличие от отрицательного log-sigmoid, потеря
$1-\sigma$ ограничена и насыщается с обеих сторон: у очень сильно неверного
примера градиент тоже может быть мал.

<a id="dpo-online"></a>

### Обновление пар в Online DPO

Функцию потерь можно оставить прежней, но менять пары во время обучения:
генерировать новые ответы текущей моделью и размечать их перед очередным
обновлением. IRPO выше — пример
такого итеративного процесса с дополнительной NLL-компонентой. Слово *online*
описывает сбор данных, а не превращает попарную потерю в градиент PPO.

<span id="практика-и-первоисточники"></span>
<a id="задания"></a>
<span id="численная-проверка-функции-потерь"></span>

Последний листинг сверяет `dpo_loss` с расчётом для двух ответов про UDP.

<details id="dpo-code-udp-gradients">
<summary>Код: потеря и градиенты для пары про UDP</summary>

```python
pi_w = torch.tensor([-3.0], requires_grad=True)
pi_l = torch.tensor([-4.0], requires_grad=True)
ref_w, ref_l = torch.tensor([-3.4]), torch.tensor([-3.6])
loss = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.5)
loss.backward()
print(round(loss.item(), 6))
print(round(pi_w.grad.item(), 6), round(pi_l.grad.item(), 6))
# 0.513015
# -0.200656 0.200656
```

</details>

При совпадающих опорных и текущих логарифмах функция возвращает $\log2$.
Если одинаково увеличить оба текущих логарифма, значение не меняется.
Для `sequence_logp` отдельная проверка должна оставлять сумму неизменной
при добавлении позиций с меткой `-100`, но менять её при исключении настоящего
EOS. Это проверки разных ошибок: формулы сравнения, сдвига токенов и маски.

<a id="проверка-понимания"></a>
Вопросы к этой главе вынесены в
[[02 Areas/ML & DL/04 Вопросы/Вопросы по LLM#Post-training и alignment|«Вопросы по LLM»]].

## Литература и реализации

<span id="первоисточник"></span>

- Rafailov et al., [Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model](https://arxiv.org/abs/2305.18290). Особенно Figure 1,
  Equations 1–7, gradient после Eq. 7, Figure 2 и Appendix A.

<span id="объяснения-и-код"></span>

- Stanford CS336, [лекция 15 — Midtraining and posttraining](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_15.pdf): связь награды и стратегии, вывод DPO, нормировка по длине.
- Nathan Lambert, [Direct-Alignment Algorithms](https://rlhfbook.com/c/08-direct-alignment): градиент, опорная модель и различия методов.
- Rafailov, Sharma, Mitchell et al., [официальная реализация DPO](https://github.com/eric-mitchell/direct-preference-optimization): `trainers.py` и правила построения масок в `preference_datasets.py`.
- Berkeley Advanced LLM Agents, [Jason Weston, meeting 2](https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf#page=74) и [Hanna Hajishirzi, meeting 4](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf#page=79): IRPO, контролируемые сравнения PPO/DPO и Tülu 3.
- Hugging Face, [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer).
- Hugging Face, [Alignment Handbook](https://github.com/huggingface/alignment-handbook) — воспроизводимые SFT/DPO recipes.

Когда ответ можно проверить программой, оценка доступна и для новых
генераций текущей модели. В следующей главе об
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|обучении с проверяемой наградой]]
разберём, из чего складывается эта проверка и что модель выучит,
если в ней есть пропуски.
