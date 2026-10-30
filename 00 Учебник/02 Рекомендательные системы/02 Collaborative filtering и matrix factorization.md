---
title: Совместная фильтрация и факторизация матрицы
type: textbook-chapter
status: canonical
last_updated: 2026-10-25
---

# Совместная фильтрация и факторизация матрицы

Аня оценила два фильма, Борис — ещё два, Вера — три. Некоторые фильмы
оценили несколько человек, другие — только один. Эти пересечения позволяют
связать их предпочтения: например, предложить Ане фильм, понравившийся
пользователям с похожей историей.

Совместная фильтрация, или *collaborative filtering*, использует такие
закономерности во взаимодействиях. В одних методах хранят сходства
между пользователями или объектами, в других обучают компактные векторы.
Описание фильма при этом не обязательно. Оно понадобится, если нужно
рекомендовать новый фильм, с которым ещё никто не взаимодействовал.

## Матрица взаимодействий

Для явных оценок удобно хранить матрицу $R\in\mathbb R^{m\times n}$
и отдельную маску $M$: строки соответствуют пользователям, столбцы — фильмам.
$M_{ui}=1$ означает, что оценка известна. Неизвестное значение не становится
нулевой оценкой.

| Кто | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Аня | 5 | ? | 1 | ? |
| Борис | 4 | 5 | ? | ? |
| Вера | ? | 1 | 5 | 4 |

Наблюдаемы семь из двенадцати пар. Средняя оценка Ани равна 3, Бориса —
4,5, Веры — $10/3$. Общая средняя равна $25/7$.

```python
import numpy as np

ratings = np.array([
    [5.0, np.nan, 1.0, np.nan],
    [4.0, 5.0, np.nan, np.nan],
    [np.nan, 1.0, 5.0, 4.0],
])
mask = np.isfinite(ratings)
target = np.where(mask, ratings, 0.0)
mu = ratings[mask].mean()
user_means = np.nanmean(ratings, axis=1)
assert mask.sum() == 7
np.testing.assert_allclose(mu, 25 / 7)
np.testing.assert_allclose(user_means, [3, 4.5, 10 / 3])
print(mu, user_means)
```

В `target` пропуски заполнены нулями только для удобства арифметики.
Маска определяет, какие ошибки входят в функцию потерь. Если убрать маску,
получится другая задача — приблизить полностью известную матрицу с нулями.

В [D2L, Matrix Factorization](https://d2l.ai/chapter_recommender-systems/mf.html)
множество известных пар обозначено $\mathcal K$. Для журнала кликов или
покупок сначала надо выбрать представление сигнала: число событий,
их взвешенную сумму или бинарное наличие взаимодействия. Эти величины
не находятся на одной шкале со звёздными оценками.

<a id="три-baseline-до-factorization"></a>
## Среднее и смещения

Прогноз общего среднего $\hat r_{ui}=\mu$ не различает ни пользователей,
ни фильмы. Среднее пользователя учитывает его привычную шкалу:
Борису модель всегда предскажет 4,5. Но два разных фильма для него
по-прежнему не различаются.

Именно среднее пользователя реализовано в
[Recommenders, Estimating Baseline Performance](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb).
Вторая часть этого notebook использует популярность для top-K.
Совместное обучение пользовательских и объектных смещений — отдельная
модель, а не действие приведённого там `groupby(...).mean()`.

Для такой модели запишем

$$
\hat r_{ui}=\mu+b_u+b_i.
$$

$b_u$ описывает склонность пользователя повышать или понижать оценки,
$b_i$ — отклонение для фильма. Два пользователя с похожими предпочтениями
могут пользоваться разной шкалой звёзд; смещение позволяет не относить
это различие целиком к вкусам.

Зафиксируем $\mu$ как среднее train и обучим остальные параметры:

$$
\begin{aligned}
L_b={}&\frac12\sum_{(u,i)\in\mathcal K}
(r_{ui}-\mu-b_u-b_i)^2\\
&+\frac{\lambda_b}{2}
\left(\sum_u b_u^2+\sum_i b_i^2\right).
\end{aligned}
$$

Если объектные смещения уже известны, производная по одному $b_u$ равна

$$
-\sum_{i\in\mathcal K_u}(r_{ui}-\mu-b_i-b_u)+\lambda_b b_u.
$$

Приравняв её нулю, получим

$$
b_u=
\frac{\sum_{i\in\mathcal K_u}(r_{ui}-\mu-b_i)}
{|\mathcal K_u|+\lambda_b}.
$$

Аналогично обновляется $b_i$ при фиксированных $b_u$.
Добавка $\lambda_b$ в знаменателе уменьшает модуль смещения по сравнению
с нерегуляризованным средним остатков. Одно наблюдение получает меньшую
свободу изменить параметр, чем множество согласованных наблюдений.

Например, при $\lambda_b=1$ и начальных $b_i=0$ у Ани сумма остатков
равна $-8/7$. Поэтому первое $b_u=-8/21$. У Бориса получится $13/21$,
у Веры — $-5/28$. Затем по уже обновлённым пользовательским смещениям
обновляются объектные. Эти два прохода повторяют.

```python
def fit_biases(target, mask, mu, penalty=1.0, sweeps=200):
    if penalty <= 0:
        raise ValueError("Use a positive bias penalty.")
    bu = np.zeros(target.shape[0])
    bi = np.zeros(target.shape[1])
    for _ in range(sweeps):
        for u in range(len(bu)):
            seen = mask[u]
            bu[u] = (target[u, seen] - mu - bi[seen]).sum() / (
                seen.sum() + penalty)
        for i in range(len(bi)):
            seen = mask[:, i]
            bi[i] = (target[seen, i] - mu - bu[seen]).sum() / (
                seen.sum() + penalty)
    return bu, bi

first_bu, first_bi = fit_biases(target, mask, mu, sweeps=1)
np.testing.assert_allclose(first_bu, [-8/21, 13/21, -5/28])
np.testing.assert_allclose(first_bi, [34/63, -133/252, -7/36, 17/56])
bu_bias, bi_bias = fit_biases(target, mask, mu)
bias_prediction = mu + bu_bias[:, None] + bi_bias[None, :]
print("First user update:", first_bu)
print("Bias-only train RMSE:",
      np.sqrt(np.mean((bias_prediction[mask] - target[mask]) ** 2)))
```

Здесь оптимизируется выпуклая квадратичная задача при фиксированном $\mu$.
Но зависимость предпочтения от конкретной пары пока отсутствует:
добавка фильма одинакова для всех пользователей. Чтобы Ане и Вере
нравились разные фильмы, нужны сходства или взаимодействие факторов.

<a id="memory-based-collaborative-filtering"></a>
## Рекомендации по соседним объектам

В item-based методе сначала находят похожие объекты, затем используют
их известные оценки или историю взаимодействий пользователя.
В user-based варианте сначала находят похожих пользователей.
Выбор сходства и правило объединения наблюдений задаются отдельно.

Для оценок одна из простых схем из [Hu, Koren, Volinsky, §3.1](https://yifanhu.net/PUB/cf.pdf)
выбирает $k$ оценённых пользователем соседей целевого фильма:

$$
\hat r_{ui}
=\frac{\sum_{j\in N_k(i;u)}s_{ij}r_{uj}}
{\sum_{j\in N_k(i;u)}s_{ij}}.
$$

Например, если сходства с двумя уже оценёнными фильмами равны $0{,}8$
и $0{,}2$, а оценки пользователя — 5 и 1, прогноз равен
$(0{,}8\cdot5+0{,}2\cdot1)/(0{,}8+0{,}2)=4{,}2$.
Это формула для положительных весов и ненулевого знаменателя.
Если сходство допускает отрицательные значения, как корреляция,
нужно отдельно определить отбор соседей и нормировку. Наивная сумма
может обнулиться; это не особенность вкусов пользователя.

Если прогнозируют остатки относительно среднего или bias baseline,
базовый прогноз целевого фильма прибавляют обратно. Для той же схемы
положительных весов и $b_{ui}=\mu+b_u+b_i$:

$$
\hat r_{ui}=b_{ui}+
\frac{\sum_{j\in N_k(i;u)}s_{ij}(r_{uj}-b_{uj})}
{\sum_{j\in N_k(i;u)}s_{ij}}.
$$

Пусть базовый прогноз целевого фильма равен 3,5, двух соседей — 4 и 2,
а наблюдаемые оценки соседей — 5 и 1. Остатки равны 1 и −1.
С прежними весами получится $3{,}5+0{,}8\cdot1-0{,}2\cdot1=4{,}1$.
Так сосед с оценкой 4 у человека, обычно ставящего 5, выражает другое
отклонение, чем 4 у человека со средним 2. Само сходство по двум общим
оценкам также менее надёжно, чем по двумстам; минимальное число пересечений
или уменьшение веса слабых пересечений — отдельные настройки метода.

### SAR: от совместных появлений к списку

[Recommenders, SAR deep dive](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/02_model_collaborative_filtering/sar_deep_dive.ipynb)
разделяет две матрицы: сходства объектов $S$ и веса истории пользователя $A$.
Итоговые оценки имеют вид $AS$.

![[00 Учебник/Assets/Figures/curated/recommender-mf-2026-10-25/sar-schema.svg]]

*Recommenders contributors, исходная схема SAR:
[notebook](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/02_model_collaborative_filtering/sar_deep_dive.ipynb),
[SVG на закреплённой ревизии](https://github.com/recommenders-team/resources/blob/e76bb39727103e45dd8011b6d0eca2f1aad3c0f7/images/sar_schema.svg).
[MIT](/licenses/recommenders-resources-e76bb397-MIT.txt).
Подписи и геометрия не изменены.*

Верхняя ветка схемы превращает взаимодействия в сходства объектов.
Нижняя — в пользовательские веса с необязательным учётом давности.
После умножения можно исключить уже виденные объекты и выбрать top-K.
Здесь score не обязан лежать на шкале от 1 до 5.

Пусть $H_{ui}$ равно 1 при наличии положительного взаимодействия
и 0 иначе. Для этого подсчёта повторные события пары сначала объединяют.
Тогда

$$
C=H^\top H,\qquad
C_{ij}=\sum_u H_{ui}H_{uj}.
$$

На диагонали стоит число пользователей, взаимодействовавших с объектом.
Вне диагонали — число пользователей, общих для двух объектов.
По этой матрице в notebook строятся, в частности, три сходства:

$$
\begin{aligned}
S_{ij}^{\mathrm{counts}}&=C_{ij},\\
S_{ij}^{\mathrm{Jaccard}}
&=\frac{C_{ij}}{C_{ii}+C_{jj}-C_{ij}},\\
S_{ij}^{\mathrm{lift\ (SAR)}}
&=\frac{C_{ij}}{C_{ii}C_{jj}}.
\end{aligned}
$$

Последняя формула соответствует функции `lift` именно этой реализации.
Вероятностный lift $P(i,j)/(P(i)P(j))$ при бинарном $H$ в $m$ раз больше.
Для фиксированного набора пользователей это общий множитель всей матрицы
$S$: порядок рекомендаций не меняется, абсолютная шкала меняется.

Counts сохраняет частоту совместных появлений. Jaccard делит пересечение
на объединение, а формула lift сильнее компенсирует большие диагональные
частоты. Если у двух редких объектов единственное общее взаимодействие,
это ещё не доказывает устойчивого сходства; нужен контроль поддержки.

<a id="worked-example-sar"></a>
### Численный пример SAR и давности

В отдельном журнале положительных действий четыре объекта расположены
в порядке `camera, tripod, lens, bag`. Пользователи взаимодействовали с наборами

- `camera, tripod`;
- `camera, lens`;
- `tripod, lens, bag`;
- `camera, lens`.

Для `camera` и `lens` диагональные частоты равны 3, общее число
пользователей — 2, поэтому Jaccard равен $2/(3+3-2)=1/2$.
Для `tripod` и `lens` получается $1/4$.
Если у нового запроса вес `camera` равен 5, а `tripod` — 1,
оценка `lens` равна $5/2+1/4=2{,}75$, `bag` — $0+1/2=0{,}5$.

Вес события с давностью $\Delta t$ можно умножить на $2^{-\Delta t/T}$.
Параметр $T>0$ — период полураспада в тех же единицах времени.
Если покупка камеры была 120 дней назад, а просмотр штатива — сейчас,
при $T=30$ веса станут $5/16$ и 1. Тогда `lens` получит $0{,}40625$,
а `bag` — $0{,}5$: порядок изменится.

```python
hits = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 1, 0],
], dtype=float)
cooccurrence = hits.T @ hits
frequency = np.diag(cooccurrence)
denominator = frequency[:, None] + frequency[None, :] - cooccurrence
similarity = np.divide(cooccurrence, denominator,
                       out=np.zeros_like(cooccurrence), where=denominator > 0)
sar_lift = cooccurrence / (frequency[:, None] * frequency[None, :])

affinity = np.array([5.0, 1.0, 0.0, 0.0])
raw_scores = affinity @ similarity
np.testing.assert_allclose(raw_scores[2:], [2.75, 0.5])

age_days = np.array([120.0, 0.0])
decayed = affinity.copy()
decayed[:2] *= np.exp2(-age_days / 30)
decayed_scores = decayed @ similarity
np.testing.assert_allclose(decayed_scores[2:], [0.40625, 0.5])
assert np.argmax(raw_scores[2:]) != np.argmax(decayed_scores[2:])
# With no new events and fixed S, shifting all event ages scales every score.
later_scores = (decayed / 2) @ similarity
np.testing.assert_allclose(later_scores, decayed_scores / 2)
print(cooccurrence)
print("Candidate scores before/after decay:", raw_scores[2:],
      decayed_scores[2:])
```

При неизменных $S$ и истории простое течение следующих 30 дней
уменьшит все веса ещё вдвое и сохранит порядок. Для изменения относительных
вкладов нужны новые события, другое правило давности или изменение $S$.

Двоичная матрица `hits` относится к положительным действиям, не к звёздам
из первой таблицы. Превращение любой оценки, даже одной звезды, в positive
было бы ещё одним допущением. Повторные события, отрицательные веса
и уже просмотренные объекты требуют заранее выбранного правила обработки.

Хранение полной $S$ требует $O(n^2)$ чисел. Для бинарного разреженного $H$
перебор пар внутри историй даёт работу порядка $\sum_u k_u^2$,
где $k_u$ — длина истории. Она может породить плотную матрицу сходств.
Ограничение числа соседей уменьшает хранение, но само по себе не возникает
из записи `A @ S`. В закреплённом SAR `score` результат для выбранных
пользователей переводится в плотный массив: память зависит также
от числа запросов в batch и размера каталога. Для обычного плотного
умножения $A\in\mathbb R^{m\times n}$ на $S\in\mathbb R^{n\times n}$
нужно $O(mn^2)$ операций; разреженность и число
хранимых соседей меняют фактическую стоимость.

## Matrix factorization

Вместо полной таблицы сходств сопоставим пользователю строку
$p_u\in\mathbb R^d$, фильму — строку $q_i\in\mathbb R^d$.
Их скалярное произведение описывает взаимодействие конкретной пары:

$$
\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i.
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-mf.svg]]

*Zhang et al., исходный рисунок
[D2L, Matrix Factorization](https://d2l.ai/chapter_recommender-systems/mf.html):
матрицы пользовательских и объектных факторов и их произведение.
[CC BY-SA 4.0](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/LICENSE-SUMMARY).
На рисунке $k$ — размерность факторов, в тексте это $d$.
Выделенные строка $u$ матрицы $P$ и столбец $i$ матрицы $Q^\top$
используются для предсказания чёрной ячейки $R_{ui}$.
Смещения на рисунке не показаны.*

Соберём строки в $P\in\mathbb R^{m\times d}$ и
$Q\in\mathbb R^{n\times d}$. Элемент $(u,i)$ произведения $PQ^\top$
равен сумме $d$ попарных произведений координат.
Например, $(0{,}2;0{,}4)$ и $(0{,}5;0{,}1)$ дают
$0{,}2\cdot0{,}5+0{,}4\cdot0{,}1=0{,}14$.

Ранг $PQ^\top$ не выше $d$; смещения добавляются отдельно.
Нужно хранить $(m+n)d$ факторных чисел вместо $mn$ предсказаний.
На крошечной матрице $3\times4$ два фактора на строку потребуют 14 чисел,
поэтому экономии ещё нет. Преимущество возникает при больших $m,n$
и малом относительно них $d$.

Координатам не обязательно соответствуют жанры. Если одновременно
повернуть все пользовательские и объектные векторы одной ортогональной
матрицей $O$, произведение сохранится:

$$
(PO)(QO)^\top=POO^\top Q^\top=PQ^\top.
$$

Сохранятся и квадраты норм. Поэтому сама обучающая задача не выбирает
единственную интерпретацию каждой оси как «комедии» или «драмы».

```python
P_demo = np.array([[0.2, 0.4], [0.8, -0.1]])
Q_demo = np.array([[0.5, 0.1], [-0.2, 0.7], [0.3, 0.2]])
theta = 0.7
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
scores_demo = P_demo @ Q_demo.T
np.testing.assert_allclose(scores_demo[0, 0], 0.14)
np.testing.assert_allclose((P_demo @ rotation) @ (Q_demo @ rotation).T,
                           scores_demo)
np.testing.assert_allclose(np.linalg.norm(P_demo @ rotation),
                           np.linalg.norm(P_demo))
print(scores_demo)
```

В реализации D2L строки выбираются через `Embedding`. Для batch из $B$
пар lookup даёт два массива $B\times d$, их поэлементное произведение
и сумма по последней оси дают $B$ предсказаний. Это не перемножение
всех $m$ пользователей со всем каталогом на каждом шаге.

### Функция потерь и регуляризация

Для явных оценок зададим цель однозначно:

$$
\begin{aligned}
L={}&\frac12\sum_{(u,i)\in\mathcal K}
(r_{ui}-\hat r_{ui})^2\\
&+\frac{\lambda}{2}
\left(\|P\|_F^2+\|Q\|_F^2\right)\\
&+\frac{\lambda}{2}
\left(\|b^{(u)}\|_2^2+\|b^{(i)}\|_2^2\right).
\end{aligned}
$$

Здесь каждая строка факторов штрафуется один раз во всей цели.
Альтернативная запись, где штраф выбранных $p_u,q_i,b_u,b_i$
добавляется к каждому событию, умножает эффективный штраф ID
на число его появлений. Это другая частотная регуляризация.
Переход от суммы ошибок к средней тоже требует согласовать $\lambda$.

Для равномерной выборки одной из $N=|\mathcal K|$ пар несмещённую
оценку градиента всей суммы даёт $N\nabla\ell_{ui}$.
К нему добавляется градиент выбранного глобального штрафа.
Обычное штрафование только двух выбранных строк на каждом шаге
не воспроизводит его автоматически: частые ID выбираются чаще.

### Один шаг SGD

Для одной пары положим $e=r_{ui}-\hat r_{ui}$.
Для функции $\tfrac12 e^2+
\tfrac{\lambda}{2}(\|p_u\|^2+\|q_i\|^2)$ при фиксированных смещениях

$$
\begin{aligned}
\nabla_{p_u}L&=-e q_i+\lambda p_u,\\
\nabla_{q_i}L&=-e p_u+\lambda q_i.
\end{aligned}
$$

Отсюда одновременное обновление

$$
\begin{aligned}
p_u'&=p_u+\eta(e q_i-\lambda p_u),\\
q_i'&=q_i+\eta(e p_u-\lambda q_i).
\end{aligned}
$$

Обе правые части используют старые векторы и одну старую ошибку.
При $p=(0{,}2;0{,}4)$, $q=(0{,}5;0{,}1)$, $r=1$,
$\eta=0{,}1$, $\lambda=0$ получаем $e=0{,}86$:

$$
\begin{aligned}
p'&=(0{,}243;0{,}4086),\\
q'&=(0{,}5172;0{,}1344).
\end{aligned}
$$

```python
p_old = np.array([0.2, 0.4])
q_old = np.array([0.5, 0.1])
error = 1.0 - p_old @ q_old
step = 0.1
p_new = p_old + step * error * q_old
q_new = q_old + step * error * p_old
np.testing.assert_allclose(p_new, [0.243, 0.4086])
np.testing.assert_allclose(q_new, [0.5172, 0.1344])
assert abs(1 - p_new @ q_new) < abs(error)
print(p_new, q_new, p_new @ q_new)
```

Новое произведение равно примерно $0{,}180595$, ошибка стала меньше.
Это проверка одного шага с заданным размером, не гарантия уменьшения loss
при любом $\eta$. Формулировка через скалярное произведение также не требует
отождествлять обучаемую релевантность с евклидовым расстоянием.

### Полное обучение маленькой модели

Ниже реализован градиент всей записанной цели: без случайных batch,
с глобальной регуляризацией и фиксированным $\mu$.
Остатки неизвестных пар обнулены маской. Все градиенты сначала вычисляются
по одному состоянию параметров и только затем применяются.

```python
def explicit_objective(P, Q, bu, bi, target, mask, mu, penalty):
    residual = np.where(mask, mu + bu[:, None] + bi[None, :]
                        + P @ Q.T - target, 0.0)
    norms = sum(np.sum(a * a) for a in [P, Q, bu, bi])
    return 0.5 * np.sum(residual ** 2) + 0.5 * penalty * norms

def train_explicit(target, mask, mu, d=2, penalty=0.1,
                   step=0.03, steps=1500, seed=7):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, (target.shape[0], d))
    Q = rng.normal(0, 0.1, (target.shape[1], d))
    bu, bi = np.zeros(len(P)), np.zeros(len(Q))
    losses = [explicit_objective(P, Q, bu, bi, target, mask, mu, penalty)]
    for _ in range(steps):
        residual = np.where(mask, mu + bu[:, None] + bi[None, :]
                            + P @ Q.T - target, 0.0)
        gP = residual @ Q + penalty * P
        gQ = residual.T @ P + penalty * Q
        gu = residual.sum(axis=1) + penalty * bu
        gi = residual.sum(axis=0) + penalty * bi
        P, Q = P - step * gP, Q - step * gQ
        bu, bi = bu - step * gu, bi - step * gi
        losses.append(explicit_objective(P, Q, bu, bi,
                                         target, mask, mu, penalty))
    return P, Q, bu, bi, np.array(losses)

P_fit, Q_fit, bu_fit, bi_fit, explicit_losses = train_explicit(target, mask, mu)
prediction = mu + bu_fit[:, None] + bi_fit[None, :] + P_fit @ Q_fit.T
assert np.all(np.diff(explicit_losses) <= 1e-10)
assert explicit_losses[-1] < explicit_losses[0]
train_rmse = np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2))
print("Objective:", explicit_losses[0], explicit_losses[-1])
print("Train RMSE:", train_rmse)
```

Малые случайные начальные факторы нужны, чтобы у обеих таблиц были
ненулевые градиенты взаимодействия. Если одновременно задать $P=0$
и $Q=0$, произведения в этих градиентах останутся нулевыми;
обучаться смогут только смещения.

Уменьшение ошибки на семи обучающих парах не устанавливает качество
рекомендаций. Значения в пропусках пока не имеют проверочной разметки.
Для выбора $d$, $\lambda$ и момента остановки нужен validation,
построенный до обучения по
[[00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline|правилам временного протокола]].
RMSE следует считать по всем проверочным парам с единым знаменателем,
а не усреднять уже накопленные или по-разному взвешенные RMSE batch.

<a id="implicit-feedback-ноль-состоит-из-разных-причин"></a>
## Неявные взаимодействия и неизвестные пары

При кликах или покупках возникает другая проблема.
Если обучать модель только на положительных парах с target 1,
то ранг-1 матрица, состоящая целиком из единиц, даст нулевую ошибку
без регуляризации. Она одинаково рекомендует всё.

![[00 Учебник/Assets/Figures/curated/recommender-mf-2026-10-25/weighted-objectives.svg]]

*Google, [Matrix factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix),
исходное сравнение observed-only, weighted MF и SVD.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Части страницы воспроизведены из работы, созданной и
[предоставленной Google](https://developers.google.com/terms/site-policies).
Надписи, матрицы и формулы не изменены.*

Слева в loss входят только отмеченные единицы; пустые клетки не учитываются.
В центре неизвестные клетки получают целевое значение 0 и отдельный вес
$w_0$. Справа все нули считаются такими же полноправными наблюдениями,
как единицы. Надпись SVD относится к низкоранговому приближению этой
полностью заполненной матрицы, а не к любому обучению факторов.

Роль веса видна без большой модели. Пусть все объектные векторы
искусственно зафиксированы равными 1, поэтому единственный параметр
пользователя $z$ задаёт общий score. Есть один positive и сто неизвестных
объектов:

$$
\begin{aligned}
L(z)&=(1-z)^2+100w_0 z^2,\\
z^*&=\frac{1}{1+100w_0}.
\end{aligned}
$$

При $w_0=1$ ответ равен $1/101$, при $w_0=0{,}01$ — $1/2$.
Это пример заданного ограниченного семейства предсказаний, не формула
решения произвольной матрицы MF.

```python
def constant_score_optimum(missing_count, missing_weight):
    if missing_count < 0 or missing_weight < 0:
        raise ValueError("Counts and weights must be nonnegative.")
    return 1.0 / (1 + missing_count * missing_weight)

observed_positive = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
all_ones = np.ones(observed_positive.shape)
assert np.sum((all_ones[observed_positive] - 1) ** 2) == 0
np.testing.assert_allclose(constant_score_optimum(100, 1), 1 / 101)
np.testing.assert_allclose(constant_score_optimum(100, 0.01), 0.5)
print(constant_score_optimum(100, 1),
      constant_score_optimum(100, 0.01))
```

Указание target 0 для неизвестной пары остаётся допущением обучения.
Оно не превращает отсутствие покупки в зарегистрированное отвержение.
Веса позволяют ослабить это допущение, но не устраняют смещение
прежнего алгоритма показа.

### Предпочтение и уверенность

[Hu, Koren, Volinsky, §4](https://yifanhu.net/PUB/cf.pdf) разделяют
сырую величину события $r_{ui}$, бинарную цель $a_{ui}$ и вес $c_{ui}$.
В статье бинарная цель названа $p_{ui}$; здесь буква $p$ уже занята
пользовательским вектором, поэтому используем $a$.

$$
\begin{aligned}
a_{ui}&=\mathbf1[r_{ui}>0],\\
c_{ui}&=1+\alpha r_{ui},\quad\alpha\ge0.
\end{aligned}
$$

Для истории $(3,0,1)$ и $\alpha=2$ получаются цели $(1,0,1)$
и веса $(7,1,3)$. Три покупки и одна покупка дают одинаковую цель 1,
но разную уверенность в ней. Ненаблюдаемая пара тоже входит в loss,
с весом 1. Вместо линейной шкалы авторы рассматривают и
$1+\alpha\log(1+r_{ui}/\varepsilon)$ при $\varepsilon>0$.
Число $\alpha=40$ из их эксперимента не является универсальной настройкой.

Полная цель weighted MF без смещений:

$$
\begin{aligned}
L_{\mathrm{imp}}={}&
\sum_{u,i}c_{ui}(a_{ui}-p_u^\top q_i)^2\\
&+\lambda\left(\|P\|_F^2+\|Q\|_F^2\right).
\end{aligned}
$$

Сумма включает весь выбранный каталог. Набор доступных объектов
и момент построения истории задаются до её оптимизации.
Эти веса не являются обратными вероятностями показа и сами по себе
не решают причинную задачу.

### Weighted ALS: обновление одной строки

Зафиксируем объектные факторы $Q$ и рассмотрим одного пользователя.
Пусть $C_u=\operatorname{diag}(c_{u1},\ldots,c_{un})$,
а $a_u$ — столбец из $n$ бинарных целей.
Часть цели, зависящая от $p_u$, имеет вид

$$
(a_u-Qp_u)^\top C_u(a_u-Qp_u)+\lambda p_u^\top p_u.
$$

После дифференцирования и деления на 2:

$$
Q^\top C_u Qp_u-Q^\top C_u a_u+\lambda p_u=0.
$$

Следовательно, надо решить линейную систему размерности $d$:

$$
(Q^\top C_u Q+\lambda I)p_u=Q^\top C_u a_u.
$$

При $c_{ui}\ge1$ и $\lambda>0$ матрица слева положительно определена:
для ненулевого $v$ квадратичная форма равна
$\sum_i c_{ui}(q_i^\top v)^2+\lambda\|v\|^2>0$.
Строковое обновление поэтому имеет единственное решение.
В программе используют `solve`, а не вычисляют обратную матрицу явно.

Для $q_1=(1;0)$, $q_2=(0;1)$, $q_3=(1;1)$ и истории $(3,0,1)$
получаем при $\alpha=2,\lambda=1$:

$$
\begin{gathered}
\begin{pmatrix}11&3\\3&5\end{pmatrix}p_u
=\begin{pmatrix}10\\3\end{pmatrix},\\
p_u=\begin{pmatrix}41/46\\3/46\end{pmatrix}.
\end{gathered}
$$

```python
def implicit_user_update(Q, counts, alpha=2.0, penalty=1.0):
    counts = np.asarray(counts, dtype=float)
    if alpha < 0 or penalty <= 0 or np.any(counts < 0):
        raise ValueError("Invalid confidence or regularization parameter.")
    if counts.shape != (len(Q),):
        raise ValueError("One count is required per item.")
    confidence = 1 + alpha * counts
    positive = counts > 0
    system = Q.T @ (confidence[:, None] * Q) + penalty * np.eye(Q.shape[1])
    rhs = Q.T @ (confidence * positive)
    return np.linalg.solve(system, rhs)

Q_als = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
counts_one = np.array([3.0, 0.0, 1.0])
p_als = implicit_user_update(Q_als, counts_one)
np.testing.assert_allclose(p_als, [41/46, 3/46])
np.testing.assert_allclose(
    implicit_user_update(Q_als, np.zeros(3)), [0, 0])
print(p_als, Q_als @ p_als)
```

### Как учесть весь каталог без полного обхода для каждого пользователя

Матрица $C_u$ имеет ненулевые элементы на всей диагонали, но отличается от единичной только
на наблюдавшихся объектах. Поэтому

$$
Q^\top C_u Q
=Q^\top Q+\sum_{i:r_{ui}>0}(c_{ui}-1)q_iq_i^\top.
$$

$Q^\top Q$ вычисляют один раз на проход. Справа в системе тоже нужны
только положительные объекты:

$$
Q^\top C_u a_u=\sum_{i:r_{ui}>0}c_{ui}q_i.
$$

При $n_u$ ненулевых взаимодействиях работа для одной строки составляет
$O(n_u d^2+d^3)$ после вычисления общей Gram matrix.
Последнее слагаемое относится к решению плотной системы.
Для обновления объектов используют ту же процедуру с пользователями
и транспонированной матрицей взаимодействий.

```python
def implicit_user_update_sparse(Q, counts, gram, alpha=2.0, penalty=1.0):
    positive = np.flatnonzero(counts > 0)
    vectors = Q[positive]
    extra = alpha * counts[positive]
    system = gram + (vectors.T * extra) @ vectors
    system = system + penalty * np.eye(Q.shape[1])
    rhs = vectors.T @ (1 + extra)
    return np.linalg.solve(system, rhs)

def implicit_objective(P, Q, counts, alpha=2.0, penalty=1.0):
    preference = counts > 0
    confidence = 1 + alpha * counts
    return np.sum(confidence * (preference - P @ Q.T) ** 2) + penalty * (
        np.sum(P * P) + np.sum(Q * Q))

implicit_counts = np.array([[3.0, 0.0, 1.0],
                            [0.0, 2.0, 1.0],
                            [1.0, 1.0, 0.0]])
Q_imp = Q_als.copy()
P_imp = np.zeros((len(implicit_counts), Q_imp.shape[1]))
implicit_losses = [implicit_objective(P_imp, Q_imp, implicit_counts)]
for _ in range(20):
    gram = Q_imp.T @ Q_imp
    P_imp = np.array([implicit_user_update_sparse(Q_imp, row, gram)
                      for row in implicit_counts])
    implicit_losses.append(implicit_objective(P_imp, Q_imp, implicit_counts))
    gram = P_imp.T @ P_imp
    Q_imp = np.array([implicit_user_update_sparse(P_imp, col, gram)
                      for col in implicit_counts.T])
    implicit_losses.append(implicit_objective(P_imp, Q_imp, implicit_counts))
assert np.all(np.diff(implicit_losses) <= 1e-10)
np.testing.assert_allclose(
    implicit_user_update_sparse(Q_als, counts_one, Q_als.T @ Q_als), p_als)
print("Implicit objective:", implicit_losses[0], implicit_losses[-1])
```

В этом маленьком примере `counts` остаётся плотным массивом для наглядности.
На большом наборе ненулевые индексы хранят разреженно; иначе уже
`flatnonzero` просматривает весь каталог и уничтожает выигрыш поиска.
Также один раз на проход строят обе ориентации истории.

Если $N$ — число ненулевых пар, один полный цикл требует порядка

$$
O\bigl((m+n)d^2+Nd^2+(m+n)d^3\bigr)
$$

арифметических операций. Слагаемое $(m+n)d^2$ относится к общим Gram
матрицам. Это не квадратичный обход всех $mn$ пар для каждого обновления.
При фиксированной размерности фактора зависимость от числа наблюдений
линейна; зависимость от $d$ скрывать нельзя.

Каждый точный блоковый шаг не увеличивает выбранную цель, которая
ограничена снизу нулём. Поэтому значения loss сходятся.
Совместная задача по $P,Q$ не выпукла: это не доказательство глобального
минимума или единственного предела самих факторов.

### Объяснение рекомендации через прошлые взаимодействия

Подставим решение пользовательской системы в score целевого объекта:

$$
\begin{aligned}
p_u&=G_u^{-1}\sum_{j:r_{uj}>0}c_{uj}q_j,\\
G_u&=Q^\top C_u Q+\lambda I,\\
s(u,i)&=\sum_{j:r_{uj}>0}
c_{uj}q_i^\top G_u^{-1}q_j.
\end{aligned}
$$

Это разложение из [Hu, Koren, Volinsky, §5](https://yifanhu.net/PUB/cf.pdf)
показывает вклад каждого положительного объекта истории.
В отличие от SAR, сходство $q_i^\top G_u^{-1}q_j$ зависит от пользователя:
$G_u$ меняется вместе с его уверенностями.

Например, пусть положительные объекты имеют векторы $(1;0)$ и $(0;1)$,
целевой объект — $(1;-1)$, их confidence равны 3,3,1, а $\lambda=1$.
Пользовательское решение равно $(0{,}75;0{,}75)$, итоговый score — 0.
Вклады двух положительных объектов равны $0{,}5$ и $-0{,}5$.

```python
Q_explain = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
counts_explain = np.array([1.0, 1.0, 0.0])
confidence_explain = 1 + 2 * counts_explain
G = Q_explain.T @ (confidence_explain[:, None] * Q_explain) + np.eye(2)
p_explain = implicit_user_update(Q_explain, counts_explain)
contributions = np.array([
    confidence_explain[j] * Q_explain[2] @ np.linalg.solve(G, Q_explain[j])
    for j in [0, 1]
])
np.testing.assert_allclose(p_explain, [0.75, 0.75])
np.testing.assert_allclose(contributions, [0.5, -0.5])
np.testing.assert_allclose(contributions.sum(), Q_explain[2] @ p_explain,
                           atol=1e-14)
print(contributions, contributions.sum())
```

Даже при неотрицательных confidence отдельный вклад может быть отрицательным.
Это алгебраическое объяснение вычисленного score, не причинный эффект
прошлой покупки. Оно не доказывает, что удаление события и переобучение
всех факторов изменило бы результат ровно на этот вклад.

## От обучения к выдаче

Для фиксированного пользователя $\mu+b_u$ не влияет на порядок.
Объектное смещение $b_i$ влияет. Его можно включить в поиск по максимальному
скалярному произведению:

$$
\begin{aligned}
\tilde p_u&=(p_u;1),\\
\tilde q_i&=(q_i;b_i),\\
\tilde p_u^\top\tilde q_i&=p_u^\top q_i+b_i.
\end{aligned}
$$

Индекс для такого *maximum inner product search* должен использовать именно
скалярное произведение. Нормировка векторов с последующим cosine search
может изменить порядок. Например, для $p=(1;0)$ объекты
$q_1=(10;10)$ и $q_2=(2;0)$ имеют произведения 10 и 2, но cosine scores
$1/\sqrt2$ и 1: лучший объект меняется.

```python
query_vector = np.array([1.0, 0.0])
item_vectors = np.array([[10.0, 10.0], [2.0, 0.0], [1.0, 1.0]])
item_bias = np.array([-9.0, 0.0, 0.5])
direct = item_vectors @ query_vector + item_bias
augmented = np.column_stack([item_vectors, item_bias]) @ np.r_[query_vector, 1]
np.testing.assert_allclose(direct, augmented)
dot_scores = item_vectors[:2] @ query_vector
cosine_scores = dot_scores / np.linalg.norm(item_vectors[:2], axis=1)
assert np.argmax(dot_scores) == 0
assert np.argmax(cosine_scores) == 1
print("With item bias:", direct, "cosine:", cosine_scores)
```

Приближённый поиск может пропустить кандидатов; фильтрация доступности
или просмотренного может дополнительно сократить список.
Чтобы получить $K$ допустимых объектов, иногда извлекают больше $K$,
повторяют запрос или используют запасной способ поиска.
Порядок с точным полным scoring служит отдельной проверкой индекса,
а не следует из качества обучения факторов.

Вместе с таблицами нужны отображения `external_id → row`:
вектор в строке 10 без версии соответствия идентификаторов может оказаться
вектором другого фильма. Признаки, факторы, смещения и индекс
версионируются как одна совместимая конфигурация обслуживания.

## Где модель ломается

У нового ID нет обученного свободного вектора. Случайная строка,
созданная для неизвестного пользователя, не становится его предпочтением.
Без истории нужен явно выбранный fallback, например популярность
из [[00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline|предыдущей главы]].
Если новая история уже появилась, при фиксированных объектных факторах
можно решить пользовательскую систему ALS, не переобучая все $Q$.
При нулевой истории приведённая система даёт $p_u=0$; персонального
порядка без дополнительных признаков она не создаёт.

Новый объект с несколькими взаимодействиями можно аналогично дообучить
по фиксированным пользовательским факторам. Для объекта без взаимодействий
потребуются признаки содержимого или иной источник кандидатов.

Редкие пользователи и объекты имеют мало ограничений на векторы.
Регуляризация уменьшает свободу подгонки, но не добавляет отсутствующую
информацию. Изменение предпочтений во времени и смещение показов тоже
не устраняются факторизацией: нужны соответствующие данные и протокол
[[00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация|оценивания]].

## Контрольный эксперимент

Сравнение методов начинается с одной задачи. Для прогноза явных оценок
сравнивают среднее, смещения, выбранный соседский прогноз и explicit MF
на одинаковых проверочных парах. Для top-K используют популярность,
SAR, implicit MF или pairwise-модель на одном доступном каталоге
и одном способе построения целей. Смешивать RMSE звёзд с качеством
выдачи по кликам в одну шкалу нельзя.

В отчёте отдельно показывают новых пользователей и объекты,
короткие и длинные истории, время ответа и долю каталога, получающую показы.
Первые семь рейтингов этой главы проверяют арифметику и оптимизацию;
они не являются бенчмарком качества или доказательством превосходства MF.

## Первоисточники и продолжение

- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/mf|D2L: Matrix Factorization]] — полный английский оригинал с рисунком, реализацией на MXNet и обучением на MovieLens.
- [[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb|Recommenders: Estimating Baseline Performance]] — среднее пользователя для rating prediction и популярность для top-K.
- [[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/sar_deep_dive.ipynb|Recommenders: SAR deep dive]] — полный оригинальный notebook; [реализация](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/recommenders/models/sar/sar_singlenode.py) закреплена отдельно.
- [Google: Matrix factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) — сравнение функций потерь и alternating least squares.
- [Hu, Koren, Volinsky: Collaborative Filtering for Implicit Feedback Datasets](https://yifanhu.net/PUB/cf.pdf), §§3–5 — соседские методы, weighted MF, ускоренное ALS и разложение рекомендации на вклады истории.

Дальше: [[00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели|обучение ранжированию: BPR и нейронные модели]].
