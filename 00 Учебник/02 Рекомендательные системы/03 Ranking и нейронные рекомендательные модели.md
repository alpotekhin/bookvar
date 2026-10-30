---
title: Ранжирование и нейронные рекомендательные модели
type: textbook-chapter
status: canonical
last_updated: 2026-10-29
---

# Ранжирование и нейронные рекомендательные модели

Пользователь купил книгу A, а книги B и C пока не покупал. Для выдачи нужно
решить, какую из доступных книг поставить выше. Предсказание звёздной оценки
и обучение такого порядка используют разные данные и функции потерь:
снижение RMSE по известным оценкам не гарантирует улучшения первых позиций.

В [[00 Учебник/02 Рекомендательные системы/02 Collaborative filtering и matrix factorization|предыдущей главе]]
оценка пары вычислялась через скалярное произведение факторов. Здесь сначала
меняется способ обучения: вместо ошибки отдельного рейтинга модель получает
сравнения объектов. Затем меняется сама функция оценки: AutoRec восстанавливает
вектор рейтингов, NeuMF объединяет произведения факторов с MLP, FM и DeepFM
добавляют взаимодействия признаков.

Для BPR достаточно градиентов и факторизации матрицы. Для нейронных частей
понадобятся [[00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|MLP]],
[[00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|обратное распространение]]
и [[00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|автоэнкодер]].

<a id="pointwise-pairwise-и-listwise-постановки"></a>
## Отдельный отклик, пара и список

Для трёх книг пусть модель возвращает числа $s_A,s_B,s_C$.
Это scores: чем больше число, тем выше объект при обычной сортировке.
Само число ещё не является вероятностью клика.

При обучении по отдельным примерам, или *pointwise learning*, каждой паре
сопоставляют метку. Для бинарного отклика используют
$p_{ui}=\sigma(s_{ui})$ и бинарную кросс-энтропию. Для одной строки

$$
\begin{aligned}
\ell_{\mathrm{BCE}}(s,y)
&=-y\log\sigma(s)\\
&\quad -(1-y)\log(1-\sigma(s))\\
&=\operatorname{softplus}(s)-ys.
\end{aligned}
$$

Здесь $s$ — свободный logit, $y\in\{0,1\}$, а
$\operatorname{softplus}(x)=\log(1+e^x)$.
Производная $\partial\ell/\partial s=\sigma(s)-y$ повышает logit
положительного примера и понижает logit отрицательного. В вычислениях
используют устойчивую функцию потерь от logits, а не вручную берут
логарифмы вероятностей, округлившихся до 0 или 1.

Однако метка 0 из показанного и не нажатого объекта и метка 0 из случайной
выборки неизвестных пар означают разные наблюдения. Во втором случае
sigmoid обучается различать распределения выбранных положительных
и отрицательных примеров. Его значение не становится автоматически
вероятностью клика в реальной выдаче.

Попарное обучение, *pairwise learning*, использует тройку $(u,i,j)$:
для пользователя $u$ объект $i$ должен стоять выше $j$. Функция потерь
зависит от разности $\Delta=s_{ui}-s_{uj}$. Если увеличить оба scores
на одно число, разность не изменится. По одним сравнениям нельзя
восстановить абсолютную вероятность отклика.

При обучении по спискам, *listwise learning*, один пример содержит
группу кандидатов и целевые сведения о группе. Например, если среди
A, B и C выбран A, можно нормировать их logits общим softmax:

$$
\begin{aligned}
p_i&=\frac{e^{s_i}}{\sum_{j\in C_u}e^{s_j}},\\
\ell_{\mathrm{list}}&=-\log p_A.
\end{aligned}
$$

Для scores $(1,0,-1)$ получаются вероятности примерно
$(0{,}6652,0{,}2447,0{,}0900)$ и loss $0{,}4076$.
Градиент по трём scores равен $p-(1,0,0)$: повышение вероятности A
связано с остальными объектами той же группы. Такой loss моделирует
выбор внутри заданного списка; он не равен NDCG и не учитывает сам
по себе эффект позиции. Есть и другие listwise objectives; группировка
данных ещё не означает прямую оптимизацию метрики выдачи.

```python
import numpy as np

scores = np.array([1.0, 0.0, -1.0])
log_normalizer = np.logaddexp.reduce(scores)
list_probabilities = np.exp(scores - log_normalizer)
list_loss = log_normalizer - scores[0]
list_gradient = list_probabilities - np.array([1.0, 0.0, 0.0])
np.testing.assert_allclose(list_probabilities.sum(), 1.0)
np.testing.assert_allclose(list_gradient.sum(), 0.0, atol=1e-15)
shifted = scores + 1000
np.testing.assert_allclose(
    np.exp(shifted - np.logaddexp.reduce(shifted)),
    list_probabilities,
)
print(list_probabilities, list_loss, list_gradient)
```

Разделение трёх постановок следует
[D2L, Personalized Ranking](https://d2l.ai/chapter_recommender-systems/ranking.html).
Архитектуру и функцию потерь выбирают отдельно: одна таблица факторов
может обучаться MSE, бинарной кросс-энтропией или BPR.

## Bayesian Personalized Ranking

Пусть на границе обучения у пользователя $u$ известны взаимодействия
с множеством $I_u^+$, а $I$ — доступный каталог. В BPR из них строят

$$
D=\{(u,i,j):i\in I_u^+,\ j\in I\setminus I_u^+\}.
$$

Для каждой тройки принимают обучающее предположение $i>_u j$.
Это не полная разметка предпочтений: внутри $I_u^+$ и внутри его дополнения
порядок из таких данных не известен.

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-ranking.svg]]

*Zhang et al., исходный рисунок из
[D2L, Personalized Ranking](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recommender-systems/ranking.md).
[CC BY-SA 4.0](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/LICENSE-SUMMARY).
SVG не изменён.*

Слева показаны наблюдаемые действия нескольких пользователей.
Справа изображены попарные ограничения для $u_1$: столбец обозначает
первый объект $i$, строка — второй объект $j$. Плюс означает требование
поставить объект столбца выше объекта строки. Для $u_1$ получаются четыре
сравнения: $i_2>i_1$, $i_3>i_1$, $i_2>i_4$ и $i_3>i_4$.
Обратные сравнения отмечены минусами. Вопросительные знаки между двумя
наблюдаемыми или двумя неизвестными объектами не задают предпочтения.
Диагональ пуста: объект с самим собой не сравнивается.

В [BPR, разделы 3–4](https://arxiv.org/abs/1205.2618) вероятность сравнения
параметризуют логистической функцией:

$$
\begin{aligned}
P(i>_u j\mid\Theta)&=\sigma(\Delta_{uij}),\\
\Delta_{uij}&=s(u,i)-s(u,j).
\end{aligned}
$$

При независимой факторизации likelihood по обучающим сравнениям
произведение таких вероятностей превращается после логарифмирования
в сумму. Гауссов prior с плотностью, пропорциональной
$\exp(-\lambda\|\Theta\|^2/2)$, добавляет L2-штраф. Максимизация posterior
эквивалентна минимизации

$$
\begin{aligned}
\mathcal L(\Theta)
&=\sum_{(u,i,j)\in D}\ell(\Delta_{uij})
  +\frac{\lambda}{2}\|\Theta\|^2,\\
\ell(\Delta)&=-\log\sigma(\Delta)
 =\operatorname{softplus}(-\Delta).
\end{aligned}
$$

Независимость сравнений здесь является предположением модели. Из одного
действия получается много зависимых по происхождению троек; байесовская
запись не превращает их в независимо измеренные ответы пользователя.

<a id="как-loss-реагирует-на-margin"></a>
### Как разность scores меняет градиент

| $\Delta$ | $\ell(\Delta)$ | $\ell'(\Delta)$ |
| ---: | ---: | ---: |
| −2 | 2,1269 | −0,8808 |
| 0 | 0,6931 | −0,5000 |
| 2 | 0,1269 | −0,1192 |

Производная равна $\ell'(\Delta)=\sigma(\Delta)-1$.
При неверном порядке она по модулю близка к 1. Когда положительный
объект уже стоит выше с большим отрывом, обновление становится меньше,
но при конечной разности не исчезает полностью.

Если $s_+=0$, $s_-=2$ и два scores были бы независимыми параметрами,
одновременный шаг со скоростью $0{,}1$ дал бы
$s_+'=0{,}088080$ и $s_-'=1{,}911920$. Порядок ещё неверен,
но разрыв уменьшился с 2 до $1{,}823841$.

```python
deltas = np.array([-2.0, 0.0, 2.0])
bpr_losses = np.logaddexp(0.0, -deltas)
bpr_derivatives = -np.exp(-np.logaddexp(0.0, deltas))
old_scores = np.array([0.0, 2.0])
g = bpr_derivatives[0]
new_scores = old_scores - 0.1 * np.array([g, -g])
np.testing.assert_allclose(new_scores, [0.0880797078, 1.9119202922])
assert new_scores[0] - new_scores[1] > -2
assert np.isfinite(np.logaddexp(0.0, 1000.0))
print(bpr_losses, bpr_derivatives, new_scores)
```

Для факторизации $s(u,i)=p_u^\top q_i$ общие параметры связаны:

$$
\begin{aligned}
\Delta&=p_u^\top(q_i-q_j),\\
\nabla_{p_u}\ell&=g(q_i-q_j),\\
\nabla_{q_i}\ell&=gp_u,\qquad
\nabla_{q_j}\ell=-gp_u,\\
g&=\sigma(\Delta)-1.
\end{aligned}
$$

Обновления меняют скалярные произведения, а не обязаны уменьшать
евклидово расстояние между векторами. Если к score добавить смещения
$b_u+b_i$, пользовательское $b_u$ сократится в разности, а объектные
$b_i-b_j$ останутся. Из чистой попарной ошибки $b_u$ не обучается.

Пусть $p=(0{,}2,0{,}4)$, $q_i=(0{,}5,0{,}1)$,
$q_j=(0{,}1,0{,}6)$. Scores равны $0{,}14$ и $0{,}26$,
$\Delta=-0{,}12$, $g\approx-0{,}52996$.
Вычисляем все градиенты по старым векторам и лишь затем обновляем:

```python
p = np.array([0.2, 0.4])
qi = np.array([0.5, 0.1])
qj = np.array([0.1, 0.6])
delta_before = p @ (qi - qj)
g = -np.exp(-np.logaddexp(0.0, delta_before))
grad_p, grad_qi, grad_qj = g * (qi - qj), g * p, -g * p
p_new = p - 0.1 * grad_p
qi_new = qi - 0.1 * grad_qi
qj_new = qj - 0.1 * grad_qj
delta_after = p_new @ (qi_new - qj_new)
assert delta_after > delta_before
assert np.logaddexp(0.0, -delta_after) < np.logaddexp(0.0, -delta_before)
print("BPR MF:", delta_before, delta_after, p_new, qi_new, qj_new)
```

В этом шаге нет регуляризации. Для полного objective выше её градиент
равен $\lambda\Theta$ и относится ко всему вектору параметров. Штрафовать
только строки, попавшие в очередную тройку, без поправки на частоту —
другая, частотно-взвешенная регуляризация, как и в предыдущей главе.

### Выбор троек задаёт веса обучения

Даже у одного пользователя может быть
$|I_u^+|(|I|-|I_u^+|)$ троек. Обычно их не материализуют.
Например, сначала выбирают пользователя, затем его положительный
и неизвестный объекты.

Пусть у первого пользователя один positive и три unknown,
а у второго три positives и один unknown. У обоих по три допустимые
тройки. Но если равномерно выбирать положительное событие из общего
журнала, первый пользователь получит четверть обновлений, второй —
три четверти. Равномерный выбор пользователя даёт им равные веса.
Оба способа возможны, но минимизируют разные взвешенные ожидания.

```python
positive_counts = np.array([1, 3])
unknown_counts = np.array([3, 1])
triple_counts = positive_counts * unknown_counts
uniform_triple_user_mass = triple_counts / triple_counts.sum()
uniform_event_user_mass = positive_counts / positive_counts.sum()
np.testing.assert_allclose(uniform_triple_user_mass, [0.5, 0.5])
np.testing.assert_allclose(uniform_event_user_mass, [0.25, 0.75])
print(uniform_triple_user_mass, uniform_event_user_mass)
```

Hard-negative sampling чаще подаёт объекты с высокими scores.
Они могут давать больший градиент, но одновременно меняют распределение
задачи и могут оказаться интересными пользователю. Если требуется
оценивать исходную сумму по всем тройкам, вероятность выбора входит
в importance weighting; без такой поправки не следует называть любое
sampling равномерной оптимизацией исходной суммы.

<a id="bpr-не-превращает-unseen-в-настоящий-dislike"></a>
### Что известно об отрицательном примере

Обычный BPR предполагает, что наблюдаемый объект предпочтительнее выбранного
неизвестного. Если неизвестную книгу не показывали, отказ от неё не наблюдался.
Изменение модели не устраняет эту неопределённость.

В offline-эксперименте множество negatives строят из данных, доступных
на границе обучения. Из него исключают positives обучающего snapshot,
а не будущие покупки из validation/test. На границе доступны A, B, C,
известна только покупка A: допустимы B и C. Будущая покупка B не позволяет
задним числом сузить выборку до C.

```python
def training_negative_pool(catalog_at_train_cutoff, train_positive_ids):
    return set(catalog_at_train_cutoff) - set(train_positive_ids)

pool = training_negative_pool({"A", "B", "C"}, {"A"})
assert pool == {"B", "C"}
# Будущий positive B намеренно не передаётся в функцию.
```

Для обучения на исторических запросах каталог и история восстанавливаются
на момент каждого запроса. Глобальная граница snapshot и доступность
отдельной строки признаков разобраны в
[[00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline|первой главе]].
Сама операция исключения наблюдаемых positives следует определению $D$
в BPR; temporal split дополнительно ограничивает, что можно знать.

## Hinge ranking loss

Попарный hinge требует заранее заданного зазора $m>0$:

$$
\ell_{\mathrm{hinge}}(\Delta)=\max(0,m-\Delta).
$$

При $m=1$ и scores $1{,}4$ против $0{,}8$ порядок правильный,
но $\Delta=0{,}6$, поэтому loss равен $0{,}4$.
Если scores равны 2 и $0{,}5$, разность $1{,}5$ уже превышает margin:
loss и его градиент равны нулю. При $\Delta<m$ производная по разности
равна −1. В точке $\Delta=m$ функция недифференцируема; оптимизатор
использует выбранный допустимый субградиент.

```python
margins = np.array([-2.0, 0.6, 1.5])
hinge_losses = np.maximum(0.0, 1.0 - margins)
np.testing.assert_allclose(hinge_losses, [3.0, 0.4, 0.0])
print(hinge_losses)
```

[D2L](https://d2l.ai/chapter_recommender-systems/ranking.html)
показывает BPR и hinge как две функции потерь для одной постановки.
Их можно подставить к одному score, но величины градиента и поведение
после достижения зазора отличаются. Margin, масштаб scores,
регуляризация и скорость обучения требуют отдельной настройки.

<a id="autorec-восстановление-строки-матрицы"></a>
## AutoRec: восстановление наблюдаемых рейтингов

AutoRec решает задачу реконструкции, а не задаёт новую попарную функцию
потерь. Item-based вариант получает столбец рейтингов одного объекта
от всех пользователей. User-based вариант получает строку одного
пользователя по всем объектам.

Следуя [D2L, AutoRec](https://d2l.ai/chapter_recommender-systems/autorec.html),
обозначим заполненный нулями столбец через $r^{(i)}\in\mathbb R^m$,
а маску наблюдений — через $M^{(i)}$. При hidden dimension $h$

$$
\begin{aligned}
z^{(i)}&=\phi(W_e r^{(i)}+b_e),\\
\hat r^{(i)}&=W_d z^{(i)}+b_d,\\
W_e&\in\mathbb R^{h\times m},
\quad W_d\in\mathbb R^{m\times h}.
\end{aligned}
$$

Encoder объединяет известные оценки в скрытые признаки,
decoder возвращает $m$ прогнозов, включая неизвестные координаты.
Одни и те же веса $W_e,W_d$ используются для разных объектов.
Это отличается от отдельного свободного вектора $q_i$ для каждого ID.

Обучение учитывает только известные targets:

$$
\begin{aligned}
\mathcal L
&=\sum_i\|M^{(i)}\odot
  (r^{(i)}-\hat r^{(i)})\|_2^2\\
&\quad+\lambda(\|W_e\|_F^2+\|W_d\|_F^2).
\end{aligned}
$$

Для трёх пользователей возьмём рейтинги $(5,?,1)$.
После технического заполнения вход равен $(5,0,1)$,
маска $(1,0,1)$. Один ReLU-нейрон с весами $(0{,}2,0,0{,}2)$
выдаёт $z=1{,}2$. Decoder с весами $(2,3,1)^\top$ предсказывает
$(2{,}4,3{,}6,1{,}2)$. Ошибка равна
$(2{,}4-5)^2+(1{,}2-1)^2=6{,}8$, а не сумме по всем трём координатам.

Градиент по выходу decoder равен $(-5{,}2,0,0{,}4)$.
Для hidden activation он равен
$2(-5{,}2)+3\cdot0+1\cdot0{,}4=-10$.
Ненулевые координаты входа передают его encoder:
градиент весов encoder равен $(-50,0,-10)$.

```python
auto_input = np.array([5.0, 0.0, 1.0])
auto_mask = np.array([True, False, True])
encoder = np.array([0.2, 0.0, 0.2])
decoder = np.array([2.0, 3.0, 1.0])
hidden = max(0.0, encoder @ auto_input)
reconstruction = decoder * hidden
residual = np.where(auto_mask, reconstruction - auto_input, 0.0)
auto_loss = residual @ residual
grad_output = 2 * residual
grad_hidden = decoder @ grad_output
grad_encoder = grad_hidden * auto_input  # ReLU активен: производная 1.
grad_decoder = grad_output * hidden
np.testing.assert_allclose(reconstruction, [2.4, 3.6, 1.2])
np.testing.assert_allclose(auto_loss, 6.8)
np.testing.assert_allclose(grad_encoder, [-50, 0, -10])
np.testing.assert_allclose(grad_decoder, [-6.24, 0, 0.48])
print(auto_loss, grad_output, grad_encoder, grad_decoder)
```

Маска зануляет вклад неизвестной выходной координаты, но общие encoder
weights всё равно получают градиент от известных targets. Во время
предсказания весь выход сохраняют; иначе прогнозы на пропусках исчезнут.

При проверке отложенного рейтинга его нет ни в loss обучения,
ни во входе encoder. D2L передаёт в evaluator обучающую матрицу
и отдельно использует тестовую матрицу как targets. Подать на вход
полный вектор со скрываемой оценкой означало бы раскрыть ответ.
Число входов/выходов зависит от числа пользователей в item-based варианте;
новый пользователь требует расширить координатное пространство.
У нового объекта без оценок encoder видит тот же нулевой вход,
что и у любого другого такого объекта.

## Neural Collaborative Filtering

NCF — семейство нейронных функций оценки взаимодействий.
Вариант NeuMF объединяет GMF и MLP. Разбор архитектуры и обучения
содержится в полных оригиналах
[[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/neumf|D2L NeuMF]]
и [[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb|Recommenders NCF]].

GMF сначала перемножает координаты факторов пользователя и объекта:

$$
x_G=p_u^G\odot q_i^G,\qquad s_G=h_G^\top x_G.
$$

При $h_G=(1,\ldots,1)$ получается обычное скалярное произведение.
Обучаемый $h_G$ меняет веса координат, но до выходной активации
эта ветвь остаётся взвешенной билинейной формой.

Другая ветвь использует собственные embeddings:

$$
\begin{aligned}
z_0&=[p_u^M;q_i^M],\\
z_{\ell+1}&=\operatorname{ReLU}(W_\ell z_\ell+b_\ell).
\end{aligned}
$$

Сама конкатенация ничего не перемножает. Если после неё оставить
только линейный слой, получится сумма отдельного пользовательского
и объектного вкладов. Скрытые нелинейные слои позволяют обучать
совместную зависимость от двух входов.

NeuMF соединяет $x_G$ и последний $z_L$ и вычисляет один logit:

$$
s_{ui}=h^\top[x_G;z_L],\qquad p_{ui}=\sigma(s_{ui}).
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-neumf.svg]]

*Zhang et al.,
[D2L, Neural Collaborative Filtering](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recommender-systems/neumf.md).
[CC BY-SA 4.0](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/LICENSE-SUMMARY).
Исходный SVG не изменён.*

Нижние четыре узла рисунка уже показывают выбранные embeddings,
а не исходные ID. Векторы $p_u,q_i$ относятся к GMF, $u_u,v_i$ — к MLP.
В наших формулах это соответственно $p_u^G,q_i^G,p_u^M,q_i^M$.
Слева выполняется покоординатное произведение, справа — конкатенация
и последовательность скрытых слоёв. Верхний слой получает результаты
обеих ветвей, а не сумму двух готовых вероятностей.

### Формы тензоров

Пусть размер batch равен $B$, размер факторов GMF — $d_G$,
размер факторов MLP — $d_M$, последний hidden layer имеет $h$ выходов.

| Величина | Форма |
| --- | --- |
| $p_u^G,q_i^G$ | $B\times d_G$ |
| $x_G$ | $B\times d_G$ |
| $p_u^M,q_i^M$ | $B\times d_M$ |
| $z_0$ | $B\times2d_M$ |
| $z_L$ | $B\times h$ |
| $[x_G;z_L]$ | $B\times(d_G+h)$ |
| logit | $B$ или $B\times1$ |

Размеры ветвей не обязаны совпадать. Без учёта biases embedding tables
содержат $(m+n)(d_G+d_M)$ параметров. Число параметров MLP и выхода
добавляется отдельно.

Для одного примера возьмём GMF-векторы $(1,2)$ и $(0{,}5,-1)$:
их произведение равно $(0{,}5,-2)$. Пусть MLP получает
$(1,0;0,1)$, использует один скрытый слой с двумя выходами
и возвращает $(2,0)$. Выходные веса
$(0{,}2,-0{,}1,0{,}3,0{,}4)$ дают logit
$0{,}1+0{,}2+0{,}6=0{,}9$.

```python
p_g, q_g = np.array([1., 2.]), np.array([0.5, -1.])
p_m, q_m = np.array([1., 0.]), np.array([0., 1.])
mlp_weights = np.array([[1., 0., 0., 1.], [-1., 0., 0., 0.]])
mlp_hidden = np.maximum(0., mlp_weights @ np.r_[p_m, q_m])
gmf_hidden = p_g * q_g
fusion = np.r_[gmf_hidden, mlp_hidden]
output_weights = np.array([0.2, -0.1, 0.3, 0.4])
ncf_logit = output_weights @ fusion
ncf_probability = np.exp(-np.logaddexp(0., -ncf_logit))
np.testing.assert_allclose(mlp_hidden, [2., 0.])
np.testing.assert_allclose(ncf_logit, 0.9)
print(gmf_hidden, mlp_hidden, ncf_logit, ncf_probability)
```

### От logit к функции потерь

В [исходной статье NCF, раздел 3](https://arxiv.org/html/1708.05031v2#S3)
и Recommenders модель обучается pointwise BCE с выбранными negatives.
В D2L архитектуру обучают pairwise BPR. Это разные эксперименты,
даже если таблицы embeddings и скрытые слои имеют одинаковые размеры.

Для BCE с logits используют $s$; sigmoid вычисляется внутри устойчивой
функции потерь. Для BPR используют $s_+-s_-$. Дополнительный sigmoid
перед этой разностью ограничил бы её интервалом $(-1,1)$.
Тогда даже при идеальном разделении нижняя грань попарного loss
была бы $\operatorname{softplus}(-1)\approx0{,}3133$ вместо 0,
а насыщение обоих sigmoid уменьшало бы градиенты.
D2L в своём коде действительно оставляет sigmoid перед BPR;
здесь для BPR мы явно выбираем свободные logits.

Монотонный sigmoid сохраняет сортировку logits в точной арифметике.
Он не делает распределение обучающих negatives равным распределению
реальных показов и не доказывает калибровку.

### Инициализация из GMF и MLP

В Recommenders §3.5 сначала обучают ветви отдельно, затем копируют
их embeddings и скрытые слои в NeuMF. Выходные веса объединяют как

$$
h=\begin{bmatrix}\alpha h_G\\(1-\alpha)h_M\end{bmatrix}.
$$

До дальнейшего обучения это даёт logit
$s=\alpha s_G+(1-\alpha)s_M$. Смесь считается до sigmoid.
При $s_G=2$, $s_M=-1$, $\alpha=0{,}5$ получается
$\sigma(0{,}5)\approx0{,}6225$, а не среднее
$\bigl(\sigma(2)+\sigma(-1)\bigr)/2\approx0{,}5749$.

```python
sigmoid = lambda x: np.exp(-np.logaddexp(0.0, -np.asarray(x)))
alpha, s_g, s_m = 0.5, 2.0, -1.0
fused_probability = sigmoid(alpha * s_g + (1-alpha) * s_m)
average_probability = alpha * sigmoid(s_g) + (1-alpha) * sigmoid(s_m)
assert abs(fused_probability - average_probability) > 0.04
print(fused_probability, average_probability)
```

Такой перенос требует совместимых размеров слоёв и одинакового
отображения user/item ID. Состояния оптимизатора не переносятся
автоматически вместе с весами. Инициализация не гарантирует
улучшения на другой задаче; его проверяют тем же протоколом,
что и обучение с нуля.

<a id="почему-comparison-часто-нечестен"></a>
### Как сравнивать NeuMF с факторизацией

При разных sampling, числе параметров, бюджете подбора и наборе кандидатов
нельзя приписать разницу метрики одной архитектуре.
В Recommenders §3.4.2 рассчитываются scores по обучающему каталогу
с удалением уже наблюдаемых пар. В §3.4.3 используется один тестовый
positive и 100 sampled negatives. Эти протоколы дают разные ранги.

Для positive со score $0{,}5$ и negatives со scores
$0{,}9,0{,}8,0{,}1,0$ полный rank равен 3.
Если в выборку попали только два последних negative, rank равен 1:
Hit@1 меняется с 0 на 1 без изменения модели.

```python
positive_score = 0.5
all_negative_scores = np.array([0.9, 0.8, 0.1, 0.0])
full_rank = 1 + np.count_nonzero(all_negative_scores >= positive_score)
sampled_rank = 1 + np.count_nonzero(all_negative_scores[2:] >= positive_score)
assert (full_rank, sampled_rank) == (3, 1)
print("Full/sampled ranks:", full_rank, sampled_rank)
```

Здесь ties пессимистично считаются выше positive; в ином протоколе
потребуется другое явно заданное правило. В опубликованной ячейке
Recommenders §3.4.3 для NDCG используется натуральный логарифм без
деления на идеальное значение. Для единственного positive стандартный
нормированный вклад равен $1/\log_2(r+1)$ при $r\le K$;
при ранге 1 он должен быть 1, а не $1/\ln2$.
Оригинальная ячейка сохранена как источник, но её число не используется
здесь как проверенная стандартная NDCG.
Полный вывод метрик находится в
[[00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация|главе об оценивании]].

<a id="factorization-machines-interactions-между-полями"></a>
## Factorization Machines: взаимодействия признаков

Свободные user/item embeddings не описывают новый объект, для которого
нет взаимодействий. Признаки категории, автора или устройства могут
разделяться между множеством объектов и запросов. Для разреженного
вектора $x\in\mathbb R^D$ факторизационная машина второго порядка задаёт score

$$
\begin{aligned}
s_{\mathrm{FM}}(x)
&=w_0+\sum_{i=1}^D w_i x_i\\
&\quad+\sum_{i<j}\langle v_i,v_j\rangle x_i x_j,\\
v_i&\in\mathbb R^k.
\end{aligned}
$$

Каждый активный признак получает линейный вес и вектор.
Коэффициент пары не хранится отдельно для каждой пары:
он равен скалярному произведению двух векторов.
Если активны только один user ID и один item ID с единичными значениями,
формула сводится к смещениям и MF. При добавлении устройства появляются
ещё взаимодействия user–device и item–device.

Один и тот же признак использует один вектор во всех своих парах.
Поэтому обновление $v_i$ по одной комбинации меняет и другие комбинации
с этим признаком. Это разделение параметров помогает использовать
разреженные наблюдения, но не создаёт знания о совершенно неизвестном
признаке без обученного представления или выбранного default.

### От суммы пар к линейному по числу признаков вычислению

[D2L, FM](https://d2l.ai/chapter_recommender-systems/fm.html)
раскрывает квадрат суммы для каждой координаты $\ell$:

$$
\begin{aligned}
\left(\sum_i v_{i\ell}x_i\right)^2
&=\sum_i v_{i\ell}^2x_i^2\\
&\quad+2\sum_{i<j}v_{i\ell}v_{j\ell}x_ix_j.
\end{aligned}
$$

Вычитание диагонали и деление на два оставляет каждую неупорядоченную
пару ровно один раз. После суммирования по $\ell$

$$
\begin{aligned}
a_\ell&=\sum_i v_{i\ell}x_i,\\
s_{\mathrm{int}}
&=\frac12\sum_{\ell=1}^k
 \left(a_\ell^2-\sum_i v_{i\ell}^2x_i^2\right).
\end{aligned}
$$

Явный перебор активных пар требует $O(kF^2)$ операций при
$F=\operatorname{nnz}(x)$. Две суммы требуют $O(kF)$,
если вход действительно хранится и обходится разреженно.

Пусть активные признаки user, item и device имеют embeddings
$(1,0)$, $(0{,}5,1)$ и $(0,1)$, а значения признаков равны единице.
Три скалярных произведения равны $0{,}5$, 0 и 1, их сумма $1{,}5$.
По второй формуле $a=(1{,}5,2)$, сумма квадратов $a$ равна $6{,}25$,
сумма квадратов отдельных элементов равна $3{,}25$.
Половина разности тоже даёт $1{,}5$.

```python
feature_vectors = np.array([[1., 0.], [0.5, 1.], [0., 1.]])
feature_values = np.ones(3)
scaled = feature_vectors * feature_values[:, None]
pairwise_fm = sum(
    scaled[i] @ scaled[j]
    for i in range(len(scaled)) for j in range(i+1, len(scaled))
)
fast_fm = 0.5 * (
    np.square(scaled.sum(axis=0)).sum() - np.square(scaled).sum()
)
np.testing.assert_allclose(pairwise_fm, 1.5)
np.testing.assert_allclose(fast_fm, pairwise_fm)
print("FM interactions:", pairwise_fm, fast_fm)
```

Для общего вещественного $x$ производная interaction-score равна

$$
\frac{\partial s_{\mathrm{int}}}{\partial v_{i\ell}}
=x_i(a_\ell-v_{i\ell}x_i).
$$

После умножения на производную функции потерь это даёт градиент
обучения. При $x_i=0$ данный пример не обновляет этот вектор
через interaction-term. Для бинарного отклика к score добавляют sigmoid
и BCE, для рейтинга можно использовать MSE, для пары — разность
двух scores и BPR.

Для нескольких категориальных полей пространство ID разделяют:
`user=7` и `item=7` не должны случайно обращаться к одной строке.
Например, при размерах словарей $(3,4,2)$ смещения полей равны
$(0,3,7)$. Локальные IDs $(1,2,0)$ обращаются к строкам $(1,5,7)$.
Эти словари и defaults обучают на train, а не пересобирают
для каждого split.

```python
field_sizes = np.array([3, 4, 2])
offsets = np.r_[0, np.cumsum(field_sizes[:-1])]
local_ids = np.array([1, 2, 0])
assert np.all((0 <= local_ids) & (local_ids < field_sizes))
global_ids = offsets + local_ids
np.testing.assert_array_equal(global_ids, [1, 5, 7])
print("Disjoint field rows:", global_ids)
```

<a id="deepfm-low-order-и-high-order-interactions"></a>
## DeepFM: общие embeddings для FM и MLP

DeepFM использует один набор embeddings в двух ветвях.
FM рассчитывает линейные веса и все взаимодействия второго порядка.
MLP получает конкатенацию тех же embeddings и возвращает ещё один logit.
Их сумма проходит через один sigmoid:

$$
\begin{aligned}
z_0&=[e_1;\ldots;e_f],\\
z_{\ell+1}&=\operatorname{ReLU}(W_\ell z_\ell+b_\ell),\\
s&=s_{\mathrm{FM}}+s_{\mathrm{DNN}},\\
p&=\sigma(s).
\end{aligned}
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-deepfm.svg]]

*Zhang et al.,
[D2L, Deep Factorization Machines](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recommender-systems/deepfm.md).
[CC BY-SA 4.0](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/LICENSE-SUMMARY).
Исходный SVG не изменён.*

На рисунке embeddings $e_1,\ldots,e_f$ общие для FM и глубокой ветви.
Это отличается от четырёх независимых embedding tables NeuMF.
Глубокая ветвь может выражать нелинейную совместную зависимость
от нескольких полей; она не является явным перечислением всех
полиномов заданной степени.

Продолжим пример user/item/device. Линейный вклад вместе с bias равен
$0{,}2$, FM-interactions равны $1{,}5$, поэтому
$s_{\mathrm{FM}}=1{,}7$. MLP получает
$(1,0,0{,}5,1,0,1)$ и использует один ReLU-нейрон с весами
$(1,0,-1,0,0,0)$. Он выдаёт $0{,}5$; выходной вес $0{,}4$
даёт $s_{\mathrm{DNN}}=0{,}2$. Итоговый logit равен $1{,}9$,
вероятность примерно $0{,}8699$.

```python
deep_weights = np.array([1., 0., -1., 0., 0., 0.])
deep_input = feature_vectors.ravel()
deep_hidden = max(0.0, deep_weights @ deep_input)
deep_logit = 0.4 * deep_hidden
fm_logit = 0.2 + fast_fm
deepfm_logit = fm_logit + deep_logit
deepfm_probability = sigmoid(deepfm_logit)
np.testing.assert_allclose([deep_hidden, fm_logit, deep_logit], [0.5, 1.7, 0.2])
np.testing.assert_allclose(deepfm_logit, 1.9)
print(deepfm_logit, deepfm_probability)
```

Градиент по общему embedding складывается из двух путей.
В примере для item-вектора $e_i=(0{,}5,1)$ FM даёт
$\partial s_{\mathrm{FM}}/\partial e_i=(1,1)$,
а MLP при активном ReLU даёт $(-0{,}4,0)$.
Суммарная производная logit равна $(0{,}6,1)$.
Для BCE с $y=1$ она умножается на
$\sigma(1{,}9)-1\approx-0{,}1301$, поэтому градиент item-вектора
примерно $(-0{,}0781,-0{,}1301)$.

```python
grad_item_logit = (
    feature_vectors[0] + feature_vectors[2]
    + 0.4 * deep_weights.reshape(3, 2)[1]
)
grad_item_bce = (deepfm_probability - 1.0) * grad_item_logit
np.testing.assert_allclose(grad_item_logit, [0.6, 1.0])
print("Shared item gradient:", grad_item_bce)
```

В [D2L](https://d2l.ai/chapter_recommender-systems/deepfm.html)
приведены полная реализация и обучение на advertising dataset.
Пример выше проверяет вычисление и общий градиент, а не воспроизводит
заявленное качество. Дорогая функция от совместных признаков
обычно применяется к уже найденным кандидатам. Её нельзя автоматически
заменить независимым индексированием item-векторов, как у чистого
скалярного произведения.

<!-- dlrm 2026-10-26 -->
## DLRM: dense features, embedding bags, and interactions

A click-prediction example can contain a price, a user ID, an item ID,
and a variable-length list of categories. Price is a numerical input;
an ID selects a row of a table. Giving an MLP the integer `item_id=1042`
would impose a numerical relationship with `item_id=1043` that the ID
assignment does not establish.

DLRM combines a bottom MLP for numerical features, embedding tables for
categorical features, pairwise dot products, and a top MLP. In the
[reference implementation](https://github.com/facebookresearch/dlrm/blob/9bc1bc3602aea22afad3206910cec50f2dbe7364/dlrm_s_pytorch.py),
`sequential_forward` runs these four stages in that order. The original
implementation ends with a sigmoid; the executable example below keeps
the final logit and uses `binary_cross_entropy_with_logits`.

### From variable-length IDs to one vector per field

The TorchRec documentation describes the pooling operation as follows:

<!-- source-excerpt: torchrec-pooling -->
> There is also the choice of pooling embeddings, often, we’re looking up
> multiple rows for a given feature which gives rise to the question of
> what we do with looking up multiple embedding vectors. Pooling is a
> common technique where we combine the embedding vectors, usually through
> sum or mean of the rows, to produce one embedding vector. This is the
> main difference between the PyTorch `nn.Embedding` and
> `nn.EmbeddingBag`.
<!-- /source-excerpt -->

*Meta Platforms, Inc. and affiliates,
[TorchRec High Level Architecture, Embedding Tables](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/docs/source/high-level-arch.rst#embedding-tables).
Original English excerpt, BSD-3-Clause; [full notice](/licenses/torchrec-BSD-3-Clause.txt).*

For a field $f$, let $E_f\in\mathbb R^{N_f\times d}$ be its table and
$I_{bf}$ the list of IDs for example $b$. Sum pooling returns

$$
e_{bf}=\sum_{i\in I_{bf}}E_f[i,:].
$$

The list can have one ID, several IDs, or no IDs. Sum pooling gives a
zero vector for an empty bag. Repeated IDs contribute repeatedly: the
bag `[2,2]` returns twice row 2. A mean instead divides a nonempty sum
by the bag length. These choices change the representation and its
gradient; a longer list has no automatic claim to greater relevance.
The ordinary embedding path in the pinned DLRM implementation explicitly
constructs `nn.EmbeddingBag(..., mode="sum", sparse=True)`.

![[00 Учебник/Assets/Figures/curated/dlrm-2026-10-26/torchrec-full-training-loop.png]]

*Meta Platforms, Inc. and affiliates,
[TorchRec High Level Architecture, Embedding Tables](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/docs/source/high-level-arch.rst#embedding-tables),
Figure 2. Complete original PNG, BSD-3-Clause;
[full notice](/licenses/torchrec-BSD-3-Clause.txt).*

In the figure, sample 1 selects rows 1, 2, and 6; sample 2 selects rows
1, 3, and 6. Each sum produces one output vector. During backpropagation,
each selected row receives the gradient of the sample using it. Rows
1 and 6 receive contributions from both samples, which must be added.
Rows 4 and 5 receive none from these samples.

The faded gradient boxes describe TorchRec's fused embedding backward
and optimizer update: it avoids materializing the full table-gradient
tensor. The numerical example below deliberately retains a conventional
gradient tensor so that the contributions can be inspected. The diagram
does not show the bottom or top MLP; it explains the embedding stage.

Suppose a product-feature table has rows $(0,0)$, $(0.5,0)$, $(0,1)$,
and $(0.5,1)$. The three bags `[1,3]`, `[]`, `[2,2]` produce
$(1,1)$, $(0,0)$, $(0,2)$. Flattening the IDs gives `[1,3,2,2]`.
With `include_last_offset=True`, offsets `[0,2,2,4]` delimit all three
bags, including the empty one. The terminal offset is the number of IDs,
not another example.

```python
import sys
import torch
from torch import nn
from torch.nn import functional as F

dlrm_dtype = torch.float64
dlrm_product = nn.EmbeddingBag.from_pretrained(
    torch.tensor([[0., 0.], [.5, 0.], [0., 1.], [.5, 1.]],
                 dtype=dlrm_dtype),
    freeze=False, mode="sum", include_last_offset=True,
)
dlrm_user = nn.EmbeddingBag.from_pretrained(
    torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]],
                 dtype=dlrm_dtype),
    freeze=False, mode="sum", include_last_offset=True,
)
dlrm_product_ids = torch.tensor([1, 3, 2, 2])
dlrm_product_offsets = torch.tensor([0, 2, 2, 4])
dlrm_user_ids = torch.tensor([2, 1, 3])
dlrm_user_offsets = torch.tensor([0, 1, 2, 3])
dlrm_products = dlrm_product(dlrm_product_ids, dlrm_product_offsets)
dlrm_users = dlrm_user(dlrm_user_ids, dlrm_user_offsets)
torch.testing.assert_close(
    dlrm_products,
    torch.tensor([[1., 1.], [0., 0.], [0., 2.]], dtype=dlrm_dtype),
)
print("Pooled product features:", dlrm_products.detach())
```

The product and user tables have separate namespaces: row 2 in one
does not mean row 2 in the other. An unseen ID needs an explicit policy,
such as an unknown row or a specified hash mapping. Hashing also creates
collisions, so it is not a way of learning an independent vector for
every possible ID. The example uses dense gradients to expose every
coordinate; it does not reproduce TorchRec's fused sparse optimizer.

### Which pairwise products reach the top MLP

The bottom MLP maps the numerical input to $v_b\in\mathbb R^d$.
All pooled categorical vectors have the same width $d$. For $F$ fields,
stack these vectors into rows:

$$
\begin{aligned}
T_b&=[v_b;e_{b1};\ldots;e_{bF}],\\
G_b&=T_bT_b^\top.
\end{aligned}
$$

The semicolons in $T_b$ mean stacking rows, so its shape is
$(F+1)\times d$. Entry $G_b[i,j]$ is the dot product of rows $i$ and $j$.
The reference implementation's default `dot` mode takes the strict lower
triangle once, excludes the diagonal, and concatenates it with $v_b$.
For the whole batch the shapes are:

| Tensor | Shape |
| --- | --- |
| Bottom MLP output | $B\times d$ |
| Stacked vectors | $B\times(F+1)\times d$ |
| Gram matrix | $B\times(F+1)\times(F+1)$ |
| Unique cross-field products | $B\times F(F+1)/2$ |
| Top MLP input | $B\times\bigl(d+F(F+1)/2\bigr)$ |

For $v=(1,2)$, product features $e_1=(1,1)$, and user features
$e_2=(0,1)$, the Gram matrix is

$$
G=\begin{pmatrix}
5&3&2\\
3&2&1\\
2&1&1
\end{pmatrix}.
$$

The selected entries, in source order, are $(3,2,1)$: product–dense,
user–dense, user–product. The top MLP receives $(1,2,3,2,1)$.
Neither $(5,2,1)$ on the diagonal nor the mirrored upper triangle is
included. With `arch_interaction_itself=True`, the source also includes
self-products; that is a different input width. Its `cat` mode replaces
dot products with concatenation and is another explicit configuration.

The following class keeps the reference `interact_features` method
unchanged. The two class attributes supply its configuration; they
are not a copy of the full DLRM class.

```python
class DLRMInteraction:
    arch_interaction_op = "dot"
    arch_interaction_itself = False

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

dlrm_interaction = DLRMInteraction()
dlrm_bottom = nn.Sequential(nn.Linear(2, 2), nn.ReLU()).to(dlrm_dtype)
with torch.no_grad():
    dlrm_bottom[0].weight.copy_(torch.eye(2, dtype=dlrm_dtype))
    dlrm_bottom[0].bias.zero_()
dlrm_dense_input = torch.tensor([[1., 2.]], dtype=dlrm_dtype)
dlrm_v = dlrm_bottom(dlrm_dense_input)
dlrm_r = dlrm_interaction.interact_features(
    dlrm_v, [dlrm_products[:1], dlrm_users[:1]],
)
torch.testing.assert_close(
    dlrm_r, torch.tensor([[1., 2., 3., 2., 1.]], dtype=dlrm_dtype),
)
print("Top MLP input:", dlrm_r.detach())
```

*`interact_features`: Facebook, Inc. and its affiliates,
[DLRM reference code](https://github.com/facebookresearch/dlrm/blob/9bc1bc3602aea22afad3206910cec50f2dbe7364/dlrm_s_pytorch.py#L483-L516),
MIT; [full notice](/licenses/dlrm-MIT.txt). Surrounding setup is the
textbook's CPU example.*

An FM sums the three pairwise products into one score. DLRM retains
their positions, so the top MLP can assign different weights to
product–dense and user–product interactions, then combine them
nonlinearly. For example, interaction vectors $(3,2,1)$ and $(1,2,3)$
both sum to 6, but weights $(0.3,0.4,-0.5)$ give contributions $1.2$
and $-0.4$. This compares the interaction branches at fixed vectors;
it is not a claim that an FM could never learn a different embedding.
DeepFM additionally gives its deep branch the concatenated embeddings.
Default DLRM gives the top MLP the processed dense vector and pairwise
dot products, not all categorical coordinates. It has no requirement
that categories called “user” and “product” be its only two fields.

With 26 categorical fields and width 128, the 27 vectors give 351
distinct cross-products. The top input has 479 coordinates, compared
with 3456 for concatenating all 27 vectors. Dot products discard
information about individual categorical coordinates; reducing the input
width is an architectural choice, not a guarantee of better predictions.
For a batch, explicitly computing all dot products costs
$O(B(F+1)^2d)$ arithmetic, and the printed implementation materializes
a $B\times(F+1)\times(F+1)$ Gram tensor. Table storage instead grows
with the number of IDs, $d\sum_f N_f$. A large vocabulary therefore
increases the parameter footprint without increasing the number of
field-pair coordinates for an example.

### Backpropagation through pooling and dot products

Use a linear top layer with weights $(0.1,-0.2,0.3,0.4,-0.5)$ and
bias $0.2$. This is the one-layer case of a top MLP. The logit is

$$
\begin{aligned}
s&=0.1v_1-0.2v_2+0.3(e_1^\top v)\\
 &\quad+0.4(e_2^\top v)-0.5(e_2^\top e_1)+0.2\\
 &=1.1.
\end{aligned}
$$

For a positive label the BCE is $\operatorname{softplus}(-1.1)$,
approximately $0.287335$. Its logit derivative is
$g=\sigma(1.1)-1\approx-0.249740$. Before multiplying by $g$,
the vector derivatives are

$$
\begin{aligned}
\nabla_vs&=(0.1,-0.2)+0.3e_1+0.4e_2\\
&=(0.4,0.5),\\
\nabla_{e_1}s&=0.3v-0.5e_2=(0.3,0.1),\\
\nabla_{e_2}s&=0.4v-0.5e_1=(-0.1,0.3).
\end{aligned}
$$

Because $e_1$ is the sum of product rows 1 and 3, each of those rows
receives $g(0.3,0.1)$. Unused rows receive zero from this example.
If an ID occurs twice in a bag, its row receives twice the pooled-vector
gradient. The dense vector also receives gradients from its direct
coordinates in the top input, not only from its dot products.

```python
dlrm_top = nn.Linear(5, 1).to(dlrm_dtype)
with torch.no_grad():
    dlrm_top.weight.copy_(torch.tensor(
        [[.1, -.2, .3, .4, -.5]], dtype=dlrm_dtype))
    dlrm_top.bias.fill_(.2)
dlrm_v.retain_grad()
dlrm_products.retain_grad()
dlrm_users.retain_grad()
dlrm_logit = dlrm_top(dlrm_r).squeeze(1)
dlrm_loss = F.binary_cross_entropy_with_logits(
    dlrm_logit, torch.ones_like(dlrm_logit))
dlrm_loss.backward()
dlrm_g = (dlrm_logit.sigmoid() - 1).detach()
torch.testing.assert_close(dlrm_logit, torch.tensor([1.1], dtype=dlrm_dtype))
torch.testing.assert_close(dlrm_v.grad, dlrm_g * torch.tensor(
    [[.4, .5]], dtype=dlrm_dtype))
dlrm_expected_product_grad = torch.zeros_like(dlrm_product.weight)
dlrm_expected_product_grad[[1, 3]] = dlrm_g * torch.tensor(
    [.3, .1], dtype=dlrm_dtype)
torch.testing.assert_close(dlrm_product.weight.grad, dlrm_expected_product_grad)
dlrm_optimizer = torch.optim.SGD(
    list(dlrm_bottom.parameters()) + list(dlrm_top.parameters())
    + list(dlrm_product.parameters()) + list(dlrm_user.parameters()), lr=.01)
dlrm_optimizer.step()
with torch.no_grad():
    dlrm_next_r = dlrm_interaction.interact_features(
        dlrm_bottom(dlrm_dense_input),
        [dlrm_product(dlrm_product_ids, dlrm_product_offsets)[:1],
         dlrm_user(dlrm_user_ids, dlrm_user_offsets)[:1]],
    )
    dlrm_next_loss = F.binary_cross_entropy_with_logits(
        dlrm_top(dlrm_next_r).squeeze(1), torch.ones_like(dlrm_logit))
assert dlrm_next_loss < dlrm_loss.detach()
print("One CPU SGD step:", dlrm_loss.item(), dlrm_next_loss.item())
```

This step checks the path from IDs to pooled vectors, interactions,
logit, loss, and updated tables. It does not measure recommendation
quality, train on Criteo, or benchmark a distributed implementation.

### Table capacity and lookup traffic are different quantities

The embedding-table example in
[Harvard ML Systems, Network Architectures](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd)
uses one billion rows, 128 coordinates, and FP32. Those weights occupy
$10^9\times128\times4=512\times10^9$ bytes: 512 GB, or approximately
476.84 GiB. That is one table, before optimizer state or other layers.
For an implementation keeping FP32 weights, a full FP32 gradient,
and two full FP32 moment tensors, these four arrays alone occupy
2.048 TB. This is a conditional accounting example, not a universal
DLRM training-memory multiplier. Sparse updates and fused optimizers
can avoid materializing the full gradient; optimizer-state layout must
still be counted for the chosen implementation.

Only selected rows are read for a request. For a batch of 1024 examples,
one field with 20 IDs per example, width 128, and four bytes per
coordinate, the selected-row payload is 10 MiB before pooling. Its
pooled output is 512 KiB. These are logical tensor sizes, not measured
HBM or network traffic: repeated IDs, caches, transaction granularity,
placement, and the implementation can change the bytes transferred.

```python
dlrm_table_bytes = 10**9 * 128 * 4
dlrm_selected_bytes = 1024 * 20 * 128 * 4
dlrm_pooled_bytes = 1024 * 128 * 4
assert dlrm_table_bytes == 512_000_000_000
assert dlrm_selected_bytes == 10 * 2**20
assert dlrm_pooled_bytes == 512 * 2**10
print("Table GB/GiB:", dlrm_table_bytes / 10**9, dlrm_table_bytes / 2**30)
print("Four FP32 arrays, TB:", 4 * dlrm_table_bytes / 10**12)
print("Selected/pooled MiB:", dlrm_selected_bytes / 2**20,
      dlrm_pooled_bytes / 2**20)
assert 27 * 26 // 2 == 351
assert 128 + 351 == 479
assert 27 * 128 == 3456
```

Increasing matrix-multiplication throughput does not make a table fit
in memory. Once it fits, lookup latency and bandwidth can still limit
the embedding stage. Index-based access also does not imply that every
lookup misses every cache: popular IDs and repeated requests can create
reuse. Pooling performs arithmetic after the gathers, so an embedding
bag is not only a memory copy. The dense MLP, sparse lookup, interaction,
and communication stages need separate measurements.

<!-- cold-ids-addition: start -->
<a id="cold-embedding-ids"></a>
### Repeated events, cold IDs, and table compaction

A training log can lose many rows while every user and item ID still occurs
somewhere in the remaining data. A fixed-size embedding table then has exactly
the same shape. Conversely, removing the only event for an ID can eliminate
that ID from the training vocabulary, but memory is freed only if the table
and its associated state are actually rebuilt or resized.

The distinction follows the lookup mechanism above: events select table rows;
events are not table rows. Let $R$ be the event log, $V_f(R)$ the distinct IDs
of field $f$, and $C_f$ its allocated row capacity. Removing events changes
$|R|$. It changes $|V_f(R)|$ only if some IDs disappear completely, and does
not by itself change $C_f$. With width $d_f$ and $b_f$ bytes per coordinate,
the allocated embedding weights occupy

$$
M_W=\sum_f C_f d_f b_f.
$$

For a dense table, setting unused rows to zero changes values, not allocation.
Taking a tensor view can also retain the original backing storage. A compact
copy must replace the old parameter and its optimizer references; memory
accounting must include any old arrays still kept alive during migration.

#### Remove duplicated events without erasing repeated behaviour

Suppose a pipeline receives ten rows, including two retries of events already
recorded. Events `e1` and `e2` are separate actions by the same user on item 10.
They are not transport duplicates. Here identical `event_id` values are the
declared evidence of retry, and a retry with conflicting contents is an error.

```python
from collections import Counter

events = [("e1", "u1", 10), ("e1", "u1", 10), ("e2", "u1", 10),
          ("e3", "u2", 20), ("e3", "u2", 20), ("e4", "u2", 30),
          ("e5", "u3", 20), ("e6", "u3", 40),
          ("e7", "u4", 50), ("e8", "u4", 60)]

def deduplicate_events(records):
    by_event = {}
    for event, user, item in records:
        value = (user, item)
        if event in by_event and by_event[event] != value:
            raise ValueError("Conflicting retry for " + event)
        by_event[event] = value
    return [(event, *by_event[event]) for event in sorted(by_event)]

clean_events = deduplicate_events(events)
counts = Counter(item for _, _, item in clean_events)
assert len(events) == 10 and len(clean_events) == 8
assert counts == {10: 2, 20: 2, 30: 1, 40: 1, 50: 1, 60: 1}
assert {item for _, _, item in events} == set(counts)
assert len({(user, item) for _, user, item in clean_events}) == 7
print("Events before/after:", len(events), len(clean_events))
print("Item IDs before/after:", 6, len(counts), "counts:", dict(counts))
```

Removing the two retries reduces events by 20% and removes zero item IDs.
An existing table with six item rows plus one unknown row still has seven
rows. Deduplicating by `(user, item)` instead would also erase `e2`, leaving
seven events and changing the frequency of item 10 from two to one. Whether
repeated actions should receive repeated loss weight depends on the target
and sampling protocol. If distinct events are aggregated, their counts,
times and labels may need to survive as features or weights. Equal user–item
pairs do not establish identical training observations.

#### A frequency cutoff is a separate modelling decision

Define the count over the cleaned training window, then keep a dedicated row
for IDs with count at least two. In this assigned example, IDs 10 and 20
survive, while 30, 40, 50 and 60 map to one out-of-vocabulary (OOV) row.
The threshold is chosen for the example, not recommended for production.
Counting the raw retried log, lifetime events, unique users, or a recent window
can produce different vocabularies. The count definition must match the policy.

All eight cleaned events can remain in training. The four cold-item events
now update a shared parameter. The new item table has three rows, not two:
two dedicated items plus OOV. Its index mapping must accompany the checkpoint
and be used identically in training and serving. User ID 10 belongs to a
different feature namespace and must not accidentally use item 10's row.

For FP32 width four, compare actual allocated arrays:

| Item-table allocation | Seven rows | Three rows |
| --- | ---: | ---: |
| Weights | 112 bytes | 48 bytes |
| First and second FP32 Adam moments | 224 bytes | 96 bytes |
| These three arrays together | 336 bytes | 144 bytes |

This counts neither gradients nor dictionary storage, step counters, other
fields, dense layers or allocator overhead. A row-wise optimizer can keep
one accumulator per row instead of two full moment vectors; a fused optimizer
may not materialize a full gradient. Use its actual state layout. The reduction
from seven to three rows is 57.1% for these arrays; it comes from the explicit
vocabulary policy, not from removing 20% of events.

The eight-event batch still selects eight item vectors before pooling.
At width four in FP32 that is 128 bytes of logical selected-row payload both
before and after remapping. The number of distinct rows touched falls from
six to three, which can improve reuse, but does not establish a measured
bandwidth or latency gain. The mapping lookup also consumes time and memory.

#### Copy weights and optimizer state by identity

Suppose the old checkpoint uses rows 1–6 for raw IDs 10–60 and row 0 for OOV.
Choose the new order OOV, 20, 10. Retained row 10 must move to new row 2
with its moments; copying only the first three old rows would swap meanings.
The OOV row was previously unused, so this example resets it and its moments
before learning from cold events. Averaging discarded Adam moments is not
equivalent to training one shared row on their combined gradients.

```python
import numpy as np

old_rows = {raw: row for row, raw in enumerate([10, 20, 30, 40, 50, 60], 1)}
old_weights = np.arange(28, dtype=np.float32).reshape(7, 4)
old_m = old_weights + 100
old_v = old_weights + 1000
source_rows = np.array([0, old_rows[20], old_rows[10]])
new_weights, new_m, new_v = [array[source_rows].copy()
                            for array in (old_weights, old_m, old_v)]
for array in (new_weights, new_m, new_v):
    array[0] = 0  # Explicit fresh OOV state, not a retained item identity.
new_rows = {20: 1, 10: 2}
lookup_row = lambda raw: new_rows.get(raw, 0)
assert [lookup_row(raw) for raw in [10, 20, 30, 999]] == [2, 1, 0, 0]
for raw in new_rows:
    for old, new in [(old_weights, new_weights), (old_m, new_m), (old_v, new_v)]:
        np.testing.assert_array_equal(new[new_rows[raw]], old[old_rows[raw]])
assert not np.shares_memory(new_weights, old_weights)
assert sum(a.nbytes for a in [old_weights, old_m, old_v]) == 336
assert sum(a.nbytes for a in [new_weights, new_m, new_v]) == 144

# Four distinct cold IDs contribute to the same OOV row in this batch.
row_gradients = np.zeros_like(new_weights)
cold_ids = [30, 40, 50, 60]
upstream = np.eye(4, dtype=np.float32)
np.add.at(row_gradients, [lookup_row(raw) for raw in cold_ids], upstream)
np.testing.assert_array_equal(row_gradients[0], np.ones(4))
np.testing.assert_array_equal(row_gradients[1:], np.zeros((2, 4)))
print("Retained mapping:", new_rows, "OOV gradient:", row_gradients[0])
```

The example intentionally keeps the old arrays for comparison, so the process
does not save 192 bytes of live memory merely by running this block. It verifies
the new arrays' size and identity mapping. A migration must release obsolete
parameters and state after validation, including references held by the old
optimizer. Training resume additionally preserves the appropriate optimizer
step counters and hyperparameters. Initializing a new OOV row can require
separate step semantics or a fresh training phase; copying a global Adam step
does not give a new row the same history as retained rows.

Changing IDs to a shared row changes model outputs for those IDs. It preserves
the retained row values but does not preserve the complete model's behaviour.
Metadata such as category or textual descriptions can still distinguish two
items whose ID vectors are shared; if no other features differ, the ranker
cannot learn separate ID-specific scores for them.

#### Hash buckets and managed collisions

The original [TorchRec zero-collision tutorial](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/examples/zch/zero_collision_hash_tutorial.ipynb)
starts with a 1000-row table and raw ID 1001. Direct indexing is out of range;
mapping `id % 1000` makes the lookup valid but maps both 1 and 1001 to row 1.
Those IDs share parameters and gradient updates. Hashing bounds capacity
without needing a full vocabulary, but does not preserve independent
representations for every input ID.

The MPZCH README describes its state as follows:

> A MPZCH module contains two essential tables: the identity table and the metadata table.
> The identity table is used to record the mapping from input hash value to the remapped ID.

*Meta Platforms, Inc. and affiliates, original English excerpt from
[Managed Collision Hash Example, Introduction of MPZCH](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/examples/zch/Readme.md#introduction-of-mpzch),
[BSD-3-Clause notice](/licenses/torchrec-BSD-3-Clause.txt).*

A managed mapping stores which raw ID currently owns each available slot.
If IDs 1 and 1001 are both resident, they can occupy different slots despite
the modulo collision. A finite table still needs admission and eviction rules
when more distinct IDs arrive than it can hold. Frequency and recency are
different signals: a historically common ID may have stopped appearing, while
a new ID can be rare only because it was introduced recently.

![[00 Учебник/Assets/Figures/curated/image-id-selection-2026-10-29/torchrec-zch-dataflow.png]]

*Meta Platforms, Inc. and affiliates, [Managed Collision Hash Example](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/examples/zch/Readme.md),
[complete original diagram](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/examples/zch/docs/mpzch_module_dataflow.png),
7351×2336 pixels, unchanged; [BSD-3-Clause](/licenses/torchrec-BSD-3-Clause.txt).
Open the figure to read the function names. It depicts the MPZCH remapping
implementation, not the embedding-gradient update or a table-compaction step.*

The left input contains raw feature IDs. The identity table records the ID
owning each slot; a parallel metadata table records eviction-related state.
The central operator searches for an existing, available or evictable slot
and returns a remapped row index. In the shown TTL configuration, a score
such as `current_time + TTL` represents expiry, and the threshold represents
the current time. The right-hand path checks slot ownership and eviction;
the returned index then selects an embedding. The identity and metadata
tables are additional allocations alongside embedding weights and optimizer
state. The drawing does not imply that these allocations cost nothing.

For two slots, suppose their owners are 10 and 20 and expiry times are 5 and 14.
At time 10, the first slot is eligible for replacement. If ID 30 takes it,
the slot's old embedding and optimizer history must not be silently attributed
to ID 30. The following assigned-state example makes an explicit reset policy:

```python
import numpy as np

owners = np.array([10, 20], dtype=np.int64)
expires = np.array([5, 14], dtype=np.int64)
weights = np.array([[7., 8.], [3., 4.]], dtype=np.float32)
moment1, moment2 = np.ones_like(weights), np.full_like(weights, 2)
now, ttl, incoming = 10, 5, 30
eligible = np.flatnonzero(expires <= now)
assert eligible.tolist() == [0]
slot = int(eligible[0])
owners[slot], expires[slot] = incoming, now + ttl
for array in (weights, moment1, moment2):
    array[slot] = 0
assert owners.tolist() == [30, 20]
assert expires.tolist() == [15, 14]
np.testing.assert_array_equal(weights[1], [3., 4.])
assert all(np.all(array[0] == 0) for array in (weights, moment1, moment2))
print("New owner and expiry:", owners, expires)
```

This demonstrates ownership and state consistency; it is not TorchRec's
concurrent probing kernel or an implementation of every eviction policy.
Expiry does not by itself prove that an ID is unimportant. Evicting a returning
user discards personal history, and training/serving must agree about whether
lookups can allocate or replace slots. A versioned snapshot, a frozen serving
mapping, or synchronized managed state is needed; independently remapping
the same IDs on two workers changes their meaning.

The linked notebook also contains a naming mismatch: its prose introduces
MPZCH, but its construction cell instantiates `MCHManagedCollisionModule`.
These are separate implementations. The diagram is from the MPZCH README;
the CPU examples here use neither implementation. Likewise,
[TorchRec RFC-0002](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/rfc/RFC-0002-Flexible-Collision-Free-Embedding-Table.md)
is a proposal for flexible key/value storage and resharding, not proof that
every fixed table already shrinks on eviction. Reusing a slot in a fixed-capacity
table maintains its allocation. Actually reducing capacity is another operation.

#### Evaluate the IDs that lost their dedicated rows

Choose the frequency cutoff using training data and a validation period.
Keep the final chronological test unchanged. Label test examples by training
history: retained known ID, known-but-pruned ID, and genuinely unseen ID.
Report their counts and metrics separately, together with aggregate ranking
quality, candidate recall, OOV rate, allocated bytes and measured latency.
An ID becoming popular in the future does not justify adding its future
frequency to the historical vocabulary decision.

For an inspectable ranking counterexample, let two retained items score 2
and 1, while cold items C, D and E all score 0 after mapping to OOV. Break
ties by item name. If the relevant item is E, it ranks fifth. Removing C,
D and E from evaluation does not repair this result; it removes the question
on which the policy failed. A held-out positive absent from the candidate
set is a retrieval miss, not a row to drop from the denominator.

```python
def ranked_items(scores):
    return sorted(scores, key=lambda item: (-scores[item], item))

before = {"A": 2., "B": 1., "C": 0., "D": 0., "E": 3.}
after = {"A": 2., "B": 1., "C": 0., "D": 0., "E": 0.}
rank_before = ranked_items(before).index("E") + 1
rank_after = ranked_items(after).index("E") + 1
assert (rank_before, rank_after) == (1, 5)
hits_retained, hits_pruned = [1, 1], [0, 0]
aggregate_hit1 = sum(hits_retained + hits_pruned) / 4
assert aggregate_hit1 == 0.5
assert sum(hits_retained) / len(hits_retained) == 1.0
print("Assigned ranks:", rank_before, rank_after, "aggregate Hit@1:", aggregate_hit1)
```

The scores and four-query outcomes are assigned counterexamples, not measured
recommendation quality. They show why reporting only surviving IDs can hide
losses in the tail. Real acceptance needs the same eligible catalogue, temporal
features, negative-sampling rules, tie policy and evaluation queries before
and after the change. A memory saving is established by the actual allocated
state; an acceptable quality trade-off requires that separate comparison.
<!-- cold-ids-addition: end -->

### Where the embedding tables live

Data parallelism replicates the same parameters and partitions examples.
Replicating a 512 GB table does not solve its capacity problem. With
model parallelism, devices own different tables or different pieces of
one table. TorchRec illustrates the alternatives as follows:

![[00 Учебник/Assets/Figures/curated/dlrm-2026-10-26/torchrec-sharding.png]]

*Meta Platforms, Inc. and affiliates,
[TorchRec Concepts, Sharding of EmbeddingTables](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/docs/source/concepts.rst#sharding-of-embeddingtables),
Figure 1. Complete original PNG, BSD-3-Clause;
[full notice](/licenses/torchrec-BSD-3-Clause.txt).*

Read the two device rows vertically. Table-wise placement puts a whole
table on one device; it works only if that table and its required state
fit there. Column-wise placement splits each vector's coordinates, so
both devices are needed to reconstruct a full vector. Row-wise placement
splits the ID space; a requested ID belongs to one shard. The final
data-parallel column keeps a full copy on both devices and saves no
per-device table capacity. The illustration shows one table; a real
plan can place different tables differently.

For row-wise sum pooling, each owner sums its selected rows and the
partial sums are added. For mean pooling, averaging the owners' means
is generally wrong because the owners may receive different numbers of
IDs. For `[1,3,2,2]` and a split after row 1, the first shard contributes
one row and the second contributes three; the correct mean divides the
total sum by four. Column-wise shards instead return complementary
coordinates, which must be concatenated in their original order.

```python
dlrm_table = torch.tensor(
    [[0., 0.], [.5, 0.], [0., 1.], [.5, 1.]], dtype=dlrm_dtype)
dlrm_ids = torch.tensor([1, 3, 2, 2])
dlrm_left_ids = dlrm_ids[dlrm_ids < 2]
dlrm_right_ids = dlrm_ids[dlrm_ids >= 2] - 2
dlrm_partial0 = dlrm_table[:2][dlrm_left_ids].sum(0)
dlrm_partial1 = dlrm_table[2:][dlrm_right_ids].sum(0)
dlrm_unsharded_sum = dlrm_table[dlrm_ids].sum(0)
torch.testing.assert_close(dlrm_partial0 + dlrm_partial1, dlrm_unsharded_sum)
dlrm_wrong_mean = (
    dlrm_table[:2][dlrm_left_ids].mean(0)
    + dlrm_table[2:][dlrm_right_ids].mean(0)
) / 2
dlrm_correct_mean = (dlrm_partial0 + dlrm_partial1) / dlrm_ids.numel()
assert not torch.allclose(dlrm_wrong_mean, dlrm_correct_mean)
dlrm_column0 = dlrm_table[:, :1][dlrm_ids].sum(0)
dlrm_column1 = dlrm_table[:, 1:][dlrm_ids].sum(0)
torch.testing.assert_close(
    torch.cat([dlrm_column0, dlrm_column1]), dlrm_unsharded_sum)
print("Row shards, sum/mean:", dlrm_unsharded_sum, dlrm_correct_mean)
print("Incorrect mean of shard means:", dlrm_wrong_mean)
```

These are CPU tensor operations that verify reconstruction, not actual
GPU collectives. Choosing a placement also changes communication.
The TorchRec documentation describes the exchange explicitly:

<!-- source-excerpt: torchrec-communication -->
> With sharding, there is an added communication cost: each GPU needs to
> ask the other GPUs for embedding vector lookup and communicate the
> gradients computed as well. This is typically referred to as `all2all`
> communication. In TorchRec, for input data on a given GPU, we determine
> where the embedding shard for each part of the data is located and send
> it to the target GPU. That target GPU then returns the embedding vectors
> back to the original GPU. In the backward pass, the gradients are sent
> back to the target GPU and the shards are updated accordingly with the
> optimizer.
<!-- /source-excerpt -->

*Meta Platforms, Inc. and affiliates,
[TorchRec Concepts, Distributed Training with TorchRec Sharded Modules](https://github.com/meta-pytorch/torchrec/blob/36ae8ed89ac78e34a18ab00a08b93afe19410b55/docs/source/concepts.rst#distributed-training-with-torchrec-sharded-modules).
Original English excerpt, BSD-3-Clause; [full notice](/licenses/torchrec-BSD-3-Clause.txt).*

![[00 Учебник/Assets/Figures/curated/dlrm-2026-10-26/torchrec-forward.png]]

*Meta Platforms, Inc. and affiliates, same source, Figure 2:
forward pass of table-wise sharded embeddings. Complete original PNG,
BSD-3-Clause; [full notice](/licenses/torchrec-BSD-3-Clause.txt).*

Follow `feature_0_ids` down the diagram. Trainer 0 owns table `emb_t0`,
but both trainers have examples needing that table. The first exchange,
`input_dist`, brings their IDs to its owner. After lookup and pooling,
`output_dist` returns the vectors to the trainers owning the examples.
Each trainer then has all fields for its local examples and can run
interactions and the dense module. The backward pass returns embedding
gradients to the table owners. Replicated dense parameters require
their own gradient synchronization; the exchange of embeddings is not
that synchronization.

In this table-wise example a bag can be pooled at one owner before its
vector is returned. Row-wise placement can need a reduction of partial
bags, as in the CPU example; column-wise placement has to assemble
coordinates. A sharding plan therefore depends on table sizes, ID and
bag-length distributions, available memory, and the interconnect, not
only on the number of parameters. The original DLRM paper describes
hybrid embedding-model/dense-data parallelism in
[§3](https://arxiv.org/html/1906.00091v1#S3); the pinned TorchRec
documentation gives the concrete placement and exchange mechanisms.
<!-- /dlrm -->

<a id="сквозной-training-protocol-для-ranker"></a>
## Обучение и проверка ранжирующей модели

Журнал показов должен позволять восстановить запрос, пользователя,
объект, время, позицию, версию алгоритма, доступные признаки и отклик.
Положительная и отрицательная метки определяются после завершения
окна атрибуции. Для неизвестных пар отдельно описывается sampling.

До обучения фиксируют временную границу, каталог, словари и признаки.
BPR получает тройки из разрешённых на этой границе данных; BCE —
строки с явно определёнными метками; групповая функция потерь —
кандидатов одного запроса. Пользователь из будущего теста не должен
попасть в обучающий словарь только потому, что файл теста уже прочитан.

После обучения одинаковый протокол применяют к исходной модели
и новым вариантам. В него входят полный или sampled каталог,
правило удаления просмотренных объектов, размер $K$ и обработка ties.
Для NCF и MF выравнивают не только данные, но и бюджет подбора,
число negatives и допустимое время ответа. Калибровку проверяют
на откликах из подходящего распределения показов, а не заключают
из наличия sigmoid.

<a id="failure-modes"></a>
## Смещения в данных и ограничения выдачи

Позиция в старой выдаче влияет на вероятность клика. Если её просто
добавить в признаки, модель может хорошо предсказывать отклики старого
алгоритма, но это не устанавливает, что произойдёт после перестановки
объектов. Нужны отдельно сформулированные предположения и способ
проверки нового алгоритма; BPR и MLP сами по себе этого не решают.

Признак `item_ctr_7d`, посчитанный с откликами после запроса, раскрывает
часть target. Граница обучающего файла не исправит утечку, если сами
признаки построены с использованием будущего. Восстановление признака
на момент решения необходимо и для offline-оценки, и для обучения.

Агрегированная метрика может улучшиться за счёт популярных объектов,
оставив новые и редкие без улучшения. Поэтому сравнивают отдельные
группы по длине истории, популярности и наличию новых ID.
Отдельно измеряют recall поиска кандидатов: последующее ранжирование
не вернёт объект, отсутствующий среди кандидатов.

<a id="что-должно-быть-в-отчёте"></a>
В отчёте сохраняют snapshot и временную границу, смысл меток и окно
атрибуции, распределение negatives, архитектуру и число параметров,
функцию потерь и регуляризацию, протокол каталога и ties, метрики по
группам и время ответа на реальном числе кандидатов.

## Практика и первоисточники

- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/ranking|D2L: Personalized Ranking]] — полный английский вывод BPR и hinge с MXNet-кодом.
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/autorec|D2L: AutoRec]] — encoder/decoder, маска, evaluator и полный training example.
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/neumf|D2L: NeuMF]] — архитектура, sampler и pairwise training; отмеченное различие sigmoid/BPR сохраняется в оригинале.
- [[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb|Recommenders: NCF deep dive]] — оригинальная теория, реализация, два evaluation protocol и pre-training.
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/fm|D2L: FM]] — вывод вычисления взаимодействий и реализация.
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/deepfm|D2L: DeepFM]] — общие embeddings и две ветви.
- [Rendle et al., BPR](https://arxiv.org/abs/1205.2618) и [He et al., NCF](https://arxiv.org/abs/1708.05031) — первичные статьи.

Дальше: [[00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки|история взаимодействий и последовательные модели]].
