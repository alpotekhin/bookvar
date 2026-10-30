---
title: Ranking и нейронные рекомендательные модели
type: textbook-chapter
status: canonical
last_updated: 2026-09-15
---

# Ranking и нейронные рекомендательные модели

Rating prediction спрашивает: «какую численную оценку поставит пользователь?»
Ranking — «какой из доступных объектов должен стоять выше?». Эти задачи связаны,
но не совпадают. Модель с меньшим RMSE может хуже упорядочивать первые десять
items, а хорошо откалиброванный CTR predictor не обязательно создаёт лучший
slate после фильтров и diversity re-ranking.

## Pointwise, pairwise и listwise постановки

**Pointwise** loss рассматривает каждую пару отдельно:

$$
\mathcal L_{\text{BCE}}
=-\sum_{(u,i)}
\left[y_{ui}\log\sigma(s_{ui})
+(1-y_{ui})\log(1-\sigma(s_{ui}))\right].
$$

Он прост, допускает calibration, но не выражает непосредственно требование
«positive должен быть выше negative».

**Pairwise** loss получает $(u,i,j)$ и сравнивает два scores. Он ближе к
относительному порядку, но результат зависит от выбора $j$.

**Listwise** методы работают со slate или аппроксимацией list metric. Они ближе
к конечному интерфейсу, но дороже и сложнее. Нельзя автоматически объявить один
класс лучшим: objective выбирают по продуктовой задаче и доступным логам.

## Bayesian Personalized Ranking

Пусть $I_u^+$ — наблюдаемые positives пользователя, а
$j\in I\setminus I_u^+$. Training set:

$$
D=\{(u,i,j)\mid i\in I_u^+,\ j\notin I_u^+\}.
$$

BPR максимизирует вероятность правильного попарного порядка:

$$
\max_\Theta
\sum_{(u,i,j)\in D}
\log\sigma\bigl(s(u,i)-s(u,j)\bigr)
-\lambda\lVert\Theta\rVert_2^2.
$$

При minimization знак меняется:

$$
\mathcal L_{\mathrm{BPR}}
=-\log\sigma(\Delta_{uij}),
\qquad
\Delta_{uij}=s(u,i)-s(u,j).
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-ranking.svg]]

Источник рисунка и вывод функции потерь: [D2L — Personalized Ranking for
Recommender Systems](https://d2l.ai/chapter_recommender-systems/ranking.html).

### Как loss реагирует на margin

| $s(u,i)$ | $s(u,j)$ | $\Delta$ | $-\log\sigma(\Delta)$ |
|---:|---:|---:|---:|
| 0.0 | 2.0 | -2 | 2.127 |
| 1.0 | 1.0 | 0 | 0.693 |
| 2.0 | 0.0 | 2 | 0.127 |

Если negative стоит выше, градиент велик. При правильном большом отрыве loss
мал, но не становится строго нулевым.

Для MF $s(u,i)=p_u^\top q_i$:

$$
\Delta_{uij}=p_u^\top(q_i-q_j).
$$

Один triple одновременно притягивает $p_u$ к $q_i$ и отталкивает от $q_j$.
Упрощённый цикл:

```python
for user, positive in observed_events:
    optimizer.zero_grad()
    negative = sampler.sample(user)
    s_pos = model(user, positive)
    s_neg = model(user, negative)
    loss = -logsigmoid(s_pos - s_neg).mean()
    loss.backward()
    optimizer.step()
```

Здесь `s_pos` и `s_neg` имеют одинаковую форму $[B]$; `.mean()` даёт
скалярный loss по $B$ triples. Определения модели, sampler и `logsigmoid`
в этом псевдокоде опущены. Проверим градиент на двух независимых обучаемых
scores $s_+=0$, $s_-=2$: $\Delta=-2$ и
$\partial\ell/\partial\Delta=\sigma(\Delta)-1\approx-0{,}880797$.
Производные по scores равны соответственно $-0{,}880797$ и $+0{,}880797$.
Одновременный шаг со скоростью 0,1 даёт $s_+'=0{,}088080$,
$s_-'=1{,}911920$ и $\Delta'=-1{,}823841$: неверный порядок ещё
не исправлен, но отрыв сократился. Для MF эти производные затем проходят
через скалярные произведения к общим embedding-параметрам.

### BPR не превращает unseen в настоящий dislike

Математика предполагает, что observed positive предпочтительнее sampled
unseen. Это удобный training signal, а не установленный факт. Если $j$ не был
показан, система не знает, что выбрал бы пользователь. Hard-negative sampler
может даже выбирать привлекательные, но ранее не exposed items и тем самым
создавать false negatives.

Поэтому фиксируют sampler, число negatives, candidate universe и временную
границу доступного feedback. В обычном offline BPR из negative pool исключают
positives **обучающего snapshot**, а не будущие positives из validation/test.
Использовать будущую покупку для выбора сегодняшнего negative — уже утечка,
даже если сама покупка не попала в loss.

Минимальный контракт множества, из которого затем семплируют:

```python
def training_negative_pool(catalog_at_train_cutoff, train_positive_ids):
    return set(catalog_at_train_cutoff) - set(train_positive_ids)
```

Например, на границе обучения доступны `{A, B, C}`, а известен только positive
`A`. Pool равен `{B, C}`. Если после границы пользователь купит `B`, это не даёт
права задним числом заменить pool на `{C}`. Возможность такого false negative —
ограничение implicit feedback, а не повод читать held-out labels. Для streaming
или point-in-time обучения граница задаётся отдельно на каждом шаге: доступна
только история к этому моменту. Правило исключения observed positives следует
из определения обучающего множества в [BPR, §3–4](https://arxiv.org/abs/1205.2618);
важность глобальной временной границы исследована в [Ji et al., A Critical Study
on Data Leakage](https://arxiv.org/abs/2010.11060).

## Hinge ranking loss

$$
\mathcal L_{\text{hinge}}
=\max\bigl(0,m-s(u,i)+s(u,j)\bigr).
$$

При margin $m=1$, $s(u,i)=1.4$, $s(u,j)=0.8$ loss равен $0.4$: порядок
правильный, но отрыв недостаточен. При $s(u,i)=2$, $s(u,j)=0.5$ loss равен
нулю. В отличие от BPR, hinge перестаёт тратить градиент после выполнения
margin.

## AutoRec: восстановление строки матрицы

С этого места меняется вопрос: до сих пор выбиралась **цель ranking**,
дальше — **архитектура score или реконструкции**. Эти решения независимы:
нейронный score можно обучать BPR либо pointwise BCE. Для первого прохода
достаточно objectives выше и training protocol в конце; подробный разбор
ветвей требует [[00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|MLP]]
и [[00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|autoencoder]].

AutoRec применяет autoencoder к вектору ratings одного пользователя
(user-based) или одного item (item-based). Для item-based AutoRec вход
$r^{(i)}\in\mathbb R^m$ содержит ratings item $i$ от всех пользователей:

$$
h=\phi(Wr^{(i)}+b),
\qquad
\hat r^{(i)}=W'h+b'.
$$

Loss считают только на наблюдаемых координатах:

$$
\mathcal L_i=
\lVert M^{(i)}\odot(r^{(i)}-\hat r^{(i)})\rVert_2^2
+\lambda\lVert\Theta\rVert_2^2.
$$

Нелинейный hidden layer способен выразить больше, чем low-rank dot product.
Но dense input размером с число users или items неудобен для огромного,
постоянно меняющегося каталога. Архитектура также не решает exposure bias и
cold start сама по себе.

Возьмём один item с ratings трёх пользователей $(5,?,1)$.
Вход после технического заполнения пропуска нулём — $(5,0,1)^\top$,
но маска наблюдений $M=(1,0,1)^\top$ хранится отдельно. Один ReLU-нейрон
с весами $(0{,}2,0,0{,}2)$ и bias 0 выдаёт $h=1{,}2$.
Линейный decoder с весами $(2,3,1)^\top$ и нулевыми biases даёт
$\hat r=(2{,}4,3{,}6,1{,}2)^\top$. Без регуляризации сумма squared
errors по наблюдаемым координатам равна
$(5-2{,}4)^2+(1-1{,}2)^2=6{,}8$.
Производная по decoder output равна $(-5{,}2,0,0{,}4)^\top$:
прогноз 3,6 на пропуске не штрафуется, как если бы настоящий rating был 0.
Это реконструкция наблюдаемых ratings, а не автоматически calibrated CTR.

## Neural Collaborative Filtering

NeuMF объединяет две ветви.

**Generalized Matrix Factorization**

$$
x_{\mathrm{GMF}}=p_u\odot q_i.
$$

**MLP**

$$
z_0=[U_u;V_i],
\qquad
z_{\ell+1}=\phi(W_\ell z_\ell+b_\ell).
$$

Итог:

$$
\hat y_{ui}
=\sigma\left(h^\top[x_{\mathrm{GMF}};z_L]\right).
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-neumf.svg]]

*Архитектура NeuMF из главы [Neural Collaborative Filtering for Personalized
Ranking](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recommender-systems/neumf.md)
книги Dive into Deep Learning, CC BY-SA 4.0; исходный SVG не изменён.*

В исходной архитектуре GMF и MLP могут иметь разные embedding tables: одной
ветви не приходится одновременно служить multiplicative и nonlinear
representation. Fusion происходит перед prediction layer.

### Формы тензоров

Для batch $B$, embedding dimension $d$ и последнего MLP layer размером $h$:

| Тензор | Форма |
|---|---|
| $p_u,q_i,U_u,V_i$ | $B\times d$ |
| $p_u\odot q_i$ | $B\times d$ |
| $[U_u;V_i]$ | $B\times2d$ |
| $z_L$ | $B\times h$ |
| fusion | $B\times(d+h)$ |
| output | $B\times1$ |

Эта таблица позволяет проверить реализацию и не спутать elementwise product
GMF с dot product классического MF.

### Почему comparison часто нечестен

NeuMF имеет больше параметров, сложнее tuning и может использовать другую
negative sampling procedure. Сравнение с MF корректно только при одинаковых:

- temporal split и train events;
- candidate universe и negatives;
- tuning budget;
- evaluation mode: full-catalog или sampled;
- features и post-processing;
- latency budget.

LF notebook отдельно показывает generic evaluation, leave-one-out evaluation и
pretraining GMF/MLP. Эти разделы нужны не для копирования итоговой цифры, а для
понимания того, сколько решений окружает саму архитектуру.

## Factorization Machines: interactions между полями

ID-only CF плохо переносится на новые items и не использует category, device,
country или campaign. Для sparse feature vector $x\in\mathbb R^d$
factorization machine задаёт

$$
\hat y(x)=w_0+\sum_{i=1}^d w_ix_i+
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j.
$$

Если активны признаки `user=42`, `item=7`, `device=mobile`, модель складывает
linear weights и три pairwise interactions:

$$
\langle v_{\text{user42}},v_{\text{item7}}\rangle,
\quad
\langle v_{\text{user42}},v_{\text{mobile}}\rangle,
\quad
\langle v_{\text{item7}},v_{\text{mobile}}\rangle.
$$

Редкая конкретная пара делит статистическую силу с другими парами через
embedding каждого признака.

Наивный расчёт interactions стоит $O(kd^2)$. D2L выводит преобразование

$$
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
=\frac12\sum_{\ell=1}^k
\left[
\left(\sum_i v_{i\ell}x_i\right)^2
-\sum_i v_{i\ell}^2x_i^2
\right],
$$

которое стоит $O(kd)$, а для sparse input — $O(k\,\mathrm{nnz}(x))$.

## DeepFM: low-order и high-order interactions

DeepFM использует общие field embeddings в двух параллельных ветвях:

1. FM моделирует linear и second-order interactions;
2. MLP получает concatenation embeddings и моделирует higher-order nonlinear
   interactions;
3. logits складываются перед sigmoid.

$$
z^{(0)}=[e_1;e_2;\ldots;e_f],
\qquad
z^{(\ell)}=\phi(W^{(\ell)}z^{(\ell-1)}+b^{(\ell)}),
$$

$$
\hat y=\sigma\left(
\hat y^{(\mathrm{FM})}+\hat y^{(\mathrm{DNN})}
\right).
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-deepfm.svg]]

Источник: [D2L — Deep Factorization
Machines](https://d2l.ai/chapter_recommender-systems/deepfm.html).

DeepFM естественен для feature-rich CTR ranking, но не для scoring миллионов
items на запрос. Обычно retrieval сначала сокращает каталог, затем ranker
вычисляет features для сотен candidates.

Продолжим пример трёх активных полей `user=42`, `item=7`, `device=mobile`.
Пусть их общие двумерные embeddings равны $(1,0)$, $(0{,}5,1)$,
$(0,1)$. Три pairwise dot products FM равны $0{,}5$, $0$ и $1$.
При суммарном linear term 0,2 FM-logit равен 1,7. DNN получает
вектор $(1,0,0{,}5,1,0,1)$ длины 6. Для наглядности возьмём один
ReLU-нейрон с весами $(1,0,-1,0,0,0)$ и нулевым bias:
он выдаёт $\max(0,1-0{,}5)=0{,}5$. Выходной вес 0,4 даёт
DNN-logit 0,2. Складываются logits: $1{,}7+0{,}2=1{,}9$,
и лишь затем получается $\sigma(1{,}9)\approx0{,}8699$.
Сумма двух отдельных sigmoid была бы другой моделью и могла бы превысить 1.

В этом искусственном расчёте один и тот же item embedding одновременно
влияет на interactions FM и на вход DNN; градиенты обеих ветвей суммируются.
Числа демонстрируют формы и fusion, а не качество обученного CTR predictor.

## Сквозной training protocol для ranker

Пусть impression log содержит `request_id`, `user_id`, `item_id`, `position`,
`policy`, `timestamp`, features и response.

1. Выбираем временную границу и замораживаем train snapshot.
2. Все vocabularies, popularity statistics и transforms fit только на train.
3. Positive определяем через окно атрибуции.
4. Negatives берём из реально exposed non-clicked items либо документируем
   sampler.
5. Для BPR строим triples; для BCE — pointwise rows.
6. Модель валидируем на более позднем времени.
7. Из test candidates исключаем train-seen только если production contract тоже
   запрещает повторы.
8. Считаем ranking, calibration и slice metrics.
9. Измеряем latency на настоящем числе candidates и с настоящими features.

```python
for impression in train_impressions:
    positives = attributed_actions(impression)
    negatives = exposed_without_action(impression)
    batch = make_pairs_or_triples(positives, negatives)
    optimize(batch)
```

Использование только clicked rows уничтожает определение отрицательного
сигнала. Использование произвольных unseen rows без описания sampler делает
эксперимент невоспроизводимым.

## Failure modes

**Position bias.** Верхние позиции кликают чаще независимо от relevance.
Добавить position как feature недостаточно для causal correction: position
назначена прежней policy.

**Feature leakage.** `item_ctr_7d`, посчитанный с событиями после impression,
неявно раскрывает target.

**False negatives.** Пользователь может не кликнуть из-за position, усталости
или потому, что уже знает item.

**Calibration drift.** Pairwise loss улучшает порядок, но score не обязан быть
вероятностью. Если downstream использует threshold или ожидаемую ценность,
calibration проверяется отдельно.

**Head domination.** Easy aggregate gains могут происходить только на
популярных items.

**Architecture without pipeline gain.** Более сильный ranker бесполезен, если
нужный item не попал в candidates. Поэтому отдельно измеряют candidate recall.

## Что должно быть в отчёте

| Область | Что зафиксировать |
|---|---|
| данные | snapshot, timestamp boundary, feedback, attribution |
| negatives | источник, количество, distribution, false-negative policy |
| модель | параметры, loss, features, regularization |
| evaluation | full/sampled catalog, $K$, seen-item rule |
| качество | Recall/NDCG/MAP, calibration, head/tail/cold slices |
| система | candidates per request, p50/p95 latency, memory |
| продукт | post-ranking, guardrails, online experiment |

Тогда название архитектуры перестаёт заменять описание эксперимента.

## Практика и первоисточники

- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/ranking|D2L: BPR and hinge loss]];
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/neumf|D2L: NeuMF]];
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/autorec|D2L: AutoRec]];
- [[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb|LF: NCF deep dive]];
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/fm|D2L: FM]];
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/deepfm|D2L: DeepFM]];
- [[00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки|следующая глава]] — время, sequence models, CTR и cold start.
