---
title: Ranking и нейронные рекомендательные модели
type: textbook-chapter
status: canonical
last_updated: 2026-07-26
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
    negative = sampler.sample(user)
    s_pos = model(user, positive)
    s_neg = model(user, negative)
    loss = -logsigmoid(s_pos - s_neg)
    loss.backward()
    optimizer.step()
```

### BPR не превращает unseen в настоящий dislike

Математика предполагает, что observed positive предпочтительнее sampled
unseen. Это удобный training signal, а не установленный факт. Если $j$ не был
показан, система не знает, что выбрал бы пользователь. Hard-negative sampler
может даже выбирать привлекательные, но ранее не exposed items и тем самым
создавать false negatives.

Поэтому фиксируют sampler, число negatives, исключение будущих positives и
candidate universe. Иначе изменение sampler легко принять за улучшение модели.

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
