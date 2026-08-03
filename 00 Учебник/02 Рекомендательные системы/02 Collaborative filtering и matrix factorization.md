---
title: Collaborative filtering и matrix factorization
type: textbook-chapter
status: canonical
last_updated: 2026-07-26
---

# Collaborative filtering и matrix factorization

Collaborative filtering строит рекомендации по совместным паттернам поведения:
если два пользователя выбирали похожие объекты, их следующие предпочтения тоже
могут оказаться похожими. Описание фильма, товара или песни при этом не
обязательно. Модель получает разреженную матрицу взаимодействий и восстанавливает
в ней структуру.

## Матрица взаимодействий

Пусть $R\in\mathbb R^{m\times n}$: строки соответствуют пользователям, столбцы
— объектам. Для explicit feedback известные элементы содержат ratings. Для
implicit feedback в матрицу можно положить strength события: например,
`view = 1`, `add_to_cart = 3`, `purchase = 5`. Большинство элементов не
наблюдается.

| | Item A | Item B | Item C | Item D |
|---|---:|---:|---:|---:|
| User 1 | 5 | ? | 1 | ? |
| User 2 | 4 | 5 | ? | ? |
| User 3 | ? | 1 | 5 | 4 |

Знак `?` принципиален. Он не означает нулевую оценку и не доказывает
неинтересность: пользователь мог не увидеть объект. Поэтому regression loss для
ratings считают только на множестве наблюдаемых пар
$\mathcal K=\{(u,i):r_{ui}\text{ is observed}\}$.

## Три baseline до factorization

Сложную модель следует сравнивать как минимум с тремя простыми правилами.

**Global mean**

$$
\hat r_{ui}=\mu
$$

показывает, сколько качества даёт одно знание масштаба target.

**User и item biases**

$$
\hat r_{ui}=\mu+b_u+b_i
$$

учитывают, что один пользователь ставит почти всем фильмам высокие оценки, а
другой — низкие; один фильм нравится почти всем, другой систематически
оценивается хуже. Biases часто объясняют неожиданно большую часть качества.

**Popularity baseline** сортирует items по числу или взвешенной сумме train
events. В LF notebook рейтинг каждого объекта для top-$K$ baseline вычисляется
только по train, затем уже виденные пользователем items исключаются. Такой
baseline особенно важен: случайный split и популярностный перекос нередко
позволяют ему выглядеть почти так же хорошо, как персонализированной модели.

Если новая модель не превосходит эти baselines на том же split, candidate set и
evaluation protocol, обсуждать её архитектуру рано.

## Memory-based collaborative filtering

Memory-based методы хранят взаимодействия и явно считают сходство. В
item-based варианте:

1. для каждой пары items находят пользователей, взаимодействовавших с обоими;
2. строят similarity $S_{ij}$;
3. для пользователя суммируют similarities к уже выбранным им items;
4. исключают seen и берут top-$K$.

Именно так устроен Simple Algorithm for Recommendation (SAR) из практического
курса Linux Foundation. Пусть $C_{ij}$ — число совместных появлений items
$i$ и $j$. Notebook рассматривает три способа превратить co-occurrence в
similarity:

$$
S_{ij}^{\text{counts}}=C_{ij},
$$

$$
S_{ij}^{\text{Jaccard}}
=\frac{C_{ij}}{C_{ii}+C_{jj}-C_{ij}},
\qquad
S_{ij}^{\text{lift}}
=\frac{C_{ij}}{C_{ii}C_{jj}}.
$$

`counts` тяготеет к предсказуемым популярным объектам, `lift` сильнее
поднимает редкие устойчивые совместные появления, Jaccard занимает
промежуточное положение.

Affinity пользователя к item может учитывать тип и давность событий:

$$
A_{ui}=\sum_{k\in E(u,i)}
w_k\,2^{-(t_0-t_k)/T},
$$

где $w_k$ — вес действия, а $T$ — период полураспада. Итоговая матрица scores:

$$
\widehat R=AS.
$$

### Worked example SAR

Пусть пользователь купил `camera` с весом $5$ и просмотрел `tripod` с весом
$1$. Для кандидата `lens`

$$
s(u,\text{lens})
=5S_{\text{camera,lens}}+1S_{\text{tripod,lens}}.
$$

Если similarities равны $0.8$ и $0.3$, score равен $4.3$. Для кандидата
`bag` при similarities $0.2$ и $0.7$ score равен $1.7$: первым будет `lens`.
Через месяц вклад старой покупки уменьшится вдвое, если $T$ выбран равным
месяцу. Здесь можно объяснить каждый score — полезное свойство baseline.

Memory-based методы прозрачны, но similarity matrix становится дорогой при
большом каталоге, а редкие объекты почти не имеют co-occurrences. Поэтому
следующий шаг — параметрическая low-rank модель.

## Matrix factorization

Matrix factorization сопоставляет пользователю вектор
$p_u\in\mathbb R^d$, объекту — $q_i\in\mathbb R^d$:

$$
\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i.
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-mf.svg]]

Источник рисунка и полная реализация: [D2L — Matrix
Factorization](https://d2l.ai/chapter_recommender-systems/mf.html).

В терминах всей матрицы произведение $PQ^\top$ имеет ранг не выше $d$.
Координаты latent space не обязаны совпадать с жанрами или категориями,
которые придумал человек. Они выбираются так, чтобы совместно объяснить
наблюдаемые ratings. Одна координата может смешивать жанр, популярность,
возраст аудитории и свойства самого датасета.

Для explicit feedback оптимизируют

$$
\mathcal L=
\sum_{(u,i)\in\mathcal K}
\left(r_{ui}-\mu-b_u-b_i-p_u^\top q_i\right)^2
+\lambda\sum_{(u,i)\in\mathcal K}
\left(\lVert p_u\rVert_2^2+\lVert q_i\rVert_2^2+b_u^2+b_i^2\right).
$$

Регуляризация особенно важна для редких IDs: без неё embedding пользователя с
одной оценкой может запомнить единственный пример.

### Один шаг SGD

Обозначим ошибку

$$
e_{ui}=r_{ui}-\hat r_{ui}.
$$

Для loss $\tfrac12e_{ui}^2+\tfrac\lambda2(\lVert p_u\rVert^2+\lVert q_i\rVert^2)$
обновления имеют вид

$$
p_u\leftarrow p_u+\eta(e_{ui}q_i-\lambda p_u),
\qquad
q_i\leftarrow q_i+\eta(e_{ui}p_u-\lambda q_i).
$$

Если модель занизила rating, $e_{ui}>0$: user и item vectors сближаются в
направлениях друг друга. Если завысила, они расходятся.

Пусть $p_u=(0.2,0.4)$, $q_i=(0.5,0.1)$, biases равны нулю, а истинный rating
$r_{ui}=1$. Предсказание равно $0.14$, ошибка $0.86$. При $\eta=0.1$ и без
регуляризации:

$$
p_u'=(0.2,0.4)+0.1\cdot0.86(0.5,0.1)
=(0.243,0.4086).
$$

Такой расчёт полезнее абстрактной фразы «embeddings обучаются
backpropagation»: видно, какая наблюдаемая пара меняет какие строки таблиц.

```python
for user, item, rating in train_loader:
    pu = user_embeddings[user]
    qi = item_embeddings[item]
    prediction = global_mean + user_bias[user] + item_bias[item]
    prediction += (pu * qi).sum(-1)
    loss = mse(prediction, rating) + regularization(...)
    loss.backward()
    optimizer.step()
```

В batch размером $B$ lookup возвращает два тензора формы $B\times d$.
Поэлементное произведение и сумма дают $B$ scores. Арифметика дёшева, но в
крупной системе случайные обращения к огромным embedding tables, их
распределённое хранение и синхронизация могут стоить больше самой формулы.

## Implicit feedback: ноль состоит из разных причин

Для implicit data бинарная матрица обычно содержит positives и ненаблюдаемые
пары. Есть три распространённых учебных постановки.

**Pointwise classification:** $(u,i)$ получает target 1, sampled unseen item —
0; обучается binary cross-entropy.

**Weighted matrix factorization:** всем парам назначают preference $p_{ui}$ и
confidence $c_{ui}$; наблюдаемые события имеют больший вес. Это позволяет
учитывать всю матрицу, но не утверждает, что каждый ноль — достоверный dislike.

**Pairwise ranking:** для triple $(u,i,j)$ positive $i$ должен оказаться выше
sampled $j$. Этому посвящена следующая глава.

Negative sampler — часть модели эксперимента:

| Sampler | Что даёт | Риск |
|---|---|---|
| uniform | простые и дешёвые negatives | большинство слишком лёгкие |
| popularity | чаще встречаются правдоподобные head items | усиливает head bias |
| in-batch | почти бесплатное переиспользование batch | false negatives, зависимость от batch |
| hard-negative | учит различать похожие candidates | ошибки sampler становятся сильным bias |
| exposed-only | ближе к наблюдаемому выбору | нужны корректные impression logs |

Сэмплировать test negatives из будущего каталога или использовать test
frequencies при popularity sampling — утечка.

## От обучения к выдаче

Для rating prediction MF достаточно вычислить score заданной пары. Для
top-$K$ нужны candidates. Если scoring — скалярное произведение, item vectors
можно поместить в индекс maximum inner product search и быстро получить top
candidates. Biases и дополнительные фильтры применяются после retrieval либо
включаются в индексируемое представление.

Минимальный serving protocol:

```text
load versioned user/item embeddings
retrieve top N by dot product
remove seen, unavailable and policy-forbidden items
apply optional feature-rich ranker
return K items
log candidate set, raw scores, final positions and model version
```

Без mappings `external_id -> embedding row` checkpoint бесполезен. При
обновлении каталога mappings и embeddings должны быть версионированы вместе.

## Где модель ломается

**Cold start.** Для нового ID нет обученного вектора. Нужны content features,
popularity fallback, onboarding или отдельный cold-start model.

**Popularity amplification.** Head items получают больше событий, лучше
обученные vectors и ещё больше exposure. Aggregate Recall@$K$ может расти,
пока catalog coverage падает.

**Sparse users.** Один общий score скрывает разницу между пользователями с
двумя и двумя тысячами событий. Нужны activity slices.

**Temporal drift.** Статические vectors смешивают давние и недавние интересы.
Time decay или последовательная модель могут помочь, но их проверяют на
temporal split.

**Exposure bias.** Наблюдаем только реакцию на прежнюю выдачу. MF не
восстанавливает counterfactual response для непоказанных объектов.

**Offline protocol mismatch.** Ranking по 99 случайным negatives и ranking по
миллионному каталогу — разные задачи.

## Контрольный эксперимент

В честном отчёте одна таблица содержит:

1. global mean, popularity и bias baseline;
2. item-based SAR;
3. MF для нескольких $d$ и $\lambda$;
4. один temporal split и неизменный candidate set;
5. RMSE для rating task либо Recall/NDCG@$K$ для top-$K$;
6. latency, размер параметров, head/tail и activity slices;
7. unknown-user/item rate и catalog coverage.

Только после этого можно утверждать, что латентные факторы добавили полезный
сигнал, а не воспользовались более лёгким split или sampler.

## Первоисточники и продолжение

- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/mf|D2L: Matrix Factorization]] — полный математический вывод, код, обучение и упражнения;
- [[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb|LF: baseline deep dive]] — rating и top-$K$ baselines;
- [[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/sar_deep_dive.ipynb|LF: SAR deep dive]] — co-occurrence, similarity, time decay и evaluation;
- [[00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели|следующая глава]] — BPR, NCF, AutoRec, FM и DeepFM.
