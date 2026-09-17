---
title: Последовательные рекомендации и признаки
type: textbook-chapter
status: canonical
last_updated: 2026-07-26
---

# Последовательные рекомендации и признаки

Агрегированная user–item matrix отвечает на вопрос, с чем пользователь
взаимодействовал, но стирает порядок. Для повторяющихся долгосрочных вкусов это
может быть приемлемо. Для новостей, музыки, коротких видео, поиска и session
commerce текущее намерение часто определяется последними несколькими действиями.

При первом чтении раннего модуля достаточно постановки и построения окон,
затем можно перейти к разделам о признаках, cold start и production contract.
Архитектурный маршрут проходит позже:
[[00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|embedding lookup]] →
[[00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|свёртка и pooling]] → Caser;
[[00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|self-attention]] и
[[00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|causal/padding masks]] → SASRec.
Эти два углублённых раздела ниже не предполагаются известными до знакомства
с нейросетями и Transformer.

## Как матрица теряет задачу

Последовательности

$$
[\text{camera},\text{lens},\text{tripod}]
$$

и

$$
[\text{tripod},\text{lens},\text{camera}]
$$

дают одинаковый набор items. Однако после покупки камеры вероятен интерес к
карте памяти, а после нескольких просмотров камер — намерение ещё может быть
исследовательским. Временной порядок, расстояние между событиями, тип действия
и session boundary несут смысл.

Sequence recommendation строит примеры

$$
(i_1,\ldots,i_t,c_t)\longrightarrow i_{t+1},
$$

где $c_t$ содержит context. Длинная история превращается во множество
prefix-target pairs.

## Сначала split, потом окна

Пусть история одного пользователя:

| Время | Item |
|---|---|
| $t_1$ | A |
| $t_2$ | B |
| $t_3$ | C |
| $t_4$ | D |
| $t_5$ | E |
| $t_6$ | F |

При leave-one-out `F` остаётся test target, `E` — validation target. Если
максимальная длина контекста $L=3$, train examples:

$$
[A]\to B,\quad[A,B]\to C,\quad[A,B,C]\to D.
$$

Validation получает $[B,C,D]\to E$, test — $[C,D,E]\to F$. Сначала проводится
временная граница, затем строятся окна. Если сначала создать все prefixes, а
потом случайно разделить их, почти одинаковые пересекающиеся истории попадут в
train и test.

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-seq-data.svg]]

Рисунок и полная реализация скользящих окон: [D2L — Sequence-Aware Recommender
Systems](https://d2l.ai/chapter_recommender-systems/seqrec.html).

```python
history = sort_by_time(events[user])
train_history = history[:-2]
valid_target = history[-2]
test_target = history[-1]

for t in range(1, len(train_history)):
    context = train_history[max(0, t - max_len):t]
    positive = train_history[t]
    negative = sample_from_catalog_at_time(user, history[t].timestamp)
    yield context, positive, negative
```

`catalog_at_time` важен: объект, появившийся после target timestamp, не должен
служить candidate или negative.

## Caser: свёртки по истории

D2L использует Caser как наглядный переход от unordered CF к sequence model.
Последние $L$ item embeddings образуют матрицу

$$
E^{(u,t)}
=
\begin{bmatrix}
q_{S^u_{t-L}}^\top\\
\vdots\\
q_{S^u_{t-1}}^\top
\end{bmatrix}
\in\mathbb R^{L\times k}.
$$

Horizontal filters высоты $h$ охватывают несколько соседних items и выделяют
union-level patterns: например, совместное появление `milk` и `butter` может
предсказывать `flour`. Vertical filters формы $L\times1$ агрегируют evolution
отдельных embedding coordinates. После pooling:

$$
o=\operatorname{HConv}(E,F),
\qquad
o'=\operatorname{VConv}(E,G),
$$

$$
z=\phi(W[o;o']+b).
$$

Уточним, что pooling относится к горизонтальной ветви: фильтр $h\times k$
при stride 1 и без padding возвращает $L-h+1$ чисел; после активации
максимум берётся по этой временной оси. Каждый горизонтальный фильтр даёт
один скаляр. Вертикальный фильтр $L\times1$ сворачивает всю историю по
каждой из $k$ координат и сохраняет $k$ чисел, без такого max pooling.
При $n_h$ фильтрах каждой высоты $h$ и $n_v$ вертикальных фильтрах
конкатенация имеет длину $\sum_h n_h+k n_v$.

Посчитаем уменьшенный вариант с $L=3$, $k=2$,
$E=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$, одним горизонтальным
фильтром высоты 2 и одним вертикальным. Горизонтальное ядро
$F=\begin{bmatrix}1&0\\0&1\end{bmatrix}$ даёт до pooling
$(2,1)$, после ReLU и max по двум позициям — $o=2$.
Вертикальное ядро $G=(1,0,-1)^\top$ даёт
$o'=(1-1,0-1)=(0,-1)$: здесь первая и последняя позиции сравниваются
отдельно для каждой координаты. Bias всех фильтров в примере равен 0.

Теперь $[o;o']=(2,0,-1)^\top$. При
$W=\begin{bmatrix}0{,}5&0&1\\0&1&-1\end{bmatrix}$ и $b=0$
получаем $z=\operatorname{ReLU}(W[o;o'])=(0,1)^\top$.
С долгосрочным $p_u=(0{,}2,0{,}3)^\top$ итоговый вектор имеет длину 4.
Для candidate weight $v_i=(1,0{,}5,1,0)^\top$ и $b_i=0$ score равен
$0+0{,}5+0{,}2+0=0{,}7$. Полный Caser использует больше фильтров;
выбранный маленький пример сохраняет оси операций и место объединения ветвей.

Краткосрочное представление $z$ объединяется с long-term user embedding $p_u$:

$$
\hat y_{uit}=v_i^\top[z;p_u]+b_i.
$$

![[Assets/Sources/Dive into Deep Learning — Recommender Systems/rec-caser.svg]]

*Архитектура Caser из главы [Sequence-Aware Recommender
Systems](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recommender-systems/seqrec.md)
книги Dive into Deep Learning, CC BY-SA 4.0; исходный SVG не изменён.*

Из схемы видно, какое inductive bias вносит модель: horizontal filter
рассматривает локальные последовательные motifs фиксированной ширины; user
embedding хранит устойчивый вкус. Недостаток — фиксированное convolutional
окно и ограниченная возможность выбирать далёкие релевантные события.

## SASRec: causal self-attention

SASRec кодирует item sequence Transformer blocks. Для позиции $t$ causal mask
запрещает видеть $i_{t+1:}$:

$$
h_t^{(0)}=e_{i_t}+p_t,
$$

$$
H^{(\ell+1)}
=\operatorname{TransformerBlock}
\left(H^{(\ell)},M_{\text{causal}},M_{\text{padding}}\right).
$$

Score следующего item часто строится скалярным произведением

$$
s(i\mid i_{\le t})=h_t^\top e_i.
$$

Для каждого prefix модель сравнивает positive next item и sampled negatives.
Self-attention позволяет текущему состоянию сильнее опираться на релевантные
прошлые действия, а не только на последние $h$ подряд.

![[Assets/Sources/Linux Foundation Recommenders/examples/00_quick_start/sasrec_amazon/cell-001-attachment.png]]

Источник рисунка и полный notebook: [Linux Foundation Recommenders — SASRec and
SSE-PT](https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/sasrec_amazon.ipynb).
В notebook зафиксированы реальные параметры experiment: Amazon Electronics,
`maxlen=50`, два Transformer blocks, hidden size 100, sampled evaluation с
100 negatives и leave-one-out split. Эти числа не являются универсальными
рекомендациями; они определяют конкретный воспроизводимый опыт.

### Padding и mask — разные вещи

В batch короткие histories дополняются PAD. Нужны две masks:

- causal mask скрывает будущие настоящие позиции;
- padding mask скрывает искусственные PAD tokens.

Если PAD имеет обучаемый embedding и не замаскирован, модель получает сигнал от
длины истории и может переносить его неконтролируемо. Если positional indices
назначаются до truncation неверно, одинаковый recent suffix получает разные
positions в train и serving.

## Long-term, short-term и context

Практический score удобно разложить:

$$
s(u,i,t)=
s_{\text{long}}(u,i)
+s_{\text{session}}(i_{\le t},i)
+s_{\text{context}}(c_t,i).
$$

Long-term component отвечает за устойчивые предпочтения, session component —
за текущее намерение, context — за surface, устройство, время и запрос. Сумма
не обязана быть буквальной: компоненты можно объединять MLP или gating. Важно,
чтобы эксперимент мог выяснить вклад каждого:

| Ablation | Что проверяет |
|---|---|
| popularity only | базовая сила head items |
| user embedding only | долгосрочная персонализация |
| recent items only | краткосрочное намерение |
| sequence + user | совместная польза двух временны́х масштабов |
| + context | дополнительный signal, а не leakage |

## Feature-rich CTR prediction

Для каждого impression доступен набор fields:

```text
user_id, item_id, category, query, device, country,
surface, position, hour, campaign, price, ...
```

Модель оценивает

$$
\hat p=P(y=1\mid u,i,c,\mathrm{shown}).
$$

Категориальные fields отображаются в embeddings, численные нормализуются или
дискретизируются, затем ranker моделирует interactions. FM эффективно
покрывает pairwise interactions, DeepFM добавляет nonlinear higher-order
ветвь.

### Worked example field interactions

Пусть активны `user=42`, `item=camera`, `device=mobile`,
`hour=late_evening`. Линейная часть может выучить, что camera популярна.
Pairwise terms различают:

- user 42 обычно выбирает cameras;
- на mobile этот item кликают чаще;
- пользователь 42 активнее вечером.

Deep branch может выучить условие более высокого порядка: сочетание
`user-group × mobile × evening` важно, хотя каждая отдельная пара слаба.

Но position одновременно является причиной клика и результатом старой policy.
Модель, обученная на observed clicks, может принять «стоял сверху» за
«релевантен». Это не исправляется простым добавлением position feature:
для causal оценки нужны randomized exposure или методы correction с явно
проверяемыми предположениями.

## Признаки без утечки

Для impression в момент $t$ каждый feature должен быть вычислим из состояния
не позднее $t$:

$$
x_{uic}(t)=f(\text{events available to the system before }t).
$$

Типичные ошибки:

- `item_ctr_7d` посчитан на всём дне, включая события после impression;
- user history содержит target event;
- category vocabulary fit на test;
- availability взята из текущего каталога, а не historical snapshot;
- negatives включают items, которых в тот момент ещё не существовало;
- feature store при offline join выбирает latest value вместо point-in-time
  value.

Нужен point-in-time correct join с проверкой времени фактической доступности
версии: `feature_available_at <= impression_timestamp`, а не только event
timestamp. Поздно доставленное старое событие не должно появляться в прошлом
состоянии. Два времени и пример задержки разобраны в
[[00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline#Data contract|контракте данных]].

## Cold start

ID embedding нового пользователя или item не обучен. Cold start делится как
минимум на четыре случая:

| Случай | Возможная стратегия |
|---|---|
| новый пользователь, нет событий | contextual popularity, onboarding |
| новый пользователь, короткая session | session model без user ID |
| новый item, есть content | text/image/category encoder |
| новый item, нет content signal | exploration и editorial priors |

Обычный random interaction split почти не проверяет ни один из них. Нужны
отдельные user-holdout и item-holdout protocols. При item cold start нельзя
случайно передать модели aggregate popularity нового item из будущего.

## Failure modes sequence models

**Repeated consumption.** Исключение seen items может быть верно для купленного
фильма и неверно для музыки или расходников.

**Session boundary.** Ночная пауза, новый запрос или смена surface могут
разделять намерения лучше фиксированного time gap.

**Truncation.** Маленький `max_len` теряет долгосрочный signal; большой
увеличивает стоимость и добавляет шум.

**Negative sampling mismatch.** Uniform negatives делают offline task слишком
лёгкой; serving сталкивается с похожими hard candidates.

**Feedback loops.** Модель чаще показывает освоенные patterns и получает ещё
больше подтверждающих sequences.

**Stale history.** Online feature обновляется позже события, поэтому train и
serving видят разные prefixes.

## Production contract

Перед deployment sequence model фиксируют:

1. определение session и сортировку равных timestamps;
2. event weights и фильтрацию bots/duplicates;
3. максимальную длину, truncation side, PAD ID и positional convention;
4. versioned item vocabulary и OOV behavior;
5. latency обновления user state;
6. candidate retrieval и число scored items;
7. fallback для пустой/устаревшей истории;
8. full-catalog или sampled offline evaluation;
9. monitoring по history length, cold users и freshness.

Этот контракт не является второстепенной инженерной деталью: его изменение
меняет функцию, которую вычисляет модель.

## Первоисточники и продолжение

- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/seqrec|D2L: Caser and sequence preparation]];
- [[05 Источники/Courses/Linux Foundation Recommenders/examples/00_quick_start/sasrec_amazon.ipynb|LF: SASRec on Amazon]];
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/ctr|D2L: CTR Prediction]];
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/fm|D2L: Factorization Machines]];
- [[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/deepfm|D2L: DeepFM]];
- [[00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация|следующая глава]] — offline metrics, beyond accuracy, A/B test и эксплуатация.
