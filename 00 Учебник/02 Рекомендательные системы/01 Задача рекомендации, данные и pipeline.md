---
title: Задача рекомендации, данные и pipeline
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# Задача рекомендации, данные и pipeline

Исходное изложение терминов и постановок находится в
[[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/recsys-intro|D2L: Overview of Recommender Systems]].
Полный разбор MovieLens, его формата и вариантов split сохранён в
[[05 Источники/Courses/Dive into Deep Learning — Recommender Systems/movielens|D2L: The MovieLens Dataset]].
Практический notebook с random, chronological и stratified splitting:
[[05 Источники/Courses/Linux Foundation Recommenders/examples/01_prepare_data/data_split.ipynb|LF Recommenders: Data splitting techniques]].

## Сначала определить событие и решение

Таблица взаимодействий не является готовой разметкой. Просмотр фильма,
добавление товара в корзину и пропуск трека означают разные вещи. Даже клик
не всегда выражает предпочтение: пользователь мог открыть объект случайно,
искать негативную информацию или быстро закрыть страницу.

**Explicit feedback** — rating, like/dislike или другой сигнал, который
пользователь сообщает намеренно. **Implicit feedback** — наблюдаемое действие:
клик, просмотр, покупка, dwell time. Explicit feedback ближе к заявленному
отношению, но редок и подвержен selection bias. Implicit feedback массовый, но
неоднозначный. Отсутствие события не является отрицательной меткой: пользователь
мог никогда не увидеть объект.

До выбора модели следует записать:

- единицу решения: item, slate, session или notification;
- candidate universe и правила доступности;
- положительный сигнал и окно атрибуции;
- способ получения negatives;
- горизонт прогноза;
- ограничения: latency, inventory, policy, diversity.

## Четыре разные задачи

### Rating prediction

Для наблюдаемых пар $(u,i)\in\mathcal K$ модель прогнозирует численную оценку
$\hat r_{ui}$. Типичная функция потерь:

$$
\mathcal L_{\mathrm{MSE}}
=\frac{1}{|\mathcal K|}
\sum_{(u,i)\in\mathcal K}(r_{ui}-\hat r_{ui})^2.
$$

Такая постановка естественна для MovieLens, но современная лента чаще не
показывает ratings и требует top-$K$ ranking.

### Top-$K$ recommendation

Для каждого пользователя система выбирает $K$ объектов из каталога. Здесь
важен относительный порядок. Модель может выдавать score $s(u,i)$, после чего
объекты сортируются. Для большого каталога scoring всех items слишком дорог,
поэтому pipeline обычно разделяют на retrieval и ranking.

### CTR или response prediction

Модель оценивает

$$
\hat p_{uic}=P(y=1\mid u,i,c),
$$

где контекст $c$ включает позицию, устройство, время, запрос и поверхность
продукта. В логах наблюдается результат прежней policy показа, поэтому
простое обучение на кликах воспроизводит её selection и position bias.

### Next-item prediction

Вместо агрегированной матрицы используется последовательность
$(i_1,\ldots,i_t)$. Цель — следующий объект $i_{t+1}$ или множество событий в
следующем окне. Такая постановка нужна, когда краткосрочное намерение важнее
долгосрочного профиля.

## Двухступенчатый serving

В каталоге из миллионов объектов полный ranking на каждом запросе невозможен.
Практический pipeline устроен так:

1. retrieval возвращает сотни или тысячи candidates;
2. ranking применяет более дорогую модель;
3. post-ranking учитывает diversity, business rules и policy constraints;
4. logger сохраняет candidates, scores, positions, exposure и response.

Логировать только клики недостаточно. Без impression и candidate logs нельзя
отличить «пользователь отверг item» от «item не был доступен модели».

## Split должен имитировать будущее

Случайный split взаимодействий часто переносит поздние события пользователя в
train, а ранние — в test. Модель получает информацию из будущего. Для
реалистичного offline protocol используют:

- **chronological split** по глобальной временной границе;
- **leave-last-one-out**: последнее событие каждого пользователя — test;
- **rolling window** для оценки стабильности во времени;
- user/item holdout для отдельного измерения cold start.

При разделении необходимо сохранять train-only fitting всех статистик,
vocabulary и negative sampler. Иначе popularity, item frequencies или
normalization уже содержат test information.

### Worked example временного split

Пусть у пользователя есть события:

| Время | Item | Действие |
|---|---|---|
| 10:00 | A | view |
| 10:05 | B | view |
| 10:11 | C | purchase |
| 10:40 | D | view |

При leave-last-one-out история `[A,B,C]` образует train, `D` — test target.
Модель не должна видеть статистику, посчитанную с учётом `D`. Если же случайно
поместить `B` в test, а `C` и `D` оставить в train, задача перестаёт имитировать
прогноз будущего.

Глобальная временная граница проверяет реальный deployment shift, но создаёт
новых users и items. Leave-one-out гарантирует test event для активных users,
однако легче production-задачи и исключает пользователей с одной записью.
Поэтому разумный отчёт содержит оба среза: warm-start ranking и отдельный
cold-start protocol.

```python
events = events.sort_values(["user_id", "timestamp"])
test = events.groupby("user_id").tail(1)
train = events.drop(test.index)

# Fit statistics strictly on train.
item_popularity = train.groupby("item_id").size()
known_users = set(train.user_id)
known_items = set(train.item_id)
```

Это схема, а не универсальная функция: для session recommendation граница
проводится между sessions, а для каталога с меняющейся доступностью дополнительно
нужно восстановить items available at test time.

## Как строятся negatives

Для observed positive $(u,i^+)$ требуется определить, с чем его сравнивать.

**Exposed negative** — item был показан, но response не наступил в заданном
окне. Такой negative ближе к продуктовой задаче, однако зависит от position и
прежней policy.

**Unexposed sampled negative** выбирается из каталога. Он помогает обучению
retrieval, но отсутствие события не доказывает неприязнь.

**In-batch negative** использует positive items других пользователей. Метод
эффективен, но один и тот же item может быть релевантен нескольким users; кроме
того, sampling distribution определяется batch construction.

Один pointwise batch можно записать так:

```python
for user, positive in positives:
    negatives = sampler(user, k=4)
    yield (user, positive, 1)
    for item in negatives:
        yield (user, item, 0)
```

В experiment report обязательно указывают sampler, число negatives и
исключение прежних positives. Иначе loss и ranking metrics невоспроизводимы.

## Candidate generation — часть задачи

Пусть каталог содержит $10^7$ items, а ranker обрабатывает $10^3$ candidates.
Даже идеальный ranker не вернёт relevant item, если retrieval его потерял.
Поэтому end-to-end recall раскладывается:

$$
P(\text{relevant in final top-}K)
=P(\text{in candidates})
P(\text{top-}K\mid\text{in candidates}).
$$

Первый множитель измеряет candidate recall, второй — качество ranker на
доступном множестве. Улучшать только второй бессмысленно, если candidate recall
низок.

Candidates часто объединяют из нескольких каналов:

- collaborative nearest items;
- two-tower ANN retrieval;
- popular/recent by segment;
- content similarity;
- editorial inventory;
- exploration candidates.

После объединения нужны deduplication и provenance: для каждого candidate
полезно знать generator, исходный score и rank.

## Data contract

Перед обучением фиксируют:

```text
observation_time < label_window_start <= label_window_end
feature_timestamp <= observation_time
item_available(observation_time) == true
```

Первое условие отделяет features от будущего target. Второе запрещает
late-arriving information. Третье исключает рекомендации объектов, которые
невозможно было показать. Нарушение любого условия даёт offline improvement,
которое не воспроизводится online.

## Baseline раньше архитектуры

Полный notebook с global mean, user bias, item bias и matrix factorization:
[[05 Источники/Courses/Linux Foundation Recommenders/examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb|LF Recommenders: Baseline recommender deep dive]].

Минимальный набор:

- random;
- most popular;
- global mean для ratings;
- global mean + user bias + item bias;
- recent/popular-by-segment.

Сложная модель полезна только если превосходит baseline на том же split,
candidate set и наборе пользователей. Иначе улучшение может объясняться
другими negatives или более лёгким test protocol.

### Bias baseline как проверка dataset

Для ratings:

$$
\hat r_{ui}=\mu+b_u+b_i.
$$

Если сложная модель едва превосходит его, latent interactions дают мало
дополнительной информации. Если даже baseline показывает неправдоподобно
высокое качество, следует искать leakage или дубликаты.

Для ranking аналогом служит popularity:

$$
s(u,i)=\log(1+\mathrm{count}_{train}(i)).
$$

Его нужно оценивать не только в среднем, но и на users с короткой историей.
Персонализированная модель часто выигрывает на активных users, но проигрывает
надёжному popularity fallback для cold users.

## Типичные ошибки постановки

- Считать все пропуски отрицательными.
- Делать random split временных взаимодействий.
- Вычислять popularity и vocabulary до split.
- Сравнивать модели на разных candidate sets.
- Удалять cold users из отчёта без отдельной метрики.
- Логировать clicks, но не impressions и positions.
- Оптимизировать CTR, не учитывая downstream purchase, satisfaction или
  повторные визиты.

Эти ошибки нельзя исправить архитектурой. Сначала строится проверяемый dataset и
evaluation contract, затем выбирается модель.
