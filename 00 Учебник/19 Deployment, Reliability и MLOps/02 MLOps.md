---
title: "MLOps: воспроизводимость, выпуск и надёжность ML-систем"
type: textbook-chapter
status: canonical
last_updated: 2026-10-29
last_verified: 2026-10-26
primary_sources:
  - https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
  - https://sre.google/workbook/canarying-releases/
  - https://sre.google/workbook/alerting-on-slos/
  - https://kubernetes.io/docs/concepts/workloads/pods/probes/
  - https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
  - https://github.com/harvard-edge/cs249r_book/tree/45ecc8d82fcae70c149cdce550d3b3d3411df913
---

# MLOps: воспроизводимость, выпуск и надёжность ML-систем

Кандидат C из примера службы поддержки выполняет два условия
на validation-наборе: recall не ниже 85% и не более 1400 обращений
к инженерам из 10 000. Теперь нужно проверить его сервисную сборку
и порядок выпуска. При экспорте могли поменяться
токенизатор или порядок классов. Новая реплика может принимать соединения,
пока веса ещё загружаются. После переключения первые жалобы могут прийти
раньше, чем появятся метки для оценки качества.

MLOps связывает эти операции с конкретной версией системы:
как собрать и проверить артефакт, допустить его к трафику, наблюдать
результат и восстановить обслуживание при ошибке. Обучающий конвейер
тоже требует выпуска: ошибка в подготовке данных способна повториться
во всех автоматически обученных кандидатах.

Ниже продолжается учебный пример службы поддержки.
Числа в расчётах условные; графики из источников описывают примеры их авторов.

<a id="что-именно-требуется-воспроизвести"></a>
<a id="три-интерфейса-на-которых-ломается-система"></a>

## Артефакт, проверка и работающая версия

В реестре находятся два выпуска: `tickets-17` обслуживает запросы,
`tickets-18` является кандидатом. У каждого выпуска есть неизменяемое
описание зависимостей и ссылки на файлы с контрольными суммами.
Указатель `production` сообщает, какой выпуск выбран сейчас.
Изменение указателя не должно менять содержимое самого `tickets-17`.

Выпуск включает веса, токенизатор, подготовку текста, порядок меток,
порог и среду исполнения. В LLM-приложении сюда могут входить chat template,
prompt, параметры генерации, инструменты и harness. Для RAG дополнительно
фиксируются корпус, embedding-модель, индекс и сборка контекста.
Идентификатор такого описания уже разобран в [[01 ML workflow]].

При выпуске проверяют соответствие файлов описанию. Например,
manifest указывает токенизатор версии 3, а контейнер по умолчанию
скачивает последнюю версию из сети. Пока артефакт не загружен
в целевое окружение, эту ошибку можно не заметить.
Фиксировать имя модели недостаточно: имя или тег могут разрешаться
в разные файлы в разные моменты.

Контрольная сумма проверяет совпадение содержимого с ожидаемым.
Цифровая подпись дополнительно позволяет проверить происхождение
при доверенном ключе и установленной политике проверки.
Ни одно из этих свойств не доказывает качество предсказаний.

### Результат проверки относится к определённому кандидату

Пусть кандидат прошёл пять проверок: данных, поведения модели,
загрузки экспортированного артефакта, нагрузки и возврата на прежний выпуск.
Если после этого изменили токенизатор, старые результаты нельзя
приписать новой сборке. Даже зелёный отчёт должен указывать,
какой именно объект и в каких условиях проверялся.

Следующая функция показывает только связь результата с кандидатом.
Она не развёртывает модель и не заменяет правила допуска.

```python
import math
import numpy as np

def release_blockers(candidate_id, results, required):
    assert candidate_id and required and len(required) == len(set(required))
    blockers = []
    for name in required:
        result = results.get(name)
        if result is None:
            blockers.append((name, 'missing'))
        elif result['subject'] != candidate_id:
            blockers.append((name, 'different-candidate'))
        elif result['passed'] is not True:
            blockers.append((name, 'failed'))
    return blockers

required = ['data', 'behavior', 'artifact', 'load', 'rollback']
checks = {
    name: {'subject': 'tickets-18', 'passed': True}
    for name in required
}
assert release_blockers('tickets-18', checks, required) == []
assert len(release_blockers('tickets-19', checks, required)) == 5
missing_load = {key: value for key, value in checks.items() if key != 'load'}
assert release_blockers('tickets-18', missing_load, required) == [('load', 'missing')]
```

Содержимое проверки задаётся отдельно: выборка и её срезы, допустимые
ошибки, оборудование, профиль нагрузки, срок действия результата.
Функция отвергает пустой список, но не может определить,
достаточны ли выбранные проверки. Их должна задавать политика выпуска,
а не сам кандидат. Подмена или неполнота этой политики является
отдельной ошибкой.

Обычно состояние кандидата меняется от зарегистрированного к проверенному,
затем к ограниченному и полному обслуживанию. Это пример жизненного цикла,
не обязательные названия стадий любого реестра. Реестр хранит решения
и происхождение; маршрутизатор применяет распределение трафика.
Запись `production` сама по себе ещё не сообщает, какая версия
фактически ответила на конкретный запрос.

<a id="интерфейс-данных-и-модели"></a>

<!-- BEGIN REVIEW: mlops feature-history 2026-10-26 -->
## Feature stores: offline history and online values

The support classifier uses `tickets_7d`, the number of tickets a customer
opened in the preceding seven days. Training needs this count for each
historical prediction time. Serving needs it for the customer making a
request now. Joining every old ticket to the customer's current count
would give the training pipeline information that the service did not have.

Even before considering time, two implementations can compute different
features. Harvard's ML Operations chapter gives this example:

<!-- source-excerpt: harvard-feature-definition-skew -->
> The data dependency debt and training-serving skew patterns described in
> the technical debt section share a common root cause: inconsistent feature
> computation across pipeline stages. Consider what typically happens without
> a feature store: a data scientist computes `user_session_length` in Python
> for training, while an engineer reimplements the same calculation in Java
> for serving. Subtle differences emerge: one uses wall-clock time, the other
> processing time; one includes idle timeouts, the other does not. The model
> trains on one definition but serves using another, and accuracy degrades
> silently.
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [ML Operations — Feature stores](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#L815),
CC BY-NC-SA 4.0. Excerpt in the original language; the internal section
reference is expanded into words and index markup is removed.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/feature-stores-2026-10-26/feast-architecture.png]]

*Feast contributors, [Architecture — Overview](https://github.com/feast-dev/feast/blob/1d5be950cf718a674cb01eaf0fd0ad6f0a4c6335/docs/getting-started/architecture/overview.md),
Feast v0.66.0, Apache-2.0. Complete original diagram, without cropping
or redrawing. The separate transformation box is part of the original.*

The two outputs on the right serve different queries. Offline retrieval
builds a table for many entities and historical timestamps; online retrieval
looks up a small set of values for the current request. A registry records
the feature definitions and the sources that supply them. Materialization
copies computed values into the serving store. It does not establish that
the computation was correct or that every historical value was available
at its stated event time.

For `tickets_7d`, the definition must specify the customer key, time zone,
window endpoints, duplicate handling and treatment of deleted tickets.
A batch job can evaluate this definition over historical events. A streaming
job can maintain the current count. Tests compare the two paths on the same
fixed history, including a ticket exactly at the window boundary and a
duplicate event. Matching column names and integer types would miss a
change from a seven-day window to a one-day window.

Feast's architecture documentation also distinguishes feature storage and
retrieval from transformation engines. A feature-store deployment does not
automatically implement every batch or streaming transformation. The team
still owns those computations, their dependencies and the comparison between
the values that training retrieves and the values that serving actually uses.

### Point-in-time retrieval

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/feature-stores-2026-10-26/time-travel.svg]]

*Harvard Edge ML Systems Book, [ML Operations at Scale — Point-in-time correctness](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4488),
CC BY-NC-SA 4.0. Complete original SVG. The timeline shows event order
schematically; positions are not an elapsed-time scale.*

The request in the figure arrives at noon, after four clicks. Six more
clicks occur later. The end-of-day total of ten is a valid daily aggregate,
but it is an invalid input to the noon prediction. Reconstructing a training
row requires the value available for that row's prediction time. Moving
the training job to a later date does not move the prediction's information
boundary.

Harvard expresses the historical lookup with this lateral join. The outer
query supplies an entity and its prediction timestamp. The inner query
discards later feature timestamps, orders the remaining rows and returns
the latest one. `LEFT JOIN` preserves events with no matching feature.

<!-- source-excerpt: harvard-point-in-time-sql -->
```sql
SELECT
    e.user_id,
    e.event_timestamp,
    e.label,
    f.feature_1,
    f.feature_2
FROM events e
LEFT JOIN LATERAL (
    SELECT feature_1, feature_2
    FROM features f
    WHERE f.user_id = e.user_id
      AND f.feature_timestamp <= e.event_timestamp
    ORDER BY f.feature_timestamp DESC
    LIMIT 1
) f ON TRUE
```
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [Point-in-Time Join](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4504),
CC BY-NC-SA 4.0. Original SQL preserved verbatim.*

This query assumes that the feature timestamp also establishes availability.
A delayed write or a later correction violates that assumption. Suppose a
value describes the state at 11:30 but first reaches the online store at
12:35. Its event timestamp passes the noon cutoff, although the service
could not have read it at noon.

The following small database separates these two times. Integers count
minutes since midnight; all rows belong to one day. `feature_1` is a click
count and `feature_2` is a monetary aggregate. Values and labels are a
teaching fixture, not a recorded production dataset.

```sql
CREATE TABLE events (
    user_id INTEGER, event_timestamp INTEGER, label INTEGER,
    PRIMARY KEY (user_id, event_timestamp)
);
INSERT INTO events VALUES
    (101, 720, 0), (101, 780, 1), (101, 900, 0), (102, 720, 1);

CREATE TABLE features (
    user_id INTEGER,
    feature_timestamp INTEGER,
    available_timestamp INTEGER,
    definition_id VARCHAR,
    source_snapshot VARCHAR,
    feature_1 INTEGER,
    feature_2 INTEGER,
    PRIMARY KEY (user_id, feature_timestamp, available_timestamp,
                 definition_id, source_snapshot)
);
INSERT INTO features VALUES
    (101, 660, 665, 'clicks:v1', 'source:A', 3, 30),
    (101, 690, 755, 'clicks:v1', 'source:A', 4, 40),
    (101, 690, 800, 'clicks:v1', 'source:A', 6, 60),
    (101, 780, 780, 'clicks:v1', 'source:A', 8, 80);
```

The second row describes 11:30 and becomes available at 12:35. The third
corrects the same 11:30 value at 13:20. At noon, neither row is eligible
for replay. At 13:00, the row stamped 13:00 is already available and is
more recent than either 11:30 revision. This example treats timestamps
as an ordered sequence in which a write stamped with the prediction time
is visible to that prediction. A real system must define ordering for
equal timestamps rather than infer it from coarse clock readings.

For this replay, add an availability cutoff, a 60-minute TTL, a pinned
definition and a pinned source snapshot. When two revisions have the same
feature timestamp, the latest eligible availability timestamp wins:

```sql
SELECT
    e.user_id, e.event_timestamp, e.label,
    f.feature_1, f.feature_2
FROM events e
LEFT JOIN LATERAL (
    SELECT feature_1, feature_2
    FROM features f
    WHERE f.user_id = e.user_id
      AND f.definition_id = 'clicks:v1'
      AND f.source_snapshot = 'source:A'
      AND f.feature_timestamp <= e.event_timestamp
      AND f.feature_timestamp >= e.event_timestamp - 60
      AND f.available_timestamp <= e.event_timestamp
    ORDER BY f.feature_timestamp DESC, f.available_timestamp DESC
    LIMIT 1
) f ON TRUE
ORDER BY e.user_id, e.event_timestamp;
```

| Prediction | Eligible value | Why |
|---|---:|---|
| User 101, 12:00 | 3 | The 11:00 value is available at 11:05 and is exactly 60 minutes old. Both 11:30 revisions arrive later than noon. |
| User 101, 13:00 | 8 | The 13:00 value is available at the prediction boundary under the fixture's ordering rule. |
| User 101, 15:00 | NULL | The most recent value is two hours old, outside the 60-minute TTL. |
| User 102, 12:00 | NULL | This user has no feature history. |

The original event-time-only query can return either 4 or 6 at noon:
both have the latest eligible feature timestamp, and its `ORDER BY` does
not break that tie. Both are unavailable at noon. It also returns 8 at
15:00 because it has no TTL. These are separate failures: tie-breaking,
historical availability and permitted age each need a rule. A missing
feature remains `NULL`; replacing it with zero would require an explicit
missing-value policy consistent with training.

### TTL and the time at which a value became available

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/feature-stores-2026-10-26/feast-point-in-time-join.png]]

*Feast contributors, [Point-in-time joins](https://github.com/feast-dev/feast/blob/1d5be950cf718a674cb01eaf0fd0ad6f0a4c6335/docs/getting-started/concepts/point-in-time-joins.md),
Apache-2.0. Complete original image. Left: prediction rows and their labels.
Right: feature history. Arrows identify the latest eligible historical row.*

In Feast's example the TTL is two hours. A prediction at 01:15:30 uses the
01:00 feature row; one at 04:32:45 uses 04:00. The 16:26:17 prediction
finds no eligible row because the latest stored features are from 05:00.
The scan goes backward from each prediction timestamp. A query run next
month must still use that row's prediction time, not next month's clock,
to apply the TTL.

Feast v0.66.0 makes the availability distinction explicit in its
[documentation](https://github.com/feast-dev/feast/blob/1d5be950cf718a674cb01eaf0fd0ad6f0a4c6335/docs/getting-started/concepts/point-in-time-joins.md#retrieving-features-as-of-the-event-time)
and [`get_historical_features` implementation](https://github.com/feast-dev/feast/blob/1d5be950cf718a674cb01eaf0fd0ad6f0a4c6335/sdk/python/feast/feature_store.py#L1972).
The default is `filter_by_created_timestamp=False`. A configured creation
timestamp normally breaks ties between rows with the same event timestamp;
it does not by itself impose a prediction-time availability cutoff.
Enabling `filter_by_created_timestamp=True` adds that cutoff for supported
offline stores. Feature views without a creation-timestamp column are
unaffected. NULL creation timestamps are excluded when filtering applies;
unsupported stores raise an error.

This setting reproduces online availability only if the recorded creation
time really represents when the value became available online. A warehouse
write timestamp can precede online materialization. In that case the replay
needs the later availability time or a log of the values actually served.
The SQL example above models the latter boundary directly; it is not an
execution of Feast or a test of all its backends.

TTL limits the age of an eligible feature row. Pipeline freshness asks
whether processing is keeping up with the source. A customer who has not
clicked for a week can have an old last-click timestamp while the pipeline
is fully caught up. Conversely, repeatedly writing a stale aggregate with
a new write timestamp can make the row appear young without incorporating
recent events. Monitor processing progress and source coverage alongside
row age. The treatment of late labels remains as in [[01 ML workflow]]:
labels may arrive after prediction, while input features must respect the
prediction's information boundary.

### Retaining history has a storage cost

Keeping only the latest online value cannot support the historical joins
above. The offline store must retain earlier versions or enough immutable
source data to reconstruct them. Harvard's [storage scenario](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4525)
assumes 100 million entities, 1000 features, one year of retention, hourly
updates and 100 bytes per retained feature value. If every feature for
every entity produces one retained value at every update, the uncompressed
size is

$$
B=N_{\mathrm{entities}}N_{\mathrm{features}}
  \frac{T_{\mathrm{retention}}}{\Delta t_{\mathrm{update}}}
  b_{\mathrm{value}}.
$$

A 365-day year contains 8760 hours. The scenario therefore retains
876 trillion feature values, or 87.6 decimal PB at 100 bytes per value.
The bytes-per-value assumption represents a stored record, not the width
of a floating-point scalar. This is a capacity calculation under the stated
assumptions, not a measurement of Feast or a typical feature-store size.

```python
entities, feature_count, hours_per_year = 100_000_000, 1000, 365 * 24
bytes_per_value = 100
retained_values = entities * feature_count * hours_per_year
retained_bytes = retained_values * bytes_per_value
assert retained_values == 876_000_000_000_000
assert retained_bytes == 87_600_000_000_000_000
assert retained_bytes / 10**15 == 87.6
```

Actual storage depends on which entities and features change, record
layout, compression, indexing, replicas and retention policy. Updating only
changed values can reduce retained records; shared timestamps can reduce
per-feature overhead. Replication and indexes add bytes. A backfill also
reads historical data and writes a new version, so its resource budget
includes that temporary overlap rather than just the final online table.

*The Harvard-derived explanations and added SQL replay example in this
section are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The two unchanged Feast images retain their Apache-2.0 license.*

<!-- END REVIEW: mlops feature-history 2026-10-26 -->

## CI, CT и CD

У конвейера обучения и у выпущенной модели разные изменения.
В понедельник прежний код может обучить модель на новой выгрузке.
Во вторник разработчик может исправить подготовку текста.
Первое создаёт нового кандидата; второе меняет процесс,
который будет создавать всех следующих кандидатов.

В [архитектуре Google Cloud](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
этот переход показан тремя уровнями автоматизации.

### Ручной процесс

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mlops-review-2026-10/google-manual.svg]]

*Jarek Kazmierczak, Khalid Salama, Valentín Huerta, Google Cloud,
[Figure 2 — Manual ML steps](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_0_manual_process),
CC BY 4.0. Полный оригинальный SVG без изменений.*

Исследователь запускает подготовку, обучение и оценивание,
затем передаёт артефакт в обслуживающую систему. На схеме граница
`ML | Ops` проходит рядом с передачей обученной модели.
Если подготовка текста осталась только в ноутбуке,
команда сервиса вынуждена восстанавливать её самостоятельно.

Ручной запуск не означает, что все результаты непроверяемы.
Редко обновляемая модель может иметь версионированные данные,
воспроизводимую сборку и строгий выпуск без автоматического обучения.
Проблема возникает, когда переходы зависят от незаписанных действий:
какой скрипт запущен, откуда взят словарь и какие строки исключены.

### Автоматический запуск обучения

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mlops-review-2026-10/google-ct.svg]]

*Jarek Kazmierczak, Khalid Salama, Valentín Huerta, Google Cloud,
[Figure 3 — ML pipeline automation for CT](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_1_ml_pipeline_automation),
CC BY 4.0. Полный исходник без изменений. Верхняя часть задаёт
реализацию конвейера, нижняя выполняет её на новых данных.*

На следующем уровне конвейер сам выполняет переходы между стадиями.
Перед обучением проверяются данные, после него кандидат.
Событие или расписание запускает процесс, а журнал выполнения
сохраняет промежуточные артефакты.

В примере поддержки новая выгрузка должна пройти проверку схемы,
временной доступности признаков и зрелости меток. Если столбец текста
стал пустым из-за ошибки выгрузки, обучение останавливают.
Если данные корректны, но состав обращений изменился,
кандидата можно обучить и сравнить с текущей моделью на нужной популяции.
Сам факт изменения распределения ещё не выбирает способ исправления.

В оригинале Google отдельно описано, что нужно сохранять о выполнении:

<!-- source-excerpt: google-mlops-metadata -->
> Information about each execution of the ML pipeline is recorded in order to help
> with data and artifacts lineage, reproducibility, and comparisons. It also helps
> you debug errors and anomalies. Each time you execute the pipeline, the ML
> metadata store records the following metadata:
>
> - The pipeline and component versions that were executed.
> - The start and end date, time, and how long the pipeline took to complete
> each of the steps.
> - The executor of the pipeline.
> - The parameter arguments that were passed to the pipeline.
<!-- /source-excerpt -->

*Google Cloud, [Metadata management](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#metadata_management),
CC BY 4.0. Слова и язык оригинала сохранены; изменена только разметка.*

Нужны и ссылки на выходы стадий: подготовленный набор, отчёт об аномалиях,
словарь, веса, оценки кандидата и предыдущей модели.
Повторный запуск может использовать сохранённую стадию только тогда,
когда совпадают её существенные входы и реализация.
Совпадение имени задачи не подтверждает пригодность старого результата.

Непрерывное обучение, continuous training или CT, означает автоматизированное
получение кандидатов. Переход кандидата к пользовательскому трафику
может оставаться отдельным решением. Частота зависит от появления
пригодных данных, изменений задачи и стоимости проверки.
Ежедневный запуск на ошибочной выгрузке лишь быстрее повторяет ошибку.

<!-- serving-decisions-completion: retraining start -->
### Retraining frequency: stale predictions and update cost

An automatic training pipeline still needs a rule for when to run. The
Harvard course separates three choices: a schedule, a trigger based on
observed degradation, and incremental updates as new labels arrive.
The scheduled policy is described as follows:

<!-- source-excerpt: harvard-retraining-schedule -->
> Those schedules are starting points, not rules. Scheduled retraining runs on a fixed cadence, such as daily, weekly, or monthly, regardless of performance metrics. It is simple to implement and guarantees that recent data eventually enters the model, but it can waste compute when the distribution is stable or respond too slowly when a shift happens between calendar runs.
<!-- /source-excerpt -->

*Harvard ML Systems, [Retraining decision framework](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#sec-ml-operations-retraining-decision-framework-e33d),
CC BY-NC-SA 4.0. Original English paragraph; publication markup only.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-decisions-2026-10-29/retraining-policies.png]]

*Vijay Janapa Reddi and Harvard ML Systems contributors,
[The Rotting Asset Curve, Figure 8](https://mlsysbook.ai/vol1/ml_ops/ml_ops.html#fig-rotting-asset-curve),
CC BY-NC-SA 4.0. Complete original course figure, unchanged. These are
illustrative curves, not observed performance of a deployed model.*

The orange curve resets at fixed times. The green curve resets when its
quality measure crosses the dashed threshold. In both cases the drawing
assumes that an update immediately restores the same accuracy. A real
update takes time to train, validate and release; the candidate can fail
validation. Input-distribution drift does not by itself imply that accuracy
has fallen. The figure illustrates two policies **under its assumed
degradation curve**, not a law that every deployed model deteriorates.

#### A cost model with explicit units

Suppose historical measurements support the local planning model

$$
A(t)=A_0e^{-\gamma t},
$$

where $t$ is days since the last accepted update, $A_0$ is accuracy just
after that update, and $\gamma$ has units day$^{-1}$. Each accepted update
is assumed to restore $A_0$. This approximation is unsuitable for a sudden
schema failure, seasonal recovery, or a task whose error cost cannot be
represented by accuracy. Those situations require a different loss curve
and possibly a data repair rather than retraining.

Let $Q$ be requests per day, $V$ the cost per request for a **unit change
in accuracy fraction**, and $C$ the expected total cost of an update.
At $V=0.50$ USD, a one-percentage-point loss costs 0.005 USD per request,
not 0.50 USD. The update cost includes training, validation, rollout work
and expected regression losses; it is not just the accelerator bill.

The instantaneous cost of stale predictions is
$QV[A_0-A(t)]$ dollars per day. Over an interval of length $T$:

$$
S(T)=QV A_0\left[T-\frac{1-e^{-\gamma T}}{\gamma}\right].
$$

Comparing $S(T)+C$ alone would favour very short intervals, because it
compares cycles of different lengths. Divide by the duration to obtain
the average cost rate:

$$
J(T)=\frac{S(T)+C}{T}.
$$

For small $\gamma T$, expansion of the exponential gives
$S(T)\approx QVA_0\gamma T^2/2$. Then

$$
\begin{gathered}
J(T)\approx\frac{QVA_0\gamma}{2}T+\frac{C}{T},\\
T_{\mathrm{approx}}^*=\sqrt{\frac{2C}{QVA_0\gamma}}.
\end{gathered}
$$

The first term increases with the interval: predictions remain stale
longer. The second decreases: the fixed update cost is paid less often.
This square-root rule is a **small-drift approximation**, not the exact
optimizer for arbitrary exponential decay. For the unapproximated model,
the stationary point satisfies

$$
\frac{QVA_0}{\gamma}
\left[1-(1+\gamma T)e^{-\gamma T}\right]=C.
$$

The left side increases from zero to $QVA_0/\gamma$. Consequently this
particular model has no finite optimum when $C\geq QVA_0/\gamma$:
$J(T)$ decreases toward its long-interval limit. That conclusion belongs
to the assumed cost function; it does not override safety or minimum-quality
constraints that the function omitted.

#### Reproduce the course's numerical example

For $Q=10^6$ requests/day, $V=0.50$ USD, $A_0=0.95$,
$\gamma=0.02$/day and $C=5000$ USD, the square-root estimate is about
1.026 days. Solving the exact equation gives about 1.033 days. Neither
number means that every fraud model should retrain daily: all five inputs
and the restoration assumption were chosen for this example.

```python
def retraining_plan(q, value, initial_accuracy, decay, update_cost):
    values = [q, value, initial_accuracy, decay, update_cost]
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('positive finite inputs required')
    if initial_accuracy > 1:
        raise ValueError('accuracy is a fraction')
    scale = q * value * initial_accuracy
    approximate = math.sqrt(2 * update_cost / (scale * decay))
    if update_cost >= scale / decay:
        return approximate, math.inf
    target = update_cost * decay / scale

    def increase(x):
        # 1 - (1+x)*exp(-x), evaluated stably for small x.
        if x < 1e-3:
            return x*x/2 - x**3/3 + x**4/8 - x**5/30 + x**6/144
        return -math.expm1(-x) - x * math.exp(-x)

    low, high = 0., 1.
    while increase(high) < target:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if increase(middle) < target:
            low = middle
        else:
            high = middle
    return approximate, ((low + high) / 2) / decay

approximate, exact = retraining_plan(1e6, .50, .95, .02, 5000.)
assert 1.025 < approximate < 1.027
assert 1.032 < exact < 1.034
assert math.isinf(retraining_plan(1., 1., .5, 1., 1.)[1])
```

The sensitivity follows from the approximation: four times the traffic
halves the interval; four times the update cost doubles it. A shorter
interval is feasible only if fresh labels, training time and validation
capacity allow it. A 24-hour label delay cannot be repaired by starting a
training job every hour on the same immature labels.

For a triggered policy, compare expected recoverable losses over a stated
horizon with the cost and risk of a new candidate. First determine whether
the signal came from changed demand, broken preprocessing, delayed labels
or genuine loss of model quality. A feature-store outage requires restoring
features; automatically teaching the model to accept corrupted inputs can
make the incident worse. Preserve the accepted model while the candidate
is evaluated, and record why the trigger fired even when no release follows.

*Derivation and parameters follow Harvard's
[Quantitative retraining economics](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#sec-ml-operations-quantitative-retraining-economics-1579).
The exact stationary equation, finite-optimum condition and executable
comparison above make the approximation's assumptions explicit.*
<!-- serving-decisions-completion: retraining end -->

### CI и CD самого конвейера

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mlops-review-2026-10/google-cicd-stages.svg]]

*Jarek Kazmierczak, Khalid Salama, Valentín Huerta, Google Cloud,
[Figure 5 — Stages of the CI/CD automated ML pipeline](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_2_cicd_pipeline_automation),
CC BY 4.0. Полный оригинал. `Pipeline continuous delivery` выпускает
конвейер, `Model continuous delivery` выпускает полученную модель.*

CI собирает и проверяет изменения компонентов. Тест подготовки текста
проверяет ожидаемые токены на фиксированных примерах; тест обучения
может проверить, что маленькая задача переобучается без NaN;
интеграционный тест проверяет передачу артефактов между стадиями.
Уменьшение loss на нескольких строках проверяет механику обучения,
но не качество на будущих обращениях.

CD доставляет проверенную реализацию конвейера в целевую среду.
Затем этот конвейер может многократно запускаться на новых данных.
Доставка полученной модели в сервис является следующим переходом.
Если смешать их, успешный выпуск кода можно ошибочно принять
за успешную проверку всех произведённых им моделей.

| Изменение | Что проверяется сначала | Что затем требует отдельного результата |
|---|---|---|
| Код токенизации | Фиксированные примеры, совместимость формата и интеграция | Качество и нагрузка нового выпуска |
| Новая выгрузка | Схема, доступность признаков, метки, покрытие и дубликаты | Качество обученного кандидата |
| Порог решения | Действия по сохранённым оценкам и нагрузка на очереди | Поведение продукта после изменения |
| Версия runtime | Загрузка, численные отличия и ресурсы | Качество и задержка в целевом пути |

Это разные проверки даже тогда, когда их выполняет одна CI-система.
Результат маленького теста не заменяет полный прогон кандидата;
полный прогон не заменяет проверку его сервисной сборки.

<a id="интерфейс-модели-и-инфраструктуры"></a>

## Загрузка модели и готовность реплики

После запуска контейнера процесс может уже слушать порт,
но ещё загружать веса, создавать CUDA-контекст или прогревать kernels.
Если маршрутизатор начнёт посылать рабочие запросы слишком рано,
первый пользователь будет ждать инициализации либо получит ошибку.

В Kubernetes эти состояния разделяют
[startup, readiness и liveness probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/).

| Проверка | Какой вопрос задаёт | Реакция на устойчивую неудачу |
|---|---|---|
| Startup | Завершилось ли начальное включение приложения? | Контейнер завершается и далее действует restart policy |
| Readiness | Можно ли сейчас направлять сюда обычный трафик? | Реплика перестаёт считаться готовым адресом сервиса |
| Liveness | Поможет ли перезапуск восстановить зависший процесс? | Контейнер перезапускается |

При наличии startup probe проверки readiness и liveness не выполняются,
пока startup не пройдена. Это позволяет отдельно задать время
для долгой загрузки. Readiness продолжает работать после запуска
и может стать отрицательной позднее.

Предположим, старая реплика A принимает запросы, а B загружает новую модель.

| Состояние | A | B | Решение |
|---|---|---|---|
| B запущена, веса читаются | Готова | Не готова | Сохранять обслуживание на A |
| B загрузила ожидаемый артефакт и прошла прогрев | Готова | Готова | Проверить рабочий запрос через целевой путь |
| Начат ограниченный выпуск | Старая доля трафика | Небольшая новая доля | Сравнивать версии отдельно |
| B нарушила критерий остановки | Остаётся доступна | Новая доля уменьшается | Остановить расширение и восстановить прежний маршрут |

Для ML-сервиса endpoint готовности может учитывать,
что загружена ожидаемая версия и выполнена предусмотренная инициализация.
Но readiness не проводит статистическую оценку классификатора.
Тестовый ответ подтверждает ограниченный путь на ограниченном входе.

Liveness не следует привязывать к любому краткому сбою общей базы данных
или очереди. Одновременный перезапуск всех реплик способен увеличить
нагрузку на зависимость и уменьшить доступную мощность.
Причины, при которых приложение снимает готовность, также проверяют:
снятие всех реплик не создаёт исправный резервный путь.

### Завершение и возврат на прежний выпуск

Перед остановкой реплики нужно прекратить назначать ей новые запросы
и дать уже принятым завершиться в установленный срок.
Изменение endpoint распространяется не мгновенно; остаются соединения,
очереди, повторные попытки и потоковые ответы.
Настройки маршрутизатора, приложения и времени завершения должны
соответствовать этому поведению.

Для LLM длинная генерация может пережить момент переключения маршрута.
Отправить её повторно на старую модель не всегда допустимо:
у нового процесса нет прежнего состояния, а вызов инструмента мог уже
совершить внешнее действие. Отмена маршрута не отменяет выполненный перевод,
созданный документ или запись в базе.

Обратный переход требует доступных старых весов, токенизатора,
совместимой схемы и достаточной мощности. Если новая версия изменила
общее хранилище несовместимым образом, смены указателя модели недостаточно.
Поэтому совместимость данных и необратимые действия проверяют до выпуска.
Две полные среды в blue–green могут упростить переключение,
но общие зависимости всё равно остаются общими.

## Безопасный выпуск

Три распространённых способа проверки отличаются воздействием на пользователя.

| Способ | Какие запросы получает кандидат | Чьё решение применяется |
|---|---|---|
| Shadow, теневая версия | Копии реальных запросов | Прежней версии |
| Canary, ограниченный выпуск | Назначенная часть рабочего трафика | Кандидата для этой части |
| Blue–green | Новый выпуск подготовлен в отдельной среде | Версии, на которую указывает маршрут |

Теневая версия позволяет сравнить ответы на тех же входах.
Записи и вызовы инструментов у неё должны быть изолированы или отключены:
копирование запроса не должно повторять действие пользователя.
Общий кэш или база могут менять результаты обеих версий.
Дополнительная нагрузка shadow тоже входит в план мощности.

Blue–green описывает организацию сред и переключения.
Canary описывает ограничение воздействия и оценку нового поведения.
Их можно сочетать, направляя только часть трафика в новую среду.
Мгновенный переход всех пользователей между двумя средами
не даёт периода ограниченного воздействия.

### Почему ошибки новой версии не видны в общей метрике

В учебном сервисе Google SRE кандидат намеренно возвращает ошибки
примерно для 20% запросов. Ему направляют 5% трафика.
На общей панели ошибка почти исчезает на фоне масштаба оси:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mlops-review-2026-10/sre-canary-aggregate.webp]]

*Alec Warner, Štěpán Davidovič et al.,
[Google SRE Workbook, Figure 16-3](https://sre.google/workbook/canarying-releases/),
CC BY-NC-ND 4.0. Полный неизменённый оригинал; искусственно заданный
дефект примера, не измерение нашего сервиса.*

Разбиение по версии показывает, где появляются ошибки:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mlops-review-2026-10/sre-canary-versions.webp]]

*Alec Warner, Štěpán Davidovič et al.,
[Figure 16-4 — HTTP response codes by application version](https://sre.google/workbook/canarying-releases/),
CC BY-NC-ND 4.0. Полный исходник. Ось доли ошибок логарифмическая:
расстояния между 0.1%, 1%, 10% и 100% одинаковы.*

Разберём отдельный расчёт с нулевой долей ошибок у старой версии.
Всего приходит 100 запросов в секунду. За пять минут это 30 000 запросов;
при доле кандидата 5% он получает в среднем 1500.
Если вероятность ошибки у него 20%, ожидаются 300 ошибок.
По всему сервису это 1%, по кандидату 20%.

Для постоянных нагрузки и долей:

$$
e_{\mathrm{all}}=(1-f)e_{\mathrm{old}}+fe_{\mathrm{new}}.
$$

Здесь $f$ — доля запросов кандидата, а $e$ — вероятность ошибки.
При фиксированных $R$ запросах в секунду и времени воздействия $T$
ожидаемое число дополнительных ошибок равно
$RTf(e_{\mathrm{new}}-e_{\mathrm{old}})$.
Это модель ожидаемого количества, не число фактически наблюдённых ошибок.

```python
def canary_impact(rate, seconds, share, old_error, new_error):
    assert math.isfinite(rate) and math.isfinite(seconds)
    assert rate >= 0 and seconds >= 0
    assert all(0 <= p <= 1 for p in [share, old_error, new_error])
    requests = rate * seconds
    mixture = (1 - share) * old_error + share * new_error
    return {
        'candidate_requests': requests * share,
        'all_errors': requests * mixture,
        'additional_errors': requests * share * (new_error - old_error),
        'overall_error_ratio': mixture,
    }

impact = canary_impact(100, 300, .05, 0., .2)
assert impact['candidate_requests'] == 1500
assert np.isclose(impact['all_errors'], 300)
assert np.isclose(impact['additional_errors'], 300)
assert np.isclose(impact['overall_error_ratio'], .01)
```

Время воздействия включает обнаружение, решение об остановке,
распространение маршрута и незавершённые запросы.
После сигнала ошибки не обязательно прекращаются мгновенно.
При переменных нагрузке и доле выпуска расчёт выполняют по интервалам.
Одна доля canary не ограничивает ущерб от утечки данных
или необратимого действия.

### Сколько запросов нужно для редкой ошибки

Пять минут без ошибок могут означать лишь малую выборку.
Если ошибка возникает независимо с вероятностью $p$ на запрос,
вероятность не увидеть её за $n$ запросов равна $(1-p)^n$.
Чтобы с вероятностью хотя бы 95% встретить одну такую ошибку, требуется

$$
n\geq\frac{\ln(0.05)}{\ln(1-p)}.
$$

Для $p=0.001$ нужны 2995 запросов кандидата.
При 5 запросах в секунду это примерно десять минут.
Это вероятность обнаружить хотя бы один случай, не доказательство
различия двух моделей и не выбор длительности полноценного A/B-теста.

```python
def requests_for_detection(probability, confidence=.95):
    assert 0 < probability <= 1 and 0 < confidence < 1
    if probability == 1:
        return 1
    return math.ceil(math.log1p(-confidence) / math.log1p(-probability))

n = requests_for_detection(.001)
assert n == 2995
assert 1 - (1 - .001) ** n >= .95
assert 1 - (1 - .001) ** (n - 1) < .95
```

Запросы часто зависимы: один клиент повторяет сбойный вызов,
а редкая ошибка появляется только в определённом языке или длине входа.
Тогда формула с независимым одинаковым $p$ неприменима буквально.
Нужны покрытие этих групп и соответствующий период наблюдения.

Canary сравнивает поведение версий и ограничивает выпуск.
Для причинного вывода о качестве продукта дополнительно нужны
подходящая рандомизация, единица сравнения и учёт взаимодействий.
Проверка HTTP-ошибок не сообщает, помогло ли пользователю
новое решение классификатора. Если метка появляется через 72 часа,
пятиминутная стадия не может проверить её окончательное значение.

<!-- serving-decisions-completion: experiment start -->
### A/B validation: effect size, sample size and decision time

A canary can expose an API error in a few requests. Establishing that a
new ranker improves purchase probability from 5.0% to 5.1% is a different
measurement problem. The difference is 0.1 percentage points, or 2% of
the baseline rate. Counting requests without specifying which of these
effects the experiment must detect leaves its duration undefined.

Before assigning traffic, record the eligible population, assignment unit,
outcome window, primary metric, minimum effect of interest, guardrails and
decision rule. For example, assign each eligible user consistently to one
ranker and record whether that user makes a purchase within seven days of
first exposure. Multiple requests from that user are not independent
Bernoulli observations. A request-level experiment answers a different
question and can expose one user to both variants.

#### Why small changes require large samples

For independent users in two equally sized groups, let $p_0$ be the
baseline probability, $p_1=p_0+\delta$ the alternative, and $n$ the number
of users **per group**. The estimated difference has approximate variance

$$
\operatorname{Var}(\hat p_1-\hat p_0)
=\frac{p_0(1-p_0)+p_1(1-p_1)}{n}.
$$

When the effect is small, replace both probabilities by the planning
baseline $p$. For a two-sided test with type-I error $\alpha$ and target
power $1-\beta$, the resulting planning approximation is

$$
n\approx\frac{2p(1-p)
(z_{1-\alpha/2}+z_{1-\beta})^2}{\delta^2}.
$$

$\alpha$ limits false rejection under the null model; power is the
probability of detecting the specified alternative. Neither is the
probability that this particular candidate is better. The normal
approximation assumes enough successes and failures and independent
assignment units. It is not a formula for correlated per-token scores,
revenue with heavy tails, or a paired offline evaluation.

Harvard uses this approximation for its 5.0% versus 5.1% example. With
$\alpha=0.05$ and power 0.8 it requires about 746 thousand users per
variant. The following implementation also computes a refinement that
uses the distinct null and alternative variances; that gives about
753 thousand. This difference is approximation error, not extra evidence
about the model.

```python
from statistics import NormalDist

def ab_sample_size(p0, p1, alpha=.05, power=.8):
    if not all(math.isfinite(x) for x in [p0, p1, alpha, power]):
        raise ValueError('finite inputs required')
    if not (0 < p0 < 1 and 0 < p1 < 1 and p0 != p1):
        raise ValueError('distinct probabilities inside (0, 1) required')
    if not (0 < alpha < 1 and .5 < power < 1):
        raise ValueError('invalid alpha or power')
    normal = NormalDist()
    za = normal.inv_cdf(1 - alpha/2)
    zb = normal.inv_cdf(power)
    delta = abs(p1 - p0)
    approximate = 2 * p0 * (1 - p0) * (za + zb)**2 / delta**2
    pooled = (p0 + p1) / 2
    refined = (
        za * math.sqrt(2 * pooled * (1 - pooled))
        + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    )**2 / delta**2
    return math.ceil(approximate), math.ceil(refined)

rough, refined = ab_sample_size(.05, .051)
assert 745000 < rough < 747000
assert 752000 < refined < 754000
assert ab_sample_size(.05, .052)[0] == math.ceil(rough / 4)
```

At 100 thousand newly eligible, independent users per day across both
variants, the refined calculation needs a little over 15 days of
enrolment. The seven-day outcome window still has to mature for the final
users. A request counter growing at 100 thousand per day is not enough
if most requests come from the same users.

*Sample-size derivation: Harvard ML Systems,
[A/B testing for model validation](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#sec-ml-operations-ab-testing-model-validation-e480).
The distinct-variance formula is the normal approximation for two
independent proportions; see [Penn State STAT 507, §10.3](https://online.stat.psu.edu/stat507/Lesson10).
It is a planning estimate, not a substitute for the experiment's analysis.*

#### The outcome must have had time to occur

<!-- source-excerpt: harvard-experiment-delayed-feedback -->
> Those controls establish the statistical envelope, but ML systems add failure modes that ordinary web experiments can hide. Conversion events may arrive days after prediction, creating delayed feedback: a recommendation shown Monday can drive a purchase Friday, so the attribution window must be part of the test design. Novelty effects can also inflate early performance as users engage with fresh recommendations, which is why mature experiments include a burn-in period before measurement.
>
> Recommendation and ranking systems add interference effects because showing an item to one user can affect what remains available or salient for another, violating the independence assumption behind standard A/B analysis. Segment heterogeneity creates a second analysis problem: an overall neutral result may hide strong positive effects for one cohort and negative effects for another. These complications do not invalidate A/B testing, but they make guardrails, segment analysis, and preregistered decisions part of the experiment rather than after-the-fact interpretation.
<!-- /source-excerpt -->

*Harvard ML Systems, [Experiment setup and decision rules](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#sec-ml-operations-ab-testing-model-validation-e480),
CC BY-NC-SA 4.0. Two complete original paragraphs, publication markup only.*

For the seven-day purchase metric, a user exposed yesterday is not yet a
seven-day non-purchaser. Either compare only fully matured cohorts or use
a specified analysis for censored observations. Compare variants over
the same eligibility and calendar rules. A burn-in period, if justified,
is fixed before looking at results; deleting an inconvenient first week
after seeing the outcome changes the experiment.

#### Decide what will stop the experiment

In a fixed-horizon design, use the predeclared sample and observation
window. Repeatedly checking an ordinary fixed-sample p-value and stopping
the first time it falls below 0.05 does not preserve a 5% false-positive
rate. A sequential design can allow interim efficacy decisions, but its
boundaries must account for repeated looks. Stopping for a serious safety
failure is still appropriate; that does not establish a statistically
valid improvement in the primary metric.

Guardrails impose constraints on a proposed improvement. A higher purchase
rate can be unacceptable if refund rates, harmful recommendations or tail
latency exceed their allowed limits. Define the guardrail's population,
denominator, observation delay and acceptable degradation before launch.
Do not treat 'not statistically significant' as proof of no degradation;
the interval may still contain a practically important loss.

| Observation at the planned decision point | What follows |
|---|---|
| Primary improvement is supported; guardrail bounds pass | The candidate is eligible for the next release gate |
| Primary improvement is supported; a guardrail fails | Investigate the trade-off; improvement alone does not approve release |
| The interval includes both useful benefit and unacceptable loss | Evidence is inconclusive; do not call the models equivalent |
| A predefined harm condition is met | Stop exposure and investigate, irrespective of the primary p-value |

Keep the assignment log, maturity rules, exclusions, analysis code and
model/feature versions with the result. Check whether observed assignment
counts match the intended split before interpreting an outcome difference.
A routing bug that selectively drops users can invalidate the comparison
even when the final p-value is small.
<!-- serving-decisions-completion: experiment end -->

<a id="интерфейс-эксплуатации-и-наблюдения"></a>

## Наблюдаемость сервиса, данных, модели и продукта

Запрос должен быть связан с фактически обслужившим выпуском.
Иначе нельзя разделить ошибки кандидата и старой версии.
Нужны также время решения, состояние преобразований и идентификатор,
по которому позднее присоединяется исход.

Метрики, журналы и трассировки решают разные задачи.
Счётчики и гистограммы дают агрегированную картину;
журнал событий позволяет восстановить конкретный случай;
трассировка показывает длительности и зависимости внутри пути.
Содержимое сообщений и персональные данные сохраняются
только в рамках разрешённого доступа и срока хранения.

### Состояние сервиса

Доля завершённых запросов, задержки и нагрузка на очереди описывают
обслуживание. Для LLM дополнительно различают время до первого токена,
интервалы выдачи и полную длительность; определения введены в
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|главе о benchmarking]].
Срез по выпуску позволяет увидеть локальный дефект canary,
а по длине входа — перегрузку на длинных запросах.

Для доли отказов знаменателем служат допустимые запросы
в выбранных границах пути. Для доли ошибок классификации нужны
подходящие размеченные примеры. Успешный HTTP-ответ
не подтверждает правильность маршрута обращения.

### Состояние данных

Проверяют схему и семантику признаков, пропуски, неизвестные категории,
свежесть и доступность источников. Столбец `tickets_7d` может оставаться
целочисленным и непустым, но после изменения запроса считать обращения
за сутки. Тест типа это пропустит; тест на фиксированной истории
с известным семидневным результатом обнаружит подмену.

Свежесть тоже требует определения. Возраст последнего события
в журнале и задержка обработки источника являются разными величинами:
пользователь мог давно ничего не делать, хотя конвейер исправен.
Для редких событий нужны время контрольного обновления, watermark
или другой признак продвижения обработки, а не только дата активности.

### Состояние модели

Качество оценивают по когорте предсказаний и меткам, доступным
к моменту расчёта. В примере с горизонтом 72 часа новые обращения
ещё не прошли полный период. Их нельзя объявить отрицательными
или сравнить с полностью завершённой старой когортой.

Допустим, у старой когорты из 1000 обращений доступны 900 меток,
из них 810 совпали с прогнозом: accuracy на размеченной части 90%,
покрытие метками 90%. У сегодняшних 1000 обращений уже есть 100 меток,
из них 95 совпали. Значение 95% относится к рано размеченной части,
не ко всем новым обращениям. Быстрые случаи могут быть проще.

Панель должна показывать момент предсказания, горизонт результата,
возраст когорты, число примеров, покрытие и правила отбора меток.
Для неполного покрытия исследуют механизм пропусков;
его нельзя исправить одним увеличением размера выборки.
Разбор выборочной разметки приведён в [[01 ML workflow]].

### Состояние продукта

Число обращений, реально назначенных инженерам, время ожидания помощи
и доля ручной обработки показывают действие всей системы.
Они меняются вместе с составом клиентов, правилами очередей и доступной
мощностью операторов. Привязка к выпуску помогает локализовать изменение,
но временное совпадение не доказывает его причину.

Проверяют и малые группы. Если 95% запросов имеют accuracy 94%,
а остальные 5% — 62%, общее значение равно 92.4%.
В отчёте нужны и качество группы, и её размер: маленькая доля
не уменьшает тяжесть ошибки для конкретного пользователя.

### Сдвиг распределения и изменение качества

Рассмотрим две группы обращений: короткие и длинные.
Ошибки модели внутри групп равны 2% и 20%.
При долях 90% и 10% общая доля ошибок равна
$0.9\cdot0.02+0.1\cdot0.20=0.038$.
Если доли станут 50% и 50%, получим 11%, хотя поведение внутри
каждой группы не изменилось.

Теперь другой случай: доли остались 90% и 10%,
но ошибка на длинных обращениях выросла с 20% до 60%.
Общая ошибка стала 7.8%. Мониторинг только долей групп
ничего не обнаружит.

| Случай | Доли коротких/длинных | Ошибки внутри групп | Общая ошибка |
|---|---|---|---:|
| Базовый | 90% / 10% | 2% / 20% | 3.8% |
| Смена состава | 50% / 50% | 2% / 20% | 11% |
| Смена качества при том же составе | 90% / 10% | 2% / 60% | 7.8% |

Первый сдвиг относится к распределению входов $p(x)$.
Во втором меняется поведение относительно истинных меток при прежнем
наблюдаемом составе; изменение $p(y\mid x)$ называют concept drift.
Таблица показывает агрегированные ошибки, а не позволяет по одним
долям групп установить полное условное распределение.

Для гистограмм часто считают Population Stability Index:

$$
\operatorname{PSI}(q,p)=
\sum_i(q_i-p_i)\ln\frac{q_i}{p_i}.
$$

$p_i$ и $q_i$ — доли в одинаково определённых корзинах базового
и текущего периода. При совпадении долей PSI равен нулю.
В первом примере сдвига он примерно 0.879; при прежних долях
и ухудшении качества — ноль.

```python
def population_stability(reference, current):
    p, q = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    assert p.ndim == q.ndim == 1 and p.shape == q.shape and len(p) > 0
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(q))
    assert np.all(p > 0) and np.all(q > 0)
    assert np.isclose(p.sum(), 1.) and np.isclose(q.sum(), 1.)
    return float(np.sum((q - p) * np.log(q / p)))

base_mix = np.array([.9, .1])
shifted_mix = np.array([.5, .5])
base_errors = np.array([.02, .20])
changed_errors = np.array([.02, .60])
assert np.isclose(base_mix @ base_errors, .038)
assert np.isclose(shifted_mix @ base_errors, .11)
assert np.isclose(base_mix @ changed_errors, .078)
assert np.isclose(population_stability(base_mix, shifted_mix), .8788898309)
assert population_stability(base_mix, base_mix) == 0.
```

В этом расчёте доли положительны. При пустых корзинах логарифм
не определён; объединение корзин или сглаживание меняют оценку
и должны быть частью записанной процедуры. Размеры выборок и выбор
границ тоже влияют на интерпретацию.

Для ещё одного примера возьмём $p=(0.5,0.5)$ и $q=(0.75,0.25)$.
Тогда $\operatorname{PSI}=0.25\ln1.5-0.25\ln0.5=0.25\ln3\approx0.275$.
Это изменение долей, не доля ошибочных прогнозов.

Один из способов обработать пустые корзины — добавить заранее выбранное
псевдосчётное значение $\alpha>0$ к **числу наблюдений** каждой корзины.
Для окна из $N$ наблюдений и $k$ общих корзин используют

$$
\tilde q_i=\frac{n_i+\alpha}{N+k\alpha}.
$$

Для опорного окна вычисляют аналогичные доли с его собственным размером.
Например, $\alpha=0.5$ делает все доли положительными и сохраняет их сумму
равной единице. Это выбранное сглаживание, а не часть определения PSI;
оценка и пороги зависят от $\alpha$, размеров окон и границ корзин.
Два пустых окна не становятся свидетельством стабильности после добавления
псевдосчётов: наблюдений для такого вывода всё ещё нет.

~~~python
reference_counts = np.array([50., 50.])
current_counts = np.array([75., 25.])
assert np.isclose(population_stability(reference_counts / 100,
                                     current_counts / 100), .25 * np.log(3))
alpha = .5
p_smooth = (reference_counts + alpha) / (reference_counts.sum() + 2 * alpha)
q_smooth = (current_counts + alpha) / (current_counts.sum() + 2 * alpha)
assert np.isclose(p_smooth.sum(), 1.) and np.isclose(q_smooth.sum(), 1.)
assert 0 < population_stability(p_smooth, q_smooth) < .25 * np.log(3)
~~~

Расчёт и правила обработки корзин продолжают
[[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/robust_ai#Population stability index (PSI)|Harvard Robust AI: PSI]].
Порог нужно проверять с тем же сглаживанием и подготовкой данных, которые
будут применяться после выпуска.

В [Harvard ML Operations, Feature distribution monitoring](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#L2479)
разобраны PSI, KS и Jensen–Shannon divergence.
Часто встречающаяся граница PSI 0.2 является соглашением для мониторинга,
не универсальным тестом качества. Сдвиг долей может не менять ошибки
идеального для этих групп классификатора. Напротив, совместное
распределение нескольких признаков может измениться при неизменных
одномерных гистограммах.

## От метрики к оповещению

Срочное оповещение должно иметь понятное действие и владельца.
Один выброс PSI может создать задачу на исследование;
быстро растущая доля отказов рабочего сервиса может требовать
немедленного восстановления маршрута. Одинаковая срочность
для всех статистических отклонений перегружает дежурного.

Для доступности удобно связать срочность с бюджетом ошибок.
Пусть SLO требует успешного выполнения 99.9% допустимых запросов
за 30 дней. Из $N$ запросов бюджет равен $0.001N$.
Нужно заранее определить допустимые запросы, условия успеха
и правила обработки повторов, иначе знаменатель будет меняться
вместе со способом учёта.

Скорость расходования бюджета, burn rate, равна

$$
b=\frac{e}{1-S},
$$

где $e$ — текущая доля ошибок, $S$ — целевая доля успеха.
При SLO 99.9% и ошибках 1% burn rate равен 10.
Если такая доля и постоянная частота запросов сохранятся,
тридцатидневный бюджет будет израсходован за три дня.

При постоянном трафике доля бюджета, потраченная за окно $w$,
равна $bw/P$, где $P$ — полный период.
Для $P=720$ часов, $b=14.4$ и $w=1$ час это 2%.
При переменном трафике используют количества допустимых и неудачных
запросов, а не автоматически заменяют их длительностями окон.

```python
def budget_burn(slo, error_ratio, window_hours, period_hours):
    assert 0 < slo < 1 and 0 <= error_ratio <= 1
    assert math.isfinite(window_hours) and math.isfinite(period_hours)
    assert 0 <= window_hours <= period_hours and period_hours > 0
    burn = error_ratio / (1 - slo)
    fraction = burn * window_hours / period_hours
    return burn, fraction

burn, spent = budget_burn(.999, .0144, 1., 30 * 24.)
assert np.isclose(burn, 14.4) and np.isclose(spent, .02)
assert np.isclose(budget_burn(.999, .01, 1., 720.)[0], 10.)
```

### Длинное и короткое окно

Длинное окно накапливает достаточно событий, но продолжает помнить
аварию после восстановления. Короткое быстрее замечает прекращение
ошибок, однако сильнее реагирует на случайные всплески.
В двухоконном правиле сигнал возникает, когда порог превышен
в обоих окнах.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mlops-review-2026-10/sre-alert-windows.webp]]

*Steven Thurgood et al.,
[Google SRE Workbook, Figure 5-6 — Short and long windows for alerting](https://sre.google/workbook/alerting-on-slos/),
CC BY-NC-ND 4.0. Полный оригинал без изменений. Синяя сплошная линия —
исходная доля ошибок, зелёная — окно 5 минут, оранжевая — 60 минут;
пунктир — порог. Вертикальная шкала логарифмическая.*

В примере SRE ошибки возникают на ограниченном интервале.
После их прекращения часовая оценка ещё выше порога,
а пятиминутная уже падает. Двухоконное правило прекращает
сигнал раньше, чем правило только по часовому окну.

Для SLO 99.9% источник предлагает 14.4-кратный burn rate
в окнах 1 час и 5 минут как одну из исходных настроек.
Это не готовая политика для любого ML-сервиса.
У сервиса с десятью запросами в час одна ошибка даёт 10%;
нужно учитывать ценность каждого запроса, малую выборку
и покрытие синтетических проверок. Успех синтетического запроса
не должен разбавлять метрику реальных пользователей.

Бюджет доступности не заменяет бюджет ошибок классификации.
Неверное, но быстрое решение с HTTP 200 может полностью проходить
первый SLO и нарушать требования к качеству.
При задержанных метках для качества задают собственный период
наблюдения и реакцию, не обещая мгновенного обнаружения.

<!-- BEGIN REVIEW: mlops fleet-alerts 2026-10-26 -->
### False threshold crossings across a model fleet

Suppose a platform checks ten metrics for each of 100 models every five
minutes. There are 288 evaluation times per day and 288,000 individual
checks. If a check has a false-positive probability of 0.003 when the
system is healthy, the expected number of false threshold crossings is
288,000 × 0.003 = 864 per day. Adding models increases this count even
when every individual detector keeps the same false-positive rate.

The calculation comes from the Harvard ML Systems slides
[Alert Fatigue: The Math](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/slides/vol2/12_ops_scale/12_ops_scale.tex#L665-L691).
The next table retains the slide's assumed probabilities. They describe
healthy checks, not the measured behavior of our services.

| Models × metrics | Checks per day | Assumed false-positive probability | Expected false crossings per day |
| --- | ---: | ---: | ---: |
| 100 × 10 | 288,000 | 0.05 | 14,400 |
| 100 × 10 | 288,000 | 0.003 | 864 |
| 100 × 10 | 288,000 | 0.0001 | 28.8 |
| 200 × 8 | 460,800 | 0.003 | 1,382.4 |
| 200 × 8 | 460,800 | 0.0001 | 46.08 |

Let $I_j$ be one when healthy check $j$ crosses its threshold and zero
otherwise. The number of false crossings is $V=\sum_j I_j$. Linearity
of expectation gives

$$
\mathbb E[V]=\sum_j \Pr(I_j=1).
$$

This equality does not require independent checks. Metrics from the same
request, models sharing a feature service, and overlapping time windows can
all be correlated. If every check has false-positive probability $\alpha$,
the expectation is still $N\alpha$.

The probability of at least one crossing is a different quantity. For
independent checks with the same $\alpha$ it is

$$
\Pr(V\geq1)=1-(1-\alpha)^N.
$$

For 1,000 checks and $\alpha=0.003$, this probability is about 95.04%.
If the 1,000 checks are exact copies of one detector instead, either all
cross or none cross. The expected count remains three, but the probability
of any crossing is only 0.3%. Applying the independence formula to these
copies would give the wrong answer.

The slides label the three probabilities as two-, three- and four-sigma
thresholds. Those are rounded approximations, not exact Gaussian tails.
For a standard normal variable and a two-sided threshold, the probability
is $\operatorname{erfc}(k/\sqrt2)$. At three sigma it is approximately
0.0026998, giving 777.54 expected crossings for 288,000 checks; at four
sigma it is approximately 0.000063342, giving 18.24. Real monitoring
residuals need not be Gaussian, stationary or correctly standardized, so
a sigma label alone does not establish a detector's false-positive rate.

```python
import math

checks_per_day = 100 * 10 * (24 * 60 // 5)
assert checks_per_day == 288_000
assert checks_per_day * .003 == 864
assert math.isclose(checks_per_day * .0001, 28.8)

# Stable evaluation of 1 - (1 - alpha)**n for independent checks.
alpha, n = .003, 1000
p_any_independent = -math.expm1(n * math.log1p(-alpha))
assert math.isclose(p_any_independent, .9504369171768446)
assert n * alpha == 3  # expectation also holds for correlated checks
p_any_identical_copies = alpha
assert p_any_identical_copies == .003

exact_three_sigma = checks_per_day * math.erfc(3 / math.sqrt(2))
exact_four_sigma = checks_per_day * math.erfc(4 / math.sqrt(2))
assert math.isclose(exact_three_sigma, 777.5412659974949)
assert math.isclose(exact_four_sigma, 18.24263529593951)
```

### A threshold crossing is not a page

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/deployment-2026-10-26/prometheus-architecture.svg]]

*The Prometheus Authors, [Overview](https://github.com/prometheus/docs/blob/3576591af32864770ad6ff73a14fc485f70afb73/docs/introduction/overview.md),
complete original SVG, unchanged. [Apache-2.0](/licenses/prometheus-3576591-Apache-2.0.txt)
and [original NOTICE](/licenses/prometheus-3576591-NOTICE.txt).*

Follow the upper branch of the original diagram: Prometheus evaluates
alerting rules and sends alerts to Alertmanager; Alertmanager sends
notifications to receivers. A raw threshold crossing can remain pending
until a rule's required duration has elapsed. Grouping can combine several
firing alerts into one notification. Inhibition and silences can suppress
notifications. Routing determines the receiver; repeated notifications also
depend on timing settings. The 864 crossings above therefore do not predict
864 pages, nor does the slide's rounded count of 29 establish a manageable
on-call load. These stages are specified separately in
[Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
and [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/).

Raising a threshold reduces false positives only if the healthy-data
distribution remains appropriate. It can also delay detection or miss a
smaller real regression. Choose the condition together with its required
response: which user-visible failure it indicates, how long that failure
can continue, who acts, and which first action is safe. SLO burn-rate rules
above address sustained user impact; model and feature metrics help locate
the cause. An exhausted resource or an imminent failure can also warrant
action before a product metric changes. There is no general rule that
infrastructure metrics must never page. See Rob Ewaschuk,
[Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/).

The worked rollout and incident calculations are in
[[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис#Budget a release and an incident|practice 17]].

*The Harvard-derived arithmetic discussion and Bookvar's added examples use
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The unchanged Prometheus figure retains Apache-2.0 and its original NOTICE.*

<!-- END REVIEW: mlops fleet-alerts 2026-10-26 -->

## Технический долг ML-систем

Два классификатора могут зависеть от одной embedding-модели,
хотя их Python-пакеты не импортируют код друг друга.
Если общий сервис меняет координаты векторов, оба потребителя
получают новые входы без изменения формы массива.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mlops-review-2026-10/harvard-dependency.svg]]

*Harvard Edge ML Systems Book,
[ML Operations at Scale — cross-model dependencies](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L177),
CC BY-NC-SA 4.0. Полный исходный SVG без изменений.
Показаны потенциальные потребители обновления, не измеренный
ущерб каждого обновления.*

Рассмотрим одну координату embedding $x=0.8$.
Классификатор вычисляет $z=2x-1=0.6$ и выбирает положительный класс.
Новая embedding-модель меняет знак координаты у всех объектов.
Расстояния между объектами внутри новой системы координат сохранятся,
но прежний классификатор получит $x=-0.8$ и $z=-2.6$.
Одинаковая размерность вектора не означает совместимость координат.

Для RAG смена embedding-модели требует проверить согласованность
представлений запросов и документов, индекс, поиск и конечные ответы.
Старые документы в одном пространстве и новые запросы в другом
не становятся совместимыми от совпадения длины векторов.

В [Hidden Technical Debt in Machine Learning Systems, разделах 2–4](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
Sculley et al. разбирают переплетение признаков, корректирующие каскады,
необъявленных потребителей и обратную связь.
Название CACE расшифровывается как *Changing Anything Changes Everything*.
Это предупреждение о связанных эффектах изменений,
не утверждение, что любая правка обязательно меняет все ответы.

Корректирующий каскад возникает, когда следующая модель учится
исправлять прежнюю. Замена предшественника меняет её вход.
Необъявленный потребитель читает оценки из таблицы или журнала
без зарегистрированного контракта. Обратная связь меняет будущие
данные через действия системы. Для этих случаев нужны разные
проверки, а не одна команда «переобучить всё».

### Найти зависимых потребителей

Реестр зависимостей должен связывать версии источников, признаки,
модели, индексы и приложения. В небольшом примере изменение encoder
затрагивает индекс и ranker напрямую, а retriever и assistant косвенно.

```python
def downstream_consumers(graph, changed):
    reached, pending = set(), [changed]
    while pending:
        producer = pending.pop()
        for consumer in graph.get(producer, []):
            if consumer != changed and consumer not in reached:
                reached.add(consumer)
                pending.append(consumer)
    return reached

dependencies = {
    'encoder': ['index', 'ranker'],
    'index': ['retriever'],
    'retriever': ['assistant'],
}
assert downstream_consumers(dependencies, 'encoder') == {
    'index', 'ranker', 'retriever', 'assistant'
}
assert downstream_consumers(dependencies, 'ranker') == set()
assert 2 * .8 - 1 > 0 and 2 * (-.8) - 1 < 0
```

Список показывает область повторной проверки, не доказывает
деградацию всех потребителей. Иногда достаточно восстановить
прежний источник, иногда нужен новый индекс или переобученный
потребитель. Старую совместимую версию сохраняют на время перехода.
Её хранение и обслуживание имеют стоимость и срок завершения миграции.

<!-- BEGIN REVIEW: mlops feature-migration 2026-10-26 -->
### Feature versions, lineage and backfills

<!-- source-excerpt: harvard-feature-version-incident -->
> A representative `user_engagement_score` incident makes the mechanics
> concrete. A platform team changes the feature from a 30-day z-score to a
> 7-day min-max scale. The output may still be a floating-point number, so
> schema validation alone can pass while the feature's meaning changes
> completely. Versioning names the semantic change before it reaches
> production; lineage identifies every model trained or served with the old
> meaning; backfill recomputes historical values for the new meaning; and
> freshness and quality checks decide whether the rewritten history is safe
> to use. Without that chain, a feature migration becomes a silent
> multi-model regression.
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [Feature versioning and lineage](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4567),
CC BY-NC-SA 4.0. Original English paragraph. The source presents a
representative scenario, not a documented incident at a named company.*

To see the semantic change numerically, take an illustrative old-window
aggregate of 40, an old mean of 30 and a standard deviation of 10. The
z-score is $(40-30)/10=1$. Suppose the new seven-day aggregate is also 40,
with minimum 0 and maximum 100; the min-max value is 0.4. An old classifier
that tests whether the feature exceeds 0.8 changes its decision. Both
inputs remain valid floating-point numbers. The equal raw aggregates are
a chosen example, not an assumption that changing the window preserves
the aggregate.

The same dependency graph used above can locate consumers, but migration
also needs to identify what each consumer expects:

<!-- source-excerpt: harvard-feature-lineage-table -->
| <span id="feature-lineage-table">Component</span> | What It Records | Operational Use |
|---|---|---|
| Definition version | Computation logic for the feature | Blocks silent semantic changes when the formula changes but the type does not. |
| Data version | Source snapshot or stream contract | Distinguishes model changes from upstream data changes. |
| Schema version | Output type, shape, units, and nullability | Preserves API compatibility checks while still naming semantic shifts. |
| Consumer declaration | Exact feature versions each model consumes | Gives the platform a place to block unsafe updates before serving. |
| Source lineage | Source tables, streams, and their versions | Traces bad predictions back to the upstream data that produced them. |
| Transformation and runtime provenance | Transformation code, timestamp, and environment | Reproduces historical values for audit, debugging, and compliance. |
| Quality and freshness metrics | Validation results observed when the value was produced | Estimates blast radius and decides whether rewritten history is safe to use. |
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [Feature Version and Lineage Components](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4577),
CC BY-NC-SA 4.0. Original table text; Markdown spacing and emphasis simplified.*

A backfill computes historical values using the new definition. Keep the
old namespace, such as `engagement:v1`, and write the new values to
`engagement:v2`. A run records its source snapshot, transformation version
and covered partitions. Otherwise a retraining job can read some days
computed by the old formula and other days computed by the new one while
the column name and type remain unchanged.

Harvard describes the incremental procedure as follows:

<!-- source-excerpt: harvard-incremental-backfill -->
> The practical pattern is incremental. Historical data is processed in
> date partitions, and each partition is validated before the next one
> starts. During a dual-write period, old and new computations run in
> parallel, which gives the platform overlapping values for comparison
> before cutover. Rollback remains possible because the previous feature
> version and its lineage stay available until dependent models have been
> retrained, evaluated, and redeployed.
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [Backfill procedures](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4588),
CC BY-NC-SA 4.0. Original English paragraph.*

For a seven-day training window, six validated partitions do not establish
that the entire window is ready. The remaining partition may be missing,
failed or computed from another source snapshot. A retraining job should
resolve one immutable manifest listing all required partitions, rather than
discover files while another job is replacing them. Validation must check
the new definition against its specification: old and new feature values
can differ intentionally, so equality between v1 and v2 is not the
acceptance criterion.

The following check only verifies partition receipts. A receipt stands for
a validation result bound to one definition and source snapshot. The code
does not inspect stored data, run a backfill or judge a model's quality.

```python
def backfill_blockers(expected_partitions, receipts, definition, snapshot):
    if not expected_partitions or len(set(expected_partitions)) != len(expected_partitions):
        raise ValueError('expected partitions must be nonempty and unique')
    if not definition or not snapshot:
        raise ValueError('definition and source snapshot are required')
    blockers = []
    for partition in expected_partitions:
        receipt = receipts.get(partition)
        if receipt is None:
            blockers.append((partition, 'missing'))
        elif receipt['definition'] != definition or receipt['snapshot'] != snapshot:
            blockers.append((partition, 'different-inputs'))
        elif receipt['passed'] is not True:
            blockers.append((partition, 'failed'))
    return blockers

days = [f'day-{i}' for i in range(1, 8)]
receipts = {day: {'definition': 'engagement:v2', 'snapshot': 'source:A',
                  'passed': True} for day in days[:-1]}
assert backfill_blockers(days, receipts, 'engagement:v2', 'source:A') == [
    ('day-7', 'missing')]
receipts['day-7'] = {'definition': 'engagement:v2', 'snapshot': 'source:B',
                     'passed': True}
assert backfill_blockers(days, receipts, 'engagement:v2', 'source:A') == [
    ('day-7', 'different-inputs')]
receipts['day-7']['snapshot'] = 'source:A'
assert backfill_blockers(days, receipts, 'engagement:v2', 'source:A') == []
```

Completing the manifest permits training on that particular v2 history.
The resulting model still needs its own evaluation, artifact checks and
controlled release. During cutover, the old model continues reading v1;
the new model reads v2. Rolling back the weights while leaving the feature
binding on v2 would restore the wrong pair. Reserve capacity for dual writes
and backfill work so that migration does not displace live feature updates.

Recomputed history also needs a declared purpose. An experiment can ask
what a new definition would have produced from data available at each
historical time. An audit can ask what the old service actually used.
Recomputing corrected values today cannot answer the second question
without preserved versions and availability records. Keep the replay
contract with the dataset so that an offline improvement is not mistaken
for evidence about past production behaviour.

*This Harvard-derived section, including the added numerical and partition
examples, is distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The original source language is retained; the worked calculations and
partition-receipt code are Bookvar additions.*

<!-- END REVIEW: mlops feature-migration 2026-10-26 -->

## Диагностика и инцидент

Пусть после обновления общего конвейера число обращений к инженерам
резко изменилось. Доступность и задержка классификатора прежние.
Модель за последние сутки не выпускали.

Отсутствие выпуска весов не исключает изменение системы:
могли поменяться признаки, tokenizer, prompt, таблица соответствий
или маршрутизация. Сначала сохраняют фактические версии и
временное окно, а затем сравнивают наблюдения по зависимостям.

| Момент | Наблюдение | Следующее проверяемое предположение |
|---|---|---|
| 08:00 | Изменился конвейер `tickets_7d` | Совпадает ли прежняя семантика признака? |
| 08:10 | У двух потребителей упала доля отправки инженерам | Есть ли общий изменённый источник? |
| 08:12 | Числа целые, пропусков нет, HTTP-ответы успешны | Схема проходит, но значение может быть другим |
| 08:15 | На фиксированной истории найдено суточное окно вместо семидневного | Проверить распространение этой версии и доступный совместимый источник |
| После ограничения воздействия | Метрика и входы восстановлены | Проверить новые запросы и накопившуюся очередь, затем разбирать причину |

Это учебный сценарий, не наблюдавшийся инцидент конкретной организации.
Указанные минуты не являются обещанием времени восстановления.

Возврат только весов `tickets-17` не исправит новый общий признак,
если старая модель получает ту же испорченную версию.
Возможные меры — восстановить совместимый источник,
отключить проблемный путь или временно направить обращения человеку.
Каждая мера требует проверки мощности и допустимого поведения.
Если резервная ручная очередь уже переполнена, она не является
безусловно безопасным выходом.

Стабилизация может начаться до окончательного установления причины,
если воздействие продолжается. При росте ошибок соединений и задержек
сначала нужны проверки соответствующего пути и ресурсов.
Для исправного протокольного пути с изменившимися решениями
приоритет получают данные и поведение. Универсального порядка,
который всегда сначала исключает инфраструктуру или модель, нет.

### Что должно быть в runbook

Инструкция связывает наблюдение с действием. Для этого примера
она должна указывать владельца `tickets_7d`, способ проверить
активную версию, контрольную историю, путь восстановления
и ожидаемое подтверждение после него.
Ссылки на панель без определения нормального состояния
не задают следующего шага.

Нужны границы полномочий, условия остановки выпуска,
контакты владельцев данных и сервиса, способ работы при недоступном
реестре или мониторинге. Процедура восстановления не должна целиком
зависеть от той же системы, которую она восстанавливает.

После инцидента добавляют проверку окна агрегации на фиксированной истории,
версионируют изменение семантики и проверяют регистрацию потребителей.
В разборе фиксируют воздействие, время обнаружения, стабилизации
и восстановления, доказательства причины и владельцев исправлений.
Исправленная выгрузка закрывает один случай; новый тест проверяет
соответствующий класс ошибок.

В [Harvard ML Operations at Scale, Production Debugging and Incident Response](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4950)
связаны классификация инцидента, зависимости, runbook и последующие изменения.
Совпавшие симптомы помогают выбрать гипотезу, но не доказывают
общую причину без проверки.

## От одной модели к платформе

Если каждая команда заново пишет загрузку артефактов, журнал выпуска
и сбор метрик, общий компонент может уменьшить эту работу.
Но он также создаёт владельца, интерфейс и общую точку отказа.
У общей платформы должны быть изоляция команд, правила доступа,
квоты и способы восстановления.

Статистическое мультиплексирование помогает, когда пики команд
не совпадают и свободная мощность действительно доступна другим.
Для распределённого обучения нужны совместимые GPU с подходящими
связями; для serving нужна мощность к моменту поступления запроса.
Свободная суммарная память не гарантирует размещение конкретной модели.

Высокая средняя загрузка не доказывает эффективность.
Перегруженные GPU могут сопровождаться очередью и нарушением SLO.
Измеряют завершённую полезную работу, задержку ожидания,
стоимость принятого результата, потери от перезапусков и доступность
нужных групп ускорителей. Эти величины дополняют аппаратную утилизацию.

<!-- BEGIN REVIEW: mlops platform-resources 2026-10-26 -->
### Resource scheduling and tenant isolation

The historical backfill and the support classifier can use the same
platform, but they cannot consume capacity on the same terms. A delayed
backfill postpones the next training run. A delayed classifier keeps a
customer waiting now. Harvard describes the training side as follows:

<!-- source-excerpt: harvard-training-resource-management -->
> Training workloads are batch-oriented, so their flexibility can be
> converted into higher utilization. Jobs have defined start and end times,
> GPU memory requirements are often known in advance, and many runs can be
> preempted and restarted if checkpointing limits lost work. A training
> scheduler uses priority queues, fair sharing, and deadline-aware placement
> to decide which jobs should wait, which should move, and which low-priority
> jobs can be preempted for urgent work. Spot/preemptible instances fit this
> control loop because automatic retry on preemption can preserve progress
> while reducing cost.
<!-- /source-excerpt -->

For serving, the same source starts from the request deadline:

<!-- source-excerpt: harvard-serving-resource-management -->
> Serving workloads bind on latency rather than batch flexibility. Demand
> fluctuates with time of day, events, and seasonality, but a live request
> cannot be preempted without user impact. The serving control loop therefore
> starts from the latency SLO and asks how much capacity must be online
> before requests arrive.
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [Training and serving resource management](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L3575),
CC BY-NC-SA 4.0. Original English paragraphs; the training paragraph's
footnote marker is removed.*

A preemptible training job needs a completed, usable checkpoint and a
tested resume path. Merely restarting its container does not preserve
training progress. The checkpoint contents, lost work and recovery sequence
are covered in [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration|Fault tolerance и fleet orchestration]].
For serving, a new replica becomes capacity only after loading and readiness;
an autoscaling decision cannot retroactively serve the requests that arrived
during that interval. These two workloads therefore need distinct admission,
placement and interruption policies even when they share a cluster.

#### Workload admission and pod placement

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/platform-resources-2026-10-26/queueing-components.svg]]

*Kubernetes Authors and Kueue contributors, [Core Kueue Concepts](https://github.com/kubernetes-sigs/kueue/blob/bca571a3dae56ebca83109ae6769bade1c73dcfa/site/content/en/docs/concepts/_index.md),
Apache-2.0. Complete original SVG. A batch job is represented by a Workload;
many Workloads can use one LocalQueue, and many LocalQueues can refer to
one ClusterQueue.*

The namespace contains the job, its Workload representation and its
LocalQueue. The ClusterQueue is cluster-scoped and governs quota and sharing
policy. This allows a platform to give teams separate submission points
while accounting for their use of a common resource pool.

Kueue explicitly distinguishes the accounting decision from placement:

<!-- source-excerpt: kueue-quota-not-pod-scheduling -->
> Quota reservation is sometimes referred to as workload scheduling or
> job scheduling, but it should not to be confused with pod scheduling.
<!-- /source-excerpt -->

*Kueue, [Quota Reservation](https://github.com/kubernetes-sigs/kueue/blob/bca571a3dae56ebca83109ae6769bade1c73dcfa/site/content/en/docs/concepts/_index.md#quota-reservation),
Apache-2.0. Original wording; emphasis and link markup removed.*

The following original configuration gives a ClusterQueue three quota
dimensions. It is a documentation example at the pinned source revision,
not a configuration applied to a cluster:

<!-- source-excerpt: kueue-basic-cluster-queue -->
```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: "cluster-queue"
spec:
  namespaceSelector: {} # match all.
  resourceGroups:
  - coveredResources: ["cpu", "memory", "pods"]
    flavors:
    - name: "default-flavor"
      resources:
      - name: "cpu"
        nominalQuota: 9
      - name: "memory"
        nominalQuota: 36Gi
      - name: "pods"
        nominalQuota: 5
```
<!-- /source-excerpt -->

*Kueue, [Cluster Queue](https://github.com/kubernetes-sigs/kueue/blob/bca571a3dae56ebca83109ae6769bade1c73dcfa/site/content/en/docs/concepts/cluster_queue.md),
Apache-2.0. Original YAML preserved verbatim.*

The admitted requests must fit all three dimensions simultaneously: at
most 9 CPUs, 36 GiB of requested memory and five pods. A proposed workload
that brings the totals to 8 CPUs, 40 GiB and four pods exceeds memory quota
even though CPUs and pod count fit. These are requested resources, not a
measurement of the processes' current consumption. Additional admission
checks and physical placement can still prevent a quota-fitting workload
from starting.

The example's empty `namespaceSelector` matches all namespaces. A shared
platform needs an explicit policy for which teams may submit through a
queue; the example is not an authorization policy for a real deployment.

If a job needs four GPUs in one node, two free GPUs on each of two nodes
do not satisfy its placement constraint. The existing
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration#Свободная память, свободные GPU и фрагментация|fragmentation example]]
works through this case. Topology-aware admission can incorporate physical
capacity and placement domains, but the actual startup still has to succeed.
Quota reservation alone neither loads weights nor establishes a healthy
distributed process group.

#### Borrowing quota without promising permanent capacity

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/platform-resources-2026-10-26/cluster-queue.svg]]

*Kubernetes Authors and Kueue contributors, [Cluster Queue — Flavors and Resources](https://github.com/kubernetes-sigs/kueue/blob/bca571a3dae56ebca83109ae6769bade1c73dcfa/site/content/en/docs/concepts/cluster_queue.md),
Apache-2.0. Complete original SVG. Dashed connections associate queue
quotas with ResourceFlavors; they are not network links between nodes.*

A ResourceFlavor can distinguish pools such as standard, spot or GPU
nodes. Sharing an unused CPU quota from a spot pool does not manufacture
an available GPU of the required model. Borrowing is accounted for by
resource and flavor; placement must match the resulting assignment.

<!-- source-excerpt: kueue-borrowing-example -->
> If ClusterQueue `team-b-cq` has no admitted Workloads, then ClusterQueue
> `team-a-cq` can admit Workloads with resources adding up to `12+9=21` CPUs
> and `48+36=84Gi` of memory.
<!-- /source-excerpt -->

*Kueue, [Borrowing example](https://github.com/kubernetes-sigs/kueue/blob/bca571a3dae56ebca83109ae6769bade1c73dcfa/site/content/en/docs/concepts/cluster_queue.md#borrowing-example),
Apache-2.0. Original first scenario, with list formatting removed. The
example has two queues, no separate shared quota and no borrowing/lending
limits.*

The nominal quotas in that example are A: 9 CPUs and 36 GiB, B: 12 CPUs
and 48 GiB. When B is idle, A can use the shared total. Setting A's CPU
`borrowingLimit` to 1 reduces A's CPU ceiling to 10, even with 12 idle
CPUs in B. Setting B's CPU `lendingLimit` to 1 can impose the same bound
from the donor side. Neither setting changes the number of physical CPUs.

The following calculation models only one resource, one flavor and two
queues at a fixed instant. A prospective allocation must fit A's nominal
quota plus its borrowing limit, the cohort's unused accounting total in
this simplified pool, and B's available lending allowance. It does not
implement Kueue's scheduler or preemption policy.

```python
def can_admit_two_queue(request, nominal_a, used_a, borrowing_a,
                        nominal_b, used_b, lending_b):
    values = [request, nominal_a, used_a, borrowing_a,
              nominal_b, used_b, lending_b]
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError('this example uses nonnegative integer resource units')
    if used_a + used_b > nominal_a + nominal_b:
        raise ValueError('existing use exceeds the modeled pool')
    next_a = used_a + request
    donor_unused = max(0, nominal_b - used_b)
    borrowed_a = max(0, next_a - nominal_a)
    return (next_a <= nominal_a + borrowing_a
            and next_a + used_b <= nominal_a + nominal_b
            and borrowed_a <= min(lending_b, donor_unused))

assert can_admit_two_queue(21, 9, 0, 12, 12, 0, 12)
assert can_admit_two_queue(10, 9, 0, 1, 12, 0, 12)
assert not can_admit_two_queue(11, 9, 0, 1, 12, 0, 12)
assert not can_admit_two_queue(11, 9, 0, 12, 12, 0, 1)
assert can_admit_two_queue(1, 9, 9, 1, 12, 0, 12)
assert not can_admit_two_queue(2, 9, 9, 1, 12, 0, 12)
assert not can_admit_two_queue(1, 9, 9, 12, 12, 12, 12)
```

With `used_a=0`, a request of 10 is the whole proposed allocation. With A
already using 9 CPUs and a borrowing limit of 1, only one further CPU fits. A vector
request must pass the corresponding check for every resource; unused
memory cannot compensate for a CPU quota violation.

A cohort can also own additional shared quota. Kueue's
[Cohort documentation](https://github.com/kubernetes-sigs/kueue/blob/bca571a3dae56ebca83109ae6769bade1c73dcfa/site/content/en/docs/concepts/cohort.md)
states that this quota is added to the ClusterQueues' nominal quotas;
it is not a second name for their sum. Hierarchical cohorts allow sharing
across a tree, subject to borrowing and lending limits. The two-queue
calculation above omits that extra pool and hierarchy deliberately.

Borrowed capacity can later be needed by its nominal owner. Whether and
which running jobs can be evicted depends on the configured reclaim and
preemption policies. A priority value does not on its own reclaim a GPU.
The platform must also allow the transition, and the job must tolerate
interruption. Kueue's WorkloadPriorityClass controls workload admission and
preemption independently of a pod's PriorityClass. Giving a batch job an
admission priority does not automatically assign that priority to its pods.

`StrictFIFO` and `BestEffortFIFO` both order workloads by priority and then
creation time. They differ when an earlier workload in that order does
not fit: StrictFIFO blocks later workloads, while BestEffortFIFO can admit
one that fits. BestEffortFIFO is the default in the pinned documentation.
Allowing smaller jobs through can improve packing; it does not give the
blocked large job a completion-time guarantee.

#### A GPU resource name does not specify its isolation

An advertised `nvidia.com/gpu` unit can represent an exclusive device,
a MIG instance under one advertisement strategy, or shared access under
a configured time-slicing policy. The resource name alone cannot tell
an application how much HBM it owns or how another process can affect it.

<!-- source-excerpt: nvidia-shared-access-not-proportional -->
> Unlike with "normal" GPU requests, requesting more than one shared
> GPU does not imply that you will get guaranteed access to a proportional
> amount of compute power. It only implies that you will get access to a
> GPU that is shared by other clients (each of which has the freedom to run
> as many processes on the underlying GPU as they want).
<!-- /source-excerpt -->

*NVIDIA, [Kubernetes device plugin — With CUDA Time-Slicing](https://github.com/NVIDIA/k8s-device-plugin/blob/1053da35d5a67fdbc3f0ea0877b7178026048769/README.md#with-cuda-time-slicing),
Apache-2.0. Original excerpt; the leading “Note” label is omitted.*

In the source's configuration example, eight physical GPUs with ten
advertised replicas each produce 80 schedulable resource units. They
remain eight physical GPUs. Requesting two shared units does not reserve
20% of one device's HBM or guarantee twice the compute of one unit.
Time-slicing in this device-plugin mode does not impose per-replica HBM
or hardware fault isolation. An ordinary process exit should not be
confused with a device-level fault that affects other users of the GPU.

<!-- source-excerpt: nvidia-mps-memory-limit -->
> In contrast with time-slicing, the amount of memory allowed per client
> (i.e. per partition) is managed by the MPS control daemon and limited to
> an equal fraction of the total device memory. In addition to controlling
> the amount of memory that each client can consume, the MPS control daemon
> also limits the amount of compute capacity that can be consumed by a client.
<!-- /source-excerpt -->

*NVIDIA, [Kubernetes device plugin — With CUDA MPS](https://github.com/NVIDIA/k8s-device-plugin/blob/1053da35d5a67fdbc3f0ea0877b7178026048769/README.md#with-cuda-mps),
Apache-2.0. Original excerpt. The pinned README marks MPS support
experimental and does not support this sharing mode on MIG-enabled devices.*

These guarantees apply to the pinned plugin implementation and configured
mode. They are not a specification for every CUDA version, virtualization
product or GPU-sharing extension.

| <span id="gpu-sharing-boundaries">Mode</span> | What the allocation controls | What still needs checking |
|---|---|---|
| Exclusive GPU | One workload receives the device allocation without configured replica sharing. | CPU, host memory, storage, network and node failures remain shared unless separately isolated. |
| Device-plugin time-slicing | Multiple advertised references permit workloads to share a physical GPU. | No per-replica HBM/fault isolation; process competition and SLO effects require measurement. |
| Device-plugin MPS | The control daemon limits each client's memory and compute allocation. | Support is version-dependent; software limits do not imply MIG's dedicated hardware partitions or a complete security boundary. |
| MIG GPU instance | A supported GPU exposes predefined hardware partitions with compute and memory resources. | The requested model must fit the profile; supported geometry, driver behaviour and remaining shared host/board resources still matter. |

For a numerical capacity check, suppose the usable application memory
budget is 40 GiB. Time-slicing ten replicas does not make ten 4 GiB memory
partitions. Two workloads with simultaneous peaks of 25 GiB each exceed
the budget. Under a four-client equal-memory MPS limit, the nominal
per-client allowance would instead be 10 GiB. A model needing 8 GiB for
weights plus 2.5 GiB for cache/workspace would exceed that allowance.
These are declared memory-budget examples, not measurements of an A100
or a promise that either configuration will meet a latency SLO.

MIG profiles also have geometry constraints. A model cannot pool two
independent small MIG instances into one larger device merely by summing
their advertised memory. It would need a supported distributed execution
configuration, with its communication and memory overhead accounted for.
Repartitioning a GPU is an operational transition; it can require stopping
workloads. The specific profiles and interruption requirements belong to
the chosen GPU and software version, not to the generic word “MIG.”

<!-- BEGIN REVIEW: mlops mig-detail 2026-10-26 -->
##### What a MIG partition separates

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mig-2026-10-26/miso-mps-vs-mig.png]]

*Baolin Li, Tirthak Patel, Siddharth Samsi, Vijay Gadepally and Devesh Tiwari,
[MISO, Figure 1 — MPS sharing mode and MIG sharing mode](https://arxiv.org/abs/2207.11428v3),
CC BY 4.0. Complete original vector figure rendered at 400 dpi; no crop,
translated labels or redrawing. The diagram is a resource-sharing
abstraction, not an A100 slice-placement map.*

On the left, the applications have different compute shares while cache
and memory remain shared. On the right, each application's colour extends
through both rows: App 1 occupies the first four drawn divisions, App 2
the next two and App 3 the last one. The additional boundaries in the
lower row depict the memory/cache separation between GPU instances.
The seven drawn lower boxes are schematic; they are not a count of
the A100's physical memory slices.

The left-hand compute bands also need a qualification. In NVIDIA's
[dynamic MPS provisioning](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html#dynamic-execution-resource-provisioning),
an active-thread limit caps a client's resources without reserving a
disjoint set of SMs; kernels from different clients may use the same SM.
Static SM partitioning is a separate mode in current MPS documentation.
Neither a dynamic thread cap nor a static SM partition creates the
dedicated memory-system paths of a MIG GPU instance. The pinned Kubernetes
device-plugin mode described above is not a contract for every MPS mode.

NVIDIA's [MIG terminology](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html#terminology)
distinguishes a GPU instance, GI, from a compute instance, CI. A GI combines
compute slices with dedicated memory-system resources. Its compute can be
subdivided into CIs, but those CIs share the parent GI's memory and engines.
Two CIs inside one GI therefore do not establish the memory boundary
between two separate GIs shown on the right of the figure.

For example, subdividing one `4g.20gb` GI into four one-compute-slice CIs
still leaves the parent GI's nominal 20 GB memory pool. It does not create
four independent 20 GB memory allocations. Memory capacity and bandwidth
remain shared within that GI even though its compute slices are divided.

For the A100-40GB example, the guide describes eight memory slices and
seven compute slices. Selected profiles make the different proportions
visible:

| <span id="mig-profile-slices">A100-40GB GI profile</span> | Compute slices | Memory slices | Nominal memory |
|---|---:|---:|---:|
| `1g.5gb` | 1 of 7 | 1 of 8 | 5 GB |
| `2g.10gb` | 2 of 7 | 2 of 8 | 10 GB |
| `3g.20gb` | 3 of 7 | 4 of 8 | 20 GB |
| `4g.20gb` | 4 of 7 | 4 of 8 | 20 GB |
| `7g.40gb` | 7 of 7 | 8 of 8 | 40 GB |

*Selected rows from the [NVIDIA A100 partitioning example](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html#partitioning),
cross-checked with MISO Table 1. These are profile names and nominal
capacities, not measured usable application memory or a complete profile
catalogue. A100-80GB and other GPU products have different profile names.*

Two `3g.20gb` GIs consume six compute slices but all eight memory slices.
The unused compute slice cannot support another `1g.5gb` GI: that profile
also needs a memory slice. Conversely, a sum that fits both slice totals
is only a necessary check; allowed placement still depends on the profile
geometry. The driver determines which instances can actually be created.

Version also changes the placement rules. MISO's 2022 text forbids
`4g.20gb` alongside `3g.20gb`; NVIDIA's guide states that this restriction
was removed starting with R510. The paper's count of 18 configurations is
not used here as a current, hardware-independent limit. Profile geometry
must come from the selected GPU and driver, not from that historical count.

##### A partition that fits can still be a poor assignment

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mig-2026-10-26/miso-workload-mix.png]]

*Li et al., [MISO, Figure 3](https://arxiv.org/abs/2207.11428v3), CC BY 4.0.
Original caption: “The system throughput of a workload mix with MPS sharing
(top two bars) and MIG sharing (bottom two bars) on A100s.” Complete source
vector figure rendered at 400 dpi without changing its data or layout.*

The four bars describe the same three-job mix: a CNN, an embedding workload
and an MLP. The top two use MPS with equal shares and approximately 4:2:1
shares. The third assigns the MIG profiles `4g`, `2g`, `1g` to the jobs;
the fourth assigns `2g`, `2g`, `3g`. In each bar, the contributions run from
CNN on the left through EMB to MLP on the right. Bar length is their sum,
so the comparison of totals does not depend on identifying the colours.

The third bar is longer than both MPS bars. The fourth is shorter than
the second MPS bar, despite also using MIG. The authors describe that
comparison explicitly:

<!-- source-excerpt: miso-configuration-dependent-performance -->
> We note that not all MIG configurations can outperform MPS configurations.
> A poorly-chosen MIG’s system throughput (e.g., a workload needing the
> smallest memory capacity but assigned the largest MIG slice) will
> underperform MPS. For example, the (57%, 29%, 14%) MPS partition outperforms
> the (2g, 2g, 3g) MIG partition.
<!-- /source-excerpt -->

*Li et al., [MISO, Section 3 — Motivation](https://arxiv.org/html/2207.11428v3#S3),
CC BY 4.0. Original wording and language; only TeX/Markdown formatting changes.*

Giving the MLP a larger slice does not compensate for the CNN's reduced
progress in this particular mix. The result supports comparing assignments
for a workload; it does not establish that enabling MIG always improves
performance.

The horizontal axis uses the paper's *system throughput*, STP. It sums
each job's progress rate relative to that same job running alone on the
full GPU:

$$
\mathrm{STP}=\sum_i\frac{q_i}{p_i}.
$$

Here $p_i$ is exclusive progress and $q_i$ is concurrent progress, measured
in the same units for job $i$. Different jobs need not share a progress
unit. For a separate teaching example, suppose exclusive rates are
100, 20 and 40 training steps per second, and concurrent rates are
80, 10 and 20. Their normalized contributions are 0.8, 0.5 and 0.5;
STP is 1.8. This number is neither 1.8 requests per second nor an 80%
reduction in each job's completion time.

```python
from fractions import Fraction

def normalized_progress(exclusive, concurrent):
    if not exclusive or len(exclusive) != len(concurrent):
        raise ValueError('nonempty matched job rates are required')
    if any(type(p) is not int or p <= 0 for p in exclusive):
        raise ValueError('exclusive rates must be positive integers')
    if any(type(q) is not int or q < 0 for q in concurrent):
        raise ValueError('concurrent rates must be nonnegative integers')
    return sum((Fraction(q, p) for p, q in zip(exclusive, concurrent)), Fraction())

assert normalized_progress([100, 20, 40], [80, 10, 20]) == Fraction(9, 5)
# Two jobs at full rate can hide a third job making no progress.
assert normalized_progress([100, 20, 40], [100, 20, 0]) == 2
```

The second assertion shows why STP alone cannot enforce fairness or a
per-job deadline. Queueing time, checkpoint/reconfiguration overhead and
the slowest jobs still affect end-to-end completion. In the paper, average
job completion time includes queueing; makespan is the interval from the
first job's start to the last job's completion. They answer different
questions from a sum of normalized instantaneous progress rates.

Figure 3 is a 2022 A100 training-workload comparison. The broader MISO
testbed has four nodes and eight A100-PCIe-40GB GPUs; its simulated
40-GPU results are separate experiments. The figure does not specify
enough architecture and batch details to reproduce this exact three-job
mix from the chart alone. Do not read it as an LLM serving p99 benchmark
or a measured result of our platform. For a live service, the co-location
acceptance case below still needs its own workload, errors, memory and
latency measurements.

*Original MISO figures retain CC BY 4.0. The added reading of the figures,
profile example and displayed calculation are Bookvar explanatory
material; no GPU experiment or MISO scheduler execution was performed.*

<!-- END REVIEW: mlops mig-detail 2026-10-26 -->

#### Tenants, namespaces and requests inside a service

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/platform-resources-2026-10-26/kubernetes-multi-tenancy.png]]

*The Kubernetes Authors, [Multi-tenancy — Tenants](https://github.com/kubernetes/website/blob/ed70e311737a341574ed5758b7af24e3bba232cb/content/en/docs/concepts/security/multi-tenancy.md#tenants),
CC BY 4.0. Complete original diagram of multi-team and multi-customer
tenancy in the same cluster. Boxes identify ownership; they do not depict
already-enforced network or hardware boundaries.*

A tenant may be a team operating several models, or a customer using one
application. Those definitions produce different boundaries. The platform
may isolate Team C from Team A while Team C's application still serves many
customers in one process. A namespace per team does not isolate the
requests, caches and logs of those customers inside that process.

<!-- source-excerpt: kubernetes-namespace-isolation-requires-controls -->
> The namespace isolation model requires configuration of several other
> Kubernetes resources, networking plugins, and adherence to security best
> practices to properly isolate tenant workloads. These considerations
> are discussed below.
<!-- /source-excerpt -->

*The Kubernetes Authors, [Multi-tenancy — Namespaces](https://github.com/kubernetes/website/blob/ed70e311737a341574ed5758b7af24e3bba232cb/content/en/docs/concepts/security/multi-tenancy.md#namespaces),
CC BY 4.0. Original paragraph.*

For two teams sharing a model registry, the control-plane check is whether
Team A's identity can read or replace Team B's model metadata. RBAC and
the registry's own authorization must enforce the intended policy. For
network isolation, the question is whether A's workload can reach B's
serving endpoint. A NetworkPolicy object needs a network implementation
that enforces it; the object alone does not establish a traffic boundary.
Allow/deny rules also do not reserve bandwidth or automatically encrypt
traffic. Storage and artifact access need their own permissions.

ResourceQuota limits namespace-level allocations and object counts.
Container resource limits address the resources they actually control;
a host-memory limit is not a GPU-HBM limit. A quota does not provide a
network-bandwidth reservation. Dedicated nodes can reduce co-location
interference while still sharing the API server, storage or physical
network. Virtual control planes isolate API resources further, but do
not themselves separate workloads on the worker nodes. Kubernetes'
[multi-tenancy guide](https://kubernetes.io/docs/concepts/security/multi-tenancy/)
discusses these boundaries separately.

An online LLM service also needs per-request admission. Kueue's batch
Workload admission is not a scheduler for each HTTP request arriving at
that service. At 10 requests per second, 100 input tokens per request
produce 1000 input tokens per second; 10,000 input tokens produce 100,000.
The request rate is unchanged while the token rate differs by 100×. This
does not imply a 100× latency ratio: batching, attention cost and cache
reuse affect the executed work.

The application must derive the tenant identity from authenticated context,
then apply its input/output limits, concurrency limits and queue budgets
to that identity. Otherwise a shared serving deployment can obey all of
its Kubernetes quotas while one customer fills its request queue or KV
cache. Cost metering can attribute the use afterward; it does not stop
the overload. Cache-access rules and logs must respect the same tenant
boundary as the request path.

Co-location therefore needs an explicit acceptance case: the model alone,
then with the intended neighbours and workload mix. Compare completed
request counts, errors, tail latency and peak memory; for an LLM, include
TTFT and inter-token timing. This specifies what to measure, not the
result of a test performed here. Billing allocation, GPU busy time and
the presence of quotas cannot supply that missing evidence.

*The Harvard-derived narrative and Bookvar examples in this section use
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Unchanged Kueue/NVIDIA excerpts retain Apache-2.0; the Kubernetes excerpt
and image retain CC BY 4.0. Source languages are preserved.*

<!-- END REVIEW: mlops platform-resources 2026-10-26 -->

### Посчитать повторяющуюся экономию

Рассмотрим 50 команд, каждая обслуживает одну модель.
Платформа экономит каждой команде 20 часов в месяц;
полная оценка часа равна 150 денежным единицам.
Стоимость платформы за тот же месяц — 120 000.
Труд оценивается в $50\cdot20\cdot150=150\,000$.

Отношение выгоды к затратам равно 1.25.
Если определить ROI как чистую выгоду, делённую на затраты,
получится $(150\,000-120\,000)/120\,000=25\%$.
Это разные показатели, а не два названия одного отношения.

```python
def platform_economics(teams, hours_saved, hourly_cost, monthly_cost):
    assert all(math.isfinite(x) for x in [teams, hours_saved, hourly_cost, monthly_cost])
    assert int(teams) == teams and teams >= 0
    assert hours_saved >= 0 and hourly_cost >= 0 and monthly_cost > 0
    benefit = teams * hours_saved * hourly_cost
    return benefit, benefit / monthly_cost, (benefit - monthly_cost) / monthly_cost

assert platform_economics(50, 20, 150, 120_000) == (150_000, 1.25, .25)
assert platform_economics(20, 20, 150, 120_000) == (60_000, .5, -.5)
```

Для 20 команд выгода равна 60 000, чистый ROI — −50%.
Если подключение платформы требует постоянной дополнительной работы,
её вычитают из сэкономленных часов или добавляют к затратам.
Учитывают инфраструктуру, дежурство, миграцию и поддержку.
Высвободившееся время не обязательно превращается в денежную
экономию бюджета; нужно показать, какую работу оно позволило выполнить.

Этот расчёт не устанавливает универсальное число моделей,
после которого необходима платформа. Общие зависимости и критичность
могут оправдать отдельные средства контроля раньше,
чем окупится полная платформа.

<!-- BEGIN REVIEW: mlops finops 2026-10-26 -->
## FinOps: lifecycle cost and the cost of a useful result

The platform calculation above values engineering time saved across teams.
A model's own budget also includes training runs, capacity kept online,
data processing and the work needed to maintain the service. Reducing one
line can increase another: a smaller serving bill may require more
engineering, and a cheaper interruptible training job may finish later.
Compare alternatives over the same period, with the same quality and
service requirements.

### Build a cost ledger before comparing models

Harvard groups lifecycle costs into four terms:

$$
C_{\mathrm{total}}=C_{\mathrm{train}}+C_{\mathrm{infer}}
                  +C_{\mathrm{data}}+C_{\mathrm{iter}}.
$$

The source defines the scope as follows:

<!-- source-excerpt: harvard-tco-definition -->
> Total Cost of Ownership (TCO) for ML Systems is the complete economic
> accounting of developing, deploying, and operating machine learning
> capabilities across their full lifecycle.
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [ML systems TCO framework](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L3817),
CC BY-NC-SA 4.0. Original wording, with emphasis and index markup removed.*

The categories need non-overlapping accounting rules. Here, training
includes the compute for initial training and later retraining; inference
includes allocated serving capacity, including idle replicas; data includes
acquisition, retained history, processing and transfer; iteration includes
engineering, validation, monitoring and incident response not already
charged elsewhere. If an ETL job's compute appears under data, it must
not appear a second time under training. If engineers' time appears under
iteration, adding the same labour as platform overhead would double count it.

A lifecycle comparison includes the initial investment and the future costs
within a declared horizon. A monthly operating report instead records that
month's charges and its stated treatment of commitments or depreciation.
Dividing a historical training bill across future months is an accounting
choice; it does not mean the cloud provider bills that training again.
Keep that choice consistent between candidates.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/finops-2026-10-26/harvard-tco-iceberg.png]]

*Harvard Edge ML Systems Book, [The TCO Iceberg](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L3847),
CC BY-NC-SA 4.0. Rendered from the complete, unchanged original plotting
cell at 300 dpi. Export bounds include labels outside the source canvas;
no data, artists or labels are changed. The eight
percentages are illustrative inputs, not measured industry budget shares.*

In this figure, GPU compute and object storage sum to 50%; the six other
rows sum to the remaining 50%. The red “waterline” separates categories
that the author calls visible and hidden. It does not identify costs that
are necessarily absent from an organization's accounts. Monitoring,
engineering and retraining can all have explicit budget lines. The figure
shows what a GPU-only estimate would omit under these chosen inputs.

Harvard's second worked scenario supplies monthly budgets for two
organizations. The amounts below retain all four source components;
ratios and shares are evaluated from the source constants.

| <span id="finops-budget">Component</span> | Startup, USD/month | Production company, USD/month | Second / first |
|---|---:|---:|---:|
| Training | 5,000 | 150,000 | 30× |
| Inference | 2,000 | 400,000 | 200× |
| Data | 500 | 80,000 | 160× |
| Iteration | 40,000 | 350,000 | 8.75× |
| Total | 47,500 | 980,000 | ≈20.63× |

*Harvard Edge ML Systems Book, [Worked example: Startup vs. production company TCO](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4099),
CC BY-NC-SA 4.0. Numerical source scenario, not a survey. The original
explanatory scaling-factor claims are not reproduced as measured causes.*

Iteration is $40{,}000/47{,}500\approx84.2\%$ of the first budget.
Inference is $400{,}000/980{,}000\approx40.8\%$ of the second and is its
largest single component, although still less than half. These calculations
tell the two hypothetical organizations where to investigate their own
spend. They do not establish that serving always dominates a mature
organization, that data costs grow superlinearly with users, or that
optimization caused the difference between these two chosen budgets.

### Allocate shared costs without losing the idle bill

An inference platform may keep four GPUs available while two teams submit
different amounts of work. The provider charges for allocated capacity;
the platform then needs a policy for assigning that charge to owners.
Harvard describes three policies in the original language:

<!-- source-excerpt: harvard-cost-attribution -->
> Cost optimization requires granular visibility into spending because
> teams cannot optimize costs they cannot see or own. Attribution policy
> is an incentive design problem, not only an accounting problem. Direct
> metering charges teams exactly for resources consumed; it is the most
> accurate model but can encourage under-provisioning when teams optimize
> their bill instead of service health. Allocation-based attribution
> charges based on reserved capacity rather than actual usage, making
> budgets predictable while potentially hiding waste. Hybrid attribution
> combines a base charge for allocation with a variable charge for excess
> usage, balancing predictability with efficiency incentives.
<!-- /source-excerpt -->

*Harvard Edge ML Systems Book, [Cost visibility and attribution](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L3733),
CC BY-NC-SA 4.0. Complete original paragraph; only line wrapping changes.*

“Exactly” in this paragraph depends on what the meter measures. A
per-process GPU-active counter does not on its own price shared HBM,
memory bandwidth, idle capacity, host resources or the reserve needed for
failures. Request counts measure another quantity again. A platform can
meter usage consistently and still need an explicit rule for these costs.
The FinOps Foundation's [Allocation capability](https://www.finops.org/framework/capabilities/allocation/)
separates ownership metadata from the policy for sharing common expenses.

For a worked calculation, allocate four identical GPUs for 100 hours at
an assumed USD 3 per allocated GPU-hour. This is a teaching rate, not a
current cloud offer. The pool contains 400 allocated GPU-hours and costs
USD 1200. A and B each reserve two GPUs. Suppose a simplified meter assigns
120 non-overlapping active GPU-hours to A and 60 to B. The remaining 220
GPU-hours are idle in this fixture; the example excludes concurrent kernels
and treats an active GPU-hour as a countable whole-device interval.

| <span id="finops-allocation">Policy</span> | Charge to A, USD | Charge to B, USD | Remaining central charge, USD |
|---|---:|---:|---:|
| Reserved capacity, equal reservations | 600 | 600 | 0 |
| Active intervals at USD 3/hour; idle kept centrally | 360 | 180 | 660 |
| Entire bill proportional to active intervals | 800 | 400 | 0 |
| Half the bill by reservation, half by active intervals | 700 | 500 | 0 |

The third policy assigns the idle bill in the same 2:1 ratio as measured
activity. It is an allocation rule, not evidence that A physically caused
two thirds of all costs. The hybrid in the last row first splits USD 600
equally, then splits the other USD 600 in a 120:60 ratio. Other hybrids
can use excess reservation or another documented basis; there is no unique
formula implied by the word “hybrid.”

```python
from fractions import Fraction

def allocate_shared(total, weights):
    # Integer currency units and integer weights; results may be fractional.
    if type(total) is not int or total < 0:
        raise ValueError('total must be a nonnegative integer')
    if not weights or any(type(w) is not int or w < 0 for w in weights):
        raise ValueError('weights must be nonnegative integers')
    if sum(weights) == 0:
        raise ValueError('a nonzero allocation basis is required')
    return [Fraction(total * w, sum(weights)) for w in weights]

pool_cost = 4 * 100 * 3
assert pool_cost == 1200
assert allocate_shared(pool_cost, [2, 2]) == [600, 600]
active_costs = [120 * 3, 60 * 3]
idle_cost = pool_cost - sum(active_costs)
assert active_costs == [360, 180] and idle_cost == 660
assert allocate_shared(pool_cost, [120, 60]) == [800, 400]
base = allocate_shared(600, [2, 2])
variable = allocate_shared(600, [120, 60])
assert [a + b for a, b in zip(base, variable)] == [700, 500]
```

With no active intervals, proportional allocation has no denominator.
Keep the bill in an explicitly owned idle or platform category, or use a
declared fallback such as reservation. Do not silently drop it. Fractional
currency allocations also need a rounding policy whose final charges
reconcile to the bill. The calculation above retains exact fractions;
it does not implement invoice rounding.

A usable cost record identifies the interval, resource pool, billed amount,
rate or commitment treatment, workload, model release, environment and
owner. Allocation metadata must survive a release or team rename. Record
unknown ownership as unknown until it is resolved; assigning it to whichever
team has the most recent request changes the accounting without evidence.
The same underlying charge may appear in views by team and by model, but
adding both views together would count it twice. Reporting costs to a team
is showback; actually transferring the charge to its budget is chargeback.

### Cost per completed request and per accepted result

Harvard defines cost per inference as total serving cost divided by
inferences served. Both numerator and denominator need the same reporting
period and service boundary. Include the capacity kept ready throughout
that period, not only the intervals when kernels run. If the comparison
claims full service cost, also include its stated CPU, storage, network,
observability and operating costs.

Suppose a service costs USD 1200 during a fixed period. It makes 1.2 million
attempts for one million logical requests because some requests are retried.
Of the logical requests, 900,000 complete within the service's SLO. A fully
labelled evaluation of that cohort finds 850,000 requests that both meet
the SLO and have an acceptable result. These are invented counts for
accounting, not a measured service or a statistical estimate of quality.

| <span id="finops-unit-cost">Denominator</span> | Count | USD per counted event |
|---|---:|---:|
| Attempts, including retries | 1,200,000 | 0.001000 |
| Logical requests received | 1,000,000 | 0.001200 |
| Logical requests completed within SLO | 900,000 | ≈0.001333 |
| Logical requests meeting both quality and SLO | 850,000 | ≈0.001412 |

Retries belong in the cost numerator even when their logical request
eventually fails. Counting each retry as another useful result would make
the first metric fall as retries increase at fixed cost, without improving
the customer's outcome. Likewise, HTTP 200 at stream start is not a
completed LLM response. Count the terminal outcome once per logical request;
the response contract is discussed in
[[02 Areas/ML & DL/06 Практика/14a HTTP streaming и жизненный цикл запроса|the HTTP SLO practice]].

```python
from fractions import Fraction

def cost_per_result(cost, results):
    if any(type(x) is not int or x < 0 for x in [cost, results]):
        raise ValueError('use nonnegative integer cost and result counts')
    return None if results == 0 else Fraction(cost, results)

cost = 1200
assert cost_per_result(cost, 1_200_000) == Fraction(1, 1000)
assert cost_per_result(cost, 900_000) == Fraction(1, 750)
assert cost_per_result(cost, 850_000) == Fraction(3, 2125)
assert cost_per_result(cost, 0) is None
# A cheaper candidate can still cost more per qualifying result.
assert cost_per_result(900, 400_000) > cost_per_result(1200, 850_000)
```

No qualifying results means there is no finite observed cost per qualifying
result; it does not mean zero cost. If quality labels arrive later, report
the measured completion/SLO cost and the separately matured quality cohort.
Estimate quality-adjusted cost only with a stated sampling and missing-label
procedure. Multiplying marginal quality and SLO rates would assume their
independence; the example instead counts their intersection directly.

For a fixed workload, measured useful throughput can be written as
$r=N_{\mathrm{good}}/H_{\mathrm{allocated}}$, in good results per allocated
GPU-hour. At a uniform GPU-hour rate $R$, the GPU component of unit cost is
$R/r$. Allocated hours already include idle time in that measurement;
dividing by another utilization factor would account for it again.
Unit cost has no denominator when there are no good results. Throughput
itself also needs a nonzero allocated interval. Extrapolation to another
load still needs a capacity test.

End-to-end request latency cannot substitute for allocated GPU time.
Consider a batch of eight requests running for one second on four GPUs:
it uses four GPU-seconds, or 0.5 GPU-seconds per request for that batch.
Two seconds of queueing raises each request's latency from one to three
seconds without adding GPU work to this particular batch. Other jobs may
use the devices while it waits. Do not multiply every request's wall time
by a device rate and then independently apply batch and utilization
corrections. The measured throughput, GPU count, queueing and reservation
interval must refer to the same execution model.

For LLMs, compare input length, generated length, cache hits and task mix
alongside request counts. A change in the proportion of long outputs can
change unit cost without a regression in the engine. Token counts are
useful workload descriptors and may be billing units; they do not uniquely
determine GPU time across different cache states, models and batches.
Quality, TTFT, inter-token timing and completions remain separate measures
as in [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLO и эксплуатация inference]].

### Training cost: energy, interruptions and billed time

Harvard's training model separates the non-energy device rate from the
facility energy term. With a fixed GPU count and $a$ denoting the energy
adder per GPU-hour, its structure is

$$
\begin{aligned}
a&=P_{\mathrm{kW}}e_{\mathrm{kWh}}\,\mathrm{PUE},\\
C&=N\,T\,(1+f)(R_{\mathrm{nonenergy}}+a).
\end{aligned}
$$

*Harvard Edge ML Systems Book, [Training cost model](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L3958),
CC BY-NC-SA 4.0. Notation compacted; the following numerical inputs are
Bookvar teaching assumptions, not the source's H100 example or market prices.*

Let $N=8$, baseline duration $T=10$ hours, non-energy rate USD 2 per GPU-hour,
accounted IT power 0.5 kW per GPU, electricity USD 0.10/kWh and PUE 1.2.
The facility energy term is $0.5\cdot0.10\cdot1.2=0.06$ USD per GPU-hour.
Suppose interruptions, restart and checkpoint work add 25% to the baseline
duration. The allocation lasts 12.5 hours, or 100 GPU-hours; its modeled
cost is $100\cdot(2+0.06)=206$ USD.

```python
from fractions import Fraction

gpus, baseline_hours = 8, 10
nonenergy_rate = Fraction(2)
power_kw, electricity_rate = Fraction(1, 2), Fraction(1, 10)
pue, extra_fraction = Fraction(6, 5), Fraction(1, 4)
energy_adder = power_kw * electricity_rate * pue
allocated_hours = gpus * baseline_hours * (1 + extra_fraction)
modeled_cost = allocated_hours * (nonenergy_rate + energy_adder)
assert energy_adder == Fraction(3, 50)
assert allocated_hours == 100 and modeled_cost == 206
# The same bill from actual allocated hours needs no second restart factor.
assert allocated_hours * Fraction(206, 100) == modeled_cost
```

The units follow directly: kW times USD/kWh gives USD/hour; multiplying
that by dimensionless PUE and GPU-hours gives currency. PUE applies to
energy, not to the entire device rental price. If an all-inclusive rental
rate already covers electricity and cooling, use that rate without adding
the same energy term again. Likewise, if the measured billed duration
already includes restarts, multiplying it by $1+f$ counts their time twice.

Here $f$ is extra time divided by baseline time. A different statement,
“20% of all allocated time is lost,” gives a useful fraction of 0.8 and
a duration multiplier $1/0.8=1.25$, not 1.20. A GPU TDP is also not a
measurement of average IT power. State what the power estimate includes
and add host/network costs only if they are not already covered.
Variable GPU counts or rates require summing the corresponding intervals.

Harvard's interruptible-compute example prices 100 useful GPU-hours at
USD 3 on demand or USD 0.90 on spot. With an assumed 20% *extra duration*,
spot costs $100\cdot1.2\cdot0.90=108$ USD rather than 90; the source rounds
this to about 110. These selected rates establish a scenario, not a
guaranteed discount. Capacity acquisition, a completion deadline, completed
checkpoints and a usable fallback can change the choice. The recovery
mechanics are already derived in
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration|Fault tolerance и fleet orchestration]].

### Quality constraints and the payback horizon

Harvard's cost-quality example uses the following input values:

| <span id="finops-cost-quality">Model</span> | Accuracy | Training cost, USD | Serving cost per 1000, USD |
|---|---:|---:|---:|
| Small | 92% | 500 | 0.10 |
| Medium | 95% | 5000 | 0.50 |
| Large | 96% | 50,000 | 2.00 |

*Harvard Edge ML Systems Book, [Cost-quality trade-offs](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L3752),
CC BY-NC-SA 4.0. First four columns of the source table, with units made
explicit. “Value Judgment” is omitted; the figures are illustrative inputs.*

Small to Medium buys three percentage points of accuracy for ten times
the training cost. Medium to Large buys one point for another tenfold
increase. At one million inferences, the stated serving rates yield
USD 100, 500 and 2000 respectively. These are serving-only amounts;
adding one training run yields USD 600, 5500 and 52,000 under the table's
limited cost boundary. They are not complete lifecycle totals.

If this hypothetical application requires accuracy of at least 95%, Small
is infeasible even though it is cheapest. Medium meets that aggregate
constraint at lower listed cost than Large. A different minimum, subgroup
requirement, latency limit or cost of a particular error can change the
decision. Accuracy divided by dollars does not encode those requirements.
The table supplies no serving hardware, latency distribution or uncertainty
interval, so it cannot establish a deployable winner on its own.

For an optimization investment, Harvard uses this constant-savings model:

$$
T_{\mathrm{payback}}=
\frac{C_{\mathrm{implementation}}}
     {S_{\mathrm{gross,month}}-C_{\mathrm{extra,month}}}.
$$

The source's simpler equation has no recurring extra cost. Its worked
example spends USD 80,000 to reduce an assumed USD 400,000 monthly
inference bill by 40%. Gross savings are USD 160,000/month and payback
is 0.5 months. A separate training optimization saves 30% of USD 150,000,
or USD 45,000/month; the same investment takes about 1.78 months to recover.
The percentages are expected savings in a scenario, not measured benefits
of quantization or a promise that one technique has priority everywhere.

*Harvard Edge ML Systems Book, [Worked example: Optimization investment decision](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/ops_scale/ops_scale.qmd#L4332),
CC BY-NC-SA 4.0. Source amounts retained; recurring-cost and rollout cases
below extend the original constant-savings calculation.*

```python
from fractions import Fraction

def payback_months(upfront, gross_monthly, extra_monthly=0):
    if any(type(x) is not int or x < 0
           for x in [upfront, gross_monthly, extra_monthly]):
        raise ValueError('costs must be nonnegative integer currency units')
    net = gross_monthly - extra_monthly
    return None if net <= 0 else Fraction(upfront, net)

assert payback_months(80_000, 160_000) == Fraction(1, 2)
assert payback_months(80_000, 45_000) == Fraction(16, 9)
assert payback_months(80_000, 160_000, 20_000) == Fraction(4, 7)
assert payback_months(80_000, 20_000, 20_000) is None

# Rollout case: actual monthly gross savings ramp up; new costs start at once.
cash = -80_000
balances = []
for gross in [0, 80_000, 160_000]:
    cash += gross - 20_000
    balances.append(cash)
assert balances == [-100_000, -40_000, 100_000]
```

Adding USD 20,000/month of maintenance to the immediate-savings case
leaves USD 140,000/month net and moves payback to about 0.57 months.
In the separate ramp-up case, savings start at zero, then reach half and
finally the full expected rate. The cash balance is still negative after
month two and positive after month three. The code checks month-end
balances; it does not locate a day within the third month or discount
future cash flows. Non-positive net savings provide no payback under
the constant-rate model.

A faster kernel produces monetary savings only when the organization can
reduce a charge or use the released capacity for work it otherwise needed
to fund. If four replicas must remain allocated for redundancy and peak
demand, a 20% latency reduction need not reduce that month's bill. An
unused reservation and a removed paid replica are different outcomes.
Before assigning savings to an optimization, compare accepted quality,
tail latency, retained capacity, rollout costs and the billing boundary on
the same workload. The cost model records the resulting saving; it cannot
supply the missing quality or capacity evidence.

*This Harvard-derived section and its Bookvar worked extensions use
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
English source explanations and numerical scenarios retain their language;
added accounting examples are identified separately from observations.*

<!-- END REVIEW: mlops finops 2026-10-26 -->

<!-- serving-operations-completion: readiness start -->
## ML Test Score: check the four areas separately

The support classifier can have reproducible training, a working registry
and tested rollbacks while nobody notices that predictions for one language
have degraded. Completing more infrastructure tests would not close that
monitoring gap. A readiness assessment must retain which part of the
system each result covers.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-operations-2026-10-29/mlops-stack-layers.svg]]

*Vijay Janapa Reddi and contributors, Harvard ML Systems,
[Development Infrastructure, Figure 5 — MLOps Stack Layers](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#sec-ml-operations-development-infrastructure-automation-de41),
[complete original SVG](https://mlsysbook.ai/vol1/ml_ops/ml_ops_files/mediabag/3665b88dc7480bf5936afba8999575d6c9eb2f29.svg),
CC BY-NC-SA 4.0. Original labels and geometry retained.*

The top row connects model/application, framework, orchestration,
infrastructure and hardware. The two middle columns separate model work
such as data management, evaluation and serving from resource work such
as scheduling, capacity and monitoring. The named products are examples
in the original figure, not prerequisites for the rubric below. A working
Kubernetes deployment does not test whether the model's features mean
the same thing at training and serving time; a correct PyTorch forward
does not test whether a failed rollout can be reversed.

Harvard groups the readiness questions as follows, retaining the source
language:

<!-- source-excerpt: harvard-readiness-four-areas -->
> - **Data section**: Validates feature expectations, privacy controls,
>   and whether each feature is beneficial relative to its operational cost.
> - **Model section**: Validates reviewed model specifications,
>   hyperparameter discipline, staleness limits, and offline-online metric alignment.
> - **Infrastructure section**: Validates reproducible training, rollback,
>   training-serving consistency, and deployment gates.
> - **Monitoring section**: Validates alerts for dependency changes,
>   data invariants, skew, and model staleness.
<!-- /source-excerpt -->

*Harvard ML Systems, [ML test score](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#sec-ml-operations-assessing-technical-debt-ml-test-score-0099),
CC BY-NC-SA 4.0. Complete four-item source passage; only wrapping changes.*

### A test needs an observable result

Harvard's representative checklist connects each area to concrete checks:

<!-- source-excerpt: harvard-readiness-selected-tests -->
| Category | Test | Implementation Example |
|---|---|---|
| Data Tests | Feature expectations are captured in schema | Great Expectations, TFX Data Validation |
| Data Tests | No feature's cost exceeds its benefit | Latency/accuracy trade-off analysis |
| Model Tests | Offline and online metrics are correlated | A/B test validation of offline improvements |
| Model Tests | Model staleness is measured and bounded | Performance decay monitoring |
| Infrastructure Tests | Model can be rolled back to previous version | Model registry with versioning |
| Infrastructure Tests | Training and serving code paths are tested for consistency | Feature store integration tests |
| Monitoring Tests | Dependency changes result in alerts | Data schema monitoring |
| Monitoring Tests | Training and serving features are not skewed | Training-serving skew detection |
<!-- /source-excerpt -->

*Harvard ML Systems, [ML Test Score Checklist](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_ops/ml_ops.qmd#tbl-ml-test-score),
CC BY-NC-SA 4.0. Eight selected rows; repeated category names are filled in
and table formatting simplified. This is not the original paper's full
28-test inventory. The listed tools are implementation examples, not proof
that the assertion has been tested.*

For `tickets_7d`, a schema checks that values are nonnegative integers
with the expected missing-value policy. The fixed-history test earlier
in the chapter checks the seven-day meaning. An ablation can evaluate
the model with and without the feature; request tracing can measure the
time and failure dependency added by its lookup. These checks answer
different questions. Passing the schema does not show that the feature
improves the model enough to justify the lookup.

Likewise, a registry proves that an old artifact is retained, not that
rollback works. A rollback test must resolve its dependencies, load it,
exercise the intended request path and verify the restored behaviour.
If the old model reads the incompatible `engagement:v2` feature, reverting
the weights alone fails the rollback condition already discussed above.

Monitoring requires a route from observation to action. In a test of the
support service, inject an invalid feature into an isolated fixture and
verify that the detector sees it, the rule changes state, the notification
reaches the expected test receiver and the runbook identifies the affected
release. Do not send test incidents to a real on-call receiver without an
agreed drill. A dashboard containing the metric establishes none of those
later steps by itself.

### Calculating the score without hiding a weak section

Breck et al.'s [original rubric, §VI-A](https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf#page=7)
has seven tests in each of four sections. A documented and shared manual
test earns 0.5 points; an automatically repeated test earns 1. Sum each
section separately, then take the minimum of the four sums. This rule
keeps one weak area from disappearing into a high total. The score is a
readiness rubric, not an estimate of the probability of a production failure.

Consider the following invented assessment. Each row contains seven test
scores; 0 means there is no qualifying evidence yet.

| Section | Seven test scores | Section sum |
|---|---|---:|
| Data | 1, 1, 1, 1, 1, 1, 1 | 7 |
| Model | 1, 1, 1, 1, 1, 1, 1 | 7 |
| Infrastructure | 1, 1, 1, 1, 1, 0, 0 | 5 |
| Monitoring | 0.5, 0.5, 0, 0, 0, 0, 0 | 1 |

The sum of all rows is 20 out of 28, but the rubric's score is 1.
Converting the two existing manual monitoring tests to repeated automated
tests raises that section to 2 and the final score to 2. Automating one
more infrastructure test instead raises its section to 6 while leaving
the final score at 1. The change can still be useful; the scalar score
does not record every improvement, so retain the per-test evidence and
the four section sums as well.

```python
def readiness_scores(sections):
    expected = {'data', 'model', 'infrastructure', 'monitoring'}
    if set(sections) != expected:
        raise ValueError('all four named rubric sections are required')
    if any(len(values) != 7 for values in sections.values()):
        raise ValueError('each section contains seven tests')
    if any(type(x) not in (int, float) or x not in (0, .5, 1)
           for values in sections.values() for x in values):
        raise ValueError('scores must be 0, 0.5 or 1')
    sums = {name: sum(values) for name, values in sections.items()}
    return sums, min(sums.values())

sections = {'data': [1] * 7, 'model': [1] * 7,
            'infrastructure': [1] * 5 + [0] * 2,
            'monitoring': [.5, .5, 0, 0, 0, 0, 0]}
sums, score = readiness_scores(sections)
assert sum(sums.values()) == 20 and score == 1
sections['infrastructure'] = [1] * 6 + [0]
assert readiness_scores(sections)[1] == 1
sections['monitoring'] = [1, 1, 0, 0, 0, 0, 0]
assert readiness_scores(sections)[1] == 2
```

The function checks arithmetic and the shape of the assessment. It cannot
decide whether a claimed automatic test runs on the actual deployed
candidate, whether a result is stale, or whether the test would detect
the intended failure. Store those facts with the result: assertion,
artifact and data versions, execution time, output, owner and response
to failure. The release checks at the start of this chapter still need
to bind evidence to `tickets-18` rather than accepting an older model's
green report.

For an LLM service, add risks not enumerated by this 2017 supervised-ML
rubric: tool authorization, prompt-injection boundaries, retrieval access
control and streaming cancellation, for example. Keep their gates explicit;
do not give them invented official ML Test Score points. No aggregate
score should waive a known blocking failure such as a rollback that
cannot restore a compatible model and feature pair.

*Harvard-derived excerpts, figure and Bookvar worked examples in this
section use CC BY-NC-SA 4.0. The original ML Test Score paper is linked
for its full rubric; its figures and full text are not republished here.*

<!-- serving-operations-completion: readiness end -->

<a id="mlops-maturity--свойства-а-не-список-покупок"></a>

## Какие свойства автоматизированы

Уровни Google 0, 1 и 2 различают ручной процесс, автоматическое
выполнение обучения и CI/CD реализации конвейера.
Это классификация автоматизации, не сертификат качества системы.
Наличие реестра или Kubernetes не отвечает на вопросы о семантике
признаков, наблюдении результата и возможности восстановления.

Для классификатора поддержки проверяемый результат выглядит так:
рабочий ответ связан с выпуском; файлы соответствуют описанию;
проверки относятся к этому кандидату; новая реплика не получает
трафик до готовности; canary наблюдается отдельно; поздние метки
присоединяются к исходной когорте; совместимый предыдущий путь
проверен и доступен.

Следующая глава рассматривает устройства, где управление версиями
и восстановление осложняются отсутствием постоянной связи,
ограниченной памятью и неодновременным обновлением парка.

## Практика и первоисточники

- Jarek Kazmierczak, Khalid Salama, Valentín Huerta,
  [MLOps: Continuous delivery and automation pipelines in machine learning](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning).
  Ручной процесс, CT, CI/CD, проверки данных и кандидатов, metadata;
  оригинальные Figures 2, 3 и 5.
- Alec Warner, Štěpán Davidovič et al.,
  [Canarying Releases](https://sre.google/workbook/canarying-releases/).
  Выделение кандидата и контроля, размер наблюдения, изоляция и сигналы выпуска.
- Steven Thurgood et al.,
  [Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/).
  Burn rate, несколько окон и ограничения при малом числе запросов.
- Kubernetes,
  [Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/).
- Sculley et al.,
  [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems).
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/ml_ops|Harvard ML Systems — полный исходный ML Operations]]
  и [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/ops_scale|ML Operations at Scale]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week07_application_deployment/Effdl26-07.pdf|Efficient DL Systems — полная лекция о развёртывании]]
  и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week07_application_deployment/03_metrics/README|стенд с метриками]]:
  оригинальные материалы без перевода. Стенд демонстрирует сервисные метрики,
  не независимую проверку качества классификации.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_14_ml_ops|Harvard MLOps lab — полный исходник]]:
  модели задержки наблюдения, порогов и воздействия отката. Его условные
  зависимости качества от drift не являются универсальными законами.
- Breck et al., [The ML Test Score](https://doi.org/10.1109/BigData.2017.8258038):
  дальнейшее чтение о проверках готовности ML-систем.
- Kreuzberger, Kühl and Hirschl,
  [Machine Learning Operations: Overview, Definition, and Architecture](https://doi.org/10.1109/ACCESS.2023.3262138):
  обзор ролей и архитектуры MLOps.
- [[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис|Практика развёртывания наблюдаемого сервиса]].

← [[01 ML workflow]] · Далее: [[03 Edge и federated deployment]]
