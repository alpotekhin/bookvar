---
title: "Жизненный цикл ML-системы: от постановки задачи до эксплуатации"
type: textbook-chapter
status: canonical
last_updated: 2026-10-27
last_verified: 2026-10-27
primary_sources:
  - https://developers.google.com/machine-learning/managing-ml-projects/pipelines
  - https://developers.google.com/machine-learning/guides/rules-of-ml
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd
  - https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment
---

# Жизненный цикл ML-системы: от постановки задачи до эксплуатации

В службе поддержки модель должна направлять сообщения о техническом сбое
инженерам, а вопросы об использовании продукта оставлять первой линии.
Для обучения есть история переписки и итоговые решения операторов.
Но в момент поступления нового обращения ещё нет ни ответа оператора,
ни окончательного маршрута. Если собрать обучающие примеры из полной истории,
можно получить высокую точность на данных, которых работающий сервис никогда
не увидит.

Чтобы проверить такую систему, недостаточно обучить классификатор.
Нужно восстановить вход в момент решения, определить метку, проверить
стоимость ошибок, пропускную способность инженерной группы и порядок
получения обратной связи. Все эти решения входят в жизненный цикл модели.

Далее служба поддержки используется как сквозной учебный пример.
Её числа и результаты условные.

<a id="два-параллельных-контура"></a>

## Контур предсказания и контур обновления

У работающей системы есть два разных вида работы. Первый выполняется
для пользовательского запроса: подготовить вход, вызвать модель,
применить правила продукта и вернуть результат. Второй создаёт следующую
версию: собрать данные, обучить кандидата, проверить его и решить,
можно ли заменить текущую модель.

На схеме Google первый вид работы выделен синим. Обновление проходит
через зелёный конвейер данных, красный конвейер обучения и фиолетовую
проверку кандидата. Между ними передаются не только файлы весов,
но и наборы данных, предсказания и версии артефактов.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/google-pipes.png]]

*Google, [ML pipelines, Figure 4](https://developers.google.com/machine-learning/managing-ml-projects/pipelines),
CC BY 4.0. Полная исходная схема без изменений. Serving обслуживает запросы;
Data, Training и Validation готовят и проверяют новую версию.
Выход `Best model` обозначает результат принятого правила сравнения,
а не гарантию, что модель лучше по любому возможному критерию.*

Для классификатора поддержки синий путь начинается с первого сообщения,
а заканчивается назначенной очередью. Журнал должен позволять выяснить,
какая версия модели и какое правило принятия решения использовались.
Позднее оператор подтверждает маршрут или исправляет его.
Эти сведения могут стать материалом для следующего обучения,
но сначала нужно проверить, что именно означает такое подтверждение.

Два контура работают с разной частотой. Предсказание может требоваться
за десятки миллисекунд, а надёжная метка появиться через несколько дней.
Переобучение не обязано происходить после каждого запроса.
Его частоту выбирают по изменению задачи, доступности меток, стоимости
обучения и результатам проверки кандидатов.

### Сначала провести данные через всю систему

Простой классификатор позволяет раньше проверить получение примеров,
предварительную обработку, экспорт и вызов модели в приложении.
В [Rules of ML, правило 4](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_4_keep_the_first_model_simple_and_get_the_infrastructure_right)
Мартин Зинкевич описывает, что должно совпасть в первом работающем конвейере:

<!-- source-excerpt: google-rules-4 -->
> Choosing simple features makes it easier to ensure that:
>
> - The features reach your learning algorithm correctly.
> - The model learns reasonable weights.
> - The features reach your model in the server correctly.
>
> Once you have a system that does these three things reliably, you have done
> most of the work. Your simple model provides you with baseline metrics and a
> baseline behavior that you can use to test more complex models.
<!-- /source-excerpt -->

*Martin Zinkevich, Google, Rules of Machine Learning, Rule 4, CC BY 4.0.
Слова и язык оригинала сохранены; изменена только Markdown-разметка.*

Для текстового маршрутизатора первым вариантом могут быть правила
по ключевым словам или TF–IDF с линейным классификатором.
Такой baseline нужен для сравнения качества и поведения всего пути.
Переход к encoder или LLM должен улучшать проверяемый результат,
а не только добавлять более сложный вызов модели.

<a id="1-постановка-задачи-как-системная-спецификация"></a>

## Что предсказывается и какое действие следует за ответом

Объектом предсказания в примере служит первое сообщение нового обращения.
Модель выдаёт оценку вероятности технического сбоя.
Приложение сравнивает её с порогом и выбирает очередь.
Правило метки, модель вероятности и порог являются разными частями системы.

Спецификация должна определить популяцию и момент решения.
Например, русскоязычные текстовые обращения без вложений обрабатываются
автоматически, а остальные идут по отдельному маршруту.
Если тестировать только эту популяцию, нельзя затем объявлять то же качество
для обращений с изображениями или многоязычной перепиской.
Доля запросов вне автоматического маршрута тоже входит в оценку продукта.

Решение связано с затратами. Пропущенный технический сбой задерживает помощь;
ошибочное направление обычного вопроса занимает время инженера.
Допустим, для учебного расчёта первая ошибка стоит пять условных единиц,
вторая одну. Это выбранная модель затрат, а не универсальные коэффициенты.

### Accuracy, стоимость ошибок и жёсткое ограничение

На validation-наборе из 10 000 обращений 1000 действительно требуют инженера.
Положительным классом считается технический сбой.

| Кандидат | TP | FN | FP | TN | Accuracy | Recall | Направлено инженерам |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 800 | 200 | 450 | 8550 | 93.5% | 80% | 1250 |
| B | 900 | 100 | 700 | 8300 | 92% | 90% | 1600 |
| C | 850 | 150 | 450 | 8550 | 94% | 85% | 1300 |

У A accuracy выше, чем у B, однако взвешенная стоимость ошибок выше:
$5\cdot200+450=1450$ против $5\cdot100+700=1200$.
Но если инженерная группа может обработать не более 1400 из этих
10 000 обращений, B не подходит по нагрузке.

Пусть дополнительно требуется recall не ниже 85%. Тогда A не проходит
по полноте, B по доступной мощности, C выполняет оба условия.
Низкую стоимость ошибок нельзя использовать, чтобы отменить жёсткое
ограничение на число обращений. На другой популяции или при других затратах
выбор может измениться.

```python
import hashlib
import json
import numpy as np

def triage_metrics(tp, fn, fp, tn, missed_cost=5., false_alarm_cost=1.):
    counts = [tp, fn, fp, tn]
    assert all(int(x) == x and x >= 0 for x in counts)
    total = sum(counts)
    assert total > 0 and tp + fn > 0
    assert missed_cost >= 0 and false_alarm_cost >= 0
    return {
        'accuracy': (tp + tn) / total,
        'recall': tp / (tp + fn),
        'engineer_load': tp + fp,
        'cost': missed_cost * fn + false_alarm_cost * fp,
    }

a = triage_metrics(800, 200, 450, 8550)
b = triage_metrics(900, 100, 700, 8300)
c = triage_metrics(850, 150, 450, 8550)
assert a['accuracy'] > b['accuracy'] and a['cost'] > b['cost']
assert [m['recall'] >= .85 and m['engineer_load'] <= 1400
        for m in [a, b, c]] == [False, False, True]
assert c['cost'] == b['cost'] == 1200
```

Это сравнение кандидатов на validation, не результат последнего независимого
теста и не доказательство эффекта в продукте. Порог и вариант выбирают до
оценки на финальном test-наборе. Если подбирать их по одному и тому же test,
он становится частью разработки и перестаёт давать независимую проверку.

Ограничения задержки, памяти и безопасности проверяют отдельно.
Для задержки задают границы пути, нагрузку и оборудование.
Запись «p99 ответа ≤80 ms при 300 запросах/с, включая подготовку входа»
имеет проверяемый смысл. Размер файла весов не является пиком памяти процесса.

<a id="2-сбор-данных-начинается-с-условий-эксплуатации"></a>

## Обучающий пример должен соответствовать моменту решения

В конвейере данных журнал приложения превращается в наборы для обучения
и проверки. Выгрузка таблицы обращений сама по себе ещё не задаёт обучающий
пример: нужно выбрать состояние переписки, признаки, метку и время их доступности.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/google-data-process.png]]

*Google, [ML pipelines, Figure 8](https://developers.google.com/machine-learning/managing-ml-projects/pipelines),
CC BY 4.0. Исходник без изменений. Из журнала приложения строятся
обучающие и проверочные наборы; их версии сохраняются в `Dataset storage`.*

Для одного обращения полезно различать три времени.

| Время | Что произошло | Для чего используется |
|---|---|---|
| Момент предсказания | Поступило первое сообщение | Определяет допустимую информацию во входе |
| Время доступности признака | Значение дошло до системы | Определяет, мог ли сервис реально его использовать |
| Время доступности метки | Результат проверен оператором | Определяет, можно ли включить метку в текущую выгрузку |

Ответ оператора, окончательная очередь и текст закрытого обращения
содержат информацию о результате. Они могут участвовать в создании метки,
но не во входе классификатора первого сообщения.

### Временная корректность признаков

Допустим, признак показывает число прошлых обращений пользователя.
В истории признака сохранены следующие записи:

| Время состояния | Доступно системе с | Значение |
|---:|---:|---:|
| 5 | 6 | 2 |
| 9 | 12 | 3 |
| 11 | 11 | 4 |

Для предсказания в момент 10 допустимо значение 2.
Состояние на момент 9 относится к прошлому, но дошло только в момент 12.
Последняя запись 4 относится к будущему.
Соединение с текущей таблицей пользователя подставит 4 и изменит задачу.

```python
def feature_as_of(rows, prediction_time):
    candidates = [
        row for row in rows
        if row['state_at'] <= prediction_time
        and row['available_at'] <= prediction_time
    ]
    assert candidates
    chosen = max(candidates, key=lambda row: (row['state_at'], row['available_at']))
    return chosen['value']

history = [
    {'state_at': 5, 'available_at': 6, 'value': 2},
    {'state_at': 9, 'available_at': 12, 'value': 3},
    {'state_at': 11, 'available_at': 11, 'value': 4},
]
assert feature_as_of(history, 10) == 2
assert feature_as_of(history, 12) == 4
```

Такое восстановление состояния называют point-in-time correct.
Для него проверяют и время события, и время, с которого
его результат был доступен конкретному пути обслуживания.
Для запоздавших исправлений и нескольких версий одного состояния
нужны явные правила выбора.

[Rules of ML, правила 29, 31 и 32](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew)
связывают эту проблему с training-serving skew: журналируют фактически
использованные признаки, учитывают меняющиеся таблицы и переиспользуют
код преобразований. Журналирование допустимо только в рамках выбранных
правил доступа и хранения; воспроизводимость не требует сохранять все
персональные тексты без ограничений.

### Метки, разбиение и исключённые примеры

Пусть метка определяется по работе с обращением в первые 72 часа.
Если выгрузка сделана в понедельник в 12:00, обращения после пятницы 12:00
ещё не прошли полный горизонт. Их незавершённость не означает отрицательный
класс. Даже у более старых обращений метка может отсутствовать из-за
задержки или сбоя; это отдельный статус, который нужно видеть в покрытии.

Определение «оператор отправил инженеру» тоже требует проверки.
Оператор мог следовать прежней модели или временно менять маршрутизацию
из-за перегрузки. Такая метка описывает действие процесса, не обязательно
независимую оценку технического содержания. Для проверки качества пригодны
явные правила разметки и выборочный независимый разбор сообщений.

Временное разбиение проверяет работу на последующем периоде.
Разбиение по пользователям или диалогам отвечает другому вопросу:
не пользуется ли модель повторяющимися историями из train.
Условия можно сочетать, но каждый фильтр меняет проверяемую популяцию.
Набор новых пользователей не заменяет тест будущего трафика
со смесью новых и возвращающихся пользователей.

Версия набора должна определять исходный snapshot, время выгрузки,
правило доступности признаков, версию разметки, разбиение, исключения
и преобразования. Для каждого исключения считают количество и причины.
Если после обновления пропали длинные обращения, улучшение общей метрики
может описывать более лёгкую выборку, а не лучшую модель.

<a id="3-разработка-модели--работа-с-несколькими-бюджетами"></a>

## Эксперимент и воспроизводимый артефакт

Обучение получает конкретный набор и выдаёт конкретного кандидата.
Сравнивать эксперименты можно, когда известно, что изменилось:
данные, модель, преобразования, параметры обучения или случайная
инициализация. Смена сразу нескольких компонентов не позволяет приписать
результат одному из них.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/google-train.png]]

*Google, [ML pipelines, Figure 9](https://developers.google.com/machine-learning/managing-ml-projects/pipelines),
CC BY 4.0. Исходник без изменений. Входом обучения служит версия
`Training dataset`, выходом кандидат `Trained model`.
Сохранение файла модели не является решением о выпуске.*

Для классификатора фиксируют данные и разбиение, код подготовки текста,
токенизатор, конфигурацию обучения и веса. Для воспроизведения пути
предсказания дополнительно нужны runtime, зависимости, порог и формат
ответа. Если две модели сравнивались с разными правилами очистки текста,
отчёт должен отражать и эту разницу.

Один seed не гарантирует побитового совпадения обучения.
Различия оборудования, порядка операций и недетерминированных kernels
могут менять результат. Различают восстановление запуска, повторяемость
его чисел в заданных условиях и устойчивость вывода на нескольких запусках.
Версия окружения помогает исследовать расхождения, но сама их не устраняет.

### Идентификатор выпуска включает больше, чем веса

В демонстрационном manifest ниже веса одинаковы, а порог меняется.
Это разные выпуски: они направят разное число обращений инженерам.
Отчёт о качестве одного порога нельзя автоматически приписать другому.

```python
def release_id(manifest):
    payload = json.dumps(
        manifest, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

release = {
    'weights': 'weights-snapshot-17',
    'tokenizer': 'tokenizer-snapshot-3',
    'preprocessing': 'text-preparation-v5',
    'label_policy': 'technical-incident-72h-v2',
    'training_data': 'tickets-train-v12',
    'validation_data': 'tickets-validation-v4',
    'runtime': 'environment-lock-v6',
    'decision_threshold': .7,
}
changed_threshold = {**release, 'decision_threshold': .8}
changed_tokenizer = {**release, 'tokenizer': 'tokenizer-snapshot-4'}
assert release_id(release) != release_id(changed_threshold)
assert release_id(release) != release_id(changed_tokenizer)
assert release_id(release) == release_id(dict(reversed(list(release.items()))))
```

В примере значения `snapshot-*` являются учебными именами.
В рабочем manifest они должны разрешаться в неизменяемые объекты,
commit или контрольные суммы содержимого. Хеш manifest идентифицирует
описание, но не проверяет, что файлы действительно соответствуют ему,
что запуск воспроизводим или что артефакт подписан доверенным владельцем.

Для LLM-приложения зависимостями выпуска могут быть также chat template,
системный prompt, параметры генерации, схемы инструментов и код harness.
Для RAG добавляются версия корпуса, индекс, retriever и правила сборки
контекста. Неизменные веса не означают неизменную систему ответов.

<!-- BEGIN REVIEW: workflow iteration 2026-10-26 -->
## How many useful experiments fit in the development window?

An experiment finishes when its result can guide the next decision. Training
may take an hour while data preparation, evaluation and review take two days.
Speeding up the training call alone does not make that whole loop hourly.

Harvard's *Quantifying the ML lifecycle* compares a slower model starting at
95% accuracy with a faster model starting at 90%. The example assumes a fixed
accuracy gain per useful experiment and a common 99% ceiling. These are chosen
simulation inputs, not measured learning curves. The original parameter and
calculation block is retained below; only its class indentation is removed.

<!-- BEGIN SOURCE: Harvard iteration calculation -->
```python
weeks_total = 26
hours_per_week = 168

# Large Model (Starts better, iterates slow)
large_start_acc = 95.0
large_gain_per_iter = 0.15  # Reduced from 0.5: SOTA gains are hard!
large_cycle_time_hours = 168  # 1 week

# Small Model (Starts worse, iterates fast)
small_start_acc = 90.0
small_gain_per_iter = 0.1
small_cycle_time_hours = 1  # 1 hour
small_effective_iters = 100  # Realistic cap on useful experiments

# ┌── 2. EXECUTE (The Compute) ─────────────────────────────────────────
# Step 1: Large model: 1 iter/week -> 26 iters
large_iters = weeks_total * (hours_per_week / large_cycle_time_hours)
large_final_acc = min(large_start_acc + (large_iters * large_gain_per_iter), 99.0)

# Small model: Potential thousands, but capped by human ability to generate ideas
# Step 2: We allow small model to reach same ceiling
small_uncapped_acc = small_start_acc + (small_effective_iters * small_gain_per_iter)
small_final_acc = min(small_uncapped_acc, 99.0)
```
<!-- END SOURCE: Harvard iteration calculation -->

*Vijay Janapa Reddi and contributors, [ML Workflow, IterationTax calculation](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd#L333),
CC BY-NC-SA 4.0. Original code fragment; dedented, not translated.*

There are 26 × 168 = 4,368 calendar hours. Weekly experiments permit 26
iterations. The faster model has 4,368 nominal hourly slots, but the example
allows only 100 useful experiments: access to compute does not supply unlimited
new hypotheses. Its assumed final accuracy is min(90 + 100 × 0.1, 99) = 99%.
The slower model reaches 95 + 26 × 0.15 = 98.9%.

| Assumptions over the same 26 weeks | Useful experiments | Assumed gain per experiment | Calculated final accuracy |
|---|---:|---:|---:|
| Source: weekly cycle, start 95% | 26 | 0.15 percentage points | 98.9% |
| Source: hourly cycle, start 90% | 100 | 0.10 percentage points | 99.0%, after the chosen cap |
| Counterexample: hourly cycle, start 90% | 100 | 0.03 percentage points | 93.0% |

The last row changes one assumption. Faster feedback creates more
opportunities to improve a model; it does not determine how useful those
opportunities will be. The source code adds fixed gains rather than
compounding percentages. Its final 0.1 percentage-point difference is not a
statistical comparison on an independent test set.

The complete cycle time sets the number of available slots; the experiment
budget limits how many slots can be used productively. This generalization
reproduces the source's two cases and the counterexample. Start and ceiling
are percentages; gain_pp is an additive change in percentage points.

```python
import math

def iteration_budget(start, gain_pp, cycle_hours, horizon_hours, useful_limit,
                     ceiling=99.):
    assert all(math.isfinite(v) for v in
               [start, gain_pp, cycle_hours, horizon_hours, useful_limit, ceiling])
    assert 0 <= start <= ceiling <= 100 and gain_pp >= 0
    assert cycle_hours > 0 and horizon_hours >= 0
    assert useful_limit >= 0 and int(useful_limit) == useful_limit
    slots = math.floor(horizon_hours / cycle_hours)
    useful = min(slots, int(useful_limit))
    return useful, min(start + useful * gain_pp, ceiling)

assert iteration_budget(95, .15, 168, 26 * 168, 100) == (26, 98.9)
assert iteration_budget(90, .10, 1, 26 * 168, 100) == (100, 99.)
assert iteration_budget(90, .03, 1, 26 * 168, 100) == (100, 93.)
assert iteration_budget(90, .10, 1, 0, 100) == (0, 90.)
```

For a real experiment log, record the duration of data preparation, training,
evaluation and the wait for review separately. Keep the failed and inconclusive
runs as well as the selected candidate. Otherwise the reported iteration rate
can count only successful training jobs while hiding the actual bottleneck.
The fixed gains above remain a planning illustration, not a forecast that a
smaller model will overtake a larger one.

*This Harvard-derived extension and its added example are distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*
<!-- END REVIEW: workflow iteration 2026-10-26 -->

<!-- BEGIN REVIEW: model export 2026-10-27 -->
## Exporting a model does not export the application

A saved `state_dict` supplies parameter and buffer values to compatible model
code. An exported graph also describes the operations to execute, their tensor
inputs and their outputs. Neither artifact automatically includes the ticket
router's text preparation, label meanings or decision threshold. Those remain
part of the release described above.

Harvard's *Framework Selection* states the feasibility rule before comparing
performance:

<!-- BEGIN SOURCE: Harvard framework constraints -->
> Rather than seeking the "best" framework, effective selection first eliminates candidates that cannot satisfy hard constraints: deployment target, required operations, compiler path, and team expertise. Only then should soft preferences such as performance, development speed, and ecosystem rank the survivors.
<!-- END SOURCE: Harvard framework constraints -->

*Vijay Janapa Reddi and contributors, [ML Frameworks, Framework selection constraints](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L4193),
CC BY-NC-SA 4.0. Complete original callout text; not translated.*

For a PyTorch model deployed through ONNX, there are several separate
compatibility questions. Can the exporter capture the computation and translate
its operators? Can the selected runtime load that ONNX version and find kernels
for those operators, dtypes and shapes? Does execution preserve the outputs
within the chosen tolerance? Does the complete application meet its resource
and quality requirements? Passing an earlier check does not answer the later
ones.

### From PyTorch operations to an ONNX graph

The PyTorch documentation describes what its `torch.export`-based exporter
captures:

<!-- BEGIN SOURCE: PyTorch ONNX capture -->
`torch.export` engine is leveraged to produce a traced graph representing only the Tensor computation of the function in an
Ahead-of-Time (AOT) fashion. The resulting traced graph (1) produces normalized operators in the functional
ATen operator set (as well as any user-specified custom operators), (2) has eliminated all Python control
flow and data structures (with certain exceptions), and (3) records the set of shape constraints needed to
show that this normalization and control-flow elimination is sound for future inputs, before it is finally
translated into an ONNX graph.
<!-- END SOURCE: PyTorch ONNX capture -->

*PyTorch contributors, [torch.onnx, torch.export-based ONNX Exporter](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/onnx.md#torchexport-based-onnx-exporter),
v2.14.1, [BSD-style license](/licenses/pytorch-v2.14.1-LICENSE.txt).
Complete explanatory paragraph; the Sphinx reference is rendered as an API name.*

The graph has named tensor inputs, tensor types, dimensions, operators and
initializers such as weights. An opset import identifies the operator definitions
the graph uses; it is not a list of kernels installed on a device. A custom
operator needs both a representation in the exported graph and an implementation
in the destination runtime. Exporting its name does not supply that implementation.

The graph's shape metadata also needs inspection. A symbolic batch dimension
can permit different batch sizes without preserving every range constraint from
the originating `ExportedProgram`. The example below checks this boundary rather
than assuming that a captured constraint becomes a runtime input validator.
Data-dependent Python branches and graph capture failures have a separate
worked treatment in [[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|the kernels and compilation chapter]].

### Which provider executes each part?

ONNX Runtime assigns supported parts of a graph to execution providers. The
original documentation explains the sequence:

<!-- BEGIN SOURCE: ORT architecture flow -->
The flow is quite simple.

1. Starting from an ONNX model, ONNX Runtime first
converts the model graph into its in-memory graph representation.
2. It performs a set of provider independent [optimizations](https://onnxruntime.ai/docs/performance/graph-optimizations.html).
3. It partitions the graph into a set of subgraphs based on the available execution providers.
4. Each subgraph is assigned to an execution provider. We ensure that a subgraph can be executed by an execution provider by querying the capability of the execution provider using the `GetCapability()` API.
<!-- END SOURCE: ORT architecture flow -->

*Microsoft and ONNX Runtime contributors, [Architecture, High-level system architecture](https://github.com/microsoft/onnxruntime/blob/e3d000e3fe1b113c4153e6250dd605c2d03da3f0/docs/reference/high-level-design.md#high-level-system-architecture),
[MIT](/licenses/onnxruntime-e3d000e3-MIT.txt).
Complete original sequence; only the relative link is resolved.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/model-export-2026-10-27/onnx-runtime-providers.png]]

*Microsoft and ONNX Runtime contributors, [Execution Providers, GetCapability diagram](https://github.com/microsoft/onnxruntime/blob/e3d000e3fe1b113c4153e6250dd605c2d03da3f0/docs/execution-providers/index.md),
[MIT](/licenses/onnxruntime-e3d000e3-MIT.txt). Complete original PNG, 1454×910,
without crop or redrawing. Follow the top arrow from the model to the partitioner;
the provider registry describes which backend can execute each supported region.
The diagram does not imply that one inference request is automatically split
across a multi-GPU cluster.*

For a session configured with `['CUDAExecutionProvider', 'CPUExecutionProvider']`,
the first provider has priority for regions it can execute. Remaining supported
regions may run on CPU. Their inputs and outputs can require device transfers
and synchronization. The resulting latency depends on the partition boundaries
as well as the kernels. A successful GPU-provider registration does not establish
that every node ran there. Conversely, fallback cannot rescue an operator for
which no registered provider has an implementation.

The provider list, its options, runtime build and installed accelerator libraries
therefore belong in the release record. A CPU run establishes CPU behavior for
that graph; it does not establish CUDA, TensorRT or mobile-delegate compatibility.
The diagram's hardware choices are possible backends, not measured equivalent
performance results.

### A complete CPU export with a checked input contract

Consider a two-feature model with two outputs:

$$
W=\begin{bmatrix}2&-1\\1&3\end{bmatrix},\qquad
b=\begin{bmatrix}0.5&-0.5\end{bmatrix}.
$$

Each input row produces $\operatorname{ReLU}(xW^T+b)$. For $x=[1,2]$, the
two pre-activation values are $0.5$ and $6.5$; ReLU preserves both. For
$x=[3,-2]$, they are $8.5$ and $-3.5$, so the output is $[8.5,0]$.
These chosen weights make the arithmetic inspectable. This is not a trained
ticket classifier or an accuracy experiment.

The program uses PyTorch 2.14.1, ONNX 1.23.2, ONNX Script 0.7.2,
ONNX Runtime 1.31.1 and NumPy 2.3.5. `dynamo=True` selects the export path
described above. The application accepts float32 matrices with two features
and batches from 2 to 8. Export uses two example rows, but the exported graph
is tested on every batch size allowed by that contract.

<!-- checked-example: model-export-cpu -->
```python
import tempfile
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch

class TwoFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[2., -1.], [1., 3.]]))
            self.fc.bias.copy_(torch.tensor([.5, -.5]))

    def forward(self, features):
        return torch.relu(self.fc(features))

model = TwoFeatureModel().eval()
sample = torch.tensor([[1., 2.], [3., -2.]])
batch = torch.export.Dim("batch", min=2, max=8)
program = torch.onnx.export(
    model, (sample,), dynamo=True, opset_version=18,
    input_names=["features"], output_names=["scores"],
    dynamic_shapes={"features": {0: batch}},
)

def checked_scores(session, features):
    if not isinstance(features, np.ndarray) or features.dtype != np.float32:
        raise ValueError("features must be a float32 NumPy array")
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must have shape [batch, 2]")
    if not 2 <= features.shape[0] <= 8:
        raise ValueError("batch must be between 2 and 8")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    return session.run(["scores"], {"features": features})[0]

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "two-features.onnx"
    program.save(path, external_data=False)
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_inputs()[0].shape == ["batch", 2]
    assert session.get_outputs()[0].name == "scores"

    actual = checked_scores(session, sample.numpy())
    np.testing.assert_allclose(actual, [[.5, 6.5], [8.5, 0.]], rtol=0, atol=0)
    generator = np.random.default_rng(27)
    for size in range(2, 9):
        features = generator.normal(size=(size, 2)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(features)).numpy()
        np.testing.assert_allclose(checked_scores(session, features), expected,
                                   rtol=1e-6, atol=1e-6)

    # This ONNX graph executes outside the captured batch range.
    outside = np.ones((9, 2), dtype=np.float32)
    assert session.run(["scores"], {"features": outside})[0].shape == (9, 2)
    try:
        checked_scores(session, outside)
    except ValueError:
        pass
    else:
        raise AssertionError("the application accepted batch 9")
```

`onnx.checker` checks the graph's structural and type rules. Creating a session
then checks that this runtime can load and prepare it. The calls for batches
2 through 8 check numerical agreement on those particular inputs. The explicit
two-row calculation checks the interpretation of rows, weights and outputs
independently of the exporter.

Batch 9 demonstrates a separate boundary. This ONNX graph has a symbolic batch
dimension and executes nine rows, although the originating export was given
the range 2–8. The application wrapper rejects nine rows deliberately. Shape
constraints recorded during capture are not a substitute for validating the
application's input contract. The wrapper also rejects float64, wrong rank,
wrong feature width and non-finite inputs before calling the runtime.

For this small float32 graph, a tolerance of `1e-6` is chosen for the random
comparisons. That value is not a universal tolerance for quantized networks,
long reductions or another provider. A changed precision or backend requires
new numerical and task-level checks. Neither these comparisons nor the
checker prove equivalence for every possible input.

### A valid graph can still have no executable kernel

An ONNX file may refer to a custom operator in its own domain. In this example,
`org.bookvar::MissingKernel` has an input and output type and an imported
domain version. The checker accepts the model, but the session cannot prepare
it because no implementation is registered:

<!-- checked-example: model-export-missing-kernel -->
```python
from onnx import TensorProto, helper

custom_graph = helper.make_graph(
    [helper.make_node("MissingKernel", ["x"], ["y"], domain="org.bookvar")],
    "missing-kernel-example",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
)
custom_model = helper.make_model(
    custom_graph, opset_imports=[helper.make_opsetid("", 18),
                               helper.make_opsetid("org.bookvar", 1)],
    ir_version=10,
)
onnx.checker.check_model(custom_model, full_check=True)
try:
    ort.InferenceSession(custom_model.SerializeToString(),
                         providers=["CPUExecutionProvider"])
except ort.capi.onnxruntime_pybind11_state.Fail as error:
    assert "MissingKernel" in str(error) and "not" in str(error).lower()
else:
    raise AssertionError("an unregistered custom kernel unexpectedly loaded")
```

This failure is not evidence that the model's mathematical result is wrong;
execution never reached an output. The remedy may be a supported decomposition,
a compatible custom implementation, or another runtime. Renaming the file or
adding CPU fallback does not implement the missing operation.

The published artifact should consequently bind the exported graph and any
external weight files to their hashes, operator imports, input/output schema,
preprocessing and postprocessing versions, runtime/provider configuration and
the exact validation report. ONNX export also does not supply the request
scheduler or KV-cache manager described in [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines]].
Those components still need their own compatible model implementation.
<!-- END REVIEW: model export 2026-10-27 -->

<!-- BEGIN REVIEW: TensorFlow SavedModel 2026-10-27 -->
## TensorFlow: saved graphs, variables and callable signatures

TensorFlow 2 executes operations eagerly by default. A `tf.function` can capture
TensorFlow computation in a graph, so the same program can have an eager path
for development and a graph path for execution or export. Calling TensorFlow a
“static production framework” leaves out this choice. Creating a graph also does
not, by itself, promise XLA compilation, a particular device placement or a speedup.

The TensorFlow guide describes the graph representation directly:

<!-- BEGIN SOURCE: TensorFlow graph definition -->
Graphs are data structures that contain a set of `tf.Operation` objects, which represent units of computation; and `tf.Tensor` objects, which represent the units of data that flow between operations. They are defined in a `tf.Graph` context. Since these graphs are data structures, they can be saved, run, and restored all without the original Python code.
<!-- END SOURCE: TensorFlow graph definition -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tensorflow-export-2026-10-27/two-layer-network.png]]

*The TensorFlow Authors, [Introduction to graphs and tf.function, What are graphs?](https://github.com/tensorflow/docs/blob/9beed83c5fe8e4ccb742762394b084cd9e2bf2e0/site/en/guide/intro_to_graphs.ipynb),
[Apache-2.0](/licenses/tensorflow-docs-9beed83c-Apache-2.0.txt).
The complete original TensorBoard screenshot shows the guide's two-layer
network, not the scalar example below. Follow the arrows upward through
MatMul, add and ReLU, then through the corresponding second-layer operations.
Some node names are abbreviated in the original; this picture illustrates
dependencies, not a complete listing of the serialized file.*

### What survives saving and loading?

The following is the original custom-module example from *Using the SavedModel
format*. The first small setup block supplies its imports and a temporary
directory. The class, save call and load calls retain the source code and comments.
The programs here were checked with TensorFlow 2.21.0 on CPU.

```python
import os
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

savedmodel_temporary = tempfile.TemporaryDirectory()
tmpdir = savedmodel_temporary.name
```

<!-- BEGIN SOURCE: TensorFlow custom module -->
```python
class CustomModule(tf.Module):

  def __init__(self):
    super(CustomModule, self).__init__()
    self.v = tf.Variable(1.)

  @tf.function
  def __call__(self, x):
    print('Tracing with', x)
    return x * self.v

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def mutate(self, new_v):
    self.v.assign(new_v)

module = CustomModule()
```
<!-- END SOURCE: TensorFlow custom module -->

<!-- BEGIN SOURCE: TensorFlow saved functions -->
When you save a `tf.Module`, any `tf.Variable` attributes, `tf.function`-decorated methods, and `tf.Module`s found via recursive traversal are saved. (See the [Checkpoint tutorial](https://www.tensorflow.org/guide/checkpoint) for more about this recursive traversal.) However, any Python attributes, functions, and data are lost. This means that when a `tf.function` is saved, no Python code is saved.

If no Python code is saved, how does SavedModel know how to restore the function?

Briefly, `tf.function` works by tracing the Python code to generate a ConcreteFunction (a callable wrapper around `tf.Graph`). When saving a `tf.function`, you're really saving the `tf.function`'s cache of ConcreteFunctions.

To learn more about the relationship between `tf.function` and ConcreteFunctions, refer to the [tf.function guide](https://www.tensorflow.org/guide/function).
<!-- END SOURCE: TensorFlow saved functions -->

<!-- BEGIN SOURCE: TensorFlow save scalar -->
```python
module_no_signatures_path = os.path.join(tmpdir, 'module_no_signatures')
module(tf.constant(0.))
print('Saving model...')
tf.saved_model.save(module, module_no_signatures_path)
```
<!-- END SOURCE: TensorFlow save scalar -->

The call with `tf.constant(0.)` creates a scalar float32 specialization of
`__call__`. The Python `print` executes while tracing; it is not an instruction
that prints on every later graph execution. `mutate` declares its scalar
`input_signature` explicitly. TensorFlow can obtain that graph while saving.
The saved state starts with `v=1`.

<!-- BEGIN SOURCE: TensorFlow load scalar -->
```python
imported = tf.saved_model.load(module_no_signatures_path)
assert imported(tf.constant(3.)).numpy() == 3
imported.mutate(tf.constant(2.))
assert imported(tf.constant(3.)).numpy() == 6
```
<!-- END SOURCE: TensorFlow load scalar -->

*The TensorFlow Authors, [Using the SavedModel format, Saving a custom model and Loading and using a custom model](https://github.com/tensorflow/docs/blob/9beed83c5fe8e4ccb742762394b084cd9e2bf2e0/site/en/guide/saved_model.ipynb),
cells 31–37, [Apache-2.0](/licenses/tensorflow-docs-9beed83c-Apache-2.0.txt).
The three code cells and the complete saved-functions explanation are retained;
only the two notebook-relative links are resolved. The explanatory paragraphs
between code cells are Bookvar additions.*

The loaded object runs the saved graph and reads its restored variable. Its
first result is $3\cdot1=3$. After the saved mutation function assigns 2, the
same graph produces $3\cdot2=6$. Loading did not call `CustomModule.__init__`
to build a fresh `v=1` instance. Changing this loaded variable also does not
rewrite the files from which it was loaded.

The exported scalar graph cannot accept a vector just because Python's original
function could have multiplied one. Before export, the polymorphic function can
trace additional input specializations. After loading, it needs a matching saved
graph. This is the source example's failure, checked without depending on a
particular version's full exception wording:

```python
assert len(imported.signatures) == 0
try:
    imported(tf.constant([3.]))
except ValueError as error:
    assert "matching" in str(error).lower()
else:
    raise AssertionError("the scalar-only SavedModel accepted a vector")
```

An empty `signatures` map does not mean that the loaded object has no callable
methods: the preceding `imported(...)` and `imported.mutate(...)` calls worked.
The map names externally exported endpoints, whereas those attributes restore
the module's saved functions. A serving application should inspect the endpoint
it actually intends to call.

The source also demonstrates differentiation through the restored function.
Starting from the loaded `v=2`, it minimizes $(10-2v)^2$:

<!-- BEGIN SOURCE: TensorFlow imported training -->
```python
optimizer = tf.keras.optimizers.SGD(0.05)

def train_step():
  with tf.GradientTape() as tape:
    loss = (10. - imported(tf.constant(2.))) ** 2
  variables = tape.watched_variables()
  grads = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(grads, variables))
  return loss
```

```python
for _ in range(10):
  # "v" approaches 5, "loss" approaches 0
  print("loss={:.2f} v={:.2f}".format(train_step(), imported.v.numpy()))
```
<!-- END SOURCE: TensorFlow imported training -->

*The TensorFlow Authors, Using the SavedModel format, complete Basic fine-tuning
code cells 39–40, Apache-2.0. Original code and comments, without translation.*

The derivative is $8v-40$. With learning rate 0.05, the update is
$v_{k+1}=0.6v_k+2$: the first step changes 2 to 3.2, and after ten steps
$v_{10}=5-3(0.6)^{10}\approx4.98186$. The printed loss is computed before
each update, while the printed `v` is read after it. This demonstrates a
trainable restored variable; it does not restore a previous optimizer state,
training dataset or full interrupted training run. The graph and saved-variable
format do not impose an inference-only restriction by themselves.

### Named endpoints and the exported state

The guide specifies how to choose an endpoint and its output keys:

<!-- BEGIN SOURCE: TensorFlow signature selection -->
To declare a serving signature, specify a ConcreteFunction using the `signatures` kwarg. When specifying a single signature, its signature key will be `'serving_default'`, which is saved as the constant `tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY`.
<!-- END SOURCE: TensorFlow signature selection -->

<!-- BEGIN SOURCE: TensorFlow output names -->
By default, the output tensor names are fairly generic, like `output_0`. To control the names of outputs, modify your `tf.function` to return a dictionary that maps output names to outputs. The names of inputs are derived from the Python function arg names.
<!-- END SOURCE: TensorFlow output names -->

*The TensorFlow Authors, [Specifying signatures during export](https://github.com/tensorflow/docs/blob/9beed83c5fe8e4ccb742762394b084cd9e2bf2e0/site/en/guide/saved_model.ipynb),
complete cells 46 and 52, Apache-2.0. Original English text.*

The next program adds a vector endpoint named `serving_default`. Its declared
input is float32, has rank one and is named `x`; `None` permits different vector lengths.
Its result is a dictionary with the key `score`. These names are part of the
calling interface, not names a client should infer from a weight file.

```python
class NamedScale(tf.Module):
    def __init__(self):
        super().__init__()
        self.v = tf.Variable(2., name="scale")
        self.python_note = "this ordinary Python attribute is not an endpoint"

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name="x")])
    def serve(self, x):
        return {"score": x * self.v}

servable = NamedScale()
servable_path = str(Path(tmpdir) / "named-scale")
tf.saved_model.save(servable, servable_path,
                    signatures={"serving_default": servable.serve})
restored = tf.saved_model.load(servable_path)
endpoint = restored.signatures["serving_default"]
assert set(restored.signatures) == {"serving_default"}
assert not hasattr(restored, "python_note")
assert endpoint.structured_input_signature[1]["x"].shape.as_list() == [None]
assert endpoint.structured_input_signature[1]["x"].dtype == tf.float32
assert set(endpoint.structured_outputs) == {"score"}
np.testing.assert_array_equal(
    endpoint(x=tf.constant([-1., 0., 3.]))["score"].numpy(), [-2., 0., 6.])
restored.v.assign(3.)
assert endpoint(x=tf.constant([3.]))["score"].numpy()[0] == 9.
```

The exported files still hold `v=2`; the loaded Python object now holds `v=3`.
A second load must recover 2 again. To publish the changed value, create and
check another saved artifact rather than treating an in-memory assignment as a
release.

The declared TensorSpec is not sufficient input validation for every call path.
In the tested TensorFlow 2.21.0 program, the flattened signature callable also
executes a scalar and a rank-two tensor: multiplication can handle both. The
restored method's polymorphic dispatch and this signature wrapper are different
interfaces. Do not infer enforcement merely by printing the declared shape.
The following application wrapper accepts finite float32 vectors of length
1–32 and rejects other inputs before the call:

```python
# Observed behavior of this particular SavedModel signature wrapper.
assert endpoint(x=tf.constant(3.))["score"].shape.rank == 0
assert endpoint(x=tf.constant([[3.]]))["score"].shape.rank == 2

def checked_scale(call, values):
    if not isinstance(values, np.ndarray) or values.dtype != np.float32:
        raise ValueError("input must be a float32 NumPy array")
    if values.ndim != 1 or not 1 <= values.shape[0] <= 32:
        raise ValueError("input must be a vector with 1–32 elements")
    if not np.isfinite(values).all():
        raise ValueError("input must be finite")
    return call(x=tf.constant(values))["score"].numpy()

np.testing.assert_array_equal(
    checked_scale(endpoint, np.array([3.], dtype=np.float32)), [9.])
```

The limit of 32 is a chosen application contract, not a SavedModel format limit.
Other models can reject wrong shapes inside their operations or through explicit
graph assertions. This example establishes the behavior of its multiplication
graph and wrapper, not a claim that all SavedModels accept arbitrary shapes.
Authorization and other non-tensor application rules remain separate controls.

The official on-disk format section identifies these parts of the directory:

| SavedModel component | What the runtime uses it for |
|---|---|
| `saved_model.pb` | Serialized computation, object structure and named signatures. |
| `variables/` | Checkpoint files containing the exported variable values. |
| `assets/` | Files explicitly tracked for graph execution, such as vocabulary tables. Our scalar model has no vocabulary asset. |
| `assets.extra/`, if present | Additional consumer files; TensorFlow does not execute this directory automatically. |
| `fingerprint.pb` | Content fingerprints; not a publisher signature or proof of model quality. |

*Based on the complete [The SavedModel format on disk](https://github.com/tensorflow/docs/blob/9beed83c5fe8e4ccb742762394b084cd9e2bf2e0/site/en/guide/saved_model.ipynb)
section, cells 23–29. The table is an editorial description, not a new graph.*

### Conversion to a smaller runtime is another checked boundary

A SavedModel is an input to conversion; it is not a universal binary for every
mobile or microcontroller device. The converter must lower the graph to operators
supported by its destination. Dynamic shapes, custom operations, precision,
runtime build and accelerator support still constrain execution. The fixed-arena
and operator-registration example in [[03 Edge и federated deployment]] handles
the separate microcontroller boundary.

For the simple multiplication above, the TensorFlow Lite converter and interpreter
bundled with TensorFlow 2.21.0 can check that next boundary on CPU. This pinned
interpreter API is deprecated in favour of `ai_edge_litert`; its use here makes
the executed dependency explicit, not a recommendation to start new device code
with a deprecated interface. No hardware delegate or device benchmark is used.

```python
converter = tf.lite.TFLiteConverter.from_saved_model(servable_path)
lite_bytes = converter.convert()
interpreter = tf.lite.Interpreter(
    model_content=lite_bytes, num_threads=1,
    experimental_op_resolver_type=(
        tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
)
lite_endpoint = interpreter.get_signature_runner("serving_default")
np.testing.assert_array_equal(
    lite_endpoint(x=np.array([-1., 0., 3.], dtype=np.float32))["score"],
    [-2., 0., 6.],
)
```

The result uses the exported value 2, not the loaded object's later assignment
to 3. This particular graph converts and executes on CPU; the result does not
establish compatibility with a phone's NPU, a microcontroller operator resolver,
TensorFlow.js or a production TensorFlow Serving process. Those paths require
their own conversion and execution evidence.

Keras 3 also distinguishes its native `.keras` model archive from inference
export. `model.save("model.keras")` is not the same operation as
`model.export(path, format="tf_saved_model")`. A SavedModel is loaded through
`tf.saved_model.load`, or through a supported Keras wrapper for a chosen endpoint;
`keras.models.load_model` is not the SavedModel importer in Keras 3. The
[Keras migration guide](https://keras.io/guides/migrating_to_keras_3/#saving-a-model-in-the-tf-savedmodel-format)
and [export API](https://keras.io/api/models/model_saving_apis/export/) specify
these separate paths. The low-level `tf.Module` example above does not depend
on reconstructing a Keras model.

### Keep a tested migration path

Harvard's framework chapter connects the tensor API to its deployment dependencies:

<!-- BEGIN SOURCE: Harvard framework exit path -->
A framework's practical utility often depends more on these surrounding paths than on the core tensor API. Model hubs, experiment trackers, serving runtimes, cloud ML services, and interchange formats can reduce lock-in or deepen optimization, but each also adds a dependency that must be maintained. These compounding effects make framework migration progressively harder: CI/CD pipelines, monitoring infrastructure, cloud integrations, and custom operators turn an API choice into an operational commitment. The measurable indicators of viability are therefore contributor diversity, backward compatibility track record, available hiring pool, and the cost of preserving an exit path through standardized formats such as ONNX, framework-agnostic data pipelines, and documented customizations.
<!-- END SOURCE: Harvard framework exit path -->

*Vijay Janapa Reddi and contributors, [ML Frameworks, Development support and long-term viability assessment](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L4262),
CC BY-NC-SA 4.0. Complete original paragraph, without translation; these are
proposed evaluation criteria, not measured rankings of framework communities.*

The unregistered ONNX operator above is a concrete dependency that an interchange
format does not remove. A destination runtime still needs its implementation.
The SavedModel example removes the need to reconstruct the original Python class,
but retains a dependency on compatible TensorFlow operations and its saved
calling interface. A migration check loads the candidate artifact in the target
environment and repeats the relevant input, output and resource tests. A file
extension or a list of nominally supported backends cannot perform that check.

The release therefore records the SavedModel contents and endpoints, target
conversion options and resulting artifact, runtime version and validation
results. The rollback procedure in [[02 MLOps]] needs that compatible set of
artifacts, not only the original training weights.

```python
savedmodel_temporary.cleanup()
```
<!-- END REVIEW: TensorFlow SavedModel 2026-10-27 -->

<a id="цена-поздно-обнаруженного-ограничения"></a>

## Что приходится проверять после изменения компонента

Зависимости определяют, какие свидетельства перестали относиться к новому
варианту. Это не означает удаление старых результатов или обязательное
переобучение после любого изменения. Старый отчёт остаётся отчётом
о старом наборе, артефакте и условиях.

| Изменение | Что требует новой проверки | Что может сохраниться |
|---|---|---|
| Порог классификации | Матрица ошибок, нагрузка на очереди и правила действия | Веса и предсказанные оценки |
| Токенизатор или очистка текста | Совпадение входов, качество, длины и задержки | Исходные тексты и прежняя версия выпуска |
| Runtime или оборудование | Совместимость, численные отличия, память и производительность | Статистическая постановка задачи |
| Определение метки | Разметка, смысл метрик и сопоставимость кандидатов | Исторические события при допустимом хранении |
| Состав трафика | Качество на новой популяции, ёмкость и стоимость | Возможность использовать старую модель, если она проходит проверки |

Если ограничение памяти обнаружено до архитектурного поиска, часть
кандидатов можно не обучать. Если после обучения, может потребоваться
компрессия и повторная оценка. Если изменён лишь лимит ответа сервиса,
прежние веса не обязательно становятся непригодными, но меняется
пользовательский результат и его проверка.

Число стадий не определяет универсальный множитель стоимости переделок.
Стоимость зависит от конкретных зависимостей, доступных артефактов
и того, какие проверки нужны заново. В
[Harvard ML Workflow](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd#L2101)
обсуждается распространение ограничений; здесь оно рассматривается
через зависимые результаты, без количественного «закона» стоимости.

<a id="4-оценивание-валидация-и-готовность-к-развёртыванию"></a>

## От качества кандидата к разрешению на выпуск

Кандидат C из примера проходит два условия на validation-наборе.
Остаются другие вопросы: совпадает ли вход в обучении и serving,
работает ли экспортированный артефакт, выполняется ли задержка
на целевой нагрузке и доступен ли обратный переход на предыдущую версию.

Проверки механики конвейера отделяют от статистической проверки модели.
Для фиксированного входа и артефакта можно проверить токены, признаки,
порядок классов и формат ответа. Для качества нужна выборка с метками,
обоснованное разбиение, значимые срезы и оценка неопределённости.

<!-- source-excerpt: google-rules-5 -->
> Machine learning has an element of unpredictability, so make sure that you
> have tests for the code for creating examples in training and serving, and
> that you can load and use a fixed model during serving.
<!-- /source-excerpt -->

*Martin Zinkevich, [Rules of ML, Rule 5](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_5_test_the_infrastructure_independently_from_the_machine_learning),
CC BY 4.0. Оригинальный текст, без перевода.*

Для маршрутизатора такой тест сохраняет несколько синтетических сообщений
и ожидаемые результаты подготовки текста. Один и тот же набор проходит
обучающий и сервисный пути. Расхождение токенов или порядка меток
нужно объяснить до сравнения общей accuracy.
Если экспорт использует другую точность, численные допуски и проверку
качества задают явно: равенство файла весов этого не доказывает.

Далее применяют несколько видов проверки с разными возможностями.

| Проверка | Что наблюдается | Чего она сама не доказывает |
|---|---|---|
| Отложенный набор | Ошибки и срезы на заданной популяции | Изменение поведения пользователей |
| Повтор сохранённого трафика | Совместимость пути и ресурсные ограничения | Точные будущие поступления и реакции |
| Теневая версия | Реальные входы и ответы кандидата без применения его решения | Последствия нового маршрута |
| Ограниченный выпуск | Состояние сервиса и действие модели на части трафика | Статистически подтверждённое улучшение продукта |
| Спланированный A/B-тест | Различие результатов при заданной единице рандомизации | Любой долгосрочный эффект вне дизайна опыта |

Теневая версия всё равно расходует ресурсы и получает данные.
Отсутствие влияния на маршрут не делает её бесплатной или освобождённой
от правил доступа. Пробный выпуск также не является автоматически A/B-тестом:
для причинного сравнения нужны подходящее распределение трафика,
единица рандомизации и учёт вмешательства между группами.

Выпуск должен ссылаться на конкретный артефакт и результаты его проверок.
Если после оценки поменялся порог, prompt или токенизатор,
сначала определяют затронутые проверки. Нельзя оставлять одобрение
от предыдущего варианта только потому, что файл весов прежний.

<a id="5-от-функции-к-сервису"></a>

## Где выполняется предсказание и что делает приложение

Частота обучения и способ получения ответа независимы.
Модель можно переобучать раз в неделю, но вызывать для каждого нового
сообщения. Можно обучать ежедневно, а результаты для каталога
считать заранее и отдавать из хранилища.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/google-offline-online.png]]

*Google, [ML pipelines, Figure 5](https://developers.google.com/machine-learning/managing-ml-projects/pipelines),
CC BY 4.0. Исходник без изменений. Слева предсказание вычисляется
по запросу, справа приложение читает заранее рассчитанный результат.
Это различие online/offline inference, а не режима обучения.*

Свободный текст нового обращения заранее неизвестен, поэтому маршрутизатор
обычно выполняет inference после его поступления.
Для конечного каталога документов можно заранее вычислить категории
или embeddings. Тогда доступность и свежесть сохранённых результатов
становятся отдельными требованиями. Пропуск в кэше должен иметь
определённое поведение, а старый результат должен быть связан
с версией модели и входа.

### Ответ модели и действие продукта

Между входом и вызовом модели находятся проверка схемы и преобразование
данных. После вызова выбирается действие: порог, доступная очередь,
ручная проверка или иной разрешённый путь.
Класс с максимальной вероятностью не обязан непосредственно становиться
пользовательским ответом.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/google-prediction-process.png]]

*Google, [ML pipelines, Figure 6](https://developers.google.com/machine-learning/managing-ml-projects/pipelines),
CC BY 4.0. Исходник без изменений. Зелёная точка обозначает подготовку
данных, оранжевая правила обработки предсказания.
Оба шага входят в проверяемый путь.*

Сервисный контракт определяет допустимый вход, версию ответа, причины
отказа и сроки выполнения. Некорректное сообщение, недоступная модель,
перегрузка и низкая уверенность в классе имеют разный смысл.
HTTP 200 сам по себе не подтверждает правильность классификации.
Для потокового ответа он также не доказывает, что генерация завершилась:
это проверяется по терминальному событию [[02 Areas/ML & DL/06 Практика/14a HTTP streaming и жизненный цикл запроса|протокола]].
Если часть обращений отправляется человеку, измеряют
и качество автоматической части, и её покрытие.

В практикуме [Efficient DL Systems, week 7](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment)
предобученная VGG16 экспортируется в файл `vgg16.pt` и вызывается из сервиса.
Скрипт `train_model.py` не обучает эту модель: он использует `torch.jit.trace`.
Пример показывает переход от объекта библиотеки к загружаемому артефакту;
название файла не является свидетельством обучения или готовности к выпуску.
Полные исходные стенды сохранены на языке курса.

Подробности контейнера, готовности реплик, мониторинга и отката
разобраны в [[02 MLOps|следующей главе]].
Здесь условия обслуживания уже должны участвовать в выборе модели,
а не появляться после завершения обучения.

<a id="6-обратная-связь-из-эксплуатации"></a>

## Как результат возвращается в данные

Журнал предсказаний связывает вход, выпуск, оценку, принятое действие
и последующее наблюдение. Метка может появиться значительно позже
и храниться в другом источнике. Для объединения нужен устойчивый
идентификатор обращения и определённая политика работы с версиями.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/google-prediction-log.png]]

*Google, [ML pipelines, Figure 7](https://developers.google.com/machine-learning/managing-ml-projects/pipelines),
CC BY 4.0. Исходник без изменений. `Prediction logging` сохраняет
результаты модели; `App logging` события приложения.
Предсказание и событие продукта не становятся истинной меткой автоматически.*

Для обращения можно сохранить версию подготовки входа, модель, порог,
назначенную очередь и время решения. После ручного разбора добавляется
метка с версией инструкции и временем её доступности.
Если исправление меняет старую метку, нужно уметь восстановить,
какая версия попала в конкретный обучающий набор.

Исправления пользователей являются выборочными наблюдениями.
Недовольные пользователи могут оставлять больше отзывов.
Операторы чаще подробно разбирают обращения, которые модель уже
отправила инженерам. Если считать такую выборку всем трафиком,
качество и следующее обучение получат смещение.

### Пример выборочной разметки

Из 1000 обращений модель отправила инженерам 100.
Разобраны все эти 100 и случайная выборка из 100 остальных 900.

| Группа | Всего в трафике | Размечено | Найдено среди размеченных |
|---|---:|---:|---|
| Назначен инженер | 100 | 100 | 90 TP, 10 FP |
| Оставлены первой линии | 900 | 100 | 20 FN, 80 TN |

Если посчитать recall только на 200 размеченных примерах,
получится $90/(90+20)\approx81.8\%$.
Но доли включения в разметку различаются.
У первой группы вероятность включения 1, у второй $1/9$.
Один размеченный пример второй группы представляет девять обращений.

Взвешенная оценка FN равна $20\cdot9=180$, TP остаётся 90.
Оценка recall по всему потоку становится $90/(90+180)=33.3\%$.
Это оценка по выборке, не знание исхода каждого из 900 обращений.
При конечной случайной выборке ей нужен интервал неопределённости.

```python
def weighted_confusion(strata):
    totals = {key: 0. for key in ['tp', 'fn', 'fp', 'tn']}
    for stratum in strata:
        probability = stratum['inclusion_probability']
        assert 0 < probability <= 1
        for key in totals:
            count = stratum.get(key, 0)
            assert count >= 0
            totals[key] += count / probability
    return totals

audit = [
    {'tp': 90, 'fp': 10, 'inclusion_probability': 1.},
    {'fn': 20, 'tn': 80, 'inclusion_probability': 1 / 9},
]
weighted = weighted_confusion(audit)
naive_recall = 90 / (90 + 20)
weighted_recall = weighted['tp'] / (weighted['tp'] + weighted['fn'])
assert np.isclose(naive_recall, 9 / 11)
assert np.isclose(weighted_recall, 1 / 3)
assert np.isclose(sum(weighted.values()), 1000.)
```

Такое взвешивание требует известного механизма отбора и ненулевой
вероятности наблюдения нужных примеров. Если обращения первой линии
вообще никто не размечает, их ошибки нельзя восстановить одним коэффициентом.
Для их оценки нужен новый источник наблюдений, например отдельный
случайный ручной аудит, который не меняет пользовательский маршрут.

Проблема связана с [Rules of ML, правила 28–35](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew):
действия модели меняют доступные ей будущие примеры.
Следующее обучение на собственных маршрутах способно закрепить
прежние решения, даже если новая модель точнее воспроизводит эти метки.

### Какое наблюдение требует какого изменения

Рост ошибок сервиса сначала требует проверки входного пути,
ресурсов и зависимостей. Несовпадение токенизации между двумя средами
требует исправления преобразований. Падение качества на зрелой
размеченной когорте требует анализа ошибок и популяции.
Переобучение не устраняет все три причины одним действием.

Изменение распределения входов является поводом исследовать данные.
Оно не доказывает падение качества без связи с задачей и метками.
Стабильная доля классов тоже не доказывает правильность предсказаний:
модель может ошибаться на других примерах с той же частотой ответов.
Ограничения на свежесть данных, задержку меток и время реакции
должны соответствовать реальному процессу, а не общему расписанию
«переобучать раз в сутки».

<!-- BEGIN REVIEW: workflow feedback 2026-10-26 -->
## Feedback loops do not share one clock

A failed request can be visible immediately, while its final task outcome may
not be known for days. Retraining on every operational alarm treats those
different observations as if they arrived together.

Harvard's *Multi-scale feedback* uses the following deployment scenario:

<!-- BEGIN SOURCE: Harvard multi-scale feedback -->
> ML systems succeed through orchestrating feedback loops across multiple timescales, each serving different optimization purposes. Our DR deployment exemplifies this pattern: minute-level loops catch a misconfigured camera before it produces a day's worth of unusable images; daily loops detect proxy shifts such as referral-rate changes, confidence drops, rejection-rate spikes, or site-specific camera failures; weekly loops aggregate labeled accuracy statistics when ground truth is available and run drift detection tests; monthly loops reveal that demographic shifts in a region require expanded training data; and quarterly loops evaluate whether the overall architecture still meets evolving clinical needs.
<!-- END SOURCE: Harvard multi-scale feedback -->

*Vijay Janapa Reddi and contributors, [ML Workflow, Multi-scale feedback](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd#L2131),
CC BY-NC-SA 4.0. Original English paragraph; only the index markup is removed.
These cadences belong to the source's illustrative scenario, not prescribed
clinical or operational intervals.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/workflow-feedback-2026-10-26/feedback-timescales.svg]]

*Vijay Janapa Reddi and contributors, ML Workflow, complete original
[feedback-timescales SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/ml_workflow/images/svg/ml_workflow_feedback_timescales.svg),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The horizontal scale is logarithmic; the bars show durations, not accuracy,
importance or required response deadlines.*

For the support-ticket example, an operational loop can stop a release after
a rise in connection errors without waiting for the 72-hour label horizon.
A data-quality loop can detect missing text before classification quality is
known. A quality review must wait for the chosen outcome horizon and label
availability. A capacity or architecture review asks whether the current
design still fits the workload and resource budget. One schedule cannot make
all four observations available sooner.

Consider three predictions and an evaluation policy requiring a full
72-hour outcome horizon. The first prediction was made at hour 0 and its
incorrect outcome became available at hour 80. The second was made at hour 60;
a correct outcome was recorded at hour 70, before that prediction's horizon
closed. The third was made at hour 0 but still has no label.

| Evaluation time | Predictions past the horizon | Usable labels | Coverage of mature predictions | Error ratio on usable labels |
|---|---:|---:|---:|---:|
| Hour 72 | 2 | 0 | 0% | Unavailable |
| Hour 80 | 2 | 1 | 50% | 100% |
| Hour 132 | 3 | 2 | 66.7% | 50% |

At hour 72, neither the unavailable old label nor the early young label is
eligible under this policy. No error ratio can be reported yet. At hour 80,
the first outcome is usable, but the missing label still reduces coverage.
At hour 132, the second prediction has completed its horizon. The change from
100% to 50% is a change in the observed cohort, not evidence that the unchanged
model improved between dashboard refreshes.

```python
import math

def mature_quality(observations, now, horizon):
    assert math.isfinite(now) and math.isfinite(horizon) and horizon >= 0
    mature = [r for r in observations if r['predicted_at'] + horizon <= now]
    usable = [r for r in mature
              if r['label_at'] is not None and r['label_at'] <= now
              and r['correct'] is not None]
    errors = sum(not r['correct'] for r in usable)
    return {
        'mature': len(mature), 'usable': len(usable),
        'coverage': len(usable) / len(mature) if mature else None,
        'error_ratio': errors / len(usable) if usable else None,
    }

cohort = [
    {'predicted_at': 0, 'label_at': 80, 'correct': False},
    {'predicted_at': 60, 'label_at': 70, 'correct': True},
    {'predicted_at': 0, 'label_at': None, 'correct': None},
]
assert mature_quality(cohort, 72, 72) == {
    'mature': 2, 'usable': 0, 'coverage': 0., 'error_ratio': None}
assert mature_quality(cohort, 80, 72) == {
    'mature': 2, 'usable': 1, 'coverage': .5, 'error_ratio': 1.}
assert mature_quality(cohort, 132, 72) == {
    'mature': 3, 'usable': 2, 'coverage': 2 / 3, 'error_ratio': .5}
```

The maturity rule is a declared part of this example, not a requirement that
every classification task wait exactly 72 hours. Changing the rule changes
which observations are comparable. When labels are corrected later, their
versions must also be preserved to reproduce a previous report. The earlier
selection-weighting example addresses a separate problem: who gets labeled,
even after enough time has passed.

The monitoring, canary and rollback procedures in [[02 MLOps]] use these
different evidence clocks. Faster collection of server metrics cannot replace
missing task outcomes.

*This Harvard-derived extension and its added example are distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*
<!-- END REVIEW: workflow feedback 2026-10-26 -->

<a id="минимальный-контракт-между-стадиями"></a>

## Что передаётся следующей стадии

Воспроизводимый цикл передаёт артефакт вместе с условиями,
при которых получены результаты его проверки.

| Граница | Передаваемый результат | Что нужно для следующего решения |
|---|---|---|
| Постановка → данные | Определение задачи и действия | Популяция, момент решения, метка, ограничения и владелец |
| Данные → обучение | Неизменяемая версия набора | Происхождение, разбиение, доступность признаков, покрытие метками |
| Обучение → проверка | Кандидат и описание запуска | Веса, код, конфигурация, окружение и версии входов |
| Проверка → выпуск | Результаты для конкретного кандидата | Срезы, неопределённость, ресурсные проверки и критерии допуска |
| Выпуск → эксплуатация | Версионированный сервисный артефакт | Распределение трафика, наблюдение, владелец и обратный переход |
| Эксплуатация → следующий цикл | Связанные события и метки | Время доступности, правила отбора, затронутая популяция и решение о правке |

Для учебного маршрутизатора этот набор позволяет ответить на конкретный
вопрос: почему обращение попало к инженеру и можно ли повторить это решение
на том же входе. После изменения порога можно пересчитать действия
по сохранённым оценкам, если остальной путь и популяция неизменны.
После изменения токенизатора прежних оценок уже недостаточно:
нужно снова получить предсказания и проверить затронутые свойства.

## Практика и первоисточники

- Google, [ML pipelines](https://developers.google.com/machine-learning/managing-ml-projects/pipelines).
  Четыре конвейера, online/offline predictions, обработка ответа,
  журналы, версии данных и моделей; исходные Figures 4–9.
- Martin Zinkevich, [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).
  Правила 1–6 о первом конвейере, 24–25 о сравнении результата,
  28–35 об обратной связи и training-serving skew.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/ml_workflow|Harvard ML Systems — полный исходный ML Workflow]].
  Постановка, воспроизводимые артефакты, проверка готовности и зависимости.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/ml_ops|Harvard ML Systems — ML Operations]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week07_application_deployment/Effdl26-07.pdf|Efficient DL Systems — полные лекционные слайды]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week07_application_deployment/README|Efficient DL Systems — маршрут по исходным стендам]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week07_application_deployment/01_python_server/README|Python-сервер]],
  [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week07_application_deployment/02_docker/README|контейнер]]
  и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week07_application_deployment/03_metrics/README|метрики]]:
  оригинальные материалы практикума без перевода.
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей]]:
  дальнейшая проверка качества и методологии сравнения.
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking и SLO]]
  и [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning|очереди и мощность сервиса]]:
  ресурсная проверка полного пути.

← [[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]
· Далее: [[02 MLOps]]
