---
title: "22. Провести scaling-law campaign"
type: practice
status: reviewed
last_updated: 2026-09-06
source_unit_id:
  - assignment-03-isoflops-fit-workflow
  - assignment-03-task-chinchilla-isoflops
  - assignment-03-isoflops-reproduction
  - assignment-03-deliverable-chinchilla-isoflops-001
  - assignment-03-deliverable-chinchilla-isoflops-002
  - assignment-03-task-scaling-laws
  - assignment-03-leaderboard-budget
  - assignment-03-deliverable-scaling-laws-003
  - assignment-03-test-tests-test-api-py-51-test-budget
  - assignment-03-test-tests-test-api-py-72-test-submit-jobs
  - assignment-03-test-tests-test-api-py-152-test-submit-rejects-duplicate-training-config
  - assignment-03-test-tests-test-api-py-193-test-final-submission-accepts-training-config-and-predicted-loss
primary_sources:
  - https://github.com/stanford-cs336/assignment3-scaling/tree/03e9372992e913061b9e78b5cfcb62ad8a87de35
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf
  - https://arxiv.org/abs/2203.15556
  - https://arxiv.org/abs/2406.19146
contract: "[[02 Areas/ML & DL/06 Практика/Contracts/stanford-cs336-a3.yml]]"
---

# 22. Провести scaling-law campaign

В этой работе нужно не подобрать красивую степенную линию к готовой таблице, а
провести полный цикл принятия решения: определить измеряемые величины,
распределить ограниченный бюджет, найти IsoFLOP-минимумы, сравнить несколько
моделей, сделать прогноз на удержанном масштабе и объяснить ошибку после
раскрытия результата.

Практика основана на Stanford CS336 Assignment 3, но не является его решением и
не публикует данные, по которым можно восстановить ответ задания. Официальное задание использует общую
очередь B200 и скрытый leaderboard. Здесь обязательный профиль работает
локально с синтетическими или собственными наблюдениями. При наличии доступа к
курсу его API можно исследовать отдельно, не сохраняя токены и не смешивая
локальные результаты с официальной отправкой.

Перед началом прочитайте [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md]].

<a id="question"></a>

## Исследовательский вопрос

Сформулируйте один прогнозируемый результат. Например:

> Для заданного target compute и неизменного training recipe какая пара
> non-embedding parameters $N$ и предъявленных токенов $D$ минимизирует
> validation loss, и насколько надёжен прогноз относительно более крупного
> удержанного запуска?

Вопрос должен называть целевую метрику, стоимость и область переноса. Нельзя
смешивать в одном ответе теоретические FLOPs и GPU-hours, dense и active MoE
parameters, разные токенизаторы или разные validation distributions.

<a id="profiles"></a>

## Два профиля выполнения

### `offline_local` — обязательный

Используйте один из двух источников точек:

1. **Синтетический стенд.** Сгенерируйте таблицу по заранее записанной гладкой
   функции $L(N,D)$, добавьте небольшой случайный шум с фиксированным seed и
   скройте один крупнейший compute budget до финальной проверки. Коэффициенты
   генератора храните отдельно от fitting notebook, чтобы не подменить оценку
   чтением правильного ответа.
2. **Собственные или открытые runs.** Импортируйте результаты небольших
   языковых моделей при условии, что известны tokenizer, data split, parameter
   convention, schedule и compute proxy. Зафиксируйте лицензию и checksum
   исходной таблицы.

Запрещено копировать или преобразовывать
`data/isoflops_curves.json` из Stanford Assignment 3: этот файл нужен для
официального упражнения, и по нему можно восстановить ответ.

### `course_api_observation` — необязательный

Если у вас есть официальный доступ, разрешается сохранить обезличенный журнал
собственных запросов и ответов API. Не добавляйте в репозиторий ключ, cookie,
полный server dump или результаты других участников. Значения Stanford
search/target budget следует подписывать как ограничения курса, а не как
рекомендуемые бюджеты реального pre-training.

<a id="pre-registration"></a>

## Шаг 1. Зафиксировать эксперимент до запусков

Создайте `artifacts/scaling-campaign/campaign-plan.yml`. В нём должны быть:

- формулировка вопроса и единица validation loss;
- train и validation data identity, tokenizer и context length;
- architecture family и точное соглашение о $N$;
- правило вычисления $D$ и число уникальных токенов/эпох;
- optimizer, batch, learning rate, warmup, decay, clipping, dtype и seed policy;
- cost metric: proxy FLOPs, measured accelerator-seconds или деньги;
- доступный **search budget** и отдельный **target budget**;
- граница диапазона, который разрешено использовать для fit;
- заранее выбранный held-out budget;
- две или более формы модели, которые будут сравниваться;
- правило раскрытия held-out результата.

Если используется приближение

$$
C_{\mathrm{proxy}}=6N_{\mathrm{nonembed}}D,
$$

запишите его именно как proxy. Для длинного контекста, MoE или нетипичной
архитектуры опишите поправку либо откажитесь от сравнения по этой формуле.

Не задавайте универсальные доли pilot/target budget и обязательное число
размеров модели. Сетка достаточна тогда, когда охватывает несколько compute
levels, позволяет увидеть обе стороны минимумов и оставляет ресурс на
адаптивное уточнение.

<a id="ledger"></a>

## Шаг 2. Вести журнал всех попыток

Каждая строка `runs.jsonl` описывает один запуск. Минимальная запись имеет вид:

```json
{
  "experiment_id": "c2-n4-s1",
  "config_hash": "sha256:...",
  "status": "completed",
  "N_nonembed": 100000000,
  "D_presented": 8000000000,
  "D_unique": 8000000000,
  "compute_proxy_flops": 4.8e18,
  "cost_reserved": 7200,
  "cost_actual": 5361,
  "cost_charged": 5361,
  "seed": 1,
  "validation_loss_final": 2.47,
  "curve_path": "curves/c2-n4-s1.json",
  "failure_reason": null
}
```

Числа в примере служат только для демонстрации схемы. Добавьте hardware,
software revision, tokenizer/data/schedule identifiers, batch, learning rate,
число шагов, wall time и MFU, если он измеряется.

Записывайте `planned`, `queued`, `running`, `completed`, `failed` и `timed_out`.
Неустойчивый или прерванный run — часть evidence. Его нельзя удалить из журнала
или представить последний checkpoint как честный final loss. Нормализованная
конфигурация должна иметь уникальный hash: случайный повтор с тем же seed не
расходует бюджет повторно из-за ошибки bookkeeping.

Модель учёта, заимствованная из серверного интерфейса Assignment 3, различает
резерв и списание. Пока job ожидает или выполняется, резервируется максимально
разрешённое время; завершённый или упавший job списывает фактическое время в
допустимых пределах; timeout списывает максимум. Для локального профиля можно
выбрать другое правило, но `reserved`, `actual` и `charged` всё равно должны
оставаться разными полями.

<a id="grid"></a>

## Шаг 3. Построить начальную IsoFLOP-сетку

Выберите несколько бюджетов $C_i$. Для каждого бюджета задайте геометрически
разнесённые $N_{ij}$ и вычислите

$$
D_{ij}=\frac{C_i}{6N_{ij}}.
$$

Если $D_{ij}$ несовместим с целым числом batches или steps, округляйте по
заранее объявленному правилу и сохраняйте как planned, так и actual compute.
Точки одной IsoFLOP-группы должны различаться allocation между моделью и
данными, но не качеством данных или процедурой оценки.

После первого раунда задайте для каждой группы один из статусов:

- `bracketed`: у лучшей точки есть меньший и больший сосед с худшим loss;
- `left_open`: loss продолжает уменьшаться к самой маленькой модели;
- `right_open`: loss продолжает уменьшаться к самой большой модели;
- `unusable`: сравнение испорчено несопоставимым schedule, failure или данными.

Для `left_open` и `right_open` расширьте сетку. Не называйте крайнее наблюдение
минимумом. Уточняющие точки выбирайте после просмотра первой серии и списывайте
из заранее оставленного adaptive budget.

<a id="fit"></a>

## Шаг 4. Оценить минимум двумя способами

### Fit A: минимумы IsoFLOP-кривых

Для каждой bracketed-группы оцените $(N_i^*,D_i^*)$. Покажите и дискретную
лучшую точку, и результат локальной интерполяции. Затем подгоните

$$
N^*(C)=k_NC^a,
\qquad
D^*(C)=k_DC^b.
$$

Если используется $C\propto ND$, проверьте $a+b\approx1$, но не заставляйте
сумму равняться единице без отчёта о свободной подгонке.

### Fit B: совместная модель

Подгоните все завершённые сопоставимые наблюдения к

$$
L(N,D)=E+A N^{-\alpha}+B D^{-\beta}.
$$

Оцените параметры нелинейно в исходном loss space и сравните с разумной
альтернативой: log-linearized fit, robust loss, weighted likelihood или модель
с другим residual term. Для каждого варианта сохраните начальные приближения,
ограничения параметров, веса точек и причину выбора.

Нельзя выбрать Fit A для одного рисунка, Fit B для другого и скрыть их
расхождение. Итоговый отчёт должен объяснить, какая часть данных заставляет
методы давать разные прогнозы.

<a id="diagnostics"></a>

## Шаг 5. Построить диагностические рисунки

Обязательны:

1. raw $L$ против $N$ для каждой IsoFLOP-группы;
2. $N^*(C)$ с исходными минимумами, fitted line и uncertainty band;
3. $D^*(C)$ с теми же элементами;
4. наблюдаемый и предсказанный $L(N,D)$;
5. residuals против $N$, $D$, $C$ и порядка запуска;
6. влияние исключения каждой compute group;
7. budget ledger: planned, reserved, actual и charged cost.

Оси должны содержать единицы и parameter convention. Если график использует
логарифмическую шкалу, это отмечается явно. Цвет одного compute budget должен
быть одинаковым на всех рисунках. Failed runs показываются отдельным символом,
а не исчезают.

<a id="uncertainty"></a>

## Шаг 6. Оценить неопределённость

Сделайте хотя бы одно повторение с другим seed на малом, среднем и крупном
доступном масштабе. Для bootstrap пересэмплируйте целые IsoFLOP-группы или
запуски с учётом структуры кампании; не объявляйте соседние checkpoints одной
траектории независимыми наблюдениями.

Сравните три источника неопределённости:

- stochastic: seed и data order;
- fitting: шум минимумов и чувствительность к точкам;
- structural: выбор математической формы.

В `fits.json` сохраните интервал для $N^*_{target}$, $D^*_{target}$ и
$L_{target}$, а также отношение target budget к крупнейшему бюджету fit. Это
отношение показывает длину экстраполяции и должно стоять рядом с прогнозом.

<a id="sealed-prediction"></a>

## Шаг 7. Зафиксировать прогноз до раскрытия

Создайте `prediction.json` с точными полями:

- target budget и cost unit;
- выбранные $N$, $D$, batch, learning rate и schedule;
- predicted validation loss и uncertainty interval;
- выбранная модель и альтернативные прогнозы;
- timestamp, source commit и hash `runs.jsonl`;
- объяснение округления до реально обучаемой конфигурации.

Получите контрольную сумму, например:

```bash
shasum -a 256 prediction.json > prediction.sha256
```

После этого файл нельзя менять. Только теперь откройте held-out точки или
запустите target configuration.

<a id="reveal"></a>

## Шаг 8. Провести post-reveal разбор

В `reveal.md` сравните prediction с наблюдением. Одного абсолютного отклонения
недостаточно. Ответьте:

1. Попал ли observation в заявленный uncertainty interval?
2. Какая модель дала наименьшую out-of-budget ошибку?
3. Сохранилась ли сумма exponents и положение IsoFLOP-минимума?
4. Есть ли систематический рисунок в residuals?
5. Не изменились ли throughput, MFU, schedule phase или data regime?
6. Какая следующая точка лучше всего различит конкурирующие объяснения?

Не переоценивайте coefficients задним числом и не заменяйте первоначальный
прогноз улучшенной версией. Можно построить post-reveal fit, но он хранится
отдельно и сравнивается с sealed prediction.

<a id="stanford-facts"></a>

## Как читать ограничения Stanford Assignment 3

В официальной постановке target равен 48 B200-hours, а общий search budget —
12 B200-hours. Эти числа описывают учебную инфраструктуру. Они не означают, что
пилоты реального проекта всегда должны стоить четверть target run, и не
переводятся в FLOPs без измерения.

Курс использует приближение

$$
N_{\mathrm{nonembed}}\approx
12\,n_{\mathrm{layer}}d_{\mathrm{model}}^2,
$$

context length 512, vocabulary 32k, фиксированный порядок DCLM tokens, no
dropout и одну B200 на run. `model_seed` меняет initialization, но не порядок
данных. Валидация в открытом интерфейсе использует $2^{18}$ токенов; запись
`218`, возникающая после неудачного PDF text extraction, неверна.

Конфигурация должна удовлетворять
`hidden_size = num_attention_heads × head_dim`; текущая проверка репозитория
требует равенства key/value и attention heads, то есть фактически не допускает
GQA. `head_dim` должен быть чётным. Число training tokens делится на
`512 × batch_size`, число шагов — на число evaluations, а validation tokens —
на `512 × validation batch size`.

Эти детали полезны как пример строгого experiment contract. Они не являются
архитектурным рецептом для другой кампании.

<a id="deliverables"></a>

## Что сдаётся

```text
artifacts/scaling-campaign/
├── campaign-plan.yml
├── environment.json
├── input-manifest.json
├── runs.jsonl
├── fits.json
├── plots/
│   ├── isoflops.png
│   ├── n-optimal.png
│   ├── d-optimal.png
│   ├── observed-vs-predicted.png
│   ├── residuals.png
│   ├── leave-one-budget-out.png
│   └── budget-ledger.png
├── prediction.json
├── prediction.sha256
└── reveal.md
```

`input-manifest.json` содержит checksum исходной таблицы или генератора и
подтверждение, что данные с ответом Stanford не использовались. `environment.json`
фиксирует код, зависимости и оборудование. Все значения на рисунках должны
восстанавливаться из `runs.jsonl` и `fits.json`.

<a id="acceptance"></a>

## Критерии прохождения

`PASS offline_local` требует:

1. pre-registration существует и предшествует held-out reveal;
2. source/fixture checksum и parameter convention указаны;
3. search и target budget разделены, перерасход отсутствует;
4. все попытки, включая failed и timed out, присутствуют в ledger;
5. duplicate configs определяются по нормализованному hash;
6. каждый минимум bracketed либо явно помечен open/unusable;
7. Fit A и Fit B построены и сравнены;
8. residual, sensitivity и uncertainty diagnostics сохранены;
9. prediction запечатан checksum до reveal;
10. post-reveal error analysis отвечает на шесть вопросов выше;
11. нигде не объявляется универсальный tokens-per-parameter ratio.

Немедленный `FAIL`: использование данных, раскрывающих ответ Stanford; незаписанное
изменение tokenizer/data/schedule; смешение active и total parameters; GPU-hours
под названием FLOPs без conversion model; удаление failed runs; заявленный
минимум на границе сетки; изменение `prediction.json` после reveal; loss без
исходных точек и единицы измерения.

Полная схема полей и границ находится в
[[02 Areas/ML & DL/06 Практика/Contracts/stanford-cs336-a3.yml]].
