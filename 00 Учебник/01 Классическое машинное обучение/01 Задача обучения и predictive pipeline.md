---
title: Задача обучения и predictive pipeline
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
primary_sources:
  - https://inria.github.io/scikit-learn-mooc/predictive_modeling_pipeline/
  - https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

# Задача обучения и predictive pipeline

В supervised learning есть таблица наблюдений, целевая величина и правило
оценивания предсказаний. Алгоритм обучения получает выборку

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^{n}
$$

и выбирает функцию $f_\theta(x)$ из заданного семейства. Параметры $\theta$
подбираются так, чтобы средняя ошибка на обучающих данных была мала:

$$
\hat R(\theta)=\frac1n\sum_{i=1}^{n}
\ell\!\left(y_i,f_\theta(x_i)\right).
$$

Это empirical risk. Конечная цель, однако, состоит не в минимизации числа,
посчитанного на знакомой таблице, а в хорошем предсказании для новых объектов
из целевой среды. Поэтому модель, preprocessing, схема разбиения и метрика
образуют одну процедуру. Оценивать только estimator недостаточно.

![[Assets/Sources/Scikit-learn MOOC/figures/supervised.png|Supervised and unsupervised learning]]

*Иллюстрация из scikit-learn MOOC:
[исходный SVG/PNG-корпус](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/figures),
CC BY 4.0.*

## Сначала определить строку таблицы

Одна строка должна соответствовать единице, для которой в эксплуатации будет
делаться одно предсказание. Для оценки кредитного риска это может быть заявка,
для диагностики — пациент, для временного прогноза — момент времени. Если
несколько строк принадлежат одному пользователю, пациенту или устройству, они
не являются независимыми. Эта структура позже определит способ split.

Столбцы делятся на:

- **features** $X$: сведения, доступные в момент предсказания;
- **target** $y$: величина, которую требуется предсказать;
- идентификаторы и служебные поля, которые обычно нельзя подавать модели как
  обычные признаки.

Target может быть непрерывным — тогда решается regression — или дискретным —
тогда classification. Название алгоритма не определяет задачу. `LogisticRegression`
решает классификацию, несмотря на слово regression в имени.

Полный исходный разбор таблиц и типов данных:
[[05 Источники/Courses/Scikit-learn MOOC/python_scripts/01_tabular_data_exploration.py|Tabular data exploration]].

## Baseline задаёт смысл метрике

Число без точки сравнения мало что сообщает. Для несбалансированной
классификации accuracy модели нужно сравнить хотя бы с правилом «всегда
предсказывать самый частый класс». Для regression простейший baseline
предсказывает среднее или медиану training target.

Пусть 95% объектов относятся к отрицательному классу. Constant classifier
получит accuracy 0.95, не обнаружив ни одного положительного объекта. Значит,
для такой задачи нужны confusion matrix, precision, recall, specificity,
balanced accuracy и PR curve; выбор зависит от стоимости двух типов ошибки.

Исходная практическая глава:
[[05 Источники/Courses/Scikit-learn MOOC/python_scripts/cross_validation_baseline.py|Baseline models in cross-validation]].

## Почему preprocessing входит в модель

Числовые признаки могут иметь разные единицы измерения; категориальные значения
нужно закодировать; пропуски — обработать. Все операции, параметры которых
оцениваются по данным, должны обучаться только на training fold.

Например, standardization вычисляет

$$
x'_j=\frac{x_j-\mu_j}{\sigma_j}.
$$

Если $\mu_j$ и $\sigma_j$ посчитаны по всей таблице до cross-validation,
validation fold уже повлиял на обучение. Это data leakage, даже если target
не использовался явно.

Правильная единица оценки — pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(),
)
```

При каждом split pipeline заново оценивает scaler на training fold, затем
преобразует validation fold уже зафиксированными параметрами.

![[Assets/Sources/Scikit-learn MOOC/figures/api_diagram-pipeline.fit.svg|Pipeline fit]]

*Во время `fit` преобразования и predictor обучаются последовательно.
Источник: scikit-learn MOOC, CC BY 4.0.*

![[Assets/Sources/Scikit-learn MOOC/figures/api_diagram-pipeline.predict.svg|Pipeline predict]]

*Во время `predict` сохранённые преобразования применяются без повторного
обучения. Источник: scikit-learn MOOC, CC BY 4.0.*

Полные исходные notebooks:

- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/02_numerical_pipeline_introduction.py|Numerical pipeline introduction]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/02_numerical_pipeline_scaling.py|Scaling numerical features]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/03_categorical_pipeline.py|Categorical pipeline]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/03_categorical_pipeline_column_transformer.py|ColumnTransformer]].

## Train, validation и test — разные полномочия

Training data изменяют параметры. Validation data влияют на выбор модели,
hyperparameters и threshold. Test data используются после завершения выбора.
Если после каждого test result меняется код или модель, test превращается в
validation.

Случайный split годится только тогда, когда строки действительно можно считать
обменными. Иначе применяется:

- group split — все строки одного пациента или пользователя остаются вместе;
- temporal split — обучение на прошлом, проверка на будущем;
- stratified split — доля классов сохраняется в folds;
- predefined split — когда эксплуатационный сценарий задаёт фиксированную
  границу.

## Полный цикл

1. Определить объект предсказания и момент, когда features доступны.
2. Зафиксировать target и loss, отражающие требуемое решение.
3. Выбрать split по структуре данных, а не по удобству.
4. Посчитать простой baseline.
5. Собрать preprocessing и estimator в один pipeline.
6. Настраивать hyperparameters только внутри training/validation procedure.
7. Один раз оценить выбранную процедуру на test.
8. Перед deployment проверить, совпадают ли online features, preprocessing и
   семантика target с экспериментом.

## Сквозной пример: кому позвонить сегодня

Представим службу удержания клиентов. Каждый рабочий день она может позвонить
лишь тысяче человек, поэтому требуется оценить вероятность ухода клиента в
следующие тридцать дней. Здесь легко начать с выбора классификатора, хотя
сначала нужно зафиксировать гораздо более важные условия эксперимента.

Строкой таблицы является пара «клиент — дата принятия решения». Признаки
содержат только сведения, существовавшие к началу этой даты: длительность
договора, обращения в поддержку, платежи и использование продукта. Целевая
метка появляется позднее и сообщает, ушёл ли клиент в заданном горизонте.
Поле `account_closed_at` нельзя превращать в признак: оно почти идеально
раскрывает ответ, но в момент звонка ещё неизвестно. Так возникает временная
утечка — особенно коварная, потому что оценка на случайном разбиении выглядит
великолепно.

Обычная точность здесь не соответствует решению. Оператору нужен ранжированный
список ограниченной длины, поэтому полезны precision среди первых 1000
клиентов, recall при доступном бюджете звонков и ожидаемая ценность:

$$
U(t)=V_{TP}TP(t)-C_{FP}FP(t)-C_{FN}FN(t),
$$

где порог $t$ определяет, кому звонить. Модель и порог решают разные задачи:
первая строит оценки риска, второй переводит их в действие с учётом стоимости
ошибок и пропускной способности процесса.

## От функции потерь к качеству решения

Функция потерь, которую минимизирует алгоритм, не обязана совпадать с итоговой
метрикой. Логистическая регрессия оптимизирует log loss, хотя продуктовая
команда может сравнивать модели по recall при фиксированном precision. Это не
противоречие. Дифференцируемая функция потерь даёт удобный способ обучить
непрерывный score, а правило принятия решения настраивается позднее на
отложенных данных.

Важно различать три уровня:

1. **задача предсказания** задаёт объект, горизонт и доступную информацию;
2. **статистическая процедура** включает преобразования, модель, подбор
   гиперпараметров и порога;
3. **прикладное решение** учитывает ограничения, стоимость и последствия.

Если меняется горизонт с 30 до 7 дней или предсказание требуется до первого
платежа, это уже новая задача и новый набор допустимых признаков. Пересчитать
только метрику недостаточно.

## Смешанные типы данных без утечки

Реальная таблица обычно содержит числовые и категориальные столбцы. Их удобно
обрабатывать раздельно, но оценивать совместно с моделью:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)
preprocess = ColumnTransformer([
    ("num", numeric, numeric_columns),
    ("cat", categorical, categorical_columns),
])
model = make_pipeline(preprocess, LogisticRegression(max_iter=1000))
```

`handle_unknown="ignore"` нужен не ради красивого кода: в эксплуатации может
появиться категория, отсутствовавшая при обучении. Однако молчаливое кодирование
её нулевым вектором следует мониторить — резкий рост неизвестных значений
может означать изменение источника данных.

![[Assets/Sources/Scikit-learn MOOC/figures/api_diagram-columntransformer.svg|Раздельная обработка столбцов в ColumnTransformer]]

*`ColumnTransformer` применяет разные преобразования к группам столбцов и
объединяет результат. Оригинальная иллюстрация из scikit-learn MOOC,
[predictive modeling pipeline](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6),
CC BY 4.0.*

## Диагностика до сравнения сложных моделей

Сначала проверяют не алгоритмы, а постановку. Неожиданно высокий результат
часто объясняется дубликатами, идентификатором, признаком из будущего или
повторением одного объекта в обеих частях разбиения. Неожиданно низкий —
ошибочным горизонтом метки, несовпадением единиц измерения, большим числом
неизвестных категорий либо тем, что доступные признаки вообще не содержат
нужного сигнала.

После обучения полезно разложить ошибки по времени, регионам, источникам данных
и важным группам объектов. Среднее значение метрики способно скрыть провал на
новых клиентах или на последнем месяце. Наконец, нужно сравнить распределения
признаков и долю пропусков между обучением и эксплуатацией. Даже корректный
offline pipeline не гарантирует, что online-процесс формирует те же значения.

Так predictive pipeline становится воспроизводимым контрактом: схема входа,
порядок преобразований, зафиксированная модель, правило выбора решения и набор
проверок после запуска.

Полный оригинальный модуль, включая упражнения и решения, начинается с
[[05 Источники/Courses/Scikit-learn MOOC/jupyter-book/predictive_modeling_pipeline/predictive_modeling_module_intro.md.md|Predictive modeling pipeline]].
