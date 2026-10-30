---
title: Bagging, random forest и gradient boosting
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
primary_sources:
  - https://inria.github.io/scikit-learn-mooc/ensemble/
  - https://scikit-learn.org/stable/modules/ensemble.html
---

# Bagging, random forest и gradient boosting

Одиночное глубокое дерево имеет low bias и high variance: небольшое изменение
выборки способно перестроить верхние splits. Ensembles используют два разных
механизма. Bagging обучает модели независимо и усредняет их ошибки. Boosting
строит модели последовательно, заставляя очередную исправлять текущий ensemble.

## Bagging

Из training set создаются bootstrap samples — выборки того же размера,
полученные случайным выбором объектов с возвращением. На каждой обучается своё
дерево. Для regression predictions усредняются:

$$
\hat f(x)=\frac1B\sum_{b=1}^{B}f_b(x).
$$

Для classification используется усреднение class probabilities или vote.

![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_grey.svg|Bagging data subsets]]
![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_grey_fitted.svg|Bagging fitted trees]]
![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_blue.svg|Bagging averaged prediction]]

*Случайные subsets порождают разные деревья; усреднение сглаживает prediction.
Источник: scikit-learn MOOC,
[bagging slides](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/slides/bagging.md),
CC BY 4.0.*

Если ошибки отдельных моделей имеют одинаковую дисперсию $\sigma^2$ и
одинаковую попарную корреляцию $\rho$, дисперсия среднего в этой модели равна

$$
\rho\sigma^2+\frac{1-\rho}{B}\sigma^2.
$$

Увеличение числа деревьев уменьшает второй член, но не устраняет общую
коррелированную ошибку.

## Random forest

Random forest добавляет к bootstrap случайность признаков. В каждом узле
рассматривается только случайное подмножество features. Сильный признак поэтому
не может автоматически определять верхний split каждого дерева; деревья
декоррелируются, и усреднение работает лучше.

Глубокие base trees в random forest допустимы: каждое переобучается по-своему,
а ensemble уменьшает variance. Больше деревьев обычно не ухудшают
generalization, но увеличивают время обучения, inference latency и размер
модели.

Объекты, не попавшие в bootstrap sample конкретного дерева, образуют
out-of-bag set. Их можно использовать для дополнительной оценки без отдельного
validation pass, хотя финальная схема оценки всё равно должна соответствовать
структуре данных.

## Gradient boosting

Boosting строит additive model:

$$
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta\,h_m(x),
$$

где $h_m$ — обычно неглубокое дерево, а $\eta$ — learning rate. В gradient
boosting новое дерево приближает отрицательный градиент loss по текущему
prediction:

$$
r_{im}=
-\left.
\frac{\partial \ell(y_i,F(x_i))}{\partial F(x_i)}
\right|_{F=F_{m-1}}.
$$

Для $\ell(y,F)=\tfrac12(y-F)^2$ псевдоостаток в точности равен обычному
остатку $y_i-F_{m-1}(x_i)$. Множитель $1/2$ выбран явно: без него
отрицательный градиент был бы вдвое больше. Новое дерево учится предсказывать
эти остатки.

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_orange1.svg|Gradient boosting iteration 1]]
![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_orange2.svg|Gradient boosting iteration 2]]
![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_orange3.svg|Gradient boosting iteration 3]]
![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_orange4.svg|Gradient boosting iteration 4]]

*Последовательные деревья корректируют остаток предыдущего ensemble. Источник:
scikit-learn MOOC,
[boosting slides](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/slides/boosting.md),
CC BY 4.0.*

Главные параметры связаны:

- `max_leaf_nodes` или depth определяют сложность отдельного дерева;
- `learning_rate` уменьшает вклад каждого шага;
- `max_iter` или `n_estimators` задаёт число шагов;
- subsampling и feature sampling добавляют regularization.

Малый learning rate обычно требует больше деревьев. Число итераций удобно
выбирать early stopping по validation score.

## Histogram gradient boosting

Точное дерево сортирует значения features при поиске thresholds. На больших
таблицах это дорого. Histogram algorithms заранее разбивают значения на
ограниченное число bins и ищут splits по агрегированным статистикам. Это
уменьшает вычисления и позволяет эффективно обучаться на миллионах объектов.
На этом принципе построены HistGradientBoosting, LightGBM, XGBoost и CatBoost,
хотя конкретные способы работы с категориями, missing values и leaf growth у
них различаются.

## Bagging и boosting — не одно и то же

| | Bagging / random forest | Gradient boosting |
|---|---|---|
| обучение base models | независимо, можно параллельно | последовательно |
| типичное дерево | глубокое, high variance | неглубокое, high bias |
| основной эффект | уменьшение variance | уменьшение bias, пошаговая оптимизация loss |
| агрегация | среднее или vote | взвешенная сумма |
| чувствительность | сравнительно устойчив | сильнее зависит от learning rate, depth и числа итераций |

## Один источник ошибок, две стратегии

Численный пример показывает, что число деревьев и их разнообразие действуют
по-разному. При $\sigma^2=1$, $B=10$ и $\rho=0{,}2$ формула выше даёт
дисперсию среднего $0{,}2+0{,}8/10=0{,}28$. Если увеличить число деревьев
до 100, получим $0{,}208$. Но при корреляции $0{,}8$ даже сто деревьев
дают $0{,}802$. Усреднение почти одинаковых ошибок мало помогает, поэтому
случайный выбор признаков важен не меньше самого размера леса.

![[Assets/Sources/Scikit-learn MOOC/figures/bagging_trees_predict.svg|Усреднение предсказаний нескольких деревьев]]

*Отдельные деревья дают разные кусочно-постоянные прогнозы, ансамбль их
усредняет. Оригинальная иллюстрация из scikit-learn MOOC,
[bagging module](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/slides/bagging.md),
CC BY 4.0.*

Для OOB-оценки каждый объект предсказывают только деревьями, которые его не
получили при обучении. Вероятность не попасть в одну выборку размера $n$
равна $(1-1/n)^n\approx e^{-1}\approx0{,}368$. При ста деревьях объект
поэтому оценивается в среднем примерно 37 деревьями, а не всем лесом.
Если же соседние строки принадлежат одному человеку, исключение лишь одной
строки не исключает знания о нём: такая OOB-оценка не измеряет перенос на
новых пользователей.

## Boosting как исправление текущих ошибок

Возьмём четыре точки $x=(1,2,3,4)$ с ответами $y=(1,3,5,7)$,
деревья-пни и шаг $\eta=1/2$. Начальная константа $F_0=\bar y=4$
минимизирует квадратичную потерю. Первые остатки равны $(-3,-1,1,3)$.
Пень с условием $x\le2$ предсказывает средние остатки в двух листьях,
то есть $h_1=(-2,-2,2,2)$. После половинного шага
$F_1=(3,3,5,5)$, а новые остатки — $(-2,0,0,2)$.

Теперь оптимальный пень уже может выбрать другую границу. Условия $x\le1$
и $x\le3$ дают одинаковую сумму квадратов; зафиксируем первое. Средние
остатки в листьях равны $-2$ и $2/3$, поэтому
$h_2=(-2,2/3,2/3,2/3)$ и $F_2=(2,10/3,16/3,16/3)$.

| Этап | Прогнозы на четырёх точках | MSE | Средняя $\ell=\tfrac12(y-F)^2$ |
|---|---|---:|---:|
| $F_0$ | $(4,4,4,4)$ | 5 | 2,5 |
| $F_1$ | $(3,3,5,5)$ | 2 | 1 |
| $F_2$ | $(2,10/3,16/3,16/3)$ | 1 | 0,5 |

Второе дерево не переобучает исходные ответы заново: оно исправляет именно
остаток после первого шага. Отдельный прогноз может даже отойти от цели —
у второй точки $3$ превращается в $10/3$, — пока уменьшается суммарная потеря.

Для бинарной классификации $F$ обозначает logit, а не вероятность.
Вероятность равна $p=\sigma(F)$, потеря —
$\ell(y,F)=\log(1+e^F)-yF$, отрицательный градиент — $y-\sigma(F)$.
Например, при $F=0$ положительная метка даёт остаток $1/2$, отрицательная —
$-1/2$. Суммируются деревья в пространстве logits, после чего применяется
sigmoid. Детали выбора значений листьев зависят от реализации, включая
возможные шаги второго порядка; сама целевая переменная градиента от этого
не становится вероятностью.

![[Assets/Sources/Scikit-learn MOOC/figures/boosting0.svg|Начальное приближение boosting]]
![[Assets/Sources/Scikit-learn MOOC/figures/boosting1.svg|Первое исправление boosting]]
![[Assets/Sources/Scikit-learn MOOC/figures/boosting2.svg|Последующее исправление boosting]]
![[Assets/Sources/Scikit-learn MOOC/figures/boosting3.svg|Сумма слабых моделей]]

*Последовательные модели исправляют остатки уже построенного ансамбля.
Оригинальная серия из scikit-learn MOOC,
[boosting slides](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/slides/boosting.md),
CC BY 4.0.*

Learning rate $\eta$ уменьшает вклад каждого дерева. Малое значение обычно
требует большего $M$, зато делает обучение плавнее. Глубина отдельного дерева
задаёт порядок взаимодействий: пни глубины 1 описывают в основном аддитивные
эффекты, более глубокие деревья способны учитывать совместные условия.
Сочетание большого learning rate, глубоких деревьев и многих итераций легко
подгоняет шум.

## Histogram boosting и вычислительный компромисс

Дискретизация делает некоторые различия неразличимыми для дерева. Если значения
признака $(1,2,3,100)$ попали в две ячейки $\{1,2,3\}$ и $\{100\}$,
гистограммный поиск может отделить 100, но уже не сможет поставить порог между
1 и 2. Увеличение числа ячеек возвращает часть точности ценой памяти и числа
кандидатов. Границы ячеек оцениваются на обучающей части и фиксируются для
предсказаний; способ построения и обработка категорий различаются между
библиотеками.

Для пропусков некоторые реализации во время обучения выбирают, в какую ветвь
их направлять. Это не означает, что качество данных можно игнорировать:
изменение механизма пропусков между train и production остаётся сдвигом
распределения.

## Практическая настройка и диагностика

Сначала полезно сравнить один и тот же pipeline для линейной модели,
случайного леса и гистограммного бустинга. Для леса основные ручки — число
деревьев, минимальный размер листа, максимальная глубина и число признаков на
split. Увеличение числа деревьев обычно стабилизирует результат, но почти не
лечит общий bias и увеличивает задержку.

Для бустинга совместно настраивают число итераций, learning rate, сложность
деревьев и регуляризацию листьев. Early stopping отслеживает validation loss и
останавливает добавление деревьев, когда улучшение исчезает. Validation-часть
при этом должна подчиняться тем же временным или групповым ограничениям, что и
финальная оценка.

Характерные симптомы помогают локализовать проблему:

- высокое качество на train и низкое на validation — уменьшить сложность
  деревьев, увеличить листья, усилить регуляризацию;
- низкое качество везде — проверить признаки и target, затем увеличить
  выразительность;
- большой разброс между folds — проверить малые подгруппы, утечку групп и
  нестабильные категории;
- хорошее ранжирование, но плохие вероятности — провести калибровку на данных,
  не участвовавших в обучении ансамбля.

Преимущество ансамблей проявляется на честной процедуре оценки. Если случайный
split смешивает прошлое и будущее или одного пользователя между folds, более
выразительный boosting лишь лучше использует утечку.

## Практика и первоисточники

- [[05 Источники/Courses/Scikit-learn MOOC/slides/bagging.md.md|Bagging slides]];
- [[05 Источники/Courses/Scikit-learn MOOC/slides/boosting.md.md|Boosting slides]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_bagging.py|Bagging]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_random_forest.py|Random forest]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_adaboost.py|AdaBoost]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_gradient_boosting.py|Gradient boosting]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_hist_gradient_boosting.py|Histogram gradient boosting]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_hyperparameters.py|Hyperparameter study]].
