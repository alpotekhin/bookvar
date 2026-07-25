---
title: Bagging, random forest и gradient boosting
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
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

Если ошибки отдельных моделей имеют одинаковую variance $\sigma^2$ и
pairwise correlation $\rho$, variance среднего приближённо равна

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

Полные notebooks:

- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_bagging.py|Bagging]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_random_forest.py|Random forest]].

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

Для squared error pseudo-residual равен обычному residual
$y_i-F_{m-1}(x_i)$. Новое дерево учится предсказывать ошибки ensemble.

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

Полный исходный модуль:

- [[05 Источники/Courses/Scikit-learn MOOC/slides/bagging.md.md|Bagging slides]];
- [[05 Источники/Courses/Scikit-learn MOOC/slides/boosting.md.md|Boosting slides]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_adaboost.py|AdaBoost]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_gradient_boosting.py|Gradient boosting]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_hist_gradient_boosting.py|Histogram gradient boosting]];
- [[05 Источники/Courses/Scikit-learn MOOC/python_scripts/ensemble_hyperparameters.py|Hyperparameter study]].
