---
title: Kernels, SVM и random Fourier features
type: textbook-chapter
status: canonical
last_updated: 2026-10-26
primary_sources:
  - https://mml-book.com/
  - https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
  - https://scikit-learn.org/stable/modules/svm.html
  - https://scikit-learn.org/stable/modules/kernel_approximation.html
  - https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html
  - https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines
---

# Kernels, SVM и random Fourier features

В предыдущей главе смесь распределений описывала плотность данных и скрытую
принадлежность объектов. Теперь у каждого обучающего объекта есть известная
метка, а задача состоит в выборе разделяющей границы. Support vector machine
(SVM), метод опорных векторов, выбирает её по расстояниям до наблюдений.
Распределение вероятностей классов для этого строить не требуется.

Линейная граница в исходных координатах подходит не всегда. Например, прямой
нельзя отделить центральное облако от окружающего кольца. Если заменить
объект $x$ признаками $\phi(x)$, зависимость может стать линейной: функция
от $(x,x^2)$ уже задаёт параболу по исходному одномерному аргументу.
Ядро позволяет вычислять скалярные произведения таких признаков, не
выписывая все их координаты. RFF и Nyström решают другую вычислительную
задачу: строят конечное явное приближение, с которым можно обучать
линейный алгоритм.

<a id="kernel-как-скалярное-произведение"></a>

## Ядро как скалярное произведение

Для отображения $\phi$ функция

$$
k(x,x')=\langle\phi(x),\phi(x')\rangle
$$

называется ядром. Оно может выражать сходство, но его значения не обязаны
быть вероятностями, лежать между 0 и 1 или даже быть неотрицательными.

Для объектов $x_1,\ldots,x_n$ матрица Грама имеет элементы
$K_{ij}=k(x_i,x_j)$. Она симметрична и положительно полуопределена:

$$
a^\top Ka
=\left\|\sum_i a_i\phi(x_i)\right\|^2\ge0.
$$

Условие должно выполняться для любого конечного набора объектов и любых
коэффициентов $a$, а не только для одной имеющейся выборки. Симметричная
функция с этим свойством допускает представление через скалярное произведение
в некотором гильбертовом пространстве. Оно может быть бесконечномерным.

Положительной диагонали недостаточно. У матрицы
$\left[\begin{smallmatrix}1&2\\2&1\end{smallmatrix}\right]$
для $a=(1,-1)^\top$ получается $a^\top Ka=-2$.
Такую матрицу нельзя получить скалярными произведениями вещественных
векторов с обычной евклидовой метрикой. Проверка собственных значений
одной матрицы выявляет подобную ошибку на данной выборке, но не доказывает
корректность функции на всех возможных входах.

<a id="полиномиальный-kernel-без-магии"></a>

### Полиномиальное ядро и явные признаки

Для $x=(x_1,x_2)^\top$ рассмотрим $k(x,x')=(x^\top x')^2$.
Раскрытие скобок даёт

$$
\begin{aligned}
(x_1x'_1+x_2x'_2)^2
&=x_1^2x_1'^2+x_2^2x_2'^2\\
&\quad+2x_1x_2x'_1x'_2.
\end{aligned}
$$

Значит, можно взять

$$
\phi(x)=
\begin{bmatrix}x_1^2\\ \sqrt2x_1x_2\\ x_2^2\end{bmatrix}.
$$

Множитель $\sqrt2$ нужен, чтобы произведение средних координат дало
именно удвоенный смешанный член. Например, для $x=(1,2)$ и $x'=(3,-1)$
исходное скалярное произведение равно 1, квадрат тоже равен 1,
а произведение трёхмерных признаков равно $9-12+4=1$.

В одном измерении неоднородное ядро $(1+xx')^2$ имеет признаки
$(1,\sqrt2x,x^2)$. Для точек $(-1,0,2)$ получается

$$
K=
\begin{bmatrix}
4&1&1\\
1&1&1\\
1&1&25
\end{bmatrix}.
$$

Проверим обе записи, а также отрицательный пример:

```python
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.svm import SVC

def quadratic_features(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2])

points = np.array([[1., 2.], [3., -1.], [-2., 1.]])
Phi = quadratic_features(points)
np.testing.assert_allclose(Phi @ Phi.T, (points @ points.T)**2, atol=1e-13)
kernel_points = np.array([[-1.], [0.], [2.]])
K_poly = polynomial_kernel(kernel_points, gamma=1., coef0=1., degree=2)
np.testing.assert_allclose(K_poly, [[4, 1, 1], [1, 1, 1], [1, 1, 25]])
bad_gram = np.array([[1., 2.], [2., 1.]])
a = np.array([1., -1.])
assert a @ bad_gram @ a == -2
print("polynomial Gram eigenvalues:", np.linalg.eigvalsh(K_poly))
```

Обычные варианты ядра:

$$
\begin{aligned}
k_{\rm linear}(x,x')&=x^\top x',\\
k_{\rm poly}(x,x')&=(\gamma x^\top x'+c_0)^p,\\
k_{\rm RBF}(x,x')&=e^{-\gamma\|x-x'\|^2}.
\end{aligned}
$$

Для приведённого полиномиального семейства достаточно целой степени
$p\ge1$ и $\gamma,c_0\ge0$; произвольные отрицательные параметры такой
гарантии не дают. Для RBF берут $\gamma>0$.
В записи $e^{-\|x-x'\|^2/(2\ell^2)}$ тот же масштаб задаётся
$\ell=1/\sqrt{2\gamma}$. При большом $\gamma$ сходство быстро убывает
с расстоянием, при малом — остаётся заметным на большем расстоянии.
Изменение единиц координат меняет эти расстояния и само ядро.

Геометрическое определение и условие на матрицы Грама разобраны в
[MML, §12.4](https://mml-book.github.io/book/mml-book.pdf).
Теперь используем скалярные произведения в задаче классификации.

## SVM: максимальный зазор

Несколько прямых могут без ошибок разделять одну обучающую выборку.
Их ответы на новые объекты при этом различаются.

![[00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/jake-separators.png]]

*Jake VanderPlas,
[Motivating Support Vector Machines](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html#Motivating-Support-Vector-Machines),
CC BY-NC-ND 3.0 Unported. Оригинал без изменений.*

На рисунке крестик попадает по разные стороны допустимых прямых.
Одной нулевой ошибки на обучающих точках недостаточно для выбора.

### Линейно разделимый случай

Пусть $y_i\in\{-1,+1\}$ и $f(x)=w^\top x+b$, где $w\ne0$.
Предсказание определяется знаком $f(x)$. Само значение $f(x)$ зависит от
масштаба параметров: умножение $w$ и $b$ на одно положительное число
не меняет границу. Поэтому величину $y_if(x_i)$, функциональный отступ,
нельзя напрямую считать расстоянием.

Проекция объекта на гиперплоскость равна

$$
x_{\rm proj}=x-\frac{f(x)}{\|w\|^2}w.
$$

Подстановка даёт $f(x_{\rm proj})=0$, а длина вычтенного вектора равна
$|f(x)|/\|w\|$. Геометрический отступ с учётом правильной стороны —
$y_if(x_i)/\|w\|$. Он не меняется при общем положительном масштабировании
параметров.

Для строго разделимой выборки можно нормировать параметры так, чтобы
$\min_i y_if(x_i)=1$. Меняются параметры уравнения, а не координаты данных.
После такой нормировки расстояние до ближайших наблюдений равно
$1/\|w\|$, а ширина полосы между $f(x)=-1$ и $f(x)=1$ равна $2/\|w\|$.
Поэтому hard-margin SVM решает

$$
\begin{aligned}
\min_{w,b}\;&\frac12\|w\|^2\\
\text{при }&y_i(w^\top x_i+b)\ge1
\quad\text{для всех }i.
\end{aligned}
$$

Множитель $1/2$ упрощает производную и не меняет минимум.
Если в выбранном пространстве признаков не существует строго разделяющей
гиперплоскости, эта система ограничений не имеет решения.

![[00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/jake-support-vectors.png]]

*Jake VanderPlas,
[Fitting a Support Vector Machine](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html#Fitting-a-support-vector-machine),
CC BY-NC-ND 3.0 Unported. Оригинал без изменений.*

Сплошная прямая — граница $f=0$, пунктирные — уровни $f=\pm1$.
Обведённые наблюдения участвуют в определении решения.
Вывод расстояния и нормировки дан в
[MML, §12.1–12.2.3](https://mml-book.github.io/book/mml-book.pdf).

### Четыре точки

Отрицательные точки расположены в $-2,-1$, положительные — в $1,2$.
Решение $w=1$, $b=0$ даёт границу в нуле:

| $x_i$ | $y_i$ | $y_if(x_i)$ |
| ---: | ---: | ---: |
| −2 | −1 | 2 |
| −1 | −1 | 1 |
| 1 | 1 | 1 |
| 2 | 1 | 2 |

Ближайшие точки $-1$ и $1$ ограничивают полосу шириной 2.
Умножение $w$ на 10 оставило бы ту же границу, но нарушило бы выбранную
нормировку ближайших отступов. Оно не увеличило бы геометрический зазор.

В этом примере точки $\pm2$ можно удалить или отодвинуть дальше,
сохранив решение. Если же положительную опорную точку передвинуть с 1
на 0,5, ближайшая пара будет $(-1,0{,}5)$. Граница окажется в −0,25,
полная ширина полосы уменьшится до 1,5, а параметры станут
$w=4/3$, $b=1/3$. Проверим исходный случай настоящим решателем:

```python
X_line = np.array([[-2.], [-1.], [1.], [2.]])
y_line = np.array([-1, -1, 1, 1])
hard = SVC(kernel="linear", C=10., tol=1e-10, shrinking=False).fit(X_line, y_line)
np.testing.assert_allclose(hard.coef_, [[1.]], atol=1e-8)
np.testing.assert_allclose(hard.intercept_, [0.], atol=1e-8)
np.testing.assert_array_equal(hard.support_, [1, 2])
np.testing.assert_allclose(y_line * hard.decision_function(X_line), [2, 1, 1, 2])
reduced = SVC(kernel="linear", C=10., tol=1e-10).fit(X_line[[1, 2]], y_line[[1, 2]])
np.testing.assert_allclose(hard.decision_function(X_line),
                           reduced.decision_function(X_line), atol=1e-8)
moved = SVC(kernel="linear", C=10., tol=1e-10).fit([[-2.], [-1.], [.5], [2.]], y_line)
np.testing.assert_allclose(moved.coef_, [[4/3]], atol=1e-7)
np.testing.assert_allclose(moved.intercept_, [1/3], atol=1e-7)
print("support indices:", hard.support_, "full margin width:", 2/np.linalg.norm(hard.coef_))
```

Здесь используется C-SVC с достаточно большим для этих данных $C$,
а совпадение с hard-margin решением проверено явно.
Большое конечное $C$ не является общим определением hard margin.

![[00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/jake-support-invariance.png]]

*Jake VanderPlas,
[Fitting a Support Vector Machine](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html#Fitting-a-support-vector-machine),
CC BY-NC-ND 3.0 Unported. Оригинал без изменений.*

В исходном эксперименте увеличение выборки с 60 до 120 объектов
не изменило три опорные точки. Добавление наблюдений, нарушающих найденные
ограничения, уже может изменить границу.

### Soft margin и hinge loss

Если в одной координате встречаются обе метки, одновременно выполнить
$f(x)\ge1$ и $f(x)\le-1$ невозможно. Чтобы допустить нарушения, вводят
для каждого объекта переменную $\xi_i\ge0$:

$$
\begin{aligned}
\min_{w,b,\xi}\;&\frac12\|w\|^2+C\sum_i\xi_i,\\
\text{при }&y_if(x_i)\ge1-\xi_i,\\
&\xi_i\ge0.
\end{aligned}
$$

При фиксированных $w,b$ минимальная допустимая переменная равна
$\xi_i=\max(0,1-y_if(x_i))$. Поскольку $C>0$, увеличивать её сверх
этого значения невыгодно. Исключив $\xi$, получаем ту же задачу:

$$
\begin{aligned}
&\min_{w,b}\Bigl\{\frac12\|w\|^2\\
&\qquad+C\sum_i\max(0,1-y_if(x_i))\Bigr\}.
\end{aligned}
$$

Это hinge loss. Для отступа $t=yf(x)$ возможны разные ситуации:

| $t$ | Потеря и положение |
| ---: | --- |
| 2 | 0; верная сторона за границей полосы |
| 1 | 0; граница полосы |
| 0,5 | 0,5; метка верна, но отступ недостаточен |
| −0,5 | 1,5; неверная сторона |

Следовательно, ненулевая $\xi$ не обязательно означает ошибку
классификации. Эта потеря штрафует и верные ответы слишком близко
к границе. При $t<1$ её производная по $t$ равна −1, при $t>1$ — 0;
в точке 1 обычной производной нет, а субградиент лежит в $[-1,0]$.
Для объектов с недостаточным отступом вклад в градиент по $w$
равен $-Cy_ix_i$, к которому добавляется производная регуляризатора $w$.

Большое $C$ увеличивает цену суммарных нарушений, малое сильнее
предпочитает небольшую норму. Это не гарантирует меньшую ошибку на новых
данных и не устраняет противоречивые метки. В этой постановке свободный
член $b$ не регуляризуется.

Иногда цель записывают как среднюю потерю плюс
$\lambda\|w\|^2/2$. После деления приведённой цели на $Cn$
получается $\lambda=1/(Cn)$. Поэтому численное значение коэффициента
регуляризации нельзя сравнивать без соглашения о сумме или среднем.

![[00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/jake-soft-margin.png]]

*Jake VanderPlas,
[Tuning the SVM: Softening Margins](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html#Tuning-the-SVM:-Softening-Margins),
CC BY-NC-ND 3.0 Unported. Оригинал без изменений.*

Панели показывают $C=10$ и $C=0{,}1$ на одинаковых данных.
При меньшем $C$ полоса шире, а больше наблюдений влияют на решение.
Геометрическая и эквивалентная постановка через потери разобраны в
[MML, §12.2.4–12.2.5](https://mml-book.github.io/book/mml-book.pdf).

<a id="от-dual-problem-к-kernel-svm"></a>

## Двойственная задача и опорные векторы

В hard-margin задаче каждому ограничению соответствует множитель
$\alpha_i\ge0$. Лагранжиан имеет вид

$$
\begin{aligned}
\mathcal J(w,b,\alpha)
&=\frac12\|w\|^2\\
&+\sum_i\alpha_i[1-y_i(w^\top x_i+b)].
\end{aligned}
$$

При фиксированных $\alpha$ минимизируем его по исходным переменным.
Производная по $w$ даёт $w=\sum_i\alpha_i y_ix_i$.
Член с $b$ равен $-b\sum_i\alpha_i y_i$: если сумма ненулевая,
инфимум по свободному $b$ равен $-\infty$. Поэтому для конечной
двойственной цели необходимо $\sum_i\alpha_i y_i=0$.

После подстановки $w$ квадратичный член даёт $\|w\|^2/2$,
а линейный член с $w$ — $-\|w\|^2$. Получается

$$
\begin{aligned}
\max_\alpha\;&\sum_i\alpha_i-\frac12\alpha^\top Q\alpha,\\
Q_{ij}&=y_iy_j x_i^\top x_j,\\
\text{при }&\alpha_i\ge0,\qquad \sum_i\alpha_i y_i=0.
\end{aligned}
$$

В soft-margin лагранжиан дополнительно содержит
$C\sum_i\xi_i-\sum_i\alpha_i\xi_i-\sum_i\beta_i\xi_i$,
где $\beta_i\ge0$ отвечает за $\xi_i\ge0$.
Минимизация по $\xi_i$ требует $C-\alpha_i-\beta_i=0$.
Отсюда ограничение $0\le\alpha_i\le C$, а остальная двойственная
цель остаётся той же.

Матрица $Q=\operatorname{diag}(y)K\operatorname{diag}(y)$
положительно полуопределена при корректном ядре. Поэтому двойственная
цель вогнута, а исходная задача выпукла. В soft-margin задаче можно
строго выполнить ограничения, выбрав достаточно большие $\xi_i$;
условие Слейтера гарантирует совпадение оптимальных значений.
В hard-margin случае для этого предполагается строгая разделимость.
В отличие от обучения
смеси EM, локальные неоптимальные минимумы здесь не являются свойством
поставленной выпуклой задачи. Численный ответ проверяют с учётом
допуска решателя. Вывод дан в
[MML, §12.3.1](https://mml-book.github.io/book/mml-book.pdf).

### Условия KKT

В оптимуме выполняются условия дополняющей нежёсткости:

$$
\begin{aligned}
\alpha_i(t_i-1+\xi_i)&=0,\\
(C-\alpha_i)\xi_i&=0,\qquad t_i=y_if(x_i).
\end{aligned}
$$

Они связывают геометрию наблюдения с двойственным коэффициентом:

| Коэффициент | Следствие |
| --- | --- |
| $\alpha_i=0$ | $\xi_i=0$, $t_i\ge1$ |
| $0<\alpha_i<C$ | $\xi_i=0$, $t_i=1$ |
| $\alpha_i=C$ | $t_i=1-\xi_i\le1$ |

Точки с $\alpha_i>0$ называются опорными: только они входят в выражение
для $w$. Опорная точка может лежать внутри полосы или быть ошибочно
классифицированной. Обратное утверждение «каждая точка точно на полосе
имеет положительный коэффициент» в вырожденных случаях неверно.

Для свободной опорной точки, где $0<\alpha_i<C$, можно вычислить
$b=y_i-w^\top x_i$. При нескольких таких точках численно усредняют
оценки. Если все коэффициенты находятся на концах отрезка, это равенство
нельзя применять без проверки: допустимый $b$ находят по неравенствам KKT.

Вернёмся к четырём точкам. При $C=0{,}02$ симметричное решение имеет
$w=0{,}12$, $b=0$, а все $\alpha_i=C$. Тогда отступы равны
$(0{,}24;0{,}12;0{,}12;0{,}24)$: метки верны, но все объекты находятся
внутри полосы. Условия допускают и другие $b$ из $[-0{,}76;0{,}76]$.
При $C=0{,}1$ получается $w=0{,}5$; внешние точки оказываются на
границе полосы, внутренние остаются внутри. При $C=10$ восстанавливается
$w=1$ и опорными остаются только две центральные точки.

```python
soft_models = []
for penalty, expected_w in [(0.02, 0.12), (0.1, 0.5), (10., 1.)]:
    model = SVC(kernel="linear", C=penalty, tol=1e-10,
                shrinking=False).fit(X_line, y_line)
    alpha = np.zeros(len(y_line))
    alpha[model.support_] = np.abs(model.dual_coef_[0])
    margin = y_line * model.decision_function(X_line)
    slack = np.maximum(0., 1 - margin)
    w = model.coef_.ravel()
    np.testing.assert_allclose(w, [expected_w], atol=1e-8)
    np.testing.assert_allclose(w, (alpha*y_line) @ X_line, atol=1e-8)
    np.testing.assert_allclose(alpha @ y_line, 0., atol=1e-9)
    np.testing.assert_allclose(alpha * (margin - 1 + slack), 0., atol=1e-8)
    np.testing.assert_allclose((penalty-alpha) * slack, 0., atol=1e-8)
    primal = 0.5*(w @ w) + penalty*slack.sum()
    dual = alpha.sum() - 0.5*(w @ w)
    np.testing.assert_allclose(primal, dual, atol=1e-8)
    soft_models.append(model)
    print("C:", penalty, "w:", w, "alpha:", alpha, "objective:", primal)
```

Для последнего решения ненулевые коэффициенты равны 0,5 и 0,5.
Их сумма — 1, $\|w\|^2=1$, поэтому двойственная цель равна 0,5.
Прямая цель также равна 0,5: нарушений нет. В первом решении обе цели
равны 0,0728, хотя все hinge losses положительны.

### Kernel SVM

Все скалярные произведения между объектами можно заменить
$k(x_i,x_j)$. Тогда вектор нормали принадлежит пространству признаков,
$w=\sum_i\alpha_i y_i\phi(x_i)$, а предсказание равно

$$
f(x)=\sum_{i:\alpha_i>0}\alpha_i y_i k(x_i,x)+b.
$$

Строка обучающей матрицы Грама содержит сходства одного объекта
со всеми обучающими объектами. Для новых $m$ наблюдений нужна матрица
формы $m\times n_{\rm train}$ с теми же обучающими столбцами
в том же порядке. Новая квадратная матрица $m\times m$ не заменяет её.

Следующий пример проверяет обычный и предварительно вычисленный
варианты RBF SVM, а также формулу предсказания по опорным векторам:

```python
from sklearn.datasets import make_circles

X_ring, y_ring = make_circles(n_samples=120, factor=0.2, noise=0.05, random_state=7)
kernel_gamma = 2.
kernel_model = SVC(kernel="rbf", gamma=kernel_gamma, C=10., tol=1e-9).fit(X_ring, y_ring)
K_ring = rbf_kernel(X_ring, gamma=kernel_gamma)
precomputed = SVC(kernel="precomputed", C=10., tol=1e-9).fit(K_ring, y_ring)
query = np.array([[0., 0.], [1., 0.], [0.5, 0.5]])
query_kernel = rbf_kernel(query, X_ring, gamma=kernel_gamma)
manual_scores = (query_kernel[:, kernel_model.support_] @ kernel_model.dual_coef_[0]
                 + kernel_model.intercept_[0])
np.testing.assert_allclose(manual_scores, kernel_model.decision_function(query), atol=1e-10)
np.testing.assert_allclose(precomputed.decision_function(query_kernel), manual_scores, atol=1e-7)
print("query scores:", manual_scores, "support count:", len(kernel_model.support_))
```

В двоичной модели scikit-learn хранит в `dual_coef_` уже произведения
$\alpha_i y_i$; умножать их на метки повторно не нужно.
Для нескольких классов `SVC` обучает попарные классификаторы, и
расположение коэффициентов сложнее. Формула кода относится к двум классам.
Подробности API — в [руководстве SVM](https://scikit-learn.org/stable/modules/svm.html).

| Линейная граница | RBF SVM |
| --- | --- |
| ![[00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/jake-circles-linear.png]] | ![[00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/jake-circles-rbf.png]] |

*Jake VanderPlas,
[Beyond Linear Boundaries: Kernel SVM](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html#Beyond-linear-boundaries:-Kernel-SVM),
CC BY-NC-ND 3.0 Unported. Два оригинала без изменений.*

В обоих случаях исходные данные двумерны. Для RBF граница нелинейна
в этих координатах, но линейна в пространстве $\phi$.
Функция $r=e^{-\|x\|^2}$ даёт наглядный отдельный пример смены признаков:

![[00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/jake-radial-lift.png]]

*Jake VanderPlas,
[Beyond Linear Boundaries: Kernel SVM](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html#Beyond-linear-boundaries:-Kernel-SVM),
CC BY-NC-ND 3.0 Unported. Оригинал без изменений.*

Центральные наблюдения поднялись по третьей координате выше кольца.
Это иллюстрация полезного дополнительного признака, а не полное
бесконечномерное отображение RBF.

<a id="от-объектов-к-gram-matrix"></a>
<a id="что-означает-kernel-trick-вычислительно"></a>

### Цена точного ядра

Вычисление одной пары RBF для плотных $d$-мерных векторов требует $O(d)$.
Полная матрица Грама содержит $n^2$ элементов: при $n=100\,000$
это $10^{10}$ чисел, 40 GB в FP32 или 80 GB в FP64, без памяти решателя.

Это размер полной плотной матрицы, а не обязательный объём памяти любого
обучения SVM. LIBSVM вычисляет строки по необходимости и использует
ограниченный кэш. При `kernel="precomputed"` полную обучающую матрицу
передаёт вызывающий код. Ограниченный кэш уменьшает память, но может
увеличить число повторных вычислений.

После обучения нужны опорные векторы и их коэффициенты. Одно предсказание
для плотных входов требует $n_{\rm SV}$ вычислений ядра,
то есть порядка $O(n_{\rm SV}d)$ арифметических операций.
Если опорных векторов много, разреженная запись по $\alpha$ всё равно
остаётся дорогой. Поэтому переход к явным конечным признакам может
снизить и стоимость обучения, и стоимость применения модели.

## Random Fourier features: явное приближение RBF

Для стационарного ядра $k(x,x')=\kappa(x-x')$ значение зависит только
от разности координат. Теорема Бохнера связывает непрерывные
положительно определённые стационарные ядра с неотрицательной
спектральной мерой. Для вещественного ядра, нормированного
$\kappa(0)=1$, эту меру можно считать распределением частот:

$$
\kappa(\delta)=
\mathbb E_{\omega\sim p}\cos(\omega^\top\delta).
$$

Для RBF $\kappa(\delta)=e^{-\gamma\|\delta\|^2}$ берут
$\omega\sim\mathcal N(0,2\gamma I_d)$. Действительно, характеристическая
функция нормального вектора с ковариацией $\Sigma_\omega$ равна
$e^{-\delta^\top\Sigma_\omega\delta/2}$; подстановка $2\gamma I$
даёт нужную экспоненту. Эта конструкция и алгоритм выборки приведены в
[Rahimi и Recht, §3 и Algorithm 1](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines).

### Парные синусы и косинусы

Выберем $D$ независимых частот и определим $2D$ признаков:

$$
z(x)=\frac1{\sqrt D}
\begin{bmatrix}
\cos(\omega_1^\top x)\\
\sin(\omega_1^\top x)\\
\vdots\\
\cos(\omega_D^\top x)\\
\sin(\omega_D^\top x)
\end{bmatrix}.
$$

Формула $\cos u\cos v+\sin u\sin v=\cos(u-v)$ даёт

$$
z(x)^\top z(x')=
\frac1D\sum_{r=1}^D\cos(\omega_r^\top(x-x')).
$$

Каждое слагаемое — несмещённая оценка ядра, а среднее уменьшает дисперсию.
При любом наборе частот $\|z(x)\|^2=1$.
Матрица $ZZ^\top$ положительно полуопределена, поскольку построена из
явных признаков, хотя отдельные её элементы могут оказаться отрицательными.

<a id="численный-пример-одного-приближения"></a>

Для $x=0$, $x'=1$ и частот 0,5 и 1,5 получаем

$$
\begin{aligned}
z(0)^\top z(1)
&=\frac{\cos(0{,}5)+\cos(1{,}5)}2\\
&\approx0{,}47416.
\end{aligned}
$$

Точное RBF при $\gamma=0{,}5$ равно $e^{-0.5}\approx0{,}60653$.
Это результат двух зафиксированных частот, а не утверждение о среднем
по правильному гауссовскому распределению. Отдельная гармоника
осциллирует; сходство, убывающее с расстоянием, получается в ожидании
после усреднения частот.

### Запись со случайной фазой

Можно использовать одну координату на частоту:

$$
\begin{aligned}
z_r(x)&=\sqrt{\frac2D}\cos(\omega_r^\top x+b_r),\\
b_r&\sim U[0,2\pi].
\end{aligned}
$$

Частоты выбираются как прежде, фазы независимы от них и друг от друга.
Из тождества произведения косинусов следует

$$
\begin{aligned}
&2\cos(u+b)\cos(v+b)\\
&\quad=\cos(u-v)+\cos(u+v+2b).
\end{aligned}
$$

Среднее второго слагаемого по равномерной фазе равно нулю.
Поэтому скалярное произведение снова несмещённо оценивает RBF.
Теперь размерность равна $D$, а не $2D$, и диагональные значения
$\|z(x)\|^2$ не обязаны точно равняться 1.

```python
from sklearn.kernel_approximation import RBFSampler, Nystroem

def paired_fourier(X, frequencies):
    projection = np.asarray(X) @ frequencies
    D = frequencies.shape[1]
    return np.concatenate([np.cos(projection), np.sin(projection)], axis=1) / np.sqrt(D)

toy_frequencies = np.array([[0.5, 1.5]])
toy_features = paired_fourier([[0.], [1.]], toy_frequencies)
toy_estimate = toy_features[0] @ toy_features[1]
np.testing.assert_allclose(toy_estimate, (np.cos(.5)+np.cos(1.5))/2)
print("two-frequency estimate:", toy_estimate, "exact:", np.exp(-0.5))

map_data = np.array([[0., 0.], [1., -1.], [2., 1.]])
sampler = RBFSampler(gamma=.5, n_components=64, random_state=7).fit(map_data)
Z_phase = sampler.transform(map_data)
manual_phase = np.sqrt(2/64)*np.cos(
    map_data @ sampler.random_weights_ + sampler.random_offset_)
np.testing.assert_allclose(Z_phase, manual_phase, atol=1e-14)
paired = paired_fourier(map_data, sampler.random_weights_)
np.testing.assert_allclose(np.diag(paired @ paired.T), 1., atol=1e-14)
print("paired dimension:", paired.shape[1], "phase dimension:", Z_phase.shape[1])
```

Код использует те же частоты для сравнения формул, но карты отличаются:
парная запись имеет 128 признаков, фазовая — 64.
`RBFSampler` реализует фазовую запись. В классическом RFF частоты
и фазы после выборки фиксированы; обучаются веса следующего линейного
алгоритма. Обучение самих частот меняет метод.

### Как число признаков влияет на ошибку

Для фиксированной пары обозначим $a=k(x,x')$.
В парной записи среднее одного слагаемого равно $a$, а его второй
момент равен $(1+a^4)/2$. Последнее следует из
$\cos^2 u=(1+\cos2u)/2$ и спектра RBF.
Поэтому

$$
\operatorname{Var}[\hat k_D(x,x')]
=\frac{(1-a^2)^2}{2D}.
$$

Стандартное отклонение убывает как $D^{-1/2}$. Это утверждение
о распределении ошибки, а не о монотонном улучшении каждого
конкретного набора частот, каждого элемента матрицы или accuracy.
Равномерная оценка сразу для всех пар требует дополнительных условий
на область входов; в статье Rahimi–Recht рассматривается компактная область.

Следующий эксперимент повторяет выборку карт, а не только сравнивает
один удачный seed:

```python
variance_rng = np.random.default_rng(11)
target_kernel = np.exp(-0.5)  # gamma=0.5, squared distance=1
rff_variance_results = []
for D in [8, 32, 128]:
    frequencies = variance_rng.normal(size=(4000, D))
    estimates = np.cos(frequencies).mean(axis=1)
    predicted_variance = (1-target_kernel**2)**2/(2*D)
    empirical_variance = estimates.var(ddof=1)
    rff_variance_results.append((D, estimates.mean(), empirical_variance, predicted_variance))
    print("D / mean / empirical variance / predicted variance:", rff_variance_results[-1])
```

Для $D=8$ предсказанная дисперсия равна примерно 0,02497,
наблюдаемая — 0,02506. Для $D=128$ получаются 0,001561 и 0,001548.
Это согласие на множестве независимых карт, а не гарантия ошибки
одной конкретной карты.

При сравнении двух записей надо учитывать число выходных координат,
а не только число частот. Ошибка аппроксимации ядра и ошибка
классификации также различаются: после построения карты остаются
выбор линейного алгоритма, его регуляризации и обучение на конечной выборке.

<a id="алгоритм-rff-pipeline"></a>

### Построение и сохранение карты

После выбора $\gamma,D$ и seed частоты и фазы сохраняют вместе со
стандартизатором и линейными весами. Новый `fit` с другой случайностью
изменит координаты, и ранее обученные веса перестанут соответствовать им.
`transform` применяет уже выбранную карту к любому пакету объектов.

Для плотного входа матрица частот содержит $dD$ чисел; вычисление
проекций требует $O(dD)$ на объект, затем считаются $D$ косинусов.
Сам линейный классификатор требует $O(D)$ операций на класс.
Нельзя учитывать только память линейных весов и забывать частоты.
Полную матрицу $n\times D$ можно не хранить, если и преобразование,
и обучение линейной модели выполняются пакетами.

<a id="nyström-другой-путь-к-явным-признакам"></a>

## Nyström: признаки по опорным объектам

Nyström выбирает $m$ объектов из обучающей выборки и использует их
как опоры. Это зависимый от данных способ, в отличие от выборки
частот RFF при заранее фиксированном числовом $\gamma$.

Пусть $S$ — индексы опор, $B=K_{:,S}$ имеет размер $n\times m$,
а $W=K_{S,S}$ — $m\times m$. Если $W$ невырождена, явные признаки
строятся как $Z=BW^{-1/2}$. Тогда

$$
\hat K=ZZ^\top=BW^{-1}B^\top.
$$

Нормировка устраняет повторный учёт корреляций между опорами.
Сами строки $B$ тоже можно использовать как признаки, но их
скалярные произведения дадут $BB^\top$, то есть другую матрицу.
Даже если взять все обучающие объекты, необработанная карта $K$
даст $K^2$, а не $K$.

Для положительно полуопределённой, но вырожденной $W$ используют
псевдообратную. При разложении $W=U\Lambda U^\top$ положительные
собственные значения заменяют на $\lambda_j^{-1/2}$,
нулевые — на 0. Тогда

$$
\begin{aligned}
Z&=B\,W^{\dagger/2},\\
\hat K&=BW^\dagger B^\top.
\end{aligned}
$$

Чтобы увидеть геометрический смысл, представим матрицу Грама конечного
набора объектов как $K=\Phi\Phi^\top$. Строки $\Phi$ — их признаки,
а матрица $A=\Phi_{S,:}$ содержит только строки опор. Тогда
$W=AA^\top$ и $B=\Phi A^\top$. Подстановка даёт

$$
\begin{aligned}
P&=A^\top(AA^\top)^\dagger A,\\
\hat K&=\Phi P\Phi^\top.
\end{aligned}
$$

По SVD матрицы $A$ можно проверить, что $P=P^\top$ и $P^2=P$:
это ортогональный проектор на пространство, натянутое на признаки опор.
Поэтому Nyström сохраняет скалярные произведения проекций исходных
признаков. Разность $K-\hat K=\Phi(I-P)\Phi^\top$
положительно полуопределена и описывает отброшенные направления.
Если все нужные направления уже представлены опорами, матрица
восстанавливается точно. В общем случае равенство $\hat K=K$ неверно.

### Расчёт для трёх объектов

Для уже построенной полиномиальной матрицы выберем первую и третью опоры:

$$
B=
\begin{bmatrix}4&1\\1&1\\1&25\end{bmatrix},
\qquad
W=
\begin{bmatrix}4&1\\1&25\end{bmatrix}.
$$

У второй точки сходства с опорами равны $(1,1)$.
Определитель $W$ равен 99, поэтому её восстановленная диагональ равна

$$
\hat K_{22}
=\frac{25-1-1+4}{99}
=\frac3{11}.
$$

Исходная диагональ равна 1. Все элементы в выбранных строках и столбцах
восстанавливаются, а недостающий остаток здесь равен $8/11$
в единственной диагональной позиции.

```python
def nystrom_from_anchors(X, anchors, kernel, tolerance=1e-12):
    B = kernel(X, anchors)
    W = kernel(anchors, anchors)
    values, vectors = np.linalg.eigh(W)
    if values.min() < -tolerance:
        raise ValueError("Anchor kernel must be positive semidefinite")
    inverse_roots = np.zeros_like(values)
    positive = values > tolerance
    inverse_roots[positive] = 1/np.sqrt(values[positive])
    normalization = (vectors * inverse_roots) @ vectors.T
    return B @ normalization

def example_kernel(X, Y):
    return polynomial_kernel(X, Y, gamma=1., coef0=1., degree=2)

anchors = kernel_points[[0, 2]]
Z_nystrom = nystrom_from_anchors(kernel_points, anchors, example_kernel)
K_approx = Z_nystrom @ Z_nystrom.T
np.testing.assert_allclose(K_approx[1, 1], 3/11)
np.testing.assert_allclose(K_poly-K_approx, np.diag([0., 8/11, 0.]), atol=1e-12)
Z_full = nystrom_from_anchors(kernel_points, kernel_points, example_kernel)
np.testing.assert_allclose(Z_full @ Z_full.T, K_poly, atol=1e-12)

library_map = Nystroem(kernel="poly", gamma=1., coef0=1., degree=2,
                      n_components=2, random_state=7).fit(kernel_points)
manual_map = nystrom_from_anchors(kernel_points, library_map.components_, example_kernel)
np.testing.assert_allclose(library_map.transform(kernel_points), manual_map, atol=1e-10)
print("anchors:", anchors.ravel(), "reconstructed middle diagonal:", K_approx[1, 1])
```

Учебный код отбрасывает собственные значения по абсолютному допуску.
В scikit-learn 1.9.1 `Nystroem` использует SVD и ограничивает сингулярные
значения снизу числом $10^{-12}$ перед вычислением обратного корня.
Для невырожденного примера результаты совпадают; поведение на почти
вырожденной матрице не следует считать буквальной псевдообратной без
учёта этого соглашения.
[Руководство Nyström](https://scikit-learn.org/stable/modules/kernel_approximation.html#nystroem-method-for-kernel-approximation)
описывает выбор опор, нормировку и последующее применение карты.

Выбор опор влияет на качество. Если они не представляют редкую область
выборки, её направления могут не попасть в приближение.
При плотных общих матрицах нужны $m$ вычислений ядра на новый объект
и умножение на матрицу нормировки размером $m\times m$.
Подготовка нормировки требует разложения матрицы $W$, обычно $O(m^3)$;
сама нормировка хранится вместе с опорами.

## Сравнение методов на данных

![[00 Учебник/Assets/Figures/curated/advanced-unsupervised-classical/kernel-approximation.png]]

*The scikit-learn developers,
[Explicit feature map approximation for RBF kernels](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html),
BSD-3-Clause. Оригинал без изменений; полный код сохранён в источниках.*

Слева показано качество классификации рукописных цифр, справа — время
обучения в исходном эксперименте. Число выходных признаков увеличивается
для RFF и Nyström. Данных здесь недостаточно много, чтобы точный SVM
стал непрактичным, поэтому рисунок не доказывает преимущество приближений
на любом объёме. Время зависит от реализации и оборудования.

В сохранённом демонстрационном скрипте центрирование вычисляется до
разбиения. Для собственного эксперимента это исправляют: все оценки
преобразований получают только на обучающей части. Также точный `SVC`
и `LinearSVC` по умолчанию оптимизируют разные потери, а `LinearSVC`
иначе обрабатывает регуляризацию свободного члена. Различие результатов
не следует целиком приписывать приближению ядра.

Следующий пример использует для всех четырёх вариантов один двоичный
C-SVC с обычной hinge loss: линейный, точный RBF, RFF и Nyström.
Объекты делятся на train, validation и test. Выбор конфигурации
проводится по validation, тестовая часть используется после выбора.

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_moons, y_moons = make_moons(n_samples=600, noise=.22, random_state=42)
X_development, X_test, y_development, y_test = train_test_split(
    X_moons, y_moons, test_size=.2, stratify=y_moons, random_state=7)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_development, y_development, test_size=.25,
    stratify=y_development, random_state=7)

candidate_rows = []
for method in ["linear", "exact", "rff", "nystrom"]:
    for penalty in [.1, 1., 10.]:
        for gamma in ([None] if method == "linear" else [.2, 2.]):
            for dimension in ([None] if method in ["linear", "exact"] else [32, 128]):
                steps = [("scale", StandardScaler())]
                if method == "rff":
                    steps.append(("map", RBFSampler(gamma=gamma, n_components=dimension,
                                                    random_state=7)))
                elif method == "nystrom":
                    steps.append(("map", Nystroem(gamma=gamma, n_components=dimension,
                                                 random_state=7)))
                steps.append(("svc", SVC(kernel="rbf" if method == "exact" else "linear",
                    C=penalty, gamma=gamma if gamma is not None else "scale", tol=1e-7)))
                estimator = Pipeline(steps).fit(X_train, y_train)
                candidate_rows.append((estimator.score(X_validation, y_validation),
                                       method, penalty, gamma, dimension, estimator))

best_by_method = {}
for method in ["linear", "exact", "rff", "nystrom"]:
    best_by_method[method] = max(
        [row for row in candidate_rows if row[1] == method], key=lambda row: row[0])
    row = best_by_method[method]
    print("validation choice:", row[:5])
selected = max(candidate_rows, key=lambda row: row[0])
selected_test_accuracy = selected[-1].score(X_test, y_test)
print("selected method:", selected[1], "test accuracy:", selected_test_accuracy)
```

В этом запуске лучшая линейная модель дала 0,8333 на validation;
лучшие точная RBF, RFF и Nyström — по 0,9583. У двух приближений
выбраны 128 признаков, $C=1$, $\gamma=2$. При равенстве оценок
код сохраняет первый вариант в порядке перебора, то есть точную RBF;
её тестовая accuracy равна 0,9583. Такое правило разрешения равенства
не доказывает превосходство точного метода над двумя приближениями.

Это небольшой пример выбора конвейера, не измерение масштабируемости.
Линейный `SVC` на явных признаках выбран ради сопоставимой целевой функции;
для больших данных вместо него используют подходящий линейный решатель,
например `SGDClassifier(loss="hinge")`, уже с его соглашением о регуляризации.

<a id="leakage-и-scaling"></a>

При фиксированном числовом $\gamma$ и одинаковом типе данных
`RBFSampler.fit` использует число координат, а не значения наблюдений.
В режиме `gamma="scale"` частоты зависят от дисперсии входной выборки.
Проверим это различие и применение уже выбранной карты к новым объектам:

```python
fixed_a = RBFSampler(gamma=.5, n_components=64, random_state=7).fit(map_data)
fixed_b = RBFSampler(gamma=.5, n_components=64, random_state=7).fit(map_data*10)
np.testing.assert_array_equal(fixed_a.random_weights_, fixed_b.random_weights_)
scaled_a = RBFSampler(gamma="scale", n_components=64, random_state=7).fit(map_data)
scaled_b = RBFSampler(gamma="scale", n_components=64, random_state=7).fit(map_data*10)
np.testing.assert_allclose(scaled_a.random_weights_, 10*scaled_b.random_weights_)
saved_frequencies = fixed_a.random_weights_.copy()
fixed_a.transform(np.array([[3., -2.]]))
np.testing.assert_array_equal(saved_frequencies, fixed_a.random_weights_)
print("numeric gamma uses a fixed map; scale gamma uses training variance")
```

Случайная карта с заранее заданными числовыми параметрами сама по себе
не оценивает статистики выборки. Но стандартизация, выбор опор Nyström,
режим `gamma="scale"` и подбор гиперпараметров зависят от данных.
В кросс-валидации эти операции должны выполняться внутри обучающих
частей. Выбор seed по тестовому качеству также использует тестовые
данные и нарушает независимость итоговой оценки.

<a id="практический-выбор"></a>
<a id="диагностика-и-типичные-ошибки"></a>

## Практический выбор и диагностика

Начальная линейная модель позволяет проверить, нужен ли нелинейный
признак вообще. Точный RBF SVM на доступной подвыборке помогает оценить
выигрыш от выбранного ядра. Если ограничивают объём данных или цена
предсказания, RFF и Nyström сравнивают по качеству, памяти, времени
преобразования и времени следующего линейного алгоритма.

<a id="совместная-роль-ccc-и-γgammaγ"></a>

$C$ и $\gamma$ подбирают совместно. При слишком локальном ядре модель
может запомнить отдельные наблюдения; увеличение $C$ сильнее наказывает
нарушения на train. Слишком широкое ядро тоже может потребовать другого
$C$. Нельзя заранее назначить одному параметру роль «качества»,
а другому — только «скорости». Число признаков приближения добавляет
ещё один источник ошибки и ещё один настраиваемый бюджет.

<a id="случайность-rff"></a>

Для RFF при небольшом числе признаков проверяют несколько независимых
карт. Для Nyström меняют выбор опор и исследуют малые собственные
значения их матрицы Грама. Хорошая аппроксимация среднего элемента
матрицы не гарантирует сохранение всех полезных для классификации
направлений.

<a id="бюджет-inference"></a>

Вместе с метрикой задачи измеряют число опорных векторов точного SVM,
объём частот RFF или опор и нормировки Nyström, размер пакета и задержку
полного конвейера.

<a id="accuracy-недостаточна"></a>

При несбалансированных классах одной accuracy
недостаточно: нужны метрики по классам и оценка подходящего рабочего порога.
Значение `decision_function` — оценка разделяющей функции, а не вероятность;
для вероятностей требуется отдельная калибровка.

## Практика и первоисточники

- MML, Chapter 12, *Classification with Support Vector Machines*,
  [книга и PDF](https://mml-book.com/), §12.1–12.3.1 и §12.4.
- Jake VanderPlas,
  [In Depth: Support Vector Machines](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html).
  Рисунки сохранены без изменений, CC BY-NC-ND 3.0 Unported
  по лицензии закреплённого репозитория.
- Rahimi, Recht, 2007,
  [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines),
  §3 и Algorithm 1.
- scikit-learn, [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
  и [Kernel Approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html),
  BSD-3-Clause.
- [[05 Источники/Courses/Advanced Unsupervised and Classical ML/sklearn-kernel-approximation.py.md|Полный официальный пример exact kernel, Nyström и RFF]]

**Назад:** [[08 Gaussian mixture и EM|Gaussian mixture и EM]]

К модулям: [[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]
