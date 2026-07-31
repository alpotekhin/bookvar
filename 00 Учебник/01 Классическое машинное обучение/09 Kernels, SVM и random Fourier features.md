---
title: Kernels, SVM и random Fourier features
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
primary_sources:
  - https://mml-book.com/
  - https://scikit-learn.org/stable/modules/kernel_approximation.html
  - https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html
  - https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines
---

# Kernels, SVM и random Fourier features

Нелинейную зависимость можно сделать линейной, если заменить исходный объект
$x$ подходящим вектором признаков $\phi(x)$. Например, линейная функция от
$(x,x^2)$ уже задаёт параболу по исходному одномерному $x$. Kernel methods
развивают эту идею: алгоритм работает со скалярными произведениями в богатом
feature space, не обязательно выписывая его координаты.

## Kernel как скалярное произведение

Функция

$$
k(x,x')=\langle\phi(x),\phi(x')\rangle
$$

измеряет сходство двух объектов в пространстве признаков. Для набора
$x_1,\ldots,x_n$ строится Gram matrix $K_{ij}=k(x_i,x_j)$. Она должна быть
симметричной и positive semidefinite:

$$
a^\top Ka\ge0\quad\text{for every }a\in\mathbb R^n.
$$

Действительно,

$$
a^\top Ka
=\left\|\sum_i a_i\phi(x_i)\right\|^2\ge0.
$$

В обратную сторону PSD-условие гарантирует существование некоторого feature
space. Поэтому не всякая интуитивная «мера похожести» является корректным
kernel.

### Полиномиальный kernel без магии

Для $x=(x_1,x_2)$ рассмотрим

$$
k(x,x')=(x^\top x')^2.
$$

Раскрывая квадрат, получаем

$$
(x_1x'_1+x_2x'_2)^2
=x_1^2x_1'^2+2x_1x_2x'_1x'_2+x_2^2x_2'^2.
$$

Это обычное скалярное произведение векторов

$$
\phi(x)=
\begin{bmatrix}
x_1^2&\sqrt2x_1x_2&x_2^2
\end{bmatrix}^{\top}.
$$

Kernel trick не создаёт другую математику: он вычисляет нужное скалярное
произведение, не материализуя $\phi(x)$.

Часто используют:

$$
k_{\text{linear}}(x,x')=x^\top x',
$$

$$
k_{\text{poly}}(x,x')=(\gamma x^\top x'+c_0)^p,
$$

$$
k_{\text{RBF}}(x,x')=\exp(-\gamma\|x-x'\|^2).
$$

RBF соответствует бесконечномерному feature space. Параметр $\gamma$ задаёт
радиус влияния: при большом $\gamma$ сходство быстро исчезает и граница может
стать очень извилистой; при малом многие объекты выглядят похожими и модель
приближается к гладкой.

## SVM: максимальный зазор

### Линейно разделимый случай

Для меток $y_i\in\{-1,+1\}$ граница имеет вид
$f(x)=w^\top x+b$. Масштаб $(w,b)$ можно выбрать так, чтобы ближайшие точки
удовлетворяли $y_if(x_i)=1$. Тогда hard-margin SVM решает

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to}\quad
y_i(w^\top x_i+b)\ge1.
$$

Расстояние от плоскости $w^\top x+b=0$ до каждой margin plane равно
$1/\|w\|$, а полная ширина полосы — $2/\|w\|$. Минимизация нормы максимизирует
зазор.

В одномерном примере отрицательные точки расположены в $-2,-1$, положительные
— в $1,2$. Решение $w=1$, $b=0$ даёт границу в нуле. Точки $-1$ и $1$
касаются margin и являются support vectors; точки $\pm2$ можно удалить, не
изменив решение.

Это легко увидеть без рисунка, вычислив signed functional margin
$y_i(w x_i+b)$:

| $x_i$ | $y_i$ | $f(x_i)=x_i$ | $y_if(x_i)$ | Роль |
|---:|---:|---:|---:|---|
| -2 | -1 | -2 | 2 | лежит за margin, $\alpha_i=0$ |
| -1 | -1 | -1 | 1 | касается левой margin plane, support vector |
| 1 | 1 | 1 | 1 | касается правой margin plane, support vector |
| 2 | 1 | 2 | 2 | лежит за margin, $\alpha_i=0$ |

Граница решения находится там, где $f(x)=0$; две границы полосы — там, где
$f(x)=-1$ и $f(x)=1$. Только две центральные точки удерживают полосу шириной
2. Если немного передвинуть $x=-2$ или $x=2$, optimum не изменится. Если
передвинуть один support vector к нулю, допустимый зазор сузится и изменятся
$w$ и $b$. Поэтому название support vector описывает не близость к центру
класса, а участие точки в определении оптимальной границы.

### Soft margin и hinge loss

Реальные данные редко разделимы без ошибок. Вводятся slack variables:

$$
\min_{w,b,\xi}
\frac12\|w\|^2+C\sum_i\xi_i,
\qquad
y_i(w^\top x_i+b)\ge1-\xi_i,\quad \xi_i\ge0.
$$

Эквивалентная unconstrained форма использует hinge loss:

$$
\min_{w,b}
\frac12\|w\|^2+
C\sum_i\max(0,1-y_if(x_i)).
$$

$C$ определяет цену нарушения margin. Большое $C$ сильнее подгоняет train,
малое допускает больше нарушений ради широкой и гладкой границы. Поскольку
$\|w\|$ зависит от масштаба координат, standardization для SVM почти всегда
часть модели.

## От dual problem к kernel SVM

Lagrange dual hard-margin задачи имеет вид

$$
\max_{\alpha}
\sum_i\alpha_i
-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^\top x_j
$$

при $\alpha_i\ge0$ и $\sum_i\alpha_i y_i=0$. В soft-margin случае добавляется
ограничение $\alpha_i\le C$. Объекты входят в objective только через
$x_i^\top x_j$, поэтому его можно заменить kernel:

$$
x_i^\top x_j\longrightarrow k(x_i,x_j).
$$

Предсказание становится

$$
f(x)=
\sum_{i:\alpha_i>0}\alpha_i y_i k(x_i,x)+b.
$$

Только точки с $\alpha_i>0$ — support vectors — участвуют в решении. Это
делает модель разреженной относительно train set, но не обязательно дешёвой:
если support vectors десятки тысяч, каждое предсказание требует десятки тысяч
kernel evaluations.

### От объектов к Gram matrix

Для трёх одномерных объектов $x=(-1,0,2)$ polynomial kernel
$k(x,x')=(1+xx')^2$ даёт

$$
K=
\begin{bmatrix}
4&1&1\\
1&1&1\\
1&1&25
\end{bmatrix}.
$$

Строка матрицы содержит сходство одного train object со всеми остальными.
Dual SVM никогда не обязан хранить координаты $\phi(x)$: ему достаточно этой
матрицы и меток. Для нового объекта $x_*$ вычисляется уже не новая квадратная
матрица, а вектор $[k(x_1,x_*),\ldots,k(x_n,x_*)]$; коэффициенты support
vectors взвешивают его элементы в формуле $f(x_*)$.

Так kernel trick меняет место вычислительной цены. Явное преобразование платит
за ширину $\phi(x)$, а exact kernel method — за попарные сходства. При $n$
объектах даже один dense Gram matrix содержит $n^2$ чисел: для
$n=100\,000$ это $10^{10}$ элементов, или примерно 40 GB в FP32, ещё до
рабочей памяти solver.

### Что означает kernel trick вычислительно

Exact kernel algorithm обычно хранит или многократно вычисляет матрицу
$n\times n$. Память растёт как $O(n^2)$, а обучение может быть ещё дороже.
Kernel SVM особенно силён на умеренных, не слишком шумных табличных данных, но
не является автоматическим выбором для миллионов объектов.

## Random Fourier features: явное приближение RBF

Теорема Бохнера утверждает: непрерывный shift-invariant positive-definite
kernel является Fourier transform неотрицательной меры. После нормировки

$$
k(x-x')=
\mathbb E_{\omega\sim p(\omega)}
\left[\cos\bigl(\omega^\top(x-x')\bigr)\right].
$$

Для Gaussian RBF спектральная плотность $p(\omega)$ также Gaussian. Выбирая
$D$ случайных частот, можно построить paired sin/cos map

$$
z(x)=\frac1{\sqrt D}
\begin{bmatrix}
\cos(\omega_1^\top x)\\
\sin(\omega_1^\top x)\\
\vdots\\
\cos(\omega_D^\top x)\\
\sin(\omega_D^\top x)
\end{bmatrix},
$$

для которого

$$
z(x)^\top z(x')
=\frac1D\sum_{r=1}^D
\cos\bigl(\omega_r^\top(x-x')\bigr)
\approx k(x-x').
$$

Это Monte Carlo approximation. Оно unbiased при правильном распределении
частот, а типичная ошибка уменьшается примерно как $1/\sqrt D$.

### Численный пример одного приближения

Пусть $x=0$, $x'=1$, а две уже sampled частоты равны
$\omega_1=0{,}5$ и $\omega_2=1{,}5$. Тогда

$$
z(0)^\top z(1)
=\frac{\cos(0{,}5)+\cos(1{,}5)}2
\approx\frac{0{,}878+0{,}071}{2}
\approx0{,}474.
$$

Для RBF с $\gamma=0{,}5$ точное значение
$e^{-0.5}\approx0{,}607$. Две частоты дают заметную ошибку; с ростом $D$
усреднение стабилизируется. Этот пример показывает, почему `n_components`
нельзя выбирать только по скорости.

Каждая sampled частота вносит один «голос» о сходстве пары. Для разности
$x-x'=-1$ два голоса из примера выглядят так:

| Частота | Фаза $\omega_r(x-x')$ | Вклад $\cos(\omega_r(x-x'))$ |
|---:|---:|---:|
| $\omega_1=0{,}5$ | $-0{,}5$ | $0{,}878$ — точки похожи на этой гармонике |
| $\omega_2=1{,}5$ | $-1{,}5$ | $0{,}071$ — почти ортогональны |
| **среднее** | — | **0,474** |

Отдельная гармоника не обязана быть похожа на RBF. Kernel возникает после
усреднения множества независимо sampled гармоник. У близких объектов мала
разность $x-x'$, поэтому большинство cosines близки к единице; у далёких фазовые
сдвиги расходятся и положительные и отрицательные вклады взаимно сокращаются.
После построения $z(x)$ kernel matrix больше не нужна: linear model работает с
явными строками признаков и может обучаться mini-batches.

В популярной записи со случайной фазой

$$
z_r(x)=\sqrt{\frac2D}\cos(\omega_r^\top x+b_r),
\qquad b_r\sim U[0,2\pi],
$$

получается то же приближение. В классическом RFF частоты и фазы не обучаются:
после случайного отображения обучается обычная линейная модель.

### Алгоритм RFF pipeline

```text
Вход: train X, число признаков D, RBF bandwidth γ
1. Оценить preprocessing только на train.
2. Sample D частот ωr из спектра RBF и D фаз br; сохранить seed.
3. Вычислять z(x) mini-batches, не строя Gram matrix.
4. Обучить linear classifier/regressor на z(X).
5. Подбирать γ, D и regularization совместно по validation.
6. Для inference сохранить scaler, ω, b и linear weights.
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/advanced-unsupervised-classical/kernel-approximation.png]]

*Качество и время обучения exact RBF SVM, Nyström и random Fourier features
при разном числе компонент. Это неизменённый результат официального примера
scikit-learn
[`plot_kernel_approximation.py`](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html),
BSD-3-Clause. Полный код примера сохранён локально.*

## Nyström: другой путь к явным признакам

RFF sample частоты из спектра kernel и не смотрит на распределение конкретной
выборки. Nyström, напротив, выбирает $m$ landmark objects и аппроксимирует Gram
matrix через соответствующие столбцы. Если

$$
C=K_{:,S},\qquad W=K_{S,S},
$$

то

$$
K\approx CW^\dagger C^\top.
$$

Фактор $CW^{\dagger/2}$ служит явным набором признаков. Nyström может быть
точнее при удачных data-dependent landmarks, но чувствителен к их выбору. RFF
удобен для streaming и stationary kernels; ни один метод не доминирует всегда.

## Практический выбор

| Метод | Память/представление | Сильная сторона | Основной риск |
|---|---|---|---|
| Linear SVM | исходные $d$ признаков | большие sparse данные | не ловит нелинейность |
| Exact kernel SVM | kernel rows/support vectors | точная богатая граница на умеренном $n$ | квадратичная память и дорогой inference |
| Nyström + linear | $m$ data-dependent features | использует структуру выборки | плохие landmarks |
| RFF + linear | $D$ random features | mini-batches, streaming, фиксированная цена inference | Monte Carlo variance |

Начинать полезно с linear baseline. Затем exact RBF SVM на подвыборке показывает,
есть ли вообще выигрыш от нелинейности. Если выигрыш есть, но exact model не
масштабируется, сравнивают Nyström и RFF на одинаковом бюджете памяти и latency.

## Диагностика и типичные ошибки

### Leakage и scaling

Scaler, kernel approximation и подбор гиперпараметров должны находиться внутри
каждого fold cross-validation. Если `RBFSampler` обучен до split, seed и
преобразование уже связаны со всей выборкой.

### Совместная роль $C$ и $\gamma$

Большое $\gamma$ создаёт локальные, быстро меняющиеся признаки; большое $C$
слабо прощает ошибки. Их сочетание легко переобучается. Малое $\gamma$ и малое
$C$ дают слишком гладкую границу. Поэтому параметры подбирают совместно, часто
по логарифмической сетке.

### Accuracy недостаточна

При несбалансированных классах нужны PR-AUC, per-class recall, calibration и
выбор рабочего threshold. `SVC` возвращает margin score, а вероятности требуют
отдельной калибровки; нельзя автоматически читать score как probability.

### Бюджет inference

Для exact SVM измеряют число support vectors и latency на реальном batch size.
Для RFF — время построения $D$ признаков и память linear weights. Быстрое
обучение не гарантирует дешёвое serving.

### Случайность RFF

При малом $D$ качество зависит от seed. Следует показать среднее и разброс
нескольких feature maps либо увеличить $D$ до устойчивого режима. Сравнивать
разные $D$ нужно при повторно настроенной regularization linear head.

## Исходные материалы и полные версии

- MML, Chapter 12, *Classification with Support Vector Machines*,
  [страница книги и PDF](https://mml-book.com/), CC BY-NC-SA 4.0.
- Rahimi & Recht, [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines).
- scikit-learn, [Kernel Approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html),
  BSD-3-Clause.
- [[05 Источники/Courses/Advanced Unsupervised and Classical ML/sklearn-kernel-approximation.py.md|Полный официальный пример exact kernel, Nyström и RFF]]

**Назад:** [[08 Gaussian mixture и EM]]  
**Дальше:** [[01 Нейрон и MLP]]
