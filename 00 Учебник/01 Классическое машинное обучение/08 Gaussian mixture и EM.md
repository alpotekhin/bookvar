---
title: Gaussian mixture и EM — обучение со скрытой принадлежностью
type: textbook-chapter
status: canonical
last_updated: 2026-10-25
primary_sources:
  - https://mml-book.com/
  - https://scikit-learn.org/stable/modules/mixture.html
  - https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
  - https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
  - https://ermongroup.github.io/cs228-notes/learning/latent/
---

# Gaussian mixture и EM — обучение со скрытой принадлежностью

Два облака наблюдений могут перекрываться, иметь разный разброс и быть
вытянутыми в разных направлениях. Одного расстояния до центра тогда
недостаточно, чтобы оценить происхождение точки. Gaussian mixture model
(GMM), смесь гауссовских распределений, описывает каждое облако средним,
ковариацией и весом. Для нового объекта вычисляется вероятность каждой
компоненты при условии, что модель смеси верна.

В отличие от PCA из предыдущей главы, здесь скрытое представление —
дискретный номер компоненты. Он не записан в данных, поэтому обучение
чередует оценку вероятностей принадлежности и переоценку параметров.
Это алгоритм expectation-maximization, EM.

## Многомерная гауссовская плотность

Для столбца $x\in\mathbb R^d$ среднее $\mu\in\mathbb R^d$ задаёт центр,
а ковариационная матрица $\Sigma\in\mathbb R^{d\times d}$ имеет элементы
$\Sigma_{ij}=\mathbb E[(X_i-\mu_i)(X_j-\mu_j)]$.
На диагонали стоят дисперсии отдельных координат; внедиагональные элементы
описывают их совместные изменения. В невырожденной модели $\Sigma$
симметрична и положительно определена: $v^\top\Sigma v>0$ при $v\ne0$.
Тогда существует обратная матрица, её определитель положителен, а плотность равна

$$
\begin{aligned}
\delta^2(x)&=(x-\mu)^\top\Sigma^{-1}(x-\mu),\\
\mathcal N(x\mid\mu,\Sigma)&=
\frac{e^{-\delta^2(x)/2}}{(2\pi)^{d/2}\sqrt{\det\Sigma}}.
\end{aligned}
$$

Квадратичная форма измеряет расстояние от центра с учётом масштаба и
корреляций. В собственном базисе $\Sigma$ это сумма квадратов координат,
делённых на соответствующие собственные значения. Поэтому поверхности равной
плотности — эллипсоиды: направления задают собственные векторы, а ширины
пропорциональны корням из собственных значений. Множитель с определителем
уменьшает высоту более широкой плотности, чтобы интеграл оставался равным 1.

Например, при $\mu=(0,0)^\top$ и $\Sigma=\operatorname{diag}(4,1)$
точка $(2,1)^\top$ имеет квадрат такого расстояния $2^2/4+1^2=2$.
Её плотность равна $e^{-1}/(4\pi)$, то есть в $e$ раз меньше плотности
в центре. Сдвиг на две единицы вдоль первой оси здесь столь же значим,
как сдвиг на одну вдоль второй. Этот механизм подробно разобран в
[MML, гл. 6 и 11](https://mml-book.github.io/book/mml-book.pdf).

Для вычисления логарифма плотности не требуется явно обращать ковариацию.
Разложим её по Холецкому: $\Sigma=LL^\top$, где $L$ — нижняя
треугольная матрица с положительной диагональю. Решение
$Lu=x-\mu$ даёт $\delta^2=u^\top u$, а
$\log\det\Sigma=2\sum_j\log L_{jj}$. Тогда

$$
\begin{aligned}
\log\mathcal N(x\mid\mu,\Sigma)
&=-\frac d2\log(2\pi)\\
&\quad-\frac12\log\det\Sigma-\frac12\delta^2.
\end{aligned}
$$

```python
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, xlogy
from scipy.stats import multivariate_normal

def gaussian_logpdf(X, mean, covariance):
    X = np.atleast_2d(X)
    L = np.linalg.cholesky(covariance)
    u = solve_triangular(L, (X - mean).T, lower=True)
    logdet = 2 * np.log(np.diag(L)).sum()
    return -0.5 * (X.shape[1] * np.log(2 * np.pi) + logdet + (u*u).sum(axis=0))

probe = np.array([[0., 0.], [2., 1.]])
covariance = np.diag([4., 1.])
log_density = gaussian_logpdf(probe, np.zeros(2), covariance)
np.testing.assert_allclose(log_density,
    multivariate_normal.logpdf(probe, mean=np.zeros(2), cov=covariance))
np.testing.assert_allclose(log_density[1] - log_density[0], -1)
print("density at (2,1):", np.exp(log_density[1]))
```

Разложение завершится ошибкой для неположительно определённой матрицы.
Это проверка предположения модели: произвольная симметричная матрица
не обязательно является допустимой невырожденной ковариацией.

## От скрытой переменной к наблюдаемой плотности

Для объекта $x_n\in\mathbb R^d$ генеративный процесс состоит из двух шагов:

$$
\begin{aligned}
z_n&\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_K),\\
x_n\mid z_n=k&\sim\mathcal N(\mu_k,\Sigma_k).
\end{aligned}
$$

$z_n$ — ненаблюдаемый номер компоненты, $\pi_k\ge0$ и
$\sum_k\pi_k=1$. После суммирования по возможным $z_n$ получаем

$$
p(x_n\mid\theta)=
\sum_{k=1}^{K}\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k).
$$

Для независимой выборки нужно максимизировать

$$
\ell(\theta)=
\sum_{n=1}^N
\log\left[
\sum_{k=1}^K\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)
\right].
$$

Сумма компонент находится внутри логарифма. Если бы $z_n$ были известны,
задача распалась бы на отдельные оценки гауссовских распределений.
При неизвестных номерах компонент каждая точка участвует в оценке
нескольких распределений с разными весами. EM пересчитывает эти веса
по текущим параметрам, а затем параметры по найденным весам.

## E-step: насколько каждая компонента объясняет объект

При текущих параметрах формула Байеса даёт апостериорную вероятность
принадлежности, или *responsibility*:

$$
\begin{aligned}
\gamma_{nk}&=p(z_n=k\mid x_n,\theta^{old})\\
&=\frac{\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal N(x_n\mid\mu_j,\Sigma_j)}.
\end{aligned}
$$

Для каждого $n$ выполняется $\sum_k\gamma_{nk}=1$. Вероятность учитывает не
только расстояние до среднего, но и исходный вес компоненты, форму
ковариации и нормировочный множитель плотности.

### Один объект, две компоненты

Пусть в одном измерении $\pi_1=\pi_2=0{,}5$, $\mu_1=0$, $\mu_2=4$, а обе
дисперсии равны 1. Для $x=1$ ненормированные вероятности пропорциональны

$$
\begin{aligned}
0{,}5e^{-(1-0)^2/2}&=0{,}5e^{-0.5},\\
0{,}5e^{-(1-4)^2/2}&=0{,}5e^{-4.5}.
\end{aligned}
$$

Общие множители сокращаются, поэтому

$$
\begin{aligned}
\gamma_1&=\frac{e^{-0.5}}{e^{-0.5}+e^{-4.5}}
\approx0{,}982,\\
\gamma_2&\approx0{,}018.
\end{aligned}
$$

В точке $x=2$, посередине между средними, обе вероятности равны 0,5.
Жёсткая метка оставила бы только один из двух вариантов.

Изменение весов смеси сдвигает границу даже при тех же средних
и дисперсиях. Если $\pi_1=0{,}9$, $\pi_2=0{,}1$, то в середине
вероятности будут 0,9 и 0,1. Равновероятная точка определяется условием
$\pi_1e^{-x^2/2}=\pi_2e^{-(x-4)^2/2}$ и равна
$x=2+\tfrac14\log9\approx2{,}5493$. Более частая компонента объясняет
и часть точек за геометрической серединой. При разных ковариациях
меняются также нормировочные множители и сами расстояния.

![[00 Учебник/Assets/Figures/curated/gmm-2026-10-25/jake-uncertainty.png]]

*Jake VanderPlas,
[Generalizing E–M: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html),
CC BY-NC-ND 3.0 Unported. Исходный рисунок без изменений.*

Размер точки в исходном коде равен $50\max_k\gamma_{nk}^2$.
Поэтому возле перекрытия двух групп точки мельче: ни одна компонента
не имеет почти единичной вероятности. Это неопределённость выбора
компоненты в данной модели, а не оценка ошибки всех её предположений.

### Уверенная принадлежность не означает высокую плотность

У объекта далеко от обеих компонент одна плотность всё равно может
превосходить другую на много порядков. После нормировки получится
почти единичная вероятность одной компоненты, хотя плотность смеси
в этой точке очень мала. Поэтому при поиске аномалий
отдельно рассматривают $\log p(x)$ и энтропию принадлежности.

```python
def expectation_1d(x, weights, means, variances):
    x = np.asarray(x, dtype=float)
    log_joint = np.log(weights)[None, :] - 0.5 * (
        np.log(2 * np.pi * variances)[None, :]
        + (x[:, None] - means[None, :]) ** 2 / variances[None, :])
    log_probability = logsumexp(log_joint, axis=1)
    responsibilities = np.exp(log_joint - log_probability[:, None])
    return responsibilities, log_probability, log_joint

weights_old = np.array([0.5, 0.5])
means_old = np.array([0., 4.])
variances_old = np.ones(2)
posterior, point_scores, point_log_joint = expectation_1d(
    [1., 2., 100.], weights_old, means_old, variances_old)
entropy = -xlogy(posterior, posterior).sum(axis=1)
print("posterior:\n", posterior)
print("log density:", point_scores, "entropy:", entropy)
assert np.exp(point_log_joint[-1]).sum() == 0
assert np.isfinite(point_scores[-1])
```

Для $x=2$ энтропия равна $\log2$, а логарифм плотности — примерно
−2,91894. Для $x=100$ логарифм плотности около −4609,61209,
но вторая компонента получает вероятность почти 1 и энтропия почти
нулевая. Низкая энтропия сама по себе не делает такой объект обычным.
Прямое вычисление двух экспонент здесь дало бы $0/0$;
вычитание `logsumexp` сохраняет нормированные вероятности.

## M-step: взвешенные оценки

Эффективное число объектов компоненты

$$
N_k=\sum_{n=1}^{N}\gamma_{nk}.
$$

Если бы номер компоненты был известен, индикатор
$a_{nk}=\mathbf1[z_n=k]$ выбирал бы одно слагаемое в полной
логарифмической функции правдоподобия:

$$
\begin{aligned}
&\log p(X,z\mid\theta)\\
&=\sum_{n,k}a_{nk}\log\pi_k\\
&\quad+\sum_{n,k}a_{nk}\log\mathcal N(x_n\mid\mu_k,\Sigma_k).
\end{aligned}
$$

На E-step вычислена условная ожидаемая величина этого индикатора:
$\mathbb E[a_{nk}\mid X,\theta^{old}]=\gamma_{nk}$.
Поэтому на M-step максимизируется функция

$$
\begin{aligned}
&Q(\theta\mid\theta^{old})\\
&=\sum_{n,k}\gamma_{nk}\log\pi_k\\
&\quad+\sum_{n,k}\gamma_{nk}\log\mathcal N(x_n\mid\mu_k,\Sigma_k).
\end{aligned}
$$

В этой функции $\gamma$ зафиксированы: они рассчитаны по старым
параметрам, а оптимизируются новые. Взвешенные оценки имеют вид

$$
\begin{aligned}
\pi_k^{new}&=\frac{N_k}{N},\\
\mu_k^{new}&=\frac{1}{N_k}\sum_n\gamma_{nk}x_n,
\end{aligned}
$$

Для ковариации обозначим отклонение от нового среднего
$r_{nk}=x_n-\mu_k^{new}$. Тогда

$$
\Sigma_k^{new}=\frac{1}{N_k}\sum_n\gamma_{nk}r_{nk}r_{nk}^\top.
$$

Для весов смеси нужно максимизировать $\sum_kN_k\log\pi_k$
при $\sum_k\pi_k=1$. Условие Лагранжа даёт
$N_k/\pi_k=\lambda$, а суммирование по $k$ — $\lambda=N$.
Отсюда вес компоненты равен её доле в общей сумме принадлежностей.

Производная $Q$ по среднему содержит
$\Sigma_k^{-1}\sum_n\gamma_{nk}(x_n-\mu_k)$.
При положительно определённой ковариации она обращается в ноль
на взвешенном среднем. Для ковариации удобно использовать матрицу
точности $\Lambda_k=\Sigma_k^{-1}$ и взвешенную матрицу рассеяния

$$
S_k=\sum_n\gamma_{nk}
(x_n-\mu_k^{new})(x_n-\mu_k^{new})^\top.
$$

Зависящая от неё часть цели равна
$\tfrac{N_k}{2}\log\det\Lambda_k-\tfrac12\operatorname{tr}(\Lambda_kS_k)$.
Её производная даёт $N_k\Lambda_k^{-1}=S_k$, то есть
$\Sigma_k=S_k/N_k$. В знаменателе стоит $N_k$, а не $N_k-1$:
это максимум правдоподобия, а не несмещённая оценка дисперсии.
Отклонения берутся от нового среднего; веса при этом остаются
найденными на предыдущем E-step. При нулевом $N_k$ такое обновление
не определено, а при вырожденном $S_k$ невырожденный максимум
может отсутствовать.

### Один цикл целиком: от старой плотности к новой

Продолжим пример с $\pi_1=\pi_2=0{,}5$, $\mu_1=0$, $\mu_2=4$ и точками
$(0,1,4,5)$. В этом численном цикле дисперсии фиксированы и равны единице:
M-step обновляет только веса смеси и средние. Это ограниченная GMM; формула
обновления ковариации выше относится к варианту, где её тоже оценивают.

Подставив старые параметры в E-step, получаем для любой точки $x$

$$
\begin{aligned}
\gamma_1(x)&=\frac{e^{-x^2/2}}{e^{-x^2/2}+e^{-(x-4)^2/2}}\\
&=\frac{1}{1+e^{4x-8}},\\
\gamma_2(x)&=1-\gamma_1(x).
\end{aligned}
$$

Так, при $x=1$ снова получается $0{,}982014$, как в предыдущем разборе.
Каждая точка участвует в обеих компонентах со своими весами:

| $x_n$ | $\gamma_{n1}$ | $\gamma_{n2}$ |
|---:|---:|---:|
| 0 | 0,999665 | 0,000335 |
| 1 | 0,982014 | 0,017986 |
| 4 | 0,000335 | 0,999665 |
| 5 | 0,000006 | 0,999994 |
| сумма | 1,982020 | 2,017980 |

Здесь округлены только показанные числа: произведения и суммы вычислены из
неокруглённых вероятностей. Взвешенные суммы координат равны
приблизительно 0,983386 и 9,016614. M-step делит их на эффективные
числа объектов:

$$
\begin{aligned}
\mu_1^{new}&\approx\frac{0{,}983386}{1{,}982020}\approx0{,}496153,\\
\mu_2^{new}&\approx\frac{9{,}016614}{2{,}017980}\approx4{,}468138.
\end{aligned}
$$

Новые веса смеси также не обязаны остаться одинаковыми:

$$
\begin{aligned}
\pi_1^{new}&=N_1/4\approx0{,}495505,\\
\pi_2^{new}&=N_2/4\approx0{,}504495.
\end{aligned}
$$

Оба средних сдвинулись вправо. Суммарный логарифм правдоподобия при
фиксированных дисперсиях вырос с $-7{,}429516$ до $-6{,}944253$. Это проверка
данного шага, а не доказательство глобальной сходимости. На следующем E-step
нужно подставить новые средние и веса смеси: прежние вероятности
принадлежности уже не соответствуют новой плотности.

Если одновременно обновлять дисперсию, точки с высокой вероятностью принадлежности сильнее
влияют и на ширину своей компоненты. Например, для первой компоненты вклад
точки $x=5$ в сумму квадратов умножается примерно на $0{,}000006$, а не считается как
полноценное наблюдение. Именно этим EM отличается от временного жёсткого
разбиения данных и отдельного обучения двух гауссовских распределений.

```python
x_train = np.array([0., 1., 4., 5.])
gamma_old, logp_old, logjoint_old = expectation_1d(
    x_train, weights_old, means_old, variances_old)
counts = gamma_old.sum(axis=0)
weights_new = counts / len(x_train)
means_new = gamma_old.T @ x_train / counts
variances_new = (gamma_old * (x_train[:, None] - means_new) ** 2).sum(axis=0) / counts
_, logp_fixed, _ = expectation_1d(x_train, weights_new, means_new, variances_old)
gamma_new, logp_full, logjoint_new = expectation_1d(
    x_train, weights_new, means_new, variances_new)
np.testing.assert_allclose(gamma_old.sum(axis=1), 1)
assert logp_fixed.sum() > logp_old.sum()
assert logp_full.sum() > logp_fixed.sum()
print("counts:", counts, "means:", means_new)
print("new variances:", variances_new)
print("log likelihood old/fixed/full:", logp_old.sum(), logp_fixed.sum(), logp_full.sum())
```

При свободном обновлении дисперсии становятся приблизительно
0,252078 и 0,359264. Суммарное правдоподобие в логарифмах возрастает
до −5,737298. В последнем расчёте сравниваются два допустимых M-step из одного
исходного E-step: с фиксированными дисперсиями и с их свободной
переоценкой. При повторении алгоритма дисперсии обновляют согласно
выбранной модели: фиксированные параметры не включают в оптимизацию.

## Полный алгоритм EM для GMM

```text
Вход: X, число компонент K, covariance_type
1. Инициализировать μk (часто k-means), πk и Σk.
2. Повторять:
   E: вычислить log πk + log N(xn | μk, Σk);
      нормализовать через log-sum-exp и получить γnk.
   M: вычислить Nk, затем обновить πk, μk и Σk.
   Посчитать суммарный логарифм правдоподобия.
3. Остановиться по заданному допуску изменения цели.
4. Повторить с несколькими инициализациями; оставить лучшее решение по цели.
```

Вычисление плотности через произведения и экспоненты в высокой размерности
может дать численный ноль даже при положительной математической плотности.
Поэтому расчёт ведут в логарифмах и нормализуют через `logsumexp`.

Следующая реализация повторяет E/M-step для одномерной смеси. Она
вычисляет правдоподобие после каждого полного обновления, включая
последний шаг. Это учебный точный EM без добавки к дисперсии:
если компонента вырождается или исчезает численно, расчёт завершается
ошибкой, а не маскирует её произвольным округлением.

```python
def fit_em_1d(x, weights, means, variances, max_iter=100, tolerance=1e-10):
    weights, means, variances = [np.array(v, dtype=float, copy=True)
                                 for v in (weights, means, variances)]
    trace = [expectation_1d(x, weights, means, variances)[1].sum()]
    for _ in range(max_iter):
        gamma, _, _ = expectation_1d(x, weights, means, variances)
        counts = gamma.sum(axis=0)
        if np.any(counts <= 0):
            raise ValueError("Numerically empty component")
        weights = counts / len(x)
        means = gamma.T @ x / counts
        variances = (gamma * (x[:, None] - means) ** 2).sum(axis=0) / counts
        if np.any(variances <= 0):
            raise ValueError("Singular Gaussian component")
        current = expectation_1d(x, weights, means, variances)[1].sum()
        change = current - trace[-1]
        if change < -1e-9:
            raise AssertionError("Exact EM decreased the likelihood")
        trace.append(current)
        if abs(change) < tolerance:
            return weights, means, variances, np.array(trace), True
    return weights, means, variances, np.array(trace), False

fit_weights, fit_means, fit_variances, likelihood_trace, converged = fit_em_1d(
    x_train, weights_old, means_old, variances_old)
print("converged:", converged, "iterations:", len(likelihood_trace) - 1)
print("final:", fit_weights, fit_means, fit_variances)
np.testing.assert_allclose(likelihood_trace[-1],
    expectation_1d(x_train, fit_weights, fit_means, fit_variances)[1].sum())
```

Здесь четыре точки образуют две пары. После сходимости средние
близки к 0,5 и 4,5, дисперсии — к 0,25. Получение маленького изменения
цели означает остановку данного запуска. Оно не исключает других
решений или вырожденных последовательностей параметров.

<a id="почему-likelihood-не-уменьшается"></a>

## Почему точный EM не уменьшает правдоподобие

Введём произвольное распределение $q(z)$:

$$
\begin{aligned}
\log p(x\mid\theta)&=\mathcal L(q,\theta)\\
&\quad+KL\!\left(q(z)\,\|\,p(z\mid x,\theta)\right),
\end{aligned}
$$

где

$$
\begin{aligned}
\mathcal L(q,\theta)&=\mathbb E_q[\log p(x,z\mid\theta)]\\
&\quad-\mathbb E_q[\log q(z)].
\end{aligned}
$$

Равенство получается подстановкой
$\log p(x,z\mid\theta)=\log p(z\mid x,\theta)+\log p(x\mid\theta)$
в определение $\mathcal L$ и суммированием по $q$.
Поскольку KL неотрицательна, $\mathcal L$ является нижней границей
логарифма правдоподобия. На E-step выбирают
$q(z)=p(z\mid x,\theta^{old})$: KL обращается в ноль, и граница касается
текущего правдоподобия. На M-step повышают эту границу по $\theta$.
Для полной выборки границы и KL складываются по наблюдениям;
функция $Q$ из вывода M-step отличается от границы только энтропией
зафиксированного $q$. Следовательно,

$$
\begin{aligned}
\ell(\theta^{new})&\ge\mathcal L(q^{old},\theta^{new})\\
&\ge\mathcal L(q^{old},\theta^{old})\\
&=\ell(\theta^{old}).
\end{aligned}
$$

Первое неравенство следует из KL, второе — из улучшения $Q$ при
фиксированных вероятностях, равенство — из точного E-step.
Параметры на E-step не меняются: уточняется распределение скрытой
переменной. На M-step меняются параметры, но не старые вероятности.
В generalized EM достаточно увеличить границу, не обязательно
найти её максимум.

Следующий код проверяет цепочку на уже разобранном обновлении всех
параметров. После M-step старая граница обычно отстаёт от нового
правдоподобия: разность равна KL между старым и новым распределениями
принадлежности. Новый E-step устраняет именно этот зазор.

```python
def bound_value(responsibilities, log_joint):
    return np.sum(responsibilities * log_joint - xlogy(responsibilities, responsibilities))

bound_old = bound_value(gamma_old, logjoint_old)
bound_after_m = bound_value(gamma_old, logjoint_new)
bound_after_e = bound_value(gamma_new, logjoint_new)
gap = np.sum(xlogy(gamma_old, gamma_old) - xlogy(gamma_old, gamma_new))
np.testing.assert_allclose(bound_old, logp_old.sum())
np.testing.assert_allclose(bound_after_e, logp_full.sum())
np.testing.assert_allclose(logp_full.sum() - bound_after_m, gap)
assert bound_old <= bound_after_m <= bound_after_e
print("old / M bound / new likelihood:", bound_old, bound_after_m, bound_after_e)
print("KL gap:", gap)
```

В примере нижняя граница после M-step равна примерно −5,953440,
новое логарифмическое правдоподобие — −5,737298, а зазор KL — 0,216142.
Оба значения выше исходных −7,429516.

Гарантия относится к точному E-step и обновлению, повышающему ту же
цель. Добавление `reg_covar` после вычисления ковариации, приближённые
вероятности или численные отсечки требуют отдельной проверки;
они не наследуют это доказательство автоматически.

Вывод через нижнюю границу подробно дан в
[Stanford CS228, EM as variational inference](https://ermongroup.github.io/cs228-notes/learning/latent/#em-as-variational-inference).
Из монотонности не следует достижение глобального максимума. Разные
инициализации могут привести к разным локальным решениям.

<a id="геометрия-covariance"></a>

## Формы ковариаций

![[00 Учебник/Assets/Figures/curated/gmm-2026-10-25/jake-covariance-types.png]]

*Jake VanderPlas,
[Choosing the covariance type](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html#Choosing-the-covariance-type),
CC BY-NC-ND 3.0 Unported. Исходный рисунок без изменений.*

На трёх панелях одни и те же точки. Диагональная ковариация слева
допускает разную ширину по координатам, но не поворот. Сферическая
посередине задаёт одну ширину по всем направлениям. Полная справа
учитывает совместное изменение координат и поворачивает эллипс
вдоль облака. Это контуры плотности, не жёсткая граница, за которой
компонента перестаёт объяснять данные.

![[00 Учебник/Assets/Figures/curated/advanced-unsupervised-classical/gmm-covariances.png]]

*Решения GMM при разных ограничениях ковариации. Изображение без изменений
взято из официального примера scikit-learn
[`plot_gmm_covariances.py`](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html),
лицензия BSD-3-Clause. В разделе источников сохранён полный скрипт.*

В этом примере Iris четырёхмерный, но показаны только два признака.
Начальные средние рассчитаны по известным классам обучающей части,
чтобы согласовать номера компонент с названиями видов; это не пример
полностью неразмеченной инициализации. Точки обозначают исходные
наблюдения, крестики дополнительно отмечают тестовые. Подписи accuracy
нельзя переносить на обычное обучение смеси без сопоставления меток.

Ограничение ковариации задаёт разные модельные семейства:

| Тип | Геометрия и число параметров формы |
|---|---|
| `spherical` | у каждой компоненты своя дисперсия, направления равноправны; $K$ параметров |
| `diag` | оси эллипсоида совпадают с координатными осями; $Kd$ |
| `tied` | все компоненты имеют одну полную форму; $d(d+1)/2$ |
| `full` | каждая компонента имеет собственный повёрнутый эллипсоид; $Kd(d+1)/2$ |

При фиксированных вероятностях принадлежности `diag` оставляет
диагональные элементы взвешенной ковариации. Для `spherical` берут
среднюю дисперсию по $d$ направлениям. Для `tied` объединяют
матрицы рассеяния всех компонент и делят на общее число объектов:

$$
\begin{aligned}
\Sigma_k^{diag}&=\operatorname{diag}(S_k)/N_k,\\
\sigma_k^2&=\frac{\operatorname{tr}S_k}{dN_k},\\
\Sigma^{tied}&=\frac1N\sum_kS_k.
\end{aligned}
$$

В первой строке `diag` означает матрицу, в которой сохранена только
диагональ $S_k$. Для `spherical` ковариация равна $\sigma_k^2I$.
Большая гибкость позволяет точнее подогнать обучающие данные,
но требует оценить больше параметров; при малой выборке они могут
описывать случайные отклонения. В текущем `GaussianMixture` тип
по умолчанию — `full`, его следует выбирать осознанно.

![[00 Учебник/Assets/Figures/curated/gmm-2026-10-25/jake-stretched.png]]

*Jake VanderPlas,
[Generalizing E–M: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html),
CC BY-NC-ND 3.0 Unported. Исходный рисунок без изменений.*

Здесь четыре вытянутые компоненты имеют собственные полные ковариации.
В исходном коде полупрозрачные эллипсы соответствуют расстояниям
Махаланобиса 1, 2 и 3, а их непрозрачность зависит от веса компоненты.
Их не следует читать как одномерные интервалы с вероятностями
68%, 95% и 99,7%: в двух измерениях масса внутри такого контура другая.

<a id="вырождение-и-другие-failure-modes"></a>

## Вырождение и ограничения модели

<a id="схлопывание-covariance"></a>

### Схлопывание ковариации

У свободной смеси с несколькими компонентами правдоподобие может
быть не ограничено сверху. Одну компоненту помещают точно на объект
и сжимают её ковариацию, а остальные оставляют с положительной
плотностью на остальных наблюдениях. В одном измерении для компоненты
с весом $\pi>0$, средним $x_1$ и стандартным отклонением $\varepsilon$
плотность в $x_1$ содержит $\pi/(\sqrt{2\pi}\varepsilon)$.
Её логарифм растёт при $\varepsilon\to0$. При этом вклады других
наблюдений остаются конечными благодаря остальным компонентам.
Монотонное улучшение цели не предотвращает такую последовательность.

```python
from sklearn.mixture import GaussianMixture

collapse_data = np.array([0., 2., 4.])
collapse_scores = []
for epsilon in [1., 0.1, 0.01, 0.001]:
    _, lp, _ = expectation_1d(collapse_data, np.array([0.5, 0.5]),
        np.array([0., 3.]), np.array([epsilon**2, 1.]))
    collapse_scores.append(lp.sum())
assert np.all(np.diff(collapse_scores) > 0)
print("shrinking component:", collapse_scores)

# Одно гауссовское распределение: для точек -1 и 1 оптимальная дисперсия равна 1.
regularization_data = np.array([-1., 1.])
exact_score = expectation_1d(regularization_data,
    np.ones(1), np.zeros(1), np.ones(1))[1].sum()
regularized_score = expectation_1d(regularization_data,
    np.ones(1), np.zeros(1), np.array([2.]))[1].sum()
assert regularized_score < exact_score
print("ML / variance plus 1:", exact_score, regularized_score)
```

`reg_covar` добавляет положительное число к диагонали оценённой
ковариации и предотвращает нулевые собственные значения в этом
обновлении. Но это уже не свободный максимум той же функции $Q$.
В примере добавка 1 меняет оптимальную дисперсию 1 на 2 и уменьшает
обычное правдоподобие. Поэтому полезно различать численную защиту,
явное ограничение на ковариацию и максимизацию с априорным распределением:
для каждой из этих задач своя цель и своя гарантия.

### Локальные оптимумы и пустые компоненты

При строго положительных весах и невырожденных гауссовских компонентах точные
вероятности положительны, но в численном расчёте маленькие значения
могут исчезнуть. Почти нулевой $N_k$ делает оценки нестабильными.
В scikit-learn к суммам принадлежностей добавлена небольшая
машинная поправка; это не означает, что компоненту удалось надёжно
оценить. Несколько инициализаций и проверка найденных весов помогают
обнаружить плохой запуск, но не доказывают единственность решения.

### Масштаб и выбросы

Для полной ковариации смена единиц сама по себе не должна менять
апостериорные вероятности. При обратимом преобразовании $x'=Ax+b$
параметры переходят в $\mu'_k=A\mu_k+b$ и
$\Sigma'_k=A\Sigma_kA^\top$. Все плотности делятся на один
$|\det A|$, который сокращается в формуле Байеса.
Логарифм правдоподобия меняется на константу $-N\log|\det A|$.

Это свойство модельного семейства не делает любой численный запуск
инвариантным. Сферическое ограничение не сохраняется при разных
масштабах координат, диагональное — при произвольном повороте;
добавка `reg_covar * I` также зависит от единиц. Инициализация и
число обусловленности могут изменить найденное решение. Если
стандартизация нужна, её оценивают только на обучающей части.

Выбросы воздействуют и на средние, и на квадраты отклонений.
Небольшое число далёких точек может получить отдельную компоненту
или растянуть существующую. Хорошее значение тренировочной цели
не подтверждает пригодность гауссовских хвостов для конкретных данных.

<a id="неидентифицируемость-labels"></a>

### Произвольность номеров компонент

Перестановка номеров компонент не меняет плотность. Нельзя сравнивать
`component 0` двух запусков без явного сопоставления их параметров.

### Компонента не обязательно равна кластеру

Несколько гауссовских компонент могут аппроксимировать одну изогнутую
или тяжёлохвостую плотность. И наоборот, одна широкая компонента
может покрывать несколько групп. Поэтому число компонент и число
содержательных групп — разные характеристики.

| Две компоненты | Шестнадцать компонент |
| --- | --- |
| ![[00 Учебник/Assets/Figures/curated/gmm-2026-10-25/jake-moons-two.png]] | ![[00 Учебник/Assets/Figures/curated/gmm-2026-10-25/jake-moons-sixteen.png]] |

*Jake VanderPlas,
[GMM as Density Estimation](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html#GMM-as-Density-Estimation),
CC BY-NC-ND 3.0 Unported. Два исходных рисунка без изменений.*

Слева два больших эллипса пересекают дуги и дают разбиение,
не совпадающее с двумя «лунами». Справа небольшие компоненты
выстраиваются вдоль дуг и лучше описывают их форму. Шестнадцать
компонент здесь не означают шестнадцать исходных классов.
Для генерации из такой модели сначала выбирают компоненту с
вероятностью $\pi_k$, затем точку из её гауссовского распределения; выбор каждой
компоненты порождает только часть общей формы.

## Как выбирать число компонент

У более богатого модельного семейства наилучшее достижимое обучающее
правдоподобие не ниже, чем у вложенного. Численно обученные модели
не обязаны соблюдать этот порядок: запуски могут остановиться
в разных локальных решениях. А для свободной смеси сама верхняя
граница может быть бесконечной из-за схлопывания. Поэтому одно
тренировочное значение не выбирает сложность модели.

Один способ сравнения — средний логарифм плотности на валидационной
выборке. Параметры обучают на train, а новые наблюдения оценивают
без переобучения. Большая модель может повысить плотность в точках
train, но занизить её между ними, что обнаружится на новых данных.

Другой способ — информационные критерии. Для числа свободных
параметров $p$ они имеют вид

$$
\begin{aligned}
\mathrm{AIC}&=-2\ell(\hat\theta)+2p,\\
\mathrm{BIC}&=-2\ell(\hat\theta)+p\log N.
\end{aligned}
$$

Меньшее значение предпочтительнее. Здесь используется суммарное
обучающее правдоподобие; подстановка среднего без умножения на $N$
меняет критерий. При $N>e^2$ штраф BIC за один параметр больше,
чем штраф AIC. Это приближённые правила выбора, не доказательство
«истинного» числа групп. У смесей возможны вырождения и совпадения
компонент, нарушающие обычные предпосылки асимптотических выводов.

Для весов, средних и полной ковариации число параметров равно

$$
p=(K-1)+Kd+K\frac{d(d+1)}2.
$$

В $d=20$ добавление одной полной компоненты даёт 231 новый параметр:
один вес, 20 координат среднего и 210 элементов симметричной
ковариации. Число компонент само по себе не описывает сложность.

![[00 Учебник/Assets/Figures/curated/gmm-2026-10-25/jake-aic-bic.png]]

*Jake VanderPlas,
[How many components?](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html#How-many-components?),
CC BY-NC-ND 3.0 Unported. Исходный рисунок без изменений.*

На рисунке критерии посчитаны для тех же двух «лун» при разном числе
компонент. Их минимумы не обязаны соответствовать двум: несколько
гауссовских компонент нужно для описания каждой дуги. В этом запуске BIC
предпочитает более простую смесь, чем AIC. Это выбор модели плотности
в данном эксперименте, не универсальная рекомендация конкретного $K$.

Следующий код сравнивает четыре ограничения ковариации и от одной
до четырёх компонент на синтетической двумерной выборке. Валидационная
часть выбирает модель; тестовая используется один раз после выбора.
AIC и BIC считаются на обучающей части и показаны как отдельные
критерии, а не как взаимозаменяемые названия валидационной оценки.

```python
def parameter_count(K, d, covariance_type):
    shape_parameters = {"full": K*d*(d+1)//2, "tied": d*(d+1)//2,
                        "diag": K*d, "spherical": K}
    return K - 1 + K*d + shape_parameters[covariance_type]

selection_rng = np.random.default_rng(7)
cloud_a = selection_rng.normal(size=(400, 2)) @ np.array([[1.2, 0.7], [0., 0.3]]) + [-2, 0]
cloud_b = selection_rng.normal(size=(400, 2)) * 0.7 + [2, 1]
selection_data = np.vstack([cloud_a, cloud_b])
selection_rng.shuffle(selection_data)
X_fit, X_validation, X_test = np.split(selection_data, [480, 640])
candidates = []
for covariance_type in ["spherical", "diag", "tied", "full"]:
    for K in range(1, 5):
        model = GaussianMixture(K, covariance_type=covariance_type, n_init=3,
            max_iter=300, tol=1e-5, reg_covar=1e-6, random_state=7).fit(X_fit)
        if not model.converged_:
            raise RuntimeError("Candidate did not converge")
        total_logp = model.score_samples(X_fit).sum()
        p = parameter_count(K, 2, covariance_type)
        np.testing.assert_allclose(model.aic(X_fit), -2*total_logp + 2*p)
        np.testing.assert_allclose(model.bic(X_fit), -2*total_logp + p*np.log(len(X_fit)))
        candidates.append(model)

best_validation = max(candidates, key=lambda m: m.score(X_validation))
best_bic = min(candidates, key=lambda m: m.bic(X_fit))
test_logp = best_validation.score_samples(X_test)
test_posterior = best_validation.predict_proba(X_test)
np.testing.assert_allclose(test_posterior.sum(axis=1), 1)
np.testing.assert_allclose(best_validation.score(X_test), test_logp.mean())
generated_points, generated_components = best_validation.sample(12)
assert generated_points.shape == (12, 2)
print("validation choice:", best_validation.n_components, best_validation.covariance_type)
print("BIC choice:", best_bic.n_components, best_bic.covariance_type)
print("test mean log density:", test_logp.mean())
```

В данном запуске и валидация, и обучающий BIC выбрали две компоненты
с полной ковариацией. Средний тестовый логарифм плотности получился
приблизительно −2,503075. Здесь оценивается плотность синтетических
точек, а не точность классификации исходных меток.

`predict_proba` возвращает вероятности компонент, `predict` — индекс
максимальной из них, `score_samples` — логарифм плотности каждого
объекта, `score` — их среднее. Сгенерированная `sample` метка сообщает,
из какой компоненты была получена точка; это не внешняя классовая метка.
Флаг `converged_` относится к критерию остановки конкретной оптимизации,
а не к корректности модели или достижению глобального максимума.

При практическом выборе дополнительно сравнивают повторные запуски
и подвыборки. Если принадлежность превращается в решение для объекта,
оценка по внешней задаче нужна отдельно от качества плотности.

<a id="gmm-k-means-и-probabilistic-predictions"></a>

## Связь GMM с K-means

Пусть веса равны $1/K$, а все ковариации равны $\sigma^2I$.
Тогда формула E-step упрощается до

$$
\gamma_{nk}=\frac{\exp[-\|x_n-\mu_k\|^2/(2\sigma^2)]}
{\sum_j\exp[-\|x_n-\mu_j\|^2/(2\sigma^2)]}.
$$

При $\sigma^2\to0$ и единственном ближайшем среднем его вероятность
стремится к 1. M-step среднего превращается в среднее точек,
назначенных этому центру. При точном равенстве расстояний предел
сохраняет деление вероятности между ближайшими компонентами;
правило разрешения равенства в K-means — дополнительный выбор.

```python
limit_points = np.array([1., 2., 3.])
for variance in [1., 0.25, 0.01]:
    gamma_limit, _, _ = expectation_1d(limit_points, weights_old,
        means_old, np.full(2, variance))
    np.testing.assert_allclose(gamma_limit[1], [0.5, 0.5])
    print("variance:", variance, "first component:", gamma_limit[:, 0])
```

Для первой и последней точек вероятности приближаются к жёстким
назначениям; середина остаётся неоднозначной. Более общая GMM
с разными весами и ковариациями уже не сводится к ближайшему центру.

K-means делит пространство на ячейки Вороного и минимизирует сумму
квадратов расстояний до назначенных центров. Его результат для объекта —
номер центра; вероятностная модель в этот критерий не входит.
GMM максимизирует логарифм правдоподобия и возвращает вектор
вероятностей $\gamma_n$, рассчитанный по гауссовским плотностям.

## Практический протокол

1. Разделить данные и обучать преобразования признаков только на train.
2. Сравнить осмысленные `covariance_type`, а не автоматически выбрать `full`.
3. Для каждого числа компонент и типа ковариации выполнить несколько
   инициализаций.
4. Сохранить обучающее и валидационное правдоподобие, AIC/BIC, веса,
   минимальные собственные значения и числа обусловленности ковариаций.
5. Исследовать объекты с высокой энтропией принадлежности и отдельно
   объекты с низкой плотностью: эти множества не обязаны совпадать.
6. Проверить устойчивость компонент и согласовать их между запусками.
7. Если гауссовские компоненты плохо описывают данные, сравнить другие
   модели плотности вместо неограниченного увеличения числа компонент.

## Практика и первоисточники

- MML, Chapter 11, *Density Estimation with Gaussian Mixture Models*,
  [страница книги и PDF](https://mml-book.com/), §11.1–11.4.
- Stanford CS228,
  [Learning in latent variable models](https://ermongroup.github.io/cs228-notes/learning/latent/),
  вывод обновлений GMM и объяснение EM через нижнюю границу.
- Jake VanderPlas,
  [In Depth: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html),
  исходные рисунки — CC BY-NC-ND 3.0 Unported по `LICENSE-TEXT`
  закреплённого репозитория. Рисунки сохранены без изменений;
  устаревшие названия API из книги в код главы не перенесены.
- scikit-learn, [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html),
  BSD-3-Clause.
- Для дальнейшего чтения: Radford Neal, Geoffrey Hinton,
  [A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants](https://www.cs.toronto.edu/~hinton/absps/emk.pdf),
  1998, трактовка EM через свободную энергию.
- [[05 Источники/Courses/Advanced Unsupervised and Classical ML/sklearn-gmm-covariances.py.md|Полный официальный пример GMM covariance types]]

**Назад:** [[07 PCA, t-SNE и UMAP]]  
**Дальше:** [[09 Kernels, SVM и random Fourier features]]
