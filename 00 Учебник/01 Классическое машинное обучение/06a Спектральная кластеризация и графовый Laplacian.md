---
title: Спектральная кластеризация и графовый Laplacian
type: textbook-chapter
status: canonical
last_updated: 2026-07-26
primary_sources:
  - https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/seminars/sem03-graph-clustering.pdf
  - https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
  - https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
---

# Спектральная кластеризация и графовый Laplacian

K-means делит пространство границами Voronoi и поэтому предпочитает компактные
выпуклые группы. Две вложенные окружности или два переплетённых полумесяца
плохо разделяются одной евклидовой границей, хотя вдоль каждой фигуры соседство
очевидно. Спектральная кластеризация меняет представление задачи: объекты
становятся вершинами графа, сходство — весом ребра, а кластером считается
множество вершин, сильно связанных внутри и слабо связанных с остальным
графом.

Метод состоит из двух различных моделей:

1. **affinity graph** определяет, какие объекты считаются соседями;
2. собственные векторы graph Laplacian переводят вершины в новое пространство,
   где обычный k-means восстанавливает разбиение.

Если граф построен неудачно, последующая линейная алгебра не исправит исходное
понятие сходства.

## От объектов к взвешенному графу

Для $n$ объектов строится симметричная матрица
$W\in\mathbb R^{n\times n}$, где $w_{ij}\ge0$ — сходство объектов $i$ и $j$.
Наиболее распространены два способа.

**RBF affinity** соединяет все пары:

$$
w_{ij}=\exp(-\gamma\|x_i-x_j\|^2).
$$

Большое $\gamma$ оставляет заметными только очень близкие связи, малое делает
граф почти однородным. Полная матрица требует $O(n^2)$ памяти.

**k-nearest-neighbour graph** оставляет ребро, если один объект входит в число
$k$ ближайших соседей другого. Его симметризуют через union или mutual
neighbours. Граф получается sparse, но малое $k$ может раздробить его на
случайные компоненты, а большое — добавить короткие пути между настоящими
группами.

Степень вершины и degree matrix равны

$$
d_i=\sum_jw_{ij},
\qquad
D=\operatorname{diag}(d_1,\ldots,d_n).
$$

Ненормированный graph Laplacian:

$$
L=D-W.
$$

Для произвольного вектора $f\in\mathbb R^n$

$$
f^\top Lf
=\frac12\sum_{i,j}w_{ij}(f_i-f_j)^2.
$$

Формула объясняет весь метод. Энергия мала, когда сильно связанные вершины
получают близкие значения $f_i$. Матрица $L$ симметрична и positive
semidefinite; постоянный вектор $\mathbf1$ имеет eigenvalue 0. Число нулевых
eigenvalues равно числу connected components графа.

## Численный пример: слабый мост между двумя группами

Возьмём четыре вершины. Рёбра $(1,2)$ и $(3,4)$ имеют вес 1, а единственный
мост $(2,3)$ — вес 0,1:

$$
W=
\begin{bmatrix}
0&1&0&0\\
1&0&0.1&0\\
0&0.1&0&1\\
0&0&1&0
\end{bmatrix},
\qquad
D=\operatorname{diag}(1,1.1,1.1,1).
$$

Для вектора $f=(1,1,-1,-1)^\top$ внутренние рёбра дают нулевой вклад, а мост
даёт

$$
f^\top Lf=0.1(1-(-1))^2=0.4.
$$

Гладкий на каждой паре и меняющий знак только на слабом мосту вектор имеет
малую Laplacian energy. Именно такую структуру выражает eigenvector со вторым
наименьшим eigenvalue — **Fiedler vector**. Его знак уже даёт бинарное
разбиение; для нескольких кластеров используют несколько eigenvectors.

## От graph cut к eigenproblem

Пусть $A$ и $\bar A$ — две части графа, а

$$
\operatorname{cut}(A,\bar A)=
\sum_{i\in A,j\in\bar A}w_{ij}.
$$

Простая минимизация cut вырождается: выгодно отделить одну слабо связанную
вершину. RatioCut штрафует маленькие части:

$$
\operatorname{RatioCut}(A,\bar A)=
\operatorname{cut}(A,\bar A)
\left(\frac1{|A|}+\frac1{|\bar A|}\right).
$$

Введём дискретный вектор

$$
f_i=
\begin{cases}
\sqrt{|\bar A|/|A|},&i\in A,\\
-\sqrt{|A|/|\bar A|},&i\in\bar A.
\end{cases}
$$

Тогда $f^\top\mathbf1=0$, $\|f\|^2=n$ и

$$
f^\top Lf=n\,\operatorname{RatioCut}(A,\bar A).
$$

Поиск лучшего дискретного $f$ является комбинаторной задачей. Спектральная
релаксация разрешает $f$ принимать любые вещественные значения при тех же
ограничениях. По Rayleigh–Ritz минимум достигается на eigenvector $L$ со
вторым наименьшим eigenvalue: первый eigenvector — константа и запрещён
условием ортогональности.

Для $K$ кластеров индикаторы собираются в матрицу $H$ и минимизируется

$$
\operatorname{tr}(H^\top LH)
\quad\text{subject to}\quad H^\top H=I.
$$

После релаксации столбцами решения становятся $K$ eigenvectors с наименьшими
eigenvalues. Строка этой матрицы — спектральное представление одной вершины.

## Нормированные Laplacians

RatioCut нормирует по числу вершин. Если degrees сильно различаются, чаще
используют Normalized Cut и один из Laplacians:

$$
L_{\mathrm{rw}}=I-D^{-1}W,
\qquad
L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}.
$$

$L_{\mathrm{rw}}$ связан со случайным блужданием: переход из $i$ в $j$ имеет
вероятность $w_{ij}/d_i$. Кластер соответствует области, из которой
блуждание редко выходит. $L_{\mathrm{sym}}$ симметричен и удобен для
eigendecomposition. В варианте Ng–Jordan–Weiss строки выбранных eigenvectors
дополнительно нормируют перед k-means.

Нельзя механически смешивать eigenvectors разных Laplacians: алгоритм
нормировки и способ извлечения labels должны соответствовать выбранной
релаксации.

## Алгоритм

```text
Вход: объекты X, число кластеров K, правило affinity
1. Оценить preprocessing только на train.
2. Построить симметричную W: RBF либо sparse k-NN graph.
3. Проверить connected components и распределение degrees.
4. Вычислить D и выбранный Laplacian L, Lrw или Lsym.
5. Найти K eigenvectors с наименьшими eigenvalues.
6. Составить U из eigenvectors; при нужном варианте нормировать строки U.
7. Запустить k-means на строках U с несколькими initializations.
8. Вернуть labels и сохранить параметры graph construction.
```

Для sparse графа не нужна полная eigendecomposition: итерационные методы
вычисляют только нижнюю часть спектра. Но стоимость всё равно может стать
ограничением при большом $n$; тогда рассматривают Nyström approximation,
landmark graphs или методы кластеризации непосредственно на sparse graph.

![[00 Учебник/Assets/Figures/curated/advanced-unsupervised-classical/spectral-clustering-comparison.png|Сравнение алгоритмов кластеризации на данных разной геометрии]]

*Сравнение clustering algorithms на наборах разной геометрии. Spectral
Clustering разделяет окружности и полумесяцы благодаря графу соседства, но
результат и время зависят от affinity. Неизменённое изображение из официальной
галереи scikit-learn,
[`plot_cluster_comparison.py`](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html),
лицензия BSD-3-Clause.*

## Диагностика и failure modes

### Affinity важнее последнего шага

Нужно исследовать histogram distances, degree distribution и connected
components. Если несколько групп уже полностью disconnected, нулевые
eigenvalues просто воспроизведут это решение. Если граф почти полный и веса
почти одинаковы, информативного разреза может не быть.

### Eigengap — подсказка, не доказательство

Разрыв между $\lambda_K$ и $\lambda_{K+1}$ означает, что $K$-мерное нижнее
eigenspace отделено от следующего направления. Он помогает выбрать $K$, но
может отсутствовать в шумных данных и не заменяет stability или прикладную
оценку.

### Eigenvectors могут быть нестабильны

При близких eigenvalues отдельные eigenvectors способны вращаться и менять
знак. Сравнивать нужно порождённое ими subspace и итоговое разбиение, а не
координаты одного столбца.

### Out-of-sample transform не бесплатен

Классическая spectral clustering получает embedding только для train graph.
Новый объект нужно присоединить к графу и продолжить eigenfunctions либо
переобучить модель. Это отличает метод от обычного feature transformer.

### Scaling и leakage

Graph construction использует расстояния, значит наследует масштаб,
representation и возможную утечку. Если clustering становится feature для
supervised model, preprocessing, affinity tuning и выбор $K$ должны происходить
внутри validation protocol.

## Когда выбирать spectral clustering

| Ситуация | Предпочтение |
|---|---|
| Компактные примерно сферические группы, большой $n$ | k-means |
| Невыпуклые группы с надёжным локальным соседством | spectral clustering |
| Группы как плотные области и нужен noise label | DBSCAN/HDBSCAN |
| Граф задан предметной областью изначально | spectral или другие graph methods |

Спектральный метод особенно убедителен, когда рёбра имеют предметный смысл:
дружба, цитирование, совместная покупка, similarity между документами. Если
граф создан произвольным RBF по плохо подготовленным признакам, красивое
разбиение остаётся гипотезой.

## Исходные материалы и практика

- [[05 Источники/Courses/HSE ML course/ml2-2026-spring/seminars/sem03-graph-clustering.pdf.md|HSE ML: полный семинар «Graph Clustering»]] —
  четырёхстраничный вывод Laplacian identity, RatioCut relaxation и алгоритма;
  сохранён оригинальный PDF и commit provenance.
- [[05 Источники/Courses/HSE ML course/ml2-2026-spring/homeworks-practice/homework-practice-08-unsupervised.ipynb.md|HSE ML: оригинальное домашнее задание по unsupervised learning]] —
  реализация spectral clustering по adjacency matrix и сравнение на
  географических и текстовых данных.
- scikit-learn,
  [Spectral clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)
  и [сравнительная галерея](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html),
  BSD-3-Clause.

**Назад:** [[06 Кластеризация и её ограничения]]  
**Дальше:** [[07 PCA, t-SNE и UMAP]]
