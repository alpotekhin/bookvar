---
title: "Sem03 Numpy"
type: external-resource
status: imported-source
source_kind: jupyter-notebook
source_commit: 4b21051531fb72dc9eef58632332ad971c92d006
language: ru
---

> [!note] Полный оригинальный материал HSE
> Источник: [`ml1-2026-spring/seminars/sem03-numpy.ipynb`](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem03-numpy.ipynb), commit `4b21051531fb72dc9eef58632332ad971c92d006`.
> В репозитории не найдено общей лицензии; материал перенесён без перевода
> по прямому разрешению владельца Bookvar для некоммерческого учебного архива.
> Ссылка на оригинал и provenance сохранены.


# Sem03 Numpy

# Машинное обучение 1, ПМИ ФКН ВШЭ

## Семинар 3 (`numpy`)

## NumPy

**NumPy** — библиотека языка Python, позволяющая (удобно) работать с многомерными массивами и матрицами. Кроме того, NumPy позволяет векторизовать многие вычисления, имеющие место в машинном обучении.

 - [numpy](http://www.numpy.org)
 - [numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)
 - [100 numpy exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/)

Кстати, про NumPy недавно вышла [публикация](https://www.nature.com/articles/s41586-020-2649-2) в Nature.

```python
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```

Основным типом данных NumPy является многомерный массив элементов одного типа — [numpy.ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.array.html). Каждый подобный массив имеет несколько *измерений* или *осей* — в частности, вектор (в классическом понимании) является одномерным массивом и имеет 1 ось, матрица является двумерным массивом и имеет 2 оси и т.д.

```python
vec = np.array([1, 2, 3])
vec.ndim # количество осей
```

```text
1
```

```python
mat = np.array([[1, 2, 3], [4, 5, 6]])
mat.ndim
```

```text
2
```

Чтобы узнать длину массива по каждой из осей, можно воспользоваться атрибутом shape:

```python
vec.shape
```

```text
(3,)
```

Чтобы узнать тип элементов и их размер в байтах:

```python
mat.dtype.name
```

```text
'int64'
```

```python
mat.itemsize
```

```text
8
```

#### Создание массивов

Есть несколько способов сформировать массив в NumPy:

* Передать итерируемый объект в качестве параметра функции array (можно также явно указать тип элементов):

```python
A = np.array([1, 2, 3])
A, A.dtype
```

```text
(array([1, 2, 3]), dtype('int64'))
```

```python
A = np.array([1, 2, 3], dtype=float)
A, A.dtype
```

```text
(array([1., 2., 3.]), dtype('float64'))
```

* Воспользоваться функциями zeros, ones, empty, identity, если вам нужен объект специального вида:

```python
np.zeros((3,))
```

```text
array([0., 0., 0.])
```

```python
np.ones((3, 4))
```

```text
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
```

```python
np.identity(3)
```

```text
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

* Воспользоваться функциями arange (в качестве параметров принимает левую и правую границы последовательности и **шаг**) и linspace (принимает левую и правую границы и **количество элементов**) для формирования последовательностей:

```python
np.arange(2, 20, 3) # аналогично стандартной функции range python, правая граница не включается
```

```text
array([ 2,  5,  8, 11, 14, 17])
```

```python
np.arange(2.5, 8.7, 0.9) # но может работать и с вещественными числами
```

```text
array([2.5, 3.4, 4.3, 5.2, 6.1, 7. , 7.9])
```

```python
np.linspace(2, 18, 14) # правая граница включается (по умолчанию)
```

```text
array([ 2.        ,  3.23076923,  4.46153846,  5.69230769,  6.92307692,
        8.15384615,  9.38461538, 10.61538462, 11.84615385, 13.07692308,
       14.30769231, 15.53846154, 16.76923077, 18.        ])
```

* Изменить размеры существующего массива с помощью reshape (при этом количество элементов должно оставаться неизменным):

```python
np.arange(9).reshape(3, 3)
```

```text
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
```

Вместо значения длины массива по одному из измерений можно указать -1 — в этом случае значение будет рассчитано автоматически:

```python
np.arange(8).reshape(2, -1)
```

```text
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])
```

* Транспонировать существующий массив:

```python
C = np.arange(6).reshape(2, -1)
C
```

```text
array([[0, 1, 2],
       [3, 4, 5]])
```

```python
C.T
```

```text
array([[0, 3],
       [1, 4],
       [2, 5]])
```

* Повторить существующий массив:

```python
a = np.arange(3)
np.tile(a, (2, 2))
```

```text
array([[0, 1, 2, 0, 1, 2],
       [0, 1, 2, 0, 1, 2]])
```

```python
np.tile(a, (4, 1))
```

```text
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]])
```

#### Базовые операции

* Базовые арифметические операции над массивами выполняются поэлементно:

```python
A = np.arange(9).reshape(3, 3)
B = np.arange(1, 10).reshape(3, 3)
```

```python
print(A)
print(B)
```

```text
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

```python
A + B
```

```text
array([[ 1,  3,  5],
       [ 7,  9, 11],
       [13, 15, 17]])
```

```python
A * 1.0 / B
```

```text
array([[0.        , 0.5       , 0.66666667],
       [0.75      , 0.8       , 0.83333333],
       [0.85714286, 0.875     , 0.88888889]])
```

```python
A + 1
```

```text
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

```python
3 * A
```

```text
array([[ 0,  3,  6],
       [ 9, 12, 15],
       [18, 21, 24]])
```

```python
A ** 2
```

```text
array([[ 0,  1,  4],
       [ 9, 16, 25],
       [36, 49, 64]])
```

Отдельно обратим внимание на то, что умножение массивов также является **поэлементным**, а не матричным:

```python
A * B
```

```text
array([[ 0,  2,  6],
       [12, 20, 30],
       [42, 56, 72]])
```

Для выполнения матричного умножения необходимо использовать функцию dot:

```python
A.dot(B)
```

```text
array([[ 18,  21,  24],
       [ 54,  66,  78],
       [ 90, 111, 132]])
```

Для умножения векторов или матриц можно также использовать оператор `@`:

```python
A @ B
```

```text
array([[ 18,  21,  24],
       [ 54,  66,  78],
       [ 90, 111, 132]])
```

```python
np.array([1, 2, 3, 4]) @ np.array([1, 1, 1, 1])
```

```text
np.int64(10)
```

Поскольку операции выполняются поэлементно, операнды бинарных операций должны иметь одинаковый размер. Тем не менее, операция может быть корректно выполнена, если размеры операндов таковы, что они могут быть расширены до одинаковых размеров. Данная возможность называется [broadcasting](http://www.scipy-lectures.org/intro/numpy/operations.html#broadcasting):

![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/02.05-broadcasting.png)

```python
np.tile(np.arange(0, 40, 10), (3, 1)).T + np.array([0, 1, 2])
```

```text
array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22],
       [30, 31, 32]])
```

* Некоторые операции над массивами (например, вычисления минимума, максимума, суммы элементов) выполняются над всеми элементами вне зависимости от формы массива, однако при указании оси выполняются вдоль нее (например, для нахождения максимума каждой строки или каждого столбца):

```python
A
```

```text
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
```

```python
A.min()
```

```text
np.int64(0)
```

```python
A.max(axis=1)
```

```text
array([2, 5, 8])
```

```python
A.sum(axis=1)
```

```text
array([ 3, 12, 21])
```

#### Индексация

Для доступа к элементам может использоваться [много различных способов](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html), рассмотрим основные.

* Для индексации могут использоваться конкретные значения индексов и срезы (slice), как и в стандартных типах Python. Для многомерных массивов индексы для различных осей разделяются запятой. Если для многомерного массива указаны индексы не для всех измерений, недостающие заполняются полным срезом (:).

```python
a = np.arange(10)
a
```

```text
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

```python
a[2:5]
```

```text
array([2, 3, 4])
```

```python
a[3:8:2]
```

```text
array([3, 5, 7])
```

```python
A = np.arange(81).reshape(9, -1)
A
```

```text
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23, 24, 25, 26],
       [27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44],
       [45, 46, 47, 48, 49, 50, 51, 52, 53],
       [54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71],
       [72, 73, 74, 75, 76, 77, 78, 79, 80]])
```

```python
A[2:4]
```

```text
array([[18, 19, 20, 21, 22, 23, 24, 25, 26],
       [27, 28, 29, 30, 31, 32, 33, 34, 35]])
```

```python
A[:, 2:4]
```

```text
array([[ 2,  3],
       [11, 12],
       [20, 21],
       [29, 30],
       [38, 39],
       [47, 48],
       [56, 57],
       [65, 66],
       [74, 75]])
```

```python
A[2:4, 2:4]
```

```text
array([[20, 21],
       [29, 30]])
```

```python
A[-1]
```

```text
array([72, 73, 74, 75, 76, 77, 78, 79, 80])
```

* Также может использоваться индексация при помощи списков индексов (по каждой из осей):

```python
A = np.arange(81).reshape(9, -1)
A
```

```text
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23, 24, 25, 26],
       [27, 28, 29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42, 43, 44],
       [45, 46, 47, 48, 49, 50, 51, 52, 53],
       [54, 55, 56, 57, 58, 59, 60, 61, 62],
       [63, 64, 65, 66, 67, 68, 69, 70, 71],
       [72, 73, 74, 75, 76, 77, 78, 79, 80]])
```

```python
A[[2, 4, 5], [0, 1, 3]]
```

```text
array([18, 37, 48])
```

* Может применяться логическая индексация (при помощи логических массивов):

```python
A = np.arange(11)
A
```

```text
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

```python
A[A % 5 != 3]
```

```text
array([ 0,  1,  2,  4,  5,  6,  7,  9, 10])
```

```python
A[np.logical_and(A != 7, A % 5 != 3)] # также можно использовать логические операции
```

```text
array([ 0,  1,  2,  4,  5,  6,  9, 10])
```

#### Зачем?

Зачем необходимо использовать NumPy, если существуют стандартные списки/кортежи и циклы?

Причина заключается в скорости работы. Попробуем посчитать скалярное произведение 2 больших векторов:

```python
SIZE = 10000000

A_quick_arr = np.random.normal(size = (SIZE,))
B_quick_arr = np.random.normal(size = (SIZE,))

A_slow_list, B_slow_list = list(A_quick_arr), list(B_quick_arr)
```

```python
%%time
ans = 0
for i in range(len(A_slow_list)):
    ans += A_slow_list[i] * B_slow_list[i]
```

```text
CPU times: user 2.16 s, sys: 3.7 ms, total: 2.16 s
Wall time: 2.16 s
```

```python
%%time
ans = sum([A_slow_list[i] * B_slow_list[i] for i in range(SIZE)])
```

```text
CPU times: user 1.66 s, sys: 142 ms, total: 1.8 s
Wall time: 1.81 s
```

```python
%%time
ans = np.sum(A_quick_arr * B_quick_arr)
```

```text
CPU times: user 11.9 ms, sys: 10.9 ms, total: 22.8 ms
Wall time: 27.6 ms
```

```python
%%time
ans = A_quick_arr.dot(B_quick_arr)
```

```text
CPU times: user 12 ms, sys: 0 ns, total: 12 ms
Wall time: 9.84 ms
```

NumPy работает быстро по нескольким причинам:
* Массивы хранятся в непрерывном участке памяти, а все элементы имеют один и тот же тип
* Для вычислений по возможности используются библиотеки линейной алгебры вроде BLAS

Посмотреть, какая библиотека используется у вас, можно в конфигурации NumPy:

```python
print(np.show_config())
```

```text
Build Dependencies:
  blas:
    detection method: pkgconfig
    found: true
    include directory: /opt/_internal/cpython-3.12.2/lib/python3.12/site-packages/scipy_openblas64/include
    lib directory: /opt/_internal/cpython-3.12.2/lib/python3.12/site-packages/scipy_openblas64/lib
    name: scipy-openblas
    openblas configuration: OpenBLAS 0.3.27  USE64BITINT DYNAMIC_ARCH NO_AFFINITY
      Zen MAX_THREADS=64
    pc file directory: /project/.openblas
    version: 0.3.27
  lapack:
    detection method: pkgconfig
    found: true
    include directory: /opt/_internal/cpython-3.12.2/lib/python3.12/site-packages/scipy_openblas64/include
    lib directory: /opt/_internal/cpython-3.12.2/lib/python3.12/site-packages/scipy_openblas64/lib
    name: scipy-openblas
    openblas configuration: OpenBLAS 0.3.27  USE64BITINT DYNAMIC_ARCH NO_AFFINITY
      Zen MAX_THREADS=64
    pc file directory: /project/.openblas
    version: 0.3.27
Compilers:
  c:
    commands: cc
    linker: ld.bfd
    name: gcc
    version: 10.2.1
  c++:
    commands: c++
    linker: ld.bfd
    name: gcc
    version: 10.2.1
  cython:
    commands: cython
    linker: cython
    name: cython
    version: 3.0.11
Machine Information:
  build:
    cpu: x86_64
    endian: little
    family: x86_64
    system: linux
  host:
    cpu: x86_64
    endian: little
    family: x86_64
    system: linux
Python Information:
  path: /tmp/build-env-8744k94k/bin/python
  version: '3.12'
SIMD Extensions:
  baseline:
  - SSE
  - SSE2
  - SSE3
  found:
  - SSSE3
  - SSE41
  - POPCNT
  - SSE42
  - AVX
  - F16C
  - FMA3
  - AVX2
  not found:
  - AVX512F
  - AVX512CD
  - AVX512_KNL
  - AVX512_KNM
  - AVX512_SKX
  - AVX512_CLX
  - AVX512_CNL
  - AVX512_ICL

None
```

### Примеры векторизации вычислений на NumPy

Разберём несколько задач (из [100 numpy exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/)), где NumPy может существенно ускорить вычисления и упростить код.

Дан четырёхмерный массив. Как получить двумерный массив, в котором элемент с индексами $(i, j)$ содержит сумму всех элементов исходного массива, у которых первые два индекса — это $(i, j)$?

```python
A = np.random.randint(0,1000,(2,5,20,25))
res = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(res)
```

```text
[[254801 246060 251801 259209 250743]
 [244386 244280 243786 257685 239709]]
```

Даны одномерные массивы A и B. Элементы массива B принимают значения от 0 до `len(A) - 1`. Требуется прибавить единицу ко всем элементам A, чьи индексы записаны в B. Если индекс встречается в B несколько раз, то надо прибавить единицу для каждого такого вхождения.

```python
A = np.ones(10)
B = np.random.randint(0,len(A),20)
print(A)
print(B)
A += np.bincount(B, minlength=len(A))
print(A)
```

```text
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
[7 3 8 7 0 8 2 6 9 2 6 5 1 7 8 9 5 4 6 4]
[2. 2. 3. 2. 3. 3. 4. 4. 4. 3.]
```

Даны одномерный массив A и число n. Вычислите массив B, в котором i-й элемент равен среднему значению элементов с i-го по (i+n-1)-й в массиве A.

```python
def moving_average(Z, n=3) :
    ret = np.cumsum(Z, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
A = np.random.randint(0, 10, 20)
print(A)
print(moving_average(A, n=3))
```

```text
[8 9 6 7 3 5 9 4 1 2 7 6 2 9 3 6 0 8 5 5]
[7.66666667 7.33333333 5.33333333 5.         5.66666667 6.
 4.66666667 2.33333333 3.33333333 5.         5.         5.66666667
 4.66666667 6.         3.         4.66666667 4.33333333 6.        ]
```
