---
title: "Homework Practice Bonus Knn"
type: external-resource
status: imported-source
source_kind: jupyter-notebook
source_commit: 4b21051531fb72dc9eef58632332ad971c92d006
language: ru
---

> [!note] Полный оригинальный материал HSE
> Источник: [`ml2-2026-spring/homeworks-practice/homework-practice-bonus-knn.ipynb`](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/homeworks-practice/homework-practice-bonus-knn.ipynb), commit `4b21051531fb72dc9eef58632332ad971c92d006`.
> В репозитории не найдено общей лицензии; материал перенесён без перевода
> по прямому разрешению владельца Bookvar для некоммерческого учебного архива.
> Ссылка на оригинал и provenance сохранены.


# Homework Practice Bonus Knn

# Машинное обучение, ФКН ВШЭ

## Бонусное практическое задание. Поиск ближайших соседей

### Общая информация

Дата выдачи: 13.06.2026

Мягкий дедлайн: 23:59 MSK 19.06.2026

Жесткий дедлайн: 23:59 MSK 19.06.2026

### Оценивание и штрафы

Каждая из задач имеет определенную «стоимость» (указана в скобках около задачи).

За работу можно набрать **6** баллов (включая 0.5 за кулинарный бонус). 
Обратите внимание - оценка не нормируется к 10, ставится именно набранный вами абсолютный балл за ноутбук. Так как это бонус - этой домашки нет в знаменателе при подсчете итоговой оценки за ДЗ. 

Сдавать задание после указанного жёсткого срока сдачи нельзя.

Задание выполняется самостоятельно. «Похожие» решения считаются плагиатом и все задействованные студенты (в том числе те, у кого списали) не могут получить за него больше 0 баллов (подробнее о плагиате см. на странице курса). Если вы нашли решение какого-то из заданий (или его часть) в открытом источнике, необходимо указать ссылку на этот источник в отдельном блоке в конце вашей работы (скорее всего вы будете не единственным, кто это нашел, поэтому чтобы исключить подозрение в плагиате, необходима ссылка на источник).

Использование генеративных языковых моделей разрешено только при явном указании: какая модель, какие промпты и где использовались.

Неэффективная реализация кода может негативно отразиться на оценке.

### Формат сдачи

Задания сдаются через систему anytask. Посылка должна содержать:

* Ноутбук `homework-practice-bonus-knn-Username.ipynb`

Username — ваша фамилия и имя на латинице именно в таком порядке

# Вам нужно реализовать LSH и NSW


В этом задании мы будем работать с датасетом [FashionMnist](https://github.com/zalandoresearch/fashion-mnist) изображений предметов одежды и обуви. В файле уже находится массив со 100 найденными соседями каждого тестового объекта по евклидовой метрике, однако мы для начала (чтобы реализовать метод случайных проекций) попробуем найти 100 ближайших соседей для каждого объекта по косинусной близости.

```python
! wget -nc -q http: // ann-benchmarks.com / fashion-mnist-784-euclidean.hdf5
```

если не работает, можно так

```python
import os
import urllib.request

fname = 'fashion-mnist-784-euclidean.hdf5'
if not os.path.exists(fname):
    url = 'https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(fname, 'wb') as out:
        out.write(resp.read())
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
import h5py


with h5py.File('fashion-mnist-784-euclidean.hdf5', 'r') as f:
    train = f['train'][()]
    test = f['test'][()]
    true_neighbors_eucl = f['neighbors'][()]

print(train.shape)
print(test.shape)
print(true_neighbors_eucl.shape)
```

```python
plt.figure(figsize=(16, 4))

plt.subplot(141)
plt.imshow(train[0].reshape(28, 28))

plt.subplot(142)
plt.imshow(train[1].reshape(28, 28))

plt.subplot(143)
plt.imshow(train[3].reshape(28, 28))

plt.subplot(144)
plt.imshow(train[5].reshape(28, 28))

plt.show()
```

В библиотеке `scikit-learn` есть модуль с алгоритмами поиска ближайших соседей, помогающий решать задачи классификации и регрессии. В этом задании, однако, нас будет интересовать не классификация, а качество и скорость решения непосредственно задачи поиска. Определим для этого две метрики:

* **Recall**: доля «настоящих соседей», которых нашёл алгоритм поиска. Пример: если вам необходимо найти 100 ближайших соседей, и из 100 возвращённых алгоритмом объектов 93 действительно находятся в множестве ближайших, то полнота равна 0.93.


* **Queries per second (QPS), также requests per second (RPS)**: число запросов, на которое алгоритм успевает ответить за одну секунду. Часто эту характеристику вычисляют как $\dfrac{num\_queries}{total\_time}$, где $num\_queries$ — общее число запросов, а $total\_time$ — суммарное время их выполнения; однако, так как некоторые реализации в задании умеют группировать запросы и выполнять несколько параллельно, мы будем осуществлять их по отдельности и усреднять величину $\dfrac{1}{request\_time}$.

Хотя в классе `NearestNeighbors` нет методов приближённого поиска ближайших соседей, мы воспользуемся им, чтобы найти 100 ближайших объектов по косинусной близости, сформировав таким образом правильные ответы для приближённых методов. Алгоритмы `kd_tree` и `ball_tree` в этом классе не поддерживают `metric=cosine`, поэтому распараллелим полный перебор:

Понормируем

```python
train = train.astype(np.float32)
test = test.astype(np.float32)
train /= np.linalg.norm(train, axis=1, keepdims=True) + 1e-8
test /= np.linalg.norm(test, axis=1, keepdims=True) + 1e-8
```

```python
from sklearn.neighbors import NearestNeighbors


sklearn_index = NearestNeighbors(
    n_neighbors=100,
    algorithm='brute',
    metric='cosine',
    n_jobs=-1
)

sklearn_index.fit(train)

true_neighbors_cosine = sklearn_index.kneighbors(
    test,
    n_neighbors=100,
    return_distance=False
)
```

```python
from tqdm.auto import tqdm
from time import perf_counter


def compute_recall_qps(query_func: callable, test_set: np.ndarray, true_neighbors: np.ndarray, **kwargs):
    """
    Given a function that returns a list of nearest neighbors, estimate its recall and speed.
    Args:
        query_func: function with signature (query, k_neighbors, **kwargs).
            Returns a list of k_neighbors approximate nearest neighbors for a query
        test_set: array with shape (num_objects, dim). Contains query vectors for recall evaluation
        true_neighbors: array of indices with shape (num_objects, num_neighbors). Contains ground truth data for
            recall evaluation. k_neighbors from query_func is inferred from its shape
        **kwargs: passed to query_func
    Returns:
        avg_recall: average recall of query_func over the test set
        qps: number of queries per second handled by query_func
    """
    recalls = []
    query_times = []

    for sample, neighbors_for_sample in zip(test_set, tqdm(true_neighbors)):
        start = perf_counter()

        approx_neighbors = query_func(sample, k_neighbors=neighbors_for_sample.shape[0], **kwargs)

        query_times.append(1 / (perf_counter() - start))

        set_true = set(neighbors_for_sample.tolist())
        hits = sum(1 for neighbor in approx_neighbors if neighbor in set_true)

        recalls.append(hits / len(set_true))

    return np.mean(recalls), np.mean(query_times)
```

**Задание 1 (0.5 балла).** Евклид vs косинус

В HDF5 уже лежат 100 ближайших соседей каждого тестового объекта по евклидовой метрике (`true_neighbors_eucl`) — их посчитали авторы [ann-benchmarks](http://ann-benchmarks.com/) на исходных пикселях. Чуть выше мы нашли top-100 по косинусу (`true_neighbors_cosine`); дальше по ноутбуку правильным ответом будем считать именно его.

Сравните два списка. Для не меньше 500 случайных объектов из `test` посчитайте $|S_{\mathrm{eucl}} \cap S_{\mathrm{cos}}|$ — сколько из 100 косинусных соседей попало в евклидовый top-100 из файла. Постройте гистограмму этих пересечений (от 0 до 100) и усредните recall@100 (среднее пересечение, делённое на 100). Здесь ответ алгоритма — соседи из HDF5, эталон — `true_neighbors_cosine`. 

В 1–2 предложениях объясните, почему для проверки LSH нельзя брать `true_neighbors_eucl` за эталон.

```python
# YOUR CODE HERE
# подсказка: overlaps[i] = len(set(true_neighbors_eucl[i]) & set(true_neighbors_cosine[i]))
# recall_eucl_vs_cos = mean(overlaps) / 100
```

*Почему не берём соседей из HDF5 как эталон:*

YOUR ANSWER HERE

Подвыборка для скорости

```python
EVAL_SIZE = 2000
test_eval = test[:EVAL_SIZE]
true_neighbors_eval = true_neighbors_cosine[:EVAL_SIZE]
```

**Задание 2. (3 балла)**

Реализуйте все методы класса `LSHIndex`, а затем постройте индекс по датасету `FashionMNIST` и подсчитайте recall вашего решения.

Чтобы для удобства вместо косинусной близости считать скалярное произведение, и обучающую, и тестовую части датасета можно нормировать заранее.

```python
# LSH


def compute_hashes(matrices, query):
    """
    Compute hash values for each hash table and a given vector.
    Args:
        matrices: np.array of shape (NUM_TABLES, NUM_BITS, dim), last axis represents
            random hyperplanes for each hash bit of each table
        query: vector to be hashed
    Returns:
        hashes: np.array of shape (NUM_TABLES, ), contains hash values for each table as unsigned integers
    """

    # YOUR CODE HERE

    raise NotImplementedError


class LSHIndex:

    def __init__(self, vectors, projection_matrices):
        """
        Build the index and store vectors for efficient neighbors search.
        Args:
            vectors: Training data, np.array (num_vectors, dim). Each k-NN query looks for neighbors in this set
            projection_matrices: np.array of shape (NUM_TABLES, NUM_BITS, dim), last axis represents
                random hyperplanes for each hash bit of each table
        """

        # YOUR CODE HERE

        raise NotImplementedError

    def query(self, query, k_neighbors):
        """
        A helper function to perform a k-NN query.
        Args:
            query: a normalized query vector the neighbors of which we need to find
            k_neighbors: the number of neighbors to return
        Returns:
            neighbors: a list of indices in the dataset
                used to build the index. Vectors at these positions represent approximate k nearest neighbors of
                the query, indices do not need to be sorted by any property
        """
        hashes = compute_hashes(np.asarray(self.matrices), query)

        return self._search_neighbors(query, k_neighbors, hashes)

    def _search_neighbors(self, query, k_neighbors, hashes):
        """
        All the fun happens here. Given a sample, its hashes and the number  of neighbors, locate nearest neighbors
            wrt cosine similarity using the hash tables built during index construction.
        Args:
            query: a normalized query vector the neighbors of which we need to find
            k_neighbors: the number of neighbors to return
            hashes: an array containing hash values of query for each hash table
        Returns:
            neighbors: a list of indices in the dataset
                used to build the index. Vectors at these positions represent approximate k nearest neighbors of
                the query, indices do not need to be sorted by any property
        """

        # YOUR CODE HERE

        raise NotImplementedError
```

Зададим дефолтные параметры

```python
NUM_TABLES = 32
NUM_BITS = 12
rng = np.random.default_rng(0)
projection_matrices = rng.standard_normal((NUM_TABLES, NUM_BITS, train.shape[1]))
```

Функции для подсчета нужных метрик

```python
# YOUR CODE HERE

# recall, qps = ...

print(recall, qps)

assert recall >= 0.95
```

Визуализируйте пример работы алгоритма: найдите 5 ближайших соседей для нескольких объектов тестовой выборки и покажите, каким изображениям они соответствуют (вместе с самим запросом).

```python
# YOUR CODE HERE
```

**Задание 3 (0.5 балла).** Карта соседей в 2D

Выберите **один** объект из test. Обучите `PCA(n_components=2)` на `train`, спроецируйте:

* сам запрос;
* 20 ближайших соседей по **brute** (`sklearn_index`);
* 20 ближайших соседей по **вашему LSH**.

На одном scatter-графике отметьте три группы точек разными цветами/маркерами. Запрос выделите отдельно. Коротко прокомментируйте: где LSH промахивается относительно эталона судя по картинке, и промахивается ли? А если посмотреть не 20 соседей?

```python
# YOUR CODE HERE
```

**Задание 4. (1 балл)**

Исследуйте и покажите на графиках зависимость recall и QPS от размера хэша и числа хэш-таблиц в случае `LSH` или `num_construction_runs` и `num_search_runs` в случае NSW.

Сравните вашу реализацию с классом `sklearn.neighbors.NearestNeighbors`: так как для косинусной близости в этом классе поддерживается только `algorithm='brute'`, вы получите время работы точного поиска ближайших соседей. Для честности сравнения укажите в конструкторе метода `n_jobs` равным 1 и при замерах времени проводите поиск соседей для каждого объекта отдельно.

Отметьте соответствующую ему точку на графике recall-QPS, а затем с помощью перебора гиперпараметров постройте кривую в этих координатах для вашей реализации (нужно протестировать хотя бы 5 различных комбинаций). При построении можете вдохновляться графиками с сайта [ann-benchmarks](http://ann-benchmarks.com/):

![](http://ann-benchmarks.com/fashion-mnist-784-euclidean_100_euclidean.png)

```python
# YOUR CODE HERE
```

**Задание 5 (0.5 балла).** NSW

Реализуйте `NSWIndex` (каркас ниже). Постройте индекс, посчитайте recall@100 и QPS на `test_eval` (как для LSH).

Построение графа на всём `train` (60k) в чистом Python может занять **очень** долго. Можно построить индекс на подвыборке `train[:N]` (укажите `N` в комментарии); эталон при этом остаётся по полному `train`. Тогда recall может быть заметно ниже, чем у LSH — это нормально, главное рабочая реализация и график.

На том же графике recall–QPS (из задания 4) добавьте **хотя бы 3** точки NSW при разных `num_search_runs` (остальные гиперпараметры зафиксируйте и укажите в подписи).

```python
# NSW


class NSWIndex:

    def __init__(self, vectors, num_neighbors, num_construction_runs):
        """
        Build the index and store vectors for efficient neighbors search.
        Args:
            vectors: Training data, np.array (num_vectors, dim). Each k-NN query looks for neighbors in this set
            num_neighbors: how many neighbors to look for during sequential insertion
            num_construction_runs: number of search attempts in k-NN search during construction
        """
        # YOUR CODE HERE

        raise NotImplementedError

    def query(self, query, k_neighbors, num_search_runs):
        """
        A helper function to perform a k-NN query. Used to ensure that _knn_search can be called with GIL lifted.
        Args:
            query: a normalized query vector the neighbors of which we need to find
            k_neighbors: the number of neighbors to return
            num_search_runs: number of search attempts (parameter m in the NSW paper)
        Returns:
            neighbors: a list of indices in the dataset used to build the index. Vectors at these positions
            represent approximate k nearest neighbors of the query, indices do not need to be sorted
        """
        return self._knn_search(query, k_neighbors, num_search_runs)

    def _knn_search(self, query, k_neighbors, num_search_runs):
        """
        Run the NSW search algorithm in a constructed graph.
        Args:
            query: a normalized query vector the neighbors of which we need to find
            k_neighbors: the number of neighbors to return
            num_search_runs: number of search attempts (parameter m in the NSW paper)
        Returns:
            neighbors: a list of indices in the dataset used to build the index. Vectors at these positions
            represent approximate k nearest neighbors of the query, indices do not need to be sorted
        """
        # YOUR CODE HERE

        raise NotImplementedError
```

```python
# YOUR CODE HERE
```

**Кулинарное задание (+0.5 балла).**

Приложите свой любимый рецепт (необязательно завтрака) и расскажите, почему он вам так нравится. Лонгриды приветствуются. Рецепт без фото не принимается. "Хоть поешь нормально..."
