---
title: "Sem13 Pca Tsne"
type: external-resource
status: imported-source
source_kind: jupyter-notebook
source_commit: 4b21051531fb72dc9eef58632332ad971c92d006
language: ru
---

> [!note] Полный оригинальный материал HSE
> Источник: [`ml1-2026-spring/seminars/sem13-pca-tsne.ipynb`](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem13-pca-tsne.ipynb), commit `4b21051531fb72dc9eef58632332ad971c92d006`.
> В репозитории не найдено общей лицензии; материал перенесён без перевода
> по прямому разрешению владельца Bookvar для некоммерческого учебного архива.
> Ссылка на оригинал и provenance сохранены.


# Sem13 Pca Tsne

# Методы понижения размерности


Довольно часто бывает так, что признаков очень много. Хочется уменьшить их число так, чтобы задача по-прежнему хорошо решалась. В этой тетрадке мы с вами посмотрим на несколько различных способов понижать размерность.

> "To deal with hyper-planes in a 14 dimensional space, visualize a 3D space and say 'fourteen' very loudly. Everyone does it." — Geoffrey Hinton

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

# 1. Отбор признаков

Самый простой способ выделения признаков &mdash; их отбор. Есть много разных стратегий отбора признаков. 


- __Одномерные стратегии:__ считаем насколько сильно признаки связаны с таргетом с помощью разных метрик, оставляем только самые связанные
- __Жадные методы отбора признаков:__ надстройки над методами обучения моделей. Они перебирают различные подмножества признаков и выбирают то из них, которое дает наилучшее качество определённой модели машинного обучения. Данный процесс устроен следующим образом. Обучение модели считается черным ящиком, который на вход принимает информацию о том, какие из его признаков можно использовать при обучении модели, обучает модель, и дальше каким-то методом оценивается качество такой модели, например, по отложенной выборке или кросс-валидации. Таким образом, задача, которую необходимо решить, — это оптимизация функционала качества модели по подмножеству признаков. Признаки обычно перебираются по какому-то аллгоритму. Например, можно попробовать все комбинации (очень долго и неэффективно).
- __Отбор признаков на основе моделей.__

Отберем признаки на основе их корреляции с целевым признаком, и сравним результаты с исходными.

```python
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# сделаем тетрадку этичнее заменив load_boston()  на fetch_california_housing()
from sklearn.datasets import fetch_california_housing
ds = fetch_california_housing()

X, y = ds.data, ds.target
indexes = np.arange(len(y))
np.random.seed(52342)
np.random.shuffle(indexes)
X = X[indexes, :]
y = y[indexes]

features_ind = np.arange(X.shape[1])
corrs = np.abs([pearsonr(X[:, i], y)[0] for i in features_ind])
importances_sort = np.argsort(corrs)
fig = plt.figure(figsize=(16,8))
plt.barh(features_ind, corrs[importances_sort])
plt.xlabel('importance', fontsize=20)
X = X[:, importances_sort]
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-4-output-1.png|Notebook output]]

```text
<Figure size 1152x576 with 1 Axes>
```

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

features_counts = np.arange(X.shape[1])

def scores_by_features_count(reg):
    scores = []
    for features_part in features_counts:
        X_part = X[:,importances_sort[features_part:]]
        scores.append(cross_val_score(reg, X_part, y).mean())
    return scores

linreg_scores = scores_by_features_count(LinearRegression())
rf_scores = scores_by_features_count(RandomForestRegressor(n_estimators=100, max_depth=3))
```

```python
plt.figure(figsize=(16,8))

plt.plot(features_counts, linreg_scores, label='LinearRegression')
plt.plot(features_counts, rf_scores, label='RandomForest')
plt.legend(loc='best', fontsize=20)
plt.xlabel('#features deleted', fontsize=20)
plt.ylabel('$R^2$', fontsize=20)
plt.grid()
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-6-output-2.png|Notebook output]]

```text
<Figure size 1152x576 with 1 Axes>
```

В общем, если мы захотим немного сократить потребление ресурсов, пожертвовав частью качества,
видно, что это можно сделать.

# 2. Метод главных компонент (Principal Component Analysis, PCA)

Выделение новых признаков путем их отбора часто дает плохие результаты, и в некоторых ситуациях такой подход практически бесполезен. Например, если мы работаем с изображениями, у которых признаками являются яркости пикселей,
невозможно выбрать небольшой поднабор пикселей, который дает хорошую информацию о содержимом картинки. Поэтому признаки нужно как-то комбинировать.

__Метод главных компонент__ &mdash; один из самых интуитивно простых и часто используемых методов для снижения размерности данных и проекции их на ортогональное подпространство признаков. В рамках метода делается два важных упрощения задачи

1. игнорируется целевая переменная;
2. строится линейная комбинация признаков.

П. 1 на первый взгляд кажется довольно странным, но на практике обычно не является таким уж плохим. Это связано с тем, что часто данные устроены так, что имеют какую-то внутреннюю структуру в пространстве меньшей размерности, которая никак не связана с целевой переменной. Поэтому и оптимальные признаки можно строить не глядя на ответ.

П. 2 тоже сильно упрощает задачу, но далее мы научимся избавляться от него.

### Теория

Кратко вспомним, что делает этот метод (подробно см. в лекции).

Пусть $X$ &mdash; матрица объекты-признаки, с нулевым средним каждого признака, а $w$ &mdash; некоторый единичный вектор. Тогда $Xw$ задает величину проекций всех объектов на этот вектор. Далее ищется вектор, который дает наибольшую дисперсию полученных проекций (то есть наибольшую дисперсию вдоль этого направления):

$$
\max_{w: \|w\|=1} \| Xw \|^2 =  \max_{w: \|w\|=1} w^T X^T X w
$$

Подходящий вектор тогда равен собственному вектору матрицы $X^T X$ с наибольшим собственным значением. После этого все пространство проецируется на ортогональное дополнение к вектору $w$ и процесс повторяется.

## 2.1 PCA на плоскости

Для начала посмотрим на метод PCA на плоскости для того, чтобы лучше понять, как он устроен. Попробуем специально сделать один из признаков более значимым и проверим, что PCA это обнаружит. Сгенерируем выборку из двухмерного нормального распределения с нулевым математическим ожиданием.

```python
np.random.seed(314512)

data_synth_1 = np.random.multivariate_normal(
    mean=[0, 0], 
    cov=[[4, 0], 
         [0, 1]],
    size=1000
)
```

Теперь изобразим точки выборки на плоскости и применим к ним PCA для нахождения главных компонент. В результате работы PCA из sklearn в `dec.components_` будут лежать главные направления (нормированные), а в `dec.explained_variance_` &mdash; дисперсия, которую объясняет каждая компонента. Изобразим на нашем графике эти направления, умножив их на дисперсию для наглядного отображения их значимости.

```python
from sklearn.decomposition import PCA

def PCA_show(dataset):
    plt.scatter(*zip(*dataset), alpha=0.5)
    
    dec = PCA()
    dec.fit(dataset)
    ax = plt.gca()
    for comp_ind in range(dec.components_.shape[0]):
        component = dec.components_[comp_ind, :]
        var = dec.explained_variance_[comp_ind]
        start, end = dec.mean_, component * var
        ax.arrow(start[0], start[1], end[0], end[1],
                 head_width=0.2, head_length=0.4, fc='r', ec='r')
    
    ax.set_aspect('equal', adjustable='box')

plt.figure(figsize=(16, 8))
PCA_show(data_synth_1)
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-13-output-3.png|Notebook output]]

```text
<Figure size 1152x576 with 1 Axes>
```

Видим, что PCA все правильно нашел. Но это, конечно, можно было сделать и просто посчитав
дисперсию каждого признака. Повернем наши данные на некоторый фиксированный угол и проверим,
что для PCA это ничего не изменит.

```python
angle = np.pi / 6
rotate = np.array([
        [np.cos(angle), - np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
data_synth_2 = rotate.dot(data_synth_1.T).T

plt.figure(figsize=(16, 8))
PCA_show(data_synth_2)
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-15-output-4.png|Notebook output]]

```text
<Figure size 1152x576 with 1 Axes>
```

Ниже пара примеров, где PCA отработал не так хорошо (в том смысле, что направления задают не очень хорошие признаки).

**Упражнение:** объясните, почему так произошло.

```python
from sklearn.datasets import make_circles, make_moons, make_blobs

np.random.seed(54242)

data_synth_bad = [
    make_circles(n_samples=1000, factor=0.2, noise=0.1)[0]*2,
    make_moons(n_samples=1000, noise=0.1)[0]*2,
    make_blobs(n_samples=1000, n_features=2, centers=4)[0]/5,
    np.random.multivariate_normal(
        mean=[0, 1.5], 
        cov=[[3, 1], 
             [1, 1]],
        size=1000),
]

plt.figure(figsize=(16,8))
rows, cols = 2, 2
for i, data in enumerate(data_synth_bad):
    plt.subplot(rows, cols, i + 1)
    PCA_show(data)
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-17-output-5.png|Notebook output]]

```text
<Figure size 1152x576 with 4 Axes>
```

## 2.2 Лица людей

Рассмотрим датасет с фотографиями лиц людей и применим к его признакам PCA. Ниже изображены примеры лиц из базы, и последняя картинка &mdash; это "среднее лицо".

```python
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces(shuffle=True, random_state=432542)
faces_images = faces.data
faces_ids = faces.target
image_shape = (64, 64)
    
mean_face = faces_images.mean(axis=0)

plt.figure(figsize=(16, 8))
rows, cols = 2, 4
n_samples = rows * cols
for i in range(n_samples - 1):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(faces_images[i, :].reshape(image_shape), interpolation='none',
               cmap='gray')
    plt.xticks(())
    plt.yticks(())
    
plt.subplot(rows, cols, n_samples)
plt.imshow(mean_face.reshape(image_shape), interpolation='none',
           cmap='gray')
plt.xticks(())
_ = plt.yticks(())
```

```text
downloading Olivetti faces from https://ndownloader.figshare.com/files/5976027 to /root/scikit_learn_data
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-19-output-6.png|Notebook output]]

```text
<Figure size 1152x576 with 8 Axes>
```

Теперь найдем главные компоненты

```python
model_pca = PCA()
faces_images -= mean_face  # отнормировали данные к нулевому среднему
model_pca.fit(faces_images)

plt.figure(figsize=(16, 8))
rows, cols = 2, 4
n_samples = rows * cols
for i in range(n_samples):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(model_pca.components_[i, :].reshape(image_shape), interpolation='none', cmap='gray')
    plt.xticks(())
    plt.yticks(())
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-21-output-7.png|Notebook output]]

```text
<Figure size 1152x576 with 8 Axes>
```

Получилось жутковато, что уже неплохо, но есть ли от этого какая-то польза?

- Во-первых, новые признаки дают более высокое качество классификации.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

gscv_rf = GridSearchCV(
    RandomForestClassifier(),
    {'n_estimators': [100, 200, 500, 800], 'max_depth': [2, 3, 4, 5]}, cv=5
)
```

```python
%%time
gscv_rf.fit(faces_images, faces_ids)
print(gscv_rf.best_score_)
```

```text
0.825
CPU times: user 3min 59s, sys: 771 ms, total: 4min
Wall time: 3min 59s
```

```python
%%time
gscv_rf.fit(model_pca.transform(faces_images)[:,:100], faces_ids)
print(gscv_rf.best_score_)
```

```text
0.9125
CPU times: user 1min 31s, sys: 580 ms, total: 1min 32s
Wall time: 1min 32s
```

На практике можно выбирать столько главных компонент, чтобы оставить $90\%$ дисперсии исходных данных. В данном случае для этого достаточно выделить около $60$ главных компонент, то есть снизить размерность с $4096$ признаков до $60$.

```python
faces_images.shape
```

```text
(400, 4096)
```

```python
plt.figure(figsize=(10,7))
plt.plot(np.cumsum(model_pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(21, c='b')
plt.axhline(0.9, c='r')
plt.show();
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-28-output-8.png|Notebook output]]

```text
<Figure size 720x504 with 1 Axes>
```

 - Во-вторых, их можно использовать для компактного хранения данных. Для этого объекты трансформируются в новое пространство, и из него выкидываются самые незначимые признаки. Ниже приведены результаты сжатия в 20 раз.

```python
base_size = image_shape[0] * image_shape[1]

def compress_and_show(compress_ratio):
    model_pca = PCA(n_components=int(base_size * compress_ratio))
    model_pca.fit(faces_images)

    faces_compressed = model_pca.transform(faces_images)
    
    # обратное преобразование
    faces_restored = model_pca.inverse_transform(faces_compressed) + mean_face

    plt.figure(figsize=(16, 8))
    rows, cols = 2, 4
    n_samples = rows * cols
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(faces_restored[i, :].reshape(image_shape), interpolation='none',
                   cmap='gray')
        plt.xticks(())
        plt.yticks(())
        
compress_and_show(0.05)
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-30-output-9.png|Notebook output]]

```text
<Figure size 1152x576 with 8 Axes>
```

И даже при сжатии в 50 раз лица остаются узнаваемыми.

```python
compress_and_show(0.02)
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-32-output-10.png|Notebook output]]

```text
<Figure size 1152x576 with 8 Axes>
```

## 2.3 PCA для визуализации химического состава рек

Посмотрим на ещё один пример применения метода главных компонент. Будем работать с подмножеством [датасета о химическом составе воды](http://data.europa.eu/euodp/en/data/dataset/data_waterbase-rivers-10) в разных реках.

```python
!wget https://github.com/FUlyankin/Intro_to_DS/raw/master/data/water_dataset
```

```text
--2021-12-12 13:10:20--  https://github.com/FUlyankin/Intro_to_DS/raw/master/data/water_dataset
Resolving github.com (github.com)... 140.82.113.3
Connecting to github.com (github.com)|140.82.113.3|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://raw.githubusercontent.com/FUlyankin/Intro_to_DS/master/data/water_dataset [following]
--2021-12-12 13:10:20--  https://raw.githubusercontent.com/FUlyankin/Intro_to_DS/master/data/water_dataset
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.111.133, 185.199.108.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 2188569 (2.1M) [application/octet-stream]
Saving to: ‘water_dataset.1’

water_dataset.1     100%[===================>]   2.09M  --.-KB/s    in 0.06s   

2021-12-12 13:10:20 (33.6 MB/s) - ‘water_dataset.1’ saved [2188569/2188569]
```

```python
import pickle
index_list, feature_list, data_matrix = pickle.load(open('water_dataset', 'rb'))
```

* `index_list` - список id рек, которые были отобраны для задания
* `feature_list` - список признаков (они имеют вид `год ПРОБЕЛ показатель`)
* `data_matrix` - собственно данные (строки соответствуют рекам из `index_list`, а столбцы - признакам из `feature_list`)

Давайте попробуем сделать PCA, визуализровать данные в простанстве первых двух компонент и проинтерпретировать первые главные компоненты.

```python
X = data_matrix.copy()
X.shape
```

```text
(500, 532)
```

```python
from sklearn.decomposition import PCA

model_pca = PCA(10) # оставим 10 компонент
X_pca = model_pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1]);
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-40-output-11.png|Notebook output]]

```text
<Figure size 432x288 with 1 Axes>
```

Прямое применение PCA  к данным не дало хорошегго результата. 

- Вспомним, что PCA пытается выделять главные компоненты, максимизируя дисперсию. Дисперсия чувствительна к выбросам, значит и метод главных компонент чувствителен к ним. 
- Судя по всемму у данных разные единицы измерения, значит разброс несопоставим. PCA запутался.

```python
plt.hist(X.max(axis=0), bins=30,log=True);
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-42-output-12.png|Notebook output]]

```text
<Figure size 432x288 with 1 Axes>
```

Срежем выбросы по $99\%$ квантилю, а затем стандартизируем данные.

```python
replace = np.percentile(X, 99, axis=0)
for i in range(X.shape[1]):
    X[:,i][X[:,i] > replace[i]] = replace[i]
```

```python
plt.hist(X.max(axis=0), bins=30,log=True);
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-45-output-13.png|Notebook output]]

```text
<Figure size 432x288 with 1 Axes>
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scal = scaler.fit_transform(X)
```

```python
model_pca = PCA(10)
X_pca = model_pca.fit_transform(X_scal)

plt.scatter(X_pca[:,0], X_pca[:,1]);
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-47-output-14.png|Notebook output]]

```text
<Figure size 432x288 with 1 Axes>
```

Визуализация стала лучше. Чтобы понять физический смысл первых признаков, посмотрим из каких колонок они формируются.

```python
sorted(list(zip(feature_list, np.abs(model_pca.components_[0]))), key = lambda w : w[1], reverse=True)[:20]
```

```text
[('1992 Nitrate', 0.07205514167463652),
 ('1994 Nitrate', 0.07145770940394157),
 ('1993 Nitrate', 0.06953996244973613),
 ('1990 pH', 0.06899899935368568),
 ('1994 Nitrite', 0.06892630086390457),
 ('1989 pH', 0.06879277214220082),
 ('1995 Nitrate', 0.06821782583892055),
 ('1997 Nitrate', 0.0679660058189696),
 ('1997 Dissolved oxygen', 0.06794410190616795),
 ('1992 Nitrite', 0.06791502888673775),
 ('1991 pH', 0.0677594568278405),
 ('1998 Total phosphorus', 0.06761510041629039),
 ('1992 pH', 0.06752486404460212),
 ('1993 Nitrite', 0.06748580450611906),
 ('2002 Nitrate', 0.06726143675036038),
 ('1999 Nitrate', 0.06705702189135081),
 ('2002 Total phosphorus', 0.06705015770807102),
 ('1996 Dissolved oxygen', 0.0669196068568655),
 ('1995 Nitrite', 0.066705747608734),
 ('1998 Dissolved oxygen', 0.06654171398359399)]
```

```python
sorted(list(zip(feature_list, np.abs(model_pca.components_[1]))), key = lambda w : w[1], reverse=True)[:20]
```

```text
[('2006 Total organic carbon (TOC)', 0.10041321449682199),
 ('2005 Total organic carbon (TOC)', 0.09772568450458845),
 ('2005 CODMn', 0.09077344156953379),
 ('2004 CODMn', 0.08934944037390763),
 ('1999 Total nitrogen', 0.08912593532340404),
 ('1996 Total organic nitrogen', 0.08896511702814078),
 ('1998 Total nitrogen', 0.08890103684332638),
 ('1997 Total nitrogen', 0.08815045390348394),
 ('2000 Total nitrogen', 0.08793498941700152),
 ('1996 Total nitrogen', 0.08785326728275304),
 ('1986 Total organic nitrogen', 0.08772444881772631),
 ('2009 CODMn', 0.08768984047796037),
 ('1995 Total nitrogen', 0.08745538919212203),
 ('2002 CODMn', 0.08727596682768261),
 ('2001 CODMn', 0.08695775051660339),
 ('1985 Total organic nitrogen', 0.0868967039344715),
 ('2009 Total organic carbon (TOC)', 0.08656463369093668),
 ('1995 Total organic nitrogen', 0.08633959718245476),
 ('2006 CODMn', 0.08616992064355723),
 ('1994 Total nitrogen', 0.08569178674510657)]
```

```python
sorted(list(zip(feature_list, np.abs(model_pca.components_[3]))), key = lambda w : w[1], reverse=True)[:20]
```

```text
[('1989 Total inorganic nitrogen', 0.1352715314191778),
 ('1991 Total inorganic nitrogen', 0.1339142131067909),
 ('1990 Total inorganic nitrogen', 0.13350701356669084),
 ('1987 Total inorganic nitrogen', 0.1330826509257205),
 ('1988 Total inorganic nitrogen', 0.13242619223171223),
 ('1989 Total oxidised nitrogen', 0.13037885722151063),
 ('1987 Total oxidised nitrogen', 0.1295302061181991),
 ('1988 Total oxidised nitrogen', 0.12840159026903256),
 ('1990 Total oxidised nitrogen', 0.126800367178203),
 ('1991 Total oxidised nitrogen', 0.12438740321787424),
 ('1992 Total oxidised nitrogen', 0.12270180024633459),
 ('1993 Total oxidised nitrogen', 0.11842012469743415),
 ('1994 Total oxidised nitrogen', 0.11596298321755777),
 ('1995 Total oxidised nitrogen', 0.11552744969636228),
 ('1996 Total oxidised nitrogen', 0.11252133360049936),
 ('1998 Total oxidised nitrogen', 0.10438609481564975),
 ('1987 CODMn', 0.10161880654931965),
 ('1989 CODMn', 0.10107353314398644),
 ('1999 Total oxidised nitrogen', 0.10059930497384939),
 ('1987 Total ammonium', 0.10042294841587188)]
```

## 2.4 PCA с ядрами

Так как PCA фактически работает не исходными признаками, а с матрицей их ковариаций, можно использовать для ее вычисления вместо скалярного произведения $\langle x_i, x_j \rangle$ произвольное ядро $K(x_i, x_j)$. Это будет соответствовать переходу в другое пространство. Единственная проблема &mdash; непонятно, как подбирать ядро.

Ниже приведены примеры объектов в исходном пространстве (похожие группы обозначены одним цветом для наглядности), и результат их трансформации в новые пространства (для разных ядер). Если результаты получаются линейно разделимыми &mdash; значит мы выбрали подходящее ядро.

```python
from sklearn.decomposition import KernelPCA

def KPCA_show(X, y):
    reds = y == 0
    blues = y == 1
    
    plt.figure(figsize=(8, 8))
    rows, cols = 2, 2
    plt.subplot(rows, cols, 1)
    plt.scatter(X[reds, 0], X[reds, 1], alpha=0.5, c='r')
    plt.scatter(X[blues, 0], X[blues, 1], alpha=0.5, c='b')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    kernels_params = [
        dict(kernel='rbf', gamma=10),
        dict(kernel='poly', gamma=10),
        dict(kernel='cosine', gamma=10),
    ]
    
    for i, p in enumerate(kernels_params):
        dec = KernelPCA(**p)
        X_transformed = dec.fit_transform(X)
        
        plt.subplot(rows, cols, i + 2)
        plt.scatter(X_transformed[reds, 0], X_transformed[reds, 1], alpha=0.5, c='r')
        plt.scatter(X_transformed[blues, 0], X_transformed[blues, 1], alpha=0.5, c='b')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
np.random.seed(54242)
KPCA_show(*make_circles(n_samples=1000, factor=0.2, noise=0.1))
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-53-output-15.png|Notebook output]]

```text
<Figure size 576x576 with 4 Axes>
```

```python
np.random.seed(54242)
KPCA_show(*make_moons(n_samples=1000, noise=0.1))
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-54-output-16.png|Notebook output]]

```text
<Figure size 576x576 with 4 Axes>
```

# 3. TSNE (t-distributed Stohastic Neighbor Embedding)

Джефри Хинтон не просто сказал цитату из эпирафа к этой тетрадке, но и вместе со своим аспирантом, в 2008 году, придумал [новый методв изуализации данных.](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) Основная идея метода состоит в поиске отображения из многомерного признакового пространства на плоскость (или в 3D, но почти всегда выбирают 2D), чтоб точки, которые были далеко друг от друга, на плоскости тоже оказались удаленными, а близкие точки – также отобразились на близкие. То есть neighbor embedding – это своего рода поиск нового представления данных, при котором сохраняется соседство.

Попробуем взять данные о рукописных цифрах и визуализируем их с помощью PCA.

```python
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target

plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i,:].reshape([8,8]), cmap='gray');
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-56-output-17.png|Notebook output]]

```text
<Figure size 1152x432 with 10 Axes>
```

```python
X.shape
```

```text
(1797, 64)
```

Получается, размерность признакового пространства здесь – 64. Но давайте снизим размерность всего до 2 и увидим, что даже на глаз рукописные цифры неплохо разделяются на кластеры.

```python
model_pca = PCA(n_components=2)
X_reduced = model_pca.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. PCA projection');
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-59-output-18.png|Notebook output]]

```text
<Figure size 864x720 with 2 Axes>
```

Попробуем сделать то же самое с помощью t-SNE. Картинка получится лучше, так как у PCA есть существенное ограничение - он находит только линейные комбинации исходных признаков (если не добавить какое-нибудь ядро).  Внутри sklearn есть реализация TSNE, но она не такая эффективная как библиотека [MulticoreTSNE,](https://github.com/DmitryUlyanov/Multicore-TSNE) в которой метод можно распараллелить.

```python
!pip install MulticoreTSNE
```

```text
Collecting MulticoreTSNE
  Downloading MulticoreTSNE-0.1.tar.gz (20 kB)
Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from MulticoreTSNE) (1.19.5)
Requirement already satisfied: cffi in /usr/local/lib/python3.7/dist-packages (from MulticoreTSNE) (1.15.0)
Requirement already satisfied: pycparser in /usr/local/lib/python3.7/dist-packages (from cffi->MulticoreTSNE) (2.21)
Building wheels for collected packages: MulticoreTSNE
  Building wheel for MulticoreTSNE (setup.py) ... [?25l[?25hdone
  Created wheel for MulticoreTSNE: filename=MulticoreTSNE-0.1-cp37-cp37m-linux_x86_64.whl size=68518 sha256=20d416c528171f03d28dded462bee2037eb2120a76e8a34e83021940ebfb20d0
  Stored in directory: /root/.cache/pip/wheels/80/47/df/c0d66e9f775f33281c422a1964de86a59c47f93bb8c37643e3
Successfully built MulticoreTSNE
Installing collected packages: MulticoreTSNE
Successfully installed MulticoreTSNE-0.1
```

```python
from MulticoreTSNE import MulticoreTSNE as TSNE

tsne = TSNE(n_jobs=4, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.title('MNIST. t-SNE projection');
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-62-output-19.png|Notebook output]]

```text
<Figure size 864x720 with 2 Axes>
```

У метода есть параметр `Perplexity`, который отвечает за то, насколько сильно точки могут разлететься друг от друга.

```python
tsne = TSNE(n_jobs=4, perplexity=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.title('MNIST. t-SNE projection');
```

![[Assets/Sources/HSE ML course/ml1-2026-spring/seminars/sem13-pca-tsne/cell-64-output-20.png|Notebook output]]

```text
<Figure size 864x720 with 2 Axes>
```

Итоговая картинка может сильно поменяться при изменении `random_state`, это усложняет интерпретацию. В целом, по таким картинкам не стоит делать далеко идущих выводов – не стоит гадать по кофейной гуще.

В 2018 году был предложен ещё один алгоритм нелинейного снижения размерности, [UMAP.](https://umap-learn.readthedocs.io/en/latest/) Он похож на TSNE, но работает быстрее и более эффективен.

- [статья](https://habr.com/ru/company/io/blog/265089/) "Как подобрать платье с помощью метода главных компонент"
- [Q&A Разбор PCA с интуицией и примерами](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues)
- [Distillpub о TSNE](https://distill.pub/2016/misread-tsne/)
- [Подробнее про UMAP](https://pair-code.github.io/understanding-umap/)
