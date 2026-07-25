---
title: "Data split"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: 6232b154548c955315650d58dca6bf1411c56020
---

> [!note] Original source material
> This page preserves [`examples/01_prepare_data/data_split.ipynb`](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/01_prepare_data/data_split.ipynb) from
> *Linux Foundation Recommenders* at commit `6232b154548c955315650d58dca6bf1411c56020`. License:
> [MIT](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



<i>Copyright (c) Recommenders contributors.</i>

<i>Licensed under the MIT License.</i>

Data splitting is one of the most vital tasks in assessing recommendation systems. Splitting strategy greatly affects the evaluation protocol so that it should always be taken into careful consideration by practitioners.

The code hereafter explains how one applies different splitting strategies for specific scenarios.

## 0 Global settings

```python
import sys
import pyspark
import pandas as pd
from datetime import datetime, timedelta

from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.datasets.download_utils import maybe_download
from recommenders.datasets.python_splitters import (
    python_random_split,
    python_chrono_split,
    python_stratified_split
)
from recommenders.datasets.spark_splitters import spark_random_split

print(f"System version: {sys.version}")
print(f"Pyspark version: {pyspark.__version__}")
```

```text
System version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0]
Pyspark version: 3.5.4
```

```python
DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
DATA_PATH = "ml-100k.data"

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_PREDICTION = "Rating"
COL_TIMESTAMP = "Timestamp"
```

## 1 Data preparation

### 1.1 Data understanding

For illustration purpose, the data used in the examples below is the MovieLens-100K dataset.

```python
filepath = maybe_download(DATA_URL, DATA_PATH)
```

```python
data = pd.read_csv(filepath, sep="\t", names=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP])
```

A glimpse at the data

```python
data.head()
```

```text
   UserId  MovieId  Rating  Timestamp
0     196      242       3  881250949
1     186      302       3  891717742
2      22      377       1  878887116
3     244       51       2  880606923
4     166      346       1  886397596
```

A little more...

```python
data.describe()
```

```text
             UserId        MovieId         Rating     Timestamp
count  100000.00000  100000.000000  100000.000000  1.000000e+05
mean      462.48475     425.530130       3.529860  8.835289e+08
std       266.61442     330.798356       1.125674  5.343856e+06
min         1.00000       1.000000       1.000000  8.747247e+08
25%       254.00000     175.000000       3.000000  8.794487e+08
50%       447.00000     322.000000       4.000000  8.828269e+08
75%       682.00000     631.000000       4.000000  8.882600e+08
max       943.00000    1682.000000       5.000000  8.932866e+08
```

And, more...

```python
print(
    "Total number of ratings are\t{}".format(data.shape[0]),
    "Total number of users are\t{}".format(data[COL_USER].nunique()),
    "Total number of items are\t{}".format(data[COL_ITEM].nunique()),
    sep="\n"
)
```

```text
Total number of ratings are	100000
Total number of users are	943
Total number of items are	1682
```

### 1.2 Data transformation

Original timestamps are converted to ISO format.

```python
data[COL_TIMESTAMP]= data.apply(
    lambda x: datetime.strftime(datetime(1970, 1, 1, 0, 0, 0) + timedelta(seconds=x[COL_TIMESTAMP].item()), "%Y-%m-%d %H:%M:%S"),
    axis=1
)
```

```python
data.head()
```

```text
   UserId  MovieId  Rating            Timestamp
0     196      242       3  1997-12-04 15:55:49
1     186      302       3  1998-04-04 19:22:22
2      22      377       1  1997-11-07 07:18:36
3     244       51       2  1997-11-27 05:02:03
4     166      346       1  1998-02-02 05:33:16
```

## 2 Experimentation protocol

Experimentation protocol is usually set up to favor a reasonable evaluation for a specific recommendation scenario. For example,
* *Recommender-A* is to recommend movies to people by taking people's collaborative rating similarities. To make sure the evaluation is statisically sound, the same set of users for both model building and testing should be used (to avoid any cold-ness of users), and a stratified splitting strategy should be taken.
* *Recommender-B* is to recommend fashion products to customers. It makes sense that evaluation of the recommender considers time-dependency of customer purchases, as apparently, tastes of the customers in fashion items may be drifting over time. In this case, a chronologically splitting should be used.

## 3 Data split

### 3.1 Random split

Random split simply takes in a data set and outputs the splits of the data, given the split ratios.

```python
data_train, data_test = python_random_split(data, ratio=0.7)
```

```python
data_train.shape[0], data_test.shape[0]
```

```text
(70000, 30000)
```

Sometimes a multi-split is needed.

```python
data_train, data_validate, data_test = python_random_split(data, ratio=[0.6, 0.2, 0.2])
```

```python
data_train.shape[0], data_validate.shape[0], data_test.shape[0]
```

```text
(60000, 20000, 20000)
```

Ratios can be integers as well.

```python
data_train, data_validate, data_test = python_random_split(data, ratio=[3, 1, 1])
```

For producing the same results.

```python
data_train.shape[0], data_validate.shape[0], data_test.shape[0]
```

```text
(60000, 20000, 20000)
```

### 3.2 Chronological split

Chronogically splitting method takes in a dataset and splits it on timestamp.

#### 3.2.1 "Filter by"

Chrono splitting can be either by "user" or "item". For example, if it is by "user" and the splitting ratio is 0.7, it means that first 70% ratings for each user in the data will be put into one split while the other 30% is in another. It is worth noting that a chronological split is not "random" because splitting is timestamp-dependent.

```python
data_train, data_test = python_chrono_split(
    data, ratio=0.7, filter_by="user",
    col_user=COL_USER, col_item=COL_ITEM, col_timestamp=COL_TIMESTAMP
)
```

Take a look at the results for one particular user:

* The last 10 rows of the train data:

```python
data_train[data_train[COL_USER] == 1].tail(10)
```

```text
       UserId  MovieId  Rating            Timestamp
1989        1       90       4  1997-11-03 07:31:40
11807       1      219       1  1997-11-03 07:32:07
50026       1      167       2  1997-11-03 07:33:03
202         1       61       4  1997-11-03 07:33:40
16314       1      230       4  1997-11-03 07:33:40
43280       1      162       4  1997-11-03 07:33:40
51295       1       35       1  1997-11-03 07:33:40
820         1      265       4  1997-11-03 07:34:01
11154       1      112       1  1997-11-03 07:34:01
45732       1       57       5  1997-11-03 07:34:19
```

* The first 10 rows of the test data:

```python
data_test[data_test[COL_USER] == 1].head(10)
```

```text
       UserId  MovieId  Rating            Timestamp
5682        1       49       3  1997-11-03 07:34:38
24493       1       30       3  1997-11-03 07:35:15
6234        1      233       2  1997-11-03 07:35:52
39865       1      131       1  1997-11-03 07:35:52
4280        1       82       5  1997-11-03 07:36:29
96699       1      152       5  1997-11-03 07:36:29
25721       1      141       3  1997-11-03 07:36:48
5842        1       72       4  1997-11-03 07:37:58
333         1       33       4  1997-11-03 07:38:19
37810       1      158       3  1997-11-03 07:38:19
```

Timestamps of train data are all precedent to those in test data.

#### 3.3.2 Min-rating filter

A min-rating filter is applied to data before it is split by using chronological splitter. The reason of doing this is that, for multi-split, there should be sufficient number of ratings for user/item in the data.

For example, the following means splitting only applies to users that have at least 10 ratings.

```python
data_train, data_test = python_chrono_split(
    data, filter_by="user", min_rating=10, ratio=0.7,
    col_user=COL_USER, col_item=COL_ITEM, col_timestamp=COL_TIMESTAMP
)
```

Number of rows in the yielded splits of data may not sum to the original ones as users with fewer than 10 ratings are filtered out in the splitting.

```python
data_train.shape[0] + data_test.shape[0], data.shape[0]
```

```text
(100000, 100000)
```

### 3.3 Stratified split

Chronogically splitting method takes in a dataset and splits it by either user or item. The split is stratified so that the same set of users or items will appear in both training and testing data sets.

Similar to chronological splitter, `filter_by` and `min_rating_filter` also apply to the stratified splitter.

The following example shows the split of the sample data with a ratio of 0.7, and for each user there should be at least 10 ratings.

```python
data_train, data_test = python_stratified_split(
    data, filter_by="user", min_rating=10, ratio=0.7,
    col_user=COL_USER, col_item=COL_ITEM
)
```

```python
data_train.shape[0] + data_test.shape[0], data.shape[0]
```

```text
(100000, 100000)
```

### 3.4 Data split in scale

Spark DataFrame is used for scalable splitting. This allows splitting operation performed on large dataset that is distributed across Spark cluster.

For example, the below illustrates how to do a random split on the given Spark DataFrame. For simplicity reason, the same MovieLens data, which is in Pandas DataFrame, is transformed into Spark DataFrame and used for splitting.

```python
spark = start_or_get_spark()
```

```text
your 131072x1 screen size is bogus. expect trouble
26/01/16 15:55:55 WARN Utils: Your hostname, unicorn resolves to a loopback address: 127.0.1.1; using 10.255.255.254 instead (on interface lo)
26/01/16 15:55:55 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
26/01/16 15:56:06 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
```

```python
data_spark = spark.read.csv(filepath)
```

```python
data_spark_train, data_spark_test = spark_random_split(data_spark, ratio=0.7)
```

Interestingly, it was noticed that Spark random split does not guarantee a deterministic result. This sometimes leads to issues when data is relatively small while users seek for a precision split.

```python
data_spark_train.count(), data_spark_test.count()
```

```text
(69941, 30059)
```

```python
spark.stop()
```

## References

1. Dimitris Paraschakis et al, "Comparative Evaluation of Top-N Recommenders in e-Commerce: An Industrial Perspective", IEEE ICMLA, 2015, Miami, FL, USA.
2. Guy Shani and Asela Gunawardana, "Evaluating Recommendation Systems", Recommender Systems Handbook, Springer, 2015.
3. Apache Spark, url: https://spark.apache.org/.
