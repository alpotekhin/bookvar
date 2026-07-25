---
title: "Apply Diversity Metrics"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: 6232b154548c955315650d58dca6bf1411c56020
---

> [!note] Original source material
> This page preserves [`examples/03_evaluate/als_movielens_diversity_metrics.ipynb`](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/03_evaluate/als_movielens_diversity_metrics.ipynb) from
> *Linux Foundation Recommenders* at commit `6232b154548c955315650d58dca6bf1411c56020`. License:
> [MIT](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



<i>Copyright (c) Recommenders contributors.</i>

<i>Licensed under the MIT License.</i>

## -- Compare ALS and Random Recommenders on MovieLens (PySpark)

In this notebook, we demonstrate how to evaluate a recommender using metrics other than commonly used rating/ranking metrics.

Such metrics include:
- Coverage - We use following two metrics defined by \[Shani and Gunawardana\]:

    - (1) catalog_coverage, which measures the proportion of items that get recommended from the item catalog;
    - (2) distributional_coverage, which measures how equally different items are recommended in the recommendations to all users.

- Novelty - A more novel item indicates it is less popular, i.e. it gets recommended less frequently.
We use the definition of novelty from \[Castells et al.\]

- Diversity - The dissimilarity of items being recommended.
We use a definition based on _intralist similarity_ by \[Zhang et al.]

- Serendipity - The "unusualness" or "surprise" of recommendations to a user.
We use a definition based on cosine similarity by \[Zhang et al.]

We evaluate the results obtained with two approaches: using the ALS recommender algorithm vs. a baseline of random recommendations.
 - Matrix factorization by [ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS) (Alternating Least Squares) is a well known collaborative filtering algorithm.
 - We also define a process which randomly recommends unseen items to each user.
 - We show two options to calculate item-item similarity: (1) based on item co-occurrence count; and (2) based on item feature vectors.

The comparision results show that the ALS recommender outperforms the random recommender on ranking metrics (Precision@k, Recall@k, NDCG@k, and	Mean average precision), while the random recommender outperforms ALS recommender on diversity metrics. This is because ALS is optimized for estimating the item rating as accurate as possible, therefore it performs well on accuracy metrics including rating and ranking metrics. As a side effect, the items being recommended tend to be popular items, which are the items mostly sold or viewed. It leaves the [long-tail items](https://github.com/microsoft/recommenders/blob/main/GLOSSARY.md) having less chance to get introduced to the users. This is the reason why ALS is not performing as well as a random recommender on diversity metrics.

From the algorithmic point of view, items in the tail suffer from the cold-start problem, making them hard for recommendation systems to use. However, from the business point of view, oftentimes the items in the tail can be highly profitable, since, depending on supply, business can apply a higher margin to them. Recommendation systems that optimize metrics like novelty and diversity, can help to find users willing to get these long tail items. Usually there is a trade-off between one type of metric vs. another. One should decide which set of metrics to optimize based on business scenarios.

**Coverage**

We define _catalog coverage_ as the proportion of items showing in all users’ recommendations:
$$
\textrm{CatalogCoverage} = \frac{|N_r|}{|N_t|}
$$
where $N_r$ denotes the set of items in the recommendations (`reco_df` in the code below) and $N_t$ the set of items in the historical data (`train_df`).

_Distributional coverage_ measures how equally different items are recommended to users when a particular recommender system is used.
If  $p(i|R)$ denotes the probability that item $i$ is observed among all recommendation lists, we define distributional coverage as
$$
\textrm{DistributionalCoverage} = -\sum_{i \in N_t} p(i|R) \log_2 p(i)
$$
where
$$
p(i|R) = \frac{|M_r (i)|}{|\textrm{reco_df}|}
$$
and $M_r (i)$ denotes the users who are recommended item $i$.

**Diversity**

Diversity represents the variety present in a list of recommendations.
_Intra-List Similarity_ aggregates the pairwise similarity of all items in a set. A recommendation list with groups of very similar items will score a high intra-list similarity. Lower intra-list similarity indicates higher diversity.
To measure similarity between any two items we use _cosine similarity_:
$$
\textrm{Cosine Similarity}(i,j)=  \frac{|M_t^{l(i,j)}|} {\sqrt{|M_t^{l(i)}|} \sqrt{|M_t^{l(j)}|} }
$$
where $M_t^{l(i)}$ denotes the set of users who liked item $i$ and $M_t^{l(i,j)}$ the users who liked both $i$ and $j$.
Intra-list similarity is then defined as
$$
\textrm{IL} = \frac{1}{|M|} \sum_{u \in M} \frac{1}{\binom{N_r(u)}{2}} \sum_{i,j \in N_r (u),\, i<j} \textrm{Cosine Similarity}(i,j)
$$
where $M$ is the set of users and $N_r(u)$ the set of recommendations for user $u$. Finally, diversity is defined as
$$
\textrm{diversity} = 1 - \textrm{IL}
$$

**Novelty**

The novelty of an item is inverse to its _popularity_. If $p(i)$ represents the probability that item $i$ is observed (or known, interacted with etc.) by users, then
$$
p(i) = \frac{|M_t (i)|} {|\textrm{train_df}|}
$$
where $M_t (i)$ is the set of users who have interacted with item $i$ in the historical data.

The novelty of an item is then defined as
$$
\textrm{novelty}(i) = -\log_2 p(i)
$$
and the novelty of the recommendations across all users is defined as
$$
\textrm{novelty} = \sum_{i \in N_r} \frac{|M_r (i)|}{|\textrm{reco_df}|} \textrm{novelty}(i)
$$

**Serendipity**

Serendipity represents the “unusualness” or “surprise” of recommendations. Unlike novelty, serendipity encompasses the semantic content of items and can be imagined as the distance between recommended items and their expected contents (Zhang et al.) Lower cosine similarity indicates lower expectedness and higher serendipity.
We define the expectedness of an unseen item $i$ for user $u$ as the average similarity between every already seen item $j$ in the historical data and $i$:
$$
\textrm{expectedness}(i|u) = \frac{1}{|N_t (u)|} \sum_{j \in N_t (u)} \textrm{Cosine Similarity}(i,j)
$$
The serendipity of item $i$ is (1 - expectedness) multiplied by _relevance_, where relevance indicates whether the item turns out to be liked by the user or not. For example, in a binary scenario, if an item in `reco_df` is liked (purchased, clicked) in `test_df`, its relevance equals one, otherwise it equals zero. Aggregating over all users and items, the overall
serendipity is defined as
$$
\textrm{serendipity} = \frac{1}{|M|} \sum_{u \in M_r}
\frac{1}{|N_r (u)|} \sum_{i \in N_r (u)} \big(1 - \textrm{expectedness}(i|u) \big) \, \textrm{relevance}(i)
$$

**Note**: This notebook requires a PySpark environment to run properly. Please follow the steps in [SETUP.md](https://github.com/Microsoft/Recommenders/blob/master/SETUP.md#dependencies-setup) to install the PySpark environment.

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import numpy as np
import pandas as pd

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, IntegerType, LongType, StructType, StructField
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF, CountVectorizer, VectorAssembler
from pyspark.ml.recommendation import ALS

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkDiversityEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

%load_ext autoreload
%autoreload 2

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))
```

```text
System version: 3.8.0 (default, Nov  6 2019, 21:49:08)
[GCC 7.3.0]
Spark version: 3.2.0
```

Set the default parameters.

```python
# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# user, item column names
COL_USER="UserId"
COL_ITEM="MovieId"
COL_RATING="Rating"
COL_TITLE="Title"
COL_GENRE="Genre"
```

### 1. Set up Spark context

The following settings work well for debugging locally on VM - change when running on a cluster. We set up a giant single executor with many threads and specify memory cap.

```python
# the following settings work well for debugging locally on VM - change when running on a cluster
# set up a giant single executor with many threads and specify memory cap

spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
spark.conf.set("spark.sql.crossJoin.enabled", "true")
```

### 2. Download the MovieLens dataset

```python
# Note: The DataFrame-based API for ALS currently only supports integers for user and item ids.
schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_ITEM, IntegerType()),
        StructField(COL_RATING, FloatType()),
        StructField("Timestamp", LongType()),
    )
)

data = movielens.load_spark_df(spark, size=MOVIELENS_DATA_SIZE, schema=schema, title_col=COL_TITLE, genres_col=COL_GENRE)
data.show()
```

```text
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.81k/4.81k [00:05<00:00, 862KB/s]
```

```text
+-------+------+------+---------+--------------------+------+
|MovieId|UserId|Rating|Timestamp|               Title| Genre|
+-------+------+------+---------+--------------------+------+
|     26|   138|   5.0|879024232|Brothers McMullen...|Comedy|
|     26|   224|   3.0|888104153|Brothers McMullen...|Comedy|
|     26|    18|   4.0|880129731|Brothers McMullen...|Comedy|
|     26|   222|   3.0|878183043|Brothers McMullen...|Comedy|
|     26|    43|   5.0|883954901|Brothers McMullen...|Comedy|
|     26|   201|   4.0|884111927|Brothers McMullen...|Comedy|
|     26|   299|   4.0|878192601|Brothers McMullen...|Comedy|
|     26|    95|   3.0|880571951|Brothers McMullen...|Comedy|
|     26|    89|   3.0|879459909|Brothers McMullen...|Comedy|
|     26|   361|   3.0|879440941|Brothers McMullen...|Comedy|
|     26|   194|   3.0|879522240|Brothers McMullen...|Comedy|
|     26|   391|   5.0|877399745|Brothers McMullen...|Comedy|
|     26|   345|   3.0|884993555|Brothers McMullen...|Comedy|
|     26|   303|   4.0|879468307|Brothers McMullen...|Comedy|
|     26|   401|   3.0|891033395|Brothers McMullen...|Comedy|
|     26|   429|   3.0|882386333|Brothers McMullen...|Comedy|
|     26|   293|   3.0|888907015|Brothers McMullen...|Comedy|
|     26|   270|   5.0|876954995|Brothers McMullen...|Comedy|
|     26|   442|   3.0|883388576|Brothers McMullen...|Comedy|
|     26|   342|   2.0|875320037|Brothers McMullen...|Comedy|
+-------+------+------+---------+--------------------+------+
only showing top 20 rows
```

#### Split the data using the Spark random splitter provided in utilities

```python
train_df, test_df = spark_random_split(data.select(COL_USER, COL_ITEM, COL_RATING), ratio=0.75, seed=123)
print ("N train_df", train_df.cache().count())
print ("N test_df", test_df.cache().count())
```

```text
N train_df 75147
```

```text
[Stage 19:================================================>     (178 + 3) / 200]
```

```text
N test_df 24853
```

#### Get all possible user-item pairs

Note: We assume that training data contains all users and all catalog items.

```python
users = train_df.select(COL_USER).distinct()
items = train_df.select(COL_ITEM).distinct()
user_item = users.crossJoin(items)
```

### 3. Train the ALS model on the training data, and get the top-k recommendations for our testing data

To predict movie ratings, we use the rating data in the training set as users' explicit feedback. The hyperparameters used in building the model are referenced from [here](http://mymedialite.net/examples/datasets.html). We do not constrain the latent factors (`nonnegative = False`) in order to allow for both positive and negative preferences towards movies.
Timing will vary depending on the machine being used to train.

```python
header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM,
    "ratingCol": COL_RATING,
}


als = ALS(
    rank=10,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=False,
    seed=42,
    **header
)
```

```python
with Timer() as train_time:
    model = als.fit(train_df)

print("Took {} seconds for training.".format(train_time.interval))
```

```text
Took 10.75137371000028 seconds for training.
```

In the movie recommendation use case, recommending movies that have been rated by the users does not make sense. Therefore, the rated movies are removed from the recommended items.

In order to achieve this, we recommend all movies to all users, and then remove the user-movie pairs that exist in the training dataset.

```python
# Score all user-item pairs
dfs_pred = model.transform(user_item)

# Remove seen items.
dfs_pred_exclude_train = dfs_pred.alias("pred").join(
    train_df.alias("train"),
    (dfs_pred[COL_USER] == train_df[COL_USER]) & (dfs_pred[COL_ITEM] == train_df[COL_ITEM]),
    how='outer'
)

top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull()) \
    .select('pred.' + COL_USER, 'pred.' + COL_ITEM, 'pred.' + "prediction")

print(top_all.count())

window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
top_k_reco = top_all.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")

print(top_k_reco.count())
```

```text
1464772
```

```text
[Stage 235:>                                                        (0 + 2) / 2]
```

```text
9430
```

### 4. Random Recommender

We define a recommender which randomly recommends unseen items to each user.

```python
# random recommender
window = Window.partitionBy(COL_USER).orderBy(F.rand())

# randomly generated recommendations for each user
pred_df = (
  train_df
  # join training data with all possible user-item pairs (seen in training)
  .join(user_item,
        on=[COL_USER, COL_ITEM],
        how="right"
  )
  # get user-item pairs that were not seen in the training data
  .filter(F.col(COL_RATING).isNull())
  # count items for each user (randomly sorting them)
  .withColumn("score", F.row_number().over(window))
  # get the top k items per user
  .filter(F.col("score") <= TOP_K)
  .drop(COL_RATING)
)
```

### 5. ALS vs Random Recommenders Performance Comparison

```python
def get_ranking_results(ranking_eval):
    metrics = {
        "Precision@k": ranking_eval.precision_at_k(),
        "Recall@k": ranking_eval.recall_at_k(),
        "NDCG@k": ranking_eval.ndcg_at_k(),
        "Mean average precision": ranking_eval.map_at_k()

    }
    return metrics

def get_diversity_results(diversity_eval):
    metrics = {
        "catalog_coverage":diversity_eval.catalog_coverage(),
        "distributional_coverage":diversity_eval.distributional_coverage(),
        "novelty": diversity_eval.novelty(),
        "diversity": diversity_eval.diversity(),
        "serendipity": diversity_eval.serendipity()
    }
    return metrics
```

```python
def generate_summary(data, algo, k, ranking_metrics, diversity_metrics):
    summary = {"Data": data, "Algo": algo, "K": k}

    if ranking_metrics is None:
        ranking_metrics = {
            "Precision@k": np.nan,
            "Recall@k": np.nan,
            "nDCG@k": np.nan,
            "MAP": np.nan,
        }
    summary.update(ranking_metrics)
    summary.update(diversity_metrics)
    return summary
```

#### ALS Recommender Performance Results

```python
als_ranking_eval = SparkRankingEvaluation(
    test_df,
    top_all,
    k = TOP_K,
    col_user=COL_USER,
    col_item=COL_ITEM,
    col_rating=COL_RATING,
    col_prediction="prediction",
    relevancy_method="top_k"
)

als_ranking_metrics = get_ranking_results(als_ranking_eval)
```

```python
als_diversity_eval = SparkDiversityEvaluation(
    train_df = train_df,
    reco_df = top_k_reco,
    col_user = COL_USER,
    col_item = COL_ITEM
)

als_diversity_metrics = get_diversity_results(als_diversity_eval)
```

```python
als_results = generate_summary(MOVIELENS_DATA_SIZE, "als", TOP_K, als_ranking_metrics, als_diversity_metrics)
```

#### Random Recommender Performance Results

```python
random_ranking_eval = SparkRankingEvaluation(
    test_df,
    pred_df,
    col_user=COL_USER,
    col_item=COL_ITEM,
    col_rating=COL_RATING,
    col_prediction="score",
    k=TOP_K,
)

random_ranking_metrics = get_ranking_results(random_ranking_eval)
```

```python
random_diversity_eval = SparkDiversityEvaluation(
    train_df = train_df,
    reco_df = pred_df,
    col_user = COL_USER,
    col_item = COL_ITEM
)

random_diversity_metrics = get_diversity_results(random_diversity_eval)
```

```python
random_results = generate_summary(MOVIELENS_DATA_SIZE, "random", TOP_K, random_ranking_metrics, random_diversity_metrics)
```

#### Result Comparison

```python
cols = ["Data", "Algo", "K", "Precision@k", "Recall@k", "NDCG@k", "Mean average precision","catalog_coverage", "distributional_coverage","novelty", "diversity", "serendipity" ]
df_results = pd.DataFrame(columns=cols)

df_results.loc[1] = als_results
df_results.loc[2] = random_results
```

```python
df_results
```

```text
   Data    Algo   K  Precision@k  Recall@k    NDCG@k  Mean average precision  \
1  100k     als  10     0.044374  0.015567  0.040657                0.004202
2  100k  random  10     0.018259  0.006516  0.018537                0.002038

   catalog_coverage  distributional_coverage    novelty  diversity  \
1          0.374158                 7.989889  11.740626   0.890659
2          0.998775                10.543160  12.180267   0.923302

   serendipity
1     0.879359
2     0.892897
```

#### Conclusion
The comparision results show that the ALS recommender outperforms the random recommender on ranking metrics (Precision@k, Recall@k, NDCG@k, and	Mean average precision), while the random recommender outperforms ALS recommender on diversity metrics. This is because ALS is optimized for estimating the item rating as accurate as possible, therefore it performs well on accuracy metrics including rating and ranking metrics. As a side effect, the items being recommended tend to be popular items, which are the items mostly sold or viewed. It leaves the long-tail less popular items having less chance to get introduced to the users. This is the reason why ALS is not performing as well as a random recommender on diversity metrics.

### 6.  Calculate diversity metrics using item feature vector based item-item similarity
In the above section we calculate diversity metrics using item co-occurrence count based item-item similarity. In the scenarios when item features are available, we may want to calculate item-item similarity based on item feature vectors. In this section, we show how to calculate diversity metrics using item feature vector based item-item similarity.

```python
# Get movie features "title" and "genres"
movies = (
    data.groupBy(COL_ITEM, COL_TITLE, COL_GENRE).count()
    .na.drop()  # remove rows with null values
    .withColumn(COL_GENRE, F.split(F.col(COL_GENRE), "\|"))  # convert to array of genres
    .withColumn(COL_TITLE, F.regexp_replace(F.col(COL_TITLE), "[\(),:^0-9]", ""))  # remove year from title
    .drop("count")  # remove unused columns
)
```

```python
# tokenize "title" column
title_tokenizer = Tokenizer(inputCol=COL_TITLE, outputCol="title_words")
tokenized_data = title_tokenizer.transform(movies)

# remove stop words
remover = StopWordsRemover(inputCol="title_words", outputCol="text")
clean_data = remover.transform(tokenized_data).drop(COL_TITLE, "title_words")
```

```python
# convert text input into feature vectors

# step 1: perform HashingTF on column "text"
text_hasher = HashingTF(inputCol="text", outputCol="text_features", numFeatures=1024)
hashed_data = text_hasher.transform(clean_data)

# step 2: fit a CountVectorizerModel from column "genres".
count_vectorizer = CountVectorizer(inputCol=COL_GENRE, outputCol="genres_features")
count_vectorizer_model = count_vectorizer.fit(hashed_data)
vectorized_data = count_vectorizer_model.transform(hashed_data)

# step 3: assemble features into a single vector
assembler = VectorAssembler(
    inputCols=["text_features", "genres_features"],
    outputCol="features",
)
feature_data = assembler.transform(vectorized_data).select(COL_ITEM, "features")

feature_data.show(10, False)
```

```text
[Stage 1441:============================================>       (172 + 2) / 200]
```

```text
+-------+------------------------------------------------------------------------------------+
|MovieId|features                                                                            |
+-------+------------------------------------------------------------------------------------+
|29     |(1043,[158,269,1025,1026,1029,1031],[1.0,1.0,1.0,1.0,1.0,1.0])                      |
|26     |(1043,[54,139,1025],[1.0,1.0,1.0])                                                  |
|1677   |(1043,[260,902,1024],[1.0,1.0,1.0])                                                 |
|964    |(1043,[416,429,1024,1025],[1.0,1.0,1.0,1.0])                                        |
|474    |(1043,[112,302,329,517,540,787,933,1032,1034],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
|1258   |(1043,[114,799,1025,1028],[1.0,1.0,1.0,1.0])                                        |
|541    |(1043,[635,910,1026,1029],[1.0,1.0,1.0,1.0])                                        |
|1224   |(1043,[978,1024],[1.0,1.0])                                                         |
|558    |(1043,[231,524,1024,1027,1041],[1.0,1.0,1.0,1.0,1.0])                               |
|191    |(1043,[206,1024,1035],[1.0,1.0,1.0])                                                |
+-------+------------------------------------------------------------------------------------+
only showing top 10 rows
```

The *features* column is represented with a SparseVector object. For example, in the feature vector (1043,[128,544,1025],[1.0,1.0,1.0]), 1043 is the vector length, indicating the vector consisting of 1043 item features. The values at index positions 128,544,1025 are 1.0, and the values at other positions are all 0.

```python
als_eval = SparkDiversityEvaluation(
    train_df = train_df,
    reco_df = top_k_reco,
    item_feature_df = feature_data,
    item_sim_measure="item_feature_vector",
    col_user = COL_USER,
    col_item = COL_ITEM
)

als_diversity=als_eval.diversity()
als_serendipity=als_eval.serendipity()
print(als_diversity)
print(als_serendipity)
```

```text
0.8742459916963194
0.8891175823541189
```

```python
random_eval = SparkDiversityEvaluation(
    train_df = train_df,
    reco_df = pred_df,
    item_feature_df = feature_data,
    item_sim_measure="item_feature_vector",
    col_user = COL_USER,
    col_item = COL_ITEM
)

random_diversity=random_eval.diversity()
random_serendipity=random_eval.serendipity()
print(random_diversity)
print(random_serendipity)
```

```text
0.896073781038039
0.8925253230847529
```

It's interesting that the value of diversity and serendipity changes when using different item-item similarity calculation approach, for both ALS algorithm and random recommender. The diversity and serendipity of random recommender are still higher than ALS algorithm.

### References
The metric definitions / formulations are based on the following references:
- P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems: choice, discovery and relevance, ECIR 2011
- G. Shani and A. Gunawardana, Evaluating recommendation systems, Recommender Systems Handbook pp. 257-297, 2010.
- E. Yan, Serendipity: Accuracy’s unpopular best friend in recommender Systems, eugeneyan.com, April 2020
- Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist: introducing serendipity into music recommendation, WSDM 2012

```python
# cleanup spark instance
spark.stop()
```
