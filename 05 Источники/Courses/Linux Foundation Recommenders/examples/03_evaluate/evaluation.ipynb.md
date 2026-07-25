---
title: "Evaluation"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: 6232b154548c955315650d58dca6bf1411c56020
---

> [!note] Original source material
> This page preserves [`examples/03_evaluate/evaluation.ipynb`](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/03_evaluate/evaluation.ipynb) from
> *Linux Foundation Recommenders* at commit `6232b154548c955315650d58dca6bf1411c56020`. License:
> [MIT](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



<i>Copyright (c) Recommenders contributors.</i>

<i>Licensed under the MIT License.</i>

Evaluation with offline metrics is pivotal to assess the quality of a recommender before it goes into production. Usually, evaluation metrics are carefully chosen based on the actual application scenario of a recommendation system. It is hence important to data scientists and AI developers that build recommendation systems to understand how each evaluation metric is calculated and what it is for.

This notebook deep dives into several commonly used evaluation metrics, and illustrates how these metrics are used in practice. The metrics covered in this notebook are merely for off-line evaluations.

## 0 Global settings

Most of the functions used in the notebook can be found in the `recommenders` directory.

```python
import sys
import pandas as pd
import pyspark
import sklearn
from sklearn.preprocessing import minmax_scale

from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from recommenders.evaluation.python_evaluation import auc, logloss

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"PySpark version: {pyspark.__version__}")
print(f"Scikit Learn version: {sklearn.__version__}")
```

```text
System version: 3.9.16 (main, May 15 2023, 23:46:34)
[GCC 11.2.0]
Pandas version: 1.5.3
PySpark version: 3.2.4
Scikit Learn version: 1.0.2
```

Note to successfully run Spark codes with the Jupyter kernel, one needs to correctly set the environment variables of `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` that point to Python executables with the desired version. Detailed information can be found in the setup instruction document [SETUP.md](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/SETUP.md).

```python
COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_PREDICTION = "Rating"

HEADER = {
    "col_user": COL_USER,
    "col_item": COL_ITEM,
    "col_rating": COL_RATING,
    "col_prediction": COL_PREDICTION,
}
```

## 1 Prepare data

### 1.1 Prepare dummy data

For illustration purpose, a dummy data set is created for demonstrating how different evaluation metrics work.

The data has the schema that can be frequently found in a recommendation problem, that is, each row in the dataset is a (user, item, rating) tuple, where "rating" can be an ordinal rating score (e.g., discrete integers of 1, 2, 3, etc.) or an numerical float number that quantitatively indicates the preference of the user towards that item.

For simplicity reason, the column of rating in the dummy dataset we use in the example represent some ordinal ratings.

```python
df_true = pd.DataFrame(
        {
            COL_USER: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            COL_ITEM: [1, 2, 3, 1, 4, 5, 6, 7, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14],
            COL_RATING: [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1],
        }
    )
df_pred = pd.DataFrame(
    {
        COL_USER: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        COL_ITEM: [3, 10, 12, 10, 3, 5, 11, 13, 4, 10, 7, 13, 1, 3, 5, 2, 11, 14],
        COL_PREDICTION: [14, 13, 12, 14, 13, 12, 11, 10, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
    }
)
```

Take a look at ratings of the user with ID "1" in the dummy dataset.

```python
df_true[df_true[COL_USER] == 1]
```

```text
   UserId  MovieId  Rating
0       1        1       5
1       1        2       4
2       1        3       3
```

```python
df_pred[df_pred[COL_USER] == 1]
```

```text
   UserId  MovieId  Rating
0       1        3      14
1       1       10      13
2       1       12      12
```

### 1.2 Prepare Spark data

Spark framework is sometimes used to evaluate metrics given datasets that are hard to fit into memory. In our example, Spark DataFrames can be created from the Python dummy dataset.

```python
spark = start_or_get_spark("EvaluationTesting", "local")

dfs_true = spark.createDataFrame(df_true)
dfs_pred = spark.createDataFrame(df_pred)
```

```python
dfs_true.filter(dfs_true[COL_USER] == 1).show()
```

```text
[Stage 0:>                                                          (0 + 1) / 1]
```

```text
+------+-------+------+
|UserId|MovieId|Rating|
+------+-------+------+
|     1|      1|     5|
|     1|      2|     4|
|     1|      3|     3|
+------+-------+------+
```

```python
dfs_pred.filter(dfs_pred[COL_USER] == 1).show()
```

```text
+------+-------+------+
|UserId|MovieId|Rating|
+------+-------+------+
|     1|      3|    14|
|     1|     10|    13|
|     1|     12|    12|
+------+-------+------+
```

## 2 Evaluation metrics

### 2.1 Rating metrics

Rating metrics are similar to regression metrics used for evaluating a regression model that predicts numerical values given input observations. In the context of recommendation system, rating metrics are to evaluate how accurate a recommender is to predict ratings that users may give to items. Therefore, the metrics are **calculated exactly on the same group of (user, item) pairs that exist in both ground-truth dataset and prediction dataset** and **averaged by the total number of users**.

#### 2.1.1 Use cases

Rating metrics are effective in measuring the model accuracy. However, in some cases, the rating metrics are limited if
* **the recommender is to predict ranking instead of explicit rating**. For example, if the consumer of the recommender cares about the ranked recommended items, rating metrics do not apply directly. Usually a relevancy function such as top-k will be applied to generate the ranked list from predicted ratings in order to evaluate the recommender with other metrics.
* **the recommender is to generate recommendation scores that have different scales with the original ratings (e.g., the SAR algorithm)**. In this case, the difference between the generated scores and the original scores (or, ratings) is not valid for measuring accuracy of the model.

#### 2.1.2 How to work with the evaluation utilities

A few notes about the interface of the Rating evaluator class:
1. The columns of user, item, and rating (prediction) should be present in the ground-truth DataFrame (prediction DataFrame).
2. There should be no duplicates of (user, item) pairs in the ground-truth and the prediction DataFrames, othewise there may be unexpected behavior in calculating certain metrics.
3. Default column names for user, item, rating, and prediction are "UserId", "ItemId", "Rating", and "Prediciton", respectively.

In our examples below, to calculate rating metrics for input data frames in Spark, a Spark object, `SparkRatingEvaluation` is initialized. The input data schemas for the ground-truth dataset and the prediction dataset are

* Ground-truth dataset.

|Column|Data type|Description|
|-------------|------------|-------------|
|`COL_USER`|<int\>|User ID|
|`COL_ITEM`|<int\>|Item ID|
|`COL_RATING`|<float\>|Rating or numerical value of user preference.|

* Prediction dataset.

|Column|Data type|Description|
|-------------|------------|-------------|
|`COL_USER`|<int\>|User ID|
|`COL_ITEM`|<int\>|Item ID|
|`COL_RATING`|<float\>|Predicted rating or numerical value of user preference.|

```python
spark_rate_eval = SparkRatingEvaluation(dfs_true, dfs_pred, **HEADER)
```

```text
/home/u/anaconda/envs/recommenders/lib/python3.9/site-packages/pyspark/sql/context.py:125: FutureWarning: Deprecated in 3.0.0. Use SparkSession.builder.getOrCreate() instead.
  warnings.warn(
```

#### 2.1.3 Root Mean Square Error (RMSE)

RMSE is for evaluating the accuracy of prediction on ratings. RMSE is the most widely used metric to evaluate a recommendation algorithm that predicts missing ratings. The benefit is that RMSE is easy to explain and calculate.

```python
print(f"The RMSE is {spark_rate_eval.rmse()}")
```

```text
The RMSE is 7.254309064273455
```

#### 2.1.4 R Squared (R2)

R2 is also called "coefficient of determination" in some context. It is a metric that evaluates how well a regression model performs, based on the proportion of total variations of the observed results.

```python
print(f"The R2 is {spark_rate_eval.rsquared()}")
```

```text
The R2 is -31.699029126213595
```

#### 2.1.5 Mean Absolute Error (MAE)

MAE evaluates accuracy of prediction. It computes the metric value from ground truths and prediction in the same scale. Compared to RMSE, MAE is more explainable.

```python
print(f"The MAE is {spark_rate_eval.mae()}")
```

```text
The MAE is 6.375
```

#### 2.1.6 Explained Variance

Explained variance is usually used to measure how well a model performs with regard to the impact from the variation of the dataset.

```python
print(f"The explained variance is {spark_rate_eval.exp_var()}")
```

```text
The explained variance is -6.4466019417475735
```

#### 2.1.7 Summary

|Metric|Range|Selection criteria|Limitation|Reference|
|------|-------------------------------|---------|----------|---------|
|RMSE|$> 0$|The smaller the better.|May be biased, and less explainable than MAE|[link](https://en.wikipedia.org/wiki/Root-mean-square_deviation)|
|R2|$\leq 1$|The closer to $1$ the better.|Depend on variable distributions.|[link](https://en.wikipedia.org/wiki/Coefficient_of_determination)|
|MAE|$\geq 0$|The smaller the better.|Dependent on variable scale.|[link](https://en.wikipedia.org/wiki/Mean_absolute_error)|
|Explained variance|$\leq 1$|The closer to $1$ the better.|Depend on variable distributions.|[link](https://en.wikipedia.org/wiki/Explained_variation)|

### 2.2 Ranking metrics

"Beyond-accuray evaluation" was proposed to evaluate how relevant recommendations are for users. In this case, a recommendation system is a treated as a ranking system. Given relency definition, recommendation system outputs a list of recommended items to each user, which is ordered by relevance. The evaluation part takes ground-truth data, the actual items that users interact with (e.g., liked, purchased, etc.), and the recommendation data, as inputs, to calculate ranking evaluation metrics.

#### 2.2.1 Use cases

Ranking metrics are often used when hit and/or ranking of the items are considered:
* **Hit** - defined by relevancy, a hit usually means whether the recommended "k" items hit the "relevant" items by the user. For example, a user may have clicked, viewed, or purchased an item for many times, and a hit in the recommended items indicate that the recommender performs well. Metrics like "precision", "recall", etc. measure the performance of such hitting accuracy.
* **Ranking** - ranking metrics give more explanations about, for the hitted items, whether they are ranked in a way that is preferred by the users whom the items will be recommended to. Metrics like "mean average precision", "ndcg", etc., evaluate whether the relevant items are ranked higher than the less-relevant or irrelevant items.

#### 2.2.2 How-to with evaluation utilities

A few notes about the interface of the Rating evaluator class:
1. The columns of user, item, and rating (prediction) should be present in the ground-truth DataFrame (prediction DataFrame). The column of timestamp is optional, but it is required if certain relevant function is used. For example, timestamps will be used if the most recent items are defined as the relevant one.
2. There should be no duplicates of (user, item) pairs in the ground-truth and the prediction DataFrames, othewise there may be unexpected behavior in calculating certain metrics.
3. Default column names for user, item, rating, and prediction are "UserId", "ItemId", "Rating", and "Prediciton", respectively.

#### 2.2.3 Relevancy of recommendation

Relevancy of recommendation can be measured in different ways:

* **By ranking** - In this case, relevant items in the recommendations are defined as the top ranked items, i.e., top k items, which are taken from the list of the recommended items that is ordered by the predicted ratings (or other numerical scores that indicate preference of a user to an item).

* **By timestamp** - Relevant items are defined as the most recently viewed k items, which are obtained from the recommended items ranked by timestamps.

* **By rating** - Relevant items are defined as items with ratings (or other numerical scores that indicate preference of a user to an item) that are above a given threshold.

Similarly, a ranking metric object can be initialized as below. The input data schema is

* Ground-truth dataset.

|Column|Data type|Description|
|-------------|------------|-------------|
|`COL_USER`|<int\>|User ID|
|`COL_ITEM`|<int\>|Item ID|
|`COL_RATING`|<float\>|Rating or numerical value of user preference.|
|`COL_TIMESTAMP`|<string\>|Timestamps.|

* Prediction dataset.

|Column|Data type|Description|
|-------------|------------|-------------|
|`COL_USER`|<int\>|User ID|
|`COL_ITEM`|<int\>|Item ID|
|`COL_RATING`|<float\>|Predicted rating or numerical value of user preference.|
|`COL_TIMESTAM`|<string\>|Timestamps.|

In this case, in addition to the input datasets, there are also other arguments used for calculating the ranking metrics:

|Argument|Data type|Description|
|------------|------------|--------------|
|`k`|<int\>|Number of items recommended to user.|
|`revelancy_method`|<string\>|Methonds that extract relevant items from the recommendation list|

For example, the following code initializes a ranking metric object that calculates the metrics.

```python
spark_rank_eval = SparkRankingEvaluation(dfs_true, dfs_pred, k=3, relevancy_method="top_k", **HEADER)
```

A few ranking metrics can then be calculated.

#### 2.2.4 Precision

Precision@k is a metric that evaluates how many items in the recommendation list are relevant (hit) in the ground-truth data. For each user the precision score is normalized by `k` and then the overall precision scores are averaged by the total number of users.

Note it is apparent that the precision@k metric grows with the number of `k`.

```python
print(f"The precision at k is {spark_rank_eval.precision_at_k()}")
```

```text
The precision at k is 0.3333333333333333
```

#### 2.2.5 Recall

Recall@k is a metric that evaluates how many relevant items in the ground-truth data are in the recommendation list. For each user the recall score is normalized by the total number of ground-truth items and then the overall recall scores are averaged by the total number of users.

```python
print(f"The recall at k is {spark_rank_eval.recall_at_k()}")
```

```text
The recall at k is 0.2111111111111111
```

#### 2.2.6 Normalized Discounted Cumulative Gain (NDCG)

NDCG is a metric that evaluates how well the recommender performs in recommending ranked items to users. Therefore both hit of relevant items and correctness in ranking of these items matter to the NDCG evaluation. The total NDCG score is normalized by the total number of users.

```python
print(f"The NDCG at k is {spark_rank_eval.ndcg_at_k()}")
```

```text
The NDCG at k is 0.3333333333333333
```

#### 2.2.7 Mean Average Precision (MAP)

MAP is a metric that evaluates the average precision for each user in the datasets. It also penalizes ranking correctness of the recommended items. The overall MAP score is normalized by the total number of users.

```python
print(f"The MAP at k is {spark_rank_eval.map_at_k()}")
```

```text
The MAP at k is 0.15
```

#### 2.2.8 ROC and AUC

ROC, as well as AUC, is a well known metric that is used for evaluating binary classification problem. It is similar in the case of binary rating typed recommendation algorithm where the "hit" accuracy on the relevant items is used for measuring the recommender's performance.

To demonstrate the evaluation method, the original data for testing is manipuldated in a way that the ratings in the testing data are arranged as binary scores, whilst the ones in the prediction are scaled in 0 to 1.

```python
# Convert the original rating to 0 and 1.
df_true_bin = df_true.copy()
df_true_bin[COL_RATING] = df_true_bin[COL_RATING].apply(lambda x: 1 if x > 3 else 0)

df_true_bin
```

```text
    UserId  MovieId  Rating
0        1        1       1
1        1        2       1
2        1        3       0
3        2        1       1
4        2        4       1
5        2        5       0
6        2        6       0
7        2        7       0
8        3        2       1
9        3        5       1
10       3        6       1
11       3        8       1
12       3        9       1
13       3       10       0
14       3       11       0
15       3       12       0
16       3       13       0
17       3       14       0
```

```python
# Convert the predicted ratings into a [0, 1] scale.
df_pred_bin = df_pred.copy()
df_pred_bin[COL_PREDICTION] = minmax_scale(df_pred_bin[COL_PREDICTION].astype(float))

df_pred_bin
```

```text
    UserId  MovieId    Rating
0        1        3  1.000000
1        1       10  0.888889
2        1       12  0.777778
3        2       10  1.000000
4        2        3  0.888889
5        2        5  0.777778
6        2       11  0.666667
7        2       13  0.555556
8        3        4  1.000000
9        3       10  0.888889
10       3        7  0.777778
11       3       13  0.666667
12       3        1  0.555556
13       3        3  0.444444
14       3        5  0.333333
15       3        2  0.222222
16       3       11  0.111111
17       3       14  0.000000
```

```python
# Calculate the AUC metric
auc_score = auc(
    df_true_bin,
    df_pred_bin,
    col_user = COL_USER,
    col_item = COL_ITEM,
    col_rating = COL_RATING,
    col_prediction = COL_RATING
)

print(f"The auc score is {auc_score}")
```

```text
The auc score is 0.33333333333333337
```

It is worth mentioning that in some literature there are variants of the original AUC metric, that considers the effect of **the number of the recommended items (k)**, **grouping effect of users (compute AUC for each user group, and take the average across different groups)**. These variants are applicable to various different scenarios, and choosing an appropriate one depends on the context of the use case itself.

#### 2.3.2 Logistic loss

Logistic loss (sometimes it is called simply logloss, or cross-entropy loss) is another useful metric to evaluate the hit accuracy. It is defined as the negative log-likelihood of the true labels given the predictions of a classifier.

```python
# Calculate the logloss metric
logloss_score = logloss(
    df_true_bin,
    df_pred_bin,
    col_user = COL_USER,
    col_item = COL_ITEM,
    col_rating = COL_RATING,
    col_prediction = COL_RATING
)

print(f"The logloss score is {logloss_score}")
```

```text
The logloss score is 5.2574953720277815
```

It is worth noting that logloss may be sensitive to the class balance of datasets, as it penalizes heavily classifiers that are confident about incorrect classifications. To demonstrate, the ground truth data set for testing is manipulated purposely to unbalance the binary labels. For example, the following binarizes the original rating data by using a lower threshold, i.e., 2, to create more positive feedback from the user.

```python
df_true_bin_pos = df_true.copy()
df_true_bin_pos[COL_RATING] = df_true_bin_pos[COL_RATING].apply(lambda x: 1 if x > 2 else 0)

df_true_bin_pos
```

```text
    UserId  MovieId  Rating
0        1        1       1
1        1        2       1
2        1        3       1
3        2        1       1
4        2        4       1
5        2        5       1
6        2        6       1
7        2        7       0
8        3        2       1
9        3        5       1
10       3        6       1
11       3        8       1
12       3        9       1
13       3       10       1
14       3       11       1
15       3       12       1
16       3       13       0
17       3       14       0
```

By using threshold of 2, the labels in the ground truth data is not balanced, and the ratio of 1 over 0 is

```python
one_zero_ratio = df_true_bin_pos[COL_PREDICTION].sum() / (df_true_bin_pos.shape[0] - df_true_bin_pos[COL_PREDICTION].sum())

print(f"The ratio between label 1 and label 0 is {one_zero_ratio}")
```

```text
The ratio between label 1 and label 0 is 5.0
```

Another prediction data is also created, where the probabilities for label 1 and label 0 are fixed. Without loss of generity, the probability of predicting 1 is 0.6. The data set is purposely created to make the precision to be 100% given an presumption of cut-off equal to 0.5.

```python
prob_true = 0.6

df_pred_bin_pos = df_true_bin_pos.copy()
df_pred_bin_pos[COL_PREDICTION] = df_pred_bin_pos[COL_PREDICTION].apply(lambda x: prob_true if x==1 else 1-prob_true)

df_pred_bin_pos
```

```text
    UserId  MovieId  Rating
0        1        1     0.6
1        1        2     0.6
2        1        3     0.6
3        2        1     0.6
4        2        4     0.6
5        2        5     0.6
6        2        6     0.6
7        2        7     0.4
8        3        2     0.6
9        3        5     0.6
10       3        6     0.6
11       3        8     0.6
12       3        9     0.6
13       3       10     0.6
14       3       11     0.6
15       3       12     0.6
16       3       13     0.4
17       3       14     0.4
```

Then the logloss is calculated as follows.

```python
# Calculate the logloss metric
logloss_score_pos = logloss(
    df_true_bin_pos,
    df_pred_bin_pos,
    col_user = COL_USER,
    col_item = COL_ITEM,
    col_rating = COL_RATING,
    col_prediction = COL_RATING
)

print(f"The logloss score is {logloss_score}")
```

```text
The logloss score is 5.2574953720277815
```

For comparison, a similar process is used with a threshold value of 3 to create a more balanced dataset. Another prediction dataset is also created by using the balanced dataset. Again, the probabilities of predicting label 1 and label 0 are fixed as 0.6 and 0.4, respectively. **NOTE**, same as above, in this case, the prediction also gives us a 100% precision. The only difference is the proportion of binary labels.

```python
prob_true = 0.6

df_pred_bin_balanced = df_true_bin.copy()
df_pred_bin_balanced[COL_PREDICTION] = df_pred_bin_balanced[COL_PREDICTION].apply(lambda x: prob_true if x==1 else 1-prob_true)

df_pred_bin_balanced
```

```text
    UserId  MovieId  Rating
0        1        1     0.6
1        1        2     0.6
2        1        3     0.4
3        2        1     0.6
4        2        4     0.6
5        2        5     0.4
6        2        6     0.4
7        2        7     0.4
8        3        2     0.6
9        3        5     0.6
10       3        6     0.6
11       3        8     0.6
12       3        9     0.6
13       3       10     0.4
14       3       11     0.4
15       3       12     0.4
16       3       13     0.4
17       3       14     0.4
```

The ratio of label 1 and label 0 is

```python
one_zero_ratio = df_true_bin[COL_PREDICTION].sum() / (df_true_bin.shape[0] - df_true_bin[COL_PREDICTION].sum())

print(f"The ratio between label 1 and label 0 is {one_zero_ratio}")
```

```text
The ratio between label 1 and label 0 is 1.0
```

It is perfectly balanced.

Applying the logloss function to calculate the metric gives us a more promising result, as shown below.

```python
# Calculate the logloss metric
logloss_score = logloss(
    df_true_bin,
    df_pred_bin_balanced,
    col_user = COL_USER,
    col_item = COL_ITEM,
    col_rating = COL_RATING,
    col_prediction = COL_RATING
)

print(f"The logloss score is {logloss_score}")
```

```text
The logloss score is 0.5108256237659907
```

It can be seen that the score is more close to 0, and, by definition, it means that the predictions are generating better results than the one before where binary labels are more biased.

#### 2.3 Summary

|Metric|Range|Selection criteria|Limitation|Reference|
|------|-------------------------------|---------|----------|---------|
|Precision|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Only for hits in recommendations.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|Recall|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Only for hits in the ground truth.|[link](https://en.wikipedia.org/wiki/Precision_and_recall)|
|NDCG|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Does not penalize for bad/missing items, and does not perform for several equally good items.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|MAP|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Depend on variable distributions.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|AUC|$\geq 0$ and $\leq 1$|The closer to $1$ the better. 0.5 indicates an uninformative classifier|Depend on the number of recommended items (k).|[link](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)|
|Logloss|$0$ to $\infty$|The closer to $0$ the better.|Logloss can be sensitive to imbalanced datasets.|[link](https://en.wikipedia.org/wiki/Cross_entropy#Relation_to_log-likelihood)|

```python
# cleanup spark instance
spark.stop()
```

## References

1. Guy Shani and Asela Gunawardana, "Evaluating Recommendation Systems", Recommender Systems Handbook, Springer, 2015.
2. PySpark MLlib evaluation metrics, url: https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html.
3. Dimitris Paraschakis et al, "Comparative Evaluation of Top-N Recommenders in e-Commerce: An Industrial Perspective", IEEE ICMLA, 2015, Miami, FL, USA.
4. Yehuda Koren and Robert Bell, "Advances in Collaborative Filtering", Recommender Systems Handbook, Springer, 2015.
5. Chris Bishop, "Pattern Recognition and Machine Learning", Springer, 2006.
