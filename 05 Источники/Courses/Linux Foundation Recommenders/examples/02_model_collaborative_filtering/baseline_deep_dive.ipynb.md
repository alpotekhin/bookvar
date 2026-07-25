---
title: "Estimating Baseline Performance"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: 6232b154548c955315650d58dca6bf1411c56020
---

> [!note] Original source material
> This page preserves [`examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb`](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/02_model_collaborative_filtering/baseline_deep_dive.ipynb) from
> *Linux Foundation Recommenders* at commit `6232b154548c955315650d58dca6bf1411c56020`. License:
> [MIT](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



<i>Copyright (c) Recommenders contributors.</i>

<i>Licensed under the MIT License.</i>

<br>
Estimating baseline performance is as important as choosing right metrics for model evaluation. In this notebook, we briefly discuss about why do we care about baseline performance and how to measure it.

The notebook covers two example scenarios under the context of movie recommendation: 1) rating prediction and 2) top-k recommendation.

### Why does baseline performance matter?
<br>
Before we go deep dive into baseline performance estimation, it is worth to think about why we need that.

As we can simply see from the definition of the word 'baseline', <b>baseline performance</b> is a minimum performance we expect to achieve by a model or starting point used for model comparisons.

Once we train a model and get results from evaluation metrics we choose, we will wonder how should we interpret the metrics or even wonder if the trained model is better than a simple rule-based model. Baseline results help us to understand those.

Let's say we are building a food recommender. We evaluated the model on the test set and got nDCG (at 10) = 0.3. At that moment, we would not know if the model is good or bad. But once we find out that a simple rule of <i>'recommending top-10 most popular foods to all users'</i> can achieve nDCG = 0.4, we see that our model is not good enough. Maybe the model is not trained well, or maybe we should think about if nDCG is the right metric for prediction of user behaviors in the given problem.

### How can we estimate the baseline performance?
<br>
To estimate the baseline performance, we first pick a baseline model and evaluate it by using the same evaluation metrics we will use for our main model. In general, a very simple rule or even <b>zero rule</b>--<i>predicts the mean for regression or the mode for classification</i>--will be a enough as a baseline model (Random-prediction might be okay for certain problems, but usually it performs poor than the zero rule). If we already have a running model in hand and now trying to improve that, we can use the previous results as a baseline performance for sure.

Most importantly, <b>different baseline approaches should be taken for different problems and business goals</b>. For example, recommending the previously purchased items could be used as a baseline model for food or restaurant recommendation since people tend to eat the same foods repeatedly. For TV show and/or movie recommendation, on the other hand, recommending previously watched items does not make sense. Probably recommending the most popular (most watched or highly rated) items is more likely useful as a baseline.

In this notebook, we demonstrate how to estimate the baseline performance for the movie recommendation with MovieLens dataset. We use the mean for rating prediction, i.e. our baseline model will predict a user's rating of a movie by averaging the ratings the user previously submitted for other movies. For the top-k recommendation problem, we use top-k most-rated movies as the baseline model. We choose the number of ratings here because we regard the binary signal of 'rated vs. not-rated' as user's implicit preference when evaluating ranking metrics.

Now, let's jump into the implementation!

```python
import sys
import itertools
import pandas as pd

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.datasets.pandas_df_utils import filter_by
from recommenders.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
```

```text
System version: 3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]
Pandas version: 2.3.3
```

First, let's prepare training and test data sets.

```python
MOVIELENS_DATA_SIZE = "100k"
TOP_K = 10
```

```python
data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["UserId", "MovieId", "Rating", "Timestamp"]
)

data.head()
```

```text
100%|██████████| 4.81k/4.81k [00:00<00:00, 5.78kKB/s]
```

```text
   UserId  MovieId  Rating  Timestamp
0     196      242     3.0  881250949
1     186      302     3.0  891717742
2      22      377     1.0  878887116
3     244       51     2.0  880606923
4     166      346     1.0  886397596
```

```python
train, test = python_random_split(data, ratio=0.75, seed=42)
```

### 1. Rating prediction baseline

As we discussed earlier, we use each user's **mean rating** as the baseline prediction.

```python
# Calculate avg ratings from the training set
users_ratings = train.groupby(["UserId"])["Rating"].mean()
users_ratings = users_ratings.to_frame().reset_index()
users_ratings.rename(columns={"Rating": "AvgRating"}, inplace=True)

users_ratings.head()
```

```text
   UserId  AvgRating
0       1   3.696970
1       2   3.837209
2       3   2.744186
3       4   4.500000
4       5   2.868217
```

```python
# Generate prediction for the test set
baseline_predictions = pd.merge(test, users_ratings, on=["UserId"], how="inner")

baseline_predictions.loc[baseline_predictions["UserId"] == 1].head()
```

```text
     UserId  MovieId  Rating  Timestamp  AvgRating
375       1      233     2.0  878542552    3.69697
385       1      159     3.0  875073180    3.69697
591       1      238     4.0  875072235    3.69697
656       1      100     5.0  878543541    3.69697
865       1       63     2.0  878543196    3.69697
```

Now, let's evaluate how our baseline model will perform on regression metrics

```python
baseline_predictions = baseline_predictions[​["UserId", "MovieId", "AvgRating"]​]

cols = {
    "col_user": "UserId",
    "col_item": "MovieId",
    "col_rating": "Rating",
    "col_prediction": "AvgRating",
}

eval_rmse = rmse(test, baseline_predictions, **cols)
eval_mae = mae(test, baseline_predictions, **cols)
eval_rsquared = rsquared(test, baseline_predictions, **cols)
eval_exp_var = exp_var(test, baseline_predictions, **cols)

print("RMSE:\t\t%f" % eval_rmse,
      "MAE:\t\t%f" % eval_mae,
      "rsquared:\t%f" % eval_rsquared,
      "exp var:\t%f" % eval_exp_var, sep='\n')
```

```text
RMSE:		1.044885
MAE:		0.836925
rsquared:	0.136491
exp var:	0.136496
```

As you can see, our baseline model actually performed quite well on the metrics. E.g. MAE (Mean Absolute Error) was around 0.84 on MovieLens 100k data, saying that users actual ratings would be within +-0.84 of their mean ratings. This also gives us an insight that users' rating could be biased where some users tend to give high ratings for all movies while others give low ratings.

Now, next time we build our machine-learning model, we will want to make the model performs better than this baseline.

### 2. Top-k recommendation baseline

Recommending the **most popular items** is intuitive and simple approach that works for many of recommendation scenarios. Here, we use top-k most-rated movies as the baseline model as we discussed earlier.

```python
item_counts = train["MovieId"].value_counts().to_frame().reset_index()
item_counts.columns = ["MovieId", "Count"]
item_counts.head()
```

```text
   MovieId  Count
0       50    419
1      181    382
2      100    381
3      258    377
4      288    371
```

```python
user_item_col = ["UserId", "MovieId"]

# Cross join users and items
test_users = test['UserId'].unique()
user_item_list = list(itertools.product(test_users, item_counts['MovieId']))
users_items = pd.DataFrame(user_item_list, columns=user_item_col)

print("Number of user-item pairs:", len(users_items))

# Remove seen items (items in the train set) as we will not recommend those again to the users
users_items_remove_seen = filter_by(users_items, train, user_item_col)

print("After remove seen items:", len(users_items_remove_seen))
```

```text
Number of user-item pairs: 1546764
After remove seen items: 1471784
```

```python
# Generate recommendations
baseline_recommendations = pd.merge(item_counts, users_items_remove_seen, on=['MovieId'], how='inner')
baseline_recommendations.head()
```

```text
   MovieId  Count  UserId
0       50    419     877
1       50    419     815
2       50    419     416
3       50    419     259
4       50    419     598
```

```python
cols["col_prediction"] = "Count"

eval_map = map_at_k(test, baseline_recommendations, k=TOP_K, **cols)
eval_ndcg = ndcg_at_k(test, baseline_recommendations, k=TOP_K, **cols)
eval_precision = precision_at_k(test, baseline_recommendations, k=TOP_K, **cols)
eval_recall = recall_at_k(test, baseline_recommendations, k=TOP_K, **cols)

print("MAP@K:\t%f" % eval_map,
      "NDCG@K:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
```

```text
MAP@K:	0.138664
NDCG@K:	0.252867
Precision@K:	0.224628
Recall@K:	0.111736
```

Again, the baseline is quite high, nDCG = 0.25 and Precision = 0.22.

<br>

### Concluding remarks

In this notebook, we discussed how to measure baseline performance for the movie recommendation example.
We covered very naive approaches as baselines, but still they are useful in a sense that they can provide reference numbers to estimate the complexity of the given problem as well as the relative performance of the recommender models we are building.

```python
# Record results for tests - ignore this cell
store_metadata("map", eval_map)
store_metadata("ndcg", eval_ndcg)
store_metadata("precision", eval_precision)
store_metadata("recall", eval_recall)
store_metadata("rmse", eval_rmse)
store_metadata("mae", eval_mae)
store_metadata("exp_var", eval_exp_var)
store_metadata("rsquared", eval_rsquared)
```

### References

[[1](https://dl.acm.org/citation.cfm?id=1401944)] Yehuda Koren,	Factorization meets the neighborhood: a multifaceted collaborative filtering model, KDD '08 pp. 426-434 2008.
[[2](https://surprise.readthedocs.io/en/stable/basic_algorithms.html)] Surprise lib, Basic algorithms
