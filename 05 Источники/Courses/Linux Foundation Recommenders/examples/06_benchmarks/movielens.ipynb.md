---
title: "Benchmark with Movielens dataset"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: 6232b154548c955315650d58dca6bf1411c56020
---

> [!note] Original source material
> This page preserves [`examples/06_benchmarks/movielens.ipynb`](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/06_benchmarks/movielens.ipynb) from
> *Linux Foundation Recommenders* at commit `6232b154548c955315650d58dca6bf1411c56020`. License:
> [MIT](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



<i>Copyright (c) Recommenders contributors.</i>

<i>Licensed under the MIT License.</i>

This illustrative comparison applies to collaborative filtering algorithms available in this repository such as Spark ALS, SAR and others using the Movielens dataset. These algorithms are usable in a variety of recommendation tasks, including product or news recommendations.

The main purpose of this notebook is not to produce comprehensive benchmarking results on multiple datasets. Rather, it is intended to illustrate on how one could evaluate different recommender algorithms using tools in this repository.

## Experimentation setup:

* Objective
  * To compare how each collaborative filtering algorithm perform in predicting ratings and recommending relevant items.

* Environment
  * The comparison is run on a machine with 4 CPUs, 30Gb of RAM, and 1 GPU GeForce GTX 1660 Ti with 6Gb of memory.
  * It should be noted that a local machine is not supposed to run scalable benchmarking analysis. Either scaling up or out the computing instances is necessary to run the benchmarking in an run-time efficient way without any memory issue.
  * **NOTE ABOUT THE DEPENDENCIES TO INSTALL**: This notebook uses CPU, GPU and PySpark algorithms, so make sure you install the `full environment` as detailed in the [SETUP.md](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/SETUP.md).

* Datasets
  * [Movielens 100K](https://grouplens.org/datasets/movielens/100k/).
  * [Movielens 1M](https://grouplens.org/datasets/movielens/1m/).

* Data split
  * The data is split into train and test sets.
  * The split ratios are 75-25 for train and test datasets.
  * The splitting is stratified based on items.

* Model training
  * A recommendation model is trained by using each of the collaborative filtering algorithms.
  * Empirical parameter values reported [here](http://mymedialite.net/examples/datasets.html) are used in this notebook. More exhaustive hyper parameter tuning would be required to further optimize results.

* Evaluation metrics
  * Ranking metrics:
    * Precision@k.
    * Recall@k.
    * Normalized discounted cumulative gain@k (NDCG@k).
    * Mean-average-precision (MAP).
    * In the evaluation metrics above, k = 10.
  * Rating metrics:
    * Root mean squared error (RMSE).
    * Mean average error (MAE).
    * R squared.
    * Explained variance.
  * Run time performance
    * Elapsed for training a model and using a model for predicting/recommending k items.
    * The time may vary across different machines.

## Globals settings

```python
# Remove warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"  # Set local IP to avoid hostname warnings
import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
```

```python
import sys
import numpy as np
import pandas as pd
import cornac

try:
    import pyspark
    from recommenders.utils.spark_utils import start_or_get_spark
except ImportError:
    pass  # skip this import if we are not in a Spark environment

try:
    import tensorflow as tf # NOTE: TF needs to be imported before PyTorch, otherwise we get an error
    tf.get_logger().setLevel("ERROR") # only show error messages
    import torch
    from recommenders.utils.gpu_utils import get_cuda_version, get_cudnn_version
except ImportError:
    pass  # skip this import if we are not in a GPU environment

# try:
#     import surprise # Put SVD surprise back in core deps when #2224 is fixed
# except:
#     pass

current_path = os.path.join(os.getcwd(), "examples", "06_benchmarks") # To execute the notebook programmatically from root folder
sys.path.append(current_path)
from benchmark_utils import *
from recommenders.datasets import movielens
from recommenders.utils.general_utils import get_number_processors
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.utils.notebook_utils import store_metadata


print(f"System version: {sys.version}")
print(f"Number of cores: {get_number_processors()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Cornac version: {cornac.__version__}")

# try:
#     print(f"Surprise version: {surprise.__version__}") # Put SVD surprise back in core deps when #2224 is fixed
# except NameError:
#     pass

try:
    print(f"PySpark version: {pyspark.__version__}")
except NameError:
    pass  # skip this import if we are not in a Spark environment

try:
    print(f"CUDA version: {get_cuda_version()}")
    print(f"CuDNN version: {get_cudnn_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"PyTorch version: {torch.__version__}")
except NameError:
    pass  # skip this import if we are not in a GPU environment

%load_ext autoreload
%autoreload 2
```

```text
System version: 3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]
Number of cores: 24
NumPy version: 1.26.4
Pandas version: 2.3.3
Cornac version: 2.3.0
PySpark version: 3.5.8
CUDA version: 13.0
CuDNN version: 91900
TensorFlow version: 2.15.1
PyTorch version: 2.11.0+cu130
```

```python
try:
    spark = start_or_get_spark("PySpark", memory="32g")
    spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
    # Suppress Spark warnings
    spark.sparkContext.setLogLevel("ERROR")
    log4j = spark._jvm.org.apache.log4j
    log4j.LogManager.getLogger("org").setLevel(log4j.Level.ERROR)
    log4j.LogManager.getLogger("akka").setLevel(log4j.Level.ERROR)
    log4j.LogManager.getLogger("org.apache.spark").setLevel(log4j.Level.ERROR)
    log4j.LogManager.getLogger("org.spark_project").setLevel(log4j.Level.ERROR)
except NameError:
    pass  # skip this import if we are not in a Spark environment
```

```text
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
26/05/12 18:10:49 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
```

```python
# Fix random seeds to make sure out runs are reproducible
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
except NameError:
    pass  # skip this import if we are not in a GPU environment
```

## Parameters

```python
data_sizes = ["100k"] # Movielens data size: 100k, 1m, 10m, or 20m
algorithms = ["als", "sar", "ncf", "embdotbias", "bpr", "bivae", "lightgcn"]
```

```python
environments = {
    "als": "pyspark",
    "sar": "python_cpu",
    # "svd": "python_cpu",
    "embdotbias": "python_gpu",
    "ncf": "python_gpu",
    "bpr": "python_cpu",
    "bivae": "python_gpu",
    "lightgcn": "python_gpu",
}

metrics = {
    "als": ["rating", "ranking"],
    "sar": ["ranking"],
    # "svd": ["rating", "ranking"],
    "embdotbias": ["rating", "ranking"],
    "ncf": ["ranking"],
    "bpr": ["ranking"],
    "bivae": ["ranking"],
    "lightgcn": ["ranking"]
}
```

Algorithm parameters

```python
als_params = {
    "rank": 10,
    "maxIter": 20,
    "implicitPrefs": False,
    "alpha": 0.1,
    "regParam": 0.05,
    "coldStartStrategy": "drop",
    "nonnegative": False,
    "userCol": DEFAULT_USER_COL,
    "itemCol": DEFAULT_ITEM_COL,
    "ratingCol": DEFAULT_RATING_COL,
}

sar_params = {
    "similarity_type": "jaccard",
    "time_decay_coefficient": 30,
    "time_now": None,
    "timedecay_formula": True,
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_timestamp": DEFAULT_TIMESTAMP_COL,
}

# svd_params = {
#     "n_factors": 150,
#     "n_epochs": 15,
#     "lr_all": 0.005,
#     "reg_all": 0.02,
#     "random_state": SEED,
#     "verbose": False
# }

embdotbias_params = {
    "n_factors": 40,
    "y_range": [0,5.5],
    "wd": 1e-1,
    "lr_max": 5e-3,
    "epochs": 15
}

ncf_params = {
    "model_type": "NeuMF",
    "n_factors": 4,
    "layer_sizes": [16, 8, 4],
    "n_epochs": 15,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "verbose": 10
}

bpr_params = {
    "k": 200,
    "max_iter": 200,
    "learning_rate": 0.01,
    "lambda_reg": 1e-3,
    "seed": SEED,
    "verbose": False
}

bivae_params = {
    "k": 100,
    "encoder_structure": [200],
    "act_fn": "tanh",
    "likelihood": "pois",
    "n_epochs": 500,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "seed": SEED,
    "use_gpu": True,
    "verbose": False
}

lightgcn_param = {
    "model_type": "lightgcn",
    "n_layers": 3,
    "batch_size": 1024,
    "embed_size": 64,
    "decay": 0.0001,
    "epochs": 20,
    "learning_rate": 0.005,
    "eval_epoch": 5,
    "top_k": DEFAULT_K,
    "metrics": ["recall", "ndcg", "precision", "map"],
    "save_model":False,
    "MODEL_DIR":".",
}

params = {
    "als": als_params,
    "sar": sar_params,
    # "svd": svd_params,
    "embdotbias": embdotbias_params,
    "ncf": ncf_params,
    "bpr": bpr_params,
    "bivae": bivae_params,
    "lightgcn": lightgcn_param,
}
```

```python
prepare_training_data = {
    "als": prepare_training_als,
    "sar": prepare_training_sar,
    # "svd": prepare_training_svd,
    "embdotbias": prepare_training_embdotbias,
    "ncf": prepare_training_ncf,
    "bpr": prepare_training_cornac,
    "bivae": prepare_training_cornac,
    "lightgcn": prepare_training_lightgcn,
}
```

```python
prepare_metrics_data = {
    "als": lambda train, test: prepare_metrics_als(train, test),
    "embdotbias": lambda train, test: prepare_metrics_embdotbias(train, test),
}
```

```python
trainer = {
    "als": lambda params, data: train_als(params, data),
    # "svd": lambda params, data: train_svd(params, data),
    "sar": lambda params, data: train_sar(params, data),
    "embdotbias": lambda params, data: train_embdotbias(params, data),
    "ncf": lambda params, data: train_ncf(params, data),
    "bpr": lambda params, data: train_bpr(params, data),
    "bivae": lambda params, data: train_bivae(params, data),
    "lightgcn": lambda params, data: train_lightgcn(params, data),
}
```

```python
rating_predictor = {
    "als": lambda model, test: predict_als(model, test),
    # "svd": lambda model, test: predict_svd(model, test),
    "embdotbias": lambda model, test: predict_embdotbias(model, test),
}
```

```python
ranking_predictor = {
    "als": lambda model, test, train: recommend_k_als(model, test, train),
    "sar": lambda model, test, train: recommend_k_sar(model, test, train),
    # "svd": lambda model, test, train: recommend_k_svd(model, test, train),
    "embdotbias": lambda model, test, train: recommend_k_embdotbias(model, test, train),
    "ncf": lambda model, test, train: recommend_k_ncf(model, test, train),
    "bpr": lambda model, test, train: recommend_k_bpr(model, test, train),
    "bivae": lambda model, test, train: recommend_k_bivae(model, test, train),
    "lightgcn": lambda model, test, train: recommend_k_lightgcn(model, test, train),
}
```

```python
rating_evaluator = {
    "als": lambda test, predictions: rating_metrics_pyspark(test, predictions),
    # "svd": lambda test, predictions: rating_metrics_python(test, predictions),
    "embdotbias": lambda test, predictions: rating_metrics_python(test, predictions)
}


ranking_evaluator = {
    "als": lambda test, predictions, k: ranking_metrics_pyspark(test, predictions, k),
    "sar": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    # "svd": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "embdotbias": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "ncf": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bpr": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bivae": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "lightgcn": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
}
```

```python
def generate_summary(data, algo, k, train_time, time_rating, rating_metrics, time_ranking, ranking_metrics):
    summary = {"Data": data, "Algo": algo, "K": k, "Train time (s)": train_time, "Predicting time (s)": time_rating, "Recommending time (s)": time_ranking}
    if rating_metrics is None:
        rating_metrics = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Explained Variance": np.nan,
        }
    if ranking_metrics is None:
        ranking_metrics = {
            "MAP@k": np.nan,
            "nDCG@k": np.nan,
            "Precision@k": np.nan,
            "Recall@k": np.nan,
        }
    summary.update(rating_metrics)
    summary.update(ranking_metrics)
    return summary
```

## Benchmark loop

```python
%%time

# For each data size and each algorithm, a recommender is evaluated.
cols = ["Data", "Algo", "K", "Train time (s)", "Predicting time (s)", "RMSE", "MAE", "R2", "Explained Variance", "Recommending time (s)", "MAP@k", "nDCG@k", "Precision@k", "Recall@k"]
df_results = pd.DataFrame(columns=cols)

for data_size in data_sizes:
    # Load the dataset
    df = movielens.load_pandas_df(
        size=data_size,
        header=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
    )
    print("Size of Movielens {}: {}".format(data_size, df.shape))

    # Split the dataset
    df_train, df_test = python_stratified_split(df,
                                                ratio=0.75,
                                                min_rating=1,
                                                filter_by="item",
                                                col_user=DEFAULT_USER_COL,
                                                col_item=DEFAULT_ITEM_COL
                                                )

    # Loop through the algos
    for algo in algorithms:
        print(f"\nComputing {algo} algorithm on Movielens {data_size}")

        # Data prep for training set
        train = prepare_training_data.get(algo, lambda x,y:(x,y))(df_train, df_test)

        # Get model parameters
        model_params = params[algo]

        # Train the model
        model, time_train = trainer[algo](model_params, train)
        print(f"Training time: {time_train}s")

        # Predict and evaluate
        train, test = prepare_metrics_data.get(algo, lambda x,y:(x,y))(df_train, df_test)

        if "rating" in metrics[algo]:
            # Predict for rating
            preds, time_rating = rating_predictor[algo](model, test)
            print(f"Rating prediction time: {time_rating}s")

            # Evaluate for rating
            ratings = rating_evaluator[algo](test, preds)
        else:
            ratings = None
            time_rating = np.nan

        if "ranking" in metrics[algo]:
            # Predict for ranking
            top_k_scores, time_ranking = ranking_predictor[algo](model, test, train)
            print(f"Ranking prediction time: {time_ranking}s")

            # Evaluate for rating
            rankings = ranking_evaluator[algo](test, top_k_scores, DEFAULT_K)
        else:
            rankings = None
            time_ranking = np.nan

        # Record results
        summary = generate_summary(data_size, algo, DEFAULT_K, time_train, time_rating, ratings, time_ranking, rankings)
        df_results.loc[df_results.shape[0] + 1] = summary

print("\nComputation finished")
```

```text
100%|██████████| 4.81k/4.81k [00:00<00:00, 5.22kKB/s]
```

```text
Size of Movielens 100k: (100000, 4)

Computing als algorithm on Movielens 100k
```

```text
Training time: 9.7989s
Rating prediction time: 0.1371s
Ranking prediction time: 0.1483s
```

```text

Computing sar algorithm on Movielens 100k
Training time: 0.4918s
Ranking prediction time: 0.1171s

Computing ncf algorithm on Movielens 100k
Training time: 91.1847s
Ranking prediction time: 10.5998s

Computing embdotbias algorithm on Movielens 100k
Training time: 107.2887s
Rating prediction time: 0.0495s
Ranking prediction time: 2.8453s

Computing bpr algorithm on Movielens 100k
Training time: 7.1731s
Ranking prediction time: 0.6734s

Computing bivae algorithm on Movielens 100k
Training time: 20.3837s
Ranking prediction time: 0.9034s

Computing lightgcn algorithm on Movielens 100k
Already create adjacency matrix.
Already normalize adjacency matrix.
Using xavier initialization.
Epoch 1 (train)3.5s: train loss = 0.47040 = (mf)0.47016 + (embed)0.00025
Epoch 2 (train)3.0s: train loss = 0.28737 = (mf)0.28673 + (embed)0.00063
Epoch 3 (train)1.5s: train loss = 0.25197 = (mf)0.25116 + (embed)0.00081
Epoch 4 (train)3.2s: train loss = 0.23655 = (mf)0.23557 + (embed)0.00098
Epoch 5 (train)3.0s + (eval)0.5s: train loss = 0.23006 = (mf)0.22896 + (embed)0.00111, recall = 0.16089, ndcg = 0.34701, precision = 0.29947, map = 0.21783
Epoch 6 (train)3.2s: train loss = 0.22270 = (mf)0.22150 + (embed)0.00121
Epoch 7 (train)3.1s: train loss = 0.21182 = (mf)0.21051 + (embed)0.00131
Epoch 8 (train)4.1s: train loss = 0.20782 = (mf)0.20639 + (embed)0.00143
Epoch 9 (train)3.2s: train loss = 0.19480 = (mf)0.19322 + (embed)0.00157
Epoch 10 (train)3.1s + (eval)0.4s: train loss = 0.18633 = (mf)0.18459 + (embed)0.00174, recall = 0.17786, ndcg = 0.38426, precision = 0.33531, map = 0.24908
Epoch 11 (train)3.2s: train loss = 0.17432 = (mf)0.17242 + (embed)0.00190
Epoch 12 (train)2.8s: train loss = 0.17215 = (mf)0.17010 + (embed)0.00205
Epoch 13 (train)1.5s: train loss = 0.17082 = (mf)0.16866 + (embed)0.00216
Epoch 14 (train)3.1s: train loss = 0.16716 = (mf)0.16489 + (embed)0.00227
Epoch 15 (train)2.9s + (eval)0.3s: train loss = 0.16167 = (mf)0.15928 + (embed)0.00239, recall = 0.19160, ndcg = 0.40435, precision = 0.35016, map = 0.26342
Epoch 16 (train)3.1s: train loss = 0.15764 = (mf)0.15514 + (embed)0.00250
Epoch 17 (train)3.2s: train loss = 0.15895 = (mf)0.15635 + (embed)0.00260
Epoch 18 (train)3.0s: train loss = 0.15448 = (mf)0.15178 + (embed)0.00270
Epoch 19 (train)3.2s: train loss = 0.15333 = (mf)0.15053 + (embed)0.00280
Epoch 20 (train)3.1s + (eval)1.1s: train loss = 0.14917 = (mf)0.14626 + (embed)0.00290, recall = 0.19623, ndcg = 0.41733, precision = 0.36182, map = 0.27628
Training time: 65.9464s
Ranking prediction time: 0.0888s

Computation finished
CPU times: user 10min 36s, sys: 52.3 s, total: 11min 28s
Wall time: 6min 17s
```

## Results

```python
df_results
```

```text
   Data        Algo   K Train time (s) Predicting time (s)      RMSE  \
1  100k         als  10         9.7989              0.1371  0.960785
2  100k         sar  10         0.4918                 NaN       NaN
3  100k         ncf  10        91.1847                 NaN       NaN
4  100k  embdotbias  10       107.2887              0.0495  0.992760
5  100k         bpr  10         7.1731                 NaN       NaN
6  100k       bivae  10        20.3837                 NaN       NaN
7  100k    lightgcn  10        65.9464                 NaN       NaN

        MAE        R2  Explained Variance Recommending time (s)     MAP@k  \
1  0.747694  0.276669            0.272568                0.1483  0.012740
2       NaN       NaN                 NaN                0.1171  0.258452
3       NaN       NaN                 NaN               10.5998  0.252473
4  0.776040  0.223344            0.223393                2.8453  0.053549
5       NaN       NaN                 NaN                0.6734  0.297456
6       NaN       NaN                 NaN                0.9034  0.329962
7       NaN       NaN                 NaN                0.0888  0.276277

     nDCG@k  Precision@k  Recall@k
1  0.037863     0.042842  0.013880
2  0.393819     0.340615  0.185377
3  0.384246     0.338176  0.176739
4  0.117810     0.104242  0.042450
5  0.444991     0.388653  0.216555
6  0.471722     0.411347  0.224300
7  0.417327     0.361824  0.196233
```

```python
# Record results for tests - ignore this cell
for algo in algorithms:
    store_metadata(algo, df_results.loc[df_results["Algo"] == algo, "nDCG@k"].values[0])
```
