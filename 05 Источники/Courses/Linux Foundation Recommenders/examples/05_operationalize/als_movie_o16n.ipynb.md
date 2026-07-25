---
title: "Building a Real-time Recommendation API"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: 6232b154548c955315650d58dca6bf1411c56020
---

> [!note] Original source material
> This page preserves [`examples/05_operationalize/als_movie_o16n.ipynb`](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/05_operationalize/als_movie_o16n.ipynb) from
> *Linux Foundation Recommenders* at commit `6232b154548c955315650d58dca6bf1411c56020`. License:
> [MIT](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



<i>Copyright (c) Recommenders contributors.</i>

<i>Licensed under the MIT License.</i>

This reference architecture shows the full lifecycle of building a recommendation system. It walks through the creation of appropriate azure resources, training a recommendation model using a Virtual Machine or Databricks, and deploying it as an API. It uses Azure Cosmos DB, Azure Machine Learning, and Azure Kubernetes Service.

This architecture can be generalized for many recommendation engine scenarios, including recommendations for products, movies, and news.
### Architecture
![architecture](https://raw.githubusercontent.com/recommenders-team/resources/main/images/reco-arch.png)

**Scenario**: A media organization wants to provide movie or video recommendations to its users. By providing personalized recommendations, the organization meets several business goals, including increased click-through rates, increased engagement on site, and higher user satisfaction.

In this reference, we train and deploy a real-time recommender service API that can provide the top 10 movie recommendations for a given user.

### Components
This architecture consists of the following key components:
* [Azure Databricks](https://docs.microsoft.com/en-us/azure/azure-databricks/what-is-azure-databricks)<sup>1)</sup> is used as a development environment to prepare input data and train the recommender model on a Spark cluster. Azure Databricks also provides an interactive workspace to run and collaborate on notebooks for any data processing or machine learning tasks.
* [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/aks/intro-kubernetes)(AKS) is used to deploy and operationalize a machine learning model service API on a Kubernetes cluster. AKS hosts the containerized model, providing scalability that meets throughput requirements, identity and access management, and logging and health monitoring.
* [Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction) is a globally distributed database service used to store the top 10 recommended movies for each user. Azure Cosmos DB is ideal for this scenario as it provides low latency (10 ms at 99th percentile) to read the top recommended items for a given user.
* [Azure Machine Learning Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/) is a service used to track and manage machine learning models, and then package and deploy these models to a scalable Azure Kubernetes Service environment.

<sup>1) Here, we are just giving an example of using Azure Databricks. Any platforms listed in [SETUP](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/SETUP.md) can be used as well.</sup>


### Table of Contents.
0. [File Imports](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/05_operationalize/als_movie_o16n.ipynb#0-File-Imports)
1. [Service Creation](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/05_operationalize/als_movie_o16n.ipynb#1-Service-Creation)
2. [Training and evaluation](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/05_operationalize/als_movie_o16n.ipynb#2-Training)
3. [Operationalization](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/05_operationalize/als_movie_o16n.ipynb#3.-Operationalize-the-Recommender-Service)

## Setup
To run this notebook on Azure Databricks, you should setup Azure Databricks by following the appropriate sections in the repository [SETUP instructions](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/SETUP.md) and import this notebook into your Azure Databricks Workspace (see instructions [here](https://docs.azuredatabricks.net/user-guide/notebooks/notebook-manage.html#import-a-notebook)).

Please note: This notebook **REQUIRES** that you add the dependencies to support **operationalization**. See [SETUP](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/SETUP.md) for details.

## 0 File Imports

```python
import os
import sys
import urllib

from azure.common.client_factory import get_client_from_cli_profile
import azure.mgmt.cosmosdb
import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import Webservice, AksWebservice
from azureml.exceptions import WebserviceException
from azureml.core import Environment
from azureml.core.environment import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.environment import SparkPackage
import pydocumentdb.document_client as document_client
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, LongType

from recommenders.datasets import movielens
from recommenders.datasets.cosmos_cli import find_collection, read_collection, read_database, find_database
from recommenders.datasets.download_utils import maybe_download
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from recommenders.utils.notebook_utils import is_databricks
from recommenders.utils.timer import Timer
from recommenders.utils.spark_utils import start_or_get_spark

print("Azure SDK version:", azureml.core.VERSION)
```

```text
Azure SDK version: 1.0.69
```

```python
# Start spark session if needed
if not is_databricks():
    cosmos_connector = (
        "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/"
        "azure-cosmosdb-spark_2.3.0_2.11/1.3.3/azure-cosmosdb-spark_2.3.0_2.11-1.3.3-uber.jar"
    )
    jar_filepath = maybe_download(url=cosmos_connector, filename="cosmos.jar")
    spark = start_or_get_spark("ALS", memory="10g", jars=[jar_filepath])
    sc = spark.sparkContext
print(sc)
```

```text
<SparkContext master=local[*] appName=ALS>
```

## 1 Service Creation
Modify the **Subscription ID** to the subscription you would like to deploy to and set the resource name variables.

#### Services created by this notebook:
1. [Azure ML Service](https://azure.microsoft.com/en-us/services/machine-learning-service/)
    1. [Azure ML Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace)
    1. [Azure Application Insights](https://azure.microsoft.com/en-us/services/monitor/)
    1. [Azure Storage](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-overview)
    1. [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/)

1. [Azure Cosmos DB](https://azure.microsoft.com/en-us/services/cosmos-db/)
1. [Azure Kubernetes Service (AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/)

**Add your Azure subscription ID**

```python
# Add your subscription ID
subscription_id = ""

# Set your workspace name
workspace_name = "o16n-test"
resource_group = "{}-rg".format(workspace_name)

# Set your region to deploy Azure ML workspace
location = "eastus"

# AzureML service and Azure Kubernetes Service prefix
service_name = "mvl-als"
```

```python
# Login for Azure CLI so that AzureML can use Azure CLI login credentials
!az login
```

```python
# Change subscription if needed
!az account set --subscription {subscription_id}
```

```python
# Check account
!az account show
```

```python
# CosmosDB
# account_name for CosmosDB cannot have "_" and needs to be less than 31 chars
account_name = "{}-ds-sql".format(workspace_name).replace("_", "-")[:31]
cosmos_database = "recommendations"
cosmos_collection = "user_recommendations_als"

# AzureML resource names
model_name = "{}-reco.mml".format(service_name)
aks_name = "{}-aks".format(service_name)
```

```python
# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'
```

```python
userCol = "UserId"
itemCol = "MovieId"
ratingCol = "Rating"

train_data_path = "train"
test_data_path = "test"
```

### 1.1 Import or create the AzureML Workspace.
This command will check if the AzureML Workspace exists or not, and will create the workspace if it doesn't exist.

```python
ws = Workspace.create(
    name=workspace_name,
    subscription_id=subscription_id,
    resource_group=resource_group,
    location=location,
    exist_ok=True
)
```

### 1.2 Create a Cosmos DB to store recommendation results

This step will take some time to create CosmosDB resources.

```python
# explicitly pass subscription_id in case user has multiple subscriptions
client = get_client_from_cli_profile(
    azure.mgmt.cosmosdb.CosmosDB,
    subscription_id=subscription_id
)

async_cosmosdb_create = client.database_accounts.create_or_update(
    resource_group,
    account_name,
    {
        'location': location,
        'locations': [{
            'location_name': location
        }]
    }
)
account = async_cosmosdb_create.result()

my_keys = client.database_accounts.list_keys(resource_group, account_name)
master_key = my_keys.primary_master_key
endpoint = "https://" + account_name + ".documents.azure.com:443/"

# DB client
client = document_client.DocumentClient(endpoint, {'masterKey': master_key})

if not find_database(client, cosmos_database):
    db = client.CreateDatabase({'id': cosmos_database })
    print("Database created")
else:
    db = read_database(client, cosmos_database)
    print("Database found")

# Create collection options
options = dict(offerThroughput=11000)

# Create a collection
collection_definition = {
    'id': cosmos_collection,
    'partitionKey': {'paths': ['/id'],'kind': 'Hash'}
}
if not find_collection(client, cosmos_database, cosmos_collection):
    collection = client.CreateCollection(
        db['_self'],
        collection_definition,
        options
    )
    print("Collection created")
else:
    collection = read_collection(client, cosmos_database, cosmos_collection)
    print("Collection found")

dbsecrets = dict(
    Endpoint=endpoint,
    Masterkey=master_key,
    Database=cosmos_database,
    Collection=cosmos_collection,
    Upsert=True
)
```

```text
Database created
Collection created
```

## 2 Training

Next, we train an [Alternating Least Squares model](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html) on [MovieLens](https://grouplens.org/datasets/movielens/) dataset.

### 2.1 Download the MovieLens dataset

```python
# Note: The DataFrame-based API for ALS currently only supports integers for user and item ids.
schema = StructType(
    (
        StructField(userCol, IntegerType()),
        StructField(itemCol, IntegerType()),
        StructField(ratingCol, FloatType()),
    )
)

data = movielens.load_spark_df(spark, size=MOVIELENS_DATA_SIZE, schema=schema)
data.show()
```

```text
100%|██████████| 4.81k/4.81k [00:00<00:00, 11.0kKB/s]
```

```text
+------+-------+------+
|UserId|MovieId|Rating|
+------+-------+------+
|   196|    242|   3.0|
|   186|    302|   3.0|
|    22|    377|   1.0|
|   244|     51|   2.0|
|   166|    346|   1.0|
|   298|    474|   4.0|
|   115|    265|   2.0|
|   253|    465|   5.0|
|   305|    451|   3.0|
|     6|     86|   3.0|
|    62|    257|   2.0|
|   286|   1014|   5.0|
|   200|    222|   5.0|
|   210|     40|   3.0|
|   224|     29|   3.0|
|   303|    785|   3.0|
|   122|    387|   5.0|
|   194|    274|   2.0|
|   291|   1042|   4.0|
|   234|   1184|   2.0|
+------+-------+------+
only showing top 20 rows
```

### 2.2 Split the data into train, test
There are several ways of splitting the data: random, chronological, stratified, etc., each of which favors a different real-world evaluation use case. We will split randomly in this example – for more details on which splitter to choose, consult [this guide](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/01_prepare_data/data_split.ipynb).

```python
train, test = spark_random_split(data, ratio=0.75, seed=42)
print("N train", train.cache().count())
print("N test", test.cache().count())
```

```text
N train 75031
N test 24969
```

### 2.3 Train the ALS model on the training data

To predict movie ratings, we use the rating data in the training set as users' explicit feedback. The hyperparameters used to estimate the model are set based on [this page](http://mymedialite.net/examples/datasets.html).

Under most circumstances, you would explore the hyperparameters and choose an optimal set based on some criteria. For additional details on this process, please see additional information in the deep dives [here](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/04_model_select_and_optimize/tuning_spark_als.ipynb).

```python
als = ALS(
    rank=10,
    maxIter=15,
    implicitPrefs=False,
    alpha=0.1,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=True,
    userCol=userCol,
    itemCol=itemCol,
    ratingCol=ratingCol,
)
```

```python
model = als.fit(train)
```

### 2.4 Get top-k recommendations for our testing data

In the movie recommendation use case, recommending movies that have been rated by the users do not make sense. Therefore, the rated movies are removed from the recommended items.

In order to achieve this, we recommend all movies to all users, and then remove the user-movie pairs that exist in the training dataset.

```python
# Get the cross join of all user-item pairs and score them.
users = train.select(userCol).distinct()
items = train.select(itemCol).distinct()
user_item = users.crossJoin(items)
dfs_pred = model.transform(user_item)
dfs_pred.show()
```

```text
+------+-------+----------+
|UserId|MovieId|prediction|
+------+-------+----------+
|   148|    148| 2.2560365|
|   463|    148|  2.936453|
|   471|    148| 3.8262048|
|   496|    148| 2.2901149|
|   833|    148| 1.7296925|
|   243|    148| 2.2667758|
|   392|    148| 2.4605818|
|   540|    148| 3.0631547|
|   623|    148| 3.1649487|
|   737|    148| 1.7344649|
|   858|    148| 1.8472893|
|   897|    148| 3.5229573|
|    31|    148| 1.9613894|
|   516|    148| 3.1411705|
|    85|    148| 2.2291098|
|   137|    148| 4.0498815|
|   251|    148| 3.2075853|
|   451|    148|  4.016654|
|   580|    148|  2.843738|
|   808|    148| 3.4666717|
+------+-------+----------+
only showing top 20 rows
```

```python
# Remove seen items.
dfs_pred_exclude_train = dfs_pred.alias("pred").join(
    train.alias("train"),
    (dfs_pred[userCol]==train[userCol]) & (dfs_pred[itemCol]==train[itemCol]),
    how='outer'
)
top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train."+ratingCol].isNull()) \
    .select("pred."+userCol, "pred."+itemCol, "pred.prediction")

top_all.show()
```

```text
+------+-------+----------+
|UserId|MovieId|prediction|
+------+-------+----------+
|     1|    587| 3.4595456|
|     1|    869|  2.967618|
|     1|   1208|  2.858056|
|     1|   1677| 2.9235902|
|     2|     80| 3.0129535|
|     2|    303| 3.0719132|
|     2|    472| 3.4143965|
|     2|    582|  4.877232|
|     2|    838|  1.529903|
|     2|    975| 2.9654517|
|     2|   1260|  3.252151|
|     2|   1325| 1.1417896|
|     2|   1381| 3.7900786|
|     2|   1530|  2.625749|
|     3|     22| 2.7082264|
|     3|     57| 2.5156925|
|     3|     89| 3.7927365|
|     3|    367| 2.7083492|
|     3|   1091| 1.5662774|
|     3|   1167| 3.2427955|
+------+-------+----------+
only showing top 20 rows
```

### 2.5 Evaluate how well ALS performs

Evaluate model performance using metrics such as Precision@K, Recall@K, [MAP@K](https://en.wikipedia.org/wiki/Evaluation_measures_\(information_retrieval\) or [nDCG@K](https://en.wikipedia.org/wiki/Discounted_cumulative_gain). For a full guide on what metrics to evaluate your recommender with, consult [this guide]../03_evaluate/evaluation.ipynb).

```python
cols = {
    'col_user': userCol,
    'col_item': itemCol,
    'col_rating': ratingCol,
    'col_prediction': "prediction",
}

test.show()
```

```text
+------+-------+------+
|UserId|MovieId|Rating|
+------+-------+------+
|     1|      2|   3.0|
|     1|      3|   4.0|
|     1|      4|   3.0|
|     1|     14|   5.0|
|     1|     17|   3.0|
|     1|     27|   2.0|
|     1|     29|   1.0|
|     1|     35|   1.0|
|     1|     36|   2.0|
|     1|     51|   4.0|
|     1|     52|   4.0|
|     1|     54|   3.0|
|     1|     56|   4.0|
|     1|     60|   5.0|
|     1|     64|   5.0|
|     1|     69|   3.0|
|     1|     77|   4.0|
|     1|     83|   3.0|
|     1|     85|   3.0|
|     1|     88|   4.0|
+------+-------+------+
only showing top 20 rows
```

```python
# Evaluate Ranking Metrics
rank_eval = SparkRankingEvaluation(
    test,
    top_all,
    k=TOP_K,
    **cols
)

print(
    "Model:\tALS",
    "Top K:\t%d" % rank_eval.k,
    "MAP:\t%f" % rank_eval.map_at_k(),
    "NDCG:\t%f" % rank_eval.ndcg_at_k(),
    "Precision@K:\t%f" % rank_eval.precision_at_k(),
    "Recall@K:\t%f" % rank_eval.recall_at_k(), sep='\n'
)
```

```text
Model:	ALS
Top K:	10
MAP:	0.003698
NDCG:	0.034331
Precision@K:	0.039343
Recall@K:	0.014976
```

```python
# Evaluate Rating Metrics
prediction = model.transform(test)
rating_eval = SparkRatingEvaluation(
    test,
    prediction,
    **cols
)

print(
    "Model:\tALS rating prediction",
    "RMSE:\t%.2f" % rating_eval.rmse(),
    "MAE:\t%f" % rating_eval.mae(),
    "Explained variance:\t%f" % rating_eval.exp_var(),
    "R squared:\t%f" % rating_eval.rsquared(), sep='\n'
)
```

```text
Model:	ALS rating prediction
RMSE:	0.95
MAE:	0.740282
Explained variance:	0.289807
R squared:	0.285394
```

### 2.6 Save the model

```python
(model
 .write()
 .overwrite()
 .save(model_name))
```

## 3. Operationalize the Recommender Service
Once the model is built with desirable performance, it will be operationalized to run as a REST endpoint to be utilized by a real time service. We will utilize [Azure Cosmos DB](https://azure.microsoft.com/en-us/services/cosmos-db/), [Azure Machine Learning Service](https://azure.microsoft.com/en-us/services/machine-learning-service/), and [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/aks/intro-kubernetes) to operationalize the recommender service.

### 3.1 Create a look-up for Recommendations in Cosmos DB

First, the Top-10 recommendations for each user as predicted by the model are stored as a lookup table in Cosmos DB. At runtime, the service will return the Top-10 recommendations as precomputed and stored in Cosmos DB:

```python
recs = model.recommendForAllUsers(10)
recs_topk = recs.withColumn("id", recs[userCol].cast("string")) \
    .select("id", "recommendations." + itemCol)
recs_topk.show()
```

```text
+---+--------------------+
| id|             MovieId|
+---+--------------------+
|471|[745, 1540, 244, ...|
|463|[64, 190, 1286, 3...|
|833|[1192, 179, 1524,...|
|496|[320, 1589, 262, ...|
|148|[1512, 718, 793, ...|
|540|[958, 1512, 1368,...|
|392|[1643, 1449, 1512...|
|243|[285, 251, 1405, ...|
|623|[390, 1643, 173, ...|
|737|[856, 60, 61, 151...|
|897|[1368, 958, 320, ...|
|858|[1154, 1129, 853,...|
| 31|[1203, 1245, 889,...|
|516|[745, 694, 1512, ...|
|580|[1368, 958, 1589,...|
|251|[1203, 1449, 253,...|
|451|[1368, 1019, 958,...|
| 85|[1643, 1449, 511,...|
|137|[1368, 1643, 958,...|
|808|[1512, 867, 1367,...|
+---+--------------------+
only showing top 20 rows
```

```python
# Save data to CosmosDB
(recs_topk.coalesce(1)
 .write
 .format("com.microsoft.azure.cosmosdb.spark")
 .mode('overwrite')
 .options(**dbsecrets)
 .save())
```

### 3.2 Configure Azure Machine Learning

Next, Azure Machine Learning Service is used to create a model scoring image and deploy it to Azure Kubernetes Service as a scalable containerized service. To achieve this, a **scoring script** should be created. In the script, we make a call to Cosmos DB to lookup the top 10 movies to recommend given an input User ID.

```python
score_sparkml = """
import json
import pydocumentdb.document_client as document_client

def init(local=False):
    global client, collection
    try:
        client = document_client.DocumentClient('{endpoint}', dict(masterKey='{key}'))
        collection = client.ReadCollection(collection_link='dbs/{database}/colls/{collection}')
    except Exception as e:
        collection = e

def run(input_json):
    try:
        # Query them in SQL
        id = str(json.loads(json.loads(input_json)[0])['id'])
        query = dict(query='SELECT * FROM c WHERE c.id = "' + id +'"')
        options = dict(partitionKey=str(id))
        document_link = 'dbs/{database}/colls/{collection}/docs/' + id
        result = client.ReadDocument(document_link, options);
    except Exception as e:
        result = str(e)
    return json.dumps(str(result))
""".format(key=dbsecrets['Masterkey'],
           endpoint=dbsecrets['Endpoint'],
           database=dbsecrets['Database'],
           collection=dbsecrets['Collection'])

# test validity of python string
exec(score_sparkml)

with open("score_sparkml.py", "w") as file:
    file.write(score_sparkml)
```

Register your model:

```python
mymodel = Model.register(
    model_path=model_name,  # this points to a local file
    model_name=model_name,  # this is the name the model is registered as
    description="AML trained model",
    workspace=ws
)

print(mymodel.name, mymodel.description, mymodel.version)
```

```text
Registering model mvl-als-reco.mml
mvl-als-reco.mml AML trained model 1
```

### 3.3 Deploy the model as a Service on AKS

#### 3.3.1 Create an Environment for your model:

```python
env = Environment(name='sparkmlenv')

# Specify a public image from microsoft/mmlspark as base image
env.docker.base_image="microsoft/mmlspark:0.15"

pip = [
    'azureml-defaults',
    'numpy==1.14.2',
    'scikit-learn==0.19.1',
    'pandas',
    'pydocumentdb'
]

# Add dependencies needed for inferencing
env.python.conda_dependencies = CondaDependencies.create(pip_packages=pip)
env.inferencing_stack_version = "latest"

# Add spark packages
env.spark.precache_packages = True
env.spark.repositories = ["https://mmlspark.azureedge.net/maven"]
env.spark.packages= [
    SparkPackage("com.microsoft.ml.spark", "mmlspark_2.11", "0.15"),
    SparkPackage("com.microsoft.azure", artifact="azure-storage", version="2.0.0"),
    SparkPackage(group="org.apache.hadoop", artifact="hadoop-azure", version="2.7.0")
]
```

#### 3.3.2 Create an AKS Cluster to run your container
This may take 20 to 30 minutes depending on the cluster size.

```python
# Verify that cluster does not exist already
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    # Create the cluster using the default configuration (can also provide parameters to customize)
    prov_config = AksCompute.provisioning_configuration()
    aks_target = ComputeTarget.create(
        workspace=ws,
        name=aks_name,
        provisioning_configuration=prov_config
    )
    aks_target.wait_for_completion(show_output = True)
    print(aks_target.provisioning_state)
    # To check any error logs, print(aks_target.provisioning_errors)
```

```text
Creating.......................................................................................................
SucceededProvisioning operation finished, operation "Succeeded"
Succeeded
```

#### 3.3.3 Deploy the container image to AKS:

```python
# Create an Inferencing Configuration with your environment and scoring script
inference_config = InferenceConfig(
    environment=env,
    entry_script="score_sparkml.py"
)

# Set the web service configuration (using default here with app insights)
aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)

# Webservice creation using single command
try:
    aks_service = Model.deploy(
        workspace=ws,
        models=[mymodel],
        name=service_name,
        inference_config=inference_config,
        deployment_config=aks_config,
        deployment_target=aks_target
    )
    aks_service.wait_for_deployment(show_output=True)
except WebserviceException:
    # Retrieve existing service.
    aks_service = Webservice(ws, name=service_name)
    print("Retrieved existing service")
```

```text
Running....................................................................................................................
SucceededAKS service creation operation finished, operation "Succeeded"
```

### 3.4 Call the AKS model service
After the deployment, the service can be called with a user ID – the service will then look up the top 10 recommendations for that user in Cosmos DB and send back the results.
The following script demonstrates how to call the recommendation service API and view the result for the given user ID:

```python
import json

scoring_url = aks_service.scoring_uri
service_key = aks_service.get_keys()[0]

input_data = '["{\\"id\\":\\"496\\"}"]'.encode()

req = urllib.request.Request(scoring_url, data=input_data)
req.add_header("Authorization","Bearer {}".format(service_key))
req.add_header("Content-Type","application/json")

with Timer() as t:
    with urllib.request.urlopen(req) as result:
        res = result.read()
        resj = json.loads(
            # Cleanup to parse into a json object
            res.decode("utf-8")
            .replace("\\", "")
            .replace('"', "")
            .replace("'", '"')
        )
        print(json.dumps(resj, indent=4))

print("Full run took %.2f seconds" % t.interval)
```

```text
{
    "MovieId": [
        320,
        1589,
        262,
        1344,
        958,
        889,
        1368,
        645,
        919,
        1137
    ],
    "id": "496",
    "_rid": "34hEAIe9pterAQAAAAAACA==",
    "_self": "dbs/34hEAA==/colls/34hEAIe9ptc=/docs/34hEAIe9pterAQAAAAAACA==/",
    "_etag": "6d006b74-0000-0100-0000-5f25f0550000",
    "_attachments": "attachments/",
    "_ts": 1596321877
}
Full run took 0.05 seconds
```

## Appendix - Realtime scoring with AzureML

In the previous cells, we utilized Cosmos DB to cache the recommendation results for realtime serving. Alternatively, we can generate recommendation results on demand by using the model we deployed. Following scripts load the registered model and use it for recommendation:

* *score_sparkml.py*
    ```
    import json
    import os
    from pyspark.ml.recommendation import ALSModel

    # Note, set `model_name`, `userCol`, and `itemCol` defined earlier.
    model_name = "mvl-als-reco.mml"
    userCol = "UserId"
    itemCol = "MovieId"

    def init(local=False):
        global model

        # Load ALS model.
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_name)
        model = ALSModel.load(model_path)

    def run(input_json):
        js = json.loads(json.loads(input_json)[0])
        id = str(js['id'])
        k = js.get('k', 10)

        # Use the model to get recommendation.
        recs = model.recommendForAllUsers(k)
        recs_topk = recs.withColumn('id', recs[userCol].cast("string")).select(
            'id', "recommendations." + itemCol
        )
        result = recs_topk[recs_topk.id==id].collect()[0].asDict()

        return json.dumps(str(result))
    ```

* Call the AKS model service
    ```
    # Get a recommendation of 10 movies
    input_data = '["{\\"id\\":\\"496\\",\\"k\\":10}"]'.encode()

    req = urllib.request.Request(scoring_url, data=input_data)
    req.add_header("Authorization","Bearer {}".format(service_key))
    req.add_header("Content-Type","application/json")

    ...
    ```

```python

```
