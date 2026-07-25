---
title: "SASRec & SSEPT"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: 6232b154548c955315650d58dca6bf1411c56020
---

> [!note] Original source material
> This page preserves [`examples/00_quick_start/sasrec_amazon.ipynb`](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/examples/00_quick_start/sasrec_amazon.ipynb) from
> *Linux Foundation Recommenders* at commit `6232b154548c955315650d58dca6bf1411c56020`. License:
> [MIT](https://github.com/recommenders-team/recommenders/blob/6232b154548c955315650d58dca6bf1411c56020/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



### Sequential Recommendation Using Transformer

![image.png](/_assets/Sources/Linux Foundation Recommenders/examples/00_quick_start/sasrec_amazon/cell-001-attachment.png)

This is a class of sequential recommendation that uses Transformer \[2\] for encoding the users preference represented in terms of a sequence of items purchased/viewed before. Instead of using CNN (Caser \[3\]) or RNN (GRU \[4\], SLI-Rec \[5\] etc.) the approach relies on Transformer based encoder that generates a new representation of the item sequence. Two variants of this Transformer based approaches are included here,

- Self-Attentive Sequential Recommendation (or SASRec [1]) that is based on vanilla Transformer and models only the item sequence and
- Stochastic Shared Embedding based Personalized Transformer or SSE-PT [6], that also models the users along with the items.

This notebook provides an example of necessary steps to train and test either a SASRec or a SSE-PT model.

```python
%load_ext autoreload
%autoreload 2
```

```text
The autoreload extension is already loaded. To reload it, use:
  %reload_ext autoreload
```

```python
import os
import sys
import pandas as pd
import torch

from recommenders.utils.timer import Timer
from recommenders.datasets.amazon_reviews import get_review_data
from recommenders.datasets.split_utils import filter_k_core
from recommenders.models.sasrec.model import SASREC
from recommenders.models.sasrec.ssept import SSEPT
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.models.sasrec.util import SASRecDataSet
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
```

```text
System version: 3.11.14 (main, Jan 14 2026, 19:35:32) [Clang 21.1.4 ]
PyTorch version: 2.10.0+cu128
CUDA available: True
CUDA version: 12.8
```

### Input Parameters

```python
num_epochs = 5
batch_size = 128
seed = 100  # Set None for non-deterministic result

# data_dir = os.path.join("tests", "recsys_data", "RecSys", "SASRec-tf2", "data")
data_dir = os.path.join("..", "..", "tests", "resources", "deeprec", "sasrec")

# Amazon Electronics Data
dataset = "reviews_Electronics_5"

lr = 0.001             # learning rate
maxlen = 50            # maximum sequence length for each user
num_blocks = 2         # number of transformer blocks
hidden_units = 100     # number of units in the attention calculation
num_heads = 1          # number of attention heads
dropout_rate = 0.1     # dropout rate
l2_emb = 0.0           # L2 regularization coefficient
num_neg_test = 100     # number of negative examples per positive example
model_name = "sasrec"  # "sasrec" or "ssept"
```

```python
reviews_name = dataset + '.json'
outfile = dataset + '.txt'

reviews_file = os.path.join(data_dir, reviews_name)
if not os.path.exists(reviews_file):
    reviews_output = get_review_data(reviews_file)
else:
    reviews_output = os.path.join(data_dir, dataset+".json_output")
```

```python
if not os.path.exists(os.path.join(data_dir, outfile)):
    df = pd.read_csv(reviews_output, sep="\t", names=["userID", "itemID", "time"])
    df = filter_k_core(df, 10)  # filter for users & items with less than 10 interactions

    user_set, item_set = set(df['userID'].unique()), set(df['itemID'].unique())
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1

    df["userID"] = df["userID"].apply(lambda x: user_map[x])
    df["itemID"] = df["itemID"].apply(lambda x: item_map[x])
    df = df.sort_values(by=["userID", "time"])
    df.drop(columns=["time"], inplace=True)
    df.to_csv(os.path.join(data_dir, outfile), sep="\t", header=False, index=False)
```

SASRec requires sequence input and sequence target. Targets are for both positive and negative examples. Inputs to the model are

* user's item history as input to the transformer
* user's item history shifted (by 1) as target to the transformer (positive examples)
* a sequence of items that are not equal to the positive examples (negative examples)

From each user's history three samples are created. If there are $N_u$ items for user-$u$ then $N_u-2$ items are used in training and the last two items are used for validation and testing, respectively.

## Dataset Format

- The input files should have the following format:
    - each row has user-id and item-id converted into integers (starting from 1)
    - the rows are sorted by user-id and time of interaction
    - for every user the last item is used for testing and the last but one is used for validation
    - for example, for user `30449` the sorted inputs are:
        - `30449 2771`
        - `30449 61842`
        - `30449 60293`
        - `30449 30047`
        - `30449 63296`
        - `30449 22042`
        - `30449 6717`
        - `30449 75780`

      then the train inputs are
        - [`2771`, `61842`, `60293`, `30047`, `63296`] (input sequence)
        - [`61842`, `60293`, `30047`, `63296`, `22042`] (target sequence for positive examples)
        - [`1001`, `50490`, `33312`, `19294`, `45342`] (sample negative examples)

      and the validation inputs are
        - [`2771`, `61842`, `60293`, `30047`, `63296`, `22042`] (input sequence)
        - [`61842`, `60293`, `30047`, `63296`, `22042`, `6717`] (target sequence for positive examples)
        - [`4401`, `60351`, `22176`, `23456`, `45342`, '1193`] (sample negative examples)

      and the test inputs are
        - [`2771`, `61842`, `60293`, `30047`, `63296`, `22042`, `6717`] (input sequence)
        - [`61842`, `60293`, `30047`, `63296`, `22042`, `6717`, `75780`] (target sequence for positive examples)
        - [`4401`, `60351`, `22176`, `23456`, `45342`, '1193`, `54231`] (sample negative examples)

```python
inp_file = os.path.join(data_dir, dataset + ".txt")
print(inp_file)

# initiate a dataset class
data = SASRecDataSet(filename=inp_file, col_sep="\t")

# create train, validation and test splits using leave-one-out strategy
# - valid_size: number of items per user for validation
# - test_size: number of items per user for testing
# - min_interactions: minimum interactions for a user to have valid/test splits
# - verbose: print split statistics
split_stats = data.split(valid_size=1, test_size=1, min_interactions=3, verbose=True)
print(split_stats)

# some statistics
num_steps = int(len(data.user_train) / batch_size)
cc = 0.0
for u in data.user_train:
    cc += len(data.user_train[u])
print('average sequence length: %.2f' % (cc / len(data.user_train)))
```

```text
../../tests/resources/deeprec/sasrec/reviews_Electronics_5.txt
Split complete:
  Total users: 36262
  Total items: 35074
  Users with train/valid/test: 36262
  Users with train only: 0
  Valid size per user: 1
  Test size per user: 1
{'num_users': 36262, 'num_items': 35074, 'users_with_splits': 36262, 'users_train_only': 0, 'valid_size': 1, 'test_size': 1, 'min_interactions': 3}
average sequence length: 15.01
```

### Model Creation

Model parameters are

    - number of items
    - maximum sequence length of the user interaction history
    - number of Transformer blocks
    - embedding dimension for item embedding
    - dimension of the attention
    - number of attention heads
    - dropout rate
    - dimension of the convolution layers, list
    - L_2-regularization coefficient

```python
if model_name == 'sasrec':
    model = SASREC(item_num=data.itemnum,
                   seq_max_len=maxlen,
                   num_blocks=num_blocks,
                   embedding_dim=hidden_units,
                   attention_dim=hidden_units,
                   attention_num_heads=num_heads,
                   dropout_rate=dropout_rate,
                   conv_dims = [100, 100],
                   l2_reg=l2_emb,
                   num_neg_test=num_neg_test
    )
elif model_name == "ssept":
    model = SSEPT(item_num=data.itemnum,
                  user_num=data.usernum,
                  seq_max_len=maxlen,
                  num_blocks=num_blocks,
                  # embedding_dim=hidden_units,  # optional
                  user_embedding_dim=10,
                  item_embedding_dim=hidden_units,
                  attention_dim=hidden_units,
                  attention_num_heads=num_heads,
                  dropout_rate=dropout_rate,
                  conv_dims = [110, 110],
                  l2_reg=l2_emb,
                  num_neg_test=num_neg_test
    )
else:
    print(f"Model-{model_name} not found")
```

### Sampler

    - the sampler creates negative samples from the training data for each batch
    - this is done by looking at the original user interaction history and creating items that are not present at all
    - the sampler generates a sequence of negative items of the same length as the original history

```python
sampler = WarpSampler(data.user_train, data.usernum, data.itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=3)
```

### Model Training

    - the loss function is defined over all the negative and positive logits
    - a mask has to be applied to indicate the non-zero items present in the output
    - we also add the regularization loss here

    - having a train-step signature function can speed up the training process

```python
# Train the model
# - val_epoch: evaluate on validation set every N epochs (0 to disable)
# - eval_batch_size: batch size for evaluation (higher = faster)
with Timer() as train_time:
    history = model.train_model(
        data,
        sampler,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        val_epoch=5,  # validate every 5 epochs
        eval_batch_size=256,
        verbose=True
    )

print(f'\nTraining time: {train_time.interval/60.0:.2f} mins')
print(f'Final training loss: {history["loss"][-1]:.4f}')
```

```text
Epoch 1: loss=1.1672
```

```text
Epoch 2: loss=1.0167
```

```text
Epoch 3: loss=0.8861
```

```text
Epoch 4: loss=0.7827
```

```text
Epoch 5: loss=0.6970, val_NDCG@10=0.3659, val_HR@10=0.5700, time=12.9s
Training complete. Total time: 12.9s

Training time: 0.22 mins
Final training loss: 0.6970
```

```python
# Evaluate on test set
# - seed: for reproducibility when sampling users (if > 10000 users)
# - eval_batch_size: batch size for evaluation
print("Evaluating on test set...")
with Timer() as eval_time:
    test_metrics = model.evaluate(data, seed=seed, eval_batch_size=256)

print(f"Evaluation time: {eval_time.interval:.2f}s")
print(f"\nTest Results:")
print(f"  NDCG@10: {test_metrics[0]:.6f}")
print(f"  HR@10:   {test_metrics[1]:.6f}")
```

```text
Evaluating on test set...
```

```text
Evaluation time: 0.47s

Test Results:
  NDCG@10: 0.342169
  HR@10:   0.538600
```

```python
# Record results for tests - ignore this cell
store_metadata("ndcg@10", test_metrics[0])
store_metadata("Hit@10", test_metrics[1])
```

## Reference
\[1\] Wang-Cheng Kang, Julian McAuley: Self-Attentive Sequential Recommendation, arXiv preprint arXiv:1808.09781 (2018) <br>

\[2\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems. 5998–6008 <br>

\[3\] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. ACM, 565–573.

\[4\] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv preprint arXiv:1406.1078. 2014.

\[5\] Zeping Yu, Jianxun Lian, Ahmad Mahmoody, Gongshen Liu, Xing Xie. Adaptive User Modeling with Long and Short-Term Preferences for Personailzed Recommendation. In Proceedings of the 28th International Joint Conferences on Artificial Intelligence, IJCAI’19, Pages 4213-4219. AAAI Press, 2019.

\[6\] Liwei Wu, Shuqing Li, Cho-Jui Hsieh, James Sharpnack. SSE-PT: Sequential Recommendation Via Personalized Transformer. In Fourteenth ACM Conference on Recommender Systems, RecSys'20:, Pages 328–337, 2020.
