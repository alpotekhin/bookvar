---
title: BERT, RoBERTa, and DeBERTa
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "7ae01f5ceabee591297dad10b1db720f2d9e4a1d96e1b01040239da6acf8e712"
translation_reviewed: "2026-11-21"
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md"
last_updated: 2026-11-21
primary_sources:
  - https://aclanthology.org/N19-1423/
  - https://arxiv.org/abs/1907.11692v1
  - https://arxiv.org/abs/2006.03654v6
  - https://aclanthology.org/2020.tacl-1.5/
  - https://arxiv.org/abs/1909.11942v6
  - https://arxiv.org/abs/2003.10555v1
  - https://arxiv.org/abs/2111.09543v4
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей]]"
---

# BERT, RoBERTa, and DeBERTa

<a id="turning-two-segments-into-one-input"></a>

## How BERT combines two segments

In Figure 2 of the BERT paper, two segments form one input.
The word `playing` occupies two positions, `play` and `##ing`.

```text
[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
```

`[CLS]` starts the input. The first `[SEP]` closes segment A and the
second closes B. Positions run from 0 to 10 across both segments.
The paper calls A and B *sentences*, though either may contain more
than one sentence of text.

Each row adds three learned vectors:

$$
\begin{aligned}
e_i&=E_{\mathrm{token}}[x_i]\\
&\quad+E_{\mathrm{position}}[i]+E_{\mathrm{segment}}[s_i].
\end{aligned}
$$

At position 4, BERT adds the embedding of `cute`, the embedding of
position 4, and the type A embedding. At positions 8 and 9, `play` and
`##ing` have different vocabulary and position rows but share type B.
The type tells the encoder which segment a token belongs to. Tokens in the two segments can still attend to each other across `[SEP]`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure2.png|Each BERT input row sums token, segment type, and position embeddings]]

*The three summands of the input representation. `play` and `##ing`
use different token and position rows but the same type B.
Source: Jacob Devlin et al., [BERT, Figure 2](https://aclanthology.org/N19-1423.pdf#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The diagram and English caption are unchanged.*

The official code applies LayerNorm and dropout to this sum before the
stack ([`embedding_postprocessor`](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L428-L521)).

<a id="architecture-and-tensor-shapes"></a>

### Encoder blocks and tensor shapes

`BERT_BASE` has 12 blocks, width $d=768$, and 12 heads. Each head has
width $768/12=64$, and the FFN intermediate width is 3072.
`BERT_LARGE` has 24 blocks, $d=1024$, 16 heads, and FFN width 4096.
The authors report approximately 110 and 340 million parameters
respectively ([paper, §3](https://aclanthology.org/N19-1423/)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-base-bert-large-encoders.png|BERT BASE and LARGE stacks have 12 and 24 bidirectional blocks]]

*Stacks of 12 and 24 encoder blocks. Source: Jay Alammar,
[The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The original figure is unchanged.*

Each block applies self-attention followed by an FFN with GELU.
In original BERT, normalization follows residual addition:

$$
U=\operatorname{LN}(H+\operatorname{Dropout}(\operatorname{MHA}(H))),
$$

$$
H'=\operatorname{LN}(U+\operatorname{Dropout}(\operatorname{FFN}(U))).
$$

This is the original Transformer’s post-norm layout. BERT keeps one
encoder stack. Within each block the same heads read both segments,
with PAD keys masked and right context available.

For a batch of $B$ examples padded to length $T$, shapes are:

| Quantity | Shape |
|---|---|
| Token and type IDs, mask | `[B,T]` |
| States at each block | `[B,T,d]` |
| Q, K, V split into heads | `[B,h,T,d_h]` |
| Attention scores | `[B,h,T,T]` |
| MLM logits at `m` positions | `[m,V]` |

For `BERT_BASE` on this pair, Q, K, and V have shape `[1,12,11,64]`.

Padding this example to 12 positions leaves `B` at 1 and increases `T`
from 11 to 12. Each attention layer masks PAD keys. The encoder still
computes an output for the PAD query; neither the MLM readout nor
classification from `[CLS]` reads that row.

<a id="mlm-obtaining-a-bidirectional-training-signal"></a>

## MLM: recovering a selected subword

Suppose the preprocessor selects position 4 and replaces `cute` with
`[MASK]`. The model has `my dog is` on the left and `he likes play ##ing`
on the right. The encoder computes a state at position 4, and the output
layer maps it to vocabulary scores. The correct class is the ID for `cute`.
This task is called *masked language modeling* (MLM). The
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|previous chapter]]
compares it with predicting the next token.

BERT selects about 15% of the WordPiece positions. The preprocessor
leaves `[CLS]` and `[SEP]` out of the candidate set
([BERT, §3.1 and Appendix A.1](https://aclanthology.org/N19-1423/)).
At a selected position it inserts `[MASK]` 80% of the time, a random
token 10% of the time, and leaves the original token there 10% of the
time. In all three cases the target is the original ID. So `cute` can
contribute to the loss even when it remains visible.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png|The state at the masked position passes to the MLM vocabulary head]]

*The selected position's state becomes a vocabulary distribution.
“All English words” in the original figure is a simplification: actual
outputs index a WordPiece vocabulary, including subwords and special tokens.
Source: Jay Alammar, [The Illustrated BERT, Masked Language Model](https://jalammar.github.io/illustrated-bert/#masked-language-model),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Bidirectional attention builds $h_4$ from the whole input after replacement. The MLM head
passes that state through a dense layer, GELU, and LayerNorm:

$$
\begin{aligned}
z_i&=\operatorname{LN}
\bigl(\operatorname{GELU}(h_iW_m+b_m)\bigr),\\
\ell_i&=z_iE_{\mathrm{token}}^\top+b_{\mathrm{vocab}}.
\end{aligned}
$$

The `cute` logit is the dot product of the transformed $z_4$ and the
same table row BERT would have looked up for `cute` at the input,
plus the vocabulary bias. The other logits use the other rows.
Source: [official `get_masked_lm_output`](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_pretraining.py#L240-L282).

Let $\mathcal M$ be selected positions and $\tilde x$ the corrupted input.
Mean MLM loss is

$$
L_{\mathrm{MLM}}=
-\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).
$$

With $\mathcal M=\{4\}$ the average is just $-\log p_\theta(\text{cute}\mid\tilde x)$. There is no separate loss term for `dog`, although position 4 can attend to it.

<details id="bert-mlm-replacement-ablation">
<summary>Why BERT does not always use MASK</summary>

The mixture of replacements reduces mismatch with downstream use,
where `[MASK]` is usually absent. Appendix C.2 separately compares
80/10/10 with 100% `[MASK]`. Full fine-tuning is relatively robust to these
variants, while the frozen-feature NER experiment gives 94.0 Dev F1 for
all-MASK versus 94.9 for 80/10/10. This is a result for a particular
experiment, not proof that this proportion is optimal for all models.

</details>

<a id="berts-second-objective-segment-relationships"></a>

## NSP: where the second segment came from

If `he likes playing` follows `my dog is cute` in the document, the NSP
label is 0. If the preprocessor randomly samples the second segment,
for example from a passage about penguins, the label is 1. This is
*next sentence prediction* (NSP). The paper describes a 50–50 split. In the
[official generator](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L234-L273),
the random branch is mandatory when a chunk has only one segment
to offer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-next-sentence-prediction.png|The encoder reads both segments and the CLS output classifies the source of the second]]

*The pair enters one bidirectional encoder, and `[CLS]` supports pair
classification. The middle Tokenized Input row shows only the first
segment; actual input includes both segments and separators, as in the
bottom row.
Source: Jay Alammar, [The Illustrated BERT, Two-sentence Tasks](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The final `[CLS]` state in the same pair passes through a dense layer and `tanh`, together called the pooler. Its output goes to a classifier with two logits, one for a genuine continuation and one for a random segment.

Each step adds mean loss over selected MLM tokens to mean NSP loss
over batch examples:

$$
L=L_{\mathrm{MLM}}+L_{\mathrm{NSP}}.
$$

Summing unnormalized losses for all MLM positions with one NSP loss
would be a different normalization: the objective ratio would then
change with text length. Original code normalizes MLM by the sum of
valid-target weights and adds $10^{-5}$ to the denominator. Weights of zero keep target padding out of the mean.

<a id="one-executable-pre-training-step"></a>

### Losses and gradients for one pair

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure1.png|A shared encoder in pre-training and separate copies for supervised tasks]]

*On the left, a shared encoder receives pair A/B: MLM outputs reconstruct
selected positions, while NSP reads `[CLS]`. On the right, the same
pretrained parameters initialize separate supervised-task models.
Arrows between the two sides represent parameter transfer, not the
transfer of current states from one task to another.
Source: Jacob Devlin et al., [BERT, Figure 1](https://aclanthology.org/N19-1423.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Both panels and the original English caption are unchanged.*

The run below has one input and one selected position. Before the
update, MLM loss is $14.243288$ and NSP loss is $0.685105$. The gradient
of their sum, $14.928394$, updates the shared encoder. These are losses
from random initialization; they do not measure how well a trained
BERT can recover words.

To check the unselected `dog` position's contribution specifically to
MLM, the code differentiates MLM loss with respect to intermediate
input vectors. Element `[0,2]` is `dog`: it is not a target, but
attention lets its representation affect reconstruction of `cute`.
NSP loss is not included in this measurement. The derivative is taken
with respect to the position's input vector, not a weight row in the
shared input/output table. The input PAD vector receives zero MLM
gradient because it cannot be read as a key at any layer. In this run, the input `dog` gradient norm is
approximately $0.118682$, and PAD's is zero.

The shared table `word.weight` has another role: its rows also participate
in vocabulary softmax. Nonzero `word.weight.grad[5]` alone would
therefore not demonstrate use of input `dog`. Even the PAD row receives
a gradient through the output classifier: the target is `cute`, but
probabilities normalize over all 12 classes, including PAD. The input
attention mask does not remove this output class.

<details id="bert-pretraining-code">
<summary>Code for the pre-training step</summary>

The code below passes a corrupted pair through a small post-norm encoder,
a weight-tied MLM head, and an NSP head, computes both losses, and updates
weights. It uses PyTorch's attention; attention-head operations are
explained in the
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Transformer chapter]].

There are 12 toy vocabulary tokens, two width-16 layers, and four heads.
Selection of `cute` is specified in advance, not produced by a real
WordPiece tokenizer. Dropout is disabled for reproducibility. This is
a computation example, not a reproduction of published BERT training.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(31)
torch.set_num_threads(1)

class TinyBertPretraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.word = nn.Embedding(12, 16)
        self.position = nn.Embedding(32, 16)
        self.segment = nn.Embedding(2, 16)
        self.input_norm = nn.LayerNorm(16, eps=1e-12)
        activation = lambda x: F.gelu(x, approximate="tanh")
        block = nn.TransformerEncoderLayer(
            16, 4, dim_feedforward=64, dropout=0.0,
            activation=activation, layer_norm_eps=1e-12,
            batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(
            block, 2, enable_nested_tensor=False)
        self.mlm_transform = nn.Sequential(
            nn.Linear(16, 16), nn.GELU(approximate="tanh"),
            nn.LayerNorm(16, eps=1e-12))
        self.vocab_bias = nn.Parameter(torch.zeros(12))
        self.pooler = nn.Linear(16, 16)
        self.nsp = nn.Linear(16, 2)

    def forward(self, ids, types, valid, chosen):
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.input_norm(
            self.word(ids)+self.position(pos)[None]+self.segment(types))
        h = self.encoder(h, src_key_padding_mask=~valid)
        z = self.mlm_transform(h[chosen])
        mlm_logits = F.linear(z, self.word.weight, self.vocab_bias)
        nsp_logits = self.nsp(torch.tanh(self.pooler(h[:, 0])))
        return h, mlm_logits, nsp_logits

# PAD=0, CLS=1, SEP=2, MASK=3, my=4, dog=5, is=6, cute=7,
# he=8, likes=9, play=10, ##ing=11.
original = torch.tensor([[1, 4, 5, 6, 7, 2, 8, 9, 10, 11, 2, 0]])
types = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]])
valid = original.ne(0)
chosen = torch.zeros_like(valid)
chosen[0, 4] = True
ids = original.clone()
ids[chosen] = 3
targets = original[chosen]         # cute, not MASK
nsp_target = torch.tensor([0])

model = TinyBertPretraining().double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
word_inputs = []
def save_word_inputs(_module, _args, output):
    word_inputs.append(output)
hook = model.word.register_forward_hook(save_word_inputs)
h, mlm_logits, nsp_logits = model(ids, types, valid, chosen)
hook.remove()
assert h.shape == (1, 12, 16) and mlm_logits.shape == (1, 12)
assert nsp_logits.shape == (1, 2)
mlm = F.cross_entropy(mlm_logits, targets, reduction="sum")/(chosen.sum()+1e-5)
nsp = F.cross_entropy(nsp_logits, nsp_target)
loss = mlm+nsp
# Differentiate MLM alone with respect to the input embedding vectors.
mlm_input_grad, = torch.autograd.grad(mlm, word_inputs[0], retain_graph=True)
assert mlm_input_grad[0, 2].norm() > 0  # unchosen dog is used as context
assert torch.count_nonzero(mlm_input_grad[~valid]) == 0
optimizer.zero_grad()
before = model.word.weight.detach().clone()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(model.word.weight.grad).all()
assert model.word.weight.grad[0].norm() > 0  # PAD remains an output class
optimizer.step()
assert not torch.equal(before, model.word.weight.detach())
print("MLM", float(mlm.detach()), "NSP", float(nsp.detach()))
print("dog input MLM gradient norm", float(mlm_input_grad[0, 2].norm()))
print("PAD input MLM gradient norm", float(mlm_input_grad[~valid].norm()))
print("hidden/MLM/NSP", tuple(h.shape),
      tuple(mlm_logits.shape), tuple(nsp_logits.shape))
```

</details>

<details id="bert-original-recipe">
<summary>BERT’s corpus and training settings</summary>

<a id="bert-data-and-recipe"></a>

The paper uses BooksCorpus, approximately 800 million words, and
English Wikipedia, approximately 2.5 billion words. Wikipedia provides
contiguous passages, excluding lists, headings, and tables. Preserved
document boundaries allow genuine continuations and long context to
be selected.

Appendix A.2 gives this recipe:

| Parameter | Original training |
|---|---|
| updates | 1 million |
| batch | 256 sequences |
| learning rate | 10,000-step warmup to $10^{-4}$, then linear decay |
| Adam | $\beta_1=0.9,\ \beta_2=0.999$ |
| weight decay | 0.01 |
| dropout | 0.1 |
| length | 128 for the first 90% of steps, 512 for the last 10% |

Switching to 512 trains distant absolute positions without paying for
long attention at every early step. Multiplying the number of processed
sequences by 512 does not give the actual number of content tokens:
part of training uses length 128, examples can be shorter, and padding
may be present.

BASE trained for four days on four Cloud TPUs in a Pod configuration
(16 TPU chips); LARGE used 16 Cloud TPUs (64 chips). This describes the
hardware of the 2018 experiment, not a promised training time on a
modern accelerator.

</details>

<a id="один-encoder--разные-способы-чтения-выхода"></a>

<a id="fine-tuning-on-a-labeled-task"></a>

## Fine-tuning BERT on labeled data

Fine-tuning starts from the pre-trained tokenizer, input tables, and
encoder weights. Each task gets its own copy of those weights and a
new output head, trained on labeled examples.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/bert-figure4.png|Four BERT readouts, including separate start and end positions for an answer]]

*Single-text and pair classification above; answer extraction and NER below.
Source: Jacob Devlin et al., [BERT, Figure 4](https://aclanthology.org/N19-1423.pdf#page=15),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
All four panels and the original caption are preserved.*

Text classification reads the `[CLS]` output. Named entity recognition
needs a label per word. If WordPiece splits a word into several tokens,
the BERT experiment reads the first subword’s state.

<a id="extractive-qa-two-projections-and-a-valid-span"></a>

### Extractive QA: finding both boundaries

Encode `Where does Marie live?` and `Marie lives in Paris` as
`[CLS] question [SEP] passage [SEP]`. The same bidirectional encoder
reads the two segments together. Two learned vectors $s,e$ replace
the vocabulary head. They give each state $h_i$ a start and end logit,
$\ell_i^{\mathrm{start}}=s^\top h_i$ and
$\ell_i^{\mathrm{end}}=e^\top h_i$. The labels are positions in the
input. The vocabulary ID of `Paris` is not a target.

Independent start and end maxima may define an impossible span.
Take start logits
$(0,0,1,3)$ and end logits $(4,0,0,2)$. Independent selection starts at
`Paris` and ends at `Marie`. Under $j\ge i$, the highest sum is
$3+2=5$, selecting the single-token answer `Paris`.

Training uses two separate softmax distributions. With the four
logits above, the probabilities assigned to the correct start and
end are about $0.8098$ and $0.1155$. The two cross-entropies sum to
$2.369681$. The paper describes a sum;
[BERT's released training code](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_squad.py#L595-L610)
averages the two losses and normalizes each distribution across the
whole input. The average for the example's four positions would be
$1.184840$. A candidate span $(i,j)$ is not one class in this loss.

At inference the
[same code](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_squad.py#L703-L776)
filters out starts and ends outside the passage, incompatible pairs,
and overlong spans. It splits long documents into overlapping windows
and maps window positions back to the original text. The example
above has only four passage positions and the constraint $j\ge i$.

In SQuAD 2.0, some questions have no answer. Original BERT assigns
`[CLS]` as both their correct start and end. At decoding, the best
nonempty span is compared with the summed `[CLS]` start/end scores
using a threshold selected on dev
([BERT, §§4.2–4.3](https://aclanthology.org/N19-1423/)).

<details id="bert-span-code">
<summary>Code for selecting a valid span</summary>

```python
import torch
import torch.nn.functional as F

words = ["Marie", "lives", "in", "Paris"]
start = torch.tensor([0., 0., 1., 3.], requires_grad=True)
end = torch.tensor([4., 0., 0., 2.], requires_grad=True)
i, j = torch.arange(4)[:, None], torch.arange(4)[None, :]
scores = (start[:, None]+end[None, :]).masked_fill(j<i, -torch.inf)
flat = int(scores.argmax())
left, right = divmod(flat, 4)
assert (int(start.argmax()), int(end.argmax())) == (3, 0)
assert (left, right) == (3, 3)
loss = (F.cross_entropy(start[None], torch.tensor([3]))+
        F.cross_entropy(end[None], torch.tensor([3])))
loss.backward()
print("answer:", " ".join(words[left:right+1]), "loss:", float(loss.detach()))
```

</details>

<details id="bert-features-and-evaluation">
<summary>Frozen features and BERT results</summary>

<a id="frozen-features-or-full-fine-tuning"></a>

<h3>Frozen features or full fine-tuning</h3>

BERT can be frozen while a separate model is trained on its states.
Features for unchanged inputs can then be computed in advance. The
last state at each position is not the only option: the paper compares
several layers and their combinations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png|CoNLL-2003 dev F1 for features taken from different frozen BERT layers]]

*Feature-extraction comparisons for CoNLL-2003 NER.
Source: Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
based on [BERT, §5.3, Table 7](https://aclanthology.org/N19-1423/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

With BERT frozen, concatenating the last four layers gave 96.1 Dev F1;
the last layer alone gave 94.9. A two-layer BiLSTM and classifier were
trained on top in that experiment. Full BASE fine-tuning gave 96.4
Dev F1. These numbers do not establish that the last four layers always
beat the last: they compare particular features, models, and an NER dataset.

Nor does `[CLS]` become a retrieval vector by definition. Independent
document representations need a learned similarity geometry; a joint
pair classifier and a bi-encoder solve different tasks.

<a id="reading-berts-results"></a>

<h3>Ablating NSP and attention direction</h3>

In Table 5, BASE reaches 88.5 SQuAD 1.1 Dev F1. Removing NSP while
retaining MLM gives 87.9; switching to LTR without NSP gives 77.8.
Adding a BiLSTM to LTR raises this to 84.9. Comparing 88.5 with 77.8
changes two parts of pre-training at once; to examine LTR, the authors
compare no-NSP MLM against no-NSP LTR. This is not a numerical estimate
of one mask's effect across all possible recipes.

Leaderboard results use another protocol. SQuAD 1.1 Test F1 of 91.8
belongs to one LARGE model with preliminary TriviaQA fine-tuning;
the table gives 90.9 Dev F1 for LARGE without TriviaQA.
One LARGE model obtains 83.1 Test F1 on SQuAD 2.0. These values should
not share an unlabeled column: dataset, external data, and ensembling
must be identified.

On small labeled datasets, fine-tuning depends on head initialization,
data order, and hyperparameters. BERT selects runs using dev results;
RoBERTa's replication tables report medians of five runs. These
protocols belong beside the number; the best run should not be
presented as expected performance.

Pre-training transfers corpus properties, including errors and social
biases. A mean benchmark result does not establish robustness to a
new domain or the correctness of every prediction.

</details>

<a id="roberta-когда-новый-результат-даёт-не-новая-архитектура"></a>

<a id="roberta-changes-to-training"></a>

## RoBERTa: masking, data, and optimization

The RoBERTa experiments retain the BERT encoder and its masked-token
task. Liu et al. first vary how they prepare and train examples at
BASE size. For LARGE they also use a larger corpus and train longer
([RoBERTa, §§3–5](https://arxiv.org/pdf/1907.11692v1)).

<a id="static-and-dynamic-masking"></a>

### Precomputed BERT masks and fresh RoBERTa masks

With static masking, the preprocessor might write a copy of this pair
with `[MASK]` in place of `cute` and `cute` as its target. The next epoch
reads the same record. BERT's data preparation makes ten copies with
different replacements rather than fixing one pattern for each text.
Over the 40 epochs described in RoBERTa, each variant appears about
four times.

RoBERTa draws the positions again whenever it presents a sequence.
On the next pass `dog` might be hidden and `cute` might appear in the
context. The encoder still predicts selected subword IDs, and its
attention still reads right context.

| Masking | SQuAD 2.0 | MNLI-m | SST-2 |
|---|---:|---:|---:|
| static masking | 78.3 | 84.3 | 92.5 |
| dynamic masking | 78.7 | 84.0 | 92.9 |

Source: [RoBERTa, Table 1](https://arxiv.org/pdf/1907.11692v1#page=4).
The table gives F1 on SQuAD 2.0 and accuracy on MNLI-m and SST-2. Scores are medians over five runs. The effect is small and not uniformly
positive across the three tasks; dynamic masking alone cannot explain
RoBERTa's later overall gain.

<a id="nsp-and-text-assembly"></a>

### Assembling inputs without NSP

Dropping NSP removes the need to sample a random second segment.
The space can instead hold more consecutive sentences. Section 4.2
compares four ways to assemble an input:

| Format | Texts included in one input | NSP |
|---|---|---|
| SEGMENT-PAIR | two segments, each possibly containing several sentences | yes |
| SENTENCE-PAIR | two individual sentences | yes |
| FULL-SENTENCES | whole sentences filling input up to 512; a separator permits crossing into another document | no |
| DOC-SENTENCES | whole sentences from one document only; its end may produce a short input | no |

A short pair of individual sentences offers fewer long-range dependencies
than long contiguous segments. The authors increase the short-example
batch to roughly preserve token count, but SENTENCE-PAIR still underperforms
SEGMENT-PAIR: 76.2 versus 78.7 F1 on SQuAD 2.0.

FULL-SENTENCES gives 79.1 SQuAD 2.0 Dev F1; DOC-SENTENCES gives 79.7;
the SEGMENT-PAIR replication with NSP gives 78.7. Later experiments
choose FULL-SENTENCES because restricting inputs to one document causes
more variation in length and examples per batch.

Both NSP and input organization change here. The table shows that this
recipe can work without NSP. It does not prove that adding any NSP
objective to any training setup must hurt. Comparison with BERT's
ablation requires checking which inputs remain after removing the loss.

<a id="batch-size-update-count-and-vocabulary"></a>

### Batch size, update count, and vocabulary size

More examples per update can distribute computation more efficiently.
But one million steps with batch 256 and one million with batch 8192
represent different amounts of presented data. In the controlled
Section 4.3 experiment, step count falls as batch size rises, and
learning rate is tuned separately:

| Batch size | Steps | Peak learning rate | MNLI-m |
|---|---:|---:|---:|
| 256 | 1 million | $10^{-4}$ | 84.7 |
| 2K | 125,000 | $7\cdot10^{-4}$ | 85.2 |
| 8K | 31,000 | $10^{-3}$ | 84.6 |

Source: [RoBERTa, Table 3](https://arxiv.org/pdf/1907.11692v1#page=6).
Similar numbers of example presentations do not make optimization
trajectories identical. Here the best MNLI variant is 2K, not the largest
batch. The final recipe uses 8K for large-scale training.

RoBERTa also replaces BERT's WordPiece with byte-level BPE of roughly
50,000 entries. A base byte alphabet represents input without `[UNK]`
caused by a missing character, but does not remove the model's length
limit. The larger vocabulary adds about 15 million token-table parameters
to BASE and 20 million to LARGE while preserving the main blocks.
Section 4.4 reports small declines on some tasks from the new tokenization;
it does not present tokenization as an independent explanation for all gains.

<a id="the-final-recipe-and-results"></a>

### 160 GB of text and half a million steps

CC-NEWS, OpenWebText, and STORIES supplement the original 16 GB of
BooksCorpus/Wikipedia, bringing the corpus above 160 GB. RoBERTa omits
BERT's initial length-128 phase and constructs inputs up to length 512.
Training duration and the number of example presentations also increase.

| LARGE data and steps, batch 8K | SQuAD 2.0 Dev F1 | MNLI-m Dev accuracy |
|---|---:|---:|
| 16 GB, 100,000 | 87.3 | 89.0 |
| 160 GB, 100,000 | 87.7 | 89.3 |
| 160 GB, 300,000 | 88.7 | 90.0 |
| 160 GB, 500,000 | 89.4 | 90.2 |

Source: [RoBERTa, Table 4](https://arxiv.org/pdf/1907.11692v1#page=7).
The rows accumulate changes. Moving from 16 to 160 GB changes corpus
composition as well as size; the authors explicitly leave separating
these factors to future work.

The often-cited GLUE 88.5 belongs to a test-set ensemble, not one model
in the dev table. Fine-tuning also changes for some tasks: RTE/STS/MRPC
start from an MNLI model, and leaderboard QNLI is reformulated as ranking.
SQuAD 2.0 Dev F1 of 89.4 and RACE Test accuracy of 83.2 follow other
protocols specified in the paper.

<a id="deberta-separate-content-from-relative-position"></a>

## DeBERTa: content and distance in an attention score

At position 4 in the BERT example, the input sums token, position,
and segment embeddings. Q and K are projected from that sum.
DeBERTa can consult a separate row for the distance from position 4
to `dog` at position 2: $4-2=2$. The key `he` at position 6 uses a
different row, for $4-6=-2$. These indices are for comparison with
the BERT example, not a run of the DeBERTa tokenizer.

In the original DeBERTa design, the encoder uses a relative-position
table $P$ rather than absolute positions at its input. It projects
content states $H$ and the position table separately. Each token pair
selects one positional row for each direction.

For one head:

$$
\begin{aligned}
Q^c&=HW_{q,c},\\
K^c&=HW_{k,c},\\
V^c&=HW_{v,c}.
\end{aligned}
$$

$$
Q^r=PW_{q,r},\qquad K^r=PW_{k,r}.
$$

Vectors are rows. If $H$ has shape $[T,d]$ and $P$ has shape $[2k,d]$,
the five projection matrices have shape $[d,d_h]$. Here $k$ limits
distinguishable distances, not attention length.

<a id="three-terms-in-one-score"></a>

### Three dot products for one pair

Following the convention in the
[paper, §3.1, Equations 3–4](https://arxiv.org/pdf/2006.03654v6#page=3),
the relative-row index is

$$
\delta(i,j)=\operatorname{clip}(i-j+k,0,2k-1).
$$

The score connecting query $i$ to key $j$ is

$$
\begin{aligned}
\tilde A_{ij}
&=Q_i^cK_j^{c\top}\\
&\quad+Q_i^cK_{\delta(i,j)}^{r\top}\\
&\quad+K_j^cQ_{\delta(j,i)}^{r\top}.
\end{aligned}
$$

The first product compares content. The second lets a query prefer a
relative key position: `learning` may score a nearby word on the left
differently from a distant word on the right. The third compares key content with the query’s location relative to that key, so it uses the reverse index $\delta(j,i)$. The authors omit the position-to-position term
from the general decomposition.

Take $k=2$, $i=1$, $j=2$, and $d_h=2$. Then $\delta(i,j)=1$ and
$\delta(j,i)=3$. Let $Q_i^c=(1,2)$, $K_j^c=(3,1)$, $K_1^r=(0,1)$,
and $Q_3^r=(1,-1)$. The products are $5$, $2$, and $2$. Before softmax,
their sum is scaled by $\sqrt{3d_h}$:
$9/\sqrt6\approx3.674235$. This is not yet an attention weight; scores
for the row's other keys are required.

The example uses three keys. Q, K, V, and the relative vectors are already projected, so only the calculation inside one head remains.

<details id="deberta-attention-code">
<summary>Code for the three-key calculation</summary>

```python
import math
import torch

q = torch.tensor([[1., 0.], [1., 2.], [0., 1.]], dtype=torch.float64)
k = torch.tensor([[1., 0.], [0., 1.], [3., 1.]], dtype=torch.float64)
v = torch.tensor([[1., 0.], [0., 1.], [2., 2.]], dtype=torch.float64)
kr = torch.tensor([[2., 0.], [0., 1.], [1., 0.], [0., 2.]],
                  dtype=torch.float64)
qr = torch.tensor([[0., 1.], [1., 0.], [0., 2.], [1., -1.]],
                  dtype=torch.float64)

def disentangled(q, k, v, kr, qr):
    distance = len(kr)//2
    i, j = torch.arange(len(q))[:, None], torch.arange(len(k))[None, :]
    forward = (i-j+distance).clamp(0, 2*distance-1)
    reverse = (j-i+distance).clamp(0, 2*distance-1)
    c2c = q @ k.T
    c2p = (q[:, None, :]*kr[forward]).sum(-1)
    p2c = (k[None, :, :]*qr[reverse]).sum(-1)
    scores = (c2c+c2p+p2c)/math.sqrt(3*q.shape[-1])
    weights = scores.softmax(-1)
    return weights @ v, weights, (c2c, c2p, p2c)

output, weights, parts = disentangled(q, k, v, kr, qr)
assert [float(x[1, 2]) for x in parts] == [5., 2., 2.]
assert [float(x[1, 0]) for x in parts] == [1., 4., 1.]
torch.testing.assert_close(sum(x[1] for x in parts),
                           torch.tensor([6., 5., 9.], dtype=q.dtype))
print("scores:", sum(x[1] for x in parts)/math.sqrt(6))
print("weights:", weights[1], "output:", output[1])
```

</details>

At row $i=1$, sums of the three terms are $(6,5,9)$. Scaling and softmax
give weights approximately $(0.1973,0.1312,0.6715)$. With values
$(1,0)$, $(0,1)$, and $(2,2)$, the output is approximately
$(1.5403,1.4742)$. The third key gets the largest weight, but both
others still contribute.

An out-of-range distance uses a boundary table row. The distant key
remains accessible: clipping is not a local-attention mask. Permuting
tokens can change $i-j$ even when token vectors themselves are unchanged.

Materializing a separate relative vector for every pair would require
$O(T^2d_h)$ intermediate memory. Algorithm 1 first computes products
with the shared table of size $2k$, then selects scores by distance
indices. Relative tables need $O(kd_h)$ memory, but the final score
matrix is still $T\times T$. DeBERTa does not make full attention linear
in sequence length.

<a id="enhanced-mask-decoder"></a>

### Enhanced Mask Decoder: absolute positions before MLM

After the encoder reads the input, the MLM head still has a blank to
fill. DeBERTa supplies absolute positions at this stage. In the
authors' example, `a new [MASK] opened beside the new [MASK]`, the
first blank is in the subject and the second in the phrase headed by `beside`.
The fact that each follows `new` does not distinguish them, though
their full relative contexts differ.

In [Figure 2](https://arxiv.org/pdf/2006.03654v6#page=20) and Appendix A.8,
EMD has two inputs: $H$ contains contextual K/V states, and $I$ supplies
query states. After one pass, updated $I$ reads the same memory $H$
again. The paper's experiment uses two passes with shared parameters,
followed by the MLM head.

With `position_biased_input=False`, the released code initializes its queries as
$H+P_{\rm abs}$, adding the position vectors to the states $H$ from
the penultimate layer. Both reads of $H$ use the final block's weights.

We can trace two reads of fixed memory numerically. Keep only ordinary
attention and a residual connection: $Q=I$, $K=V=H$, with identity
projections. Relative terms, LayerNorm, and the FFN are disabled in this
small example. Memory has rows $(1,0)$ and $(0,1)$. Position vectors are
$(0,0)$ and $(1,0)$, so initial queries $I^{(0)}=H+P$ are $(1,0)$ and
$(1,1)$. Each pass applies the same transformation:

$$
\begin{aligned}
A^{(r)}&=\operatorname{softmax}
\left(\frac{I^{(r)}H^\top}{\sqrt2}\right),\\
I^{(r+1)}&=I^{(r)}+A^{(r)}H.
\end{aligned}
$$

The first query gives scores $(1/\sqrt2,0)$ and weights approximately
$(0.669762,0.330238)$. Adding the read vector to the query gives the first
row of $I^{(1)}$ as $1.669762$ and $0.330238$. On the second read, the
score difference changes; weights become approximately
$(0.720549,0.279451)$. The first row of $I^{(2)}$ becomes $2.390310$ and
$0.609690$. K/V remain unchanged in both passes.

<details id="deberta-emd-code">
<summary>Code for both reads of the fixed memory</summary>

```python
import math
import torch

memory = torch.eye(2, dtype=torch.float64)
positions = torch.tensor([[0., 0.], [1., 0.]], dtype=memory.dtype)
queries = memory+positions
history, attention_weights = [], []
for _ in range(2):
    weights = (queries @ memory.T/math.sqrt(2)).softmax(-1)
    queries = queries+weights @ memory
    attention_weights.append(weights)
    history.append(queries.clone())

torch.testing.assert_close(history[0][0], torch.tensor(
    [1.669761549, 0.330238451], dtype=memory.dtype))
torch.testing.assert_close(history[1][0], torch.tensor(
    [2.390310430, 0.609689570], dtype=memory.dtype))
torch.testing.assert_close(history[1][1], torch.tensor(
    [2., 2.], dtype=memory.dtype))
print("first read", history[0], "second read", history[1])
```

</details>

The first read changed the query, so the second read has different
weights even though the keys and values stay fixed. In DeBERTa, the
final encoder block performs both reads with learned projections,
relative scores, normalization, and the FFN. The code computes the
query states and gathers the prediction positions for the vocabulary
head. This decoder fills blanks in a known sequence without generating
a new sequence.

The [authors’ ICLR 2021 slides](https://iclr.cc/media/Slides/iclr/2021/virtual%2804-00-00%29-04-00-00UTC-2562-deberta_decodi.pdf) show the same two inputs.

<details id="deberta-original-emd">
<summary>Microsoft’s implementation</summary>

The following original excerpt preserves its operators and language
([Microsoft DeBERTa, `emd_context_layer`](https://github.com/microsoft/DeBERTa/blob/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098/DeBERTa/apps/models/masked_language_model.py#L51-L71)).
Copyright Microsoft Corporation; [full MIT license](/licenses/microsoft-deberta-MIT.txt).

```python title="Original implementation — MIT"
    hidden_states = encoder_layers[-2]
    if not self.position_biased_input:
      layers = [encoder.layer[-1] for _ in range(2)]
      z_states +=  hidden_states
      query_states = z_states
      query_mask = attention_mask
      outputs = []
      rel_embeddings = encoder.get_rel_embedding()

      for layer in layers:
        # TODO: pass relative pos ids
        output = layer(hidden_states, query_mask, return_att=False, query_states = query_states, relative_pos=relative_pos, rel_embeddings = rel_embeddings)
        query_states = output
        outputs.append(query_states)
```

</details>

<details id="deberta-setting-and-versions">
<summary>DeBERTa results and changes in V2</summary>

<a id="fine-tuning-results-and-versions"></a>

SiFT in Section 4 is fine-tuning regularization, not part of the three
attention terms. The model compares answer distributions for the
original example and a small perturbation of its input representations.
SiFT perturbs normalized vectors, reducing dependence of perturbation
scale on the original embedding norm. The authors apply SiFT to the
1.5B SuperGLUE model; this description must not be assigned to every
original DeBERTa checkpoint.

The main LARGE configuration has 24 layers, width 1024, and 16 heads.
After deduplication it uses about 78 GB of Wikipedia, BooksCorpus,
OpenWebText, and STORIES. In Table 4 of paper version v6, removing EMD
changes SQuAD 2.0 Dev F1 from 82.5 to 81.3 and RACE accuracy from
71.7 to 70.3. This is a BASE ablation, not a LARGE result or a measurement
of one EMD pass.

For LARGE, the authors report Avg 90.00 in
[GLUE dev Table 1](https://arxiv.org/pdf/2006.03654v6#page=6), plus 90.7
SQuAD 2.0 F1 and 86.8 RACE accuracy in Table 2. However, the simple
mean of the eight printed GLUE metrics is $719/8=89.875$, not 90.00.
That calculation does not reproduce the reported Avg; the paper does
not explain the discrepancy there. RoBERTa's ensemble GLUE Test 88.5
uses another protocol. The 89.9 result versus a human baseline of 89.8
belongs to the 1.5B model's particular SuperGLUE protocol, not all
language tasks.

The [official repository](https://github.com/microsoft/DeBERTa)
separately describes V2: a new 128K SentencePiece vocabulary, an added
convolution, shared content/position projections, and relative buckets.
V3 changes the pre-training objective and how embeddings are updated.

</details>

<a id="limitations-and-legacy"></a>

## Input length and answer extraction

BERT's 512-position limit includes subwords, the question, and
separators. If the passage spans several windows, at least one must
contain the entire answer, and scores from different windows have to
be compared. A benchmark with short GLUE examples does not test this.

BERT attends across a known input. MLM fills selected positions in it; that loss does not teach ordinary left-to-right continuation.

On SQuAD 2.0 the model finds a span already present in the passage or
chooses `[CLS]`. The encoder in
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5]]
also reads the question and passage in both directions. A separate
decoder uses those states to generate an answer, so its output need
not be bounded by two positions in the passage.

<details id="bert-corruption-variants">
<summary>Whole-word masking, SpanBERT, and ALBERT</summary>

<a id="whole-word-masking"></a>
<h3 id="whole-word-masking-selects-targets-together">Whole-word masking selects targets together</h3>

If a name is tokenized as `phil ##am ##mon`, predicting only
`##am` leaves the other parts visible. Whole-word masking groups those
three positions when selecting prediction targets. It keeps WordPiece
tokenization and the same vocabulary-sized prediction head; it does not
turn the name into one new vocabulary item.

The original BERT release explains the unchanged prediction task:

> The training is identical -- we still predict each masked WordPiece token
> independently. The improvement comes from the fact that the original prediction
> task was too 'easy' for words that had been split into multiple WordPieces.

— Google Research, [BERT README, Whole Word Masking Models, May 31, 2019](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/README.md#L74-L118),
Apache-2.0. This is the authors' motivation, not a claim that every
partially visible word has only one possible completion.

In the pinned generator, a piece beginning with `##` joins the preceding
candidate group. The groups are shuffled, then selected under the
prediction budget. A group that would exceed the remaining budget is
skipped rather than partly selected. Consequently, grouping can leave a
budget underfilled; selecting exactly 15% of the positions in every short
example is not a requirement.

The following reduced selector receives an explicit group order instead
of drawing a random shuffle. It isolates grouping and budget handling
from token replacement:

```python
def whole_word_groups(pieces):
    groups = []
    for index, piece in enumerate(pieces):
        if piece in ("[CLS]", "[SEP]"):
            continue
        if piece.startswith("##") and groups:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups

def choose_word_targets(pieces, group_order, budget):
    groups = whole_word_groups(pieces)
    chosen = []
    for group_index in group_order:
        candidate = groups[group_index]
        if len(chosen) + len(candidate) <= budget:
            chosen.extend(candidate)
    return sorted(chosen)

pieces = ["[CLS]", "the", "man", "likes", "play", "##ing", "[SEP]"]
assert whole_word_groups(pieces) == [[1], [2], [3], [4, 5]]
order = [3, 1, 0, 2]
assert choose_word_targets(pieces, order, budget=2) == [4, 5]
assert choose_word_targets(pieces, order, budget=1) == [2]
assert choose_word_targets(["[CLS]", "play", "##ing", "[SEP]"],
                           [0], budget=1) == []

targets = choose_word_targets(pieces, order, budget=2)
corrupted = pieces.copy()
corrupted[4] = "[MASK]"
# An unchanged draw is allowed for the other selected piece.
assert corrupted[5] == "##ing"
assert [pieces[i] for i in targets] == ["play", "##ing"]
```

Both selected positions retain their original labels, even though one
input piece is unchanged. The actual BERT generator applies its 80/10/10
replacement decision separately inside the selected group. Thus “select
the whole word” does not mean “all its input pieces must be literal
`[MASK]` tokens.” The README's all-MASK example illustrates one outcome,
while [the generator, lines 341–415](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L341-L415)
defines the replacement procedure.

This code assumes a valid WordPiece sequence and a permutation of its
candidate groups. Other tokenizers need their own word-boundary mapping;
a SentencePiece boundary marker is not the same convention as `##`.
The existing MLM head still predicts each selected subword separately
from its contextual state. Whole-word selection changes the corruption
distribution, not the encoder architecture or the attention visibility
mask.

The original release reports improved SQuAD and MultiNLI results for its
new BERT-Large checkpoints. That does not establish a universal gain:
[SpanBERT, Table 6](https://aclanthology.org/2020.tacl-1.5/)
reports 77.7 coreference F1 for subword masking and 76.6 for whole-word
masking in its particular bi-sequence ablation. Dataset, training budget,
and the other objectives belong with any such comparison.

<a id="spanbert-boundary-objective"></a>
<h3 id="spanbert-predict-a-span-from-its-boundaries">SpanBERT: predict a span from its boundaries</h3>

An extractive QA model locates a contiguous answer, and a coreference
model compares mentions such as `the champion` and `Denver Broncos`.
Their downstream classifiers often read a span's boundary states.
SpanBERT adds a pretraining loss that asks those states to retain enough
information to predict the words between them.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/spanbert-boundary-objective.png|Original SpanBERT span-boundary objective and token predictions]]

*Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer
and Omer Levy, [SpanBERT, Figure 1, printed p. 65](https://aclanthology.org/2020.tacl-1.5.pdf#page=2).
Original figure and complete caption, CC BY 4.0. Only outer page material
was cropped; the formula, all token states and all arrows are retained.*

The paper describes the auxiliary prediction directly:

> Given a masked span of tokens
> $(x_s,\ldots,x_e)\in Y$, where $(s,e)$ indicates its start
> and end positions, we represent each token $x_i$
> in the span using the output encodings of the
> external boundary tokens $\mathbf{x}_{s-1}$ and $\mathbf{x}_{e+1}$, as well
> as the position embedding of the target token
> $\mathbf{p}_{i-s+1}$:

$$
y_i=f(h_{s-1},h_{e+1},p_{i-s+1}).
$$

— [Joshi et al., section 3.2](https://aclanthology.org/2020.tacl-1.5/),
CC BY 4.0. Bold contextual vectors in the source equation are written
as $h$ here to distinguish them from input token IDs.

In Figure 1, positions 5–8 contain `an American football game`.
The external boundary states are $h_4$ for `was` and $h_9$ for `to`.
To predict `football` at position 7, the boundary head receives
$[h_4;h_9;p_3]$. It does not read $h_7$ directly. The ordinary MLM head
does read $h_7$, so the same target has two prediction paths.

The offset embedding distinguishes the first, second, third and fourth
outputs for one common pair of boundaries. Without an offset or another
position-dependent input, a deterministic head receiving the same two
vectors would return the same vocabulary distribution at every position
of the span. The offset specifies a location; the target word itself
remains a label outside the input.

The paper uses two dense transformations, each followed by GELU and
LayerNorm:

$$
\begin{aligned}
u_i&=[h_{s-1};h_{e+1};p_{i-s+1}],\\
a_i&=\operatorname{LN}(\operatorname{GELU}(W_1u_i)),\\
y_i&=\operatorname{LN}(\operatorname{GELU}(W_2a_i)).
\end{aligned}
$$

The input vocabulary embedding matrix supplies the tied output weights
for both predictions. For each selected token, the objective adds

$$
L_i=-\log P_{\mathrm{MLM}}(x_i\mid h_i)
    -\log P_{\mathrm{SBO}}(x_i\mid y_i).
$$

For illustrative probabilities 0.5 from MLM and 0.25 from SBO, this
token contributes $-\log0.5-\log0.25=\log8\approx2.0794$.
Dropping the second term would remove the training signal to the
boundary head. The predictions within the span are not fed to one
another as decoder inputs; this remains encoder pretraining, not
autoregressive generation of the missing phrase.

The small module below checks the two read paths. It starts with supplied
final encoder states rather than running another Transformer. Its widths
are 4 for a state and 2 for an offset, not the original model dimensions.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(131)
dtype = torch.float64
final_states = nn.Parameter(torch.randn(12, 4, dtype=dtype))
offsets = nn.Embedding(4, 2, dtype=dtype)
vocabulary = nn.Embedding(12, 4, dtype=dtype)
boundary_head = nn.Sequential(
    nn.Linear(10, 8, bias=False, dtype=dtype), nn.GELU(),
    nn.LayerNorm(8, dtype=dtype),
    nn.Linear(8, 4, bias=False, dtype=dtype), nn.GELU(),
    nn.LayerNorm(4, dtype=dtype),
)

# Zero-based rows 3 and 8 are h_4 and h_9 in the original figure.
boundary_inputs = torch.cat([
    final_states[3].expand(4, -1),
    final_states[8].expand(4, -1),
    offsets(torch.arange(4)),
], dim=-1)
sbo_states = boundary_head(boundary_inputs)
sbo_logits = F.linear(sbo_states, vocabulary.weight)
mlm_logits = F.linear(final_states[4:8], vocabulary.weight)
span_targets = torch.tensor([4, 5, 6, 7])
sbo_loss = F.cross_entropy(sbo_logits, span_targets)
mlm_loss = F.cross_entropy(mlm_logits, span_targets)
direct_sbo_gradient, = torch.autograd.grad(
    sbo_loss, final_states, retain_graph=True
)
assert direct_sbo_gradient[3].norm() > 0
assert direct_sbo_gradient[8].norm() > 0
assert torch.count_nonzero(direct_sbo_gradient[4:8]) == 0

same_offset_inputs = boundary_inputs.clone()
same_offset_inputs[:, -2:] = boundary_inputs[0, -2:]
same_offset_logits = F.linear(
    boundary_head(same_offset_inputs), vocabulary.weight
)
torch.testing.assert_close(
    same_offset_logits, same_offset_logits[0].expand_as(same_offset_logits)
)
(mlm_loss + sbo_loss).backward()
assert final_states.grad[4:8].norm() > 0
assert offsets.weight.grad.norm() > 0
assert vocabulary.weight.grad.norm() > 0
assert math.isclose(-math.log(.5) - math.log(.25), math.log(8))
```

The zero internal-state derivative above concerns direct inputs to
the SBO head at the final encoder output. In a complete encoder, the
boundary states themselves depend on other positions through attention.
SBO can therefore backpropagate through those earlier dependencies.
The check does not say that all internal-position computations receive
zero gradient throughout SpanBERT.

<a id="spanbert-span-sampling"></a>
<h4 id="sampling-several-words-without-introducing-a-decoder">Sampling several words without introducing a decoder</h4>

Whole-word masking can select isolated words. SpanBERT instead starts
from a word boundary and extends a contiguous span. Its proposed
length distribution favors short spans but includes lengths up to ten
words; the total target budget is about 15% of subword positions.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/spanbert-span-lengths.png|Original SpanBERT distribution of sampled span lengths]]

*Joshi et al., [Figure 2, printed p. 66](https://aclanthology.org/2020.tacl-1.5.pdf#page=3),
CC BY 4.0. Complete original plot and caption. The horizontal axis counts
words, not WordPiece positions or decoder tokens.*

In the released implementation, weights $0.2(0.8)^{\ell-1}$ for
$\ell=1,\ldots,10$ are normalized over that finite range:

$$
P(\ell)=\frac{0.2(0.8)^{\ell-1}}{1-0.8^{10}},
\qquad \ell\in\{1,\ldots,10\}.
$$

This produces mean length about 3.8, as in the paper. Sampling an
unbounded geometric variable and replacing every value above ten
by ten would instead accumulate a tail probability at ten. That is a
different distribution from the plotted and implemented one.

```python
import math

weights = [.2 * .8**(length - 1) for length in range(1, 11)]
length_probabilities = [weight / sum(weights) for weight in weights]
mean_words = sum(length * probability for length, probability
                 in zip(range(1, 11), length_probabilities))
assert math.isclose(sum(length_probabilities), 1.)
assert 3.79 < mean_words < 3.80
clamped_geometric_mean = sum(.8**step for step in range(10))
assert 4.46 < clamped_geometric_mean < 4.47
assert not math.isclose(mean_words, clamped_geometric_mean)
```

See [the pinned sampling code, lines 74–132](https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/pretraining/fairseq/data/masking.py#L74-L132).
The paper describes spans of whole words; the released loop also stops
at the exact subword budget, so its final selected word can be cut
short. The sampled length distribution does not imply that every
realized span has that many complete words after budget and document
boundaries are applied.

The original span-level replacement setting draws the corruption type
once for a span: MASK, random replacements, or unchanged text.
This differs from BERT's per-piece replacement loop. Adjacent selected
intervals can also merge before SBO boundary targets are prepared.
External boundaries must exist; the implementation does not fabricate
a context token outside the document.

The model also uses a single contiguous training segment without NSP.
That input change and the auxiliary objective must be separated when
reading results. In the paper's 1.2M-step ablation, Table 7 reports:

| Span-masking setting | SQuAD 2.0 Dev F1 | NewsQA Dev F1 | Coreference F1 |
|---|---:|---:|---:|
| two segments + NSP | 85.4 | 73.0 | 76.4 |
| one segment | 86.7 | 73.4 | 76.3 |
| one segment + SBO | 86.8 | 74.1 | 79.0 |

The first transition changes both the segment construction and NSP.
The second keeps single-segment span masking and adds SBO.
Its coreference gain is much larger than its SQuAD gain in this ablation.
These are not the final 2.4M-step test results, and they do not establish
one fixed improvement from adding SBO to any model.

In [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5]],
a deleted span is replaced by one sentinel and a separate causal decoder
generates the missing text. SpanBERT retains the corrupted sequence's
positions and predicts its selected tokens through encoder heads.
The two methods share span-based corruption, but their target format,
conditional dependencies and computation are different.

<a id="sentence-order-prediction"></a>
<h3 id="albert-classify-the-order-of-two-document-segments">ALBERT: classify the order of two document segments</h3>

NSP's negative example can combine a paragraph about unlocking a door
with a paragraph about penguins. A classifier can use the change of topic
without learning which of two related events came first. ALBERT's
sentence-order prediction (SOP) changes how the negative pair is built.

Take consecutive segments from one document:

| Label | First segment | Second segment |
|---|---|---|
| ordered | She unlocked the door. | Then she entered. |
| swapped | Then she entered. | She unlocked the door. |

Both rows contain the same words. A topic histogram of the combined
text is identical, so that feature alone cannot distinguish these two
labels. Token order, positions and segment membership do change.
The segments need not be single linguistic sentences.

The released ALBERT generator implements the swapped case as follows:

```python title="Original ALBERT pair swap — Apache-2.0"
        elif not FLAGS.random_next_sentence and rng.random() < 0.5:
          is_random_next = True
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
          # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
          tokens_a, tokens_b = tokens_b, tokens_a
```

— Google Research, [create_pretraining_data.py, lines 338–343](https://github.com/google-research/albert/blob/b772393d3dae115b493258ce8e37c17b2cc62100/create_pretraining_data.py#L338-L343),
Copyright 2019 The Google Research Authors, Apache-2.0.
The historical variable name `is_random_next` also marks a swapped SOP
pair; it does not mean that this branch sampled another document.

Input assembly still produces `[CLS] A [SEP] B [SEP]`.
The first segment and its initial special tokens receive type 0,
the second segment and final separator type 1. Positional IDs run
through the assembled sequence. The binary head reads the pooled
`[CLS]` state, and its cross-entropy is added to MLM loss.

The following calculation uses supplied logits to check target labels,
mean cross-entropy and its gradient:

```python
from collections import Counter
import math
import torch
import torch.nn.functional as F

def assemble_sop(first, second):
    tokens = ["[CLS]", *first, "[SEP]", *second, "[SEP]"]
    types = [0] * (len(first) + 2) + [1] * (len(second) + 1)
    return tokens, types

first = ["she", "unlocked", "the", "door", "."]
second = ["then", "she", "entered", "."]
ordered, ordered_types = assemble_sop(first, second)
swapped, swapped_types = assemble_sop(second, first)
assert Counter(ordered) == Counter(swapped)
assert ordered != swapped and ordered_types != swapped_types
assert len(ordered) == len(ordered_types) == 12
assert len(swapped) == len(swapped_types) == 12
sentence_order_labels = torch.tensor([0, 1])
sop_logits = torch.tensor([[math.log(3), 0.],
                          [0., math.log(3)]], dtype=torch.float64,
                         requires_grad=True)
sop_loss = F.cross_entropy(sop_logits, sentence_order_labels)
assert math.isclose(sop_loss.item(), -math.log(.75))
sop_loss.backward()
torch.testing.assert_close(
    sop_logits.grad,
    torch.tensor([[-.125, .125], [.125, -.125]], dtype=torch.float64)
)
```

The supplied logits assign probability 0.75 to the correct label in
each row, giving mean loss about 0.28768. In training those scores
would come from the shared encoder and sentence head; the derivative
would reach both. SOP does not predict the next segment's words and
does not turn the encoder into a document generator.

For compatibility, the pinned preprocessor writes the label under the
feature name `next_sentence_labels` even for SOP. It maps ordered to 0
and swapped to 1. It also retains a random-document fallback when a
chunk contains only one segment. The ordered/swapped definition
describes the intended two-segment task; exact preprocessing must also
account for that fallback and truncation.

[Lan et al., ALBERT v6, sections 3.1 and 4.6, Table 5](https://arxiv.org/abs/1909.11942v6)
compare three losses in an ALBERT-base configuration. The following
are their reported measurements, not results of the small example:

| Sentence loss | NSP accuracy, % | SOP accuracy, % | SQuAD 2.0 Dev F1 | RACE accuracy, % |
|---|---:|---:|---:|---:|
| none | 52.4 | 53.3 | 78.1 | 61.7 |
| NSP | 90.5 | 52.0 | 77.2 | 62.3 |
| SOP | 78.9 | 86.5 | 80.0 | 64.0 |

High NSP accuracy did not transfer to the SOP diagnostic in this
experiment. The downstream columns support the authors' choice for
that setup; they do not prove that every NSP objective harms every
model, or that solving reversed pairs establishes general discourse
understanding. A useful diagnostic must test held-out documents and
control for punctuation, segment length and other ordering cues.

</details>

<details id="bert-rtd-variants">
<summary>ELECTRA and DeBERTaV3: detecting replacements</summary>

<a id="electra-replaced-token-detection"></a>

<a id="electra-detecting-which-tokens-changed"></a>

<h3>ELECTRA: identifying replaced tokens</h3>

BERT computes states for the entire input, but its vocabulary head
has direct targets only at selected positions. ELECTRA uses those
positions to construct a corrupted input. The main encoder receives
one binary label at every valid position: did the token change?
The resulting encoder can be fine-tuned for the same classification
and labeling tasks as BERT.

The small generator first solves the familiar MLM task. Suppose the
original text is `She cooked rice and washed dishes`, with `cooked`
and `dishes` masked for the generator. It returns two vocabulary
distributions and trains against the original `cooked` and `dishes`.
Draw one token from each distribution. In this example the draws
are `ate` and `dishes`. The second draw matches the original token,
so the discriminator reads `She ate rice and washed dishes` with
labels $(0,1,0,0,0,0)$, where 1 means the token changed. The
`dishes` position has an MLM target and an RTD label of zero.

Let $x_i$ be the original token, $\hat{x}_i$ the discriminator input
token, and $\mathcal V$ the set of valid positions excluding padding.
For the target $y_i=\mathbb 1[\hat{x}_i\ne x_i]$ and the logit $s_i$
from the discriminator state, let $p_i=\sigma(s_i)$. The mean loss is

$$
L_{\mathrm{RTD}}=-\frac{1}{|\mathcal V|}
\sum_{i\in\mathcal V}\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big].
$$

Here $p_i$ is the probability of a replacement. The original paper
uses $D$ for the probability that the token is original. Reversing
the class convention leaves the binary loss unchanged. If the
discriminator gives $p_i=0.8$ to `ate` and $p_i=0.2$ to the five
original tokens, each correct label receives probability $0.8$ and
the mean loss is $-\log 0.8\approx0.2231$.

The main encoder receives more direct targets per sequence than
in MLM, at the cost of also running and training the generator.
Sampling plausible replacements asks the discriminator to judge the
token in context. The generator learns to reconstruct the original
text; the binary loss does not differentiate through the discrete
sample. With a shared embedding table, the loss has a separate path
to shared parameters. [ELECTRA Figure 2](https://arxiv.org/html/2003.10555v1#S2.F2)
shows the full path from masks to replacement labels.

<a id="debertav3-gradient-disentangling"></a>

<a id="debertav3-shared-values-separate-gradients"></a>

<h3>DeBERTaV3: sharing embeddings with stop-gradient</h3>

DeBERTaV3 retains DeBERTa's attention with separate content and
relative-position signals. During pre-training the main encoder becomes
the RTD discriminator, while MLM remains the auxiliary generator's
objective. The upper arrows in the authors' figure show the token
path; the arrows below show gradient flow. $\theta_G$ and $\theta_D$
denote network parameters apart from the embedding tables shown.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/debertav3-2026-11-13/debertav3-figure1-sharing.png|Three ways to share generator and discriminator embeddings: ES, NES, and GDES]]

*Pengcheng He, Jianfeng Gao, and Weizhu Chen, [DeBERTaV3, Figure 1, p. 5](https://arxiv.org/pdf/2111.09543v4#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
All three panels and the original caption are retained; only the surrounding page is cropped.*

In panel (a), ES, one table $E$ receives the sum of the
$L_{\mathrm{MLM}}$ and $\lambda L_{\mathrm{RTD}}$ gradients. Where
one gradient component is negative and the other positive, they
partly cancel. In panel (b), NES, each network has a separate table:
updating $E_G$ no longer changes $E_D$. This removes the competition
over a shared parameter, but the discriminator no longer reads the
values learned by the generator.

In panel (c), GDES, each discriminator forward pass constructs its
table from two terms:

$$
E_D=\operatorname{stopgrad}(E_G)+E_\Delta.
$$

$E_\Delta$ starts at zero, so both networks initially read the same
rows. The stop-gradient operation passes the value of $E_G$ forward
without differentiating through it. RTD updates $E_\Delta$ and the
discriminator parameters; MLM updates $E_G$ and the generator
parameters. After an MLM step, the effective $E_D$ also changes:
the discriminator reads the new value of $E_G$ on its next pass.
The [pinned Microsoft implementation](https://github.com/microsoft/DeBERTa/blob/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098/DeBERTa/apps/tasks/rtd_task.py#L99-L131)
uses `.detach()` on the sampled input IDs and, separately, on the
generator's contribution to the discriminator's embedding table.

Consider one embedding coordinate instead of a full table. Use
quadratic stand-ins for MLM and RTD to isolate their gradients:
the targets are 3 and 0 respectively, the initial value is 1,
$\lambda=2$, and the SGD learning rate is $0.1$. Under ES the
derivatives are $1-3=-2$ and $2(1-0)=2$, leaving the shared value
at 1. Under NES the generator updates its value to $1.2$ and the
discriminator updates its own value to $0.8$. Under GDES the MLM
step first updates $E_G$ to $1.2$. The discriminator then reads
$1.2+0$ and sends a gradient of $2.4$ only to $E_\Delta$. The
correction becomes $-0.24$, so after both updates
$E_D=1.2-0.24=0.96$ while $E_G=1.2$. This is one illustrative step,
not a simulation of training on a corpus.

```python
import math
import torch
import torch.nn.functional as F

# One fixed outcome of sampling: cooked -> ate, dishes -> dishes.
original = torch.tensor([1, 2, 3, 4, 5, 6, 0])  # 0 is PAD
chosen = torch.tensor([1, 5])
sampled = torch.tensor([7, 6])
valid = original.ne(0)
masked = original.clone()
masked[chosen] = 8  # MASK
corrupted = original.clone()
corrupted[chosen] = sampled
replaced = corrupted.ne(original)
assert masked.tolist() == [1, 8, 3, 4, 5, 8, 0]
assert original[chosen].tolist() == [2, 6]  # generator's two MLM labels
assert replaced[valid].int().tolist() == [0, 1, 0, 0, 0, 0]

# Supplied RTD logits give the true label probability 0.8 at each position.
logits = torch.where(replaced, math.log(4), -math.log(4)).double()
rtd = F.binary_cross_entropy_with_logits(logits[valid], replaced[valid].double())
assert math.isclose(rtd.item(), -math.log(.8), abs_tol=1e-8)

# Scalar losses isolate gradient flow; they are not MLM or RTD losses.
def scalar(value):
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)

lr, weight = .1, 2.
e = scalar(1.)
loss_es = .5 * (e - 3).square() + weight * .5 * e.square()
grad_es, = torch.autograd.grad(loss_es, e)
assert grad_es.item() == 0.

g, delta, d = scalar(1.), scalar(0.), scalar(1.)
grad_g, = torch.autograd.grad(.5 * (g - 3).square(), g)
g_next = (g - lr * grad_g).detach().requires_grad_()
grad_d, = torch.autograd.grad(weight * .5 * d.square(), d)
d_nes = d - lr * grad_d

d_gdes = g_next.detach() + delta
grad_g_from_rtd, grad_delta = torch.autograd.grad(
    weight * .5 * d_gdes.square(), (g_next, delta), allow_unused=True
)
assert grad_g_from_rtd is None
assert math.isclose(grad_delta.item(), 2.4)
delta_next = delta - lr * grad_delta
exported = g_next.detach() + delta_next.detach()
assert math.isclose(g_next.item(), 1.2)
assert math.isclose(d_nes.item(), .8)
assert math.isclose(exported.item(), .96)
```

The authors compare all three methods with the same Base setup.
Figure 2 plots the **generator's** MLM loss against training steps,
rather than discriminator accuracy or elapsed time. In this
experiment NES and GDES track each other and remain below ES
after the initial drop.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/debertav3-2026-11-13/debertav3-figure2-mlm-loss.png|Generator MLM loss over training steps for ES, NES, and GDES]]

*He, Gao, and Chen, [DeBERTaV3, Figure 2, p. 6](https://arxiv.org/pdf/2111.09543v4#page=6),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original plot, logarithmic loss axis, and English caption are retained.*

[Table 2, p. 5](https://arxiv.org/pdf/2111.09543v4#page=5) separately
compares downstream results after fine-tuning. MNLI-m accuracy is
88.8 for ES, 88.3 for NES, and 89.3 for GDES; SQuAD 2.0 F1 is
86.3, 85.3, and 87.2 respectively. Here NES reduced the MLM loss
more quickly but trailed ES on both downstream tasks. The final V3
model family in §4.1 also changes the training recipe, so those
results cannot be attributed to GDES alone.

The discriminator can be exported with the folded table
$E_G+E_\Delta$. Fine-tuning the encoder for answer extraction, for
example, requires neither the generator nor the RTD head.

</details>

## Sources and further reading

- Jacob Devlin et al., [BERT, §§3–5, Appendices A.1–A.5 and C.2](https://aclanthology.org/N19-1423/).
- Yinhan Liu et al., [RoBERTa v1, §§3–5, Tables 1–7](https://arxiv.org/abs/1907.11692v1).
- Pengcheng He et al., [DeBERTa v6, §§3–5, Appendices A.7–A.8](https://arxiv.org/abs/2006.03654v6).
- Kevin Clark et al., [ELECTRA, §2 and Figure 2](https://arxiv.org/html/2003.10555v1#S2), and the [authors' Google Research explanation](https://research.google/blog/more-efficient-nlp-model-pre-training-with-electra/).
- Pengcheng He, Jianfeng Gao, and Weizhu Chen, [DeBERTaV3, §§2.3 and 3–4.1](https://arxiv.org/pdf/2111.09543v4): RTD, three embedding-sharing methods, Figures 1–2, and Table 2.
- Google Research, [BERT Whole Word Masking release and generator](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/README.md#L74-L118), Apache-2.0. The original explanation is quoted above; the reduced selector and budget examples are added. [Full license](/licenses/google-bert-eedf5716-Apache-2.0.txt).
- Mandar Joshi et al., [SpanBERT, sections 3, 4.2–4.3 and 6, Figures 1–2, Tables 6–7](https://aclanthology.org/2020.tacl-1.5/), CC BY 4.0. Original figures/captions and the boundary-input passage retain their language. The equations use $h$ for contextual states; the small PyTorch module and probability calculations are added.
- Zhenzhong Lan et al., [ALBERT v6, sections 3.1 and 4.6, Table 5](https://arxiv.org/abs/1909.11942v6). Pair construction and legacy feature naming are also checked against [the pinned original generator](https://github.com/google-research/albert/blob/b772393d3dae115b493258ce8e37c17b2cc62100/create_pretraining_data.py), Copyright 2019 The Google Research Authors, Apache-2.0. The swap excerpt is unchanged; the door example and supplied-logit calculation are added. [Full code license](/licenses/google-albert-b772393d-Apache-2.0.txt).
- Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/).
- Lena Voita, [Transfer Learning — BERT and fine-tuning](https://lena-voita.github.io/nlp_course/transfer_learning.html).
- [Official BERT: model and data preparation](https://github.com/google-research/bert/tree/eedf5716ce1268e56f0a50264a88cafad334ac61).
- [Official DeBERTa: attention and MLM](https://github.com/microsoft/DeBERTa/tree/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098).
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-only|Encoder-only reference]].
- [[02 Areas/ML & DL/04 Вопросы/100 вопросов по NLP|NLP questions]].

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three Transformer architectures]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей|Named entity recognition]].
