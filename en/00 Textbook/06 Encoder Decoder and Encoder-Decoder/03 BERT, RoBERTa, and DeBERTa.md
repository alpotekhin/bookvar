---
title: BERT, RoBERTa, and DeBERTa
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "0cce42e115bc2a8c453047ce060f7a67dac104a9eae45ed1b899a49f4d921741"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md"
last_updated: 2026-10-29
primary_sources:
  - https://aclanthology.org/N19-1423/
  - https://arxiv.org/abs/1907.11692v1
  - https://arxiv.org/abs/2006.03654v6
  - https://aclanthology.org/2020.tacl-1.5/
  - https://arxiv.org/abs/1909.11942v6
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение]]"
---

# BERT, RoBERTa, and DeBERTa

BERT uses one pretrained encoder as the starting point for classification,
token labeling, and answer extraction. It first learns from ordinary text;
a task-specific output layer is then added and the model fine-tuned on
annotations. For a question–paragraph pair, for example, the model locates
the answer's start and end in the paragraph. Attention connects both texts
inside the encoder.

The [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|previous chapter]]
explains why a bidirectional encoder needs an objective other than NTP.
Here we follow BERT through pair preparation, input representations,
MLM/NSP, weight updates, and supervised application. RoBERTa changes the
training recipe; DeBERTa changes attention computation and masked-token
reconstruction.

## Architecture and tensor shapes

`BERT_BASE` has 12 blocks, width $d=768$, and 12 heads. Each head has
width $768/12=64$, and the FFN intermediate width is 3072.
`BERT_LARGE` has 24 blocks, $d=1024$, 16 heads, and FFN width 4096.
The authors report approximately 110 and 340 million parameters
respectively ([paper, §3](https://aclanthology.org/N19-1423/)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-base-bert-large-encoders.png]]

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

This is post-norm. BERT has no separate decoder stack or cross-attention.
Joint self-attention over a question and paragraph nevertheless lets
positions in either text read the other. The mask excludes PAD keys,
not right context.

### Turning two segments into one input

The paper uses “sentence” for an arbitrary contiguous text segment,
which may contain several linguistic sentences. Two segments A/B are
selected and WordPiece-tokenized. Figure 2 gives this example:

```text
[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
```

There are 11 positions. `##ing` continues a word rather than being a
separate word. `[CLS]` starts the complete input; the first `[SEP]`
ends A and the second ends B. Positions run continuously from 0 to 10;
the beginning of B does not reset the counter.

Each row adds three learned vectors:

$$
\begin{aligned}
e_i&=E_{\mathrm{token}}[x_i]\\
&\quad+E_{\mathrm{position}}[i]+E_{\mathrm{segment}}[s_i].
\end{aligned}
$$

`[CLS]` and the first `[SEP]` have type A; the second segment and final
`[SEP]` have type B. A segment embedding identifies the token's part,
but does not block attention between parts.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure2.png]]

*The three summands of the input representation. `play` and `##ing`
use different token and position rows but the same type B.
Source: Jacob Devlin et al., [BERT, Figure 2](https://aclanthology.org/N19-1423.pdf#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The diagram and English caption are unchanged.*

The official code applies LayerNorm and dropout to this sum before the
stack ([`embedding_postprocessor`](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L428-L521)).
The sum of three tables therefore describes feature assembly, not the
whole input layer.

For a batch of $B$ examples padded to length $T$, shapes are:

| Quantity | Shape |
|---|---|
| token IDs, type IDs, and real-position mask | `[B,T]` |
| states before and after each block | `[B,T,d]` |
| Q/K/V split into heads | `[B,h,T,d_h]` |
| attention scores | `[B,h,T,T]` |
| MLM logits for $m$ selected positions | `[m,V]` |

The example has `B=1,T=11`; BASE's Q/K/V shapes are `[1,12,11,64]`.
Adding PAD enlarges the batch tensor, not the number of real tokens or
MLM targets. Excluding PAD keys does not itself zero PAD-query outputs;
those rows must be excluded from subsequent loss or pooling.

## MLM: obtaining a bidirectional training signal

Positions to reconstruct are selected among ordinary subwords.
Original BERT selects approximately 15%; `[CLS]` and `[SEP]` are
not candidates in the official generator. Selection follows WordPiece
tokenization: the original recipe may select one piece of a word without
the others ([BERT, §3.1, A.1–A.2](https://aclanthology.org/N19-1423/)).
Whole-word masking is a later data-preparation variant.

Suppose `cute` is selected. Its input may become `[MASK]`, a random
token, or unchanged `cute`; the correct target is always the original
ID for `cute`. Among selected positions, these cases have probabilities
80%, 10%, and 10%. This does not mean that 80% of the whole text is hidden.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png]]

*The selected position's state becomes a vocabulary distribution.
“All English words” in the original figure is a simplification: actual
outputs index a WordPiece vocabulary, including subwords and special tokens.
Source: Jay Alammar, [The Illustrated BERT, Masked Language Model](https://jalammar.github.io/illustrated-bert/#masked-language-model),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Bidirectional attention builds $h_{\mathrm{cute}}$ from the entire
corrupted pair. The original implementation's MLM head applies another
transformation:

$$
\begin{aligned}
z_i&=\operatorname{LN}
\bigl(\operatorname{GELU}(h_iW_m+b_m)\bigr),\\
\ell_i&=z_iE_{\mathrm{token}}^\top+b_{\mathrm{vocab}}.
\end{aligned}
$$

The output matrix shares weights with the input token table. The extra
dense/GELU/LayerNorm transformation and vocabulary bias belong to the
MLM head; they are unnecessary when switching to classification.
Source: [official `get_masked_lm_output`](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_pretraining.py#L240-L282).

Let $\mathcal M$ be selected positions and $\tilde x$ the corrupted input.
Mean MLM loss is

$$
L_{\mathrm{MLM}}=
-\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).
$$

The [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|introductory chapter]]
already checks why an unchanged selected token also contributes a loss.
The same principle applies here: labels are stored separately from
corrupted IDs. Unselected rows have no direct MLM loss, but their states
participate in selected positions' attention and receive gradients
through those dependencies.

The mixture of replacements reduces mismatch with downstream use,
where `[MASK]` is usually absent. Appendix C.2 separately compares
80/10/10 with 100% MASK. Full fine-tuning is relatively robust to these
variants, while the frozen-feature NER experiment gives 94.0 Dev F1 for
all-MASK versus 94.9 for 80/10/10. This is a result for a particular
experiment, not proof that this proportion is optimal for all models.

## BERT's second objective: segment relationships

For NSP, half the pairs contain a genuine next document segment, and
half contain a randomly selected second segment. For example,
`my dog is cute` followed by `he likes playing` and followed by a
passage about penguins would have different labels. These illustrate
target construction, not evaluation data.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-next-sentence-prediction.png]]

*The pair enters one bidirectional encoder, and `[CLS]` supports pair
classification. The middle Tokenized Input row shows only the first
segment; actual input includes both segments and separators, as in the
bottom row.
Source: Jay Alammar, [The Illustrated BERT, Two-sentence Tasks](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

In the official model, the last `[CLS]` state passes through a pooler:
`dense → tanh`. A binary head maps it to two logits. The code uses label
0 for a genuine continuation and 1 for a random segment. NSP neither
classifies every neighboring word pair nor scores the whole sequence
using the NTP rule.

Each step adds mean loss over selected MLM tokens to mean NSP loss
over batch examples:

$$
L=L_{\mathrm{MLM}}+L_{\mathrm{NSP}}.
$$

Summing unnormalized losses for all MLM positions with one NSP loss
would be a different normalization: the objective ratio would then
change with text length. Original code normalizes MLM by the sum of
valid-target weights and adds $10^{-5}$ to the denominator. That
protects division, but does not make a batch without training targets useful.

### One executable pre-training step

The code below passes a corrupted pair through a small post-norm encoder,
a weight-tied MLM head, and an NSP head, computes both losses, and updates
weights. It uses PyTorch's attention; attention-head operations are
explained in the
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Transformer chapter]].

There are 12 toy vocabulary tokens, two width-16 layers, and four heads.
Selection of `cute` is specified in advance, not produced by a real
WordPiece tokenizer. Dropout is disabled for reproducibility. This is
a computation example, not a reproduction of published BERT training.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure1.png]]

*On the left, a shared encoder receives pair A/B: MLM outputs reconstruct
selected positions, while NSP reads `[CLS]`. On the right, the same
pretrained parameters initialize separate supervised-task models.
Arrows between the two sides represent parameter transfer, not the
transfer of current states from one task to another.
Source: Jacob Devlin et al., [BERT, Figure 1](https://aclanthology.org/N19-1423.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Both panels and the original English caption are unchanged.*

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

The same encoder weights receive two signals: reconstruct `cute` and
classify the segment relationship. The vocabulary table receives
gradients through input representations and through the shared output
projection. PAD is not a target and is masked as a key. Loss values
from a randomly initialized network are not a language-task result.

To check the unselected `dog` position's contribution specifically to
MLM, the code differentiates MLM loss with respect to intermediate
input vectors. Element `[0,2]` is `dog`: it is not a target, but
attention lets its representation affect reconstruction of `cute`.
NSP loss is not included in this measurement. The derivative is taken
with respect to the position's input vector, not a weight row in the
shared input/output table. The input PAD vector receives zero MLM
gradient because it cannot be read as a key at any layer. With the
specified seed on CPU in float64, the input `dog` gradient norm is
approximately $0.118682$, and PAD's is zero.

The shared table `word.weight` has another role: its rows also participate
in vocabulary softmax. Nonzero `word.weight.grad[5]` alone would
therefore not demonstrate use of input `dog`. Even the PAD row receives
a gradient through the output classifier: the target is `cute`, but
probabilities normalize over all 12 classes, including PAD. The input
attention mask does not remove this output class.

## BERT data and recipe

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

<a id="один-encoder--разные-способы-чтения-выхода"></a>

## Fine-tuning on a labeled task

After pre-training, tokenization, input tables, and the encoder are
retained. MLM/NSP heads are replaced by the required output format.
In full fine-tuning, supervised-task gradients change the entire stack.
Each task gets its own fine-tuned copy of the weights: one pretrained
checkpoint does not mean one shared ready-made classifier.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/bert-figure4.png]]

*Single-text and pair classification above; answer extraction and NER below.
Source: Jacob Devlin et al., [BERT, Figure 4](https://aclanthology.org/N19-1423.pdf#page=15),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
All four panels and the original caption are preserved.*

Classification uses `[CLS]` or its corresponding pooled output and
learns class scores. NER applies a projection to token states. When a
word splits into several WordPieces, the label policy must be specified:
BERT's experiment uses the first subword's state. Special positions and
PAD must not silently acquire training labels.

### Extractive QA: two projections and a valid span

For a question–paragraph pair, learned vectors $s,e$ produce
$\ell_i^{\mathrm{start}}=s^\top h_i$ and
$\ell_i^{\mathrm{end}}=e^\top h_i$.
Softmax normalizes over allowed positions, not vocabulary items.
The correct start and end are indices in the source paragraph.

Independent start and end maxima may define an impossible span.
Suppose the paragraph is `Marie lives in Paris`, with start logits
$(0,0,1,3)$ and end logits $(4,0,0,2)$. Independent selection starts at
`Paris` and ends at `Marie`. Under $j\ge i$, the highest sum is
$3+2=5$, selecting the single-token answer `Paris`.

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

A real batch must also exclude the question, separators, and padding.
Long documents are split into overlapping windows with indices mapped
back to the original text. The code only checks valid-span selection
from given logits and its loss; it does not train an encoder.

In SQuAD 2.0, some questions have no answer. Original BERT assigns
`[CLS]` as both their correct start and end. At decoding, the best
nonempty span is compared with the summed `[CLS]` start/end scores
using a threshold selected on dev. This differs from requiring some
paragraph word as an answer
([BERT, §§4.2–4.3](https://aclanthology.org/N19-1423/)).

### Frozen features or full fine-tuning

BERT can be frozen while a separate model is trained on its states.
Features for unchanged inputs can then be computed in advance. The
last state at each position is not the only option: the paper compares
several layers and their combinations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png]]

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

## Reading BERT's results

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

<a id="roberta-когда-новый-результат-даёт-не-новая-архитектура"></a>

## RoBERTa: changes to training

RoBERTa retains a bidirectional stack and MLM. Liu et al. first study
example preparation and optimization at BASE size, then combine changes
in a LARGE model. The sequence of comparisons separates individual
settings from the final recipe's result
([§§3–5](https://arxiv.org/pdf/1907.11692v1)).

### Static and dynamic masking

Original BERT preprocessing writes corrupted examples in advance.
To avoid giving a text only one mask, data are duplicated ten times
with different variants. In the 40-epoch regime described by RoBERTa,
each variant appears about four times.

Dynamic masking selects positions again before presenting the input.
The same tokenized string might first reconstruct `cute`, later `dog`,
then `likes`. The training example changes, not the bidirectional
attention visibility mask.

| BASE replication variant | SQuAD 2.0 F1 | MNLI-m accuracy | SST-2 accuracy |
|---|---:|---:|---:|
| static masking | 78.3 | 84.3 | 92.5 |
| dynamic masking | 78.7 | 84.0 | 92.9 |

Source: [RoBERTa, Table 1](https://arxiv.org/pdf/1907.11692v1#page=4).
Values are medians over five runs. The effect is small and not uniformly
positive across the three tasks; dynamic masking alone cannot explain
RoBERTa's later overall gain.

### NSP and text assembly

Section 4.2 compares four formats:

| Format | Texts included in one input | NSP |
|---|---|---|
| SEGMENT-PAIR | two segments, each possibly containing several sentences | yes |
| SENTENCE-PAIR | two individual sentences | yes |
| FULL-SENTENCES | whole sentences filling input up to 512; a separator permits crossing into another document | no |
| DOC-SENTENCES | whole sentences from one document only; its end may produce a short input | no |

A short pair of individual sentences offers fewer long-range dependencies
than long contiguous segments. The authors increase the short-example
batch to roughly preserve token count, but SENTENCE-PAIR still underperforms
SEGMENT-PAIR in their experiment.

FULL-SENTENCES gives 79.1 SQuAD 2.0 Dev F1; DOC-SENTENCES gives 79.7;
the SEGMENT-PAIR replication with NSP gives 78.7. Later experiments
choose FULL-SENTENCES because restricting inputs to one document causes
more variation in length and examples per batch.

Both NSP and input organization change here. The table shows that this
recipe can work without NSP. It does not prove that adding any NSP
objective to any training setup must hurt. Comparison with BERT's
ablation requires checking which inputs remain after removing the loss.

### Batch size, update count, and vocabulary

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

### The final recipe and results

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

<a id="whole-word-masking"></a>
## Whole-word masking selects targets together

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
## SpanBERT: predict a span from its boundaries

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
### Sampling several words without introducing a decoder

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
## ALBERT: classify the order of two document segments

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

## DeBERTa: separate content from relative position

BERT adds absolute position to token vectors at the input. After projection,
the Q/K dot product mixes both signals. DeBERTa separately projects
content states $H$ and a relative-position table $P$; a relative vector
is selected for a position pair, not permanently attached to a word.

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

### Three terms in one score

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
differently from a distant word on the right. The third depends on key
content and the query's relative location. It uses the reversed
distance $\delta(j,i)$. The authors omit the position-to-position term
from the general decomposition.

Take $k=2$, $i=1$, $j=2$, and $d_h=2$. Then $\delta(i,j)=1$ and
$\delta(j,i)=3$. Let $Q_i^c=(1,2)$, $K_j^c=(3,1)$, $K_1^r=(0,1)$,
and $Q_3^r=(1,-1)$. The products are $5$, $2$, and $2$. Before softmax,
their sum is scaled by $\sqrt{3d_h}$:
$9/\sqrt6\approx3.674235$. This is not yet an attention weight; scores
for the row's other keys are required.

The code expands the example to three keys, applies softmax, and obtains
a weighted sum of V. Q/K/V and relative vectors are already projected;
this is one head's computation, not a complete DeBERTa implementation.

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

### Enhanced Mask Decoder

EMD adds absolute positions when reconstructing MLM tokens, after the
main relative-position processing. The authors motivate this with
`a new [MASK] opened beside the new [MASK]`: both blanks follow
`new`, but serve different roles. This motivates an additional feature;
it does not prove that a complete relative-position model is fundamentally
unable to distinguish the blanks.

In [Figure 2 and Appendix A.8](https://arxiv.org/pdf/2006.03654v6#page=20),
EMD has two inputs: $H$ contains contextual K/V states, and $I$ supplies
query states. After one pass, updated $I$ reads the same memory $H$
again. The paper's experiment uses two passes with shared parameters,
followed by the MLM head.

In the pinned official implementation, when
`position_biased_input=False`, $H$ comes from the penultimate encoder
output. Initial queries add absolute positional vectors to that $H$.
The final Transformer block is called twice with the same memory, each
time replacing query states with its output for the next pass.

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

For example, two blanks receive different initial position vectors and
can therefore form different Q with the same contextual memory. On the
second pass, Q also depends on the first read's result. This is another
attention transformation, not the addition of two finished vocabulary
logits or one bias before softmax.

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

This demonstrates changing queries with fixed memory, not a numerical
reproduction of the full DeBERTa block. EMD reads retained context using
the final Transformer block's learned projections, relative scores,
normalization, and FFN. No new token is inserted between reads; the
vocabulary head receives the second pass's result.

There is no separate target sequence or causal generation here. After
training, classification or NER uses the encoder; “decoder” in EMD
refers to reconstruction of MLM targets. The original mechanism is
also explained in the [authors' ICLR 2021 slides](https://iclr.cc/media/Slides/iclr/2021/virtual%2804-00-00%29-04-00-00UTC-2562-deberta_decodi.pdf).

### Fine-tuning, results, and versions

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

Version boundaries also matter. The [official repository](https://github.com/microsoft/DeBERTa)
separately describes V2: a new 128K SentencePiece vocabulary, an added
convolution, shared content/position projections, and relative buckets.
V3 changes pre-training to ELECTRA-style replaced-token detection and
gradient-disentangled embedding sharing. The formulas and MLM step
above describe the original version, not a complete V3 recipe.

## Limitations and legacy

Full attention processes known text, but MLM does not define an ordinary
autoregressive factorization of its probability. Free generation needs
another procedure or different training. Only selected positions receive
direct MLM loss, while the full attention matrix is still computed
for every real position.

The 512 limit concerns subword input length including special tokens,
not 512 document words. For long text, window selection, overlap,
answer merging, and index offsets become part of the method. Quality
on short GLUE examples does not evaluate these decisions.

On small labeled datasets, fine-tuning depends on head initialization,
data order, and hyperparameters. BERT selects runs using dev results;
RoBERTa's replication tables report medians of five runs. These
protocols belong beside the number; the best run should not be
presented as expected performance.

Pre-training transfers corpus properties, including errors and social
biases. A mean benchmark result does not establish robustness to a
new domain or the correctness of every prediction.

The [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|next chapter, GPT-1]],
examines causal pre-training and supervised fine-tuning.

## Sources and further reading

- Jacob Devlin et al., [BERT, §§3–5, Appendices A.1–A.5 and C.2](https://aclanthology.org/N19-1423/).
- Yinhan Liu et al., [RoBERTa v1, §§3–5, Tables 1–7](https://arxiv.org/abs/1907.11692v1).
- Pengcheng He et al., [DeBERTa v6, §§3–5, Appendices A.7–A.8](https://arxiv.org/abs/2006.03654v6).
- Google Research, [BERT Whole Word Masking release and generator](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/README.md#L74-L118), Apache-2.0. The original explanation is quoted above; the reduced selector and budget examples are added. [Full license](/licenses/google-bert-eedf5716-Apache-2.0.txt).
- Mandar Joshi et al., [SpanBERT, sections 3, 4.2–4.3 and 6, Figures 1–2, Tables 6–7](https://aclanthology.org/2020.tacl-1.5/), CC BY 4.0. Original figures/captions and the boundary-input passage retain their language. The equations use $h$ for contextual states; the small PyTorch module and probability calculations are added.
- Zhenzhong Lan et al., [ALBERT v6, sections 3.1 and 4.6, Table 5](https://arxiv.org/abs/1909.11942v6). Pair construction and legacy feature naming are also checked against [the pinned original generator](https://github.com/google-research/albert/blob/b772393d3dae115b493258ce8e37c17b2cc62100/create_pretraining_data.py), Copyright 2019 The Google Research Authors, Apache-2.0. The swap excerpt is unchanged; the door example and supplied-logit calculation are added. [Full code license](/licenses/google-albert-b772393d-Apache-2.0.txt).
- Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/).
- Lena Voita, [Transfer Learning — BERT and fine-tuning](https://lena-voita.github.io/nlp_course/transfer_learning.html).
- [Official BERT: model and data preparation](https://github.com/google-research/bert/tree/eedf5716ce1268e56f0a50264a88cafad334ac61).
- [Official DeBERTa: attention and MLM](https://github.com/microsoft/DeBERTa/tree/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098).
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-only|Encoder-only reference]].
- [[02 Areas/ML & DL/04 Вопросы/100 вопросов по NLP|NLP questions]].

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three architectural patterns]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1]].
