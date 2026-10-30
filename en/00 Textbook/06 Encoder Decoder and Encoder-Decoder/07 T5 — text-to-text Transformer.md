---
title: T5 — text-to-text Transformer
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "1bced617c9985be26919d79c4d01a87e4d0a3087d59e65d2e200fd2f84608dfc"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer.md"
last_updated: 2026-10-29
primary_sources:
  - https://jmlr.org/papers/v21/20-074.html
  - https://github.com/google-research/text-to-text-transfer-transformer/tree/42909aa15d1fbd16c69333aa474a00ae8ed9fde4
  - https://github.com/huggingface/transformers/tree/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/t5
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning]]"
next: "[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура]]"
---

# T5: the text-to-text Transformer

To judge grammatical acceptability, T5 is trained to output
`acceptable` or `not acceptable`. Translation outputs a translated
sentence; summarization outputs a summary. In every case, the decoder
predicts text tokens, including an end token. No separate classification
head is needed for each dataset.

Text-to-Text Transfer Transformer names both this representation of
tasks and an encoder–decoder model. Raffel et al. use the common
format to compare architectures, pre-training objectives, corpora,
transfer strategies, and model sizes.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/t5-figure1.png]]

*Raffel et al., [Exploring the Limits of Transfer Learning, Figure 1, page 3](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=3).
Original translation, CoLA, STS-B, and summarization examples. English
strings and caption are preserved,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

A prefix such as `translate English to German:` or `cola sentence:`
is part of training input. Its relationship to the desired answer is
learned from such pairs. This differs from
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3]],
where examples specify a task in current context without weight
updates. A short prefix alone does not make original T5 reliably
follow any new instruction.

## Architecture: source and answer have different roles

In translation, the whole source sentence is available before generation.
The encoder therefore reads it with bidirectional self-attention.
The decoder receives previous translation tokens and accesses encoder
output through cross-attention. Future translation words remain hidden.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/t5-figure4.png]]

*Raffel et al., [Figure 4, page 16](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=16).
Encoder–decoder on the left, ordinary language model in the center,
prefix LM on the right. Lines represent available attention connections,
not weight-update operations. Original figure and caption, CC BY 4.0.*

Let input $x$ contain $S$ tokens. A shared token table maps them to
width-$d$ vectors, and the encoder returns memory
$H^{enc}\in\mathbb R^{B\times S\times d}$. While generating
$y_1,\ldots,y_T$, the decoder uses this memory at each layer:

$$
P_\theta(y\mid x)=\prod_{t=1}^{T}
P_\theta(y_t\mid y_{<t},H^{enc}(x)).
$$

Decoder self-attention obtains Q/K/V from decoder states.
Cross-attention obtains queries from the decoder and keys/values from
encoder memory. With $h$ heads, scores have shapes $[B,h,T,T]$ and
$[B,h,T,S]$ respectively. After the last decoder block, normalization
and a shared vocabulary projection produce logits $[B,T,V]$.

### Input and answer visibility

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/t5-figure3.png]]

*Raffel et al., [Figure 3, page 15](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=15).
Dark cells allow attention. Rows are numbered bottom to top, which
must be accounted for when reading the causal triangle. Original
figure and caption, CC BY 4.0.*

The encoder uses the left pattern. All connections between real input
tokens are permitted; added padding is excluded separately.
Decoder row $i$ reads positions $0,\ldots,i$ under a causal mask.
In cross-attention, every decoder position may read all nonpadding
encoder positions. Its rectangular mask does not impose $j\le i$:
source and target lengths and positions are independent.

With three source tokens and two answer positions, cross-attention
has two rows and three columns, both rows able to read all three
source tokens. Decoder self-attention's first row reads only the
first decoder input; the second reads the first two. The source may
contain words to the right of a blank. That is permitted reconstruction
context, not future-target leakage.

The prefix LM on Figure 3's right gives a prefix bidirectional
attention within one stack. T5 has a separate encoder stack and
separate cross-attention projections. Their information access is
similar, but changing one mask does not make them the same implementation.

### Shift decoder input, not the original targets

The released tokenizer encodes `not acceptable` as `[59, 9961, 1]`,
where 1 is `</s>`. Training feeds the decoder `[0, 59, 9961]`.
ID 0, ordinarily PAD, serves as a start token here.

| Position | Input | Target |
|---|---|---|
| 0 | PAD, 0 | not, 59 |
| 1 | not, 59 | acceptable, 9961 |
| 2 | acceptable, 9961 | EOS, 1 |

At position 1, the model knows `not` but has not seen `acceptable`.
All three positions are computed in parallel; causal masking preserves
the same conditioning as sequential generation. Decoder input is
shifted, while cross-entropy compares logits at $t$ with the original
target at the same $t$.

If conditional probabilities of the three correct tokens are $0.4$,
$0.5$, and $0.8$, the full-string probability is $0.16$.
Total negative log-likelihood is $-\ln0.16\approx1.8326$, or $0.6109$
averaged over three tokens. These are illustrative values, not T5
measurements. The EOS target teaches stopping after the label.

Target-text padding must be excluded from batch loss. Transformers
uses label `-100` for this, not the vocabulary PAD ID.
These have different roles: `-100` is not an input token and becomes
PAD when targets shift. The starting PAD remains a real, visible
decoder position. A `decoder_input_ids != 0` mask would wrongly
remove it.

## How a T5 block differs from the original Transformer

An encoder block applies self-attention followed by an FFN;
a decoder block inserts cross-attention between them. Each sublayer
receives normalized input, then adds its output to the unnormalized
residual path. Without dropout, an encoder block is

$$
\begin{aligned}
u&=x+\operatorname{SelfAttn}(R(x)),\\
z&=u+\operatorname{FFN}(R(u)).
\end{aligned}
$$

Here $R$ normalizes without mean subtraction or an added bias:

$$
R(x)_i=\gamma_i\frac{x_i}
{\sqrt{\frac1d\sum_{j=1}^{d}x_j^2+\epsilon}}.
$$

For $x=(1,2)$, unit $\gamma$, and negligible $\epsilon$, the output
is approximately $(0.6325,1.2649)$. Its mean is not zero.
Ordinary LayerNorm first subtracts 1.5 and gives approximately
$(-1,1)$. Substituting it would change T5's function even if layer
placement stayed the same.

Original T5's FFN is $\operatorname{ReLU}(xW_i)W_o$.
In the pinned implementation, attention and FFN projections have no
bias. Each stack also ends with normalization. Encoder token embeddings,
decoder token embeddings, and vocabulary output weights are tied.
This shares one table; ordinary T5's encoder and decoder blocks have
separate parameters. Sharing whole-stack parameters is a separate ablation.

Before tied vocabulary output, states are multiplied by $d^{-1/2}$.
The pinned attention implementation computes $QK^\top$, then adds
bias and mask; it has no extra division by $\sqrt{d_{kv}}$.
Projection scaling is accounted for in initialization. Mechanically
importing another implementation's scale factor would change logits
([`modeling_t5.py`](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/t5/modeling_t5.py);
general architecture: paper §2.1).

### Relative positions change scores, not token vectors

T5 does not add absolute position vectors to embeddings. For each
query/key pair, it computes offset $r=j-i$ and bucket $b(r)$.
Each head learns a table of scalar biases:

$$
s_{ij}^{(a)}=q_i^{(a)}k_j^{(a)\top}+
\beta_{a,b(j-i)}+M_{ij}.
$$

$M$ forbids unavailable connections; $\beta$ changes preferences
among allowed ones. If two content scores are zero and biases are
0 and $\ln3$, softmax gives weights $0.25$ and $0.75$.
Positions can affect attention before content scores differ.

The encoder uses 32 buckets split by direction. Within a direction,
distances 0–7 are exact; farther distances enter logarithmically
widening intervals. Positive $j-i$ gives:

| Offset $j-i$ | Encoder bucket |
|---:|---:|
| 0 | 0 |
| 1 | 17 |
| 2 | 18 |
| 8 | 24 |
| 16 | 26 |
| 64 | 30 |
| 128 | 31 |

Negative offsets use the other half. The decoder has only the
past-facing direction, so its 32 buckets are not halved and exact
distances cover 0–15. A causal mask blocks future positions;
bucket assignment itself does not replace that mask.

All distances of 128 or more use the boundary bucket for their
direction; rounding may place some shorter distances there too.
This is not a 128-token attention window. An encoder position can
still read the whole source; several distant positions simply get
the same relative bias. A stack shares its bias table across its
layers, but encoder and decoder have separate tables. Cross-attention
in the pinned implementation adds no such learned bias.

### Head width need not be model width divided by head count

For T5-Base, $d=768$, $h=12$, and $d_{kv}=64$, so combined head width
$a=h d_{kv}=768$. But T5-3B uses $d=1024$, $h=32$, and $d_{kv}=128$,
giving $a=4096$. At 11B, 128 heads give $a=16384$.
$W_Q,W_K,W_V$ project width $d$ into $a$; output projection returns
$a$ to $d$. Block counts below are for each of the two stacks.

| Model | Blocks | $d$ | FFN |
|---|---:|---:|---:|
| Small | 6 | 512 | 2048 |
| Base | 12 | 768 | 3072 |
| Large | 24 | 1024 | 4096 |
| 3B | 24 | 1024 | 16384 |
| 11B | 24 | 1024 | 65536 |

These configurations are described in
[§3.7](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=36)
and fixed in official gin files. A Base encoder block contains
$4da+2df+2d$ parameters; a decoder block contains $8da+2df+3d$.
Here $f$ is FFN width and the last term counts normalization scales.
The shared token table adds $Vd$, final normalizations $2d$, and
two bias tables $64h$. With $V=32128$, total parameters are
222,903,552. “220M” is a rounded name.

SentencePiece has 32,000 entries; with 100 sentinels, the tokenizer
exposes 32,100 IDs. The configuration pads table rows to 32,128.
This distinguishes tokenizer size from matrix size; the extra 28
are not ordinary subwords. The tokenizer is trained on English,
German, French, and Romanian at ratio 10:1:1:1, while the model's
main C4 pre-training corpus remains English.

## Pre-training: reconstruct deleted spans

Ordinary text supplies a task without manual labels: delete some
tokens and use them as correct answers. T5 replaces every contiguous
deleted span with its own sentinel, allowing the decoder to associate
reconstructed content with the right location.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5-2026-10-22/t5-figure2-spans.png]]

*Raffel et al., [Figure 2, page 13](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=13).
The encoder sees two gaps; the decoder reconstructs the removed spans.
CC BY 4.0.*

<details>
<summary>Original Figure 2 caption</summary>

> Schematic of the objective we use in our baseline model. In this example, we
> process the sentence “Thank you for inviting me to your party last week.” The
> words “for”, “inviting” and “last” (marked with an ×) are randomly chosen for
> corruption. Each consecutive span of corrupted tokens is replaced by a sentinel
> token (shown as `<X>` and `<Y>`) that is unique over the example. Since “for” and
> “inviting” occur consecutively, they are replaced by a single sentinel `<X>`. The
> output sequence then consists of the dropped-out spans, delimited by the sentinel
> tokens used to replace them in the input plus a final sentinel token `<Z>`.

</details>

The example removes `for inviting` and `last`. Using the released
tokenizer's notation, encoder input is:

~~~text
Thank you <extra_id_0> me to your party <extra_id_1> week.
~~~

The decoder target is:

~~~text
<extra_id_0> for inviting <extra_id_1> last <extra_id_2>
~~~

EOS is appended to both tokenized sequences. With the archived
SentencePiece model and T5Tokenizer's `legacy=True`, exact IDs are:

~~~text
encoder: [1562, 25, 32099, 140, 12, 39, 1088, 32098, 471, 5, 1]
target:  [32099, 21, 14256, 32098, 336, 32097, 1]
~~~

`<extra_id_0>` has ID 32099, `<extra_id_1>` has 32098, and
`<extra_id_99>` has 32000: IDs descend. A sentinel is one special
token despite its long textual spelling. The final `<extra_id_2>`
does not identify a third deleted word here; it closes the target
list before EOS.

### Converting a mask into two sequences

First check the transformation on short toy IDs. Let original tokens
be `[2,3,4,5,6,7,8]`, deleting `3,4` and `7`.
Reserve 14, 13, and 12 for sentinels, and 1 for EOS.

~~~python
def replace_runs(tokens, replace_mask, first_sentinel=14):
    result = []
    sentinel = first_sentinel
    for i, (token, replace) in enumerate(zip(tokens, replace_mask)):
        if replace:
            if i == 0 or not replace_mask[i - 1]:
                result.append(sentinel)
                sentinel -= 1
        else:
            result.append(token)
    return result

tokens = [2, 3, 4, 5, 6, 7, 8]
noise = [False, True, True, False, False, True, False]
source = replace_runs(tokens, noise) + [1]
target = replace_runs(tokens, [not x for x in noise]) + [1]
print(source)  # [2, 14, 5, 6, 13, 8, 1]
print(target)  # [14, 3, 4, 13, 7, 12, 1]
~~~

Source replaces deleted runs; target replaces retained runs.
The target therefore contains the missing text, while the clean
suffix becomes a closing sentinel. Reconstruction is straightforward:
between 14 and 13 lies the first gap's text; between 13 and 12 lies
the second's. Replacing source sentinels with those spans recovers
the original sequence.

This is a teaching transformation for a specified mask. It neither
samples the mask nor reproduces the official TensorFlow processor's
random seeds. The manually chosen example also does not demonstrate
15% noise or a mean span length of 3.

### Independent baseline deletions versus final T5's sampled spans

In the paper's baseline (§3.1.4), each token is independently deleted
with probability 15%, then neighboring deleted tokens are merged.
The resulting number and average length of spans are not fixed in advance.

Final models (§3.3.4, §3.7) specify noisy-token count $m$ and noisy-span
count $s$. For $N\ge2$, noise rate 0.15, and mean length 3,
the standard helper uses

$$
\begin{aligned}
m&=\operatorname{clip}(\operatorname{round}(0.15N),1,N-1),\\
s&=\max(1,\operatorname{round}(m/3)).
\end{aligned}
$$

It partitions $m$ into $s$ positive lengths, independently partitions
$N-m$ clean tokens into $s$ positive lengths, and alternates the parts.
At $N=40$, $m=6$ and $s=2$. Noise lengths $(2,4)$ and clean lengths
$(17,17)$ are possible: 17 clean, 2 noisy, 17 clean, 4 noisy.
Mean noise length is 3 although neither span has length 3.

[Official `random_spans_noise_mask`](https://github.com/google-research/text-to-text-transfer-transformer/blob/42909aa15d1fbd16c69333aa474a00ae8ed9fde4/t5/data/preprocessors.py)
with `random_roll=False` starts clean and ends noisy.
Paired `nonnoise_span_to_unique_sentinel` then yields a target
ending with the last reconstructed span and EOS; there may be no
separate closing sentinel afterward. Figure 2 ends with clean text
and therefore includes an extra sentinel. Reproducing preprocessing
requires checking the mask, transformation, and EOS addition together,
not insisting that every example literally end in `<extra_id_2>`.

### Why the decoder gets a short target

For $N$ original tokens, $m$ deletions, and $s$ noisy spans,
source length before EOS is usually $N-m+s$. If the mask starts clean
and ends noisy, target length is $m+s$. At $N=40$, lengths are 36
and 8 before EOS rather than regenerating all 40. This reduces
decoder positions, FFN work, and attention matrices.

The standard length helper selects 568 original tokens to obtain
encoder input length 512 including EOS. It deletes 85 tokens in
28 spans; target length including EOS is 114. Thus “source length
512” and “512 original tokens before corruption” may refer to
different pipeline stages.

## A teaching training step: from batch to weight update

The two pairs below use the preceding example's toy vocabulary.
In the second, original `[3,9,4]` becomes source `[3,14,4,1]`
and target `[14,9,13,1]`: 14 identifies the only gap and 13 closes
it before EOS. This shorter pair is padded. The model is small
and randomly initialized; no pretrained weights are needed.
The code uses Transformers 4.57.6 and ordinary CPU PyTorch.

~~~python
import torch
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

torch.manual_seed(7)
config = T5Config(
    vocab_size=16, d_model=8, d_kv=4, d_ff=16,
    num_layers=2, num_decoder_layers=2, num_heads=2,
    dropout_rate=0., pad_token_id=0, eos_token_id=1,
    decoder_start_token_id=0, feed_forward_proj="relu",
)
model = T5ForConditionalGeneration(config)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
inputs = torch.tensor([[2, 14, 5, 6, 13, 8, 1],
                       [3, 14, 4, 1, 0, 0, 0]])
labels = torch.tensor([[14, 3, 4, 13, 7, 12, 1],
                       [14, 9, 13, 1, -100, -100, -100]])
source_mask = inputs.ne(0).long()
decoder_mask = labels.ne(-100).long()
decoder_ids = model._shift_right(labels)
print(decoder_ids.tolist())
# [[0, 14, 3, 4, 13, 7, 12], [0, 14, 9, 13, 1, 0, 0]]

def forward():
    return model(
        input_ids=inputs, attention_mask=source_mask,
        decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask,
        labels=labels, use_cache=False,
    )

out = forward()
manual_loss = F.cross_entropy(
    out.logits.reshape(-1, 16), labels.reshape(-1), ignore_index=-100)
torch.testing.assert_close(out.loss, manual_loss)
assert model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()
before = model.shared.weight.detach().clone()
optimizer.zero_grad()
out.loss.backward()
optimizer.step()
with torch.no_grad():
    after_loss = forward().loss
print(tuple(out.logits.shape), int(labels.ne(-100).sum()))
print(float(out.loss.detach()), float(after_loss))
assert not torch.equal(before, model.shared.weight)
~~~

The encoder processes two length-7 sources; the second has four real
tokens. The decoder also has seven positions in each row, but loss
uses only 11 target tokens: seven in the first, four in the second.
Both rows use the same parameters. Gradients update encoder, decoder,
and shared vocabulary table; address equality checks weight tying.

`_shift_right` is shown as a detail of this pinned implementation,
not a universal public API across versions. Supplying only `labels`
would let this implementation prepare the shift itself. It is made
explicit here to check input and mask.

This executes one step on an artificial batch. SGD and the small
configuration are chosen to check computation. The paper uses
AdaFactor, large corpora, and many more updates; this example does
not reproduce released T5 quality.

<span id="почему-авторы-выбрали-именно-эту-цель"></span>

## How pre-training objectives were compared

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5-2026-10-22/t5-figure5-objectives.png]]

*Raffel et al., [Figure 5, page 24](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=24).
Objective experiments proceed through general formulation, corruption
method, noise rate, and span length. CC BY 4.0.*

<details>
<summary>Original Figure 5 caption</summary>

> A flow chart of our exploration of unsupervised objectives. We first consider a
> few disparate approaches in Section 3.3.1 and find that a BERT-style denoising
> objective performs best. Then, we consider various methods for simplifying the
> BERT objective so that it produces shorter target sequences in Section 3.3.2.
> Given that replacing dropped-out spans with sentinel tokens performs well and
> results in short target sequences, in Section 3.3.3 we experiment with different
> corruption rates. Finally, we evaluate an objective that intentionally corrupts
> contiguous spans of tokens in Section 3.3.4.

</details>

The paper first compares suffix prediction, BERT-style corruption,
and reconstruction of shuffled text. In the encoder–decoder
BERT-style task, the decoder reconstructs the whole original text.
This is not original BERT's loss at selected encoder positions.
Deshuffling performs substantially worse in this comparison.

Next, targets are shortened: predict only removed spans identified
by sentinels, or only removed tokens, instead of the complete text.
Table 5 combines similar results with different costs. For example,
span replacement gives SuperGLUE 71.36 versus 68.67 for deletion
without markers, although deletion does better on GLUE. One favorable
aggregate therefore does not select an objective for every task.

Mean span length 3 gives SuperGLUE 72.53 and SQuAD EM 81.84, versus
71.36 and 80.88 for independent-deletion baseline (Table 7).
Translation does not improve in every direction. Choosing spans
around length 3 therefore balances transfer and short sequences,
rather than proving optimality across every dataset and scale.

The architectural comparison has a separate limitation (§3.2.2):
a model with $L$ encoder and $L$ decoder layers cannot simultaneously
be matched to one stack in parameters and computation. The authors
approximately compare $L+L$ encoder–decoder with an $L$-layer LM
by input/target-pair processing cost; the latter has substantially
fewer parameters. Sharing stack weights reduces parameters but not
executed layers. These conditions must not be omitted to turn
Table 2 into an unconditional architecture ranking.

<span id="c4-масштаб-начинается-с-отбора-данных"></span>

## C4 and the training regime

C4 starts from the April 2019 Common Crawl snapshot. Filters retain
English with langdetect confidence at least 0.99, lines of at least
three words with terminal punctuation, and pages with at least five
sentences. Pages containing `lorem ipsum`, a curly bracket, or a
word from a selected prohibited-word list are removed; lines containing
`Javascript` are also removed. Duplicate three-sentence spans are
kept once ([§2.2](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=6)).

These mechanical rules do not verify truth. A curly bracket, for
example, can eliminate a useful explanation containing a short code
sample. Word-list filtering ignores the context of word use.
“Clean” means the result of chosen heuristics, not error-free or
neutral documents.

In Table 8, ordinary C4 occupies 745 GB; a version without these
heuristics occupies 6.1 TB while retaining language filtering.
Switching the baseline to this larger uncleaned dataset reduces
SuperGLUE from 71.36 to 68.04. Wikipedia plus Toronto Books Corpus,
on the other hand, gives 73.24: alignment between pre-training domain
and task also affects results.

### The controlled baseline and final series

The baseline trains for $2^{19}=524288$ steps with 128 length-512
sequences per batch: roughly $2^{35}=34.36$ billion token positions.
It uses AdaFactor and dropout 0.1. Its learning-rate schedule is
$1/\sqrt{\max(n,10000)}$: constant 0.01 initially, followed by
inverse-square-root decay after 10,000 steps. This is not linear
warmup to 0.01.

Baseline fine-tuning runs $2^{18}=262144$ steps at constant rate 0.001.
Checkpoints are saved every 5000 steps and selected by validation.
At this stage, GLUE and SuperGLUE are each treated as mixtures of
their constituent tasks.

Final models pretrain for one million steps with batches of $2^{11}$
length-512 sequences, approximately 1.05 trillion token positions.
The final variant replaces independent deletions with span corruption,
mixes labeled and unlabeled tasks, then undergoes separate fine-tuning.
For individual GLUE/SuperGLUE tasks, the batch falls to eight sequences
and checkpoints are saved every 1000 steps to better capture the
state before overfitting.

A large C4 supports this data stream without hundreds of repeats
of a small corpus. In Table 9, keeping total training at $2^{35}$
tokens while using only $2^{23}$ distinct tokens means 4096 passes.
Training loss falls, but SuperGLUE drops to 59.29 from the full
corpus's 71.36.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5-2026-10-22/t5-figure6-repeated-data.png]]

*Raffel et al., [Figure 6, page 29](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=29).
Pre-training losses on full and artificially reduced C4 corpora.
Legend sizes are in tokens; the horizontal axis is thousands of steps.
Original plot and full English caption, CC BY 4.0.*

Every curve has the same total training budget, but smaller corpora
repeat more often. The $2^{23}$ curve falls lowest: the model predicts
hidden spans better in texts seen repeatedly. The plot itself shows
training loss only. Reduced transfer is established separately by
Table 9's downstream results. Together, these observations are
consistent with corpus memorization; remasking does not create new
source documents. Quality declines little after 64 repeats in this
series, so the result does not mean that all data repetition is harmful.

<span id="перенос-задача-становится-данными-а-не-новой-головой"></span>

## Fine-tuning, task mixtures, and answer evaluation

Supervised fine-tuning prepares source/target pairs. CoLA generates a
label; SQuAD generates answer text from a question and context;
CNN/Daily Mail generates a summary. STS-B rounds scores to increments
of 0.2 and writes them as strings: 2.57 becomes `2.6`.
This discretizes regression; nearby numbers may now share one target.

After generation, outputs are converted to metric format. A label
string is matched against allowed classes; an invalid string counts
as an error. SQuAD uses exact match and token F1, summaries use
ROUGE, and translation uses BLEU. A common vocabulary head does not
make these criteria identical. EOS and label wording also enter
training probability.

One encoder–decoder can train on mixed records. But sampling all
datasets proportional to size would nearly displace small supervised
tasks with enormous C4. Section 3.5.2 introduces an artificial
dataset-size cap $K$:

$$
\begin{aligned}
p_i&=\frac{\min(n_i,K)}{\sum_j\min(n_j,K)},\\
p_i^{(T)}&=\frac{p_i^{1/T}}{\sum_jp_j^{1/T}}.
\end{aligned}
$$

For toy datasets of 100 million, 1 million, and 10,000 examples,
ordinary mixing gives approximately 99.00%, 0.99%, and 0.01%.
Capping at $K=1$ million gives 49.75%, 49.75%, and 0.50%.
Applying $T=2$ afterward gives 47.62%, 47.62%, and 4.76%.
The small dataset appears more often, but its examples repeat more.
This illustrates the formula, not final T5 mixture weights.
The temperature-mixing experiment uses $K=2^{21}$.

Final models choose the unlabeled task's effective size separately:
710,000 for Small through 133 million for 11B, with 2.62 million
for Base. En–Fr and En–De effective pre-training sizes are capped at
one million examples. These choices and separate fine-tuning are part
of Section 3.7's result. “Multi-task” does not mean that every table
entry comes from one checkpoint without additional training.

### Full updates, adapters, and gradual unfreezing

Section 3.5.1 compares fine-tuning strategies. The ordinary approach
updates every parameter. Adapters insert extra dense–ReLU–dense blocks
after existing FFNs; adapters and normalization parameters change,
while main weights stay fixed. A narrow intermediate representation
reduces trainable parameter count.

At model width 768 and adapter width 32, its two matrices contain
$2\cdot768\cdot32=49152$ weights excluding biases. Width 512 raises
this to 786,432. Small adapters perform worse on high-resource tasks
in the experiment; fewer trainable parameters do not automatically
preserve quality.

Gradual unfreezing starts with upper blocks of both stacks, adding
another encoder and decoder layer approximately every $2^{18}/12$
steps. Shared embeddings update throughout. In this comparison,
gradual unfreezing slightly trails full fine-tuning while reducing
some computation. That concerns a particular schedule and datasets,
not a prohibition on frozen layers in other regimes.

<span id="как-читать-ключевые-эксперименты"></span>

## Scaling and results

Section 3.6 compares ways to spend approximately $4\times$ additional
compute. The following values are validation results from one
experimental series, not final test results:

| Method | GLUE | SuperGLUE |
|---|---:|---:|
| baseline | 83.28 | 71.36 |
| same model, four times as many steps | 85.33 | 74.72 |
| twice as large, twice as many steps | 86.18 | 77.18 |
| four times as large, original step count | 85.91 | 78.04 |
| ensemble of four complete runs | 84.77 | 71.74 |

Increasing batch size alone also helps but is not equivalent to more
updates. Ensembling improves some generative tasks more than other
uses of compute, but yields little SuperGLUE gain here. Larger models
also raise later fine-tuning and inference cost; extra pre-training
of a smaller model does not enlarge it in deployment.

Final Table 14 gives T5-11B GLUE 90.3 and SuperGLUE 88.9 on test
servers, SQuAD EM 91.26 on validation, and CNN/Daily Mail ROUGE-2
21.55 on test. Translation uses separate WMT test sets and does not
exceed the listed best specialized systems. Their training data
differ, including use of back-translation, so this is not a clean
architecture ablation.

Table 15 separately compares baseline, baseline trained on a trillion
tokens, and T5-Base on validation. SuperGLUE scores are 71.36, 73.90,
and 75.64. Longer baseline training alone therefore does not reproduce
all T5-Base changes. Nor do these differences isolate each setting's
individual contribution: several changes enter the final recipe.

SuperGLUE 88.9 beside human baseline 89.8 does not mean an 88.9%
probability of understanding arbitrary text. It aggregates particular
tasks. For example, T5-11B has COPA 94.8 and WSC 93.8 versus the
reported human 100%. ROUGE likewise checks reference overlap,
not factual support for every summary sentence.

## Inference cost and practical trade-offs

At inference, the encoder processes the source once. Each decoder
layer separately projects encoder memory into cross-attention keys
and values and retains them. Each new token then uses those cross-KV
and extends its own self-KV cache.

### What can be reused across requests?

For unchanged source, encoder memory and each decoder layer's cross-KV
may be reused. A new answer to the same document starts a fresh self-KV
cache because that cache records the previous answer's history.
Weights do not change. Model, positions, and source masks must match;
equal length or a few matching initial tokens is insufficient.

Suppose source changes from `[a,b,c]` to `[a,b,d]`. In a causal GPT,
the first two states do not depend on the third, so their KV may
be retained under otherwise equal conditions. In T5's bidirectional
encoder, early positions also read the last token. Replacing `c`
with `d` can change all completed encoder memory, including states
of `a` and `b`. A shared text prefix therefore does not authorize
reuse of its completed encoder states or cross-KV.

Changing the source also requires clearing decoder self-KV.
Upper-layer decoder states were already computed using cross-attention
to the old source. Replacing only cross-KV does not remove that dependency.

Transformers 4.57.6 does not automatically enforce this requirement
on every call. Once cross-cache is filled,
`EncoderDecoderCache.is_updated` permits retained keys and values
without reprojecting `encoder_hidden_states`. Supplying new encoder
memory with an old cache can therefore make the decoder continue
answering from the old source. New source requires a fresh decoder
cache, not merely new `encoder_outputs`
([pinned T5Attention.forward branch](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/t5/modeling_t5.py#L496)).

### Memory and computation

For Base, $S=512$, $T=128$ already processed decoder positions, and
FP16, self-KV requires
$2\cdot12\cdot128\cdot768\cdot2=4718592$ bytes, or 4.5 MiB.
Cross-KV requires
$2\cdot12\cdot512\cdot768\cdot2=18874368$ bytes, or 18 MiB.
Encoder memory adds 0.75 MiB. These exclude weights and temporary
tensors and are not measured peak serving memory.

Training includes quadratic terms in $S$ for encoder self-attention
and in $T$ for decoder self-attention, plus an $ST$ cross-attention
term. A short target reduces the last two, not the cost of encoding
a long document. For 3B and 11B, KV calculations use $h d_{kv}$,
not automatically $d_{model}$.

Final experiments use beam width 4 and length penalty 0.6 for long
answers; controlled experiments use greedy decoding. Beams retain
different continuations and add decoder work. Shared source permits
reuse of encoder computation and cross-KV; actual memory duplication
depends on implementation.

For closed classification, an encoder-only linear head may be cheaper
than sequential label generation. For translation and summaries,
encoder–decoder separates known input from generated text. Selection
requires source and answer lengths, latency constraints, and quality
on relevant data; parameter count alone does not settle it.

## Limitations and legacy

Original T5 trains on a web corpus with specific filtering and limited
language coverage. Textual answers may contain errors; a shared loss
does not verify each conclusion's truth. Text-to-text formatting also
does not itself eliminate possible pre-training/evaluation overlap.

Later model names identify different changes.
[Official T5.1.1 documentation](https://github.com/google-research/text-to-text-transfer-transformer/blob/42909aa15d1fbd16c69333aa474a00ae8ed9fde4/released_checkpoints.md)
lists GEGLU instead of ReLU, disabled pre-training dropout, C4-only
training without a downstream mixture, and untied embedding/output
matrices. Dropout is recommended again for fine-tuning. XL/XXL also
have different shapes from original 3B/11B. Such a checkpoint cannot
silently substitute for the original recipe in a reproduction.

mT5, ByT5, and FLAN-T5 are covered separately in the
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/T5|T5 atlas]]:
they change language data, input representation, or post-training.
Task prefixes, supervised-task mixture training, and generalization
to unseen instructions must be evaluated as distinct properties.

The reading route now returns to decoder-only models:
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA]].

## Sources and further reading

- [Raffel et al., Exploring the Limits of Transfer Learning](https://jmlr.org/papers/v21/20-074.html): §2 architecture, C4, formats; §§3.1–3.7 baseline, objectives, data, transfer, final experiments. Figures 1–6 are reused under CC BY 4.0.
- [Official T5 repository, pinned revision](https://github.com/google-research/text-to-text-transfer-transformer/tree/42909aa15d1fbd16c69333aa474a00ae8ed9fde4): preprocessors, postprocessors, model configurations, and checkpoint descriptions.
- [Transformers T5, pinned implementation](https://github.com/huggingface/transformers/tree/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/t5): norms, relative buckets, decoder shift, weight tying, and loss. The CPU example uses this implementation.
- [Stanford CS224N 2023, Lecture 9, slides 36–38](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf#page=36): from continuation to encoder–decoder infilling.
- [[02 Areas/ML & DL/01 Справочник/Attention/Self-Attention|Self-attention reference]].
- [[02 Areas/ML & DL/02 Атлас моделей/Семейства/T5|T5 family]].
- [[02 Areas/ML & DL/Papers/T5|Detailed paper and comparison-protocol note]].

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3]].
**Next:** [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA as a baseline architecture]].
