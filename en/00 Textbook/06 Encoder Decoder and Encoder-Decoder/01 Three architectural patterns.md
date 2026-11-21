---
title: Three Transformer architectures
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "506410800209b6d39a9f1e3d005f84cea0d2721faf7c8aabff5d859fd61cc60f"
translation_reviewed: "2026-11-21"
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна.md"
last_updated: 2026-11-21
primary_sources:
  - https://aclanthology.org/N19-1423/
  - https://jmlr.org/papers/v21/20-074.html
  - https://arxiv.org/abs/1905.03197
  - https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa]]"
---

# Three Transformer architectures

<a id="one-example-for-the-whole-chapter"></a>

Take the Russian sentence `Кошка не стала есть корм` (“The cat did not eat the food”). BERT can fill in `есть` (“eat”) when given words on either side of `[MASK]`. A causal language model can predict the same word after `Кошка не стала`, before it has seen `корм` (“food”). In translation, the whole Russian sentence is available. The decoder can predict `not` from it and the English prefix `The cat did`.

| Where the prediction is made | Tokens it can read |
| --- | --- |
| BERT: `[MASK]` → `есть` | `Кошка не стала [MASK] корм` |
| Causal model: `стала` → `есть` | `Кошка не стала` |
| Encoder–decoder: `did` → `not` | Full Russian source and `The cat did` |

An encoder and a causal decoder each use one stack. The translator uses two:
its decoder reads encoder output through cross-attention, as in the
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|previous chapter]].
Here each word stands for one token. BERT and T5 positions refer to subwords.

<a id="the-visibility-matrix"></a>

<a id="главный-переключатель-матрица-видимости"></a>

## Which keys a query can read

Row $i$ of the attention matrix is a query; column $j$ is a key.
With mask $M$, attention weights are

$$
A=\operatorname{softmax}_{j}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right).
$$

For an allowed connection, $M_{ij}=0$; for a forbidden connection,
$M_{ij}=-\infty$. If a row has at least one allowed key, softmax gives
forbidden connections zero weight and normalizes the rest among
themselves. Ordinary softmax cannot handle a row with no allowed keys;
a separate convention is needed. This chapter's code treats that input
as an error.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/t5-figure3.png|Three T5 attention masks: full, causal, and causal with a bidirectional prefix]]

*Full, causal, and causal-with-bidirectional-prefix masks. The authors
place $y_1$ at the bottom: rows are numbered bottom to top. A printed
tensor usually places its first row at the top, so the triangle appears
vertically reflected. Causality has the same condition in either view:
$j\le i$.
Source: Colin Raffel et al., [T5, Figure 3, §3.2.1](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=15),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The figure and its English caption are unchanged.*

A full mask allows all connections within the real input. A causal mask
lets position $i$ read only keys $j\le i$. In a prefix LM, the first $P$
positions read one another in both directions; subsequent positions read
the entire prefix and the known part of the continuation. This last
variant uses one stack, not a separate encoder with added cross-attention.

Cross-attention usually has a rectangular matrix:
$T_{tgt}\times T_{src}$. Every target row may read all real source keys.
The causal mask restricts links within the target. It does not mask
source words by their position, because the entire source is already known. PAD-key exclusion is added to any of these masks; it does
not replace the visibility rule.

<a id="encoder-only-build-a-contextual-representation"></a>

<a id="encoder-only-bidirectional"></a>

## Encoder-only: context on both sides

In `Кошка не стала есть корм, потому что он был испорчен` (“The cat did not eat the food because it was spoiled”), the query at `он` (“it”) can read `корм` (“food”) to its left and `испорчен` (“spoiled”) to its right in the first layer. In the next layer it can read their updated states. This is appropriate for classification because the full text is known. During BERT pre-training, some inputs are deliberately changed before the model reconstructs the selected tokens.

<a id="how-bert-learns-without-task-labels"></a>

### Reconstructing selected tokens in BERT

Masked language modeling selects a set of positions $\mathcal M$.
The correct tokens are retained separately; the model receives corrupted
sequence $\tilde x$:

```text
original: Кошка не стала есть   корм
input:    Кошка не стала [MASK] корм
targets:                есть   корм
```

This small example selects the last two positions to demonstrate different
replacement cases, not the actual selection rate. One becomes `[MASK]`;
the other stays unchanged. Original BERT selects 15% of WordPiece
positions. Of those selected positions, 80% become `[MASK]`, 10% become
a random token, and 10% remain unchanged. These are proportions of the
**selected** set, not three proportions of the entire text. The vocabulary
classifier reconstructs the original token in all three cases:

$$
\mathcal L_{\mathrm{MLM}}
=-\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).
$$

If the correct tokens `есть` and `корм` receive probabilities $0.2$
and $0.8$, their summed loss is
$-\ln0.2-\ln0.8\approx1.832581$. The unchanged `корм` also contributes.
Labels must therefore not be constructed using `input == MASK`:
that would discard randomly replaced and unchanged selected positions.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png|A vocabulary classifier reconstructs the hidden token from its BERT state]]

*The selected position's state passes through a vocabulary classifier.
The words and the label “all English words” simplify actual WordPiece
tokenization. The target is the original token, not its random replacement;
15% refers to position selection.
Source: Jay Alammar, [The Illustrated BERT, Masked Language Model](https://jalammar.github.io/illustrated-bert/#masked-language-model),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The following code computes this loss for the specified probabilities.
It does not train BERT; the logit table checks target selection and derivatives.

<details>
<summary>Code: MLM loss and shifted language-model targets</summary>

```python
import math
import torch
import torch.nn.functional as F

original = torch.tensor([3, 4, 5, 6, 7])  # Кошка не стала есть корм
selected = torch.tensor([False, False, False, True, True])
corrupted = original.clone()
corrupted[3] = 9                         # [MASK]; position 4 is unchanged
labels = original.masked_fill(~selected, -100)
probs = torch.full((5, 10), 0.1, dtype=torch.float64)
for row, target_prob in [(3, 0.2), (4, 0.8)]:
    probs[row].fill_((1-target_prob)/9)
    probs[row, original[row]] = target_prob
logits = probs.log().requires_grad_()
loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
loss.backward()
torch.testing.assert_close(loss.detach(), torch.tensor(
    -math.log(0.2)-math.log(0.8), dtype=logits.dtype))
assert torch.count_nonzero(logits.grad[~selected]) == 0
assert logits.grad[4, 7] < 0              # unchanged selected token is trained
print("MLM labels", labels.tolist(), "sum loss", float(loss.detach()))

# A causal LM supervises every next token here, including EOS.
BOS, EOS = 1, 2
sequence = torch.tensor([BOS, 3, 4, 5, 6, 7, EOS])
decoder_input, next_labels = sequence[:-1], sequence[1:]
causal_logits = torch.zeros(6, 10, dtype=torch.float64, requires_grad=True)
causal_loss = F.cross_entropy(causal_logits, next_labels, reduction="sum")
causal_loss.backward()
torch.testing.assert_close(causal_loss.detach(), torch.tensor(
    6*math.log(10), dtype=causal_logits.dtype))
print("NTP input/targets", decoder_input.tolist(), next_labels.tolist())
print("NTP supervised positions", len(next_labels))
```

</details>

Zero gradient with respect to output logits at unselected rows does not
mean their input tokens take no part in learning. Their keys and values
may affect predictions at selected positions, carrying gradients back
to their representations.

Original BERT also uses next sentence prediction: given a pair of
segments, distinguish a real continuation from a randomly selected second
segment. This is an additional task on the `[CLS]` state, not a property
of a bidirectional mask. The full MLM/NSP recipe and its changes in
RoBERTa are covered in the
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|next chapter]].
The primary source is [Devlin et al., §3–3.2](https://aclanthology.org/N19-1423/).

<a id="what-comes-out-of-an-encoder"></a>

### Classification, labeling, and span extraction

An encoder returns $H\in\mathbb R^{T\times d}$, one row per position. A document label, a token label, and an answer span are read out from that matrix in different ways.

Whole-text classification can use $h_{\mathrm{[CLS]}}$ and compute
$\operatorname{softmax}(h_{\mathrm{[CLS]}}W_c+b_c)$. For example, with
zero bias, $h=(1,2)$, and columns of $W_c$ equal to $(1,0)$ and $(-1,0)$,
the two logits are $(1,-1)$ and probabilities approximately
$(0.880797,0.119203)$. Training changes this projection and, in ordinary
BERT fine-tuning, the encoder parameters as well.

For token labeling, a classifier acts on every row $h_i$. For extractive
QA, separate projections score the start and end of a span. The model
returns indices in known text instead of generating answer words from
the vocabulary.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/bert-figure4.png|Four BERT task layouts: two classification tasks, answer-span extraction, and token labeling]]

*Two classification variants above; answer-span extraction and token
labeling below. The added output layer changes prediction format while
text passes through the same type of encoder.
Source: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova,
[BERT, Figure 4](https://aclanthology.org/N19-1423.pdf#page=15),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
All four panels and the original caption are preserved.*

For retrieval, rows can be pooled into one vector and trained so that
similarity corresponds to relevance. Merely averaging an MLM model's
outputs does not guarantee that geometry.
[Sentence-BERT](https://arxiv.org/abs/1908.10084) learns independently
computed representations of text pairs. Jointly processing a pair and
encoding its members independently are different encoder uses; neither
follows simply from the name BERT.

<a id="why-ordinary-generation-is-awkward-for-an-encoder"></a>

### Why MLM does not give a continuation probability

For a `[MASK]` inside a sentence, the encoder receives known right
context. In free continuation, those words do not exist yet. Moreover,
summed MLM losses over selected positions are not an autoregressive
factorization of the whole sentence's probability.

One can repeatedly append `[MASK]`, choose a predicted token, and
recompute. That algorithm can be defined and run, but it does not turn
BERT's pre-training into prefix-continuation training. Iterative
infilling, the order of position updates, and ordinary left-to-right
generation define different procedures. The difficulty concerns
training objective and available context, not whether a class name
contains “decoder.”

<a id="decoder-only-next-token"></a>

## Decoder-only: predict the next token

A causal language model learns

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

For a five-token phrase, add BOS to the first input and EOS to the
targets. There are six predictions: BOS predicts `Кошка`, `Кошка`
predicts `не`, followed by `стала`, `есть`, `корм`, and the end
of the phrase. The code above constructs two shifted slices of one sequence.

The row containing `стала` may read BOS, `Кошка`, `не`, and its own
token, but not the following input row `есть`. Allowing the diagonal
does not reveal the correct next token: that token is one position to
the right. Both the mask and the shift are required.

All input tokens are known during training, so one pass computes all
six distributions. During generation, future input tokens are absent:
a chosen continuation is first appended to the prefix, after which the
model computes the next distribution. One stack produces position
states and logits; architecture does not determine how tokens are chosen.

<a id="causal-head-to-logits"></a>

### From a head output to vocabulary logits

For the prefix `a cat`, the next token is still unknown. Take the
[[en/00 Textbook/05 Attention and Transformer/03 Masking, multi-head attention, and tensor shapes#The query for cat in a cat sleeps|head output at cat]],
$z_2\approx(0.330238,1.339523)$. The example uses a block with normalization
after each residual sum, as in [the original GPT-1 `block` implementation](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/train.py#L140-L147).

The input state at `cat` is $x_2=(0,1,0)$. Let the output projection append a zero to the two head coordinates. Adding the input gives $(0.330238,2.339523,0)$. The mean is $0.889921$ and the variance is $1.068850$. LayerNorm subtracts that mean and divides by $\sqrt{\operatorname{Var}+10^{-5}}$, giving approximately $(-0.541354,1.402131,-0.860777)$.

For this calculation, position additions and the FFN output are zero. Both LayerNorms have unit scale and zero bias. The second residual sum therefore equals the first LayerNorm output, and the second LayerNorm gives $h_2\approx(-0.541353,1.402130,-0.860777)$. Neither normalization nor the position-wise FFN can read the hidden `sleeps` input row.

In [GPT-1, §3.1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=3), the output layer takes a dot product with each row of the input embedding table. Let the vocabulary be `a`, `cat`, and `sleeps`, with an identity embedding matrix. The coordinates of $h_2$ are then the three logits:

| Token | Logit | Probability |
| --- | ---: | ---: |
| `a` | $-0.541353$ | $0.114816$ |
| `cat` | $1.402130$ | $0.801763$ |
| `sleeps` | $-0.860777$ | $0.083421$ |

The first softmax in attention is over input positions. The second is over the vocabulary. The model can assign `sleeps` a positive probability even though the input position containing that token is hidden. The chosen weights prefer `cat`; the correct token `sleeps` is used to compute the loss. The [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение#gpt1-decoder-architecture|GPT-1 chapter]] uses this distinction to explain the difference between the language-model head and a head for a labeled task.

<a id="different-tasks-in-one-textual-format"></a>

<a id="почему-из-next-token-prediction-получается-general-purpose-модель"></a>

### Text outputs and task training

Classification can be serialized as `Review: ... Rating: positive`,
and translation as `Russian: ... English: ...`. For a language model,
answers become text continuations. GPT-1 adds labeled-task training after
pre-training; GPT-2 studies transfer without weight changes by specifying
tasks through textual context. For example, it appends `TL;DR:` to a
document and generates a summary. The GPT chapters examine these transfer methods.

The format alone does not guarantee instruction following. NTP fits
the distribution of continuations to training data; instruction-following
behavior depends on data and subsequent training. Textual output does
not require decoder-only architecture either: T5 also produces text
answers, but has a separate encoder.

Appending a token to a causal decoder does not alter earlier states when the weights, positions, and masking rule stay fixed. This lets us keep old keys and values. In the [numerical check](#architecture-mask-cache), full attention changes rows that have already been computed, while causal attention leaves them unchanged.

<a id="encoderdecoder-read-one-text-write-another"></a>

<a id="encoder-decoder-conditional"></a>

## Encoder–decoder: a separate source for translation

For translation, an encoder builds memory
$H_{\mathrm{src}}\in\mathbb R^{T_{src}\times d}$. The decoder receives a
target prefix and reads that memory at each layer. Source and translation
lengths are independent.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/t5-figure4.png|Separate encoder and decoder stacks beside two single-stack variants: causal and prefix LM]]

*On the left, two separate stacks: a green encoder and a pink decoder.
The middle uses one causal stack for concatenated input and answer.
The right uses one stack with bidirectional input and a causal continuation.
T5's comparison distinguishes both stack separation and visibility.
Source: Colin Raffel et al., [T5, Figure 4, §3.2.1](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=16),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The diagram and its original English caption are preserved.*

<a id="one-translation-step"></a>

### Where the did state gets its context

Take source tokens `Кошка не стала есть корм` and decoder input `<BOS> The cat did`. In self-attention, the query at `did` reads four target positions. In cross-attention, a new query from that row reads five source positions. The score matrix has shape $4\times 5$, omitting batch and head axes. Row four belongs to `did`; the column for `не` is not hidden because of where it occurs in the Russian phrase.

$$
\begin{aligned}
Q&=H_{\mathrm{dec}}W_Q,\\
K&=H_{\mathrm{src}}W_K,\\
V&=H_{\mathrm{src}}W_V.
\end{aligned}
$$

For example, this head may use source positions `не стала` while computing
the state used to choose `not`. This illustrates a possible alignment,
not a claim about a particular head: actual weights are learned and need
not follow word-for-word translation.

Changing a chosen English token changes future queries, but not the
encoder output, so source memory can be computed once. The same translation can be requested from a causal model with one prefix: `Кошка не стала есть корм <SEP> The cat did`. Now `did` reads Russian and English keys in the same self-attention operation, with one softmax per head. In the two-stack model, target-prefix and source attention are normalized separately.

<a id="teacher-forcing"></a>

### Training on the reference prefix

During training, the known correct translation is shifted:

```text
decoder input: <BOS> The cat did not eat the food
targets:        The cat did not eat the food <EOS>
```

The mask prevents reading targets to the right, while cross-attention
preserves access to the whole known Russian source. For pair $x,y$,
training minimizes

$$
\mathcal L_{\mathrm{seq2seq}}
=-\sum_t\log p_\theta(y_t\mid y_{<t},x).
$$

This is the negative log conditional probability of the translation given its source.
Losses for Russian input words do not enter this sum. Gradients still
reach the encoder through cross-attention, as in the complete model
of the previous chapter. EOS is trained alongside ordinary tokens;
PAD targets are excluded.

<a id="t5-every-task-as-text-to-text"></a>

### Text outputs in T5

T5 uses an encoder–decoder and expresses task results as text. Its
internal blocks differ from the 2017 architecture: normalization precedes
sublayers, and relative biases in attention scores replace added sinusoidal
positions ([paper, §2.1](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=4)).
Translation inputs may begin with `translate English to German:`;
summarization inputs with `summarize:`. For classification, the decoder
generates a label name instead of a separate classifier's logit.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/t5-figure1.png|T5 examples of translation, acceptability, similarity, and summarization as input and output text]]

*Four inputs and their textual targets. For example, `not acceptable`
is a CoLA text label, and `3.8` is the textual representation of an
STS-B numerical target.
Source: Colin Raffel et al., [T5, Figure 1 and §2.4](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The figure's original English text is unchanged.*

A task prefix indicates the required answer but does not switch the
attention mask to another type. The encoder retains full access to the
input; the decoder retains causal access to target tokens. Text-to-text
does not mean that any checkpoint will follow an arbitrary wording
without training: the corresponding behavior must be supported by its
data and training.

<a id="t5-pre-training-reconstructing-spans"></a>

### Which tokens T5 reconstructs

Pre-training does not require translation pairs. T5 hides spans of
ordinary text and generates their contents as a separate target sequence.
One special sentinel token replaces each contiguous hidden span. Different
spans within an example receive distinct markers so the decoder can
distinguish insertion locations.

In the authors' example, `for inviting` and `last` are removed from
`Thank you for inviting me to your party last week.`. The input becomes
`Thank you <X> me to your party <Y> week.`, and the target becomes
`<X> for inviting <Y> last <Z>`. Three removed tokens form two spans;
`<Z>` ends the list of reconstructed spans.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/t5-figure2.png|T5 replaces spans with sentinels in the input and generates their contents in sentinel order]]

*The contiguous span `for inviting` is replaced by one marker. Its
contents are reconstructed by the decoder; unchanged source words are
not copied into the target.
Source: Colin Raffel et al., [T5, Figure 2, §3.1.4](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=13),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The authors' small example hides 3 of 10 words; it illustrates the
operation, not exact compliance with a 15% rate.*

Sentinels are vocabulary entries, not ordinary words. The model must
generate both them and span contents. Unlike BERT's MLM, predicted
elements occupy a new sequence, and the decoder conditions on its
already known beginning. Grouping adjacent hidden positions can shorten
both input and output.

The paper's baseline independently selects tokens and then merges
neighboring selected positions. Section 3.3.4 separately studies selecting
whole spans: for 500 original tokens, 15% corruption, and 25 spans, mean
length is $75/25=3$. Independent selection followed by merging and
sampling spans of a prescribed average length produce different
distributions of hidden regions.

The code below reconstructs the figure's transformation for preselected
positions. It does not reproduce T5's random sampler or actual
SentencePiece vocabulary.

<details>
<summary>Code: sentinels, target shift, and the PAD start token</summary>

```python
import torch

def corrupt_spans(words, noise):
    if len(words) != len(noise) or not any(noise):
        raise ValueError("provide one mask entry per token and a nonempty corruption")
    inputs, targets = [], []
    i, span = 0, 0
    while i < len(words):
        if not noise[i]:
            inputs.append(words[i])
            i += 1
        else:
            sentinel = f"<extra_id_{span}>"
            inputs.append(sentinel)
            targets.append(sentinel)
            while i < len(words) and noise[i]:
                targets.append(words[i])
                i += 1
            span += 1
    targets.append(f"<extra_id_{span}>")
    return inputs, targets

words = "Thank you for inviting me to your party last week.".split()
noise = [False, False, True, True, False, False, False, False, True, False]
inputs, targets = corrupt_spans(words, noise)
assert inputs == ["Thank", "you", "<extra_id_0>", "me", "to", "your",
                  "party", "<extra_id_1>", "week."]
assert targets == ["<extra_id_0>", "for", "inviting", "<extra_id_1>",
                   "last", "<extra_id_2>"]
print("encoder input", " ".join(inputs))
print("target", " ".join(targets))

# Toy IDs only; EOS is made explicit here, although absent from Figure 2.
symbols = ["<pad>", "</s>"] + list(dict.fromkeys(words + inputs + targets))
vocab = {s: i for i, s in enumerate(symbols)}
PAD, EOS = 0, 1
labels = torch.tensor([vocab[s] for s in targets] + [EOS, -100, -100])
decoder_ids = torch.full_like(labels, PAD)
decoder_ids[1:] = labels[:-1].masked_fill(labels[:-1] == -100, PAD)
valid = labels.ne(-100)
allowed = torch.ones(len(labels), len(labels), dtype=torch.bool).tril()
allowed &= valid[None, :]
assert decoder_ids[0] == PAD and allowed[0, 0]
wrong = allowed & decoder_ids.ne(PAD)[None, :]
assert not wrong[0].any()  # PAD-ID at position zero is the real decoder start
print("supervised targets", int(valid.sum()), "decoder IDs", decoder_ids.tolist())
```

</details>

A common T5 implementation uses the PAD ID as the decoder's start token.
Here it is 0. It is nevertheless a real prefix start. A mask based on
`decoder_ids != PAD` would incorrectly block every key in the first
row. Real positions are instead determined separately from target length.
The two added `-100` values are label padding only; seven supervised
targets remain: the six elements in the authors' target sequence plus EOS.

Shifting and replacement of `-100` with PAD follow
[`_shift_right` in Transformers v4.57.1](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/t5/modeling_t5.py#L859-L885).
This is a rule of that implementation, not a universal encoder–decoder rule.

<a id="one-model-can-combine-three-masks"></a>

## A prefix mask in one stack

In a prefix LM, known input and answer share one stack. The first $P$
positions form a bidirectional prefix and must not read any answer
position. Answer positions see the whole prefix, their own row, and
preceding answer rows. With zero-based indexing, a connection is allowed when

$$
j<P\quad\text{or}\quad(i\ge P\ \text{and}\ j\le i).
$$

This graph differs from both a fully causal stack and two stacks with
separate cross-attention. It is the right-hand matrix in T5 Figure 3
and the yellow stack on the right in Figure 4. Block parameters are
shared across source and target positions; an encoder–decoder usually
has different weights for those roles.

[UniLM, §2.1–2.4](https://arxiv.org/pdf/1905.03197v3#page=3) trains shared
parameters under several visibility patterns. The authors use four tasks:
left-to-right, right-to-left, bidirectional, and sequence-to-sequence.
The two unidirectional tasks each receive $1/6$ of training; bidirectional
and sequence-to-sequence each receive $1/3$. This is multimode training,
not merely changing a mask in an arbitrary pre-trained BERT.

UniLM pre-training uses cloze tasks: selected tokens are replaced and
reconstructed at the same positions. In sequence-to-sequence pre-training,
tokens in both segments can be hidden; during supervised generative
fine-tuning, tokens in the target segment are hidden. The same prefix
mask therefore does not make UniLM's objective identical to ordinary
shifted NTP.

Source positions must be prevented from reading the answer at **every**
layer. Otherwise, a future token first enters a source state. At the
next layer, a target query reads that source position and indirectly
receives future information, even if the triangle within the target
segment is correct.

A numerical example exposes this path. Take scalars $1,\ldots,6$, with
the first three forming a prefix. Each sublayer averages allowed values
and adds the average to its input. With the correct mask, prefix states
after one layer are $3,4,5$, and the first target row is $6.5$. At layer
two it becomes $6.5+(3+4+5+6.5)/4=11.125$.

If source rows are incorrectly allowed all six keys, their first-layer
states are $4.5,5.5,6.5$. The second layer gives the first target row
$12.25$. Changing the last future value from 6 to 66 increases it by
another $7.5$. With the correct mask, the result stays $11.125$.

<details>
<summary>Code: two-layer leakage through source positions</summary>

```python
import torch

n, prefix = 6, 3
i, j = torch.arange(n)[:, None], torch.arange(n)[None, :]
good = (j < prefix) | ((i >= prefix) & (j <= i))
bad = good.clone()
bad[:prefix] = True

def two_layers(values, allowed):
    weights = allowed.to(values.dtype)
    weights = weights / weights.sum(-1, keepdim=True)
    h = values
    for _ in range(2):
        h = h + weights @ h
    return h

x = torch.arange(1., 7., dtype=torch.float64)[:, None]
changed = x.clone()
changed[-1] += 60
correct = two_layers(x, good)
leaking = two_layers(x, bad)
torch.testing.assert_close(correct[3, 0], torch.tensor(11.125, dtype=x.dtype))
torch.testing.assert_close(leaking[3, 0], torch.tensor(12.25, dtype=x.dtype))
torch.testing.assert_close(two_layers(changed, good)[:4], correct[:4])
torch.testing.assert_close(
    two_layers(changed, bad)[3]-leaking[3], torch.tensor([7.5], dtype=x.dtype))
print("correct first target", correct[3], "leaking first target", leaking[3])
```

</details>

These are deliberately simplified layers with uniform weights, sufficient
for a counterexample. Normalization, an FFN, and learned scores do not
automatically repair a missing restriction in the dependency graph.
[UniLM v1 code](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/unilm-v1/src/biunilm/seq2seq_loader.py#L269-L275)
opens source-segment columns to all queries and copies a triangular
matrix into the target square.

<a id="контроль-понимания"></a>

<a id="three-independent-choices"></a>

Next comes [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa, and DeBERTa]]: how models keep the bidirectional encoder while changing the inputs, pre-training targets, and results.

## Further examples

<details>
<summary>Numerical example: masks and caching</summary>

<a id="architecture-mask-cache"></a>

<a id="минимальный-causal-attention-в-коде"></a>

<a id="computing-with-an-explicit-mask"></a>

<h3>Averaging the allowed values</h3>

Assign four V rows to `не`, `стала`, `есть`, and `корм`: $(1,0)$, $(0,1)$, $(2,2)$, and $(4,0)$. To isolate
the mask, set Q and K to zero. All allowed scores are then equal, so
attention averages the available V rows.

A full mask gives every row mean $(1.75,0.75)$. With a causal mask, the `стала` row receives $(0.5,0.5)$, the third $(1,1)$, and the last the mean of all
four. Replacing the last value with $(0,4)$ changes the full mean to
$(0.75,1.75)$ but leaves the first three causal rows unchanged.

```python
import math
import torch

def visibility(n, mode="causal", prefix=0):
    i, j = torch.arange(n)[:, None], torch.arange(n)[None, :]
    if mode == "full":
        return torch.ones(n, n, dtype=torch.bool)
    if mode == "causal":
        return j <= i
    if mode == "right-to-left":
        return j >= i
    if mode == "prefix":
        if not 1 <= prefix <= n:
            raise ValueError("prefix must contain between 1 and n positions")
        return (j < prefix) | ((i >= prefix) & (j <= i))
    raise ValueError("unknown visibility mode")

def masked_attention(q, k, v, allowed):
    if allowed.dtype != torch.bool or not allowed.any(-1).all():
        raise ValueError("each query needs an allowed key")
    scores = (q @ k.T / math.sqrt(q.shape[-1])).masked_fill(~allowed, -torch.inf)
    weights = scores.softmax(-1)
    return weights @ v, weights

v = torch.tensor([[1., 0.], [0., 1.], [2., 2.], [4., 0.]],
                 dtype=torch.float64)
q, k = torch.zeros_like(v), torch.zeros_like(v)
full, _ = masked_attention(q, k, v, visibility(4, "full"))
causal, _ = masked_attention(q, k, v, visibility(4))
changed = v.clone()
changed[-1] = torch.tensor([0., 4.])
new_full, _ = masked_attention(q, k, changed, visibility(4, "full"))
new_causal, _ = masked_attention(q, k, changed, visibility(4))
torch.testing.assert_close(full[1], torch.tensor([1.75, 0.75], dtype=v.dtype))
torch.testing.assert_close(causal[1], torch.tensor([0.5, 0.5], dtype=v.dtype))
torch.testing.assert_close(new_full[1], torch.tensor([0.75, 1.75], dtype=v.dtype))
torch.testing.assert_close(new_causal[:3], causal[:3])
print("full", full, "causal", causal)

# Append a fifth position without changing the first four input vectors.
extended = torch.cat([v, torch.tensor([[0., 4.]], dtype=v.dtype)])
q5, k5 = torch.zeros_like(extended), torch.zeros_like(extended)
full5, _ = masked_attention(q5, k5, extended, visibility(5, "full"))
causal5, _ = masked_attention(q5, k5, extended, visibility(5))
prefix4, _ = masked_attention(q, k, v, visibility(4, "prefix", 2))
prefix5, _ = masked_attention(q5, k5, extended, visibility(5, "prefix", 2))
new_query, _ = masked_attention(
    q5[-1:], k5, extended, torch.ones(1, 5, dtype=torch.bool))
torch.testing.assert_close(full5[0], torch.tensor([1.4, 1.4], dtype=v.dtype))
torch.testing.assert_close(causal5[:4], causal)
torch.testing.assert_close(prefix5[:4], prefix4)
torch.testing.assert_close(new_query[0], causal5[-1])
print("full after append", full5[0], "new causal query", new_query[0])
```

This checks permitted dependencies with deliberately chosen projections;
it is not an experiment on a trained model's text understanding. A real
block uses learned Q/K/V, residual connections, and an FFN; the mask
still excludes the same paths. The code uses 2D matrices without PAD
and does not implement a complete multilayer cache. Cached rectangular
attention needs masks based on absolute query and key positions, as
explained in the
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|masking chapter]].

<a id="which-states-can-be-reused"></a>

<h3>Reusing states</h3>

Appending a fifth value $(0,4)$ changes the full mean from $(1.75,0.75)$
to $(1.4,1.4)$. With bidirectional attention, outputs at already read
positions change: the new key participates in their softmax and the
new value in their weighted sums. A multilayer encoder passes these
updated rows to its next layer. In general, the old rows cannot simply
be retained without recomputation.

With a causal mask, the first four outputs remain unchanged. The fifth
row reads all five values and gets $(1.4,1.4)$, but earlier rows cannot
access the new key. That restriction holds at every layer: if input
vectors and positional indices are unchanged, the new token does not
change earlier states. This assumes unchanged model parameters and
position-encoding rules, and evaluation without dropout or other
random changes.

When extending a causal prefix, one can therefore retain old positions'
keys and values at every layer and compute queries only for new positions.
This is a KV cache. The example's final call uses one query and five keys;
it produces the same new-position output as full recomputation. Ordinary
BERT's mask provides no corresponding saving when input is appended:
new right context may change previous states.

A prefix LM fixes the boundary of its bidirectional region in advance.
If the first two positions form that prefix, they receive $(0.5,0.5)$.
Appended answer positions do not change them, so the prefix can also be
computed once. Moving the boundary from two to three positions, however,
changes the original rows: their mean becomes $(1,1)$. The old cache no
longer matches the mask. A fixed bidirectional prefix is compatible
with sequential generation; extending the bidirectional prefix itself
requires recomputation.

In an encoder–decoder, the source remains fixed during generation.
Its output is computed once; each cross-attention layer can also retain
its own K/V projections of that memory. Queries from the changing target
prefix must be computed anew.
[T5 v4.57.1](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/t5/modeling_t5.py#L493-L535)
maintains separate `self_attention_cache` and `cross_attention_cache`.
Changing the source requires updating both its memory and dependent caches.

</details>

<details>
<summary>Model outputs and training data</summary>

<a id="what-to-choose-in-practice"></a>

<h3>Model outputs and training data</h3>

A task first specifies the required output and available context.
A family name alone neither selects a model nor evaluates its quality.

| Required output | Computation | Training signal |
|---|---|---|
| whole-text label | output layer over pooled states or CLS | labeled texts |
| position labels or answer boundaries | projections of individual rows | token or span annotations |
| retrieval vector | independent encoding and pooling | pairs, ranking, or another similarity objective |
| pair score | joint encoding of two texts | labeled pairs |
| text continuation | causal states and vocabulary logits | next tokens |
| textual answer to a known source | conditional autoregressive distribution | source–answer pairs or span reconstruction |

If a document is encoded independently of a query, its vector can be
stored and reused. With joint query–document encoding, document states
depend on the query; retaining a vector no longer replaces the full
joint pass. This is the bi-encoder/cross-encoder distinction, not proof
that any encoder is faster than any decoder.

A causal model's final state can also use its entire known prefix.
Using it as a retrieval vector requires suitable pooling and similarity
training; NTP alone does not guarantee this. Model selection should
check quality on relevant data, length limits, and the cost of the
required regime, not only the mask.

<a id="сравнение-без-рекламных-упрощений"></a>

<a id="comparing-the-three-variants"></a>

<h3>Stacks, masks, and output heads</h3>

| Property | Encoder-only | Decoder-only | Encoder–decoder |
|---|---|---|---|
| source text | entire known input | part of a shared prefix | separate encoder input |
| self-attention | usually bidirectional | usually causal | full in encoder, causal in decoder |
| separate cross-attention to memory | absent in ordinary BERT | absent in ordinary GPT | present |
| output | states and a chosen output head | states and vocabulary head | decoder states and vocabulary head |
| pre-training examples | MLM, contrastive learning | NTP | span reconstruction |
| original model examples | BERT | GPT | T5, BART |

Prefix LMs qualify the second column: known input may be bidirectional
within one stack. Stack count and attention-mask type should therefore
both be specified.

<a id="ошибки-которые-стоит-научиться-замечать"></a>

<a id="what-these-distinctions-do-and-do-not-establish"></a>

<h3>What changing the right context tests</h3>

Full context access permits dependencies but does not prove that the
network uses them meaningfully. Equality of causal states after future
tokens change checks a forbidden connection; differing bidirectional
states do not by themselves demonstrate semantic understanding.

A decoder does not need cross-attention to answer questions about a
document: the document can occupy its shared prefix. Conversely,
cross-attention does not guarantee correct use of a source. Training
and results on relevant tasks must be evaluated.

MLM, NTP, and span reconstruction specify different sets of conditional
predictions. None defines “understanding.” The same architecture can
participate in classification, retrieval, and generation with appropriate
training and an appropriate output head.

<a id="why-the-word-decoder-is-confusing"></a>

<h3>The decoder in GPT and in the translator</h3>

The original translation Transformer's decoder contains causal
self-attention, cross-attention, and an FFN. GPT has neither cross-attention
nor a separate encoder, but “decoder-only” remains the name for its
causal continuation-predicting stack.

The reverse inference from a component name also fails. A vision encoder
beside a language model does not establish T5-like cross-attention between
them. [LLaVA](https://llava-vl.github.io/) projects image features into
language-model input representations. For a composite model, the
connection between parts must be identified rather than inferred from
their names.

</details>

<details>
<summary>Checking a mask and reading the implementation</summary>

<a id="practice-without-a-large-gpu"></a>

<a id="1-нарисовать-маски"></a>

<a id="1-visibility-matrices"></a>

<h3>Mask axes</h3>

The example's `visibility` function returns Boolean matrices where
`True` means an allowed connection. For six positions, it produces four
variants: full access, two causal directions, and a three-position
prefix. The prefix mask's upper-right square must be completely closed.
The two-layer check above shows the error that opening it would cause.

A self-attention square and a cross-attention rectangle have different
axes. For two queries and four source keys, a full mask has shape
`[2,4]`; it is not constructed with `tril(2,4)`.

<a id="2-сравнить-representations"></a>

<a id="2-compare-representations"></a>

<h3>Right context for bank</h3>

Two sentences suitable for a pretrained English-model experiment are:

```text
They reached the bank beside the river.
They reached the bank beside the office.
```

The left prefix and token `bank` match; only the right continuation
changes. Check subword alignment, disable dropout, and compare the same
layer and position. A causal model's `bank` state must match within
numerical tolerance; a bidirectional model's state may change. The
causal model's last word does not have this invariance: the changed
continuation lies in its prefix.

The four-V numerical example checks the same information access without
loading weights. It does not replace an experiment with pretrained
models and makes no claims about results for any checkpoint.

<a id="3-read-real-code"></a>

<h3>Inputs, targets, and masks in code</h3>

In official BERT, MLM position selection and the full attention mask
live in different parts of data preparation and modeling. T5's
`_shift_right` builds decoder input from targets, and decoder attention
acts on that shifted input. UniLM's `input_mask` changes available
context while stack parameters are shared.

These locations answer different questions: what was changed in the
input, where loss is computed, and which connections are allowed.
A comparison must check all three, not merely the presence of a causal flag.

</details>

<a id="literature-and-additional-material"></a>

## Sources

<a id="primary-papers"></a>

### Primary sources

- Jacob Devlin et al., [BERT, §3–3.2 and Figure 4](https://aclanthology.org/N19-1423/).
- Colin Raffel et al., [T5, §§2.1, 2.4, 3.1.4, 3.2.1, 3.3.4](https://jmlr.org/papers/v21/20-074.html).
- Li Dong et al., [UniLM, §2](https://arxiv.org/abs/1905.03197).
- Alec Radford et al., [Generative Pre-Training, §3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- Nils Reimers, Iryna Gurevych, [Sentence-BERT, §3](https://arxiv.org/abs/1908.10084).

<a id="лучшие-объяснения-и-код"></a>

### Materials and code

- Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/).
- Lena Voita, [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html).
- [UniLM v1: mask preparation](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/unilm-v1/src/biunilm/seq2seq_loader.py).
- [T5 in Transformers v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/t5).

### Related Bookvar notes

- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-only]]
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Decoder-only]]
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder]]

Questions on these topics: [[02 Areas/ML & DL/04 Вопросы/100 вопросов по NLP|100 NLP questions]].

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|The complete Transformer]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa, and DeBERTa]]
