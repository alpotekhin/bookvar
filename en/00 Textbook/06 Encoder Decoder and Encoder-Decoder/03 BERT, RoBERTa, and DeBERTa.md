---
title: BERT, RoBERTa, and DeBERTa
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources:
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/1907.11692
  - https://arxiv.org/abs/2006.03654
---

# BERT, RoBERTa, and DeBERTa

A word's meaning is often revealed by what comes after it. The same token
`bank` means something different in `withdrew money from the bank` and `sat on
the river bank`, and right context helps decide which sense is present. A
causal language model cannot use that context in the current token state: its
next-token objective permits only left context. That is necessary for
generation but artificial when labeling known text for classification, entity
recognition, relation extraction, or extractive question answering.

BERT removes this restriction by using a Transformer encoder stack with no
causal mask. Every position sees the complete input, so a token's upper-layer
state includes both left and right context. Simply predicting the current token
would now collapse into copying because the answer is already present. BERT
therefore hides selected tokens and reconstructs them from the rest of the
text. This combination—bidirectional encoder plus masked language modeling
(MLM)—became the standard encoder-only pre-training pattern.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png]]

*The same word receives different contextual representations in different
sentences. Illustration: Jay Alammar, [The Illustrated
BERT](https://jalammar.github.io/illustrated-bert/), [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

## Architecture and tensor shapes

BERT is not a full encoder–decoder Transformer. It contains encoder blocks
only: multi-head self-attention with full visibility, a position-wise
feed-forward network, residual connections, and LayerNorm. `BERT_BASE` has 12
layers, hidden width 768, 12 heads, and about 110 million parameters;
`BERT_LARGE` has 24 layers, width 1024, 16 heads, and about 340 million. BASE was
matched in size to GPT-1 so that differences could not be attributed solely to
parameter count.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-base-bert-large-encoders.png]]

*BERT BASE and LARGE differ in depth and width but use the same encoder-only
path. Illustration: Jay Alammar, [The Illustrated
BERT](https://jalammar.github.io/illustrated-bert/), [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Input is split into WordPieces from a vocabulary of roughly 30,000 units.
`[CLS]` begins the sequence; `[SEP]` ends one segment or separates a pair. At
each position, the model adds a token embedding, a learned position embedding,
and a segment embedding identifying segment A or B.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-input-output.png]]

*Token, position, and segment vectors are added, and the top layer returns one
state per position. Illustration: Jay Alammar, [The Illustrated
BERT](https://jalammar.github.io/illustrated-bert/), [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

For batch size $B$, sequence length $T$, model width $d$, $h$ heads, and head
width $d_h=d/h$, token IDs have shape `[B,T]` and summed embeddings
`[B,T,d]`. Split Q, K, and V have shape `[B,h,T,d_h]`, attention scores
`[B,h,T,T]`, and every layer again outputs `[B,T,d]`. Unlike a decoder, there
is no triangular causal mask; only padding is hidden. Row $i$ may therefore use
positions both below and above $i$.

## MLM: obtaining a bidirectional training signal

BERT selects 15% of WordPiece positions. For the selected set $M$,

$$
\mathcal L_{MLM}=-\sum_{i\in M}\log p_\theta(x_i\mid \tilde x),
$$

where $x_i$ is the original token and $\tilde x$ the corrupted sequence. Loss
is computed only at selected positions even though the encoder computes all
states. A causal LM usually obtains direct supervision at every position but
has one-sided context; MLM has sparse supervision but bidirectional context.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png]]

*Masked language modeling reconstructs a hidden token using both sides of its
context. Illustration: Jay Alammar, [The Illustrated
BERT](https://jalammar.github.io/illustrated-bert/), [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Consider `the cat sits on the mat`, with `sits` selected.

1. In 80% of selected cases the input becomes `the cat [MASK] on the mat`; 10%
   are replaced by a random token, and 10% remain unchanged.
2. Adding `[CLS]`, `[SEP]`, position, and segment embeddings produces
   `[1,T,d]`.
3. Every encoder layer builds the selected position state $h_i$, which may use
   both `the cat` and `on the mat`.
4. The MLM head transforms $h_i$ and projects it to vocabulary logits `[V]`.
5. Cross-entropy compares the distribution with the original ID for `sits`;
   unselected positions do not enter this sum.

The 80/10/10 mixture reduces the pre-training–application mismatch because
`[MASK]` is normally absent during fine-tuning. An unchanged selected token is
not useless: the model does not know which positions enter the loss and must
maintain informative ordinary-token states. Random replacement prevents the
presence of `[MASK]` from being the sole signal that an answer is required.

## BERT's second objective: segment relationships

Next Sentence Prediction (NSP) uses A/B pairs. In half the examples B genuinely
follows A in the corpus (`IsNext`); in the other half B is sampled randomly
(`NotNext`). The `[CLS]` state feeds a binary head, and the total loss is MLM
plus NSP. This requires corpora with document boundaries.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-next-sentence-prediction.png]]

*The original BERT NSP objective. Illustration: Jay Alammar, [The Illustrated
BERT](https://jalammar.github.io/illustrated-bert/), [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The original motivation is not an established law. Removing NSP slightly hurt
several tasks in BERT's ablation, and the authors connected it to segment-
relation tasks. RoBERTa later removed NSP without losing quality under longer
sequences and a different recipe. The experiments are not formally
contradictory because data construction and training regime changed together.
They show why an objective cannot be interpreted outside its data pipeline.

## BERT data and recipe

Pre-training combined BooksCorpus—about 800 million words—and English
Wikipedia—about 2.5 billion words. `BERT_BASE` trained on 4 Cloud TPUs and
`BERT_LARGE` on 16 for four days. The recipe used one million steps, batches of
256 sequences, Adam with initial learning rate $10^{-4}$, dropout 0.1, and
maximum length 512. Most steps used length 128; the final 10% used length 512
because full attention makes long examples much more expensive.

These details are not historical bookkeeping. Quality cannot be attributed to
“BERT” as architecture alone: token volume, context length, masking, and compute
determine how much useful signal the parameters receive. This becomes
RoBERTa's central lesson.

## One encoder, several ways to read its output

After pre-training, the stack remains. Single-text or pair classification feeds
`[CLS]` of shape `[B,d]` into a linear `[d,C]` head. Token classification
applies one projection to `[B,T,d]` and returns `[B,T,C]`. Extractive QA uses
two trainable vectors to produce start and end logits `[B,T]`; the answer is a
source span, not newly generated text. Usually the entire encoder and the small
head are updated together.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-tasks.png]]

*One encoder and four ways to read its output: pair classification, text
classification, answer extraction, and token labeling. Illustration: Jay
Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

`[CLS]` is not “the sentence meaning vector” by definition. It is a service
position that pre-training and a downstream task may make useful for a
particular head. Cosine distance between raw `[CLS]` vectors need not measure
semantic similarity; sentence embeddings require an appropriate objective and
pooling method.

## Reading BERT's results

BERT set new results on eleven tasks. `BERT_LARGE` substantially improved GLUE
over GPT-1; one model reached 91.8 F1 on SQuAD 1.1 and 83.1 F1 on SQuAD 2.0.
The mechanism is better supported by controlled comparisons than by leaderboard
rows. Replacing bidirectional MLM with a left-to-right objective reduced SQuAD
F1 from 88.5 to 77.8; adding a BiLSTM to the unidirectional representation
recovered only part of the gap. Larger models improved even on small labeled
datasets.

These numbers are not a pure causal estimate of “bidirectionality” under every
condition. Each configuration fixes a tokenizer, corpus, pre-training regime,
and fine-tuning protocol; some leaderboard systems use ensembles or additional
data. The narrower, defensible conclusion is that deep bidirectional
pre-training produced strong transferable features, and simple end-to-end
fine-tuning worked across varied NLU tasks.

## RoBERTa: a new result without a new architecture

RoBERTa keeps the BERT encoder and MLM but revises the recipe. It removes NSP,
packs long sequences, samples masks dynamically on each presentation,
increases batch size and update count, expands the corpus, and uses byte-level
BPE with roughly 50,000 units. Data grows to about 160 GB by adding CC-NEWS,
OpenWebText, and STORIES to BooksCorpus and Wikipedia.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/roberta-dynamic-masking-table1.png]]

*Static and dynamic masking in a controlled comparison. Source: Liu et al.,
[RoBERTa, Table 1](https://arxiv.org/abs/1907.11692), 2019.*

BERT prepared masks in advance and reused ten data variants across epochs. In
RoBERTa, the same string may receive a new mask each time. This change alone had
a small effect; the table does not support the claim that dynamic masking
explains the whole gain. The strong result comes from accumulated changes. With
batches near 8,000 sequences and 500,000 steps, `RoBERTa_LARGE` achieved 88.5
on test GLUE, 89.4 F1 on SQuAD 2.0 validation, and 83.2 accuracy on RACE under
the paper's protocols.

RoBERTa is therefore a lesson in experimental design. Comparing a new model to
an undertrained baseline mixes architecture with data and optimization. Here,
a stronger BERT-like system showed how much performance remained in the older
mechanism. “RoBERTa outperforms BERT” does not mean its self-attention is
architecturally different.

## DeBERTa: separate content from relative position

DeBERTa does change attention. BERT adds an absolute position vector directly
to each token embedding. DeBERTa keeps content representations and a table of
relative offsets separate. For query position $i$ and key position $j$, the
unnormalized score has three interactions:

$$
\tilde A_{ij}=Q_i^cK_j^{c\top}
+Q_i^cK_{\delta(i,j)}^{r\top}
+K_j^cQ_{\delta(j,i)}^{r\top}.
$$

The first term compares content with content. The second asks where the key is
relative to this query; the third asks where the query is relative to this key.
The authors omit a pure position-to-position term. With head width $d_h$, the
score is scaled by $\sqrt{3d_h}$ because it sums three dot products rather than
one.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/deberta-disentangled-attention-equation4.png]]

*Disentangled attention in the authors' notation. Source: He et al., [DeBERTa,
Eq. 4](https://arxiv.org/abs/2006.03654), 2020.*

Relative position is sufficient for many interactions, but MLM may also need
the absolute location of a missing token. The Enhanced Mask Decoder (EMD)
reintroduces absolute position representations near the MLM output head. This
explains *decoding-enhanced* in the name: it is improved decoding of a masked
token, not a generative decoder with causal self-attention.

`DeBERTa_LARGE` retains 24 layers, $d=1024$, and 16 heads, and trains on about
78 GB from Wikipedia, BookCorpus, OpenWebText, and STORIES. The paper reports
90.0 average on GLUE dev, 90.7 F1 on SQuAD 2.0, and 86.8 on RACE, ahead of
comparable encoders. Removing EMD or the positional interactions hurts several
tasks. A 1.5B version with scale-invariant fine-tuning reaches 89.9 on SuperGLUE
against the authors' cited human baseline of 89.8. This aggregate comparison
does not mean that the system “understands language better than humans” in
general; it applies to one benchmark, metric set, and protocol.

## Limitations and legacy

Encoder-only models naturally analyze known text but do not define an
autoregressive distribution over a complete string. Independent MLM
predictions require an additional procedure for open generation. Only roughly
15% of positions receive direct loss per step, `[MASK]` introduces a
pre-training mismatch, and full attention requires memory on the order of
$T^2$. Length 512 limits documents that can be processed in one pass.

Books, Wikipedia, news, and web pre-training transfer factual errors and social
bias along with knowledge. High mean GLUE does not guarantee robustness to a
domain shift, counterexample, or rephrasing. Fine-tuning on small datasets is
sensitive to seed and hyperparameters, so reporting one best run overstates
expected performance.

The legacy has three parts. BERT established bidirectional pre-training plus a
small task head. RoBERTa showed that architectures must be compared under a
strong, transparent recipe. DeBERTa made position an explicit part of token
interaction. All three remain encoder-only and are especially appropriate when
the answer is a class, tag, score, or span of known input.

The next chapter changes visibility direction. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1]] returns to a causal decoder so that one objective can both learn features and continue a sequence.

## Sources and further reading

- [Devlin et al., BERT](https://arxiv.org/abs/1810.04805) — architecture, MLM/NSP, recipe, tasks, and ablations.
- [Liu et al., RoBERTa](https://arxiv.org/abs/1907.11692) — effects of data, batch size, training length, NSP, and dynamic masking.
- [He et al., DeBERTa](https://arxiv.org/abs/2006.03654) — disentangled attention, EMD, and scaling experiment.
- [Jurafsky & Martin, Speech and Language Processing, Chapter 10](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — masked language models and transfer of encoder representations.
- [Jay Alammar, The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — visual explanation of input, MLM, and task heads.
- [[02 Areas/ML & DL/Papers/BERT|Local BERT note]], [[02 Areas/ML & DL/Papers/RoBERTa|RoBERTa]], and [[02 Areas/ML & DL/Papers/DeBERTa|DeBERTa]] — recipe, result, and ablation tables.
- [BERT, §3.1](https://arxiv.org/abs/1810.04805) — original MLM formulation.
- [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three architectural patterns]].
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|Next chapter: GPT-1]].
