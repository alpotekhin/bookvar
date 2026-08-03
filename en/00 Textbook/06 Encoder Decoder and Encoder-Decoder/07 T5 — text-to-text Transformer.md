---
title: T5 — text-to-text Transformer
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources:
  - https://jmlr.org/papers/v21/20-074.html
  - https://github.com/google-research/text-to-text-transfer-transformer
---

# T5: the text-to-text Transformer

Classification, translation, extractive question answering, and summarization
have naturally different outputs. A traditional pipeline consequently used
different task heads, batch formats, losses, and inference procedures: a
classifier returned a class index, extractive QA returned two positions, and a
translator returned a sequence. This made it difficult to tell whether an
improvement came from general transfer or a benchmark-specific architecture.

T5—the Text-to-Text Transfer Transformer—gives every task one contract: input
is text and output is text. A textual prefix names the task. `translate English
to German:` requests translation, `summarize:` requests a summary, and a
classifier generates a verbal label. One encoder–decoder model, one
autoregressive loss, and one decoding procedure apply to every task.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/text-to-text-framework.png]]

*The unified text-to-text interface: the task name is part of the source text,
and every answer is represented as target text. Source: Raffel et al., [T5,
Figure 1](https://jmlr.org/papers/v21/20-074.html), 2020; open-access JMLR
article.*

Read the figure by rows, not as a collection of different heads. For
translation, the source contains a prefix and sentence and the target contains
the translation. For CoLA, the target is `acceptable` or `not acceptable`. For
STS-B, the numeric score is quantized and written as a string. Data and the
post-decoding metric change, but the model output always remains a next-token
distribution.

## Architecture: source and answer have different roles

T5 uses a full encoder–decoder Transformer. The encoder reads the source with
bidirectional self-attention. The decoder creates the target left to right: its
self-attention is causal, while cross-attention in every layer can access all
encoder states. The conditional probability is

$$
p_\theta(y\mid x)=\prod_{t=1}^{T_y}p_\theta(y_t\mid y_{<t},H^{enc}(x)).
$$

Unlike a decoder-only model, it need not repeat the source in the generated
prefix. The encoder computes an input memory once, and the decoder reads that
memory repeatedly while producing the answer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/encoder-decoder-architecture.png]]

*The encoder reads the condition in full; a causal decoder creates the answer
left to right and accesses encoder memory through cross-attention. The diagram
accompanies T5 materials: Colin Raffel et al., [Exploring the Limits of
Transfer Learning](https://jmlr.org/papers/v21/20-074.html).*

Let source IDs have shape `[B,S]` and target IDs `[B,T]`. After embedding, the
encoder receives `[B,S,d]` and returns memory of the same shape. Decoder
embeddings have shape `[B,T,d]`. Decoder self-attention logits are
`[B,h,T,T]` with a triangular mask; cross-attention logits are `[B,h,T,S]`.
Projection into the shared vocabulary produces `[B,T,V]`, and cross-entropy is
computed against a shifted target. Teacher forcing makes all target positions
parallel during training; at inference, tokens are generated sequentially.

The paper's Base configuration has 12 encoder and 12 decoder layers, width 768,
12 heads, FFN width 3072, and about 220 million parameters. The family includes
Small 60M, Base 220M, Large 770M, 3B, and 11B. These are coordinated
configurations, not one trained model truncated afterward.

## How a T5 block differs from the original Transformer

T5 retains multi-head attention and feed-forward sublayers, but uses a
normalization that neither subtracts the mean nor adds a bias. It appears before
the sublayer, outside the residual path. The original T5 FFN uses ReLU. Token
embeddings are shared by the encoder, decoder, and output projection. Later
T5.1.1 configurations change some details, so their recipe should not silently
be attributed to the model in the paper.

Instead of absolute positional embeddings, T5 adds a learned scalar relative
position bias to attention logits. Distances are grouped into buckets: small
distances are distinguished exactly, large distances more coarsely and
logarithmically. The bias is head-specific and shared across layers within one
stack. It conveys order and distance without adding an absolute vector to each
token representation.

The 32,000-piece SentencePiece vocabulary is shared between input and output.
It was trained mainly for English but includes German, French, and Romanian
needed by the translation tasks. This does not make the original T5 fully
multilingual; mT5 later broadens language coverage.

## Pre-training: reconstruct deleted spans

Task prefixes appear when transferring to labeled datasets. T5 pre-training
does not use a list of instructions. It uses a denoising objective called span
corruption. About 15% of tokens are grouped into contiguous spans with mean
length near three tokens. Each deleted input span is replaced with a unique
sentinel token such as `<extra_id_0>`, `<extra_id_1>`, and so on. The target
lists deleted spans in their original order, prefixing each with its matching
sentinel and ending with the next sentinel.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/t5-span-corruption-figure2.png]]

*Original string, corrupted encoder input, and decoder target for span
corruption. Source: Raffel et al., [T5, Figure
2](https://jmlr.org/papers/v21/20-074.html), 2020.*

Suppose the original text is `Thank you for inviting me to your party last
week`, with `for inviting` and `last` removed.

1. The tokenizer constructs the sequence. A sampler selects positions and
   merges neighboring selected tokens into two spans.
2. The encoder receives `Thank you <extra_id_0> me to your party
   <extra_id_1> week`.
3. The decoder target is `<extra_id_0> for inviting <extra_id_1> last
   <extra_id_2>`. The final sentinel closes the final reconstructed span before
   EOS.
4. At target position $t$, the causal decoder sees previous target tokens and
   all encoder memory, but no future answer token.
5. The loss sums $-\log p(y_t\mid y_{<t},x_{corrupt})$ over target tokens. The
   undeleted source words need not be generated again.

A sentinel has two jobs. In the input, it identifies one particular gap; in the
target, it associates reconstructed text with that gap. Unique sentinels keep
two locations distinct. Masking spans requires the model to reconstruct
variable-length phrases, while the short target saves decoder computation
compared with reproducing the entire original string.

Span corruption is not causal language modeling of the whole document. The
encoder sees corrupted source context on both sides of every gap, and the
decoder assigns probability to a separate target string. Nor is it BERT MLM:
BERT classifies the original token at every selected position, whereas T5
generates all deleted spans consecutively.

## Why this objective was selected

The T5 paper is primarily a controlled study of transfer learning. Within one
codebase, the authors compare architectures, corruption objectives, corruption
rates, span lengths, corpora, transfer strategies, and scale. Many denoising
objectives performed similarly. The conclusion is not that sentinels are the
only correct objective. The chosen variant combined good transfer with a short
target sequence—a practical quality–compute trade-off.

At comparable compute, encoder–decoder was a strong general architecture for
text-to-text tasks. A decoder-only model can concatenate source and target, but
uses causal attention on the source and does not separate input and answer
roles. An encoder-only model is natural for classification and span extraction,
but without a generative decoder it does not cover translation and
summarization through the same interface.

## C4: scale begins with data selection

The authors created the Colossal Clean Crawled Corpus (C4) from the April 2019
Common Crawl snapshot. After cleaning, the English version occupied roughly
750 GB. Filters required sufficiently long connected text and terminal
punctuation; removed code, boilerplate, `lorem ipsum`, some pages containing
blocked terms, and duplicate fragments; and retained text classified as English
with high confidence.

Here *clean* means the output of particular heuristics, not an objectively
neutral corpus. A blocked-word filter may disproportionately remove writing
about certain groups and topics. Common Crawl retains factual errors,
duplicates, social bias, and the distribution of the open web. English-only
filtering limits how far conclusions transfer to other languages.

The baseline controlled experiments ran for $2^{19}=524\,288$ steps with a
batch of 128 sequences of length 512—about 34 billion tokens—and AdaFactor.
Final T5 models used a much longer regime approaching one trillion tokens. A
final-series `T5-Base` result therefore cannot be compared directly with one
ablation baseline without checking token count and accompanying improvements.

## Transfer: the task becomes data, not a new head

For supervised fine-tuning, every record becomes a pair of strings. For
example, `cola sentence: The course is jumping well.` maps to `not acceptable`;
a CNN/Daily Mail article receives `summarize:` and a reference summary. Encoder,
decoder, and shared output projection are updated with the same
maximum-likelihood objective used in pre-training.

In multi-task training, examples from several datasets are mixed. Sampling
probability is part of the method. Proportional sampling lets a huge dataset
overwhelm a small one; uniform sampling over tasks repeatedly reuses small
datasets. T5 compares capped and temperature-scaled mixtures, so “training on
all tasks” is not a complete recipe without a sampling rule.

After generation, text is converted to the metric's format. Classification
matches a string against allowed labels; SQuAD uses exact match;
summarization uses ROUGE. A common model does not make the metrics identical.
An invalid string can count as an error, a multi-token verbal label requires
several decoding steps, and an alternative label wording changes the optimized
probability.

## Reading the key experiments

The final T5-11B reported a GLUE average of 90.3, SuperGLUE 88.9 against the
paper's human baseline of 89.8, SQuAD validation exact match of 91.26, and
CNN/Daily Mail ROUGE-2 of 21.55. It did not surpass specialized translation
systems using back-translation: a universal interface does not guarantee the
best result in every domain.

Small, Base, Large, 3B, and 11B show broad improvement with size, but final
numbers combine several factors. A larger model, longer pre-training, C4, the
chosen objective, and the multi-task recipe form one system. In a separate
comparison, Base trained on one trillion tokens improved over the 34-billion-
token baseline, and the other choices added further gains. The correct reading
is that scale mattered greatly but did not exhaust the effects of architecture,
data, and transfer design.

A SuperGLUE aggregate should not be paraphrased as “T5 almost understands
language like a human.” Human and model baselines are measured on particular
tasks and metrics, some with exploitable artifacts. Likewise, ROUGE measures
fragment overlap with a reference rather than factual faithfulness of any
summary, while exact match penalizes a correct paraphrase.

## Inference cost and practical trade-offs

At serving time, the encoder runs once. During autoregressive decoding, the
decoder stores a KV cache of its own past and repeatedly uses cached encoder
memory for cross-attention. Cost grows with source and target length; beam
search multiplies decoder work by beam count. Full encoder self-attention
remains quadratic in a long source length $S$.

Text-to-text is especially natural when input and output have distinct roles:
translation, summarization, correction, and conditional generation. For closed
classification, an encoder-only model with a linear head can be cheaper and
cannot generate an invalid label. A decoder-only model is convenient when
condition and answer form one continuation or when a general conversational
API is required. There is no workload-independent winner.

## Limitations and legacy

A unified string interface unifies software, not task semantics.
Classification remains class selection even when a class is written as a
word; discretized regression loses precision; generation quality depends on
decoding. Original T5 prefixes are short task identifiers, not evidence of
reliable understanding of arbitrary natural-language instructions.

C4 is limited to the web domain and English. Full attention limits length. The
model can reproduce corpus errors and generate plausible but incorrect text.
Eleven billion parameters improved average metrics at substantial training and
inference cost; that result does not specify the best model under a restricted
budget.

T5's durable legacy is threefold: a common text-to-text API, sentinel-based span
corruption, and systematic comparison of transfer-learning components. mT5
replaces the corpus with multilingual mC4, ByT5 moves to UTF-8 bytes, and
FLAN-T5 adds instruction fine-tuning. These are separate changes to data,
tokenization, and post-training; FLAN properties should not be attributed to
the original T5.

Historically, T5 complements [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT]] and [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3]]: BERT builds bidirectional representations of a known input, GPT continues one causal sequence, and T5 explicitly separates the condition from the generated answer.

## Sources and further reading

- [Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) — primary paper: interface, architecture, C4, objectives, controlled study, and final results.
- [Google Research, Exploring Transfer Learning with T5](https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/) — official introduction.
- [Official T5 repository](https://github.com/google-research/text-to-text-transfer-transformer) — reference implementation, preprocessing, and checkpoints.
- [[02 Areas/ML & DL/Papers/T5|Local T5 paper note]] — architecture, recipe, ablation, and result tables.
- [[02 Areas/ML & DL/02 Атлас моделей/Семейства/T5|T5 family atlas]] — differences among T5, mT5, ByT5, and FLAN-T5.
- [BERT, § 3.1](https://arxiv.org/abs/1810.04805) — the original token-level MLM objective for comparison with span corruption.
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|Previous chapter: GPT-3]].
