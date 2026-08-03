---
title: GPT-2 — zero-shot through language
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources: [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]
---

# GPT-2: zero-shot through language

GPT-1 transferred general language knowledge into a task by updating its
weights. GPT-2 investigated a more radical hypothesis: many tasks are already
represented inside ordinary text, so a sufficiently large language model may
recognize a task from a prefix and continue in the required format without
task-specific fine-tuning. Translation pages contain language pairs, articles
end in summaries, and interviews alternate questions and answers. If a model
learns to predict all of this text, a visible textual pattern can specify the
task instead of a new output head.

This changes the interface, not the loss. GPT-2 still minimizes

$$
\mathcal L=-\sum_t\log p(x_t\mid x_{<t}).
$$

During training it is not told that a fragment is an example of translation or
summarization. At inference time, a phrase such as `TL;DR:` or a pattern such as
`English = French` changes the conditional distribution of the continuation.
The paper's claim of “unsupervised multitask learning” should therefore be read
as a hypothesis about latent structure in web text, not as a hidden catalog of
labeled tasks.

## What changed from GPT-1

The architectural principle remains the same: a causal decoder-only Transformer
with learned token and positional embeddings. The model became deeper, context
grew from 512 to 1024 tokens, and the largest configuration grew from 117
million to 1.5 billion parameters. Layer normalization moved before attention
and the FFN, a final normalization was added after the stack, and residual
projection initialization was scaled with depth. These changes made a larger
stack easier to train; no separate module creates zero-shot behavior.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-sizes-hyperparameters-3.png]]

*The four GPT-2 sizes and their block parameters. Illustration: Jay Alammar,
[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), [CC
BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The 117M, 345M, 762M, and 1.5B series is part of the experimental design. It
tests not only the largest model's absolute score but the direction of change
as depth and width increase. In the report, WebText loss and most evaluations
improved with scale without clear saturation. That observation motivated the
next, much more expensive step toward GPT-3.

## WebText and byte-level BPE

BooksCorpus supplied long prose but covered a narrow range of textual forms.
For GPT-2, the authors assembled WebText: about eight million documents and
roughly 40 GB of text. Pages entered the corpus if their links had received at
least three Reddit votes. Wikipedia was deliberately excluded, in part to avoid
training directly on the source of some language benchmarks.

User votes were an inexpensive quality heuristic, but they also transferred
the preferences of an English-speaking Reddit community: choices of sites,
topics, styles, and political viewpoints. WebText is not a random sample of
language or of the world. A broader corpus increased the number of task-like
patterns, while also expanding the range of biases, questionable claims, and
potentially memorized passages.

Text was encoded with byte-level BPE and a 50,257-token vocabulary. Every UTF-8
string can first be represented as bytes, so no separate unknown token is
required; frequent byte sequences are merged into longer units. This is a
useful compromise between byte universality and subword efficiency. Different
languages still require different numbers of tokens, so equal-length texts may
consume different fractions of the context window and compute budget. The
tokenizer is part of the model and its economics, not a neutral preprocessing
step.

## How one loop becomes several tasks

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-autoregression-2.gif]]

*GPT-2 autoregression: the selected token is appended to the prefix and becomes
part of the condition for the next step. Source: Jay Alammar, [The Illustrated
GPT-2](https://jalammar.github.io/illustrated-gpt2/), [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

There is no task switch in the animation. Computation is identical for every
prefix:

1. A causal Transformer turns prefix tokens into contextual states.
2. The final state is projected to scores for every vocabulary token.
3. The next token is selected greedily or by sampling.
4. That token is appended, and the loop repeats until stopping.

For translation, the prefix may look like `English sentence = French sentence`.
For summarization, an article ends in `TL;DR:`. For question answering, the
model receives a question and, when needed, a context. The difference lives in
the string rather than in the weights or output head. A modern server may cache
previous attention keys and values, but that is an inference optimization; the
underlying autoregressive conditional distribution is unchanged.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png]]

*Representation flow through a GPT-2-like block: masked self-attention,
residual paths, and the FFN. Source: Jay Alammar, [The Illustrated
GPT-2](https://jalammar.github.io/illustrated-gpt2/), [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Here *zero-shot* means no gradient training on the target dataset. It does not
mean that WebText contained no information about the task: the model may have
seen other questions, summaries, and translations. A more precise description
is transfer to a new dataset and format through conditional text context.

## What the experiments showed

The authors first evaluated pure language modeling. Without fine-tuning, the
largest GPT-2 improved the result on seven of eight examined corpora. On
LAMBADA, which requires recovering the final word from long-range context,
accuracy reached 63.24%. This establishes transfer across text distributions,
but it is not yet arbitrary instruction following.

The model was then applied to the Children's Book Test, Winograd Schema, CoQA,
CNN/DailyMail summarization, and translation. Results were mixed. On CoQA it
scored 55 F1, compared with 89.8 for a strong supervised system. Summaries after
`TL;DR:` were often coherent but substantially behind specialized models on
ROUGE. Translation appeared without parallel-data fine-tuning, but 5 BLEU for
English-to-French remained far below task-specific systems. The important
result was not victory on every benchmark; it was non-zero competence across
different tasks from one unchanged generator.

The task metric must not be confused with the training objective. GPT-2
optimizes the likelihood of every next WebText token, whereas BLEU, F1, and
ROUGE measure a narrow property of the final answer. This mismatch explains
why fluent text can receive a weak task score and why lower perplexity need not
produce a proportional gain in translation quality.

## Generation, memorization, and model release

For open-ended generation, decoding changes the observed text substantially.
Low temperature narrows the distribution and makes continuations more
predictable; overly permissive sampling increases diversity but breaks
coherence more often. The report used top-k sampling, so a few striking samples
should not be treated as a neutral sample of model behavior.

The authors searched for long matches with training documents and showed that
some examples could be memorized. Fluency proves neither generalization nor
truth: a continuation may reconstruct a seen passage or produce a plausible
fabrication. Concern about large-scale generation of deceptive content also led
to staged release of the weights—smaller configurations first, then 1.5B.
GPT-2 therefore made model-release policy part of the research discussion.

## Limitations and the transition to GPT-3

The task is specified by an implicit pattern, so a small change in phrasing can
change the result sharply. One marker is often insufficient to communicate
labels, style, and answer boundaries unambiguously. Autoregression accumulates
errors, and likelihood does not rule out toxic or false text. A 1024-token
window is small for long documents, WebText is biased toward the English web,
and the largest model is more expensive for every generated token.

GPT-2 framed a task as textual context and showed that this kind of transfer
improves with model scale. It still lacked a reliable way to explain a new
format. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3]] places several demonstrations directly in the context window and systematically compares zero-, one-, and few-shot regimes.

## Sources and further reading

- [Radford et al., Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — primary technical report: WebText, architecture, transfer evaluations, and memorization analysis.
- [OpenAI, GPT-2 code and model card](https://github.com/openai/gpt-2) — released weights, tokenizer, and reference implementation.
- [Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — the source of the autoregression and block illustrations used above.
- [[02 Areas/ML & DL/Papers/GPT 2.0|Local GPT-2 note]] — configuration and result tables.
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|Previous chapter: GPT-1]].
