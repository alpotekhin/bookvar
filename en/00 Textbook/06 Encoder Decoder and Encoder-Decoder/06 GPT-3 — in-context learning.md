---
title: GPT-3 — scaling and in-context learning
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources: [https://arxiv.org/abs/2005.14165]
---

# GPT-3: scaling and in-context learning

GPT-2 showed that a text prefix can sometimes specify a task without
fine-tuning, but one instruction or format marker is often ambiguous. A new
classifier must communicate both the task and the allowed labels; translation
must specify direction and style; a string transformation must reveal the exact
rule. GPT-3 asked whether a much larger language model could infer such a rule
from several input–answer pairs placed directly before a new query.

The architectural novelty was modest. The model remained a causal Transformer
trained to predict the next token. The largest configuration contained 175
billion parameters, and the paper systematically compared models from 125
million to 175 billion. The scientific question was not whether a new block
could solve the task, but how the behavior of one paradigm changed with
computational scale.

## Where the “learning” happens

Let $D$ be a text containing demonstrations, $q$ a new query, and $y$ the
answer. The model defines

$$
p(y\mid D,q)=\prod_t p(y_t\mid D,q,y_{<t}).
$$

Every element of $D$, $q$, and the generated prefix of $y$ belongs to one token
sequence. There is no backward pass, no parameter update, and no new output
head. Only hidden states in the current forward pass change: attention lets
each new token use the task description and demonstrated pairs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-3-paper/zero-one-few-shot-vs-finetuning.png]]

*Four adaptation regimes illustrated with translation. In the three left
panels only the input text changes; on the right, training examples update the
weights. Source: Brown et al., [Language Models are Few-Shot Learners, Figure
2.1](https://arxiv.org/abs/2005.14165), 2020.*

The paper distinguishes three regimes. **Zero-shot** contains a natural-language
description and a query. **One-shot** adds one complete demonstration.
**Few-shot** adds as many demonstrations as reasonably fit in the window,
usually several to several dozen. Fine-tuning sits outside this scale: examples
act through gradients and remain encoded in the weights after the update.

*In-context learning* names an observed adaptation of behavior, not ordinary
optimizer-based training. When the demonstrations are removed, the model does
not preserve the inferred rule as a new parameter state. This separates GPT-3
from GPT-1: GPT-1 changed its weights using downstream examples, whereas GPT-3
used examples as temporary memory within the prompt.

## Mechanics of a few-shot prompt

Consider sentiment classification with labels `positive` and `negative`. The
prefix contains an instruction, several reviews with correct labels, and one
new review without an answer.

1. The tokenizer turns the instruction, separators, demonstrations, and query
   into one sequence of length $T$.
2. The causal Transformer builds every position's state. The last position can
   see all preceding pairs but no token of the future answer.
3. Attention can relate the new review to the wording and format of the
   examples; the code constructs no explicit feature-to-label table.
4. The next-token distribution supplies the start of the label. If the label
   spans several tokens, the remainder is generated autoregressively.
5. Scoring must include the whole label string rather than only its first
   token.

Demonstration order, separators, class balance, and label wording become
interface hyperparameters. If the most recent examples all belong to one class,
the model may follow a local pattern instead of the content. One successful
prompt is therefore not a reliable evaluation of the method.

## Architecture: a familiar block at a new scale

GPT-3 retains a GPT-2-like decoder-only path with a 2048-token context window.
Its configurations alternate full attention with locally banded sparse
attention borrowed from Sparse Transformer. This reduces the cost of some
layers without changing causality or the sequence interface.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt-2-transformer-xl-bert-3.png]]

*The decoder-only GPT path in a comparative diagram. It shows the causal
backbone shared by GPT-2 and GPT-3, not the exact 175B configuration. Source:
Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

In few-shot use, context length directly limits the number of examples.
Full-attention scores have shape `[B,h,T,T]`; demonstrations consume memory and
are recomputed for each new request. They can be cheaper than fine-tuning when
tasks are numerous and examples scarce, but they are not free: a longer prompt
increases latency and leaves less room for the answer.

## The scaling experiment

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-3-paper/model-sizes-table-2-1.png]]

*The eight GPT-3 configurations. Source: Brown et al., [Language Models are
Few-Shot Learners, Table 2.1](https://arxiv.org/abs/2005.14165), 2020.*

The authors trained eight sizes from 125M to 175B to observe a curve rather
than one striking endpoint. Parameter count, depth, width, head count, and token
batch size grew together, while the learning rate decreased. Every
configuration processed roughly 300 billion tokens. The table therefore does
not isolate the causal effect of parameter count alone; it describes matched
training regimes for different compute budgets.

The 175B model had 96 layers, hidden width 12,288, and 96 attention heads. These
numbers determine cost but do not by themselves explain transfer. The tested
hypothesis was that lower language-modeling error in a larger network would
produce stronger zero- and few-shot behavior without gradient adaptation.

## Data and sampling regime

The training mixture combined filtered Common Crawl, an expanded WebText, two
book corpora, and English Wikipedia. After filtering and deduplication, the
available corpus was far larger than 300 billion tokens, so sampling was not
uniform. Smaller, higher-quality sources were sampled more often: WebText2 and
books could repeat several times, while much of Common Crawl was seen less than
once.

Data scale is therefore more than a byte count. Quality filters,
deduplication, and source weights determine which genres the model sees
repeatedly. Common Crawl broadens knowledge and task formats while importing
errors, social biases, and personal data from the open web. English dominates,
so strong English results cannot be assumed to transfer automatically to other
languages.

## Reading the results without oversimplification

The paper covers translation, question answering, cloze tests, commonsense,
arithmetic, letter rearrangement, reading, and news generation. On many tasks,
few-shot performance improved with model size much faster than zero-shot
performance. The largest model set a new closed-book result on TriviaQA;
few-shot prompting sharply improved LAMBADA accuracy; and the number of digits
handled in simple addition grew with scale.

A general interface did not create general reliability. GPT-3 still trailed
fine-tuned models on some datasets, and scale helped little on tasks such as
ANLI. In arithmetic it often reproduced surface patterns and failed at lengths
or formats outside its distribution. Few-shot demonstrations reduced task
ambiguity but did not guarantee a robust algorithm.

The comparison across sizes is particularly revealing. Extra demonstrations
sometimes did little for small configurations, while the gap between zero- and
few-shot grew for large models. This observation made in-context learning a
research topic of its own. It does not prove that a capability suddenly appears
at one threshold: the shape of the curve depends on the task, metric, and
prompt.

## Benchmark contamination and honest evaluation

At web scale, a test example or a near-duplicate may enter pre-training. The
authors searched for benchmark overlap and compared contaminated and clean
subsets, but a filtering error prevented complete removal of all detected
matches before training. Post-hoc analysis reduces uncertainty; it cannot
restore an ideal controlled experiment.

Evaluating in-context learning therefore requires deduplication against the
test set, templates specified in advance, several demonstration orders, and
variance reporting. If a prompt is chosen after inspecting test answers, the
prompt itself becomes a channel for overfitting even though model weights are
frozen.

## Limitations of base GPT-3

The pre-trained model optimizes plausible continuation rather than user intent.
Given a question, it may continue discussing the question instead of answering
briefly; given a false premise, it may accept and elaborate it. It has no built-
in verification of facts, sources, or calculations. It reproduces corpus
stereotypes and can generate abusive text.

The cost of training 175B and of sequential inference limits experimental
access. A 2048-token window forces a trade-off among demonstration count, task
length, and answer length. Sensitivity to wording and order destabilizes
evaluation. Finally, a few-shot prompt does not correct knowledge in the
weights and does not create persistent memory between requests.

## Legacy and the next transition

GPT-3 made a prompt with demonstrations the primary API of a frozen model and
showed that scale affects not only perplexity but also how a new task can be
specified. Prompting research subsequently separated from fine-tuning, and
context came to be viewed as temporary working memory.

A base language model nevertheless remains a text completer. The next stage—
instruction tuning and learning from human preferences—changes the weights so
that a natural-language instruction more reliably elicits a useful answer. It
does not replace in-context learning: the tuned model still uses examples and
data from the current window.

## Sources and further reading

- [Brown et al., Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — primary paper: architecture, data mixture, prompting regimes, experiments, and contamination analysis.
- [OpenAI, Language models are few-shot learners](https://openai.com/index/language-models-are-few-shot-learners/) — publication page.
- [[02 Areas/ML & DL/Papers/GPT 3.0|Local GPT-3 note]] — result tables and historical context.
- [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 10 — Prompting, RLHF and DPO|CS224N: prompting, instruction tuning, and preference learning]] — the transition to the next stage.
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|Previous chapter: GPT-2]].
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|Next chapter: T5 and an explicit text-to-text interface]].
