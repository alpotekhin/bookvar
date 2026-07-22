---
title: "Decoding and next-token selection"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена.md"
last_updated: 2026-07-20
last_verified: 2026-07-22
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/
  - https://huggingface.co/blog/how-to-generate
  - https://huggingface.co/docs/transformers/generation_strategies
  - https://arxiv.org/abs/1904.09751
---

# Decoding and next-token selection

A language model does not emit a finished sentence. Given a prefix, it returns
one number for every token in its vocabulary. Decoding turns those numbers into
decisions: which token to append, when to stop, and which alternatives to retain
for the next step. The same model parameters can consequently produce an exact
translation, a repetitive loop, or several plausible continuations depending on
the selection procedure.

## From logits to a distribution

For prefix $x_{<t}$, the model computes logits $z_1,\ldots,z_{|V|}$. Softmax
defines the next-token distribution:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Two independent operations can follow.

1. A **distribution transformation** changes the eligible tokens or their relative probabilities: temperature, top-k, top-p, repetition penalties, or token bans.
2. A **search strategy** selects one or more paths: greedy decoding, random sampling, or beam search.

This distinction prevents a common confusion. `top_p=0.9` is not a complete
search method: it first truncates the distribution, after which a token must
still be selected. Likewise, `temperature=0.7` introduces no randomness if the
decoder ultimately takes `argmax`.

## Greedy decoding: the locally best token

Greedy decoding chooses

$$
x_t=\arg\max_{w\in V}p(w\mid x_{<t}).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-greedy_search.png]]

*The greedy path in Patrick von Platen's educational example,
[How to generate text](https://huggingface.co/blog/how-to-generate). At every
level the most probable branch is selected, giving a final probability of
$0.5\cdot0.4=0.2$.*

A local maximum need not belong to the most probable complete sequence. Choosing
`nice` with probability 0.5 appears better than `dog` with 0.4, but at the next
step `dog has` has probability $0.4\cdot0.9=0.36$ and overtakes `nice woman`.

Greedy decoding is useful when one short, reproducible output is required and
the distribution is sharply concentrated: classification through generation,
some extraction formats, or a quick smoke test. In long open-ended generation,
it often falls into repetitive high-probability patterns.

## Beam search: several hypotheses at once

Beam search retains the $B$ highest-scoring unfinished sequences. At each step,
it expands every sequence with candidate tokens, computes cumulative scores,
and keeps only the best $B$ continuations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-beam_search.png]]

*A beam width of two preserves both initial branches and discovers that the less
probable first token leads to the more probable pair. Source: Hugging Face,
[How to generate text](https://huggingface.co/blog/how-to-generate).*

For a sequence $y_{1:T}$, the natural score is the sum of log probabilities:

$$
s(y_{1:T})=\sum_{t=1}^{T}\log p(y_t\mid y_{<t},x).
$$

Because every log probability is non-positive, longer sequences receive lower
total scores. Translation and speech recognition therefore use length
normalization, for example

$$
s_\alpha(y)=\frac{s(y)}{|y|^\alpha}.
$$

Beam search is best suited to tasks tightly constrained by the input and with a
limited set of good realizations: machine translation, speech recognition, and
sometimes summarization. In open dialogue, model probability does not coincide
with human judgments of interest. A wider beam may amplify generic safe phrases
and repetition rather than improve the answer.

Beam search remains approximate: a branch pruned early cannot return. Unless the
beam is as wide as the set of all possible prefixes, it does not guarantee the
global maximum.

## Random sampling

Multinomial sampling draws a token from the distribution:

$$
x_t\sim p(\cdot\mid x_{<t}).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-sampling_search.png]]

*A branch is chosen randomly according to the displayed probabilities, so a
second run may follow a different path. Source: Hugging Face,
[How to generate text](https://huggingface.co/blog/how-to-generate).*

Unrestricted sampling preserves the model distribution, but the vocabulary has
a long tail of weakly plausible tokens. Each has low probability on one step,
yet the chance of at least one poor choice grows across a long sequence. This
motivates transformations that control the tail's shape or size.

## Temperature changes probability contrast

Temperature divides the logits by $T>0$:

$$
p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

For $T<1$, logit differences are amplified and the distribution sharpens. For
$T>1$, it flattens. The limit $T\to0$ approaches greedy selection, but APIs
should disable sampling explicitly for determinism rather than receive a zero
temperature.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-sampling_search_with_temp.png]]

*The same branches before and after lowering temperature. This diagram from
[Hugging Face](https://huggingface.co/blog/how-to-generate) depicts a concrete
redistribution of probability mass, not an abstract “creativity level.”*

Temperature does not know which tokens are factually wrong. It only changes the
confidence of the existing ranking. A low value can entrench an incorrect
maximum; a high value can increase the probability of a poor tail.

## Top-k: a fixed candidate count

Top-k retains the $k$ highest-probability tokens, zeros the rest, and
renormalizes the remaining mass:

$$
\tilde p_i=
\begin{cases}
p_i / \sum_{j\in K}p_j,&i\in K,\\
0,&i\notin K.
\end{cases}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-top_k_sampling.png]]

*Top-k applied to a real GPT-2 distribution; retained candidates are red.
Source: Patrick von Platen,
[How to generate text](https://huggingface.co/blog/how-to-generate).*

Fixed $k$ behaves poorly in two limiting cases. If the model is nearly certain
of one token, the other $k-1$ candidates add needless noise. If the distribution
is genuinely broad, the same $k$ may discard substantial plausible mass.

## Top-p: a variable set with fixed mass

Nucleus sampling sorts tokens and chooses the smallest set $V^{(p)}$ such that

$$
\sum_{w\in V^{(p)}}p(w\mid x_{<t})\ge p.
$$

Probabilities within this set are then renormalized and sampled.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-top_p_sampling.png]]

*With top-p, the number of retained red bars depends on distribution shape
rather than being fixed in advance. Source: Hugging Face,
[How to generate text](https://huggingface.co/blog/how-to-generate).*

This adaptive nucleus motivated Holtzman et al. in
[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751):
for open-ended generation, the tail of the model distribution is often
unreliable, while beam search favors dull high-probability continuations.

Modern APIs also expose `min_p`, which retains tokens whose probability is at
least a specified fraction of the leading token's probability. Like top-k and
top-p, it is a truncation heuristic, not a guarantee of factual correctness.

## Constraints and penalties

Logits may be modified further before selection.

- A **repetition penalty** reduces the appeal of tokens already seen.
- A **frequency penalty** grows with repetition count; a **presence penalty** depends only on whether the token appeared.
- **No-repeat n-gram** hard-bans repetition of a specified $n$-gram.
- **Constrained decoding** permits only tokens compatible with a grammar, JSON Schema, or required phrase list.

Penalties do not cure the cause of repetition and may damage necessary anaphora,
code, or tabular output. For structured output, a finite-state machine or grammar
is more reliable than arbitrary logit reductions: it guarantees syntax, though
not the truth of field values.

## Stopping is part of decoding

Generation ends at EOS, a length limit, or an external stopping criterion. A
string stop sequence requires care: it can cross token boundaries, occur inside
code, or already have been streamed to the user. Minimum length, maximum length,
and length penalties alter the distribution over complete answers, so they must
be recorded alongside temperature and top-p.

## Choosing a strategy

| task | reasonable starting point | what to test |
|---|---|---|
| exact extraction / JSON | greedy or low randomness plus a grammar | schema validity and semantics |
| translation / ASR | beam search with length normalization | quality versus beam width and latency |
| open dialogue | top-p or min-p with moderate temperature | diversity, factuality, repetition |
| several candidates | independent samples | genuine diversity, not paraphrases |
| code | sample candidates, then run tests | pass@k and verification cost |

Parameter values cannot be transferred mechanically between models. Logit
calibration, vocabulary, and chat templates differ. Even a model update behind
the same API changes the effective distribution.

## A reproducible experiment

Record the model and tokenizer identifiers, the prompt after applying the chat
template, the random seed, all `GenerationConfig` parameters, stopping criteria,
and library version. One seed is not enough to evaluate stochastic generation;
use multiple independent runs and report metric distributions.

Strategies should be compared at an equal candidate budget. One greedy answer
and 64 sampled answers followed by best-of selection use different amounts of
compute and constitute different systems, even if each finally displays one
string.

## Sources and further reading

- Jurafsky & Martin, [Speech and Language Processing, chapter 8](https://web.stanford.edu/~jurafsky/slp3/) — sequential generation, sampling, temperature, top-k, and top-p.
- Patrick von Platen, [How to generate text](https://huggingface.co/blog/how-to-generate) — ready-made visual examples of greedy, beam, and sampling methods.
- Hugging Face, [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) and [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation) — the exact mapping from methods to API parameters.
- Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — the motivation for nucleus sampling and an analysis of degeneration.
