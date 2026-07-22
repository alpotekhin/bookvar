---
title: "Speculative decoding"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md"
last_updated: 2026-07-20
last_verified: 2026-07-22
primary_sources:
  - https://arxiv.org/abs/2211.17192
  - https://arxiv.org/abs/2302.01318
  - https://research.google/blog/looking-back-at-speculative-decoding/
  - https://cs336.stanford.edu/
---

# Speculative decoding

A large autoregressive model normally adds one token per forward pass. Although
several adjacent positions can be evaluated in parallel, they do not yet exist:
each next position depends on the token selected immediately before it.
Speculative decoding temporarily removes this dependency. A fast **draft**
proposes several tokens; in one pass, the main **target** model computes
distributions for every proposed position and determines which prefix can be
accepted. A correct acceptance procedure preserves the target distribution
rather than merely copying successful guesses from a smaller model.

## Where the speedup comes from

Decode for a large Transformer is often limited by reading parameters from HBM.
Almost the entire model must be read to produce one new token. Adding several
positions to the pass enlarges the matrix operations, but does not require
rereading the weights once per position. Verifying $k$ draft tokens can therefore
cost only slightly more than one ordinary target step.

The draft, in turn, must be cheap enough that generating $k$ candidates
sequentially does not consume the gain. The resulting pipeline is:

1. the draft autoregressively proposes $\tilde x_1,\ldots,\tilde x_k$;
2. the target obtains distributions for $k+1$ positions in one parallel pass;
3. proposals are checked left to right until the first rejection;
4. after a rejection, the next token is sampled from a correction distribution;
5. if all $k$ proposals are accepted, the target contributes one additional token.

One expensive target pass advances generation by between one and $k+1$ tokens.

## Why matching argmax is not enough

For greedy decoding, the rule is simple: accept each draft token while it equals
the target argmax; at the first mismatch, use the target argmax. This produces
exactly the same output as ordinary greedy target decoding.

For sampling, “the token appears sufficiently probable” is not a valid test.
Let the draft distribution be $p$, the target distribution $q$, and the proposal
$\tilde x\sim p$. Accept it with probability

$$
a(\tilde x)=\min\left(1,\frac{q(\tilde x)}{p(\tilde x)}\right).
$$

If the draft underestimates a token relative to the target, $q/p\ge1$, so the
proposal is always accepted. If the draft proposes a token too frequently, only
a fraction $q/p$ is accepted. On rejection, the replacement is not sampled
directly from $q$, but from the residual distribution

$$
q'(x)=\frac{(q(x)-p(x))_+}{\sum_y(q(y)-p(y))_+}.
$$

The residual is needed only after rejection. If $p=q$, its denominator is zero,
but rejection also has zero probability, so this branch is unreachable.

The accepted portion accounts for the mass shared by $p$ and $q$; the residual
restores the target's remaining mass. Together they reproduce probability
$q(x)$ for every token. This is why speculative sampling can be **lossless with
respect to the specified target decoding procedure**.

## The full algorithm in the original figure

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/speculative-sampling-algorithm.png]]

*Algorithm 2 from Leviathan et al., [Fast Inference from Transformers via
Speculative Decoding](https://arxiv.org/abs/2211.17192), as shown in
[Stanford CS336: Inference](https://cs336.stanford.edu/). The upper portion has
the draft generate $K$ tokens sequentially and the target compute $K+1$ sets of
logits in parallel; the lower portion gives the acceptance ratio and residual
distribution.*

The statement that the target verifies tokens in parallel needs qualification.
One target forward pass receives the prompt followed by all draft tokens. The
causal mask ensures that logits at position $i$ depend only on earlier proposals,
so the correct conditional distribution is available at every verification
point.

After the first rejection, all later draft positions are invalid because they
were conditioned on a prefix containing the rejected token. They cannot be
accepted independently even if they receive high target scores.

## Acceptance connects quality to speed

Suppose the average probability of accepting the next draft token is
approximately $\alpha$. Under a rough independence assumption, the expected
number of accepted draft tokens is

$$
\mathbb E[A]=\sum_{i=1}^{k}\Pr(A\ge i)
\approx\sum_{i=1}^{k}\alpha^i.
$$

At $\alpha=0.9$, a long draft often pays off; at $\alpha=0.4$, later positions
are rarely reached. Increasing $k$ also lengthens both target verification and
draft generation. The best length depends on the prompt, domain, batch size,
and hardware.

Experiments in the original paper show speed and preserved quality together on
XSum and HumanEval.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/speculative-sampling-results.png]]

*Table 1 from Leviathan et al. for $K=4$: speculative sampling reduces time per
token by roughly 1.9–2.46 times, while metrics remain within the statistical
variation of ordinary sampling. Source:
[Leviathan et al.](https://arxiv.org/abs/2211.17192); table reproduced in the
Stanford CS336 slides.*

This is not a universal speedup. The table represents particular models,
hardware, tasks, and batch size one. On another stack, extra kernels, KV-cache
copies, and scheduler behavior may change the result.

## Why more draft tokens are not always better

The paper's plots show three competing effects. Raising $K$ initially lowers
mean sampling time, but the improvement then saturates. Acceptance falls for
later draft positions, while each verification pass grows longer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/speculative-sampling-stats.png]]

*Mean time to generate 128 tokens (left), acceptance by draft position (center),
and duration of one verification cycle (right). Figure from Leviathan et al.,
[Speculative Decoding](https://arxiv.org/abs/2211.17192), reproduced in CS336.*

Choose $K$ by minimum end-to-end latency, not by the largest number of tokens
per target pass. Modern implementations may adapt draft length according to
recent acceptances.

## What can serve as a draft

A smaller model is only the first option.

- A **separate draft model** shares the tokenizer and is trained to approximate the target.
- **Early exit** uses intermediate layers of the target model.
- **Additional heads**, as in Medusa, propose several future tokens in parallel from a shared hidden state.
- **Prompt lookup** takes a likely continuation from a repeated input fragment; it is useful in summarization and editing, where output copies the document.
- **N-gram or speculative candidates** reuse sequences already seen without another neural network.

In every case, target verification remains the authority for the final
distribution. The proof of exactness depends on the procedure, however: trees
of candidates, multiple heads, and stochastic sampling each require an
appropriate acceptance algorithm.

## The two models' KV caches

Draft and target maintain separate KV caches. After $a$ tokens are accepted,
both caches must represent the same confirmed prefix; states corresponding to
rejected draft positions are removed or overwritten. A large draft model adds
parameters and cache memory that can reduce the available batch size.

If draft and target use different tokenizers, “one draft token” no longer maps
to one target position. Universal assisted generation retokenizes text across
vocabularies and must synchronize boundaries carefully. This broadens
applicability but adds computation and edge cases.

## When speculation does not help

Speculation is most attractive at batch size one or under light interactive
load, where target decode is memory-bound. Its advantage diminishes when:

- a large continuous batch already amortizes parameter reads effectively;
- the draft is expensive or runs on the same saturated device;
- the target domain differs greatly from draft training and acceptance is low;
- responses are short enough that setup cost matters;
- constrained decoding often forbids draft proposals;
- the verification kernel performs poorly for the selected $K$;
- tensor-parallel communication grows with verification length.

Speculation may raise single-request throughput yet reduce aggregate goodput if
the draft occupies memory that previously held more target requests.

## How to measure it

A comparison with ordinary target decoding must hold the output distribution
fixed: temperature, top-p, seed policy, and stop conditions. Then measure:

- acceptance rate by draft position, domain, and response length;
- mean confirmed tokens per target pass;
- draft, verification, and correction time separately;
- p50 and p95 TTFT and TPOT;
- additional parameter memory and both KV caches;
- throughput and goodput at several batch sizes, not only batch-one latency;
- statistical agreement with baseline target sampling.

The final check is fundamental. An implementation that merely accepts probable
tokens without residual correction may be fast, but it implements a different
decoding distribution and may change quality.

## Sources and further reading

- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — acceptance/rejection sampling and the original experiments.
- Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — an independent derivation of exact sampling.
- Google Research, [Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/) — a visual history of the method and an interactive video.
- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/) — the connection to prefill/decode and memory-bound serving.
- Hugging Face, [Assisted generation](https://huggingface.co/docs/transformers/generation_strategies#speculative-decoding) — practical draft-model and prompt-lookup variants.
