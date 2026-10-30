---
title: "Speculative decoding"
type: textbook-chapter
status: canonical
last_updated: 2026-10-19
last_verified: 2026-10-19
source_language: mixed
source_unit_id:
  - lecture-10-section-341-use-shortcuts-but-double-check-lossless
  - lecture-10-figure-step-353-rendering-1
  - lecture-10-figure-step-366-rendering-1
  - lecture-10-figure-step-367-rendering-1
primary_sources:
  - https://arxiv.org/abs/2211.17192v2
  - https://arxiv.org/abs/2302.01318v1
  - https://arxiv.org/html/2401.15077v3
  - https://aclanthology.org/2024.emnlp-main.422/
  - https://arxiv.org/html/2503.01840v3
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
locale: en
translation_status: current
canonical_revision: "32085af4415e2aa3904eded4a14c42e7ced65c81d4288c42af4a5d099ff597ad"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md"
---

<a id="спекулятивное-декодирование"></a>

# Speculative decoding

Suppose a language model greedily continues a prefix with A, B, D.
Ordinary decoding runs it for A, then B, then D. If a small model proposes
A, B, C in advance, the main model can check the entire proposal in one
pass: accept A and B, reject C, and emit D. One expensive pass produces
three new tokens.

The small model produces a **draft**; the main **target** model determines
the final continuation. Greedy decoding can compare proposals with target
argmax tokens. Sampling needs a different rule that preserves the entire
distribution, including unlikely outcomes.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speculative-review-2026-10/leviathan-generation.png]]

*Yaniv Leviathan, Matan Kalman and Yossi Matias,
[Fast Inference from Transformers via Speculative Decoding, Figure 1](https://arxiv.org/abs/2211.17192v2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Each row is one cycle. Green marks accepted proposals, red strikethrough
marks rejections, and blue marks replacements. In the first row, four
proposals are accepted, then “n” replaces “bond”: five tokens from one
target pass. Unmodified original from the authors' archive.*

<a id="откуда-берётся-возможность-ускорения"></a>

## Where the speedup comes from

For a single request, dense-model decode often spends more time reading
weights than multiplying them. One activation row passes through the layers
for each new token. Processing several known tokens together combines rows
into matrix operations and reuses loaded weights across positions.

The continuation is initially unknown, so the draft proposes it first.
The target then processes known text. For prefix $s$ and candidates
$d_1,d_2,d_3$, it needs

$$
\begin{aligned}
q_1&=q(\cdot\mid s),\\
q_2&=q(\cdot\mid s,d_1),\\
q_3&=q(\cdot\mid s,d_1,d_2),\\
q_4&=q(\cdot\mid s,d_1,d_2,d_3).
\end{aligned}
$$

Causal masking prevents an earlier position from seeing later proposals.
One pass over the known continuation therefore supplies the same conditional
distributions required for sequential verification. It evaluates known inputs;
it does not independently guess several unknown future tokens.

The draft itself is generated sequentially. A length-$K$ draft normally
requires $K$ cheap steps and one expensive verification pass. Each cycle
emits the accepted prefix plus one token: a correction on rejection or a
bonus after complete acceptance. Before EOS or the length limit intervenes,
that is between one and $K+1$ tokens.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speculative-review-2026-10/leviathan-timeline.png]]

*Leviathan et al.,
[Figure 5: simplified trace diagram](https://arxiv.org/abs/2211.17192v2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The bottom row is ordinary decoding; upper rows insert small blue draft
steps before large purple verification blocks. Here $M_p$ is the target
and $M_q$ the draft, opposite to this chapter's Chen et al. convention:
draft $p$, target $q$. Yellow and orange blocks belong to the experiment's
encoder-decoder model; decoder-only LLMs have no separate encoders.
Complete rendering of the authors' vector PDF, uncropped.*

[Chen et al., Conditional Scoring](https://arxiv.org/abs/2302.01318v1)
separately discusses linear layers, KV reads and device communication.
For small $K$, linear layers reuse loaded weights, and small collectives
may remain latency-bound rather than transfer-volume-bound. Increasing
$K$ adds arithmetic, activations and traffic. Verification need not remain
as cheap as one ordinary step. See
[[55a Физика LLM inference — prefill, decode и roofline|prefill, decode and roofline]].

<a id="почему-нельзя-просто-принять-совпавший-argmax"></a>

## Why matching argmax is not enough

<a id="жадное-декодирование"></a>

### Greedy decoding

Accept candidates while each matches the target argmax conditioned on the
accepted prefix. At the first mismatch, emit the target argmax and end
the cycle. Later distributions were evaluated under the wrong prefix
and cannot be used.

| Position | Verification prefix | Candidate | Target argmax | Action |
|---|---|---|---|---|
| 1 | $s$ | A | A | Accept A |
| 2 | $s$, A | B | B | Accept B |
| 3 | $s$, A, B | C | D | Emit D and end the cycle |

A, B, D matches three ordinary greedy steps. Continuing beyond C would
be invalid even if a later proposal matched the argmax conditioned on
$s$, A, B, C.

<!-- checked-example: speculative-greedy -->
```python
import numpy as np

def greedy_verify(candidates, target_rows):
    output = []
    for i, candidate in enumerate(candidates):
        chosen = int(np.argmax(target_rows[i]))
        output.append(chosen)
        if chosen != candidate:
            return output
    output.append(int(np.argmax(target_rows[-1])))
    return output

# A=0, B=1, C=2, D=3. Последняя строка нужна только при полном принятии.
rows = np.array([[.7, .1, .1, .1],
                 [.1, .7, .1, .1],
                 [.1, .1, .1, .7],
                 [.1, .1, .7, .1]])
assert greedy_verify([0, 1, 2], rows) == [0, 1, 3]
```

With fixed logits and identical tie-breaking, the argument gives identical
greedy sequences. On hardware, multi-position verification may use different
kernels and rounding orders from a single-position step. Mathematical
equivalence does not guarantee bitwise-identical logits.

<a id="случайная-выборка"></a>

### Sampling

Sampling must sometimes emit a token other than the most likely one.
Accepting only argmax matches would lose those outcomes.
Consider vocabulary A, B, C:

| Token | Draft $p$ | Target $q$ |
|---|---:|---:|
| A | 0.6 | 0.2 |
| B | 0.3 | 0.5 |
| C | 0.1 | 0.3 |

The draft proposes A three times as often as the target requires.
Accept one third of A proposals. Always accept B and C proposals because
the draft proposes them less often than required.

For candidate $x\sim p$:

$$
a(x)=\min\left(1,\frac{q(x)}{p(x)}\right).
$$

Acceptance probabilities are $1/3,1,1$. Unconditional probabilities of
“proposed and accepted” are

$$
\min(p,q)=(0.2,0.3,0.1).
$$

These sum to $0.6$. Rejection occurs with probability $0.4$.
The target mass still missing is $(0,0.2,0.2)$; normalized, it becomes
$(0,0.5,0.5)$. Sample the replacement from that distribution.

In general, the residual or correction distribution is

$$
r(x)=
\frac{\max(0,q(x)-p(x))}
{\sum_y\max(0,q(y)-p(y))}.
$$

The calculation can be checked directly:

<!-- checked-example: speculative-mass -->
```python
proposal = np.array([.6, .3, .1])
target = np.array([.2, .5, .3])
acceptance = np.minimum(1, target / proposal)
accepted_mass = proposal * acceptance
reject_probability = 1 - accepted_mass.sum()
residual = np.maximum(target - proposal, 0)
residual /= residual.sum()

np.testing.assert_allclose(acceptance, [1/3, 1, 1])
np.testing.assert_allclose(accepted_mass, [.2, .3, .1])
np.testing.assert_allclose(residual, [0, .5, .5])
np.testing.assert_allclose(accepted_mass + reject_probability * residual, target)

# Если после отказа заново выбирать из target, получится другое распределение.
wrong = accepted_mass + reject_probability * target
np.testing.assert_allclose(wrong, [.28, .5, .22])
```

Sampling directly from $q$ after rejection would give A probability
$0.28$ instead of $0.2$. The accepted portion already supplied part of
$q$; correction must supply only the missing mass.

For any vocabulary, accepted mass for $x$ is $\min(p(x),q(x))$.
Rejection probability equals the sum of positive differences $q-p$,
so the unconditional replacement mass is $\max(0,q(x)-p(x))$.
Together:

$$
\begin{aligned}
&\min(p(x),q(x))\\
&\quad+\max(0,q(x)-p(x))=q(x).
\end{aligned}
$$

See [Leviathan et al., Appendix A.1](https://arxiv.org/abs/2211.17192v2)
and [Chen et al., Theorem 1](https://arxiv.org/abs/2302.01318v1).
When $p=q$, rejection is impossible and the zero residual is never normalized.
When $p(x)=0$, the draft cannot propose $x$, but correction can still emit it.

<a id="какие-вероятности-подставлять"></a>

### Which probabilities belong in the test

$p$ must be the actual proposal distribution, and $q$ the desired target
decoding distribution. Apply temperature, top-k, top-p, token constraints
and other transformations before computing the ratio and residual.
Raw softmax probabilities are wrong if sampling used a truncated,
renormalized vocabulary. Draft settings may differ from target settings.

Leviathan et al. states this convention in §2.2. The original paragraph is
reproduced under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It uses $p$ for target and $q$ for draft, opposite to our notation.

<!-- source-excerpt: speculative-standardized-sampling:start -->
> First, note that while there are many methods and parameters of sampling, like argmax, top-k, nucleus, and setting a temperature, and popular implementations usually treat them differently at the logits level, they can all easily be cast into standard sampling from an adjusted probability distribution. For example, argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing. We can therefore only deal with standard sampling from a probability distribution, and cast all of the other types of sampling into that framework. Going forward we'll assume that $p(x)$ and $q(x)$ are the distributions from $M_p$ and $M_q$ respectively, adjusted for the sampling method.
<!-- source-excerpt: speculative-standardized-sampling:end -->

Preserving the distribution means preserving the sequence law under the
same target policy and stopping rules. Equal seeds need not yield the same
string: ordinary and speculative algorithms consume random numbers differently.

<a id="один-цикл-принять-b-отклонить-a-выдать-замену"></a>

## One cycle: accept B, reject A, emit a correction

Take draft B, A, C. Store each proposal and the distribution that produced
it. The target supplies four distributions for the corresponding prefixes:

| Prefix | $p$ over A, B, C | $q$ over A, B, C | Candidate |
|---|---|---|---|
| $s$ | 0.6, 0.3, 0.1 | 0.2, 0.5, 0.3 | B |
| $s$, B | 0.5, 0.2, 0.3 | 0.1, 0.6, 0.3 | A |
| $s$, B, A | 0.2, 0.3, 0.5 | 0.2, 0.3, 0.5 | C |
| $s$, B, A, C | — | 0.1, 0.2, 0.7 | — |

B is always accepted because $0.5/0.3>1$. A's ratio is $0.1/0.5=0.2$.
With verification uniform $0.7$, A is rejected. Under prefix $s$, B,
the residual is $(0,0.4,0)$, normalized to $(0,1,0)$.
The correction is B, so the cycle emits B, B.

Distributions conditioned on $s$, B, A and $s$, B, A, C are discarded.
The actual continuation now begins $s$, B, B and needs another verification.

The following function checks prepared distributions. It does not execute
neural networks and is not a GPU implementation.

<!-- checked-example: speculative-cycle -->
```python
def categorical_from_uniform(probability, u):
    assert 0 <= u < 1
    return int(np.searchsorted(np.cumsum(probability), u, side="right"))

def verify_chain(candidates, proposals, targets, uniforms, replacement_uniform):
    accepted = []
    for i, token in enumerate(candidates):
        p, q = proposals[i], targets[i]
        assert p[token] > 0
        if uniforms[i] < min(1.0, q[token] / p[token]):
            accepted.append(token)
            continue
        residual = np.maximum(q - p, 0)
        residual /= residual.sum()  # Эта ветвь достижима только при ненулевом остатке.
        correction = categorical_from_uniform(residual, replacement_uniform)
        return accepted + [correction], len(accepted), "reject"
    bonus = categorical_from_uniform(targets[-1], replacement_uniform)
    return accepted + [bonus], len(accepted), "all-accepted"

p_rows = np.array([[.6, .3, .1], [.5, .2, .3], [.2, .3, .5]])
q_rows = np.array([[.2, .5, .3], [.1, .6, .3],
                   [.2, .3, .5], [.1, .2, .7]])
# A=0, B=1, C=2
assert verify_chain([1, 0, 2], p_rows, q_rows, [.8, .7, .4], .4) == (
    [1, 1], 1, "reject"
)
assert verify_chain([1, 0, 2], p_rows, q_rows, [.8, .1, .4], .4) == (
    [1, 0, 2, 2], 3, "all-accepted"
)
```

In the second call, A's verification uniform is $0.1$, so A is accepted.
C is then accepted, and the final target row supplies bonus C.
The result B, A, C, C contains $K+1=4$ tokens.

An accepted EOS or the output length limit stops emission. Later tokens
from that cycle are not appended. The $K+1$ statement therefore assumes
no early stop.

<a id="полный-алгоритм-на-исходном-слайде"></a>

## The full algorithm in the original slide

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/speculative-sampling-algorithm.png]]

*Charlie Chen et al.,
[Algorithm 2, Speculative Sampling](https://arxiv.org/abs/2302.01318v1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
also in [Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L507-L535).
The upper loop builds the draft, the target supplies $K+1$ distributions,
and the lower loop accepts a prefix until the first rejection. The code
above separates correction and bonus-token branches.*

Target logits are computed in parallel, but prefix verification is logically
sequential. A position's distribution is valid only after all earlier
proposals have been accepted. Otherwise its conditioning prefix changes.

The classical algorithm uses complete $p$ and $q$ distributions.
Specialized algorithms and kernels may organize storage differently,
but the chosen token's probability alone is insufficient for residual correction.

<a id="kv-cache-двух-моделей"></a>

## The two models' KV caches

Verification leaves unconfirmed states in KV storage. After rejection,
those states must become invisible to future attention. Physical pages may
be freed later; correctness depends on visible length and correspondence
to the confirmed prefix.

A selected token may not yet have been processed. Its logits came from
the previous input; its own KV appears only on the next forward.
[[55 KV-cache, пакетирование и PagedAttention|Selecting a token and writing its KV]]
must be distinguished in ordinary and speculative decoding alike.

Consider a target with five confirmed tokens u, v, x, y, z.
Its cache contains the first four and z is pending. With draft B, A, C,
one target pass receives z, B, A, C, supplying the four rows above.

| Moment | Confirmed text | Target KV |
|---|---|---|
| Before verification | u v x y z | u v x y |
| After forward, before acceptance | u v x y z | u v x y z B A C; the suffix is temporary |
| B accepted, A rejected, correction B | u v x y z B B | u v x y z B |
| Before next verification | Final B is pending | Rejected A, C must be invisible |

The last B in text and the last B in cache are different positions.
The cache contains the accepted B; the correction B is still unprocessed.

<!-- checked-example: speculative-kv -->
```python
confirmed = list("uvxyz")
cached_before = confirmed[:-1]
pending = confirmed[-1]
draft = list("BAC")
temporary_cache = cached_before + [pending] + draft
accepted_count = 1
committed_cache = temporary_cache[:len(confirmed) + accepted_count]
new_confirmed = confirmed + draft[:accepted_count] + ["B"]

assert temporary_cache == list("uvxyzBAC")
assert committed_cache == list("uvxyzB")
assert new_confirmed == list("uvxyzBB")
assert committed_cache == new_confirmed[:-1]
```

The draft maintains its own state, synchronized to the accepted path.
Its cache need not have the same length as the target's at every instant.
After sequential drafting, the final proposal may not yet have KV.
Complete acceptance must account for that token and the target's bonus.
An implementation may process them together, retain a pending suffix,
or synchronize them another way.

A tree may require gathering one accepted path from noncontiguous cache
positions rather than truncating a simple suffix. Other branches must
remain invisible.

<a id="вероятность-принятия-и-число-токенов-за-цикл"></a>

<a id="процент-принятия-связывает-качество-и-скорость"></a>

## Acceptance probability and progress per cycle

For one prefix, proposal acceptance probability is

$$
\begin{aligned}
\beta&=\sum_x\min(p(x),q(x))\\
&=1-\frac12\sum_x|p(x)-q(x)|.
\end{aligned}
$$

The second expression uses total variation distance.
The three-token example has $\beta=0.6$. Sharing an argmax does not determine
this value: distributions can agree on the leading token while differing
substantially elsewhere.

Let $A$ count consecutive accepted candidates out of $K$.
Accepting at least $i$ requires passing all first $i$ checks. For any such $A$:

$$
\mathbb E[A]=\sum_{i=1}^{K}\Pr(A\ge i).
$$

If every conditional acceptance probability is approximated by a constant
$\alpha$, then

$$
\begin{aligned}
\mathbb E[N]
&=1+\mathbb E[A]\\
&=\sum_{i=0}^{K}\alpha^i\\
&=\frac{1-\alpha^{K+1}}{1-\alpha}.
\end{aligned}
$$

$N$ includes the correction or bonus token. At $\alpha=1$, the limit is
$K+1$. Real conditional probabilities depend on prefixes; multiplying
observed positionwise averages need not give the exact path-survival probability.

[Leviathan et al., §3.1–3.3](https://arxiv.org/abs/2211.17192v2) relates
progress to time. With ordinary target step $T_0$, draft step $T_d$ and
verification time $T_v(K)$:

$$
S(K)\approx
\frac{T_0\,\mathbb E[N]}
{K T_d+T_v(K)}.
$$

This approximates long-run stationary execution without extra overhead.
It is not a TTFT equation or guaranteed speedup.

<a id="почему-больше-draft-токенов-не-всегда-лучше"></a>

## Why more draft tokens are not always better

Use illustrative, unmeasured values:
$T_0=10$ ms, $T_d=1$ ms, $T_v(K)=10+0.5K$ ms and $\alpha=0.8$.

| $K$ | Expected output tokens per cycle | Cycle time, ms | Speedup |
|---|---:|---:|---:|
| 1 | 1.8000 | 11.5 | 1.57× |
| 2 | 2.4400 | 13.0 | 1.88× |
| 4 | 3.3616 | 16.0 | 2.10× |
| 8 | 4.3289 | 22.0 | 1.97× |

Doubling proposals from four to eight gains about one token per cycle but
adds six milliseconds. Under these costs, $K=8$ loses to $K=4$.

<!-- checked-example: speculative-speed -->
```python
def expected_progress(alpha, k):
    return sum(alpha**i for i in range(k + 1))

def toy_speedup(alpha, k):
    return 10 * expected_progress(alpha, k) / (k + 10 + .5*k)

np.testing.assert_allclose(
    [expected_progress(.8, k) for k in [1, 2, 4, 8]],
    [1.8, 2.44, 3.3616, 4.32891136]
)
assert toy_speedup(.8, 4) > toy_speedup(.8, 8)
assert expected_progress(0, 4) == 1
assert expected_progress(1, 4) == 5
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speculative-review-2026-10/chen-lookahead.png]]

*Chen et al., [Figure 1](https://arxiv.org/abs/2302.01318v1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Left: time to generate 128 tokens. Center: the authors' acceptance
efficiency, accepted output divided by $K+1$. Right: cycle time.
The center curve is not the individual-check $\alpha$ used above.
XSum latency is lowest around $K=3$. Full rendering of the vector
figure included in the paper, uncropped.*

Different tasks already had different optimal settings in the original
work. A longer draft helps only while additional progress pays for
drafting and verification. Dynamic lookahead estimates this per request
instead of fixing one $K$ for every case.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/speculative-sampling-results.png]]

*Chen et al., Table 1, also used in CS336.
Chinchilla 70B, 4B draft, batch one, $K=4$, 16 TPU v4:
reported speedups are about 1.9–2.0× on XSum and 2.46× on HumanEval.
Sampling settings are listed in the table and paper.
These are the authors' experiment, not Bookvar measurements.
[Original](https://arxiv.org/abs/2302.01318v1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The paper chose a shallow draft partly to reduce communication latency
on the same TPU devices. Fewer parameters alone do not imply the fastest
step on a particular device configuration.

<a id="чем-может-быть-draft"></a>

## What can serve as a draft

A separate small language model already supplies vocabulary probabilities.
It may use an existing checkpoint or be trained to imitate the target.
Closer distributions reduce rejection on the intended inputs, but a
larger draft costs more per proposal.

Other approaches use intermediate target layers, early-exit self-speculation,
additional prediction heads, or repeated text. Prompt lookup matches the
current suffix to an earlier input fragment and proposes its continuation.

For example, a document may already contain “payment is due within
30 days.” After another “payment is due within,” the system can propose
“30 days” without a draft neural network. The target must still verify it.
A deterministic proposal can be represented as a point distribution;
sampling must use that actual proposal mechanism, not imaginary logits.

Medusa and EAGLE can propose trees instead of chains. Trees cover alternatives
from one prefix but increase verification positions, temporary KV and mask
construction costs. A tree by itself does not establish distribution preservation;
that depends on the verifier.

Different tokenizers create another problem: a draft-token boundary may
fall inside a target token. Retokenization must synchronize prefixes.
[Universal Assisted Generation](https://huggingface.co/blog/universal_assisted_generation)
describes this extension; two incompatible vocabularies cannot be inserted
directly into the same $q/p$ formula.

<a id="eagle-draft-на-уровне-признаков"></a>

## EAGLE: drafting in feature space

<a id="eagle-1-почему-нужен-уже-выбранный-следующий-токен"></a>

### EAGLE-1: why the selected next token is an input

Suppose the target processed “How can” and selected “I.”
Its state after “can” defines a next-token distribution, but is not yet
the state after “I.” Predicting that next state requires knowing which
token was selected: “I” or another allowed continuation.

[EAGLE, §3.1](https://arxiv.org/html/2401.15077v3#S3.SS1) gives its small draft
features $f_1,\ldots,f_t$ from the target Transformer and shifted embeddings
$e_2,\ldots,e_{t+1}$. At width $d$, concatenation has shape $[B,L,2d]$.
A linear projection reduces it to $d$; one decoder block predicts
$\hat f_{t+1}$. The frozen target LM head converts that feature into the
distribution for token $t+2$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speculative-review-2026-10/eagle-shifted-input.png]]

*Li et al., [EAGLE, Figure 6, ICML 2024, PMLR 235](https://proceedings.mlr.press/v235/li24bt.html),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The bottom drops $t_1$. Embeddings $e_2,e_3$ align with $f_1,f_2$, not
$f_2,f_3$: a one-position shift. Projection and the decoder predict
$f_3$, whose LM-head output supplies $t_4$.
The complete figure is extracted from the published PDF, not redrawn.*

After sampling that token, its embedding and predicted feature feed the
next cheap step. The target has not run again, so feature errors can
accumulate through the draft.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speculative-review-2026-10/eagle-feature-tree.png]]

*Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang,
[EAGLE: original official-repository diagram](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/figs/fig1.png),
[Apache-2.0](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/LICENSE).
Left: target processing “How can.” Center: the autoregressive head receives
features and shifted embeddings. Right: alternative branches after three
cheap passes. Unmodified EAGLE-1 diagram, not EAGLE-3.*

Training in [§3.2](https://arxiv.org/html/2401.15077v3#S3.SS2) has two losses.
Smooth L1 matches predicted and target features. Cross-entropy matches
distributions produced by the same frozen LM head from true and predicted
features. The latter is a soft teacher target, not merely a text token ID.

The original experiment weights classification loss by $0.1$.
The projection and autoregressive block train; embeddings and LM head come
from the target. Small feature noise exposes the draft to imperfect inputs
during training rather than only at generation time.

<a id="eagle-2-расширение-и-повторный-отбор-дерева"></a>

### EAGLE-2: tree expansion and reranking

[EAGLE-2, §4](https://aclanthology.org/2024.emnlp-main.422/) retains the draft
model and training but changes tree construction. It expands selected
nodes at each depth, then reranks nodes across the whole tree before
verification.

Draft probability serves as an approximate acceptance estimate, an empirical
observation rather than an identity. Node value includes its entire path:

$$
V(v)\approx\prod_{u\in\operatorname{path}(v)}c(u),
$$

$c$ is the draft's conditional token probability. A confident child of an
unlikely parent can still have low path value.

Let root children A and B have probabilities $0.6$ and $0.4$.
A has C, D, E with $0.5,0.3,0.2$; B has F, G with $0.9,0.1$:

| Node | Path value $V$ |
|---|---:|
| A | 0.60 |
| B | 0.40 |
| C | 0.60 × 0.50 = 0.30 |
| D | 0.60 × 0.30 = 0.18 |
| E | 0.60 × 0.20 = 0.12 |
| F | 0.40 × 0.90 = 0.36 |
| G | 0.40 × 0.10 = 0.04 |

At depth two, an expansion budget of two selects F and C.
A final verification budget of four proposal nodes selects A, B, F, C;
the root is separate. Every selected node retains its parent.

<!-- checked-example: speculative-rerank -->
```python
value = {"A": .6, "B": .4, "C": .6*.5, "D": .6*.3,
         "E": .6*.2, "F": .4*.9, "G": .4*.1}
level = ["C", "D", "E", "F", "G"]
assert sorted(level, key=lambda node: -value[node])[:2] == ["F", "C"]
selected = sorted(value, key=lambda node: -value[node])[:4]
assert selected == ["A", "B", "F", "C"]
parent = {"A": None, "B": None, "C": "A", "D": "A",
          "E": "A", "F": "B", "G": "B"}
assert all(parent[node] is None or parent[node] in selected for node in selected)
```

A child's value cannot exceed its parent's because each factor lies in
$[0,1]$. Break equal-value ties in favor of shallower nodes.
Selecting the largest values then preserves connectivity.
The published argument is reproduced from
[EMNLP 2024, §4.2](https://aclanthology.org/2024.emnlp-main.422/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

<!-- source-excerpt: speculative-eagle2-closure:start -->
> The value of a node is always less than or equal to that of its parent node. For nodes with the same value, we prioritize selecting shallower nodes. This ensures that the top m tokens selected after reranking still form a connected tree.
<!-- source-excerpt: speculative-eagle2-closure:end -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/speculative-review-2026-10/eagle2-tree-mask.png]]

*Li et al., [EAGLE-2, Figure 7](https://aclanthology.org/2024.emnlp-main.422.pdf#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Top: expansion. Middle: reranking. Bottom: packed nodes and attention mask.
The authors' root is “It”; this is separate from the A–G calculation.
The complete figure is extracted from page five; the caption is in text
and the diagram itself is unchanged.*

<a id="proposal-verification-и-rollback-дерева"></a>

### Tree proposal, verification and rollback

Packing a tree into one array does not turn it into a sequence.
For root R, children A and B, A's children C and D, and B's child E,
D can see R and A but not B, C or E.

The square mask below concerns tree nodes. Already processed confirmed
prefix positions are separately visible to every node.

| Query \ key | R | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| R | 1 | 0 | 0 | 0 | 0 | 0 |
| A | 1 | 1 | 0 | 0 | 0 | 0 |
| B | 1 | 0 | 1 | 0 | 0 | 0 |
| C | 1 | 1 | 0 | 1 | 0 | 0 |
| D | 1 | 1 | 0 | 0 | 1 | 0 |
| E | 1 | 0 | 1 | 0 | 0 | 1 |

If R occupies position $n$, position IDs are
$n,n+1,n+1,n+2,n+2,n+2$. Packed array position does not replace depth.
An ordinary lower-triangular mask would let B read A and change its
conditioning prefix.

<!-- checked-example: speculative-tree -->
```python
parents = [-1, 0, 0, 1, 1, 2]
depth = np.zeros(len(parents), dtype=int)
tree_mask = np.zeros((len(parents), len(parents)), dtype=bool)
for node, par in enumerate(parents):
    if par >= 0:
        depth[node] = depth[par] + 1
    ancestor = node
    while ancestor >= 0:
        tree_mask[node, ancestor] = True
        ancestor = parents[ancestor]

np.testing.assert_array_equal(depth, [0, 1, 1, 2, 2, 2])
np.testing.assert_array_equal(tree_mask[4], [1, 1, 0, 0, 1, 0])
np.testing.assert_array_equal(tree_mask[5], [1, 0, 1, 0, 0, 1])
assert np.tril(np.ones((6, 6), dtype=bool))[2, 1] and not tree_mask[2, 1]

# Greedy-проверка: из корня выбран B, затем E, затем отсутствующий ребёнок F.
children = {"R": {"A", "B"}, "A": {"C", "D"}, "B": {"E"}, "E": set()}
target_next = {"R": "B", "B": "E", "E": "F"}
node, emitted = "R", []
while True:
    chosen = target_next[node]
    emitted.append(chosen)
    if chosen not in children[node]:
        break
    node = chosen
assert emitted == ["B", "E", "F"]
```

Greedy tree verification selects the target's next argmax.
If that child exists, accept it and continue; otherwise emit the token
and end the cycle. Keep only KV on the selected path, with the last emitted
token potentially still pending. The example discards A, C and D.

Sampling requires proposal probabilities, not the longest path of matches.
[EAGLE, Appendix A.2](https://arxiv.org/html/2401.15077v3#A1.SS2) describes
multiple child-verification rounds at one node. After rejecting the first
candidate, a second can be tested instead of immediately drawing a
replacement, but it must be tested against the updated residual.

For $q=(0.2,0.5,0.3)$ and $p_1=(0.6,0.3,0.1)$, a rejection leaves
$r_1=(0,0.5,0.5)$. If the next candidate is actually drawn from
$p_2=(0,0.75,0.25)$ and is B, acceptance probability is
$0.5/0.75=2/3$. After a second rejection, all residual mass is on C.

The second ratio uses $r_1/p_2$, not original $q/p_1$.
$p_2$ must describe the real second-proposal mechanism, including sampling
without replacement if used. Deterministic top-k is not a random draw from
the original $p$. Without these conditions, the chain formula does not
prove correctness for an arbitrary tree.

<a id="eagle-3-объединение-слоёв-и-обучение-на-собственных-состояниях"></a>

### EAGLE-3: combining layers and training on self-generated states

[EAGLE-3, §3](https://arxiv.org/html/2503.01840v3#S3) removes the requirement
to reconstruct a particular upper-layer feature. It trains token prediction
without a separate feature-regression loss.

Features come from lower, middle and upper parts of the target.
Three width-$d$ vectors are concatenated to $3d$ and projected to $d$.
The resulting $g$ is combined with the selected next-token embedding.
A projection and small decoder produce an internal state $a$, from which
the LM head obtains token probabilities.

In the paper's example, the target processed “How can” and selected “I.”
The first cheap step uses $g_{\mathrm{how}},g_{\mathrm{can}}$ and shifted
embeddings to produce $a_{\mathrm I}$ and token “do.”
For the second step, true $g_{\mathrm I}$ is unavailable: the target has
not processed “I.” The draft substitutes its own $a_{\mathrm I}$.
Then $a_{\mathrm{do}}$ substitutes for unavailable $g_{\mathrm{do}}$,
and drafting continues.

| Version | Features | Training target | Tree |
|---|---|---|---|
| EAGLE-1 | One upper-level feature stream and shifted embeddings | Feature regression plus soft cross-entropy | Fixed structure |
| EAGLE-2 | Same | Same | Context-dependent expansion and reranking |
| EAGLE-3 | Fused multilevel features, then self-generated states | Token prediction without separate feature regression | EAGLE-2's dynamic tree |

**Training-time test** feeds self-generated $a$ states back during training.
Training only on true target $g$ would expose later inference steps to a
different input-state distribution. Unrolled training steps expose the
draft to that regime in advance.

These steps need their own attention mask. For training text “How can I,”
predictions “are,” “we,” “do” continue three different prefixes:
“How,” “How can,” “How can I.” They are not the sequence “are we do.”
At the next step, “are” sees “How → are,” “we” sees “How can → we,”
and “do” sees “How can I → do.” Attention between new branches is
therefore diagonal; access to original text follows the relevant prefix.
Position labels come from the training data.

Original [Figure 5, inference, and Figure 6, training masks](https://proceedings.neurips.cc/paper_files/paper/2025/file/c7b5a35ea98b62512a869c19ea7b03cb-Paper-Conference.pdf#page=5)
show both on one published page. Symbols $f,g,a$ are not interchangeable:
EAGLE-1 ties its predicted vector to a teacher feature; EAGLE-3 has no
such regression constraint.

<a id="пример-расчёта-выигрыша-дерева"></a>

<a id="worked-trade-off"></a>

### Calculating a tree's latency trade-off

Suppose an ordinary target step takes 10 ms, a twelve-node draft tree
takes 2.5 ms, and verification takes 13 ms. With mean progress 4.2 tokens
including the extra token,

$$
\mathrm{TPOT}\approx\frac{2.5+13}{4.2}
=3.69\text{ ms/token}.
$$

That is about 2.71× relative to 10 ms/token. If another domain lowers
progress to 1.4 tokens at the same cost, the result is 11.07 ms/token,
slower than baseline. These are teaching values, not an EAGLE run.

With larger batches, the target can reuse weights effectively without
drafting. Single-request speedups and large-batch throughput describe
different conditions. EAGLE-3 separately reports 1.38× throughput at
batch 64 in SGLang v0.4.4, using one H100, LLaMA 3.1 8B and MT-bench.
That experiment used a length-three chain, not a tree. It is an author
measurement, not a universal multiplier.

<a id="когда-ускорения-не-будет"></a>

## When speculation does not help

If ordinary decoding already uses compute units well, additional verification
positions consume scarce resources. A large continuous batch can reduce
speculation's advantage, depending on kernels and scheduling.

An expensive draft, poor domain acceptance and a large tree increase the
time until the next token emission. Long contexts add KV reads and
attention work. Distributed verification changes collective message sizes.
Speeding up one linear layer does not determine the whole cycle.

Extra weights and temporary KV reduce capacity for other requests.
One request can finish sooner while fewer total requests meet the SLO.
Short answers, EOS and output limits further reduce useful proposals.

<a id="как-измерять"></a>

## Measuring speculative decoding

First fix the target decoding policy: model, precision, temperature,
top-p/top-k, token constraints and stopping rules. Record draft version,
chain length or tree budget, kernel modes and scheduler settings.

Separate drafting, verification, token selection, KV reconciliation and
other overhead. State whether progress includes the extra token:
accepted proposals $A$ and emitted tokens $N=A+1$ differ.
An average over all proposed tree nodes is also different from the
probability of passing the next conditional check.

Measure TTFT, TPOT, inter-token gaps, memory, throughput and goodput under
the same requests and load. Emitting several tokens per cycle can be
bursty; average TPOT does not describe every user-visible pause.

Small discrete examples allow exact enumeration and residual checks.
Statistical tests on real generations help detect bugs, but similar
XSum or HumanEval scores do not prove equal distributions.
For greedy decoding, first check tokens under fixed logits, then separately
investigate numerical differences in real execution.

See [[58b Benchmarking, SLO и эксплуатация inference|benchmarking and SLOs]]
for matched-load comparisons.

<a id="практика-и-первоисточники"></a>

## Practice and primary sources

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/lecture.pdf|Efficient DL Systems week 9: inference algorithms]] and [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/homework/homework.ipynb|speculative-decoding assignment]].
- [[02 Areas/ML & DL/06 Практика/16 Проверить speculative decoding и rollback KV|Practice: acceptance, correction and KV rollback]].
- [[02 Areas/ML & DL/Papers/Speculative Decoding|Original paper source note]].
- [Leviathan, Kalman, Matias: Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192v2): §2, §3.1–3.3, Appendix A.1.
- [Chen et al.: Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318v1): Conditional Scoring, Modified Rejection Sampling, Table 1, Figure 1 and Theorem 1.
- [Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L507-L543): mechanism and two-token proof.
- [Google Research: Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/): authors' visualization.
- [EAGLE, §3 and Appendix A.2](https://arxiv.org/html/2401.15077v3#S3), [EAGLE-2, §3–4](https://aclanthology.org/2024.emnlp-main.422/), [EAGLE-3, §3](https://arxiv.org/html/2503.01840v3#S3): feature drafting and tree variants.
- [Universal Assisted Generation](https://huggingface.co/blog/universal_assisted_generation): tokenizer synchronization.
- Zhao et al., [FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](https://aclanthology.org/2025.acl-long.198/): reducing the draft LM head's vocabulary; [authors' implementation](https://github.com/thunlp/FR-Spec).
- Hu et al., [GRIFFIN: Effective Token Alignment for Faster Speculative Decoding](https://arxiv.org/abs/2502.11018v3): token and feature alignment when training the drafter; [authors' implementation](https://github.com/hsj576/GRIFFIN).
- Ringel and Romano, [Accelerating Speculative Decoding with Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989v1): DDTree builds a tree from block-diffusion draft distributions.
- Iso et al., [Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding](https://arxiv.org/abs/2604.26779v1): draft verification within RL rollout generation.

Previous: [[57b Сжатие моделей — pruning, distillation и low-rank|Model compression]].

Next: [[58a Распределённый inference и disaggregated serving|Distributed inference]].
