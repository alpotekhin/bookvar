---
title: Masked diffusion — MDLM, LLaDA and block generation
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - anchor: masking-process
    label: "MDLM: поглощающее маскирование и обучающий вход"
    label_en: "MDLM: absorbing masking and the training input"
  - anchor: weighted-mask-loss
    label: "SUBS и взвешенная ошибка masked diffusion"
    label_en: "SUBS and the weighted masked-diffusion loss"
  - anchor: diffusion-bert-boundary
    label: "BERT и генеративная модель с маскированием"
    label_en: "BERT versus a generative masking model"
  - anchor: reverse-transition
    label: "Обратный переход: вероятность токена и вероятность раскрытия"
    label_en: "Reverse transitions: token probability and reveal probability"
  - anchor: complete-toy-sampler
    label: "Полный sampler с точными условными распределениями"
    label_en: "A complete sampler with exact conditional distributions"
  - anchor: diffusion-pair-factorization
    label: "Ошибка независимой генерации зависимых токенов"
    label_en: "Independent sampling of dependent tokens"
  - anchor: llada-training-and-selection
    label: "LLaDA: pre-training, SFT, EOS и выбор позиций"
    label_en: "LLaDA: pre-training, SFT, EOS and position selection"
  - anchor: block-diffusion-and-kv
    label: "Block Diffusion — блочная генерация и KV-cache"
    label_en: "Block Diffusion — block generation and KV caching"
  - anchor: diffusion-stream-leakage
    label: "Block Diffusion: чистый и зашумлённый потоки без утечки targets"
    label_en: "Block Diffusion: clean/noisy streams without target leakage"
  - anchor: remasking-and-refinement
    label: "ReMDM — повторное маскирование"
    label_en: "ReMDM — remasking and refinement"
  - anchor: remdm-loop
    label: "ReMDM-loop: доработка при постоянном уровне шума"
    label_en: "ReMDM-loop: refinement at a fixed noise level"
  - anchor: likelihood-and-serving
    label: "NELBO, perplexity и стоимость диффузионной генерации"
    label_en: "NELBO, perplexity and diffusion-serving cost"
  - anchor: diffusion-prediction-reuse
    label: "Повторное использование прогноза и prefix KV-cache"
    label_en: "Prediction reuse versus prefix KV caching"
primary_sources:
  - https://arxiv.org/html/2406.07524v2
  - https://arxiv.org/html/2502.09992v3
  - https://arxiv.org/html/2503.09573v3
  - https://arxiv.org/html/2503.00307v4
  - https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/
---

# Masked diffusion — MDLM, LLaDA and block generation

An autoregressive model fills the positions of `The cat sleeps here` in
order. A masked diffusion model can fill position 3 first and then use
`sleeps` to help predict the missing tokens before and after it. It starts
with four output positions, all masked.

| Reverse-process state | Sequence (positions 1–4) |
|---|---|
| Initial | `[MASK]` `[MASK]` `[MASK]` `[MASK]` |
| One possible step | `[MASK]` `[MASK]` `sleeps` `[MASK]` |
| Later | `The` `[MASK]` `sleeps` `here` |
| Complete | `The` `cat` `sleeps` `here` |

This toy trajectory treats each word as one token. A reverse step can
reveal several positions or none. The basic MDLM sampler keeps a token
once revealed; ReMDM later relaxes that constraint.

The predictor uses [[00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|bidirectional self-attention]],
as in [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT]].
Unlike the [[00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|shifted targets of a next-token model]],
the target at a masked position is the clean token at that same position.

<a id="masking-process"></a>
<span id="corrupting-a-sequence-creates-the-training-input"></span>
## The masked training input

![[00 Учебник/Assets/Figures/curated/masked-diffusion-2026-10-29/mdlm-overview.png|Original MDLM training diagram and historical LM1B comparison]]

*Subham Sekhar Sahoo et al., [MDLM v2, Figure 1](https://arxiv.org/html/2406.07524v2#S1.F1),
CC BY 4.0. Complete original PNG, without redrawing or cropping. Read the
left side from a clean sequence through different masking rates to token
predictions. The right-hand benchmark is the paper's historical comparison,
not a current ranking; the diffusion likelihood estimates are bounds.*

Let $x=(x_1,\ldots,x_L)$ be a clean training sequence and let $m$ denote
the mask token. Define $\alpha_t$ as the probability that a token remains
visible at noise level $t$. In the linear schedule, $\alpha_t=1-t$:

$$
q(z_t^i\mid x_i)=
\alpha_t\,\delta_{x_i}+(1-\alpha_t)\,\delta_m.
$$

Here $\delta_a$ assigns probability one to token $a$. At $t=0$ the sequence
is clean; at $t=1$ it is entirely masked. At $t=0.75$, each position is
independently masked with probability 0.75. In a four-token sequence the
expected number of masks is three, but a particular sample can contain
zero, one, two, three or four masks.

To construct a complete forward trajectory, a masked position must remain
masked as noise increases. For $s<t$, a still-visible token is removed
with probability $(\alpha_s-\alpha_t)/\alpha_s$. This is an absorbing
state process. Training usually does not execute every intermediate
transition: it samples $t$ and creates $z_t$ directly from the marginal
above. The original clean sequence remains available as the target.

A bidirectional Transformer reads $z_t$ and predicts a distribution over
clean vocabulary tokens at each position. All predictions use the same
input state; a prediction for one missing position is not already an
observation for another position in that forward pass.

Some denoisers also receive the noise level $t$ explicitly; it is omitted
from the probability notation here. MDLM evaluates both time-conditioned
and time-independent variants. LLaDA uses its bidirectional predictor
without a separate time input. These choices matter when reusing a
prediction at another noise level.

For `The [MASK] sleeps [MASK]`, position 2 can use `sleeps` to its right,
and position 4 can use it to its left. The model distinguishes the two
positions through positional information even though they share the
same mask-token embedding.

<a id="weighted-mask-loss"></a>
<span id="training-averages-over-masking-rates"></span>
## The weighted denoising loss

MDLM uses a substitution-based parameterization (SUBS). The predicted
clean token cannot itself be `[MASK]`, so its mask-token logit is set to
$-\infty$. At visible positions the clean-token distribution copies the
input token. At an unmasked position the copy has probability one, so its negative
log-probability is zero.

For the linear schedule, the continuous-time negative evidence lower
bound, normalized by sequence length, becomes

$$
\begin{aligned}
\widehat\ell(x,z_t,t)&=-\frac{1}{Lt}
\sum_{i:z_t^i=m}\log p_\theta(x_i\mid z_t),\\
\ell(x)&=\mathbb E_{t,z_t}[\widehat\ell(x,z_t,t)].
\end{aligned}
$$

Here $t\sim U(0,1)$ and $z_t\sim q(\cdot\mid x)$; the first line is the
loss for one sampled noise level and corruption, and the second averages
over those draws.

The unnormalized quantity $L\ell(x)$ bounds the negative log-likelihood
under the endpoint and SUBS assumptions of MDLM. For a general smooth
schedule, the positive cross-entropy weight is
$-\alpha'_t/(1-\alpha_t)$ rather than $1/t$.
The source derivation is [MDLM §§3.2–3.5](https://arxiv.org/html/2406.07524v2#S3).

At $t=0.25$, one possible draw again gives `The [MASK] sleeps [MASK]`.
Suppose the network assigns probabilities 0.8 and 0.5 to the targets
`cat` and `here`. Only positions 2 and 4 contribute to the sampled
per-token loss:

$$
\frac{-\log0.8-\log0.5}{4\cdot0.25}
\approx0.916291.
$$

Dividing by the *observed* number of masks instead would give 0.458145.
With this Bernoulli corruption, that is a different estimator: $Lt=1$
is the expected count, while this draw contains two masks. A draw with
no masks contributes zero. Forcing it to have a mask would change the
corruption distribution. Later we will see a likelihood estimator that
deliberately samples the mask count first.

<details id="mdlm-weighted-code">
<summary>Code: loss for a sampled mask</summary>

```python
import math

def masked_loss(correct_probabilities, mask_probability, length):
    if not 0 < mask_probability <= 1 or length < 1:
        raise ValueError("Invalid probability or length")
    if len(correct_probabilities) > length:
        raise ValueError("Too many selected positions")
    if any(not 0 < p <= 1 for p in correct_probabilities):
        raise ValueError("Expected positive target probabilities")
    return -sum(map(math.log, correct_probabilities)) / (length * mask_probability)

loss = masked_loss([0.8, 0.5], 0.25, 4)
assert math.isclose(loss, 0.916290731874155)
assert masked_loss([], 0.25, 4) == 0
assert math.isclose(loss / 2, -math.log(0.4) / 2)
```

</details>

This code checks the reduction and weighting of supplied probabilities;
it does not train a Transformer. A training implementation obtains those
probabilities from logits and differentiates the corresponding weighted
cross-entropy. Near $t=0$, the factor $1/t$ and rare masking events can
produce high-variance estimates. Practical schedules, lower cutoffs and
low-discrepancy sampling need to be specified when reproducing training.
Block Diffusion §§4–5 studies this variance explicitly.

<a id="diffusion-bert-boundary"></a>

### BERT and masked diffusion

The BERT objective also predicts tokens at their original positions.
Its corruption recipe selects a fixed fraction of positions and can
replace a selected token with a mask or a random token, or leave it
unchanged. The original objective does not specify how to go from an
all-mask string to a sample from a sequence distribution. MDLM specifies
that chain and trains against its likelihood bound.

<a id="reverse-transition"></a>
<span id="a-reverse-step-decides-both-what-to-predict-and-what-to-reveal"></span>
## The reverse transition

Assume first that the clean token $x_i$ is known. If $z_t^i$ is visible,
then $z_s^i=z_t^i$: reversing an absorbing masking process cannot change
its visible value. If $z_t^i=m$, Bayes' rule gives

$$
\begin{aligned}
q(z_s^i=x_i\mid z_t^i=m,x_i)&=r,\\
r&=\frac{\alpha_s-\alpha_t}{1-\alpha_t},\quad s<t.
\end{aligned}
$$

To obtain this ratio, divide the probability that a token is visible at
$s$ but masked at $t$, namely $\alpha_s-\alpha_t$, by the probability
that it is masked at $t$, namely $1-\alpha_t$. With the linear schedule,
the reveal probability is $(t-s)/t$.

At generation time $x_i$ is unknown. Replace its point mass by the model's
predicted clean-token distribution. Write $\pi_i(v)=p_\theta(x_i=v\mid z_t)$
for the predicted probability of clean token $v$ at position $i$.
For a currently masked position,

$$
p_\theta(z_s^i=v\mid z_t)=
\begin{cases}
 r\,\pi_i(v),&v\ne m,\\
 1-r,&v=m,
\end{cases}
.
$$

Take the same visible pattern, `The [MASK] sleeps [MASK]`, during
generation. For $t=0.75$, $s=0.25$, the reveal probability is $2/3$.
If the predictor assigns probability 0.6 to `cat` and 0.4 to `dog`
at position 2, that position becomes `cat` with probability 0.4,
`dog` with probability $4/15$, or stays masked with probability $1/3$.
Position 4 stays masked with probability $1/3$ regardless of its
token predictions. The MDLM step samples the two positions independently,
so the probability of `The cat sleeps [MASK]` after this step is
$0.4\cdot(1/3)=2/15$. On the next pass the predictor can use the newly
visible `cat` as context.

<details id="mdlm-reverse-code">
<summary>Code: reverse distribution at one position</summary>

```python
MASK = "[MASK]"

def reverse_token(current, clean_probabilities, alpha_t, alpha_s):
    if not 0 <= alpha_t < 1 or not alpha_t <= alpha_s <= 1:
        raise ValueError("Invalid reverse schedule")
    if MASK in clean_probabilities or any(p < 0 for p in clean_probabilities.values()):
        raise ValueError("Clean distribution cannot contain a mask or negative mass")
    if not math.isclose(sum(clean_probabilities.values()), 1):
        raise ValueError("Clean probabilities must sum to one")
    if current != MASK:
        return {current: 1.0}
    reveal = (alpha_s - alpha_t) / (1 - alpha_t)
    return {**{v: reveal * p for v, p in clean_probabilities.items()}, MASK: 1 - reveal}

transition = reverse_token(MASK, {"cat": 0.6, "dog": 0.4}, 0.25, 0.75)
assert math.isclose(transition["cat"], 0.4)
assert math.isclose(transition["dog"], 4 / 15)
assert math.isclose(transition[MASK], 1 / 3)
assert reverse_token("cat", {"dog": 1.0}, 0.25, 0.75) == {"cat": 1.0}
```

</details>

The last assertion expresses carry-over: even a new prediction preferring
`dog` cannot overwrite a committed `cat` in this sampler. At the final
step $s=0$, $r=1$ and all remaining masks are filled. Predicted probability
and reveal probability have distinct roles throughout the trajectory.

<a id="complete-toy-sampler"></a>
<span id="a-complete-sampler-on-a-two-token-distribution"></span>
## Simultaneous reveals on a two-token dataset

Consider a dataset containing only `red fox` and `blue whale`, each with
probability one half. The clean-token predictor below computes the exact
conditional marginal distributions from that dataset. It substitutes for
a learned neural network so that sampling errors can be separated from
errors in learning.

With both positions masked, each predicts two alternatives with probability
one half. After `red` is visible, the second position predicts `fox` with
probability one. The two positions are dependent even though each
individual marginal is easy to compute.

<details id="mdlm-toy-code">
<summary>Code: the two-token sampler</summary>

```python
import random

DATA = {("red", "fox"): 0.5, ("blue", "whale"): 0.5}

def toy_denoiser(state):
    compatible = {x: p for x, p in DATA.items()
                  if all(z == MASK or z == v for z, v in zip(state, x))}
    if not compatible:
        raise ValueError("The observed context has zero probability in this toy dataset")
    total = sum(compatible.values())
    result = []
    for position in range(len(state)):
        probabilities = {}
        for x, probability in compatible.items():
            token = x[position]
            probabilities[token] = probabilities.get(token, 0) + probability / total
        result.append(probabilities)
    return result

def draw(probabilities, rng):
    threshold = rng.random()
    cumulative = 0.0
    for token, probability in probabilities.items():
        cumulative += probability
        if threshold < cumulative:
            return token
    raise AssertionError("Invalid categorical distribution")

def sample_pair(steps, seed):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    rng = random.Random(seed)
    state = [MASK, MASK]
    history = [state.copy()]
    for remaining in range(steps, 0, -1):
        if MASK not in state:
            break
        predictions = toy_denoiser(state)
        alpha_t = 1 - remaining / steps
        alpha_s = 1 - (remaining - 1) / steps
        # Both transitions are computed from the same old state.
        state = [draw(reverse_token(z, p, alpha_t, alpha_s), rng)
                 for z, p in zip(state, predictions)]
        history.append(state.copy())
    assert MASK not in state
    return tuple(state), history

assert toy_denoiser(["red", MASK])[1] == {"fox": 1.0}
for steps in (1, 2, 4, 16):
    for seed in range(20):
        result, history = sample_pair(steps, seed)
        for before, after in zip(history, history[1:]):
            assert all(a == MASK or a == b for a, b in zip(before, after))
```

</details>

The sampler starts from pure masks, recomputes the conditional predictions
after its state changes, and samples each position from the reverse
transition. It performs no model loading or training. The carry-over check
holds even for an invalid final pair such as `red whale`: a committed
mistake also remains committed.

An invalid pair can appear when both positions are revealed in one step.
Independent draws from their marginals produce four equally probable
pairs. `red whale` and `blue fox` are outside the dataset. More steps reduce the chance of simultaneous revelation and let
one sampled token condition the other.

<a id="diffusion-pair-factorization"></a>

For a linear schedule with $T$ equally spaced reverse steps, each position
has a uniform reveal-step index in $\{1,\ldots,T\}$. When there are
$k$ steps left, the reveal probability for a still-masked position is
$1/k$. Thus each of the $T$ steps gets the same unconditional probability:
for example, the second step has probability $(1-1/T)/(T-1)=1/T$ when
$T>1$. The independent indices coincide with probability $1/T$. When that happens, the two
independent token draws disagree with probability $1/2$. Otherwise the
second reveal sees the first and is correct. Therefore, in this particular
toy model,

$$
P(\text{invalid pair})=\frac{1}{2T}.
$$

This gives 0.5, 0.25 and 0.125 at one, two and four steps. It is an exact
calculation for the specified dataset and sampler, not an error rate for
LLaDA or a general scaling law. A Transformer can model dependencies
through its shared input, but independent token draws in a finite reverse
step do not become a joint draw merely because attention is bidirectional.

<a id="llada-training-and-selection"></a>
<span id="llada-a-bidirectional-predictor-trained-at-language-model-scale"></span>
## LLaDA training and decoding

LLaDA pre-trains a bidirectional Transformer by masking tokens
independently at a sampled rate. For instruction fine-tuning, the prompt
stays visible and only the response can be masked. In
[§2.3, Eq. (5)](https://arxiv.org/html/2502.09992v3#S2.SS3) the loss sums
the negative log-probabilities of the masked response tokens and weights
the sum by the inverse masking rate. The accompanying guidelines show a
per-answer reduction that also divides by the padded response length.

For example, suppose a prompt is followed by the padded response
`red fox [EOS] [EOS]`. If the draw gives `red [MASK] [EOS] [MASK]`,
the targets are `fox` at position 2 and `[EOS]` at position 4. The
guidelines count all four response positions in the length. The prompt
provides context, and the visible EOS is already known. Neither adds a
loss term on this draw. In this recipe the EOS padding is part of the
training target, unlike padding excluded from a next-token loss.

In the pinned [LLaDA guidelines](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/GUIDELINES.md),
masking probability is bounded below by a small epsilon. The guidelines
illustrate the loss and data format; they are not a complete released
pre-training pipeline. Their snippets also contain inconsistent variable
names, so they should not be presented as a program that runs unchanged.

The [reference generator](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py)
divides a fixed output length into blocks and allocates a number of token
commitments to each step. In `low_confidence` mode, candidate tokens are
ranked by their predicted probabilities; the highest-confidence candidates
are committed, while other proposals remain masks. `random` uses random
scores instead. Suppose the current block has two masks, the proposed tokens have
probabilities 0.6 and 0.5, and the step can commit one position. The
confidence rule keeps the 0.6 proposal and leaves the other mask in
place. The MDLM ancestral step makes separate random reveal decisions:
it can reveal zero, one or both positions.

The name `remasking` in this implementation does not mean that it can
erase any previously committed token. The code restricts selection to
currently masked positions and copies visible values. Proposing a value
and then leaving that position masked differs from remasking a value
already included in the persistent sequence. ReMDM makes the latter
operation explicit.

LLaDA's original paper studies an 8B model trained from scratch. Its
results are evidence for that model and protocol, not proof that replacing
every autoregressive objective with masked diffusion improves quality.
See [LLaDA §2 and Appendix B.4](https://arxiv.org/html/2502.09992v3#S2).

<a id="block-diffusion-and-kv"></a>
<span id="block-diffusion-changes-the-factorization-and-the-attention-mask"></span>
## Block diffusion and block-causal attention

![[00 Учебник/Assets/Figures/curated/masked-diffusion-2026-10-29/block-diffusion-overview.svg|Original Block Diffusion sequence of completed context and an active noisy block]]

*Marianne Arriola et al., [Block Diffusion v3, Figure 1](https://arxiv.org/html/2503.09573v3#S1),
CC BY 4.0. Complete original SVG. Earlier blocks are clean context; the
active block is generated by diffusion before the next block begins.*

The figure's parallelism label refers to positions within a block;
successive blocks still depend on earlier ones. Its quality labels
summarize the paper's comparisons, not guarantees for every checkpoint.

A fixed mask array fixes the number of output positions. Block diffusion
instead factorizes a sequence into successive blocks:

$$
p_\theta(x)=\prod_{b=1}^{B}p_\theta(x^{(b)}\mid x^{(<b)}).
$$

Each conditional is a diffusion process within the current block. Blocks
are generated in order, while positions inside a block can be revealed
together or in arbitrary order. A block size of one recovers a tokenwise
factorization; a single full-sequence block gives the opposite endpoint.
Sampling efficiency and training variance still depend on the recipe.

In BD3-LMs, attention is block-causal. A token can see all positions in
its own block and earlier blocks, but not later blocks. With four tokens
in blocks of size two, the permitted attention rows are

```text
1 1 0 0
1 1 0 0
1 1 1 1
1 1 1 1
```

The completed first block cannot see the changing second block.
With fixed positions and noise conditioning for that clean context, its
hidden states, keys and values remain valid while the second block is
denoised. The active block's own representations change
when its masks are filled and normally require recomputation.

Merely generating blocks with a *fully bidirectional* Transformer does
not establish this invariance. At deeper layers, even an unchanged
prompt token has attended to changing answer positions. For example,
with uniform attention and scalar values 2 for a prefix token and 0 for
an active token, their mean is 1. Replacing the active value by 6 changes
the mean to 4. A subsequent projection of the prefix state changes too.
Scheduling the updates in blocks does not make these prefix states
invariant. Exact prefix caching requires the block-causal mask or another
mechanism that prevents the dependency.

<a id="diffusion-stream-leakage"></a>

### Clean and noisy streams during training

Take the clean blocks `The cat` and `sleeps here`, and suppose their
noisy copies are `The [MASK]` and `sleeps [MASK]`. To predict `here`,
the second noisy block can read its own noisy copy and the clean first
block. It cannot read the clean second block, where the answer is still
present, or the noisy first block. The clean stream attends only to the
clean stream with a block-causal mask. Earlier clean states cannot
relay `here` to the predictor because they cannot attend forward.

The vectorized algorithm concatenates the noisy and clean streams in
one forward pass. It processes two copies of each position, and the
mask makes each noisy block see only its own noisy inputs and the clean
prefix it will have at generation time. See
[Block Diffusion §3.2](https://arxiv.org/html/2503.09573v3#S3.SS2) and
[Appendix B.6](https://arxiv.org/html/2503.09573v3#A2.SS6).

<a id="remasking-and-refinement"></a>
<span id="remdm-permits-a-committed-token-to-become-uncertain-again"></span>
## ReMDM and remasking

![[00 Учебник/Assets/Figures/curated/masked-diffusion-2026-10-29/remasking-overview.svg|Original ReMDM example correcting She sell to She sells]]

*Guanghan Wang et al., [ReMDM v4, Figure 1](https://arxiv.org/html/2503.00307v4#S1.F1),
CC BY 4.0. The left-hand trajectory runs upward from $t=1$ to $t=0$:
`sell` becomes a mask and is then replaced by `sells`. The right-hand
MAUVE values belong to the original OpenWebText experiment, not to our toy sampler.*

ReMDM removes a visible token with probability $\sigma_t$. To keep the
next marginal visible probability at $\alpha_s$, it must reveal more
of the masked positions. Under the known-clean-data posterior in [ReMDM §3.1](https://arxiv.org/html/2503.00307v4#S3.SS1),
a visible token survives with probability $1-\sigma_t$, while a masked
position is revealed with probability

$$
r_\sigma=\frac{\alpha_s-(1-\sigma_t)\alpha_t}{1-\alpha_t}.
$$

For $\alpha_t=0.5$, $\alpha_s=0.75$, and $\sigma_t=0.25$, one quarter
of the visible positions are removed. Masked positions are filled with
probability 0.75. The next expected visible fraction is
$0.5\cdot0.75+0.5\cdot0.75=0.75$. The MDLM reveal probability here was
0.5. Keeping it while adding remasking would leave an expected visible
fraction of only $0.5\cdot0.75+0.5\cdot0.5=0.625$.

The probabilities are valid only when
$0\le\sigma_t\le\min(1,(1-\alpha_s)/\alpha_t)$ for $\alpha_t>0$.
At $\alpha_t=0$, there are no visible positions to remove. Previously
visible tokens must not be remasked at the final clean endpoint.

<details id="mdlm-remask-code">
<summary>Code: remasking and reveal probabilities</summary>

```python
def remask_rates(alpha_t, alpha_s, sigma):
    if not 0 <= alpha_t < 1 or not alpha_t <= alpha_s <= 1:
        raise ValueError("Invalid schedule")
    maximum = min(1.0, (1 - alpha_s) / alpha_t) if alpha_t else 1.0
    if not 0 <= sigma <= maximum:
        raise ValueError("Invalid remasking probability")
    reveal = (alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t)
    return sigma, reveal

remove, reveal = remask_rates(0.5, 0.75, 0.25)
assert reveal == 0.75
assert 0.5 * (1 - remove) + 0.5 * reveal == 0.75
assert remask_rates(0.5, 0.75, 0) == (0, 0.5)
assert remask_rates(0.5, 1, 0) == (0, 1)
```

</details>

These are posterior schedule calculations, not a complete ReMDM sampler.
During generation the network supplies the clean-token probabilities.
A later step can replace a remasked mistake using its new context, but
it can also replace a correct token.

![[00 Учебник/Assets/Figures/curated/masked-diffusion-2026-10-29/remasking-loop.svg|Original ReMDM-loop noise and remasking schedules]]

*Wang et al., [ReMDM v4, Figure 2 and §4.2](https://arxiv.org/html/2503.00307v4#S4),
CC BY 4.0. Generation runs from right to left as $t$ decreases. On the
flat part of $\alpha$, remasking spends additional steps revising a
partially completed sequence rather than increasing its visible fraction.*

<a id="remdm-loop"></a>

In ReMDM-loop, ordinary unmasking first creates a partial candidate.
During an intermediate phase, $\alpha_s=\alpha_t$ is held constant and
remasking is enabled. At $\alpha=0.5$, $\sigma=0.25$ gives a masked-to-visible
probability of 0.25: the expected removed and restored masses balance.
The final phase disables remasking and fills the remaining positions.
This separates spending computation on refinement from progressing
toward a fully visible sequence. The paper also studies capped, rescaled
and confidence-dependent remasking schedules; they are distinct choices,
not consequences of the name “diffusion”.

<a id="likelihood-and-serving"></a>
<span id="likelihood-latency-and-what-to-compare"></span>
## Likelihood estimates and generation cost

For an autoregressive model, summing the log-probabilities of the
observed tokens given their prefixes gives the sequence log-likelihood.
A masked diffusion model defines the sequence probability by summing over
latent trajectories. The expected NELBO bounds the negative log-likelihood
from above; exponentiating its per-token value bounds perplexity from above.
A Monte Carlo estimate of that bound still has sampling error.

<a id="llada-fixed-mask-likelihood"></a>
### Likelihood estimation with a fixed number of masks

For conditional likelihood evaluation, [LLaDA §2.4, Eq. (6)](https://arxiv.org/html/2502.09992v3#S2.SS4)
estimates the same bound as Eq. (5) by sampling the mask count instead of
the noise level. Its predictor needs only the masked response and prompt.
For a response of length $L$,
first draw the mask count $\ell$ uniformly from 1 to $L$. Then choose
exactly $\ell$ positions uniformly without replacement and multiply
their summed loss by $L/\ell$. Dividing by $L$ for a per-token result
leaves $1/\ell$ times the sum.

For the same toy response, take `The [MASK] sleeps [MASK]` with
$\ell=2$ and reuse the illustrative target probabilities 0.8 and 0.5. The
per-token estimate is $(-\log 0.8-\log 0.5)/2$, about 0.458145. That
division is part of the fixed-count scheme. The training example drew
a masking probability of 0.25 and then happened to get two masks; its
denominator was $L t=1$, giving 0.916291. The two draws look the same,
but their mask counts were sampled differently. Neither draw by itself
gives a sequence likelihood.

<details id="mdlm-estimator-weights">
<summary>Why averaging over the masking rate gives the same weight</summary>

Fix a particular mask set of size $\ell>0$. Under Bernoulli corruption
its probability at rate $t$ is $t^\ell(1-t)^{L-\ell}$. Multiplying by the
per-token weight $1/(Lt)$ and integrating $t$ from 0 to 1 gives
$1/(L\ell\binom{L}{\ell})$. If the predictor does not take $t$ as an input,
the loss for this mask pattern is the same throughout that integral.
The fixed-count sampler gives the pattern probability
$1/(L\binom{L}{\ell})$ and uses weight $1/\ell$. The resulting coefficient
is identical for each nonempty mask set.

</details>

### Denoiser calls and cache reuse

An autoregressive decode step can process one new token against a cached
prefix. A fully bidirectional denoising step may process the entire
sequence again. To compare them, count network evaluations and tokens
per evaluation, then measure latency on the same workload. Classifier-free
guidance evaluates an unconditional input too, even if the implementation
batches it with the conditional input.

<a id="diffusion-prediction-reuse"></a>

In [MDLM §4.1](https://arxiv.org/html/2406.07524v2#S4.SS1), if a step
leaves every mask in place, the next step receives the same sequence. A
time-independent denoiser can reuse its predicted distributions; only the
reverse-transition probabilities need updating. A newly revealed token
or a different time embedding invalidates that prediction. A prefix KV
cache, by contrast, can stay valid while the active block changes.

For an output of $L$ tokens split into $B$ blocks, with $K$ denoising passes
per block, the nominal denoiser count is $BK$, not merely $K$.
Cache-building passes and stopping checks may add work. This count is
still not FLOPs or elapsed time: a full block pass can reuse weights
across many positions and use the GPU differently from tokenwise decode.
See [[00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|prefill and decode]]
and [[00 Учебник/10 ML Systems/03 Измерение производительности и roofline|performance measurement]]
for the corresponding hardware model.

Masked reconstruction gives the predictor part of the reference response.
To test instruction-following, start with a masked response and score
the generated answers. Report the prompts, tokenizer, length budget,
and sampling policy so the results can be compared.

<span id="sources-and-adjacent-chapters"></span>
## References

- Sahoo et al., [Simple and Effective Masked Diffusion Language Models, v2](https://arxiv.org/html/2406.07524v2): §§3.2–3.5, §4, Appendix B.3. Forward process, SUBS, weighted loss and ancestral sampling; CC BY 4.0.
- Nie et al., [Large Language Diffusion Models, v3](https://arxiv.org/html/2502.09992v3): §2 and Appendices B.4–B.7. LLaDA training, SFT, sampling and evaluation. The model-weight license is not used as a license for the paper's figures.
- Arriola et al., [Block Diffusion, v3](https://arxiv.org/html/2503.09573v3): §§2–3, §§4–5 and Appendix B.6. Block factorization, clean/noisy training streams, caching and gradient variance; CC BY 4.0.
- Wang et al., [Remasking Discrete Diffusion Models, v4](https://arxiv.org/html/2503.00307v4): §§3–4.2. Valid remasking transitions, marginal preservation and loop schedules; CC BY 4.0.
- Kuleshov, Arriola, Schiff and Wang, [How to Build a Diffusion Language Model](https://kuleshov-group.github.io/blog/blog/2026/how-to-build-a-diffusion-language-model/): the lecture-based overview connecting these papers. Its broader claims about later releases are not assumed to hold for every masked sampler.

Related: [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT]],
[[00 Учебник/11 Pre-training и Scaling/41h Masked autoencoders — MAE и восстановление патчей|MAE]],
[[00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|diffusion and flow models for images]].

Previous: [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Scaling laws]].
Next: [[00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Decoding and selecting the next token]].
