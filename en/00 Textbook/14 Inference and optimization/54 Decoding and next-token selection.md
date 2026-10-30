---
title: "Decoding and next-token selection"
type: textbook-chapter
status: canonical
last_updated: 2026-10-30
last_verified: 2026-10-25
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|Next-token objective]]"
next: "[[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей и контаминация]]"
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/
  - https://huggingface.co/blog/how-to-generate
  - https://web.stanford.edu/~jurafsky/slp3/7.pdf
  - https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation
  - https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/logits_process.py
  - https://arxiv.org/abs/1904.09751v2
locale: en
translation_status: current
canonical_revision: "c10fe3a763d44cf1f0e8d2e4103bcba7adae026fc7cf910e673515aaa51ab486"
translation_reviewed: "2026-10-30"
translation_of: "00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена.md"
---

<a id="декодирование-и-выбор-следующего-токена"></a>

# Decoding and next-token selection

A language model does not emit a finished sentence. After each prefix it
returns one number for every token in its vocabulary. Decoding turns those
numbers into decisions: which token to append, when to stop, and which
alternatives to retain for the next step. The same model parameters can
therefore produce a translation, a repetitive loop, or several plausible
continuations depending on the selection procedure.

<a id="от-логитов-к-распределению"></a>

## From logits to a distribution

For prefix $x_{<t}$, the model computes logits $z_1,\ldots,z_{|V|}$.
Softmax defines the next-token distribution:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

A logit may be negative or greater than one. For example, $[2,1,0]$
becomes approximately $[0.665,0.245,0.090]$. Adding the same constant
to every logit does not change the distribution: the common exponential
factor cancels between numerator and denominator. Computation usually
subtracts the maximum logit $m$ first:

$$
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
\qquad m=\max_j z_j.
$$

The exponentials then do not exceed one. The logits $[1002,1001,1000]$
give the same distribution without requiring the exponential of 1002.
Library softmax implementations use such numerically stable operations.

The subsequent procedure includes both transforming this distribution
and selecting a continuation.

1. A **distribution transformation** changes eligible tokens or their
   relative probabilities: temperature, top-k, top-p, repetition penalties
   and token bans.
2. A **search strategy** selects one or more paths: greedy decoding,
   random sampling or beam search.

`top_p=0.9` is not a complete search method: it truncates the
distribution, after which a token must still be selected.
`temperature=0.7` introduces no randomness if selection still uses
`argmax`.

The selected token becomes part of the prefix. The next model call
computes a different distribution, conditional on that choice. The model
does not receive its entire future answer in advance or automatically
revise earlier choices.

If transformations replace $p$ with $q$, sampling follows $q$. The
probability of the resulting string **under the original model** still
uses $p$. These differ: top-k makes some strings impossible for the
generation procedure even though the model assigns them nonzero
probability. This distinction is needed in
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|speculative decoding]],
which must preserve the chosen target distribution.

<a id="greedy-локально-лучший-токен"></a>

## Greedy decoding: the locally best token

Greedy decoding selects

$$
x_t=\arg\max_{w\in V}p(w\mid x_{<t}).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-greedy_search.png]]

*The greedy path in Patrick von Platen's
[How to generate text](https://huggingface.co/blog/how-to-generate).
Taking the highest-probability branch at each level gives
$0.5\cdot0.4=0.2$.*

A local maximum need not belong to the most probable complete sequence.
Choosing `nice` with probability 0.5 looks better than `dog` with 0.4,
but at the next step `dog has` has probability $0.4\cdot0.9=0.36$
and overtakes `nice woman`.

The figure compares continuations of exactly two words. A **completed**
answer also includes the termination event: the probability of
`nice woman EOS` multiplies three conditional probabilities, including
$p(\mathrm{EOS}\mid\text{The nice woman})$. Cutting generation off after
two words imposes a length limit; it is not a model-selected EOS.

Greedy decoding is useful when one short reproducible result is needed
and the distribution is concentrated: classification through generation,
some extraction formats, or a smoke test. For long open-ended text it
often enters repetitive high-probability patterns.

<a id="beam-search-несколько-гипотез-одновременно"></a>

## Beam search: several hypotheses at once

Beam search retains the $B$ highest-scoring unfinished sequences.
Each step expands them with candidate tokens, computes cumulative
scores, and retains the best $B$ continuations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-beam_search.png]]

*A beam width of two preserves both initial branches and discovers that
the less probable first token leads to the more probable pair.
Patrick von Platen, [How to generate text](https://huggingface.co/blog/how-to-generate).*

For a sequence $y_{1:T}$, the natural score is the sum of log probabilities:

$$
s(y_{1:T})=\sum_{t=1}^{T}\log p(y_t\mid y_{<t},x).
$$

With width two, the first step retains `nice` and `dog`.
The second step produces six candidates:

| Continuation after The | Two-step probability |
|---|---:|
| dog has | $0.4\cdot0.9=0.36$ |
| nice woman | $0.5\cdot0.4=0.20$ |
| nice house | $0.5\cdot0.3=0.15$ |
| nice guy | $0.5\cdot0.3=0.15$ |
| dog and | $0.4\cdot0.05=0.02$ |
| dog runs | $0.4\cdot0.05=0.02$ |

The first two rows survive. Selection uses the combined pool of
continuations from all hypotheses. If both best continuations began
with `dog`, both slots would go to them; `nice` has no reserved
slot. Summing logarithms produces the same ordering without multiplying
many small numbers.

<a id="завершённые-гипотезы-и-нормировка-длины"></a>

### Finished hypotheses and length normalization

A hypothesis ending in EOS is no longer expanded. An implementation may
store it separately while continuing to expand unfinished paths. Finding
the first EOS does not require returning that answer: another path may
finish later with a higher final score.

Every log probability is non-positive, so longer sequences accumulate
lower scores. Translation and speech recognition therefore use length
normalization, for example

$$
s_\alpha(y)=\frac{s(y)}{|y|^\alpha}.
$$

Suppose one completed answer has probability 0.3 and length two, and
another has probability 0.2 and length three. Without normalization the
first wins: $\log0.3\approx-1.204$ versus $\log0.2\approx-1.609$.
At $\alpha=1$, the scores become $-0.602$ and $-0.536$, reversing
their order. The normalized score is no longer the log probability of
the string. A positive coefficient weakens the preference for short
answers; it neither specifies a desired length nor measures content
quality.

Token counting follows the implementation. In Transformers 4.57.1,
ordinary beam search uses generated-continuation length, excluding the
prompt and counting EOS when the sequence finishes. See
[_update_finished_beams](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/utils.py#L3056-L3110)
and the corresponding calculation in
[BeamHypotheses](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/beam_search.py#L943-L966).

Early stopping also depends on the score. With positive length
normalization, a path's current average score can rise as non-positive
log probabilities are appended, because the denominator increases.
Comparing only current scores therefore does not give a strict bound.
In that version, `early_stopping=True` stops after enough finished
candidates have been collected; `False` uses a heuristic; and
`"never"` uses an upper bound on the attainable score that accounts
for maximum length. Even the last mode cannot recover branches pruned
earlier.

Beam search is suited to tasks whose answers are closely constrained
by the input and have a limited set of good realizations: translation,
speech recognition and sometimes summarization. In open dialogue,
model probability does not coincide with human judgments of interest.
A wider beam may amplify generic phrases and repetition.

Beam search remains approximate. Once pruned, a branch cannot return.
It does not guarantee the global maximum when its width is smaller
than the set of all possible prefixes.

Even increasing the width does not guarantee a better final result.
Consider a three-step teaching tree. Initially $p(A)=0.5$,
$p(B)=0.4$, $p(C)=0.1$. After A there are five equiprobable
continuations; after B, two with probability 0.5 each. At step two,
a width-two beam keeps both B-prefixes with probability 0.2 and
discards all A-prefixes with probability 0.1. Suppose each B-prefix
then has ten equiprobable continuations, whereas the A-prefix chosen
by greedy decoding has a continuation with probability 0.99.
The beam result has probability 0.02; the greedy result has 0.099.
These are fixed-length paths without EOS. The cause is prefix pruning,
not length normalization.

<a id="случайная-выборка"></a>

## Random sampling

Multinomial sampling draws a token from the distribution:

$$
x_t\sim p(\cdot\mid x_{<t}).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-sampling_search.png]]

*A branch is selected randomly according to its displayed probability;
another run may take a different path. Patrick von Platen,
[How to generate text](https://huggingface.co/blog/how-to-generate).*

Unrestricted sampling preserves the model distribution, but a vocabulary
has a long tail of weakly plausible tokens. Each may be unlikely on
one step, while the chance of at least one poor choice increases over
a long sequence. Distribution transformations control this tail.

A rare token is not necessarily wrong: a surname, technical term or
code symbol can also have low probability. Truncation changes the balance
between these candidates and undesirable continuations without checking
their meaning. In a hypothetical task with a 0.01 probability of an
undesirable choice on each of 100 independent steps, the probability
of at least one such choice is $1-0.99^{100}\approx0.634$.
This illustrates risk accumulation, not the measured error rate of an LLM.

<a id="temperature-меняет-контраст-вероятностей"></a>

## Temperature changes probability contrast

Temperature divides logits by $T>0$:

$$
p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

At $T<1$, logit differences become stronger and the distribution sharpens.
At $T>1$, it flattens. The probability ratio shows why:

$$
\frac{p_i(T)}{p_j(T)}
=\exp\!\left(\frac{z_i-z_j}{T}\right).
$$

For a logit difference of one, the ratio is $e$ at $T=1$,
$e^2$ at $T=0.5$, and $\sqrt e$ at $T=2$. A positive temperature
preserves logit order and hence their `argmax`; it changes sampling
probabilities.

With a unique maximum, $T\to0$ concentrates mass on it. If several
tokens share the maximum, they retain equal probabilities in the limit;
this does not specify greedy tie-breaking. As $T\to\infty$, the
distribution becomes uniform over allowed tokens with finite logits.
Tokens with logit $-\infty$ remain forbidden.

Zero cannot be substituted into the formula. In Transformers, turn off
sampling explicitly with `do_sample=False`. Other servers may treat
`temperature=0` as a separate switch; that is an API convention,
not a softmax value.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-sampling_search_with_temp.png]]

*Lowering temperature makes the car branch less probable.
Patrick von Platen, [How to generate text](https://huggingface.co/blog/how-to-generate).*

The figure's probabilities 0.75, 0.23 and 0.02 illustrate concentration,
but they are not an exact transformation of 0.5, 0.4 and 0.1 using
one temperature. The nice/dog ratio would require $T\approx0.189$,
whereas nice/car would require $T\approx0.444$. The next example
calculates the distribution exactly for stated temperatures.

Temperature does not identify factually incorrect tokens. It changes
the contrast of an existing ranking. A low value may reinforce an
incorrect maximum; a high value may increase an undesirable tail.

<a id="расчёт-на-одном-распределении"></a>

### Calculate one distribution

Let tokens A–E have probabilities $[0.4,0.3,0.2,0.08,0.02]$.
Choosing logits $z_i=\log p_i$ makes softmax recover those probabilities.
Changing temperature then amounts to raising probabilities to $1/T$
and renormalizing.

| Token | $T=1$ | $T=0.5$ | $T=2$ |
|---|---:|---:|---:|
| A | 0.4000 | 0.5391 | 0.3083 |
| B | 0.3000 | 0.3032 | 0.2670 |
| C | 0.2000 | 0.1348 | 0.2180 |
| D | 0.0800 | 0.0216 | 0.1379 |
| E | 0.0200 | 0.0013 | 0.0689 |

Run the following Python blocks sequentially in one session.
First compute this table:

~~~python
import math
import torch

torch.set_num_threads(1)
probabilities = torch.tensor(
    [[0.4, 0.3, 0.2, 0.08, 0.02]], dtype=torch.float64
)
logits = probabilities.log()
history = torch.tensor([[0]])

def probabilities_at(temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return torch.softmax(logits / temperature, dim=-1)

for temperature in (0.5, 1.0, 2.0):
    print(temperature, probabilities_at(temperature).tolist())
~~~


<a id="top-k-фиксированное-число-кандидатов"></a>

## Top-k: a fixed candidate count

Top-k retains the $k$ most probable tokens, zeros the rest and
renormalizes the retained mass:

$$
\tilde p_i=
\begin{cases}
p_i / \sum_{j\in K}p_j,&i\in K,\\
0,&i\notin K.
\end{cases}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-top_k_sampling.png]]

*At $k=6$, the illustration retains 0.68 of the original mass on the
left and 0.99 on the right. The retained bars are blue. The two
distributions show how excluded mass changes at a fixed k.
Patrick von Platen, [How to generate text](https://huggingface.co/blog/how-to-generate).*

Fixed $k$ has different costs in two limiting contexts. If the model
is almost certain about one token, the other $k-1$ candidates add
unnecessary noise. If the distribution is broad, the same $k$ can
discard substantial plausible mass.

For A–E at $k=2$, A and B remain with total mass 0.7.
After normalization their probabilities are $4/7$ and $3/7$,
not $1/2$ each. Candidate count is fixed, but selection remains weighted.

The implementation's boundary rule matters. Transformers'
`TopKLogitsWarper` removes logits **smaller** than the kth-largest.
Ties can therefore retain more than k tokens. With logits $[2,1,1,0]$
and $k=2$, three survive. In `generate`, `top_k=0` disables
top-k. The warper constructor itself accepts a positive k, so a disabled
filter is omitted rather than constructed.

<a id="top-p-переменный-набор-заданной-массы"></a>

## Top-p: a variable set with a mass threshold

Nucleus sampling sorts tokens and chooses the smallest set $V^{(p)}$
whose mass reaches the threshold:

$$
\sum_{w\in V^{(p)}}p(w\mid x_{<t})\ge p.
$$

The retained probabilities are renormalized before sampling.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-top_p_sampling.png]]

*At threshold 0.92, nine candidates remain in the first context and
three in the second. The original mass of the retained bars may exceed
the threshold. Patrick von Platen and Hugging Face,
[How to generate text](https://huggingface.co/blog/how-to-generate).*

This adaptive nucleus motivates Holtzman et al.'s
[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751):
in the open-ended generation they studied, the tail was often unreliable,
while beam search favored repetitive high-probability continuations.

For A–E at threshold 0.8, cumulative masses are 0.4, 0.7 and 0.9.
A, B and C must survive. Stopping at A and B would not reach the threshold;
retaining only a fraction of C is not an option because selection is
between whole tokens. Renormalization gives $[4/9,3/9,2/9,0,0]$.

The threshold refers to mass **before** final normalization. It means
neither “an 80% chance of a correct answer” nor retaining 80% of the
vocabulary. On another prefix, 0.8 may retain one token or thousands.
Ties and boundary rounding need a specified rule. The examples below
use Transformers 4.57.1 with at least one retained token.

Holtzman et al.'s conclusions concern their GPT-2 open-generation setup
and selected metrics. They do not establish one best decoder for every
modern model, translation, program or verifiable task.

<a id="min-p-порог-относительно-лидера"></a>

## Min-p: a threshold relative to the leading token

Min-p retains token $i$ when its probability is at least fraction $m$
of the largest probability:

$$
p_i\ge m\,p_{\max}.
$$

For A–E at $m=0.15$, the threshold is $0.15\cdot0.4=0.06$.
A, B, C and D survive with mass 0.98. D's normalized probability
is $0.08/0.98\approx0.0816$. Neither candidate count nor total
retained mass is fixed.

At $m=0.1$ with a leading probability of 0.9, the threshold is 0.09.
Only the leader survives from $[0.9,0.04,0.03,0.02,0.01]$.
For the flatter $[0.24,0.22,0.20,0.18,0.16]$, the threshold is 0.024
and all five survive.

Temperature also changes this filter. After temperature scaling,
$p_i/p_{\max}=\exp((z_i-z_{\max})/T)$. For $0<m\le1$, the condition
is $z_i\ge z_{\max}+T\log m$. The same `min_p` at different
temperatures need not retain the same candidates. See
[MinPLogitsWarper](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/logits_process.py).

Check all three filters on the same logits:

~~~python
from transformers import (
    TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper,
)

filters = {
    "top-k=2": TopKLogitsWarper(top_k=2),
    "top-p=0.8": TopPLogitsWarper(top_p=0.8),
    "min-p=0.15": MinPLogitsWarper(min_p=0.15),
}
filtered_probabilities = {}
for name, transform in filters.items():
    scores = transform(history, logits.clone())
    result = scores.softmax(dim=-1)
    filtered_probabilities[name] = result
    print(name, result.tolist())
~~~


The filters write $-\infty$ into excluded logits, making their softmax
probabilities zero. If constraints forbid every token, there is no mass
to normalize. Define how to handle that error rather than choosing an
arbitrary token.

<a id="почему-порядок-преобразований-меняет-результат"></a>

## Transformation order changes the result

Apply `top_k=2` and `top_p=0.5` to A–E.
Top-k first retains A and B with probabilities $4/7$ and $3/7$.
Top-p now needs only A because $4/7>0.5$. Selection becomes deterministic.

Top-p first instead needs both A and B, since A's original mass is only
0.4. Subsequent top-k removes nothing, leaving $4/7$ and $3/7$.

~~~python
top_k = TopKLogitsWarper(top_k=2)
top_p = TopPLogitsWarper(top_p=0.5)

k_then_p = top_p(history, top_k(history, logits.clone())).softmax(-1)
p_then_k = top_k(history, top_p(history, logits.clone())).softmax(-1)
print(k_then_p.tolist())  # [[1.0, 0.0, 0.0, 0.0, 0.0]]
print(p_then_k.tolist())  # [[4/7, 3/7, 0.0, 0.0, 0.0]]
~~~


For sampling, Transformers 4.57.1 applies temperature → top-k → top-p
→ min-p when those filters are configured. Repetition penalties and
most hard constraints are added earlier. The order is specified in
[_get_logits_processor](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/utils.py).
A server with a different order or defaults can produce a different
distribution from parameters with the same names.

To sample the original untruncated distribution in this version, explicitly
set `do_sample=True, temperature=1.0, top_k=0, top_p=1.0` and enable
no other filters. `do_sample=True` alone is insufficient: the standard
`GenerationConfig` has `top_k=50`, and a model's own configuration
can specify other values.

<a id="ограничения-и-штрафы"></a>

## Constraints and penalties

<a id="repetition-penalty-и-знак-логита"></a>

### Repetition penalty and logit sign

Transformers' repetition penalty divides a previously seen token's
non-negative logit by $r$ and multiplies its negative logit by $r$:

$$
z_i'=
\begin{cases}
z_i/r,&z_i\ge0,\\
r z_i,&z_i<0.
\end{cases}
$$

At $r>1$, both operations lower the logit. For example, 2 becomes
$2/1.2\approx1.667$ and $-0.5$ becomes $-0.6$. Dividing a negative
logit by a factor greater than one would move it toward zero and
make that token more attractive instead.

This penalty is applied once per encountered token ID, not repeatedly
for every occurrence. For a decoder-only model, history includes the
prompt by default. A term from a document can therefore be penalized
even when the answer must repeat it. `RepetitionPenaltyLogitsProcessor`
can exclude an initial portion using `prompt_ignore_length`.

Because it treats signs differently, this penalty depends on the absolute
logit offset. Softmax agrees for $z$ and $z+c$, but the penalized
distributions may differ. This particular operation cannot be viewed
as a universal transformation of probabilities alone.

<a id="frequency-presence-и-запрет-n-грамм"></a>

### Frequency, presence and n-gram bans

Additive frequency and presence penalties are commonly written in terms
of occurrence count $c_i$:

$$
z_i'=z_i-\lambda_f c_i-\lambda_p\mathbf1[c_i>0].
$$

At $c_i=2$, $\lambda_f=0.3$ and $\lambda_p=0.2$, the deduction is 0.8.
The first and tenth repetitions incur different frequency penalties
but the same presence component. The API defines the counted history,
tokenization and transformation order. This is a different formula
from the multiplicative repetition penalty above.

No-repeat n-gram imposes a hard ban. With token history `[0,1,0]`
and n-gram size two, appending 1 would repeat `[0,1]`, so it is
forbidden. Token 1 is not forbidden in every context: the ban depends
on the current sequence's suffix.

~~~python
from transformers import (
    RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor,
)

penalty_history = torch.tensor([[1, 1, 3]])
penalty_logits = torch.tensor(
    [[-2.0, 2.0, 0.5, -0.5, 1.0]], dtype=torch.float64
)
penalized = RepetitionPenaltyLogitsProcessor(1.2)(
    penalty_history, penalty_logits.clone()
)
without_prompt = RepetitionPenaltyLogitsProcessor(
    1.2, prompt_ignore_length=2
)(penalty_history, penalty_logits.clone())
print(penalized.tolist())       # token 1: 2/1.2; token 3: -0.5*1.2
print(without_prompt.tolist())  # only token 3 is penalized

ngram_history = torch.tensor([[0, 1, 0]])
ngram_scores = NoRepeatNGramLogitsProcessor(2)(
    ngram_history, torch.zeros((1, 5))
)
assert torch.isneginf(ngram_scores[0, 1])
~~~


Penalties can disrupt a required repetition of a name, pronoun, code
line or table pattern. Token-level n-gram bans need not coincide with
word-level repetitions, especially for short token pieces.

<a id="ограничение-формата"></a>

### Constraining the format

For structured output, a finite-state machine or grammar can enforce
the syntax of a completed output when it reaches an accepting state.
A permitted prefix is not necessarily valid JSON: `max_new_tokens`,
cancellation or an external stop can end it before a closing bracket.
Syntax also says nothing about the truth of field values. Check normal
completion, truncation, abort and refusal separately. A consumer accepts
the result only after checking its finish reason and validating the
complete schema.

At each step the grammar specifies eligible token IDs $A(y_{<t})$.
Forbidden logits become $-\infty$; eligible ones are normalized.
Checking a token requires all its characters or bytes: one token may
contain several punctuation marks and part of the next field.

A local mask does not produce the global conditional distribution over
valid strings. Suppose an illustrative answer always has two characters
and only ab and cd are valid. Initially $p(a)=0.6$ and $p(c)=0.4$.
Then $p(b\mid a)=0.1$ and $p(d\mid c)=0.9$; the remaining mass goes
to forbidden continuations. The second-step mask forces b or d, giving
output probabilities 0.6 and 0.4. Under the original model those strings
have probabilities 0.06 and 0.36. Conditioning on validity of the
**entire** string would instead give $1/7$ and $6/7$.
These are different distributions even with correct format validation.

<a id="остановка-тоже-относится-к-декодированию"></a>

## Stopping is part of decoding

Generation ends at EOS, a length limit or an external criterion.
A string stop sequence may cross token boundaries, occur inside code
or already have been streamed to the user. Minimum/maximum length and
length penalties change the distribution of completed answers, so
record them alongside temperature and top-p.

EOS is a special token ID that the model can select.
`max_new_tokens` bounds newly generated tokens; `max_length`
bounds total length including the prompt. They are different parameters.
A chat model may use several special IDs to end a turn; obtain them
from its configuration and chat template.

String stops examine decoded text and need not be a single token.
[StopStringCriteria](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/stopping_criteria.py)
illustrates these cases for the marker `stop`:

| Received token pieces | What happens on the latest step |
|---|---|
| st, then op | The second token completes the marker |
| st, then opera | The marker ends inside the second token; the trailing era does not prevent detection |
| las, then topper | The marker spans the boundary, using s from the first token and top from the second |
| stop, then at | The latest step creates no new match; an ordinary loop would have stopped one step earlier |

Stopping and removing the marker are separate operations.
`StopStringCriteria` does not itself delete the marker from returned
IDs. Already streamed text cannot be recalled. To hide the marker,
a server needs a buffer for possible stop-string prefixes and a defined
rule for the trailing text. Checking only the latest token or each
client chunk separately is insufficient.

Test the four cases with the real `StopStringCriteria`. The small
vocabulary supplies an ID-to-string mapping; the token splits are
specified explicitly, not produced by encoding ordinary text.

~~~python
from tokenizers import Tokenizer, models, decoders
from transformers import PreTrainedTokenizerFast, StopStringCriteria

pieces = [
    "abcdef", "st", "op", "opera", "las", "topper", "at", "stop", "<unk>"
]
vocabulary = {piece: index for index, piece in enumerate(pieces)}
backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
backend.decoder = decoders.Fuse()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend, unk_token="<unk>"
)
criterion = StopStringCriteria(tokenizer, ["stop"])
stop_cases = [
    (["st", "op"], True),
    (["st", "opera"], True),
    (["las", "topper"], True),
    (["stop", "at"], False),
]
for tokens, expected in stop_cases:
    ids = torch.tensor([[vocabulary[token] for token in tokens]])
    assert criterion(ids, None).item() is expected
~~~


<a id="как-выбирать-стратегию"></a>

## Choosing a strategy

| Task | Starting point | What to test |
|---|---|---|
| Exact extraction / JSON | Greedy or low randomness plus a grammar | Schema validity and semantics |
| Translation / ASR | Beam search with length normalization | Quality versus beam width and latency |
| Open dialogue | Top-p or min-p with moderate temperature | Diversity, factuality and repetition |
| Multiple candidates | Independent samples | Distinct alternatives rather than paraphrases |
| Code | Sample candidates and execute tests | pass@k and verification cost |

Parameter values do not transfer mechanically between models. Logit
calibration, vocabulary and chat templates differ. Updating a model
behind the same API also changes the effective distribution.

<a id="воспроизводимый-эксперимент"></a>

## A reproducible experiment

Record model/tokenizer identifiers, the prompt after the chat template,
random seed, all `GenerationConfig` parameters, stop criteria and
library version. A single seed is insufficient to evaluate stochastic
generation: use independent runs and report distributions of metrics.

Compare strategies at an equal candidate budget. One greedy answer and
64 sampled answers followed by best-of selection use different amounts
of computation and form different systems, even if each displays one
final string.

<a id="проверка-настоящего-generate-без-загрузки-весов"></a>

### Test real generate without downloading weights

A tiny randomly initialized model is sufficient to test the API.
It has not learned language; its outputs are IDs in an artificial
vocabulary, not language-quality measurements. This example runs real
forward and `generate` calls in Transformers 4.57.1 on CPU.

~~~python
import copy
from transformers import LlamaConfig, LlamaForCausalLM, GenerationConfig

torch.manual_seed(17)
model = LlamaForCausalLM(LlamaConfig(
    vocab_size=8, hidden_size=16, intermediate_size=24,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    max_position_embeddings=32,
    bos_token_id=1, eos_token_id=None, pad_token_id=7,
)).eval()
prompt = torch.tensor([[1, 4, 5]])
attention_mask = torch.ones_like(prompt)
config = GenerationConfig(
    max_new_tokens=4, do_sample=False,
    bos_token_id=1, eos_token_id=None, pad_token_id=7,
)

with torch.inference_mode():
    first_logits = model(prompt).logits[:, -1, :]
    greedy = model.generate(
        prompt, attention_mask=attention_mask, generation_config=config
    )
assert greedy[0, prompt.shape[1]] == first_logits.argmax(-1).item()
assert greedy.shape[1] == prompt.shape[1] + 4

eos_config = copy.deepcopy(config)
eos_config.eos_token_id = first_logits.argmax(-1).item()
with torch.inference_mode():
    stopped = model.generate(
        prompt, attention_mask=attention_mask, generation_config=eos_config
    )
assert stopped.shape[1] == prompt.shape[1] + 1

beam_config = copy.deepcopy(config)
beam_config.num_beams = 2
beam_config.num_return_sequences = 2
beam_config.length_penalty = 0.7
with torch.inference_mode():
    beams = model.generate(
        prompt, attention_mask=attention_mask,
        generation_config=beam_config,
        return_dict_in_generate=True, output_scores=True,
    )
transitions = model.compute_transition_scores(
    beams.sequences, beams.scores, beams.beam_indices,
    normalize_logits=False,
)
reconstructed = transitions.sum(-1) / 4**0.7
torch.testing.assert_close(reconstructed, beams.sequences_scores)

sample_config = copy.deepcopy(config)
sample_config.do_sample = True
sample_config.top_k = 0
sample_config.top_p = 0.9
sample_config.temperature = 0.8
sample_config.num_return_sequences = 3
torch.manual_seed(123)
with torch.inference_mode():
    samples = model.generate(
        prompt, attention_mask=attention_mask,
        generation_config=sample_config,
        return_dict_in_generate=True, output_scores=True,
    )
assert samples.sequences.shape == (3, prompt.shape[1] + 4)
print("greedy", greedy.tolist())
print("EOS", stopped.tolist())
print("beam scores", beams.sequences_scores.tolist())
print("sampling", samples.sequences.tolist())
~~~


The first run disables EOS and stops at the new-token limit. The second
deliberately assigns EOS to the most probable next-token ID, testing
termination after one selection. That is a mechanism test, not a way
to configure a real model's EOS.

The third run returns two beam hypotheses. Each score is reconstructed
from its own transitions, using `beam_indices` because a hypothesis
can move between beam slots. All continuations have length four here;
the input prompt does not enter the normalization denominator.
The last run produces three independent samples for one prompt,
with beam search disabled.

A seed fixes the pseudorandom source, not bitwise agreement across
hardware, library versions or concurrent-request execution orders.
Language quality must be evaluated with a trained model, not this
random instance.

<a id="практика-и-первоисточники"></a>

## Practice and primary sources

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/lecture.pdf|Efficient DL Systems: inference acceleration lecture]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/seminar.ipynb|Efficient DL Systems: speculative sampling and Triton seminar]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/homework/homework.ipynb|Efficient DL Systems: decoding exercises]].
- Jurafsky & Martin, [Speech and Language Processing, chapter 7, §7.6](https://web.stanford.edu/~jurafsky/slp3/7.pdf#page=20), 19 August 2026 revision: greedy, sampling, temperature and truncation.
- Patrick von Platen, [How to generate text](https://huggingface.co/blog/how-to-generate): the original greedy, beam and sampling illustrations.
- Hugging Face, [GenerationConfig](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation) and [generation utilities](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils): version 4.57.1 used by these executable examples, not a designation of the latest release.
- Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751): nucleus sampling and degeneration analysis.

Previous: [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Test-time compute]].
Next: [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Model evaluation and contamination]].
