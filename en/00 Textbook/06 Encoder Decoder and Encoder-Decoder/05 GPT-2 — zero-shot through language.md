---
title: GPT-2 — tasks as text continuations
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "0ef00e2cac0e5944a6415ff5775ff247c5450892ad54ae8810fd8ba5777ee4ec"
translation_reviewed: "2026-11-21"
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md"
last_updated: 2026-11-21
primary_sources:
  - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - https://github.com/openai/gpt-2/tree/9b63575ef42771a015060c964af2c3da4cf7c8ab
  - https://jalammar.github.io/illustrated-gpt2/
  - https://aclanthology.org/Q19-1016/
  - https://aclanthology.org/P16-1144/
  - https://arxiv.org/html/2005.14165v4#S3.SS6
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning]]"
---

# GPT-2: tasks as text continuations

In the [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training#Textual entailment|GPT-1 chapter]], a causal Transformer reads the pair “A cat sleeps”
and “An animal rests.” A trained classifier takes the `Extract` state
and returns three scores. The released GPT-2 instead returns vocabulary
scores. To apply it to the same pair without fitting a new head on
labeled examples, write the label as a continuation.

<a id="gpt2-relation-prefix"></a>

<a id="from-a-classification-head-to-a-label-in-text"></a>

## Comparing text labels

Leave the label blank after the colon. This prompt has no `Extract`
token and no other pairs with completed answers.

~~~text
Premise: A cat sleeps.
Hypothesis: An animal rests.
Relation:
~~~

The [OpenAI encoder](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py)
produces 20 tokens. The last is the colon (ID 25) at position 19.
The model computes 50,257 logits from that state, one per vocabulary
entry. The labels ` neutral` and ` contradiction` each take one token
(8500 and 25741). The label ` entailment` takes two: ` entail` (39793)
and `ment` (434). The leading space is part of the first token.

Suppose the next-token probabilities after `Relation:` are $0.40$
for ` entail`, $0.25$ for ` neutral`, and $0.10$ for ` contradiction`.
The first-token scores favor ` entailment`. To score the whole label,
feed token 39793 at position 20 and read the probability of token 434.
If it is $0.50$, the probability after prompt $p$ is

$$
P(y=\texttt{entailment}\mid p)=0.40\cdot0.50=0.20.
$$

Comparing full labels now selects ` neutral`: $0.25>0.20>0.10$.
The score is the probability of starting the continuation with that label;
we do not require the answer to end there. The probabilities are illustrative, but the token
split comes from the released encoder. The [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-relation-shots|GPT-3 chapter]]
adds labeled examples to the same prompt without changing the scoring rule.

Pre-training still compares predicted next tokens with continuations
from the corpus:

$$
\mathcal L=-\sum_t\log P(x_t\mid x_{<t}).
$$

The authors' hypothesis is that varied web text already contains
translation pairs, questions with answers, and other task examples.
When a document contains a question followed by its answer, the model
learns to predict that answer as text. [Section 2 of the report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=2)
sets out this mechanism separately from the benchmark results:
next-token training does not guarantee that the model will recognize
an unfamiliar format.

<a id="what-changed-from-gpt-1"></a>

## GPT-2 blocks and model sizes

Small GPT-2 retains GPT-1's 12 blocks, 12 heads, and width 768.
The blocks still use causal attention and GELU; the learned token
table is shared between input and output. The context window grows
from 512 to 1024 tokens, and the vocabulary to 50,257. Even with the
same number of blocks, the computations change: LayerNorm now precedes
each sublayer, with a final LayerNorm after the whole stack.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-sizes-hyperparameters-3.png|Four GPT-2 stacks contain 12, 24, 36 and 48 blocks, with hidden widths 768, 1024, 1280 and 1600.]]

*Jay Alammar, [The Illustrated GPT-2, “Transformers for Language Modeling”](https://jalammar.github.io/illustrated-gpt2/).
GPT-2's four depths and widths.
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
original unchanged.*

| Model | Blocks | Heads | $d$ / FFN |
| --- | ---: | ---: | --- |
| 124M | 12 | 12 | 768 / 3072 |
| 355M | 24 | 16 | 1024 / 4096 |
| 774M | 36 | 20 | 1280 / 5120 |
| 1.5B | 48 | 25 | 1600 / 6400 |

The report uses 117M, 345M, 762M, and 1542M.
[OpenAI's README](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/README.md)
corrects the initial parameter counts. Released models are named 124M,
355M, 774M, and 1.5B. Layer counts and widths in the original figure
remain correct.

<details id="gpt2-param-count">
<summary>Exact parameter count</summary>

The implementation makes the count reproducible. With tied output
weights, token and position tables contain $Vd+Td$ parameters. One
block contains $12d^2+13d$: QKV and attention output projections, two
FFN projections, their biases, and two LayerNorms. Final LayerNorm
adds $2d$. Thus

$$
N=Vd+Td+L(12d^2+13d)+2d.
$$

For $V=50257$, $T=1024$, $d=768$, and $L=12$, this gives 124,439,808
parameters. A separate $[d,V]$ output matrix must not be added again:
it is the transposed token table. The other configurations give
354,823,168; 774,030,080; and 1,557,611,200 parameters.

</details>

<a id="gpt2-pre-norm"></a>

<a id="pre-norm-and-residual-paths"></a>

### LayerNorm before each sublayer

GPT-1 applied LayerNorm after residual addition. In GPT-2, each sublayer
first normalizes its input, computes an update, and adds it to the
original state:

$$
\begin{aligned}
\widehat H_{\ell-1}&=\operatorname{LN}_1(H_{\ell-1}),\\
A_\ell&=H_{\ell-1}+\operatorname{Attention}(\widehat H_{\ell-1}),\\
\widehat A_\ell&=\operatorname{LN}_2(A_\ell),\\
H_\ell&=A_\ell+\operatorname{FFN}(\widehat A_\ell).
\end{aligned}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-pre-post-norm.png|Xiong et al.: post-LN after addition on the left; pre-LN before attention and FFN on the right]]

*Ruibin Xiong et al., [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html),
[Figure 1](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Read both panels from bottom to top. In post-LN, LayerNorm follows
the addition. In pre-LN, the residual stream bypasses LayerNorm,
which instead precedes attention or the FFN. This is one block;
the final norm of the stack is not shown.*

Another LayerNorm follows the final block. Changing operation order
also changes the direct derivative path: `x + sublayer(LN(x))`
retains an identity-Jacobian term. This neither removes the sublayer's
derivative nor guarantees trouble-free training, but distinguishes
pre-norm from `LN(x + sublayer(x))`.

<details id="gpt2-original-block">
<summary>Excerpt from OpenAI's original block</summary>

The original `block` function shows the order directly:

```python title="Original implementation — Modified MIT"
def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present
```

*OpenAI, [`src/model.py`, commit `9b63575e`](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py).
Software Copyright (c) 2019 OpenAI;
[Modified MIT License](/licenses/openai-gpt2-Modified-MIT.txt).
The excerpt is unchanged.*

</details>

Report Section 2.3 also describes scaling residual-branch weight
initialization by $1/\sqrt{N_{\mathrm{res}}}$, where $N_{\mathrm{res}}$
counts such branches. With two sublayers per block, this is $2L$.
This initialization belongs to the report's training description.
Released `model.py` primarily defines the network for weight loading
and generation; its default initializers do not by themselves reproduce
the complete historical training recipe.

### One query before and after moving LayerNorm

Reuse the tokens `a cat sleeps` and the
[[en/00 Textbook/05 Attention and Transformer/02 Self-Attention — Q, K, V#4. A numerical example by hand|projection matrices from the Q, K and V example]].
Set positional additions to zero. With GPT-1's operation order,
the query at `cat` is projected from $x=(0,1,0)$ and equals $(0,1)$.
LayerNorm is applied later, after adding the attention output to $x$.

With GPT-2's order, LayerNorm maps $x$ to
$(-0.7071,1.4142,-0.7071)$ first, using unit scale, zero bias and
$\epsilon=10^{-5}$. The same projection now produces
$q_{\text{cat}}\approx(-1.4142,0.7071)$. The keys are also projected
from normalized rows. After scaling by $1/\sqrt{2}$, the scores for
`a` and `cat` are about $-1.4141$ and $1.7677$. The weight of `cat`
changes from $0.6698$ to $0.9601$. The mask still excludes `sleeps`;
the scores differ because the projections received different inputs.
This calculation uses the same toy weights, not the trained models.

## WebText and byte-level BPE

WebText contains pages reached by outbound Reddit links with at least
three karma, not Reddit comments themselves. About 45 million links
became just over eight million documents, approximately 40 GB of text,
after extraction, deduplication, and heuristic cleaning. The version
used excludes links created after December 2017
([report, Section 2.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).

Wikipedia was deliberately excluded because many tests are built from
it. This reduces one source of overlap but does not eliminate reprints
or other matches. Reddit votes helped select links; they did not verify claims on the pages.

### From bytes to tokens

GPT-1 preprocessing lost casing and some formatting. GPT-2 preserves
the string and starts with UTF-8 bytes. For example, `Привет` uses
twelve bytes despite having six letters. Each of the 256 byte values
maps reversibly to a printable character. BPE merges frequent sequences
of those characters.

Before merging, a regular expression separates letters, numbers,
punctuation, contractions, and whitespace. It restricts merge regions
without deleting string content. A leading space often shares a piece
with the following word. GPT-2 displays the space byte as `Ġ` in its
vocabulary; that is a byte's internal display, not an extra source letter.

The released encoder gives:

| Original string | IDs |
|---|---|
| `Hello world` | `[15496, 995]` |
| `Hello` | `[15496]` |
| ` hello` | `[23748]` |
| ` hello!` | `[23748, 0]` |
| `Привет` | `[140, 253, 21169, 18849, 38857, 16843, 20375]` |

Token 995 includes the space before `world`. Casing, spacing, and
punctuation affect tokenization; adding `strip()` or `lower()` is
not neutral. ID 0 means `!`, not padding. The vocabulary also contains
`<|endoftext|>` with ID 50256; unconditional generation uses it as
the starting context.

Every byte is supported, so normal encoding of valid Unicode requires
no unknown token. Decoding must first assemble bytes across tokens,
then read UTF-8: one token may contain only part of a multibyte letter.
An arbitrary generated ID sequence need not form valid UTF-8; default
`decode` replaces invalid sequences. Learning merges is examined in
the [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|tokenization chapter]].

<a id="how-one-loop-becomes-several-tasks"></a>

<a id="from-an-input-string-to-a-distribution"></a>

## Computing next-token logits

Return to `Relation:`. On the first pass, the input has shape
$[1,20]$ and its positions run from 0 to 19. At the colon, the
model adds row 25 of the token table `wte` to row 19 of the position
table `wpe`. Their sum has width 768 in GPT-2 Small.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt2-2026-10-22/gpt2-input-embedding-positional-encoding-3.png|The first token’s embedding row is added to a separate learned row for the first position.]]

*Jay Alammar, [The Illustrated GPT-2, “Input Encoding”](https://jalammar.github.io/illustrated-gpt2/).
Adding token and learned position rows.
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
unchanged.*

The figure uses `<s>` as an explanatory start token. OpenAI's script
uses `<|endoftext|>`, not the literal string `<s>`. Four cells in a
vector are also schematic: small GPT-2 has 768 coordinates. The figure
numbers positions from one; code indexes the first position as zero.

States pass through all blocks, each with its own weights. A causal
mask prevents row $t$ from receiving right-side information at every depth.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png|A vertical arrow follows the first position through masked self-attention and the FFN up to decoder block twelve.]]

*Jay Alammar, [The Illustrated GPT-2, “A journey up the Stack”](https://jalammar.github.io/illustrated-gpt2/).
One position's state across blocks.
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
unchanged.*

This diagram simplifies internals: LayerNorm and residual paths are
not shown. Their order is shown in the GPT-2 block above. Vertical arrows connect layers at one position;
self-attention inside each block also connects accessible positions.

After final LayerNorm, state $h_t\in\mathbb R^d$ multiplies transposed
`wte`:

$$
\begin{aligned}
z_t&=h_tW_{\mathrm{te}}^\top,\\
P(x_{t+1}\mid x_{\le t})&=\operatorname{softmax}(z_t).
\end{aligned}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt2-2026-10-22/gpt2-output-projection-2.png|The 768-wide output state is compared with each of 50,257 embedding rows, giving one logit per token.]]

*Jay Alammar, [The Illustrated GPT-2, “Model Output”](https://jalammar.github.io/illustrated-gpt2/).
Projection into 50,257 vocabulary scores.
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
unchanged.*

The original table has shape $[50257,768]$. Multiplying a state row
by its transpose gives $[1,768]@[768,50257]\to[1,50257]$.
The figure's vector orientation is schematic and final LayerNorm is
omitted. GPT-1's separate classification head is absent.

### Token selection, temperature, and top-k

Suppose four continuations have logits $(2,1,0,-1)$. At temperature
$\tau=1$, softmax gives approximately $(0.644,0.237,0.087,0.032)$.
Greedy selection always returns the first ID. Sampling may return
other IDs at frequencies specified by the distribution.

Temperature replaces logits with $z/\tau$ for $\tau>0$.
At $\tau=0.5$, differences double and probabilities become approximately
$(0.865,0.117,0.016,0.002)$. Lower temperature strengthens the model's
preference; it does not verify its most likely answer.

Top-k retains candidates with the largest logits and renormalizes.
At $k=2$, the example gives $(0.731,0.269,0,0)$, a different distribution
from the original full-vocabulary softmax.

<details id="gpt2-sampling-code">
<summary>Code: temperature and top-k</summary>

```python
import torch

def distribution(logits, temperature=1., k=0):
    assert temperature > 0
    assert 0 <= k <= logits.numel()
    scaled = logits / temperature
    if k:
        threshold = scaled.topk(k).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
    return scaled.softmax(-1)

z = torch.tensor([2., 1., 0., -1.], dtype=torch.float64)
p = distribution(z)
cold = distribution(z, temperature=.5)
top_two = distribution(z, k=2)
torch.testing.assert_close(
    top_two, torch.tensor([.7310585786, .2689414214, 0., 0.],
                         dtype=torch.float64))
assert p.argmax() == cold.argmax() == top_two.argmax() == 0
assert cold[0] > p[0]
print([round(v, 3) for v in p.tolist()])
print([round(v, 3) for v in cold.tolist()])
# [0.644, 0.237, 0.087, 0.032]
# [0.865, 0.117, 0.016, 0.002]
```

</details>

Here `k=0` disables truncation, as in OpenAI's `top_k_logits`.
The original threshold-based implementation retains all values equal
to the threshold: with logits $(2,1,1,0)$ and $k=2$, three candidates
remain. This differs from forcibly retaining exactly two IDs.

A chosen token is appended and becomes the condition for the next
step. During training, continuation is known from the corpus and many
positions' losses can be computed together. During generation, the
particular next continuation is unknown until its ID is selected.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-autoregression-2.gif|Each frame appends the next item on the right. The following prediction uses the appended text.]]

*Jay Alammar, [The Illustrated GPT-2, “One Difference From BERT”](https://jalammar.github.io/illustrated-gpt2/).
Appending generated tokens to a prefix.
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
original animation.*

The animation explains the loop using words and a schematic `$`
separator. It specifies neither exact BPE boundaries nor a reproduced
experiment demonstrating GPT-2's ability to follow that instruction.

<a id="retaining-keys-and-values"></a>

### The KV cache when continuing a prefix

Causal masking permits reuse of prefix computation. Appending a token
does not add accessible right neighbors to previous positions.
Their keys and values at every layer stay unchanged. A new position
computes its own Q/K/V and reads retained K/V.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt2-2026-10-22/gpt2-self-attention-qkv-3-2.png|The layers retain keys and values for a, then compute new queries, keys and values for robot.]]

*Jay Alammar, [The Illustrated GPT-2, “Evaluation Time: Processing One Token at a Time”](https://jalammar.github.io/illustrated-gpt2/).
Each block retains K/V for `a` on the left; the new column computes
Q/K/V for `robot`.
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
unchanged.*

The first pass over the `Relation:` prompt caches 20 positions.
To score the second piece of ` entailment`, process ` entail` at
position 20. Its query reads the 20 old K/V pairs and its own.
Scoring a one-token label does not require that extra pass.

For $L$ layers, $H$ heads, and head width $d_h=d/H$, a cache may have
shape $[B,L,2,H,T,d_h]$. The 2 distinguishes K from V.
Old Q is unnecessary: each new query compares with all available keys.

Small GPT-2 with one length-1024 input has

$$
\begin{aligned}
N_{\mathrm{KV}}&=2LTd\\
&=2\cdot12\cdot1024\cdot768\\
&=18\,874\,368.
\end{aligned}
$$

Storing each element in FP32 would require 72 MiB; FP16 would require
36 MiB, excluding weights and temporary tensors. Doubling sequence
length doubles this cache size; saving computation costs memory.

For a three-token prefix, the first pass uses positions $(0,1,2)$.
Its final distribution predicts the token at position 3. Once selected,
that token is processed as one ID with positional index 3 and four
available K/V positions, not with a reset positional index 0.

If the cache contains $P$ tokens and a new chunk has length $Q$, its
row $i$ may read keys $j\le P+i$. For example:

```text
P = 3, Q = 2, key columns = 0 1 2 3 4
new row 0, position 3:       1 1 1 1 0
new row 1, position 4:       1 1 1 1 1
```

The rectangular `[2,5]` mask is bottom-right aligned. A square
`tril(2)` does not represent the three old positions. This matches
the released `attention_mask` function. `sample.py` first processes
the complete prefix, then new IDs.

Here `past` is returned with new K/V already appended. Original
OpenAI `model.py` has a different contract: `present` contains
only the current chunk, and `sample.py` concatenates it with old
`past`. Mixing these contracts duplicates the prefix.

For an initial prefix at positions 0–2, original `sample.py`
accumulates memory as follows:

| Input | Old `past` | New `present` | Total |
|---|---:|---:|---:|
| 0, 1, 2 | 0 | 3 | 3 |
| 3 | 3 | 1 | 4 |
| 4 | 4 | 1 | 5 |

On the first call, `present` becomes the initial cache. Later,
`sample.py` concatenates only the new part along the token axis.
Teaching `TinyGPT2` does this inside `forward`, so its caller
replaces the old cache with the returned one instead of concatenating
again. Concatenating the old three positions with a returned full
cache of four would produce seven instead of four. The initial
prefix would be duplicated, and the next input would receive wrong
position index 7 instead of 4. See
[`sample_sequence`, original `body`](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/sample.py#L56-L68).

The cache contains states of processed inputs. The distribution after
the prefix selects position-3's token, but that token's K/V have not
yet been computed: the cache still has three positions. The next call
reads the chosen ID, grows the cache to four, and selects position 4.
If the loop stops immediately after selecting $G\ge1$ new tokens for
a length-$P$ prefix, it retains $P+G-1$ positions. The last selected
ID is in the returned text but enters the cache only on another
forward pass. Text length and cache length then differ by one.

Full and chunked passes use the same position indices and allowed
keys, so their logits agree within numerical precision. Caching saves
repeated computation but consumes memory and does not expand the
trained 1024-position window. Beyond the position table, indices
cannot silently be reset to zero. The teaching code does not implement
right padding, dropout, variable-length batched generation, or loading
actual weights.

<details id="gpt2-cache-code">
<summary>Model code and cache check</summary>

<a id="a-complete-small-model-full-and-cached-passes-agree"></a>

The following example retains pre-norm, GELU, two residual branches,
final LayerNorm, learned positions, and a tied output table.
Width is reduced to 16, blocks to two, and the window to 16.
Weights are random, so this checks computation, not text quality.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(11)
torch.set_num_threads(1)

class Block(nn.Module):
    def __init__(self, d=16, heads=4):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4*d), nn.Linear(4*d, d)

    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        split = lambda z: z.reshape(b, t, self.heads, d//self.heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        p = 0 if past is None else past[0].shape[-2]
        if past is not None:
            k, v = (torch.cat((old, new), dim=-2)
                    for old, new in zip(past, (k, v)))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d//self.heads)
        query_pos = p + torch.arange(t, device=x.device)
        key_pos = torch.arange(k.shape[-2], device=x.device)
        allowed = key_pos[None, :] <= query_pos[:, None]
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        a = (weights @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(a)
        x = x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))
        return x, (k, v)

class TinyGPT2(nn.Module):
    def __init__(self, vocab=17, d=16, length=16, layers=2):
        super().__init__()
        self.word, self.position = nn.Embedding(vocab, d), nn.Embedding(length, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d)
        for name, param in self.named_parameters():
            if param.ndim >= 2:
                nn.init.normal_(param, std=.02)
            elif name.endswith("bias"):
                nn.init.zeros_(param)
            else:
                nn.init.ones_(param)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=.02/math.sqrt(2*layers))
            nn.init.normal_(block.out.weight, std=.02/math.sqrt(2*layers))

    def forward(self, ids, past=None):
        p = 0 if past is None else past[0][0].shape[-2]
        t = ids.shape[1]
        assert 0 < t and p + t <= self.position.num_embeddings
        pos = p + torch.arange(t, device=ids.device)
        x = self.word(ids) + self.position(pos)
        cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if past is None else past[i])
            cache.append(kv)
        logits = F.linear(self.final_norm(x), self.word.weight)
        return logits, cache

model = TinyGPT2().double().eval()
assert sum(p.numel() for p in model.parameters()) == 7120
ids = torch.tensor([[1, 2, 3, 4, 5]])
full, _ = model(ids)
prefix, past = model(ids[:, :3])
parts = [prefix]
with torch.no_grad():
    for t in range(3, 5):
        step, past = model(ids[:, t:t+1], past=past)
        parts.append(step)
        assert past[0][0].shape[-2] == t + 1
cached = torch.cat(parts, dim=1)
torch.testing.assert_close(full, cached, rtol=1e-10, atol=1e-10)

# The unchanged next-token objective can still train the same network.
loss = F.cross_entropy(full[:, :-1].reshape(-1, 17), ids[:, 1:].reshape(-1))
loss.backward()
assert model.word.weight.grad.norm() > 0
assert model.blocks[0].qkv.weight.grad.norm() > 0
print(tuple(full.shape), len(past), tuple(past[0][0].shape))
print("same logits; nonzero language-model gradients")
# (1, 5, 17) 2 (1, 4, 5, 4)
# same logits; nonzero language-model gradients
```

</details>

## How GPT-2 prompts for translation and answers

<a id="gpt2-translation-prefix"></a>

<a id="translation-and-answers-through-prefixes"></a>

### Translation from sentence pairs

[Section 3.7 of the report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=6) specifies translation using pairs
`english sentence = french sentence` followed by an unfinished pair.
This teaching prompt follows that format:

```text
The cat sleeps. = Le chat dort.
The dog runs. = Le chien court.
The bird sings. =
```

In the first two lines, English is on the left of `=` and French is
on the right. The right side of the third line is empty. The
[GPT-2 tokenizer](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py)
encodes all three lines as 27 tokens, including two newlines. The last
token, ` =` (ID 796), sits at zero-based position 26. Its output gives
the distribution for the first translation token. If it selects
` L` (ID 406), the next pass takes that token as input and scores,
among other tokens, the apostrophe (ID 6). The string
` L'oiseau chante.` takes seven tokens: `[406, 6, 25678, 559, 442, 12427, 13]`.
These are the token IDs of the intended translation, not a GPT-2
generation. In the experiment, the authors chose each token greedily
and kept the first generated sentence.

Translation pairs also occur in the training text. In
[Table 1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3), the authors show WebText passages where
a French quotation is followed by its English translation. During
pre-training, the model predicts words of that translation from the
preceding text, including the quotation. GPT-2 was not fine-tuned on
a separate parallel corpus, but WebText did contain translations.
The authors estimated its French text at about 10 MB. On WMT-14, GPT-2
scored 5 BLEU from English to French and 11.5 in the reverse direction.
That is consistent with the predominance of English in WebText,
but comparing directions does not isolate the effect of the corpus mix.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-coqa-2026-11-11/liu22-figure3-translation-prefix.png|Three pairs of English and German expressions form the context. The translation after mountain == is blank on the fourth line.]]

*Jiachang Liu et al., [What Makes Good In-Context Examples for GPT-3?, Figure 3, p. 111](https://aclanthology.org/2022.deelio-1.10.pdf#page=12),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original diagram from Appendix A.*

In Liu et al.'s diagram, the first three lines are already in the prompt.
The model continues only the fourth, after `mountain ==`; the dots mark
the blank and are not input. This illustration translates into German
and stops at a newline. It shows a GPT-3 prompt format, not the result
of GPT-2's French translation experiment.

In the report, zero-shot transfer means no training of weights on the
downstream task. Translation and Natural Questions nevertheless use
demonstrations inside the prompt. In modern terminology, this teaching
prompt is two-shot. Demonstration count and gradient fine-tuning must
be tracked separately.
The [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#Translation: direction and parallel examples|GPT-3 chapter]]
shows the same sentence with zero, one, and two completed pairs; the
paper compares zero, one, and 64 pairs.

Summarization appends `TL;DR:` to an article. The reported experiment
generates 100 tokens with top-k at $k=2$ and takes the first three
sentences. Results therefore depend on the marker, decoding, and truncation.

<a id="gpt2-gpt3-coqa"></a>

<a id="coqa-dialogue-history-and-demonstrations"></a>

### Answering from prior turns in CoQA

In [CoQA's first example](https://arxiv.org/html/1808.07042v2#S1.F1), three
people are coming to Jessica's birthday. The questions establish whether
she expects visitors and how many, then the fifth question asks only
`Who?` The earlier turns tell you who to look for; the passage supplies
the visitors' names.

To ask the fifth question, give the model the passage and the four
previous questions with their reference answers. End the prompt with
`Q: Who?` and a blank `A:`. The fifth reference answer and the evidence
spans marked by CoQA's annotators are withheld. In the
[GPT-2 experiment, §3.5](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=6),
the model decoded greedily after `A:`. For the earlier “how many” question,
it could produce `Three` even though the word does not appear in the passage.
The answer need not be a contiguous span.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-coqa-2026-11-11/coqa-figure4-conversation-flow.png|Ten columns show the first ten turns. Early answers draw more evidence from the beginning of the passage; later answers draw more from the end. Bands show transitions between parts of the passage.]]

*Siva Reddy, Danqi Chen, and Christopher D. Manning,
[CoQA: A Conversational Question Answering Challenge, Figure 4](https://aclanthology.org/Q19-1016.pdf#page=8).
Full plot from the authors' copy of the paper, not redrawn;
[CC BY 4.0](https://transacl.org/ojs/index.php/tacl/article/view/1572).*

Each column shows ten equal parts of the passage in order, from top to
bottom. The thick band linking the top blocks of columns one and two
means both early questions often use evidence from the beginning.
By turn ten, the bottom blocks are taller: answers draw on the end
more often. The plot summarizes spans selected by annotators. It does
not show GPT's attention weights.

In the [GPT-3 experiment](https://arxiv.org/html/2005.14165v4#A7.F18),
earlier turns about the same passage remain part of the input even in
zero-shot. One-shot prepends a completed example about a different
passage, and few-shot uses five such examples on CoQA. The weights stay
fixed. These are the word-overlap F1 scores on the development set
from [GPT-3 Table H.1](https://arxiv.org/html/2005.14165v4#A8.T1).
Columns 0, 1, and 5 count separate examples prepended to the dialogue.

| Model | 0 | 1 | 5 |
| --- | ---: | ---: | ---: |
| GPT-2 1.5B | 55 | — | — |
| GPT-3 350M | 55.0 | 52.1 | 52.0 |
| GPT-3 175B | 81.5 | 84.0 | 85.0 |

Five examples add 3.5 F1 points for GPT-3 175B over its own zero-shot
score but cost GPT-3 350M three points. The difference between the
generations cannot be attributed to parameter count: the training data,
window, and prompt format also differ. The
[[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-relation-shots|worked classification prompt]]
shows exactly which lines a demonstration adds.

<a id="selecting-a-candidate-by-text-probability"></a>

### Using right context to fill a blank

Children's Book Test asks the model to choose a missing word from ten
candidates. GPT-2 scores each word and the remaining sentence. In the
second factor, the candidate is part of the context, so substituting a
different word changes the suffix probability too:

| Word | Word $P$ | Suffix $P$ | Product |
|---|---:|---:|---:|
| A | 0.6 | 0.2 | 0.12 |
| B | 0.4 | 0.8 | 0.32 |

Scoring only the blank would choose A; scoring the blank and suffix
chooses B. These numbers illustrate the rule, not CBT measurements.
An actual sequence sums log-probabilities of every candidate and suffix
token. This uses right context for selection while every computation
remains causal.

CBT sums log-probabilities without dividing by length. Replacing the sum
with a mean can change the winner when the choices have different token counts.
The [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#gpt-llama-completion-scoring|comparison with GPT-3 and LLaMA]]
shows how normalizing by tokens or by characters can select different endings.

Winograd Schema compares full and partial scoring from
[Trinh and Le, Section 3](https://arxiv.org/abs/1806.02847v2).
Each candidate replaces the pronoun. Full scoring includes the whole
resulting string; partial scoring includes only the text after the
replacement, conditioning on preceding text and the candidate.
For the trophy/suitcase sentence, for example, it compares the suffix
`is too big` after `the trophy` and `the suitcase`.

Candidate probability can change the decision. Suppose A has probability
0.8 and a conditional suffix probability 0.4, while B has 0.2 and 0.5.
After canceling the shared left prefix, full scores are 0.32 and 0.10,
choosing A; partial scores are 0.4 and 0.5, choosing B. This is a
separate constructed example. GPT-2 Figure 3's 70.70% belongs to
partial scoring, not arbitrary replacement-ranking rules.

<a id="what-the-experiments-showed"></a>

## Metrics and evaluation protocols

Section 3.1 reports improvements over prior best results on seven of
eight language-modeling datasets without fine-tuning. It does not
claim success on seven of eight arbitrary NLP tasks. Perplexity
comparisons require a normalization unit: byte, character, word,
and BPE token give different quantities.

For example, total NLL of 6 nats over three words gives word-level
perplexity $e^{6/3}\approx7.389$. If those words comprise five BPE
tokens, token-level perplexity is $e^{6/5}\approx3.320$.
The smaller second number does not mean a better model. The report
normalizes log-probability to each dataset's units and describes
reversible detokenizers that reduce preprocessed-test artifacts.

| Test | GPT-2 1.5B and evaluation protocol |
| --- | --- |
| LAMBADA | 52.66% accuracy; 63.24% with an added stop-word filter<br>final-word prediction; 63.24% is not the unfiltered result |
| CBT, common nouns / named entities | 93.3% / 89.05%<br>validation split: one test book was found in WebText |
| Winograd Schema Challenge | 70.70% accuracy<br>partial scoring; small set of 273 examples |
| CoQA | 55 F1<br>development set; document, dialogue history, greedy decoding |
| CNN/DailyMail | ROUGE-1/2/L: 29.34 / 8.27 / 26.58<br>`TL;DR:`, top-k=2, 100 tokens, first three sentences |
| WMT-14 En→Fr / Fr→En | 5 / 11.5 BLEU<br>translation pairs in context; first greedy sentence |
| Natural Questions | 4.1% exact match<br>question–answer examples in context; no document retrieval |

Numbers and protocols come from Sections 3.2–3.8 and Tables 3–4.
Word accuracy, BLEU, and F1 here belong to different datasets.
Scores are comparable within their protocols, not across rows.

The summarization table shows the method's limits: ROUGE-AVG is 21.40
with `TL;DR:`, 15.03 without it, 20.98 for Random-3, and 31.55 for
Lede-3. The model responds to a task marker but still trails simply
taking the article's first three sentences. The authors note confused
event details in examples.

Natural Questions' overall 4.1% exact match differs from 63.1% on the
1% of questions receiving the highest model confidence. Selecting
a small confident subset does not give 63.1% over the full dataset.
Table 5 includes confident wrong answers. String probability and
factual correctness remain different quantities.

<a id="gpt2-gpt3-lambada"></a>

### LAMBADA: choosing the last word

LAMBADA removes the last word from a passage. In building the dataset,
[Paperno et al., §3.1](https://aclanthology.org/P16-1144.pdf#page=4)
kept a passage if two readers recovered the word from the full passage
and none of ten other readers guessed it from the final sentence alone.
“In the morning he rode away on his…” leaves open what Anton rode.
An earlier sentence mentioning his bicycle gives the reader a clue.
A plausible answer still counts as wrong if it does not match the
original word.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-lambada-2026-11-13/paperno-lambada-figure2a.png|Two stacked bars. The light segment shows that the target word occurred earlier in over 80% of LAMBADA passages and under 15% of passages sampled before human filtering.]]

*Denis Paperno et al., [The LAMBADA dataset, Figure 2(a)](https://aclanthology.org/P16-1144.pdf#page=6),
[CC BY 4.0](https://aclanthology.org/faq/copyright/). Original panel, not redrawn.*

The light portion of each bar represents passages that already
contain the target word. In LAMBADA it exceeds 80%. The bar labeled `input`
represents 5,000 randomly sampled passages shown to readers before
filtering on their answers. There, the share is under 15%. The figure
shows how filtering changed the dataset, not how accurately a model
can find the right word among the words it has read. If the answer
does not appear earlier, a copy-only rule rules it out entirely.
In the GPT-2 report this affected 19% of answers, and restricting
candidates to words in the passage lowered accuracy.

There is a separate problem: the language model does not know that
the next word must end the sentence. Give it this prefix:

~~~text
Anton left his bicycle outside. In the morning he rode away on his
~~~

Suppose `very` has probability 0.50, `bicycle` 0.30, and `scooter` 0.20.
Unrestricted selection can pick `very` and go on to write `very old bicycle`.
In LAMBADA, `very` is wrong: the missing word was `bicycle`. If the
selection rule excluded `very`, `bicycle` would win without changing
the scores the model produced. These numbers are for illustration,
not a GPT-2 run, and the example does not reproduce OpenAI's list.

In the [GPT-2 report, §3.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=5),
the authors exclude stop words as an approximation to the final-word
constraint. Accuracy rises from 52.66% to 63.24%. That filter does
not check whether every remaining word can end a sentence. The
[[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#lambada-gpt3-fill-blank|GPT-3 experiment with demonstrations]]
instead changes the input: the same passage can be written with a blank,
with a completed blank-and-answer pair above it. GPT-3 175B scores 76.2%
on plain continuation and 86.4% with 15 fill-in-the-blank examples.
Between those two results, both the format and the number of examples
change. Compared with GPT-2's 63.24%, the model and answer-selection
procedure also differ.

<a id="generation-memorization-and-model-release"></a>

## Which benchmark texts appeared in WebText

The authors lowercase the text and keep alphanumeric words. They
look for matching eight-word spans in WebText using a Bloom filter
([report, Section 4](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=8)).
Eight matching words may be a common phrase, not a copied document.
The authors therefore inspect the matches by hand.

About 15% of CoQA's news documents already appeared in WebText.
Scores on those documents were roughly 3 F1 higher; the estimated
increase across development-set domains was approximately 0.5–1 F1.
CoQA appeared after the link cutoff. The authors distinguish the
passage overlap they measured from overlap of the annotated dialogues.

Removing every LAMBADA example with detected overlap changes perplexity
from approximately 8.6 to 8.7 and accuracy from 63.2% to 62.9%.
This demonstrates small sensitivity to that particular exclusion
procedure. It does not establish absence of all duplicates, semantic
overlap, or memorization.

Similar WebText training and held-out losses continued improving
with model size. This concerns fit to the corpus distribution.
Producing a coherent fictional news article does not verify its contents.

GPT-2 weights were released in stages during 2019, ending with the
largest 1.5B model.
[OpenAI's model card](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/model_card.md)
describes research use, limitations, and biases. The release is a base language model without instruction tuning or
RLHF; the generation script takes text, not formatted chat messages.

<a id="limitations-and-the-transition-to-gpt-3"></a>

## Demonstrations in GPT-2 and GPT-3

The 1024-token window limits demonstrations, the new input, and the
answer together. If demonstrations and a question take 900 positions,
124 remain for the continuation. Those positions are BPE tokens, not words.

GPT-2's translation prompts already used completed sentence pairs.
[[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#Translation: direction and parallel examples|GPT-3]]
varies their number experimentally: the model performs the same task
with zero, one, or several examples while its weights stay fixed.
The next chapter compares those settings and traces changes in the
model's size and training data.

## Sources and further reading

<details id="gpt2-nanogpt-reference">
<summary>Additional code: nanoGPT</summary>

The [[06 Практика/20 Собрать языковую модель с нуля#nanogpt-worked-reference|worked nanoGPT example]]
shows the training path for a similar model. The original nanoGPT
sampler recomputes a cropped context without the OpenAI K/V cache.

</details>

- [Radford et al., Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019. Section 2: data, representation, and model; Section 3: task protocols; Section 4: data overlap.
- [OpenAI GPT-2, commit 9b63575e](https://github.com/openai/gpt-2/tree/9b63575ef42771a015060c964af2c3da4cf7c8ab). `model.py`: pre-norm, positions, cache; `encoder.py`: byte-level BPE; `sample.py`: generation; README and model card: corrected sizes and limitations.
- [Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), August 12, 2019. Original input, stack, output, autoregression, and K/V figures; their simplifications are noted beside them.
- [Trinh and Le, A Simple Method for Commonsense Reasoning, v2](https://arxiv.org/abs/1806.02847v2), Section 3. Full/partial scoring definitions used by GPT-2's Winograd experiment.
- [Stanford CS224n, Lecture 9: Pretraining, slides 51–52](https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture09-pretraining-updated.pdf#page=51). Continuations with human prefixes and generated text separately identified, in the original English.
- [[02 Areas/ML & DL/Papers/GPT 2.0|Earlier GPT-2 note]] — additional material, not a substitute for the report and released code.

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1]].
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3]].
