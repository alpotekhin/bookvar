---
title: LLaMA as a baseline architecture
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "23084f7630399202af0453352e3ca725394da6e5a2bde20c3a42d7adcf8cdb93"
translation_reviewed: "2026-11-21"
translation_of: "00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура.md"
last_updated: 2026-11-21
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning]]"
next: "[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block]]"
primary_sources:
  - https://arxiv.org/abs/2302.13971v1
  - https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
  - https://arxiv.org/abs/1910.07467
  - https://arxiv.org/abs/2104.09864v5
  - https://arxiv.org/abs/2002.05202v1
  - https://aclanthology.org/2023.findings-acl.527/
  - https://arxiv.org/abs/2203.15556v1
  - https://proceedings.mlr.press/v235/sardana24a.html
---

# LLaMA as a baseline architecture

<a id="llama-relation-prefix"></a>

Give the first LLaMA the same prompt as in the
[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-relation-shots|GPT-3 chapter]],
without demonstrations or special tokens. After the colon, compare
three continuations: ` entailment`, ` contradiction` and ` neutral`.

~~~text
Premise: A cat sleeps.
Hypothesis: An animal rests.
Relation:
~~~

The reversible GPT-2 tokenizer retained by GPT-3 encodes the prompt
in 20 tokens. The first LLaMA's SentencePiece tokenizer uses 23.
GPT-3 splits ` entailment` into two tokens and LLaMA splits it into
three. Call LLaMA's tokens $y_1,y_2,y_3$. The logits at the last
position of the prompt give the probability of $y_1$. Add $y_1$ to
score $y_2$, then add $y_2$ to score $y_3$. The product of those
three conditional probabilities scores the whole label.
` contradiction` and ` neutral` each use a single token in both models.
This scores the start of a continuation. More text can follow the
label. The [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-candidate-scores|GPT-3 example]] shows how to compare labels
of different lengths.


<a id="gpt-llama-comparison"></a>
<a id="от-gpt-3-к-первой-llama"></a>

<a id="от-gpt-1-к-первой-llama"></a>

## From GPT-1 to the first LLaMA

GPT-1 applied a trained classifier to the last position of the pair.
From GPT-2 onward, the vocabulary projection can score the answer.
GPT-3 and LLaMA v1 share a 2,048-token window and a pretraining
objective: predict the next token. They encode position at different stages and use different
feed-forward layers.

| Model | Decoder | Training and task interface |
| --- | --- | --- |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение#gpt1-decoder-architecture|GPT-1 (2018)]] | Post-norm, absolute position embeddings; 512 tokens. | About 117M parameters. BooksCorpus, then task-specific fine-tuning. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-pre-norm|GPT-2 (2019)]] | Pre-norm, final LayerNorm; 1,024 tokens. | Up to 1.56B parameters. WebText. The prefix specifies the task. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-full-local-attention|GPT-3 (2020)]] | Alternating full and local attention layers; 2,048 tokens. | 125M to 175B parameters, 300B training tokens per model. Evaluation varies the number of demonstrations. |
| LLaMA v1 (2023) | RMSNorm, RoPE, SwiGLU, full attention; 2,048 tokens. | 7B to 65B parameters, 1T to 1.4T training tokens. The prefix specifies the task. |

Sources: [GPT-1, §§3–4](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=3),
[GPT-2, §§2–3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3)
and the [corrected release counts](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/README.md),
[GPT-3, §2](https://arxiv.org/html/2005.14165v4#S2), and
[LLaMA, §§2–3](https://arxiv.org/html/2302.13971v1#S2).

<details id="llama-gpt-history">
<summary>How GPT‑1, GPT‑2 and GPT‑3 handle the same pair</summary>

<a id="gpt1-to-gpt2"></a>

### GPT-1 → GPT-2: from a classifier to text completion

In [Figure 1 of the GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=4),
the `Entailment` branch wraps the premise and hypothesis in special tokens:
`[Start, A cat sleeps, Delim, An animal rests, Extract]`.
At the last position, `Extract` can attend to both sentences.
The `Task Classifier` maps its 768-dimensional state to three
logits, one each for entailment, contradiction and neutrality.
The labels are used to fine-tune the head and decoder. The correct
label is not appended to the sentence pair.

GPT-2 can take the task as ordinary text:

~~~text
Premise: A cat sleeps.
Hypothesis: An animal rests.
Relation:
~~~

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt2-2026-10-22/gpt2-output-projection-2.png|The 768-dimensional output of small GPT-2 is compared with all 50,257 rows in its token embedding table.]]

*Jay Alammar, [The Illustrated GPT-2, Model Output](https://jalammar.github.io/illustrated-gpt2/#model-output),
[unmodified original](https://jalammar.github.io/images/gpt2/gpt2-output-projection-2.png),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Alammar illustrates the first position. For the sentence pair, use
the state of the last input token in `Relation:`. It is compared with all 50,257
vocabulary rows. The projection scores possible next tokens rather than
three classes. The [released GPT-2 code](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py)
applies a final LayerNorm before this projection, omitted in the figure.
In the [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-relation-prefix|worked GPT-2 example]],
` entailment` is two tokens. If its first token has probability 0.4
and its second has conditional probability 0.5, the whole label scores
$0.4\cdot0.5=0.20$, below the single-token ` neutral` at 0.25.
These probabilities illustrate the calculation; they are not
predictions from the released GPT-2.

Text completion and the move to pre-norm are separate changes.
The smallest GPT-2 still has GPT-1's 12 layers, 12 heads, and width
of 768, while the window grows from 512 to 1,024 tokens.
Inside the block, LayerNorm moves from the sum to the input of the sublayer:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-pre-post-norm.png|Two residual paths: on the left, LayerNorm follows each addition; on the right, it precedes attention and the FFN.]]

*Ruibin Xiong et al.,
[On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html),
[Figure 1, p. 2](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The diagram
compares the placement of LayerNorm, not the sizes or other details of GPT.*

Follow the lower skip path. In the GPT-1 ordering on the left, the input
bypasses attention, but the sum still passes through LayerNorm. In GPT-2
on the right, the input goes straight to the addition, and LayerNorm
prepares the input to attention. If attention adds zero, its
residual addition on the right preserves the input. On the left the
same sum still passes through LayerNorm.

<a id="gpt2-to-gpt3"></a>

### GPT-2 → GPT-3: demonstrations and scale

GPT-2 already received demonstrations in some experiments, including
[[#natural-questions-gpt-llama|closed-book question answering]].
GPT-3 varied the number of demonstrations systematically. The model
could receive a task description alone, the task with one solved example,
or the task with several solved examples. The weights stay fixed across
these evaluations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-continuity-2026-11-10/min-figure2-demonstrations.png|Three labeled texts and a fourth with its label blank go into the model. Positive below the model is its prediction.]]

*Sewon Min et al., [Rethinking the Role of Demonstrations, Figure 2](https://aclanthology.org/2022.emnlp-main.759.pdf#page=2).
Complete figure, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The figure puts three texts labeled `Positive`, `Neutral`, and
`Negative` before the test input. All four lines form one input
sequence. The `Positive` below the model is its prediction, outside
the prompt. Remove the first three lines and the same test input
remains. Make the same change to the example about the cat: in
one-shot, put a labeled pair about a door before it. Both prompts
start with the same instruction to choose one of the three labels.
Here is the text after the instruction:

~~~text
Premise: The door is open.
Hypothesis: The door is shut.
Relation: contradiction

Premise: A cat sleeps.
Hypothesis: An animal rests.
Relation:
~~~

Remove the first pair and its blank separator for zero-shot. Both prompts score
the same continuations after the final `Relation:`. The scores can
change because the first pair's tokens are now in the context; there
is no backward pass. The
[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-relation-shots|GPT-3 chapter]]
adds a second demonstration.

Brown et al. repeated the evaluation on eight model sizes, from
125 million to 175 billion parameters. Each model processed
300 billion training tokens. This lets us compare how much the
demonstrations helped at different sizes on the same task. On the
ANLI R3 test split, another sentence-pair classification task,
GPT-3 175B scored 34.5% with no demonstrations, 35.1% with one and
40.2% with 50. The 13B model scored 34.5% with 50, close to the 33.3%
expected from guessing among three labels.
[Brown et al., Table H.1](https://arxiv.org/html/2005.14165v4#A8.T1).
The [original plot, Figure 3.9](https://arxiv.org/html/2005.14165v4#S3.F9)
traces the three conditions across model sizes. Compare the curves
at the same horizontal position to see the effect of demonstrations.
The plot's caption says `dev`; the numbers above come from the
`test` row of Table H.1. The [ANLI prompt](https://arxiv.org/html/2005.14165v4#A7.F10)
used `True`, `False` and `Neither` rather than the labels in our example.

GPT-3 kept the main GPT-2 block but changed the mask in some layers.

</details>

<a id="какие-токены-читает-один-слой"></a>

### Which tokens can one layer read?

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/long-context-review-2026-09/mistral-sliding-window.png|Full causal attention, a three-position causal window, and the receptive field across several local layers.]]

*Albert Q. Jiang et al., [Mistral 7B, Figure 1](https://arxiv.org/html/2310.06825v1#S2.F1).
[Unmodified original](https://arxiv.org/html/2310.06825v1/swa.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The two masks illustrate the principle, not GPT-3's configuration.*

The row for `on` lists the keys its query can read. In the left matrix,
those are `The`, `cat`, `sat` and `on` itself. In the middle, `The`
drops out: the pictured window holds three positions including the
current one. The final `the` is to the right of the query and is
inaccessible in both masks. A one marks an allowed connection,
not a softmax weight.

GPT-1, GPT-2 and the first LLaMA use the left-hand mask at every layer.
GPT-3 alternates full and local layers. If `cat` already read `The` in
a full layer, the next local layer can receive information about `The`
through the state at `cat`, even though `on` has no direct connection
to `The`. The right panel shows a different case: several consecutive
local layers, as in Mistral. The GPT-3 authors do not specify their
local window width. The
[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-full-local-attention|GPT-3 example]]
computes a query's weights before and after removing a distant key.

<a id="gpt3-to-llama1"></a>

### GPT-3 → LLaMA: a new block at the same context length

The first LLaMA uses full causal attention in every layer. In the
example above, `on` can attend to `The` in each layer. The window
stays at 2,048 tokens: the mask and the length of the window are
separate choices. Its [[#Pre-norm and residual connections|pre-norm block]]
keeps the GPT-2 and GPT-3 ordering, but RMSNorm rescales without
subtracting the mean.

<a id="gpt-llama-position-shift"></a>

In GPT-2 and GPT-3, position enters the token state before the first
block. The [GPT-2 implementation](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L141-L165)
adds a row of the learned `wpe` table, selected by position, to each
token embedding. GPT-3 keeps that design. Its local layers change which
keys a query can read, not how positions are encoded
([Brown et al., §2.1](https://arxiv.org/html/2005.14165v4#S2.SS1)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt2-2026-10-22/gpt2-input-embedding-positional-encoding-3.png|GPT-2 adds the token embedding and the position embedding before the first block.]]

*Jay Alammar, [The Illustrated GPT-2, Input Encoding](https://jalammar.github.io/illustrated-gpt2/#input-encoding),
[original PNG](https://jalammar.github.io/images/gpt2/gpt2-input-embedding-positional-encoding-3.png),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The figure labels the first position #1; the code indexes it at 0.*

Take two toy tokens, `a` and `cat`, with embeddings $(1,0)$ and $(0,1)$.
Use one head and identity Q/K/V projections, and leave out normalization
to isolate the effect of position. Let the rows $p_0,p_1,p_2$ be
$(0,0),(1,0),(0,1)$. At positions 0 and 1, the vectors after addition are
$(1,0)$ and $(1,1)$. The dot product of the query for `cat` and the key
for `a` is 1. Give the same tokens indices 1 and 2 instead and their
vectors become $(2,0)$ and $(0,2)$, with a dot product of 0. The gap is
unchanged, but the table rows are different. The values V change too:
they are projected from the same input states.

The [first LLaMA, §2.2](https://arxiv.org/html/2302.13971v1#S2.SS2) has
no `wpe` table. Each layer computes Q, K and V first, then rotates Q
and K. In RoFormer's upper panel, `Query / Key` is on the left and
`Position Encoded Query / Key` is on the right. Between them is the
rotation: position $m$ supplies the angle $m\theta_1$, and the input
is the query or key that has already been projected.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png|RoFormer rotates one pair of coordinates in the upper panel and uses different frequencies for several pairs in the lower panel.]]

*Jianlin Su et al., [RoFormer v5, Figure 1, p. 5](https://arxiv.org/pdf/2104.09864v5#page=5).
Original figure and caption, unaltered;
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).*

Return to `a` and `cat`. Without the added position vectors, the key
and query before rotation are $(1,0)$ and $(0,1)$. At one radian per
position, the key at 0 is unchanged and the query at 1 becomes
$(-\sin 1,\cos 1)$. Their dot product is $-\sin 1\approx-0.8415$.
At positions 1 and 2, both vectors rotate by one additional radian,
so the dot product stays the same. The shift also preserves the other
allowed dot products and the value vectors V. Softmax gives the same
weights, and their weighted sum of V gives the same attention output.
We only renumbered the tokens. Prepending another token would give
the queries an additional key and could change the attention output.

<a id="gpt-llama-ffn"></a>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/tajima-figure1-ffn.svg|Left: GELU FFN with two matrices. Center: SwiGLU with separate input matrices, Swish on one branch and an elementwise product before the output matrix. Right: the separate SwiMGLU method.]]

*Yukito Tajima, Nakamasa Inoue, Yusuke Sekikawa, Ikuro Sato and Rio Yokota,
[Masked Gated Linear Unit, Figure 1](https://arxiv.org/html/2506.23225v1#S1.F1),
[original SVG](https://arxiv.org/html/2506.23225v1/fig_swimlu_v2.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Complete, unmodified figure.*

Read the arrows from bottom to top. On the left, the token vector passes
through the expansion matrix $W$, then GELU, then the output matrix $W_o$.
GPT-3 uses this FFN. In the center, the same input goes to two matrices,
$W_g$ and $W_v$. Swish (SiLU at $\beta=1$) acts only on the output of
$W_g$. The two branches are multiplied elementwise before $W_o$.
This is the first LLaMA's FFN. The right panel shows SwiMGLU, a later
method from the figure's authors.

GPT-3 6.7B and LLaMA-7B both have 32 layers, 32 heads and a
model width of 4,096. At that width, GPT-3's FFN expands to 16,384
coordinates; each of LLaMA's two branches expands to 11,008.
The narrower branches keep the increase in FFN weights to about
0.78% despite the third matrix. The [[#FFN width and rounding|matrix count below]]
shows how these widths arise.

<a id="gpt-llama-tokenization"></a>

<a id="сколько-позиций-занимает-одна-и-та-же-строка"></a>

## One string in three tokenizers

Consider `A cat sleeps 123`. In the released GPT-1 code,
pre-tokenization is followed by lowercasing; BPE operates within words,
and `</w>` marks a word boundary. GPT-2 starts from UTF-8 bytes.
It preserves case and spaces; a leading space can be part of the same
token as the letters or digits that follow. GPT-3 retains GPT-2's
reversible tokenizer. The first LLaMA uses SentencePiece BPE and splits
numbers into individual digits.
See the [GPT-1 code](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/text_utils.py),
[GPT-2, §2.2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4),
[GPT-3, §2.1](https://arxiv.org/html/2005.14165v4#S2.SS1) and
[LLaMA, §2.1](https://arxiv.org/html/2302.13971v1#S2.SS1).

The table counts only the string's tokens, without special tokens.

| Model | Count | Pieces |
| --- | ---: | --- |
| GPT-1 | 5 | `a</w>` · `cat</w>` · `sleeps</w>` · `1` · `23</w>` |
| GPT-2 / GPT-3 | 4 | `A` · `␠cat` · `␠sleeps` · `␠123` |
| LLaMA v1 | 8 | `▁A` · `▁cat` · `▁sle` · `eps` · `▁` · `1` · `2` · `3` |

The symbol `␠` displays a literal space inside a GPT-2 token.
The LLaMA row shows SentencePiece vocabulary entries, where `▁` marks
a space. The encoder inserts a dummy leading space that decoding removes.
That space is not BOS; adding BOS separately takes another position.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-token-embeddings-wte-2.png|GPT-2's token embedding table, with 50,257 rows and width 768, 1,024, 1,280 or 1,600.]]

*Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[original Token Embeddings (wte) figure](https://jalammar.github.io/images/gpt2/gpt2-token-embeddings-wte-2.png),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
unchanged. The words at the right are examples, not a map of actual IDs.*

GPT-2 assigns `␠cat` ID 3797 and reads row 3797 of the embedding table.
In the small model that row has 768 coordinates. LLaMA assigns `▁cat`
ID 6635, and the 7B model reads a row of width 4,096. The vocabularies
and weights belong together; swapping the tokenizer does not preserve
the mapping from text to model inputs.

A 2,048-position window limits the number of tokens in the encoded
prompt. GPT-3 spends one position on the space and `123`; LLaMA spends
four. This is one string, not an estimate of context capacity on a
corpus. To count how many demonstrations fit, tokenize the actual
prompt with its separators and leave room for the answer.

<span id="сначала-вся-модель-на-одной-странице"></span>
<span id="что-унаследовано-от-gpt"></span>

<a id="от-токенов-к-вероятности-продолжения"></a>

## From token IDs to next-token probabilities

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/cs336-model-overview.png|CS336 overview of a decoder-only Transformer from token embeddings to vocabulary logits]]

*Stanford CS336 Staff, Assignment 1, version 26.0.3,
[Figure 1, p. 13](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=13).
Original course diagram,
[MIT](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/LICENSE).
It shows the same main sublayers in the course's decoder-only model, rather than a figure from the LLaMA paper.*

<details>
<summary>How CS336 describes the model (§3.1)</summary>

> Given a sequence of token IDs, the Transformer language model uses an input
> embedding to convert token IDs to dense vectors, passes the embedded tokens
> through num_layers Transformer blocks, and then applies a learned linear
> projection (the “output embedding” or “LM head”) to produce the predicted
> next-token logits.

</details>

A batch of $B$ sequences of length $T$ starts as an integer tensor of shape
$[B,T]$. Each ID selects a row from a learned table
$E\in\mathbb R^{V\times d}$, giving a $[B,T,d]$ tensor: one vector of
width $d$ per position. Unlike the early GPTs, LLaMA does not add a learned
absolute-position embedding to each vector.

Each of the $L$ blocks preserves this shape. Attention reads from the
positions permitted by its mask. The FFN transforms each position separately,
using the same matrices at every position. Different blocks have their
own attention and FFN weights and normalization scales.

After the last block, a final RMSNorm and a vocabulary projection
$W_U\in\mathbb R^{d\times V}$ produce logits of shape $[B,T,V]$.
Softmax converts the logits at a position to next-token probabilities.
Generation needs the logits at the final position; training normally
includes the loss at every valid target position.

Take inputs `[BOS, A, B]` and targets `[A, B, EOS]`. The output at
`A` predicts `B`, so it may read `BOS` and `A`, but not `B`.
The causal mask enforces this rule even when all positions are computed
in parallel. The sampled token becomes part of the input to the next step.

<span id="изменение-1-нормализация-стоит-перед-подслоем"></span>

<a id="pre-norm-и-остаточные-связи"></a>

## Pre-norm and residual connections

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/cs336-prenorm-block.png|CS336 pre-norm Transformer block with residual paths around attention and the FFN]]

*Stanford CS336 Staff,
[Figure 2, p. 13](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=13).
Original diagram, MIT. Follow both paths to each addition. The assignment
uses RMSNorm for the boxes labeled Norm.*

In the lower half of the figure, the input splits. One path goes straight
to the addition; the other passes through normalization and causal
attention with RoPE. The upper half repeats the pattern with the FFN.
Writing RMSNorm as $R$, the two updates are

$$
\begin{aligned}
h&=x+\operatorname{Attn}(R_1(x)),\\
y&=h+\operatorname{FFN}(R_2(h)).
\end{aligned}
$$

$R_1$ and $R_2$ have separate learned scales. The addition keeps the
incoming state and adds the output of the sublayer. Attention must
therefore return width $d$ after working in separate heads. The FFN also
returns width $d$ after expanding to its wider intermediate state.

<span id="post-norm-в-исходном-transformer"></span>
<span id="pre-norm-в-llama"></span>

In the original Transformer, LayerNorm comes after the addition. Use $N$
for the same arbitrary normalization in both expressions to isolate the
effect of moving it:

$$
\begin{aligned}
\text{post-norm: }&y=N(x+F(x)),\\
\text{pre-norm: }&y=x+F(N(x)).
\end{aligned}
$$

If the sublayer $F$ outputs zero, pre-norm returns $x$ unchanged;
post-norm still returns $N(x)$. For instance, RMSNorm with unit gain
and $\epsilon=0$ maps $(1,2,2)$ and $(2,4,4)$ to the same vector. The
pre-norm residual path retains the difference at that addition.

The difference also appears in backpropagation. The pre-norm Jacobian
includes a direct term $I$: $I+J_FJ_N$. In post-norm the entire result is
multiplied by $J_N$: $J_N(I+J_F)$. This is the motivation discussed in
[CS336 §3.4](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=19)
and §2.2 of the LLaMA paper. The identity term alone does not guarantee
stable training; initialization, sublayer scale and the optimizer still
matter. LLaMA also applies a final RMSNorm after the entire stack.

<span id="изменение-2-rmsnorm-вместо-layernorm"></span>
<span id="численный-пример"></span>
<span id="2-реализовать-rmsnorm-и-проверить-инвариант"></span>

<a id="rmsnorm-масштаб-без-вычитания-среднего"></a>

## RMSNorm: rescaling without subtracting the mean

For $(1,2,2)$, the mean square is $(1+4+4)/3=3$, so the root mean square
is $\sqrt3$. Dividing by it gives $(0.5774,1.1547,1.1547)$. Its mean
is still nonzero; its mean square is one, up to rounding.

[RMSNorm §4](https://arxiv.org/abs/1910.07467) generalizes this operation
to $d$ coordinates. The LLaMA implementation adds $\epsilon$ under the
square root:

$$
R(x)_i=
g_i\frac{x_i}
{\sqrt{\frac1d\sum_{j=1}^{d}x_j^2+\epsilon}}.
$$

The learned $g_i$ gives each coordinate its own scale. The unit-RMS
statement applies before multiplication by $g$ and is exact only at
$\epsilon=0$. With positive $\epsilon$ it is a good approximation when
the input mean square is much larger than $\epsilon$.

LayerNorm first subtracts the mean $\mu$ and then divides by a standard
deviation. This difference persists even if both implementations omit
bias. At $\epsilon=0$, RMSNorm returns the same result for $x$ and $10x$,
but generally not for $x$ and $x+1$. Negative scaling flips the sign;
near zero, $\epsilon$ can no longer be ignored.

<span id="минимальная-реализация"></span>

This example keeps the original operation order: compute the squares
and mean in FP32, then cast back to the input dtype before multiplying
by the learned gain.

~~~python
import torch

def rmsnorm(x, gain, eps=1e-6):
    scaled = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps)
    return scaled.to(x.dtype) * gain

x = torch.tensor([[1., 2., 2.]])
gain = torch.ones(3)
print(rmsnorm(x, gain))
# tensor([[0.5774, 1.1547, 1.1547]])
torch.testing.assert_close(rmsnorm(x, gain), rmsnorm(10*x, gain))
print(rmsnorm(x + 1, gain))
# tensor([[0.7385, 1.1078, 1.1078]])
~~~

Squaring a sufficiently large FP16 value can overflow FP16. Doing that
part in FP32 does not require storing the whole model in FP32. The
multiplication's result also depends on the dtype of `gain`: an FP32
gain can promote a low-precision input again.

In the [first `model.py`](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py),
the `RMSNorm` constructor defaults to `1e-6`, but
`TransformerBlock` passes `args.norm_eps`, whose default in
`ModelArgs` is `1e-5`. The class signature by itself does not tell
you which setting a released checkpoint used. The formula and placement
are independent choices: GPT-2 uses pre-norm with LayerNorm; LLaMA uses
pre-norm with RMSNorm.

Compare the normalization in the [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-pre-norm|GPT-2 block]]
on a toy input: let the state for `cat` be $(0,1,0)$. With unit gain
and $\epsilon=10^{-5}$, RMSNorm gives $(0,1.7320,0)$.
LayerNorm with unit gain and zero bias gives $(-0.7071,1.4142,-0.7071)$:
subtracting the mean changes even the two initially zero coordinates.
To keep the next calculation small, use the first two coordinates
as the query. After RMSNorm it is $(0,1.7320)$ before RoPE.

<span id="изменение-3-rope-вместо-прибавления-position-embedding"></span>
<span id="3-увидеть-относительную-позицию-rope"></span>

<a id="rope-положение-меняет-сравнение-запросов-и-ключей"></a>

## RoPE: position changes how queries and keys compare

Suppose one two-dimensional pair has identical content in the query
and key, $q=k=(1,0)$. At frequency $\omega=1$, the query at position
$m=3$ rotates by three radians to $q'=(-0.9900,0.1411)$. The key
at $n=1$ rotates by one radian to $k'=(0.5403,0.8415)$.
Their dot product is $\cos(1-3)\approx-0.4161$. Adding 7 to both
positions leaves it unchanged.

The [[#gpt-llama-position-shift|RoFormer figure above]] shows one
rotation in the upper panel and several pairs with different
frequencies in the lower panel.

For a single pair the rotation is

$$
R(\alpha)=
\begin{bmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{bmatrix}.
$$

Since $R(\alpha)^\top=R(-\alpha)$,

$$
\begin{aligned}
&(R(m\omega)q)^\top R(n\omega)k\\
&\quad=q^\top R((n-m)\omega)k.
\end{aligned}
$$

The positional factor depends on the offset $n-m$. The vectors $q$ and
$k$ still come from the token states, so the score also depends on content.
RoPE does not turn attention into a function of distance alone.

<span id="несколько-частот"></span>

Within a head of width $d_h$, adjacent coordinates form $d_h/2$ pairs.
The original LLaMA uses these frequencies:

$$
\begin{aligned}
\omega_i&=10000^{-2i/d_h},\\
i&=0,\ldots,d_h/2-1.
\end{aligned}
$$

For $d_h=4$, the frequencies are 1 and 0.01 radians per position.
At position 100, the first pair rotates by 100 radians and the second
by one. Different pairs therefore change phase at different rates.

The original code treats each pair as a complex number and multiplies
it by $\cos(m\omega)+i\sin(m\omega)$. It never builds a
$d_h\times d_h$ rotation matrix. Each layer rotates Q and K, not V.
The precomputed sines and cosines are not learned weights.

In the running example, `a` and `cat` have positions 0 and 1 and
the toy head has one coordinate pair, with $\omega=1$. The projected
key for `a` is $(1.7320,0)$ and does not rotate. The query and key
for `cat` both rotate to $(-1.4574,0.9358)$. After division by
$\sqrt2$, the score for `a` goes from zero to about $-1.7850$.
The score for the key at `cat` stays $2.1213$: rotating both
vectors by the same angle preserves their dot product. The attention
weight on `cat` is then $0.9803$ and the future `sleeps` remains
masked. These values depend on the chosen toy weights; they are not
measurements from a LLaMA checkpoint.

Different arrangements of the coordinate pairs require corresponding
weight layouts. Two checkpoints that both use RoPE are not necessarily
interchangeable.

<span id="чего-rope-не-обещает"></span>

Sines and cosines are defined at any position. That does not tell us
whether the model learned to use phases and offsets outside its training
range. Nor does an individual attention score have to decrease with
distance: even $\cos(n-m)$ oscillates. The
[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание|long-context chapter]]
covers window extension and frequency changes separately.

<span id="изменение-4-swiglu-вместо-обычной-ffn"></span>

<a id="swiglu-две-проекции-перед-поэлементным-произведением"></a>

## SwiGLU: two projections before the elementwise product

A conventional FFN projects to a wider space, applies a nonlinearity,
then projects back to $d$. SwiGLU starts with two different projections
of the same input:

$$
\begin{aligned}
z_g&=xW_g,&z_u&=xW_u,\\
v&=\operatorname{SiLU}(z_g)\odot z_u,\\
\operatorname{FFN}(x)&=vW_d.
\end{aligned}
$$

Here $\operatorname{SiLU}(z)=z\sigma(z)$ and $\odot$ is elementwise
multiplication. Both branches use the vector at the current position.
The FFN does not read neighboring positions.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png|CS336 comparison of SiLU, ReLU and the identity function]]

*Stanford CS336 Staff,
[Figure 3, p. 21](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=21).
Original course figure, MIT. SwiGLU applies SiLU to only one of its
two input projections.*

Suppose the two projections give the following vectors:

| Stage | Vector |
| --- | --- |
| $z_g$ | $(-1,0,2)$ |
| $\operatorname{SiLU}(z_g)$ | $(-0.2689,0,1.7616)$ |
| $z_u$ | $(2,-3,0.5)$ |
| Product | $(-0.5379,0,0.8808)$ |

With the output matrix

$$
W_d=\begin{bmatrix}
1&2\\
-1&1\\
0.5&-0.5
\end{bmatrix}
$$

the result is about $(-0.0975,-1.5162)$. In the product, the second
coordinate goes to zero, the first changes sign relative to $z_u$,
and the third grows from $0.5$ to $0.8808$.
The SiLU factor can be negative or greater than one; it is not a
probability.

[GLU Variants Improve Transformer, §§2–3](https://arxiv.org/abs/2002.05202v1)
compares FFN variants, not just activation functions in isolation.
In its T5-like experiment with matched parameters and compute, held-out
log-perplexity after 524,288 steps was 1.677 for ReLU, 1.683 for
ungated Swish and 1.636 for SwiGLU. Those are results for that setup,
not a general explanation of why SwiGLU must work better.

<span id="почему-внутренняя-ширина-выглядит-необычно"></span>
<span id="4-посчитать-параметры-ffn"></span>

<a id="ширина-ffn-и-округление"></a>

### FFN width and rounding

The two matrices in an ordinary FFN of hidden width $4d$ contain
$8d^2$ weights. SwiGLU with hidden width $f$ has three matrices and
$3df$ weights. Equalizing the counts gives $f=8d/3$.

The first LLaMA implementation truncates $8d/3$ to an integer, then
rounds it up to a multiple of 256. For $d=4096$, that is
$10922\longrightarrow11008$. [GPT-3 6.7B, §2.1 and Table 2.1](https://arxiv.org/html/2005.14165v4#S2.SS1)
uses width $4\cdot4096=16384$.
[LLaMA-7B, §2.2 and Table 2](https://arxiv.org/html/2302.13971v1#S2.SS2)
uses two input matrices and one output matrix at the rounded width
of $11008$. Counting matrix weights only, without biases:

$$
\begin{aligned}
\text{GPT-3:}\quad 2\cdot4096\cdot16384 &= 134\,217\,728,\\
\text{LLaMA:}\quad 3\cdot4096\cdot11008 &= 135\,266\,304.
\end{aligned}
$$

The extra 1,048,576 weights per block come from rounding. Without it,
each FFN would have $8d^2$ weights. Similar weight counts do not imply
the same runtime: the matrix shapes and intervening operations differ.
The multiple of 256 is specific to this code. CS336 asks for a
multiple of 64, and later Llama code includes a width multiplier.

The names `w1/w3/w2` in `FeedForward` correspond to $W_g/W_u/W_d$.
Swapping `w1` and `w3` changes which projection goes through SiLU
and therefore changes the function, even though their shapes match.

<span id="изменение-5-меньше-необязательных-параметров"></span>

<a id="матрицы-смещения-и-связывание-весов"></a>

## Matrices, biases and weight tying

In the original `model.py`, the attention and FFN linear projections
and the vocabulary output all set `bias=False`. RMSNorm learns a
scale but no bias. These choices are part of the implementation, not
a consequence of being a decoder-only Transformer.

LLaMA creates separate weights for the input token embeddings and the
vocabulary output. GPT-2 and T5 reuse the token embedding weights
instead. At $V=32000$ and $d=4096$, the two LLaMA matrices each
contain 131,072,000 weights. Having transposed shapes does not make
them share storage or gradients.

RoPE needs no learned table with one row per position. Its sine and
cosine buffer can be reconstructed from the head width, the frequency
base and the position indices. Each normalization contributes only
$d$ weights, but acts on the states passed to the much larger sublayers.


<span id="один-токен-проходит-через-блок"></span>

<a id="полный-числовой-проход-через-малый-блок"></a>

## A numerical pass through the entire block

Take two toy token vectors $x_0=(1,0)$ and $x_1=(0,1)$.
There is one block and one head of width 2. The attention and FFN
matrices are all identities, the RMSNorm scales are all ones, and the
FFN width is 2. Set $\epsilon=0$ to make the arithmetic easier to
follow. These are toy weights, not a configuration of LLaMA-7B.

The first RMSNorm gives $(\sqrt2,0)$ and $(0,\sqrt2)$.
These are also Q, K and V before RoPE. Position 0 does not rotate;
at frequency 1, the query and key at position 1 become
$(-1.1900,0.7641)$. The values V do not rotate.

The last query is compared with both keys. After division by $\sqrt2$,
the scores are about $(-1.1900,1.4142)$ and softmax gives
$(0.068866,0.931134)$. At position 0, the mask permits only the first
key, giving weights $(1,0)$ regardless of the second token.

Add the original, unnormalized input $x_1=(0,1)$ to the attention
output. Apply the second RMSNorm before the FFN, then add the FFN
output to $h$:

| Stage | Value |
| --- | --- |
| Weighted sum of V | $(0.097392,1.316822)$ |
| First addition, $h$ | $(0.097392,2.316822)$ |
| Second RMSNorm | $(0.059397,1.412966)$ |
| SwiGLU output | $(0.001816,1.605629)$ |
| Second addition, $y$ | $(0.099208,3.922451)$ |

The final RMSNorm gives $(0.035757,1.413761)$. Let the three columns
of the vocabulary projection be $(1,0)$, $(0,1)$ and $(-1,-1)$.
The logits are then $(0.035757,1.413761,-1.449519)$ and the
probabilities of the three toy tokens are about
$(0.192552,0.763847,0.043601)$.

<span id="практика"></span>
<span id="1-сопоставить-формулу-и-код"></span>
<span id="после-главы-нужно-уметь"></span>

<a id="код-прохода-и-одного-обучающего-шага"></a>

### Code for the forward pass and one training step

This example specifies all 46 trainable numbers. The input table `E`
and output matrix `U` are separate. The input IDs `[0,1]` are
paired with targets `[1,2]`: each position predicts the next ID.
The example uses FP64 so the intermediate values can be checked.

<details id="llama-block-listing">
<summary>Block and training step (Python)</summary>

~~~python
import math
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=6, sci_mode=False)
dtype = torch.float64
p = {
    "E": torch.nn.Parameter(torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=dtype)),
    "U": torch.nn.Parameter(torch.tensor([[1., 0., -1.], [0., 1., -1.]], dtype=dtype)),
}
for name in ("q", "k", "v", "o", "gate", "up", "down"):
    p[name] = torch.nn.Parameter(torch.eye(2, dtype=dtype))
for name in ("norm1", "norm2", "final"):
    p[name] = torch.nn.Parameter(torch.ones(2, dtype=dtype))

def rms(x, gain):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True)) * gain

def rope(x, start_pos=0):
    pos = torch.arange(start_pos, start_pos + x.shape[1], dtype=x.dtype)
    c, s = pos.cos(), pos.sin()
    return torch.stack((x[..., 0]*c - x[..., 1]*s,
                        x[..., 0]*s + x[..., 1]*c), dim=-1)

def forward(ids, cache=None):
    x = p["E"][ids]
    r = rms(x, p["norm1"])
    old = 0 if cache is None else cache["k"].shape[1]
    q, k = rope(r @ p["q"], old), rope(r @ p["k"], old)
    v = r @ p["v"]
    if cache is not None:
        k = torch.cat((cache["k"], k), dim=1)
        v = torch.cat((cache["v"], v), dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(2)
    t = ids.shape[1]
    query_pos = old + torch.arange(t)
    key_pos = torch.arange(old + t)
    forbidden = key_pos[None, :] > query_pos[:, None]
    a = scores.masked_fill(forbidden, -torch.inf).softmax(-1)
    h = x + (a @ v) @ p["o"]
    z = rms(h, p["norm2"])
    ff = (F.silu(z @ p["gate"]) * (z @ p["up"])) @ p["down"]
    y = h + ff
    logits = rms(y, p["final"]) @ p["U"]
    return logits, {"a": a, "h": h, "z": z, "ff": ff, "y": y,
                    "cache": {"k": k, "v": v}}

ids = torch.tensor([[0, 1]])
targets = torch.tensor([[1, 2]])
logits, trace = forward(ids)
loss = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
print("attention:", trace["a"].detach())
print("h:", trace["h"].detach())
print("second norm:", trace["z"].detach())
print("ff:", trace["ff"].detach())
print("y:", trace["y"].detach())
print("logits:", logits.detach())
print("probabilities:", logits.softmax(-1).detach())
optimizer = torch.optim.SGD(p.values(), lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()
with torch.no_grad():
    new_logits, _ = forward(ids)
    new_loss = F.cross_entropy(new_logits.reshape(-1, 3), targets.reshape(-1))
print("loss:", float(loss.detach()), "->", float(new_loss))
~~~

</details>

With these initial weights, the loss falls from 2.405477 to 2.366481.
The gradient flows through both residual additions, the normalizations,
the Q/K/V projections and SwiGLU. A lower loss on these two positions
says nothing about whether the toy model can model language.

The example's `rms` omits $\epsilon$ because all the vectors it
normalizes here are nonzero. A production model uses $\epsilon$ and
gets the FFN width and the head count from its configuration. The
toy decoder uses the same kinds of operations as a pretrained LLaMA,
but not its dimensions, training data or learned weights.

<span id="проверка-форм-тензоров-по-официальному-коду"></span>

<a id="формы-и-параметры-исходных-моделей"></a>

## Shapes and parameters of the original models

Section 2.2 and Table 2 of the paper give the four sizes. The FFN
width $f$ below follows the rounding rule in the original
`FeedForward` implementation. All four use $d_h=128$.

| Model | $d$ | $L$ | $h$ | $f$ |
| --- | ---: | ---: | ---: | ---: |
| 7B | 4096 | 32 | 32 | 11008 |
| 13B | 5120 | 40 | 40 | 13824 |
| 33B | 6656 | 60 | 52 | 17920 |
| 65B | 8192 | 80 | 64 | 22016 |

For 7B, a batch with $B=2$ and $T=3$ enters a block at shape
$[2,3,4096]$. Each Q/K/V projection initially preserves the width.
The original code reshapes it to $[2,3,32,128]$ and transposes
to $[2,32,3,128]$ for attention. The scores have shape $[2,32,3,3]$.
After mixing V, the heads are joined back to $[2,3,4096]$ and the
output projection produces the input to the residual addition.

The FFN forms two branches of shape $[2,3,11008]$, multiplies
them elementwise and projects back to $[2,3,4096]$. Normalization
preserves shapes; the final vocabulary projection gives $[2,3,32000]$.
In the notation $xW$, $x$ is a row vector. PyTorch stores
`Linear.weight` as `[out_features, in_features]`, the transpose of
$W$ as written here.

Without biases or weight tying, the parameter count is

$$
P=2Vd+L(4d^2+3df+2d)+d.
$$

The terms count the two vocabulary matrices, the four Q/K/V/O
projections, the three FFN matrices, two normalization scales per block,
and the final RMSNorm scale. For 7B with $V=32000$, this gives
6,738,415,616 parameters. The FP16 weights alone occupy about 12.55 GiB.
Gradients, optimizer states, the KV cache and temporary tensors are
not included.

The 7B model's FP16 self-attention cache for one request with 2048
processed positions takes

$$
2\cdot32\cdot2048\cdot32\cdot128\cdot2
=2^{30}\ \text{bytes}.
$$

That is 1 GiB: two arrays (K and V), 32 layers, 2048 positions,
32 heads per layer, 128 values per head and two bytes per value.
Doubling the number of concurrent sequences doubles this memory if
all sequences are equally long; the weights can still be shared.
The parameter count alone cannot determine the maximum serving batch.

<a id="что-выполняет-выпущенный-код"></a>

## What the released code computes

The pinned first
[`model.py`](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py)
implements inference. The `Attention` constructor creates CUDA
caches. `Transformer.forward` is decorated with
`@torch.inference_mode()` and projects only `h[:, -1, :]` into
the vocabulary. It returns $[B,V]$, not the training shape $[B,T,V]$.
It cannot be dropped into the training loop above as written.

The first pass caches K/V for the prompt. A later pass adds a token
that reads the saved states as well as its own. The saved K already
includes RoPE. The new query must use its position after the prefix,
not start at zero; otherwise its relative angles to the saved keys
are wrong.

The mask in that first snapshot covers an initial prompt and then
one token at a time. If several new tokens arrive after a cached
prefix, the scores are rectangular:
$[T_{\rm new},T_{\rm old}+T_{\rm new}]$. A square mask over the new
tokens alone is insufficient. In
[Llama 3's later `model.py`](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py),
the prefix columns are explicitly added on the left and left unmasked.
This difference is an implementation detail of the two snapshots.

Defaults for `max_seq_len`, `norm_eps` and the head count do not
replace the settings for a particular checkpoint. For example, Table 1
of the Llama 2 paper lists a 2K training context for the first LLaMA,
although the early `ModelArgs` defaults to `max_seq_len=1024`.

<a id="генерация-с-кэшем-на-том-же-малом-блоке"></a>

### Generation with a cache on the same toy block

The toy `forward` accepts an optional `cache`. Without one, it
processes the whole input. With one, it computes only the new positions
and appends their K/V. The number of saved keys supplies the RoPE
`start_pos`. The mask compares the absolute positions of new queries
and all keys. Old queries are not needed: the next step computes
attention for the new query.

For three prefix tokens and two new tokens, the allowed pairs are:

| Query / key | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 3 | yes | yes | yes | yes | no |
| 4 | yes | yes | yes | yes | yes |

Both new positions may read the entire prefix; position 3 cannot yet
read position 4. The mask is 2×5: three open prefix columns followed
by a causal triangle over the two new keys. Shifting a diagonal in
a standalone 2×2 matrix would still leave out the prefix columns.
The rectangular case occurs, for example, when a prompt is processed
in chunks.

After the training step above, generate three IDs by choosing the
largest logit at the final position on each call:

<details id="llama-cache-listing">
<summary>Generation loop with a cache (Python)</summary>

~~~python
with torch.no_grad():
    sequence = torch.tensor([[0, 1]])
    pending = sequence
    cache = None
    processed = 0
    for _ in range(3):
        logits, trace = forward(pending, cache)
        cache = trace["cache"]
        processed += pending.shape[1]

        # Check the same states against a complete, uncached pass.
        complete, _ = forward(sequence)
        torch.testing.assert_close(logits[:, -1], complete[:, -1])

        chosen = logits[:, -1].argmax(-1, keepdim=True)
        sequence = torch.cat((sequence, chosen), dim=1)
        pending = chosen

    print(sequence.tolist())
    print(processed, cache["k"].shape[1], sequence.shape[1])  # 4, 4, 5
~~~

</details>

The first call processes the two input tokens and chooses the first
new one. The second call processes that choice, and the third
processes the next one. The final chosen ID is already in
`sequence`, but it has not gone through the block and is not in
the cache. There are five IDs and only four computed positions.
Without the cache, the three passes would process $2+3+4=9$
positions. The attention matrices differ in shape, so 9/4 is a ratio
of position counts, not a measured speedup. The toy vocabulary has no
separate stop rule for EOS.

A multi-layer model needs one K/V cache per layer. This example
does not cache logits or completed answers. Its use of `torch.cat`
allocates fresh memory at every append; the code shows the calculation,
not an efficient serving implementation.

<a id="когда-можно-использовать-сохранённый-префикс"></a>

### When a saved prefix can be reused

In a causal decoder, the states of a prefix do not depend on what
comes after it. For `[a,b,c]` and `[a,b,d]`, the first two
positions can use the same saved K/V. The third position and anything
after it must be computed separately. This does not hold for the
bidirectional T5 encoder, where changing the third token can change
the first two states.

The token IDs, weights, positions, RoPE configuration and visibility
mask must agree. For example, keys rotated with one set of RoPE
frequencies cannot be reused after changing those frequencies.
Identical text is not enough if it tokenizes differently. Changing
the weights through fine-tuning invalidates the old cache too.

<a id="llama-pretraining-data"></a>

<a id="данные-и-объём-предобучения"></a>

## Pretraining data and token budget

The LLaMA team trained 7B and 13B on one trillion tokens each, and
33B and 65B on 1.4 trillion. Training a smaller model for longer
can make sense if it will serve many requests: another training
token does not add any weights that must be loaded for inference.
Its BPE tokenizer [[#gpt-llama-tokenization|splits numbers into individual digits]]
and falls back to UTF-8 bytes for unknown characters.
[Llama 2, §2.2](https://arxiv.org/pdf/2307.09288v1#page=6) confirms
that both generations use a vocabulary of 32,000 tokens.

Table 1 of the first paper gives these sampling proportions:

| Source | Share |
| --- | ---: |
| English Common Crawl / CCNet | 67% |
| C4 | 15% |
| GitHub | 4.5% |
| Wikipedia | 4.5% |
| Gutenberg and Books3 | 4.5% |
| arXiv | 2.5% |
| Stack Exchange | 2% |

CCNet deduplicates lines in Common Crawl, identifies the language and
filters for quality with an n-gram model. A further classifier
distinguishes pages used as Wikipedia references from randomly selected
pages. C4 applies other cleaning heuristics to web data of the same
kind; the authors included it to diversify the mixture.

For GitHub, the team kept Apache-, BSD- and MIT-licensed projects,
applied file-quality heuristics and removed exact duplicates.
Wikipedia comes from June–August 2022 snapshots in 20 languages.
The books are deduplicated by work using an overlap threshold of
more than 90%. For arXiv, the team starts from LaTeX sources, removes
preambles, bibliographies and comments, and expands user macros.
For Stack Exchange it removes HTML and sorts answers by score.

A sampling share is different from a corpus's size on disk or its
number of repetitions. At 1.4 trillion tokens, Wikipedia's 4.5%
accounts for about 63 billion sampled tokens; Table 1 reports
2.45 passes over that subset. At one trillion tokens, the share
remains 4.5%, but only 45 billion tokens are sampled. The epoch counts
in the 1.4T column do not carry over to all model sizes.

The [[#Model size and pre-training tokens|two 13B models compared earlier]]
have similar parameter counts but saw different numbers of training
tokens. At the other end, storing GPT-3 175B at two bytes per
parameter takes about 350 GB, against about 26 GB for LLaMA-13B.
Neither figure tells us the KV-cache size or the response latency.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/llama1-training-loss.png|LLaMA Figure 1: training loss against tokens seen for the four original model sizes]]

*Touvron et al., [LLaMA, Figure 1, p. 3](https://arxiv.org/pdf/2302.13971v1#page=3).
Original training loss curves, CC BY 4.0.*

<details>
<summary>Original caption to LLaMA Figure 1</summary>

> Training loss over train tokens for the 7B, 13B, 33B, and 65 models.
> LLaMA-33B and LLaMA-65B were trained on 1.4T tokens. The smaller models
> were trained on 1.0T tokens. All models are trained with a batch size
> of 4M tokens.

</details>

The 7B curve is still falling near one trillion tokens. The [[#llama-v1-results|evaluation results]] track accuracy on individual tasks over the same run. To assess
[[#training-and-inference-budget|whether further training pays off]],
first choose the quality level at which to compare models, then
estimate how many tokens they will process in service.

<a id="оптимизация-и-обучение-на-нескольких-gpu"></a>

### Optimization and training on multiple GPUs

The run used AdamW with $\beta_1=0.9$, $\beta_2=0.95$, weight
decay 0.1 and gradient-norm clipping at 1.0. After a 2,000-step
warmup, cosine decay reduced the learning rate to 10% of its peak.
The peak was $3\cdot10^{-4}$ for 7B and 13B and $1.5\cdot10^{-4}$
for 33B and 65B. The global batch contained about four million tokens.

Section 2.4 describes several execution choices. The causal attention
implementation from xFormers avoids storing the full matrix of attention
weights and computing masked scores. Saving some expensive activations
reduces recomputation during backpropagation. The implementation also
uses model and sequence parallelism and overlaps computation with
communication between GPUs. These are ways of running the same model;
they do not change MHA to GQA or introduce a second loss.

The authors report about 380 tokens per second per GPU for 65B on
2,048 A100 80GB GPUs, and about 21 days for 1.4T tokens. Those
measurements are for the code and cluster in the 2023 paper, not
an estimate for current hardware or a result of the CPU example.

<span id="что-llama-не-изобрела"></span>

<a id="что-подтверждают-результаты-и-чего-они-не-выделяют"></a>

<span id="what-the-results-establish-and-what-they-leave-open"></span>

<a id="llama-v1-results"></a>

## LLaMA v1 results

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/llama1-downstream-training.png|LLaMA Figure 2: six downstream accuracy curves across the pretraining run]]

*Touvron et al., [LLaMA v1, Figure 2, p. 8](https://arxiv.org/pdf/2302.13971v1#page=8).
Accuracy on six tasks during pretraining. The horizontal axis is
billions of training tokens; the dashed line marks Chinchilla.
Original plots and caption, CC BY 4.0.*

On HellaSwag and the question-answering tasks, the longer runs usually
improve the result. SIQA fluctuates, while 33B and 65B are close on
WinoGrande despite differing in training loss. Section 3.7 discusses
both exceptions. Figure 1's declining loss does not predict an equal
gain on each downstream task. The dashed line comes from another
model, not another point on the LLaMA training curves.

Section 2.2 identifies where the parts came from: pre-normalization
from earlier GPT-style models, RMSNorm from Zhang and Sennrich, RoPE
from RoFormer, and SwiGLU from Shazeer's work and its use in PaLM.
LLaMA combines them with a different training recipe. Its GPT-3
comparisons do not isolate the contribution of each component.

For example, the zero-shot results in Table 3 show LLaMA-13B at
79.2 on HellaSwag versus 78.9 for GPT-3 175B, but 80.1 on PIQA
versus 81.0. The models differ in training volume, data, tokenization
and architecture. The paper scores candidates with likelihood
normalized by character count, except on BoolQ and OpenBookQA, where
it divides the candidate likelihood in context by its likelihood
after the background prefix `Answer:`. The resulting accuracy
is for that evaluation, not for an unconstrained chat reply.
In the [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#gpt-llama-completion-scoring|HellaSwag walkthrough]],
the same probabilities select different endings under token and character normalization.

The original LLaMA is a base model. In the separate experiment in §4,
instruction-tuned LLaMA-I 65B scores 68.9 on 5-shot MMLU against
63.4 for the base 65B. The training changes; the basic block does not.
The [original model card](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/MODEL_CARD.md)
notes that the base model was not trained with human feedback and
can produce inaccurate, harmful or unhelpful continuations.

<span id="ошибки-при-чтении-схем"></span>

When loading a checkpoint, match its generation, number of layers and
heads, FFN width and multiplier, `norm_eps`, RoPE frequencies and
layout, vocabulary and weight tying. Some errors cause a shape
mismatch. Others, such as reordering RoPE coordinate pairs or swapping
the two SwiGLU input branches, can keep all the shapes and still
change the computation.

<a id="gpt-llama-beyond-block"></a>

<span id="токенизация-объём-обучения-и-оценка-задач"></span>
<span id="tokenization-training-volume-and-evaluation"></span>

## GPT and LLaMA in comparative experiments

<a id="размер-модели-и-объём-предобучения"></a>

### Model size and pre-training tokens

The smallest GPT-2 and GPT-3 have almost the same parameter count,
124M and 125M, but GPT-3 doubles the context window and changes the data.
Its mixture adds filtered Common Crawl, two book corpora and Wikipedia
alongside WebText2. The 300B in the table counts tokens processed by
each model, not bytes on disk or the size of the unweighted mixture.
Sources were sampled at different rates.
See [GPT-3, Tables 2.1–2.2](https://arxiv.org/html/2005.14165v4#S2).

GPT-3 13B and LLaMA 13B form another near-size match. The former
processed 300B tokens and the latter 1T, about 3.3 times as many.
Because their tokenizers differ, this is a ratio of token counts,
not of raw text volumes. Moving along the 13B curve in
[[#llama-pretraining-data|LLaMA Figure 1]] means the same model has
seen more tokens. Moving to the 65B curve at the same horizontal
position means more weights at the same token count.

A rough pre-training estimate for a dense decoder is $6ND$, where
$N$ is the parameter count and $D$ the number of tokens processed.
With $N=13\cdot10^9$, 300 billion tokens cost about
$2.34\cdot10^{22}$ FLOP and one trillion costs $7.8\cdot10^{22}$ FLOP,
again a factor of 3.3. Compared with LLaMA-13B, GPT-3 175B uses
fewer tokens but comes to about $3.15\cdot10^{23}$ FLOP by the same
estimate. These are approximate operation counts, not measurements.
They do not separately account for attention dot products,
implementation differences, or GPU utilization.
[Approximation: Sardana et al., §2](https://arxiv.org/html/2401.00448v3#S2).

<a id="gpt-llama-parameter-budget"></a>

<a id="сколько-весов-приходится-на-словарь-и-позиции"></a>

### How many weights go to vocabulary and positions?

GPT-1, the smallest GPT-2 and the smallest GPT-3 each have twelve
layers, width 768 and twelve heads. GPT-2 gains roughly eight million
parameters over GPT-1 without changing the depth or width of its stack.
The released tokenizer vocabulary grows from 40,478 to 50,257 entries;
at 768 weights per row, that adds 7,510,272 parameters. Expanding the
position table from 512 to 1,024 entries adds 393,216, and GPT-2's
final LayerNorm adds 1,536. The count excludes the special tokens and
head added for GPT-1 task-specific fine-tuning.
See the [GPT-1 vocabulary](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/model/encoder_bpe_40000.json)
and the [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2 parameter count]].

From the smallest GPT-2 to the smallest GPT-3, the window doubles
again. The extra 1,024 rows at the same width account for about 0.79M
weights. That estimates the contribution of the position table, not
the exact count of GPT-3's unreleased weights. Its local attention
changes [[#Which tokens can one layer read?|access to keys]], not the
smallest model's head count or hidden width
([GPT-3, §2.1 and Table 2.1](https://arxiv.org/html/2005.14165v4#S2.SS1)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/gulordava-figure1-ab.png|Top: separate input embedding E and output projection W. Bottom: E is reused, transposed, for the output.]]

*Kristina Gulordava, Laura Aina and Gemma Boleda,
[How to represent a word and predict it, too, Figure 1(a–b), p. 2937](https://aclanthology.org/D18-1323.pdf#page=2),
EMNLP 2018. Two original panels, unchanged within the panels;
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The paper shows an LSTM. Transformer decoders face the same choice
of whether to share the two vocabulary matrices.*

In the bottom panel, the same table serves two roles. GPT-2 looks up
row 3797 of $E$ for input `␠cat`; at the output, the dot product of
the final state with that same row is the logit for `␠cat`. The
50,257 rows of width 768 count once: 38,597,376 parameters. LLaMA-7B
uses two independently learned rows for input `▁cat` and its output
logit, as in the top panel. Its vocabulary is smaller (32,000), but
with width 4,096 the two matrices contain 262,144,000 weights together.
There is no learned position table: RoPE acts on Q and K inside each
block. The [[#Shapes and parameters of the original models|full LLaMA count]]
adds attention projections, FFN and normalizations.

<a id="где-задаётся-нужный-ответ"></a>

### Where the answer comes from

In the [[#llama-relation-prefix|opening example]], we score three
candidate labels. In GPT-3 and base LLaMA one can prepend a solved
example such as `Premise: The door is open. Hypothesis: The door is shut. Relation: contradiction`.
The example takes up part of the window and changes the token states
without changing the weights. Natural Questions gives the model no
answer choices. It generates a continuation, which is compared with
the reference answers.

<a id="natural-questions-gpt-llama"></a>

<a id="как-менялась-оценка-на-natural-questions"></a>

### How Natural Questions was evaluated

In §3.8 of the GPT-2 paper, several question–answer pairs already
preceded the test question. The prompt asked for a short continuation;
the model did not receive a document containing the answer. The
largest GPT-2 scored 4.1% exact match. The paper does not break out
zero, one and multiple demonstrations, so 4.1% does not belong in
a zero-shot column as defined in the GPT-3 paper.
[Radford et al., §3.8](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=7).

GPT-3 and LLaMA varied the number of demonstrations. The figure from
the LLaMA paper shows one. The prompt includes an instruction, a
question answered by “Frank Sinatra,” and a new question ending in
`A:`. The `Target` line below the rule is the answer for scoring and
does not enter the prompt. With zero demonstrations, only the
instruction precedes the test question. In the 64-shot condition,
64 completed pairs appear there.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/llama1-figure3-natural-questions.png|Original Natural Questions prompt: the Frank Sinatra answer enters the prompt; Charles Darwin remains the reference answer.]]

*Hugo Touvron et al., [LLaMA v1, Figure 3, Appendix A, p. 17](https://arxiv.org/pdf/2302.13971v1#page=17).
Original left panel (Natural Questions), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The LLaMA authors generated greedily, extracted a short answer,
normalized case, punctuation, articles and whitespace, and compared
it against the accepted strings. For example, if the continuation
were `Charles Darwin, an English naturalist`, the text before the
comma would match the reference after normalization. This demonstrates
the scoring rule, not a generation run.

The column headings give the number of solved pairs before the test
question. Each cell reports exact match.

| Model | 0 | 1 | 64 |
| --- | ---: | ---: | ---: |
| GPT-3 175B | 14.6% | 23.0% | 29.9% |
| LLaMA 7B | 16.8% | 18.7% | 26.1% |
| LLaMA 13B | 20.1% | 23.4% | 31.9% |

LLaMA was evaluated on the 3,610-question Natural Questions test set
([Table 4](https://arxiv.org/pdf/2302.13971v1#page=4) and
[Appendix A](https://arxiv.org/html/2302.13971v1#A1)).
The GPT-3 row is from [Brown et al., Table 3.3](https://arxiv.org/html/2005.14165v4#S3.SS2).
GPT-3 gains 15.3 percentage points from zero to 64 demonstrations
with fixed weights. The published LLaMA 13B results are competitive
with GPT-3 175B on this task, with about 13.5 times fewer parameters.
These are results from two papers, not a matched-prompt run. They
cannot isolate the effect of RoPE, training tokens or corpus composition.
TriviaQA, on the right of the original Figure 3, differs in another
way: LLaMA used a filtered development set; the 71.2% GPT-3 few-shot
result was separately confirmed on the wiki-split test server.
Those numbers should not be treated as evaluations on identical questions.

<a id="gpt3-demo-order"></a>

<a id="что-меняет-порядок-примеров"></a>

### What changes when examples are reordered?

Zhao et al. evaluated GPT-3 2.7B on SST-2 sentiment classification.
Four labeled reviews preceded the test review. The researchers
permuted their order while holding the reviews, prompt template and
weights fixed. They recomputed accuracy on the same evaluation
examples for each permutation, and repeated this for ten sets
of demonstrations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/zhao21c-figure2.png|GPT-3 2.7B accuracy on SST-2 across permutations of four examples in each of ten demonstration sets.]]

*Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein and Sameer Singh,
[Calibrate Before Use, PMLR 139](https://proceedings.mlr.press/v139/zhao21c.html),
[Figure 2, p. 3](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original plot was extracted without changing axes or labels.*

The horizontal axis is the set index, not the number of examples.
Every set contains four. Dots within one column show the results
for different permutations; the box marks the interquartile range.
For the first set, the lowest and highest points lie near 54% and 93%.
A reproducible four-shot comparison therefore needs the examples,
their order and the prompt template. The plot measures GPT-3 on SST-2,
not the variation to expect from LLaMA.

<a id="gpt-llama-label-mapping"></a>

<a id="когда-примеры-задают-новое-соответствие-меток"></a>

### When examples define a new label mapping

A model that labels a review `positive` may already know what that
word means. A correct prediction alone does not show that it learned
a rule from the prompt. Pan et al. compare three labeling schemes to
separate the hint in the class names from the input–output pairs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/pan-icl-figure1.png|Pan et al.'s Random, Abstract and Gold prompts. The last column gives the expected label for each.]]

*Jane Pan, Tianyu Gao, Howard Chen and Danqi Chen,
[What In-Context Learning “Learns” In-Context, Figure 1, p. 8298](https://aclanthology.org/2023.findings-acl.527.pdf#page=1),
Findings of ACL 2023. Original artwork,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

In **Gold**, at the bottom, a favorable review gets `positive`.
In **Random**, each demonstration label is drawn uniformly from
`positive` and `negative`. The reference for the test review
is unchanged. This measures what a model can infer from the text and
class names when the demonstration labels do not provide a reliable
mapping.

**Abstract** keeps a consistent mapping within the prompt but replaces
class names with digits. Suppose `0` marks positive reviews and `1`
marks negative ones. The correct answer for a new favorable review
is `0`. Swap the digits in every demonstration, and the correct answer
becomes `1` while the review and model weights stay fixed. The authors
choose a new label permutation per test example and omit terms such
as `Sentiment` from the abstract templates. The middle row of the
illustration retains the task name so the prompts can be compared
visually. The labels are single tokens in these experiments, so the
evaluator can compare their logits after the prompt.
[Method: §§2–3 and Appendix B](https://aclanthology.org/2023.findings-acl.527.pdf#page=2).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/pan-icl-figure2.png|Six plots for GPT-3, LLaMA and OPT. Top: model size at 32 examples. Bottom: example count at fixed model size.]]

*Pan et al., [Figure 2, p. 8300](https://aclanthology.org/2023.findings-acl.527.pdf#page=3).
All six original panels and their legend,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

In the bottom row, the weights of each model are fixed. Moving from
8 to 32 demonstrations increases average **Abstract** accuracy from
about 45% to 60% for GPT-3 175B, and from about 48% to 57% for LLaMA-65B.
**Random** does not show that gain. The results average across 16
classification datasets; the error bars show variation across three
templates. GPT-3 and LLaMA were evaluated on different numbers of
examples, so compare the changes within a panel before comparing
heights across families. The **Abstract** improvement shows that more
pairs can help the model follow a supplied mapping. The plot does not
tell us which layers recover it or which architectural component
explains the difference between models.

<a id="training-and-inference-budget"></a>

<a id="бюджет-обучения-и-бюджет-вывода"></a>

### Training compute and inference compute

In 2022, [Hoffmann et al., §4](https://arxiv.org/html/2203.15556v1#S4)
reallocated Gopher's training budget. Gopher had 280 billion parameters
and processed 300 billion tokens. Chinchilla had 70 billion parameters
and processed 1.4 trillion tokens. Under the $6ND$ approximation, a
token costs about a quarter as much for 70B as for 280B. The rounded
counts give $5.0\cdot10^{23}$ and $5.9\cdot10^{23}$ FLOP, respectively.
These are rough estimates; the paper uses the more detailed accounting
in [Appendix F](https://arxiv.org/html/2203.15556v1#A6) to match the budgets.
On WikiText-103, perplexity fell from 7.75 to 7.16; on zero-shot LAMBADA,
accuracy rose from 74.5% to 77.4%. The authors also changed the optimizer,
tokenizer, and proportions of MassiveText (§4.1), so the comparison
does not isolate the effect of training tokens.

GPT-3 held training tokens fixed while varying model size. Hoffmann
et al. fixed training compute and looked for the model size and token
count with the lowest loss. LLaMA starts with a constraint on the size
of the model that will serve requests. For a 10B model, Hoffmann et al.
estimated about 200 billion training tokens at the training-only optimum.
On the [[#llama-pretraining-data|LLaMA training curves]], 7B loss was
still falling at one trillion tokens. These numbers come from different
models and corpora. They illustrate different training goals, not a
head-to-head quality comparison.

Sardana et al. formalize the serving decision. They set a target
loss $\ell$ and an expected workload $T$, the total number of input
and output tokens over the model's lifetime. Among models that reach
$L(N,D)=\ell$, they compare $6ND+2NT$, where $N$ is the parameter count
and $D$ is the pre-training token count. A larger model may reach the
same loss with less training compute, but each request erodes that lead.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-scaling-2026-11-09/sardana24a-figure1.png|Three panels compare 7B and 13B at equal loss: training cost, inference cost, and total cost for two trillion served tokens.]]

*Nikhil Sardana, Jacob Portes, Sasha Doubov and Jonathan Frankle,
[Beyond Chinchilla-Optimal, PMLR 235 (2024)](https://proceedings.mlr.press/v235/sardana24a.html),
[Figure 1, p. 2](https://raw.githubusercontent.com/mlresearch/v235/main/assets/sardana24a/sardana24a.pdf#page=2).
All three original panels, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The horizontal dashed line in panel **(a)** sets a common loss target.
13B reaches it at $4.5\cdot10^{22}$ FLOP and 7B at $5.2\cdot10^{22}$ FLOP.
Before the first request, 7B has spent an extra $0.7\cdot10^{22}$ FLOP.
Panel **(b)** shows how serving can make up the difference. Under
the paper's approximation, a forward pass costs $2N$ FLOP per token
for a model with $N$ parameters. On each input or output token, 7B
saves $2(13-7)\cdot10^9=12\cdot10^9$ FLOP. The rounded numbers on
the figure therefore give a break-even point near 580 billion tokens:

$$
T_*=\frac{0.7\cdot10^{22}}{12\cdot10^9}\approx5.8\cdot10^{11}.
$$

At two trillion served tokens, 7B spends $2.8\cdot10^{22}$ FLOP
on inference and 13B spends $5.2\cdot10^{22}$. Adding training gives
$8.0\cdot10^{22}$ and $9.7\cdot10^{22}$ FLOP, the totals in panel
**(c)**. This is an equal-loss comparison, not a measurement of two
released LLaMA models. At 100 billion served tokens, the same calculation
gives $5.34\cdot10^{22}$ FLOP for 7B and $4.76\cdot10^{22}$ for 13B,
making 13B cheaper. Equal loss does not guarantee equal quality
on a particular task. Comparing costs in money also requires input
and output lengths, throughput on the chosen hardware and resource
prices ([§2](https://arxiv.org/html/2401.00448v3#S2),
[§6](https://arxiv.org/html/2401.00448v3#S6)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-scaling-2026-11-09/sardana24a-figure2b.png|Contour plot with inference tokens on the horizontal axis and target loss on the vertical axis. Each line gives a percentage of the parameter count optimal for training alone.]]

*Nikhil Sardana et al., [Beyond Chinchilla-Optimal, Figure 2(b), p. 3](https://raw.githubusercontent.com/mlresearch/v235/main/assets/sardana24a/sardana24a.pdf#page=3).
Original panel with axes and legend, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Hold the target loss at 2.2 and follow the horizontal line to the right.
As demand grows, it crosses the contours labeled 75% and 50%: the
calculation selects a smaller fraction of the model size that minimizes
training compute alone. The percentages describe parameter counts,
not FLOP savings. The authors computed the curves from a fitted loss
function. Their long-training runs covered models from 150M to 6B
without repeating data ([§3–4](https://arxiv.org/html/2401.00448v3#S3)).
The [[00 Учебник/11 Pre-training и Scaling/43 Scaling laws#joint-model-data-scaling|scaling laws chapter]]
shows how to estimate $L(N,D)$ in the first place.

<span id="llama-1-llama-2-и-llama-3--не-одна-конфигурация"></span>

<a id="переходы-к-llama-2-и-llama-3"></a>

<a id="llama-generations"></a>

## What changed in Llama 2 and Llama 3

Llama 3 8B has about 1.3 billion more weights than Llama 2 7B,
although each layer computes a quarter as many key and value
coordinates. Both models have 32 layers, width 4,096, causal
attention, and the same RMSNorm → sublayer → residual addition order.
The difference comes from three places: attention projections,
the FFN, and the two vocabulary matrices.

The last column counts query heads and K/V heads separately.

| Model | Context | Q / KV |
| --- | ---: | ---: |
| LLaMA v1 7B | 2,048 | 32 / 32 |
| Llama 2 7B | 4,096 | 32 / 32 |
| Llama 3 8B | 8,192 | 32 / 8 |
| Llama 3.1 8B | 131,072 | 32 / 8 |

For both 7B releases, each SwiGLU branch has width 11,008 and the
vocabulary has 32,000 rows. For both 8B releases the counts are
14,336 and 128,256.

See [Llama 2, Table 1](https://arxiv.org/html/2307.09288v2#S2.T1)
and [Llama 3 Herd, §3.2](https://arxiv.org/html/2407.21783v3#S3.SS2).
The table compares the smallest releases; larger variants have
different head counts. The 128,256 rows include the 256 slots
that the [Meta tokenizer](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/tokenizer.py)
reserves in addition to 128,000 BPE entries.

<a id="llama-v1--llama-2-длиннее-последовательности-и-gqa-у-70b"></a>

### LLaMA v1 → Llama 2: longer sequences and GQA at 70B

The 7B and 13B models keep the full set of K/V heads and the
same SentencePiece tokenizer. Their context grows from 2,048 to
4,096 tokens. Pretraining also gets longer: from one to two
trillion tokens at 7B and 13B, and from 1.4 trillion at LLaMA-65B
to two trillion at Llama 2 70B. Both quantities count tokens,
but one limits a single request and the other counts what the
model processed during all of pretraining.

Llama 2 70B uses GQA. Table 1 also lists a 34B GQA model, but its
weights did not ship in the original release. Of the 7B, 13B,
and 70B models released then, only 70B shares keys and values.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png|In Ainslie et al.'s diagram, several Q heads in a group share K and V; with MHA each Q head has its own K and V.]]

*Ainslie et al., [GQA, Figure 2](https://aclanthology.org/2023.emnlp-main.298/),
[original PNG](https://arxiv.org/html/2305.13245v3/images/gmq_architecture.png),
CC BY 4.0. In MHA each query head has its own K/V pair.
In GQA, several query heads share a pair. The head counts in
the diagram are illustrative.*

Why keep more than one K/V pair per layer? In
[Meta's experiment, Appendix A.2.1](https://arxiv.org/html/2307.09288v2#A1.SS2.SSS1),
a 30B model split attention across eight A100s. With GQA each GPU
held one of eight distinct K/V heads. With MQA each GPU held a copy
of the same head. Both runs stored one head per position on each
GPU, so MQA had no cache-size advantage in that setup.
In [Figure 24 of the Llama 2 report](https://arxiv.org/html/2307.09288v2#A1.F24),
GQA and MQA have similar results, and both fit larger batches than
MHA. The experiment used the 30B ablation models, not the released
70B model. Meta also widened the FFN in the GQA and MQA quality
ablations to compensate for their smaller attention projections.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gqa-2026-10-22/groups-speed.png|In T5-XXL, time per sample at eight groups is close to MQA; at 64 groups it reaches the MHA line.]]

*Joshua Ainslie et al., [GQA, Figure 6, p. 4898](https://aclanthology.org/2023.emnlp-main.298.pdf#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original plot and caption, without redrawing.*

Ainslie et al. ran a separate experiment on T5-XXL. They varied the
number of groups, used 2,048 input tokens, and generated 512.
The blue squares at one and eight groups are almost level. At
64 groups, one per query head, the point reaches the MHA line.
The reduction in cache size does not translate directly into a
speedup, and the T5 timings are not measurements of Llama.

<a id="llama-2-7b--llama-3-8b-меньше-kv-больше-весов"></a>

### Llama 2 7B → Llama 3 8B: fewer K/V heads, more weights

The April Llama 3 models use GQA at both 8B and 70B. In 8B,
the first four query heads read the first K/V pair, the next four
read the second pair, and so on. All 32 queries are still computed
separately. At head width 128, the output of each K and V projection
shrinks from $32\cdot128=4096$ to $8\cdot128=1024$ coordinates.
The Q and O projection shapes stay the same. This removes
$2\cdot4096\cdot(4096-1024)=25\,165\,824$ weights per layer.
At the same time, each of the three SwiGLU matrices grows from
11,008 to 14,336 intermediate coordinates. The FFN adds
$3\cdot4096\cdot(14336-11008)=40\,894\,464$ weights per layer.

| Source of the difference | Change in weights across the model |
| --- | ---: |
| K and V projections: 32 layers | −805,306,368 |
| Three SwiGLU matrices: 32 layers | +1,308,622,848 |
| Input embedding and vocabulary projection | +788,529,152 |
| Total | +1,291,845,632 |

The last addition is $2\cdot4096\cdot(128256-32000)$.
The input embedding and vocabulary projection are untied, so
both matrices gain rows. The totals are 6,738,415,616 and
8,030,261,248 weights, matching the published weight counts in the
[Llama 2 7B](https://huggingface.co/api/models/meta-llama/Llama-2-7b-hf/revision/01c7f73d771dfac7d292323805ebc428287df4f9)
and [Llama 3 8B](https://huggingface.co/api/models/meta-llama/Meta-Llama-3-8B/revision/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920)
metadata. Llama 2's metadata includes another 2,048 F32 values.
The [conversion code](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/convert_llama_weights_to_hf.py#L197)
shows where that count comes from: 64 RoPE inverse frequencies
are saved per layer. The frequencies are computed, not trained.

<a id="llama-gqa-cache-step"></a>

#### One decoding step in Llama 3 8B

Suppose four positions are already cached and the model receives a
fifth token. Use one sequence and one layer, without sharding.
In [Meta's code](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L153-L190),
the projections produce Q of shape `[1, 1, 32, 128]` and K and V of
shape `[1, 1, 8, 128]`. The axes are batch, position, head, and head
dimension. The code applies RoPE to Q and K and writes the new K and
V at position 4. The occupied portion of each cache now has shape
`[1, 5, 8, 128]`.

Only after the write does `repeat_kv` expand the K and V slices to
`[1, 5, 32, 128]` for matrix multiplication. Query heads 0 and 3
compare different queries with the same set of keys from group 0.
Head 4 reads group 1. Every head computes its own five scores and
weighted sum, even when its keys and values are shared. The expanded
tensors are not written back: the sixth token adds eight heads each
for K and V again. A kernel with native GQA support can address the
group without explicitly repeating the input tensors.

The cache contains neither the FFN nor the vocabulary matrices.
One position across all 32 layers needs $2\cdot32\cdot h_{KV}\cdot128$
values. At two bytes per value, that is 512 KiB per token in
Llama 2 7B and 128 KiB in Llama 3 8B. Filling the original context
takes **2 GiB at 4,096 tokens** and **1 GiB at 8,192 tokens**,
respectively. These are K/V payloads for one sequence, excluding
weights and runtime overhead. The [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|GQA chapter]]
derives the same calculation for arbitrary configurations.

Llama 3 also changes the input IDs: SentencePiece and the new BPE
have different vocabularies and segmentation rules. The same ID
can denote a different piece of text in each vocabulary. The
[April model card](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/MODEL_CARD.md)
reports over 15 trillion pretraining tokens, versus two trillion
for Llama 2. Comparing the finished models cannot isolate the
effect of GQA, the vocabulary, or the training mix.

<a id="llama31-long-context"></a>

<a id="llama-3--31-что-понадобилось-для-окна-128k"></a>

### Llama 3 → 3.1: what it took to reach 128K

In the July release, the smaller model's matrix sizes stay the
same, but the context is 16 times longer. At 131,072 tokens in
Llama 3.1 8B, the same 128 KiB per position requires **16 GiB**
of cache even with GQA. Filling the window also costs compute.
In one layer, each query head compares the first position with
one key, the second with two, and so on. That gives 33,558,528
allowed pairs at 8,192 positions and 8,590,000,128 at 131,072.
There are almost 256 times as many pairs per head; that is not
a wall-clock measurement. When generating a single next token,
only the new attention row is computed, so the pair count for
that step grows linearly.

April's Llama 3 already used a RoPE base of 500,000. The July release
kept that base but also
[[00 Учебник/07 Анатомия современной LLM/04 RoPE#rope-llama31-scaling|slows the rotation of some coordinate pairs]].
This changes the angles applied to queries and keys without adding
rows to any trained matrix. Changing the frequencies alone does not
show whether the model can use a key tens of thousands of positions away.

In the [July Meta report, §3.4.2](https://arxiv.org/pdf/2407.21783v1#page=14),
the team increased the window in stages and continued pretraining
at each length. Before moving on, they checked that performance had
recovered on short tasks and that the model could retrieve a fact
inserted in a long document. For 405B, they report six stages from
8K to 128K and about 800 billion tokens for this phase; they do not
report the same budget for 8B. In [Table 21](https://arxiv.org/html/2407.21783v3#S5.T21),
8B has an average multi-needle recall of 98.8 and an En.QA F1 of 27.1
on questions about novels. They are different tasks and metrics:
98.8 does not tell us how reliably the model answers questions about
the whole long text.

Retrieving an inserted string and answering a question over a
long set of documents are different tests. In a separate
experiment, the Princeton team continued pretraining April
Llama-3-8B-Base on long sequences. At several checkpoints they
evaluated the checkpoint itself and a copy fine-tuned on UltraChat.
This compares models at the same continued-pretraining token count.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-generations-2026-11-10/gao25-figure1-sft.png|RAG and re-ranking scores before and after SFT at checkpoints spaced by billions of continued-pretraining tokens.]]

*Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen,
[How to Train Long-Context Language Models (Effectively), Figure 1, p. 7377](https://aclanthology.org/2025.acl-long.366.pdf#page=2).
Both panels and the original legend, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The left panel scores answers using retrieved documents.
The **Before SFT** curve barely moves from one to five billion
tokens. On **After SFT**, the score rises from about 49 to 52.
The right panel asks the model to re-rank documents. Its curves
are closer and vary more. The two panels have different scales.
The shaded regions summarize variation across two runs.
The figure illustrates how to evaluate
intermediate checkpoints; it does not measure the gain of Meta's
Llama 3.1 over Llama 3. See the
[[05 Источники/Papers/Gao et al. — How to Train Long-Context Language Models|ProLong source note]]
for the protocol and other experiments.

<a id="llama31-long-short-mix"></a>

#### Why keep short texts in the mix?

In [the next ProLong experiment, §3.2](https://aclanthology.org/2025.acl-long.366.pdf#page=4),
each model saw five billion more tokens with a maximum sequence
length of 64K. The team varied the fraction drawn from books
and repositories; the rest came from ShortMix. At 60%, three
billion tokens came from the long-data mix and two billion from
ShortMix. At 100%, all five billion came from the long-data mix.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-generations-2026-11-10/gao25-figure2-data-mix.png|Six panels plot the share of long data. After SFT, the long-task average peaks at 60% and drops at 100%.]]

*Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen,
[How to Train Long-Context Language Models (Effectively), Figure 2, p. 7379](https://aclanthology.org/2025.acl-long.366.pdf#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All six panels, the legend, and the original caption.*

In the fifth panel, the star marks 60% long data, where the
post-SFT long-task average is about 55. At 100% it falls to about
49. Compare the two RAG curves: moving from 60% to 100% raises
the score before SFT and lowers it after SFT. Choosing a checkpoint
from the blue RAG curve alone can favor a mix that does worse in the
finished model. In the rightmost panel, the pre-SFT short-task
average also falls as the long-data share rises. The 60/40 split
is a result from this experiment on Llama-3-8B-Base, not Meta's recipe.

<a id="llama-base-chat-instruct"></a>

<a id="base-chat-и-instruct-что-получает-декодер"></a>

### Base, Chat, and Instruct: what the decoder sees

A base model can take
`Premise: A cat sleeps. Hypothesis: An animal rests. Relation:`
and score possible continuations. Adding solved examples changes
the input without changing the weights. Chat and Instruct are
different checkpoints: their weights were updated on dialogue
and preference data. At inference time the conversation still
becomes one token sequence.

Here is the same turn without a system message. In Llama 2 Chat,
`[INST]` and `[/INST]` enclose the user text. The reply begins
after the second marker.

```text title="Llama 2 Chat"
<s>[INST] Does “An animal rests” follow from “A cat sleeps”? [/INST]
```

In Llama 3 Instruct, `<|eot_id|>` ends the message. The encoder
adds an `assistant` header and two newlines. The reply begins
immediately after them.

```text title="Llama 3 Instruct"
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Does “An animal rests” follow from “A cat sleeps”?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Here `<s>` and the angled markers stand for control tokens inserted
by the encoder, not arbitrary user text. In
[Llama 2](https://github.com/meta-llama/llama/blob/c7c0a30d686b0743352692e0f7736ea1701591f4/llama/generation.py#L259)
the wrapper adds BOS and tokenizes the string with `[INST]`; in
[Llama 3](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/tokenizer.py#L206)
it assembles headers from special IDs and the role tokens.
Llama 3 generation stops on either the end-of-turn `<|eot_id|>`
or the end-of-text `<|end_of_text|>` token. Using a template from
a different release changes the input IDs; it does not switch
the decoder into a separate chat mode.

In [Llama 2, §3](https://arxiv.org/html/2307.09288v2#S3) the authors
describe SFT followed by iterations using selected responses
and PPO policy updates. In the
[July Llama 3 report, §4.1 and Figure 7](https://arxiv.org/html/2407.21783v3#S4.SS1),
a reward model still selects generated responses for SFT. The
subsequent preference stage uses [[00 Учебник/12 Post-training и Alignment/05 DPO|DPO]]
instead of PPO. That is a
change in how the weights are updated, separate from both GQA
and the chat template.

<span id="литература-и-дополнительные-материалы"></span>
<span id="основные-статьи"></span>
<span id="код-и-хорошие-разборы"></span>
<span id="внутри-базы"></span>

<a id="источники-и-дальнейшее-чтение"></a>

## Sources and further reading

- [Touvron et al., LLaMA v1](https://arxiv.org/abs/2302.13971v1): §1 on inference costs; §2 on data, architecture and training; §§3–4 on evaluation and the separate instruction-tuning experiment.
- [[05 Источники/Papers/Pan et al. 2023 — What in-context learning learns|Pan et al., What In-Context Learning “Learns” In-Context]]: §§2–4 and Appendix B on testing label mappings in GPT-3, LLaMA and OPT.
- [Sardana et al., Beyond Chinchilla-Optimal](https://proceedings.mlr.press/v235/sardana24a.html): Figure 1 and §2 on costs at matched loss; §6 on monetary costs.
- [Original LLaMA implementation, February 2023](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py): projection shapes, norms, RoPE, FFN rounding, separate input and output embeddings, and the inference-only forward method.
- [[05 Источники/Papers/Gulordava et al. 2018 — Tied language model architectures|Gulordava, Aina and Boleda, EMNLP 2018]]: Figure 1 and §2 compare tied and separate vocabulary matrices.
- [Stanford CS336, Assignment 1, version 26.0.3](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf): §§3.1, 3.4–3.5 build a decoder-only Transformer; Figures 1–3 and the excerpt above are from the MIT-licensed assignment.
- [Zhang and Sennrich, RMSNorm](https://arxiv.org/abs/1910.07467): §§3–4 on the normalization and its properties.
- [Su et al., RoFormer v5](https://arxiv.org/abs/2104.09864v5): §§3.1–3.2 and 3.4.2–3.4.3 on coordinate pairs, relative offsets and computing rotations.
- [Shazeer, GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1): §§2–3 compare the three-matrix FFN and alternatives at matched compute.
- [Touvron et al., Llama 2](https://arxiv.org/abs/2307.09288v1): §2 and Table 1; [pinned implementation](https://github.com/meta-llama/llama/blob/c7c0a30d686b0743352692e0f7736ea1701591f4/llama/model.py).
- [Meta's original Llama 3 model card](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/MODEL_CARD.md) and [implementation](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783v1), §§3.2 and 3.4.
- [Sebastian Raschka, Converting GPT to Llama](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama): notebooks on the steps from GPT to Llama 2 and Llama 3.
- [[02 Areas/ML & DL/02 Атлас моделей/Семейства/Llama|Llama in the model atlas]]; [[02 Areas/ML & DL/01 Справочник/Позиционные представления/RoPE|RoPE in the reference]]; [[02 Areas/ML & DL/01 Справочник/FFN и MoE/SwiGLU|SwiGLU in the reference]].
- [[02 Areas/ML & DL/Papers/LLaMA|Detailed notes on LLaMA v1 and the evaluation setup]].

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3]].
**Next:** [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block|The modern decoder block]].
