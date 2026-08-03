---
title: Three Transformer architectural patterns
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources:
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/1910.10683
  - https://arxiv.org/abs/1905.03197
---

# Encoder-only, decoder-only, and encoder–decoder

The names *encoder* and *decoder* are often reduced to a misleading slogan:
the first “understands” text and the second “generates” it. A decoder also
builds meaningful representations, and an encoder can reconstruct missing
content. The real distinction is which positions attention connects, which
objective was optimized during training, and which data are available at
inference.

All three families use the same computational machinery. An encoder lets each
position use the complete known input. A causal decoder hides future positions
and learns to continue a prefix. An encoder–decoder reads a source sequence in
full and then generates a new sequence while accessing stored source
representations through cross-attention. Following one sentence through all
three keeps masks, losses, and inference in one explanation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt-2-transformer-xl-bert-3.png]]

*From [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/):
BERT represents known text; GPT continues a prefix. This is a useful first map;
the exact distinction comes from attention masks and training objectives.*

## One example for the whole chapter

Consider the sentence:

```text
The cat did not eat the food because it was spoiled.
```

Now consider three tasks:

1. classify the sentiment or resolve what `it` refers to;
2. continue the text after `The cat did not`;
3. translate the sentence into another language.

They use the same text but require different access to context. Classification
may use words on both sides immediately. Next-token generation must not inspect
an unseen continuation. Translation knows the entire source while the target
appears left to right. These constraints produce the three patterns.

## The main switch: the visibility matrix

Before softmax, attention computes pairwise scores

$$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
A=\operatorname{softmax}(S+M).
$$

If an edge is allowed, $M_{ij}=0$; if query $i$ may not read key $j$,
$M_{ij}=-\infty$, and softmax assigns zero weight. For four tokens:

```text
FULL / BIDIRECTIONAL        CAUSAL                 CROSS-ATTENTION

      K1 K2 K3 K4              K1 K2 K3 K4              source K1 K2 K3
 Q1   ✓  ✓  ✓  ✓          Q1   ✓  ·  ·  ·          target Q1  ✓  ✓  ✓
 Q2   ✓  ✓  ✓  ✓          Q2   ✓  ✓  ·  ·                 Q2  ✓  ✓  ✓
 Q3   ✓  ✓  ✓  ✓          Q3   ✓  ✓  ✓  ·                 Q3  ✓  ✓  ✓
 Q4   ✓  ✓  ✓  ✓          Q4   ✓  ✓  ✓  ✓
```

This is not a metaphor. It is the actual constraint inside attention, and a
first approximation to identifying an architectural family.

## Encoder-only: build a contextual representation

An encoder receives the complete input at once. Every position can read every
other position:

$$
M_{ij}=0\quad\forall i,j.
$$

A token representation therefore depends on both directions. In the example,
`it` can use both `food` and `spoiled` in one forward pass.

### How BERT learns without task labels

BERT hides selected input tokens and reconstructs them:

```text
The cat did not eat the [MASK] because it was spoiled.
                         ↓
                       “food”
```

Probability is predicted only at masked positions:

$$
\mathcal L_{\text{MLM}}
=-\sum_{i\in\mathcal M}\log p_\theta(x_i\mid x_{\setminus\mathcal M}).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png]]

*Masked language modeling from [The Illustrated
BERT](https://jalammar.github.io/illustrated-bert/). The hidden token is
reconstructed from left and right context.*

*Bidirectional* does not mean two separate passes as in a bidirectional RNN.
One self-attention layer connects a position to both sides simultaneously.

### What comes out of an encoder

The encoder returns one vector per position:

$$
H=(h_1,\ldots,h_n),\qquad H\in\mathbb R^{n\times d}.
$$

These vectors can feed `[CLS]` or pooled text classification, token-level NER,
bi-encoder retrieval, joint query–document cross-encoder reranking, and
start/end heads for extractive QA.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-tasks.png]]

*Devlin et al., [BERT](https://arxiv.org/abs/1810.04805): one pre-trained model
with small task-specific heads. The encoder's architectural role is to produce
representations of known input, not “to classify.”*

### Why ordinary generation is awkward for an encoder

During MLM, the model saw `[MASK]` and right context. In open generation, right
context does not yet exist and literal mask tokens do not occur naturally. An
encoder can fill blanks iteratively, but that is not the simple autoregressive
process it was trained to perform.

:::caution[Common mistake]
“BERT cannot generate because it has no decoder” is imprecise. The central
problem is the mismatch between objective and inference procedure, not the
block's name.
:::

## Decoder-only: predict the next token

A decoder-only model receives a prefix and predicts the next token at every
position:

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

To prevent copying from the answer, it uses a causal mask:

$$
M_{ij}=\begin{cases}0,&j\le i,\\-\infty,&j>i.\end{cases}
$$

For `The | cat | did | not | eat`, one forward pass trains:

```text
The                 → cat
The cat             → did
The cat did         → not
The cat did not     → eat
```

Targets are shifted by one position, so all training predictions run in
parallel. At inference there are no future targets, and the model appends one
token at a time.

### Why next-token prediction can produce a general model

Almost any task can be serialized as text:

```text
Text: ...
Sentiment: positive

Question: ...
Answer: ...

Translate into English:
...
```

The next-token objective is shared by code, dialogue, documents, and labeled
examples. This is one reason GPT, Llama, Qwen, and DeepSeek use decoder-only
architectures. The objective itself only models continuations from the training
distribution. Reliable instruction following, safety refusal, and preferred
style are shaped later through SFT and preference or reinforcement learning;
they do not follow from the causal mask alone.

### Minimal causal attention

```python
import torch
import torch.nn.functional as F

def causal_attention(q, k, v):
    # q, k, v: [batch, heads, tokens, head_dim]
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    tokens = q.size(-2)
    forbidden = torch.triu(
        torch.ones(tokens, tokens, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    scores = scores.masked_fill(forbidden, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v
```

In [nanoGPT's `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py),
the same idea appears as `scaled_dot_product_attention(..., is_causal=True)`.
Matching that flag to the triangular matrix makes “decoder-only” operational
rather than terminological.

## Encoder–decoder: read one text, write another

For translation, summarization, and speech recognition, the source is fully
known while the output must be generated. Encoder–decoder separates those
roles.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/encoder-decoder-architecture.png]]

*Raffel et al., [T5](https://arxiv.org/abs/1910.10683): bidirectional encoder on
the left, causal decoder on the right. Cross-attention connects generated
output to source representations.*

### One translation step

Let the source be `The cat did not eat the food.` The encoder computes once:

$$
H_{\text{src}}=\operatorname{Encoder}(x_{1:n}).
$$

Suppose the decoder has generated `Die Katze hat`. In each decoder layer:

1. causal self-attention connects the current token to the generated prefix;
2. cross-attention uses decoder state as query and encoder states as keys and
   values;
3. the FFN transforms the resulting representation.

For cross-attention,

$$
Q=H_{\text{dec}}W_Q,\qquad
K=H_{\text{src}}W_K,\qquad
V=H_{\text{src}}W_V.
$$

The query comes from the translation being built; memory comes from the source.
The model can attend to the negation while choosing the negative construction
and to `eat` while selecting the corresponding verb.

### Teacher forcing

During training, the decoder receives the correct output shifted by one token:

```text
decoder input:  <BOS> Die Katze hat nicht
targets:               Die Katze hat nicht gegessen
```

$$
\mathcal L_{\text{seq2seq}}
=-\sum_t\log p_\theta(y_t\mid y_{<t},x).
$$

As in decoder-only training, a causal mask hides the current target. The
difference is the additional conditional memory $x$.

### T5: every task as text-to-text

T5 adds a textual task prefix:

```text
translate English to German: That is good.
summarize: long document...
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/text-to-text-framework.png]]

*The text-to-text framework from Raffel et al.,
[T5](https://arxiv.org/abs/1910.10683). A label is also text, but source and
target are handled by different sides of the architecture.*

## Why the word “decoder” is confusing

In the original Transformer, a decoder layer contains causal self-attention,
cross-attention to encoder output, and a position-wise FFN. GPT is called
decoder-only even though it has no separate encoder and no cross-attention.
Historically, it is the generative half of the Transformer with the connection
to a source encoder removed.

Ask these questions instead of merely asking whether a “decoder” exists:

- Is self-attention full or causal?
- Is there separate source memory?
- Is there cross-attention?
- Which objective was used?

## One stack can combine all three masks

UniLM demonstrated that one Transformer stack can act bidirectionally,
unidirectionally, or sequence-to-sequence by changing the attention mask. In
seq2seq mode, source tokens see one another, while target tokens see the source
and preceding target tokens.

This is a useful thought experiment: much of the difference among families is
encoded not in a Python class name but in the graph of allowed connections.

## What to choose in practice

| Task | Reasonable starting point | Why |
|---|---|---|
| classification, NER | encoder | complete context in one pass |
| dense retrieval | encoder / dual encoder | documents can be encoded in advance |
| precise pair reranking | encoder / cross-encoder | all query–document interactions are available |
| open chat, code, continuation | decoder-only | objective matches generation |
| translation, summarization | encoder–decoder or decoder-only | explicit source/target split or a single LM interface |
| speech-to-text | encoder–decoder | audio encoder + text decoder |
| VLM | often encoder + decoder | vision encoder supplies visual tokens; LLM generates |

Decoder-only became the most general interface but did not eliminate the other
families. For repeated retrieval over a fixed collection, an encoder is often
cheaper because document embeddings are computed once. When output is tightly
conditioned on a large source, encoder–decoder provides an explicit boundary
between reading and generation.

## Comparison without slogans

| Property | Encoder-only | Decoder-only | Encoder–decoder |
|---|---|---|---|
| self-attention | full | causal | full in encoder, causal in decoder |
| cross-attention | no | usually no | yes |
| typical objective | MLM / contrastive | next token | conditional next token |
| output | representations / scores | autoregressive tokens | conditional tokens |
| source prefill | one pass | part of common prefix | separate encoder pass |
| natural strength | known-input representation | general generation | source → target |
| examples | BERT, DeBERTa, E5 | GPT, Llama, Qwen, DeepSeek | T5, BART, Whisper |

## Mistakes worth learning to recognize

:::danger[“The encoder understands; the decoder generates”]
This is a mnemonic, not a definition. Decoder states are representations too,
and an encoder can participate in iterative generation. Masks, objectives, and
inference procedures define the exact difference.
:::

:::danger[“A bidirectional model sees the answer”]
In MLM, the correct token is replaced or hidden. The model sees right context,
but not the target at the same position.
:::

:::danger[“Decoder-only training is sequential”]
Generation is sequential; training runs all positions in parallel under the
causal mask.
:::

:::danger[“Every embedding model is BERT”]
An encoder may be trained with MLM, contrastive loss, distillation, or a
supervised retrieval objective. Architecture and training are separate axes.
:::

## Practice without a large GPU

### 1. Draw the masks

For sequence length six, construct a full mask, a lower-triangular causal mask,
and a UniLM-style mask with three source and three target tokens. Verify that
every row answers: “which keys may this query read?”

### 2. Compare representations

Take a ready-made BERT encoder and a causal LM. Feed sentences containing an
ambiguous word in different contexts. Compare representations with and without
context. The goal is not a benchmark, but direct evidence that a “word vector”
now depends on its sentence.

### 3. Read real code

In nanoGPT, find the QKV projection, causal mask or `is_causal=True`, and the
logit/target shift in the loss. Then open T5 and locate separate self-attention
and encoder–decoder attention. A distinction becomes operational when it can be
pointed to in code.

## Understanding check

After this chapter, you should be able to explain without memorized slogans:

- why BERT uses right context without copying its target;
- why GPT trains in parallel but generates sequentially;
- where Q, K, and V come from in cross-attention;
- why changing a mask can change the mode of the same stack;
- why decoder-only is a strong default for a general LLM but not a universally
  optimal architecture.

## Literature and additional material

### Primary papers

- [Devlin et al., BERT](https://arxiv.org/abs/1810.04805)
- [Raffel et al., T5](https://arxiv.org/abs/1910.10683)
- [Dong et al., UniLM](https://arxiv.org/abs/1905.03197)
- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Explanations and code

- [Jay Alammar, The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Lena Voita, Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)
- [Andrej Karpathy, nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py)

### Related Bookvar notes

- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-only]]
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Decoder-only]]
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder]]

**Next:** [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|a decoder-only block in LLaMA: RMSNorm, RoPE, and SwiGLU.]]
