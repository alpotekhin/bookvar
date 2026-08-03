---
title: Building the Transformer — from the original block to BERT, GPT, and LLaMA
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/2302.13971
---

# Building the Transformer: from the original block to BERT, GPT, and LLaMA

Attention performs only one part of the computation: it transfers information
between positions. After that exchange, every position must still transform the
features it received, and a deep network must carry a signal across many layers
without destroying the original representation. A complete Transformer block
therefore combines attention, a position-wise feed-forward network, residual
connections, and normalization.

The ordering of these components has changed over time. The original
Transformer was an encoder–decoder with normalization after residual addition.
BERT retained the encoder; GPT retained a causal decoder without
cross-attention; LLaMA changed normalization, position handling, and the FFN.
The 2017 architecture is the common starting point, while BERT, GPT, and LLaMA
are defined by specific changes to its blocks and connections.

## 1. The central decomposition

Every layer performs two different operations:

1. **Attention:** a token receives information from other tokens.
2. **FFN:** a small neural network processes the resulting vector.

In one sentence:

> Attention moves information between tokens; the FFN transforms each token's
> features independently.

With only an FFN, tokens cannot use context. With only attention, the model
lacks a sufficiently expressive nonlinear transformation of collected
information.

## 2. The residual stream

It is useful to view the Transformer not as a tower of boxes but as a stream
$x\in\mathbb{R}^{B\times T\times d}$ to which sublayers add updates:

$$
x\leftarrow x+\operatorname{Attention}(\operatorname{Norm}(x)),
$$

$$
x\leftarrow x+\operatorname{FFN}(\operatorname{Norm}(x)).
$$

This is the modern **pre-norm** form. The residual path preserves the interface
between layers: each sublayer consumes and returns `B × T × d_model`.

The original Transformer used **post-norm**:

$$
x\leftarrow\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
$$

A pre-norm LLaMA block should not be drawn and labeled “the original
Transformer.” Operation order affects optimization and is an architectural
difference.

## 3. Feed-forward network

The original FFN is

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2.
$$

The same FFN is applied independently to every position. Transformer Base uses
$d_{model}=512$, hidden width $d_{ff}=2048$, and ReLU. The FFN expands the
vector, applies a nonlinearity, and compresses it back. In modern LLMs this
sublayer often contains a large fraction of all parameters. LLaMA replaces the
ordinary ReLU MLP with gated SwiGLU:

$$
\operatorname{SwiGLU}(x)=
\big(\operatorname{SiLU}(xW_g)\odot xW_u\big)W_d.
$$

## 4. Positional information

Without position information, self-attention cannot know which token is first:
permuting input rows merely permutes output rows in the same way. The model
therefore needs a positional signal.

The original Transformer added sinusoidal encodings:

$$
PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d}\right),
$$

$$
PE_{(pos,2i+1)}=\cos\left(pos/10000^{2i/d}\right).
$$

This is not a universal Transformer property:

- BERT and early GPT models use learned absolute position embeddings;
- LLaMA applies RoPE to Q and K inside each layer;
- other families use relative bias, ALiBi, and related methods.

## 5. The original encoder block

The encoder receives token and position representations. Each of its $N$
blocks contains full multi-head self-attention, residual addition plus
LayerNorm, a position-wise FFN, and another residual addition plus LayerNorm.

Full attention means that every non-padding input position may read every other
input position. The encoder returns contextual representations of the complete
input.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/Transformer_encoder.png]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): an encoder
as self-attention and FFN connected by a residual path. Trace one block here
before moving to the complete stack.*

## 6. The original decoder block

The decoder has three sublayers:

1. masked self-attention over the known target prefix;
2. cross-attention to encoder output;
3. an FFN.

In cross-attention,

$$
Q=\text{decoder states},\qquad K,V=\text{encoder outputs}.
$$

This develops Bahdanau attention: the decoder reads a memory of the source
sequence. The compatibility function, multi-head organization, and absence of
a recurrent decoder state are different.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/The_transformer_encoder_decoder_stack.png]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): the expanded
encoder–decoder stack. Encoder outputs serve as K and V in cross-attention at
every decoder block.*

## 7. Training and generation

During training, the target sequence is shifted:

```text
decoder input:  <BOS> The black cat sleeps
labels:               The black cat sleeps <EOS>
```

The causal mask prevents position $t$ from using labels on its right. All
positions can nevertheless be processed in one tensor operation.

During generation, the next token is unknown. At each step the model computes
next-token logits, selects or samples a token, appends it to the prefix, and
continues. A KV cache stores K/V for earlier positions rather than recomputing
them. Dependence of token $t+1$ on the selected token $t$ remains, so
autoregressive generation is sequential in time.

## 8. Three architectural branches

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt-2-transformer-xl-bert-3.png]]

*Jay Alammar, [The Illustrated
GPT-2](https://jalammar.github.io/illustrated-gpt2/): decoder-only GPT-2,
encoder-only BERT, and the recurrent extension Transformer-XL.*

From GPT-2 to LLaMA the overall decoder silhouette remains, while its internal
mechanisms change: absolute positional embeddings give way to RoPE, LayerNorm
to RMSNorm, and a two-layer FFN to gated SwiGLU. Comparing these mechanisms is
more informative than drawing another nearly identical tower.

### Encoder-only: BERT

BERT retains the encoder stack: full bidirectional self-attention; token,
segment, and learned position representations; masked language modeling; and,
in the original model, next sentence prediction.

For MLM, 15% of WordPieces are selected. Of these, 80% are replaced by
`[MASK]`, 10% by a random token, and 10% remain unchanged. Loss is computed on
the selected positions. BERT is not an autoregressive generator: its
pre-training objective reconstructs hidden pieces rather than continuing a
prefix from left to right.

### Decoder-only: GPT

GPT uses a causal stack with no encoder and no cross-attention: masked
self-attention, an FFN, learned absolute positions in early GPT models, and
next-token prediction.

GPT-1 did not invent the decoder-only Transformer. It demonstrated that
generative pre-training transfers to downstream NLP tasks. GPT-2 then scaled
the model and data and emphasized task performance without task-specific
training examples.

### Modern decoder-only: LLaMA

LLaMA retains a causal decoder but changes its details:

- pre-normalization with RMSNorm;
- RoPE rather than absolute position embeddings;
- SwiGLU rather than a ReLU FFN;
- no bias in several linear layers;
- grouped-query attention in later Llama versions.

LLaMA 1 used ordinary multi-head attention. GQA must not be projected backward
onto the entire family or the first version.

## 9. Architectural diff

| Component | Transformer 2017 | BERT Base | GPT-1 | LLaMA 1 |
|---|---|---|---|---|
| architecture | encoder–decoder | encoder | causal decoder | causal decoder |
| norm layout | Post-LN | Post-LN | Post-LN-like | Pre-RMSNorm |
| activation / FFN | ReLU | GELU | GELU | SwiGLU |
| positions | sinusoidal | learned absolute | learned absolute | RoPE |
| cross-attention | decoder only | no | no | no |
| training objective | translation | MLM + NSP | next token + fine-tuning | next token |

This table illustrates a useful way to describe a new model: record what is
preserved and what changes instead of repeating the whole Transformer.

## 10. Why Transformer displaced the recurrent backbone

The original paper compares:

| Layer | Complexity per layer | Sequential operations | Maximum path length |
|---|---:|---:|---:|
| self-attention | $O(T^2d)$ | $O(1)$ | $O(1)$ |
| recurrent | $O(Td^2)$ | $O(T)$ | $O(T)$ |
| convolution | $O(kTd^2)$ | $O(1)$ | $O(\log_k T)$ |

The key gain is parallel training computation and a short path between any two
positions. The price is a quadratic interaction matrix. At long context,
$T^2$ becomes the central limitation and motivates FlashAttention,
sparse/sliding-window attention, linear attention, and SSMs.

:::caution[The table depends on the regime]
$O(1)$ sequential operations describes processing an already known sequence
inside a layer, not generation of unknown future tokens. Self-attention is not
always cheaper than a recurrent network either; the comparison depends on
$T$, $d$, hardware, and implementation.
:::

## 11. A minimal modern causal block

```python
class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

This is a LLaMA-like block, not the original Transformer. An exact 2017 decoder
also requires an encoder, cross-attention, post-residual LayerNorm, a ReLU FFN,
and sinusoidal positions.

## 12. How to read real code

Three levels of difficulty:

1. [Karpathy's `ng-video-lecture`](https://github.com/karpathy/ng-video-lecture)
   keeps the complete mechanism small enough to hold in your head.
2. [Karpathy's `build-nanogpt`](https://github.com/karpathy/build-nanogpt)
   reproduces GPT-2 124M, including the data pipeline, initialization, and
   training.
3. [Meta Llama 3 reference code](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
   shows RoPE, RMSNorm, SwiGLU, and GQA in a modern implementation.

`nanoGPT` remains useful as a compact historical implementation, although its
author now points readers to the newer `nanochat`.

## 13. Tensor-shape check

For a causal LLM:

```text
token ids                 [B, T]
token embeddings          [B, T, d]
Q, K, V                   [B, h, T, dh]
attention scores          [B, h, T, T]
attention result          [B, T, d]
residual after attention  [B, T, d]
FFN hidden                [B, T, dff]
residual after FFN        [B, T, d]
vocabulary logits         [B, T, |Vocab|]
```

An unexpected shape change along the residual stream almost always indicates a
missing projection or incorrect head concatenation.

## Summary

- The original Transformer is an encoder–decoder, not a GPT-like tower.
- An encoder block contains full self-attention and an FFN.
- The original decoder contains causal self-attention, cross-attention, and an
  FFN.
- Attention transfers information between positions; the FFN transforms each
  position nonlinearly and independently.
- The residual stream preserves shape `B × T × d_model`.
- BERT, GPT, and LLaMA differ not only in masking, but also in training
  objectives, positional mechanisms, normalization, and the FFN.
- A modern LLM diagram must not be presented as the exact 2017 architecture.

## Sources and interactive explanations

### Primary sources

- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Radford et al., Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Devlin et al., BERT](https://arxiv.org/abs/1810.04805)
- [Touvron et al., LLaMA](https://arxiv.org/abs/2302.13971)

### Code and explanations

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/13_transformers|TinyTorch 13 — Transformers]] — executable construction of `TransformerBlock` from attention, MLP, and layer normalization, with autoregressive generation.
- [Harvard, The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy, Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy, Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder|Encoder–Decoder reference]]

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention from the inside]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three Transformer architectural patterns]]
