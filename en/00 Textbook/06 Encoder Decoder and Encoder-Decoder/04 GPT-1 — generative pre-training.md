---
title: GPT-1 — generative pre-training
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "318e040db226fabac04ba6576bf629775600e4626a03e2ac7ce0a4ec16967499"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md"
last_updated: 2026-10-29
primary_sources:
  - https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
  - https://github.com/openai/finetune-transformer-lm/tree/a69b5c43b0452462890bca8ff92fb75dee9290cf
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык]]"
---

# GPT-1: generative pre-training

Suppose a labeled dataset contains the premise “A cat sleeps” and the
hypothesis “An animal rests.” The task is to choose entailment,
contradiction, or insufficient information. An ordinary language model
instead predicts the next token. GPT-1 connects these tasks through
shared parameters: first a causal Transformer learns to continue books,
then the model and a small classification head are fine-tuned on labeled pairs.

Fine-tuning changes all network weights. Sentiment classification and
story-ending selection need their own data and separate fine-tuned
parameter versions. GPT-1's main results follow this training on each
task; experiments without weight updates are discussed separately below.
Pre-training followed by fine-tuning already existed; the authors discuss
Dai and Le and ULMFiT. Their 2018 work examines transfer from a causal
Transformer and ways to present several task types to it
([paper, Sections 1–3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)).

## One architecture for both stages

GPT-1 has twelve blocks, state width $d=768$, and twelve attention heads.
Each head uses $768/12=64$ coordinates; the FFN intermediate width is
3072. A learned position representation is added to each token
representation. Maximum length is 512 tokens, including downstream
task special tokens.

The model has approximately 117 million parameters
([CS224n, slide 48](https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture09-pretraining-updated.pdf#page=48)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/gpt1-figure1-block.png]]

*GPT-1 block, the left panel of Figure 1 in
[Radford et al., page 4](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=4).
Computation proceeds upward: embeddings, twelve blocks, then one of two
output heads. Normalization follows each residual addition. The panel
is extracted without redrawing; the complete figure appears below.*

Input $U$ of shape $[B,T]$ contains token IDs. Looking up and adding
token-table and position-table rows gives $H_0$ of shape $[B,T,768]$:

$$
H_0=E_{\mathrm{token}}[U]+E_{\mathrm{position}}[0:T].
$$

Each block then applies causal self-attention and an FFN.
In the original code, normalization comes **after** residual addition:

$$
\begin{aligned}
A_\ell&=\operatorname{LN}
  (H_{\ell-1}+\operatorname{Attention}(H_{\ell-1})),\\
H_\ell&=\operatorname{LN}(A_\ell+\operatorname{FFN}(A_\ell)).
\end{aligned}
$$

This is post-norm, as in the original Transformer. GPT-2 changes the
order; its pre-norm layout must not be projected backward onto GPT-1.
The FFN uses GELU; there is no RMSNorm, RoPE, or SwiGLU here. In the
code, `conv1d` with kernel size 1 is an ordinary affine transformation
of the last axis, not a mixture of adjacent tokens
([`block`, `mlp`, and `conv1d`](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/train.py)).

At every layer, row $t$ may read positions $0,\ldots,t$. The last
token's state can therefore use the entire input, while the first
cannot receive words on its right. There is no separate encoder or
cross-attention. “Decoder-only” refers to causal blocks, not to
receiving memory from another encoder.

For language modeling, each state is projected into the vocabulary.
In the paper's mathematical notation, the input token table is also
the output matrix: weights are tied.

$$
P(u_{t+1}\mid u_{\le t})
=\operatorname{softmax}(h_tE_{\mathrm{token}}^\top).
$$

Classification places a different, small matrix above a selected
position's state. Both heads receive representations from the same
Transformer, but their distributions cover different sets: vocabulary
tokens and task labels.

## Stage one: what BooksCorpus teaches

The authors pretrain on BooksCorpus, containing more than seven thousand
unpublished books. Connected prose places names, events, and related
sentences in one context. The network nevertheless sees segments of
at most 512 tokens, not a whole book in one pass. The paper contrasts
this corpus with the shuffled sentences of the 1 Billion Word Benchmark.
That motivates corpus selection; it is not an isolated experiment
establishing the effect of textual continuity alone.

Published preprocessing uses `ftfy` for Unicode repair, standardizes
spacing and punctuation, splits with spaCy, lowercases each extracted
token, and applies BPE. This already loses original casing and is not
a reversible encoding of arbitrary text. The authors train BPE with
40,000 merges. Merge count is not exact final vocabulary size: initial
characters remain and special entries are added
([`TextEncoder`](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/text_utils.py)).

In the toy segment `a cat sleeps the dog runs`, let each word be one
token. The output at `a` predicts `cat`, the next predicts `sleeps`,
and so on:

```text
input:   a    cat     sleeps  the  dog   runs
target:  cat  sleeps  the     dog  runs  —
```

The final output has no known target because the segment's continuation
is absent from this example. Six input tokens yield five loss terms.
The network computes them in parallel; the causal mask prevents reading
the target from the next position. Gradients pass through the output
projection, all blocks, and input representations.

The paper writes language modeling as maximizing a sum:

$$
L_1(U)=\sum_t\log P(u_t\mid u_{t-k},\ldots,u_{t-1};\Theta).
$$

Minimization reverses its sign to give negative log-likelihood.
Averaging over real targets reduces the loss magnitude's dependence on
length. Both sign and reduction must be explicit because they affect
the language-model objective's weight during fine-tuning.

The historical pre-training recipe in Section 4.1 uses Adam, a peak
learning rate of $2.5\cdot10^{-4}$, warmup over the first 2000 updates,
then cosine decay to zero. It specifies 100 epochs, batches of 64
length-512 segments, dropout 0.1, and normal initialization with standard
deviation 0.02. These describe a particular experiment; one short
forward pass below does not reproduce GPT-1 pre-training.

Most settings remain for fine-tuning. For most tasks, the authors give
learning rate $6.25\cdot10^{-5}$, batch 32, three epochs, and linear decay
after warmup over the first 0.2% of updates. Dropout 0.1 precedes the
classification head. New special tokens and the head train together
with transferred weights.

## Stage two: express each task as a sequence

Fields of a labeled task become one or more text inputs. Special tokens
mark boundaries and the position from which a representation is extracted.
The figure retains the authors' names `Start`, `Delim`, and `Extract`.
Published code calls them `_start_`, `_delimiter_`, and `_classify_`.

<details>
<summary>Complete Figure 1: the block and four task formats — click the figure to enlarge</summary>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/gpt1-figure1.png]]

*Radford et al., [Improving Language Understanding by Generative Pre-Training,
Figure 1, page 4](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
The left shows the block and two output heads. On the right, from top
to bottom: classification, entailment, semantic similarity, and answer
selection. Linear denotes a learned output projection, not generation
of a textual label. The original is preserved without redrawing.*

</details>

### Single-text classification

For review sentiment, input is `[Start, review, Extract]`.
When `Extract` is processed, causal attention permits the whole review.
Its last-layer state is multiplied by
$W_y\in\mathbb R^{768\times K}$ and softmaxed over $K$ classes.

With right padding, `Extract` may precede the last array element.
Its position must be located instead of taking `h[:, -1]`. Original
code finds it using the `_classify_` ID. Positive or negative sentiment
is stored as a separate label, not inserted into the input from which
it must be predicted.

### Textual entailment

An ordered pair becomes
`[Start, premise, Delim, hypothesis, Extract]`. Our example is:

```text
index:  0      1  2   3      4      5  6      7     8
input:  Start  A  cat sleeps Delim  An animal rests Extract
label:  entailment
```

State $h_8$ uses both sentences. A $[768,3]$ head produces three logits.
Swapping premise and hypothesis can change the correct answer:
“A cat sleeps” entails a resting animal, but the reverse does not
establish that the animal is a cat.

For hand calculation, reduce the state to $h=(1,2)$ and use columns
$(1,0)$, $(0,0)$, and $(-1,0)$ in $W_y$. Logits are $(1,0,-1)$.
Exponentials are approximately $(2.7183,1,0.3679)$, giving probabilities
$(0.6652,0.2447,0.0900)$ after division by their sum. The correct
entailment label has loss $-\ln0.6652\approx0.4076$.

These numbers illustrate head arithmetic; they are not GPT-1 outputs.
The CE derivative with respect to three logits is $g=p-(1,0,0)$.
Writing $h$ and $g$ as row vectors gives

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial W_y}&=h^\top g,\\
\frac{\partial\mathcal L}{\partial h}&=gW_y^\top.
\end{aligned}
$$

For these values, $g\approx(-0.334759,0.244728,0.090031)$.
The first row of the $W_y$ gradient equals $g$, and the second equals
$2g$; the state gradient is approximately $(-0.424790,0)$.
The optimizer changes $W_y$ and Transformer weights on which $h$
depends. The hidden state is recomputed on the next pass, not retained
as an independent trainable parameter.

### Semantic similarity

For a symmetric task, a pair is read twice:
`[Start, A, Delim, B, Extract]` and
`[Start, B, Delim, A, Extract]`. These sequences independently pass
through the same network. Their two `Extract` states are then added
and sent to a shared output head
([paper, Section 3.3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)).

If $h_{AB}=(1,2)$ and $h_{BA}=(3,-1)$, their sum is $(4,1)$.
A head with weights $(1,-1)$ gives score $3$. Swapping inputs exchanges
the two states but preserves their sum. This assumes deterministic
evaluation; dropout can introduce randomness during training.

Two passes do not make internal causal states bidirectional. In each
pass, an early word still cannot read its right context. Symmetry
arises when the final representations are combined. Adding probabilities
after softmax instead of states would change the described architecture.

### Choosing an answer or story ending

Each candidate gets its own input:
`[Start, context, question, Delim, answer, Extract]`.
Story Cloze has no separate question: one of two possible endings follows
the context. A shared linear head gives one scalar per sequence;
softmax normalizes those scalars across candidates.

For scores $s_1=2$ and $s_2=0.5$,

$$
P(y=1)=\frac{e^2}{e^2+e^{0.5}}\approx0.8176.
$$

If the first option is correct, CE is approximately 0.2014.
This does not require its complete text to have continuation probability
0.8176. The scalars come from a separate trained head; they are not
summed language log-probabilities of the ending.

With $B$ tasks and $C$ candidates, input shape is $[B,C,T]$. The first
two axes flatten to $[BC,T]$ before the Transformer. After extracting
$BC$ states and applying a scalar head, scores reshape to $[B,C]$.
Candidates do not attend to each other; they interact through output
normalization and shared weights.

```python
import torch
import torch.nn.functional as F

h = torch.tensor([1., 2.], requires_grad=True)
head = torch.tensor([[1., 0., -1.], [0., 0., 0.]], requires_grad=True)
logits = h @ head
nli_loss = F.cross_entropy(logits[None], torch.tensor([0]))
nli_loss.backward()
torch.testing.assert_close(
    logits.softmax(0), torch.tensor([.66524096, .24472847, .09003057]))
torch.testing.assert_close(
    h.grad, head.detach() @ (logits.detach().softmax(0) - torch.tensor([1., 0., 0.])))
g = logits.detach().softmax(0) - torch.tensor([1., 0., 0.])
torch.testing.assert_close(head.grad, h.detach()[:, None] * g[None, :])

ab, ba = torch.tensor([1., 2.]), torch.tensor([3., -1.])
w = torch.tensor([1., -1.])
assert (ab + ba) @ w == (ba + ab) @ w == 3
choice_scores = torch.tensor([[2., .5]])
choice_loss = F.cross_entropy(choice_scores, torch.tensor([0]))
print(round(nli_loss.item(), 6), round(choice_loss.item(), 6))
# 0.407606 0.201413
```

### Two objectives and their normalization

The paper's $L_1,L_2,L_3$ are log-likelihoods to maximize.
$L_2$ scores the correct task label; the auxiliary $L_1$ predicts
next tokens in task texts:

$$
L_3=L_2+\lambda L_1.
$$

The equivalent minimized objective has positive terms
$\mathcal L_{\mathrm{class}}+\lambda\mathcal L_{\mathrm{LM}}$.
The authors give $\lambda=0.5$ for standard fine-tuning. Published code
first averages LM loss over real targets **within each** sequence, then
averages across all serialized candidates in the batch:

$$
\begin{aligned}
\ell_{bc}&=
-\frac{\sum_{t=1}^{T-1}m_{bct}
 \log P(x_{bct}\mid x_{bc,<t})}
 {\sum_{t=1}^{T-1}m_{bct}},\\
\mathcal L&=\frac1B\sum_b\operatorname{CE}(s_b,y_b)
+\frac{\lambda}{BC}\sum_{b,c}\ell_{bc}.
\end{aligned}
$$

$m_{bct}=1$ for a real target and 0 for right padding. This computation
does not predict `Start` from an empty context; `Delim` and
`Extract` do enter the target sequence. LM loss includes all candidate
texts, including incorrect endings; the separate label $y_b$ specifies
the correct choice.

If all eight targets in a nine-token pair have probability $0.5$,
summed LM loss is $8\ln2\approx5.5452$ and mean LM loss is
$\ln2\approx0.6931$. With the NLI example's CE, the mean gives
$0.4076+0.5\cdot0.6931\approx0.7542$, while the sum gives
$0.4076+0.5\cdot5.5452\approx3.1802$. These assign different effective
weights to the auxiliary task.

A mean over sequences also differs from a mean over all tokens.
For one candidate with loss $2$ and another with losses $(1,1,1)$,
the first method gives $(2+1)/2=1.5$; the second gives
$(2+1+1+1)/4=1.25$. Original code uses the first method.

The following excerpt preserves OpenAI's `model` implementation.
Its last line averages each sequence's losses. The subsequent
`mgpu_train` averages the resulting array and adds `lm_coef` times
language loss to classification loss.

```python title="Original implementation — MIT"
        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
```

*OpenAI, [`train.py`, commit `a69b5c43`](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/train.py),
Copyright (c) 2018 OpenAI, [MIT license](/licenses/openai-gpt1-MIT.txt).
The excerpt is unchanged.*

This code differs from the paper's separate embedding matrices:
`we` combines token, special-token, and position rows. Input adds the
selected token and position rows, while LM projection uses all of
`we`, including position rows. An implementation with separate tables
and token-only output is therefore not a bitwise copy of this code.

### Complete teaching example: pre-training and fine-tuning

The next example performs one update at each stage. A small post-norm
Transformer first predicts a continuation from six ordinary tokens.
Three randomly initialized special embeddings and a scalar head are
then added. Old token weights and the position table are retained.

Fine-tuning uses two contexts with two candidates each. Every
`[Start, context, Delim, candidate, Extract]` input has a valid-length
mask. ID zero denotes ordinary token `a`; in padding positions it is
only an array filler. Its presence on the right must neither alter
the preceding `Extract` state nor add LM targets.

```python
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_num_threads(1)
vocabulary = ["a", "cat", "sleeps", "the", "dog", "runs",
              "an", "animal", "rests", "child", "plays", "outside"]

class TinyGPT(nn.Module):
    def __init__(self, vocab=12, width=16, length=12):
        super().__init__()
        self.word = nn.Embedding(vocab, width)
        self.position = nn.Embedding(length, width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                width, 4, 64, dropout=0., batch_first=True,
                norm_first=False,
                activation=lambda x: F.gelu(x, approximate="tanh"))
            for _ in range(2)
        ])
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, std=.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)
            else:
                nn.init.ones_(p)

    def forward(self, ids):
        t = ids.shape[1]
        assert 1 <= t <= self.position.num_embeddings
        pos = torch.arange(t, device=ids.device)
        h = self.word(ids) + self.position(pos)
        mask = torch.ones(t, t, dtype=torch.bool, device=ids.device).triu(1)
        for block in self.blocks:
            h = block(h, src_mask=mask)
        return h

    def lm_logits(self, h):
        return F.linear(h, self.word.weight)  # tied token weights

model = TinyGPT()
optimizer = torch.optim.SGD(model.parameters(), lr=.05)
book = torch.tensor([[0, 1, 2, 3, 4, 5]])  # a cat sleeps the dog runs
before_pretrain = model.word.weight.detach().clone()
h = model(book)
pretrain_loss = F.cross_entropy(
    model.lm_logits(h[:, :-1]).reshape(-1, 12), book[:, 1:].reshape(-1))
pretrain_loss.backward()
optimizer.step()
assert not torch.equal(before_pretrain, model.word.weight)

# Add specials only after the language-model step.
START, DELIM, EXTRACT = 12, 13, 14
old_words = model.word.weight.detach().clone()
extended = nn.Embedding(15, 16)
nn.init.normal_(extended.weight, std=.02)
with torch.no_grad():
    extended.weight[:12].copy_(old_words)
model.word = extended
torch.testing.assert_close(model.word.weight[:12], old_words)
choice_head = nn.Linear(16, 1)
nn.init.normal_(choice_head.weight, std=.02)
nn.init.zeros_(choice_head.bias)

contexts = [[0, 1], [3, 4]]  # a cat; the dog
candidates = [[[2], [5, 11]], [[2, 11], [5]]]
labels = torch.tensor([0, 1])
ids = torch.zeros(2, 2, 12, dtype=torch.long)
valid = torch.zeros_like(ids, dtype=torch.bool)
for b, options in enumerate(candidates):
    for c, answer in enumerate(options):
        seq = [START] + contexts[b] + [DELIM] + answer + [EXTRACT]
        ids[b, c, :len(seq)] = torch.tensor(seq)
        valid[b, c, :len(seq)] = True
flat = ids.flatten(0, 1)
mask = valid.flatten(0, 1)
assert ((flat == EXTRACT) & mask).sum(1).eq(1).all()
pool = (flat == EXTRACT).long().argmax(1)

# A new optimizer must include the newly created parameters.
optimizer = torch.optim.SGD(
    list(model.parameters()) + list(choice_head.parameters()), lr=.05)
optimizer.zero_grad()
h = model(flat)
scores = choice_head(h[torch.arange(4), pool]).reshape(2, 2)
scores.retain_grad()
classification_loss = F.cross_entropy(scores, labels)
token_loss = F.cross_entropy(
    model.lm_logits(h[:, :-1]).reshape(-1, 15),
    flat[:, 1:].reshape(-1), reduction="none").reshape(4, -1)
token_loss.retain_grad()
target_mask = mask[:, 1:]
per_candidate = (token_loss * target_mask).sum(1) / target_mask.sum(1)
lm_loss = per_candidate.mean()
total_loss = classification_loss + .5 * lm_loss
before_finetune = model.word.weight.detach().clone()
before_head = choice_head.weight.detach().clone()
total_loss.backward()
assert model.blocks[0].self_attn.in_proj_weight.grad.norm() > 0
assert choice_head.weight.grad.norm() > 0
assert torch.isfinite(total_loss)
optimizer.step()
assert not torch.equal(before_finetune[:12], model.word.weight[:12])
assert not torch.equal(before_head, choice_head.weight)
print("scores:", tuple(scores.shape))
print("targets:", target_mask.sum(1).tolist())
print("mean LM loss per candidate:", per_candidate.detach().tolist())
print("classification / LM / total:",
      *[round(float(x.detach()), 6) for x in
        (classification_loss, lm_loss, total_loss)])
print("updated: backbone and task head")
# scores: (2, 2)
# targets: [5, 6, 6, 5]
# classification / LM / total: 0.718351 2.733433 2.085067
# updated: backbone and task head
```

State `h[bc, pool[bc]]` is extracted before right padding. The causal
mask blocks future positions, so this particular layout needs no
additional key-padding mask for real states. Left padding or interspersed
gaps require a separate analysis of masks and positions.

The first update changes weights using LM loss. The second uses CE
over two candidates and mean LM loss over four sequences. Both objectives
send gradients into the shared Transformer. Labels `[0,1]` identify
correct candidate indices and do not enter `ids`.

Flattening axes $B,C$ gives the row order below. The table omits only
special tokens and padding; target counts include `Delim` and
`Extract`, but not `Start`.

| Task, candidate | Context and continuation | LM targets | Correct choice |
|---|---|---:|---|
| 0, 0 | `a cat` → `sleeps` | 5 | yes |
| 0, 1 | `a cat` → `runs outside` | 6 | no |
| 1, 0 | `the dog` → `sleeps outside` | 6 | no |
| 1, 1 | `the dog` → `runs` | 5 | yes |

Labels `[0,1]` define correctness in this toy dataset. Without more
context, all these continuations are possible. The batch demonstrates
computation, not knowledge about cats and dogs.

With the specified CPU seed, row-mean LM losses are approximately
$2.666036$, $2.762151$, $2.767778$, and $2.737765$. Their mean is
$2.733433$. Classification loss is already averaged across two tasks
and equals $0.718351$, giving total
$0.718351+0.5\cdot2.733433\approx2.085067$.
Library versions may change the last few digits.

Each LM row has total weight $0.5/4=0.125$. A token in a five-target
row has coefficient $0.125/5=0.025$; one in a six-target row has
$0.125/6\approx0.020833$. Right padding gets zero. Averaging over
candidates thus equalizes sequence weights, not individual token
weights across lengths. This follows the normalization in the quoted
`model` and `mgpu_train` code.

An incorrect candidate also trains the LM: its next tokens are targets
just like those of the correct candidate. The separate classification
loss teaches selection between them. It sends gradients to both scores
in each task, not just the correct option's score.
After backward, `token_loss.grad` contains the coefficients $0.025$,
$0.020833$, and zero for padding. Each `scores.grad` row is one half
of candidate probabilities minus the correct-label one-hot vector;
the factor $1/2$ comes from averaging across two tasks.

Sizes, corpus, update count, and SGD are chosen for a short executable
example. It follows causal post-norm computation and the transfer
scheme, but reproduces neither GPT-1 training nor quality. It has no
dropout, BPE, or distributed training. Its LM output uses token rows
only, unlike the published TensorFlow code's combined table.

## Reading the key experiments

Main results follow separate supervised fine-tuning on each task.
The paper evaluates twelve datasets and reports improvements over the
previous best on nine. Results from Tables 2–4 show both successful
transfer and limitations:

| Dataset | Metric | GPT-1 | Paper's comparison |
|---|---|---|---|
| MultiNLI | accuracy, matched / mismatched | 82.1 / 81.4 | previous 80.6 / 80.1 |
| Story Cloze | ending-selection accuracy | 86.5 | previous 77.6 |
| RACE | answer-selection accuracy | 59.0 | previous 53.3 |
| RTE | entailment accuracy | 56.0 | previous 61.7 |
| MRPC | paraphrase F1 | 82.3 | previous 86.0 |

These are the authors' historical comparisons, not one shared rerun.
MRPC F1 cannot be directly compared with RTE accuracy. Likewise, the
GLUE score of 72.8 in Table 4 is not the mean in the separate ablation table.

Section 5, “Ablation studies,” removes parts of the recipe.
Table 5 reports these average scores over selected tasks:

| Variant | Table 5 average |
|---|---:|
| pretrained Transformer with auxiliary LM objective | 74.7 |
| no pre-training | 59.9 |
| no auxiliary LM objective | 75.0 |
| pretrained single-layer LSTM with LM objective | 69.1 |

Pre-training gives a 14.8-point average difference in this comparison.
The auxiliary LM objective does not improve the mean: removing it raises
the mean by 0.3 points. The authors nevertheless observe benefits on
large datasets and declines on some small ones. The conclusion is
limited to this experiment; positive $\lambda$ does not guarantee
improvement on every task.

The LSTM comparison uses a particular single-layer network with 2048
hidden units and the authors' training regime. It does not establish
that every Transformer beats every recurrent network at equal budget.
The left panel of Figure 2 shows transfer improving as more pretrained
layers are used, so results do not arise from the initial word table alone.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/gpt1-figure2-transfer.png]]

*Radford et al., [Figure 2, left panel, page 7](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=7).
The horizontal axis counts transferred layers. Blue lines show RACE
against the left scale; orange lines show MultiNLI against the right.
Solid lines are held-out results; dashed lines are training results.
Heights of different-colored curves are not scores on one common scale.
Both axes and the complete legend are preserved without redrawing.*

The [implementation README](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/README.md)
reports a separate Story Cloze reproduction: median accuracy over ten
runs is 85.8%, whereas the paper's 86.5% is one run. The authors warn
about GPU-operation nondeterminism. Published `train.py` implements
ROCStories/Story Cloze, not all twelve experimental tasks.

### What zero-shot meant in this work

Besides fine-tuning, the authors tried fixed rules without task-specific
weight updates. Section 5, “Zero-shot Behaviors,” lists four ways to
derive answers from language probabilities:

- CoLA compares mean sentence log-probability with a threshold to assess acceptability.
- SST-2 appends `very` to a review and compares the next-word probabilities
  of `positive` and `negative`.
- RACE scores each answer's mean token log-probability given context and question.
- DPRD substitutes a possible referent for a pronoun and scores mean
  log-probability of the text remaining after the substitution.

RACE's normalization affects ranking across answer lengths.
Suppose answer A has token probabilities $(0.8,0.8)$, while one-token
answer B has probability $0.7$. Full-string probabilities are $0.64$
and $0.7$, so the product prefers B. Mean log-probabilities are
$\ln0.8\approx-0.2231$ and $\ln0.7\approx-0.3567$, so the mean prefers A.
This reduces the direct length penalty but does not itself guarantee
correctness or comparability across every candidate type.

Figure 2's right panel tracks normalized heuristic performance during
pre-training. The authors scale between random performance and the
then-best result; the axis is not ordinary accuracy. These observations
must be separated from the main table, where weights and the output
head are trained on labels.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/gpt1-figure2-zero-shot.png]]

*Radford et al., [Figure 2, right panel, page 7](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=7).
The horizontal axis is the number of language-pretraining updates on
a logarithmic scale. Color identifies tasks; line style distinguishes
Transformer and LSTM. The vertical axis is a relative score between
random performance and the then-best single-model result, not accuracy.
The original panel retains its axes and legend.*

## What GPT-1 did not solve

The 512-token window limits both book segments and the total fields
of downstream tasks. For
`Start + context + Delim + answer + Extract`, no more than 509 positions
remain for content. If preprocessing truncates a sentence or important
fact, causal attention cannot recover the missing input.
`transform_roc` truncates context and candidate separately, so actual
limits must be read together with data-preparation code.

Classification at `Extract` uses all preceding text, but early-word
states remain one-sided. Two semantic-similarity passes remove the
final score's dependence on input order without changing visibility
inside blocks.

A new task requires labels, an output head, and parameter training.
A shared initialization does not mean that one fine-tuned network
performs all twelve tasks unchanged. Finally, selecting a plausible
story ending measures success on that test, not the factual reliability
of arbitrary text continuations.

## Legacy and the transition to GPT-2

GPT-1 combines causal language pre-training with transfer of all parameters
to different labeled tasks. Input formats retain one Transformer, while
heads specify the required answer: a label, similarity score, or candidate index.

In [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2]],
the authors enlarge model and corpus and investigate task execution
through a text prefix without separate supervised fine-tuning.
That interface differs from GPT-1's trained heads described here.

## Sources and further reading

- [Radford et al., Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), 2018. Sections 3.1–3.3: objectives and four formats; 4.1: training; Section 5: zero-shot and ablations.
- [OpenAI official code, commit a69b5c43](https://github.com/openai/finetune-transformer-lm/tree/a69b5c43b0452462890bca8ff92fb75dee9290cf). `train.py`: post-norm, pooling, LM normalization, and ROCStories serialization; `text_utils.py`: text preparation and BPE.
- [OpenAI, Improving language understanding with unsupervised learning](https://openai.com/index/language-unsupervised/), June 11, 2018. Accompanying publication and experimental context.
- [Stanford CS224n, Lecture 9: Pretraining, slides 46–50](https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture09-pretraining-updated.pdf#page=46). Language and classification heads, GPT-1, NLI formatting, and results, in the original English.
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three architectural patterns]] — input visibility and training objectives.
- [[02 Areas/ML & DL/Papers/GPT 1.0|Earlier GPT-1 note]] — additional notes, not a replacement for the primary paper.

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa, and DeBERTa]].
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2]].
