---
title: GPT-3 — scaling and in-context learning
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "3cd49ae19d8bc01c4c68d05a823ffd5c1fd7fee1106332a9a53843491566b0a2"
translation_reviewed: "2026-10-30"
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md"
last_updated: 2026-10-30
primary_sources:
  - https://arxiv.org/abs/2005.14165v4
  - https://github.com/openai/gpt-3/tree/d7a9bb505df6f630f9bab3b30c889e52f22eb9ea
  - https://stanford-cs324.github.io/winter2022/lectures/capabilities/
  - https://jalammar.github.io/how-gpt3-works-visualizations-animations/
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer]]"
---

# GPT-3: scaling and in-context learning

A language model can classify a review after reading several reviews
with completed labels. It receives all this text as one request and
continues the last, unfinished pair. Switching from reviews to
translation changes the input examples; the weights remain fixed.

GPT-3 investigates how well this way of specifying tasks works as model
size increases. Brown et al. train eight autoregressive Transformers,
from 125 million to 175 billion parameters, and evaluate zero-, one-,
and few-shot settings. Every model sees 300 billion pre-training tokens.
These experiments do not train on individual benchmarks. Architecture,
data, prompt preparation, and scoring are described in
[Section 2](https://arxiv.org/abs/2005.14165v4).

## Where the “learning” happens

In ordinary fine-tuning, an input–correct-answer pair defines a loss.
Backward computes gradients, an optimizer changes parameters, and the
saved checkpoint retains those updates. This is how
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1]]
trains a classifier for a new task.

In GPT-3's few-shot experiments, completed pairs become part of the
input sequence, followed by a new question without an answer.
Let $D$ denote demonstrations, $q$ the question, and $y$ the answer:

$$
P_\theta(y\mid D,q)=\prod_{t=1}^{|y|}
P_\theta(y_t\mid D,q,y_{<t}).
$$

Parameters $\theta$ are the same for every request. Replacing $D$
changes tokens, hidden states, and continuation probabilities. Removing
examples from the next request removes them from the model's available
context. The saved weights do not themselves memorize those examples.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/figure-replacements-2026-10-30/gpt3-demonstrations-frame-171.png]]

*Jay Alammar, [How GPT3 Works — examples in the prompt](https://jalammar.github.io/how-gpt3-works-visualizations-animations/).
Two completed description–code pairs precede an unfinished third pair.
The continuation is generated below; no weight update is shown.
Complete frame 171 of the [original animation](https://jalammar.github.io/images/gpt3/09-gpt3-generating-react-code-example.gif),
exported to lossless PNG without cropping or relabeling,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Alammar presents the prompt format as his reconstruction, not a recorded GPT-3 run.
The paper's zero-/one-/few-shot comparison remains available in
[Figure 2.1](https://arxiv.org/pdf/2005.14165v4#page=7).*

Shots count complete demonstrations, not sentences or instructions.
A task description plus one new question is zero-shot. One completed
pair before it is one-shot; two pairs are two-shot, a few-shot setting.
The paper usually uses 10–100 examples, as far as the 2048-token window
permits. GPT-3 fine-tuning is outside this study.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt3-2026-10-22/10-gpt3-fine-tuning.gif]]

*Jay Alammar, [How GPT3 Works — Fine-tuning](https://jalammar.github.io/how-gpt3-works-visualizations-animations/).
The animation shows additional training of pretrained parameters.
The preceding image instead places examples in the input at fixed weights.
Original unchanged,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

“Learning” in in-context learning describes behavior changing after
examples are presented. It does not yet explain how the task is solved.
For translation, pre-training must supply knowledge of both languages;
a few pairs may specify translation direction and answer format.
For an unusual string transformation, examples may specify a more
particular rule. Section 5 discusses this ambiguity: recognizing a
familiar task, adapting to a variant, and acquiring a new rule cannot
be distinguished from one successful answer.

## Mechanics of a few-shot prompt

Consider a two-shot request with two allowed labels. This is a teaching
example, not a prompt or measured response from GPT-3 experiments.

~~~text
Classify the review. Answer only: positive sentiment or negative sentiment.
Review: The acting was wonderful and I enjoyed every minute.
Label: positive sentiment
Review: The story was dull and the dialogue was painful.
Label: negative sentiment
Review: The beautiful scenery could not rescue the boring plot.
Label:
~~~

The first two reviews contain answers; the last ends at `Label:`.
The instruction specifies the classification task and permitted strings,
while demonstrations show the review–label mapping. Newlines and colons
also enter the tokenized text. No special two-class head is created.

After tokenization, the Transformer processes the complete prefix.
At every layer, its final position uses accessible earlier positions.
The causal mask prevents reading the unwritten answer. Vocabulary
projection of the final state gives next-token probabilities. The
selected token is appended; the next pass predicts a continuation
conditioned on that choice.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt3-2026-10-22/tokens-through-stack-frame-385.png]]

*Jay Alammar, [How GPT3 Works — input and response through the layer stack](https://jalammar.github.io/how-gpt3-works-visualizations-animations/).
Complete frame 385 of the [original animation](https://jalammar.github.io/images/gpt3/08-gpt3-tokens-transformer-blocks.gif),
without cropping or relabeling,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Columns show positions processed through every block. The output after
the last input token specifies `Okay`; reading `Okay` produces
`human`. Each word is schematically treated as one token. This is
an explanatory figure, not a recorded experiment with GPT-3 weights.*

### Reusing demonstrations

Suppose instruction and demonstrations occupy positions 0–79, and
the new question occupies 80–99. Position 35's first-layer state
depends only on inputs 0–35. At the next layer, it again reads only
positions through 35, now processed by the previous layer. No layer
therefore receives information from the new question at this position.
The answer may depend on demonstrations, but demonstration states
are not recomputed in light of the answer.

If several questions begin with the same 80 tokens, their K/V can be
computed once and reused as a shared prefix. Each question then gets
its own states at positions 80 onward. Memory for question A cannot
be substituted for question B: after the common prefix, inputs differ.
Changing a label, separator, demonstration order, or position indices
also changes the corresponding states. What is reused is computation
of an identical prefix, not a model-training result. This follows
from GPT-2's causal mask; it is not a claim about a particular GPT-3 service.

Without reuse, 100 requests with an 80-token common prefix and 20-token
questions process 10,000 input positions. A common cache reduces new
positions to 80 + 100 × 20 = 2080. Speedup is not automatically the
ratio of these counts: every question still reads allowed K/V, and
runtime depends on attention implementation, batching, and data transfer.
The prefix still occupies 80 of each request's 2048 context positions.

### What changes in attention when weights stay fixed?

Projection matrices $W_Q,W_K,W_V$ remain unchanged but act on new hidden
states. Q/K/V, attention weights, and weighted outputs therefore change.
Freezing parameters does not make a function constant with respect to input.

A small numerical example separates this input dependence from GPT-3's
sentiment-classification quality. Let one head's already projected
query and keys be $q=(1,0)$, $k_1=(\sqrt2\ln3,0)$, and $k_2=(0,2)$.
Division by $\sqrt2$ gives scores $(\ln3,0)$ and weights $(0.75,0.25)$.
With $v_1=1$, $v_2=-1$, the output is $0.5$. Swapping values in response
to new input would change it to $-0.5$.

~~~python
import math
import torch

q = torch.tensor([1., 0.], dtype=torch.float64)
keys = torch.tensor([[2**0.5 * math.log(3), 0.],
                     [0., 2.]], dtype=torch.float64)
scores = keys @ q / 2**0.5
weights = scores.softmax(dim=0)
values_a = torch.tensor([1., -1.], dtype=torch.float64)
values_b = values_a.flip(0)
print(weights.tolist())
print((weights @ values_a).item(), (weights @ values_b).item())
# approximately [0.75, 0.25], then 0.5 and -0.5
~~~

This checks the attention formula, not a trained in-context-learning
model. Keys and values are set by hand. It shows where input can change
a result without a gradient step. Useful representations of reviews
and labels must already have been learned; their quality is tested
on new task examples.

### Generating a label versus scoring supplied candidates

With few permitted answers, one can score each complete label instead
of generating arbitrary text. Its tokens are forced in sequence and
their conditional log-probabilities summed. This uses teacher forcing
without weight updates.

Suppose a toy tokenizer splits labels into `[positive, sentiment]`
and `[negative, sentiment]`. First-token probabilities are $0.6$
and $0.4$; probabilities of `sentiment` after them are $0.3$ and
$0.8$. The first token favors positive. Complete probabilities are
$0.6\cdot0.3=0.18$ and $0.4\cdot0.8=0.32$, favoring negative.
Their logs are $-1.7148$ and $-1.1394$.

The code reconstructs that calculation from artificial vocabulary
logits. The logits at position $i$ predict position $i+1$.
Candidates have equal length and no padding; the four-ID vocabulary
is not GPT-3's actual vocabulary.

~~~python
import torch

# ID: 0 = end of prompt, 1 = positive, 2 = negative, 3 = sentiment
ids = torch.tensor([[0, 1, 3], [0, 2, 3]])
probs = torch.tensor([
    [[0., .6, .4, 0.], [.7, 0., 0., .3], [1., 0., 0., 0.]],
    [[0., .6, .4, 0.], [.2, 0., 0., .8], [1., 0., 0., 0.]],
], dtype=torch.float64)
logits = probs.log()

def completion_logprob(logits, ids, prompt_length):
    # The first answer token uses the final prompt position.
    logp = logits[:, :-1].log_softmax(dim=-1)
    target = ids[:, 1:]
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    answer_logp = token_logp[:, prompt_length - 1:]
    return answer_logp.sum(dim=-1), answer_logp.mean(dim=-1)

total, mean = completion_logprob(logits, ids, prompt_length=1)
print(total.tolist())                 # [-1.714798..., -1.139434...]
print(total.exp().tolist())           # [0.18, 0.32]
print(total.softmax(dim=0).tolist())  # [0.36, 0.64]
~~~

The last line renormalizes only the two selected candidates.
$0.64$ is neither the original probability among all possible texts
nor measured classifier reliability on data. A real model requires
checking tokenization, leading spaces, answer start, and padding.
If scoring includes a terminating newline, its probability also counts.
The same terminator may have different conditional probabilities
after different labels.

### How the paper compares different answer lengths

The sum $s_{\rm sum}=\sum_t\log P(y_t\mid D,q,y_{<t})$ scores the
whole string. For most multiple-choice tasks, GPT-3 normalizes by
token count: $s_{\rm mean}=s_{\rm sum}/|y|$. Maximizing mean log-probability
is equivalent to maximizing the geometric mean of conditional
probabilities. Dividing the string probability itself by length is
a different criterion.

For example, a one-token answer with probability $0.4$ beats a two-token
answer with probability $0.3$ under full-string scoring. Length
normalization instead compares $0.4$ with $\sqrt{0.3}\approx0.548$,
reversing the ranking. The scoring choice changes results even when
model and prompt are identical.

For ARC, OpenBookQA, and RACE, the authors apply another correction:

$$
s_{\rm ratio}(y)=\log P(y\mid D,q)-\log P(y\mid a_0),
$$

Here $a_0$ is a generic answer prefix such as `Answer: `.
This reduces the advantage of strings frequently continued even
without the question. If conditional answer probabilities are $0.4$
and $0.3$, and their background probabilities are $0.5$ and $0.05$,
ratios are $0.8$ and $6$, favoring the second answer. Ratios need not
sum to one; this is a ranking rule, not a new probability of the
correct class. Application conditions are given in
[§2.4](https://arxiv.org/pdf/2005.14165v4#page=10).

<span id="архитектура-знакомый-блок-в-новом-масштабе"></span>

## GPT-3 architecture

GPT-3 retains GPT-2's decoder-only architecture: next-token prediction,
pre-sublayer normalization, modified initialization, and reversible
tokenization. FFN width is $4d_{\rm model}$. Attention remains causal.
Compared with GPT-2, context grows from 1024 to 2048 tokens, and full
attention layers alternate with locally banded sparse layers
([§2.1](https://arxiv.org/pdf/2005.14165v4#page=8)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt3-2026-10-22/07-gpt3-processing-transformer-blocks.gif]]

*Jay Alammar, [How GPT3 Works — stack of 96 decoder layers](https://jalammar.github.io/how-gpt3-works-visualizations-animations/).
One token passes through all 96 blocks of the largest model. Dots
represent omitted blocks between 4 and 96. Internal attention and FFNs
are not drawn; the animation does not specify local-layer masks.
Original animation,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The 175B model has hidden width 12,288, 96 heads, head dimension 128,
and FFN width 49,152. For one length-$T$ request, block input has
shape $[1,T,12288]$. Projected and head-split Q/K/V have shape
$[1,96,T,128]$. Head outputs concatenate and project back to 12,288;
the FFN then applies its two dense projections at each position.
Sparse attention connections do not mean MoE: this GPT-3 has no router
or selection of several experts.

### Full and local attention

Full causal attention at $i$ uses every $j\le i$. A local layer
restricts the allowed region to a band near $i$. With teaching band
width 3, position 7 would read 5, 6, and 7 instead of 0–7.
This illustrates a local mask, not a published GPT-3 configuration:
Section 2.1 does not specify exact band width.

A local band restricts direct reading in one layer, not all input
positions capable of affecting the final output. Suppose the first
layer is full and the second local with width 3. Position 7 in layer
two reads states 5, 6, and 7. State 5 already depends on inputs 0–5,
state 6 on 0–6, and state 7 on 0–7. A path from input position 0 to
final position 7 therefore remains through the preceding layer.
This permits information transfer but does not guarantee that the
intermediate vector retains everything needed. Two width-3 local
layers without a full layer reach only positions 3–7: each read
extends the region two positions leftward.

For eight positions, a full causal mask has 36 allowed pairs; a local
mask has 21. Two full layers have 72 pairs, while one full plus one
local has 57. These count connections per head; Q/K/V projections,
FFNs, and residual additions remain. They describe the teaching example,
not measured savings in a GPT-3 run.

Alternation retains direct access to distant tokens in full layers
while reducing allowed pairs in local layers. Savings require an
implementation that avoids computing discarded elements. Masking a
dense $T\times T$ matrix after computing it is insufficient.
Fully materializing scores for one 175B full-attention layer at
$T=2048$ in FP16 would require
$96\cdot2048^2\cdot2=805306368$ bytes, or 768 MiB. This is one tensor's
size, not measured peak training memory.

### Space for demonstrations and the answer

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt3-2026-10-22/context-window-frame-229.png]]

*Jay Alammar, [How GPT3 Works — context window](https://jalammar.github.io/how-gpt3-works-visualizations-animations/).
Complete frame 229 of the [original animation](https://jalammar.github.io/images/gpt3/05-gpt3-generate-output-context-window.gif),
without cropping or redrawing,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Input and continuation occupy one sequence. Words are drawn as tokens
for clarity; actual BPE can split a word into several pieces.*

Suppose an instruction takes 80 tokens, each demonstration plus separator
takes 36, and the new question takes 50. Reserve 128 for the answer.
The 2048-token window then fits

$$
K_{\max}=\left\lfloor
\frac{2048-80-50-128}{36}\right\rfloor=49
$$

demonstrations. The complete sequence uses 2022 tokens; a fiftieth
demonstration would make 2058. These are assumed lengths, not a
measurement of the sentiment prompt above. Long documents and answers
leave less room for examples.

Demonstrations also increase inference cost. With ordinary MHA, a
full 175B KV cache for 2048 positions in FP16 would require
$2\cdot96\cdot2048\cdot12288\cdot2$ bytes, or 9 GiB per request.
This includes all layers' keys and values, excluding weights and
temporary tensors. It assumes the entire history is retained at
every layer; local layers may retain fewer positions. Storing
175 billion parameters at two bytes each alone needs roughly
350 GB. Prompting avoids task-specific training, but every request
still executes a large model.

## The scaling experiment

| Model | Parameters | Layers | $d_{\rm model}$ | Heads | $d_{\rm head}$ | Batch size, tokens | Peak learning rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | $6.0\times10^{-4}$ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | $3.0\times10^{-4}$ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | $2.5\times10^{-4}$ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | $2.0\times10^{-4}$ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | $1.6\times10^{-4}$ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | $1.2\times10^{-4}$ |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 | 2M | $1.0\times10^{-4}$ |
| GPT-3 175B | 175.0B | 96 | 12288 | 96 | 128 | 3.2M | $0.6\times10^{-4}$ |

Data: Brown et al., [Table 2.1, page 8](https://arxiv.org/pdf/2005.14165v4#page=8).
All rows use a 2048-token context. Batches are measured in tokens; learning
rates are peak values. Published entries are retained, including XL's
24 heads and 13B's width 5140; the inconsistencies are discussed below.

All eight models train on 300 billion tokens. Larger configurations
use larger batches and lower peak learning rates. For 125M, these
are 0.5 million tokens and $6\cdot10^{-4}$; for 175B, 3.2 million
and $6\cdot10^{-5}$. Depth, width, and GPU computation distribution
also change. This compares coordinated configurations at different
scales, not an isolated depth-versus-width effect at fixed budget.

Some published entries cannot be reconstructed from the usual identity
$d_{\rm model}=h d_{\rm head}$: 1.3B lists 2048, 24, and 128; 13B lists
5140, 40, and 128. They should be treated as reported values, not
silently corrected when transcribed. Full training code and GPT-3
weights are not released in the
[publication repository](https://github.com/openai/gpt-3/tree/d7a9bb505df6f630f9bab3b30c889e52f22eb9ea).
The detailed shapes above use the internally consistent 175B row.

Appendix B specifies Adam with $\beta_1=0.9$, $\beta_2=0.95$,
$\epsilon=10^{-8}$, global gradient-norm clipping at 1.0, and weight
decay 0.1. Learning rate warms up linearly over 375 million tokens,
then follows cosine decay to 10% of its maximum by 260 billion tokens;
remaining training uses that rate. Batch size grows from 32,000 tokens
to its final size over the first 4–12 billion tokens depending on
configuration. Converting the entire schedule into steps by dividing
by the final batch is therefore inaccurate.

Training uses V100s with matrix operations split across GPUs and
layers split across GPUs. In current terminology, these are widthwise
and depthwise forms of model parallelism. The paper does not publish
a concrete TP/PP topology for every configuration.

## Data and sampling

Raw Common Crawl contains copies, boilerplate, and documents unlike
the desired training text. The authors filter it, remove near duplicates,
and mix it with selected corpora. Table 2.2 distinguishes available
source size, mixture weight, and actual epochs.

| Source | Tokens, billions | Weight | Epochs |
|---|---:|---:|---:|
| filtered Common Crawl | 410 | 60% | 0.44 |
| WebText2 | 19 | 22% | 2.9 |
| Books1 | 12 | 8% | 1.9 |
| Books2 | 55 | 8% | 0.43 |
| English Wikipedia | 3 | 3% | 3.4 |

The caption defines weight as the fraction of sampled examples.
Rounded percentages sum to 101%; they are not an exact sampler
configuration. Epoch counts are reported from the table, not
recomputed from rounded weights. WebText2, Books1, and Wikipedia
repeat; Common Crawl and Books2 receive less than one complete pass.
Uniformly shuffling all source text would produce a different mixture.

### Quality filtering and near duplicates

Appendix A takes positive classifier examples from curated WebText,
Wikipedia, and books, and negatives from raw Common Crawl.
Logistic regression uses HashingTF features. Its score measures
similarity to curated corpora, not the truth of each document's claims.

A document with score $s$ is retained if a draw from
`np.random.pareto(9)` exceeds $1-s$. Retention probability is
$(2-s)^{-9}$: 1 at $s=1$, approximately 0.424 at $s=0.9$, and
$1/512$ at $s=0$. Stochastic selection keeps some low-scoring documents
rather than imposing one hard threshold, retaining some diversity
outside the positive-example distribution.

MinHashLSH with ten hashes then finds near duplicates within and
across corpora; WebText is additionally removed from Common Crawl.
Appendix A reports an average dataset-size reduction of 10%.
Corpus deduplication and test-overlap removal solve different problems:
one reduces repeated text, the other prevents training on test answers.

### Document packing

Training uses length-2048 sequences. Short documents are packed together
with an end-of-text token between them. In `a b EOT c d`, token
`c` can still attend to `a`, `b`, and `EOT`: the authors do not
apply document-boundary block-diagonal masks. EOT signals a boundary
but does not itself block attention across it. Within each source,
examples are drawn without replacement until the end of a pass;
the source may then be reused
([Appendix B](https://arxiv.org/pdf/2005.14165v4#page=43)).

<span id="ключевые-результаты-без-упрощения"></span>

## Results: task, format, and metric

For each test example, demonstrations are randomly sampled from the
task's training split. Exceptions are specified: LAMBADA and StoryCloze
use development examples; original Winograd uses its single available
set. When separate dev and test splits exist, $K$ is tuned on dev and
then fixed for test. Test labels must not guide that choice.

Prompt format can change what the scored continuation means.
[Stanford CS324, Capabilities](https://stanford-cs324.github.io/winter2022/lectures/capabilities/)
separates task definition, reduction to language modeling, and results.
The discussion below similarly separates LAMBADA, closed-book answers,
and synthetic arithmetic. For free-form answers, Section 2.4 describes
beam search of width 4 with length penalty 0.6; metrics depend on
the task: F1, BLEU, or exact match.

### LAMBADA: specifying exactly one answer

LAMBADA asks for a paragraph's final word. Simply truncating before
that word also assigns probability to continuations that leave the
sentence open. Few-shot formatting supplies examples with a blank
and a separate answer, demonstrating an infilling task.

This short example illustrates the format; its words are not from
the test set.

~~~text
Fill in blank:
Mira took a pear from the bowl. The fruit in her hand was a ___. -> pear
Anton left his bicycle outside. In the morning he rode away on his ___. ->
~~~

The intended answer is `bicycle`, not a repetition of the paragraph.
Appendix G.21 shows the actual format; Appendix H specifies $K=15$
for few-shot LAMBADA.

| GPT-3 175B, LAMBADA test | Zero-shot | One-shot | Few-shot |
|---|---:|---:|---:|
| accuracy | 76.2% | 72.5% | 86.4% |
| target-word perplexity | 3.00 | 3.35 | 1.92 |

Source: [§3.1.2 and Table 3.2](https://arxiv.org/pdf/2005.14165v4#page=12).
Zero-shot uses a different format from one/few-shot. Differences cannot
therefore be attributed solely to demonstration count under fixed
wording. One example hurts here; fifteen help. The 125M model has
few-shot accuracy 22.0%, versus zero-shot 42.7%. Added text helps only
to the extent that the model can use its format. LAMBADA's pre-training
overlap is examined below.

### Question answering without retrieval

In closed-book QA, the model receives a question and answer examples,
but no factual source documents or search tool. Answering a new
question requires pre-training knowledge. `Q: … A: …` pairs establish
answer form without supplying each new question's fact.

A short teaching one-shot example is:

~~~text
Q: What is the capital of Japan?
A: Tokyo
Q: What is the capital of Canada?
A:
~~~

The intended `Ottawa` cannot be extracted from the single Japan
demonstration. That example asks for a short city-name answer; the
model must already have learned the Canadian fact. This is neither
a test-set example nor a measured GPT-3 response.

| Task | Zero-shot | One-shot | Few-shot, $K=64$ |
|---|---:|---:|---:|
| Natural Questions | 14.6 | 23.0 | 29.9 |
| WebQuestions | 14.4 | 25.3 | 41.5 |
| TriviaQA | 64.3 | 68.0 | 71.2 |

Percentages are from [Table 3.3](https://arxiv.org/pdf/2005.14165v4#page=13).
Appendix H identifies NQs and WebQs as test and TriviaQA as dev;
175B's few-shot 71.2 is also confirmed on the TriviaQA wiki-split test
server. These should not become one “GPT-3 accuracy”: questions and
answer requirements differ.

On TriviaQA, few-shot GPT-3 exceeds the paper's closed-book T5-11B+SSM
comparison of 60.5. On Natural Questions, that T5 reaches 36.6, above
GPT-3's 29.9. Success on one question format does not guarantee transfer
to another. Retrieval-based systems need a separate comparison:
they receive extra information when answering, and their index and
retriever also consume resources.

### Translation: direction and parallel examples

Translation demonstrations pair source sentences with translations;
a new pair stops immediately after the separator. Unlike closed-book
QA, the answer must preserve the input sentence's content rather than
retrieve one fact from parameters.
[CS324, Translation](https://stanford-cs324.github.io/winter2022/lectures/capabilities/#translation)
shows this reduction to text continuation. GPT-3 one-shot uses one
pair; few-shot uses 64.

| Direction | Zero-shot | One-shot | Few-shot |
|---|---:|---:|---:|
| En→Fr | 25.2 | 28.3 | 32.6 |
| Fr→En | 21.2 | 33.7 | 39.2 |
| En→De | 24.6 | 26.2 | 29.7 |
| De→En | 27.2 | 30.4 | 40.6 |
| En→Ro | 14.1 | 20.6 | 21.0 |
| Ro→En | 19.9 | 38.6 | 39.5 |

These are BLEU scores from [Table 3.4](https://arxiv.org/pdf/2005.14165v4#page=15):
WMT'14 for French–English, WMT'16 for the other pairs, and
`multi-bleu.perl` with XLM tokenization. Appendix H separately gives
SacreBLEU, so these variants must not be mixed. For example, De→En
40.6 in Table 3.4 and 43.0 in contamination Table C.1 use different
BLEU variants and demonstration seeds.

Into-English few-shot results exceed the listed unsupervised NMT
systems. GPT-3 nevertheless receives parallel pairs at evaluation,
so available supervision differs. From-English results are weaker:
French scores 32.6 versus 39.2 in reverse. The pre-training corpus
is 93% English words, an asymmetry in data. Without a separate
experiment, the difference cannot be assigned solely to BPE.
A reproduction must fix language, dataset, examples, and metric
implementation, not compare an unlabeled “translation quality” number.

### Arithmetic and limits of generalization

Synthetic tasks ask numerical questions and use exact answers,
without partial credit for closeness. The authors generate 2000
examples for each of ten operation types, drawing operands uniformly
from specified ranges. “Two-digit” addition, for example, uses
integers 0–99, not only 10–99.

| GPT-3 175B, accuracy | Zero-shot | One-shot | Few-shot |
|---|---:|---:|---:|
| addition of two numbers in $[0,100)$ | 76.9 | 99.6 | 100.0 |
| addition in $[0,1000)$ | 34.2 | 65.5 | 80.4 |
| addition in $[0,10000)$ | 4.0 | 14.0 | 25.5 |
| multiplication in $[0,100)$ | 19.8 | 27.4 | 29.2 |

Source: [Table 3.9](https://arxiv.org/pdf/2005.14165v4#page=23).
Section 3.9.1's prose gives 80.2 for three-digit addition, while the
table gives 80.4; the table value is retained here. Perfect accuracy
on these short additions does not establish a general error-free
addition algorithm: accuracy falls rapidly with digit count. The
authors observe errors such as missed carries.

For $83+59$, the ordinary algorithm computes $3+9=12$, writes 2 in
the units place, carries 1, then computes $8+5+1=14$, producing 142.
Missing the carry gives 132. A language model generates answer tokens;
its architecture does not define a call to that algorithm. Returning
142 on one input does not establish how the result was obtained.

Searching two textual formats, the authors find 17 of 2000 three-digit
additions and 2 of 2000 subtractions in training data. This limits an
explanation based on verbatim memorization of those found strings,
but does not exclude other formats or related training examples.

### Word transformations and new meanings

Section 3.9.2 reconstructs words after letter permutations or insertion
of irrelevant characters. Removing insertions preserves letter order;
unscrambling an anagram additionally requires choosing a word, which
may not be uniquely specified. Demonstrations place the corrupted word
left of `=` and its reconstruction right of it. The new input stops
after `=`. Demonstrations specify transformation type; word knowledge
helps choose a result.

In teaching example `h.e!a/r t`, removing insertions gives `heart`;
all five letters are already ordered. Permutation `htrae` permits
both `heart` and `earth`, so recovering the original needs another
criterion. These strings illustrate operations, not measured GPT-3 answers.

A cyclic shift preserves circular letter order but changes the start.
Two anagram tasks preserve different clues: A1 keeps first and last
letters; A2 keeps the first two and last two. Other letters shuffle.
Higher A2 accuracy thus concerns a task with more preserved information.

For 10,000 words in each of five tasks,
[Table 3.10](https://arxiv.org/pdf/2005.14165v4#page=23) gives few-shot
accuracies: cyclic shift 37.9%; anagrams A1/A2 15.1%/39.7%; insertion
removal 67.2%; reversed letters 0.44%. Prose and Table C.1 contain
slightly different values for some tasks; they must not silently be
combined into one row. Table C.1 uses a different demonstration seed.
Success at removing insertions does not imply mastery of all
letter operations: reversal is almost unsolved.

There is also a representation mismatch: the model reads BPE tokens,
while the operation concerns letters. A token can contain several
letters, so reversing token order is not reversing the word.
[CS324, Novel tasks](https://stanford-cs324.github.io/winter2022/lectures/capabilities/#novel-tasks)
also discusses defining an invented word and requesting a sentence
using it. This checks use of a meaning in current context, without
separate vocabulary or weight updates.

In Section 3.9.5, a human writes definitions and an example sentence
for the first word; GPT-3 continues subsequent sentences. Thus the
general task already has a demonstration, while each particular new
word is defined once. A few successful continuations show feasibility
but do not provide accuracy on a large independent novel-word sample.

One example defines `screeg` as waving a sword at something. The model
uses `screeghed at each other`, applying the definition in a new
sentence and forming a past tense. The authors call the phrase
plausible but slightly unnatural. This is an observation about one
continuation, not proof of durable word learning.

Grammar correction ([§3.9.6](https://arxiv.org/pdf/2005.14165v4#page=30))
likewise uses input–corrected-string pairs. One example changes
`eated` to `ate`. A house-rental example exposes a limitation:
the model may remove a statement about cheapness while repairing
word order. Grammatical acceptability and preservation of meaning
are different criteria; editing evaluation must check both.

### News generation and human evaluation

News prompts supply a title and subtitle after three completed articles.
These examples encourage a news article instead of, for example,
a discussion of the topic. Section 3.9.4 then evaluates people's
ability to distinguish generated text from human text, not news
factual accuracy.

For short texts, the study uses 25 newser.com titles and subtitles.
Original articles average 215 words; generated articles average 200.
Participants indicate confidence in human or machine authorship,
including a neutral “don't know.” Accuracy is computed for each
participant over non-neutral answers and then averaged. It is not
the fraction of all articles universally accepted as human-written.

The 175B mean is 52%, with a 95% confidence interval of 49–54%
([Table 3.11](https://arxiv.org/pdf/2005.14165v4#page=26)); an intentionally
degraded control generator gives 86%. A separate experiment with
12 longer Reuters articles and continuations averaging 498 words
also gives 52%, but with interval 48–57%
([Table 3.12](https://arxiv.org/pdf/2005.14165v4#page=27)).
These are two samples, not one measurement over all text genres.
The result concerns those participants, lengths, genre, and procedure.
It establishes neither truth of generated news, a universal authorship
test, nor the quality of a modern detector.

### Where scale did not solve the task

ANLI asks for entailment, contradiction, or neutrality between statements.
[Table C.1](https://arxiv.org/pdf/2005.14165v4#page=45) gives 175B few-shot
test accuracies of 36.8%, 34.0%, and 40.2% for three rounds. This is
a contamination-analysis reevaluation with a different demonstration
seed; Round 3 in Figure 3.9 uses dev and does not replace those rows.
Uniform random choice among three classes gives 33.3%. Smaller models
often remain near that level. LAMBADA success cannot be generalized
to all sentence-comparison and understanding tasks.

The scaling series shows lower language-model error alongside
improvements on several tasks, particularly few-shot. Tasks improve
unevenly. Figure 1.2 shows demonstration-count effects on insertion
removal; Figure 1.3 aggregates 42 accuracy measurements. The aggregate
does not replace separate protocols or tell whether a particular
instruction will work and how many examples it needs.

## Benchmark contamination and honest evaluation

At web scale, test questions, answers, or near copies may already
appear in training. Frozen weights exclude test-time fine-tuning,
not test exposure during pre-training.

Before training, the authors attempt removal of matches of 13
consecutive words, lowercased with punctuation ignored. Matches
and their surrounding 200-character windows are removed; short
remaining fragments are discarded. Very common 13-grams found in
more than ten training documents are ignored as probable boilerplate.
An implementation error leaves some overlaps, particularly in long
documents. Retraining is too expensive.

After training, each benchmark is filtered to a subset without found
matches and reevaluated. This uses $N$ chosen by example length, with
a maximum of 13 and minimum of 8 words for nonsynthetic tasks, not
one fixed 13-gram rule for all datasets. Matching covers the full
available corpus, including text the model might never have sampled.
Methods and limitations appear in
[§4 and Appendix C](https://arxiv.org/pdf/2005.14165v4#page=31).

For LAMBADA, 2944 of 5153 examples remain without found overlap.
Accuracy is 86.0% there and 86.4% on the full set (Table C.1).
The other 2209 overlapping examples have 86.9% accuracy.
The caption specifies a different demonstration seed from the main
results. Similar scores do not prove no memorization effect: filtering
may change difficulty distribution. Winograd has 132 schemas found
in pre-training and is marked with an asterisk; nearly fully overlapping
language-model benchmarks are omitted from reported results.

Text overlap also needs semantic inspection. A reading-comprehension
corpus may contain the source article without its test question and
answer, unlike a complete question–answer pair. Conversely, a paraphrased
test may evade exact $N$-gram matching.

In our own comparisons, prompts should be selected on dev, with test
used only after fixing template, example count, and scoring.
Several prespecified demonstration sets and orders measure prompt
sensitivity separately. Replacing unsuccessful examples after inspecting
test answers already adapts the experiment to test, even without
gradients. This is a requirement for our protocol, not a claim that
the paper evaluated every possible demonstration order.

## Limitations of base GPT-3

Pre-training optimizes corpus-token probability. It does not define
a separate objective to answer truthfully, verify arithmetic, or
follow every user request. A model may continue a discussion of a
question instead of answering briefly. Section 5 notes repetition,
contradiction, and loss of coherence in long text; Section 6 examines
stereotypes and other generative biases.

Few-shot examples can clarify format and change answers but provide
no built-in access to fresh facts. If a fact is absent from learned
regularities and current context, replacing `Q:` with another label
does not create its source. External documents require a retrieval
mechanism; arithmetic may be checked with an executable tool.

Context limits demonstration count and task length. Large parameter
counts increase every experiment's cost. The reported results concern
base GPT-3 from 2020, not automatically a later chat service,
instruction-tuned model, or tool-using system.

<span id="наследие-и-следующий-переход"></span>

## Transition to T5

GPT-3 tasks use one autoregressive continuation of a shared sequence.
Examples and question form its prefix; the answer is appended on
the right, with fixed weights during evaluation.
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5]]
also uses textual inputs and outputs but separates them architecturally:
an encoder builds bidirectional input states, and a decoder generates
an answer with cross-attention to them. The next chapter continues
an architectural comparison, not publication chronology: T5 preceded GPT-3.

Instruction tuning and preference learning are covered later in
post-training. They update weights and can coexist with demonstrations
in the current context.

## Sources and further reading

- [Brown et al., Language Models are Few-Shot Learners, v4](https://arxiv.org/abs/2005.14165v4): §2 architecture and evaluation; §3.1.2 LAMBADA; §§3.2–3.3 closed-book QA and translation; §§3.8–3.9 ANLI, synthetic tasks, qualitative experiments; §§4–5 overlap and limitations. Appendices A–C and G–H give recipe details, formats, and full results.
- [OpenAI publication repository, pinned revision](https://github.com/openai/gpt-3/tree/d7a9bb505df6f630f9bab3b30c889e52f22eb9ea): synthetic tasks, statistics, generation examples, and model card, not GPT-3 training code.
- [Stanford CS324 — Capabilities](https://stanford-cs324.github.io/winter2022/lectures/capabilities/): NLP task definitions and reductions to scoring or generation. This chapter's formulas and numbers follow the primary paper.
- [Jay Alammar — How GPT3 Works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/): original parameter, context, layer, and fine-tuning illustrations, CC BY-NC-SA 4.0. Figures explain mechanisms rather than reproduce measured GPT-3 answers.
- [[02 Areas/ML & DL/Papers/GPT 3.0|Earlier GPT-3 note]]: additional comparisons with each row's protocol identified separately.

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2]].
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5 and text-to-text]].
