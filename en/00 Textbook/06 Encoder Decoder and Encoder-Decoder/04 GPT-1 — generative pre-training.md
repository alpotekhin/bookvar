---
title: GPT-1 — generative pre-training
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources: [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf]
---

# GPT-1: generative pre-training

Before 2018, a strong language-processing system was usually built for one
particular task. Text classification, question answering, and reasoning over a
pair of sentences used different input formats and often different models.
Labeled examples were scarce for each task, while unlabeled books and web pages
were plentiful. GPT-1 tested a practical way to transfer regularities learned
from a large unlabeled corpus into several small labeled datasets: first train
one model to continue text, then adapt all of its weights to the target task.

The order matters. GPT-1 was not yet a general assistant that could be steered
with a written instruction. Every downstream task still required labeled data,
a separate fine-tuning run, and a linear output head. The core architecture and
nearly all parameters, however, were shared. This established a reproducible
`pre-training → fine-tuning` recipe that separated expensive language learning
from comparatively inexpensive specialization.

## One architecture for both stages

GPT-1 is a stack of 12 causal Transformer blocks with no encoder and no
cross-attention. Its hidden width is 768, attention is split across 12 heads,
and the inner feed-forward layer has width 3072, for roughly 117 million
parameters in total. Learned positional embeddings are added to token
embeddings. The context is limited to 512 tokens, and the vocabulary is built
with BPE.

For a token sequence $u_1,\ldots,u_n$, pre-training maximizes

$$
L_1(U)=\sum_i \log P(u_i\mid u_{i-k},\ldots,u_{i-1};\Theta).
$$

The causal mask prevents position $i$ from using $u_{i+1}$ or any later token.
Token IDs enter as a tensor of shape `[B,T]`; after embedding, the representation
has shape `[B,T,768]`; the output projection produces vocabulary scores of shape
`[B,T,V]`. All positions in a training segment are processed in parallel, but
each loss term predicts the next token from left context only.

The term *decoder-only* does not mean that the model consumes a representation
from a separate encoder—there is no encoder. GPT-1 retains masked self-attention
and the feed-forward network from the original Transformer decoder while
removing cross-attention. The same computation can therefore score text
probability and produce features for a downstream task.

## Stage one: what BooksCorpus teaches

GPT-1 was pre-trained on BooksCorpus: about seven thousand unpublished books,
or roughly 800 million words. Long, connected prose matters for more than its
size. Characters, events, and themes persist across many sentences, so
next-token prediction rewards the use of dependencies beyond a local phrase.

One training step can be read in sequence:

1. A book fragment is split into BPE tokens and assigned positions.
2. The causal mask leaves position $t$ only the prefix $u_{\le t}$.
3. Twelve blocks turn every prefix into a contextual state.
4. The output matrix defines a distribution for the next token.
5. Cross-entropy over all positions updates the Transformer, embeddings, and
   output projection.

There are no topic, sentiment, or sentence-relation labels at this stage. The
supervision comes from the text itself. It is therefore more accurate to speak
of parameters that have accumulated transferable syntactic, semantic, and
discourse features than of a finished solver for twelve benchmarks.

## Stage two: express each task as a sequence

The main engineering problem in transfer was to present structurally different
inputs to one causal Transformer. Instead of building a new encoder for every
benchmark, the authors serialized the fields using special tokens.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/gpt1-figure1.png]]

*Left: the shared pre-trained Transformer and output head. Right: four ways to
serialize labeled tasks. Source: Radford et al., [Improving Language
Understanding by Generative Pre-Training, Figure
1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
2018.*

For classification, an `Extract` token follows the text and its final hidden
state is passed to a linear head. For textual entailment, the premise and
hypothesis are joined by a delimiter. A similarity pair is read in both orders
so that a directional model does not make a symmetric task depend on order. In
multiple-choice tasks, the context is joined with each candidate in turn and
the candidate scores are normalized together.

Let $h_l^m$ be the state of the final service token after the last layer. The
classification head computes

$$
P(y\mid x^1,\ldots,x^m)=\operatorname{softmax}(h_l^mW_y).
$$

Fine-tuning updates the entire Transformer, not only $W_y$. The authors also
retain an auxiliary language-modeling objective on the downstream text:

$$
L_3(C)=L_2(C)+\lambda L_1(C),
$$

where $L_2$ is the likelihood of the correct label and $L_1$ is next-token
likelihood. This term supplies additional supervision and keeps the
representation closer to what was learned from books. In the paper it helped
particularly on larger labeled datasets; that observation is not a universal
guarantee for every fine-tuning regime.

## Reading the key experiments

The paper evaluates twelve datasets covering textual entailment, question
answering, semantic similarity, classification, and commonsense reasoning. The
pre-trained model improved the previous best result on nine of the twelve.
Gains were especially notable on RACE and Story Cloze, where the model must
connect several parts of a context rather than recognize one keyword.

The control experiments matter more than the leaderboard positions. The same
architecture performed much worse without pre-training. The Transformer
transferred better than a comparable recurrent model. Performance improved as
higher layers of the pre-trained network were used, so the transfer cannot be
explained by token embeddings alone. The auxiliary language-modeling objective
made a smaller but measurable contribution.

The authors also tracked behavior without gradient-based adaptation. Simple
scores on some downstream tasks improved during pre-training, suggesting that
language modeling itself creates useful capabilities. This was not GPT-1's
primary interface, however. The paper's main results use supervised
fine-tuning; describing them as zero-shot prompting would be historically
incorrect.

## What GPT-1 did not solve

Left-to-right context is natural for generation but is not always ideal for
feature extraction: an early token representation cannot use words to its
right. Later in 2018, BERT demonstrated the strength of bidirectional masked
language-model pre-training for understanding tasks. GPT-1's 512-token context
also cannot read a long book at once, and BooksCorpus covers a narrow slice of
genres and languages.

Every new task still requires a labeled dataset, hyperparameter selection, a
separate set of fine-tuned weights, and its own output head. `Delim` and
`Extract` unify the computation graph but do not provide a natural-language
interface. Finally, accurate classification is not verified world knowledge:
the loss rewards statistical prediction of text, not the truth of statements.

## Legacy and the transition to GPT-2

GPT-1 showed that one causal Transformer could first learn from raw text and
then transfer to varied tasks with minimal architectural change. This made the
scale of pre-training a resource in its own right: new abilities could come
from better general parameters, not only from new task-specific architectures.

GPT-1's limitation immediately suggests the next question. If web text already
contains translations, questions, answers, and summaries, must the weights be
updated for every task? [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2]] will increase the model and corpus diversity and test whether a text prefix itself can specify the task.

## Sources and further reading

- [Radford et al., Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — the primary paper: architecture, objectives, input transformations, results, and ablations.
- [OpenAI, Improving language understanding with unsupervised learning](https://openai.com/index/language-unsupervised/) — publication page and accompanying material.
- [[02 Areas/ML & DL/Papers/GPT 1.0|Local GPT-1 note]] — task table and historical context.
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|Previous chapter: BERT, RoBERTa, and DeBERTa]].
