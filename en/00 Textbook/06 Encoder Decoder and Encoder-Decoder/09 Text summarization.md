---
title: Text summarization
description: "Sentence selection, summary generation, and factual consistency, worked through on one short report."
type: textbook-chapter
status: canonical
locale: en
source_language: en
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/09 Суммаризация текста.md"
translation_status: current
canonical_revision: "2bcfcd3c6452747ff0ceca05ad5f274bd0a7c43b39fc9779edf7e6841246b969"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Text summarization tasks"
    anchor: sum-task
  - label: "Extractive summarization and BERTSum"
    anchor: sum-extractive
  - label: "Abstractive summarization and pointer-generators"
    anchor: sum-generation
  - label: "Measuring summary overlap with ROUGE"
    anchor: sum-rouge
  - label: "Summary coverage and faithfulness"
    anchor: sum-faithfulness
  - label: "Summarizing long documents"
    anchor: sum-long-documents
primary_sources:
  - https://aclanthology.org/D19-1387/
  - https://aclanthology.org/P17-1099/
  - https://aclanthology.org/2020.acl-main.703/
  - https://aclanthology.org/D18-1206/
  - https://aclanthology.org/W04-1013/
  - https://aclanthology.org/2020.acl-main.173/
  - https://aclanthology.org/2020.acl-main.450/
---

> The repair desk received 12 requests on Monday. By Wednesday it had
> closed 8 of those requests. On Thursday two of the remaining four
> went to a contractor. Opening hours did not change.

A weekly report can replace those sentences with one: “By Wednesday
the desk had closed 8 of the 12 requests it received on Monday; two
of the remaining four went to a contractor on Thursday.” The 12
refers to requests received and the 8 to requests closed. A summary
that swaps their roles is wrong even if it retains both numbers.

<a id="sum-task"></a>

## What belongs in the summary

The weekly report needs three figures: requests received, closed,
and passed to the contractor. A notice for customers would have
more use for the unchanged opening hours. The intended output also
shapes the training data. [XSum](https://aclanthology.org/D18-1206/)
uses a one-sentence description of what a news article is about.
CNN/DailyMail supplies several editorial highlights. Learning either
format does not teach a model what a repair desk needs from a weekly
report.

An extractive system can return only parts of the document, such
as two of its sentences. An abstractive system can combine information
from different parts in a new sentence. In
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/08 Машинный перевод|machine translation]], we try to preserve
the whole message. A summary must leave some of it out. The reference
summary records which information its author selected.

<a id="sum-extractive"></a>

## How BERTSum selects sentences

Taking the first two sentences is a useful baseline. It retains the
numbers received and closed but omits the handoff to the contractor.
A model that selects by content can first encode the whole report
and give each sentence its own vector.

On the right of the figure, every sentence starts with a separate
[CLS]. The bottom row shows the corresponding Transformer outputs
in blue. The segment embeddings alternate: the first and third
sentences receive $E_A$, and the second receives $E_B$. The context
of each [CLS] extends beyond its sentence; the bidirectional encoder
reads the document as a whole.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/summarization-2026-11-22/liu-bertsum-fig1.png|Original BERT on the left uses one CLS. BERTSum on the right uses one CLS per sentence, alternating segment embeddings A, B, A, and three contextual CLS outputs.]]

*Yang Liu and Mirella Lapata, [Text Summarization with Pretrained Encoders, Figure 1](https://aclanthology.org/D19-1387.pdf#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Diagram cropped from the paper; the text above explains how to read it.*

In BERTSumExt, extra Transformer layers let the sentence vectors
attend to one another. A linear layer and sigmoid turn each
updated vector into an inclusion score. The scores need not
sum to one: several sentences can appear in the same summary.

To train the classifier, we need an include-or-skip label for each
sentence. An editor may use words that never occur in the document,
so Liu and Lapata build approximate labels by greedily adding source
sentences that improve ROUGE against the reference. They train with
binary cross-entropy. At inference they rank sentences by score and
skip a candidate if it contains a trigram already present in the
selected text. Trigram blocking removes some verbatim repetition,
but two sentences can still state the same fact in different words.

Suppose our model gives the four sentences scores of 0.85, 0.93,
0.89, and 0.20. With a two-sentence budget, we return the second
and third sentences in document order. The contractor handoff is
now present, but the report no longer says when the requests came
in, and “those requests” has lost its explicit antecedent. A
sentence selector cannot fix that reference or rewrite the two
sentences into one.

<a id="sum-generation"></a>

## Copying words and generating new text

At each step of See, Liu, and Manning's pointer-generator, the decoder
forms two distributions. One is a softmax over a fixed vocabulary.
The other is attention over the positions in the source text.
When a word appears more than once, each occurrence contributes
its attention weight to that word's copy probability. If $g$ is
the generation weight,

$$
\begin{aligned}
p(w) &= g\,p_{\mathrm{vocab}}(w) + (1-g)c(w),\\
c(w) &= \sum_{i:x_i=w} a_i.
\end{aligned}
$$

The blue bars on the left are attention weights over source
positions. The green bars on the right give the vocabulary
distribution. The bars at the top combine them by word rather than
by position. For *Argentina*, the two routes both contribute.
The paper's example *2‑0* is outside the fixed vocabulary but can
still be copied.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/summarization-2026-11-22/see-pointer-generator-fig3.png|The pointer-generator weights the source attention by 1 minus p_gen and the vocabulary distribution by p_gen. Both distributions contribute to the final probability of Argentina.]]

*Abigail See, Peter J. Liu, and Christopher D. Manning, [Get To The Point, Figure 3](https://aclanthology.org/P17-1099.pdf#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original diagram and labels.*

The word *requests* appears twice in our report. Suppose those
positions receive attention weights 0.20 and 0.15, and the
vocabulary assigns *requests* a probability of 0.05. With $g=0.60$,
the total is $0.60\cdot0.05+0.40\cdot(0.20+0.15)=0.17$.
The model recomputes $g$ from the context vector, decoder state,
and decoder input at each step. The training loss uses the final
probability of the target word; there is no separate copy label.

The same paper adds coverage, the sum of earlier attention weights
at each source position. If a position received 0.20 and 0.30 in
earlier steps, it has coverage 0.50. At a new step with attention
0.25, that position contributes $\min(0.25,0.50)=0.25$ to the
coverage loss. The attention mechanism also takes the accumulated
weights as input. Returning to the same position becomes costly,
but the model can still attach 8 and 12 to the wrong actions, or
repeat a fact after reading it elsewhere in the document.

[BART](https://aclanthology.org/2020.acl-main.703/) takes a
different approach. In pretraining, the encoder receives a corrupted
text and the decoder learns to reconstruct the original sequence.
For example, text infilling can replace a span with one mask token.
The figure shows that the encoder input and the decoder target need
not align position for position. The full, uncorrupted text is the
target at this stage.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/summarization-2026-11-22/lewis-bart-fig1c.png|The bidirectional encoder on the left takes a corrupted sequence. The autoregressive decoder on the right reconstructs A, B, C, D, E in order.]]

*Mike Lewis et al., [BART, Figure 1(c)](https://aclanthology.org/2020.acl-main.703.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Only the BART panel is shown.*

During summarization fine-tuning, the encoder sees the uncorrupted
document, and the target tokens come from an editor's shorter summary.
The training step gives the decoder the preceding target tokens.
At inference it must continue its own output, as in
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/08 Машинный перевод#mt-objective|translation]].
BART has no separate output for copying a word at position $i$.
If summaries in its training data usually mention only the 12
requests received, reconstruction pretraining does not make the
number closed a required part of the output.

<a id="sum-rouge"></a>

## Why ROUGE can prefer the wrong answer

ROUGE-N counts matching $n$-grams against a reference.
[Chin-Yew Lin's paper](https://aclanthology.org/W04-1013/)
defines ROUGE-N as recall: divide the overlap by the number of
reference $n$-grams. We can also report precision relative to the
candidate length and their F1. Comparisons should name the metric
variant and tokenization. ROUGE-2 counts adjacent pairs of words.
ROUGE-L uses the longest common subsequence, which preserves order
but can skip words.

Take a reference that covers only the first two facts, leaving out
the handoff: *The desk received 12 requests and closed 8 by
Wednesday*. After lowercasing and stripping punctuation, it has
ten words.

| Candidate | Words matched | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| The desk received 12 requests and closed 12 by Wednesday | 9 | 9/10 | 9/10 | 0.900 |
| Of the 12 requests, 8 were resolved by Wednesday | 6 | 6/10 | 6/9 | 0.632 |

Repeated words only count up to the occurrence count in the
reference. The first candidate uses *12* twice, but only one of
those instances matches. The second matches *the*, *12*, *requests*,
*8*, *by*, and *Wednesday*. The faithful paraphrase scores lower
because it uses a different sentence structure. A gain in ROUGE on
a benchmark does not tell us whether numbers, dates, or causal
relations became more accurate.

<a id="sum-faithfulness"></a>

## Checking claims against the document

We can evaluate the repair summary by listing the facts we expect
it to cover and the claims it actually makes. The opening summary
makes three claims, all supported by the report. The version with
12 closures makes two claims, only one of them supported. The
denominator for coverage is the number of expected facts. The
denominator for the fraction of supported claims is the number of
claims made. A summary with one correct claim can have perfect
precision while leaving out most of the report.

In [Maynez et al.'s annotation scheme](https://aclanthology.org/2020.acl-main.173/),
“closed 12” is an *intrinsic hallucination*: it distorts something
in the source. “The contractor worked for free” is an *extrinsic
hallucination*: the report cannot verify it. That does not mean the
contractor was in fact paid. Checking faithfulness to a source does
not establish what happened outside it.

[QAGS](https://aclanthology.org/2020.acl-main.450/) generates questions
from the summary, then asks the same extractive QA model to answer
each question once from the source and once from the summary. It
compares answers with token-level F1 and averages the scores.
We can illustrate the comparison with two questions of our own:

| Question | Source answer | Summary answer | F1 |
| --- | --- | --- | ---: |
| How many requests came in? | 12 | 12 | 1 |
| How many requests were closed by Wednesday? | 8 | 12 | 0 |

The mean is 0.5 for these two hand-written questions. It is not a
score computed by QAGS on the example. The comparison catches the
wrong number only if the question is asked and the QA model
extracts both answers correctly. It need not catch the missing
contractor handoff, since the summary makes no claim about that
event to ask a question about.

<a id="sum-long-documents"></a>

## When the document exceeds the context window

Suppose the Monday entry and the Thursday entry appear on different
pages. Splitting the report between them may leave “two of the remaining
four” without the record of which requests remained. Summarizing each
page reduces the length, but a second pass can recover an omitted
connection only by going back to the underlying text.

In a report intended for review, keep a reference to the source
next to each claim. “Two of the remaining four went to a contractor”
requires the handoff entry. Linking the remaining requests to the
Monday group may also require the preceding record. If the task is
to recover the state of the requests on a particular date, select
passages that contain dates and changes in status, not just those
that mention “request.”

<details>
<summary>Run the numerical examples in Python</summary>

```python
from collections import Counter
from math import isclose


def overlap(a, b):
    return sum((Counter(a) & Counter(b)).values())


def rouge1(candidate, reference):
    candidate, reference = candidate.split(), reference.split()
    hits = overlap(candidate, reference)
    precision = hits / len(candidate)
    recall = hits / len(reference)
    f1 = 2 * hits / (len(candidate) + len(reference))
    return precision, recall, f1


scores = [0.85, 0.93, 0.89, 0.20]
selected = sorted(
    sorted(range(1, 5), key=lambda i: scores[i - 1], reverse=True)[:2]
)
assert selected == [2, 3]

g = 0.60
requests_probability = g * 0.05 + (1 - g) * (0.20 + 0.15)
assert isclose(requests_probability, 0.17)
assert min(0.25, 0.20 + 0.30) == 0.25

reference = "the desk received 12 requests and closed 8 by wednesday"
wrong = "the desk received 12 requests and closed 12 by wednesday"
faithful = "of the 12 requests 8 were resolved by wednesday"
assert rouge1(wrong, reference) == (0.90, 0.90, 0.90)
assert isclose(rouge1(faithful, reference)[2], 12 / 19)

pairs = [("12", "12"), ("8", "12")]
answer_f1 = sum(
    2 * overlap(a.split(), b.split()) / (len(a.split()) + len(b.split()))
    for a, b in pairs
) / len(pairs)
assert answer_f1 == 0.50
print(selected, round(requests_probability, 2), round(12 / 19, 3), answer_f1)
```

</details>

## Sources

- Yang Liu and Mirella Lapata, [Text Summarization with Pretrained Encoders](https://aclanthology.org/D19-1387/), §§ 3.1–3.2 and 4: sentence representations, labels, and decoding.
- Abigail See, Peter J. Liu, and Christopher D. Manning, [Get To The Point](https://aclanthology.org/P17-1099/), §§ 2.2–2.3: the mixture distribution and coverage.
- Mike Lewis et al., [BART](https://aclanthology.org/2020.acl-main.703/), §§ 2–3: corruption functions and fine-tuning for generation.
- Shashi Narayan, Shay B. Cohen, and Mirella Lapata, [Extreme Summarization](https://aclanthology.org/D18-1206/), § 3: the XSum one-sentence target.
- Chin-Yew Lin, [ROUGE](https://aclanthology.org/W04-1013/), §§ 2–3: $n$-gram and subsequence overlap.
- Joshua Maynez et al., [On Faithfulness and Factuality in Abstractive Summarization](https://aclanthology.org/2020.acl-main.173/), § 2: annotating factual errors.
- Alex Wang, Kyunghyun Cho, and Mike Lewis, [QAGS](https://aclanthology.org/2020.acl-main.450/), §§ 3–4: asking the same questions of the source and summary.

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/08 Машинный перевод|Machine translation]].
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/10 Ответы на вопросы|Question answering]].
