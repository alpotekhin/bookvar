---
title: Question answering
description: "Passage retrieval, BERT span extraction, abstention when evidence is missing, and generation from several sources."
type: textbook-chapter
status: canonical
locale: en
source_language: en
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/10 Ответы на вопросы.md"
translation_status: current
canonical_revision: "09ec2334e7f677d69c11d2a1ee58b54c305f119a575313acd542ebfeb547d8d1"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Question answering over documents"
    anchor: qa-task
  - label: "Passage retrieval for question answering and DPR"
    anchor: qa-retrieval
  - label: "Extracting answer spans with BERT"
    anchor: qa-span
  - label: "Abstention in SQuAD 2.0"
    anchor: qa-no-answer
  - label: "Exact Match and token F1 for question answering"
    anchor: qa-metrics
  - label: "Generative question answering and Fusion-in-Decoder"
    anchor: qa-generation
primary_sources:
  - https://aclanthology.org/P17-1171/
  - https://aclanthology.org/2020.emnlp-main.550/
  - https://aclanthology.org/N19-1423/
  - https://aclanthology.org/D16-1264/
  - https://aclanthology.org/P18-2124/
  - https://rajpurkar.github.io/SQuAD-explorer/
  - https://github.com/google-research/bert/blob/master/run_squad.py
  - https://aclanthology.org/2020.emnlp-main.437/
  - https://aclanthology.org/2021.eacl-main.74/
---

How many requests were forwarded to the contractor on Thursday?
The workshop's log says, “Staff sent the contractor two requests
Thursday.” A reader can take “two requests” straight from the log.
If the question is asked through a search of all the workshop's
records, the system must find that entry first. Search might miss
the entry, and even the right document might not name the contractor.

<a id="qa-task"></a>

## The question, the passage, and the answer

In reading comprehension, the system gets a question and a
passage. In [SQuAD 1.1](https://aclanthology.org/D16-1264/) the
answer is a contiguous span in that passage. The annotation
records the answer text and its offset. The offset distinguishes
between multiple occurrences of the same phrase. “Two requests”
in Thursday's entry answers our question; the same phrase in
last week's log does not.

In open-domain QA, the relevant passage must be found in a
collection. [DrQA](https://aclanthology.org/P17-1171/) first
retrieved Wikipedia articles, then extracted an answer from
their text. To explain an error, we can check the two stages
separately: did search find the right text, and did the reader
extract the right answer from it?

<a id="qa-retrieval"></a>

## Finding the relevant entry

Suppose the archive contains these three passages:

| Passage | Text |
| --- | --- |
| A. Procedure | “Request intake on Thursday. Before forwarding a request to the contractor, check the address.” |
| B. Dispatch entry | “Staff sent the contractor two requests Thursday.” |
| C. Closure entry | “By Wednesday the workshop had closed eight requests from the same batch.” |

Matching terms help a fragment score well with BM25. The
procedure contains words from the question but gives no count.
If A ranks first and search returns one passage, the reader
never sees B. If B ranks second, returning two passages gives
the reader access to the evidence.

[Dense Passage Retrieval](https://aclanthology.org/2020.emnlp-main.550/)
uses separate encoders for questions and passages. Passage vectors
can be computed and indexed ahead of time. At search time DPR
encodes the question and looks up passages with large dot
products. During training, the score of a question paired with
a supporting passage competes with the scores of negative passages
in a softmax. Among the negatives are high-ranking BM25 results
that share terms with the question but do not contain its answer.

Take two-dimensional vectors for the example. The question has
$v_q=(1,0.50)$ and B has $v_B=(0.30,0.84)$. Their dot product
is $1\cdot0.30+0.50\cdot0.84=0.72$.

| Passage | Vector | Retrieval score |
| --- | --- | ---: |
| A | (0.40, 0.20) | 0.50 |
| B | (0.30, 0.84) | 0.72 |
| C | (0.28, 0.36) | 0.46 |

B now ranks first. The score of 0.72 orders it among the
candidates; it is not the probability of answering correctly.

If the supporting passages for three questions rank 2, 1, and
3, search hits the first result in $1/3$ of cases and one of
the first two in $2/3$. In the DPR plot, the vertical axis
is the share of questions whose reference answer string appears
in at least one of the first $k$ passages. Even that is an
approximation of supporting evidence: “two requests” might
refer to a different week.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/question-answering-2026-11-22/karpukhin-dpr-fig1.png|DPR on Natural Questions: the horizontal axis is the number of retrieved passages k. The vertical axis is top-k retrieval accuracy. Curves compare BM25 with DPR at different training set sizes.]]

*Vladimir Karpukhin et al., [Dense Passage Retrieval for Open-Domain Question Answering, Figure 1](https://aclanthology.org/2020.emnlp-main.550.pdf#page=5), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Full plot cropped from the paper.*

Read across one curve to see how the same retriever changes
with $k$. Compare curves at a fixed $k$ to compare BM25
with different DPR models. These are retrieval results on
the Natural Questions development set. The reader can still
answer incorrectly after retrieval succeeds. Passing it more
text also raises the cost of reading and gives it more
similar-looking spans to choose from.

<a id="qa-span"></a>

## How BERT finds the start and end

BERT receives the question and passage in a single sequence:
[CLS], question tokens, [SEP], passage tokens, [SEP]. Unlike
DPR's separate encoders, the reader lets the representation
of each passage token depend on the question. The same token
“Thursday” can have different features for “Which day?” and
“How many requests?”.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/question-answering-2026-11-22/devlin-bert-qa-fig4c.png|BERT for SQuAD 1.1: a SEP token separates the question from the passage. Outputs over the passage positions supply start and end scores.]]

*Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, [BERT, Figure 4(c)](https://aclanthology.org/N19-1423.pdf#page=15), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The question answering panel only.*

In the original BERT formulation, two learned vectors score
each output state $h_i$: $a_i=w_s^\top h_i$ for the start
and $b_i=w_e^\top h_i$ for the end. During training, each
set of scores gets its own softmax. For annotated boundaries
$i^\star$ and $j^\star$, the loss is

$$
\mathcal L=-\log p_s(i^\star)-\log p_e(j^\star).
$$

At inference, we score pairs. Within one window, adding the
logits $a_i+b_j$ ranks pairs in the same order as adding
the two log probabilities: the normalization constants do
not depend on the pair. A valid pair must be in the passage,
have $i\le j$, and have length $j-i+1\le L_{\max}$. An
answer cannot start in the question or join boundaries from
different windows.

In passage B, count one token per word and number the passage
separately from the question. The first four words are “Staff
sent the contractor.” The table gives the three strongest
candidates; other positions score lower.

| Position | Word | Start $a_i$ | End $b_i$ |
| ---: | --- | ---: | ---: |
| 5 | two | 2.8 | 1.6 |
| 6 | requests | 1.2 | 2.4 |
| 7 | Thursday | 3.1 | 0.5 |

The highest-scoring start is 7 and the highest-scoring end
is 6. We must discard $(7,6)$. With a three-token limit,
$(5,6)$ scores $2.8+2.4=5.2$, while $(5,5)$ scores 4.4.
“Thursday” by itself scores 3.6. The resulting pair $(5,6)$
returns “two requests.”

A real model works with subtokens. The selected token indices
must map back to character offsets to return the original
string. Overlapping windows help on long documents: without
overlap, the start and end of a short answer can fall on
opposite sides of a window boundary.

<a id="qa-no-answer"></a>

## When the passage does not contain an answer

Change the question to “What was the contractor's name?” B
contains the word “contractor,” but no name. A system forced
to extract a span may return the closest-looking word anyway.
[SQuAD 2.0](https://aclanthology.org/P18-2124/) adds plausible
questions of this kind, written by annotators so that the
paragraph has no answer.

In BERT for SQuAD 2.0, [CLS] denotes no answer. In training,
both labels point to [CLS] for an unanswerable example. At
inference we compare the sum of [CLS] logits, $s_\varnothing$,
with the score of the best valid span, $s_{\mathrm{span}}$.
The [authors' implementation](https://github.com/google-research/bert/blob/master/run_squad.py)
returns an empty answer if $s_\varnothing-s_{\mathrm{span}}>\tau$.
The threshold is tuned on held-out data. Increasing it makes
the system return a span more often, including for questions
the passage cannot answer.

| Question about B | Best span | Span score | [CLS] score | Difference |
| --- | --- | ---: | ---: | ---: |
| How many requests were sent? | two requests | 5.2 | 2.0 | −3.2 |
| What was the contractor's name? | contractor | 3.3 | 4.1 | 0.8 |

At $\tau=0.5$ the first answer stays and the second becomes
empty. At $\tau=1$ the system would return “contractor.” These
are illustrative values for one window, not confidence
percentages. With several windows, the implementation also
needs a rule to compare span and null scores across windows.

An empty answer tells us what the system decided given the
passages it saw. It cannot establish that the name is absent
from the entire archive. First check whether the retrieved results
contained a passage that answers the question. If not, give that
passage to the reader to see whether fixing retrieval would suffice.
If the passage was already there, inspect reading and answer selection.

<a id="qa-metrics"></a>

## What Exact Match and F1 measure

After normalization, SQuAD's Exact Match is one if the
prediction matches any reference exactly. The English
[evaluation script](https://rajpurkar.github.io/SQuAD-explorer/)
ignores case, punctuation, articles, and excess whitespace.
It does not account for Russian inflection; a different
collection needs its own normalization rules.

Suppose the only reference is “two requests.” “Two” does not
match it exactly, but its one token appears in the reference.
Precision is $1/1$, recall is $1/2$, and F1 is $2/3$. “Two”
could be a perfectly natural answer in conversation. The
difference in score here follows from how we chose the reference.

| Prediction | Reference | Exact Match | F1 |
| --- | --- | ---: | ---: |
| two requests | two requests | 1 | 1 |
| two | two requests | 0 | 0.667 |
| eight | two requests | 0 | 0 |
| empty | empty | 1 | 1 |

With several references, each metric takes the best score
before averaging over questions. SQuAD 2.0 also reports
answerable and unanswerable subsets separately. A combined
average can hide a reader that guesses too often or abstains
too often. When there is a retrieval stage, measure evidence
hits as well as final answer quality.

<a id="qa-generation"></a>

## An answer that needs two entries

A generative model can answer a question without any documents.
In [closed-book QA](https://aclanthology.org/2020.emnlp-main.437/) it relies on information learned during training.
It might have learned a well-known date, but the question alone
cannot tell it what the workshop recorded on Thursday. To answer
a question about the log, give the model access to the log.

“How many requests had been closed by Wednesday, and how many
were sent to the contractor on Thursday?” requires B and C.
Neither has a contiguous span that answers both parts. We
can combine them: “Eight were closed by Wednesday [C]; two
went to the contractor on Thursday [B].”

[Fusion-in-Decoder](https://aclanthology.org/2021.eacl-main.74/)
pairs the question with each retrieved passage separately.
The authors also include the passage title. The same encoder
processes each input independently. The “concat” box joins
their sequences of hidden states, and the decoder can attend
to all of them through cross-attention. It predicts the answer
one token at a time, as in
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/09 Суммаризация текста#sum-generation|the chapter on summarization]].

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/question-answering-2026-11-22/izacard-fid-fig2.png|Question-plus-passage inputs pass independently through a shared encoder. Their hidden states are concatenated for the decoder, which produces the answer.]]

*Gautier Izacard and Edouard Grave, [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering, Figure 2](https://aclanthology.org/2021.eacl-main.74.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original diagram.*

This avoids encoder self-attention over tokens from all
passages at once. The decoder has more states to read as
passages are added, so cross-attention gets more expensive.
Access to both entries does not guarantee the decoder takes
eight from C and two from B. The labels in our example point
to the supporting passages; the original model learned to
produce the answer string without such citations. The
[[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер|RAG chapter]] covers checking claims and their sources.

<details>
<summary>Check the retrieval, span, and metric calculations in Python</summary>

~~~python
from collections import Counter
from math import isclose


def dot(q, p):
    return sum(a * b for a, b in zip(q, p))


q = (1.0, 0.50)
passages = {"A": (0.40, 0.20), "B": (0.30, 0.84), "C": (0.28, 0.36)}
scores = {key: round(dot(q, vector), 2) for key, vector in passages.items()}
assert scores == {"A": 0.50, "B": 0.72, "C": 0.46}
assert sorted(scores, key=scores.get, reverse=True) == ["B", "A", "C"]
ranks = (2, 1, 3)
assert [sum(r <= k for r in ranks) for k in (1, 2)] == [1, 2]


def select_span(start, end, threshold=0.5):
    # Position 0 encodes no answer. Other keys are passage positions.
    score, span = max(
        (a + b, (i, j))
        for i, a in start.items() for j, b in end.items()
        if 0 < i <= j and j - i + 1 <= 3
    )
    difference = start[0] + end[0] - score
    return (None if difference > threshold else span), score, difference


span, score, delta = select_span(
    {0: 1.0, 5: 2.8, 6: 1.2, 7: 3.1},
    {0: 1.0, 5: 1.6, 6: 2.4, 7: 0.5},
)
assert span == (5, 6) and isclose(score, 5.2) and isclose(delta, -3.2)
missing_ru = ({0: 2.1, 7: 1.4}, {0: 2.0, 7: 1.9})
missing_en = ({0: 2.1, 4: 1.4}, {0: 2.0, 4: 1.9})
for missing, distractor in ((missing_ru, (7, 7)), (missing_en, (4, 4))):
    span, score, delta = select_span(*missing)
    assert span is None and isclose(score, 3.3) and isclose(delta, 0.8)
    assert select_span(*missing, threshold=1)[0] == distractor


def token_f1(answer, gold):
    if not answer or not gold:
        return float(answer == gold)
    hits = sum((Counter(answer) & Counter(gold)).values())
    return 2 * hits / (len(answer) + len(gold))


assert isclose(token_f1(["two"], ["two", "requests"]), 2 / 3)
assert token_f1(["two", "requests"], ["two", "requests"]) == 1
assert token_f1([], []) == 1
print("top B; span 5–6; null margin 0.8; partial F1 0.667")
~~~

</details>

## Sources

- Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes, [Reading Wikipedia to Answer Open-Domain Questions](https://aclanthology.org/P17-1171/), §§ 3.1–3.2: separate retrieval and reading.
- Vladimir Karpukhin et al., [Dense Passage Retrieval](https://aclanthology.org/2020.emnlp-main.550/), §§ 3 and 5: dual encoders, negatives, and retrieval evaluation.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, [BERT](https://aclanthology.org/N19-1423/), §§ 4.2–4.3 and [implementation](https://github.com/google-research/bert/blob/master/run_squad.py): boundaries, [CLS], and thresholding.
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang, [SQuAD](https://aclanthology.org/D16-1264/), § 6.1; Pranav Rajpurkar, Robin Jia, and Percy Liang, [SQuAD 2.0](https://aclanthology.org/P18-2124/), §§ 2–4: answer spans, unanswerable questions, and evaluation.
- Gautier Izacard and Edouard Grave, [Fusion-in-Decoder](https://aclanthology.org/2021.eacl-main.74/), § 3: independent encoding and joint reading in the decoder.

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/09 Суммаризация текста|Text summarization]].
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1]].
