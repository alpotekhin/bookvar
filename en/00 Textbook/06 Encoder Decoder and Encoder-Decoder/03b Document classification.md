---
title: Document classification
description: "Follow one classification task from TF-IDF and logistic regression through fastText and BERT."
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03b Классификация документов.md"
canonical_revision: "ce4198c891a166efc1b940abd3d677f01457b1845d392a05b6d65d5f861a9644"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "TF-IDF and bigrams for text classification"
    anchor: doccls-tfidf
  - label: "Logistic regression and multinomial naive Bayes"
    anchor: doccls-linear
  - label: "fastText: averaging feature vectors"
    anchor: doccls-fasttext
  - label: "BERT: document classification with CLS"
    anchor: doccls-bert
  - label: "Evaluating a text classifier and choosing a threshold"
    anchor: doccls-evaluation
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/4.pdf
  - https://scikit-learn.org/1.9/modules/feature_extraction.html#tfidf-term-weighting
  - https://aclanthology.org/E17-2068/
  - https://aclanthology.org/N19-1423/
  - https://d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html#fig-bert-one-seq
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03c Извлечение информации из документов]]"
---

“Great screen, awful battery” and “Awful screen, great battery” contain
the same words. If the task is to classify sentiment about the
screen, the correct labels are positive and negative, respectively.
A bag of words gives both reviews the same features. A classifier
cannot distinguish them by changing its weights alone. In the
previous chapter, labels belonged to words and spans. Here each
review gets one label, and the annotation instructions give it
a particular meaning: sentiment about the screen, not about the
device as a whole.

<a id="doccls-tfidf"></a>

## What a bag of words loses

Start with four toy reviews. Lowercasing and removing punctuation
gives the following tokens:

| Review | Tokens | Screen sentiment |
| --- | --- | --- |
| 1 | `great screen awful battery` | positive |
| 2 | `awful screen great battery` | negative |
| 3 | `bright screen weak speakers` | positive |
| 4 | `dim screen loud speakers` | negative |

A unigram bag has one coordinate for each of the nine known words.
Its value counts occurrences of that word in a review. TF-IDF
reweights those counts. A term that appears in fewer documents
gets a larger multiplier, but that multiplier does not encode
which class the term predicts.

In [scikit-learn's smoothed convention](https://scikit-learn.org/1.9/modules/feature_extraction.html#tfidf-term-weighting),
$N$ is the number of documents, $\operatorname{df}(t)$ is the
number of documents that contain term $t$, and $c_t$ is the
term's count in the current document:

$$
\begin{aligned}
\operatorname{idf}(t)&=1+\log\frac{1+N}{1+\operatorname{df}(t)},\\
v_t&=c_t\,\operatorname{idf}(t),\\
x_t&=\frac{v_t}{\|v\|_2}.
\end{aligned}
$$

`screen` appears in all four reviews, so its IDF is 1. The terms
`great`, `awful`, and `battery` each appear in two, giving
$q=1+\log(5/3)\approx1.511$. Before normalization, the first
review has four nonzero weights $q,1,q,q$ in word order.
The norm is $\sqrt{1+3q^2}\approx2.801$. After division, the weight
for `screen` is about $0.357$; the other three weights are $0.539$
each. The second review has the same coordinates and values.
IDF and normalization cannot recover order after counting has
discarded it.

Bigrams retain the order of adjacent words. The first review
has `great screen`, while the second has `awful screen`. Using
only bigrams, each of these reviews has three active features.
Each feature appears in one document, giving
$1+\log(5/2)\approx1.916$ before normalization and $1/\sqrt{3}$
afterward. Now the vectors differ. A classifier can keep both
unigrams and bigrams: the former carry information to new combinations;
the latter distinguish known phrases. Bigrams do not connect distant
words. `the screen was far from great` has neither `great screen`
nor `awful screen`.

Fit the vocabulary and IDF on the training documents, then freeze
them. An unseen bigram does not add a coordinate at prediction time.
If no features survive, the vector is zero and a linear model scores
the input using only its intercept. In cross-validation, fit the
vectorizer inside each training fold so that the held-out documents
do not contribute to the statistics.

<a id="doccls-linear"></a>

## From features to a probability

For $y=1$, “the writer liked the screen,” logistic regression
uses $z=w^\top x+b$ and $p=\sigma(z)$. Suppose the only nonzero
weights are $+1.2$ for `great screen` and $-1.2$ for
`awful screen`, with $b=0.1$. Use the bigram-only vectors above.

| Review | Logit | $p(y=1\mid x)$ |
| --- | --- | ---: |
| 1 | $0.1+1.2/\sqrt{3}\approx0.793$ | 0.688 |
| 2 | $0.1-1.2/\sqrt{3}\approx-0.593$ | 0.356 |

The weights are illustrative. To see how a model could learn
their signs, start with weights and intercept at zero. At $p=0.5$,
the derivative of binary cross entropy with respect to a weight
is $(p-y)x_t$. The positive review contributes
$-0.5/\sqrt{3}\approx-0.289$ to the gradient for `great screen`,
so gradient descent raises that weight. The negative review
contributes $+0.289$ for `awful screen`, so the weight falls.
A penalty on large weights limits the effect of rare bigrams
that happen to occur with just one label.

With unigrams alone, $x_1=x_2$ forces the model to assign the same
probability to opposite labels. The pair's average cross entropy
is minimized at $p=0.5$. Any deterministic decision from that
vector alone gets at least one of the two reviews wrong. The
representation sets that limit before optimization begins.

### Naive Bayes on the same reviews

Multinomial naive Bayes counts words within each class instead of
adjusting weights to reduce classification loss. With add-one
smoothing, the probability of `bright` in the positive class is
$(1+1)/(8+9)=2/17$. It is $1/17$ in the negative class: `bright`
never appears there, but smoothing keeps its probability above zero.
There are eight tokens per class, nine words in the vocabulary,
and two reviews per class, so the class priors are equal.

In a comparison of log likelihoods, `bright` adds $\log2$ in favor
of the positive class. `screen` occurs twice in each class and
makes no difference to the comparison. Add the contributions
over the words, counting repeats. The model treats word occurrences
as independent draws conditional on the class. It also gives the
same scores to the first two reviews because their bags are
identical. [Jurafsky and Martin, Chapter 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf)
derive the model and discuss its assumptions.

<details>
<summary>Check the feature vectors, probabilities, and gradients in Python</summary>

~~~python
from collections import Counter
from math import exp, isclose, log, sqrt

docs = [
    "great screen awful battery", "awful screen great battery",
    "bright screen weak speakers", "dim screen loud speakers",
]
labels = [1, 0, 1, 0]

def terms(text, n):
    words = text.lower().split()
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def fit_idf(texts, n):
    df = Counter(t for text in texts for t in set(terms(text, n)))
    return {t: 1 + log((1 + len(texts)) / (1 + d)) for t, d in df.items()}

def transform(text, n, idf):
    weights = {t: count * idf[t] for t, count in Counter(terms(text, n)).items()
               if t in idf}
    norm = sqrt(sum(v*v for v in weights.values()))
    return {t: v/norm for t, v in weights.items()} if norm else {}

idf1, idf2 = (fit_idf(docs, n) for n in [1, 2])
x1, x2 = (transform(d, 1, idf1) for d in docs[:2])
assert len(idf1) == 9 and idf1["screen"] == 1
assert all(isclose(x1[t], x2[t]) for t in x1)
assert isclose(x1["screen"], 0.35696573415957816)
assert transform("unseen vocabulary", 1, idf1) == {}

v1, v2 = (transform(d, 2, idf2) for d in docs[:2])
assert isclose(v1["great screen"], 1/sqrt(3))
assert "great screen" not in v2 and "awful screen" not in v1

sigmoid = lambda z: 1/(1 + exp(-z))
w = {"great screen": 1.2, "awful screen": -1.2}
score = lambda row: 0.1 + sum(w.get(t, 0) * value for t, value in row.items())
assert [f"{sigmoid(score(v)):.3f}" for v in (v1, v2)] == ["0.688", "0.356"]
grad = [-0.5 * v1["great screen"], 0.5 * v2["awful screen"]]
assert [f"{g:.3f}" for g in grad] == ["-0.289", "0.289"]

counts = [Counter(t for text, y in zip(docs, labels) if y == k
                  for t in terms(text, 1)) for k in (0, 1)]
def nb_prob(term, k):
    return (counts[k][term] + 1) / (sum(counts[k].values()) + len(idf1))
assert isclose(log(nb_prob("bright", 1) / nb_prob("bright", 0)), log(2))
assert nb_prob("screen", 1) == nb_prob("screen", 0)

print("Bigrams: p(positive) = " + ", ".join(f"{sigmoid(score(v)):.3f}" for v in (v1, v2)))
~~~

</details>

<a id="doccls-fasttext"></a>

## fastText: averaging learned features

The fastText classifier learns a vector for each feature,
such as a word or an n-gram. Instead of multiplying a sparse
feature row by one class matrix, it looks up the active vectors,
averages them, and feeds the average to the output layer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/document-classification-2026-11-22/joulin-fasttext-figure1.png|The n-gram features x1, x2, and the remaining inputs feed one common hidden representation, which feeds the output.]]

*Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov, [Bag of Tricks for Efficient Text Classification, Figure 1](https://aclanthology.org/E17-2068.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The complete diagram is cropped from the PDF; the labels are unchanged.*

If the first review's three bigram vectors are $(1,0)$,
$(0,0.5)$, and $(0.5,-0.5)$, their average is $(0.5,0)$. An
output weight $(1,-1)$ without bias gives a logit of $0.5$.
In the actual model, matrix $A$ stores the vectors and $B$
maps the pooled representation to class logits. With a normalized
bag of features $x$, the computation is $B(Ax)$. The product
has rank at most the hidden dimension. Token vectors do not
depend on their neighbors; local order comes in through inputs
such as `great screen`. The original [§2.2](https://aclanthology.org/E17-2068.pdf#page=2)
explains the addition and hashing of n-gram features.

<a id="doccls-bert"></a>

## BERT: one output for the sequence

BERT receives ordered subwords with position information,
including `[CLS]` at the beginning and `[SEP]` at the end. In NER
we read several token states to get a tag per word. Review
classification reads a single vector associated with `[CLS]` and
trains a head on the document label. The two reviews have the
same bag of words but different token sequences, so their
`[CLS]` states can differ. Preserving order does not guarantee
that a classifier learns to focus on the screen.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/document-classification-2026-11-22/d2l-bert-one-seq.svg|Six tokens plus CLS and SEP enter BERT. Only the CLS output feeds the classification head and the class probability distribution.]]

*Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, [Dive into Deep Learning, Figure 16.6.1](https://d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html#fig-bert-one-seq), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original SVG, unmodified.*

The diagram has an output at every position, but the arrow to
the classifier comes from `[CLS]`. With two classes, use two logits
and a softmax, or one difference logit and a sigmoid. The binary
cross entropy agrees with the two-class cross entropy for
$z=z_+-z_-$. Some implementations transform the sequence state
before classification; D2L draws a small fully connected network.
If a review can receive both a `screen` label and a `battery`
label, those targets are not exclusive. One option is a separate
logit and sigmoid per label instead of a shared softmax.

Suppose the two sequence vectors after reading `[CLS]` are
$(0.6,-0.2)$ and $(-0.2,0.6)$. A head with weight $(1,-1)$
and bias $0.2$ produces logits $1$ and $-0.6$, or probabilities
about $0.731$ and $0.354$. This checks the head's arithmetic
with toy inputs; it is not a measurement of BERT. Full fine-tuning
updates the encoder and head. A frozen-encoder experiment trains
only the head. Compare the two on the same split. See
[BERT, §3 and Figure 4b](https://aclanthology.org/N19-1423/) for
the sequence classification setup.

Truncation can remove the sentence that describes the screen.
One way to retain it is to encode windows and train a classifier on
a pooled document representation. Simply copying the document
label onto every window may label “awful battery” as negative
screen sentiment even though the screen is not mentioned. Check
long documents separately on validation data, especially those
whose relevant phrase lies outside the first window.

<a id="doccls-evaluation"></a>

## Testing on new reviews

Compare unigram TF-IDF, TF-IDF with bigrams, and BERT on the same
set of documents. Changing both the split and the model hides
the reason for any difference. If a review appears in training
and is copied or quoted in test data, even a simple model can
carry over a rare labeled phrase. Group duplicates before splitting.
If the goal is future reviews of unseen devices, test a split by
time or device. Vocabulary, model settings, and thresholds must
be selected without the test set.

For a linear model, inspect contributions $w_tx_t$ from
active features. In the [20 Newsgroups example](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html),
the scikit-learn authors find `caltech` predictive of
`alt.atheism` because it appears in email addresses in quoted
threads. A review template or product ID could play the same
role here. Inspect where a feature comes from before dropping it.
A large coefficient is not proof of causation.

Suppose a team wants to flag the 50 negative reviews among
1,000 for manual review. Flagging none gives 95% accuracy
and zero recall on negatives. In this hypothetical example,
changing the negative-class threshold gives:

| Threshold | Correct flags | False flags | Missed | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 40 | 60 | 10 | 0.40 | 0.80 |
| 0.8 | 30 | 10 | 20 | 0.75 | 0.60 |

The higher threshold sends 40 reviews instead of 100 to the team,
but misses ten additional negatives. Choose the threshold by how
many negatives may be missed and how many documents the team can
inspect. Once it is fixed, report test errors separately for reviews
with conflicting screen and battery sentiment, long reviews, and
unseen devices. One aggregate score cannot tell which of those
cases the model stops distinguishing.

## Sources

- Dan Jurafsky and James H. Martin, [Speech and Language Processing, Chapter 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf): multinomial naive Bayes, text classification, and metrics.
- scikit-learn, [TF-IDF weighting](https://scikit-learn.org/1.9/modules/feature_extraction.html#tfidf-term-weighting) and [diagnosing a sparse text classifier](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html).
- Armand Joulin et al., [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/), §§2–2.2: averaged vectors, n-grams, and the output matrix.
- Jacob Devlin et al., [BERT](https://aclanthology.org/N19-1423/), §3 and Figure 4b; [D2L, §16.6.1](https://d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html#single-text-classification): one label per input sequence.
- In this textbook: [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|word counts and TF-IDF]] and the [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация|derivation of logistic regression]].

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей|Named entity recognition]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03c Извлечение информации из документов|Information extraction from documents]].
