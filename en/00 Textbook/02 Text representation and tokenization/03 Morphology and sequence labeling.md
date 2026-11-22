---
title: Morphology and sequence labeling
description: "Word forms, lemmas, part-of-speech tags, and features. A worked hidden Markov model, Viterbi decoding, and linear-chain CRF."
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/02 Представление текста и токенизация/03 Морфология и разметка последовательностей.md"
canonical_revision: "d35b5a6fca7515682059360074b9b5f8c45f0bbcc251d52885a46c086033a979"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Morphological annotation: lemma, UPOS, and FEATS"
    anchor: morphology-tags
  - label: "Hidden Markov models and the Viterbi algorithm"
    anchor: hmm-viterbi
  - label: "Linear-chain CRF and sequence scores"
    anchor: linear-chain-crf
  - label: "Evaluating morphological annotation"
    anchor: pos-evaluation
primary_sources:
  - https://universaldependencies.org/format.html
  - https://www.kamperh.com/nlp817/notes/05_hmm_notes.pdf
  - https://arxiv.org/abs/1503.07283
  - https://aclanthology.org/A00-1031/
  - https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf
---

In the Russian sentence «Мы стали работать» (we began to work), `стали`
is a form of the verb `стать` (to become or begin). In «кусок стали»
(a piece of steel), the same spelling is the genitive of the noun
`сталь` (steel). A tagger that sees only `стали` has no evidence
in the spelling alone to distinguish these occurrences. A tagger that sees the sentence
can use both `мы` (we) and `работать` (to work).

<a id="morphology-tags"></a>

## Lemmas, parts of speech, and features

A lemma groups inflected forms of one lexical item: `сталь`, `стали`,
and `сталью` can all have the lemma `сталь`. A part-of-speech
tag gives the broader category, such as noun or verb. Morphological
features describe the inflection. In the last two rows below, the
lemma and part of speech match, but the case and number differ.

| Context | Form | Lemma | Part of speech | Selected features |
| --- | --- | --- | --- | --- |
| Мы стали работать (we began to work) | стали | стать | `VERB` | past tense, plural |
| кусок стали (a piece of steel) | стали | сталь | `NOUN` | genitive, singular |
| На столе лежат книги (books lie on the table) | книги | книга | `NOUN` | nominative, plural |
| обложка книги (the cover of a book) | книги | книга | `NOUN` | genitive, singular |

[Universal Dependencies](https://universaldependencies.org/format.html#morphological-annotation)
records these in separate fields. `FORM` retains the surface form,
including capitalization, `LEMMA` holds the lemma, and `UPOS`
gives the universal part of speech. The `FEATS` field for `книги`
in «обложка книги» can include `Case=Gen|Gender=Fem|Number=Sing`.
A vertical bar separates features; an underscore represents an
empty list. `XPOS` is available for an additional language- or
corpus-specific tag. These sentences were written for this chapter.
[SynTagRus](https://universaldependencies.org/treebanks/ru_syntagrus/index.html)
shows how a Russian corpus represents the annotations.

Labels attach to words as defined by the annotation scheme. Their
boundaries need not match the previous chapter's subword tokens.
In [CoNLL-U](https://universaldependencies.org/format.html#words-tokens-and-empty-nodes),
even a surface token can stand for multiple words. Spanish `al`
has a row for the surface token and two word rows, `a` and `el`.
The part-of-speech decisions belong to those two words.

A lexical analyzer can list more than one analysis for `стали`.
Both the inflection table for `сталь` and the one for `стать`
contain that form. For unseen words, an analyzer can look for
analogies. [pymorphy2, §§3–4](https://arxiv.org/abs/1503.07283)
combines a dictionary with heuristics, including suffix matching
against dictionary forms. A suffix gives a candidate, not a
guarantee. If the tagger is restricted to candidates and the
analyzer omits the correct one, sentence context cannot recover it.

<a id="hmm-viterbi"></a>

## Choosing tags with a hidden Markov model

Take the words $x_1,\ldots,x_T$ and a part-of-speech tag $y_t$ for each
word. A hidden Markov model (HMM) assigns a joint probability to
the two sequences. At each position it uses a transition
$a_{uv}=P(y_t=v\mid y_{t-1}=u)$ and an emission
$b_v(x_t)=P(x_t\mid y_t=v)$. In tagging, the words are observed and
the states corresponding to the tags are hidden.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/pos-tagging-2026-11-22/kamper-hmm-transitions.png|Solid arrows connect states such as NOUN, DET, and VERB. Dashed arrows point from a tag to its word probabilities.]]

*Herman Kamper, [Hidden Markov Models, p. 5](https://www.kamperh.com/nlp817/notes/05_hmm_notes.pdf#page=5), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The original diagram, cropped to remove the page margins.*

The solid arrow from `NOUN` to `VERB` refers to adjacent tags.
The dashed arrow from `NOUN` points to probabilities of particular
words given that tag, such as $P(\text{can}\mid\mathrm{NOUN})$.
The start state transitions into the first tag. The transition
to the end state also contributes a probability, without emitting a word.

### Estimating transitions and emissions

Suppose a labeled corpus contains 100 transitions out of `PRON`,
50 of them to `VERB`. The relative-frequency estimate of
$a_{\mathrm{PRON},\mathrm{VERB}}$ is $0.5$. For the emission of
`стали` as a verb, divide the number of occurrences of `стали`
tagged `VERB` by the number of **all** word occurrences tagged
`VERB`. This denominator is different from the number of
occurrences of `стали`.

A first-order HMM uses only the previous tag for a transition
and only the current tag for an emission. Seeing `работать`
(to work) later in the sentence does not raise the emission
probability of `стали` under `VERB`. It can change which complete
tag sequence scores highest. In [TnT, §2](https://aclanthology.org/A00-1031.pdf#page=1),
Brants used two preceding tags, interpolated estimates of different orders,
and used suffix probabilities for unknown words. The second tag gives
more context; interpolation helps with sparse counts; suffixes
provide evidence when the word is absent from the lexicon.

### A calculation for «мы стали работать»

Keep two paths, `PRON NOUN VERB` and `PRON VERB VERB`, to make
every step visible. The probabilities are illustrative; words
and transitions not shown can take up the remaining probability
mass. Both paths start with a $0.6$ transition to `PRON` and a
$0.5$ emission probability for `мы` (we). The shared prefix has
probability $0.30$.

For a full path, multiply all transitions, including start and end,
and one emission per word. The resulting $P(x,y)$ is divided by $P(x)$
to get $P(y\mid x)$. For a fixed sentence the denominator is shared
by all paths, so comparing joint probabilities suffices for decoding.

| Step | `стали` as `NOUN` | `стали` as `VERB` |
| --- | ---: | ---: |
| Transition from `PRON` | 0.4 | 0.5 |
| Emission of `стали` | 0.4 | 0.2 |
| Score after `стали` | $0.30\cdot0.4\cdot0.4=0.048$ | $0.30\cdot0.5\cdot0.2=0.030$ |
| Transition to `VERB` and emission of `работать` | $\times 0.15\cdot0.5$ | $\times 0.6\cdot0.5$ |
| Score after `работать` | 0.0036 | 0.009 |
| Transition to the end with probability 0.2 | 0.00072 | 0.0018 |

A greedy choice after `стали` would retain only the noun path.
After `работать`, both paths end in the same state, `VERB`, and
the path that used a verb at the second position wins. The weaker
path can now be discarded. Any continuation of two paths ending
with the same tag at the same time multiplies them by the same factors.

Let $d_t(v)$ be the score of the best prefix ending at tag $v$:

$$
\begin{aligned}
d_1(v)&=a_{\mathrm{BOS},v}\,b_v(x_1),\\
d_t(v)&=b_v(x_t)\max_u\bigl[d_{t-1}(u)a_{uv}\bigr].
\end{aligned}
$$

At the last position, select the tag maximizing $d_T(v)a_{v,\mathrm{EOS}}$.
Store the maximizing predecessor $u$ at each cell so that the full
path can be recovered. The final `VERB` points to `VERB` at the second
position, then to `PRON` at the first. This is the Viterbi algorithm.
For $T$ words and $K$ tags, the $TK$ cells have up to $K$ predecessors
each, for $O(TK^2)$ work instead of enumerating $K^T$ paths. Scores need
only two columns at a time; backtracking normally keeps all $TK$
pointers.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/pos-tagging-2026-11-22/kamper-viterbi-trellis.png|Columns m, o, h and state rows C, V. The C cell under h contains 0.02016 and points to V in the preceding column.]]

*Herman Kamper, [Hidden Markov Models, p. 14](https://www.kamperh.com/nlp817/notes/05_hmm_notes.pdf#page=14), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The original table is cropped from the page; [p. 12](https://www.kamperh.com/nlp817/notes/05_hmm_notes.pdf#page=12) gives the transition and emission probabilities.*

Kamper's observations are the symbols `m o h` and the two states
are consonant `C` and vowel `V`. The `C` cell under `h` compares
$0.024\cdot0.2\cdot0.2=0.00096$ from `C` with
$0.144\cdot0.7\cdot0.2=0.02016$ from `V`. The backpointer goes
to `V`. After the end transition, the best path `C V C` has
score $0.02016\cdot0.4=0.008064$. The states do not follow
deterministically from the letters: even `V` can emit `h` in the
example model.

For long sequences, sum log probabilities to avoid underflow.
Taking logs preserves the ordering of positive probabilities;
a zero-probability transition gets $-\infty$. To find the
probability of the observations $P(x)$, the forward algorithm preserves
all paths, replacing each max by a sum (`logsumexp` in log space).
For «мы стали работать», the two permitted paths sum to $0.00252$.
The conditional probability of the verb path, given the words
and the restriction to these two paths, is $0.0018/0.00252=5/7$.
Allowing other paths changes the denominator.
[Kamper, pp. 8–19](https://www.kamperh.com/nlp817/notes/05_hmm_notes.pdf#page=8)
derives both recurrences.

<details>
<summary>Viterbi decoding with exact fractions</summary>

~~~python
from fractions import Fraction as F
from itertools import product

words = ("мы", "стали", "работать")
tags = (("PRON",), ("NOUN", "VERB"), ("VERB",))
transition = {
    ("BOS", "PRON"): F("0.6"),
    ("PRON", "NOUN"): F("0.4"), ("PRON", "VERB"): F("0.5"),
    ("NOUN", "VERB"): F("0.15"), ("VERB", "VERB"): F("0.6"),
    ("VERB", "EOS"): F("0.2"),
}
emission = {
    ("PRON", "мы"): F("0.5"),
    ("NOUN", "стали"): F("0.4"), ("VERB", "стали"): F("0.2"),
    ("VERB", "работать"): F("0.5"),
}

scores, back = {"BOS": F(1)}, []
for word, states in zip(words, tags):
    column, prev = {}, {}
    for state in states:
        best, parent = max(
            (score * transition.get((tag, state), F(0)), tag)
            for tag, score in scores.items()
        )
        column[state] = best * emission[state, word]
        prev[state] = parent
    scores = column
    back.append(prev)

prob, last = max(
    (p * transition.get((tag, "EOS"), F(0)), tag)
    for tag, p in scores.items()
)
path = []
for prev in reversed(back):
    path.append(last)
    last = prev[last]
path = tuple(reversed(path))

def path_probability(path):
    result, prev = F(1), "BOS"
    for state, word in zip(path, words):
        result *= transition.get((prev, state), F(0)) * emission[state, word]
        prev = state
    return result * transition.get((prev, "EOS"), F(0))

paths = [(path_probability(candidate), candidate) for candidate in product(*tags)]
assert (prob, path) == max(paths)
assert prob == F("0.0018") and path == ("PRON", "VERB", "VERB")
assert prob / sum(p for p, _ in paths) == F(5, 7)
print(path, float(prob))
~~~

</details>

<a id="linear-chain-crf"></a>

## Linear-chain CRF: features of the full sentence

An HMM can estimate its transitions and emissions from counts.
But the tagger may want evidence from the next word itself,
not only from its tag. The local score for `VERB` can use a
feature for “the next form is `работать`,” or consult a dictionary
to see whether the next word has an infinitive analysis. A
linear-chain conditional random field (CRF) models the probability
of the tags given the text. It does not model the probability of
the text, so its features can depend on several observations at
once. [Lafferty, McCallum, and Pereira, §§1–3](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf#page=1)
derive the model with vertex and edge features.

One common parameterization uses $u_t(v;x)$ for the local score of
tag $v$ at position $t$ and $m_{uv}$ for the score of adjacent tags.
Start and end can be included as special states. Add the scores
along the chain, then normalize over the tag sequences for this
sentence:

$$
\begin{aligned}
S(y,x)&=\sum_{t=1}^T u_t(y_t;x)+\sum_{t=2}^T m_{y_{t-1},y_t},\\
P(y\mid x)&=\frac{\exp S(y,x)}{\sum_z\exp S(z,x)}.
\end{aligned}
$$

At `стали`, suppose the local scores for `NOUN` and `VERB` are $1.1$
and $0.8$, and the respective transition scores into the last
`VERB` are $-0.8$ and $0.3$. With identical contributions from the
other terms, the totals are $1.1-0.8=0.3$ and $0.8+0.3=1.1$. The noun
wins locally; the verb wins on the full path. With only those two
paths, the verb path has probability $e^{1.1}/(e^{0.3}+e^{1.1})\approx0.690$.
Transition scores can be negative and do not sum to one. They
contribute to a path score rather than defining HMM transitions.

Decoding uses Viterbi again: add the transition and local scores,
take a maximum, and store the predecessor. The shared denominator
does not affect which path wins. Training needs the denominator:
the loss is $\log\sum_z e^{S(z,x)}-S(y_{\mathrm{gold}},x)$. For
the gold verb path, the loss is about $0.371$. The forward
recurrence uses `logsumexp` instead of a max to avoid enumerating
the paths. For a linear weight, the gradient is the model's
expected feature value minus the gold feature value. The
`VERB → VERB` transition appears once in the gold path and not
in the alternative. The derivative with respect to its weight
is $0.690-1\approx-0.310$, so a gradient descent step raises
that weight and makes the gold path more probable.

A contextual encoder can produce the $u_t$ scores. One decoder can
choose the highest-scoring label at each word; another can add
a CRF and score adjacent output tags explicitly. Even in the
first decoder, the score for `стали` can depend on `работать` through
the encoder. When the encoder uses subwords, align them with
the labeled words. One option is to use the first subword's
vector and compute the loss only at those positions; otherwise a
word split into four pieces can get four loss terms in place of one.

<a id="pos-evaluation"></a>

## Evaluating morphological annotation

Part-of-speech accuracy is the fraction of words with the correct
`UPOS`. Exact feature accuracy requires the whole `FEATS` bundle to
match. For a finer measure, compute micro-F1 over triples of
word position, feature name, and value. A wrong number value
counts as both an extra prediction and a missed gold feature.

For «Они читают книги.» (they read books), keep only the
listed features and count punctuation. Matching empty bundles
at the period count as an exact match.

| Word | Gold → predicted UPOS | Gold → predicted features |
| --- | --- | --- |
| Они (they) | `PRON → PRON` | plural, third person → singular, third person |
| читают (read) | `VERB → VERB` | plural, present → plural, present |
| книги (books) | `NOUN → PROPN` | plural, accusative → singular, accusative |
| . | `PUNCT → PUNCT` | empty → empty |

The accuracy is $3/4$ for parts of speech and $2/4$ for complete
feature bundles. The system predicts six feature triples and
matches four of the six gold triples. Precision and recall
are both $4/6$, so micro-F1 is $2/3$. The period contributes to
exact bundle accuracy, but contributes no feature triple.
Lemmatization accuracy is a separate count of words whose predicted
lemma matches the gold lemma.

These metrics assume aligned word positions. If the system also
segments the text, the evaluation protocol must account for
segmentation errors instead of silently dropping unmatched words.
A separate score for forms unseen during training exposes errors
that frequent forms can hide. When comparing across genres,
split documents rather than sharing fragments of a document
between training and test sets.

## Sources

- Universal Dependencies: [CoNLL-U](https://universaldependencies.org/format.html), [UPOS](https://universaldependencies.org/u/pos/index.html), and [morphological features](https://universaldependencies.org/u/feat/index.html).
- Herman Kamper, [NLP817: Hidden Markov Models](https://www.kamperh.com/nlp817/notes/05_hmm_notes.pdf), pp. 2–19: part-of-speech tagging, Viterbi, and the forward algorithm.
- Mikhail Korobov, [Morphological Analyzer and Generator for Russian and Ukrainian Languages](https://arxiv.org/abs/1503.07283), 2015: dictionaries, inflection patterns, and unknown forms.
- Thorsten Brants, [TnT: A Statistical Part-of-Speech Tagger](https://aclanthology.org/A00-1031/), 2000, §§2–3: the second-order model and evaluation.
- John Lafferty, Andrew McCallum, and Fernando Pereira, [Conditional Random Fields](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf), 2001, §§1–4: conditional chain models and parameter estimation.

**Previous:** [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece, and Unigram]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|how a recurrent network carries information from earlier words]].
