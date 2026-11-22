---
title: Machine translation
description: "Parallel data, conditional generation, beam search, back-translation, and translation quality on one example."
type: textbook-chapter
status: canonical
locale: en
source_language: en
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/08 Машинный перевод.md"
translation_status: current
canonical_revision: "b59c7f5e7c02a14e29b77a7cb123e012087e769e54b93f35f5185826ea306172"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Parallel data and bitext mining"
    anchor: mt-parallel-data
  - label: "Conditional translation probability and training"
    anchor: mt-objective
  - label: "Greedy and beam decoding in translation"
    anchor: mt-decoding
  - label: "Back-translation and target-language control"
    anchor: mt-back-translation
  - label: "Evaluating machine translation"
    anchor: mt-evaluation
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/13.pdf
  - https://aclanthology.org/2022.findings-emnlp.154/
  - https://aclanthology.org/P16-1009/
  - https://aclanthology.org/D18-1045/
  - https://aclanthology.org/Q17-1024/
  - https://www.nature.com/articles/s41586-024-07335-x
  - https://aclanthology.org/W18-6319/
  - https://aclanthology.org/2020.emnlp-main.213/
  - https://aclanthology.org/2021.tacl-1.87/
---

The Russian instruction *Не закрывайте клапан до полудня* can be
translated as *Do not close the valve before noon* or *Before noon,
do not shut the valve*. The second version changes the word order
and the verb, but preserves the meaning. Removing *not* from the
first leaves almost all the words in place and reverses the instruction.

<a id="mt-parallel-data"></a>

## Where parallel sentences come from

Supervised training needs pairs $(x,y)$: source text $x$ and its
translation $y$. In a bilingual manual, section numbers may match,
but the sentences still need to be aligned. The translator may
combine two sentences, move an explanation to a footnote, or leave
out a line. Pairing sentences by position shifts every later pair
after the first omission.

To find pairs in separate collections, a system can compare
[[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|sentence embeddings]].
An encoder produces one vector per sentence, then a search retrieves
neighbors from the other language. A Russian line about a valve may
retrieve both a correct instruction not to close it and a reminder
to open it. Similarity alone does not make both strings translations.
Mining systems compare the score with nearby alternatives and filter
the pairs. Useful checks include language, length, matching numbers
or codes, and duplicate detection. A pair with *до 12:00* on one
side and *before 2 PM* on the other merits a check even if the
rest of the sentence matches.

LASER3 addresses a step before the search: align vectors across
languages when the teacher encoder handles a new language poorly.
The student receives a sentence in the new language and the teacher
receives its translation. Training brings their vectors closer under
cosine similarity. Once the student can encode new sentences in the
shared space, it can help mine pairs from collections that lack known
translations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/machine-translation-2026-11-22/heffernan-laser3-fig1.png|On the right, student and teacher encode the two sides of a pair and cosine loss brings the vectors together. On the left, the student predicts a masked token.]]

*Kevin Heffernan, Onur Çelebi, and Holger Schwenk,
[Bitext Mining Using Distilled Sentence Representations for Low-Resource Languages, Figure 1](https://aclanthology.org/2022.findings-emnlp.154.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Figure cropped from the surrounding PDF page.*

The right panel shows one pair consisting of a Turkish question
about a watch and its English translation. Another pair uses the
same English sentence on both sides to anchor the student to a
known space. In the left panel, the student predicts a masked
token in a Turkish sentence. No translation is used for that
step. Both losses train an encoder for finding pairs; it does
not produce translations.

<a id="mt-objective"></a>

## What the encoder–decoder learns

In a [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Transformer]],
the encoder reads the full source sentence and returns one
contextual vector per token. At each step the causal mask lets the
decoder see earlier target tokens. Cross-attention gives it access
to the encoder outputs. When choosing *not*, the decoder has the
prefix *do* and can attend to the source containing *не*. An
attention map may show which positions a head used on that step.
It does not establish whether the output preserves negation.

The probability of the sequence is the product of conditional
probabilities. Training sums their negative logarithms:

$$
\mathcal L=-\sum_{t=1}^{T}\log p_\theta(y_t\mid y_{<t},x).
$$

Here $T$ includes the end token. For this example, treat each word
as one token; a real tokenizer can split it into several pieces.
The decoder receives the target shifted by one position.
The start token in the table is schematic. T5, for instance, uses
`pad_token_id` at that position.

| Step | Tokens already in the decoder input | Target on this step |
| ---: | --- | --- |
| 1 | `<start>` | `do` |
| 2 | `<start> do` | `not` |
| 3 | `<start> do not` | `close` |
| Final | `<start> do not close the valve before noon` | `<eos>` |

At step two, if $p_\theta(\text{not}\mid\text{do},x)=0.4$, the
loss contribution is $-\log(0.4)\approx0.916$. The third step of
the training example still receives the gold prefix `do not`.
That is teacher forcing. The mask prevents the third position
from seeing its own target. At generation time there is no gold
prefix. Step three uses the choice made at step two. After
`do close` it cannot insert `not` before `close`.

<a id="mt-decoding"></a>

## What beam search keeps

Consider two steps after a fixed prefix `do`. These probabilities
are made up to show how a locally less likely token can lead to
a more likely complete translation.

| Prefix | Conditional probabilities of continuations |
| --- | --- |
| `do` | `close`: $0.60$; `not`: $0.40$ |
| `do close` | `the`: $0.40$; `that`: $0.35$; `this`: $0.25$ |
| `do not` | `close`: $0.95$; `open`: $0.05$ |

Greedy search keeps `do close` and chooses `the` next.
The prefix has probability $0.60\cdot0.40=0.24$. A beam of width
two also keeps `do not`. After expanding both candidates,
`do not close` has probability $0.40\cdot0.95=0.38$ and ranks
first. Assume that the remaining tokens up to `<eos>` are
deterministic in this example, so the ranking does not change.

Real decoders add log probabilities instead. Hypotheses can finish
at different steps. Longer translations have more nonpositive
terms in the raw sum. Decoders can therefore account for length
when ranking completed outputs; the setting is tuned on held-out
translations. A wider beam may find a higher score under the
*model*. It cannot repair a model that scores a complete translation
without negation above the correct one. Constrained decoding can
require a term, but the presence of *valve* alone says nothing
about whether the instruction prohibits closing it.

<a id="mt-back-translation"></a>

## Why translate in the opposite direction?

There may be few Russian-to-English pairs but plenty of good
English text. Take the human-written sentence *Do not close the
valve before noon*. A temporary English-to-Russian model creates a
synthetic source, for example *Не перекрывайте клапан раньше
полудня*. The Russian string becomes the input for training the
forward model, and the original English is the target. The
synthetic side is the input. These pairs can be mixed with human
bitext. If the reverse model drops *не*, a contradictory pair
enters the training data.

How the reverse model generates its outputs can affect the forward
model. In the Edunov et al. plot, the horizontal axis gives the
number of training pairs and the vertical axis gives the trained
models' BLEU on `newstest2012` for English-to-German translation.
The curves differ in how the synthetic **input** was created
earlier. At the same large data size on the right, sampling and
beam search with added noise scored higher than plain beam search.
That compares methods of preparing training data in one experiment,
not decoding strategies for a user-facing translation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/machine-translation-2026-11-22/edunov-backtranslation-fig1.png|Five WMT curves: after adding synthetic pairs, sampling and beam plus noise rise above greedy and beam on newstest2012.]]

*Sergey Edunov, Myle Ott, Michael Auli, and David Grangier,
[Understanding Back-Translation at Scale, Figure 1](https://aclanthology.org/D18-1045.pdf#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Plot cropped from the surrounding PDF page.*

### Specifying the target language

A multilingual model can share weights across translation
directions. The same input can have a correct output in different
languages, so the request must specify the target language. In
Johnson et al., a special input token specifies that language.
This design can be tested on directions missing from the training
pairs (zero-shot translation). Quality still needs to be measured
per direction. A strong English-to-German result says little
about a direction between two low-resource languages. In
[NLLB-200](https://doi.org/10.1038/s41586-024-07335-x), bitext
mining, multilingual training, and testing directions on FLORES-200
address different parts of the problem.

<a id="mt-evaluation"></a>

## Checking whether the meaning survived

Compare two translations with one reference. Strip punctuation and
case; this calculation uses whitespace splitting rather than an
external tokenizer.

| Role | String |
| --- | --- |
| Reference | `do not close the valve before noon` |
| Meaning changed | `do close the valve before noon` |
| Meaning preserved | `before noon do not shut the valve` |

All six unigrams in the wrong output match. For longer n-grams
the clipped precisions are $4/5$, $3/4$, and $2/3$. The brevity
penalty is $\exp(1-7/6)$. The geometric mean of the four
precisions, multiplied by the penalty, gives about $67.3$ on a
0–100 scale. The valid translation reorders the words and changes
the verb. It has no trigram in common with the reference. Its
unsmoothed toy BLEU-4 is zero. For short strings this is one
comparison, not a corpus score. The
[[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация#bleu-reference-overlap|evaluation chapter]]
aggregates clipped matches and lengths over a corpus before
scoring. For a reproducible result, report the data, translation
direction, and sacreBLEU signature.

chrF uses character n-grams and can give credit for shared stems
or nearby spellings. It still does not verify that the prohibition
survived. COMET can account for the source as well as the
hypothesis and reference. In the *estimator* version of Rei et al.,
a shared encoder encodes the three strings separately.
A regressor predicts a score from combined representations and
pairwise features. Training compares it with a quality annotation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/machine-translation-2026-11-22/rei-comet-estimator-fig1.png|The three inputs to the COMET estimator are Hypothesis, Source, and Reference. A shared encoder and pooling layer supply vectors to the regressor; MSE is a training loss.]]

*Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie,
[COMET: A Neural Framework for MT Evaluation, Figure 1](https://aclanthology.org/2020.emnlp-main.213.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Figure cropped from the surrounding PDF page.*

The figure shows a reference-based estimator, not the interface
of every COMET model. Its score depends on the checkpoint.
It does not enforce a requirement to include *not* in every
prohibition; a translator can express a prohibition in other ways.
With [MQM](https://aclanthology.org/2021.tacl-1.87/), a human
annotator marks the error and its severity. Losing negation is a
meaning error here; replacing *close* with *shut* can be acceptable.
Separate slices for negation, numbers, names, and terminology
surface errors that a corpus score can hide. In document
translation, neighboring sentences can also clarify a pronoun or
help keep the name of a part consistent.

<details>
<summary>Verify the two search steps and BLEU calculation in Python</summary>

~~~python
from collections import Counter
from math import exp, isclose, log

# The prefix "do" is fixed. The remaining tails are deterministic in this toy.
edges = {
    ("do",): [("close", 0.60), ("not", 0.40)],
    ("do", "close"): [("the", 0.40), ("that", 0.35), ("this", 0.25)],
    ("do", "not"): [("close", 0.95), ("open", 0.05)],
}
def beam(width):
    active = [(("do",), 1.0)]
    for _ in range(2):
        candidates = [(seq + (word,), p * q)
                      for seq, p in active for word, q in edges[seq]]
        active = sorted(candidates, key=lambda item: -item[1])[:width]
    return active

assert beam(1)[0][0] == ("do", "close", "the")
assert isclose(beam(1)[0][1], 0.24)
assert beam(2)[0][0] == ("do", "not", "close")
assert isclose(beam(2)[0][1], 0.38)
assert round(-log(0.40), 3) == 0.916

def grams(words, n):
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def toy_bleu4(candidate, reference):
    cand, ref = candidate.split(), reference.split()
    precision = []
    for n in range(1, 5):
        counts = grams(cand, n)
        matches = sum((counts & grams(ref, n)).values())
        precision.append(matches / sum(counts.values()))
    bp = exp(min(0, 1 - len(ref) / len(cand)))
    score = (bp * exp(sum(log(p) for p in precision) / 4)
             if all(precision) else 0)
    return precision, 100 * score

ref = "do not close the valve before noon"
wrong = "do close the valve before noon"
safe = "before noon do not shut the valve"
precision, score = toy_bleu4(wrong, ref)
assert precision == [1, 4/5, 3/4, 2/3]
assert round(score, 1) == 67.3
assert toy_bleu4(safe, ref)[1] == 0
~~~

</details>

## Sources

- Daniel Jurafsky and James H. Martin, [Speech and Language Processing, Chapter 13](https://web.stanford.edu/~jurafsky/slp3/13.pdf): conditional generation, search, and evaluation.
- Kevin Heffernan, Onur Çelebi, and Holger Schwenk, [LASER3](https://aclanthology.org/2022.findings-emnlp.154/), Section 3 and Figure 1: learning embeddings for bitext mining.
- Rico Sennrich, Barry Haddow, and Alexandra Birch, [Improving Neural Machine Translation Models with Monolingual Data](https://aclanthology.org/P16-1009/), Section 3.2; Sergey Edunov et al., [Understanding Back-Translation at Scale](https://aclanthology.org/D18-1045/), Section 5.1 and Figure 1.
- Melvin Johnson et al., [Google’s Multilingual Neural Machine Translation System](https://aclanthology.org/Q17-1024/), Sections 2 and 4: selecting the target language and zero-shot translation; [NLLB-200](https://doi.org/10.1038/s41586-024-07335-x): mining and evaluating translation directions.
- Matt Post, [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/); Ricardo Rei et al., [COMET](https://aclanthology.org/2020.emnlp-main.213/), Section 2.3 and Figure 1; Markus Freitag et al., [Experts, Errors, and Context](https://aclanthology.org/2021.tacl-1.87/).

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5 and text-to-text tasks]].
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1]].
