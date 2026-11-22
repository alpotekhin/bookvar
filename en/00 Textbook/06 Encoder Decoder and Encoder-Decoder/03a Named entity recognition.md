---
title: Named entity recognition
description: "BIO tags, BERT's classification head, and exact matching of entity boundaries."
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей.md"
canonical_revision: "774771c0d0ccd8543f4cb482872d8bc2508a14231f89501d5cbead0f841e2e1c"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "BIO (IOB2): named entity boundaries and types"
    anchor: ner-bio
  - label: "NER with BERT: word-to-subword alignment"
    anchor: ner-wordpiece-alignment
  - label: "NER: constrained sequence decoding"
    anchor: ner-constrained-decoding
  - label: "Exact-span precision and recall"
    anchor: ner-span-metrics
  - label: "Nested entities and span classification"
    anchor: ner-nested
primary_sources:
  - https://aclanthology.org/W03-0419/
  - https://aclanthology.org/N19-1423/
  - https://huggingface.co/learn/llm-course/chapter7/2
  - https://doi.org/10.1186/s12859-021-04248-8
  - https://doi.org/10.3390/app14209302
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03b Классификация документов]]"
---

In “Maya Chen joined Hugging Face in New York last spring,” we want the
person `Maya Chen`, the organization `Hugging Face`, and the location
`New York`. Returning only `New` as a location gets the type right
and the boundary wrong. Named entity recognition (NER) returns a
type and a span for each mention. Picking a catalogue entry for
one of several places called New York is a separate task, entity linking.

<a id="ner-bio"></a>

## Beginning, inside, and outside tags

With non-overlapping mentions, one label per word is enough.
Use the types `PER`, `ORG`, and `LOC` for this example. In the
BIO scheme used here, also called IOB2, `B-ORG` starts an organization
mention, `I-ORG` continues it, and `O` marks a word outside any entity
in the label set.

| Words | Tags | Mention |
| --- | --- | --- |
| Maya Chen | `B-PER I-PER` | person |
| joined | `O` | — |
| Hugging Face | `B-ORG I-ORG` | organization |
| in | `O` | — |
| New York | `B-LOC I-LOC` | location |
| last spring . | `O O O` | — |

A one-word entity also starts with `B`, as in `Berlin/B-LOC`.
This distinguishes two consecutive names, `Paris/B-LOC London/B-LOC`,
from one name, `New/B-LOC York/I-LOC`. The corpus defines the types:
our example leaves `last spring` outside because dates are not in
the inventory. CoNLL-2003 used `PER`, `ORG`, `LOC`, and `MISC`.
Other tasks may choose a different inventory.

Check what a dataset means by “BIO.” In
[the original CoNLL-2003, §2.3](https://aclanthology.org/W03-0419.pdf#page=2),
a location could start with `I-LOC`; `B-LOC` separated adjacent
mentions of the same type. That is IOB1. The
[Hugging Face course](https://huggingface.co/learn/llm-course/chapter7/2)
shows CoNLL data with `B` at the beginning of every mention.
Do not decode an IOB2 dataset as though an `I` at the beginning
were a valid start.

<a id="ner-wordpiece-alignment"></a>

## Aligning words with BERT's subwords

The corpus assigns one tag to `Hugging`. The pinned
[`bert-base-cased` tokenizer](https://huggingface.co/google-bert/bert-base-cased/blob/cd5ef92a9fb2f889e972770a36d4ed042daf221e/tokenizer.json)
splits it into `Hu` and `##gging`. Both pieces belong to
word 3, counting from zero, and `Face` is word 4.
The special tokens `[CLS]` and `[SEP]` have no word number.
A fast tokenizer exposes this correspondence through `word_ids()`
when given the annotated words as pretokenized input.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ner-2026-11-22/kim-biobert-bio.png|The first piece mit has tag I, while its continuation ##otic has x. The pieces spin and ##dle show the same pattern.]]

*Kwangmin Kim and Doheon Lee, [Context-aware multi-token concept recognition, Figure 3](https://link.springer.com/article/10.1186/s12859-021-04248-8/figures/3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure, unmodified.*

The encoder reads both `mit` and `##otic`. The diagram places
`I` over the first piece and uses `x` for a continuation without
a target tag. The [BERT paper, §5.3](https://aclanthology.org/N19-1423.pdf#page=9)
likewise takes the first subword's representation for NER.

| Position | Subword | Word ID | First piece only | All pieces |
| --- | --- | ---: | --- | --- |
| 0 | `[CLS]` | — | `-100` | `-100` |
| 1–3 | `Maya Chen joined` | 0–2 | `B-PER I-PER O` | `B-PER I-PER O` |
| 4 | `Hu` | 3 | `B-ORG` | `B-ORG` |
| 5 | `##gging` | 3 | `-100` | `I-ORG` |
| 6 | `Face` | 4 | `I-ORG` | `I-ORG` |
| 7–12 | `in New York last spring .` | 5–10 | `O B-LOC I-LOC O O O` | `O B-LOC I-LOC O O O` |
| 13 | `[SEP]` | — | `-100` | `-100` |

In the all-pieces variant, `##gging` gets `I-ORG` because it
continues the entity. A word that already has `I-ORG` gets
`I-ORG` on all its pieces; an outside word gets `O` on all of them.
Averaging the loss over active positions gives words split into more
pieces greater weight. To compare models on the same words, reduce
piece predictions to word predictions by a fixed rule. With first-piece
training, read the first piece. With all-pieces training, the same rule
is an option, or you can evaluate a different aggregation separately.

### Training the classification head

BERT returns a vector $h_i$ per subword. A shared linear head maps
each vector to tag logits, $z_i=Wh_i+b$. Three entity types give
seven scores, two per type and one for `O`. Our example has 14 input
positions including the special tokens, so the head returns a
$14\times7$ array before the mask is applied. First-piece cross
entropy uses 11 of those rows.

Suppose the model gives probability $0.7$ to `B-ORG` at `Hu`
and $0.6$ to `I-ORG` at `Face`. Their loss terms are
$-\log0.7$ and $-\log0.6$. Sum those with the other labeled
words' contributions and divide by 11, not 14. The `-100` in
the table is a cross-entropy `ignore_index` for class-index
targets, not an eighth class. Ignoring the loss at `##gging`
does not remove that piece from attention. Its representation
can help the model predict the label at `Hu`. Padding can be
masked in both places; the second piece is part of the input
even though it has no target.

<a id="ner-constrained-decoding"></a>

## Decoding a valid sequence

An independent maximum at each word can return `O I-ORG`
for `Hugging Face`. In IOB2, `I-ORG` must follow either `B-ORG`
or `I-ORG`. One repair turns the `I-ORG` into `B-ORG` and returns
only `Face` as an organization. A decoder can instead search for
the highest-scoring legal path. The two rules can give different
boundaries.

| Word | `O` | `B-ORG` | `I-ORG` |
| --- | ---: | ---: | ---: |
| Hugging | 3 | 2 | — |
| Face | 1 | 0.9 | 4 |

These toy logits cover two words after an `O`; other tags are
excluded. The path `O I-ORG` would win with $3+4=7$, but its
transition is illegal. The legal path `B-ORG I-ORG` scores
$2+4=6$, the path `O O` scores $4$, and `O B-ORG` scores $3.9$.
Choosing `B-ORG` at the first word pays off at the second.
The [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/03 Морфология и разметка последовательностей#hmm-viterbi|Viterbi algorithm]]
finds this path without enumerating all paths. Set prohibited edges
to $-\infty$ and allowed edges to zero. A learned CRF can also score
compatibility among legal transitions; a fixed constraint mask
does not require CRF training.

Convert a legal sequence to spans by opening a span on `B`,
continuing it on `I`, and closing it on `O`, the next `B`, or the
end of the sentence. The half-open interval `[17, 29)` contains
`Hugging Face` in the original string: 17 is the position of `H`,
and 29 is the space after `Face`. Take boundaries from the first
and last **words** in the mention. If the mention were just one
word, `Hugging`, it would still end after `gging`, even if its
tag came from the representation of `Hu`.

Offsets must refer to the same string. With pretokenized input, offsets
can be relative to individual words; map them through each word's
position to return absolute boundaries. For non-ASCII strings,
specify whether the coordinates count Unicode code points,
UTF-8 bytes, or UTF-16 code units. A window over a long document
can cut through a mention. With overlapping windows, reconcile
scores or predictions at the same document position before joining
tags, or deduplicate complete spans by their offsets. An `I` at
a truncated boundary does not establish that the mention ends there.

<a id="ner-span-metrics"></a>

## Scoring exact spans

[CoNLL-2003, §2.4](https://aclanthology.org/W03-0419.pdf#page=3)
credits exact matches. Compare triples of start, end, and type.
If a prediction gives just the first word of a longer mention,
it adds one extra predicted span and misses the gold span.

| Type | Gold | Predicted | Match |
| --- | --- | --- | --- |
| PER | Maya Chen `[0, 9)` | Maya `[0, 4)` | no |
| ORG | Hugging Face `[17, 29)` | Hugging `[17, 24)` | no |
| LOC | New York `[33, 41)` | New `[33, 36)` | no |

Only `Chen`, `Face`, and `York` have the wrong tag, so word
accuracy is $8/11$. Add 89 correctly classified `O` positions
and accuracy rises to $97/100$. Exact-span scores do not change:
$\mathrm{TP}=0$, $\mathrm{FP}=3$, and $\mathrm{FN}=3$.
Correcting only the tag for `Chen` matches one of the three spans
and raises $F_1$ to $1/3$.

$$
\begin{aligned}
P&=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\\
R&=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\\
F_1&=\frac{2\mathrm{TP}}{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}.
\end{aligned}
$$

For micro-F1, sum the true positives, false positives, and false
negatives over documents and types before taking the ratio. An `O`
does not form an entity and adds nothing to those counts. Break
scores down by type and span length to diagnose errors. For
example, if performance only drops on multiword names, inspect
the boundaries. With a tool such as
[`seqeval`](https://github.com/chakki-works/seqeval#usage), state
the scheme and mode: a permissive decoder for malformed `I` tags
does not implement strict IOB2. Use the same handling of invalid
transitions on validation and test.

<details>
<summary>Check the BIO tags, spans, and counts in Python</summary>

~~~python
import re
from itertools import product

text = "Maya Chen joined Hugging Face in New York last spring."
words = list(re.finditer(r"[A-Za-z]+|\.", text))
gold = "B-PER I-PER O B-ORG I-ORG O B-LOC I-LOC O O O".split()
pred = "B-PER O O B-ORG O O B-LOC O O O O".split()
word_ids = (None, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, None)

first_labels, all_labels = [], []
for i, w in enumerate(word_ids):
    label = gold[w] if w is not None else -100
    first = w is not None and (i == 0 or w != word_ids[i - 1])
    first_labels.append(label if first else -100)
    if w is not None and not first and label.startswith("B-"):
        label = "I-" + label[2:]
    all_labels.append(label)
assert first_labels[4:7] == ["B-ORG", -100, "I-ORG"]
assert all_labels[4:7] == ["B-ORG", "I-ORG", "I-ORG"]
assert sum(label != -100 for label in first_labels) == 11

def spans(tags):
    active, result = None, set()
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, kind = tag.partition("-")
        if prefix == "I":
            if active is None or active[1] != kind:
                raise ValueError(f"Invalid IOB2 tag at word {i}: {tag}")
            continue
        if active is not None:
            start, old_kind = active
            result.add((words[start].start(), words[i - 1].end(), old_kind))
        active = (i, kind) if prefix == "B" else None
    return result

g, p = spans(gold), spans(pred)
assert g == {(0, 9, "PER"), (17, 29, "ORG"), (33, 41, "LOC")}
assert p == {(0, 4, "PER"), (17, 24, "ORG"), (33, 36, "LOC")}
tp, fp, fn = len(g & p), len(p - g), len(g - p)
assert (tp, fp, fn) == (0, 3, 3)
assert sum(a == b for a, b in zip(gold, pred)) == 8

def valid(tags):
    prev = "O"
    for tag in tags:
        if tag.startswith("I-") and prev not in ("B-" + tag[2:], tag):
            return False
        prev = tag
    return True

scores = ({"O": 3, "B-ORG": 2}, {"I-ORG": 4, "B-ORG": 0.9, "O": 1})
free = tuple(max(row, key=row.get) for row in scores)
legal = [tags for tags in product(*scores) if valid(tags)]
best = max(legal, key=lambda tags: sum(row[tag] for row, tag in zip(scores, tags)))
assert free == ("O", "I-ORG") and not valid(free)
assert best == ("B-ORG", "I-ORG")
print(f"{tp=} {fp=} {fn=}; F1={2 * tp / (2 * tp + fp + fn):.1f}")
~~~

</details>

<a id="ner-nested"></a>

## When one entity contains another

A single BIO sequence cannot put two tags on the same word.
Suppose the annotation guidelines mark `New York` as a location
and `New York University` as an organization. Both start at `New`.
The answer in word coordinates is `{(0, 2, LOC), (0, 3, ORG)}`.
One tag at word 0 must lose one of these spans.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ner-2026-11-22/park-flat-nested.png|Panel b shows separate boxes for flat entities. Panel c adds boxes inside some of those entities.]]

*Yesol Park, Gyujin Son, and Mina Rho, [Biomedical Flat and Nested Named Entity Recognition, Figure 1](https://www.mdpi.com/2076-3417/14/20/9302#fig_body_display_applsci-14-09302-f001), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure, unmodified.*

In panel (c), an inner box starts at the same position as its
outer box and ends earlier. The flat annotation in panel (b)
retains only the outer interval. One way to predict both is to
classify individual pairs of boundaries. For `New York University`,
there are six contiguous candidates: three one-word spans, two
two-word spans, and one three-word span. The encoder reads the text
once; a classifier can score each pair using the representations
of its endpoints. Classifying `[0, 2)` as `LOC` need not rule out
classifying `[0, 3)` as `ORG`. For $T$ words there are $O(T^2)$
contiguous candidates. A maximum span length reduces that count
but rules out any longer mentions. Discontinuous mentions need a
different representation, since an interval includes everything
between its boundaries.

## Sources

- Erik F. Tjong Kim Sang and Fien De Meulder, [CoNLL-2003, §§2.3–2.4](https://aclanthology.org/W03-0419/): original tags and evaluation.
- Jacob Devlin et al., [BERT, §5.3 and Figure 4d](https://aclanthology.org/N19-1423/): classifying the first wordpiece.
- Hugging Face, [Token classification](https://huggingface.co/learn/llm-course/chapter7/2): `word_ids()`, label alignment, and evaluation.
- Kwangmin Kim and Doheon Lee, [Context-aware multi-token concept recognition](https://doi.org/10.1186/s12859-021-04248-8), Figure 3.
- Yesol Park, Gyujin Son, and Mina Rho, [Biomedical Flat and Nested Named Entity Recognition](https://doi.org/10.3390/app14209302), §§2.1–2.3 and 4.2: BIO, nested annotations, and span classification.

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa, and DeBERTa]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03b Классификация документов|Document classification]].
