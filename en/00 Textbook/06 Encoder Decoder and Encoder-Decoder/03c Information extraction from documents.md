---
title: Information extraction from documents
description: "Follow mentions through coreference, entity linking, relation extraction, event arguments, and time normalization."
type: textbook-chapter
status: canonical
locale: en
source_language: en
translation_of: "00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03c Извлечение информации из документов.md"
translation_status: current
canonical_revision: "ad9a00cbb0b6986fc450b347c2b72c968f498d7ee8a29a555cf5911692c8c4d6"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Mention boundaries and entity IDs"
    anchor: ie-mentions
  - label: "Coreference and antecedent scoring"
    anchor: ie-coreference
  - label: "Entity linking to a catalog"
    anchor: ie-linking
  - label: "Document-level relation extraction"
    anchor: ie-relations
  - label: "Events and time normalization"
    anchor: ie-events
  - label: "Evaluating information extraction"
    anchor: ie-evaluation
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/21.pdf
  - https://web.stanford.edu/~jurafsky/slp3/24.pdf
  - https://aclanthology.org/D17-1018/
  - https://aclanthology.org/2020.emnlp-main.519/
  - https://aclanthology.org/P19-1074/
  - https://aclanthology.org/D19-1585/
  - https://aclanthology.org/L12-1122/
  - https://aclanthology.org/P14-2006/
---

In 2021, biologist Anna Levina went to work at the North Institute.
Two years later, she took over as director of the institute.

The two sentences report different facts about the same person and
organization. Levina first joined the institute and later became its
director. The second sentence gives no personal name, organization
name, or calendar year. Recording the appointment
requires information from both sentences.

<a id="ie-mentions"></a>

## Mentions and entities

[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей|Named entity recognition]] can mark *Anna Levina*
as a person's name and *North Institute* as an organization. Coreference
also needs *she* and *the institute* in sentence 2. Those are mentions too:
they occur at different positions but refer to the entities already named.

| Mention | Text span | Sentence | Document entity |
| --- | --- | ---: | --- |
| $m_1$ | Anna Levina | 1 | `person:1` |
| $m_2$ | North Institute | 1 | `org:1` |
| $m_3$ | she | 2 | `person:1` |
| $m_4$ | the institute | 2 | `org:1` |

In a data set, each mention also has boundaries in the original
document, such as character offsets. Identical strings at different
positions can refer to different entities. The IDs `person:1` and
`org:1` are local to this document. To combine records across
documents, a system can map these entities to entries in a catalog.

<a id="ie-coreference"></a>

## Scoring antecedents for “she”

An antecedent is an earlier mention that a later expression refers
back to. For `she`, the model compares `Anna Levina`, `North Institute`,
and the option of no antecedent. It can also link `the institute`
to `North Institute`. The resulting clusters are $\{m_1,m_3\}$
and $\{m_2,m_4\}$.

The model of Lee et al. takes a document rather than a list of names.
It enumerates spans up to a length limit and represents each span
using the contextual vectors at its ends, an attention-weighted
sum of the vectors inside it, and a width feature. A score $s_m(i)$
estimates how likely a span is to be a mention. The model discards
low-scoring spans before comparing pairs: a document of $n$ tokens
already has on the order of $nL$ candidates with maximum width $L$,
when $L\le n$.

For a surviving pair $(i,j)$ with $j$ earlier than $i$, the model
scores the compatibility $s_a(i,j)$. It adds both mention scores:

$$
s(i,j)=s_m(i)+s_m(j)+s_a(i,j).
$$

Suppose the mention scores are $0.4$ for *she*, $0.7$ for the
name, and $0.5$ for the institute. Give the first pair a
compatibility score of $1.1$ and the second $-1.2$. The resulting
scores are $2.2$ and $-0.3$. The dummy antecedent has a fixed
score of zero. Softmax over these three options assigns the name
about $0.838$. These numbers are illustrative. In the learned model,
the scores depend on context, distance, and learned parameters.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/information-extraction-2026-11-22/lee-antecedents-fig2.png|For the company, two pairwise paths compare General Electric and the Postal Service; a third path supplies the dummy antecedent score of zero to softmax.]]

*Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer,
[End-to-end Neural Coreference Resolution, Figure 2](https://aclanthology.org/D17-1018.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Figure cropped from the surrounding PDF page.*

In the figure, *the company* plays the role of *she*. Two paths
combine pair and mention scores; the third contributes zero.
A dummy antecedent is needed, for instance, for the first mention
of a new entity. Training need not pick the closest earlier mention
as the only correct answer. If there are several matching gold
antecedents, Lee et al. train the model to raise their total
probability. At inference, the chosen links form clusters: if $m_5$
links to $m_3$ and $m_3$ links to $m_1$, all three belong to one
cluster.

Annotation guidelines determine what counts as one entity.
*The institute* refers to the organization, but *its lab* may
denote a separate unit. A part-whole relation is not coreference
even if the two expressions occur close together.

<a id="ie-linking"></a>

## Finding Anna Levina in a catalog

Two people can share a name. In a toy catalog, `P17` is Anna Levina
the biologist, `P54` is an architect with the same name, and `P81`
is an athlete with a similar name. The document does not list these
records. Entity linking returns `P17` using the name and the context
that she is a biologist who joined the North Institute. The pronoun
*she* can inherit the ID found for a more informative mention in
the same cluster.

BLINK encodes the mention in context as a vector $q$ and encodes a
record's title and description separately as $e$. The catalog vectors
can be computed in advance. The system retrieves the top $K$ candidates
by dot product $q^\top e$. Take $q=(0.8,0.6)$ and three two-dimensional
catalog vectors:

| Record | Vector | Retrieval score |
| --- | --- | ---: |
| `P54`, architect | $(0.8,0.6)$ | $1.00$ |
| `P17`, biologist | $(0.6,0.8)$ | $0.96$ |
| `P81`, athlete | $(0,1)$ | $0.60$ |

With $K=2$, the right record reaches the reranker in second place.
A cross-encoder reads the context and each description together.
It can match *biologist* in the document to the profession in the
record. With logits of $1.1$ for `P54` and $2.7$ for `P17`,
`P17` ranks first and has about $0.832$ of the softmax mass over
these two candidates. With $K=1$, only `P54` would be available.
The reranker cannot recover a record excluded by retrieval.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/information-extraction-2026-11-22/wu-blink-fig1.png|The bi-encoder places the Jaguar context and entity records in one space. The cross-encoder compares the context with the roller coaster and car records.]]

*Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer,
[Scalable Zero-shot Entity Linking with Dense Entity Retrieval, Figure 1](https://aclanthology.org/2020.emnlp-main.519.pdf#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Figure cropped from the surrounding PDF page.*

In the authors' example, retrieval keeps two senses of *Jaguar*
and the cross-encoder prefers the roller coaster. Unlike a coreference
model, it compares document text with a description from outside the
document. Wu et al. evaluate given mentions whose gold entities are
in the catalog. Mentions can still corefer when no record exists.
A system with an incomplete catalog needs a `NIL` outcome or a
review path. Softmax over retrieved descriptions cannot tell whether
the correct entity is in the set. With one candidate it gives that
candidate probability one even if the candidate is wrong.

<a id="ie-relations"></a>

## Which relations does the document support?

Suppose the database has predicates `works_for`, `heads`, and
`founded`. The pair (Levina, North Institute) supports the first
two; neither sentence says she founded the institute. The schema
fixes the argument order: `heads(Levina, North Institute)`, not
`heads(North Institute, Levina)`.

The first sentence supports employment on its own. A local rule can
look for a construction like *went to work at* between a person and
an organization. [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/04 Синтаксический разбор|A syntactic parse]] helps identify the arguments when
word order changes. But the sentence *Levina did not work at the
North Institute* contains the same names. The negation has to be
interpreted. So do expressions like *may take over*. If the schema
cannot represent plans or uncertainty, the extractor should not
store those cases as events that took place.

The sentence about the appointment has only *she* and *the institute*.
A document-level model works with clusters of mentions, so it can assign
`heads` to the same local pair. The record can cite sentence 2 for the
appointment and sentence 1 for the arguments and time anchor. The
[DocRED annotations](https://aclanthology.org/P19-1074/) distinguish
relation triples from the sentences that support them. Seeing two
names together does not establish a specific relation.

A learned classifier can pool the contextual vectors of each entity's
mentions and score the ordered pair. Several predicates may apply at
once. One design uses a logit and sigmoid per predicate. It trains
each predicate with binary cross-entropy. A single-label classifier
would miss one of the two valid relations. Type
constraints reduce the search space: `heads` takes a person first
and an organization second in this schema. Type compatibility does
not establish the relation itself.

<a id="ie-events"></a>

## Event roles and time

`heads(Levina, North Institute)` does not separate a 2023 appointment
from a statement about who runs the institute now. A database that
tracks changes can store an event triggered by *took over as director*.
The arguments fill roles: new director, organization, and start time.
The phrase *two years later* adds two years to the 2021 anchor from
the first sentence. Neither a month nor a day nor an end date is given.

~~~yaml
event: assume_directorship
person: P17
organization: O01
year: 2023
evidence_sentences: [1, 2]
~~~

`O01` stands for the North Institute record in the same toy catalog.
Time expressions can yield intervals or unresolved references.
For example, *last week* may need a publication date; without an
anchor the calendar interval remains unknown. [SUTime](https://aclanthology.org/L12-1122/)
distinguishes detecting a time expression from normalizing it with
a reference time. Connecting *two years later* to the earlier event
is another step, which cannot be done from the phrase alone.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/information-extraction-2026-11-22/wadden-dygie-fig1.png|DyGIE++ feeds sentences to BERT, enumerates spans, and passes span representations through a graph. Separate outputs score entities, relations, events, and auxiliary coreference.]]

*David Wadden, Ulme Wennberg, Yi Luan, and Hannaneh Hajishirzi,
[Entity, Relation, and Event Extraction with Contextualized Span Representations, Figure 1](https://aclanthology.org/D19-1585.pdf#page=1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Figure cropped from the surrounding PDF page.*

DyGIE++ illustrates how several tasks can share span representations.
BERT encodes a sentence with its neighbors, then the model enumerates
candidate spans. Predicted connections can update their vectors
through a graph. Separate heads score entity types, relations,
event triggers, and argument roles; coreference is auxiliary.
In the paper, the relation classifier operates on span pairs
within a sentence. Context from nearby sentences does not remove
that output constraint. Catalog linking and calendar normalization
are also outside the architecture shown.

<a id="ie-evaluation"></a>

## Measuring errors at each stage

If all four mention boundaries are correct but *she* is a singleton,
named entity scores miss the coreference error. For $B^3$, compare
each mention's predicted cluster $C$ to its gold cluster $G$.
Its precision is $|C\cap G|/|C|$ and its recall is $|C\cap G|/|G|$.
Average over mentions. Precision is 1 for all four here. Recall is
$1/2$ for each Levina mention and 1 for each institute mention.
That gives $P=1$, $R=3/4$, and $F_1=6/7$. These calculations use
aligned spans. With predicted mentions, alignment and singleton
handling must follow the evaluation protocol. This example is not
the full CoNLL score.

For linking, measure whether the gold record reached the candidate
set (recall@$K$), then evaluate the top-ranked choice. Suppose
retrieval finds the gold record for eight of ten linkable mentions
and reranking selects six correctly. That is $8/10$ candidate
recall, $6/8$ accuracy conditional on retrieval success, and $6/10$
end-to-end linking accuracy on the ten aligned mentions. Score
missing catalog records and mention boundary errors separately.

For relation extraction with fixed entities, compare sets of triples.
Suppose the classifier scores `works_for` at $0.86$, `heads` at $0.75$,
and `founded` at $0.58$. A $0.7$ threshold keeps the two gold triples.
A $0.5$ threshold also keeps the unsupported one: precision drops to
$2/3$, recall stays at 1, and $F_1=0.8$. In the full system, a wrong
entity ID can turn a correct predicate into both a false positive
and a missed gold triple. To separate the errors, compare results
with gold entities and with predicted ones. For events, score the
trigger, argument roles, and normalized time separately.

<details>
<summary>Verify the scores, clusters, and triples in Python</summary>

~~~python
from math import exp, isclose

def softmax(scores):
    shift = max(scores)
    weights = [exp(s - shift) for s in scores]
    return [w / sum(weights) for w in weights]

# m3: rank m1, m2, and the empty antecedent.
p = softmax([0.4 + 0.7 + 1.1, 0.4 + 0.5 - 1.2, 0.0])
assert round(p[0], 3) == 0.838

q = (0.8, 0.6)
catalog = {"P17": (0.6, 0.8), "P54": (0.8, 0.6), "P81": (0.0, 1.0)}
dense = {k: sum(a * b for a, b in zip(q, v)) for k, v in catalog.items()}
assert isclose(dense["P17"], 0.96)
top2 = sorted(dense, key=dense.get, reverse=True)[:2]
assert top2 == ["P54", "P17"]
cross = dict(zip(top2, softmax([1.1, 2.7])))
assert max(cross, key=cross.get) == "P17"
assert round(cross["P17"], 3) == 0.832

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0

gold_clusters = [{1, 3}, {2, 4}]
pred_clusters = [{1}, {3}, {2, 4}]
def membership(groups):
    return {m: group for group in groups for m in group}
g, c = membership(gold_clusters), membership(pred_clusters)
bp = sum(len(c[m] & g[m]) / len(c[m]) for m in g) / len(g)
br = sum(len(c[m] & g[m]) / len(g[m]) for m in g) / len(g)
assert (bp, br) == (1.0, 0.75) and isclose(f1(bp, br), 6 / 7)

pair = ("P17", "O01")
gold = {(pair[0], r, pair[1]) for r in ("works_for", "heads")}
scores = dict(zip(("works_for", "heads", "founded"), (0.86, 0.75, 0.58)))
for threshold, expected in [(0.7, (1, 1, 1)), (0.5, (2 / 3, 1, 0.8))]:
    pred = {(pair[0], r, pair[1]) for r, score in scores.items() if score >= threshold}
    p, r = len(pred & gold) / len(pred), len(pred & gold) / len(gold)
    assert all(isclose(a, b) for a, b in zip((p, r, f1(p, r)), expected))

# The second sentence provides the offset; the first provides its anchor.
records_by_mention = {1: "P17", 2: "O01"}
def verified_ids(groups):
    result = {}
    for group in groups:
        ids = {records_by_mention[m] for m in group if m in records_by_mention}
        if len(ids) == 1:
            result.update({m: next(iter(ids)) for m in group})
    return result

linked = verified_ids(gold_clusters)
event = ("assume_directorship", linked[3], linked[4], 2021 + 2, (1, 2))
assert event == ("assume_directorship", "P17", "O01", 2023, (1, 2))
assert 3 not in verified_ids(pred_clusters)
~~~

</details>

A model that generates the whole record can be evaluated with the same
metrics. Its output can be valid YAML and still name the wrong person or
year. The next chapter treats different NLP tasks as instances of one
format: text in, text out.

## Sources

- Daniel Jurafsky and James H. Martin, *Speech and Language Processing*: [Chapter 21](https://web.stanford.edu/~jurafsky/slp3/21.pdf) on relations, events, and time; [Chapter 24](https://web.stanford.edu/~jurafsky/slp3/24.pdf) on coreference and linking.
- Kenton Lee et al., [End-to-end Neural Coreference Resolution](https://aclanthology.org/D17-1018/), Sections 2–4: spans, antecedent scores, and training.
- Ledell Wu et al., [Scalable Zero-shot Entity Linking with Dense Entity Retrieval](https://aclanthology.org/2020.emnlp-main.519/), Sections 3–4: given mention spans, retrieval, and reranking.
- Yuan Yao et al., [DocRED](https://aclanthology.org/P19-1074/), Section 3: entity clusters, relations, and supporting sentences.
- David Wadden et al., [DyGIE++](https://aclanthology.org/D19-1585/), Section 2: task outputs and contextual span representations.
- Angel X. Chang and Christopher Manning, [SUTime](https://aclanthology.org/L12-1122/), Sections 2–3: rules and normalization; Sameer Pradhan et al., [Scoring Coreference Partitions of Predicted Mentions](https://aclanthology.org/P14-2006/), Section 2: evaluating gold and predicted mentions.

**Previous:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03b Классификация документов|Document classification]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5 and text-to-text tasks]].
