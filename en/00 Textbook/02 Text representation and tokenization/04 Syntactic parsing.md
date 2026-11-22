---
title: Syntactic parsing
description: "Constituency and dependency trees, CKY, arc-standard parsing, and how to score a parse."
type: textbook-chapter
status: canonical
locale: en
source_language: en
translation_of: "00 Учебник/02 Представление текста и токенизация/04 Синтаксический разбор.md"
translation_status: current
canonical_revision: "b78766a01357f0c168e57b7d8e6732febce78e86c3352866499112f137b30e99"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: "Constituency and dependency trees"
    anchor: syntax-two-trees
  - label: "CKY constituency parsing"
    anchor: syntax-cky
  - label: "Arcs in Universal Dependencies"
    anchor: syntax-ud
  - label: "The arc-standard dependency parser"
    anchor: syntax-arc-standard
  - label: "Evaluating syntactic parses"
    anchor: syntax-evaluation
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/19.pdf
  - https://web.stanford.edu/~jurafsky/slp3/20.pdf
  - https://link.springer.com/article/10.1007/s10579-025-09826-3
  - https://aclanthology.org/P17-1076/
  - https://aclanthology.org/D14-1082/
  - https://universaldependencies.org/format.html
---

In *I saw a man with a telescope*, the observer may have used the
telescope, or the man may have had it. On the first reading, *with a
telescope* describes how the observer saw the man. On the second,
it describes which man the observer saw. The words and part-of-speech
tags are the same. The phrase attaches to a different part of the
sentence.

<a id="syntax-two-trees"></a>

## Constituents and dependencies

A constituency tree divides a sentence into nested phrases. `NP`
stands for noun phrase, `VP` for verb phrase, and `PP` for
prepositional phrase. On both readings, `[PP with [NP a telescope]]`
is one phrase. The brackets show where it attaches within the verb
phrase. Part-of-speech tags are omitted.

~~~text
The observer has the telescope:
[VP [VP saw [NP a man]]
    [PP with [NP a telescope]]]

The man has the telescope:
[VP saw
    [NP [NP a man] [PP with [NP a telescope]]]]
~~~

In the first tree, the `PP` and the shorter verb phrase `saw a man`
are siblings under `VP`. In the second, the `PP` sits inside an
`NP` that covers `a man with a telescope`. In either case, the full
sentence `S` joins the subject `I` with the outer verb phrase.

A dependency tree uses words as nodes and draws arcs from heads to
dependents. The figure shows both annotations for one Italian sentence.
Above the words, `PP` covers *in puro stile scream*. Below the words,
the corresponding attachment is the `obl` arc from *cantata* to
*stile*. The preposition *in* has its own `case` arc from *stile*.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/syntax-2026-11-22/parallel-trees-fig1.png|A phrase tree above La canzone è cantata in puro stile scream contains a PP. Dependency arcs below run from cantata to stile and from stile to in.]]

*Chiara Alzetta, Alessio Miaschi, Felice Dell’Orletta, Giulia Venturi, and Simonetta Montemagni,
[Parallel Trees, Figure 1](https://link.springer.com/article/10.1007/s10579-025-09826-3#Fig1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure, unchanged.*

<a id="syntax-cky"></a>

## How CKY builds parses bottom up

Number the boundaries between the words. A span $[i,j)$ runs from
boundary $i$ up to boundary $j$; the first word occupies $[0,1)$.

~~~text
0 I 1 saw 2 a 3 man 4 with 5 a 6 telescope 7
~~~

A small context-free grammar is enough for this sentence. The symbol
on the left of an arrow names a category. The right side contains
either a word or two adjacent categories. For example, `NP → D N`
allows a determiner and a noun to form a noun phrase. The vertical bar
separates alternative rules. `NP → I` stands in for a more detailed
analysis of the pronoun.

~~~text
S  → NP VP
VP → V NP  | VP PP
NP → D N   | NP PP
PP → P NP

NP → I     V → saw     D → a
N  → man | telescope   P → with
~~~

Fill a chart with the single words first. Span $[2,3)$ has category
`D` and $[3,4)$ has `N`. For a longer span $[i,j)$, try each interior
boundary $k$. If the chart contains `B` on $[i,k)$ and `C` on
$[k,j)$ and the grammar has `A → B C`, add `A` on $[i,j)$.
Save $k$ and the two child categories with the entry. They are enough
to recover the tree without parsing those words again.

| Span | Children | Result |
| --- | --- | --- |
| $[2,4)$ | `D` *a* and `N` *man* | `NP` |
| $[5,7)$ | `D` *a* and `N` *telescope* | `NP` |
| $[1,4)$ | `V` *saw* and `NP` *a man* | `VP` |
| $[4,7)$ | `P` *with* and `NP` *a telescope* | `PP` |
| $[2,7)$ | `NP` *a man* and `PP` *with a telescope* | `NP` |
| $[1,7)$, $k=2$ | `V` and the longer `NP` | `VP` |
| $[1,7)$, $k=4$ | The shorter `VP` and `PP` | `VP` |
| $[0,7)$ | `NP` *I* and `VP` with its two alternatives | `S` |

At $k=2$ the prepositional phrase stays inside the longer `NP`.
At $k=4$ it attaches to the shorter `VP`. Keeping just one backpointer
for `VP` on $[1,7)$ would discard one reading. Keeping both lets
us recover two trees from a single `S` entry.

The Cocke–Kasami–Younger algorithm (CKY) reuses results for smaller
spans, much as [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/03 Морфология и разметка последовательностей#hmm-viterbi|Viterbi]]
reuses results for prefixes. Here a state has two boundaries and a
category, and each recurrence combines two children. Trying every split
and binary rule takes $O(n^3|R_{\mathrm{bin}}|)$ time. A chart of
Boolean entries or best scores uses $O(n^2|\mathcal N|)$ space,
where $\mathcal N$ is the set of categories. Storing every split can
take cubic space, and enumerating all the trees can take exponential
time. Other grammars may have rules such as `NP → Pron` or rules with
more than two children. Those rules need conversion or a parser that
handles them directly.

<a id="syntax-scoring"></a>

### Choosing a parse

The grammar admits both readings. Give `VP → VP PP` a score of
$-0.2$, `NP → NP PP` a score of $-0.7$, and all other rules a score
of zero. These are illustrative scores. Score each tree by adding its
rule scores. Now each chart entry can keep the maximum and the
backpointer that produced it. For `VP` on $[1,7)$ we get $-0.2$ at
$k=4$ and $-0.7$ at $k=2$. The verb attachment wins.

Write $S_A(i,j)$ for the best score of category $A$ on $[i,j)$.
For a single word it is the score of the lexical rule; an absent
category has score $-\infty$. For a longer span, compare splits and
rules:

$$
\begin{aligned}
S_A(i,j)
&=\max_{\substack{i<k<j\\A\to BC}}\big[s(A\to BC)\\
&\qquad{}+S_B(i,k)+S_C(k,j)\big].
\end{aligned}
$$

A probabilistic context-free grammar instead adds log probabilities of the rules;
the probabilities of alternatives with the same left-hand category
must sum to one.

A learned parser can score spans using context from both sides.
Stern, Andreas, and Klein subtract states at the span boundaries
in each direction of a bidirectional network. One network maps the
resulting vector to label scores; another gives the span itself a score.
The two scores for $[i,k)$ and $[k,j)$ enter the choice of split
$k$. The chart decoder adds the best subtree scores to these.
The authors also give a greedy top-down decoder.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/syntax-2026-11-22/stern-top-down-fig1.png|The left panel splits She enjoys playing tennis plus a period, first at 1 and then at 4. The right panel shows the S tree with NP, VP, and the period.]]

*Mitchell Stern, Jacob Andreas, and Dan Klein,
[A Minimal Span-Based Neural Constituency Parser, Figure 1](https://aclanthology.org/P17-1076.pdf#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Figure from the PDF, cropped from the surrounding page.*

The decoder starts with $[0,5)$ and splits after *She* at $k=1$.
The right span *enjoys playing tennis.* receives the empty label
$\varnothing$: it is needed for splitting but does not become a node
in the output tree. A split at $k=4$ separates the period from the
`VP`, so the period ends up directly under `S`. Greedy top-down
decoding chooses a split before building its children. Bottom-up search
can compare splits using the best scores of both subtrees.

<a id="syntax-ud"></a>

## Both readings in Universal Dependencies

In the basic UD tree, each word's row records the ID of its
governor (`HEAD`) and a dependency label (`DEPREL`). CoNLL-U numbers
words from 1, so `I` is word 1 and `0` is the artificial root.
The main word has `HEAD=0` and label `root`. Every other word in
the basic tree is reachable from the root, and there are no cycles.

| ID | Word | Observer has telescope: HEAD, DEPREL | Man has telescope: HEAD, DEPREL |
| ---: | --- | --- | --- |
| 1 | I | `2, nsubj` | `2, nsubj` |
| 2 | saw | `0, root` | `0, root` |
| 3 | a | `4, det` | `4, det` |
| 4 | man | `2, obj` | `2, obj` |
| 5 | with | `7, case` | `7, case` |
| 6 | a | `7, det` | `7, det` |
| 7 | telescope | `2, obl` | `4, nmod` |

Only the row for *telescope* changes. When the phrase describes the
means of seeing, its head *telescope* depends on the verb *saw* with
label `obl`. When it modifies *man*, the dependency is `nmod`.
In both cases, *with* depends on *telescope* with label `case`.
That is a UD convention; other annotation schemes can put a
preposition elsewhere in the tree. CoNLL-U stores these arcs alongside
the parts of speech and features introduced in the morphology chapter.

<a id="syntax-arc-standard"></a>

## Dependency parsing with a stack

The arc-standard algorithm has a stack, a buffer of unread words,
and a set of arcs. Start with only `0` on the stack. `SHIFT` moves
the next word onto the stack. If the top two words are `[…, a, b]`
with `b` on top, `LEFT-ARC` adds `b → a` and pops `a`.
`RIGHT-ARC` adds `a → b` and pops `b`. The operation names
indicate which side the dependent is on. The artificial root cannot
be a dependent; the final arc attaches the main word to the root.

Two `SHIFT` operations put `[0, 1, 2]` on the stack.
`LEFT-ARC(nsubj)` removes `I` and adds `2 → 1`.
Read `a man` and use `LEFT-ARC(det)` to attach the article to `man`.
The stack is now `[0, 2, 4]` and the buffer is `[5, 6, 7]`.

| Point in the parse | Observer has the telescope | Man has the telescope |
| --- | --- | --- |
| Stack `[0, 2, 4]` | `RIGHT-ARC(obj)`: `2 → 4`, leaving `[0, 2]` | Keep `4` on the stack |
| Three `SHIFT` operations | `[0, 2, 5, 6, 7]` | `[0, 2, 4, 5, 6, 7]` |
| `LEFT-ARC(det)`, `LEFT-ARC(case)` | Add `7 → 6` and `7 → 5`, leaving `[0, 2, 7]` | Same arcs, leaving `[0, 2, 4, 7]` |
| Attach `7` | `RIGHT-ARC(obl)`: `2 → 7` | `RIGHT-ARC(nmod)`: `4 → 7`, then `RIGHT-ARC(obj)`: `2 → 4` |
| `RIGHT-ARC(root)` | `0 → 2`, leaving `[0]` | `0 → 2`, leaving `[0]` |

Once a greedy parser pops `man` with `RIGHT-ARC(obj)`, it can no
longer make `man` the head of `telescope`. A transition classifier
scores the valid next actions using the stack, buffer, and arcs
built so far. Chen and Manning train a neural network to predict
transitions from gold trees, using features from words, parts of speech,
and existing dependency labels. Either trace above takes seven shifts
and seven attachments. This counts transitions, not the work the
classifier performs for each transition.

Both trees are projective: every word between a head and its dependent
is a descendant of that head. Basic arc-standard builds only projective
trees. Non-projective dependencies require another transition system or
graph-based decoding. Scoring pairs of words also needs a structural
constraint. If independent choices make `telescope` the head of `man`
and `man` the head of `telescope`, they form a cycle. A global decoder
searches for a tree, even when that means giving up a locally best arc.

<a id="syntax-evaluation"></a>

## What parsing metrics count

Suppose the gold tree gives the telescope to the observer and the
parser attaches it to the man. For dependencies, unlabeled attachment
score (`UAS`) credits a correct head. Labeled attachment score (`LAS`)
requires both the head and the label. The mistake in row 7 yields
$6/7$ for both metrics. If the prediction were `2, nmod` instead of
`2, obl`, UAS would rise to $7/7$ but LAS would stay at $6/7$.

For constituency, compare triples of the form (label, start, end).
Count spans longer than one word here, including `S`. Five out of
six match: `S` $[0,7)$, `VP` $[1,7)$, `NP` $[2,4)$, `PP` $[4,7)$,
and `NP` $[5,7)$. The prediction omits `VP` $[1,4)$ and adds
`NP` $[2,7)$. Precision and recall are both $5/6$, so $F_1=5/6$.
The $6/7$ attachment score and the $5/6$ span score measure different
outputs of the same attachment decision. When comparing real parsers,
fix the corpus, word alignment, punctuation handling, and any handling
of artificial nodes and unary chains before comparing scores.

<details>
<summary>Verify the two parses and their scores in Python</summary>

~~~python
from collections import defaultdict, deque
from functools import cache
from itertools import product

words = "I saw a man with a telescope".split()
lex = {"I": "NP", "saw": "V", "a": "D", "man": "N", "with": "P", "telescope": "N"}
rules = [
    ("S", "NP", "VP"),
    ("VP", "V", "NP"), ("VP", "VP", "PP"),
    ("NP", "D", "N"), ("NP", "NP", "PP"),
    ("PP", "P", "NP"),
]
n = len(words)
chart = defaultdict(list)
for i, word in enumerate(words):
    chart[i, i + 1, lex[word]].append(None)
for width in range(2, n + 1):
    for i in range(n - width + 1):
        j = i + width
        for k in range(i + 1, j):
            for a, b, c in rules:
                if chart.get((i, k, b)) and chart.get((k, j, c)):
                    chart[i, j, a].append((k, b, c))


@cache
def trees(i, j, a):
    result = []
    for back in chart[i, j, a]:
        if back is None:
            result.append(((a, i, j),))
        else:
            k, b, c = back
            children = product(trees(i, k, b), trees(k, j, c))
            result.extend(((a, i, j),) + left + right for left, right in children)
    return tuple(result)


assert chart[1, 7, "VP"] == [(2, "V", "NP"), (4, "VP", "PP")]
noun, verb = [set(node for node in tree if node[2] - node[1] > 1)
              for tree in trees(0, n, "S")]
assert noun - verb == {("NP", 2, 7)}
assert verb - noun == {("VP", 1, 4)}
assert len(noun & verb) == 5 and len(verb) == len(noun) == 6

# S moves the next word onto the stack. L/R attach and remove the dependent.
prefix = ["S", "S", "L:nsubj", "S", "S", "L:det"]
pp = ["S", "S", "S", "L:det", "L:case"]
verb_actions = prefix + ["R:obj"] + pp + ["R:obl", "R:root"]
noun_actions = prefix + pp + ["R:nmod", "R:obj", "R:root"]


def parse(actions):
    stack, buffer, arcs = [0], deque(range(1, n + 1)), {}
    for action in actions:
        if action == "S":
            stack.append(buffer.popleft())
            continue
        direction, label = action.split(":")
        dep_slot = -2 if direction == "L" else -1
        head = stack[-1 if direction == "L" else -2]
        dep = stack.pop(dep_slot)
        assert dep != 0 and dep not in arcs
        if head == 0:
            assert label == "root" and stack == [0] and not buffer
        arcs[dep] = (head, label)
    assert stack == [0] and not buffer and len(arcs) == n
    return arcs


gold, pred = parse(verb_actions), parse(noun_actions)
assert (gold[7], pred[7]) == ((2, "obl"), (4, "nmod"))
assert sum(gold[i][0] == pred[i][0] for i in gold) == 6  # UAS numerator
assert sum(gold[i] == pred[i] for i in gold) == 6         # LAS numerator
~~~

</details>

An application can use a parse to propose relations between words.
For instance, the arc from *saw* to *telescope* suggests an instrument,
while the arc from *man* to *telescope* suggests a modifier of the
person. A sentence parser cannot determine the intended reading from
this sentence alone when that reading depends on earlier context.

## Sources

- Dan Jurafsky and James H. Martin, *Speech and Language Processing*, [Chapter 19](https://web.stanford.edu/~jurafsky/slp3/19.pdf), Sections 19.5–19.8: ambiguity, CKY, span scoring, and evaluation; [Chapter 20](https://web.stanford.edu/~jurafsky/slp3/20.pdf), Sections 20.1–20.4: dependencies, transition and graph-based parsing, and evaluation.
- Chiara Alzetta et al., [Parallel Trees](https://link.springer.com/article/10.1007/s10579-025-09826-3), Section 2.1 and Figure 1: two annotations of the same sentence.
- Mitchell Stern, Jacob Andreas, and Dan Klein, [A Minimal Span-Based Neural Constituency Parser](https://aclanthology.org/P17-1076.pdf), Sections 2–4: scores, the chart decoder, and top-down parsing.
- Danqi Chen and Christopher D. Manning, [A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082.pdf), Sections 2–3: stack transitions and the transition classifier.
- Universal Dependencies, [CoNLL-U](https://universaldependencies.org/format.html#syntactic-annotation), [obl](https://universaldependencies.org/u/dep/obl.html), [nmod](https://universaldependencies.org/u/dep/nmod.html), and [case](https://universaldependencies.org/u/dep/case.html).

**Previous:** [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/03 Морфология и разметка последовательностей|Morphology and sequence labeling]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|RNNs and BPTT]].
