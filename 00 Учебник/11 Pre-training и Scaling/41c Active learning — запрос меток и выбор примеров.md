---
title: Active learning — запрос меток и выбор примеров
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-28
last_verified: 2026-10-28
primary_sources:
  - https://www.cs.cmu.edu/~bsettles/pub/settles.activelearning.pdf
  - https://arxiv.org/abs/1708.00489v4
  - https://github.com/modAL-python/modAL/tree/bba6f6fd00dbb862b1e09259b78caf6cffa2e755
  - https://mlsysbook.ai/vol1/09_data_selection/09_data_selection.html
---

# 41c. Active learning — запрос меток и выбор примеров

Suppose a document classifier starts with 200 annotated documents and a pool
of 10,000 documents whose labels have not been collected. An annotator can
review another 100 documents. Selecting those documents changes what the model
can learn from the annotation budget: a hundred near-duplicates may provide
less new information than a hundred examples of different unresolved cases.
These counts describe a selection problem, not a measured advantage over random
sampling.

In pool-based active learning, the learner repeatedly chooses examples from the
unlabelled pool, requests their labels, and trains on the enlarged labelled set.
The labels are unavailable when the query is chosen. This differs from the
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting|labelled pruning problem]]:
an EL2N calculation can compare predictions with known targets, while an active
learner must decide which targets are worth obtaining.

<a id="annotation-loop"></a>
## The pool, the labelled set, and the annotation loop

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/active-learning-2026-10-28/active-learning.png]]

*The active learning workflow from Tivadar Danka's modAL documentation:
gather data, train a model, measure uncertainty, request labels, and train again.
The “Is my model accurate?” decision requires an evaluation protocol; confidence
on the queried examples alone cannot establish deployment accuracy.
[Original documentation and figure](https://github.com/modAL-python/modAL/blob/bba6f6fd00dbb862b1e09259b78caf6cffa2e755/docs/source/content/overview/modAL-in-a-nutshell.rst),
[MIT license](/licenses/modal-bba6f6fd-MIT.txt). Complete original image, unchanged.*

Let $L_t$ contain the labelled examples available in round $t$, and let $U_t$
contain the remaining candidate IDs. After fitting a model $\theta_t$ on $L_t$,
the query strategy selects a batch $Q_t\subseteq U_t$. An oracle supplies labels
for that batch. An oracle can be a human annotator, a laboratory experiment, or
another procedure that provides the required target; it is not assumed to be
infallible.

After successful annotation, the sets change as follows:

$$
\begin{aligned}
L_{t+1}&=L_t\cup\{(x_i,y_i):i\in Q_t\},\\
U_{t+1}&=U_t\setminus Q_t.
\end{aligned}
$$

The next model uses the expanded labelled set. It may be trained from scratch
or updated from the previous checkpoint, but the choice changes both computation
and the experiment. A comparison between query strategies should use the same
training policy, initial labelled set, and annotation budget.

In a real annotation queue, a requested label may be pending, rejected, or judged
unresolvable. Such an item has not become a labelled training example. Separating
pending IDs from available pool IDs prevents two annotators from receiving the
same request unintentionally; only accepted annotations enter $L_{t+1}$.
The executable loop later in the chapter uses an immediate, reliable toy oracle
so that these engineering delays do not obscure the set updates.

Settles distinguishes three acquisition settings. Pool-based selection compares
many available candidates before querying. Stream-based selection decides on
each incoming item whether to request its label, often without retaining the
whole stream. Membership-query synthesis constructs a new input to be labelled.
The last setting can produce inputs a human cannot interpret, so a mathematically
informative query need not be a usable annotation task. The examples here use
a fixed pool of real-valued inputs rather than query synthesis.

<a id="uncertainty"></a>
## Uncertainty sampling uses the predicted label distribution

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/active-learning-2026-10-28/motivating-example.png]]

*The modAL motivating example. Black points are unlabelled candidates; blue and
red outlined points already have labels. The background shows the classifier's
probability regions. Candidates in the pale transition band are uncertain under
this model, while candidates deep inside either coloured region are predicted
more confidently. This is an illustration of query selection, not a learning
curve or proof that a queried label improves accuracy. Tivadar Danka,
[original figure](https://github.com/modAL-python/modAL/blob/bba6f6fd00dbb862b1e09259b78caf6cffa2e755/docs/source/content/overview/img/motivating-example.png),
[MIT license](/licenses/modal-bba6f6fd-MIT.txt). Complete original image, unchanged.*

A probabilistic classifier supplies $p_\theta(y=c\mid x)$ for each class.
No true label is needed to calculate the following three scores. They retain
different amounts of the predicted distribution.

Least-confidence sampling uses

$$
U(x)=1-\max_c p_\theta(y=c\mid x).
$$

It queries the largest value of $U$. If the most likely class has probability
0.7, this score is 0.3, regardless of how the remaining probability is distributed.
For a calibrated classifier, that has the interpretation of predicted
misclassification probability; an uncalibrated classifier may assign 0.99 to
many incorrect predictions, so the score does not establish actual error risk.

The modAL documentation defines the next measure in these terms:

> Classification margin is the difference in probability of the first and
> second most likely prediction, that is, it is defined by

$$
M(x)=p_\theta(\hat y_1\mid x)-p_\theta(\hat y_2\mid x).
$$

> where $\hat y_1$ and $\hat y_2$ are the first and second most likely classes.

Tivadar Danka, [Uncertainty sampling](https://github.com/modAL-python/modAL/blob/bba6f6fd00dbb862b1e09259b78caf6cffa2e755/docs/source/content/query_strategies/uncertainty_sampling.ipynb),
MIT. The source's $\hat x$ notation for a predicted class is replaced by
$\hat y$ in the displayed definition and its explanatory sentence.

Margin sampling queries the smallest $M$. A small gap means that the two leading
classes are hard to distinguish under this model. For many classes, however,
it still ignores how probability is shared among all other labels.

Entropy sampling uses the entire vector:

$$
H(x)=-\sum_c p_\theta(c\mid x)\log p_\theta(c\mid x).
$$

The convention $0\log0=0$ gives the limiting contribution of an impossible class.
For an outcome with probability $p_c$, $-\log p_c$ is its surprisal; averaging
these values using $p_c$ as the weights gives entropy.
The largest entropy occurs at a uniform distribution; a distribution concentrated
on one class has zero entropy. The query strategy selects the largest $H$.
This is uncertainty under the model, not information gain from a label already
observed, and not a count of the guesses an annotator would need.

### Three strategies need not select the same example

The original modAL example uses the following three probability vectors:

```python
import numpy as np

# Original probability matrix and margin computation from modAL (MIT).
proba = np.array([[0.1 , 0.85, 0.05],
                  [0.6 , 0.3 , 0.1 ],
                  [0.39, 0.61, 0.0 ]])
part = np.partition(-proba, 1, axis=1)
margin = - part[:, 0] + part[:, 1]

# The following calculations and checks make all three outputs explicit.
uncertainty = 1 - proba.max(axis=1)
safe = np.where(proba > 0, proba, 1.)
entropy = -(proba * np.log(safe)).sum(axis=1)
assert np.allclose(proba.sum(axis=1), 1.)
assert np.allclose(uncertainty, [.15, .4, .39])
assert np.allclose(margin, [.75, .3, .22])
assert np.allclose(entropy, [.51818621, .89794572, .66874809])
assert uncertainty.argmax() == 1
assert margin.argmin() == 2
assert entropy.argmax() == 1
print("least confidence", uncertainty, "margin", margin, "entropy", entropy)
print("selected rows, counted from 1:",
      uncertainty.argmax()+1, margin.argmin()+1, entropy.argmax()+1)
```

The first row has a dominant class, so all three criteria find it relatively
confident. The second row has a larger total probability outside the leading
class than the third row: 0.4 versus 0.39. Least confidence therefore chooses
row 2. The third row has the smallest gap between its two leading probabilities,
0.22, so margin sampling chooses row 3. Entropy chooses row 2 because its
probability is spread across three classes rather than two.

The array and two-line `partition` calculation are reproduced from the
[pinned notebook](https://github.com/modAL-python/modAL/blob/bba6f6fd00dbb862b1e09259b78caf6cffa2e755/docs/source/content/query_strategies/uncertainty_sampling.ipynb),
[MIT license](/licenses/modal-bba6f6fd-MIT.txt); the checks and printed output
are additions. No model is trained in this score comparison.

For binary classification, write $p=p_\theta(y=1\mid x)$. Least confidence is
$\min(p,1-p)$ and the margin is $|2p-1|$. Binary entropy increases as $p$
approaches 0.5 from either side. All three rankings therefore prefer the examples
closest to 0.5, apart from ties. This equivalence does not extend to three or
more classes, as the original example demonstrates.

<a id="committee"></a>
## Query-by-committee compares competing predictions

A single model can be confidently wrong in a region poorly represented by the
labelled set. Query-by-committee trains several models that fit the current
labelled data but can disagree on unlabelled candidates. Their disagreement
provides another query criterion.

In the version-space interpretation described by Settles §3.2, the committee
approximates a set of hypotheses consistent with the observed labels. A queried
label can rule out hypotheses that predicted another class. Practical ensembles
may use different initializations or bootstrap samples rather than explicitly
enumerating consistent hypotheses. This is an approximation: several models
with the same blind spot can agree on a wrong label.

### Vote entropy

The original modAL example asks three classifiers to predict five candidates.
Its first candidate receives votes `[0, 1, 0]`, while the third receives
`[0, 1, 2]`. The documentation explains the first vote distribution as follows:

> Each instance has a corresponding probability distribution to it: the
> distribution of class labels when picking a classifier by random. In the
> first instance, there are two votes for `0`, one votes for `1` and zero votes
> for `2`. In this case, this distribution is `[0.6666, 0.3333, 0.0]`.

Tivadar Danka, [Disagreement sampling](https://github.com/modAL-python/modAL/blob/bba6f6fd00dbb862b1e09259b78caf6cffa2e755/docs/source/content/query_strategies/Disagreement-sampling.rst),
MIT. Original English passage, unchanged.

If $v_c(x)$ is the fraction of committee votes for class $c$, vote entropy is

$$
H_{\mathrm{vote}}(x)=-\sum_c v_c(x)\log v_c(x).
$$

Three different votes give a uniform distribution and entropy $\log3$.
Unanimous votes have entropy zero. The following program uses the source's
five vote rows, calculates their distributions exactly, and selects the third
candidate rather than treating rounded decimal counts as exact probabilities.

```python
import numpy as np

# Original hard-vote rows from modAL's Disagreement sampling example (MIT).
votes = np.array([[0, 1, 0], [1, 1, 2], [0, 1, 2],
                  [2, 2, 2], [1, 2, 2]])
vote_probabilities = np.eye(3)[votes].mean(axis=1)
safe = np.where(vote_probabilities > 0, vote_probabilities, 1.)
vote_entropy = -(vote_probabilities * np.log(safe)).sum(axis=1)
assert np.allclose(vote_probabilities.sum(axis=1), 1.)
assert np.allclose(vote_entropy, [.63651417, .63651417, np.log(3),
                                 0., .63651417])
assert vote_entropy.argmax() == 2
print("vote distributions", vote_probabilities)
print("vote entropies", vote_entropy)
```

Hard votes discard confidence. A classifier assigning `(0.51, 0.49)` casts
the same vote as one assigning `(0.99, 0.01)`, although the distributions differ.
Probabilistic committee measures retain that information.

### Consensus entropy and disagreement with the consensus

For $R$ committee members, define the average prediction
$\bar p(c\mid x)=R^{-1}\sum_r p_r(c\mid x)$. Consensus entropy is $H(\bar p)$.
It can be high either because members disagree, or because every member is
uncertain in the same way.

Compare two binary committees. In the first, both members predict `(0.5, 0.5)`.
In the second, one predicts `(0.99, 0.01)` and the other `(0.01, 0.99)`.
Both committees have consensus `(0.5, 0.5)` and consensus entropy $\log2$.
Only the second committee contains a disagreement between members.

Settles also describes average KL divergence from member predictions to the
consensus. For discrete distributions,
$\operatorname{KL}(p\|q)=\sum_c p_c\log(p_c/q_c)$; equal distributions give zero
because every nonzero term has a probability ratio of one. The committee score is

$$
D(x)=\frac1R\sum_{r=1}^{R}
\operatorname{KL}(p_r(\cdot\mid x)\,\|\,\bar p(\cdot\mid x)).
$$

The identical members in the first committee give $D=0$. In the second,
$D=\log2-H(0.99,0.01)\approx0.6371$. This measure responds to differences
among the members rather than uncertainty shared by all of them. It is not
automatically a calibrated estimate of epistemic uncertainty for an arbitrary
ensemble.

For either confident member of the second committee, the calculation is
$0.99\log(0.99/0.5)+0.01\log(0.01/0.5)$. Averaging the two equal values gives
the same 0.6371. More generally, average KL to the consensus equals consensus
entropy minus the mean of the individual entropies, an identity checked below.

```python
import numpy as np

def committee_statistics(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or not all(p.shape) or np.any(p < 0)
            or not np.isfinite(p).all()):
        raise ValueError("expected finite member-by-class probabilities")
    if not np.allclose(p.sum(axis=1), 1.):
        raise ValueError("each member distribution must sum to one")
    consensus = p.mean(axis=0)
    safe_p = np.where(p > 0, p, 1.)
    safe_consensus = np.where(consensus > 0, consensus, 1.)
    individual_entropy = -(p * np.log(safe_p)).sum(axis=1)
    consensus_entropy = -(consensus * np.log(safe_consensus)).sum()
    average_kl = (p * (np.log(safe_p)-np.log(safe_consensus))).sum(axis=1).mean()
    assert np.isclose(average_kl, consensus_entropy-individual_entropy.mean())
    return consensus_entropy, average_kl

same = committee_statistics([[.5, .5], [.5, .5]])
opposed = committee_statistics([[.99, .01], [.01, .99]])
assert np.isclose(same[0], opposed[0])
assert np.isclose(same[1], 0.) and opposed[1] > .63
print("same uncertain predictions", same)
print("opposed confident predictions", opposed)
```

Every member distribution must be normalized before this calculation. A row
such as `[0.8, 0.1, 0.0]` sums to 0.9 and fails the check. That row appears in
the pinned documentation's consensus example; the valid distributions above
avoid treating the missing probability mass as a class prediction.

<a id="expected-gradient"></a>
## Expected model change averages over unknown labels

A gradient can depend on a label that has not been obtained yet. Expected
gradient length handles this by enumerating possible labels, computing the
gradient each would produce, and averaging the lengths using the model's
current label probabilities:

$$
G(x)=\sum_c p_\theta(c\mid x)
\|\nabla_\theta\ell(\theta;x,c)\|_2.
$$

Settles §3.3 motivates this per-example approximation when the current training
gradient is close to zero after fitting the labelled data. The exact gradient
of an enlarged training objective also depends on loss normalization and
regularization. A large expected per-example gradient is a query heuristic,
not a proof that the next optimization run improves held-out accuracy.

For binary logistic regression without a bias, $p=\sigma(w^\top h)$ and
$\nabla_w\ell=(p-y)h$. The two possible labels give

$$
\begin{aligned}
G(x)&=p(1-p)\|h\|_2+(1-p)p\|h\|_2\\
&=2p(1-p)\|h\|_2.
\end{aligned}
$$

The probability term is largest at 0.5, but feature magnitude also contributes.
Take $w=(\log9/5,0)$ and features $(0,1)$ and $(5,0)$. Their probabilities
are 0.5 and 0.9. Expected gradient lengths are 0.5 and 0.9: the more confident
example ranks higher because its feature norm is larger.

```python
import numpy as np

w = np.array([np.log(9)/5, 0.])
features = np.array([[0., 1.], [5., 0.]])
p = 1 / (1 + np.exp(-(features @ w)))
expected_norms = []
for h, probability in zip(features, p):
    gradient_if_zero = probability * h
    gradient_if_one = (probability-1) * h
    expected_norms.append((1-probability)*np.linalg.norm(gradient_if_zero)
                          + probability*np.linalg.norm(gradient_if_one))
expected_norms = np.array(expected_norms)
assert np.allclose(p, [.5, .9])
assert np.allclose(expected_norms, [.5, .9])
assert np.allclose(expected_norms, 2*p*(1-p)*np.linalg.norm(features, axis=1))
print("probabilities", p, "expected gradient lengths", expected_norms)
```

This dependence on feature scaling is one reason expected gradient length needs
careful preprocessing. A confidently wrong model can also assign a small weight
to the true class in the expectation, since the weights come from its own
predictions. The criterion requires no oracle label at query time, but inherits
errors in those predictive probabilities.

<a id="executable-loop"></a>
## A complete two-round annotation loop

The next program fits a binary logistic classifier, scores the remaining pool,
asks an oracle for one label, and moves that ID into the labelled set. It repeats
the process twice. The oracle is a deterministic function of the queried input;
it is evaluated only for the chosen ID, not for every pool item before ranking.

The classifier is fitted from zero in each round using 200 gradient steps.
The labelled seed contains one example of each class. This is a small execution
example, not an experiment demonstrating that uncertainty sampling outperforms
random selection. An independent test set and repeated query runs would be
needed for such a comparison.
Scores within $10^{-12}$ are treated as ties in this example; the smaller ID wins.

```python
import numpy as np

features = {10: -2., 20: 2., 101: -.2, 205: .2, 307: -1., 409: 1.}
labelled = {10: 0, 20: 1}
pool = {101, 205, 307, 409}
oracle_calls = []

def sigmoid(z):
    return np.exp(-np.logaddexp(0., -z))

def fit_labelled(known):
    ids = sorted(known)
    X = np.array([[features[i], 1.] for i in ids])
    y = np.array([known[i] for i in ids])
    parameters = np.zeros(2)
    for _ in range(200):
        probabilities = sigmoid(X @ parameters)
        parameters -= .1 * (X.T @ (probabilities-y) / len(y))
    return parameters

def oracle(example_id):
    oracle_calls.append(example_id)
    return int(features[example_id] > 0)

trace = []
for round_index in range(2):
    parameters = fit_labelled(labelled)
    candidates = sorted(pool)
    X_pool = np.array([[features[i], 1.] for i in candidates])
    p = sigmoid(X_pool @ parameters)
    uncertainty = np.minimum(p, 1-p)
    # Use the smallest ID among scores tied within the stated tolerance.
    tied = np.flatnonzero(np.isclose(uncertainty, uncertainty.max(),
                                   rtol=0., atol=1e-12))
    chosen_position = int(tied[0])
    chosen_id = candidates[chosen_position]
    before = len(labelled), len(pool)
    new_label = oracle(chosen_id)
    labelled[chosen_id] = new_label
    pool.remove(chosen_id)
    assert set(labelled).isdisjoint(pool)
    assert (len(labelled), len(pool)) == (before[0]+1, before[1]-1)
    trace.append((round_index, chosen_id, new_label, len(labelled), len(pool)))
assert len(oracle_calls) == len(set(oracle_calls)) == 2
assert len(labelled) == 4 and len(pool) == 2
final_parameters = fit_labelled(labelled)
print("round, queried ID, accepted label, labelled count, pool count")
for row in trace:
    print(*row)
print("final parameters", final_parameters)
```

The seed labels and oracle labels agree in this toy problem. A real oracle can
return an incorrect label, request more context, or abstain. Annotation quality
therefore needs its own checks: clear instructions, repeated review where needed,
and a record of disagreements. Model uncertainty does not identify which human
reviewer is correct.

The code scores only the current `pool` and removes an ID after receiving its
label. In production, the persistent record should also identify the dataset
version, model checkpoint, query score, query round, and annotation status.
Otherwise an interrupted job can repeat requests or attach a label to a changed
example. The stable-ID requirement is the same as in labelled data scoring,
but active learning adds a state transition between unlabelled, pending, and
accepted-labelled records.

<a id="batch-and-diversity"></a>
## A batch needs diversity as well as high individual scores

Suppose a pool contains 100 nearly identical documents about one ambiguous
phrase. A model can give all 100 high uncertainty. Taking the top 100 scores
then spends the entire batch on that one unresolved pattern. After receiving
the first few labels, the remaining labels may contribute little new information.
A batch chosen before any of its labels arrive cannot adapt to that feedback.

Settles §6.1 discusses this overlap between high-scoring queries. Sener and
Savarese study it in the batch acquisition setting for CNNs and formulate
selection as coverage of a representation space. Their k-center strategy picks
points far from already covered points, rather than taking the largest individual
uncertainties. Its geometric objective and approximation guarantee concern
distances in the chosen representation; they do not give a universal accuracy
guarantee or establish that uncertainty sampling fails for every deep model.

The two criteria can favour different inputs. Uncertainty may concentrate near
one decision boundary, while coverage can select a distant but confidently
predicted cluster. A distant point may also be an outlier rather than a useful
new mode. Representation choice, outlier handling, the existing labelled seed,
and the size of the query batch affect the result. Similarity filtering or
clustering can reduce duplicates, but each proposed batch should be evaluated
against a same-budget random batch rather than assumed to be more informative.

Annotation batch size also affects latency. Larger batches can keep multiple
annotators occupied and amortize a costly training run. Smaller batches allow
the model to incorporate labels sooner before choosing the next questions.
Neither query count nor training time alone captures that trade-off.

<a id="evaluation-and-cost"></a>
## Evaluate on data that was not selected by the query rule

The queried examples deliberately follow a selection distribution. Their
unweighted accuracy need not match accuracy on the deployment distribution.

For example, suppose easy cases constitute 90% of traffic and are all classified
correctly, while the remaining 10% are hard cases with 50% accuracy. Overall
accuracy is $0.9\cdot1+0.1\cdot0.5=0.95$. A query set containing only hard cases
has 50% accuracy. Neither arithmetic result is wrong, but they measure performance
under different input distributions.

Hold out independently sampled, genuinely labelled evaluation data. A comparison
should use the same initial labelled seed, the same model-training procedure,
and several query/training random seeds for each strategy. Compare quality after
equal numbers of accepted labels, and also after equal annotation time or cost
when examples require different effort. For imbalanced tasks, report the groups
or classes that a global score can hide.

If the labelled seed lacks a class, the model may assign those examples
confidently to its existing classes. Uncertainty-only selection can then overlook
the new class. If the candidate pool contains none of its examples, no query
rule can obtain its labels from that pool. When the examples are present,
random exploration and explicit coverage constraints can help expose them;
the right mixture depends on the task and needs evaluation.

Settles §6.3 reports why a smaller number of queried examples need not mean less
annotation effort. A difficult document can take much longer to label than an
easy one, and the cost can depend on the annotator. Dividing entropy by estimated
cost is a possible heuristic, not a generally proven improvement-per-dollar
objective. The score is not itself measured model improvement, and predicted
annotation time can be wrong.

For a concrete accounting example, 100 queries taking two minutes each use
200 annotation minutes, while 60 queries taking five minutes each use 300.
The second strategy requests fewer labels but consumes more time. Model fitting,
scoring the pool, annotation review, and queue delays add further costs. A claimed
label saving should state whether it includes those stages.

Stopping can follow an exhausted annotation budget, a task-quality target, or
a lack of useful gains on held-out data. Rising confidence on the remaining pool
alone is insufficient: a model can become confidently wrong or remain unaware
of underrepresented cases. The loop in the first figure ends only after its
accuracy decision has been given a concrete, independently checked meaning.

---

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting|Отбор обучающих примеров — EL2N, GraNd и forgetting]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding]]

**Источники:** Burr Settles,
[Active Learning Literature Survey](https://www.cs.cmu.edu/~bsettles/pub/settles.activelearning.pdf),
§§2.1–2.3, 3.1–3.3, 6.1–6.3 and 6.7;
Tivadar Danka and Peter Horvath,
[modAL documentation](https://github.com/modAL-python/modAL/tree/bba6f6fd00dbb862b1e09259b78caf6cffa2e755/docs/source/content);
Ozan Sener and Silvio Savarese,
[A Core-Set Approach, v4](https://arxiv.org/abs/1708.00489v4), §§3–4.
