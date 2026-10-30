---
title: Геометрический отбор — k-center, coreset и herding
type: textbook-chapter
status: canonical
source_language: en
license: CC BY-NC-SA 4.0
last_updated: 2026-10-28
last_verified: 2026-10-28
primary_sources:
  - https://jeffe.cs.illinois.edu/teaching/algorithms/notes/J-approx.pdf
  - https://arxiv.org/abs/1708.00489v4
  - https://github.com/google/active-learning/blob/efedd8f1c45421ee13af2b9ff593ad31f3835942/sampling_methods/kcenter_greedy.py
  - https://icml.cc/Conferences/2009/papers/447.pdf
  - https://sarielhp.org/book/chapters/coreset_cluster.pdf
---

# 41d. Геометрический отбор — k-center, coreset и herding

A batch of uncertain documents can contain many versions of the same document.
Querying all of them spends several annotations on a region already represented
by the first query. A geometric selector instead compares examples with the
examples already selected. Its decisions depend on a representation of each
document and a distance between representations; labels need not be available.

For a pool with one-dimensional representations $0,1,4,5,9,10$, suppose the
example at $0$ is already labelled and two more labels can be requested.
Selecting $4$ and $5$ leaves the example at $10$ five units from any labelled
example. Selecting $10$ and then $5$ leaves every example at most one unit
from the labelled set. These are statements about the chosen distance, not
about the accuracy of a classifier trained on the resulting labels.

The distinction from
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|uncertainty-based querying]]
is the selection criterion. Uncertainty asks where the current model cannot
decide between labels. Geometric coverage asks which examples remain far from
the selected set. Both criteria can be computed on the same unlabelled pool,
but they need not choose the same examples.

<a id="coverage-objective"></a>
## Distances to a set and the covering radius

Let $P$ be a finite pool and let $z_i=\phi(x_i)\in\mathbb R^d$ be a fixed
representation of example $i$. For a nonempty set of selected examples $S$,
define the distance to the selected set and its covering radius by

$$
\begin{aligned}
 a_i(S)&=\min_{j\in S}d(z_i,z_j),\\
 r(S)&=\max_{i\in P}a_i(S).
\end{aligned}
$$

The inner minimum assigns each example to its nearest selected example. The
outer maximum finds the worst represented example. A radius of $1$ means
that every pool example has a selected neighbour within distance $1$; it
does not mean that the selected examples have the correct labels or that
the model predicts neighbouring examples alike.

If $L$ contains the examples already labelled, and $b$ new examples may be
queried, the fixed-seed selection problem is

$$
\begin{aligned}
 A^*&\in\mathop{\arg\min}_{A\subseteq P\setminus L,\ |A|\le b}
             r(L\cup A),\\
 r^*&=r(L\cup A^*).
\end{aligned}
$$

The existing set $L$ is fixed in this optimization. A method that is allowed
to replace $L$ solves a different problem. When there is no labelled seed,
the usual $k$-center problem chooses all $k$ centres, starting with an
arbitrary first example.

The choice of a maximum makes coverage different from
[[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения|k-means]].
The k-means objective sums squared distances to centres, while k-center
minimizes the largest distance. A dense region contributes many terms to
the k-means sum. A single isolated example can determine the entire k-center
radius. Also, ordinary k-means centres can lie between observations; an
annotation query must identify an actual example.

<a id="farthest-first"></a>
## Farthest-first traversal

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/geometric-coresets-2026-10-28/erickson-farthest-first.png|Six iterations of Gonzalez's k-center algorithm]]

*Jeff Erickson, [Approximation Algorithms, §J.10, p. 14](https://jeffe.cs.illinois.edu/teaching/algorithms/notes/J-approx.pdf#page=14).
Complete six-panel figure and its original caption, extracted from the PDF;
CC BY-NC-SA 4.0 under the author's
[permissions](https://jeffe.cs.illinois.edu/teaching/algorithms/#more-information).
Read left to right, then continue on the lower row. Filled points are the
centres already selected. The arrows identify a point farthest from its
nearest selected centre; the dashed arcs show the corresponding coverage.
The final green arrow marks the remaining distance after six selections.*

Erickson describes the greedy construction as follows:

> Choose the $k$ center points one at a time, starting with an arbitrary input
> point as the first center. In each iteration, choose the input point that is
> farthest from any earlier center point to be the next center point.

Here “farthest” refers to distance from the selected set: maximize the
minimum distance to an earlier centre. Maximizing distance from only the
last centre would make the algorithm alternate between distant regions
without accounting for the other centres already present.

Maintain one number $a_i$ for each pool example. After selecting a new centre
$c$, update

$$
 a_i\leftarrow\min\{a_i,d(z_i,z_c)\}.
$$

The new centre can improve an example's nearest distance or leave it
unchanged. It cannot make that distance larger. The next query is an
unselected example with the largest current $a_i$.

For the six-point example, all distances are absolute differences:

<a id="coverage-trace"></a>

| Selected coordinates | $a(0)$ | $a(1)$ | $a(4)$ | $a(5)$ | $a(9)$ | $a(10)$ | Radius |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $\{0\}$ | 0 | 1 | 4 | 5 | 9 | 10 | 10 |
| $\{0,10\}$ | 0 | 1 | 4 | 5 | 1 | 0 | 5 |
| $\{0,10,5\}$ | 0 | 1 | 1 | 0 | 1 | 0 | 1 |

Initially $10$ is the farthest point. After adding it, $9$ is only one unit
from a centre, while $5$ is still five units from either centre. Thus the
second query is $5$, not $9$. The distance update accounts for redundancy
within the batch before the new labels have arrived.

<a id="approximation-guarantee"></a>
## Why the radius is at most twice optimal

The guarantee uses a metric: distances are nonnegative and symmetric, and
obey the triangle inequality. Distinct examples may have identical
representations and distance zero; the argument still applies to the
representations, although it cannot distinguish those examples' labels.

For the problem without a fixed seed, Erickson gives the following proof
(§J.10, Theorem J.7; original English, mathematical typesetting restored):

> Let $\mathrm{OPT}$ denote the optimal k-center clustering radius for $P$.
> For any index $i$, let $c_i$ and $r_i$ denote the $i$th center point and
> $i$th clustering radius computed by GonzalezKCenter.
>
> By construction, each center point $c_j$ has distance at least $r_{j-1}$
> from any center point $c_i$ with $i<j$. Moreover, for any $i<j$, we have
> $r_i\ge r_j$. Thus, $|c_i c_j|\ge r_k$ for all indices $i$ and $j$.
>
> On the other hand, at least one cluster in the optimal clustering contains
> at least two of the points $c_1,c_2,\ldots,c_{k+1}$. Thus, by the triangle
> inequality, we must have $|c_i c_j|\le2\cdot\mathrm{OPT}$ for some indices
> $i$ and $j$. We conclude that $r_k\le2\cdot\mathrm{OPT}$, as claimed.

The pairwise-distance statement concerns distinct centres. The extra point
$c_{k+1}$ is the next farthest point, used as a witness for the proof; it
is not an additional centre returned to the user.

The same reasoning extends to a fixed labelled set $L$. Suppose greedy
adds $b$ centres and finishes with radius $r>2r^*$. Consider these $b$ new
centres and a point still at distance $r$ from the final selected set.
At the moment each new centre was chosen, its distance from every earlier
centre was at least $r$: the sequence of covering radii never increases.
Thus these $b+1$ points are pairwise at least $r$ apart.

Each is also at least $r$ from every centre in $L$. No radius-$r^*$ ball
centred in $L$ can cover any of them. An optimal extension must cover all
$b+1$ using its $b$ new centres. Two points must share a new centre, giving
distance at most $2r^*<r$, a contradiction. If $r=0$, the bound is immediate.

This is an approximation bound for the radius in the specified metric.
It gives no factor-two bound on test error, required label count, training
time or the loss after retraining. A brute-force comparison on a small pool
can check an implementation against the discrete optimum, but it cannot
replace the proof for arbitrary pools.

<a id="implementation"></a>
## An implementation with stable IDs and explicit exclusion

Pool positions change when data are shuffled or filtered. Query results
should therefore contain stable example IDs, not positions in a temporary
feature array. The implementation below uses Euclidean distance, keeps the
already selected IDs fixed, and explicitly excludes selected rows even
when every distance is zero. Exact distance ties go to the smallest ID.

```python
import numpy as np


def farthest_first(features, ids, labelled_ids, budget):
    z = np.asarray(features, dtype=float)
    ids = np.asarray(ids)
    if z.ndim != 2 or min(z.shape) == 0 or not np.isfinite(z).all():
        raise ValueError("features must be a nonempty finite matrix")
    if ids.shape != (len(z),) or ids.dtype.kind not in "iu":
        raise ValueError("one integer ID is required per row")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("example IDs must be unique")
    seeds = list(labelled_ids)
    position = {int(identifier): i for i, identifier in enumerate(ids)}
    if any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) for s in seeds):
        raise ValueError("seed IDs must be integers")
    if len(set(seeds)) != len(seeds) or any(s not in position for s in seeds):
        raise ValueError("seed IDs must be unique members of the pool")
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)):
        raise ValueError("budget must be an integer")
    if not 0 <= budget <= len(z) - len(seeds):
        raise ValueError("budget exceeds the available pool")
    if not seeds and budget == 0:
        raise ValueError("coverage requires at least one centre")

    available = np.ones(len(z), dtype=bool)
    nearest = np.full(len(z), np.inf)

    def add_centre(row):
        available[row] = False
        with np.errstate(over="raise", invalid="raise"):
            distance = np.linalg.norm(z - z[row], axis=1)
        np.minimum(nearest, distance, out=nearest)

    for identifier in seeds:
        add_centre(position[identifier])

    chosen = []
    for _ in range(budget):
        rows = np.flatnonzero(available)
        farthest = nearest[rows].max()
        ties = rows[nearest[rows] == farthest]
        row = ties[np.argmin(ids[ties])]
        chosen.append(int(ids[row]))
        add_centre(row)
    return chosen, nearest


z = np.array([0., 1., 4., 5., 9., 10.])[:, None]
ids = np.array([101, 205, 309, 412, 516, 620])
chosen, nearest = farthest_first(z, ids, [101], 2)
assert chosen == [620, 412]
assert np.array_equal(nearest, [0., 1., 1., 0., 1., 0.])
print(chosen, nearest.tolist(), nearest.max())
```

With no seed, all initial distances are infinite and the first centre is
the smallest ID. This arbitrary first choice counts against the budget.
After it is added, distances become finite and the usual traversal begins.
An overflow raises an error instead of silently treating an unrepresentable
distance as an ordinary selection score.

For a stored $N\times d$ feature matrix, each new centre requires $N$
Euclidean distances, or $O(Nd)$ arithmetic. Initializing from $|L|$ seeds
and adding $b$ centres costs $O((|L|+b)Nd)$ with this implementation. The
running nearest-distance and availability arrays require $O(N)$ storage;
the representations themselves require $O(Nd)$. There is no need to store
the $N\times N$ distance matrix. Model inference to produce the
representations is an additional cost.

The historical
[Google implementation](https://github.com/google/active-learning/blob/efedd8f1c45421ee13af2b9ff593ad31f3835942/sampling_methods/kcenter_greedy.py)
updates nearest distances but relies on already selected points having
distance zero rather than masking them at `argmax`. If all features are
identical, all points have distance zero. Its assertion can then reject
the selected seed; with a different seed position it can return a newly
chosen position repeatedly. Explicit exclusion is needed even though
selecting more zero-distance points cannot improve coverage.

```python
import numpy as np

# This uses farthest_first from the preceding block.
same_features = np.zeros((4, 2))
stable_ids = np.array([40, 10, 30, 20])
chosen, nearest = farthest_first(same_features, stable_ids, [40], 3)
assert chosen == [10, 20, 30]
assert len(set(chosen)) == 3
assert nearest.max() == 0.

# Reordering rows must not change stable-ID results.
order = np.array([2, 0, 3, 1])
again, _ = farthest_first(same_features[order], stable_ids[order], [40], 3)
assert again == chosen
print(chosen, "radius", nearest.max())
```

<a id="representation-and-metric"></a>
## The representation determines what coverage means

Two documents can be close under a bag-of-words representation and far
apart under a contextual encoder, or vice versa. Changing the encoder,
pooling rule, feature normalization or layer changes the distances being
optimized. Representations should be fixed during one traversal; after
retraining the encoder, previously cached distances may no longer describe
the new feature space.

![[Assets/Sources/Scikit-learn MOOC/figures/clustering_quiz_kmeans_not_scaled.svg|K-means on original features]]

*scikit-learn MOOC authors,
[Quiz M4.01](https://inria.github.io/scikit-learn-mooc/clustering/clustering_quiz_m4_01.html),
CC BY 4.0. Original synthetic k-means example with three clusters and no
feature scaling. Horizontal variation dominates the assignments. The original
figure has no numerical axis labels; it is not a k-center experiment.*

![[Assets/Sources/Scikit-learn MOOC/figures/clustering_quiz_kmeans_scaled.svg|K-means on scaled features]]

*scikit-learn MOOC authors,
[Quiz M4.01](https://inria.github.io/scikit-learn-mooc/clustering/clustering_quiz_m4_01.html),
CC BY 4.0. The same original example after feature scaling.
The assignments change because scaling changes the geometry. These figures
illustrate dependence on the metric, not a comparison of query strategies.*

For Euclidean distances after standardization, feature $j$ contributes
$(z_{ij}-z_{qj})^2/s_j^2$ to squared distance, where $s_j$ is the fitted
standard deviation. Scaling does not establish that every resulting
coordinate is equally useful for the annotation task. A noisy feature can
receive a large weight if its original variation is small.

On unit-normalized nonzero vectors, squared Euclidean distance and cosine
dissimilarity satisfy

$$
 \|u-v\|_2^2=2(1-u^\top v).
$$

Consequently they give the same farthest-first ordering when ties are
resolved alike. The numerical approximation factor depends on which
distance is reported. Euclidean distance is a metric, so the radius is
at most $2r^*$. Squaring this bound gives a factor of $4$ for squared
Euclidean radius. Cosine dissimilarity $1-\cos(u,v)$ does not in general
satisfy the triangle inequality, so the metric proof cannot be applied
to it unchanged. On unit vectors its equivalence to squared Euclidean
distance gives the corresponding factor-four bound instead.

For angles $0^\circ,60^\circ,120^\circ$, cosine dissimilarities between
neighbours are $0.5$ and $0.5$, while the endpoint dissimilarity is $1.5$.
The triangle inequality would require $1.5\le1$, which is false.

```python
import numpy as np

angles = np.deg2rad([0., 60., 120.])
unit = np.column_stack([np.cos(angles), np.sin(angles)])
cosine = 1. - unit @ unit.T
euclidean_squared = ((unit[:, None] - unit[None, :]) ** 2).sum(axis=-1)
assert np.allclose(euclidean_squared, 2. * cosine)
assert cosine[0, 2] > cosine[0, 1] + cosine[1, 2]
assert np.isclose(cosine[0, 2], 1.5)
print(cosine.round(6))
```

<a id="outliers-and-labels"></a>
## Outliers and the limits of a geometric guarantee

Consider coordinates $0,1,2,3,100$, a fixed seed at $0$, and one query.
Farthest-first chooses $100$ and leaves radius $3$. If one pool point may
be left uncovered, choosing $2$ covers $0,1,2,3$ within radius $1$ and
leaves $100$ out. This changes the optimization problem: an outlier budget
permits the algorithm to stop representing some examples.

A point far from the rest might be a corrupted document, or it might be
the only example of a valid rare category. Distance alone cannot decide
which interpretation is correct. Dropping distant points is therefore not
a generally valid data-cleaning rule. Coverage should be checked against
the task's known groups and the independently held-out evaluation data.

Sener and Savarese's [§4.3](https://arxiv.org/abs/1708.00489v4)
formulates a robust k-center problem with a specified number of exceptions.
Their mixed-integer feasibility procedure searches candidate radii; it is
separate from the ordinary greedy update. The factor-two proof above does
not justify arbitrary outlier removal or an approximate nearest-neighbour
implementation that can miss the true farthest point.

```python
import numpy as np

points = np.array([0., 1., 2., 3., 100.])


def radius_after_ignoring(points, centre_positions, ignored):
    distances = np.abs(points[:, None] - points[list(centre_positions)])
    nearest = distances.min(axis=1)
    return np.sort(nearest)[len(points) - 1 - ignored]


# Exhaustive one-query solutions with fixed seed position 0.
plain = [(radius_after_ignoring(points, (0, q), 0), q) for q in range(1, 5)]
trimmed = [(radius_after_ignoring(points, (0, q), 1), q) for q in range(1, 5)]
assert min(plain) == (3., 4)
assert min(trimmed) == (1., 2)
print("ordinary", min(plain), "one exception", min(trimmed))
```

The loss analysis in Sener and Savarese's §4.1 also has explicit assumptions:
bounded and Lipschitz loss, regularity of the regression function, a
covering condition, and zero training error on the selected set. Its
core-set loss compares empirical averages on the subset and the pool;
it is only one term in the analysis of population risk. Their CNN lemma
uses the $L_2$ difference between target probabilities and softmax outputs.
The authors explicitly do not extend that lemma to cross-entropy, although
their experiments train with cross-entropy. A small embedding-space radius
alone does not establish the lemma's conditions or a bound on a trained
LLM's downstream accuracy.

<a id="coreset-versus-centres"></a>
## A set of centres and a coreset are different objects

A $k$-centre solution answers one optimization problem: which centres
achieve a small covering radius for this pool? A coreset replaces the
pool by a smaller, possibly weighted set on which a specified family of
queries can be evaluated with a controlled error.

For a nonnegative additive query loss, a common coreset requirement is
the following approximation for every $\theta\in\Theta$:

$$
\begin{aligned}
 F_P(\theta)&=\sum_{i\in P}\ell_i(\theta),\\
 F_{Q,w}(\theta)&=\sum_{i\in Q}w_i\ell_i(\theta),\\
 (1-\varepsilon)F_P(\theta)&\le F_{Q,w}(\theta)\\
                         &\le(1+\varepsilon)F_P(\theta).
\end{aligned}
$$

The query family $\Theta$ and the loss are part of the statement. A
coreset for a k-means distance sum is not automatically a coreset for
cross-entropy over all neural-network parameters. Har-Peled's
[§§31.2.1–31.2.2](https://sarielhp.org/book/chapters/coreset_cluster.pdf)
defines weighted geometric clustering costs and their query-specific
approximation. In active-learning literature, “core-set selection” also
names a subset-selection approach motivated by a loss bound; the name
alone does not establish the uniform inequality above.

For k-center in the Euclidean plane, Erickson's §J.11 gives a geometric
compression construction. First compute a greedy radius $R$, so
$r^*\le R\le2r^*$. Overlay square cells of side
$\delta=\varepsilon R/(2\sqrt2)$, and keep one point from each occupied
cell. A discarded point and its representative differ by at most the
cell diameter $\delta\sqrt2$. For any proposed centres $C$,

$$
 r_Q(C)\le r_P(C)\le r_Q(C)+\delta\sqrt2.
$$

The first inequality follows from $Q\subseteq P$. For the second, take
a point $p$, its representative $q$, and the nearest centre to $q$.
The triangle inequality bounds the distance from $p$ to that centre by
$d(p,q)+d(q,C)$. The nearest centre to $p$ can only be closer.

If $C_Q$ optimally covers $Q$ with $k$ centres that may lie anywhere in the
plane, then $r_Q(C_Q)\le r^*$. Hence

$$
\begin{aligned}
 r_P(C_Q)&\le r^*+\delta\sqrt2\\
         &=r^*+\varepsilon R/2\\
         &\le(1+\varepsilon)r^*.
\end{aligned}
$$

Solving the compressed problem remains a separate step; simply keeping
representatives is not an exact solver. Allowing arbitrary centre positions
here also differs from querying only pool examples. If $R=0$, all points
are already covered exactly and no positive-width grid is needed.

For fixed dimension, the number of occupied cells intersecting the $k$
covering balls is $O(k/\varepsilon^d)$, with dimension-dependent constants.
The dependence on $d$ makes this direct grid construction unsuitable as
a promise of compression for high-dimensional transformer embeddings.
The planar illustration and proof do not remove that dependence.

The next example uses a fixed cell width rather than estimating an
optimal radius. It checks the additive distance bound directly, including
the representatives of each occupied cell. No labels are requested.

```python
import numpy as np

points = np.array([[.01, .01], [.02, .02], [.04, .03],
                   [1.01, .01], [1.02, .02], [1.04, .03]])
width = .1
occupied = {}
for row, cell in enumerate(np.floor(points / width).astype(int)):
    occupied.setdefault(tuple(cell), row)
representatives = np.array(list(occupied.values()))
small = points[representatives]
assert representatives.tolist() == [0, 3]


def radius(data, centres):
    return np.linalg.norm(data[:, None] - centres[None, :], axis=-1).min(axis=1).max()


centres = np.array([[0., 0.]])
full_radius = radius(points, centres)
small_radius = radius(small, centres)
assert small_radius <= full_radius <= small_radius + width * np.sqrt(2.)
print("pool", len(points), "representatives", len(small))
print("radii", full_radius, small_radius, "allowed gap", width * np.sqrt(2.))
```

If the intended query is a distance sum rather than a maximum, the
population of a cell must not disappear. A representative for a cell
containing 100 examples and a representative for a cell containing one
example should not automatically receive equal weight. Assigning cell
counts supplies weights, but a sum-loss error guarantee still needs its
own proof; the maximum-distance argument above does not provide one.

<a id="herding"></a>
## Herding matches feature averages

Herding uses a different target from coverage. Suppose the population
feature mean $\mu$ is known, and a sequence of examples should reproduce
it. Welling's [§§2–3](https://icml.cc/Conferences/2009/papers/447.pdf)
defines deterministic state selection and a dynamic weight update. In
vector notation, with unit step size,

$$
\begin{aligned}
 x_t&\in\mathop{\arg\max}_{x\in\mathcal X}w_t^\top\phi(x),\\
 w_{t+1}&=w_t+\mu-\phi(x_t).
\end{aligned}
$$

The weights accumulate the difference between the desired features and
the features already generated. Selecting a state with large current
inner product responds to that accumulated deficit. Summing the weight
updates makes the mean-matching statement explicit:

$$
 \mu-\frac1T\sum_{t=0}^{T-1}\phi(x_t)
       =\frac{w_T-w_0}{T}.
$$

If the weights remain bounded, the feature-mean error is $O(1/T)$.
More generally, sublinear weight growth makes this difference tend to
zero. Welling's recurrence argument in §4 assumes bounded features,
exact maximization and a target mean obtained from a distribution positive
on every allowed state in the finite-state setting; replacing
the maximization by an arbitrary local search does not automatically
preserve its boundedness result. The telescoping identity itself holds
for every sequence produced by the stated update.

For a binary feature $\phi(x)\in\{0,1\}$ with target $\mu=1/4$, start
at $w_0=1/4$. Positive $w$ favours $1$; negative $w$ favours $0$.
Resolve a tie at zero in favour of $0$.

| Step | Weight before selection | Selected feature | Weight after update |
| --- | ---: | ---: | ---: |
| 0 | $1/4$ | 1 | $-1/2$ |
| 1 | $-1/2$ | 0 | $-1/4$ |
| 2 | $-1/4$ | 0 | 0 |
| 3 | 0 | 0 | $1/4$ |

The four selected values average to $1/4$, and the weights return to
their starting value. The sequence may select the same state repeatedly.
This is not the same operation as returning four distinct examples from
a two-example annotation pool.

```python
import numpy as np

features = np.array([0., 1.])
target = .25
initial_weight = .25
weight = initial_weight
sequence, weights = [], [weight]
for _ in range(8):
    chosen = int(np.argmax(weight * features))
    value = features[chosen]
    sequence.append(value)
    weight += target - value
    weights.append(weight)
    mean_error = target - np.mean(sequence)
    assert np.isclose(mean_error, (weight - initial_weight) / len(sequence))
assert sequence == [1., 0., 0., 0., 1., 0., 0., 0.]
assert np.mean(sequence) == target
print("sequence", sequence, "mean", np.mean(sequence))
print("weights", weights)
```

There is no Gaussian-distribution assumption in this update. Matching
only the raw feature mean can, however, discard information about
dispersion. The population $\{-1,0,1\}$ has mean zero and second moment
$2/3$. Replacing it by the single point $0$ matches the mean exactly but
has second moment zero and leaves covering radius $1$. To constrain a
second moment, the feature map must include the corresponding squared
coordinate; matching an unspecified “representative distribution” does
not follow from matching one average.

Restricting each new choice to previously unselected examples is a useful
subset-selection variant, but it changes the state space during the
process. It can exhaust the candidates, and the original infinite-sequence
convergence claim does not transfer unchanged.

<a id="mean-error"></a>

Greedily minimizing the next feature-mean error also gives a different
criterion. With the initialization $w_0=\mu$ used above and
$s_t=\sum_{j=0}^{t-1}\phi(x_j)$, the herding weight is
$w_t=(t+1)\mu-s_t$. Expanding the candidate's next squared error gives

$$
\begin{aligned}
 E_{t+1}(x)&=\left\|\mu-\frac{s_t+\phi(x)}{t+1}\right\|^2,\\
 (t+1)^2E_{t+1}(x)&=\|w_t\|^2-2w_t^\top\phi(x)\\
                  &\quad+\|\phi(x)\|^2.
\end{aligned}
$$

The last norm depends on the candidate. When it is constant across
candidates, minimizing this error reduces to maximizing the herding
inner product under the stated initialization. With unequal norms the
rules need not agree. In the binary
example, the first herding choice is $1$, with immediate mean error $3/4$;
choosing $0$ would have immediate error $1/4$. Herding compensates through
subsequent weight updates, rather than minimizing every intermediate mean
error. The prefix error rises from $0$ after four states to $3/20$ after
five, before returning to $0$ after eight.

For a dataset-selection experiment, retain the feature extractor version,
normalization, metric, seed IDs, requested budget, tie rule, selected IDs
and representation/selection costs. Evaluate the trained model on data
not selected by this criterion. Report both the task metric and the
geometric or moment error being optimized; one cannot substitute for the
other.

---

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|Active learning — запрос меток и выбор примеров]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|Обучение с частичной разметкой — pseudo-labeling и FixMatch]]

**Связанные главы:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting|Отбор по EL2N, GraNd и forgetting]];
[[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения|Кластеризация и её ограничения]].

**Источники:** Jeff Erickson,
[Approximation Algorithms, §§J.10–J.11](https://jeffe.cs.illinois.edu/teaching/algorithms/notes/J-approx.pdf),
including the original six-panel figure and the attributed construction passage,
CC BY-NC-SA 4.0. English exposition and worked examples in this chapter are
available under the same license; the two scikit-learn MOOC figures retain
their CC BY 4.0 license. Ozan Sener and Silvio Savarese,
[Active Learning for Convolutional Neural Networks: A Core-Set Approach, v4](https://arxiv.org/abs/1708.00489v4),
§§3–4.4; Max Welling,
[Herding Dynamical Weights to Learn](https://icml.cc/Conferences/2009/papers/447.pdf),
§§2–4; Sariel Har-Peled,
[Coresets for Clustering](https://sarielhp.org/book/chapters/coreset_cluster.pdf),
§§31.1, 31.2.1–31.2.2 and 31.5. No figures or extended excerpts from
Sener–Savarese, Welling or the current Har-Peled PDF are reproduced here.
