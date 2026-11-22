---
title: Scientific discovery agents
description: "Symbolic regression, visual concept libraries, and exact checks of constructions found by program search."
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery.md"
canonical_revision: "e2b076b6e583c8ff0578fceb78eace665878bd16f5a085812b909252a14d61ab"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: FunSearch — program search and cap-set verification
    label_en: FunSearch — program search and cap-set verification
    anchor: funsearch-program-search
  - label: AlphaEvolve — evaluation cascade and construction verification
    label_en: AlphaEvolve — evaluation cascade and construction verification
    anchor: alphaevolve-evaluation-cascade
source_unit_id:
  - meeting-11-slides-visual-concept-discovery
  - meeting-11-slides-neural-and-formal-systems
  - meeting-11-slides-scientific-process-symbolic-regression
  - meeting-11-slides-swarat-chaudhuri-lasr-concept-library-sequence
  - meeting-11-slides-lasr-examples-and-scaling
  - meeting-11-slides-abstraction-discovery-synthesis
  - meeting-11-reading-02-catalogue-record
primary_sources:
  - https://arxiv.org/abs/2504.00185v1
  - https://trishullab.github.io/escher-web/
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
  - https://arxiv.org/abs/2409.09359v3
  - https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/tree/de6b22d5282e7de3432b0d52758a6ec3f583527d
  - https://github.com/trishullab/escher/tree/f5c96df60ebda4dff1e933a54c9f6f8f03677cf3
  - https://trishullab.github.io/lasr-web/
  - https://doi.org/10.1038/s41586-023-06924-6
  - https://arxiv.org/abs/2506.13131v1
---

# Scientific discovery agents

<span id="71-агенты-научного-поиска-и-discovery"></span>

Two measurements give $y=4$ at distance $r=1$ and $y=1$ at $r=2$.
We could fit $a/r$, $b/r^2$, or a polynomial. If the form is unknown,
we need to choose the operations and how to combine them, as well as
fit the constants. This is *symbolic regression*: finding expressions
from data.

Both families have a coefficient we can fit to the two points. They
then make different predictions at $r=3$. We use the prediction error
and the complexity of the expression to decide which candidate to keep.

<span id="нейронный-proposal-символическая-проверка-and-search"></span>
<span id="нейронная-генерация-кандидатов-символическая-проверка-и-поиск"></span>
<span id="как-отбирают-кандидатов"></span>

## Selecting candidates

Start with $a/r$. The generator supplies the inverse-proportional form.
The evaluator fits $a$ on the training points and measures the error.
The search procedure decides whether to keep that expression in the
population. An LLM could propose replacing $r$ by $r^2$; the evaluator
would fit and score the new expression in the same way.

The evaluator depends on the task. LaSR computes expressions on
observed data. ESCHER scores how well descriptions match images.
FunSearch runs a program and checks the mathematical object it constructs.

In the [[en/00 Textbook/17 Tools and Agents/70 Formal proofs and mathematical agents|previous chapter]],
a tactic contributed to a proof that could be checked under stated
axioms. In regression, many expressions may agree on a finite set of
measurements. Even exact agreement proves neither a causal relationship
nor that the prediction remains correct beyond the measured range.

<span id="научный-поиск-как-пространство-проверяемых-гипотез"></span>
<span id="как-сравнить-две-формулы"></span>

## Comparing two expressions

Suppose we have the measurements

$$
\mathcal D = \{(x_i, y_i)\}_{i=1}^{n}
$$

and a grammar that specifies variables, constants, allowed operators
and how to combine them. The tree for $b/r^2$ has division at its root,
the constant $b$ on the left and the square of $r$ on the right.
Fitting $b$ for that tree is numerical optimization. Choosing the tree
is a discrete search.

Let $\mathcal H$ be the expressions the grammar permits and $\ell$
the error on one prediction. One objective combines average error
with an expression cost $C(f)$:

$$
\begin{aligned}
J(f)&=\frac{1}{n}\sum_i\ell(f(x_i),y_i)+\lambda C(f),\\
f^*&=\arg\min_{f\in\mathcal H}J(f).
\end{aligned}
$$

The cost might count nodes or free parameters. When it increases
by one, the average error must fall by more than $\lambda$ for
the objective to improve. The grammar matters: $r^8$ uses one power
operator, whereas a product of eight copies of $r$ uses seven
multiplication operators if leaves are not counted.

In the [LaSR formulation, sections 2–3](https://arxiv.org/html/2409.09359v3#S2),
the data likelihood favors expressions that fit and the prior favors
simpler expressions. With independent Gaussian errors of equal
variance, the negative log likelihood is a scaled sum of squared
residuals plus a constant. The negative log prior supplies the
complexity penalty.

Search can also keep a Pareto frontier. Drop an expression if another
is no worse on either error or complexity and strictly better on
at least one. A short approximation and a longer, more accurate
expression can both survive until the final choice.

<span id="один-проход-на-условном-примере"></span>
<span id="подбор-коэффициентов-и-проверка-на-новых-точках"></span>

### Fitting coefficients and checking new points

Use synthetic, noiseless data with dimensionless distance $r>0$.
Fit on the first two points and compare predictions on the other two:

| Split | $r$ | $y$ |
|---|---:|---:|
| Training | 1 | 4 |
| Training | 2 | 1 |
| Validation | 3 | $4/9$ |
| Validation | 4 | $1/4$ |

The candidates are $f_A(r)=a/r$ and $f_B(r)=b/r^2$.
For either fixed exponent $q$, let $z_i=r_i^{-q}$. Least squares
minimizes $\sum_i(cz_i-y_i)^2$ with respect to the single coefficient
$c$. Set the derivative to zero:

$$
\sum_i z_i(cz_i-y_i)=0,
\qquad c=\frac{\sum_i z_i y_i}{\sum_i z_i^2}.
$$

For $q=1$, $z_1=1$ and $z_2=1/2$, so
$a=(4+1/2)/(1+1/4)=3.6$. For $q=2$, $z_2=1/4$, and
$b=(4+1/4)/(1+1/16)=4$. A predicts 3.6 and 1.8. Its training
mean squared error is $((3.6-4)^2+(1.8-1)^2)/2=0.4$.
B fits both points exactly.

Count division and exponentiation as operators, and ignore leaves:
$C(A)=1$ and $C(B)=2$. With $\lambda=0.05$, the training
criterion $\mathrm{MSE}+0.05C$ is 0.45 for A and 0.10 for B.
The reduction in error pays for the extra operator.

Keep both coefficients fixed when evaluating on the validation points.
A predicts 1.2 and 0.9, giving $\mathrm{MSE}_{val,A}\approx0.49668$.
B predicts $4/9$ and $1/4$, both correct. The same criterion selects B,
with 0.10 against approximately 0.54668. A validation set used to
choose operators or concepts is part of model selection; it cannot
also give an independent final estimate.

Fix $4/r^2$, then examine a held-out point at $r=8$ with $y=1/16$.
It predicts $4/64=1/16$. One check outside the training and validation
range cannot establish a physical law. On noisier, dimensional data
we would also need independent observations, units and uncertainty
estimates.

Both A and B lie on the Pareto frontier: A is simpler and B has lower
error. The expression $f_C(r)=3.6/r+0$ simplifies to A. Before
simplification it has the same error as A but one extra operator,
so A dominates it. Counting before and after simplification changes
the objective.

```python
from fractions import Fraction as F

train = [(F(1), F(4)), (F(2), F(1))]
validation = [(F(3), F(4, 9)), (F(4), F(1, 4))]

def fit_inverse_power(points, power):
    # For f(r) = coefficient * z, z = 1 / r**power.
    numerator = sum(y / r**power for r, y in points)
    denominator = sum(1 / r**(2 * power) for r, _ in points)
    return numerator / denominator

def mse(points, coefficient, power):
    return sum((coefficient / r**power - y)**2 for r, y in points) / len(points)

a = fit_inverse_power(train, 1)
b = fit_inverse_power(train, 2)
assert (a, b) == (F(18, 5), F(4))
assert mse(train, a, 1) == F(2, 5)
assert mse(validation, a, 1) == F(6437, 12960)
assert mse(validation, b, 2) == 0
assert b / F(8)**2 == F(1, 16)
```

The code fits two supplied exponents. To discover an exponent or a
different form, the search has to change the expression tree and
fit its constants again.

<span id="деревья-выражений-и-островные-популяции"></span>
<span id="как-pysr-перебирает-деревья"></span>

### How PySR searches expression trees

PySR keeps several populations of trees. A mutation might change a
constant or operator, add a node, or delete a subtree. Crossover
exchanges subtrees. For example, copying $r^2$ from $1+r^2$ into
the denominator of $b/r$ gives $b/r^2$. The search can simplify the
new tree and fit its constants after the change.

Tournament selection compares a sample of the population to choose
a parent. In age-based replacement an accepted candidate can displace
the oldest even if its score is worse. A separate archive retains
the best expressions, and migration moves promising trees between
populations. See the [LaSR authors' description of PySR](https://trishullab.github.io/lasr-web/).

Changing one node can turn bounded, periodic $\sin r$ into the
growing $\exp r$. Conversely, $r^2$ and $(r+1)^2-2r-1$ denote the
same function but have different trees. LaSR keeps the local PySR
operators and adds proposals conditioned on descriptions of the
desired relationship.

To reproduce a search, pin the training, validation and test
points and their ranges; allowed operations and units; constant
fitting, error and complexity; number of evaluations, wall time
and random seed. Specify when the search simplifies expressions
and how it tests extrapolation or noise. The words “charge” and
“distance” give a model information missing from anonymous feature
names $x_1,x_2$. Equal iteration counts do not control for different
inputs and hints.

<span id="lasr-совместная-эволюция-формул-и-библиотеки-понятий"></span>
<span id="как-lasr-использует-описания-зависимостей"></span>

## How LaSR uses descriptions of relationships

[LaSR](https://arxiv.org/abs/2409.09359v3) stores phrases such as
“power law,” “symmetric in two variables” and “exponential decay.”
They guide the prompt; they do not add operators to the grammar.
The LLM must still produce expressions built from the permitted
operations.

The LLM's weights stay fixed. First LaSR searches with the current
library. Then it revises the library using accurate and inaccurate
expressions, and feeds the revised descriptions into the next round.
An “inverse power” hint could suggest changing $a/r$ to $b/r^2$.
A controlled run at comparable cost, without the library, would
help determine how much the hint contributed.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/lasr-overview.svg|LaSR searches for expressions, extracts descriptions from good and bad candidates, and evolves the library.]]

*Arya Grayeli et al., [LaSR, Figure 1](https://arxiv.org/html/2409.09359v3#S1.F1),
NeurIPS 2024, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original SVG. Search is at the top, concept extraction on the right,
and generation of new concepts from the library on the left.*

<span id="эволюция-гипотез"></span>
<span id="как-lasr-предлагает-новые-формулы"></span>

### How LaSR proposes expressions

Each island has its own population. With probability $p$, an LLM
operation replaces an ordinary symbolic operation. This is a choice
of search operator, not the probability that a candidate is correct.
The operations differ in the input they give the model:

- `LlmInit` uses variable names, the grammar and initial concepts.
- `LlmMutate` also takes one expression to change.
- `LlmCrossover` takes two expressions and asks for a candidate that
  reuses their subexpressions.

The implementation parses the reply, checks the grammar and computes
the expression. At $r=2$, $1/(r-2)$ divides by zero and cannot be
scored as a finite prediction. A candidate may have a useful structure
and poor starting coefficients; numerical optimization still fits
its constants after generation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/lasr-evolution.svg|A LaSR iteration links island populations, the Pareto frontier, and operations on the concept library.]]

*Grayeli et al., [LaSR, Figure 2](https://arxiv.org/html/2409.09359v3#S3.F2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original SVG.
Initialization and expression search are at left, the frontier at center
and the library at right. The bottom arrow feeds sampled concepts back
into search.*

<span id="абстракция-понятий"></span>
<span id="как-lasr-извлекает-понятия-из-формул"></span>

### Extracting concepts from expressions

After a round of search, LaSR collects candidates on the Pareto
frontier and candidates with the largest loss. The model tries
to describe common features of the former without repeating patterns
in the latter. If both sets frequently contain division, division
by itself is not an informative description.

If $a/r$ and $b/r^2$ fit better than growing polynomials, the model
might propose “decreasing inverse power.” A narrow measurement
range can make unrelated families look similar, so the phrase can
also be misleading. Good fit belongs to the evaluated expressions,
not automatically to the whole family named in the hint.

<span id="эволюция-понятий"></span>
<span id="как-lasr-комбинирует-понятия"></span>

### Combining concepts

The model can generate concepts from those already in the library.
Figure 1 combines “exponential growth/decay” with a dependence on
temperature and proposes `Boltzmann Distribution`. That phrase can
guide a new expression, which still has to fit the data.

LaSR does not score a new concept separately before putting it in
the library. The authors describe the decision:

> The implementation of this function follows that of the LlmCrossover operation in that we are using multiple concepts as a reference to generate new ones, with the key distinction that, unlike in the LlmCrossover operation, the fitness of each generated concept here is difficult to quantify. Thus, we include all the generated responses in the concept library. While these concepts may sometimes be inaccurate, they increase the evolutionary algorithm’s exploration ability.

*Grayeli et al., [LaSR, section 3, Concept Evolution](https://arxiv.org/html/2409.09359v3#S3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Expressions generated from a poor concept can be rejected later,
while the concept remains available. LaSR samples recent concepts
to guide expression search and also samples older ideas to evolve
the library. [Appendix A.3.2](https://arxiv.org/html/2409.09359v3#A1.SS3.SSS2)
uses a window of 20 recent concepts. Changing the window or how
often the LLM is called changes both the proposals and the cost.


<span id="что-показали-feynman-equations-and-scaling-law-case"></span>
<span id="что-показали-эксперименты-на-формулах-фейнмана-и-scaling-laws"></span>
<span id="результаты-lasr-на-формулах-и-данных-big-bench"></span>

## LaSR on known equations and BIG-Bench

<span id="восстановление-известных-формул"></span>

### Recovering known equations

[Section 4.1 of v3](https://arxiv.org/html/2409.09359v3#S4.SS1)
uses 100 Feynman equations with a small amount of noise. Table 1
reports 72 exact recoveries for LaSR with GPT-3.5-turbo-0125,
$p=0.01$ and 40 iterations, versus 59 for PySR. The other
baselines largely come from earlier published results, not a
fresh run of every method on the same hardware.

Table 2 runs a cascade. It starts with Llama 3 8B at $p=1\%$,
then passes only the unsolved problems to the next setting.
The cumulative totals are 67, 69 and 71 at $p=1\%,5\%,10\%$,
then 72 after GPT-3.5 at $p=1\%$. The later stages have access
to a different set of problems and add to the total budget.
They do not give four independent solve rates.

Exact recovery compares the expression with the symbolic reference.
Section 4.3 instead counts runs whose loss goes below an MSE threshold
of $10^{-11}$. On noisy data the right equation may have nonzero
loss, while an overfit expression can have low loss without
matching the reference.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/lasr-ablations.svg|LaSR ablations on the left and a comparison of user hints on the right.]]

*Grayeli et al., [LaSR, Figure 3](https://arxiv.org/html/2409.09359v3#S4.F3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original SVG.
The plots use the MSE threshold, not the exact-match criterion of
Tables 1–2.*

<span id="закон-кулона-и-вид-найденного-выражения"></span>
<span id="приближение-к-закону-кулона"></span>

### Approximating Coulomb's law

Two lecture slides compare individual PySR and LaSR candidates for
$F\propto q_1q_2/r^2$. With charges fixed, doubling $r$ should cut
the force to one quarter. Swapping the charges should preserve it.
These properties offer checks beyond the error on the observed data.

The slides count ten simplification steps for PySR and four for LaSR.
Those counts describe the algebra shown, not the number of search
iterations or the runtime. One LaSR step drops a small term from the
denominator. It is an approximation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/discovery/m11-p75-pysr-coulomb.png|A Berkeley slide shows the original PySR expression for Coulomb's law and its simplification.]]

*Swarat Chaudhuri,
[Abstraction and Discovery with Large Language Model Agents, page 75](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=75),
Berkeley, Spring 2025. Original slide. The slide credits the PySR
expression with ten simplification steps.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/discovery/m11-p76-lasr-coulomb.png|The LaSR slide drops a small term from the denominator to approximate Coulomb's law.]]

*Swarat Chaudhuri,
[Abstraction and Discovery with Large Language Model Agents, page 76](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=76),
Berkeley, Spring 2025. Original slide. The step labeled `Negligible`
is approximate. See [LaSR, Appendix A.8](https://arxiv.org/html/2409.09359v3#A1.SS8)
for the primary account.*

Rearranging the denominator on the second slide gives

$$
\begin{aligned}
F_{\mathrm{cand}}&=\frac{kq_1q_2}{\epsilon r(r+\delta/q_2)},\\
k&=0.07957782,\\
\delta&=1.9181636\cdot10^{-5}.
\end{aligned}
$$

The coefficient $k$ is close to $1/(4\pi)$, but differs from it.
Substituting $k\approx1/(4\pi)$ gives the ratio to Coulomb's law:

$$
\frac{F_{\mathrm{cand}}}{F_{\mathrm{Coulomb}}}
\approx\frac{1}{1+\delta/(q_2r)}.
$$

At positive $q_2r=1$ in the units of the example, the correction has
order $2\cdot10^{-5}$, so it may be hard to see in a numerical test.
At $q_2r=10^{-5}$ the ratio is about 0.343, a relative error of about
66%. Dropping the term requires $|\delta/(q_2r)|\ll1$. Neither slide
by itself establishes the paper's `Exact Solve` criterion.

Known equations may appear in LLM training data. In section 4.5 the
authors test 41 synthetic formulas with arbitrary variables and unusual
combinations of operators. At $p=0.1\%$, LaSR with Llama 3 8B has
test $R^2=0.913$ against PySR's $0.070$. This measures predictive fit,
not exact recovery. It makes direct recall of a known equation a less
likely explanation without ruling out all effects of pretraining.

<span id="поиск-зависимости-для-big-bench"></span>

### Fitting a relationship on BIG-Bench

In [section 4.6 and Appendix A.6](https://arxiv.org/html/2409.09359v3#S4.SS6),
LaSR looks for a relationship between language-model performance,
training settings and the number of examples in the prompt. BIG-Bench
has 204 tasks and 55 models. The study selects only records with the
`multiple choice grade` metric: 53,812 observations. The authors fit
parameters on 43,049 and validate on 10,763.

Search used 3,840 populations of 200 candidates and ran for seven hours.
The setup also encourages the score to increase monotonically with the
number of training steps. Searching over forms still needs choices by
the researcher: variables, grammar, complexity and constraints.

The compact expression has the form

$$
s(t,k)=E+A\left(\frac{B}{t}\right)^k,
$$

where $t>0$ is the number of training steps and $k$ the number of
in-context examples. Equation 5 of v3 gives $A=-0.0248235$,
$B=116050.999$ and $E=0.360124$. Since $A$ is negative, when $t>B$, the
base $B/t$ is below one and increasing $k$ moves the score toward $E$
from below. When $t<B$, the base exceeds one and increasing $k$ makes
the negative term larger in magnitude. At $t=B$, the score is
independent of $k$.

| Training steps | $k=1$ | $k=2$ | $k=4$ |
|---|---:|---:|---:|
| $t=B/2$ | 0.310477 | 0.260830 | −0.037052 |
| $t=2B$ | 0.347712 | 0.353918 | 0.358573 |

The first row includes a negative predicted grade, though a proportion
correct lies between zero and one. Nothing in the expression enforces
that bound under extrapolation. The numbers above come from substitution
in the published equation, not a new language-model experiment.

```python
def lasr_score_v3(steps, shots):
    if steps <= 0 or shots < 0 or int(shots) != shots:
        raise ValueError("Positive steps and a nonnegative integer shot count required")
    return 0.360124 - 0.0248235 * (116050.999 / steps)**shots

assert round(lasr_score_v3(116050.999 / 2, 4), 6) == -0.037052
assert round(lasr_score_v3(116050.999 * 2, 4), 6) == 0.358573
assert lasr_score_v3(116050.999, 1) == lasr_score_v3(116050.999, 4)
```

Appendix A.6 of v3 still mentions a change of behavior between odd
and even shot counts. The printed $B$ is positive, so $(B/t)^k$ cannot
alternate in sign. In [v1](https://arxiv.org/abs/2409.09359v1), $B$ was
negative and parity could change the sign. The analysis here follows
the v3 coefficients.

Table 4 of v3 gives a validation MSE of $0.03598\pm0.00265$ for the
three-parameter expression and $0.03610\pm0.00268$ for the authors'
five-parameter `Chinchilla` form. The difference is small relative
to the reported spread. Both curves predict a BIG-Bench grade, not
Chinchilla's original pretraining loss.

To assess transfer, keep the table of models, tasks and measurements,
and report how the task grades are combined. Hold out whole models or
families, compare with simple prespecified functions, and inspect
whether the form survives removing one family or one sample of tasks.
A held-out set of records does not establish that an entire family
was kept out of the fit.

<span id="когда-абстракция-ускоряет-поиск-а-когда-закрепляет-ошибку"></span>
<span id="когда-удачная-подсказка-сужает-поиск"></span>

## When a useful hint narrows the search

At $r=1$, $r^{-2}$ and $\exp(-2(r-1))$ both equal 1 and have derivative
$-2$. At $r=2$ they predict $1/4$ and about $0.1353$. Measurements
clustered near 1 can make the two formulas hard to distinguish.

Suppose “inverse power” enters the library. The next round proposes
more candidates of that form, and the best among them become examples
for the next extraction. An inverse power may win because few
exponentials are left in the population. A description may also omit
conditions, such as the range in which an expression works. These are
possible failure modes, not effects separately measured by LaSR.

Keep the source of a concept: which expressions produced it, which
data scored those expressions, and at what budget it helped or hurt.
Compare searches without the library and without the LLM as separate
controls. Equal wall time, equal candidate counts and equal token
budgets answer different questions.


<span id="escher-visual-concept-library"></span>
<span id="escher-библиотека-понятий-для-визуального-распознавания"></span>
<span id="как-escher-уточняет-признаки-классов"></span>

## How ESCHER refines class descriptions

<!-- source_unit_id: meeting-11-slides-visual-concept-discovery -->

A mug may look like a bowl if the handle is hidden. “Ceramic surface”
describes both, whereas “handle on the side” can help separate them.
To use the phrase as a feature, the image encoder must respond to it
in photographs where that feature is visible.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/escher-overview.svg|ESCHER uses vision-language scores to revise descriptions for classes that look alike.]]

*Atharva Sehgal et al., [ESCHER, Figure 1](https://arxiv.org/html/2504.00185v1#S1.F1),
CVPR 2025, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original SVG. Shape and texture descriptions help distinguish similar
classes. The lower half shows feedback from the vision-language model
as the library is revised.*

<span id="как-понятия-превращаются-в-оценку-класса"></span>

### Turning descriptions into class scores

[ESCHER, section 3](https://arxiv.org/html/2504.00185v1#S3)
uses a *concept bottleneck model* (CBM). CLIP encodes the feature
descriptions and the image. Similarities form the vector
$s(x,\mathcal C)\in\mathbb R^{|\mathcal C|}$, one component per
description. CLIP computes similarity with a dot product of normalized
vectors. A similarity value alone does not tell us the probability
that the feature is present.

A linear score combines the similarities for each class:

$$
\begin{aligned}
f(x,y)&=w_y^\top s(x,\mathcal C),\\
\hat y&=\arg\max_y f(x,y).
\end{aligned}
$$

In the version without training labels, give equal weight to each
description of a class. Suppose the mug has “ceramic surface” and
“handle,” while the bowl has “ceramic surface” and “open top.” The
encoder returns 0.7, 0.4 and 0.6 for the three phrases on one image.
The mug scores $(0.7+0.4)/2=0.55$, and the bowl scores
$(0.7+0.6)/2=0.65$. The classifier chooses bowl even if the image
contains a mug.

Suppose “handle attached to the side” scores 0.95. Adding it to the
mug descriptions raises their mean to $2.05/3\approx0.6833$, changing
the decision. These are invented scores, not measurements from CLIP
or from the paper. The same phrase could mislead on another image;
evaluate the library across images.

With labels, fit $w_y$ with cross-entropy and regularization. The
linear adapter can reduce the contribution of a description that
scores as highly on bowls as on mugs. In the few-shot setting it
learns from a small number of labeled images per class. The text and
image encoders can stay fixed while the adapter trains.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/course-follow-through/escher-figure2-original.svg|A fixed concept bottleneck is shown at left; ESCHER updates the library using scores and history at right.]]

*Sehgal et al., [Self-Evolving Visual Concept Library using Vision-Language Critics, Figure 2](https://arxiv.org/html/2504.00185v1#S1.F2),
CVPR 2025, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Full original SVG from the HTML edition (page 3 of the PDF).*

<span id="один-цикл-уточнения-библиотеки"></span>

### One round of library revision

[Algorithm 1 and section 4](https://arxiv.org/html/2504.00185v1#S4)
alternate between a fixed library and a fixed current classifier:

1. The LLM proposes initial descriptions for the class names.
2. The CBM scores the images. If labels are available for adapter
   training, fit the adapter.
3. Use the score matrix to pick pairs that are difficult to separate.
4. Give the LLM both classes' current descriptions and the history
   of changes and resulting scores. Ask for new features that
   distinguish the pair.
5. Add the new descriptions. At the next round, score the images
   again and record the result of the change.

A side handle can help distinguish mugs from bowls while making mugs
more similar to pitchers. After changing two classes, score all the
classes again.

<span id="путаница-без-меток--сигнал-для-поиска-не-ошибка-по-эталону"></span>
<span id="как-выбрать-пары-классов-без-меток"></span>

### Selecting class pairs without labels

Without labels we have an $N\times J$ score matrix with one row for
each image and one column for each class. *Top-k pseudo-confusion*
counts pairs that appear among the top $k$ scores on the same image:

| Image | Classes in descending score order | Top-2 pair |
|---|---|---|
| 1 | A, B, C, D | A–B |
| 2 | B, A, C, D | A–B |
| 3 | A, B, C, D | A–B |
| 4 | C, B, D, A | B–C |

A and B appear together three times, B and C once. That makes A–B a
candidate for refinement. With no ground-truth labels, the count of
three is not three errors. A shared texture or a mistake by the
encoder can give both classes a high score.

The pinned implementation makes a different choice.
[`get_confusion_matrix`, lines 48–89](https://github.com/trishullab/escher/blob/f5c96df60ebda4dff1e933a54c9f6f8f03677cf3/escher/cbd_utils/metrics.py#L48)
drops the highest-scoring class with `[:, 1:]`, then takes the next
`topk`. For `topk=2` the same rows give B–C, A–C, B–C and B–D, a
different order of priorities. The first table follows section 4 of
the paper. Reproducing results requires the implemented heuristic,
not its name alone.

The code below counts unordered pairs of distinct classes in those
two variants. It does not implement the full ESCHER matrix, its
diagonal handling, or softmax:

```python
from collections import Counter
from itertools import combinations

rankings = ["ABCD", "BACD", "ABCD", "CBDA"]

def count_pairs(rows, k, skip_best=False):
    counts = Counter()
    start = int(skip_best)
    for row in rows:
        selected = row[start:start + k]
        for left, right in combinations(selected, 2):
            counts[tuple(sorted((left, right)))] += 1
    return counts

paper_pairs = count_pairs(rankings, 2)
code_pairs = count_pairs(rankings, 2, skip_best=True)
assert paper_pairs == {("A", "B"): 3, ("B", "C"): 1}
assert code_pairs == {("B", "C"): 2, ("A", "C"): 1, ("B", "D"): 1}
```

Another option is Pearson correlation between score columns. It
measures whether scores rise together on the same images, not the
classification error rate. Agglomerative clustering instead
groups columns by their similarity. An ordinary confusion matrix
requires labeled images: it counts pairs of the true class and
the predicted class.

Here $k$ is the number of class scores kept per image. The $K$ in
Algorithm 1 is the number of pairs to revise in one round. Requesting
descriptions for all $J^2$ pairs becomes expensive at hundreds of
classes, so the heuristic supplies sampling weights.

<span id="история-уточнений-и-повторяющиеся-пары"></span>
<span id="как-история-влияет-на-выбор-пар"></span>

### How history affects pair selection

The effect of a new description is unknown when the LLM proposes it.
Record the change at iteration $t$ together with the score at $t+1$,
after the CBM evaluates it. Keeping only the LLM replies would omit
which changes helped.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/escher-history-prompt.svg|The ESCHER prompt includes the two classes' earlier descriptions and a measure of their confusion.]]

*Sehgal et al., [ESCHER, Figure 6](https://arxiv.org/html/2504.00185v1#S7.F6),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original English
prompt.*

The template calls its input a `misprediction rate`. Depending on the
heuristic, the value can be a normalized correlation rather than an
error rate computed from labels.

When the handle is hidden, another description may still fail to
separate the mug and the bowl. ESCHER samples pairs by confusion and
uses $\gamma$ to penalize repeated selections. Reducing a pair's
weight assigns fewer requests to it; it does not indicate fewer errors
on that pair.

The procedure returns the library and, in trained settings, its
adapter. Measure accuracy on separate labeled images. For the
tableware example, those could include unseen materials and views
with the handle obscured. This is a proposed transfer test for
the teaching example, not an experiment reported by ESCHER.

In the NABirds example below, the initial descriptions give a mean
score of 0.2869 for a female pheasant and 0.2860 for a male, though the
image shows a male. After revision, the male descriptions average
0.2877 and the classifier selects the male. These are differences
between compatibility scores, not changes in the probability of a
correct prediction from 28.60% to 28.77%.

One new description is `teetering walking motion`. A still photograph
does not directly record motion, but the words can affect similarity
in CLIP. The right class does not verify each description as a fact
about the image.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/escher-bird-concepts.svg|Revised text about the pheasant's head and neck changes the scores of two similar classes.]]

*Sehgal et al., [ESCHER, Figure 3](https://arxiv.org/html/2504.00185v1#S5.F3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original SVG.
The image illustrates one NABirds result; the authors measure
test accuracy separately.*

<span id="что-сравнивалось-в-опытах-escher"></span>
<span id="точность-escher-на-отложенных-изображениях"></span>

### ESCHER accuracy on held-out images

In [section 5](https://arxiv.org/html/2504.00185v1#S5), the authors extend
three existing CBMs. The main experiments use GPT-3.5-turbo-0125 for
descriptions and CLIP ViT-L/14 for visual scores. Validation images
guide the library updates; test images measure top-1 accuracy.
“No additional annotations” refers to updating the library. LM4CV
still trains its adapter on labeled data.

The table shows test accuracy in percent and differences in percentage
points:

| Setting | Original CBM | CBM + ESCHER | Change (pp) |
|---|---:|---:|---:|
| LM4CV, CUB | 63.26 | 83.17 | +19.91 |
| LM4CV, Food101 | 94.77 | 94.90 | +0.13 |
| LaBO, 8-shot, CIFAR100 | 74.23 | 73.62 | −0.61 |
| CbD, zero-shot, CUB | 62.00 | 63.33 | +1.33 |

These are selected rows from Tables 1–3. The large gain on one LM4CV
row does not carry over to every few-shot setting. Direct CLIP scores
64.83 on CUB in Table 3, above both CbD variants in that setting.

Table 4 examines whether adding independent descriptions suffices.
On CUB, LM4CV with more independently generated concepts scores
66.09, against 83.17 with ESCHER. That supports a contribution from
feedback beyond description count. It does not control for all
compute: history increases prompt length, and each round requires
rescoring images and, where applicable, training the adapter.
See [section 7.7](https://arxiv.org/html/2504.00185v1#S7.SS7) for the
reported cost.


<a id="funsearch-program-search"></a>
<span id="funsearch-поиск-функции-которая-строит-решение"></span>
<span id="как-funsearch-меняет-порядок-перебора"></span>

## How FunSearch changes the order of search

Suppose we want to choose as many points as possible without taking
three on one line. Keep a greedy constructor fixed and change the
order in which it considers points. In FunSearch the model writes
the `priority` function that defines that order. The constructor
builds the set, and the evaluator checks the constraint before
reporting its size.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/program-search-2026-11-09/funsearch-figure-1.png|FunSearch puts the task specification and earlier programs in the prompt and returns scored candidates to the database.]]

*Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov et al.,
[Mathematical discoveries from program search with large language models, Figure 1](https://www.nature.com/articles/s41586-023-06924-6/figures/1).
Publisher's original, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The dashed arrows lead from the program database to the new prompt.
The LLM sees earlier scored functions and writes another. The evaluator
runs it inside the fixed program. Runtime errors, time limits and invalid
outputs rule out candidates. Programs with valid results go back to the
database with their scores. The database keeps separate populations so
that the best function in one population does not determine every prompt.
The loop updates the stored programs, not the weights of the LLM.
See [FunSearch and Methods in the paper](https://doi.org/10.1038/s41586-023-06924-6).

<span id="один-проход-построение-и-независимая-проверка"></span>
<span id="построение-множества-без-троек-на-прямой"></span>

### Constructing a cap set

Take points in $\{0,1,2\}^3$ and add coordinates modulo 3. A *cap set*
contains no three **distinct** points $a,b,c$ with $a+b+c=0$.
For any two distinct points the third point on their line is $-a-b$.
Once the constructor chooses $(0,0,0)$ and $(0,0,1)$, it must skip
$(0,0,2)$.

A constant priority gives lexicographic order. The greedy constructor
returns eight points. Another ordering puts the nine points satisfying
$z=x^2+y^2\pmod 3$ first, one for each $(x,y)$. The constructor and
tie-breaking rule stay the same. An independent checker goes through
every triple in the output.

```python
from itertools import combinations, product

def third_point(a, b):
    return tuple((-x - y) % 3 for x, y in zip(a, b))

def construct(n, priority):
    universe = product(range(3), repeat=n)
    order = sorted(universe, key=lambda p: (-priority(p), p))
    chosen, blocked = [], set()
    for point in order:
        if point in blocked:
            continue
        blocked.update(third_point(point, prev) for prev in chosen)
        chosen.append(point)
    return chosen

def evaluate(points, n):
    domain = set(product(range(3), repeat=n))
    if len(set(points)) != len(points) or not set(points) <= domain:
        return None
    for a, b, c in combinations(points, 3):
        if all((x + y + z) % 3 == 0 for x, y, z in zip(a, b, c)):
            return None
    return len(points)

def paraboloid_priority(p):
    x, y, z = p
    return int(z == (x*x + y*y) % 3)

baseline = construct(3, lambda p: 0)
candidate = construct(3, paraboloid_priority)
invalid = candidate + [(0, 0, 1)]
assert baseline[:3] == [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
assert third_point(baseline[0], baseline[1]) == (0, 0, 2)
assert evaluate(baseline, 3) == 8
assert evaluate(candidate, 3) == 9
assert len(invalid) == 10 and evaluate(invalid, 3) is None
```

Add $(0,0,1)$ to the nine points and it forms a line with $(0,1,1)$
and $(0,2,1)$: their sum is $(0,3,3)$, zero modulo 3. The list has
length ten, but the evaluator rejects it. The nine checked points
establish that size nine is achievable in dimension 3. This run
does not rule out a larger set.

The quadratic priority in the example was chosen by hand. FunSearch's
published run found a function that constructs a set of 512 points
in dimension 8 ([Cap sets and Figure 4](https://doi.org/10.1038/s41586-023-06924-6)).
The [published list](https://github.com/google-deepmind/funsearch/blob/cc53f274237d7ab05c19df939edbc1f9616a7c19/cap_set/n8_size512.txt)
passes all 130,816 pair checks: no pair has its third point in the
list. Checking pairs suffices because each pair fixes that third point.
This establishes a lower bound, not an upper bound. In
[Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/#Sec19),
the authors report that four of 140 direct search runs in dimension 8
found a set of size 512. The witness check needs no LLM, whereas the
cost and success rate refer to the search that produced it.

<a id="alphaevolve-evaluation-cascade"></a>
<span id="alphaevolve-правки-в-нескольких-частях-программы"></span>
<span id="как-alphaevolve-меняет-программу-поиска"></span>

## How AlphaEvolve changes the search program

A single priority function works when the rest of the constructor
can stay fixed. In some tasks the initializer, objective and optimizer
need to change together. In AlphaEvolve the task author marks editable
blocks. The model sees the program and examples from the database
and proposes replacements. The controller applies them before running
the evaluator. A proposed edit that cannot be applied or executed
does not earn the score of a working program.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/program-search-2026-11-09/alphaevolve-figure-2.png|The AlphaEvolve controller selects a parent, applies the edit, and stores the evaluator's results with the new program.]]

*Alexander Novikov, Ngân Vũ, Marvin Eisenberger et al.,
[AlphaEvolve: A coding agent for scientific and algorithmic discovery, Figure 2](https://arxiv.org/html/2506.13131v1#S2.F2).
Unmodified original, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).*

An evaluation can return more than one score. For tensor decomposition,
the authors record the fewest terms found and the fraction of random
initializations that reached that count. In an evaluation cascade, run
a cheap check first. Only promising candidates move on to harder or
more expensive checks. Passing the first stage keeps a program in
consideration; it does not establish the result of the next stage.
See [sections 2.1–2.6 and 3.1](https://arxiv.org/html/2506.13131v1#S2).

<span id="почему-4096-проверок-достаточно-для-одной-конструкции"></span>
<span id="точная-проверка-разложения-для-матриц-4--4"></span>

### Checking the 4 × 4 decomposition exactly

For matrix multiplication, AlphaEvolve modified a search program.
The program found coefficients for a different algorithm: it
multiplies two $4\times4$ matrices with 48 scalar multiplications
over the complex numbers. Each term evaluates a linear form in $A$
and one in $B$, multiplies the two values, and contributes the result
to the output entries with fixed coefficients.

The rule is bilinear, so it suffices to test pairs of basis matrices.
For example, $E_{12}E_{21}=E_{11}$ and $E_{12}E_{12}=0$.
Each input has 16 basis matrices. Compare 16 output entries for
each of the $16\cdot16$ pairs, giving 4096 scalar comparisons.

To see how the checks extend beyond basis matrices, take
$A=2E_{12}+3E_{21}$ and $B=E_{21}$. Bilinearity gives
$T(A,B)=2T(E_{12},E_{21})+3T(E_{21},E_{21})$.
The first term is $2E_{11}$ and the second is zero. The same expansion
works for every pair of inputs. Testing only basis pairs would not
suffice for an arbitrary program that is not bilinear.

The real and imaginary parts of the
[published factors](https://github.com/google-deepmind/alphaevolve_results/blob/4226acbf237ff9ad10ba7673a2af127a2d8a5971/mathematical_results.ipynb)
are multiples of $1/2$. For the local check, double each of the
three factor matrices and compare the resulting integer tensor
coefficients with eight times the reference. All 4096 agree exactly;
a deliberately corrupted coefficient fails the check.
This verifies the published construction without rerunning the search.
It does not prove 48 is minimal or measure speed on a GPU. Additions,
layout and arithmetic types affect runtime separately.

<span id="от-генерации-гипотез-к-эксперименту"></span>
<span id="выбор-следующего-измерения"></span>

## Choosing the next measurement

LaSR searches the data it has. Choosing where to measure next is an
*active experiment design* problem.

The earlier fits were $f_A(r)=3.6/r$ and $f_B(r)=4/r^2$.
Their predictions differ by

$$
d(r)=\left|\frac{3.6}{r}-\frac{4}{r^2}\right|.
$$

At $r=1,2,4,8$, the differences are respectively 0.4, 0.8, 0.65 and
0.3875. With equal cost and additive noise across those points,
$r=2$ separates the two fixed predictions better than $r=8$.
The original measurement at $r=2$ was used in fitting, so this
would require a fresh independent observation at the same distance.
The point at $r=8$ tests transfer past the measured range.
Choosing by disagreement is a rule for this fixed pair, not a
measurement-planning stage in LaSR.

In a real search, account for coefficient uncertainty, other
plausible expressions and variation in noise with $r$. Refitting
$a$ and $b$ after each observation changes the models being compared.
Data selected adaptively do not replace a final independent test of
the chosen relationship.

Set the calibration procedure and the criteria for a usable
measurement before taking it. An agent that proposes a measurement
does not thereby gain permission to control the instrument.
The [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|next chapter]]
separates proposing a tool action, approving it, and executing it.

<span id="минимальный-отчёт-об-агентном-научном-поиске"></span>
<span id="что-сохранить-для-повторения-поиска"></span>

## What to save when reporting a search

To reproduce a symbolic regression run, keep the measurements and split, the noise
assumptions, the operators, the procedure for fitting constants and
the stopping rule. Keep the expressions before and after
simplification so that an approximation can be distinguished from
an exact rewrite.

When comparing searches, report the evaluated candidates, token
usage and wall time. Two methods can evaluate the same number of
expressions but spend different amounts of compute on generation.
Record human choices such as the grammar, hints, metric and final
check, along with failed runs and limits on transfer.

Keep the evaluation logs, the concept library and the code version
with the final equation. LaSR's
[`lasr-experiments` branch at the pinned commit](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/tree/de6b22d5282e7de3432b0d52758a6ec3f583527d)
uses modified versions of PySR and SymbolicRegression.jl.
Using the latest API without checking settings does not reproduce
that study.

<span id="источники-и-дальнейшее-чтение"></span>
<span id="источники"></span>

## Sources

- Swarat Chaudhuri, [Abstraction and Discovery with Large Language Model Agents](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf), Berkeley Meeting 11.
- Grayeli et al., [Symbolic Regression with a Learned Concept Library, v3](https://arxiv.org/abs/2409.09359v3), NeurIPS 2024: sections 2–3 for the objective and search, section 4 for experiments, Appendix A.6 for BIG-Bench. See [v1](https://arxiv.org/abs/2409.09359v1) for the earlier coefficients.
- [LaSR project](https://trishullab.github.io/lasr-web/) and [official repository](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl).
- Sehgal et al., [Self-Evolving Visual Concept Library using Vision-Language Critics](https://arxiv.org/abs/2504.00185v1), CVPR 2025: sections 3–4 for the concept bottleneck and the library loop, section 5 for the results, and sections 7.7–7.10 for costs and history.
- [ESCHER project](https://trishullab.github.io/escher-web/) and [pinned implementation](https://github.com/trishullab/escher/tree/f5c96df60ebda4dff1e933a54c9f6f8f03677cf3).
- [[05 Источники/Papers/FunSearch|Romera-Paredes et al., FunSearch]]: specification, program database and cap-set construction.
- [[05 Источники/Papers/AlphaEvolve|Novikov et al., AlphaEvolve]]: sections 2–3.1, the architecture and the published factors.

Previous: [[en/00 Textbook/17 Tools and Agents/70 Formal proofs and mathematical agents|Formal proofs and mathematical agents]].
Next: [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Agent system safety]].
