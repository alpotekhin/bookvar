---
title: Formal proofs and mathematical agents
description: "From Lean goals and tactics to autoformalization, learned proof search, and checking which axioms a proof uses."
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md"
canonical_revision: "a4fbd9e8006b0a44a207afbf551957c28d8302025141abfdb7ea481b5b715fd7"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: AlphaProof — search and adaptation budgets
    label_en: AlphaProof — search and adaptation budgets
    anchor: alphaproof-search-vs-ttrl
  - label: Lean — checking the statement and its axioms
    label_en: Lean — checking the statement and its axioms
    anchor: lean-proof-acceptance
primary_sources:
  - https://github.com/leanprover-community/mathlib4/blob/c44e0c8ee63ca166450922a373c7409c5d26b00b/Mathlib/NumberTheory/EulerProduct/Basic.lean
  - https://terrytao.wordpress.com/2021/02/12/246b-notes-4-the-riemann-zeta-function-and-the-prime-number-theorem/
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
  - https://doi.org/10.1038/s41586-025-09833-y
  - https://arxiv.org/abs/2306.15626
  - https://arxiv.org/abs/2405.17216
  - https://arxiv.org/abs/2407.10040
  - https://arxiv.org/abs/2408.03350
  - https://arxiv.org/abs/2310.04353
  - https://arxiv.org/abs/2502.13834v3
  - https://arxiv.org/abs/2210.12283v1
  - https://arxiv.org/abs/2506.07477v1
  - https://arxiv.org/abs/2410.04753v1
  - https://lean-lang.org/doc/reference/latest/ValidatingProofs/
source_unit_id:
  - meeting-08-slides-riemann-zeta-demo
  - meeting-08-slides-formal-proving-setup
  - meeting-08-slides-lean-proof-demo
  - meeting-08-slides-rl-and-alphazero
  - meeting-08-slides-alphaproof-premise
  - meeting-08-slides-imo-setting
  - meeting-08-slides-alphaproof-imo-results
  - meeting-08-slides-alphaproof-methods
  - meeting-08-slides-test-time-rl
  - meeting-08-reading-01-catalogue-record
  - meeting-08-reading-02-catalogue-record
  - meeting-09-slides-autoformalization-sft-rlvr
  - meeting-09-slides-informal-formal-gaps
  - meeting-09-slides-lean-foundations
  - meeting-09-slides-kaiyu-yang-lean-theorem-proving-pipeline
  - meeting-09-slides-formalizing-inequalities
  - meeting-09-slides-autoformalization-challenges
  - meeting-09-slides-leaneuclid
  - meeting-09-slides-euclid-logical-gap
  - meeting-09-slides-equivalence-and-diagrams
  - meeting-09-slides-end-to-end-autoformalization
  - meeting-09-slides-autoformalization-takeaway
  - meeting-09-reading-01-catalogue-record
  - meeting-09-reading-02-catalogue-record
  - meeting-09-reading-03-catalogue-record
  - meeting-10-slides-informal-and-formal-math
  - meeting-10-slides-proving-gap-and-agenda
  - meeting-10-slides-lean-star
  - meeting-10-slides-modular-prover-motivation
  - meeting-10-slides-draft-sketch-prove
  - meeting-10-slides-leanhammer
  - meeting-10-slides-advanced-prover-recap
  - meeting-10-slides-proving-research-workflow
  - meeting-10-slides-minictx
  - meeting-10-reading-01-catalogue-record
  - meeting-10-reading-02-catalogue-record
  - meeting-10-reading-03-catalogue-record
  - meeting-10-reading-04-catalogue-record
  - meeting-11-slides-copra
  - meeting-11-slides-compiler-verification
  - meeting-11-slides-formal-reasoning-summary
  - meeting-11-reading-01-catalogue-record
---

# Formal proofs and mathematical agents

Given `h : P ∧ Q`, prove `Q ∧ P`. Take the proof of `Q`
from `h` first, then the proof of `P`. The order appears in
the Lean proof:

```lean
theorem swap_and (P Q : Prop) (h : P ∧ Q) : Q ∧ P := by
  constructor
  · exact h.right
  · exact h.left
```

`constructor` leaves two goals, first `Q` and then `P`.
`exact h.right` has the type needed for the first goal.
`exact h.left` would fail there: it proves `P`. Once both
goals are closed, the Lean kernel checks the resulting term.

For a harder theorem, the model may not know which lemma to try.
It proposes tactics, Lean returns new goals or an error, and a
search procedure chooses which branch to continue. A successful
proof establishes the statement written in Lean. The translation
from the original problem still has to be checked.

<span id="часть-i-формальная-среда-и-базовый-контракт"></span>
<span id="часть-i-формальная-среда-состояния-действия-и-доказательства"></span>
<span id="как-lean-проверяет-доказательство"></span>

## How Lean checks a proof

<span id="неформальное-решение-формальная-теорема-и-проверяемое-доказательство"></span>
<span id="записать-условие-и-доказать-его"></span>

### State the claim, then prove it

A mathematical argument can refer to a diagram or leave an algebraic
step implicit. Its formal version needs the types and assumptions,
the exact statement and definitions, and a proof that elaborates
to a term the kernel can check. Autoformalization produces that
formal statement or translates an informal proof. A prover may
instead start with a fixed statement. Losing a hypothesis and
running out of search time are different failures; keep the
outputs from the two stages separately.

Lean is a programming language and proof assistant. Mathlib
provides definitions and lemmas, including primality and divisibility.
Search can use only what the current project makes available.
If a familiar theorem has a different statement or is not imported,
the agent needs another lemma or an intermediate proof.

<span id="один-шаг-lean-proof-state-tactic-kernel-check"></span>
<span id="один-шаг-lean-состояние-доказательства-тактика-и-проверка-ядром"></span>
<span id="что-меняет-один-шаг-в-lean"></span>

### What one tactic changes

```text
n : ℕ
⊢ ∃ p, n ≤ p ∧ Nat.Prime p
```

The line above `⊢` supplies a natural number `n`. The goal
following `⊢` asks for a prime `p`
at least as large as `n`. This is a goal view, not a complete Lean
file. Producing it requires a declaration and Mathlib imports
that provide `Nat.Prime`.

A tactic constructs part of a proof. It may close a goal, create
subgoals, or fail. A mathematical plan here is to choose a prime
divisor of `n! + 1`. A prime at most `n` would divide `n!`
and therefore could not also divide `n! + 1`. A formal proof
needs lemmas about factorials, the chosen prime divisor and
divisibility of a difference.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m08-p79.png|The suggested divisor of n! + 1 leaves obligations for primality and the lower bound.]]

*Thomas Hubert, [Berkeley lecture 8, slide 79](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=79).*

The agent gives the model the goal, local hypotheses and any
retrieved lemmas. It parses the reply as a term or tactic.
When Lean executes a tactic, it can report an error or return
the remaining obligations. Choosing a witness still leaves
primality and the lower bound. Closing all obligations produces
a term for the kernel to check.

<span id="интуиция-предлагает-структуру-kernel-принимает-только-формальный-терм"></span>

The kernel checks the term without repeating the search or reading
the model's explanation. A term can type-check and still depend on an axiom such as
`sorryAx`, which stands in for an unfinished step. Inspect
its dependencies before accepting it,
as the LeanEuclid example below illustrates.

<span id="lean-термы-tactics-proof-state-and-environment"></span>
<span id="lean-термы-тактики-состояние-доказательства-и-окружение"></span>
<span id="из-тактик-получается-терм"></span>

### From tactics to a proof term

A proposition has type `Prop`. To prove proposition `P`, construct
a term of type `P`. `And.intro` accepts a proof of each conjunct,
so the opening example has a direct term version:

```lean
theorem swap_and_term (P Q : Prop) (h : P ∧ Q) : Q ∧ P :=
  And.intro h.right h.left
```

Tactics build the term while showing what is left to prove.
The environment includes imports, definitions, notation, type
class instances and theorems. The same printed goal can use
different definitions in different projects.

Induction replaces a goal about a natural number with a base case
and a successor case:

```lean
theorem zero_add_example (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add, ih]
```

The base goal `0 + 0 = 0` reduces by the definition of addition.
In the successor case, the induction hypothesis is `ih : 0 + n = n`.
Unfolding addition gives
`Nat.succ (0 + n) = Nat.succ n`; rewriting with `ih` makes both
sides equal. The examples use the standard library in Lean 4.19.0,
without Mathlib. See
[Theorem Proving in Lean 4, chapters 3 and 5](https://lean-lang.org/theorem_proving_in_lean4/Propositions-and-Proofs/).

<span id="часть-ii-автоформализация-и-смысл-постановки"></span>
<span id="как-записать-исходную-задачу-на-формальном-языке"></span>

## Translating the problem into a formal statement

<span id="три-разрыва-theorem-statement-proof-steps-and-semantic-equivalence"></span>
<span id="три-разрыва-постановка-шаги-доказательства-и-смысловая-эквивалентность"></span>
<span id="порядок-кванторов-меняет-утверждение"></span>

### Quantifier order changes the claim

“Every natural number has a larger one” allows a different choice
for each `n`. Choose `m = n + 1`:

```lean
theorem larger_exists : ∀ n : Nat, ∃ m : Nat, m > n := by
  intro n
  exact ⟨n + 1, Nat.lt_succ_self n⟩
```

Swapping the quantifiers asks for one number larger than all
naturals. Substituting that number for `n` refutes the claim:

```lean
theorem no_largest : ¬ (∃ m : Nat, ∀ n : Nat, m > n) := by
  intro h
  obtain ⟨m, hm⟩ := h
  exact (Nat.lt_irrefl m) (hm m)
```

Both statements parse; only the first matches the original sentence.
Formalizing a proof presents another problem: an informal
argument can omit an intermediate fact that Lean requires.
Even when both formal statements have proofs, checking whether
they express the same claim can require a further proof relating
their definitions and hypotheses.

<span id="почему-bleu-и-компиляция-не-доказывают-эквивалентность-теорем"></span>
<span id="как-сравнивать-две-формулировки"></span>

### Comparing two theorem statements

Swapping `∀` and `∃` moves little text but changes the claim.
Renaming a bound variable changes the text without changing the
claim. BLEU and successful parsing each miss one side of this
distinction. In [Autoformalization with Large Language Models, §4](https://arxiv.org/html/2205.12615v1#S4),
38 of 150 inspected translations were correct; errors included
different definitions, missing assumptions and type errors.
That count belongs to the 2022 protocol. In a new evaluation,
record parsing, correspondence to the original problem and
successful proof under the allowed axioms as separate outcomes.

<!-- BEGIN REVIEW: Berkeley finite products 2026-10-26 -->
<!-- source_unit_id: meeting-08-slides-riemann-zeta-demo -->
<a id="finite-products-and-missing-assumptions"></a>
<span id="worked-example-finite-products-and-missing-assumptions"></span>
<span id="какие-условия-нужны-для-произведения-эйлера"></span>

### Conditions on the Euler product

In the AIME example, multiplying the congruence by 4 leaves three
candidate moduli, but only two satisfy the original condition.
The Euler product has a different missing assumption: a sum over
integers formed from 2 and 3 is not a sum over all naturals, and
passing to infinite products requires convergence.
[Thomas Hubert, Berkeley lecture 8, slides 94–107](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=94)
uses these examples. The Mathlib comment below is unchanged.

#### AIME warm-up: a necessary condition can admit an extra candidate

The warm-up on slide 95 asks for positive integers $n$ such that
the remainder of $1^3+2^3+\cdots+n^3$ modulo $n+5$ is 17.
The remainder must be smaller than the divisor, so $m=n+5>17$.
Using the sum-of-cubes identity gives

$$
\begin{aligned}
S_n&=\left(\frac{n(n+1)}{2}\right)^2,\\
4S_n&=((m-5)(m-4))^2.
\end{aligned}
$$

Modulo $m$, the product $(m-5)(m-4)$ is 20. Thus the original condition
implies $400\equiv68\pmod m$, or $m\mid332$.
The divisors of $332=4\cdot83$ exceeding 17 are 83, 166 and 332.
They give $n=78,161,327$ respectively. Substitution into the original
remainder condition retains the first two and rejects the third:

| $n$ | $m=n+5$ | $S_n\bmod m$ |
|---:|---:|---:|
| 78 | 83 | 17 |
| 161 | 166 | 17 |
| 327 | 332 | 100 |

Multiplication by 4 preserved a necessary condition, not an equivalence
for every modulus. For $m=332$, the difference $100-17=83$ is not
divisible by 332, but four times that difference is. Cancelling 4 modulo
$m$ would require an additional invertibility argument. Testing the
original condition avoids making that invalid reverse step.
The requested sum is therefore $78+161=239$.

```python
def cube_sum(n):
    return (n * (n + 1) // 2) ** 2

moduli = [m for m in range(18, 333) if 332 % m == 0]
assert moduli == [83, 166, 332]
candidates = [m - 5 for m in moduli]
remainders = [cube_sum(n) % (n + 5) for n in candidates]
assert remainders == [17, 17, 100]
answers = [n for n in candidates if cube_sum(n) % (n + 5) == 17]
assert answers == [78, 161]
assert sum(answers) == 239
```

This code checks the finite candidate list after the divisibility argument.
It is not a search over an arbitrarily chosen upper bound for $n$:
the necessary condition $m\mid332$ supplies the complete list.

#### Expanding a product means choosing one exponent for each prime

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/formal-zeta-2026-10-26/prime-factorization.jpg|The denominators in the harmonic series are written as products of prime powers.]]

*Thomas Hubert, [Berkeley Meeting 8, slide 99](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=99). Each denominator is factored into primes. For example, $12=2^2\cdot3$. The expansion chooses the exponent of each prime independently. The picture depicts the divergent harmonic series. Page 97 credits the [zetamath channel](https://www.youtube.com/@zetamath); the exact video has not been identified.*

To remove the convergence question temporarily, take just the primes 2 and
3 and allow the exponents 0, 1 and 2. Distributing the two finite sums gives

$$
\begin{gathered}
\left(1+\frac12+\frac14\right)
\left(1+\frac13+\frac19\right)\\
=\sum_{a=0}^{2}\sum_{b=0}^{2}\frac{1}{2^a3^b}
=\frac{91}{36}.
\end{gathered}
$$

Each pair of exponents selects one term from each factor. The denominators
form the following finite set:

| | $b=0$ | $b=1$ | $b=2$ |
|---|---:|---:|---:|
| $a=0$ | 1 | 3 | 9 |
| $a=1$ | 2 | 6 | 18 |
| $a=2$ | 4 | 12 | 36 |

Unique prime factorization makes the map $(a,b)\mapsto2^a3^b$ injective.
Thus collecting terms by their integer denominator does not lose a
multiplicity. Replacing the primes by 2 and 4 would break that argument:
the denominator 4 can arise from choosing $2^2$ or from choosing $4^1$.
Distributivity still holds, but a sum over distinct denominators would
no longer equal the product.

The right-hand side is also not the ordinary partial sum
$1+1/2+1/3$. It includes numbers larger than 3, such as 18 and 36,
and excludes 5, 7 and every other integer with a prime factor outside
$\{2,3\}$. A correct formal statement must describe this indexing set.

```python
from fractions import Fraction as F

denominators = [2**a * 3**b for a in range(3) for b in range(3)]
assert len(denominators) == len(set(denominators)) == 9
left = (1 + F(1, 2) + F(1, 4)) * (1 + F(1, 3) + F(1, 9))
right = sum((F(1, d) for d in denominators), F(0))
assert left == right == F(91, 36)
assert right != 1 + F(1, 2) + F(1, 3)
nonprime_denominators = [2**a * 4**b for a in range(3) for b in range(3)]
assert len(set(nonprime_denominators)) < len(nonprime_denominators)
```

#### A finite set of primes can still index an infinite sum

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/formal-zeta-2026-10-26/prime-geometric-factors.jpg|Each geometric series contributes powers of a single prime.]]

*Thomas Hubert, [Berkeley Meeting 8, slide 100](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=100), repeated on slide 104. Each factor contains the powers of one prime. Choosing one term from each factor fixes the denominator $2^a3^b5^c\cdots$. The shading at the bottom is present in the original image.*

Keeping the primes 2 and 3 while allowing every nonnegative exponent gives
all positive integers of the form $2^a3^b$, an infinite set. For the exponent
$s=2$, the two geometric series converge and their product is

$$
\frac{1}{1-2^{-2}}\frac{1}{1-3^{-2}}
=\frac43\frac98=\frac32.
$$

This is a sum over integers whose prime factors belong to $\{2,3\}$,
not over all positive integers. For example, its terms include $1/36^2$
but not $1/5^2$. Adding more primes changes the indexing set; increasing
the exponent cutoff changes it in a different way.

Mathlib separates these steps explicitly. The following is the complete
introductory explanation from Michael Stoll's
[`EulerProduct/Basic.lean`, Mathlib v4.19.0](https://github.com/leanprover-community/mathlib4/blob/c44e0c8ee63ca166450922a373c7409c5d26b00b/Mathlib/NumberTheory/EulerProduct/Basic.lean),
retained in its original English under Apache 2.0:

<!-- BEGIN SOURCE: Mathlib Euler Products introduction -->
> The main result in this file is `EulerProduct.eulerProduct_hasProd`, which says that
> if `f : ℕ → R` is norm-summable, where `R` is a complete normed commutative ring and `f` is
> multiplicative on coprime arguments with `f 0 = 0`, then
> `∏' p : Primes, ∑' e : ℕ, f (p^e)` converges to `∑' n, f n`.
>
> `ArithmeticFunction.IsMultiplicative.eulerProduct_hasProd` is a version
> for multiplicative arithmetic functions in the sense of
> `ArithmeticFunction.IsMultiplicative`.
>
> There is also a version `EulerProduct.eulerProduct_completely_multiplicative_hasProd`,
> which states that `∏' p : Primes, (1 - f p)⁻¹` converges to `∑' n, f n`
> when `f` is completely multiplicative with values in a complete normed field `F`
> (implemented as `f : ℕ →*₀ F`).
>
> There are variants stating the equality of the infinite product and the infinite sum
> (`EulerProduct.eulerProduct_tprod`, `ArithmeticFunction.IsMultiplicative.eulerProduct_tprod`,
> `EulerProduct.eulerProduct_completely_multiplicative_tprod`) and also variants stating
> the convergence of the sequence of partial products over primes `< n`
> (`EulerProduct.eulerProduct`, `ArithmeticFunction.IsMultiplicative.eulerProduct`,
> `EulerProduct.eulerProduct_completely_multiplicative`.)
>
> An intermediate step is `EulerProduct.summable_and_hasSum_factoredNumbers_prod_filter_prime_tsum`
> (and its variant `EulerProduct.summable_and_hasSum_factoredNumbers_prod_filter_prime_geometric`),
> which relates the finite product over primes `p ∈ s` to the sum of `f n` over `s`-factored `n`,
> for `s : Finset ℕ`.
<!-- END SOURCE: Mathlib Euler Products introduction -->

Here `HasSum` and `HasProd` state convergence to a particular value;
`Summable` requires the existence of a sum. The condition on the norm
requires absolute convergence. The finite-prime lemma uses `factoredNumbers`
for the positive integers supported on the selected primes. A second lemma
controls the omitted tail as the finite set eventually contains every
prime below a sufficiently large bound. The final theorem combines those
two statements.

The module also assumes $f(1)=1$; this hypothesis and coprime
multiplicativity are declared before the final theorem. Copying only its
last two lines would hide those inherited assumptions.

For the zeta series, take $f(0)=0$ and $f(n)=n^{-s}$ for positive $n$.
The extra zero value is explicit because the Mathlib sum ranges over
natural numbers including 0. The series and Euler product converge
absolutely when $\operatorname{Re}s>1$
([DLMF 25.2.1 and 25.2.11](https://dlmf.nist.gov/25.2#E11)).

$$
\begin{aligned}
\zeta(s)&=\sum_{n=1}^{\infty}n^{-s}\\
&=\prod_{p\ \mathrm{prime}}(1-p^{-s})^{-1},\\
&\hspace{1em}\operatorname{Re}s>1.
\end{aligned}
$$

The raw $s=1$ expressions on slides 98–102 illustrate the combinatorics;
they are not equalities between finite complex values. The harmonic
series diverges, and the analytically continued zeta function has a pole
at 1. Its divergence also gives a proof that there are infinitely many
primes, the topic named on slide 94. Suppose instead that all primes
belonged to a finite set $P$. Each geometric series with ratio $1/p$
would have the finite sum $(1-1/p)^{-1}$, so their finite product would
be some constant $M$. Every integer from 1 to $N$ factors over $P$;
choosing a sufficiently large exponent cutoff includes each of their
reciprocals in the expanded product. All terms are positive, hence
$H_N=\sum_{n=1}^{N}1/n\leq M$ for every $N$.

But the harmonic series has no such bound. The block of terms with
$2^{j-1}<n\leq2^j$ contains $2^{j-1}$ terms, each at least $2^{-j}$.
Each block therefore contributes at least $1/2$, giving
$H_{2^r}\geq1+r/2$. For large enough $r$ this exceeds $M$,
contradicting the assumption of finitely many primes. This argument uses
finite partial sums and bounds; it does not assign a finite value to
$\zeta(1)$.

Substituting a point in the critical strip into the convergent
Euler-product argument would also violate its assumptions.

#### What the final slide leaves implicit

The last three frames relate products over primes to products or sums over
zeros of the analytically continued zeta function. These are different
indexing sets. Trivial zeros, nontrivial zeros and the pole at 1 must be
accounted for consistently; a product only over nontrivial zeros requires
additional factors, as in [DLMF 25.2.12](https://dlmf.nist.gov/25.2#E12).

For the final prime-counting relation, Terence Tao's
[246B Notes 4, Theorem 8](https://terrytao.wordpress.com/2021/02/12/246b-notes-4-the-riemann-zeta-function-and-the-prime-number-theorem/)
gives a precise version for non-integer $x>1$:

$$
\begin{aligned}
\psi(x)&=\sum_{p^k\leq x}\log p,\\
Z(x)&=\lim_{T\to\infty}
\sum_{\substack{\rho:\,|\operatorname{Im}\rho|\leq T}}
\frac{x^\rho}{\rho},\\
\psi(x)&=x-Z(x)-\log(2\pi)\\
&\quad-\frac12\log(1-x^{-2}).
\end{aligned}
$$

The sum defining $\psi$ uses primes $p$ and positive integers $k$;
the sum defining $Z$ uses nontrivial zeros, counted with multiplicity.
The symmetric limit specifies how that sum is taken. At $x=2.5$ the
prime-power sum is $\log2$; at $x=3.9$ it is $\log2+\log3$; at $x=4.1$
it additionally includes $\log2$ from $2^2$. This jump explains why a
statement at a prime power needs an endpoint convention. The logarithmic
correction in the displayed version accounts for trivial zeros that are
not included in $Z$.

These analytic statements are not consequences of the nine-term finite
calculation alone. A formal prover would also need the relevant definitions,
convergence results and analytic-continuation theorems in its environment.
The finite Python checks above do not establish those theorems or reproduce
the live AlphaProof demonstration. The full Mathlib proof is available in
the pinned module; the original source and reuse conditions are recorded in
[[05 Источники/Papers/Mathlib Euler products|the Mathlib source note]].
<!-- END REVIEW: Berkeley finite products 2026-10-26 -->


<span id="lips-сузить-бесконечное-пространство-действий-символическим-инструментом"></span>
<span id="какие-преобразования-ищет-lips"></span>

### How LIPS proposes new inequalities

[LIPS, §§2–4](https://arxiv.org/html/2502.13834v3#S2) restricts the
operations a model can consider. It distinguishes bounding an
expression (*scaling*) from rewriting it. Scaling can mean
applying the arithmetic–geometric mean inequality, not
multiplying the expression by a constant.

To prove `F ≤ G`, a known upper bound `F ≤ U` reduces the problem
to `U ≤ G`. The reduced claim can be false even when the original
is true. Take `F = x²`, `G = x² + 1` and `U = x² + 2`. At `x = 0`,
the old goal holds, but the new one asks for `2 ≤ 1`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m09-p58.png|The symbolic module proposes new inequality goals using rules for bounds and rewriting.]]

*Kaiyu Yang, [Berkeley lecture 9, slide 58](https://rdi.berkeley.edu/adv-llm-agents/slides/mathverification.pdf#page=58);
slides 61–64 and [LIPS, §3](https://arxiv.org/html/2502.13834v3#S3)
describe filtering candidates and involving the language model.*

The symbolic module matches rules to expressions to produce
new goals. Cylindrical algebraic decomposition, stored
counterexamples and numerical optimization reject some false
goals. Numerical search can find a counterexample; failure to
find one is not a proof. Taylor-based and separability
heuristics rank the surviving goals, and the model considers
the ten most promising.

Rewriting can expose a useful bound. Under `a² + b² + c² = 1`,
substitute `3(1 − c²)` for `3a² + 3b²`:

$$
\frac{1}{3a^2+3b^2+c^2}
= \frac{1}{3-2c^2}.
$$

For `a = 1/3` and `b = c = 2/3`, both denominators are `19/9`,
so both fractions equal `9/19`. The identity also requires a
nonzero denominator. Since `c² ≤ 1`, the new denominator is
at least 1, and the old one equals it. The v3 paper, §5.1,
describes 96 scaling tactics and 16 rewriting tactics
formalized in Lean 4. A different mathematical domain would
need its own operations and side conditions.

<span id="leaneuclid-текст-диаграмма-domain-system-and-smt"></span>
<span id="leaneuclid-как-связать-текст-диаграмму-формальную-систему-и-smt"></span>
<span id="какие-предпосылки-leaneuclid-извлекает-из-диаграммы"></span>

### What LeanEuclid extracts from a diagram

A geometry problem statement may give equal lengths and
parallel lines without naming which side of a line a point
lies on. [LeanEuclid, §§3–4](https://arxiv.org/html/2405.17216v1#S3)
formalizes points, lines, circles, order and construction
rules in system `E`. The paper's 173 problems comprise 48
from Book I of the *Elements* and 125 from UniGeo. The 100
held-out UniGeo tasks named in the repository are a subset,
not the size of the original benchmark.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/formal-math-2026-10-21/leaneuclid-autoformalization.png|The task and diagram yield a Lean statement and proof; SMT helps compare statements and infer premises.]]

*Logan Murphy et al., [Autoformalizing Euclidean Geometry, Figure 1, ICML 2024](https://proceedings.mlr.press/v235/murphy24a.html),
CC BY 4.0. Original labels retained.*

System `E` separates constructions from deductions about a
constructed configuration. For an equilateral triangle on
`AB`, draw a circle centered at `A` through `B` and another
centered at `B` through `A`. Before introducing an intersection
point `C`, show that the circles intersect. Membership in the
circles then gives `AC = AB` and `BC = BA`. The model selects
constructions and theorems. SMT can infer some of the required
facts from the context.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/formal-math-2026-10-21/leaneuclid-construction-rules.png|Solid arrows are explicit construction rules; dashed arrows infer missing premises.]]

*Murphy et al., [Figure 2, ICML 2024](https://proceedings.mlr.press/v235/murphy24a.html),
CC BY 4.0. Original figure and caption.*

<span id="что-именно-проверено-в-реализации-leaneuclid"></span>
<span id="от-какой-аксиомы-зависит-результат-leaneuclid"></span>

#### The axiom behind LeanEuclid's SMT result

The pinned [LeanEuclid revision `33a72c0`](https://github.com/loganrjmurphy/LeanEuclid/tree/33a72c0fa83914202532998a2e7b8771295c3cc7)
does not reconstruct a Lean proof from the result returned by
SMT. Its [solver wrapper](https://github.com/loganrjmurphy/LeanEuclid/blob/33a72c0fa83914202532998a2e7b8771295c3cc7/SystemE/Meta/Smt/Solver.lean#L44-L47)
declares `SMT_VERIF (P : Prop) : P`. On `unsat` the tactic
uses this axiom to close the goal. The axiom itself can prove
any proposition, including `False`.

The authors flag this limitation in the
[README](https://github.com/loganrjmurphy/LeanEuclid/blob/33a72c0fa83914202532998a2e7b8771295c3cc7/README.md#acknowledgements).
The translation to SMT, solver and wrapper are part of what
the user must trust. A successful build is therefore not an
axiom-free verification of the geometry. The limitation is
specific to the implementation, not to the possibility of
formalizing Euclidean geometry soundly.

<span id="разобранный-пример-скрытая-посылка-на-диаграмме"></span>
<span id="что-меняется-когда-точки-лежат-по-разные-стороны-прямой"></span>

### When the points lie on opposite sides of a line

Euclid I.24 compares the third sides of two triangles given
two pairs of equal sides and different included angles. The
proof introduces `G` and uses the isosceles triangle `DGF`.
The locations of `D` and `G` relative to `EF` require separate
cases; one diagram does not cover them all.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m09-p89.png|Euclid I.24 introduces G and uses the equal angles of the isosceles triangle DGF.]]

*Kaiyu Yang, [Berkeley lecture 9, slide 89](https://rdi.berkeley.edu/adv-llm-agents/slides/mathverification.pdf#page=89).*

[Appendix B](https://arxiv.org/html/2405.17216v1#A2)
works through the omitted configuration, with `D` and `G`
on opposite sides of `EF`:

$$
\begin{aligned}
\alpha &= \angle DGF = \angle DFG,\\
x &= \angle EFG,\qquad y = \angle EGF.
\end{aligned}
$$

The order of the rays gives

$$
\begin{aligned}
\angle DGE &= \alpha+y < \pi,\\
\angle DFE &= 2\pi-\alpha-x < \pi.
\end{aligned}
$$

The first inequality says `y < π − α`; the second says
`x > π − α`. Hence `x > y`. In triangle `EFG` the larger
angle faces the larger side, so `EG > EF`. The construction
gives `EG = BC` and therefore `BC > EF`. This step uses both
the equal base angles and the order of rays in this case.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m09-p97.png|With D and G on opposite sides of EF, the angles α, x and y establish x > y.]]

*Kaiyu Yang, [Berkeley lecture 9, slide 97](https://rdi.berkeley.edu/adv-llm-agents/slides/mathverification.pdf#page=97).*

<span id="семантическая-эквивалентность-и-диаграммные-следствия"></span>
<span id="как-e3-сравнивает-геометрические-утверждения"></span>

### How E3 compares geometry statements

LeanEuclid's E3 checker tries to prove both implications
between a reference and a candidate statement in language
`E`. Dropping a hypothesis can make a claim stronger; adding
one can make it weaker. Success in just one direction does
not establish equivalence. A consistency check guards
against accepting a result for incompatible hypotheses.

When full equivalence cannot be established, E3 can compare
parts of the statements. It matches variables heuristically
and checks corresponding conditions for semantic agreement.
That score is not a proof that the statements are equivalent.
The same SMT trust boundary applies.

<span id="end-to-end-autoformalization-without-conflating-translation-and-proof"></span>
<span id="сквозная-автоформализация-без-смешения-перевода-и-доказательства"></span>
<span id="что-сохранять-при-формализации-и-доказательстве"></span>

### What to save when formalizing and proving

```text
informal statement
  → candidate formal statement
  → semantic/domain checks
  → candidate proof
  → proof-assistant execution trace
  → checked term + axiom dependencies
```

Reject a wrong quantifier order before spending time on proof
search. If the statement is correct but search times out, keep
the last goals and attempted tactics. A single `solved` field
cannot distinguish those cases.

<span id="граница-автоматической-проверки"></span>
<span id="кто-проверяет-смысл-постановки"></span>

### Who checks what the statement means

<span id="что-даёт-формальная-среда-и-что-остаётся-человеческой-работой"></span>

The kernel checks the term's type in its formal environment.
A dependency check identifies the axioms it uses. A person or
another validation procedure still has to check whether the
formal statement represents the original problem.

<span id="часть-iii-обучение-поиск-и-модульные-prover"></span>
<span id="часть-iii-обучение-и-организация-поиска"></span>
<span id="поиск-и-обучение-математических-агентов"></span>

## Search and training for theorem provers

<span id="neural-prover-premise-retrieval-tactic-proposal-search-kernel"></span>
<span id="нейросетевой-prover-от-поиска-посылок-до-проверки-ядром"></span>
<span id="как-reprover-подбирает-леммы-и-тактики"></span>

### How ReProver retrieves premises and proposes tactics

[LeanDojo, §3](https://arxiv.org/html/2306.15626v1#S3)
extracts goals, tactics and referenced premises from Lean
projects. A premise can be imported or introduced earlier in
the module. Even a semantically perfect lemma from later in
the file is unavailable at the current proof step.

[ReProver, §§4–5](https://arxiv.org/html/2306.15626v1#S4)
encodes the goal and accessible premises with a Transformer. It learns from premises used in human proofs;
other premises in the file can supply hard negatives. In v1
the contrastive objective uses squared error. The retriever
returns up to 100 premises but only text that fits the model's
input is kept. In the prime example, a lemma establishing
primality does not automatically give the lower bound.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m09-p55.png|Candidate tactics branch from a Lean proof state; the proof assistant computes the new goals.]]

*Kaiyu Yang, [Berkeley lecture 9, slide 55](https://rdi.berkeley.edu/adv-llm-agents/slides/mathverification.pdf#page=55).
This is a search diagram, not the training procedure for ReProver's retriever.*

ReProver keeps a queue of states with priorities derived
from tactics' probabilities:

```text
formal theorem
  → initial proof state
  → retrieve accessible premises
  → sample/rank tactics
  → execute in Lean
  → enqueue valid successor states; reject errors and loops
  → repeat until no goals remain
  → kernel checks the reconstructed term
```

An `apply` tactic can be valid yet leave three obligations,
one of which cannot be proved from the available context.
Finding a relevant lemma does not guarantee that the model
will use it or that search will have time for that branch.

<span id="sft-и-rlvr-для-формальных-действий"></span>
<span id="обучение-на-готовых-доказательствах-и-собственных-попытках"></span>

### Training on supplied proofs and generated attempts

Supervised training pairs proof states with the next tactic.
In `swap_and` the three pairs precede `constructor`,
`exact h.right` and `exact h.left`. They contain no example
of recovering from a wrong first branch.

With verifiable rewards the model generates its own attempts.
Lean can supply a terminal reward for a checked proof. Syntax
errors, intermediate goals and cost can guide search or be
used for intermediate rewards; those are separate design
choices. A single remaining goal can be impossible, so
reducing the goal count does not always mean progress. The
formal checker can verify tactics and proofs but does not
repair the task statement.

<span id="почему-формальная-математика-допускает-alphazero-подобный-цикл"></span>
<span id="чему-поиск-доказательств-учится-у-alphazero"></span>

### What proof search borrows from AlphaZero

Proof assistants provide actions, state transitions and a
check for success. Search can supply successful continuations
to train a stronger model. Thomas Hubert compares that cycle
with AlphaZero:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m08-p30.png|The lecture compares experience, verifiable feedback, search and task selection in games and proof search.]]

*Thomas Hubert, [Berkeley lecture 8, slide 30](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=30).
The analogy does not make the algorithms identical.*

A game already comes with rules and an initial position.
Training a prover also needs formal statements, a library
and tasks of suitable difficulty. Running out of time on
`Q ∧ P` establishes neither that claim nor its negation.

<span id="alphaproof-как-формализатор-prover-и-поиск"></span>
<span id="alphaproof-формализатор-модель-доказательства-и-поиск"></span>
<span id="две-модели-в-alphaproof"></span>

### The two models in AlphaProof

One model translates informal problems into Lean statements;
a second proposes proof steps. Search keeps a tree of states
and decides what to explore. An autoformalizer can create more
exercises than the library has human proofs for.

<span id="от-mathlib-sft-к-rl-с-lean-verifier"></span>
<span id="обучение-на-доказательствах-mathlib-и-на-результатах-поиска"></span>
<span id="какие-доказательства-попадают-в-обучение-alphaproof"></span>

### Which proofs enter AlphaProof's training data

The initial model learns from Mathlib proofs. The formalizer
adds candidate problems from an informal corpus. The prover
tries to prove or disprove them. Successful checked traces
supply training data, and the updated model returns to search.
Easier tasks provide successful traces while harder tasks
are still unsolved.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m08-p84.png|The formalizer produces statements and the prover supplies checked traces for training.]]

*Thomas Hubert, [Berkeley lecture 8, slide 84](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=84).
`O(1M)` and `O(100M)` are scales shown on the slide, not independently counted totals.*

Feeding found proofs back into training is *expert iteration*.
Search together with the formal checker supplies the new
examples; a person need not label every trace.

<span id="test-time-rl-как-локальная-специализация-вокруг-трудной-задачи"></span>
<span id="как-alphaproof-дообучается-на-вариантах-задачи"></span>

### How AlphaProof adapts to variants of a problem

When search fails to find a proof of the target, AlphaProof
generates related statements: weaken a condition, isolate a
sublemma, or vary the conclusion. Lean can check that a variant
is well-formed even if it is false. The prover tries to prove
or refute the variants and keeps the successful traces.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m08-p91.png|Related problems supply training traces; the specialized prover keeps returning to the original goal.]]

*Thomas Hubert, [Berkeley lecture 8, slide 91](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=91).
The expanding region is conceptual, not a guarantee of solving every new goal.*

A specialist starts from the general prover's weights and
trains on state–tactic pairs from successful traces. It also
tries the original theorem: solving a variant does not close
the target. Once that target is proved, the scheduler stops
issuing its tasks and variants. This is the test-time
reinforcement learning (TTRL) described in
[Hubert et al., Methods → Inference](https://doi.org/10.1038/s41586-025-09833-y).

<a id="alphaproof-search-vs-ttrl"></a>
<span id="сколько-вычислений-уходит-на-поиск-и-на-дообучение"></span>

#### How much compute goes to search and adaptation

Panel **a** keeps weights fixed and spends compute on more
search. Panel **b** spends it on adaptation, then measures
solve rate with 4000 search simulations. Follow the orange
`formal-imo` curve in both panels:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/alphaproof-nature/figure-4-search-vs-ttrl.png|Panel a varies search in TPU hours; panel b varies adaptation in TPU days. The orange curve is formal-imo.]]

*Thomas Hubert, Rishi Mehta, Laurent Sartran et al.,
[Olympiad-level formal mathematical reasoning with reinforcement learning, Fig. 4](https://www.nature.com/articles/s41586-025-09833-y/figures/4).
Publisher's original, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Moving from two TPU minutes to twelve TPU hours per problem
raises the solve rate from 33.1% to 43.7%, a 10.6-point gain.
At an average of 50 TPU days of adaptation per problem the
rate is 53.9%; near 500 TPU days it is 58.3%. The last number
is 14.6 points above the twelve-hour search result at a
different budget. The horizontal axis in panel **a** is logarithmic in hours;
its counterpart in panel **b** is linear in days. Device-days measure allocated
hardware time rather than elapsed time when devices run in
parallel. Each per-problem figure is total compute divided
by the number of problems.

<span id="что-именно-проверял-imo-protocol"></span>
<span id="что-именно-проверял-протокол-imo"></span>
<span id="что-было-автоматизировано-на-imo-2024"></span>

### What was automated at IMO 2024

Experts translated problem statements into the formal language.
According to lecture 8, for problems that asked for an answer,
Gemini proposed about a hundred candidates. Checking and
filtering those candidates could guide further work around
the target. AlphaGeometry 2 handled geometry; AlphaProof
handled other areas. This protocol did not test fully
autonomous translation of all contest problems.

<span id="результат-и-границы-утверждения"></span>
<span id="какие-задачи-решили-на-imo-2024"></span>

### Which IMO 2024 problems were solved

In the [official report](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/),
AlphaProof solved problems 1, 2 and 6, and AlphaGeometry 2
solved problem 4. Four complete solutions earned 28 of 42
points; the gold threshold was 29. One problem took minutes
and others took up to three days. Contestants had two sessions
of 4.5 hours each.

The lecture notes that problem 5 was not formalized on the
day of the contest. Neither problem 3 nor 5 earned points.
A comparison needs the same translation assumptions, system
split, answer-candidate procedure and time budget.


<span id="lean-star-thought-before-tactic-and-expert-iteration"></span>
<span id="lean-star-рассуждение-перед-тактикой-и-expert-iteration"></span>
<span id="как-lean-star-обучается-на-успешных-траекториях"></span>

### How Lean-STaR trains on successful attempts

[Lean-STaR, §3](https://arxiv.org/html/2407.10040v1#S3)
predicts informal reasoning before a tactic. For `Q ∧ P`
it might say to construct the conjunction and take both
parts from `h : P ∧ Q`. Lean executes the tactics; it does
not check the accompanying explanation.

Human thoughts before every tactic are not available in
the training records. The authors show GPT-4 the proof
state and the correct tactic, then ask it for a retrospective
explanation. Lean-CoT trains on state–explanation–tactic
triples. The extra text is synthetic annotation, not evidence
of how the original proof author or the model decided.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/formal-math-2026-10-21/leanstar-pipeline.png|Lean-STaR labels known steps, trains Lean-CoT and learns again from successful proof trajectories.]]

*Lin et al., [Lean-STaR, Figure 4](https://arxiv.org/html/2407.10040v1#S3.F4),
CC BY 4.0. Original labels retained.*

The model then attempts the training theorems itself.
Lean identifies completed proofs. The authors deduplicate
and use the kept triples for the next round of fine-tuning.
A proved theorem does not certify the informal text.

<span id="почему-изменён-алгоритм-поиска"></span>
<span id="почему-lean-star-запускает-параллельные-попытки"></span>

#### Why Lean-STaR samples parallel attempts

The authors found that average token log probability did
not rank tactic candidates well once they also contained
informal text. A likely explanation can accompany an
unhelpful tactic. Instead, run `K` attempts from the initial
state. Each proposes a thought and tactic, advances on a
valid tactic and retries from the same state after failure.
Every proposal, including a rejected one, spends part of
the `N`-generation limit.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/formal-math-2026-10-21/leanstar-search-sampling.png|Branching search is on the left. Independent attempts with retries are on the right.]]

*Lin et al., [Lean-STaR, Figure 5](https://arxiv.org/html/2407.10040v1#S3.F5),
CC BY 4.0.*

The original diagram labels some positive numbers `log p`,
although a probability cannot have a positive logarithm.
Use §3.3 of the paper for the priority rule, not the numbers
in the illustration.

Table 3 uses InternLM2-Math-plus-7B, `K = 64`, `N = 50` and
temperature 0.7. SFT scores 43.4%, Lean-CoT 45.5% and one
iteration of Lean-STaR 46.3%. The 64 attempts combine two
sets of 32. Compare those rows with one another; changing
the base model or the search budget changes the experiment.

<span id="почему-prover-стоит-разложить-на-retrieval-atp-and-search"></span>
<span id="почему-prover-стоит-разделить-на-retrieval-atp-и-поиск"></span>
<span id="зачем-разделять-поиск-посылок-и-автоматическое-доказательство"></span>

### Why separate premise retrieval from automated proof

A model can choose a useful plan and get the lemma name
wrong. Automated algebra can close a local goal without
finding the intermediate construction for the whole theorem.
Separate the plan, library retrieval and local proof so that
each can use a different method.

<span id="draftsketchprove-отделить-математический-план-от-заполнения-holes"></span>
<span id="draftsketchprove-отделить-математический-план-от-заполнения-пропусков"></span>
<span id="draftsketchprove-от-плана-к-доказательству"></span>

### Draft–Sketch–Prove: from a plan to a checked result

[Draft–Sketch–Prove, §3](https://arxiv.org/html/2210.12283v1#S3)
starts with an informal solution. The model turns it into a
formal sketch with intermediate statements and gaps. In the
paper, Isabelle automation including Sledgehammer fills those
gaps. A gap in a search sketch is not an accepted proof.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p64.png|An informal solution becomes a formal sketch with marked proof gaps.]]

*Sean Welleck, [Berkeley lecture 10, slide 64](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=64).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p66.png|Several informal solutions produce different formal sketches whose obligations are attempted separately.]]

*Sean Welleck, [Berkeley lecture 10, slide 66](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=66).*

Figure 2 of the paper asks for an inequality for real `a`:

$$
10a \leq 28a^2+1.
$$

The plan completes the square: start from `(a − 5/28)² ≥ 0`.
Expanding and multiplying by 28 gives

$$
28a^2-10a+\frac{25}{28}\geq 0.
$$

Adding `3/28` gives `28a² − 10a + 1 ≥ 0`. The choice of
`5/28` belongs to the mathematical plan. Local automation
can check the square, the expansion and the rational
arithmetic. If a sketch uses the wrong coefficient or omits
an assumption, an attempted gap fill can fail. The sketch
then needs a revision.

<span id="leanhammer-premise-selector-atp-aesop-and-reconstruction"></span>
<span id="leanhammer-выбор-посылок-atp-aesop-и-реконструкция-доказательства"></span>
<span id="как-leanhammer-восстанавливает-доказательство-в-lean"></span>

### How LeanHammer reconstructs a proof in Lean

A hammer connects a proof assistant to an automatic theorem
prover (ATP). It retrieves premises and translates the goal
for the external prover. To retain trust in the Lean kernel,
it then reconstructs a proof in Lean instead of accepting
the external answer as a proof term.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p78.png|LeanHammer selects premises, translates with Lean-auto, searches with Zipperposition and reconstructs with Duper.]]

*Thomas Zhu, [Berkeley lecture 10, slide 78](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=78).*

[LeanHammer, §3](https://arxiv.org/html/2506.07477v1#S3)
encodes the goal and candidate premises. Full constant names
replace overloaded notation such as `gcd`. Library embeddings
can be precomputed, and local hypotheses are added at search
time. Contrastive training pulls relevant premises closer;
masking prevents another known positive for the same query
from being treated as a negative.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p82.png|The premise encoder learns to rank used lemmas above candidate negatives.]]

*Thomas Zhu, [Berkeley lecture 10, slide 82](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=82).*

Aesop manages the search. Lean-auto translates the goal,
Zipperposition reports premises sufficient for a proof, and
Duper uses them to construct a Lean proof. Aesop can also
apply lemmas directly. The paper supplies 16 premises to
the external route and 32 to direct application.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p85.png|Aesop combines direct premise application with external proof search and Lean reconstruction.]]

*Thomas Zhu, [Berkeley lecture 10, slide 85](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=85).*

[Appendix B](https://arxiv.org/html/2506.07477v1#A2)
proves `Associated y (gcd x y) ↔ y ∣ x`. Associated means equal
up to multiplication by a unit. For natural numbers the
result is familiar: `gcd(x,y) = y` exactly when `y` divides
`x`. Both sides hold for `x = 6, y = 3`.

Aesop splits the equivalence into two directions. The first
uses premises ranked 1 and 6 through the external route,
then Duper. The first 16 are insufficient for the second
direction, but `associated_of_dvd_dvd` at rank 18 is available
for direct application. It creates two divisibility goals.
One uses the external route with a new set of premises;
the other uses `gcd_dvd_right`, ranked 4, directly.

<span id="minictx-контекст-реального-проекта-и-time-split"></span>
<span id="где-ломается-переход-от-олимпиадной-задачи-к-проекту-mathlib"></span>
<span id="minictx-контекст-реального-проекта-и-временное-разделение-данных"></span>
<span id="что-меняется-если-модель-видит-предыдущий-код"></span>

### What changes when the model sees earlier code

[miniCTX, §§2–4](https://arxiv.org/html/2408.03350v1#S2)
tests problems whose definitions or lemmas can occur just
above the theorem or in another module. Its v1 collection
includes 376 theorems: 87 from PrimeNumberTheoremAnd,
54 from PFR, 50 from newer Mathlib additions and 185 from HTPI.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p109.png|miniCTX combines four sources and uses a time cutoff to assess novelty.]]

*Sean Welleck, [Berkeley lecture 10, slide 109](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=109).
Whether a task is unseen depends on the model's training cutoff.*

Earlier definitions and lemmas belong to the in-file context.
Cross-file dependencies require imports and retrieval in
other modules. The answer proof must be excluded. Holding
out a theorem in a known file, a new definition and a whole
project tests different kinds of generalization.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p110.png|In-file tasks depend on earlier declarations; cross-file tasks can depend on imported modules.]]

*Sean Welleck, [Berkeley lecture 10, slide 110](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=110).*

File tuning trains on preceding code, proof state and next
tactic. The body of a just-defined predicate tells the model
what needs to be unfolded; its printed name alone may not.
The simple setup truncates context at 1024 tokens, which
can remove a definition farther away.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p111.png|The file-conditioned input adds earlier source code to the proof state.]]

*Sean Welleck, [Berkeley lecture 10, slide 111](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=111).*

[Table 2, §5.3](https://arxiv.org/html/2408.03350v1#S5.SS3)
compares state-tactic and file tuning on DeepSeek-Coder-1.3B.
The miniF2F rates are 32.79% and 33.61%; the miniCTX rates
are 9.31% and 31.65%. Both use 32 proposals per expanded
state and at most 100 expansions. The complete-proof GPT-4o
row uses a different procedure and budget.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m10-p112.png|The lecture plots results with and without file context; its exact values differ from the paper's Table 2.]]

*Sean Welleck, [Berkeley lecture 10, slide 112](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf#page=112).
The slide's baseline is near 20%, not the 9.31% in the cited paper table;
the slide alone does not establish the reason for the difference.*

The difficult PFR subset does not improve with file tuning.
Sections 5.4–5.5 also show reliance on existing automation
such as `simp` and `nlinarith`. Access to more context does
not establish that the model can replace those tactics.

<span id="copra-stateful-backtracking-with-prover-feedback"></span>
<span id="copra-поиск-с-состоянием-возвратом-и-обратной-связью-prover"></span>
<span id="copra-поиск-с-состоянием-возвратом-и-формальной-обратной-связью"></span>
<span id="как-copra-возвращается-к-прежней-цели"></span>

### How COPRA returns to a previous goal

[COPRA, §§2–3](https://arxiv.org/html/2310.04353v1#S2)
uses an existing LLM without updating its weights in the
search loop. The prompt contains the goal, some history,
failed actions and retrieved lemmas when needed. The proof
assistant executes the proposed tactic and returns a state
or error.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p16.png|The COPRA loop proposes and executes a tactic, retrieves premises as needed and backtracks after failed branches.]]

*Swarat Chaudhuri, [Berkeley lecture 11, slide 16](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=16).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p17.png|The prompt records current and failed actions; the response proposes a tactic in the required format.]]

*Swarat Chaudhuri, [Berkeley lecture 11, slide 17](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=17).*

Depth-first search maintains a stack of states and a set
`Bad(O)` for each state. On failure or exhaustion it restores
the parent and excludes that continuation there. The tactic
can still be legal in a different context. Backtracking resets
the proof state, not merely a line of chat history. A format
error may call for reparsing or a fresh response. A tactic
error calls for a different action. The partial-order filter
prunes repeats and some harder states, but it is not a
complete test for dead ends. Calls and depth remain bounded.

In the lecture example,
`3(14n + 3) − 2(21n + 4) = 1` proves coprimality:
a common positive divisor must divide 1. The proposed
formal proof cites `Nat.gcd_sub_right`, unavailable in that
environment. Verified sublemmas enable a different last
step. This is repair during search at fixed weights.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p24.png|The completed coprimality proof uses checked sublemmas; an unavailable lemma name is noted below.]]

*Swarat Chaudhuri, [Berkeley lecture 11, slide 24](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=24).
The bottom annotation identifies the failed lemma reference; the slide does not show
the Lean error log.*

<span id="improver-улучшение-уже-найденного-доказательства"></span>
<span id="как-improver-сравнивает-доказательства-одной-теоремы"></span>

### How ImProver compares proofs of the same theorem

[ImProver, §3](https://arxiv.org/html/2410.04753v1#S3)
takes the statement, context, existing proof and a metric.
A candidate has to pass Lean before it can be ranked.
The opening proof uses three tactic calls. One
`exact And.intro h.right h.left` can accomplish the same
result; this is a teaching example, not an ImProver run.
Fewer calls do not guarantee faster checking.

Another metric measures the share of `have` tactics with
an explicit type. The explicit intermediate claims can make
a proof easier to follow, but the metric is only a proxy.

ImProver can insert states extracted from Lean's `InfoTree`
as comments. These Chain-of-States comments report actual
obligations, unlike synthetic explanations in Lean-STaR.
It can choose among candidates or give one attempt's error
and score to the next attempt. With `m` rounds and `n` samples
per round the budget is up to `m × n` generations. When
comparing variants, keep proof acceptance, checking cost
and the readability criterion with the reported length.


<span id="часть-iv-проектный-пример-и-приёмка-результата"></span>
<span id="доказательство-корректности-компилятора"></span>

## Proving a stack compiler correct

<span id="сквозной-пример-корректность-компилятора"></span>
<span id="от-вычисления-выражения-к-стековой-программе"></span>

### From an expression to a stack program

We can evaluate `(2 + 3) × 4` by following its expression tree, or compile
the tree into stack instructions and run them. Both executions give 20.
In [Swarat Chaudhuri's Coq example, Berkeley Meeting 11, slides 25–39](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=25),
the goal is to prove that the two methods agree for every expression.
The proof gets stuck on a subexpression: when its code runs, the stack
may already hold values and some instructions still remain.

The definitions follow the lecture. The proof below uses the standard
`List.app_assoc`. In order, the Coq blocks form one file for Coq 8.20.1
and require no external tactic libraries.

<span id="1-исходный-язык-и-его-семантика"></span>
<span id="как-вычисляется-исходное-выражение"></span>

#### Evaluating a source expression

A source expression is a natural number or a binary node for addition
or multiplication. `expDenote` recursively evaluates both children
of a binary node and applies the operation to their values.

```coq
Require Import Coq.Lists.List.

Inductive binop : Set := Plus | Times.

Inductive exp : Set :=
| Const : nat -> exp
| Binop : binop -> exp -> exp -> exp.

Definition binopDenote (b : binop) : nat -> nat -> nat :=
  match b with Plus => plus | Times => mult end.

Fixpoint expDenote (e : exp) : nat :=
  match e with
  | Const n => n
  | Binop b e1 e2 => binopDenote b (expDenote e1) (expDenote e2)
  end.
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p27.png|The Coq definitions of expressions and their recursive evaluation.]]

*`expDenote` returns the number represented by a constant, or combines the values of the two children of a binary node. This is the value the compiler must preserve. Source: [Berkeley, Meeting 11, slide 27](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=27).*

<span id="2-целевой-стековый-язык"></span>
<span id="как-исполняются-стековые-инструкции"></span>

#### Executing a stack instruction

The head of the list is the top of the stack. `iConst` pushes a number.
`iBinop` pops the first operand from the top and the second from the next
position. It pushes their result. With fewer than two operands,
`instrDenote` returns `None`, and `progDenote` stops instead of executing
the remaining instructions.

```coq
Inductive instr : Set :=
| iConst : nat -> instr
| iBinop : binop -> instr.

Definition prog := list instr.
Definition stack := list nat.

Definition instrDenote (i : instr) (s : stack) : option stack :=
  match i with
  | iConst n => Some (n :: s)
  | iBinop b =>
      match s with
      | arg1 :: arg2 :: s' => Some (binopDenote b arg1 arg2 :: s')
      | _ => None
      end
  end.

Fixpoint progDenote (p : prog) (s : stack) : option stack :=
  match p with
  | nil => Some s
  | i :: p' =>
      match instrDenote i s with
      | None => None
      | Some s' => progDenote p' s'
      end
  end.
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p30.png|The stack interpreter distinguishes a push, a binary operation, and stack underflow.]]

*Swarat Chaudhuri, [Berkeley, Meeting 11, slide 30](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=30). The next slide defines the recursive program interpreter, `progDenote`, included in the code above.*

<span id="3-компилятор"></span>
<span id="в-каком-порядке-компилируются-аргументы"></span>

#### Compiling the right operand first

For `Binop b e1 e2`, the compiler emits the code for `e2`, then for `e1`,
then the operation. This leaves the value of `e1` on top of the stack
when the operation begins, where the interpreter expects the first operand.

```coq
Fixpoint compile (e : exp) : prog :=
  match e with
  | Const n => iConst n :: nil
  | Binop b e1 e2 => compile e2 ++ compile e1 ++ iBinop b :: nil
  end.
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p32.png|A binary node compiles to the right operand, the left operand, and the binary instruction.]]

*`compile` turns the expression tree into a list of stack instructions. Emitting the right operand first puts the left operand at the top when the instruction pops its arguments. Source: [Berkeley, Meeting 11, slide 32](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=32).*

Starting with an empty stack, `(2 + 3) × 4` runs as follows:

| Instruction completed | Stack, top at left |
|---|---|
| Initial state | `[]` |
| `iConst 4` | `[4]` |
| `iConst 3` | `[3, 4]` |
| `iConst 2` | `[2, 3, 4]` |
| `iBinop Plus` | `[5, 4]` |
| `iBinop Times` | `[20]` |

Swapping the two operands would give the same number for these two
operations because addition and multiplication commute. The compiler
and interpreter agree on operand order, so the proof need not rely on
either operation being commutative.

<span id="4-естественная-теорема-оказывается-слишком-слабой-для-индукции"></span>
<span id="почему-гипотеза-о-пустом-стеке-не-помогает"></span>

#### Why the empty-stack hypothesis is too weak

Start by stating correctness for the whole program on an empty stack:

```text
Theorem compile_correct :
  forall e,
    progDenote (compile e) nil = Some (expDenote e :: nil).
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p33.png|The first theorem compares expression evaluation with execution on an empty stack.]]

*Both stacks in the statement start at `nil`: the first is the input, and the other is the tail of the single-value result. The statement runs `compile e` alone. Source: [Berkeley, Meeting 11, slide 33](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=33).*

Slide 34 reports that the automated attempt failed, without showing its
trace. To see where direct induction stalls, follow the binary case.
After `compile e2` has run, its result is on the stack and `compile e1`
and `iBinop b` still have to execute. The induction hypothesis describes
running `compile e1` alone on an empty stack. It says nothing about this
execution.

<span id="5-усиливаем-индукционное-утверждение"></span>
<span id="лемма-для-произвольного-стека-и-продолжения"></span>

#### A lemma for any stack and continuation

Let `s` be the stack before evaluating an expression and `p` the program
that follows it. Running `compile e` before `p` should have the same effect
as pushing `expDenote e` onto `s` and running `p` directly:

```coq
Lemma compile_one_instr :
  forall e p s,
    progDenote (compile e ++ p) s =
    progDenote p (expDenote e :: s).
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p37.png|The stronger lemma quantifies over an arbitrary stack and continuation.]]

*The lemma specifies the effect of the compiled expression inside a larger program. It leaves both the initial stack `s` and the continuation `p` free. Source: [Berkeley, Meeting 11, slide 37](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=37).*

With those parameters free, the induction hypotheses apply to
the running program. Associativity of `++` exposes the suffix after
each compiled subexpression:

```coq
Proof.
  induction e as [n | b e1 IH1 e2 IH2].
  - intros p s. simpl. reflexivity.
  - intros p s. simpl.
    repeat rewrite <- app_assoc.
    rewrite IH2.
    rewrite IH1.
    reflexivity.
Qed.
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p38.png|The binary case applies the induction hypotheses for the two operands in execution order.]]

*Swarat Chaudhuri, [Berkeley, Meeting 11, slide 38](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=38). The slide uses `app_assoc_reverse`. The code above rewrites with the standard `app_assoc` instead.*

Despite its name on the slide, `compile_one_instr` describes the compiled
code of an entire expression. In the constant case this code pushes a number
and continues with `p`. In the binary case, apply the hypothesis for `e2`
to stack `s` and continuation `compile e1 ++ iBinop b :: p`. Then apply
the hypothesis for `e1` to stack `expDenote e2 :: s` and continuation
`iBinop b :: p`. The binary instruction pops the two values, combines them
in the right order, and resumes `p` with the value of the whole expression
on the stack.

<span id="6-исходная-теорема-становится-следствием"></span>
<span id="частный-случай-пустого-стека"></span>

#### The empty-stack corollary

Set `p = nil` and `s = nil` in the lemma. The right-hand side reduces to
`Some (expDenote e :: nil)`. On the left, `app_nil_r` removes the empty
continuation:

```coq
Theorem compile_correct :
  forall e,
    progDenote (compile e) nil = Some (expDenote e :: nil).
Proof.
  intro e.
  pose proof (compile_one_instr e nil nil) as H.
  rewrite app_nil_r in H.
  exact H.
Qed.
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-math/m11-p39.png|The original compiler theorem follows from the generalized lemma.]]

*Swarat Chaudhuri, [Berkeley, Meeting 11, slide 39](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf#page=39). The slide proves the result by cases and applies the lemma to the subexpressions. The code above applies it directly with an empty stack and continuation.*

The stronger lemma also covers an existing stack. Start with `[7]`.
The compiled expression places 20 on top, giving `[20, 7]`. A continuation
consisting of one addition then gives `[27]`.

```coq
Definition sample_expr : exp :=
  Binop Times (Binop Plus (Const 2) (Const 3)) (Const 4).

Example sample_value : expDenote sample_expr = 20.
Proof. reflexivity. Qed.

Example sample_run :
  progDenote (compile sample_expr) nil = Some (20 :: nil).
Proof. reflexivity. Qed.

Example sample_with_continuation :
  progDenote (compile sample_expr ++ iBinop Plus :: nil) (7 :: nil)
    = Some (27 :: nil).
Proof. reflexivity. Qed.

Example insufficient_stack :
  progDenote (iBinop Times :: nil) nil = None.
Proof. reflexivity. Qed.

Print Assumptions compile_one_instr.
Print Assumptions compile_correct.
```

The sample computations check particular inputs, including underflow.
The lemma covers every expression and every finite stack and continuation.
`Print Assumptions` reports whether the resulting theorem depends on
unproved assumptions.

<span id="сравнение-архитектур-formal-prover"></span>
<span id="сравнение-архитектур-формального-prover"></span>
<span id="сравнение-способов-построения-доказательства"></span>

### Comparing proof construction methods

| System | What it proposes and how the result is checked |
|---|---|
| ReProver | Retrieves available lemmas and proposes tactics. Lean executes the tactics; the search queue keeps the states the search was able to reach. |
| LIPS | Symbolic rules produce and filter inequalities. A model helps choose what to try next. Rewrites and bounds are formalized in Lean 4. |
| LeanEuclid | Formalizes geometry in system `E`. In the pinned implementation, some SMT-backed steps rely on the `SMT_VERIF` axiom. |
| Lean-STaR | Generates a rationale and a tactic on each independent attempt. Lean checks the action and proof term; the rationale remains unchecked prose. |
| Draft–Sketch–Prove | Drafts an informal solution and a formal sketch. Isabelle's automation fills the holes; a sketch with gaps is not accepted as a proof. |
| LeanHammer | Retrieves premises, tries direct application, or calls an external prover. Duper reconstructs an accepted proof in Lean. |
| COPRA | Proposes a tactic, uses error feedback, and backtracks to try another route. The proof assistant executes the steps. |
| AlphaProof | Translates problems into Lean and searches for proofs. Checked trajectories are used for training, including those found on variants of a hard problem. |
| ImProver | Proposes revisions to an existing proof. It first type-checks each candidate in Lean, then compares length or another metric. |

These studies report results on different tasks, libraries, base models,
and search budgets. Their published percentages do not give a common ranking.

<span id="воспроизводимый-research-workflow"></span>
<span id="воспроизводимый-исследовательский-процесс"></span>
<span id="что-записывать-при-сравнении-систем"></span>

### What to record when comparing systems

Keep the project commit, Lean or Coq version, imports, and model checkpoints.
Record whether the data split allows the training run to see the theorem
itself, its preceding context, local premises, or other files in the same
project. A library update can make a premise available or change a tactic's
behavior.

Measure the budget in operations the system actually performs: model
requests, proposed tactics, expanded proof states, checking time and cost.
An invalid Lean-STaR tactic uses a generation but creates no state.
One LeanHammer node may require several external calls and reconstruction.
A count of “steps” cannot be compared until those steps are defined.

A search log should record the proposed action, its input state, the result
or error, and any backtrack. Distinguish completed proofs, exhausted
budgets, environment errors, and forbidden dependencies. Reaching a timeout
or failing to find a proof provides no evidence that the theorem is false.

<span id="acceptance-contract-без-пропусков-и-подмены-постановки"></span>
<a id="lean-proof-acceptance"></a>
<span id="приёмка-зависимости-доказательства-и-неизменность-постановки"></span>
<span id="как-обнаружить-пропуск-или-подмену-постановки"></span>

### Checking for gaps or a changed statement

Return to the conjunction from the opening example. `Given.Goal` is
the fixed type of the proof to be checked. Three candidates have that
type: one swaps the two parts of the conjunction, one leaves a hole,
and one derives the result from a deliberately introduced axiom
`invented : False`. A fourth proves a different proposition, also named
`Goal`.

```lean
namespace Given
def Goal : Prop := ∀ P Q : Prop, P ∧ Q → Q ∧ P
end Given

theorem valid : Given.Goal := by
  intro P Q h
  exact ⟨h.right, h.left⟩

theorem unfinished : Given.Goal := by sorry

axiom invented : False
theorem helper : False := invented
theorem forged : Given.Goal := False.elim helper

namespace Candidate
def Goal : Prop := True
theorem easy : Goal := True.intro
end Candidate

#print axioms valid
#print axioms unfinished
#print axioms forged
#print axioms Candidate.easy

example : Given.Goal := valid
example : Given.Goal := unfinished
example : Given.Goal := forged

#check Candidate.easy
```

In the recorded run, Lean 4.19.0 accepted the file and warned about
`sorry`. Each of the three `example` declarations type-checked.
Adding `example : Given.Goal := Candidate.easy` gives `type mismatch`.
Unfolding `Candidate.Goal` yields `True`; unfolding `Given.Goal`
yields the proposition about swapping the conjuncts.

| Candidate | Agrees with `Given.Goal` | Axioms |
|---|---|---|
| `valid` | Yes | None |
| `unfinished` | Yes | `sorryAx` |
| `forged` | Yes | `invented` |
| `Candidate.easy` | No | None |

`#print axioms` follows dependencies transitively. The body of `forged`
mentions neither `sorry` nor `invented`, but the command reaches
`invented` through `helper`. The same check exposes a dependency on
`SMT_VERIF` when a geometry proof uses an intermediate lemma relying on it.

[Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)
lists `propext`, `Classical.choice` and `Quot.sound` as ordinary axioms
to allow. Agree in advance on any domain-specific assumptions. `sorryAx`
and a custom axiom that can prove any proposition do not belong to that
list. Native evaluation adds version-dependent assumptions that need
a separate decision.

Fix the target type and the definitions it refers to in a trusted
environment, independently of the candidates. A theorem name or printed
header does not establish that two types denote the same proposition.

Building the file and inspecting dependencies can catch accidental
errors. Untrusted code needs stronger isolation because building a
project can execute tactics and metaprograms. Keep the build isolated
from secrets and check its output outside that process. The Lean
reference's `comparator` workflow checks the statement and its referenced
definitions against a trusted challenge, then checks an exported proof
term and its axiom dependencies outside the sandbox. The guarantees
depend on the tool versions, configured checkers, trusted imports,
and the isolation of the build. The code in this chapter was checked
with Lean 4.19.0 and Coq 8.20.1; those runs did not use `comparator`,
run the full LeanEuclid implementation, or train any models.

<span id="первоисточники-и-страницы-курса"></span>
<span id="источники"></span>

## Sources

- [Berkeley Advanced LLM Agents, Meeting 8: AlphaProof](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf), PDF pages 5–107. The chapter uses images from pages 30, 79, 84 and 91, and the embedded JPEGs from pages 99–100.
- [[05 Источники/Papers/AlphaProof|Hubert et al., AlphaProof, Nature]], Figure 4, the section on scaling inference compute, and Methods → Inference for TTRL.
- [Mathlib: Euler products](https://github.com/leanprover-community/mathlib4/blob/c44e0c8ee63ca166450922a373c7409c5d26b00b/Mathlib/NumberTheory/EulerProduct/Basic.lean), the full module introduction, the lemma for a finite set of primes and the bound on the tail. Apache 2.0; see the [[05 Источники/Papers/Mathlib Euler products|source note]] for the pinned version.
- [Berkeley Advanced LLM Agents, Meeting 9: autoformalization and theorem proving](https://rdi.berkeley.edu/adv-llm-agents/slides/mathverification.pdf), PDF pages 7–114. Images from pages 55, 58, 89 and 97. Published LeanEuclid Figures 1–2 replace the partially shaded or overlapped material on pages 84 and 105.
- [Berkeley Advanced LLM Agents, Meeting 10: advanced theorem proving](https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf), PDF pages 4–117. Images from pages 64, 66, 78, 82, 85 and 109–112. Lean-STaR Figures 4–5 replace the introductory slides on pages 39 and 42.
- [Berkeley Advanced LLM Agents, Meeting 11: abstraction and discovery](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf), PDF pages 14–44. Images from pages 16, 17, 24, 27, 30, 32–33 and 37–39.
- [AlphaZero](https://arxiv.org/abs/1712.01815v1), [AlphaProof and AlphaGeometry at the IMO](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/), [LeanDojo](https://arxiv.org/abs/2306.15626v1), [Autoformalization](https://arxiv.org/abs/2205.12615v1), [LIPS](https://arxiv.org/abs/2502.13834), [LeanEuclid](https://arxiv.org/abs/2405.17216v1), [Lean-STaR](https://arxiv.org/abs/2407.10040v1), [Draft–Sketch–Prove](https://arxiv.org/abs/2210.12283v1), [LeanHammer](https://arxiv.org/abs/2506.07477), [miniCTX](https://arxiv.org/abs/2408.03350v1), [ImProver](https://arxiv.org/abs/2410.04753v1) and [COPRA](https://arxiv.org/abs/2310.04353v1).
- [Theorem Proving in Lean 4](https://lean-lang.org/theorem_proving_in_lean4/), chapters 3 and 5; [Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
- [LeanEuclid implementation](https://github.com/loganrjmurphy/LeanEuclid/tree/33a72c0fa83914202532998a2e7b8771295c3cc7), MIT. The trust caveat refers to this commit; the benchmark description refers to version 1 of the paper.

Previous: [[en/00 Textbook/17 Tools and Agents/69 Coding, web, and computer-use agents|Coding, web, and computer-use agents]].
Next: [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Scientific discovery agents]].
