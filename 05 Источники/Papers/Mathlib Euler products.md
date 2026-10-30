---
title: "Mathlib: Euler products"
type: source-note
status: canonical
last_verified: 2026-10-26
source_url: https://github.com/leanprover-community/mathlib4/blob/c44e0c8ee63ca166450922a373c7409c5d26b00b/Mathlib/NumberTheory/EulerProduct/Basic.lean
year: 2023
---

# Mathlib: Euler products

Michael Stoll's `Mathlib/NumberTheory/EulerProduct/Basic.lean` formalizes
Euler products for multiplicative arithmetic functions. This is a source-code
and documentation reference, not an LLM benchmark paper.

The pinned Mathlib v4.19.0 revision is
`c44e0c8ee63ca166450922a373c7409c5d26b00b`. The original file and its
Apache 2.0 license are preserved in
`05 Источники/Originals/formal-zeta-2026-10-26/`.
The textbook retains the complete English module introduction without
translation or stylistic editing.

The proof separates a finite product over primes and a sum over numbers
factored by those primes, then controls the omitted tail. `HasProd` states
convergence, while `tprod` variants express the resulting equality.
Hypotheses include multiplicativity, the value at zero, completeness and
norm-summability; the compact equality alone does not state all of them.

[[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты#finite-products-and-missing-assumptions|Worked example: AIME side conditions, a finite Euler product and convergence]].

The associated lecture is Thomas Hubert's
[AlphaProof: when reinforcement learning meets formal mathematics](https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=94),
Berkeley Advanced LLM Agents, 31 March 2025, physical pages 94–107.
The slides contain the mathematical illustration, not a saved Lean file
or a complete AlphaProof execution trace. The channel referenced on page 97
is [zetamath](https://www.youtube.com/@zetamath); the exact original video
for each embedded image has not been established from the PDF.

For the analytic conditions, see
[NIST DLMF §25.2](https://dlmf.nist.gov/25.2) and
[Terence Tao, 246B Notes 4, Theorem 8](https://terrytao.wordpress.com/2021/02/12/246b-notes-4-the-riemann-zeta-function-and-the-prime-number-theorem/).
The textbook does not reproduce Tao's full notes. Its finite rational
checks are not a proof of the infinite product or the explicit formula.
