---
title: "AlphaEvolve: A coding agent for scientific and algorithmic discovery"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2506.13131v1
year: 2025
---

# AlphaEvolve

Alexander Novikov, Ngân Vũ, Marvin Eisenberger et al.,
[arXiv:2506.13131v1](https://arxiv.org/html/2506.13131v1), 16 June 2025.
The preprint is CC BY-NC-ND 4.0. Bookvar uses the complete, unmodified
[Figure 2](https://arxiv.org/html/2506.13131v1#S2.F2).

Sections 2.1–2.6 describe editable blocks, program diffs, evaluation
budgets, a program database, and asynchronous workers. Section 3.1
separates an evolved search program from the matrix multiplication
tensor decomposition it finds. A decomposition with 48 terms is
an upper bound on the rank, not a proof of the minimum rank.

The [authors' results notebook, pinned revision](https://github.com/google-deepmind/alphaevolve_results/blob/4226acbf237ff9ad10ba7673a2af127a2d8a5971/mathematical_results.ipynb)
contains the three factor matrices for the 48-term construction.
The local Bookvar check reads those literals without executing the notebook
and checks all 4,096 coefficients exactly. This verifies the published
construction, not a rerun of AlphaEvolve's search or a speed comparison.

[[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery#alphaevolve-evaluation-cascade|Textbook: editable blocks, evaluation cascade and a checked mathematical witness]].
