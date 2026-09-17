# Primary review: post-handoff content corrections

Date: 2026-09-15. These are additional corrections after the foundations and
advanced handoffs, not replacements for their original 102/55 finding records.
All three pages were read in full before editing and reread after the main edit;
the final small deltas were checked in their surrounding sections. Snapshot root:
`.superpowers/sdd/2026-09-15-close-editorial-findings/task-primary-review-before/`.
No existing authored English counterpart maps to these three pages.

## Derivative and gradient

Path: `00 Учебник/00 Математические и ML-основания/02 Производная и градиент.md`.

- Qualified steepest ascent by unit Euclidean directions and a nonzero gradient.
- Qualified strict descent by differentiability, nonzero gradient, and a
  sufficiently small positive step; the zero-gradient case does not move.
- Replaced unnecessary English nouns in explanatory Russian; clarified reverse
  mode, Jacobian orientation, and numerical versus training differentiation.
- All five original figures retained, with their original links and captions.
- Sources: https://d2l.ai/chapter_preliminaries/calculus.html and
  https://cs231n.github.io/optimization-2/; the existing MML/Google figures remain.
- Before SHA256: `56f43f60427f6fe476df0d71ccdd54b88500792fa578f28ddce46712b8b868b3`.
- After SHA256: `454937e034fa5ed30480d19fb8d5ae49350389505aadf5e75f2a7a2f9daaa69d`.

## Spectral clustering

Path: `00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian.md`.

- Explained union/mutual graph symmetrization, normalized cluster indicators,
  nonzero Rayleigh vectors, eigenvalue multiplicity, and out-of-sample limits.
- Corrected the eigenvector direction claim: negative eigenvalues reverse the
  direction; zero maps the vector to zero.
- Distinguished arbitrary eigenvector signs from instability of nearly
  degenerate eigenspaces; completed the Russian explanation throughout.
- One original figure retained. Eight explicit old anchors preserve incoming links.
- Source: https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf
  and the existing HSE seminar/homework source routes.
- Before SHA256: `4a8537f3b145fb4c71f375c6ccb59438411bfd0e561009dfafb6f790d01b1ea8`.
- After SHA256: `52af70baf5e70420404e3ee70366a638857ae7b3cc21c8b2b6567f08799cae82`.

## DPO

Path: `00 Учебник/12 Post-training и Alignment/05 DPO.md`.

- Corrected log-probabilities in the dataflow diagram and response-only causal
  shifting/masking assumptions in the implementation explanation.
- Distinguished saturation from the beta-dependent derivative magnitude;
  at zero margin the magnitude is beta/2, while a very negative margin does not
  yield a vanishing margin gradient.
- Removed an incorrect guarantee that concatenating chosen/rejected batches
  reduces padding. One forward call can still require more padding.
- Qualified the finite partition function/reference-support derivation,
  parameter-gradient claims, length normalization, and training-set accuracy.
- Rewrote the Berkeley/Tulu sections and practice in continuous Russian prose.
  Preserved the two renamed headings as explicit old anchors, all four wiki
  figure embeds, and every course source-unit identifier.
- Source: https://arxiv.org/abs/2305.18290 plus the existing Berkeley and Tulu
  source routes. No new external figure or long source quotation was added.
- Before SHA256: `9147d179a9edd437dbf164910c62cf270a828d433730863e11cb6b6dffcc57fb`.
- After SHA256: `2c01405ddf084d092b9838dc2b26efb43f8962fc1ee1b5b8190ea87c68f5a4ec`.

## Executed checks and limits

- `test_primary_review_examples.py`: three CPU test groups passed, including
  the actual current DPO `sequence_logp`, masks/padding/backward, 16 beta/margin
  cases, the badly ordered pair, and nonzero/zero gradient descent.
- `task-1-numerical-checks.cjs`: passed again; includes the spectral graph and
  its second eigenvalue `0.09501243788791114`.
- `task-1-page-code-checks.py`: current foundations examples passed on CPU.
- `git diff --check` passed for these three pages.
- These checks are not full DPO training, GPU validation, or browser rendering.
  Final emitted-link/math/image checks remain a separate acceptance step.
