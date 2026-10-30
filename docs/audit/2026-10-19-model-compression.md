# Chapter 57b: model compression — editorial and visual review

Date: 2026-10-19

## Scope and delivered revision

Canonical chapter: `00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank.md`

SHA-256: `e818fccc317a2025b020f453a2e578e43ea1a2ba39ae893a86c6ee0a89bed5d3`

Local route: <http://127.0.0.1:4330/textbook/inference/model-compression-pruning-distillation-low-rank/>

The entire old chapter and the complete replacement were read. The final editorial pass retained mechanisms, derivations, counterexamples, source excerpts, and illustrations. It did not compress the subject into definitions. No English chapter, commit, push, or deployment was performed. This audit covers this chapter, not completion of the book or the full source courses.

## Source reading and factual corrections

- Harvard ML Systems at commit `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`: selected sections on target structures, unstructured and structured pruning (lines 650–1106; graphic code handled separately), knowledge distillation (1829–2033), low-rank factorization (2087–2141), and NAS formulation/search (2567–2615). One full English workflow paragraph is reproduced verbatim under CC BY-NC-SA 4.0.
- TinyTorch Module 16 at the same commit: selected magnitude-pruning, SVD, and distillation introductions/implementation. The upstream module was not executed; the chapter uses its own checked NumPy demonstrations. A problematic illustrative temperature table in the upstream educational material was not copied.
- Minitron arXiv:2407.14679v2: sections 2, 3, 4.1, 4.3, 6 and Appendix A.4. The investigated combined loss is distinguished from the final width-pruned models' selected logit-only KL recipe. Table 5 dimensions and the full-family training-cost calculation are retained in context. Recommendations 5–7 remain in their original English.
- Wanda arXiv:2306.11695v1, section 3: score orientation, per-output selection, activation collection after preceding pruning, and absence of surviving-weight updates. The activation example includes a calibration-distribution counterexample.
- Hinton, Vinyals, Dean arXiv:1503.02531v1, pages 1–3: soft targets, temperature, hard/soft loss combination, and temperature-squared scaling. No long excerpt or figure from that paper is reproduced.
- Local pinned CS336 lecture 10, lines 490–502: original pruning section and the two existing Minitron figure references.
- Source snapshots and the two exact excerpt selectors/hashes are under `05 Источники/Originals/model-compression-2026-10-19/`. Paper source PDF/archive provenance is recorded separately in the figure manifest.

## Explanation and numerical acceptance

Six displayed Python blocks execute from the actual Markdown, in reading order:

1. Magnitude masking, exact count, stable ties, unchanged dense shape.
2. Wanda: activation-weighted score and reconstruction errors 25 versus 0.01.
3. Physical MLP neuron removal: both adjacent matrices and the hidden bias; 22 parameters become 12. Equality is with the masked model, not the original unpruned model.
4. Rank-one SVD: singular values 4 and 2; squared Frobenius error 4; an input direction loses its entire output despite retaining 80% of squared singular-value energy.
5. Distillation: teacher/student distributions, forward KL, analytic gradient, and one improving update.
6. Explicitly hypothetical architecture selection, intermediate fusion traffic of 1 MiB, and the paper's family-cost ratio of about 1.803.

`publishing/tools/check_model_compression_examples.py` passes:

- 6 displayed blocks;
- 300 random Wanda score/reconstruction trials;
- 400 masked-versus-physically-compact MLP trials;
- 400 SVD error/shape trials;
- 150 finite-difference gradient trials, including hard-label CE at temperature 1;
- 9 source-file hashes and 2 exact English excerpts;
- 8 embedded figures, 5 byte-unchanged Harvard SVGs, and 1 byte-identical rerender of the new source PDF;
- legacy fragment anchors and all 3 previous source-unit IDs retained.

These are CPU algebra/provenance checks. No GPU kernel performance, compressed-LLM quality, or serving speedup was measured.

## Original illustrations and actual UI findings

All seven previous illustrations remain; Minitron Figure 4 was added from the author-supplied standalone vector PDF. Its full 3200 × 906 Poppler render is not cropped, redrawn, or translated. CC BY 4.0 evidence, the source PDF, the versioned archive member, hashes, and dimensions are recorded in `00 Учебник/Assets/Figures/curated/model-compression-review-2026-10/assets.json` and the shared registry.

The visual review found and corrected the following defects:

- Five Harvard SVGs had print-sized intrinsic dimensions and displayed as small images. Five narrowly scoped CSS selectors now use a responsive 36rem width and white background. Original SVG bytes remain unchanged.
- The existing Harvard animal-probability chart was incorrectly captioned as a teacher/student training flow. It now identifies the actual 85%/10%/5% soft-target chart and explicitly separates those illustrative values from the chapter's logit example.
- The Harvard low-rank figure's own symbols are mapped to the notation used in the prose.
- A 369 px-wide MLP equation exceeded the 343 px mobile content column. It is now split into two aligned lines.

Actual browser checks on the final built revision:

- Desktop 1365 × 900: all eight source figures were inspected in context; all loaded. The five Harvard SVGs display at 576 px without clipping. The new vector-source PNG and both existing Minitron figures retain their aspect ratios.
- Mobile 390 × 844: content width 343 px; document width 375 px, with no page-level horizontal overflow. All eight images fit the content column without cropping.
- 17 display formulas, zero KaTeX errors, zero display-formula overflows after the MLP equation fix.
- Opening the new image follows the actual asset link, displays the 3200 × 906 original, and supports native click-to-zoom. Wide multi-part diagrams require enlargement for small labels on a phone; the inline view is an overview, not a claim that every label is readable at 343 px.
- Desktop screenshot: /tmp/bookvar-model-compression-review-desktop-2026-10-19.png
- Mobile screenshot: /tmp/bookvar-model-compression-review-mobile-2026-10-19.png

## Publication checks

The final content build completed successfully: 1961 pages. After that build:

- built-link checker: zero broken internal routes, fragments, or files;
- site-output tests: 12 passed;
- asset-registry tests: 37 passed (registry and assets unchanged after this run);
- targeted `git diff --check`: clean.

The no-ai-slop pass checked for empty framing, unsupported contrasts, mixed-language filler, and definitions without mechanisms. Technical contrasts, equations, limitations, source quotations, and useful summaries were preserved. This is an editorial review, not a claim of an automated style/AI-authorship score.

## Remaining work

The page is available only in the local demo; publishing is outside this turn. The next chapter in the editorial sequence is 58, speculative decoding. It has not been audited or rewritten as part of this change.

