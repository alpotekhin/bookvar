# Semi-supervised learning: source and reader review

Scope: new chapter 41e, original figures, CPU examples, navigation between
accepted 41d and 42. No RL/English-body/translation/git publication changes.
The previous goal turn made content progress: 41d accepted; it is not reopened.

## Sources and source comparison

- Existing 514-line pretraining overview read fully; unchanged. It covers
  corpus/target origins, OLMo and mid-training, not pseudo-label mechanisms.
- Harvard H-V1-09 2031–2240 reread fully; 2241–2371 newly read including both
  cost/plot programs. Source cumulative 1–2371, next 2372. No claim to all QMD.
- FixMatch v2 §§1–4.1 and Tables 1–2 fully read; original repo README and
  fixmatch.py fully read, Apache license retained. Table 2's error percentages,
  RA/CTA distinction, folds and optimization protocol preserved.
- Full scikit-learn self-training/propagation guide and circles.py read, both
  circles images visually inspected; LabelSpreading is not relabelled as the
  hard-clamped harmonic example. Source threshold.py also fully read and its
  total-pool/training-fold offset caught; misleading count image not imported.
- no-ai-slop SKILL/eval reread completely. The saved 569-line reader was read
  fully again in contiguous ranges 1–210, 211–430 and 431–end after recovery.

| Mechanism | Source basis | Worked explanation |
| --- | --- | --- |
| Unlabelled-data assumptions | Guide, Harvard discussion, explicit counterexample | Two target functions on four corners agree with the known labels but disagree elsewhere |
| Self-training | Original guide paragraph retained in English under BSD | Threshold/argmax, iterative fit, provenance of pseudo-labels, confidence vs correctness |
| Consistency | FixMatch §§2.1–2.3 | Distribution targets, augmentation assumptions, collapse without labelled anchoring |
| FixMatch loss | Paper Eq.3–4; original code stop_gradient/mask/mean | Weak vs strong, shared parameters, accepted fraction and zero gate; explicit four-row loss/gradient |
| Training computation | Full model() source; paper §4 | One actual linear SGD step, batch/view cost, BN/interleave and EMA-eval boundary |
| Confirmation bias | Objective differentiated on wrong target | True-class probability decreases after an accepted wrong pseudo-label update |
| Graph inference | Guide and complete circles example; explicit Laplacian derivation | Energy derivative, clamped five-node chain, iterative/direct solution, edge change and singular unanchored component |
| Cost and protocol | Primary Table 2; assigned Harvard programs | Error vs accuracy, RA/CTA, labelled image reuse; 16 vs 200 label ratios, 8.1 assigned cost ratio, ten/eleven task crossover |

## Reuse

Two complete unchanged PNG originals and the original FixMatch diagram rendered
from its native PDF version at 400 dpi; source/page/license SHA and public
license copies in assets.json/central registry. The same diagram is explicitly
released in the Apache author repo and its PNG is archived/hashed as permission
evidence. Its native diagram and repository version were visually compared,
including all labels/branches/bars. That permission is not applied to other
paper figures or the whole arXiv paper. Final complete-diagram box gives
1850×700 with margins; no bitmap enlargement or redrawing. Circles sources
are BSD; original 400×400 resolution is not called a high-resolution rerender.
No synthetic replacement infographic, altered label or translated figure.

## Accepted revision and executable examples

Accepted reader SHA-256:
`7a35177afdf70c69f947345525297e3fd9bf7aa85933d646032f6d41c4a48ce6`.
At acceptance the saved file was hashed again and matched the executed revision.
The subsequent insertion of 41f changes only this chapter's next-navigation
edge. `simclr.test.ts` restores that edge and verifies the same accepted hash;
the explanation, programs and figures remain unchanged.
No unresolved explanatory gap was found within this chapter's stated scope.
This does not close general self-supervised representation learning.

All five actual Python blocks executed on CPU with NumPy 2.3.5 / Torch 2.14.1.
Independent checks cover 36 masked-gradient scenarios, 64 weighted chains,
the singular unanchored component, the changed last-edge case and one autograd
check of the graph-energy derivative. A wrong accepted target reduced true-class
probability from 0.017986209962091555 to 0.0176712637680298 as explained.
The checker log records the exact reader SHA. Historical TensorFlow training,
image-augmentation benchmarking and reproduction of the CIFAR result were not
executed or claimed.

## Publication and visual evidence

- Full unit suite: 613 tests / 94 files passed in 48.91 s. After the final
  cost-accounting sentence changed, all 15 affected tests / four files passed.
- Accepted build: 1,977 pages in 24.62 s; output suite 161 tests passed in
  4.47 s; built links: zero broken routes, fragments or files. Coverage passed
  for 210 structural rows / 59 destinations; this is not teaching completeness.
  Logs are `/var/tmp/bookvar-semi-supervised-{examples,scoped,build,output,links,coverage}-accepted-2026-10-28.log`
  and `/var/tmp/bookvar-semi-supervised-unit-final-2026-10-28.log`.
- The previously running output/link handles 86571/69729 are now absent.
  Their final logs contain the passing summaries; neither was restarted just
  because its earlier observation was lost.
- At desktop 1280 px the body is 1280/1280. The native 1850×700 FixMatch
  diagram displays at approximately 632×239 without changing aspect ratio;
  each circles original displays at its actual 400×400 resolution. The saved
  desktop screenshot was opened and inspected after recovery.
- At mobile 360 px the body is 360/360; all nine display equations fit 328/328,
  with no KaTeX error or overwide inline equation. All three figures load.
  At 2.5× the diagram viewer reaches both horizontal ends, 0/516 and 516/516,
  with the full 611/611 height and the caption end accessible. Both circles
  were opened with their complete axes, legend, titles and borders.
- The widest loss table was actually scrolled to 282/282 and back to 0/282:
  both the final loss contribution and first columns remain accessible without
  page overflow. Footer clicks passed 41e→41d→41e→42→41e.
- The `/en/` route was opened and explicitly reports the unavailable-language
  fallback. It contains the same nine rendered equations, not an authored
  English translation. Only navigation edges in accepted neighbours changed;
  tests restore their prior accepted hashes.

Screenshots: `screenshots/2026-10-28-semi-supervised-desktop.png`,
`-circles-desktop.png`, `-fixmatch-mobile-left.png`, `-fixmatch-mobile-right.png`,
`-circles-raw-mobile.png`, `-circles-spread-mobile.png`,
`-loss-table-mobile-right.png` (all use the same date/topic prefix).
No stage, commit, push, deployment, delegation, live infrastructure mutation or
English-body editing was performed. The overall goal remains active.

## Next source boundary

After accepting this chapter, Harvard 2372–2600 was read fully, including the
synthetic-gap TikZ and the complete KWS calculation. Cumulative source reading
is now 1–2600; next unread line is 2601. Those later mechanisms are not counted
as integrated merely because their source was read. Unsupported general claims
about smaller models overfitting more and universal synthetic/real ratios must
not be imported as established facts.

The existing 930-line embeddings chapter was read in full and left unchanged.
It already covers contrastive matrix loss, SimCSE, hard negatives and evaluation;
a generic second explanation would duplicate it. SimCLR's two-view objective,
MoCo's queue/momentum update and masked-image reconstruction still need explicit
thematic reconciliation. The VLM-training chapter has not yet been fully read
in this pass. The independent native figure inventory and Harvard 2601–4741
remain open; H-V1-09/H-F1-09 therefore remain re-review.
