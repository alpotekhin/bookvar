# Harvard Neural Architectures: selection, constraints and source review

The new worked practice integrates Lab 06 with the book's decision framework.
It does not close the full Network Architectures chapter, the native image
directory or the whole course. The preceding RL status reply was no progress
toward that outstanding work; accepted PPO/DPO/RLVR/GRPO remain unchanged.

## Delivered revision

- Canonical practice: `06 Практика/18a Сравнить архитектуры по ресурсным ограничениям.md`,
  370 lines, SHA-256
  `d9871982d0377913404e35754d102f03a2b87bf3c245527612e6062fdabd6db6`.
- Reader route: `/practice/18a-architecture-selection/`.
- Source execution: `publishing/tools/check_architecture_lab.py`, SHA-256
  `5c62a282cc2323448367fc1d10398243b2d6b8795bd34bc707d7ff50e0bdfc47`.
- Original plot export: `publishing/tools/render_architecture_source_figure.py`,
  SHA-256 `e61e32079317ba429a9ffa9eb178967d1523ebe324766b454eb987b91c430f0a`.
- Public result: `site/public/examples/architecture-lab-source-checks-2026-10-26.json`.
  The final three unit tests bind it to the current page, checker and originals.

The practice appears after the Harvard labs guide in navigation and both
practice indexes. The ML Systems introductory chapter received only an
additional further-reading link and updated date; its explanation was not
replaced. Its resulting SHA-256 is
`c958fbd61b171e67163951269d76375e26376e7a54c69d76b4e3ad24c7b6676d`.
No English translation body, original course page or immutable raw file was
edited. English reader routes use the explicit fallback notice.

## Sources actually read

The source archive is `05 Источники/Originals/nn-architectures-2026-10-26/`.
Its README gives exact repository paths, licenses and reproduction commands.
The CPU result records each original's hash.

Two revisions are kept separate:

- book chapter: `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`;
- lab, deck, helper, variants, style and helper tests:
  `45ecc8d82fcae70c149cdce550d3b3d3411df913`.

The complete 1721-line lab, 1896-line deck including notes/backup,
413-line helper and original helper test were read. Variant review covered
the factory and all four Lab 06 records, not unrelated labs. Style review
covered the palette and `apply_plotly_theme`, not unrelated Marimo widgets.

At the initial practice acceptance, book review covered the opening 1–145
and selected 4560–4993 material, including the decision framework, native
TikZ figure and wildlife sizing example. The subsequent continuation read
architectural principles and the full MLP/CNN explanations through 1990.
The zebra's long numeric vector paths were inspected in the complete original
SVG, not read coordinate-by-coordinate. The current MLP and CNN chapters were
read in full; the missing rotation-equivariance explanation was then integrated
into CNN with original escnn visuals. Its separate acceptance is recorded in
`2026-10-26-cnn-equivariance.md`. Sequential book reading resumes at line 1991,
RNNs. The full 5256-line chapter and native figure directory remain open;
H-V1-06/H-F1-06 are not closed by these partial integrations.

## Explanation and examples

The complete final 370-line practice was read end-to-end after the last
opening-paragraph edit. The explanation keeps five conditions rather than
reducing architecture choice to a quality ranking.

| Mechanism | Source and worked reasoning | Boundary retained |
| --- | --- | --- |
| Candidate selection and constraint iteration | Complete native book figure and verbatim four-step procedure; the mobile path checks memory, speed and quality again after capacity changes | Family labels suggest candidates, not exclusive permissible architectures |
| Five-condition feasibility | All three mobile candidates and assigned 220 MB / 25 ms / 4.5 W / 86% / 85% limits; each inequality is evaluated | Scenario inputs, not measured phone specifications or model performance |
| Memory accounting | 3.5 M and 18 M FP32 parameters give 14 MB and 72 MB; partial weight-plus-activation totals are 30 MB and 142 MB | The helper checks activations only; totals omit runtime/workspaces and are not measured peaks |
| Operations and bytes | 0.32 GMAC / 16 MB is separated from HBM traffic; 32 MB versus 320 MB transferred gives 20 versus 2 FLOP/byte | No source measurement of traffic; family name does not fix arithmetic intensity |
| Input scaling | Complete unchanged `_scaled`; doubling resolution changes operations, activations, latency and power by different exponents | Assigned exponents do not derive attention complexity or thermal behaviour |
| Failure scale | Analytic power boundary 372.314683 px versus last passing sample 371.2 and first failing sample 380.8 | A sampled first failure can also be a fixed quality/kernel violation, not a capacity crossing |
| Different decision rules | Quality/kernel score selects efficient transformer; feasible resource headroom and default bias index select CNN | Finite infeasibility penalties are not hard filtering; assumed bias factors are not learning curves |
| Four tracks | All twelve candidates evaluated, with mobile, wearable, vehicle and cloud failures explained | Quality units are task-specific; scalar inputs establish neither p99 nor safety |

The final style check removed a generic opening in favour of the actual
mobile numbers. It preserved the source quote, full function, reasoning,
caveats, figures and terminology. The checklist was not used as evidence of
source completeness or teaching quality.

## Executed source and checks

The checker executes the reviewed standard-library helper without its
type-only relative import and eight named pure local lab functions. It
extracts only the four literal Lab 06 variant records. Hardware/model metadata
is explicitly unresolved because those fields do not drive this arithmetic.
No Marimo bootstrap, UI, ledger, ML model or hardware registry was executed.

Observed results:

- twelve source candidates and four default tracks;
- 300 independently derived input scales, checking all three candidates at
  each scale (900 feasibility comparisons), plus the original 36-point curves;
- both printed Python blocks executed; copied `_scaled` is AST-identical to
  the source in the checking interpreter;
- four-step prose quote matches the pinned book;
- eight original plot statements, original theme and source trace values
  verified. Plot source-text SHA-256:
  `7e097ec3195471135814668c6922200e881d85900343569f113c8ae5e40f5461`.

Logs under `/var/tmp/`:

- `bookvar-architecture-assets.log`: 148 asset/provenance tests passed;
- `bookvar-architecture-all-unit.log`: 62 files / 511 unit tests passed on
  the complete first revision; only the opening paragraph changed afterward;
- `bookvar-architecture-final-unit.log`: all three affected tests passed
  after the final page/public-result update;
- `bookvar-architecture-build-final.log`: final 1969-page Astro build,
  22.41 seconds, including sitemap and completed search index;
- `bookvar-architecture-output.log`: 135 built-output tests passed;
- `bookvar-architecture-links.log`: zero broken internal routes, fragments
  or files;
- `bookvar-architecture-coverage.log`: structural coverage passed,
  210 matrix rows / 58 destinations. This does not prove semantic coverage.

Scoped `git diff --check` passed. No stage, commit, push, deployment or live
infrastructure mutation was performed.

## Original figures and browser inspection

Two figures have exact local and central provenance records:

1. The complete original architecture-selection SVG, copied byte-for-byte
   from the imported book asset. SHA-256:
   `d537844e880c3cec792d3ccc273f5aaad782a289c4c78edb1076a54cab3dd30a`.
   Its historical image URL now returns 404; the unchanged import manifest,
   original source-page archive and pinned chapter TikZ identify the local
   original. A clipped Quick Look thumbnail was not mistaken for the SVG:
   the browser viewer displays the full bottom state.
2. The lab's normalized-budget chart, rendered from its unchanged eight
   plotting statements and original palette/theme. Only the export canvas
   is set to 1000×480. SVG SHA-256:
   `f6b2401aae6bff5584bda2d3f8f221c19e0e20ea3f9ecf32cfe840a1db5f9835`.
   Plotly 6.3.0 / Kaleido 1.1.0 and existing Chrome produced the static
   export; saved trace JSON and source-text checksums establish derivation.
   Internal generated SVG IDs are not claimed reproducible across runs.

Both use the source's CC BY-NC-SA 4.0 license. No substitute diagram was
drawn. Captions identify authors, exact source and the scenario assumptions.

The final built page was inspected in the actual browser at 1280×900 and
360×800. At desktop width both full originals are readable in the viewer.
At mobile width the body is 360/360 pixels, both images load, all three
display formulas fit the 328-pixel text column and there are no KaTeX errors.
The eight-column table reaches 453.5 pixels of its rounded 454-pixel scroll
range; the last cell ends at 344.20 in the 16–344 content column.

At 3×, both horizontal ends of each original were inspected. The selection
flow's lower deployment/selected states and change-architecture branches are
present and readable. The budget chart's y-axis, CNN values, ViT values and
complete legend are accessible without a crop. Inline mobile figures are
overviews; the viewer is needed for their small labels.

The labs guide → new practice → original book links were followed with
navigation waits. A failed heading query used an unsupported `level` option;
the resulting source URL and exact `Network Architectures` heading were
then checked directly. No failed query was counted as navigation proof.
The English route displays “This content is not available in your language
yet”; the same three formulas fit and both figures load. No translation
was authored. Temporary viewport overrides were reset.

Screenshots:

- `screenshots/2026-10-26-architecture-selection-desktop.png`;
- `screenshots/2026-10-26-architecture-budgets-desktop.png`;
- `screenshots/2026-10-26-architecture-boundary-mobile.png`;
- `screenshots/2026-10-26-architecture-table-mobile-right.png`;
- `screenshots/2026-10-26-architecture-selection-mobile-left.png`;
- `screenshots/2026-10-26-architecture-selection-mobile-right.png`;
- `screenshots/2026-10-26-architecture-budgets-mobile-right.png`.

## Source contradictions not adopted

The deck was read rather than accepted as an infallible numerical reference.
Its fixed family intensity/utilization tables and unsupported anecdotes were
not transferred as measured facts. Concrete findings include:

- A 40 FLOP/byte example lies below a stated 156 FLOP/byte ridge. That pair
  cannot establish compute-bound execution.
- 1,048,576 / 9,408 is about 111.455, not 47. The deck mixes two different
  convolution examples.
- The displayed `12 L H² S` formula with the displayed GPT-2/GPT-3 shapes
  gives about 86.97 G / 356.24 T and a ratio of 4096, not the stated pair
  of costs and ratio. Counting convention and attention terms also require
  qualification.
- Five images per 25 ms gives 200 images/s, not 1000. Concurrency does not
  by itself prove a throughput or latency gain, and shared model weights
  do not multiply by inference batch size.
- FP32 to INT8 raw weight storage changes by fourfold, not twofold.
- The backup KV example is 1 MiB per stated head and 1 GiB in total under
  its dimensions, not 2 MB and 2 GB. Numerical format and unit conventions
  are explicit in the independent calculation.
- Materialized attention-score storage is quadratic; that does not prove
  every attention implementation requires that resident matrix. Recurrent
  inference state and training-time unrolled activations are different.
- Per-family fixed GPU utilization, universal hardware capacity limits and
  an untraceable cloud-provider deployment story lack a reproducible result.

The source archive remains unchanged. H-SL1-06 is a reviewed cross-link with
these findings, not a claim that every slide was integrated as fact.

## Remaining work

Continue Network Architectures at line 1991 in order, compare its
mechanisms with accepted MLP/CNN/RNN/Transformer/recommender/system chapters,
and integrate genuinely missing explanations and useful original figures.
Do not reopen completed RL or substitute the new practice for full chapter
integration. H-F1-06 still requires the native figure-directory review.
The full course and whole-book objective remain active. Source/rights findings
from earlier units remain open; this acceptance does not close them.
