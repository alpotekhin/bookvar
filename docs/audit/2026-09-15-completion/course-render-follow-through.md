# Primary follow-through: courses, rendering and navigation

Date: 2026-09-15. This report supplements the per-page editorial ledgers; it
does not add newly discovered rendering defects to the original 442 findings.

## Stanford additions

The primary editor fully read and reread the NTP and long-context chapters.
An independent reviewer subsequently read both complete pages and executed
`test_course_attention_mtp.py` (3 CPU test groups). The changes explain:

- Independent multi-token heads versus DeepSeek's sequential prediction
  modules; causal targets, head-specific valid-token counts, normalization,
  memory, and the distinction between training and speculative decoding.
- Local causal attention, receptive-field growth, ring-cache eviction,
  hybrid local/full layers, and persistent positional indices.
- Cross-layer KV sharing, its combination with GQA, memory arithmetic,
  independent query projections, and architectural/training constraints.

The original MTP Figure 1 and CLA Figure 1 are retained without cropping or
redrawing, with author, source, license and SHA records in the asset registry.
The Meta MTP figure is not misreported as reuse of a DeepSeek course diagram.
FP4 and evaluation additions, including authored English quantization parity,
are detailed in `course-fp4-eval.md`.

The durable Stanford overlay promotes exactly three semantic units: MTP,
cross-layer KV sharing and local attention. Regeneration reports 807 units:
557 integrated, 24 covered-existing, 21 contract-only, 151 source-only and
54 excluded. Visual records remain 159: 73 integrated, 9 covered-existing,
45 source-only and 32 excluded. FP4 and selected evaluation image rows retain
their exact partial status; prose coverage does not imply complete visual reuse.

## Berkeley additions and corrected source metadata

`course-berkeley-follow-through.md` records complete-page work on RU/EN video
models and RU scientific discovery. Root applied its three reviewed coverage
decisions for xGen, GenS and ESCHER and the five reasoned visual dispositions.
The new semantic totals are 140 integrated, 6 covered-existing, 38 source-only
and 28 excluded; the 161 visual records are unchanged (47/87/27).

GenS had an incorrect editorial title, “generalist foundation agents”. Root
read all deterministic page text and visually inspected all seven original
slides, physical pages 99–105, before correcting it to “Generative Frame
Sampler for long video understanding”. No PDF bytes, page range, semantic ID,
kind, visual kind or exclusion decision changed. Following the documented
manual workflow, the same single title was corrected in `semantic-review.json`
and `audit-contract.json`; the importer now pins the new contract SHA rather
than weakening its equality or completeness checks. All 31 focused Berkeley
regressions pass, including tamper/collapse rejection.

The published ESCHER asset is the byte-exact original SVG. The unused raster
preview and original PDF were moved out of the publication figure directory
into `raw/papers/escher/figure2-preview.png` and `2504.00185v1.pdf`; no binary
was discarded. The raw SVG copy is `raw/papers/escher/figure2-original.svg`.
These are final storage paths and supersede the agent report's handoff paths.
The registry names the primary publication, CC BY 4.0 and exact output hash.

Plan–Sequence–Learn remains in the source collection pending an appropriate
robotics chapter. The source-marked confidential page and unreviewed adjacent
material are not promoted by course-level permission. Both course catalogues
and the Berkeley hub distinguish integrated explanations from original-only
material and link directly to the relevant textbook chapters.

## Primary rendering and editorial repairs

- Replaced the cropped Jamba raster on the family page with the original
  vector figure. Inspected the full source and its rendered layout. Dark-theme
  review exposed disappearing black arrows on a transparent background;
  source figures now receive a white CSS background without modifying artwork.
- Fully read/reread the Triton chapter. Corrected the distinction between
  program instances and kernel launches, reduction accumulation, approximate
  GeLU, incomplete executable templates, and pervasive mixed-language prose.
  An independent full-page review caught a missing historical softmax heading
  anchor; root restored it and the reviewer verified the correction.
- The publication adapter now resolves preserved explicit heading aliases and
  copies approved linked course code artifacts even when linked from a textbook
  page. Focused red-to-green tests cover alias/fence/collision behavior and the
  local Lecture 6 Python download.
- Full-site output exposed a single-dollar closing delimiter in Scaling Laws.
  It swallowed subsequent headings and prose into mathematics. The chapter
  was fully reread and repaired; 37 figures, 22 headings, 15 explicit anchors
  and the original formula bodies are retained. See `scaling-render-fix.md`.
- Corrected the double-subscript syntax in the legacy quantization formula
  after reading the complete page; this is not a full new editorial acceptance
  of the entire legacy Concepts archive.
- Browser review exposed doubled chapter/file numbers in the main index.
  Explicit display aliases now omit file ordinals while preserving every target
  and the sidebar sequence. Six index tests pass, including the regression
  observed failing before this final alias repair.

Snapshots for primary course changes are under
`.superpowers/sdd/2026-09-15-close-editorial-findings/task-6-course-before/`;
Triton and legacy-formula snapshots are under `task-7-render-before/`.
Earlier root derivative/spectral/DPO work has its own report and snapshots in
`primary-content-review.md`. Final build, route and viewport results are
recorded separately in the final acceptance report, not inferred here.

No model training, GPU benchmark, production deployment, commit or push was
performed by this follow-through. English fallback pages remain explicitly
labelled; an untranslated Russian chapter is not counted as authored English.
