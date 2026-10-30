# MIG: original figures, profiles and workload-dependent placement

Canonical file: `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md`.
Final: 2240 lines, SHA-256
`ac9a516bc886263afb6ebd7303dbd2d12be6533d167842b1fa6c713d26a370ee`.
Route: `/textbook/deployment/mlops/#what-a-mig-partition-separates`.

This is the previously open MIG visual/mechanism addition. Removing only
the bounded `mlops mig-detail 2026-10-26` section restores the exact
accepted 2080-line revision, SHA-256
`6b8633c35ce4c25317ca731feefc9e4811c7ee1caaf7e252624086f7626f6c20`.
The complete preceding chapter was read in the immediately preceding
FinOps pass; its hash was freshly checked before this addition. The new
section and its surrounding memory-budget/geometry/tenant explanation
were reread in full. FinOps, feature history and accepted RL were not
rewritten. This is not a new full-course completion claim.

## Source comparison and explanation

MISO, arXiv:2207.11428v3 (6 October 2022), by Baolin Li, Tirthak Patel,
Siddharth Samsi, Vijay Gadepally and Devesh Tiwari. The saved arXiv page
explicitly links CC BY 4.0. The source archive and complete paper are
preserved, with selected native vector figures and original TeX.

Read: full Background, Motivation and Methodology; figure captions,
profile/workload tables, STP/JCT/makespan definitions and the distinction
between the real testbed and simulation. The first real-system evaluation
section was inspected for context, not newly transferred as headline gains.
Complete paper pages 2 and 3 were visually inspected.

NVIDIA MIG Concepts was read through terminology, partitioning, the
A100-40GB table and placement/R510 notes. Current MPS Dynamic Execution
Resource Provisioning and Static SM Partitioning descriptions were checked.
Both HTML snapshots are retained. No long NVIDIA text or illustration
was newly reproduced; source links accompany the explanatory facts.

| Mechanism | Added explanation | Checked example / limit |
|---|---|---|
| MPS vs MIG | Full MISO Figure 1 and reading of both rows | Dynamic active-thread caps do not reserve disjoint physical SMs; static SM partitioning is a separate mode |
| GI vs CI | GI memory boundary versus compute subdivision inside one GI | Four CIs inside `4g.20gb` still share the parent nominal 20 GB pool, not four 20 GB pools |
| Profile proportions | Five selected A100-40GB rows, not a complete hardware catalogue | Two `3g.20gb` GIs use six compute slices but all eight memory slices, so the remaining compute slice cannot support another `1g.5gb` GI |
| Placement/version | Necessary slice budgets versus actual allowed placement | MISO's old 4g+3g prohibition is not carried forward: NVIDIA says it was removed from R510 onward |
| Workload-dependent assignment | Full Figure 3 and an exact English source excerpt | 4g/2g/1g beats the shown MPS configurations in this mix; 2g/2g/3g does not beat the second MPS bar |
| Relative progress | Definition and worked STP calculation | Rates 100/20/40 vs 80/10/20 give 0.8+0.5+0.5=1.8; STP=2 can coexist with a third job making no progress |

The source's seven schematic memory boxes are not promoted to a count of
the A100's physical memory slices. Its simplified MPS compute bands are
not used as a physical SM-reservation guarantee. NVIDIA's own introductory
`4g.5gb` typo is not copied; table/caption/subsequent example say `4g.20gb`.
Figure 3 is a 2022 training-workload comparison, not LLM serving p99 or a
test of this platform. The exact CNN/MLP architectures and batch sizes are
not guessed from the broader workload table.

## Original illustrations and provenance

Both PDFs below are byte-identical members of the authors' source archive,
rendered in full by pdftoppm at 400 dpi. No crop, redraw, relabeling or
changed data. The first native PDF names Tirthak Patel as author and is
vector artwork; pdfimages lists no embedded raster substitute.

| Figure | Native member | PNG | SHA-256 |
|---|---|---|---|
| MISO Figure 1 | `figures/bkgd1_mps_vs_mig.pdf` | 2917×750 | `cb9c6ff13f784acdc4d3c6ccb9e2caf1d7a0bad7e2185745148ad035fa9cc18b` |
| MISO Figure 3 | `figures/motiv2_MPSvsMIG.pdf` | 2612×990 | `8f1314e7f3797c45718b34c962b93400d419053e2f6a2e38a32633d20b3be3e1` |

All full-paper/archive/native-member/license/PNG identities are recorded
in `curated/mig-2026-10-26/assets.json` and the central asset registry.
The source archive is 5.4 MB uncompressed; only selected members were
extracted. No repeated large repository clone or full-page render set.
Three previously rejected low-resolution/UI-only NVIDIA images remain
unpublished. Source captions and original English are retained in context.

no-ai-slop and eval.md were read in full. The added prose explains the
diagram, a slice-budget example and the measured comparison's meaning;
it does not replace mechanisms with definitions or claim universal gains.
The exact source excerpt is protected from stylistic rewriting.

## Executed evidence

- `check_mig_visual_examples.py --render`: actual displayed Python fence,
  300 common-baseline progress cases, invalid inputs, five source table
  rows, exact source excerpt and two native archive members. Both PNGs
  regenerated byte-identically from the full native PDFs.
- All 15 Python fences in the chapter and the existing 1400 arithmetic
  cases passed on the final content. Other preservation/source checks
  for FinOps, feature history and platform quotas passed. The unchanged
  SQL content keeps the preceding DuckDB evidence; it was not claimed as
  a new GPU or service execution.
- Full unit suite: 506 tests / 60 files passed before the final source
  excerpt. The five affected suites were repeated on the final revision:
  19 tests passed, including publication policy and source-fidelity checks.
- Final Astro build: 1967 pages, 22.48 s. After that build completed,
  135 output tests passed; links: zero broken. Structural source coverage
  passed 210 rows / 56 destinations; this does not establish semantic
  completion of the courses.
- Scoped `git diff --check` passed. No stage/commit/push/deploy or live
  GPU/Kubernetes mutation.

Logs: `/var/tmp/bookvar-mig-*-2026-10-26.log`; final build/output/links/
coverage/affected/example logs use the explicit `final` suffix.

## Browser inspection

1280×900: both diagrams and all 16 chapter images load. The complete
figures fit the article column without cropping. The computed image filter
is `none`, opacity 1 and normal blend. No site-side visual transformation
was added to the originals.

360×800: inline figures are an overview, not a claim that every small
label is readable without magnification. At 3× the viewer exposes both
sides: SM/cache labels and App 1/2/3 legend for Figure 1; configuration
labels and the full axis/legend for Figure 3. All labels stay in the source
image. The selected profile table has 576 px content inside a 328 px
scroller; keyboard input reaches 248/248 px. The final header bounds
are x=216..344 inside the x=16..344 reading column.

All nine display formulas render without KaTeX errors or local overflow;
the document is 360 px wide at a 360 px viewport. The EN route explicitly
shows fallback and includes both figures, with no math/page overflow.
No EN translation was authored. The RU tab remains on the MIG mechanism
heading; viewport override reset.

Screenshots in `docs/audit/screenshots/` use prefix `2026-10-26-mig-`:
`mechanism-desktop`, `comparison-desktop`, `mechanism-mobile-left/right`,
`comparison-mobile-left/right`, `profiles-mobile-left/right`.

## Remaining course work

The MIG visual gap is closed on this revision. Do not enqueue it again
as missing, or restart the accepted RL/FinOps passes. The next independent
unit is H-L2-12 (`labs/vol2/lab_12_ops_scale.py`, currently preview-audited)
and H-SL2-12 (`slides/vol2/12_ops_scale/12_ops_scale.tex`). H-L2-11 is Edge,
not the scaled MLOps lab. Full Harvard chapter/lab/slides transfer and
practice 17's unexecuted container/RPC/scrape/load path remain open.
