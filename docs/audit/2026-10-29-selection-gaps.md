# Data Selection: image duplicates and cold embedding IDs

29 October 2026. Worktree editorial-route-alignment. No EN body, Git staging,
commit, push, deployment or model training is part of this change.

## Delivered reader explanations

| Destination | Addition | Acceptance scope |
| --- | --- | --- |
| 41a, image-deduplication | 267 lines, 3 executable examples, complete SemDeDup figure | File versus decoded-pixel matching; transform and annotation policy; dHash and pHash with colour collisions; embedding/cluster/threshold semantics; all-earlier versus retained-representative deletion; false-removal and leakage checks. |
| Ranking and neural recommender models, cold-embedding-ids | 288 lines, 4 executable examples, complete TorchRec MPZCH figure | Events versus vocabulary versus capacity; retries versus real repeated events; OOV cutoff; weights/moments remapping and actual allocation; fixed capacity versus eviction; ownership reset; chronological tail-quality evaluation. |

Both complete chapters were read before editing and reread after saving:
1207 and 1593 lines respectively. Existing explanations, code, figures and
links were preserved byte-for-byte outside the two bounded additions and
their dates. Exact before/after/block hashes are in
`2026-10-29-selection-gaps-additions.json`. Preservation is tested separately
from explanation quality.

## Source comparison and editorial review

The packet README lists exact files and reading scope. ImageHash README,
SemDeDup implementation, TorchRec ZCH README/tutorial and RFC were compared
with the relevant Harvard span, not used as titles for short summaries.

The explanation follows a decision through its representation, computation,
counterexample and evaluation. Original quoted English is unchanged. Worked
examples are explicitly identified as textbook fixtures, not source results.
The no-ai-slop checklist was applied to the complete final chapters. Technical
contrasts, derivations, caveats and necessary summaries were retained.

Source corrections are explicit:

1. Increasing epsilon lowers SemDeDup's strict similarity threshold for fixed
   vectors, clusters and order. The README's reversed direction is not repeated.
2. SemDeDup can discard C for resembling discarded B even if C does not
   resemble survivor A above threshold. The three-vector example distinguishes
   this from comparing only retained representatives.
3. Twenty percent fewer event records need not remove any distinct ID.
   Allocated memory changes only when the table and associated state change.
   The source's unsupported percentage transfer is not used.
4. TorchRec tutorial prose says MPZCH while its construction cell uses MCH.
   The diagram and source excerpts are attributed to their actual origins.

## Execution evidence

`publishing/tools/check_selection_gap_examples.py` executes all seven new
reader programs. It compares both image-hash functions with the pinned source
functions on 30 random RGB inputs each, executes the complete pinned SemDeDup
similarity method on assigned CPU vectors, checks the strict boundary and
epsilon direction, 30 event permutations, conflicting retries, six ID-order
permutations, all three state arrays and a weights-only remapping failure.

The existing ranking checker executed all 23 chapter programs, with the
existing finite-difference, FM, protocol and DLRM gradient checks retained.
The existing text-dedup checker keeps its original derivation and collision
checks; its block count now includes the three separately executed image
programs.

No CLIP inference, Criteo training, TorchRec execution, GPU benchmark,
recommendation-quality measurement or SemDeDup accuracy reproduction is
claimed. The paper plot remains an attributed source result.

## Figures and rendered acceptance

Both new figures are complete original PNGs, not redrawn, cropped, relabelled
or recompressed. Source/reader byte equality, dimensions, license notices and
central-registry equality are covered by selection-gaps.test.ts.

SemDeDup is 2330×912, CC BY-NC 4.0. TorchRec is 7351×2336, BSD-3-Clause.
The dense TorchRec implementation diagram requires enlargement to read small
function names; the adjacent prose walks through the input, identity table,
metadata, operator and returned embedding index.

Rendered checks used the local browser at 1280×720 and 390×844. Both figures
load in the page and viewer. SemDeDup is readable enlarged; TorchRec's small
function labels require zoom (1.5× desktop, 4× mobile checked). Horizontal
panning reaches the right-hand operators; this is viewer navigation, not a
cropped source file. The memory table fits the narrow page. The deduplication
page measured scrollWidth=clientWidth=1280 on desktop and 390 on mobile;
the ranking page measured 390 on mobile. Neither page had broken images or
KaTeX error elements in the checked DOM.

Screenshots are saved locally in /var/tmp with the prefix
`bookvar-selection-gaps-` and suffix `-2026-10-29.png`:
`reader-desktop`, `semdedup-desktop`, `semdedup-mobile`,
`torchrec-desktop`, `torchrec-mobile`, and `allocation-mobile`.
The original viewport was restored. One request during static regeneration
returned the temporary 404 page; after completion, reload returned the reader,
and its figures and links were checked again.

## Project checks

- Unit suite: 108 files, 661 tests passed. The first run detected the new
  asset records using the wrong provenance category; the records now use the
  existing pinned-repository convention and exact source-image URLs. No
  validator was weakened.
- Rendered-output suite: 165 tests passed, including five deduplication
  figures, five Python programs plus its JSON record, seven ranking figures
  and all 23 ranking programs.
- Build: 1989 pages generated successfully.
- Internal link check: zero broken routes, fragments or files.
- Source coverage check passed; its inventory count is not a new claim that
  every source or chapter was read in this turn.
- Scoped git diff --check passed. Nothing was staged or published.

Logs are in /var/tmp under the same dated prefix: `examples`,
`text-dedup`, `ranking`, `unit-final`, `output`, `links`,
`coverage`, and `build-final`. The ranking execution preceded only the
one-word prose correction from “event set” to “event log”; all 23 code blocks
and inputs were unchanged. The new-example checker and final unit suite bind
the corrected current bytes. Source-version and execution limitations above
still apply.

## Data Selection map reconciliation

The original 31 contiguous ranges still cover all 4741 pinned source lines,
with the same 19 destinations. The three recorded reader gaps are Confident
Learning, image duplicate selection and cold embedding IDs. Their closure is
bound to each saved chapter revision and its scoped review.

This closes the gaps recorded in that Data Selection map, not every Harvard
unit, the English textbook or the whole-book backlog. Existing source
corrections and keep/link-only/omit figure decisions remain explicit.
