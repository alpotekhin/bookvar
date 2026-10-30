# Collective destinations: complete comparison and uneven routing

2026-10-27, local worktree `editorial-route-alignment`.
Chapter: `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md`.
Route: `/textbook/pretraining/processes-collectives-ddp/`.

Before: all 1411 lines read, SHA-256
`e33ee25e5047a10c3814e0f41c38fef164abe5364c1a30c7407a8fb3a91a9211`.
After: 1632 lines, SHA-256
`e1e9b7999ca10a61e9a48a24484f704b13cb7918cede4d5ba050b06876cbc8db`.
The final addition and its transitions into/out of the existing collective
explanation were reread. Removing just the bounded addition, its two primary
source entries and date changes restores the entire previous SHA. All old
DDP, ring, PowerSGD, join, overlap, stream and compiler explanations remain
unchanged; the earlier overlap-preservation test still checks its old SHA.

## Source and explanation comparison

The current full chapter already covers processes and subgroups, data
partitioning, correctly normalized accumulation, buckets, ring reduce-scatter/
all-gather, scheduling dependencies, streams, compilation, compression and
failure diagnosis. Those sections were not replaced by shorter definitions.

Harvard Network Architectures, Data movement primitives 4241–4434, was
reread completely, including its TikZ source and surrounding examples.
The remaining destination distinction was added to 44a: a root receiving a
list, every rank receiving that list, and every destination receiving a
different slice. Dense attention is not classified as necessarily random
gather; the source's universal 10× access penalty and near-peak claims are
not adopted. Cross-device collectives remain distinct from local indexed
addressing covered by S8.

Read completely: Séb Arnold's PyTorch tutorial Collective Communication
(RST 179–266), its six figures, the all-reduce example and complete API list;
PyTorch v2.14.1 full docstrings/examples for gather, scatter, all_gather and
all_to_all_single. Exact revisions, source hashes, rights and the untransferred
complex-example dtype finding are in
`05 Источники/Originals/collective-destinations-2026-10-27/README.md`.
The rest of the archived c10d source is not represented as read.

Original English gather/all-gather rules are retained, changing only RST
code markup and list indentation. They are followed by complete worked
examples rather than treated as the explanation itself:

- `[0,1,2,3]` assembled at one rank vs one-element `[6]` from reduction;
- scatter's separate parts vs broadcast's identical tensor;
- gather output at one root vs all-gather output on every rank; for four
  n-element inputs, 4n output elements at the root vs 16n across all four
  replicated output buffers, explicitly not a traffic count;
- equal all-to-all: source rows `[0..3]`, `[10..13]`, `[20..23]`, `[30..33]`
  become destination 2's `[2,12,22,32]`;
- uneven all-to-all: all original PyTorch inputs, counts and four outputs
  preserved. Rank 2's contributions `[4]`, `[15,16]`, `[23]`, `[34,35]`
  are traced using source-prefix offsets, producing `[4,15,16,23,34,35]`.
  The count matrix has row sums 6/9/5/7 and column sums 9/7/6/5; send rows
  become receive columns. Allocating output with input size is wrong here.

Named API assumptions cover equal-size gather/scatter buffers, root-only
lists, group-rank list order, synchronous return values, matching split sizes,
dimension-0 rows vs flattened scalar counts and the versioned experimental
API notice. Non-root reduce contents are not promised to contain the result.
The example starts from an initialized CPU/Gloo group, not an unexplained
pretend process setup; the existing chapter introduces setup separately.

`no-ai-slop` and its full eval were read. Editorial passages were checked
for inflated framing, generic transitions and over-compression. Original
source wording, numerical reasoning, caveats, code and figures were retained.
That editorial checklist is not substituted for the mechanism checks above.

## Native diagrams and permission evidence

Three complete unchanged original PNGs from PyTorch Tutorials revision
`c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6`:

| Figure | Native dimensions | SHA-256 |
| --- | --- | --- |
| scatter.png | 959×416 | `41aa1a57a36a1533676a3425284bfb5bc55d5d819bc371dc45cd4bb0fefa72b1` |
| gather.png | 958×417 | `7280d97d339aee4e27eec068f2afd4797c183c5429adc6d649086a94de2ef0d3` |
| all_gather.png | 957×429 | `637db0ff9288c02809e1d961b22cd72c47cfbc317730f13276a85db2b4a10a0b` |

All three source PNGs were inspected before adoption and again on the site.
The remaining reduce/all-reduce/broadcast figures were also inspected in the
source; they duplicate operations already illustrated in this chapter and
were not added. The three chosen figures label tensors and ranks, so colour
is not the only information carrier. Captions distinguish before/after rows
from additional processes, and logical arrows from physical topology.

The full BSD-3-Clause license was read and preserved in source storage and
`site/public/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt`.
Attribution and direct original links are adjacent to each illustration.
Original/copy bytes, dimensions, source-page hash and full license hash are
checked through the existing strict copied-source registry branch. No crop,
redrawing, translated labels or recompression. The first registry validation
caught an underscore in an asset ID; it was changed to a hyphen in both
registries. Validation was not loosened and the full unit suite passed again.

## Executed evidence

`publishing/tools/check_collective_destinations.py` executes both actual
printed fences sequentially on four real local PyTorch 2.14.1 CPU/Gloo ranks.
It then checks:

- independent reconstruction from every source's prefix offsets;
- all send and receive totals, conserving all 27 transmitted elements;
- gather to rank 2 and scatter from rank 3;
- rank-ordered output despite deliberate arrival skew;
- same numeric values remaining separate inputs, not silently reduced;
- ordinary gather return value None and root-only output lists;
- zero-sized all-to-all pieces, including empty receive buffers.

The pre-existing DDP checker also passed on two actual CPU/Gloo processes:
collectives and crossed-meaning control, subgroup, actual flat-gradient
fence, both CS336 models, masked-token accumulation, gradient/parameter/Adam
state parity, no_sync, join and early termination. Its numerical code was
not changed, only the total AST-fence count 13→15. Existing overlap checks
passed both actual model fences, 1000 independent schedule comparisons and
the readiness/compilation counterexamples. Last chapter changes were prose
only; none of those old printed snippets changed.

Logs:

- `/var/tmp/bookvar-collective-destinations-examples-final.log`: four-rank
  printed examples and extra cases PASS on final chapter SHA;
- `/var/tmp/bookvar-collective-destinations-ddp.log`: existing DDP checker PASS;
- `/var/tmp/bookvar-collective-destinations-overlap.log`: existing model checks PASS;
- `/var/tmp/bookvar-collective-destinations-unit-final.log`: 71 files / 535 tests PASS;
- `/var/tmp/bookvar-collective-destinations-build-final.log`: 1969 pages,
  23.89 s, completed 03:32:08, terminal exit 0;
- `/var/tmp/bookvar-collective-destinations-output.log`: 140 output tests PASS,
  including the new section and all three images in root/EN output;
- `/var/tmp/bookvar-collective-destinations-links.log`: 0 broken internal
  routes, fragments or files;
- `/var/tmp/bookvar-collective-destinations-coverage.log`: structural coverage
  PASS, 210 rows / 58 existing destinations, not semantic approval of all rows.

All worker groups completed and were destroyed. Hostname warnings explicitly
fell back to loopback; no host or network settings were changed. No external
cluster, GPU, NCCL, DDPOptimizer execution or throughput benchmark was used.

## Actual rendering

The completed local build was opened at the actual 44a route. Desktop
1280×900: body 1265/1265; three new figures at about 617×268, 617×269 and
617×277, full bounds visible. All three were read in their surrounding prose.
Mobile 360×800: body 345/345; figures about 313×136, 313×136 and 313×140.
All 14 chapter images loaded, all 16 display formulas and inline KaTeX fit,
KaTeX errors 0. Existing narrow source plots are not called newly re-reviewed
merely because they loaded.

At fit size on a phone the tensor subscripts are small. Each new illustration
was therefore inspected in the viewer at 2×, including both edges, where its
labels are readable. No source edge or arrow was lost. The two new tables
have internal horizontal scrolling: operation table 313/315, split-count
table 313/394. The latter was scrolled to 80.5 px to inspect the rightmost
rank/total columns without moving or widening the page.

Saved actual screenshots:

- `screenshots/2026-10-27-collective-destinations-desktop.png`;
- `screenshots/2026-10-27-collective-all-gather-mobile-viewer.png`;
- `screenshots/2026-10-27-collective-split-counts-mobile.png`.

The EN route is an explicit fallback, not a completed translation. Existing
source English is visible through that fallback; no EN body was edited.

## Scope left open

The inter-device destination-comparison gap is closed. H-V1-06 remains
integration-partial pending individual disposition of all 24 native source
objects; H-F1-06 remains re-review. Do not approve the native cross-attention
drawing merely because its missing-image render defect was fixed. Other
Harvard/HSE/EDLS/course/rights tasks and the full textbook goal remain open.
Ready RL chapters were not edited. No staging, commit, push, deployment,
delegation or live infrastructure mutation.
