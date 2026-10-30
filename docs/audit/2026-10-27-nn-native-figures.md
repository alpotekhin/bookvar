# Harvard Network Architectures: individual native-figure review

## Scope and revisions

The complete native directory contains the same 24 distinct objects referenced
by the imported chapter. Each full image and its adjacent caption was inspected
individually, including marginal figures and the cover. The source chapter's
5256 lines had already been read sequentially; this pass does not restart that
reading or treat an image count as semantic approval.

- Immutable imported page:
  `05 Источники/Courses/Harvard ML Systems/vol1/nn_architectures.md`, SHA-256
  `52c77d7f6d386e97af65decf4f55523be785930571759fe5a906069ad13fcfac`.
- Source QMD:
  `05 Источники/Originals/nn-architectures-2026-10-26/chapter-2bd97c5.qmd`, SHA-256
  `64ef2c6bab3a09846bdb089a33c0ee355ee32ecbf291447493708c276f18d9a6`.
- Exact per-object paths, hashes, full captions, findings and destinations:
  [native disposition JSON](2026-10-27-nn-native-figures.json).

Inspection previews were rendered from the complete original bytes using
`publishing/tools/audit_nn_native_figures.mjs`. They remain in
`/var/tmp/bookvar-nn-native-2026-10-27`, outside publication. No source artwork
was redrawn, relabeled or cropped. The original archived files and source
page remain unchanged, including their documented mistakes.

## Individual decisions

“Archive” means the original remains available in the course source. It does
not mean the corresponding knowledge is omitted from the textbook. The JSON
records the thematic destination; the accepted chapters use source figures
and calculations described in the earlier content-comparison audits.

| No. | Image | Decision and specific reason |
| --- | --- | --- |
| 1 | Blueprint cover | Archive. Decorative pipeline, not a tensor or algorithm walkthrough. No claim about how the artwork was produced. |
| 2 | Inductive-bias dots | Archive. Three gray dots do not justify a universal sample-efficiency ordering of CNN, Transformer and MLP. |
| 3 | Algorithm axis | Archive. Data/Algorithm/Machine is a mnemonic, not an operation-count derivation. |
| 4 | Efficiency frontier | Archive. The caption explicitly mixes representative published points and an illustrative frontier. This is not a controlled benchmark. No new verification of every plotted report value is claimed. |
| 5 | Arithmetic-intensity bars | Archive. The source calculation divides FLOPs by FP32 weight storage, not measured traffic; no batch, inference phase or hardware ridge point is specified in the image. |
| 6 | MLP graph and matrices | Archive. The graph is legible but already covered more fully. Its caption counts three fully-connected layers while the drawing contains two weight transformations. |
| 7 | FC 205M / Conv 37K | Archive. The image omits the dimensions that produce its parameter counts. CNN already derives counts and distinguishes them from applications and activations. |
| 8 | Zebra CNN classifier | Archive. A high-level pipeline; existing LeNet/VGG/ResNet figures are tied to actual tensor shapes and code. |
| 9 | Convolution receptive field | Archive. Spatial-support sketch without values or channel labels; the existing D2L calculation shows the actual products and outputs. |
| 10 | RNN marginal curves | Archive. No axes, units or legend identify the two quantities. A constant recurrent state must be separated from training activations and sequence latency. |
| 11 | RNN unfolding | Archive. Useful generic diagram, duplicated by the original D2L unfolding/BPTT figures already attached to full examples. |
| 12 | Pronoun attention | Archive. Thick arrows without numeric weights, checkpoint or intervention do not establish causal pronoun resolution. The finite-Q/K counterexample remains in self-attention. |
| 13 | Q/K/V attention | Archive. Circle hues have no numeric scale and do not represent the caption's darkness scale. The graphic does not establish mandatory quadratic storage. |
| 14 | Packed QKV projection | Archive. The affine shapes are useful and consistent, but only one token row is filled and the Q/K/V regions are unlabeled. Existing canonical projection/shape explanations are retained. |
| 15 | Attention memory wall | Archive. An unlabeled knee cannot make score materialization unavoidable or conflate it with linear KV storage. |
| 16 | Encoder–decoder Transformer | Archive with a wiring finding. Encoder output reaches all three cross-attention ports, including the decoder query port. Do not promote it as correct wiring. |
| 17 | Context timeline | Archive as historical. Ends in 2024, mixes encoder limits and decoder/product announcements; the “RAG Era” line is not a consequence of a particular context size. |
| 18 | Four-block all-to-all | Archive. No embedding ownership, payload or split counts. The full TorchRec and PyTorch diagrams show the relevant placements and delivery results. |
| 19 | Capacity wall | Archive. Rounded decimal-GB weight sizes alone do not establish total training/runtime fit. The underlying 100M × 128 × 4 calculation is valid as weight sizing. |
| 20 | Residual block | Archive with a label finding. `F(x)+x` appears after the last ReLU, although it is the pre-ReLU sum. Existing D2L diagrams and equations distinguish these states. |
| 21 | im2col / window matrix | **Integrated in CNN**, complete native SVG. Correct the explanatory caption's row/column convention; work through both channels and every output. Do not adopt the unsupported universal speedup. |
| 22 | Broadcast/scatter/gather/reduction | Archive. Most payload identity is color-only, with no rank/index/axis contract; reduction is explicitly 1+3+5+7=16. Use the reviewed indexed-movement and collective-destination explanations. |
| 23 | Energy bars | Archive. Operand width, process/technology and model conditions are absent from the image. The ratio is not a hardware-independent law. |
| 24 | Selection flowchart | **Retained in practice 18a**, unchanged. The text already treats family choices as initial candidates and constraints as scenario checks, not proof of best-model selection. |

These decisions close the individual native inventory, not every transfer or
license question in Harvard/HSE/EDLS. They do not approve a source because it
is a university course, and do not remove disputed material from its archive.

## CNN integration and explanation check

The current 1093-line CNN chapter was read in full before editing. The complete
selected Harvard section, QMD lines 4042–4240 including the im2col TikZ matrix,
was reread against the native SVG. The four matrix rows and filter values agree
with the drawing. The source caption and prose reverse their orientation and
state a 5–10× speedup without a benchmark configuration; those claims are not
copied as facts.

The new English section is inserted inside the existing matrix-lowering
explanation, after the original `F.unfold` example. It has one complete original
SVG with adjacent author, pinned source, license and explicit caption correction.
The matrix is decomposed into two 4×4 channel blocks for reading width. The
red outlines are followed by their actual values, so color is not the only
way to identify a window.

For the first output, channel contributions are 37 and 319, giving 356.
The other positions give 47+345=392, 67+397=464 and 77+423=500. The code
compares the literal diagram rows with transposed `F.unfold`, then compares
their dot products with `F.conv2d`. Materialized storage has 32 entries for
18 input values; a channel's center appears four times, a corner once.
There is no timing or GPU execution claim.

Removing only the bounded addition and restoring the update date recovers
the whole accepted CNN SHA
`d2fe8eaa5ceda18400dbbfae92d1fdcb13a61c07a35964adae0ef76cda65ee6e`.
The older equivariance baseline remains enforced after that restoration.
Other chapter explanations, code and figures are preserved. RL and EN bodies
were not edited.

The 17 actual chapter Python blocks ran on the existing cached PyTorch runtime.
No new dependency download was needed. The first attempt to use the bundled
Python selected Python 3.12 rather than the cached PyTorch's Python 3.14;
that offline resolution failed and was replaced by the existing compatible
runtime. It did not trigger an installation or a disk cleanup.

## Final revision and evidence

The complete expanded chapter was reread. Final notation-only changes were
then checked in their surrounding explanation: stride/dilation are both 1,
the long second-channel sum is wrapped, and the output is displayed as a
2×2 matrix rather than a long inline list. The final file has 1194 lines,
SHA-256 `faf4455eb24ef822f831521f1549b44f9f5fc5770a74c69f9b3f61f068bd66da`.
No definitions replaced the worked explanation; source quotations and the
previous whole-chapter baseline are preserved. The no-ai-slop review found
no generic introduction or unsupported emphasis to retain in the addition.

Final executed checks, all terminal exit 0:

- `check_cnn_examples.py`: all 17 actual chapter fences; literal window rows,
  independently indexed patches, both channel contributions, transposed
  unfold and overlap multiplicities; the previous 720 rotation cases,
  80 original/library correlations, 396 geometry cases and gradient checks
  also pass. Log: `/var/tmp/bookvar-nn-native-cnn-examples-release.log`.
- Unit suite: 72 files, 539 tests, 47.26 s.
  `/var/tmp/bookvar-nn-native-unit-release.log`.
- Final Astro build: 1969 pages, 26.34 s.
  `/var/tmp/bookvar-nn-native-build-accepted.log`.
- Built output suite: 141 tests, 4.91 s, including the new image, caption,
  code and equations in both root and EN routes.
  `/var/tmp/bookvar-nn-native-output-release.log`.
- Built links: zero broken internal routes/fragments/files.
  `/var/tmp/bookvar-nn-native-links-release.log`.
- Source coverage: 210 matrix rows and 58 destinations, structural PASS.
  `/var/tmp/bookvar-nn-native-coverage-final.log`. The first run exposed
  two destinations incorrectly combined into one table cell; the row now
  has one valid destination and names the existing practice in its figure
  description. The parser was not weakened.
- Scoped `git diff --check`: pass; no staging or publication.

The first native-inventory test run caught an erroneous `map(basename)`
callback in the new test. It was corrected to a one-argument callback;
the complete suite subsequently passed. This was a test implementation
error, not an alteration of native assets.

## Browser observations

Actual root route:
`http://127.0.0.1:4330/textbook/neural-networks/cnn-lenet-resnet/#two-channels-window-rows-and-a-filter-vector`.

- At 1280×900, body client/scroll widths are 1265/1265. The complete native
  figure displays at 617×265, including both inputs and the right-hand vector.
- At 360×800, body widths are 345/345. All 14 chapter images load; the new SVG
  displays at 313×134. Fine labels need the viewer at this width; inline
  rendering is not claimed to make every number comfortably readable.
- At 2× in the mobile viewer, the left input/kernel panels and right matrix/
  vector panels were each inspected after panning to the corresponding edge.
  The original labels and numbers remain sharp; no source crop was substituted.
- All 14 display equations and inline math fit at 360 px; KaTeX errors 0.
  The initial sum was 18 px too wide and was wrapped before final acceptance.
- The actual EN route shows “This content is not available in your language
  yet.” and includes the new original figure. This is a labeled fallback,
  not a completed English translation.

Saved screenshots:

- [Desktop](screenshots/2026-10-27-nn-native-desktop.png).
- [Mobile viewer, right edge](screenshots/2026-10-27-nn-native-mobile-viewer.png).
- [Mobile calculation and output](screenshots/2026-10-27-nn-native-mobile-formula.png).

The native-directory disposition is complete. The full textbook, remaining
course integration and source/rights findings are not declared complete.
H-F1-06 now records selective reuse with findings; H-V1-06 remains partial
pending reconciliation of its section-level transfer evidence. No GPU/NCCL
benchmark, EN body edit, agent work, commit/push or deployment occurred.
