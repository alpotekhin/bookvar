# Computational Primitives: chapter comparison and missing-figure repair

Scope: the already fully read Harvard Network Architectures source,
pin `2bd97c5`, sections 4036–4559. This unit compares the whole current
tensor, GPU, roofline and kernel chapters. It does not close all systems
primitives or the native-figure inventory.

## Current artifacts and whole-text reading

- `00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md`:
  all 747 original lines read sequentially, including the full NumPy fence
  and source-figure explanations. Its initial SHA matched the accepted
  `06ffb472ba85aa805651996971e02a7774895c85787762560e134301228753a8`.
  The only repair splits the explicit numerical QK-transpose product and
  result into separate aligned rows, plus updates the date. Current:
  749 lines, SHA
  `f8b39f3482e80bc07ff4cafcdddd287f7fe89cb31d00c1f9d0db0d125dd6a536`.
- `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md`:
  all 910 lines read sequentially, including both Python examples and the
  limited CUDA C++ launch fragment. Unchanged SHA
  `d432d06ee1687fca66fe6151028a1c4a2bb2313a7e23d9d2b7d877b96cb48d81`.

The new `publishing/tests/tensor-mobile.test.ts` reverses the exact equation
wrap and update date, then verifies the entire accepted tensor-chapter hash.
No substantive explanation, original excerpt, example or illustration is
replaced by a short overview.

## Mechanism mapping

| Harvard mechanism | Existing explanation and worked case |
| --- | --- |
| Matrix products and shared maps | Tensor chapter: explicit row-by-column product, batched/shared weights, rectangular Q/K contraction, semantic axis error when batch equals heads |
| Broadcast and reduction | Tensor chapter: row/feature broadcasting, incompatible shapes, dual expansion, sum/mean/keepdims; no claim that array broadcasting is itself a network collective |
| Layout changes | Tensor chapter: transpose versus reshape, head split/permutation and copy qualification; S8: explicit row/column strides, sliced-row versus transposed-input contracts and tiled address grids |
| Reuse and local windows lowered to GEMM | GPU chapter: full tiled GEMM sequence, loaded elements versus arithmetic, masked incomplete tiles and zero-fill/barrier requirements; convolution lowering itself was compared in the CNN unit |
| Sequential/strided access | GPU chapter: explicit sector counts for five strides, offset crossing, same address-set permutation and cache/HBM distinction |
| Shared-memory data movement | GPU chapter: bank degree, broadcast of one word, 32×33 padding, XOR permutation and diagonal counterexample |
| Compute/memory resources | GPU chapter: block/SM/warp assignment, resident versus ready warps, register/shared limits, spills, Tensor Cores and scoped TMEM example |
| Avoiding intermediate writes | GPU chapter: three-kernel versus fused elementwise payload, FMA rounding and occupancy caveats |
| Host/device transfer and dependencies | GPU chapter: actual Tensor.to object/copy semantics, streams/events/pinned memory, reuse of two input buffers; one- versus two-stream original course figures |

The comparison was then extended through every line of the two remaining
local systems destinations:

- Roofline: 1105 lines, unchanged SHA
  `ee3b591a0085982c18c2ae691889e38f2f624b505cb2a414520c947c1f57323b`.
  The explanation already distinguishes work from traffic, mandatory
  storage from observed transfers, HBM from L2, sequential kernel bounds
  from pooled bounds, and type-compatible peaks. Its worked GEMM reuse,
  wave-quantization, MFU/HFU, scaling and retained-progress examples are
  preserved. No new short summary replaces those explanations.
- Kernels: 1699 lines, unchanged SHA
  `1aaeef7e9387d9840ac2361926df475179f23150db0c25771aea2bad0f06b33e`.
  Program instances, address vectors, strides, masks and neutral elements,
  row reductions, online rescaling, tiled GEMM, L2 ordering and fusion
  limits already have worked cases and original course figures. The full
  FLCE, Dynamo and CUDA Graphs additions were also read, not skipped because
  the current source comparison concerns lower-level primitives.

Specific remaining explanatory checks: a local reduction tree's work and
dependency depth; indexed gather/scatter, repeated destinations and the
difference from network collectives; full comparison of the distributed
destination. The current S8 explains chunk accumulation and delegates the
within-block reduction to the compiler, but does not work through a binary
reduction tree. Harvard's four-way fan-in picture is not such a derivation.
Do not close that gap with the source's unqualified near-peak assertion.

## Source contradictions and boundaries retained

The source's im2col picture stores window positions in rows despite its
column-based caption; the source also gives an unsupported generic 5–10×
speedup. Existing CNN/GPU explanations retain their explicit conventions
and distinguish algorithmic loads from measured traffic.

Input-dependent dense attention weights do not alone require a dynamic
execution graph or address-random access. The source's gather prose and
universal slowdown are not copied as hardware facts. Neither quadratic
score storage nor a particular energy ratio is mandatory for every
implementation. Batch reuse, tiling, caches and checkpointing must be
accounted for. INT8 TOPS/W divided by FP32 TFLOP/s/W is not an application
throughput-per-watt comparison. MAC and FLOP are not interchangeable units.

Live NVIDIA spot-check on 2026-10-27:

- [PTX ISA 9.4, Tensor Memory and allocation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory):
  the existing sm_100a/sm_100f example remains consistent with 128 rows,
  32-bit cells and power-of-two column allocation in units of 32. It is
  not presented as a layout for every GPU generation.
- [CUDA Best Practices 13.4, Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory):
  the existing CC 6.0+ sector model and permutation qualification match
  the documented example. Counted sectors are not unconditionally HBM bytes.

Only these selected sections were spot-checked live, not all 31,060 lines
of PTX ISA or all hardware contracts. The chapter's previous broad
last_verified date is not silently upgraded.

## Executed examples and visual findings

- `check_tensor_foundations_examples.py`: actual NumPy fence plus independent
  rational contraction, broadcasting/shape failures, norms, head permutation,
  rectangular batched scores and a reshape-copy counterexample. Log
  `/var/tmp/bookvar-harvard-tensors.log` before the notation-only repair.
- `check_gpu_foundations_examples.py`: both printed Python fences, 25 tiled
  products against scalar sums, sectors/banks/XOR, occupancy arithmetic,
  traffic accounting and a two-buffer dependency schedule. Log
  `/var/tmp/bookvar-harvard-gpu.log`. CPU branch executed; CUDA transfer
  explicitly False. The CUDA C++ fragment is not compiled or benchmarked.
- At viewport 360×800, GPU's seven display equations already fit; tensor's
  explicit QK-transpose example was 320px wide inside 313px. All 6 and 15
  retained chapter figures loaded with their original aspect ratios.
  Loading is not a new blanket figure-readability acceptance; previous
  provenance and figure QA remain evidence for unchanged objects.
- Before the adapter repair, the Harvard source page rendered 23 images
  although all 24 archived objects were referenced. The missing object is
  the encoder–decoder Transformer figure, not an unreferenced asset.
- Final tensor inspection at 360×800: the repaired QK product and result
  are both visible; all 22 display formulas fit the content width to within
  one pixel of browser rounding. Body client/scroll widths are 345/345.
  Screenshot: `screenshots/2026-10-27-tensor-qk-mobile.png`.
- `check_roofline_examples.py` passed on the unchanged current chapter:
  three actual CPU fences, both tables, 25 rational cost checks and 108
  strong-scaling identity cases. Three CUDA fences were parsed, not run.
- `check_triton_examples.py` passed on the unchanged current chapter:
  four actual CPU fences, 1026 index/mask lengths, stable and online
  softmax, all eleven printed GEMM matrices, 48 chunked-CE gradient cases,
  aliasing and launch-queue models. Log:
  `/var/tmp/bookvar-primitives-triton-compare.log`. No Triton compilation,
  CUDA execution or speedup is established.
- `check_compile_examples.py` also executed all three actual Dynamo fences
  on CPU: five static calls / three backend invocations, two cached regions
  around the explicit break, and three dynamic-shape calls / one backend
  invocation. The printed FX function matched the captured function. Log:
  `/var/tmp/bookvar-primitives-dynamo-compare.log`. The recording backends
  return the graph unchanged; this is not Inductor or CUDA performance.

## Missing source figure: reproduction, repair and limits

At source line 1285, a bare DOI is immediately followed by a wiki image:
`...w19-4828.![[Assets/.../8f27999...svg]]`. GFM autolinking swallowed the
beginning of the generated Markdown image. The old HTML's DOI ended in
`%5B!%5B8f27999...`, and it contained only 23 actual source images.

The first attempted change in `convertWikiSyntax` passed its isolated
tests but did not affect the real publication path: `prepareAssets`
already replaces images before wiki-link conversion. The built-output
regression test still failed, so that ineffective change and its isolated
tests were removed. The final repair separates the bare-URL/image seam
inside `prepareAssets`; ordinary inline images remain inline.

Observed RED: the new `buildPublication` fixture failed before that repair;
68 other tests in that file passed. Observed GREEN: build and link-converter
fixtures passed, followed by the complete suite. The built-output test
checks the immutable source hash, all 24 actual image elements, the restored
SVG and the correct DOI in both root and EN routes.

Final source-page SHA remains
`52c77d7f6d386e97af65decf4f55523be785930571759fe5a906069ad13fcfac`.
The SVG's SHA is
`e963a6f0a16b2dee19272589a175e0c1ccfac5b078126df6578917123de23f40`,
matching its archived import-manifest entry. A separate read-only output
inventory verified all 17 bare-URL-adjacent image inserts across ten Harvard
source pages, in both route locales. It did not accept their subject matter.

The restored figure was inspected in context at 1280×900 and in the mobile
viewer at 360×800, including both edges at 2×. Labels need enlargement on
the narrow page; no content is cropped from the SVG. Screenshot:
`screenshots/2026-10-27-harvard-transformer-restored-desktop.png`.

Subject-matter finding: the original TikZ connects the encoder output PT1
to all three decoder cross-attention ports, including `RB4.270`, where a
decoder-originating line also arrives. Thus the restored original must not
be adopted as a correct Q/K/V wiring diagram in the textbook. Preserve it
as an immutable course object and use the already retained primary
Transformer illustration for teaching. Restored rendering is not factual
approval of a source figure.

## Final checks and cursor

Final adapter revision: 68 unit files / 523 unit tests; 137 built-output
tests; 1969-page build; zero broken internal links. Source coverage passed
for the existing 210 matrix rows and 58 destinations; that is structural
evidence, not acceptance of every course mechanism. Logs:
`/var/tmp/bookvar-doi-figure-{unit,output,build,links}-final.log` and
`/var/tmp/bookvar-doi-figure-coverage.log`.

After the published transfer-map update, the build was repeated: 1969 pages
in 21.19 s; all 137 output tests, link check and source-coverage check passed.
Those logs are `/var/tmp/bookvar-primitives-map-{build,output,links,coverage}.log`.
No unit input changed after the 523-test adapter run.

The tensor formula-wrap repair and missing-image adapter repair are locally
verified. Continue with the specific reduction/gather/scatter gaps, then the
distributed destination and native-figure disposition. Overall source
integration and the wider textbook goal remain active. No EN body editing,
staging, commit, push or deployment is part of this unit.
