# RNN: input projections and layer dependencies

## Delivered revision

Canonical chapter:
`00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT.md`.
1035 lines, SHA-256:
`9e7990cd4670623c0f8b4cc78fc233d30dfca3696143dee5ad01f87f2fa136b4`.
Route:
`/textbook/recurrent-networks/rnn-bptt/#parallel-work-inside-a-recurrent-network`.

The whole accepted 824-line chapter was read before deciding what to add.
It already contains the scalar forward pass and manual BPTT, a complete
96-parameter language model, teacher forcing, Jacobian products, clipping,
truncated BPTT and padding. Those explanations and all five source figures
were retained. Removing the marked `rnn-execution` insertion and restoring
only the update date recovers the accepted revision, SHA-256
`d9cb27f7d76de93f41fe3f252ce1da1bcf53897197decbabaf257dbddfabaa25`.
The Python checker and unit test enforce this preservation.

The resulting explanation was reread in four consecutive chunks through
the end of the chapter. The external NVIDIA fragment was then corrected
from the guessed `#gemm-dimensions` to the observed `#gemm-dim` and checked
in the final rendered page. No further mathematical or prose changes
followed that read.

## Source comparison

Harvard *Network Architectures*, pin
`2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`, archived source SHA-256
`64ef2c6bab3a09846bdb089a33c0ee355ee32ecbf291447493708c276f18d9a6`.
Sequential reading now covers lines 1–3690, including RNN, attention,
Transformer and sparse RecSys. Selected Decision Framework material at
4560–4993 was read earlier. The next sequential source cursor is 3691,
Shared Building Blocks. This is not full chapter or native-image-directory
acceptance. Comparison of the current attention/Transformer/RecSys chapters
remains outstanding; a read source section does not prove its integration.

For the RNN gap, the complete 380-line pinned D2L Deep RNN chapter was
read, including its framework variants. The complete original SVG was
viewed directly and then inside the local chapter. A short English source
passage is retained verbatim, with line wrapping only. The original
figure is a byte copy, not a replacement diagram.

Supporting primary references are Appleyard's 2016 cuDNN article and
NVIDIA's 2023 Recurrent Layers User's Guide §§1–5/Notices. Their historical
performance results are not generalized to another device or treated as
measurements of the new correctness example. No NVIDIA figure or full
passage was copied. Source scope and pins are in
`05 Источники/Originals/rnn-execution-2026-10-26/README.md`.

| Mechanism | Explanation and concrete check | Boundary retained |
| --- | --- | --- |
| Input projections | The existing scalar inputs yield `[0.5, 0, 0.5]` before any state is computed; flattening `[B,T,d]` gives the input matrix product | The nonlinear hidden-state chain still depends on its predecessor |
| Known inputs and generation | Teacher-forced inputs and a supplied prompt can be projected ahead; future generated tokens are unavailable | Preparing future projections does not feed them to earlier states |
| Projection chunks | FP32 buffer example: 62.5 MiB for all 1000 positions, 1.25 MiB for 20 | These are projection-buffer sizes, not total BPTT memory |
| Chunking versus truncation | Same states and all input/parameter/h0 gradients without detach | `F` changes scheduling; a detach at `K` changes derivative paths |
| Multiple layers | Complete D2L dependency grid; each cell needs the previous same-layer state and current lower-layer state | First-layer input width is `d`; later equal-width inputs have width `m` |
| Wavefront schedule | Three layers × four positions, all 12 cells explicitly assigned to six ideal slots | Equal-cost slots and unlimited concurrency are assumptions, not GPU speedup evidence |
| Weight reductions | Flattened delta/input/previous-state matrices produce the same shared-weight gradients as summed outer products | Reverse recurrence must first supply the deltas |

The no-ai-slop check preserves the original chapter rather than compressing
its explanation. The new English section uses the existing scalar example,
states dimensions before matrix products and separates scheduling from
truncation. A source-shape inconsistency was recorded in the source note,
not inserted as editorial commentary in the textbook. Necessary technical
contrasts, equations and source attribution remain.

## Executed evidence

Command:

```bash
LC_ALL=C LANG=C uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  python publishing/tools/check_rnn_bptt_examples.py
```

Final checker SHA-256:
`0f8714bbce0a80eef00ac135a48db31ef40056a1bccdf955da9ad9140de81dc0`.
It executes all six printed Python blocks independently. Previous checks
remain: full LM training, entropy lower bound, causal prefixes, state
chunks and batches, manual/finite-difference BPTT, every CE gradient,
noncommuting Jacobians, clipping/nonfinite handling and aligned TBPTT.

New checks exercise 216 projection-chunk cases over 48 independently
constructed shape configurations, including singleton axes, incomplete
last chunks and chunks longer than the sequence. All states and every
input/parameter/h0 gradient match an independent reference recurrence.
For all 48 configurations the two flattened weight-gradient products
match autograd. Sixty-four dependency grids check each prerequisite
before admitting a cell to its slot. Both memory calculations, the
unchanged D2L excerpt and original SVG hash also pass.

This does not execute a parallel CUDA schedule, measure latency or train
the source D2L GRU exercise.

Final logs:

- `/var/tmp/bookvar-rnn-execution-examples-final.log`: all checks pass.
- `/var/tmp/bookvar-rnn-execution-unit.log`: 64 files / 515 tests pass on
  the full insertion; only the external fragment and output assertion
  were subsequently corrected.
- `/var/tmp/bookvar-rnn-execution-unit-final.log`: two affected tests pass
  after the final fragment correction.
- `/var/tmp/bookvar-rnn-execution-build-final.log`: 1969 pages, 23.11 s,
  complete at 23:05:57 local time.
- `/var/tmp/bookvar-rnn-execution-output-final.log`: 136 output tests pass.
- `/var/tmp/bookvar-rnn-execution-links-final.log`: zero broken internal
  routes, fragments or files.
- `/var/tmp/bookvar-rnn-execution-coverage-final.log`: structural coverage
  passes, 210 matrix rows / 58 destinations. This is not semantic course
  completion.

The first new output test counted seven `<img>` elements, because the
viewer contains a hidden empty image. It was corrected to require the
six actual `/assets/` source images; no figure or old assertion was removed.

## Figure provenance and rendered result

New image: `d2l-deep-rnn.svg`, native `viewBox="0 0 200 226"`, SHA-256
`518484464a1fd5fce1bde4fefdbbf68402fde18f62c460ee1ab680fe3965d8e0`.
Authors: Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola;
D2L pin `23d7a5aecceee57d1292c56e90cce307f183bb0a`, CC BY-SA 4.0.
The source text, SVG, license and publication copy are hashed in both
the curated manifest and central registry. Byte equality is tested.

The final local build was reloaded before inspection. At desktop
1280×900 the original appears with its source excerpt, attribution and
interpretation. At 360×800 the body is 360/360 pixels, all 25 display
formula containers fit their 328-pixel content width, and the six
source images load. The new schedule table fits 328 pixels and includes
the final `(3,4)` cell. The original scalar table is independently
scrollable; it does not widen the document.

The new SVG is complete in the mobile fit view. At 3× both the upper-left
output/state cells and lower-right final input/state cells can be reached;
this was checked by scrolling, not inferred from file dimensions.
Saved screenshots:

- `screenshots/2026-10-26-rnn-grid-desktop.png` — final local page.
- `screenshots/2026-10-26-rnn-grid-mobile-fit.png` — final complete fit view.
- `screenshots/2026-10-26-rnn-grid-mobile-right-bottom.png` — reachable
  right/bottom at 3× (same unchanged SVG/caption).

The EN route was visited: it explicitly says the content is not available
in that language and shows the current canonical content, including the
new English source section and original SVG. This is verified fallback,
not a completed English translation. No EN body was edited.

The browser was restored to the RU RNN chapter, the viewer closed and
the temporary viewport reset. No staging, commit, push, deployment,
infrastructure mutation or RL editing was performed. The full textbook
and course-integration goal remains active.
