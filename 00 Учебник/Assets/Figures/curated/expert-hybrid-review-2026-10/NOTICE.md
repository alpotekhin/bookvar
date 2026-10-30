# Original figures for chapter 44f

Imported and visually inspected 2026-10-13. No diagram was redrawn, translated
or generated. The two EDLS panels remove only the presentation title and outer
margins; all diagram labels, arrows, matrices and device labels remain intact.

## Efficient DL Systems: TP versus EP

Authors: Efficient DL Systems course authors, `mryab/efficient-dl-systems`.
Pinned commit: `e632aa89ca9e6638d52e1b686095e7442faffbb0`.

- Source lecture: [Week 6, lecture.pdf](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf).
- Exact download: [raw PDF](https://raw.githubusercontent.com/mryab/efficient-dl-systems/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf).
- License: **MIT**, [pinned LICENSE](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE).
- `EDLS-LICENSE.txt` preserves the complete license, including copyright
  `(c) 2021 course authors`; SHA-256
  `01908feead9275d5043e082e9fa37e415a2972dca45fd62b8489e1fd2a058a13`.
- The unchanged source copy is integrated at
  `05 Источники/Papers/Source PDFs/edls-week06-e632aa89.pdf`; SHA-256
  `64b6f4285f9fe9bc62bf8293f3585eb26e738d493723b22c2301c58a5a191215`.

Both diagrams are the course's own TP/EP comparison drawings on PDF pages
129 and 130, not the visibly borrowed third-party image on page 119.
The latter was inspected and rejected for both readability and attribution.

| Asset | PDF page (1-based) | SHA-256 |
|---|---|---|
| `edls-moe-tensor-parallel.png` | 129 | `5118bb2f49454b22504c4915a2b9bedf002b4215c3764a16a6744939b5b32e08` |
| `edls-moe-expert-parallel.png` | 130 | `f74a6f8fd2d83d702bb21082fd6b17bd730dbfbdaab96b6e2b21af239eefc7ce` |

Derivation: `pdf-page-render-crop`. Full page rendering: 2400 x 1350 px.
Crop box `[left, top, right, bottom]`: `[70, 230, 2240, 1220]`.
Output: 2170 x 990 px, PNG. Reproducible Poppler command (substitute page):

```sh
pdftoppm -f 129 -l 129 -scale-to 2400 -x 70 -y 230 -W 2170 -H 990 \
  -png -singlefile '05 Источники/Papers/Source PDFs/edls-week06-e632aa89.pdf' edls-moe-tensor-parallel
```

The two final crops were opened at full output size and inspected. No clipped
labels or lost matrix panels. The chapter explains that the EP panel's B1-B4
labels identify source batches, not an assertion that every entire batch is
delivered to every expert.

## NVIDIA Megatron Core: dropless, drop, padding

Author: NVIDIA CORPORATION.
Pinned version: `core_v0.19.2`, commit
`4b4acac9a1d28ea6829c8d4f566d75698a21249d`.

- Original: [token_drop.png](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/docs/images/moe/token_drop.png).
- Exact download: [raw PNG](https://raw.githubusercontent.com/NVIDIA/Megatron-LM/4b4acac9a1d28ea6829c8d4f566d75698a21249d/docs/images/moe/token_drop.png).
- License: **BSD-3-Clause**, repository default in the [pinned LICENSE](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/LICENSE).
  Its introductory rule applies to all files unless otherwise noted. No
  separate image notice or more specific directory license was found for
  this asset. This does not assign BSD to every document or borrowed file in
  the repository; the copied complete LICENSE records its other exceptions.
- `MEGATRON-LICENSE.txt` is the complete original LICENSE; SHA-256
  `39f1df137a306d31698db206063f462b69c19e536f012db09a66997ad2cb96f0`.
- `mcore-token-capacity.png`: exact original bytes, `source-raster-copy`,
  3104 x 1510 px, SHA-256
  `d5200aa91f34f4a09b6f5cce988dea53586e30965f3b4cecf3cc878c83ee2206`.

All three panels were visually inspected, and their counts were checked:
16 initial assignments, 13 retained under capacity 4, 3 dropped, 3 padding
rows. Original image flag names were not retouched; the chapter explicitly
points out the capacity flag spelling in the pinned current README.

## Rejected or retired illustrations

- EDLS p.119: blurred third-party permutation picture with visible attribution
  to a different author. Not imported under the course's MIT license.
- Harvard `moe-all-to-all-routing.svg`: top-2 illustration with a cost label
  lacking the corresponding top-k multiplier. Kept in its existing archive,
  removed from this chapter; not repaired by altering the source diagram.
- Harvard `3d-parallelism-cube.svg` and `parallelism-decision-tree.svg`:
  absolute interconnect requirements and fixed indicative speeds are not a
  reliable universal rule. Kept in archive, not embedded here.
- CS336 Lecture 8 p.66: the Llama 3 table is discussed directly from the
  primary report; the full presentation frame is no longer embedded.
- Ultra-Scale Playbook `5d_nutshell_ep.svg`: inspected but not selected;
  the complete wide graphic adds little to the explicit rank-group table.

Use in Bookvar: only the canonical Russian chapter
`00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md`.
Central asset registry and EN chapter are integrated by the parent task.
`assets.json` contains three proposed central records. The NVIDIA image uses
the registry's `pinned-upstream-repository` route for a GitHub original;
the two EDLS PDF crops use `source-license-verified`. Both carry verified
license metadata and the original notices above.
