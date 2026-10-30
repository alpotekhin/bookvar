# Tensor Memory Layout

Vijay Janapa Reddi and Harvard ML Systems contributors, *ML Frameworks*,
Figure `fig-tensor-memory-layout`, source revision
`2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`, QMD lines 2904–2951.

The complete published SVG is copied from the existing source archive without
modification. Its source and teaching copies have the same SHA-256:
`f0d88143e5561c8061bfba1f5ebc9f79e7c9d712e4e79fd195ce06a8e1a4f036`.
The QMD contains the corresponding original TikZ. The published SVG is not
claimed to be a Git blob in the native image directory.

License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The repository license at the source revision is retained in
`05 Источники/Originals/nn-architectures-2026-10-26/LICENSE-2bd97c5.md`.
The chapter reproduces the first sentence of the source caption in English.
The adjacent worked example corrects the element/byte-address qualification;
it does not modify the original drawing or adopt its unsourced bandwidth
percentages. See `assets.json` for exact paths, source links and hashes.

# Device Placement Dimensions

The same authors and source revision, `fig-3d-parallelism`, QMD lines
3339–3440. `device-placement-dimensions.svg` is a byte-identical copy of
archived mediabag object `2989ddaa72253cc48b5b80d50c512a87ca09ff9b.svg`,
SHA-256 `6c3e1e350e5653c517ac8b82ec4a7184eceb7573ae8320b56c4ca222861c6b12`.
CC BY-NC-SA 4.0; same pinned license evidence as the layout figure above.
This generated publication SVG is not claimed to be a native Git image blob.

The complete SVG/viewBox and original English caption are unchanged. No
crop, redraw, translated label or raster replacement. Added explanation
assigns one logical numbering consistent with the eight visible GPU labels,
enumerates TP/DP/PP groups and distinguishes 32 devices from eight stacks
and from a global sequence batch. It makes no claim about physical GPU IDs,
network topology, collective implementation, pipeline timing or measured
performance. Added on 2026-10-28 to the distributed-training overview.
