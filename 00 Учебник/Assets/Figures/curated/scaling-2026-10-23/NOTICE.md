# Original scaling-law and coordinate-check figures

Verified 2026-10-23 for chapter 43. No redraw, crop, recoloring,
translated labels, or invented observations.

## Porian Figure 1

Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, Yair Carmon,
*Resolving Discrepancies in Compute-Optimal Scaling of Language Models*,
arXiv:2406.19146v1, Figure 1.

- Paper and license evidence: https://arxiv.org/html/2406.19146v1
- Original SVG: https://arxiv.org/html/2406.19146v1/figure1.svg
- License: CC BY 4.0, explicitly declared on this paper version.
- Terms: https://creativecommons.org/licenses/by/4.0/
- SHA-256: c6de58a59c3203a2e4ba9f7574513f6a6c60e6da426f35dd42ec91b7c1f9b5ba.
- All five panels, branch arrows, axes, legends and English annotations
  remain in the byte-identical SVG; viewBox 0 0 844.67727 553.03189.
  All SVG href references are internal font paths; no scripts or remote
  dependencies are required.

The panels concern RefinedWeb. The first panel reproduces a configuration
close to Kaplan's unadjusted scaling; it is not the adjusted exponent 0.73.
The cosine-decay branch and tuned-optimizer-without-decay branch are
separate experiments. The adjacent chapter explanation retains that distinction.
The English excerpt comes from §3.5 of the same paper version, unchanged.
The new chapter arithmetic is a separate synthetic example, not a repetition
of the authors' training runs.

## microsoft/mup coordinate checks

Greg Yang and microsoft/mup contributors.
Copyright (c) Microsoft Corporation. MIT License.
Pinned repository revision: 19814971934ef91dd546f88e913fc963e096d11c.

- Explanation: https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/README.md#coord-check
- SP source: https://raw.githubusercontent.com/microsoft/mup/19814971934ef91dd546f88e913fc963e096d11c/coord_checks/sp_mlp_adam_lr0.001_nseeds5_bn0_coord.png
- μP source: https://raw.githubusercontent.com/microsoft/mup/19814971934ef91dd546f88e913fc963e096d11c/coord_checks/%CE%BCp_mlp_adam_lr0.001_nseeds5_bn0_coord.png
- License: https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/LICENSE
- Public full license: /licenses/mup-19814971-MIT.txt.
- sp-coordinates.png SHA-256: 8569933d788dc2ad7ced8bada362b29c17472fd54f8da6e4a8fb2b2f48cd22f6.
- mup-coordinates.png SHA-256: 260297652888aacc47a458333eae437ac669e3cf9330204aed389c61ee83288a.

Both unchanged PNGs are 1500 × 400. They show t=1 initialization and
t=2/3 after one/two updates, not three optimizer updates. The vertical
scales differ between panels. The μP example uses MuAdam according to
the README; the original figure title still says `adam`.
The English excerpt about output/attention exceptions is copied unchanged.
No mup package installation, model training, or coordinate-check reproduction
was performed in this pass.

Original figure bytes, full pinned README and MIT license, and the Porian
HTML/license snapshot are retained in
`05 Источники/Originals/scaling-2026-10-23/`.
Exact per-file provenance is in `assets.json` and the central asset registry.
Earlier course crops retain their existing rights records; neither this MIT
license nor Porian's CC license is applied to unrelated course/paper figures.
