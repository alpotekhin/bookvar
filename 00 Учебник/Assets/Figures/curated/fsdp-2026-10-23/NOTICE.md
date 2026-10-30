# ZeRO and FSDP: complete original source diagrams

Authors: Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro von Werra and Thomas Wolf; Hugging Face, The Ultra-Scale Playbook. Source commit: c93890dd4b22356ee18ed9b863e6f2d0909729bc. The pinned README declares Apache-2.0; its complete text is distributed at [/licenses/ultrascale-Apache-2.0.txt](/licenses/ultrascale-Apache-2.0.txt).

Three full diagrams are reproduced from zero_memory.svg, dp_zero3_fwd.svg and dp_zero3_bwd.svg. The originals are retained byte-for-byte in 05 Источники/Originals/fsdp-2026-10-23/. The latter two also matched the existing SVGs used by other chapters. No existing SVG has been overwritten.

The SVGs embed equations and GPU labels as PNG objects. Direct browser inspection lost these objects. An initial MuPDF rasterization also failed visual QA because clipping was not respected and GPU labels overlapped. Neither that output nor the cropped Quick Look thumbnail is the delivered diagram.

The accepted files are complete CairoSVG 2.9.1 renders, width 2400 px, white background, with the original aspect ratios and full viewBoxes. No figure has been redrawn, internally cropped, relabeled, translated, recolored or recomposed. The PNG format is a technical rendering of the original diagram. Labels, embedded equations and both extreme edges were checked visually. Exact source/output/code hashes and URLs are in assets.json and the central registry. The renderer is publishing/tools/render_fsdp_figures.py; host Cairo/fonts can affect output bytes, so a different environment must be visually checked rather than presumed bit-identical.

The source-page HTML and pinned license declaration already retained in tensor-sequence-2026-10-23 are reused. The Apache notice applies to these Playbook objects, not to the separate FSDP paper figures already present in the chapter under CC BY-NC-ND 4.0. Those files and their original attribution are unchanged.
