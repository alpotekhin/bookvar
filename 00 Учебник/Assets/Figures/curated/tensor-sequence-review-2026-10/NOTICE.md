# Original figures for chapter 44c

Imported 2026-10-13. These are byte-for-byte original assets, not redrawn,
translated, cropped or re-encoded by Bookvar.

Source: **The Ultra-Scale Playbook**, Hugging Face / Nanotron.
Authors: Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen,
Mohamed Mekkouri, Leandro von Werra, Thomas Wolf.

Pinned revision: `c93890dd4b22356ee18ed9b863e6f2d0909729bc`.
License: **Apache-2.0**, declared in the source Space's
[pinned README metadata](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/README.md).
`HF-LICENSE-METADATA.txt` is an exact local copy of that README, not a
replacement license drafted by Bookvar. This record covers the selected
Playbook artwork; it does not assign this license to external papers or
every third-party image linked from the Playbook.

| Local file | Exact original | SHA-256 |
|---|---|---|
| `hf-attention-tp.png` | [`tp_full_diagram.png`](https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/tp_full_diagram.png) | `01f9fd3bc4b0a97b167d6ce2d47b0a207492441592d1e2a51fab1e8bfad9962e` |
| `hf-tp-sp-regions.png` | [`tp_sp_diagram.png`](https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/tp_sp_diagram.png) | `d2463f346a6a3e16d447329a94eb8e9120d38effacf1230637ca25cd35d4c250` |
| `hf-causal-zigzag.svg` | [`cp_zigzagmask.svg`](https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/cp_zigzagmask.svg) | `8b8894a769d91f542743f7757122a0c0e8987d1fac3cbb4ddaca0f398a35ab0f` |

README SHA-256: `973209127acd43db26172be9130198447c7404e3206629487a0f42e3994aae58`.
PNG SHA-256 values also match the pinned upstream LFS object IDs.

Visual/content inspection:

- Attention figure: complete two-rank/head flow, output all-reduce and dropout.
- TP/SP figure: both complete comparison panels; labels f/f*/g/g* are explained
  directly in the chapter and do not silently inherit different paper notation.
- Zigzag figure: preserved as vector; caption names exact rows per GPU and
  independently verifies 34 allowed causal pairs per process. Color is not
  the sole means of finding a process's rows.

Rejected during curation: upstream `tp_diagram4.png` has inconsistent numbers
in its second matrix multiplication; `tp_sp_diagram_zoomed.png` has a less
complete comparison and ambiguous intermediate-width labels. Neither is
embedded or retained in this folder. The chapter keeps the already tracked
Harvard `tensor-parallel-split.svg` for the column/row comparison; that asset
retains its own CC BY-NC-SA 4.0 provenance.
