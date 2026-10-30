# Original DDP timeline diagrams

Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri,
Leandro von Werra and Thomas Wolf; Hugging Face.
*The Ultra-Scale Playbook: Training LLMs on GPU Clusters*, 2025.

Pinned Space revision: c93890dd4b22356ee18ed9b863e6f2d0909729bc.
The Space README declares `license: apache-2.0`; that exact README,
the original article, and the unchanged SVGs are retained under
`05 Источники/Originals/distributed-overview-2026-10-23/`.
Full Apache License 2.0 is included there and at
`/licenses/ultrascale-Apache-2.0.txt`.

- `hf-ddp-sequential.svg`: original `assets/images/dp_overlap1.svg`,
  SHA-256 d622641a5e1ffa561888558cdb1b310ff5081562b5edeaf0abc153a1bedb7b66.
- `hf-ddp-overlap.svg`: original `assets/images/dp_overlap2.svg`,
  SHA-256 43e29a970b78972837688fe9dd5f8c2fd581f39eb9e8ebf1f59488b6b465cf23.

Both complete SVGs are byte-identical copies. No crop, redraw, recoloring,
translated label, or removed timeline component. Their viewBoxes are
0 0 944.31891 166.11024 and 0 0 856.93304 166.11024 respectively.
White-background PNGs used during inspection are previews only, not
replacement publication assets. New textbook timings are explicitly
synthetic examples and not measurements from this source.

The arithmetic figures `tp_diagram.svg` and `tp_diagram4.png` were
examined but are not used in this chapter. The two-matrix figure has
inconsistent intermediate products: its first branch shows a value 20
multiplied by [10,30], while the corresponding result is drawn as [0,0]
instead of [200,600]. The existing Harvard split diagram is retained;
the separate ReLU example in chapter 44 is checked by autograd on CPU.
Apache licensing here does not extend to third-party paper illustrations
embedded elsewhere in the Playbook or to the Llama/DeepSeek papers.
