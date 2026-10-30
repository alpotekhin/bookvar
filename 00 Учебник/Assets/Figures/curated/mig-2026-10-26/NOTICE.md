# MISO GPU-sharing figures

Baolin Li, Tirthak Patel, Siddharth Samsi, Vijay Gadepally and Devesh Tiwari,
*MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant Systems
for Machine Learning*, arXiv:2207.11428v3, 6 October 2022, Figures 1 and 3.
[Paper and license declaration](https://arxiv.org/abs/2207.11428v3).
CC BY 4.0; the saved arXiv page links the license explicitly.

Complete native vector PDFs are extracted byte-for-byte from the authors'
source archive, then rendered with pdftoppm at 400 dpi. No data, drawing,
labels or composition is changed. Captions are retained beside the figures.
Native PDFs and the full paper/source archive are retained for provenance.
`assets.json` records hashes, source members and dimensions.

Figure 1 is schematic. Its seven lower boxes are not an A100 memory-slice
count, and its compute bands do not promise disjoint physical SMs for
dynamic MPS active-thread limits. The chapter distinguishes this from
current static SM partitioning, and GI memory isolation from CIs inside
one GI. Figure 3 is the paper's particular 2022 training-workload comparison,
not our GPU benchmark or a guarantee about LLM serving latency.
