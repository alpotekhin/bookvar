# English index and inference physics

The English index now follows the same core and specialist modules as the canonical index. It includes every chapter link and marks the 28 chapters with an existing English body. Unmarked links remain explicitly described as original-language fallbacks. A complete index is not a complete translation of the book.

The physics chapter now contains the omitted read/write ledgers for separate and fused gated MLPs; MHA/GQA arithmetic intensity; the Llama 2 storage and compulsory-read calculation; fixed-batch MBU; the union of routed experts; and unequal per-rank traffic. It preserves the original JAX explanation and plots, and distinguishes analytical limits from measured bandwidth or throughput. Five actual Python examples are byte-identical to the canonical programs.

The parent editor read both chapter bodies, worked through the numerical examples, reread the completed English text, and corrected the dense-parameter convention, a dimensionally invalid comparison of FLOPs with a rate, and overlap in TTFT component definitions. The example checker additionally enumerates transfer operations and routing pairs independently.

## Evidence

- Exact paths and before/after hashes: [JSON journal](2026-10-29-en-index-physics-parity.json).
- Five executable programs, independent MLP/GQA/cache/latency arithmetic, 16 expert-routing pairs, and uneven-rank MBU: `python3 publishing/tools/check_english_physics_parity.py`, passed.
- All seven canonical figures retained; the existing PagedAttention fragmentation illustration is also retained.
- Index and reading-path tests: 14 passed.
- Display-math delimiters checked after correcting a replacement-string escaping error.

No GPU measurement, inference-engine benchmark, Git publication, or deployment is claimed. The parent subsequently split one Q/K/V display across lines; the JSON retains its pre-render SHA and pins the new exact revision. All five CPU programs passed again. Physics and contents passed the final desktop/mobile checks, recorded in [the integrated report](2026-10-29-english-reconciliation.md).
