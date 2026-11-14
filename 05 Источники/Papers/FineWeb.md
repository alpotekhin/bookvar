---
title: "FineWeb: web extraction, deduplication and filtering"
type: source-note
status: canonical
last_verified: 2026-11-13
source_url: https://arxiv.org/html/2406.17557v2
year: 2024
---

# FineWeb: web extraction, deduplication and filtering

[Penedo et al., The FineWeb Datasets, arXiv v2](https://arxiv.org/html/2406.17557v2)
describe the original 15-trillion-token English release, built from 96
Common Crawl snapshots. That count refers to the final corpus in this paper,
not to the intermediate collections or the number of tokens consumed in one run.

| Part of the paper | What it provides |
| --- | --- |
| [§§3.1–3.3](https://arxiv.org/html/2406.17557v2#S3.SS1) | Evaluation setup, text extraction and the baseline filters. |
| [§3.4, Figures 3–5](https://arxiv.org/html/2406.17557v2#S3.SS4) | Cross-snapshot and per-snapshot MinHash, including a separate look at the oldest snapshot. |
| [§§3.5–3.7](https://arxiv.org/html/2406.17557v2#S3.SS5) | C4 rules, additional heuristics and the final release. |
| [§4](https://arxiv.org/html/2406.17557v2#S4) and [Appendix F](https://arxiv.org/html/2406.17557v2#A6) | FineWeb-Edu classifier, thresholds, and changes in topic and domain coverage. |
| [Appendix D](https://arxiv.org/html/2406.17557v2#A4) and [Appendix E](https://arxiv.org/html/2406.17557v2#A5) | Model and training settings, MinHash collision probabilities, and ablation scale. |

The [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
and the [Datatrove implementation](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py)
are separate resources. Consult the paper's
[license section](https://arxiv.org/html/2406.17557v2#A2) for the database
license and Common Crawl terms. The **figures from the paper** use
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); that license
does not carry over to the contents of the crawl.

The experiment and original Figures 4–5 are explained in
[[00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных#fineweb-deduplication-scope|the textbook chapter on corpus construction]].
