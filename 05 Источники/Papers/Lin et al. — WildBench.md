---
title: "WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild"
type: source-note
status: canonical
source_language: en
source_url: https://arxiv.org/abs/2406.04770v2
year: 2024
last_verified: 2026-11-09
---

# WildBench

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha
Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi.
[Author version, arXiv v2, 5 October 2024](https://arxiv.org/html/2406.04770v2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- [§2 and Figure 3](https://arxiv.org/html/2406.04770v2#S2.F3):
  how the 1,024 conversations were selected and how task categories differ across benchmarks.
- [§3 and Figure 4](https://arxiv.org/html/2406.04770v2#S3.F4):
  checklist-guided individual scoring and pairwise judgments.
  The [authors’ larger rendering](https://github.com/allenai/WildBench/blob/d6b8dcaf377d173d031980f97c16e1a82618c03d/docs/wb_eval.png)
  uses a different figure number; Bookvar embeds the arXiv v2 original.
- [Appendix D](https://arxiv.org/html/2406.04770v2#A4) and
  [Appendix E](https://arxiv.org/html/2406.04770v2#A5):
  exact original prompts, respectively for WB-Reward and WB-Score.
- [§4.2 and Table 3](https://arxiv.org/html/2406.04770v2#S4.T3):
  historical correlations, reported separately by model subset, metric and baseline.

The released table code at `d6b8dcaf` uses a strict
[winning-length margin](https://github.com/allenai/WildBench/blob/d6b8dcaf377d173d031980f97c16e1a82618c03d/leaderboard/data_dir/_create_tables.py#L109-L153)
and distinguishes the
[internal score](https://github.com/allenai/WildBench/blob/d6b8dcaf377d173d031980f97c16e1a82618c03d/leaderboard/data_dir/_create_tables.py#L233-L253)
from the
[displayed scale](https://github.com/allenai/WildBench/blob/d6b8dcaf377d173d031980f97c16e1a82618c03d/leaderboard/data_dir/_merge_results.py#L94-L99).
The revision is evidence about those transformations, not a reproduction of the paper's experiments.

[[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md#wildbench-checklist|Bookvar: the checklist and a three-pair worked example]] ·
[Stanford CS336 Lecture 12](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_12.py#L194-L199).
