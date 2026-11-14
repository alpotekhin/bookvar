---
title: "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators"
type: source-note
status: canonical
source_language: en
source_url: https://arxiv.org/abs/2404.04475v2
year: 2025
last_verified: 2026-11-09
---

# Length-Controlled AlpacaEval

Yann Dubois, Balázs Galambosi, Percy Liang and Tatsunori B. Hashimoto.
[Author version, arXiv v2, 10 March 2025](https://arxiv.org/html/2404.04475v2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
This is the version with four authors and the three original figures reproduced
in Bookvar. The version used here is pinned; the current leaderboard is not.

- [§2](https://arxiv.org/html/2404.04475v2#S2) specifies the 805 instructions,
  the reference responses, the automatic annotator and the raw win rate.
- [§3 and Figure 2](https://arxiv.org/html/2404.04475v2#S3.F2) describe the
  regression and the prediction at zero length difference.
- [§4.1 and Figure 3](https://arxiv.org/html/2404.04475v2#S4.F3) test sensitivity
  to the same model being asked for a concise or verbose answer.
- [§4.2 and Figure 1](https://arxiv.org/html/2404.04475v2#S1.F1) compare model
  rankings to the historical Chatbot Arena sample. The 0.94 and 0.98 values are
  rank correlations, not the share of correctly judged pairs.
- [§5](https://arxiv.org/html/2404.04475v2#S5) discusses remaining judge biases
  and the limits of the equal-length comparison.

[[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md#alpacaeval-length-control|Bookvar: the length-control calculation and original figures]] ·
[Stanford CS336, Lecture 12, chat benchmarks](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_12.py#L183-L193).
