---
title: "The Complexity Trap: observation masking and LLM summarization"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2508.21433v3
authors: [Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, Yaroslav Zharov]
year: 2025
source_language: en
---

# The Complexity Trap: observation masking and LLM summarization

[Paper v3](https://arxiv.org/html/2508.21433v3) ·
[Code and configurations](https://github.com/JetBrains-Research/the-complexity-trap) ·
[Released trajectories](https://huggingface.co/datasets/JetBrains-Research/the-complexity-trap).

The main experiment compares three ways of building the next input in
SWE-agent: retain the trajectory, replace old observations with placeholders,
or summarize older turns. [Section 3](https://arxiv.org/html/2508.21433v3#S3)
defines the observation window and the summary trigger. The main experiments
use a ten-turn tail in both reduced-context variants; the summary variant
accumulates another 21 turns before its first summary. Figure 3 illustrates
a one-turn tail, not the benchmark configuration.

[Table 1](https://arxiv.org/html/2508.21433v3#S4.T1) reports solve rate and
mean cost per instance. [Appendix C.1](https://arxiv.org/html/2508.21433v3#A3.SS1)
reports paired comparisons against the raw agent on the same 500 issues.
Those tests do not establish equivalence between masking and summarization.
[Section 5.1](https://arxiv.org/html/2508.21433v3#S5.SS1) tests a separate
50-issue OpenHands slice and varies the observation window. Its tuning result
does not determine a window for other agents.

Bookvar explains the transformation and reads the reported numbers in
[[00 Учебник/17 Tools и Agents/66 Agent harness и context engineering#harness-observation-masking|Observation masking and summarization]].
It reproduces the original Figure 3 PNG with attribution under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
