---
title: "DeepSpeed-FastGen: Dynamic SplitFuse"
type: source-note
status: reviewed
source_type: paper
source_language: en
source_url: https://arxiv.org/abs/2401.08671v1
published: 2024-01-09
last_verified: 2026-11-11
authors: [Connor Holmes, Masahiro Tanaka, Michael Wyatt, Ammar Ahmad Awan, Jeff Rasley, Samyam Rajbhandari, Reza Yazdani Aminabadi, Heyang Qin, Arash Bakhtiari, Lev Kurilenko, Yuxiong He]
---

# DeepSpeed-FastGen: Dynamic SplitFuse

[Section 3.2](https://arxiv.org/html/2401.08671v1#S3.SS2) describes
how the scheduler builds passes of a target token count from ongoing
generation and parts of prompts. A long prompt can span several passes;
a short prompt can be split when the remaining space is too small.
A partially processed prompt does not produce its first output token.

[Figure 3](https://arxiv.org/html/2401.08671v1#S3.F3) compares three schedules. The early vLLM row
alternates whole prefills with generation; Orca mixes an entire new
prompt with generation, so that pass is still long. In the SplitFuse
row, a partial prompt shares a pass with decode. The rectangles and
arrows describe the schedule, not measured durations. The paper's
[benchmarks, §4](https://arxiv.org/html/2401.08671v1#S4) use historical
versions and specific model, accelerator and workload configurations.

[Original Figure 3 PNG](https://arxiv.org/html/2401.08671v1/figures/fastgen-overview-light.png) is
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
[[00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching#Почему длинный prefill мешает decode|Bookvar scheduling chapter]]
reads the three rows and works through a separate cache and latency example.
