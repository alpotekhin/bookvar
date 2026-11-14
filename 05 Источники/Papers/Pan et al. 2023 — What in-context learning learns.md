---
title: "What In-Context Learning “Learns” In-Context: Disentangling Task Recognition and Task Learning"
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://aclanthology.org/2023.findings-acl.527/
authors: [Jane Pan, Tianyu Gao, Howard Chen, Danqi Chen]
year: 2023
license: CC BY 4.0
---

# What In-Context Learning “Learns” In-Context

[Pan et al., Findings of ACL 2023, pp. 8298–8319](https://aclanthology.org/2023.findings-acl.527/),
DOI [10.18653/v1/2023.findings-acl.527](https://doi.org/10.18653/v1/2023.findings-acl.527).

[§2 and Figure 1](https://aclanthology.org/2023.findings-acl.527.pdf#page=1)
set up three versions of the same classification problem. Random draws
demonstration labels from the original label set but leaves the test
targets intact. Abstract assigns a fresh one-to-one mapping from classes
to symbols per test prompt and scores the model against that mapping.
Gold uses the original, correct labels. The figure illustrates these
interventions; Appendix B lists the task-agnostic Abstract templates
actually used in the experiment.

[§3–4 and Figure 2](https://aclanthology.org/2023.findings-acl.527.pdf#page=3)
compare legacy GPT-3, LLaMA v1 and OPT across 16 classification
datasets. The top row fixes the demonstration count at 32 and varies
model size. The bottom row fixes the model and varies the count.
The three prompts contribute to the reported variation. GPT-3 was
evaluated on fewer examples than the other two families. The experiment
tests how reliably a model follows a mapping in its current prompt;
it does not locate the mechanism in particular layers.

The two figures and a worked example appear in
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-label-mapping|the GPT-to-LLaMA comparison]].
The figures retain their original labels and legends under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([ACL Anthology copyright policy](https://aclanthology.org/faq/copyright/)).
[Author code](https://github.com/princeton-nlp/WhatICLLearns).
