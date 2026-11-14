---
title: "AlphaProof: Olympiad-level formal mathematical reasoning with reinforcement learning"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://doi.org/10.1038/s41586-025-09833-y
year: 2026
---

# AlphaProof

Thomas Hubert, Rishi Mehta, Laurent Sartran et al., *Nature* 651, 607–613.
[Paper](https://doi.org/10.1038/s41586-025-09833-y), published online
12 November 2025; issue dated March 2026.
[Full text and licence](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12999475/fullTextXML):
CC BY 4.0. The publisher's Figure 4 has no separate third-party credit line.

The sections “Inference-time scaling” and Methods → “Inference” describe
two ways to spend additional compute. A fixed proof network can run more
tree search. Alternatively, test-time reinforcement learning (TTRL) updates
a specialist initialized from the generalist network, using proof and
disproof experience on target statements and their generated Lean variants.
The success criterion remains a proof of the target.

[Figure 4](https://www.nature.com/articles/s41586-025-09833-y/figures/4)
uses average TPU hours per problem for tree search and average TPU days per
problem for TTRL. Its miniF2F curve uses the validation split; Table 1 uses
the corrected test split. The separate IMO 2024 evaluation includes expert
formalization and an independent geometry system.

[[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты#alphaproof-search-vs-ttrl|Textbook: compare the two budgets using the original Figure 4]].
