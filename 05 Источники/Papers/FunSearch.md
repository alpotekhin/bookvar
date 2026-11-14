---
title: "FunSearch: Mathematical discoveries from program search with large language models"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://doi.org/10.1038/s41586-023-06924-6
year: 2024
---

# FunSearch

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov et al.,
*Nature* 625, 468–475. [Paper](https://doi.org/10.1038/s41586-023-06924-6),
published online 14 December 2023.
[Publisher text and licence](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10794145/fullTextXML):
CC BY 4.0. [Figure 1](https://www.nature.com/articles/s41586-023-06924-6/figures/1)
has no separate credit-line restriction.

The specification fixes the evaluator and can provide a program skeleton.
The model proposes a new function, such as the priority rule inside a
greedy cap-set constructor. Valid executed programs return to a population
that supplies examples for later prompts. The cap-set result gives a
constructive lower bound; the bin-packing experiments instead compare
heuristics on instances. Neither experiment proves that the search
will find an optimum in general.

[Official code and released constructions](https://github.com/google-deepmind/funsearch).
[[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery#funsearch-program-search|Textbook: one cap-set search step and a separate checker]].
