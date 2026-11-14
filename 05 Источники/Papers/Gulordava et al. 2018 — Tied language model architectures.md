---
title: "How to represent a word and predict it, too: Improving tied architectures for language modelling"
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://aclanthology.org/D18-1323/
authors: [Kristina Gulordava, Laura Aina, Gemma Boleda]
year: 2018
license: CC BY 4.0
---

# How to represent a word and predict it, too

In §2.1 the authors compare a language model with separate input and
output tables with one that uses the input table again for prediction.
Their recurrent example uses an LSTM to produce the hidden state. The
choice of one or two vocabulary matrices also applies when a Transformer
produces that state. In §2.2 they add a projection before the tied output
table so that the hidden state and the embeddings can have different
widths; this is the third panel of Figure 1, not the setup of GPT-2.

[Figure 1, page 2937](https://aclanthology.org/D18-1323.pdf#page=2) has
three panels. Panels (a) and (b) appear in
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-parameter-budget|the GPT–LLaMA parameter comparison]].
Both panels retain their original labels and colors.
[ACL Anthology copyright policy](https://aclanthology.org/faq/copyright/)
applies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) to
ACL materials published from 2016 onward.
