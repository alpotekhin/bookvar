---
title: "TinyBERT: Distilling BERT for Natural Language Understanding"
type: source-note
status: canonical
last_verified: 2026-11-05
source_url: https://aclanthology.org/2020.findings-emnlp.372/
authors: [Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu]
year: 2020
source_language: en
---

# TinyBERT: Distilling BERT for Natural Language Understanding

[Published paper, Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.372/),
DOI: 10.18653/v1/2020.findings-emnlp.372.
[Author manuscript, arXiv v5](https://arxiv.org/abs/1909.10351v5).
[Official code, pinned revision 0598f02d](https://github.com/huawei-noah/Pretrained-Language-Model/tree/0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9/TinyBERT).

The integrated explanation, original figures and numeric example are in
[[00 Учебник/12 Post-training и Alignment/08 Reasoning distillation#tinybert-transformer-distillation|Transformer-layer distillation]].
It covers layer correspondence, pre-softmax attention scores, hidden-state
projections, embedding targets and the two training stages.
The source concerns BERT encoders and sentence-level tasks, not reasoning-model
distillation.

Figure provenance is recorded in
`00 Учебник/Assets/Figures/curated/distillation-tinybert-2026-11-05/assets.json`.
The published ACL figures use CC BY 4.0 under the
[ACL Anthology copyright policy](https://aclanthology.org/faq/copyright/).
The preserved implementation files retain their Apache-2.0 notices;
the software license is not used as the justification for copying paper figures.
