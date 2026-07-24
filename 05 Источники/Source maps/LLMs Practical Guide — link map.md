---
title: LLMs Practical Guide — source map
type: source-note
status: link-only
last_verified: 2026-07-24
---

# LLMs Practical Guide — source map

Repository:
[`Mooler0410/LLMsPracticalGuide`](https://github.com/Mooler0410/LLMsPracticalGuide/tree/c4a39847f5455b8383dffee55c4fe9e5e16966a4),
commit `c4a39847f5455b8383dffee55c4fe9e5e16966a4`.

The inspected repository has no license file. Bookvar therefore links to the
catalogue and figures but does not copy them. The source is mainly a curated
2023 bibliography; it is useful for historical coverage and for checking
which older papers a topic map has missed, not as evidence for the current
state of model families, APIs, prices, or licenses.

## What the catalogue contributes

| Source section | Useful material | Bookvar destination | Editorial treatment |
|---|---|---|---|
| [BERT-style models](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#bert-style-language-models-encoder-decoder-or-encoder-only) | BERT, RoBERTa, DistilBERT, ALBERT, UniLM, ELECTRA, T5, early GLM and ST-MoE | model atlas; encoder-only and encoder–decoder chapters | Use as a completeness checklist; cite primary papers. |
| [GPT-style models](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#gpt-style-language-models-decoder-only) | GPT through GPT-4, OPT, PaLM, BLOOM, GLaM, Gopher, Chinchilla, LLaMA, LaMDA | GPT history; model timeline; scaling | Preserve historical ordering; update later families from primary reports. |
| [Pretraining data](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#pretraining-data) | RedPajama, The Pile, data-centric AI, objective choice | pretraining data pipeline | Follow links to dataset cards and papers. |
| [Finetuning and test data](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#finetuning-data) | synthetic data, zero-shot evaluation, shortcut learning, OOD robustness | post-training data; evaluation and contamination | Separate training evidence from evaluation evidence. |
| [Knowledge-intensive tasks](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#knowledge-intensive-tasks) | MMLU, BIG-bench, Atlas, retrieval-augmented models | RAG and evaluation | Link to primary benchmark and retrieval papers. |
| [Abilities with scaling](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#abilities-with-scaling) | Kaplan and Chinchilla scaling, CoT, emergence, inverse scaling | scaling laws; reasoning | Present the emergence debate, not only the original claim. |
| [Specific tasks](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#specific-tasks) | VLMs, annotation, augmentation, LLM-as-evaluator | VLM, synthetic data, evaluation | Split by mechanism instead of retaining the broad source bucket. |
| [Efficiency](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#efficiency) | cost, latency, PEFT, ZeRO and Megatron | inference, PEFT, distributed training | Replace stale pricing; retain primary systems papers. |
| [Trustworthiness](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#trustworthiness) | calibration, shortcuts, bias, safety and detection | evaluation, safety and robustness | Expand into separate calibrated evaluation and safety chapters. |
| [Instruction tuning](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#benchmark-instruction-tuning) | FLAN, T0, Natural Instructions, OPT-IML | SFT and instruction data | Use primary papers and dataset documentation. |
| [Alignment](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#alignment) | preference learning, summarization from feedback, InstructGPT, HH-RLHF, reward overoptimization | SFT, reward modeling, PPO, DPO, RLVR | Source stops before DPO/RLVR; treat it as historical foundation only. |
| [Usage restrictions](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/README.md#usage-and-restrictions) | model/data license matrix as of 2023 | model cards and source registry | Never reuse as current legal status without checking official model cards. |

## Visual references

- [Model evolution tree](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/imgs/tree.jpg)
- [Model-or-finetuning decision flow](https://github.com/Mooler0410/LLMsPracticalGuide/blob/c4a39847f5455b8383dffee55c4fe9e5e16966a4/imgs/decision.png)
- [Editable PowerPoint sources](https://github.com/Mooler0410/LLMsPracticalGuide/tree/c4a39847f5455b8383dffee55c4fe9e5e16966a4/source)

These remain external until reuse permission is explicit. Their conceptual
content can still be reconstructed from cited primary sources when Bookvar
needs an updated timeline or decision diagram.
