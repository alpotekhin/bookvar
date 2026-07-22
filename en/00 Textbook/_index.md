---
title: NLP and LLM Textbook
type: textbook-chapter
status: active
locale: en
translation_of: "00 Учебник/_index.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
---

# NLP and LLM Textbook

Bookvar follows one continuous route from the mathematical foundations of
machine learning to language-model architecture, training, inference,
retrieval, and agentic systems. Russian remains the root edition in Obsidian;
English chapters share the same stable routes, formulas, figures, and source
provenance.

When an English chapter has not yet passed editorial review, the site displays
the Russian source as an explicitly marked fallback. A fallback is part of the
navigation, but it is not counted as a completed translation.

## Curriculum

1. **Mathematical and machine-learning foundations** — vectors, gradients,
   probability, loss functions, validation, and generalization.
2. **Neural networks** — multilayer perceptrons, backpropagation, optimization,
   normalization, convolutional networks, and autoencoders.
3. **Text before the Transformer** — distributional representations,
   tokenization, n-gram models, recurrent networks, and sequence-to-sequence
   learning.
4. **Transformer, BERT, and GPT** — attention, positional information,
   encoder–decoder models, and the development from GPT-1 to GPT-3.
5. **Anatomy of a modern LLM** — decoder blocks, RoPE, grouped-query and latent
   attention, dense feed-forward layers, mixture of experts, and alternative
   sequence models.
6. **Training language models** — data, next-token prediction, scaling laws,
   distributed training, supervised fine-tuning, preference learning, RLHF,
   DPO, RLVR, and reasoning.
7. **Inference and systems around the model** — decoding, serving engines,
   efficient kernels, quantization, distributed inference, evaluation,
   retrieval, multimodality, and agents.

## Complete English inference sequence

The inference sequence is the first section available as a fully edited English
edition. Read it in order:

1. [[00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Decoding and next-token selection]]
2. [[00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|The physics of LLM inference: prefill, decode, and roofline]]
3. [[00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV cache, batching, and PagedAttention]]
4. [[00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Scheduling: continuous batching, chunked prefill, and prefix caching]]
5. [[00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines: vLLM, SGLang, TensorRT-LLM, and FlashInfer]]
6. [[00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]]
7. [[00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Quantizing language models]]
8. [[00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Speculative decoding]]
9. [[00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving|Distributed inference and disaggregated prefill/decode serving]]
10. [[00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLOs, and LLM inference operations]]

The English prose returns to the terminology used by the original papers and
courses rather than translating Russian sentence structure literally. Claims
remain attributed to their primary sources, and long quotations are kept out of
the textbook unless their exact wording is necessary.
