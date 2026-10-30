---
title: Machine Learning and Language Models
type: textbook-chapter
status: active
locale: en
translation_of: "00 Учебник/_index.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
---

# Machine Learning and Language Models

Bookvar follows one continuous route from the mathematical foundations of
machine learning to language-model architecture, training, inference,
retrieval, and agentic systems. Russian remains the root edition in Obsidian;
English chapters use the same routes with an `/en/` prefix.

This is a partial English edition. When no separate English page exists, the
site displays the original page with a fallback notice. That notice describes
language availability, not editorial quality. Original English courses also
appear in the [[05 Источники/Курсы|course collection]]; those archives are not
translations of Bookvar chapters.

## Curriculum

The fourteen parts below follow the textbook sidebar. They describe the whole
curriculum, including chapters currently available only in Russian. Links open
the first chapter or the module map; the sidebar contains the full sequence.

1. [[00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|Mathematical and ML foundations]] — vectors, gradients, probability, losses, validation, and generalization.
2. [[00 Учебник/01 Классическое машинное обучение/00 Карта модуля|Classical machine learning]] — linear models, trees, ensembles, clustering, and dimensionality reduction.
3. [[00 Учебник/02 Рекомендательные системы/00 Карта модуля|Recommender systems]] — collaborative filtering, ranking, sequential models, and evaluation. Neural recommenders require Part IV; Transformer-based recommenders also require Part VI.
4. [[00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|Neural networks]] — backpropagation, optimization, normalization, CNNs, and autoencoders.
5. [[00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|Text before the Transformer]] — embeddings, tokenization, language modeling, recurrence, and sequence-to-sequence learning.
6. [[00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Transformer, BERT, and GPT]] — attention, positional information, encoder–decoder models, GPT-1 through GPT-3, and T5.
7. [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|Anatomy of a modern LLM]] — decoder blocks, RoPE, efficient attention, MoE, and alternative sequence models.
8. [[00 Учебник/10 ML Systems/01 Модель как часть системы|Computational foundations of ML systems]] — GPUs, memory, roofline analysis, model arithmetic, numerical formats, profiling, and kernels.
9. [[00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|Training language models]] — data, scaling laws, distributed training, SFT, preference learning, RLHF, DPO, RLVR, and reasoning.
10. [[00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Inference and serving]] — decoding, KV cache, scheduling, efficient kernels, compression, and distributed serving.
11. [[00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow|Deployment and evaluation]] — the ML lifecycle, MLOps, reliability, evaluation, and contamination.
12. [[00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|Retrieval and RAG]] — embeddings, sparse and dense retrieval, reranking, and retrieval-augmented generation.
13. [[00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Multimodal models]] — vision–language architectures, training, documents, video, audio, image generation, and serving.
14. [[00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля|Tools and agents]] — tool use, context, memory, planning, coding and GUI agents, formal reasoning, scientific applications, and safety.

## Transformer chapters available in English

These ten chapters explain attention before introducing complete architectures
and the pre-training objectives of BERT, GPT, and T5. Their order matches Part VI
of the sidebar; the architectural-patterns chapter connects the common building
blocks to the individual model families.

1. [[00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention: queries, keys, and values]]
2. [[00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Masking, multi-head attention, and tensor shapes]]
3. [[00 Учебник/05 Attention и Transformer/04 Позиционная информация|Positional information]]
4. [[00 Учебник/05 Attention и Transformer/03 Полный Transformer|The complete encoder–decoder Transformer]]
5. [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three architectural patterns]]
6. [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa, and DeBERTa]]
7. [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1: generative pre-training]]
8. [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2: zero-shot through language]]
9. [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3: in-context learning]]
10. [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5: text-to-text learning]]

## Inference chapters available in English

These eleven chapters have separate English text. Additional inference chapters
in the sidebar may still use the original-language fallback. Read this sequence
in order:

1. [[00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Decoding and next-token selection]]
2. [[00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|The physics of LLM inference: prefill, decode, and roofline]]
3. [[00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV cache, batching, and PagedAttention]]
4. [[00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Scheduling: continuous batching, chunked prefill, and prefix caching]]
5. [[00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines: vLLM, SGLang, TensorRT-LLM, and FlashInfer]]
6. [[00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]]
7. [[00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Quantizing language models]]
8. [[00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Speculative decoding]]
9. [[00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving|Parallelism and collective operations]]
10. [[00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|Disaggregated prefill and decode serving]]
11. [[00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLOs, and LLM inference operations]]

## Multimodal chapters available in English

The multimodal route follows the data path from pixels to language-model
states, then separates visual tokenization, training, grounded perception,
temporal modalities, evaluation, and serving:

1. [[00 Учебник/16 Multimodal Models/64 Мультимодальные модели|From ViT and CLIP to multimodal language models]]
2. [[00 Учебник/16 Multimodal Models/64a Connectors и fusion|Connectors and fusion]]
3. [[00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции|Resolution, tiling, and spatial positions]]
4. [[00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Training VLMs: alignment, instruction tuning, and data]]
5. [[00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding|Documents, OCR, and visual grounding]]
6. [[00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|Video, audio, and omni models]]
7. [[00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|Evaluation, failure modes, and VLM serving]]

The additional chapters on diffusion and unified multimodal sequences are part
of the curriculum, but do not yet have separate English editions.

## Courses alongside the textbook

Use the [[05 Источники/Курсы|course reading map]] to find both a textbook chapter
and its source material. Stanford CS336 is organized around building and
training language models; Berkeley's advanced agents course connects reasoning
and post-training to agent applications. Efficient DL Systems and Harvard ML
Systems provide the broader systems material. The original lectures, figures,
and notebooks remain available in their source language.
