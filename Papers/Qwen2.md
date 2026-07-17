---
title: "Qwen2 Technical Report"
url: https://arxiv.org/abs/2407.10671
authors: "An Yang, Baosong Yang, Binyuan Hui et al."
year: 2024
date_reviewed: 2026-04-07
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/qwen2/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
---

# Qwen2 Technical Report

**Authors:** An Yang, Baosong Yang, Binyuan Hui et al. (Qwen Team, Alibaba Group)
**Published:** 2024 (arXiv:2407.10671v4, Sep 2024)
**URL:** https://arxiv.org/abs/2407.10671

## TL;DR

Qwen2 -- серия LLM от Alibaba: 5 моделей от 0.5B до 72B dense + MoE модель 57B-A14B. Обучены на 7T+ токенов с поддержкой ~30 языков. Flagship Qwen2-72B достигает 84.2 MMLU, 64.6 HumanEval, 89.5 GSM8K, 82.4 BBH, превосходя Llama-3-70B в большинстве бенчмарков. Instruction-tuned Qwen2-72B-Instruct: 9.12 MT-Bench, 48.1 Arena-Hard, 35.7 LiveCodeBench.

## Problem

Open-weight LLM быстро приближаются к proprietary моделям (GPT-4, Claude-3 Opus). Qwen2 -- следующая итерация серии Qwen, направленная на:
- Улучшение coding и math capabilities через enrichment данных
- Расширение multilingual support (~30 языков vs ограниченный набор в Qwen1.5)
- Эффективный deployment на разных масштабах: от smartphones (0.5B) до GPU clusters (72B)
- MoE модель для баланса quality/efficiency

## Method

### Tokenizer

Byte-level BPE tokenizer (тот же что в Qwen1). Vocabulary 151,646 токенов. Высокая compression rate, что помогает multilingual capabilities.

### Архитектура Dense моделей

Transformer decoder с causal attention. Ключевые отличия от Qwen1:

| Config | 0.5B | 1.5B | 7B | 72B |
|--------|------|------|----|----|
| Hidden Size | 896 | 1536 | 3584 | 8192 |
| Layers | 24 | 28 | 28 | 80 |
| Query Heads | 14 | 12 | 28 | 64 |
| KV Heads | 2 | 2 | 4 | 8 |
| Head Size | 64 | 128 | 128 | 128 |
| Intermediate Size | 4864 | 8960 | 18944 | 29568 |
| Trained Tokens | 12T | 7T | 7T | 7T |

Ключевые решения:
- **Grouped Query Attention (GQA)**: снижает KV cache, повышает throughput при inference.
- **Dual Chunk Attention (DCA)**: разбивает длинные последовательности на chunks, сохраняя relative positional info между ними.
- **YARN**: rescaling attention weights для length extrapolation.
- **SwiGLU** activations, **RoPE** positional embeddings, **QKV bias**, **RMSNorm** + pre-normalization.

### MoE модель (57B-A14B)

- Upscaled из Qwen2-7B.
- 64 routed experts + 8 shared experts, 8 activated per token.
- **Fine-grained experts**: каждый expert меньше стандартного FFN, но активируется больше экспертов одновременно -- richer combinations.
- **Expert routing**: shared experts для общих задач + specialized для routing-specific.
- **Expert initialization**: upcycling из dense модели с diversification (shuffle параметров, 50% random reinit).

### Pre-training Data

7T токенов высококачественных данных: web, math, code, ~30 языков. Улучшения над Qwen1.5:
- Quality filtering с Qwen models (model-based classifiers + heuristics)
- Больше code и math данных
- Distribution optimization через scaled-down experiments
- 12T dataset (для 0.5B) не дал значимого улучшения над 7T -- quality > quantity.

### Long-context Training

Финальная фаза pre-training: context extension 4096 -> 32768 токенов. RoPE base frequency: 10000 -> 1000000. С YARN + DCA -- обработка до 131072 токенов.

### Post-training

1. **SFT**: 500K+ examples (instruction following, coding, math, reasoning, role-playing, multilingual, safety). 2 эпохи, seq len 32768, LR 7e-6 -> 7e-7.

2. **RLHF**: двухэтапный:
   - **Offline DPO**: на собранном preference dataset.
   - **Online DPO**: модель сама генерирует responses, reward model выбирает best/worst, формируются preference пары.
   - **Online Merging Optimizer** для снижения alignment tax.

### Data construction

Сложный pipeline:
- Automatic ontology extraction (InsTag)
- Instruction selection по diversity/complexity/completeness
- Instruction evolution (self-evolution через Qwen)
- Human annotation (ranking multiple responses)
- Automated synthesis: rejection sampling (math), execution feedback (code), data repurposing (literary works), constitutional feedback (safety)

## Key Results

### Base model: Qwen2-72B vs конкуренты (Table 2)

| Dataset | Mixtral-8x22B | Llama-3-70B | Qwen2-72B |
|---------|-------------|------------|----------|
| MMLU | 77.8 | 79.5 | **84.2** |
| MMLU-Pro | 49.5 | 52.8 | **55.6** |
| GPQA | 34.3 | 36.3 | **37.9** |
| Theorem QA | 35.9 | 32.3 | **43.1** |
| BBH | 78.9 | 81.0 | **82.4** |
| HumanEval | 46.3 | 48.2 | **64.6** |
| MBPP | 71.7 | 70.4 | **76.9** |
| GSM8K | 83.7 | 83.0 | **89.5** |
| MATH | 41.7 | 42.5 | **51.1** |
| C-Eval | 54.6 | 65.2 | **91.0** |

Qwen2-72B значительно превосходит Llama-3-70B в coding (+16.4 HumanEval) и math (+6.5 GSM8K, +8.6 MATH), а также в Chinese (91.0 vs 65.2 C-Eval).

### Base model: Qwen2-7B vs конкуренты (Table 4)

| Dataset | Mistral-7B | Gemma-7B | Llama-3-8B | Qwen2-7B |
|---------|-----------|---------|-----------|---------|
| MMLU | 64.2 | 64.6 | 66.6 | **70.3** |
| HumanEval | 29.3 | 37.2 | 33.5 | **51.2** |
| GSM8K | 52.2 | 46.4 | 56.0 | **79.9** |
| MATH | 13.1 | 24.3 | 20.5 | **44.2** |

Qwen2-7B показывает массивное преимущество в math и coding над всеми 7B-конкурентами.

### MoE: Qwen2-57B-A14B (Table 3)

Активируя лишь 14B параметров, конкурирует с 30B+ dense моделями (Yi-1.5-34B). Превосходит в coding (HumanEval 53.0 vs 46.3) и Chinese (C-Eval 87.7, CMMLU 88.5).

### Instruction-tuned: Qwen2-72B-Instruct (Table 6)

| Dataset | Llama-3-70B-Inst | Qwen2-72B-Inst |
|---------|-----------------|---------------|
| MMLU | 82.0 | **82.3** |
| HumanEval | 81.7 | **86.0** |
| MATH | 50.4 | **69.0** |
| MT-Bench | 8.95 | **9.12** |
| Arena-Hard | 41.1 | **48.1** |
| LiveCodeBench | 29.3 | **35.7** |

Особенно впечатляющий прирост на MATH (69.0 vs 50.4) -- отражает инвестиции в math data и rejection sampling.

### Малые модели (Table 5)

Qwen2-0.5B (300M non-emb params) конкурентоспособен с Gemma-2B и Qwen1.5-1.8B. Qwen2-1.5B превосходит Phi-2 в language understanding при меньшем размере.

## My notes

