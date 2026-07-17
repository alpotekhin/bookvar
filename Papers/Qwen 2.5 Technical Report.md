---
title: "Qwen2.5 Technical Report"
url: https://arxiv.org/abs/2412.15115
authors: "Qwen Team"
year: 2024
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/qwen-25/source]]"
concepts: [Scaling Laws, Data Filtering, Synthetic Data, DPO, GRPO, Reinforcement Learning, Mixture of Experts, Long Context]
---
# Qwen2.5 Technical Report

## TL;DR

Qwen2.5 -- серия dense LLM (0.5B-72B) + MoE модели (Turbo, Plus), обученных на **18T токенов**. Ключевые улучшения: масштабирование данных (7T→18T), better synthetic data, two-stage RL (offline DPO + online GRPO), longer generation (до 8K токенов). Flagship Qwen2.5-72B-Instruct конкурирует с Llama-3-405B-Instruct при 5x меньшем размере.

## Problem

Qwen2 имел ограничения: короткая генерация (2K токенов), слабая работа со structured data, limited tool use. Нужно было улучшить quality через data scaling, расширить capabilities (long generation, structured understanding, instruction following) и выпустить модели разных размеров для разных use cases.

## Method

### Архитектура
- Dense Transformer decoder: GQA, SwiGLU, RoPE, QKV bias, RMSNorm
- 7 размеров: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- MoE: fine-grained expert segmentation + shared experts routing (по аналогии с DeepSeek)
- Tokenizer: BBPE, 151,643 tokens + 22 control tokens
- Context: 32K (small) / 128K (7B+), generation: 8K tokens

### Pre-training Data (18T tokens)
- **Better filtering**: Qwen2-Instruct как quality filter с multi-dimensional analysis
- **Better math/code**: данные из Qwen2.5-Math и Qwen2.5-Coder интегрированы в pre-training
- **Better synthetic data**: Qwen2-72B-Instruct + Qwen2-Math-72B-Instruct для генерации, reward models для фильтрации
- **Better data mixture**: downsampling e-commerce/social media, upsampling tech/science/academic
- Staged pre-training с разными data mixtures

### Post-training
**SFT** (1M+ примеров):
1. Long-sequence generation: back-translation для queries к long-text pre-training data
2. Math: chain-of-thought из Qwen2.5-Math, rejection sampling + reward model
3. Code: multi-language agents, sandbox validation, 40+ languages
4. Instruction-following: code-based verification + unit tests
5. Structured data understanding: tables, JSON, reasoning chains
6. Logical reasoning: 70K queries, deductive/inductive/analogical reasoning
7. Cross-lingual transfer: translation model для low-resource languages
8. Robust system instructions: hundreds of diverse system prompts

**Two-Stage RL**:
- **Offline RL (DPO)**: 150K training pairs на math/code/instruction-following. Evaluation through execution feedback + answer matching.
- **Online RL (GRPO)**: reward model trained на labeling criteria (truthfulness, helpfulness, conciseness, relevance, harmlessness, debiasing). Qwen2.5-72B-Instruct как base для RM.

### Long Context
- 4096→32K during pre-training, YARN + Dual Chunk Attention для 4x inference extension
- Qwen2.5-Turbo: progressive extension 32K→256K, inference до 1M tokens

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/qwen25-figure1-hq.png]]

## Key Results

| Benchmark | Qwen2.5-72B-Instruct | Llama-3-405B-Instruct | GPT-4o |
|---|---|---|---|
| MMLU-Pro | 71.1 | 73.3 | 74.0 |
| MATH (0-shot) | 83.1 | 73.8 | -- |
| HumanEval | 86.6 | 89.0 | 90.2 |
| IFEval | 86.1 | 88.6 | 88.0 |
| GPQA | 49.0 | 51.1 | 53.6 |

- 72B конкурирует с 405B Llama 3 при 5x меньшем размере
- Qwen2.5-Turbo: competitive with GPT-4o-mini, поддержка 1M token context
- 100+ моделей на HuggingFace (base + instruct + quantized)

## My notes

- **Data scaling 7T→18T** -- основной драйвер improvement. Использование specialized models (Math, Coder) для генерации pre-training data -- это self-improvement loop.
- Two-stage RL (offline DPO + online GRPO) -- pragmatic подход. DPO для задач с verifiable answers (не нужен RM), GRPO для subjective quality.
- SFT pipeline впечатляет масштабом: 1M+ примеров, 9 capability areas, code-based verification для instruction following. Это massive engineering effort.
- 7 размеров моделей (0.5B-72B) -- стратегия покрытия всего рынка. 3B и 14B -- underserved ниши, верное решение.
- Scaling laws для hyperparameters (не только model size) -- полезный вклад. Позволяет предсказать optimal lr/batch size для новых архитектур.
- YARN + DCA для 4x context extension на inference -- practical solution, но quality degradation на extreme lengths неизбежен.
