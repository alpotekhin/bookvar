---
title: "Yi: Open Foundation Models by 01.AI"
url: "https://arxiv.org/abs/2403.04652"
authors: [01.AI]
year: 2024
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/yi/paper.pdf|PDF]]"
concepts:
  - LLM Pre-training
  - Data Engineering
  - Long Context
  - Vision-Language
---

# Yi: Open Foundation Models by 01.AI

## TL;DR

Yi — серия bilingual (Chinese-English) моделей от 01.AI (Kai-Fu Lee) на 6B и 34B параметров. Обучены на 3.1T токенов с упором на **data quality over quantity**. Центральный тезис: при достаточно чистых данных стандартная Transformer архитектура (без экзотических модификаций) даёт near-GPT-3.5 результаты. Yi-34B достигает 76.3 MMLU, 83.7 CMMLU. Расширения: 200K context, vision-language, depth-upscaling. Finetuning dataset < 10K samples, каждый проверен ML-инженерами.

## Проблема

Как создать модель, которая:
1. Достигает GPT-3.5 уровня на benchmarks и human preference
2. Может работать на consumer GPU (RTX 4090, 24GB) после quantization
3. Билингвальная (Chinese + English) без деградации ни на одном языке

Design trade-offs: 34B (вместо стандартных 70B) дает inference efficiency + помещается в 24GB после INT4, но требует больше training data для компенсации.

## Метод

### Data Engineering — центральная идея

Авторы формулируют принцип: **3T quality tokens > 10T unfiltered tokens**. Cascaded cleaning pipeline:

**Heuristic Rule Filters:**
- URL/domain/word blocklists + garbled text
- Document length, special symbols ratio, short/consecutive/incomplete lines ratio
- Repeated words, n-grams, paragraphs (Nguyen et al.)
- PII anonymization

**Learned Filters (4 scorer):**
1. **Perplexity Scorer** (KenLM via CCNet) — отсев документов с high perplexity
2. **Quality Scorer** — классификатор, обученный на Wikipedia-quality pages
3. **Document Coherence Scorer** — детекция incoherent документов из разрозненных предложений
4. **Safety Scorer** — violence, pornography, political propaganda removal

**Cluster-based Filters:**
- Unsupervised semantic clustering web documents
- Quality labels через automatic + manual verification
- Удаление low-quality кластеров

**Deduplication (cascaded):**
- Document-level MinHash
- Sub-document exact-match
- Topic modeling для downsampling (ads)

Removal ratio значительно выше чем у CCNet, RefinedWeb, RedPajama.

### Tokenizer

BPE через SentencePiece, vocab 64,000. Числа split на digits. No dummy prefix (не помогает для Chinese). Character coverage 0.9999, rare chars -> UTF-8 bytes. Max token length 32 (для длинных Chinese phrases).

### Архитектура

Стандартный decoder-only Transformer (LLaMA-based):

| Config | Yi-6B | Yi-34B |
|--------|-------|--------|
| Hidden | 4096 | 7168 |
| Q-heads | 32 | 56 |
| KV-heads (GQA) | 4 | 8 |
| Layers | 32 | 60 |
| Seq length | 4096 | 4096 |

Отличия от LLaMA:
- **GQA для всех размеров** (LLaMA 2 только для 70B)
- **SwiGLU** с reduced activation $\frac{8}{3}h$
- **RoPE ABF** (Adjusted Base Frequency) для 200K context support

### Finetuning: Quality is All You Need

< 10K multi-turn instruction-response pairs. Каждый sample проверен и отшлифован ML-инженерами через множественные итерации с user feedback.

Техники:
- **Compound instructions** (inspired by WizardLM) — progressive complexity evolution
- **LIMA-style response formatting** — introduction-body-conclusion, bullet points
- **Step-Back CoT** — abstraction -> higher-level solution -> concrete reasoning
- **Hallucination reduction** — проверка, что knowledge не contained within model
- **Repetition reduction** — rewrite repetitive turns

**Data mixing:** InsTag-inspired tagging system, diversity-focused sampling, grid search ($\{1, 1/2, ... 1/64\}$ proportions per ability).

Training: AdamW, seq 4096, batch 64, 300 steps, constant LR 1e-5, NEFTune noise (45 для 34B, 5 для 6B).

### Long Context: 200K

Continue pretrain на ~5-10B length-upsampled data (mostly books). RoPE ABF для extension. Key observation: **1-2B tokens достаточно** для convergence на 4K-200K lengths + lightweight finetuning для near-perfect retrieval.

Авторы утверждают: способность моделировать longer dependency — **intrinsic capability** base model, а не injected post-train. Base model уже может, post-train просто releases эту capability.

### Vision-Language

Vision encoder + chat model. Multi-stage training для alignment visual representations с language model semantic space.

### Depth Upscaling

Увеличение глубины модели через continual pretraining — confirmed effectiveness для дальнейшего улучшения performance.

### Infrastructure

- Cross-cloud elastic task scheduling с automatic failure recovery
- Topology-aware resource allocation
- Hierarchical finetuning framework (Megatron for policy + DeepSpeed for reward)
- 4-bit model + 8-bit KV cache quantization + PagedAttention + Dynamic Batching
- 200K context support через computation-communication overlapping + sequence parallelism

## Ключевые результаты

### Base Model

| Benchmark | Yi-6B | Yi-34B | LLaMA 2-70B | GPT-3.5 | GPT-4 |
|-----------|-------|--------|-------------|---------|-------|
| MMLU | 63.2 | **76.3** | 69.7 | 69.1 | 83.0 |
| CMMLU | 75.5 | **83.7** | 53.3 | 55.5 | 71.0 |
| C-Eval | 72.0 | **81.4** | 50.1 | 52.5 | 69.9 |
| BBH | 42.8 | 54.3 | 64.9 | 70.1 | 86.7 |
| GSM8K | 32.5 | 67.2 | 56.8 | 57.1 | 92.0 |
| MATH | 4.6 | 14.4 | 13.5 | 14.0 | 40.2 |
| HumanEval | 15.9 | 23.2 | 31.7 | 48.1 | 67.0 |

Yi-34B **превосходит LLaMA 2-70B** на Chinese benchmarks при половине параметров.

### Chat Model

На AlpacaEval и Chatbot Arena: near GPT-3.5 human preference rate.

### Quantization

INT4 quantization: < 1% MMLU/CMMLU drop. 34B Chat помещается на 24GB GPU.

## Мои заметки

**"Data quality is all you need"** — центральный тезис, и данные его подтверждают. Yi-34B превосходит LLaMA 2-70B (вдвое больше параметров) на Chinese и comparable на English. Ключ — не архитектурные инновации (стандартный LLaMA), а cascaded data cleaning с removal ratio выше чем у всех конкурентов.

**< 10K finetuning samples** — каждый проверен ML-инженерами. Это LIMA-style philosophy: quality >> quantity для alignment. При таком малом размере возможен exhaustive grid search по data mixing, что невозможно с миллионами samples.

**34B sweet spot:** Достаточно большой для complex reasoning, но помещается на consumer GPU после INT4. Post-Chinchilla overtrain (3.1T vs optimal ~1T) компенсирует reduced size и снижает inference cost.

**Long context как intrinsic capability** — provocative claim. Если base model (trained на 4K) уже "может" 200K и нужно только 1-2B tokens для unlock — это говорит о том, что positional encoding (RoPE ABF) покрывает extrapolation, а few long samples достаточно для calibration.

**4 learned scorer** — один из самых детальных описаний data filtering pipeline в open-source LLM papers. Особенно интересен Document Coherence Scorer: детекция документов из разрозненных предложений — типичная проблема web crawl данных.

**GQA для 6B модели** — Yi первыми показали, что GQA не деградирует даже на малых моделях (LLaMA 2 использовал GQA только для 70B). Это потом стало стандартом.

**NEFTune** (noise injection в embeddings) — используется при finetuning. 45 noise scale для 34B vs 5 для 6B. Интересная разница — большая модель требует больше regularization?

**Open for research AND commercial use** — выгодная лицензия, что помогло adoption.
