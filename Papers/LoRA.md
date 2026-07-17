---
title: "LoRA: Low-Rank Adaptation of Large Language Models"
url: https://arxiv.org/abs/2106.09685
authors: [Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen]
year: 2021
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - PEFT
  - fine-tuning
  - LLM
  - efficiency
Organization: Microsoft
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]]"
raw: "[[02 Areas/ML & DL/raw/papers/lora/paper.txt]]"
---

# LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
**Published:** 2021 (arXiv:2106.09685v2, Oct 2021)
**URL:** https://arxiv.org/abs/2106.09685

## TL;DR

**LoRA** (Low-Rank Adaptation) — метод parameter-efficient fine-tuning: замораживает предобученные веса модели и вводит обучаемые матрицы низкого ранга BA в параллель к весовым матрицам Transformer. На GPT-3 175B: 10,000× меньше обучаемых параметров, 3× меньше VRAM, **нулевая дополнительная задержка при инференсе** (матрицы сливаются с W), качество на уровне или лучше full fine-tuning.

## Problem

Full fine-tuning LLM (GPT-3 175B) нецелесообразен в продакшне:
- Нужно хранить отдельную копию модели (350GB) на каждую задачу
- Высокий GPU memory barrier для обучения (Adam хранит optimizer states для всех 175B параметров)
- Adapter layers (альтернатива): добавляют inference latency (+20–30% при batch=1), т.к. обрабатываются последовательно
- Prefix tuning: нестабилен, сокращает доступную длину последовательности

## Method

### Ключевая гипотеза

Веса предобученных LM имеют низкую "intrinsic dimension" (Aghajanyan et al., 2020). Изменение весов при адаптации ∆W также имеет низкий intrinsic rank → можно параметризовать его через матрицы низкого ранга.

### LoRA reparametrization (§4.1)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig1.png]]
*Figure 1: LoRA reparametrization. Замороженная матрица W₀ + обучаемые матрицы B (d×r) и A (r×k). B инициализируется нулями, A — случайно. ∆W = BA = 0 в начале обучения.*

```
Forward pass с LoRA:
  h = W₀x + ∆Wx = W₀x + BAx

где:
  W₀ ∈ ℝ^(d×k)  — замороженные предобученные веса
  B  ∈ ℝ^(d×r)  — обучаемая матрица (init: zeros)
  A  ∈ ℝ^(r×k)  — обучаемая матрица (init: random Gaussian)
  r ≪ min(d, k) — rank (типично 1–8)

Scaling: ∆Wx умножается на α/r (α = const, обычно = первое r)
```

**Inference:** W = W₀ + BA вычисляется один раз перед деплоем → **0 дополнительной latency**.

**Переключение задач:** вычесть BA, прибавить B'A' — быстрая операция.

### Применение в Transformer (§4.2)

LoRA применяется к матрицам attention: Wq, Wk, Wv, Wo (и опционально MLP). По умолчанию в экспериментах — только Wq и Wv.

Количество обучаемых параметров: `|Θ| = 2 × L_LoRA × d_model × r`

**Экономия памяти (GPT-3 175B):**
- VRAM при обучении: 1.2TB → 350GB (r≪d → не нужны optimizer states для W₀)
- Checkpoint size с r=4, Wq+Wv: 350GB → **35MB** (10,000× сжатие)
- Training throughput: +25% vs full fine-tuning (нет градиентов для замороженных весов)

## Key Results

### RoBERTa / DeBERTa на GLUE (Table 2)

| Model | Method | Params | Avg GLUE |
|-------|--------|--------|----------|
| RoBERTa-base | FT | 125M | 86.4 |
| RoBERTa-base | AdapterD | 0.3M | 84.4 |
| **RoBERTa-base** | **LoRA** | **0.3M** | **87.2** |
| RoBERTa-large | FT | 355M | 88.9 |
| **RoBERTa-large** | **LoRA** | **0.8M** | **89.0** |
| DeBERTa-XXL | FT | 1500M | 91.1 |
| **DeBERTa-XXL** | **LoRA** | **4.7M** | **91.3** |

LoRA с 0.3M параметров превышает full FT 125M и Adapter 0.3M на RoBERTa-base.

### GPT-2 на E2E NLG (Table 3)

| Method | Params | BLEU |
|--------|--------|------|
| FT | 354.9M | 68.2 |
| AdapterL | 0.37M | 66.3 |
| AdapterL | 11.1M | 68.9 |
| **LoRA** | **0.35M** | **70.4** |

LoRA с 0.35M параметров превосходит полный FT 354.9M на BLEU.

### GPT-3 175B (Table 4)

| Method | Params | WikiSQL | MNLI-m | SAMSum R-L |
|--------|--------|---------|--------|-----------|
| FT | 175B | 73.8 | 89.5 | 44.5 |
| AdapterH | 40.1M | 73.2 | 91.5 | 45.1 |
| **LoRA** | **4.7M** | **73.4** | **91.7** | **45.9** |
| **LoRA** | **37.7M** | **74.0** | **91.6** | **45.1** |

LoRA с 4.7M (0.003% от 175B) превышает или сопоставим с FT на всех трёх датасетах.

### Inference Latency (Table 1, lora/fig5.png)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig5.png]]
*Figure 5: Задержка инференса GPT-2 medium (мс). LoRA = нулевой overhead (merged weights). AdapterL/H добавляют +20% при batch=1/seq=128.*

| Batch / Seq | FT / LoRA | AdapterL (+%) | AdapterH (+%) |
|-------------|-----------|---------------|---------------|
| 32 / 512 | 1449 ms | +2.2% | +3.0% |
| 16 / 256 | 338 ms | +5.0% | +8.4% |
| 1 / 128 | 19.8 ms | **+20.7%** | **+30.3%** |

## Ablation Studies

### Какие матрицы адаптировать? (Table 5, §7.1)

На GPT-3 175B, 18M параметров бюджет:

| Адаптируемые | r | WikiSQL | MNLI |
|-------------|---|---------|------|
| Wq only | 8 | 70.4 | 91.0 |
| Wv only | 8 | 73.0 | 91.0 |
| Wq + Wv | 4 | **73.7** | **91.3** |
| Wq + Wk + Wv + Wo | 2 | 73.7 | 91.7 |

**Вывод:** лучше адаптировать больше матриц с меньшим r, чем одну матрицу с большим r.

### Оптимальный rank r? (Table 6, §7.2)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig2.png]]
*Figure 2: GPT-3 175B validation accuracy vs trainable parameters. LoRA доминирует над Adapter/Prefix методами.*

| r | WikiSQL (Wq+Wv) | MNLI (Wq+Wv) |
|---|-----------------|--------------|
| 1 | 73.4 | 91.3 |
| 2 | 73.3 | 91.4 |
| 4 | **73.7** | **91.3** |
| 8 | 73.8 | 91.6 |
| 64 | 73.5 | 91.4 |

**Вывод:** r=1 уже конкурентоспособен! Оптимум ≈ r=4–8. Увеличение r сверх не даёт значимого прироста → ∆W действительно имеет низкий intrinsic rank.

### Subspace analysis (§7.3, lora/fig3.png, fig4.png)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig3.png]]
*Figure 3: Subspace similarity между Ar=8 и Ar=64 для Wq и Wv. Высокая схожесть верхних сингулярных векторов подтверждает: low-rank adaptation учит intrinsically low-dimensional subspace.*

Φ_Fr (нормализованная схожесть подпространств) между Ar=8 и Ar=64 ≈ 0.5–0.7 для верхних d=1 сингулярных векторов. Нижние → ≈ 0. Вывод: лишь несколько значимых направлений в ∆W.

## My notes

- LoRA стал de facto стандартом PEFT в LLM. Используется везде: Alpaca, Vicuna, PEFT library (HuggingFace). QLoRA (2023) объединил LoRA с 4-bit quantization → обучение LLaMA-65B на одной 48GB GPU.
- Практически всегда адаптируют Wq и Wv (реже Wk, Wo). MLP layers обычно замораживают.
- r=4 или r=8 — стандартные значения в практике. LoRA-rank выше 64 — почти никогда не нужен.
- Ключевой insight про нулевой inference overhead: merged weights — это не просто оптимизация, это архитектурное решение, отличающее LoRA от Adapters.
- α/r scaling: при изменении r рекомендуется сохранять α/r = const (или просто задать α=r).
