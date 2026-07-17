---
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
url: https://arxiv.org/abs/2305.14314
authors: [Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer]
year: 2023
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - PEFT
  - quantization
  - fine-tuning
  - LLM
  - efficiency
Organization: University of Washington
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]]"
raw: "[[02 Areas/ML & DL/raw/papers/qlora/paper.txt]]"
---

# QLoRA: Efficient Finetuning of Quantized LLMs

**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington)
**Published:** 2023 (arXiv:2305.14314, May 2023)
**URL:** https://arxiv.org/abs/2305.14314

## TL;DR

**QLoRA** объединяет 4-bit квантизацию предобученных весов с LoRA-адаптерами. Результат: LLaMA 65B обучается на **одной 48GB GPU** (вместо >780GB с full FT) без потери качества относительно 16-bit LoRA. Лучшая модель семейства Guanaco-65B достигает **99.3% от ChatGPT** на Vicuna benchmark.

## Problem

- Full 16-bit fine-tuning LLaMA 65B: >780 GB GPU памяти → недоступно для исследователей
- Существующая квантизация снижает качество только для inference, не для training
- LoRA снижает VRAM, но 4-bit квантизация + backprop через замороженные веса ранее была невозможна без деградации

## Method: три инновации

### 1. 4-bit NormalFloat (NF4) (§3)

Предобученные веса LLM имеют **нормальное распределение с нулевым центром** → для таких данных существует информационно-оптимальный дискретный тип данных.

NF4 = квантильное квантизование N(0,1):
```
qi = ½ · [Q_N(i/(2^k+1)) + Q_N((i+1)/(2^k+1))]

где Q_N(·) — квантильная функция стандартного нормального распределения
```
Нормализация весов в [-1, 1] → квантизация по квантилям → равномерное распределение значений по бинам.

**Сравнение типов данных (Таблица 2, mean PPL на Pile Common Crawl):**

| Тип данных | Mean PPL |
|-----------|----------|
| Int4 | 34.34 |
| Float4 (E2M1) | 31.07 |
| Float4 (E3M0) | 29.48 |
| **NFloat4 + DQ** | **27.41** |

NF4 значительно превосходит FP4 и Int4 при тех же 4 битах.

### 2. Double Quantization (DQ) (§3)

Проблема: квантизация блоками требует хранения квантизационных констант (32-bit float на блок 64 веса → **0.5 бит/параметр overhead**).

DQ — квантизовать сами константы квантизации:
- Первый уровень: веса → NF4, константы c₂^FP32 (blocksize=64)
- Второй уровень: c₂^FP32 → c₂^FP8 (blocksize=256)

Результат: 32/64 = 0.5 бит/параметр → **8/64 + 32/(64·256) = 0.127 бит/параметр**
**Экономия: ~0.37 бит/параметр ≈ 3 GB для модели 65B**

### 3. Paged Optimizers (§3)

NVIDIA unified memory: состояния оптимизатора автоматически вытесняются в CPU RAM при OOM и возвращаются при обновлении весов. Позволяет обучать 33B/65B на GPU 24/48GB без OOM при длинных последовательностях.

### QLoRA forward pass (§3, формула 5)

```
Y^BF16 = X^BF16 · doubleDequant(c₁^FP32, c₂^FP8, W^NF4) + X^BF16 · L₁^BF16 · L₂^BF16

doubleDequant(c₁, c₂, W) = dequant(dequant(c₁, c₂), W^4bit) → W^BF16
```

- **Storage dtype**: NF4 (4-bit)
- **Computation dtype**: BFloat16 (16-bit)
- Градиенты вычисляются только для LoRA параметров L₁, L₂
- W замороженный, но проходит dequantization в BF16 для forward/backward

### Критично: LoRA на ВСЕХ слоях (§4)

Стандартный LoRA (только Wq, Wv) не восстанавливает 16-bit качество при QLoRA. **Применение LoRA ко всем linear layers Transformer** — необходимое условие для match 16-bit performance.

## Key Results

### QLoRA vs 16-bit FT (Table 3)

| Метод | GLUE (RoBERTa-large) | SNI RougeL (T5-3B) |
|-------|----------------------|-------------------|
| BF16 FT | 88.6 | 54.3 |
| LoRA BF16 | 88.8 | 55.4 |
| QLoRA Int8 | 88.8 | 56.5 |
| QLoRA FP4 | 88.6 | 55.6 |
| **QLoRA NF4+DQ** | **- / ≈88.6** | **55.3** |

QLoRA восстанавливает 16-bit производительность при 4-bit весах.

### QLoRA vs 16-bit LoRA на LLaMA 7–65B (Table 4)

| Модель | BFloat16 | Float4 | **NFloat4+DQ** |
|--------|----------|--------|----------------|
| LLaMA-7B Alpaca | 38.4 | 37.2 | **39.0** |
| LLaMA-65B Alpaca | 61.8 | 61.3 | **61.8** |
| LLaMA-65B FLAN v2 | 62.5 | 63.3 | **63.9** |

NF4+DQ воспроизводит BFloat16 качество; FP4 отстаёт примерно на 1 пп.

### Guanaco: State-of-the-art Open-Source Chatbot (Table 6, Table 1 Elo)

| Модель | Размер | Память | Vicuna score |
|--------|--------|--------|--------------|
| GPT-4 | - | - | 114.5% |
| **Guanaco-65B** | 65B | **41 GB** | **99.3%** |
| Guanaco-33B | 33B | 21 GB | 97.8% |
| Open Assistant | 33B | **66 GB** | 94.9% |
| Vicuna | 13B | 26 GB | 94.9% |
| Guanaco-7B | 7B | **5 GB** | ~88% |

Elo рейтинги (GPT-4 судья):
- GPT-4: 1348
- Guanaco-65B: 1022
- Guanaco-33B: 992
- ChatGPT: 966
- Guanaco-7B: 879

### Память при обучении

| Метод | LLaMA-65B |
|-------|-----------|
| Full FT 16-bit | >780 GB |
| LoRA 16-bit | ~350 GB |
| **QLoRA 4-bit** | **<48 GB** |

### Качество данных важнее объёма (§5)

OASST1 (9K samples) > FLAN v2 (450K samples) для chatbot performance.
MMLU performance ≠ chatbot performance → benchmark selection matters.

## Ablations

- r (LoRA rank) не влияет на производительность значительно (критично — **покрытие слоёв**)
- NF4 > FP4 > Int4 при одинаковом битрейте
- DQ не снижает качество, экономит ~3 GB для 65B

## My notes

- QLoRA сделал fine-tuning 65B моделей доступным на одной потребительской GPU — революция в open-source LLM сообществе.
- Главный практический вывод: применяй LoRA ко ВСЕМ linear layers, а не только Wq/Wv.
- NF4 → стал стандартом в bitsandbytes library; де факто стандарт для 4-bit QLoRA.
- Guanaco-7B в 5GB → вписывается в смартфоны; Guanaco-33B в 21GB → на RTX 3090/4090.
- GPT-4 как судья в Elo: коррелирует с human evaluation но имеет ordering bias → рекомендуют усреднять по обоим порядкам.
- QLoRA + Alpaca/OASST1 = шаблон для создания instruction-tuned моделей в 2023.
