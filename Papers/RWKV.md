---
title: "RWKV: Reinventing RNNs for the Transformer Era"
url: https://arxiv.org/abs/2305.13048
authors: "Bo Peng, Eric Alcaide, Quentin Anthony et al."
year: 2023
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/rwkv/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
---

# RWKV: Reinventing RNNs for the Transformer Era

**Authors:** Bo Peng, Eric Alcaide, Quentin Anthony et al. (EleutherAI, множество университетов)
**Published:** 2023 (arXiv:2305.13048v2, Dec 2023)
**URL:** https://arxiv.org/abs/2305.13048

## TL;DR

RWKV (Receptance Weighted Key Value) -- архитектура, совмещающая параллелизуемый training трансформеров с O(Td) inference RNN. Модели обучены до 14B параметров на 330B токенов (The Pile) и показывают сопоставимое качество с Transformer-моделями аналогичного размера (Pythia, OPT, BLOOM). Это крупнейшая dense RNN, когда-либо обученная.

## Problem

Transformer-ы масштабируются хорошо, но inference стоит O(T^2 d) по времени и O(T^2 + Td) по памяти из-за self-attention. RNN экономны при inference (O(Td) / O(d)), но не параллелизуются при обучении и плохо масштабируются из-за vanishing gradient. Нужна архитектура, которая совмещает оба преимущества: параллельный training + линейный inference.

## Method

### Архитектура

Модель строится из стека residual-блоков, каждый содержит два sub-блока:

1. **Time-mixing** (аналог attention): использует четыре ключевых вектора:
   - **R (Receptance)** -- receiver прошлой информации
   - **W (Weight)** -- trainable positional decay vector (неотрицательный, e^{w_{t,i}} <= 1)
   - **K (Key)**, **V (Value)** -- аналогичны K, V в attention

   Все линейные проекции (R, K, V) применяются к линейной интерполяции текущего и предыдущего входа (**token shift**): `r_t = W_r * (mu_r * x_t + (1 - mu_r) * x_{t-1})`.

   Оператор **WKV** вычисляет взвешенную сумму values с channel-wise exponential decay по relative position (формула 16), вдохновлённую Attention Free Transformer (AFT). Ключевое отличие от AFT: W -- не парный matrix, а channel-wise vector, умноженный на relative position.

   Output gating: `o_t = W_o * (sigmoid(r_t) * wkv_t)`.

2. **Channel-mixing** (аналог FFN): `o'_t = sigmoid(r'_t) * (W'_v * max(k'_t, 0)^2)` -- squared ReLU activation.

### Два режима работы

- **Time-parallel mode** (training): WKV можно вычислять параллельно как матричные операции. Complexity O(BTd^2) -- доминируют матричные умножения W_r, W_k, W_v, W_o. Сам WKV update -- O(BTd), параллелизуем по batch и channel dimensions.
- **Time-sequential mode** (inference): формулируется рекуррентно, O(d) per step -- состояние передаётся от токена к токену.

### Оптимизации

- **Custom CUDA kernel** для WKV -- один kernel на GPU вместо последовательных операций.
- **Small Init Embedding** -- маленькая инициализация embedding + дополнительный LayerNorm для ускорения сходимости.
- **Custom initialization** -- большинство весов инициализируются нулями, без bias в линейных слоях.

### Обучение

Модели от 169M до 14B параметров обучены на одну эпоху (330B токенов) на The Pile. Adam без weight decay, bfloat16, context length 1024. Exponential decay learning rate + auxiliary loss из PaLM.

## Key Results

### Scaling Laws

RWKV следует тем же log-log linear scaling laws, что и Transformers (r^2 = 0.994 на Pareto-оптимальных точках). При экстраполяции на порядок дальше -- r^2 = 0.875.

### NLP benchmarks (FLOP-matched сравнение)

Сравнение с Pythia, OPT, BLOOM на 12 задачах (ARC, BoolQ, COPA, HellaSwag, LAMBADA, PIQA и др.):
- RWKV **сопоставим** с Transformer-моделями при том же compute budget.
- На некоторых задачах (e.g. ARC-Challenge) RWKV немного отстаёт, на других (e.g. PIQA) -- на уровне.

### Extended context

Finetuning с progressively увеличивающимся context length (1024 -> 2048 -> 4096 -> 8192) снижает test loss на Pile -- RWKV эффективно использует длинный контекст.

### Inference

- **Линейное** масштабирование cumulative time при генерации (vs квадратичное у Transformers).
- Значительное преимущество по memory при длинных последовательностях.

### Long-Range Arena

RWKV занимает второе место после S4 на LRA benchmark.

### Ограничения

- Linear attention может ограничивать recall точной информации из длинного контекста -- вся информация сжимается в один state vector.
- Повышенная чувствительность к prompt engineering: порядок информации в промпте критичен (F1 от 44.2% до 74.8% при переупорядочении).

## My notes
