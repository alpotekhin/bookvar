---
title: "LLaMA"
aliases: [Llama, LLaMA-1]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2]]"
courses: []
sources:
  - "[Sebastian Raschka — The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)"
  - "[Lilian Weng — The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)"
  - "[d2l.ai — Large-Scale Pretraining with Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html)"
---

# LLaMA — Open and Efficient Foundation Language Models

## Зачем это нужно: inference-budget-optimal scaling

К началу 2023 года ландшафт LLM выглядел так:
- **GPT-3** (175B, OpenAI) — закрытый, дорогой в inference
- **PaLM** (540B, Google) — закрытый, ещё дороже
- **Chinchilla** (70B, DeepMind) — показал, что меньшая модель на большем количестве данных лучше, но тоже закрытый
- **OPT** (175B, Meta) — открытый, но не конкурентоспособный с PaLM/Chinchilla

**Hoffmann et al. (2022) — Chinchilla scaling laws** установили: для фиксированного training compute бюджета оптимально обучать модель, где число параметров и число токенов масштабируются пропорционально. Для 175B модели оптимум — ~3.5T токенов (GPT-3 обучен только на 300B — **в 10x меньше оптимума**).

Но Touvron et al. (2023) заметили фундаментальный недостаток Chinchilla: **она оптимизирует training budget, а не inference budget**. На практике модель обучается один раз, но используется миллиарды раз. Поэтому **лучше обучить меньшую модель на большем количестве токенов** — она дешевле при inference.

Результат: **LLaMA-13B превосходит GPT-3 (175B) на большинстве бенчмарков** — при 13x меньшем размере и возможности inference на **одном GPU**.

## Архитектура: три ключевых улучшения

LLaMA = стандартный decoder-only [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] + три модификации, каждая заимствованная из успешных предшественников:

### 1. Pre-normalization с RMSNorm (от GPT-3)

Оригинальный Transformer использует **Post-LN**: $\text{LayerNorm}(x + \text{Sublayer}(x))$. GPT-3 переключился на **Pre-LN**: $x + \text{Sublayer}(\text{Norm}(x))$ — это стабильнее при большом масштабе, т.к. residual stream не проходит через нормализацию.

LLaMA заменяет LayerNorm на **RMSNorm** (Zhang & Sennrich, 2019):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}} \cdot \gamma$$

**Зачем RMSNorm вместо LayerNorm?** RMSNorm убирает mean-centering (вычитание среднего) — остаётся только масштабирование по RMS. Это быстрее (меньше операций) и эмпирически работает не хуже. Нет bias correction членов ($\beta = 0$).

### 2. SwiGLU activation (от PaLM)

Оригинальный Transformer FFN: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$ с hidden dim = $4d$.

LLaMA заменяет ReLU на **SwiGLU** (Shazeer, 2020):

$$\text{SwiGLU}(x, W, V, W_2) = (\text{Swish}(xW) \odot (xV)) \cdot W_2$$

где $\text{Swish}(x) = x \cdot \sigma(x)$, а $\odot$ — поэлементное умножение.

**Зачем SwiGLU?** GLU-варианты (Gated Linear Units) эмпирически показывают лучшее качество, чем ReLU/GeLU при том же compute. Механизм: один путь ($xW$) генерирует «контент», другой ($xV$) — «гейт», определяющий какую часть контента пропустить. Hidden dim = $\frac{2}{3} \times 4d$ (вместо $4d$) для компенсации дополнительной матрицы.

### 3. Rotary Positional Embeddings — RoPE (от GPT-Neo)

Оригинальный Transformer: абсолютные позиционные эмбеддинги, суммируемые с токенными. BERT, GPT-2: обучаемые абсолютные.

LLaMA убирает абсолютные позиции и добавляет **RoPE** (Su et al., 2021) **на каждом слое**:

$$f(q, m) = q_m \cdot e^{i \cdot m \cdot \theta}$$

где $m$ — позиция, $\theta$ — частоты по размерностям.

**Зачем RoPE?**
- Кодирует **относительные** позиции (dot product $q_m \cdot k_n$ зависит только от $m - n$)
- Применяется на **каждом слое** (а не только на входе) — позиционная информация не затухает
- Лучше обобщается на длинные контексты (можно экстраполировать за пределы training context window)

### Сравнительная таблица: оригинальный Transformer vs LLaMA

| Компонент | Original Transformer | LLaMA |
|-----------|---------------------|-------|
| Normalization | Post-LayerNorm | **Pre-RMSNorm** |
| Activation | ReLU | **SwiGLU** |
| FFN hidden dim | $4d$ | $\frac{2}{3} \times 4d$ |
| Positional encoding | Sinusoidal (вход) | **RoPE (каждый слой)** |
| Attention | Full bidirectional (encoder) | Causal only |
| Architecture | Encoder-Decoder | **Decoder-only** |

Эта тройка (**RMSNorm + SwiGLU + RoPE**) стала **де-факто стандартом** для всех последующих открытых LLM: LLaMA 2, Mistral 7B, Falcon, Qwen, Yi, DeepSeek.

## Размеры моделей (Table 2 из статьи)

| Параметры | $d_{\text{model}}$ | Heads | Layers | Learning Rate | Batch Size | Tokens |
|-----------|---------------------|-------|--------|--------------|------------|--------|
| **7B** | 4096 | 32 | 32 | 3e-4 | 4M | **1T** |
| **13B** | 5120 | 40 | 40 | 3e-4 | 4M | **1T** |
| **33B** | 6656 | 52 | 60 | 1.5e-4 | 4M | **1.4T** |
| **65B** | 8192 | 64 | 80 | 1.5e-4 | 4M | **1.4T** |

**Обрати внимание на tokens**: 1T-1.4T — это **в 3-5x больше**, чем Chinchilla-optimal для этих размеров. Суть LLaMA: **перетренировать** маленькую модель, чтобы она была дешевле при inference.

## Training Data — только публичные данные

| Источник | Доля | Эпохи | Объём |
|----------|------|-------|-------|
| CommonCrawl (CCNet filtered) | 67% | 1.10 | 3.3 TB |
| C4 (Raffel et al., 2020) | 15% | 1.06 | 783 GB |
| Github (Apache/BSD/MIT) | 4.5% | 0.64 | 328 GB |
| Wikipedia (20 языков) | 4.5% | 2.45 | 83 GB |
| Gutenberg + Books3 | 4.5% | 2.23 | 85 GB |
| ArXiv | 2.5% | 1.06 | 92 GB |
| StackExchange | 2% | 1.03 | 78 GB |

**~1.4T токенов** после токенизации. Токенайзер: **BPE через SentencePiece** (числа разбиваются на отдельные цифры, unknown UTF-8 — на байты).

**Критически важно**: в отличие от GPT-3, PaLM, Chinchilla — **только публичные данные**. Это сделало модель совместимой с open sourcing и позволило исследовательскому сообществу воспроизводить и развивать результаты.

## Обучение: оптимизатор и эффективность

**Оптимизатор**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay = 0.1, gradient clipping = 1.0)

**LR schedule**: cosine decay до 10% от максимального LR, 2000 warmup шагов.

### Efficient Implementation

- **Flash Attention** (causal) через xformers — не хранит attention weights, не вычисляет masked scores
- **Gradient checkpointing**: сохраняем только дорогие активации (выходы linear layers), пересчитываем остальное при backward pass
- **Model + sequence parallelism** (Korthikanti et al., 2022)
- Overlap compute и communication (all_reduce)

**Скорость**: ~380 tokens/sec/GPU на 2048 A100 80GB. Training LLaMA-65B на 1.4T токенов ≈ **21 день**.

## Ключевые результаты

### Common Sense Reasoning (Table 3, zero-shot)

| Модель | Params | BoolQ | PIQA | HellaSwag | WinoGrande |
|--------|--------|-------|------|-----------|-----------|
| GPT-3 | 175B | 60.5 | 81.0 | 78.9 | 70.2 |
| Chinchilla | 70B | 83.7 | 81.8 | 80.8 | 74.9 |
| PaLM | 540B | 88.0 | 82.3 | 83.4 | 81.1 |
| **LLaMA-13B** | **13B** | 78.1 | 80.1 | 79.2 | 73.0 |
| **LLaMA-65B** | **65B** | **85.3** | **82.8** | **84.2** | **77.0** |

**LLaMA-13B превосходит GPT-3 (175B)** на HellaSwag, WinoGrande, ARC-e, ARC-c — при **13x меньшем размере**.

**LLaMA-65B** конкурентна с Chinchilla-70B и приближается к PaLM-540B (который в 8x больше).

### Closed-Book QA (NaturalQuestions, Table 4)

| Модель | 0-shot | 1-shot | 5-shot | 64-shot |
|--------|--------|--------|--------|---------|
| GPT-3 175B | 14.6 | 23.0 | — | 29.9 |
| Chinchilla 70B | 16.6 | — | 31.5 | 35.5 |
| PaLM 540B | 21.2 | 29.3 | — | 39.6 |
| **LLaMA-13B** | 20.1 | 23.4 | 28.1 | 31.9 |
| **LLaMA-65B** | 23.8 | 31.0 | 35.0 | **39.9** |

LLaMA-65B few-shot **превосходит PaLM-540B** на NaturalQuestions 64-shot.

### Code Generation (HumanEval, pass@1)

| Модель | 0-shot |
|--------|--------|
| PaLM 62B | 15.9% |
| PaLM-Coder 62B | 36.0% |
| **LLaMA-13B** | 15.8% |
| **LLaMA-65B** | **23.7%** |

Включение Github кода в training data значительно помогает — LLaMA-13B конкурирует с PaLM-62B на code generation.

## Inference-budget vs Training-budget

Это центральная идея LLaMA, часто неправильно понимаемая:

**Chinchilla** оптимизирует: «какую модель обучить за X FLOPs обучения?» Ответ: 70B на 1.4T tokens (для бюджета ~$10^{24}$ FLOPs).

**LLaMA** оптимизирует: «какую модель **использовать** при inference за Y FLOPs/запрос?» Ответ: 13B, обученную на 1T tokens (больше, чем Chinchilla-optimal) — потому что 13B **в 13x дешевле** при inference, чем 175B.

Формально: пусть $C_{\text{train}}$ — стоимость обучения, $C_{\text{inference}} \times N$ — стоимость $N$ inference запросов. При достаточно большом $N$ (миллиарды запросов для production LLM) суммарная стоимость доминируется inference. Поэтому лучше потратить больше на training (обучить дольше), чтобы сэкономить на inference (меньшая модель).

## Влияние: open-weight революция

LLaMA запустила цепную реакцию в открытом AI-сообществе:

1. **Fine-tuning взрыв**: Alpaca (Stanford, $600), Vicuna, WizardLM, CodeLlama — десятки fine-tuned вариантов в первые месяцы
2. **Quantization**: GPTQ, GGML, AWQ — методы запуска 13B-65B моделей на consumer hardware
3. **Архитектурный стандарт**: RMSNorm + SwiGLU + RoPE стали default для новых LLM
4. **Открытая конкуренция**: Mistral 7B, Falcon, Qwen, Yi — все строят на наследии LLaMA

### Эволюция LLaMA

| Версия | Год | Ключевые изменения |
|--------|-----|-------------------|
| LLaMA 1 | Feb 2023 | Baseline, 7B-65B, public data only |
| LLaMA 2 | Jul 2023 | 70B, GQA, extended context (4K), RLHF chat version, 2T tokens |
| LLaMA 3 | Apr 2024 | 8B/70B, 15T tokens, 128K context, multilingual |
| LLaMA 3.1 | Jul 2024 | 405B, native tool use, 128K context |

## Key papers

- [[02 Areas/ML & DL/Papers/LLaMA]] — оригинал (Touvron et al., Feb 2023)
- [[02 Areas/ML & DL/Papers/LLaMA 2]] — LLaMA 2 (70B, GQA, RLHF chat, Jul 2023)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектурная парадигма
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — базовая архитектура
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — Chinchilla laws и их расширение
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — alignment в LLaMA 2 Chat
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA 2|LLaMA 2]] — наследник
- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — использует тот же архитектурный стек
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — используется при обучении

## Дополнительные ресурсы

- [Sebastian Raschka — The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — детальное сравнение архитектур GPT/LLaMA/Mistral
- [Lilian Weng — The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — обзор вариантов Transformer
- [Meta AI — LLaMA repo](https://github.com/facebookresearch/llama) — официальный код
- [d2l.ai — Large-Scale Pretraining](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html) — контекст pre-training
