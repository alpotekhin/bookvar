---
title: "nanoGPT"
aliases: [nanogpt, nano-GPT]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
courses: []
sources:
  - "[GitHub — karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)"
  - "[GitHub — karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)"
  - "[GitHub — karpathy/nanochat](https://github.com/karpathy/nanochat)"
  - "[Simon Willison — Running nanoGPT on MacBook M2](https://til.simonwillison.net/llms/nanogpt-shakespeare-m2)"
---

# nanoGPT

## What it is

[nanoGPT](https://github.com/karpathy/nanoGPT) -- образовательная реализация GPT от Andrej Karpathy. Самый минималистичный репозиторий для обучения/файнтюнинга medium-sized GPT. Духовный наследник char-rnn (2015), но на Transformer-архитектуре.

> **Update Nov 2025:** nanoGPT deprecated в пользу [nanochat](https://github.com/karpathy/nanochat).

## Архитектура

Весь код -- два файла:
- `train.py` (~300 строк) -- training loop с DDP, gradient accumulation, mixed precision
- `model.py` (~300 строк) -- GPT model definition, может загружать веса GPT-2 от OpenAI

Зависимости: PyTorch, numpy, transformers, datasets, tiktoken, wandb

## Чему учит nanoGPT

### 1. Transformer training от нуля

- Полный цикл: data preparation -> tokenization -> training -> sampling
- Можно обучить char-level GPT на Shakespeare за **3 минуты на GPU** (val loss 1.47)
- Или воспроизвести GPT-2 (124M) на OpenWebText за **4 дня на 8xA100**

### 2. Практические аспекты масштабирования

| Режим | Конфигурация | Результат |
|-------|-------------|-----------|
| CPU/MacBook | 4 layers, 4 heads, 128 embd, 2K iters | val loss ~1.88 (3 мин) |
| Single GPU | 6 layers, 6 heads, 384 embd, context 256 | val loss ~1.47 (3 мин) |
| 8xA100 | GPT-2 124M config, OpenWebText | val loss ~2.85 (4 дня) |

### 3. Baselines GPT-2

| Model | Params | Val loss (OWT) |
|-------|--------|----------------|
| gpt2 | 124M | 3.12 |
| gpt2-medium | 350M | 2.84 |
| gpt2-large | 774M | 2.67 |
| gpt2-xl | 1558M | 2.54 |

Domain gap: GPT-2 обучался на закрытом WebText; fine-tune на OpenWebText дает ~2.85 для 124M.

### 4. Fine-tuning

- Fine-tune GPT-2 на Shakespeare за минуты на одном GPU
- Параметры: меньший LR, короче обучение, `init_from='gpt2'`
- Результат: правдоподобный Shakespeare-стиль (значительно лучше char-level)

### 5. Инженерные практики

- **torch.compile()** (PyTorch 2.0): ускорение ~2x (250ms -> 135ms/iter)
- **DDP** (Distributed Data Parallel) для multi-GPU
- **MPS** на Apple Silicon: 2-3x ускорение на MacBook
- **Mixed precision** через PyTorch AMP
- **Gradient accumulation** для эффективного batch size

## Код: пошаговый разбор ключевых компонентов

### model.py: что внутри

**CausalSelfAttention class (~40 строк):**
```python
# Ключевые элементы:
self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # Q, K, V в одной проекции
self.c_proj = nn.Linear(n_embd, n_embd)       # output projection
# Flash Attention (PyTorch 2.0+):
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
**Что это учит:** attention — это просто три линейных проекции + матричное умножение + softmax. Flash Attention — одна строка в PyTorch 2.0.

**MLP class (~10 строк):**
```python
self.c_fc = nn.Linear(n_embd, 4 * n_embd)   # expansion
self.gelu = nn.GELU()
self.c_proj = nn.Linear(4 * n_embd, n_embd)  # projection
```
**Что это учит:** FFN в Transformer — это два линейных слоя с нелинейностью. Expansion ratio 4x — convention из оригинальной статьи.

**Block class (~5 строк):**
```python
x = x + self.attn(self.ln_1(x))   # pre-norm + attention + residual
x = x + self.mlp(self.ln_2(x))    # pre-norm + FFN + residual
```
**Что это учит:** pre-normalization (LayerNorm перед sublayer, не после) + residual connections. Два паттерна, определяющих стабильность обучения глубоких моделей.

**GPT class — generate method:**
```python
for _ in range(max_new_tokens):
    logits, _ = self(idx_cond)          # forward pass
    logits = logits[:, -1, :] / temp    # take last position, scale
    probs = F.softmax(logits, dim=-1)   # convert to probabilities
    idx_next = torch.multinomial(probs, 1)  # sample
    idx = torch.cat((idx, idx_next), dim=1) # append
```
**Что это учит:** авторегрессивная генерация — это цикл forward pass → sample → append. Temperature scaling контролирует «креативность».

### train.py: что внутри

**Ключевые компоненты:**

1. **Learning Rate Schedule:** cosine decay с warmup — стандарт для LLM training
2. **Gradient Accumulation:** имитация большого batch size на маленьком GPU
3. **Mixed Precision:** `torch.amp.autocast` для ускорения через float16/bfloat16
4. **DDP (Distributed Data Parallel):** multi-GPU обучение через `torch.distributed`
5. **Eval Loop:** периодическая оценка val loss для мониторинга overfitting
6. **Checkpointing:** сохранение model + optimizer state для resume

**Что это учит:** production-quality training loop — это ~300 строк boilerplate. Все LLM training loops следуют этому шаблону.

## Training curves: что наблюдается

### Shakespeare char-level (single GPU, 3 min)

```
Step 0:    train loss 4.17, val loss 4.17  (random)
Step 250:  train loss 2.49, val loss 2.50  (learning structure)
Step 500:  train loss 2.05, val loss 2.08  (learning words)
Step 1000: train loss 1.68, val loss 1.72  (learning grammar)
Step 2000: train loss 1.48, val loss 1.52  (learning style)
Step 5000: train loss 1.31, val loss 1.47  (diminishing returns)
```

**Наблюдения:**
- Loss падает быстро в начале (low-hanging fruit: пробелы, пунктуация, частые слова)
- Gap train/val растёт — начало overfitting на маленьком датасете
- Plateau ~1.47 val loss — «потолок» для char-level модели на Shakespeare

### GPT-2 124M на OpenWebText (8xA100, 4 days)

```
Step 0:     train loss ~11.0 (random, vocab 50K → log(50K) ≈ 10.8)
Step 1K:    train loss ~5.5  (learning common tokens)
Step 10K:   train loss ~3.8  (learning phrases)
Step 100K:  train loss ~3.1  (learning patterns)
Step 300K:  train loss ~2.85 (approaching GPT-2 baseline)
```

**Наблюдения:**
- Initial loss ~log(vocab_size) — модель начинает с uniform distribution
- Первые 1K шагов — максимальный прогресс (learning token frequencies)
- После 100K шагов — slow improvement, каждый 0.1 loss стоит всё дороже

## build-nanogpt: видео-лекция

В 2024 Karpathy выпустил [build-nanogpt](https://github.com/karpathy/build-nanogpt) — пошаговую видео-лекцию (4 часа), где GPT-2 строится **с нуля** commit за commit:

| Commit | Содержание | Чему учит |
|--------|-----------|-----------|
| 1 | Empty file → simple bigram model | Baseline: predict next token from previous |
| 2 | Add self-attention | Контекст: модель «видит» больше одного токена |
| 3 | Add multi-head attention | Параллельные «точки зрения» |
| 4 | Add feedforward network | Non-linear transformation |
| 5 | Add residual connections | Градиенты через глубокие стеки |
| 6 | Add layer norm | Стабилизация обучения |
| 7 | Scale up → GPT-2 config | От toy model к real architecture |
| 8 | Add FlashAttention | 2x speedup одной строкой |
| 9 | Add torch.compile | 2x speedup ещё |
| 10 | Reproduce GPT-2 124M | Финал: ~1 час, ~$10 |

**Ключевой вывод:** GPT-2 (124M) в 2024 году воспроизводится за **~1 час и ~$10**. То, что в 2019 стоило OpenAI значительных ресурсов, стало trivial задачей.

## nanochat: следующая итерация

В ноябре 2025 Karpathy deprecated nanoGPT в пользу **nanochat** — реализации, которая идёт дальше pre-training и включает **полный chat pipeline**:

- Pre-training → SFT → Chat capabilities
- Современные оптимизации (FlashAttention 2, better tokenizer)
- Цель: **лучший chatbot за $100**

## Ключевые уроки

1. **GPT — это просто**: ~600 строк кода для полной реализации. Вся «магия» LLM — в масштабе данных и compute, не в сложности кода.
2. **Данные решают**: тот же код, разные данные → от Shakespeare до GPT-2 level
3. **Fine-tuning дешёвый**: pretrained GPT-2 + несколько минут → качественные результаты
4. **Tokenization matters**: char-level (Shakespeare) vs BPE (OpenWebText) — принципиально разные результаты и скорость обучения
5. **Reproducibility**: фиксированные seeds, wandb logging, чёткие конфиги
6. **PyTorch 2.0 revolution**: torch.compile + Flash Attention = ~4x speedup без изменения модели
7. **Scaling intuition**: наблюдая training curves, формируется интуиция о том, как loss зависит от model size, data size, compute

## How it works (char-level Shakespeare)

```bash
# 1. Prepare data
python data/shakespeare_char/prepare.py  # -> train.bin, val.bin

# 2. Train
python train.py config/train_shakespeare_char.py

# 3. Sample
python sample.py --out_dir=out-shakespeare-char
```

Выход:
```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.
```

## Связанные заметки

- [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]] -- архитектура, которую воспроизводит nanoGPT
- [[02 Areas/ML & DL/Papers/Karpathy — Unreasonable Effectiveness of RNNs|Karpathy — RNN Effectiveness]] -- предшественник (char-rnn, 2015)
- [[02 Areas/ML & DL/Papers/Karpathy — Recipe for Training Neural Networks|Karpathy — Recipe for Training NNs]] -- методология обучения
- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization / BPE]]
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]

## Дополнительные ресурсы

- [GitHub — nanoGPT](https://github.com/karpathy/nanoGPT) — оригинальный репозиторий
- [GitHub — build-nanogpt](https://github.com/karpathy/build-nanogpt) — видео-лекция: GPT-2 с нуля за 4 часа
- [GitHub — nanochat](https://github.com/karpathy/nanochat) — следующая итерация: полный chat pipeline
- [Simon Willison — Training nanoGPT on my blog](https://til.simonwillison.net/llms/training-nanogpt-on-my-blog) — практический пример custom training
