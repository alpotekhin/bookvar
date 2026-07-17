---
title: Decoder-only
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/2302.13971
---

# Decoder-only

**Decoder-only Transformer** моделирует следующий токен по уже известному префиксу:

$$p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).$$

```mermaid
flowchart LR
  P["Префикс x₁…xₜ"] --> M["Masked self-attention"]
  M --> L["Логиты следующего токена"]
  L --> S["Выбор xₜ₊₁"]
  S -. "добавить к префиксу" .-> P
```

## Главное отличие

Causal mask запрещает позиции `t` видеть будущие позиции. При обучении все позиции считаются параллельно с teacher forcing; при генерации токены появляются последовательно. Уже рассчитанные keys и values сохраняет [[02 Areas/ML & DL/01 Справочник/Inference/KV-cache|KV-cache]].

Современный блок обычно сочетает pre-normalization, masked [[02 Areas/ML & DL/01 Справочник/Attention/Self-Attention|self-attention]], residual stream и gated FFN наподобие [[02 Areas/ML & DL/01 Справочник/FFN и MoE/SwiGLU|SwiGLU]]. Конкретные Llama, Qwen и DeepSeek различаются attention, FFN, позиционным кодированием и training recipe.

## Почему паттерн доминирует в LLM

- один objective подходит для любого текста;
- prompt, ответ и tool trace можно представить одной последовательностью;
- модель естественно генерирует продолжение произвольной длины;
- архитектура проста для масштабирования.

Цена — последовательный decode и отсутствие прямого двунаправленного просмотра внутри генерируемой части.

## Не путать

Decoder оригинального Transformer содержит masked self-attention **и cross-attention к encoder**. В decoder-only модели отдельного encoder и обязательного cross-attention нет.

## Источники

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Legacy: Decoder-only]]
