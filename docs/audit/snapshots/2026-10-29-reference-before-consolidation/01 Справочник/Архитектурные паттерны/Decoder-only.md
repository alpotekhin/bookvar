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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-autoregression-2.gif]]

*GPT-2 обрабатывает префикс, выбирает следующий токен из выходного
распределения, добавляет его к последовательности и повторяет проход. Источник:
Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Анимация показывает генерацию, а не обучение. При генерации следующий токен
выбирается после обработки уже известного префикса; при обучении используется
готовая последовательность правильных токенов.

## Главное отличие

При teacher forcing входная позиция `t` предсказывает правильный токен `t+1`.
Каузальная маска разрешает внимание к позициям `j ≤ t`, включая текущую,
но скрывает будущие входные токены. Поэтому состояния и потери по всем
размеченным позициям можно вычислить параллельно без утечки ответа. При
генерации будущих токенов ещё нет: уже рассчитанные ключи и значения сохраняет
[[02 Areas/ML & DL/01 Справочник/Inference/KV-cache|KV-cache]].

Современный блок обычно сочетает pre-normalization, masked [[02 Areas/ML & DL/01 Справочник/Attention/Self-Attention|self-attention]], residual stream и gated FFN наподобие [[02 Areas/ML & DL/01 Справочник/FFN и MoE/SwiGLU|SwiGLU]]. Конкретные Llama, Qwen и DeepSeek различаются attention, FFN, позиционным кодированием и training recipe.

## Почему паттерн доминирует в LLM

- одна задача предсказания следующего токена подходит для разных видов текста;
- запрос, ответ и журнал вызовов инструментов можно представить одной последовательностью;
- модель естественно генерирует продолжение произвольной длины;
- архитектура проста для масштабирования.

Цена — последовательное декодирование и отсутствие прямого двунаправленного просмотра внутри генерируемой части.

## Не путать

Decoder оригинального Transformer содержит masked self-attention **и cross-attention к encoder**. В decoder-only модели отдельного encoder и обязательного cross-attention нет.

## Подробнее

Разницу трёх топологий и связь causal mask с обучением и генерацией развивает
глава [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Три архитектурных паттерна]].

## Источники

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA](https://arxiv.org/abs/2302.13971)
