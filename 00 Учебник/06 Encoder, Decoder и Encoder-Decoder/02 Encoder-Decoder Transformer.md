---
title: Encoder-Decoder Transformer
type: textbook-chapter
status: legacy
last_updated: 2026-07-20
primary_sources: [https://arxiv.org/abs/1706.03762]
---

# Encoder-Decoder Transformer

При переводе исходное предложение и перевод играют разные роли. Исходный текст
можно прочитать целиком; перевод приходится порождать слева направо. Transformer
решает эти две задачи разными стеками: encoder строит контекстные представления
всех исходных токенов, decoder создаёт выход и на каждом шаге обращается к этим
представлениям.

Encoder принимает `src: [B,S]`, добавляет позиционную информацию и многократно применяет полный self-attention и FFN. Результат — память $H\in\mathbb R^{B\times S\times d}$. Decoder обрабатывает сдвинутую вправо цель `tgt: [B,T]`: causal self-attention не видит будущих target-токенов, а cross-attention использует

$$Q=Y W_Q,\qquad K=H W_K,\qquad V=H W_V.$$

Поэтому scores имеют shape `[B,h,T,S]`, а выход cross-attention — `[B,T,d]`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/The_transformer_encoder_decoder_stack.png]]

*Стек encoder–decoder по визуальному разбору Jay Alammar: выход каждого
encoder-блока передаётся каждому decoder-блоку как память для cross-attention.
Источник: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
автор Jay Alammar, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

На схеме выход encoder — не одно резюме предложения, а таблица из $S$ строк.
Каждый decoder-блок может заново выбрать нужные строки этой таблицы. Так исчезает
фиксированный bottleneck старого recurrent seq2seq: при переводе очередного
слова decoder не обязан довольствоваться одним вектором, созданным после чтения
всего входа.

## Три вида attention в одной модели

В encoder работает полный self-attention: каждый исходный токен видит каждый.
В decoder первый attention причинный: позиция перевода видит только уже
известный префикс. Третий слой — cross-attention — получает запросы из decoder,
но ключи и значения из encoder memory. Поэтому именно cross-attention связывает
порождаемое слово с релевантными словами источника.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoding_2.gif]]

*Пошаговое декодирование в [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/):
готовый префикс возвращается на вход decoder, тогда как encoder memory остаётся
неизменной. Автор — Jay Alammar, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

## Что происходит при обучении и генерации

При teacher forcing все позиции правильного перевода известны и обучаются
параллельно:

$$\mathcal L=-\sum_{t=1}^{T}\log p(y_t\mid y_{<t},x).$$

На инференсе истинного префикса нет: decoder генерирует токен, добавляет его к префиксу и повторяет шаг. Encoder memory вычисляется один раз; KV decoder self-attention кэшируются, а cross-attention K/V можно также подготовить заранее.

Модель получает перевод, сдвинутый на один токен: перед первым словом стоит
`BOS`, а последнее слово становится только целевой меткой. Причинная маска не
даёт позиции подсмотреть правильные слова справа. На генерации правильного
префикса уже нет: выбранный моделью токен добавляется ко входу следующего шага,
как на анимации выше.

Внутри decoder причинная маска имеет размер `[T,T]`, а маска padding источника
расширяется до `[B,1,T,S]`. Это разные маски: первая закрывает будущие токены
перевода, вторая — пустые позиции входа. Padding перевода, кроме того, исключают
из функции потерь.

## Как из состояния получается слово

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoder_output_softmax.png]]

*Последнее состояние decoder проецируется в один logit на токен словаря, после
чего softmax задаёт распределение следующего токена. Источник: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Параллельность обучения относится к вычислению logits для уже известных
целевых токенов. Выбор следующего токена при генерации всё равно последователен.

Greedy decoding выбирает argmax локально и может пропустить лучшую последовательность. Beam search хранит несколько префиксов и сравнивает суммы log-probabilities, обычно с length normalization. Sampling нужен для разнообразия, но для перевода может ухудшать точность. Стратегия decode не является частью обученных весов.

В исходном Transformer encoder–decoder улучшил WMT translation при существенно большей параллельности обучения по сравнению с recurrent системами. Paper одновременно ввёл несколько решений — multi-head attention, sinusoidal positions, residual/LayerNorm и FFN — поэтому результат нельзя приписать одному cross-attention.

## Где архитектура выигрывает и где платит

Она особенно уместна, когда source известен целиком и output должен быть условной последовательностью: перевод, summarization, transcription, structured generation. Encoder один раз строит двунаправленное представление; decoder не обязан тратить causal depth на «прочтение» source как части префикса.

Цена — два стека параметров и отдельная encoder memory. При длинном source
cross-attention читает много K/V на каждом шаге. Есть и расхождение между
обучением и применением: на обучении decoder всегда получает правильный префикс,
а при генерации — собственные предыдущие решения. Ошибка может перейти в
следующий шаг и изменить всё продолжение.

T5 и BART наследуют паттерн, но меняют pretraining objective и детали блока. Поэтому `encoder-decoder` описывает топологию, а не конкретный способ обучения.

## Источники

- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Jurafsky & Martin, SLP3, Transformers](https://web.stanford.edu/~jurafsky/slp3/)
- [D2L: The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
- [Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — наиболее ясная пошаговая схема потока encoder memory
