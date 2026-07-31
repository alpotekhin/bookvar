---
title: Seq2Seq и bottleneck фиксированного вектора
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
previous: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]"
primary_sources:
  - https://arxiv.org/abs/1406.1078
  - https://arxiv.org/abs/1409.3215
---

# Seq2Seq и bottleneck фиксированного вектора

Классификатор выдаёт один ответ, а переводчик должен породить
последовательность, длина которой не обязана совпадать с длиной входа. Нельзя
просто сопоставить первой позиции источника первую позицию перевода: порядок
слов различается, а одному слову могут соответствовать несколько. Seq2Seq
разделяет задачу на чтение и генерацию: encoder строит представление входа,
decoder условно генерирует выход слева направо.

## Условная языковая модель

Для источника $x_{1:n}$ и перевода $y_{1:m}$ модель задаёт

$$
p(y_{1:m}\mid x_{1:n})=
\prod_{t=1}^{m}p(y_t\mid y_{<t},x_{1:n}).
$$

Рекуррентный encoder читает вход:

$$h_j=f_{\mathrm{enc}}(x_j,h_{j-1}),\qquad c=h_n.$$

Финальное состояние $c$ служит фиксированным контекстом. Decoder начинает с
состояния, полученного из $c$, и на каждом шаге использует предыдущий токен:

$$s_t=f_{\mathrm{dec}}(E_y[y_{t-1}],s_{t-1},c),$$

$$p(y_t\mid y_{<t},x)=\operatorname{softmax}(W_os_t+b_o).$$

В конкретной реализации $c$ может инициализировать скрытое состояние decoder,
подаваться на каждый шаг или делать и то и другое. У многослойной LSTM
интерфейс включает пары $(h,c)$ для каждого слоя; фраза «один вектор» описывает
фиксированный по длине канал, а не обязательно один одномерный тензор.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq.svg]]

*Recurrent encoder-decoder без attention: encoder завершает чтение, после чего
decoder генерирует токены, начиная с `<bos>` и заканчивая `<eos>`. Между ними —
фиксированный контекст, не зависящий размером от длины входа. Источник: Zhang et
al., [Dive into Deep Learning,
§10.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html), CC BY-SA 4.0.*

## Формы на одном примере

Разбор форм и сдвига `<bos>`/`<eos>` адаптирован из пошаговой реализации D2L
§10.7; конкретные размеры добавлены как проверяемый пример.

Пусть пакет содержит 32 английских предложения длиной не более 12 токенов,
embedding имеет размер 256, а скрытое состояние двухслойной GRU — 512. Тогда
вход encoder имеет форму `[32, 12, 256]`, его последовательность состояний —
`[32, 12, 512]`, а финальные состояния слоёв — `[2, 32, 512]`. В раннем Seq2Seq
decoder получает именно последние состояния, хотя encoder уже вычислил 12
позиционных представлений.

Для пары

> `black cat sleeps` → `чёрная кошка спит`

обучающие входы decoder будут `<bos>, чёрная, кошка, спит`, а цели —
`чёрная, кошка, спит, <eos>`. Сдвиг на один шаг реализует условную языковую
модель; padding маскируется при суммировании cross-entropy.

```python
enc_outputs, enc_state = encoder(src_ids, src_lengths)
dec_input = tgt_ids[:, :-1]       # начинается с <bos>
targets = tgt_ids[:, 1:]          # заканчивается <eos>
logits, _ = decoder(dec_input, enc_state)
loss = cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    ignore_index=PAD,
)
```

Во время обучения decoder обычно получает истинный предыдущий токен — teacher
forcing. Во время применения истинного продолжения нет: следующий вход берётся
из собственного предсказания. Это расхождение называют exposure bias. Оно не
тождественно информационному bottleneck: первое относится к режиму генерации,
второе — к интерфейсу между encoder и decoder.

## Почему фиксированный контекст становится узким местом

При росте $n$ количество исходной информации растёт, но размер $c$ остаётся
неизменным. Далёкий исходный токен должен сначала сохраниться через всю цепочку
encoder, а затем косвенно влиять на много шагов decoder. Возникают два длинных
пути: временной путь внутри encoder и путь от единственного контекста через
генерацию. LSTM улучшает перенос градиента, но не расширяет канал.

Это не утверждение, что фиксированный вещественный вектор имеет строго конечное
число смыслов. Практическое ограничение появляется из конечной точности,
размерности, шума оптимизации и необходимости сделать представление удобным для
decoder. Sutskever et al. показали, что большой LSTM encoder-decoder уже способен
на сильный перевод; изменение порядка исходных слов сокращало вычислительное
расстояние между соответствующими начальными словами. Но качество ранних
encoder-decoder систем ухудшалось на длинных предложениях, что мотивировало
прямой доступ к позиционным состояниям.

## Декодирование — не обучение

На шаге применения greedy decoding выбирает
$y_t=\arg\max_v p(v\mid y_{<t},x)$. Локально лучший токен не обязан давать
наиболее вероятную целую последовательность. Beam search хранит $B$ лучших
частичных гипотез по сумме логарифмов вероятностей и расширяет их параллельно.
Нужны завершение по `<eos>` и нормализация длины: необработанная сумма логарифмов
часто предпочитает короткие ответы.

```python
beams = [([BOS], 0.0, init_state)]
for _ in range(max_len):
    candidates = expand_each_beam(beams)  # добавить log p(token)
    beams = top_k(candidates, k=beam_size, length_normalize=True)
    if all(seq[-1] == EOS for seq, _, _ in beams):
        break
```

Beam search не исправляет потерянную encoder-информацию: он лишь лучше ищет в
распределении, которое уже задала модель.

## Мост к attention

Переход адаптирован из курса Lena Voita «Seq2seq and Attention»: сохранить все
encoder states и строить новый взвешенный контекст на каждом шаге decoder.

Естественный выход — сохранить не только $h_n$, а все состояния
$h_1,\ldots,h_n$ и позволить decoder на каждом шаге строить свой контекст. При
генерации слова «кошка» он сможет сильнее опереться на состояние у `cat`, а при
генерации «спит» — на `sleeps`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq-attention.svg]]

*Seq2Seq с attention. В отличие от предыдущей схемы, decoder получает не только
последнее состояние encoder: на каждом выходном шаге он обращается ко всей
последовательности encoder states. Проследите стрелки от верхнего ряда к
decoder — набор доступной памяти один, но веса доступа меняются вместе с
генерируемой позицией. Источник: Zhang et al., [Dive into Deep Learning,
§11.4](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html),
CC BY-SA 4.0.*

Attention формализует это как взвешенную сумму позиционных состояний. Тем самым
интерфейс перестаёт быть фиксированным вектором: память encoder растёт с длиной
входа, а путь от нужной позиции к текущему ответу становится коротким.

Для шага decoder $t$:

$$
e_{tj}=a(s_{t-1},h_j),\qquad
\alpha_{tj}=\frac{\exp e_{tj}}{\sum_{k=1}^{n}\exp e_{tk}},
$$

$$
c_t=\sum_{j=1}^{n}\alpha_{tj}h_j.
$$

В старой модели один и тот же $c=h_n$ обслуживал все выходные позиции. Теперь
$c_t$ меняется: запросом служит состояние decoder, а значениями — состояния
encoder. Если строка attention weights для слова `кошка` равна
$(0.05,0.85,0.10)$, контекст почти совпадает с представлением позиции `cat`;
для следующего слова веса могут переместиться к `sleeps`.

| Интерфейс encoder → decoder | Память о входе | Путь от позиции источника к выходу |
|---|---|---|
| $c=h_n$ | один фиксированный контекст | через остаток encoder и decoder chain |
| $c_t=\sum_j\alpha_{tj}h_j$ | все позиционные состояния | прямой взвешенный доступ на каждом шаге |

Attention не отменяет autoregressive decoder и не устраняет teacher forcing.
Он меняет только канал доступа к источнику. Это различие станет важным дальше:
Transformer сохранит attention, но заменит рекуррентный encoder и decoder
параллельно вычисляемыми блоками.

Следующая глава выводит этот механизм из bottleneck, а затем показывает, как
self-attention устраняет обязательную рекуррентную цепочку при кодировании
известной последовательности.

## Ограничения и частые смешения

- Encoder-decoder — архитектурный паттерн, а Seq2Seq — способ моделировать
  преобразование последовательностей; ни один термин сам по себе не означает
  LSTM.
- Teacher forcing относится к обучению decoder, beam search — к поиску при
  инференсе, attention — к доступу к представлениям входа.
- Bidirectional encoder можно использовать, потому что весь источник известен;
  autoregressive decoder не может читать будущие целевые токены.
- Padding не должен изменять финальное состояние и входить в потерю: нужны
  длины, packing или корректные маски.

## Источники и дальнейшее чтение

Условная факторизация, teacher forcing и beam search следуют SLP3 и Stanford
CS224N Notes 6; вычислительная схема — D2L §10.7; объяснение fixed-vector
bottleneck и переход к attention — курсу Lena Voita. Архитектурные детали
проверены по Cho et al. и Sutskever et al.

- Cho et al., [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078), 2014 — ранний encoder-decoder и GRU.
- Sutskever, Vinyals, Le, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), 2014 — LSTM Seq2Seq, reversing source order и beam search.
- Jurafsky, Martin, [Speech and Language Processing, ch. 13](https://web.stanford.edu/~jurafsky/slp3/13.pdf) — машинный перевод и encoder-decoder.
- Stanford CS224N, [Notes 6: Neural Machine Translation, Seq2Seq and Attention](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/cs224n-2023-notes06-NMT_seq2seq_attention.pdf).
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-recurrent-neural-networks.pdf|Stanford CS230 — Recurrent Neural Networks Cheatsheet]] — компактная англоязычная сводка по Seq2Seq, beam search и attention.
- Zhang et al., [Dive into Deep Learning, §10.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html), CC BY-SA 4.0.
- Lena Voita, [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — визуальная линия от fixed-vector bottleneck к attention.

**Назад:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]
