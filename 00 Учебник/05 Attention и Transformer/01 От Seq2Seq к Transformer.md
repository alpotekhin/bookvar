---
title: От Seq2Seq к Attention
type: textbook-chapter
status: canonical
last_updated: 2026-07-18
previous: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN, LSTM и Seq2Seq]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]"
primary_sources:
  - https://arxiv.org/abs/1409.3215
  - https://arxiv.org/abs/1409.0473
  - https://arxiv.org/abs/1508.04025
---

# От Seq2Seq к Attention

> [!abstract] Идея главы
> Attention возник не как украшение Transformer. Сначала он решил конкретную
> проблему машинного перевода: decoder перестал получать всё исходное предложение
> через один фиксированный вектор и научился заново читать нужные части входа на
> каждом шаге генерации.

## Одна задача на всю главу

Пусть модель переводит:

> **The black cat sleeps on the sofa**  
> **Чёрный кот спит на диване**

Это conditional language modeling. Выход раскладывается по правилу цепочки:

$$
p(y\mid x)=\prod_{t=1}^{m}p(y_t\mid y_{<t},x).
$$

Когда модель выбирает слово «кот», ей особенно важен `cat`; когда выбирает
«чёрный» — `black`; когда генерирует «на диване» — `on the sofa`. Разным шагам
decoder нужна разная информация из одного и того же входа.

## 1. До attention: один вектор на всё предложение

В классическом recurrent encoder-decoder encoder читает вход слева направо:

$$
h_j=f_{\text{enc}}(h_{j-1},x_j).
$$

Последнее состояние $h_n$ используется как представление всего предложения.
Decoder начинает с него генерацию:

$$
s_t=f_{\text{dec}}(s_{t-1},y_{t-1},h_n).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bottleneck-min.png]]

*Lena Voita, «Seq2seq and Attention»: bottleneck старого encoder-decoder.
[Оригинальная глава курса](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html).*

Схема проста и элегантна, но интерфейс между encoder и decoder очень узок:
независимо от длины входа decoder получает один вектор фиксированного размера.
Это не означает, что «вектор физически не может хранить длинное предложение».
Проблема практичнее:

- ранняя информация должна пройти через длинную recurrent-цепочку;
- градиент от поздних decoder steps идёт к ранним input tokens длинным путём;
- decoder не может запросить у encoder разные детали в разные моменты;
- качество ранних seq2seq-систем заметно ухудшалось на длинных предложениях.

> [!warning] Не превращаем историю в миф
> LSTM и reverse-source trick уже смягчали проблему дальних зависимостей.
> Attention не «сделал RNN рабочими с нуля» — он создал гораздо более удобный
> интерфейс чтения encoder memory.

## 2. Оставим не резюме, а память

Encoder и раньше вычислял последовательность состояний:

$$H=(h_1,h_2,\ldots,h_n).$$

Радикально новое решение Bahdanau et al. — не выбрасывать $h_1,\ldots,h_{n-1}$.
Каждое $h_j$ становится annotation позиции $j$: в оригинальной модели оно
объединяет состояния двунаправленного RNN и потому содержит левый и правый
контекст.

Теперь на decoder step $t$ строится свой context vector:

$$
c_t=\sum_{j=1}^{n}\alpha_{tj}h_j.
$$

Это не последний encoder state. Это новый вектор, собранный специально для
текущего слова перевода.

### Как получить веса

Сначала модель оценивает совместимость decoder state и каждой annotation:

$$
e_{tj}=a(s_{t-1},h_j).
$$

В additive attention Bahdanau:

$$
e_{tj}=v_a^\top\tanh(W_as_{t-1}+U_ah_j).
$$

Затем softmax нормализует scores по всем input positions:

$$
\alpha_{tj}=
\frac{\exp(e_{tj})}{\sum_{k=1}^{n}\exp(e_{tk})}.
$$

И наконец decoder получает $c_t$:

$$
s_t=f_{\text{dec}}(s_{t-1},y_{t-1},c_t).
$$

Смысл этих трёх стадий лучше формулы:

1. **score** — насколько элемент памяти подходит текущему запросу;
2. **softmax** — какую долю чтения дать каждому элементу;
3. **weighted sum** — прочитать несколько элементов мягко, сохранив
   дифференцируемость.

## 3. Проследим один перевод

Представим упрощённые attention weights:

| target step | The | black | cat | sleeps | on | the | sofa |
|---|---:|---:|---:|---:|---:|---:|---:|
| Чёрный | .05 | **.76** | .12 | .02 | .01 | .02 | .02 |
| кот | .03 | .11 | **.78** | .04 | .01 | .01 | .02 |
| спит | .01 | .02 | .09 | **.80** | .02 | .02 | .04 |
| на | .01 | .01 | .01 | .08 | **.66** | .10 | .13 |
| диване | .01 | .01 | .02 | .04 | .09 | .13 | **.70** |

Каждая строка суммируется в единицу. Веса перемещаются по source sentence вместе
с генерацией. Так появляется **soft alignment**: соответствие между source и
target не размечалось вручную, но возникло как полезная внутренняя структура.

Однако attention matrix не надо называть доказательством рассуждения модели.
Это коэффициенты конкретного вычисления; одинаковый output иногда можно получить
при существенно отличающихся распределениях.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bahdanau_examples-min.png]]

*Примеры soft alignment из Bahdanau et al.; визуальный материал сохранён в базе
из курса Lena Voita. Светлая клетка означает больший вес соответствия между
source token и target token. Здесь важна не идеальная диагональ, а то, что
модель сама восстанавливает перестановки и составные соответствия.*

## 4. Bahdanau и Luong — похожий принцип, разные сборки

| Свойство | Bahdanau | Luong |
|---|---|---|
| score | additive MLP | dot, general или concat |
| encoder в исходной работе | bidirectional | обычно stacked unidirectional |
| момент вычисления | использует предыдущее decoder state | часто после текущего decoder state |
| область памяти | global | global или local window |

Популярные score functions:

$$
\text{dot}(s,h)=s^\top h,
$$

$$
\text{general}(s,h)=s^\top Wh,
$$

$$
\text{additive}(s,h)=v^\top\tanh(W_ss+W_hh).
$$

Нет универсально «настоящей» функции attention. Общий механизм — score,
нормализация и weighted aggregation; score можно параметризовать по-разному.

## 5. Что именно улучшил attention

### Динамический доступ

Один decoder использует одну и ту же encoder memory, но создаёт разные $c_t$.
Интерфейс стал содержательно-зависимым.

### Короткий путь для информации и градиента

Decoder output соединён с нужным $h_j$ через weighted sum. Сигналу больше не
обязательно полностью проходить через финальный encoder state.

### Переменная длина памяти

Число annotations растёт вместе с входом. Размер каждого $h_j$ фиксирован, но
число адресуемых ячеек не фиксировано.

### Наблюдаемый alignment

Weights можно рисовать как heatmap и исследовать. Это полезный диагностический
инструмент, хотя не полное объяснение поведения.

## 6. Почему этого всё ещё недостаточно

RNN + attention решил bottleneck, но сохранил recurrent backbone:

- $h_j$ зависит от $h_{j-1}$ — encoder нельзя полностью вычислить параллельно;
- $s_t$ зависит от $s_{t-1}$ — decoder training можно ускорить teacher forcing,
  но сама recurrence остаётся;
- путь между далёкими input positions всё ещё проходит через RNN states;
- attention используется главным образом как decoder-to-encoder read.

Следующий вопрос оказался важнее машинного перевода:

> Если weighted aggregation так хорошо передаёт информацию от encoder к decoder,
> почему бы не дать позициям одной последовательности общаться тем же способом?

Это и есть переход к **self-attention**.

### Два объяснения, которые стоит прочитать в оригинале

> Chris Olah и Shan Carter предлагают думать об attention как о чтении из
> набора annotations: сеть на каждом шаге выбирает, какие части памяти собрать
> в текущий ответ.

- [Distill — Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
- [Lena Voita — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)

Первый источник особенно хорош как визуальная история механизмов памяти; второй
— как строгий учебный маршрут с формулами, вариантами score и анализом heads.

## 7. Три термина, которые нельзя смешивать

| Механизм | Кто задаёт запрос | Где находятся читаемые элементы |
|---|---|---|
| Bahdanau cross-attention | recurrent decoder | encoder annotations |
| Transformer cross-attention | Transformer decoder | encoder outputs |
| self-attention | позиция последовательности | другие позиции той же последовательности |

Термины Query, Key и Value удобно ретроспективно применить к старому attention,
но в статье Bahdanau механизм описан через decoder state, annotations и alignment
model. Исторически точнее не делать вид, что Q/K/V уже были исходным языком.

## Мини-реализация recurrent attention

```python
# decoder_state: [B, d_s]
# encoder_states: [B, T, d_h]

query = W_s(decoder_state).unsqueeze(1)       # [B, 1, d_a]
memory = W_h(encoder_states)                  # [B, T, d_a]
scores = v(torch.tanh(query + memory)).squeeze(-1)  # [B, T]
weights = scores.softmax(dim=-1)              # [B, T]
context = torch.einsum("bt,btd->bd", weights, encoder_states)
```

Обратите внимание: здесь decoder создаёт один запрос к encoder memory. В
self-attention следующей главы все $T$ позиций создадут запросы одновременно.

## Что должно остаться после главы

- Attention появился до Transformer.
- Первая ключевая роль — decoder-to-encoder soft alignment.
- Context vector стал зависеть от decoder step.
- Score, softmax и weighted sum — общий вычислительный шаблон.
- Bahdanau attention — cross-attention, не self-attention.
- Transformer родился из следующего шага: убрать recurrence и сделать
  communication между позициями основным примитивом.

## Источники и хорошие продолжения

### Первичные

- [Sutskever et al. — Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Bahdanau et al. — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Luong et al. — Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025)

### Объяснения

- [Lena Voita — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)
- [Distill — Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
- [Jay Alammar — Visualizing Neural Machine Translation](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 07 — Attention|CS224N: Attention]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/00 Источники и визуальный стандарт|Паспорт источников модуля]]

**Назад:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN, LSTM и Seq2Seq]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]
