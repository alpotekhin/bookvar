---
title: Seq2Seq с фиксированным контекстом
type: textbook-chapter
status: canonical
last_updated: 2026-11-21
source_language: mixed
reference_terms:
  - label: "Условная вероятность перевода и EOS"
    label_en: "Conditional translation probability and EOS"
    anchor: seq2seq-factorization
  - label: "Инициализация декодера и постоянный контекст"
    label_en: "Decoder initialization and fixed context"
    anchor: seq2seq-source-context
  - label: "Слои, состояния и формы"
    label_en: "Layers, states and tensor shapes"
    anchor: seq2seq-state-shapes
  - label: "Согласование состояний энкодера и декодера"
    label_en: "Bridging encoder and decoder states"
    anchor: seq2seq-state-bridge
  - label: "Сдвиг целей и teacher forcing"
    label_en: "Target shift and teacher forcing"
    anchor: seq2seq-teacher-forcing
  - label: "EOS, PAD и нормировка потери"
    label_en: "EOS, PAD and loss reduction"
    anchor: seq2seq-loss-mask
  - label: "Padding источника и конечное состояние"
    label_en: "Source padding and final encoder state"
    anchor: seq2seq-source-padding
  - label: "Полная двухслойная GRU-модель"
    label_en: "Complete two-layer GRU model"
    anchor: seq2seq-complete-model
  - label: "Два пути градиента в энкодер"
    label_en: "Two gradient paths into the encoder"
    anchor: seq2seq-encoder-gradients
  - label: "Ограничение фиксированного контекста"
    label_en: "Fixed-context bottleneck"
    anchor: seq2seq-bottleneck
  - label: "Обращение источника и расстояния"
    label_en: "Source reversal and dependency distances"
    anchor: seq2seq-source-reversal
  - label: "Авторегрессивный цикл и EOS"
    label_en: "Autoregressive decoding and EOS"
    anchor: seq2seq-generation
  - label: "Ошибка префикса и exposure bias"
    label_en: "Prefix errors and exposure bias"
    anchor: seq2seq-exposure-bias
  - label: "Локальный выбор и вероятность всего ответа"
    label_en: "Local choices versus sequence probability"
    anchor: seq2seq-greedy
  - label: "Beam search и общий отбор кандидатов"
    label_en: "Beam search and global candidate selection"
    anchor: seq2seq-beam-search
  - label: "Завершённые гипотезы и остановка"
    label_en: "Finished hypotheses and stopping"
    anchor: seq2seq-beam-stop
  - label: "Родители гипотез и перенос состояний"
    label_en: "Beam parents and state routing"
    anchor: seq2seq-beam-states
  - label: "Нормализация длины и усечение"
    label_en: "Length normalization and truncation"
    anchor: seq2seq-length-score
  - label: "Контекст по всем позициям источника"
    label_en: "Context over all source positions"
    anchor: seq2seq-attention
  - label: "Маска источника до softmax"
    label_en: "Source mask before softmax"
    anchor: seq2seq-attention-mask
  - label: "Обучение весов внимания"
    label_en: "Learning attention weights"
    anchor: seq2seq-attention-gradients
  - label: "Память и вычисления attention"
    label_en: "Attention storage and computation"
    anchor: seq2seq-memory-cost
  - label: "Seq2Seq, encoder–decoder и причинность"
    label_en: "Seq2Seq, encoder–decoder and causality"
    anchor: seq2seq-scope
previous: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]"
primary_sources:
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html
  - https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
  - https://arxiv.org/abs/1406.1078
  - https://arxiv.org/abs/1409.3215
  - https://d2l.ai/chapter_recurrent-modern/seq2seq.html
---

# Seq2Seq с фиксированным контекстом

Будем считать слова и точку отдельными токенами. Тогда английское
предложение `They are watching .` содержит четыре токена,
а его французский перевод `Ils regardent .` — три. При переводе
меняются длина и порядок слов: английскому сочетанию `are watching` здесь
соответствует одно `regardent`. Поэтому нельзя решить задачу, назначив каждому
входному токену ровно один выходной, как при разметке частей речи.

При переводе одной последовательности в другую энкодер (encoder) читает
исходный текст, а декодер (decoder) по его представлению и уже известной
части перевода предсказывает следующий токен. Такая постановка
sequence-to-sequence (Seq2Seq) подходит и для других задач, например
для составления краткого содержания по документу.

В ранней рекуррентной модели декодер получает только конечное состояние
энкодера. Если в нём потерялась информация о первом слове, декодер уже
не может запросить состояние энкодера именно на этой позиции.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-bridge-2026-11-03/d2l-encoder-decoder.svg|Энкодер, передаваемое состояние и декодер в общей архитектуре D2L]]

*The original encoder–decoder architecture from D2L. The diagram separates
the source encoder, the transferred state and the target decoder; it does
not specify a recurrent cell or require equal source and target lengths.
Source: Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola,
[D2L, §10.6](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html),
CC BY-SA 4.0. Complete original SVG, without cropping or redrawing.*

<a id="seq2seq-factorization"></a>

## Условная языковая модель

Пусть источник состоит из $n$ токенов $x_{1:n}$, а перевод содержит $m$
обычных токенов $y_{1:m}$. Добавим $y_0=\mathrm{BOS}$, знак начала генерации,
и $y_{m+1}=\mathrm{EOS}$, знак завершения. Тогда вероятность **завершённого**
перевода равна

$$
p(y_{1:m},\mathrm{EOS}\mid x)=
\prod_{t=1}^{m+1}p(y_t\mid y_{<t},x).
$$

EOS входит в произведение наравне с остальными целями. Без него модель
оценивает вероятность префикса: например, вероятность начать с «кошка спит»,
но не обязательно закончить после этих слов. BOS задаёт начальный вход;
в этом соглашении предсказывать его не требуется.

Для пары `black cat sleeps` → `чёрная кошка спит` четыре предсказания выглядят так:

| Шаг | Известная часть перевода | Что предсказывается |
|---|---|---|
| 1 | BOS | чёрная |
| 2 | BOS, чёрная | кошка |
| 3 | BOS, чёрная, кошка | спит |
| 4 | BOS, чёрная, кошка, спит | EOS |

На каждом шаге дополнительно известен весь источник `black cat sleeps`.
Если вероятности правильных целей равны $0.7$, $0.8$, $0.9$ и $0.6$, то

$$
\begin{gathered}
p(\text{чёрная кошка спит},\mathrm{EOS}\mid x)\\
=0.7\cdot0.8\cdot0.9\cdot0.6=0.3024.
\end{gathered}
$$

Суммарная потеря равна $-\ln(0.3024)\approx1.1960$, а средняя по четырём
целям — $0.2990$ ната. Чтобы найти эту потерю, не нужно генерировать
возможные переводы: все четыре распределения считаются вдоль правильного
продолжения.

<a id="seq2seq-source-context"></a>

<a id="как-источник-попадает-в-предсказание"></a>

### Два способа передать состояние энкодера

Энкодер последовательно читает эмбеддинги исходных токенов:

$$
h_j=f_{\mathrm{enc}}(E_x[x_j],h_{j-1}),\qquad v=h_n.
$$

После `black` состояние описывает прочитанное начало; после `cat` оно
обновляется с учётом обоих слов; после `sleeps` получаем представление всей
фразы $v$. Это обучаемое представление: его координаты заранее не назначены
цвету, животному и действию. Функция потерь перевода определяет, какие
признаки сети выгодно сохранять.

У декодера собственное состояние $s_t$ и собственные параметры. В варианте,
который использует D2L, конечное состояние энкодера инициализирует декодер,
а вектор $v$ также подаётся на каждый его шаг:

$$
s_0=h_n,\qquad
s_t=f_{\mathrm{dec}}([E_y[y_{t-1}];v],s_{t-1}).
$$

Квадратные скобки с точкой с запятой обозначают конкатенацию. Затем линейный
слой переводит состояние в оценки токенов целевого словаря:

$$
\begin{gathered}
\ell_t=W_os_t+b_o,\\
p(y_t=j\mid y_{<t},x)=[\operatorname{softmax}(\ell_t)]_j.
\end{gathered}
$$

Например, для выбора «кошка» декодер уже обработал BOS и «чёрная», но
информацию о `cat` получает из $v$. При выборе «спит» изменятся предыдущее
слово и состояние декодера; $v$ останется тем же.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq.svg|Обучение encoder–decoder: правильные предыдущие токены и постоянный контекст]]

*Обучение рекуррентного encoder–decoder на английско-французской паре.
Нижняя длинная стрелка передаёт одно и то же конечное состояние энкодера
на каждый шаг декодера. Под декодером стоят правильные предыдущие токены,
над ним — цели со сдвигом на одну позицию. Источник: Zhang, Lipton, Li, Smola,
[Dive into Deep Learning, §10.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html),
CC BY-SA 4.0. Исходный SVG, без изменения содержания.*

На схеме D2L конец источника также отмечен EOS. В коде ниже конец входа
задаётся явной длиной, без отдельного исходного EOS; целевой EOS остаётся
обязательной обучающей целью в обоих случаях.

У [Sutskever et al., §2](https://arxiv.org/abs/1409.3215) энкодер
только задаёт начальное состояние декодера. У
[Cho et al., §2.2](https://arxiv.org/abs/1406.1078) контекст участвует
и в последующих переходах. В обоих случаях декодер зависит от всего
источника, но при передаче только через $s_0$ эта зависимость проходит
ещё и через цепочку состояний декодера.

Для многослойной GRU состояние включает вектор каждого слоя. Для LSTM нужно
передать и скрытые состояния $h$, и состояния ячеек $c$, рассмотренные
в предыдущей главе. Поэтому «один вектор» здесь означает фиксированное
число координат, а не обязательный тензор с одной осью.
Обозначением $v$ мы отделяем контекст источника от состояния ячейки LSTM $c$.

<a id="seq2seq-teacher-forcing"></a>

<a id="сдвиг-целей-и-teacher-forcing"></a>

## Обучение декодера на правильном переводе

Для нашего русского перевода вход декодера равен

`BOS, чёрная, кошка, спит`,

а цели на тех же четырёх позициях равны

`чёрная, кошка, спит, EOS`.

Для первой цели декодер читает BOS и оценивает вероятность «чёрная».
Для второй он получает «чёрная» из обучающей пары, независимо от того,
какое слово имело наибольшую вероятность на первом шаге. Это teacher
forcing: история для каждой позиции берётся из правильного перевода.
Все условные вероятности можно вычислить за один прямой проход, но
внутри RNN расчёт остаётся последовательным: $s_t$ зависит от $s_{t-1}$.

Правильный токен $y_t$ нельзя подать на вход той же позиции, на которой
он предсказывается. Иначе вместо условия $y_{<t}$ модель увидит ответ.
Сдвиг отделяет известную историю от текущей цели. Обучающие входы содержат
текст перевода, но только его предшествующую часть для каждого предсказания.

Для пакета с переводами разной длины дополним короткие последовательности:

| Перевод | Вход декодера | Цели |
|---|---|---|
| чёрная кошка спит | BOS, чёрная, кошка, спит | чёрная, кошка, спит, EOS |
| кошка спит | BOS, кошка, спит, EOS | кошка, спит, EOS, PAD |

Последний выход второй строки не относится к переводу. Вход EOS на этой
технической позиции не вредит более ранним выходам однонаправленной GRU;
потерю на соответствующем PAD нужно исключить. В генерации после EOS
нового шага уже не будет.

<a id="seq2seq-loss-mask"></a>

Если $M_{bt}$ равно единице на правильных целях, включая EOS, и нулю на
дополнении, средняя потеря по токенам равна

$$
L=-\frac{\sum_{b,t}M_{bt}
\log p_\theta(y_{bt}\mid y_{b,<t},x_b)}
{\sum_{b,t}M_{bt}}.
$$

Здесь в знаменателе 7, а не 8: четыре цели первой фразы и три второй.
Усреднение сначала внутри каждого предложения, а затем по предложениям
даёт другую весовую схему: короткая и длинная пара вносят одинаковый вес.
Оба варианта можно использовать, но нужно понимать, какой реализован.

PAD и EOS должны иметь разные ID, если маска строится по равенству PAD.
Иначе из потери исчезнут настоящие завершения. `ignore_index=PAD`
исключает позиции с такой целью; он не удаляет класс PAD из softmax
на остальных позициях. Для текущего примера достаточно маски по целям.

<a id="seq2seq-source-padding"></a>

<a id="padding-источника-меняет-состояние"></a>

### Длина источника и конечное состояние

У фразы `cat sleeps` истинная длина равна двум, даже если тензор содержит
ещё десять PAD. Нулевой эмбеддинг PAD не означает «сохранить состояние»:
рекуррентная матрица и смещения продолжают работать. Поэтому последнее
состояние обычного прохода по дополненному пакету в общем случае уже
не является состоянием после `sleeps`.

Ниже `pack_padded_sequence` сообщает GRU настоящие длины. Возвращаемое
конечное состояние каждого предложения соответствует последнему
действительному токену. Маска целевой потери эту проблему не решает:
она применяется позже, после кодирования источника.

<a id="seq2seq-generation"></a>

<a id="декодирование--не-обучение"></a>

## Генерация перевода до EOS

При переводе `black cat sleeps` энкодер вычисляет тот же контекст $v$,
а первый вход декодера по-прежнему BOS. Дальше правильный русский текст
модели неизвестен. Если алгоритм выбрал «чёрная», именно «чёрная» пойдёт
на следующий вход. Декодер обновит $s$, но снова получит тот же $v$.
Когда алгоритм выберет EOS, перевод закончится.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/seq2seq-predict.svg|Генерация перевода с возвратом выбранного токена в декодер]]

*Генерация в той же модели. Петли справа возвращают выбранный токен на
следующий вход декодера; на схеме обучения эти входы брались из правильного
перевода. Длинная нижняя стрелка по-прежнему несёт постоянный контекст
источника. Источник: [D2L, §10.7, Prediction](https://d2l.ai/chapter_recurrent-modern/seq2seq.html),
Zhang et al., CC BY-SA 4.0; исходный SVG.*

<a id="seq2seq-exposure-bias"></a>

Допустим, при генерации первым вышло «кошка». Тогда второе распределение
отвечает на вопрос «что идёт после “кошка” при данном источнике», а при
обучении на той же паре оно считалось после «чёрная». В частности,
декодер может продолжить «спит» и пропустить цвет.

Расхождение между правильными обучающими префиксами и префиксами,
которые модель строит сама, называют exposure bias. Teacher forcing
считает правдоподобие именно правильной последовательности; подстановка
сгенерированных токенов меняет обучающий расчёт.

<a id="seq2seq-greedy"></a>

<a id="почему-локально-лучший-токен-может-проиграть"></a>

### Жадный выбор и вероятность ответа

Жадное декодирование выбирает наиболее вероятный следующий токен.
Допустим, сначала $p(A)=0.6$, $p(B)=0.4$. После $A$ распределение равно
$p(\mathrm{EOS}\mid A)=0.4$, $p(A\mid A)=0.3$, $p(B\mid A)=0.3$.
После $B$ вероятность EOS равна $0.9$, а $A$ и $B$ — по $0.05$.

Жадный алгоритм выбирает $A$, затем EOS. Вероятность завершённого
ответа — $0.6\cdot0.4=0.24$. Ответ $B,\mathrm{EOS}$ вероятнее:
$0.4\cdot0.9=0.36$, хотя первый токен $B$ локально уступал $A$.
Первое решение изменяет распределение следующего шага.

В D2L приведён похожий пример с четырьмя выходными позициями:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/s2s-prob1.svg|Жадный путь A B C EOS и его условные распределения]]

*Жадный путь $A,B,C,\mathrm{EOS}$ имеет вероятность
$0.5\cdot0.4\cdot0.4\cdot0.6=0.048$. Каждый столбец — распределение
для уже выбранного префикса, а не независимая таблица по времени.
Источник: [D2L, §10.8, Greedy Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html),
Zhang et al., CC BY-SA 4.0; исходный SVG.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/s2s-prob2.svg|Другой префикс меняет распределения и повышает вероятность ответа]]

*Выбор $C$ вместо $B$ на втором шаге меняет последующие распределения.
Путь $A,C,B,\mathrm{EOS}$ получает $0.054$, что больше $0.048$.
Этот контрпример опровергает оптимальность жадного пути, но сам по себе
не доказывает, что второй путь лучший среди всех возможных.
Источник: [D2L, §10.8](https://d2l.ai/chapter_recurrent-modern/beam-search.html),
Zhang et al., CC BY-SA 4.0; исходный SVG.*

<details>
<summary>Beam search: отбор, состояния и оценка длины</summary>

<a id="seq2seq-beam-search"></a>

<h3 id="как-работает-beam-search">Отбор гипотез и условие остановки</h3>

Полный перебор рассматривает все возможные продолжения. При словаре
размера $V_y$ число ветвей до длины $T$ растёт экспоненциально.
Без учёта раннего EOS грубая оценка для $10^4$ токенов и десяти шагов
составляет $10^{40}$ кандидатов максимальной длины.
**Beam search** ограничивает число одновременно сохраняемых гипотез
шириной $B$.

В числовом примере при $B=2$ первый шаг сохраняет $A$ с вероятностью
$0.6$ и $B$ с $0.4$. На втором шаге сравниваются все их продолжения
вместе: $B,\mathrm{EOS}$ с $0.36$, $A,\mathrm{EOS}$ с $0.24$,
$A,A$ и $A,B$ с $0.18$ каждое, $B,A$ и $B,B$ с $0.02$ каждое.
Остаются два завершённых ответа; лучший — $B,\mathrm{EOS}$.
Выбор двух токенов отдельно для каждого родителя дал бы до четырёх
гипотез, а не beam ширины два.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/beam-search.svg|Глобальный отбор двух гипотез после расширения beam]]

*Ширина beam равна двум: после расширения сохраняются две гипотезы
среди всех кандидатов. На втором шаге это AB и CE, на третьем — ABD и CED.
Схема иллюстрирует отбор префиксов, а не отдельное правило завершения по EOS.
Источник: [D2L, §10.8, Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html),
Zhang et al., CC BY-SA 4.0; исходный SVG.*

На практике складывают логарифмы:

$$
S(y_{1:t})=\sum_{k=1}^{t}
\log p(y_k\mid y_{<k},x).
$$

Это предотвращает численное обнуление произведений и сохраняет порядок
вероятностей. Каждой гипотезе нужны её префикс, оценка и состояние декодера.
В рекуррентной модели продолжения разных префиксов нельзя вычислять
с одним общим изменяемым состоянием. При пакетной реализации состояния
переставляют по индексам выбранных родителей; фиксированный контекст
источника можно переиспользовать.

Следующий код полностью задаёт маленькое распределение из примера.
После двух обычных букв оно выдаёт EOS с вероятностью один. Поэтому
полный перебор конечен и позволяет проверить результат beam search.

```python
import math

EOS = "<eos>"

def distribution(prefix):
    if not prefix:
        return {"A": 0.6, "B": 0.4}
    if len(prefix) >= 2:
        return {EOS: 1.0}
    if prefix == ("A",):
        return {EOS: 0.4, "A": 0.3, "B": 0.3}
    return {EOS: 0.9, "A": 0.05, "B": 0.05}

def beam_search(width, max_steps=3):
    assert width >= 1 and max_steps >= 0
    beams = [((), 0.0)]
    for _ in range(max_steps):
        candidates = []
        for prefix, score in beams:
            if prefix and prefix[-1] == EOS:
                candidates.append((prefix, score))
                continue
            for token, prob in distribution(prefix).items():
                candidates.append(
                    (prefix + (token,), score + math.log(prob))
                )
        beams = sorted(candidates, key=lambda item: item[1],
                       reverse=True)[:width]
        if all(prefix[-1] == EOS for prefix, _ in beams):
            break
    finished = [item for item in beams if item[0] and item[0][-1] == EOS]
    return max(finished, key=lambda item: item[1]) if finished else None

def enumerate_finished(prefix=(), score=0.0):
    if prefix and prefix[-1] == EOS:
        return [(prefix, score)]
    result = []
    for token, prob in distribution(prefix).items():
        result.extend(enumerate_finished(
            prefix + (token,), score + math.log(prob)))
    return result

all_answers = enumerate_finished()
assert math.isclose(sum(math.exp(s) for _, s in all_answers), 1.0)
exact = max(all_answers, key=lambda item: item[1])
greedy = beam_search(1)
beam = beam_search(2)
assert greedy[0] == ("A", EOS)
assert beam[0] == exact[0] == ("B", EOS)
assert beam_search(2, max_steps=1) is None
for name, answer in [("greedy", greedy), ("beam", beam), ("exact", exact)]:
    print(name, answer[0], round(math.exp(answer[1]), 6))
```

Этот вариант оставляет завершённые гипотезы в beam без расширения и
возвращает `None`, если лимит достигнут до появления EOS.
Другой распространённый вариант, в том числе у Sutskever et al. в §3.2,
хранит завершённые ответы отдельно. Правила отбора и остановки нужно
рассматривать вместе: первая найденная EOS-гипотеза не обязательно
лучшая.

<a id="seq2seq-beam-stop"></a>

Если ранжирование использует исходную сумму логарифмов, оценка любого
продолжения не больше оценки его префикса: новые слагаемые неположительны.
Поэтому поиск можно остановить, когда лучшая завершённая оценка не ниже
всех живых оценок. Такая проверка доказывает оптимальность только среди
ещё не отброшенных ветвей. Сам beam search мог раньше удалить лучший путь.

<a id="seq2seq-beam-states"></a>

<h4 id="reordering-states-after-beam-selection">Reordering states after beam selection</h4>

The global top-$B$ selection may keep two children of the same parent
and none from another parent. Suppose the live prefixes are A and C.
If the selected continuations are CB and CD, both inherit the state
after C:

| Selected prefix | Parent row | State to carry | Token to consume next |
|---|---:|---|---|
| CB | 1 | state after C | B |
| CD | 1 | state after C | D |

The parent state already includes the old prefix. The selected new
token is then consumed by each child to obtain its state and predict
the following token. Both children initially receive equal state
values, but the different B and D embeddings generally make the next
states different.

The code below checks this one state-routing operation after selection.
It does not choose the beam or compute sequence probabilities; the
complete finite-distribution search above handles that separate step.

```python
import torch
from torch import nn

torch.manual_seed(67)
dtype = torch.float64
embedding = nn.Embedding(4, 3, dtype=dtype)  # A=0, B=1, C=2, D=3
decoder_cell = nn.GRUCell(3 + 2, 4, dtype=dtype)
parent_states = torch.tensor([
    [.1, .2, .3, .4],  # state after A
    [.5, .6, .7, .8],  # state after C
], dtype=dtype)
selected_parents = torch.tensor([1, 1])
selected_tokens = torch.tensor([1, 3])  # CB and CD
source_context = torch.tensor([[-.2, .4]], dtype=dtype)

with torch.no_grad():
    inherited = parent_states.index_select(0, selected_parents)
    child_inputs = torch.cat([
        embedding(selected_tokens), source_context.expand(2, -1),
    ], dim=-1)
    child_states = decoder_cell(child_inputs, inherited)
    for child in range(2):
        single = decoder_cell(child_inputs[child:child+1], parent_states[1:2])
        torch.testing.assert_close(child_states[child:child+1], single)
    assert not torch.allclose(child_states[0], child_states[1])
    wrong_states = decoder_cell(child_inputs, parent_states)
    assert not torch.allclose(child_states[0], wrong_states[0])
```

For `nn.GRUCell`, beam rows occupy axis 0 of `[beam,m]`.
For a sequence module’s hidden state `[layers,beam,m]`, the
corresponding gather is on axis 1. An LSTM must reorder both hidden
and cell states with the same parent indices. When decoding several
source inputs together, the indices and source contexts must also
remain grouped by their source.

An EOS child is placed among finished hypotheses according to the
chosen search rule; it is not fed back merely to continue that finished
prefix. The D2L
[beam-search tree](https://d2l.ai/chapter_recurrent-modern/beam-search.html#fig-beam-search)
illustrates global prefix selection. The tensor example here makes the
associated recurrent-state routing explicit.

<a id="seq2seq-length-score"></a>

<h4 id="длина-ответа-и-качество-перевода">Нормализация по длине</h4>

Максимум вероятности модели и качество перевода — разные критерии.
Даже точный поиск не исправит распределение, которое слишком уверенно
завершает ответы или пропускает части источника. Увеличение beam
не гарантирует улучшения перевода: оно меняет поиск в уже обученной модели.

Иногда для ранжирования применяют нормализацию длины

$$
S_\alpha(y)=\frac{S(y)}{L^\alpha},
$$

где $L$ здесь включает EOS, но не BOS. Это настраиваемая на validation
эвристика, не обязательная часть beam search и не логарифм новой
нормированной вероятности. При $\alpha=0$ остаётся исходный критерий.

Поскольку $S(y)\leq0$, деление на большее $L^\alpha$ при $\alpha>0$
делает оценку менее отрицательной. Например, для ответа вероятности
$0.30$ и длины 2 при $\alpha=1$ оценка равна $-0.6020$;
для вероятности $0.25$ и длины 4 — $-0.3466$.
После нормализации более длинный ответ с меньшей вероятностью выигрывает.
Уже выбранный префикс при дальнейшем удлинении может получить
лучшую нормализованную оценку, поэтому прежнее правило остановки
для необработанного $S$ сюда напрямую не переносится.

Достижение `max_steps` означает техническое прерывание, а не предсказание
EOS. Если интерфейс возвращает незавершённый текст, он должен явно
сообщать об усечении. В нейросетевом примере выше это флаг `ended=False`.

</details>

<a id="seq2seq-complete-model"></a>

<a id="полная-небольшая-модель"></a>

## Учебный переводчик из двух GRU

Код сохраняет схему D2L: две отдельные GRU, инициализация декодера конечными
состояниями энкодера и конкатенация постоянного контекста с входным
эмбеддингом. Добавлены packing и явный останов по EOS. Это учебная модель
для четырёх пар, без attention, dropout и оценки качества на тестовой выборке.

<details>
<summary>Полный код обучения и пошагового перевода</summary>

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

torch.manual_seed(21)
torch.set_num_threads(1)
PAD, BOS, EOS = 0, 1, 2
src_vocab = {"<pad>": 0, "black": 1, "cat": 2, "dog": 3, "sleeps": 4}
tgt_vocab = {
    "<pad>": PAD, "<bos>": BOS, "<eos>": EOS,
    "чёрная": 3, "кошка": 4, "собака": 5, "спит": 6,
}
pairs = [
    ("black cat sleeps", "чёрная кошка спит"),
    ("cat sleeps", "кошка спит"),
    ("black dog sleeps", "чёрная собака спит"),
    ("dog sleeps", "собака спит"),
]
src_rows = [[src_vocab[w] for w in x.split()] for x, _ in pairs]
tgt_rows = [
    [BOS] + [tgt_vocab[w] for w in y.split()] + [EOS]
    for _, y in pairs
]
lengths = torch.tensor([len(row) for row in src_rows])
src = torch.tensor([row + [PAD] * (3 - len(row)) for row in src_rows])
tgt = torch.tensor([row + [PAD] * (5 - len(row)) for row in tgt_rows])
dec_input, targets = tgt[:, :-1], tgt[:, 1:]

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(len(src_vocab), 8, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(len(tgt_vocab), 8, padding_idx=PAD)
        self.encoder = nn.GRU(8, 16, num_layers=2, batch_first=True)
        self.decoder = nn.GRU(8 + 16, 16, num_layers=2, batch_first=True)
        self.output = nn.Linear(16, len(tgt_vocab))

    def encode(self, ids, valid_lengths):
        packed = pack_padded_sequence(
            self.src_emb(ids), valid_lengths.cpu(),
            batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        return hidden, hidden[-1]

    def decode(self, ids, hidden, context):
        embeddings = self.tgt_emb(ids)
        repeated = context[:, None, :].expand(-1, ids.size(1), -1)
        inputs = torch.cat([embeddings, repeated], dim=-1)
        outputs, hidden = self.decoder(inputs, hidden)
        return self.output(outputs), hidden

    def forward(self, ids, valid_lengths, target_input):
        hidden, context = self.encode(ids, valid_lengths)
        logits, _ = self.decode(target_input, hidden, context)
        return logits

    @torch.no_grad()
    def greedy(self, ids, valid_lengths, max_steps=8):
        assert ids.size(0) == 1  # This loop decodes one sentence.
        hidden, context = self.encode(ids, valid_lengths)
        token = torch.tensor([[BOS]], device=ids.device)
        result = []
        for _ in range(max_steps):
            logits, hidden = self.decode(token, hidden, context)
            token = logits[:, -1:].argmax(-1)
            next_id = token.item()
            if next_id == EOS:
                return result, True
            result.append(next_id)
        return result, False  # The limit is not a generated EOS.

model = Seq2Seq()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

def objective():
    logits = model(src, lengths, dec_input)
    return F.cross_entropy(
        logits.reshape(-1, len(tgt_vocab)), targets.reshape(-1),
        ignore_index=PAD,
    )

before = objective().item()
for _ in range(300):
    optimizer.zero_grad()
    loss = objective()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
after = objective().item()
model.eval()
words = {i: word for word, i in tgt_vocab.items()}
print(f"NLL: {before:.6f} -> {after:.6f}")
for row, length in zip(src, lengths):
    ids, ended = model.greedy(row[None], length[None])
    print(" ".join(words[i] for i in ids), "EOS:", ended)
```

</details>

В пакете 14 целей без PAD: $4+3+4+3$. За 300 обновлений средняя потеря
на этих целях снизилась с $1.932506$ до $0.000583$ ната. После BOS
декодер сам восстановил четыре обучающих перевода и для каждого выбрал
EOS. На новых фразах модель не проверяли.

<a id="seq2seq-state-shapes"></a>

<a id="формы-на-одном-примере"></a>

### Тензоры на входе и выходе двух GRU

В коде четыре пары, эмбеддинг имеет 8 координат, а скрытое состояние
каждого слоя GRU — 16. Ось пакета идёт первой у последовательностей
(`batch_first=True`), но не у начальных и конечных состояний.

| Тензор | Форма при обучении |
|---|---|
| ID источника | `[4, 3]` |
| Эмбеддинги источника до упаковки | `[4, 3, 8]` |
| Конечные состояния энкодера и начальные декодера | `[2, 4, 16]` |
| Контекст $v$ из верхнего слоя | `[4, 16]` |
| Эмбеддинг и контекст на входе декодера | `[4, 4, 24]` |
| Логиты на четырёх целевых позициях | `[4, 4, 7]` |

Во входе декодера $8+16=24$ координаты. При генерации
одной фразы за раз этот вход имеет форму `[1, 1, 24]`: один выбранный
токен и тот же контекст. Состояния всех исходных позиций можно было бы
распаковать в `[4, 3, 16]`, но этот декодер их не читает.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/seq2seq-details.svg|Embedding, рекуррентные слои и выходной классификатор encoder–decoder]]

*Слои энкодера и декодера. Embedding заменяет ID векторами, Recurrent
обрабатывает последовательность, FC вычисляет оценки целевого словаря.
Обозначение $n\times$ на исходной схеме относится к числу рекуррентных
слоёв, а не к длине предложения $n$ в наших формулах. Источник:
[D2L, §10.7, Layers in an RNN encoder–decoder model](https://d2l.ai/chapter_recurrent-modern/seq2seq.html),
Zhang et al., CC BY-SA 4.0; исходный SVG.*

Передача всех слоёв напрямую возможна при совпадающих числе слоёв и размерах
состояния. Если энкодер двунаправленный или его размер отличается, нужна
явная операция согласования: например, конкатенация направлений и обучаемая
линейная проекция в размер состояния декодера. Простое присваивание тензоров
несовместимых форм не реализует такой переход.

<a id="seq2seq-encoder-gradients"></a>

Обратное распространение проходит из потерь декодера через его начальное
состояние и через все использования `context` в энкодер. `expand` не создаёт
новых обучаемых копий вектора: при backward вклады шагов суммируются.
Если отделить и начальное состояние, и контекст через `detach()`,
эта функция потерь перестанет обучать энкодер. Если отделить только один
из двух каналов, второй по-прежнему может передавать градиент.

При пошаговом декодировании переменная `hidden` меняется, а `context`
остаётся состоянием энкодера. Если на каждом шаге заменять контекст
текущим `hidden[-1]` декодера, получится уже другая вычислительная схема.

<details>
<summary>Как согласовать разные формы состояний</summary>

<a id="seq2seq-state-bridge"></a>

<h3 id="bridging-different-state-shapes">Bridging different state shapes</h3>

D2L separates decoding from the operation that prepares its initial state:

<!-- source-excerpt: d2l-decoder-init-state -->
> In the following decoder interface,
> we add an additional `init_state` method
> to convert the encoder output (`enc_all_outputs`)
> into the encoded state.
<!-- /source-excerpt -->

*Zhang, Lipton, Li, Smola,
[D2L, §10.6, Decoder](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html#decoder),
original English excerpt, CC BY-SA 4.0.*

Consider a two-layer bidirectional GRU encoder with hidden width 3 and
a one-layer unidirectional GRU decoder with hidden width 4. Copying the
encoder’s `[4,B,3]` final state into the decoder cannot satisfy its
`[1,B,4]` state contract.

One possible bridge takes the final forward and reverse states of the
top encoder layer, concatenates them, and applies a learned affine map
followed by tanh. This is a design choice, not an operation that the GRU
API performs automatically.

| Operation | Shape |
|---|---|
| Encoder final states | `[4,B,3]` |
| Expose layer and direction axes | `[2,2,B,3]` |
| Select the top layer | `[2,B,3]` |
| Concatenate its directions for each input | `[B,6]` |
| Apply the learned bridge and tanh | `[B,4]` |
| Add the decoder’s layer axis | `[1,B,4]` |

The six features for each input must come from that same input. For
example, directly reshaping `[2,B,3]` into `[B,6]` when $B>1$
would mix the batch and direction axes. Concatenating the two direction
slices avoids that error.

For a concatenated vector $(0.1,0.2,0.3,-0.1,0,0.4)$, a bridge row
$(1,0,0,1,0,0)$ and bias 0.05 give $0.1-0.1+0.05=0.05$, then
$\tanh(0.05)\approx0.049958$. Each of the decoder’s four initial-state
coordinates has its own learned row. The affine map has $4\cdot6+4=28$
parameters.

The following standalone program isolates this interface. It processes
two source sequences of different lengths, constructs a correctly
shaped initial decoder state, then checks that a loss on two decoder
steps reaches both the bridge and the encoder. As in the four-pair GRU model
above, the fixed source context accompanies every target input. Here the
decoder takes supplied feature vectors instead of token embeddings so the
check can isolate the state transfer.

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(61)
dtype = torch.float64
B, encoder_layers, encoder_width, decoder_width = 2, 2, 3, 4
source_features = torch.randn(B, 3, 2, dtype=dtype, requires_grad=True)
source_lengths = torch.tensor([2, 3])
encoder = nn.GRU(2, encoder_width, num_layers=encoder_layers,
                 bidirectional=True, batch_first=True, dtype=dtype)
packed = pack_padded_sequence(source_features, source_lengths,
                             batch_first=True, enforce_sorted=False)
_, encoded = encoder(packed)
assert encoded.shape == (4, B, 3)
by_layer = encoded.reshape(encoder_layers, 2, B, encoder_width)
top = by_layer[-1]
fixed_context = torch.cat([top[0], top[1]], dim=-1)
torch.testing.assert_close(fixed_context, torch.cat([encoded[-2], encoded[-1]], -1))

state_bridge = nn.Linear(6, decoder_width, dtype=dtype)
initial_decoder = state_bridge(fixed_context).tanh()[None]
assert initial_decoder.shape == (1, B, 4)
assert sum(parameter.numel() for parameter in state_bridge.parameters()) == 28
decoder = nn.GRU(2 + 6, decoder_width, batch_first=True, dtype=dtype)
target_features = torch.randn(B, 2, 2, dtype=dtype)
decoder_inputs = torch.cat([
    target_features, fixed_context[:, None].expand(-1, 2, -1),
], dim=-1)
decoded, final_decoder = decoder(decoder_inputs, initial_decoder)
assert decoded.shape == (B, 2, 4) and final_decoder.shape == (1, B, 4)
decoded.square().mean().backward()
assert state_bridge.weight.grad.abs().sum() > 0
assert source_features.grad[0, :2].abs().sum() > 0
assert source_features.grad[0, 2].count_nonzero() == 0
assert encoder.weight_hh_l1_reverse.grad.abs().sum() > 0

example_context = torch.tensor([.1, .2, .3, -.1, 0., .4], dtype=dtype)
example_row = torch.tensor([1., 0., 0., 1., 0., 0.], dtype=dtype)
torch.testing.assert_close((example_row @ example_context + .05).tanh(),
                           torch.tensor(.05, dtype=dtype).tanh())
```

Only the top-layer states enter this bridge directly. Lower encoder
layers still influence them through the recurrent network. A different
model could project all layer states or initialize several decoder
layers; it must define that mapping explicitly. For an LSTM decoder,
both $h_0$ and $c_0$ need compatible shapes, so mapping only the hidden
state is insufficient.

This bridge still transfers a fixed number of source coordinates.
Changing its dimensions does not give the decoder access to individual
source positions. The direction/state axis ordering is specified in
[PyTorch 2.14, GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)
and worked through in the
[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU#recurrent-bidirectional|previous chapter]].

</details>

<a id="seq2seq-bottleneck"></a>

<a id="почему-фиксированный-контекст-становится-узким-местом"></a>

## Одно представление для всего источника

Допустим, между `black` и концом фразы появилось длинное уточнение.
Чтобы перевести цвет, энкодер должен пронести сведения о `black` до
последнего состояния, а декодер — извлечь их из того же $v$, из
которого он берёт информацию о подлежащем и действии.

С ростом длины источника увеличиваются число деталей и расстояния между
связанными словами, но число передаваемых координат остаётся прежним.
При передаче контекста только в начальное состояние влияние раннего
исходного слова проходит ещё и через последовательность состояний
декодера. Подмешивание $v$ на каждом шаге сокращает эту вторую цепочку,
но не возвращает прямого доступа к состояниям отдельных исходных позиций.

LSTM и GRU помогают переносить информацию и градиенты по рекуррентной
цепочке. Они не дают декодеру заново обратиться к `cat`: после завершения
энкодера у него всё ещё только фиксированное представление источника.
Ограничение здесь практическое: размерность, конечная точность,
обучаемость и необходимость извлекать разные детали из одного
представления. Это не математическое доказательство невозможности
закодировать длинную последовательность вещественным вектором.

В опыте [Cho et al., §5.1](https://aclanthology.org/W14-4012/) у двух
моделей с фиксированным представлением, RNNenc и grConv, качество
перевода заметно падало на длинных фразах. На графиках (a) и (b)
сравните сплошные кривые: у обеих моделей BLEU для источника около
40 слов ниже, чем для источника около 20 слов. У оси (c) другой смысл:
там меняется допустимое число неизвестных слов в наборе фраз, а не их
длина. BLEU тоже снижается.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-attention-bridge-2026-11-10/cho-2014-figure4.png|Графики BLEU в зависимости от длины фразы и числа неизвестных слов]]

*Сплошная линия — отбор по исходным фразам, линия из точек — по
эталонным переводам, штриховая — по обоим. Для (a) и (b) авторы
сгладили кривые окном в 10 слов. Cho et al.,
[On the Properties of Neural Machine Translation, рис. 4](https://aclanthology.org/W14-4012.pdf#page=7),
CC BY-NC-SA 3.0; три панели из PDF, без перерисовки.*

Авторы предположили, что вектору не хватает ёмкости, чтобы удерживать
информацию о длинной фразе. График сам по себе не выделяет эту причину:
размер вектора здесь не варьировали, а качество зависело и от словаря.

<a id="seq2seq-source-reversal"></a>

В [Sutskever et al., §3.3 и §3.7](https://arxiv.org/abs/1409.3215)
глубокая LSTM с обращённым порядком источника успешно работала и на длинных
фразах. Авторы разворачивали только источник, не перевод:
`a b c` → `α β γ` заменяли на `c b a` → `α β γ`.
Без BOS и EOS в последовательности `a b c α β γ` расстояние между
позициями каждой соответствующей пары равно 3. В `c b a α β γ`
эти расстояния становятся 1, 3 и 5. Первая пара сближается, но
среднее остаётся равным 3, а декодер всё так же получает конечное
состояние энкодера.

<a id="seq2seq-attention"></a>

<a id="мост-к-attention"></a>

## Чтение исходных позиций с attention

При переводе слова «кошка» декодеру может понадобиться состояние энкодера
после `cat`, а при переводе «спит» — после `sleeps`. Если сохранить
$h_1,\ldots,h_n$, можно вычислять отдельный контекст $v_t$ для каждого
выходного шага вместо постоянного $v=h_n$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq-attention.svg|Attention по выходам энкодера перед каждым шагом декодера]]

*В этой модели attention получает все выходы энкодера и предыдущее состояние
декодера. Его результат поступает в следующий рекуррентный шаг вместе
с эмбеддингом предыдущего целевого токена. Стрелка к начальному состоянию
декодера сохраняется: attention добавляет доступ к позиционным представлениям,
а не запрещает инициализацию от энкодера. Источник:
[D2L, §11.4, The Bahdanau Attention Mechanism](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html),
Zhang et al., CC BY-SA 4.0; исходный SVG.*

Для каждого $h_j$ функция совместимости вычисляет скалярную оценку.
В варианте, где внимание предшествует обновлению состояния декодера,

$$
e_{tj}=a(s_{t-1},h_j),\qquad
\alpha_{tj}=
\frac{\exp e_{tj}}{\sum_{k=1}^{n}\exp e_{tk}}.
$$

Затем декодер получает взвешенную сумму

$$
\begin{gathered}
v_t=\sum_{j=1}^{n}\alpha_{tj}h_j,\\
s_t=f_{\mathrm{dec}}([E_y[y_{t-1}];v_t],s_{t-1}).
\end{gathered}
$$

Пусть для `black cat sleeps` состояния равны $h_1=(1,0)$, $h_2=(0,2)$,
$h_3=(-1,1)$. Чтобы увидеть, как устроено чтение, зададим веса вручную:
$(0.05,0.85,0.10)$. Получаем

$$
\begin{aligned}
v_t&=0.05(1,0)+0.85(0,2)\\
&\quad+0.10(-1,1)\\
&=(-0.05,1.80).
\end{aligned}
$$

Со старой схемой декодер получил бы $v=h_3=(-1,1)$ и при выборе «кошка»,
и при выборе «спит». Если на втором шаге задать веса $(0.1,0.2,0.7)$,
смесь станет $(-0.6,1.1)$. В обоих случаях передаётся двухмерный вектор;
различается то, какие исходные состояния вошли в него с большим весом.
Здесь веса заданы вручную. Как их вычисляет модель, покажет
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer#translation-two-reads|расчёт двух шагов перевода]].
В нём те же $h_1,h_2,h_3$; меняется только состояние декодера.

<a id="seq2seq-attention-mask"></a>

Маска исключает правое дополнение из нормировки. Добавим к трём
позициям четвёртую с оценкой 100: без маски она заберёт почти весь
вес. Если обнулить её после softmax, веса настоящих токенов не
прибавятся сами. Присваиваем оценке PAD $-\infty$ до softmax, чтобы
она внесла ноль в сумму экспонент. Для каждого источника нужна хотя бы
одна действительная позиция.

```python
import math
import torch

states = torch.tensor([
    [1., 0.], [0., 2.], [-1., 1.], [100., 100.],
], dtype=torch.float64)
scores = torch.tensor([
    math.log(.05), math.log(.85), math.log(.10), 100.,
], dtype=torch.float64, requires_grad=True)
valid = torch.tensor([True, True, True, False])
weights = scores.masked_fill(~valid, -torch.inf).softmax(0)
context = weights @ states
loss = (context - torch.tensor([0., 2.])).square().sum()
loss.backward()
assert weights[-1].item() == 0
assert scores.grad[-1].item() == 0
print("weights:", weights.detach().tolist())
print("context:", context.detach().tolist())
print("score gradients:", scores.grad.tolist())
```

<a id="seq2seq-attention-gradients"></a>

Все три действительные позиции участвуют в сумме, а градиент проходит
через softmax к оценкам. Отдельная разметка «на каком слове фокусироваться»
для такого обучения не обязательна: сигнал может поступать из функции
потерь перевода. Веса не следует автоматически считать точным словарным
выравниванием или полным объяснением решения. Каждое $h_j$ уже зависит
от контекста; одно соответствие может распределяться по нескольким
позициям.

<a id="seq2seq-memory-cost"></a>

Теперь размер хранимой памяти $[h_1;\ldots;h_n]$ растёт с длиной входа.
Сам $v_t$ всё ещё имеет фиксированную размерность, но декодер на каждом
шаге заново выбирает его по всей памяти. Для простого скалярного
произведения подсчёт оценок и суммы требует порядка $n d_h$ операций
на целевой шаг и памяти порядка $n d_h$ для выходов энкодера
(без учёта промежуточных данных обучения).

| Свойство | Фиксированный контекст | Attention |
|---|---|---|
| Память | Конечные состояния | Все $h_j$ без PAD |
| Вход на шаге | Тот же $v$ | Смесь $v_t$ |
| Первая позиция | Только через $v$ | Вес для $h_1$ |

Attention устраняет необходимость передавать сведения об источнике
только через его последнее состояние. Он не устраняет рекуррентное
вычисление самого энкодера и декодера. Transformer заменит эти
рекуррентные слои: при известной последовательности состояния позиций
можно вычислять параллельно внутри слоя. Авторегрессивная генерация
при этом останется последовательной — следующий выбранный токен
станет известен только после предыдущего.

<a id="seq2seq-scope"></a>

<a id="ограничения-и-частые-смешения"></a>

## Какой текст доступен декодеру

До первого токена перевода модель уже прочитала весь источник, поэтому
энкодер может быть двунаправленным. Декодер ограничен с другой стороны:
при расчёте $p(y_t\mid y_{<t},x)$ нельзя использовать будущие токены
перевода. При потоковом переводе модель начинает отвечать до конца
источника, поэтому и доступ к исходному тексту нужно задавать для
каждого шага отдельно.

## Источники и дальнейшее чтение

- [PyTorch, NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), The Encoder and Simple Decoder — a second implementation of the initialization-only decoder and teacher forcing. Its fixed-length padding/attention code is not substituted for the explicit masks and packed source lengths used here.
- Zhang, Lipton, Li, Smola, **Dive into Deep Learning**:
  [§10.6 — The Encoder–Decoder Architecture](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html),
  [§10.7 — Sequence-to-Sequence Learning](https://d2l.ai/chapter_recurrent-modern/seq2seq.html),
  [§10.8 — Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html),
  [§11.4 — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html).
  Первые разделы связывают постановку задачи с кодом двух RNN, сдвигом целей
  и маской потери; §11.4 меняет фиксированный контекст на вычисляемый attention.
  Иллюстрации воспроизведены по CC BY-SA 4.0; полный код курса остаётся
  в оригинале, пример выше явно отличается packing и остановкой.
- Jurafsky, Martin, [**Speech and Language Processing**, глава 14,
  §14.7–14.8](https://web.stanford.edu/~jurafsky/slp3/14.pdf)
  — условная рекуррентная LM, обучение encoder–decoder и переход
  к контексту, меняющемуся на каждом шаге. Ссылка относится к редакции
  главы от 19 августа 2026 года.
- Lena Voita, [**Seq2seq and Attention**](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)
  — разделы Conditional Language Models, The Simplest Model, Training,
  Inference и The Problem of Fixed Encoder Representation.
  Дальше курс разбирает внимание и архитектуру Transformer.
- Stanford CS224N, [**Notes 6: Neural Machine Translation, Seq2seq and
  Attention**, 2019](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf)
  — постановка NMT и устройство рекуррентного переводчика.
- Cho et al., [**Learning Phrase Representations using RNN Encoder–Decoder**](https://arxiv.org/abs/1406.1078),
  2014, §2.2 — фиксированное представление и совместное обучение двух RNN.
  В экспериментах этой работы модель оценивала пары фраз внутри
  статистической системы перевода; это не та же постановка, что
  самостоятельный перевод целого предложения.
- Sutskever, Vinyals, Le, [**Sequence to Sequence Learning with Neural
  Networks**](https://arxiv.org/abs/1409.3215), 2014, §2, §3.2–3.3 —
  многослойная LSTM, передача состояния, beam search и обращение источника.
- Cho et al., [**On the Properties of Neural Machine Translation:
  Encoder–Decoder Approaches**](https://arxiv.org/abs/1409.1259),
  2014, §5.1 и рис. 4 — отдельное исследование зависимости качества
  двух конкретных моделей от длины и неизвестных слов.
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-recurrent-neural-networks.pdf|Stanford CS230 — Recurrent Neural Networks Cheatsheet]]
  — краткая сводка после подробного разбора.

**Назад:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]
