---
title: LSTM и GRU
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
previous: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT]]"
next: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]"
primary_sources:
  - https://www.bioinf.jku.at/publications/older/2604.pdf
  - https://arxiv.org/abs/1406.1078
---

# LSTM и GRU

Полная исходная памятка Stanford с уравнениями всех gates, схемами ячейки и
сравнением LSTM/GRU сохранена в
[[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-recurrent-neural-networks.pdf|Recurrent Neural Networks Cheatsheet]].

Простая RNN переносит память через повторяющееся нелинейное преобразование.
Чтобы факт с первого шага повлиял на сотый, сигнал должен пережить длинное
произведение Якоби. LSTM меняет геометрию этого пути: наряду с обычным скрытым
состоянием она вводит состояние ячейки, которое обновляется главным образом
сложением. Вентили учатся решать, какую часть старой памяти оставить, что
записать и что показать наружу.

## LSTM: память и три решения

Сначала вычисляются коэффициенты вентилей, затем обновляется состояние ячейки,
и лишь после этого получается видимое скрытое состояние. Такой порядок важнее
перечня названий вентилей: он показывает, какая операция создаёт прямой путь
памяти и где этот путь может быть закрыт.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-chain.png]]

*Chris Olah сопоставляет повторяющуюся цепочку простой RNN и LSTM. Горизонтальная
ветвь состояния ячейки проходит через всю последовательность, а вентили
управляют её изменением. Источник: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

Пусть $z_t=[h_{t-1};x_t]$ — конкатенация прежнего скрытого состояния и нового
входа. Современная LSTM вычисляет

$$
f_t=\sigma(W_fz_t+b_f),\quad
i_t=\sigma(W_iz_t+b_i),\quad
\tilde c_t=\tanh(W_cz_t+b_c),
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,
$$

$$
o_t=\sigma(W_oz_t+b_o),\qquad
h_t=o_t\odot\tanh(c_t).
$$

$f_t$ — вентиль забывания, $i_t$ — входной вентиль, $o_t$ — выходной; все они
имеют форму `[batch, hidden]` и принимают значения от 0 до 1 покомпонентно.
$\tilde c_t$ содержит кандидата на запись. Названия описывают операции, но не
гарантируют, что отдельная координата после обучения будет человекочитаемым
«фактом».

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-f.png]]

*Вентиль забывания умножает прежнее состояние $c_{t-1}$ покоординатно: значение
около единицы сохраняет координату, около нуля — стирает. Источник: Chris Olah,
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-i.png]]

*Входной вентиль выбирает, какие координаты кандидата допустимо записать.
Источник: Chris Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-C.png]]

*Состояние обновляется суммой сохранённой старой памяти и отфильтрованного
кандидата. Именно аддитивное обновление отличает этот путь от повторного полного
преобразования состояния в простой RNN. Источник: Chris Olah,
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-o.png]]

*Выходной вентиль определяет, какую часть $\tanh(c_t)$ сделать видимой как
$h_t$. Память $c_t$ и выход $h_t$ поэтому нельзя считать одним объектом.
Источник: Chris Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-gates.svg]]

*Три вентиля LSTM вычисляются из $x_t$ и $h_{t-1}$. Нужно читать схему как
подготовку коэффициентов для последующих поэлементных операций, а не как три
независимые памяти. Источник: Zhang et al., [Dive into Deep Learning,
§10.1](https://d2l.ai/chapter_recurrent-modern/lstm.html), CC BY-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-hidden-state.svg]]

*Обновление состояния ячейки и скрытого состояния. Верхняя аддитивная ветвь
$c_{t-1}\to c_t$ создаёт более прямой путь для сигнала; выходной вентиль решает,
какую часть памяти сделать видимой в $h_t$. Источник: Zhang et al., [Dive into
Deep Learning, §10.1](https://d2l.ai/chapter_recurrent-modern/lstm.html),
CC BY-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-C-line.png]]

*Отдельно показан путь $c_{t-1}\rightarrow c_t$: он состоит из умножения на
вентиль забывания и сложения с новой записью. Источник: Chris Olah,
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

## Вычислимый пример одной координаты

Значения вентилей позволяют увидеть не только новый выход, но и коэффициент
прямого пути $\partial c_t/\partial c_{t-1}$.

Пусть для одной координаты $c_{t-1}=2.0$, $f_t=0.9$, $i_t=0.2$,
$\tilde c_t=-0.5$, $o_t=0.7$. Тогда

$$c_t=0.9\cdot2.0+0.2\cdot(-0.5)=1.7,$$

$$h_t=0.7\tanh(1.7)\approx0.655.$$

Сеть сохранила большую часть прежней памяти, внесла небольшую отрицательную
поправку и открыла наружу 70% её нелинейного представления. Производная
$\partial c_t/\partial c_{t-1}=f_t$ вдоль прямой ветви равна $0.9$. Если вентиль
научится держаться около единицы, память и градиент могут пройти далеко. Это не
гарантия: произведение малых $f_t$ всё равно затухает.

В реализации четыре аффинных преобразования объединяют в одно умножение:

```python
gates = linear(torch.cat([h_prev, x], dim=-1))  # [B, 4H]
f, i, g, o = gates.chunk(4, dim=-1)
f, i, o = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
g = torch.tanh(g)
c = f * c_prev + i * g
h = o * torch.tanh(c)
```

Порядок частей в готовом API может отличаться; переносить веса между
реализациями без проверки соглашения нельзя. Исторически исходная LSTM
Hochreiter и Schmidhuber 1997 года отличалась от привычной схемы: отдельный
вентиль забывания был добавлен позднее. Здесь описан современный стандартный
вариант.

## GRU: объединить память и выход

Gated Recurrent Unit из encoder-decoder Cho et al. оставляет одно состояние
$h_t$ и два вентиля:

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z),
$$

$$
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r),
$$

$$
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})+b_h),
$$

$$
h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t.
$$

Здесь принято соглашение D2L: $z_t=1$ означает сохранить старое состояние.
Некоторые тексты меняют местами $z_t$ и $1-z_t$; это не архитектурное
различие, но источник частых ошибок при чтении кода. Reset gate $r_t$ управляет
тем, сколько прошлого участвует в кандидате, update gate $z_t$ — пропорцией
между старым состоянием и кандидатом.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/gru.svg]]

*Полный вычислительный граф GRU. Аддитивная смесь в последней строке даёт
состоянию короткий путь, а reset gate влияет только на построение кандидата.
Источник: Zhang et al., [Dive into Deep Learning,
§10.2](https://d2l.ai/chapter_recurrent-modern/gru.html), CC BY-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-var-GRU.png]]

*Схема GRU подчёркивает главное отличие: отдельного состояния ячейки нет, а
старое состояние и кандидат смешиваются напрямую. Источник: Chris Olah,
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

Если $h_{t-1}=0.6$, $z_t=0.75$, а $\tilde h_t=-0.2$, то

$$h_t=0.75\cdot0.6+0.25\cdot(-0.2)=0.4.$$

Три четверти старого состояния сохранились, четверть заменилась кандидатом.
При $z_t\approx1$ GRU почти копирует память и создаёт прямой путь градиента.

## Что выбирать

LSTM разделяет внутреннюю память $c_t$ и видимый выход $h_t$ и имеет три
вентиля; GRU компактнее и смешивает прошлое с кандидатом в одном состоянии.
У GRU обычно меньше параметров: при размере входа $d$ и состояния $h$ основной
вклад — $3h(d+h+1)$ против $4h(d+h+1)$ у LSTM. Но это не доказывает общего
превосходства: качество зависит от данных, масштаба, оптимизации и допустимой
задержки. Выбор проверяют экспериментом при сопоставимом бюджете.

Обе архитектуры смягчают затухание градиента, но не устраняют последовательность
вычислений. Обе сжимают прочитанный префикс в состояния фиксированного размера.
Dropout между слоями, gradient clipping, маски для padding и корректная
инициализация состояния остаются инженерно важными; `detach` состояния при
потоковом обучении определяет горизонт BPTT.

Следующая глава использует LSTM или GRU как строительный блок encoder-decoder.
Там проявится другое ограничение: даже хорошо обученная память должна будет
упаковать целое исходное предложение в единственный фиксированный вектор.

## Источники и дальнейшее чтение

Порядок «путь памяти → три вентиля → численный проход» следует CS224N Notes 5
и D2L §10.1. Формулы GRU и соглашение об update gate сверены с D2L §10.2 и
Cho et al.; исторические детали — с первичной работой LSTM.

- Hochreiter, Schmidhuber, [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf), 1997 — первичная работа LSTM.
- Cho et al., [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078), 2014 — encoder-decoder и GRU.
- Jurafsky, Martin, [Speech and Language Processing, ch. 8](https://web.stanford.edu/~jurafsky/slp3/8.pdf).
- Stanford CS224N, [Notes 5: Language Models, RNN, GRU and LSTM](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/cs224n-2023-notes05-LM_RNN.pdf).
- Zhang et al., [Dive into Deep Learning, ch. 10](https://d2l.ai/chapter_recurrent-modern/index.html), CC BY-SA 4.0.
- Chris Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — пошаговая визуальная линия операций LSTM и сопоставление с GRU.

**Назад:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]
