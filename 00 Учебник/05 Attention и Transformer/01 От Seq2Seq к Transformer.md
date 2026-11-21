---
title: Attention для перевода
type: textbook-chapter
status: canonical
last_updated: 2026-11-21
previous: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]"
primary_sources:
  - https://arxiv.org/abs/1409.0473
  - https://arxiv.org/abs/1508.04025
  - https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html
  - https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html
---

# Attention для перевода

В переводе `black cat sleeps` → «чёрная кошка спит» для слова «кошка»
полезно представление `cat`, а для «спит» — представление `sleeps`.
В предыдущей модели декодер извлекал обе детали из одного конечного
состояния энкодера. С вниманием (attention) энкодер передаёт
представления всех исходных позиций, а декодер на каждом шаге
собирает из них контекст с новыми весами.

## Перевод как условная языковая модель

Пусть $x_{1:n}$ — источник, $y_{1:m}$ — обычные токены перевода,
$y_0=\mathrm{BOS}$, а $y_{m+1}=\mathrm{EOS}$. Как и в предыдущей главе,

$$
p(y_{1:m},\mathrm{EOS}\mid x)=
\prod_{t=1}^{m+1}p(y_t\mid y_{<t},x).
$$

Attention меняет способ получить сведения об $x$, а не эту факторизацию.
Во время обучения декодер получает правильную предшествующую часть
перевода; при генерации — собственные выбранные токены.
BOS, EOS, сдвиг целей и маска PAD подробно разобраны в
[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора#seq2seq-teacher-forcing|главе о Seq2Seq]].

В этом переводе порядок слов совпадает. В общем случае декодеру
не обязательно читать исходные позиции слева направо: в переводе слова
могут менять порядок, а у служебного слова может не быть отдельного
соответствия.

<a id="1-до-attention-один-вектор-на-всё-предложение"></a>

## Что получает декодер без внимания

Энкодер без attention вычисляет $h_1,\ldots,h_n$, но передаёт декодеру
только конечное состояние $v=h_n$ или фиксированный набор состояний
слоёв. Декодер использует его для инициализации и, в некоторых вариантах,
как дополнительный вход на каждом шаге.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-attention-bridge-2026-11-10/d2l-seq2seq-state.svg|Фиксированное состояние между рекуррентными энкодером и декодером]]

*Блок State — единственный путь от энкодера слева к декодеру справа.
Стрелка Targets ведёт в декодер отдельно: она обозначает входные токены
на языке перевода. Сравните с рисунком в конце следующего раздела:
там у декодера появится второй путь к источнику. Zhang et al.,
[D2L, рис. 11.4.1](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html#fig-s2s-attention-state),
CC BY-SA 4.0; исходный SVG.*

Чтобы использовать раннее слово, сеть должна сохранить нужные сведения
до конца чтения. При генерации разных слов декодер извлекает разные
детали из одного $v$. Состояния $h_1,\ldots,h_{n-1}$ он непосредственно
прочитать уже не может.

У [Sutskever et al., §3.7](https://arxiv.org/abs/1409.3215) LSTM с
обращённым источником переводила и длинные фразы. Переход к вниманию
меняет то, что может прочитать декодер, а не вводит абсолютную
границу допустимой длины. Сравнение результатов есть в разделе
[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора#seq2seq-bottleneck|о фиксированном контексте]].

<a id="2-оставим-не-резюме-а-память"></a>

<a id="2-позиционные-представления-и-чтение-памяти"></a>

<a id="2-представления-исходных-позиций-и-чтение-памяти"></a>

## Представления исходных позиций

В модели [Bahdanau, Cho и Bengio, §3](https://arxiv.org/abs/1409.0473)
для каждой исходной позиции сохраняется **annotation** — её контекстное
представление $h_j$. Двунаправленный энкодер объединяет два состояния:

$$
h_j=[\overrightarrow h_j;\overleftarrow h_j].
$$

Прямое направление уже прочитало $x_1,\ldots,x_j$, обратное —
$x_n,\ldots,x_j$. Поэтому представление у `cat` содержит сведения
не только о самом слове, но и об окружающей фразе. Это допустимо:
весь источник известен до начала перевода. Будущие целевые слова
такой энкодер не читает.

Если состояние каждого направления имеет размер $d$, то $h_j$ имеет
размер $d_h=2d$, а память одного предложения — форму $[n,2d]$.
Декодер может иметь другую размерность состояния $d_s$.

<a id="как-получить-веса"></a>

### Оценки, веса и контекст

Пусть перед следующим предсказанием состояние декодера равно $s_{t-1}$.
Оно используется как запрос к памяти. Функция совместимости получает
запрос и каждое $h_j$ и возвращает число:

$$
e_{tj}=a(s_{t-1},h_j).
$$

Число $e_{tj}$ само по себе не является вероятностью. В аддитивном
варианте оно вычисляется небольшой сетью:

$$
\begin{gathered}
z_{tj}=\tanh(W_qs_{t-1}+W_hh_j),\\
e_{tj}=w_a^\top z_{tj}.
\end{gathered}
$$

Обе проекции имеют одинаковую выходную размерность $d_a$:
$W_q$ — матрица $[d_a,d_s]$, $W_h$ — $[d_a,d_h]$,
$w_a$ — вектор длины $d_a$. За счёт проекций размерности состояния
декодера и памяти не обязаны совпадать. Параметры общие для всех
исходных и целевых позиций; длина предложения не создаёт новых матриц.

Softmax по действительным позициям источника превращает оценки в веса:

$$
\alpha_{tj}=
\frac{\exp e_{tj}}{\sum_{k=1}^{n}\exp e_{tk}}.
$$

Все веса неотрицательны, а сумма одной строки равна единице. Теперь
декодер получает контекст

$$
v_t=\sum_{j=1}^{n}\alpha_{tj}h_j.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/d2l-attention-output.svg|Один запрос: оценки по ключам, softmax и взвешенная сумма значений]]

*Один запрос сравнивается с каждым ключом, softmax нормирует оценки,
а взвешенная сумма собирает значения. В рассматриваемом рекуррентном
attention состояния энкодера используются и как ключи, и как значения.
Zhang, Lipton, Li, Smola,
[Dive into Deep Learning, Attention Scoring Functions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html),
CC BY-SA 4.0; исходный SVG.*

В $v_t$ участвуют все разрешённые позиции, а не одна случайно выбранная.
При конечных оценках в точной арифметике softmax даёт им положительные веса;
несколько весов могут быть одновременно большими. Это **мягкое чтение**
памяти. Жёсткий выбор индекса через argmax или выборку является другой
операцией с другими свойствами градиента.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/distill-rnn_read.svg|Векторы памяти входят в результат с разными весами]]

*Взвешенное чтение памяти: $r=\sum_i a_iM_i$. Здесь $a_i$ соответствуют
нашим весам $\alpha_{tj}$, а $M_i$ — читаемым векторам $h_j$.
Chris Olah, Shan Carter,
[Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/),
CC BY 4.0. Сохранён исходный статический SVG; интерактивное изменение
весов доступно в оригинальной статье.*

### От контекста к следующему слову

В схеме с attention до рекуррентного шага декодер обновляется так:

$$
s_t=f_{\mathrm{dec}}
([E_y[y_{t-1}];v_t],s_{t-1}).
$$

Обозначение $v_t$ отделяет контекст чтения от состояния ячейки LSTM $c$.
В исходных статьях и рисунках контекст часто обозначен $c_t$.

Выходной слой строит распределение следующего токена из состояния
декодера. В исходной модели RNNsearch выход также использовал
предыдущее слово и контекст через дополнительную выходную сеть;
ниже будет упрощённая реализация с линейным выходом.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq-attention.svg|Контекст по состояниям энкодера поступает во вход рекуррентного декодера]]

*Рекуррентная архитектура с attention. Блок Attention вычисляет веса,
Aggregate собирает контекст; результат поступает в рекуррентный декодер.
Стрелка между рекуррентными слоями также сохраняет инициализацию
декодера от энкодера. Zhang et al.,
[D2L, The Bahdanau Attention Mechanism](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html),
CC BY-SA 4.0; исходный SVG.*

<a id="translation-two-reads"></a>

<a id="3-проследим-один-перевод"></a>

## Два шага перевода

Для `black cat sleeps` возьмём те же три представления, что и в
[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора#seq2seq-attention|предыдущей главе]]:

$$
\begin{gathered}
h_1=(1,0),\quad h_2=(0,2),\\
h_3=(-1,1).
\end{gathered}
$$

Матрицы и состояния ниже заданы вручную. Так можно проверить, что
происходит при чтении одной и той же памяти с двумя разными запросами,
не запуская рекуррентные слои. Возьмём

$$
\begin{gathered}
W_q=I,\qquad w_a=(1,1),\\
W_h=\begin{pmatrix}1&1\\-2&0\end{pmatrix}.
\end{gathered}
$$

Проекция $W_h$ превращает исходные векторы в $(1,-2)$, $(2,0)$
и $(0,2)$. Эти координаты нужны для вычисления оценок. Усреднять будем
исходные $h_j$, а не их проекции.

В схеме Bahdanau запрос берётся **до** обновления состояния текущей
ячейкой. Предыдущий целевой токен уже известен, но его эмбеддинг
поступит в рекуррентный слой вместе с результатом чтения $v_t$.

| Цель | Запрос | Токен на входе RNN |
|---|---:|---|
| «чёрная» | $s_0$ | BOS |
| «кошка» | $s_1$ | «чёрная» |
| «спит» | $s_2$ | «кошка» |

### Веса для «кошка»

Пусть $s_1=q=(-1,1)$. Это состояние после шага с BOS, на котором
декодер предсказывал «чёрная». При следующем чтении для позиции
`cat` получаем

$$
\begin{aligned}
W_qq+W_hh_2&=(1,1),\\
z_2&=(0.761594,0.761594),\\
e_2&=1.523188.
\end{aligned}
$$

Для `black` и `sleeps` суммы до tanh равны $(0,-1)$ и $(-1,3)$,
откуда $e_1=-0.761594$ и $e_3=0.233461$. Вычтем максимум $e_2$
перед экспонентой: получим $(0.101796,1,0.275346)$. После деления
на их сумму веса равны

$$
\alpha\approx(0.073918,0.726142,0.199940).
$$

Контекст равен

$$
\begin{aligned}
v_t&=0.073918(1,0)\\
&\quad+0.726142(0,2)\\
&\quad+0.199940(-1,1)\\
&\approx(-0.126022,1.652223).
\end{aligned}
$$

### Веса для «спит»

Для выбора «спит» используем $s_2=q=(1,-1)$. Это состояние уже
учло вход «чёрная»; «кошка» поступит в RNN после вычисления $v_3$.
Память и матрицы прежние. Теперь до tanh получаются $(2,-3)$, $(3,-1)$
и $(1,1)$, а оценки равны $(-0.031027,0.233461,1.523188)$. Наибольший
вес переходит к `sleeps`.

| Веса при выборе | `black` | `cat` | `sleeps` |
|---|---:|---:|---:|
| «кошка» | 0.074 | **0.726** | 0.200 |
| «спит» | 0.142 | 0.185 | **0.673** |

Новый контекст равен $(-0.530466,1.043042)$. В модели с фиксированным
контекстом декодер на обоих шагах имел бы один и тот же вектор источника
$v=h_3=(-1,1)$; его собственное состояние при этом всё равно менялось бы.
Здесь меняется и то, какую смесь исходных представлений получает
декодер. Предсказание русского токена на этом расчёте не проверяется:
оно потребовало бы рекуррентного перехода и выходного слоя.

<details>
<summary>Что меняет tanh в аддитивной оценке</summary>

<h3 id="почему-в-функции-совместимости-нужна-нелинейность">Роль tanh в аддитивной оценке</h3>

Если удалить tanh из выбранной аддитивной функции,

$$
e_j=w_a^\top W_qq+w_a^\top W_hh_j.
$$

Первое слагаемое одинаково для всех $j$. Softmax не меняется
при добавлении одной константы ко всем оценкам, поэтому такой
вариант **потеряет зависимость весов от запроса**.
Tanh смешивает обе проекции до последнего скалярного произведения.
Это объясняет его роль именно в данной функции; например,
билинейный $q^\top Wh_j$ зависит от запроса и без tanh.

</details>

<details>
<summary>Производные по оценкам и значениям</summary>

<h3 id="как-сеть-учится-выбирать-веса">Производные по оценкам и значениям</h3>

Вернёмся к шагу «кошка». Для проверки производных возьмём желаемый
контекст $u=(0,2)$ и потерю $L=\frac12\|v_t-u\|^2$.
Это простая проверка производных; при обучении перевода далее
будет использоваться cross-entropy по токенам.

Обозначим $g=\partial L/\partial v_t=v_t-u$. Через операцию суммы
каждое представление получает прямой вклад $\alpha_jg$.
Дополнительно потеря меняет сами веса. Производная по оценке равна

$$
\frac{\partial L}{\partial e_j}
=\alpha_j\,g^\top(h_j-v_t).
$$

Она сравнивает вклад отдельного вектора с уже собранным контекстом.
В нашем примере получаются
$(0.031985,-0.099358,0.067373)$.
При градиентном шаге в пространстве оценок вторая оценка увеличится,
первая и третья уменьшатся. Через $z_j$ этот сигнал обучает
$W_q,W_h,w_a$, состояние запроса и память энкодера.
Одно обновление общих параметров не обязано буквально повторить
независимый шаг по всем $e_j$.

```python
import torch
torch.set_num_threads(1)
dtype = torch.float64
q = torch.tensor([-1., 1.], dtype=dtype, requires_grad=True)
H = torch.tensor([[1., 0.], [0., 2.], [-1., 1.]],
                 dtype=dtype, requires_grad=True)
Wq = torch.eye(2, dtype=dtype, requires_grad=True)
Wh = torch.tensor([[1., 1.], [-2., 0.]], dtype=dtype, requires_grad=True)
v = torch.ones(2, dtype=dtype, requires_grad=True)
z = torch.tanh((q @ Wq.T)[None, :] + H @ Wh.T)
scores = z @ v
weights = scores.softmax(0)
context = weights @ H
target = torch.tensor([0., 2.], dtype=dtype)
loss = .5 * (context - target).square().sum()
loss.backward()

with torch.no_grad():
    g = context - target
    delta = weights * ((H - context) @ g)
    dz = delta[:, None] * v * (1 - z.square())
    torch.testing.assert_close(q.grad, dz.sum(0) @ Wq)
    torch.testing.assert_close(H.grad, weights[:, None] * g + dz @ Wh)
    torch.testing.assert_close(Wq.grad, dz.sum(0)[:, None] @ q[None, :])
    torch.testing.assert_close(Wh.grad, dz.T @ H)
    torch.testing.assert_close(v.grad, z.T @ delta)
    torch.testing.assert_close(weights, (scores + 1000).softmax(0))

print("scores:", scores.detach().tolist())
print("weights:", weights.detach().tolist())
print("context:", context.detach().tolist())
print("loss:", loss.item())
print("score gradient:", delta.tolist())
with torch.no_grad():
    q2 = torch.tensor([1., -1.], dtype=dtype)
    scores2 = torch.tanh((q2 @ Wq.T)[None, :] + H @ Wh.T) @ v
    weights2 = scores2.softmax(0)
    context2 = weights2 @ H
    assert weights.argmax().item() == 1  # cat
    assert weights2.argmax().item() == 2  # sleeps
    linear1 = (((q @ Wq.T)[None, :] + H @ Wh.T) @ v).softmax(0)
    linear2 = (((q2 @ Wq.T)[None, :] + H @ Wh.T) @ v).softmax(0)
    torch.testing.assert_close(linear1, linear2)
    print("second read:", weights2.tolist(), context2.tolist())

```

Код сопоставляет ручные производные всех пяти групп входов и параметров
с autograd, а также проверяет инвариантность softmax к общей константе.
Сумма производных по оценкам равна нулю: одинаковый сдвиг всех оценок
не меняет ни веса, ни контекст.

В этой проверке $q$ и $H$ заданы как независимые входы. В полной сети
они могут зависеть от общих параметров и предыдущих состояний;
backward суммирует все соответствующие пути.

</details>

<a id="наблюдаемый-alignment"></a>

<a id="таблица-весов-и-наблюдаемое-выравнивание"></a>

### Соответствия слов на карте весов

У реального перевода веса удобно показывать матрицей. Одна строка
соответствует предсказываемому целевому токену, один столбец —
исходной позиции. Строка получается из текущего состояния декодера;
её нужно вычислить заново на следующем шаге.

<details>
<summary>Условный пример весов для более длинной фразы</summary>

Для `The black cat sleeps on the sofa` → «Чёрный кот спит на диване»
веса заданы вручную для чтения матрицы. Они не получены при обучении.

Веса в каждой строке идут в порядке `The black cat sleeps on the sofa`.

| Цель | Веса исходных позиций |
|---|---|
| Чёрный | .05, .76, .12, .02, .01, .02, .02 |
| кот | .03, .11, .78, .04, .01, .01, .02 |
| спит | .01, .02, .09, .80, .02, .02, .04 |
| на | .01, .01, .01, .08, .66, .10, .13 |
| диване | .01, .01, .02, .04, .09, .13, .70 |

Сумма каждой строки равна единице. Эта таблица не показывает строку EOS;
для завершения перевода модель также должна предсказать EOS.

</details>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/seq2seq-attention-bridge-2026-11-10/d2l-bahdanau-pytorch-weights.svg|Матрица весов обученного декодера D2L]]

*Матрица из запуска обученной модели D2L на фразе `i'm home .`.
Четыре столбца слева направо: `i'm`, `home`, точка и EOS;
строки — шаги французского декодера. Темнее означает больше:
на шаге 0 больше всего веса у `home`, на шаге 5 — у точки.
Цвета надо читать по шкале справа: даже самые тёмные клетки здесь
далеки от веса 1. Zhang et al.,
[D2L, §11.4.3, PyTorch](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html#training),
CC BY-SA 4.0; исходный SVG.*

У [Bahdanau, Cho и Bengio, рис. 3](https://arxiv.org/html/1409.0473v7#S4.F3)
оси подписаны словами, а больший вес, наоборот, отмечен светлым цветом.
Там можно проследить перестановку: `European Economic Area` →
`zone économique européenne`, `marine environment` →
`environnement marin`. Эти примеры разбираются и в
[курсе Lena Voita](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#attention_alignment).

На таких матрицах часто видны языковые соответствия, хотя отдельные
метки выравнивания не задавались при обучении. Но одинаковый контекст
можно получить разными весами. Например, для памяти
$(1,0),(0,1),(0.5,0.5)$ обе строки $(0.4,0.4,0.2)$ и
$(0.2,0.2,0.6)$ дают $(0.5,0.5)$.
Следовательно, даже фиксированный результат чтения не всегда
однозначно определяет распределение attention.

Кроме того, $h_j$ содержит окружающий контекст. Большой вес позиции
не означает использование только соответствующего слова.

<a id="мини-реализация-recurrent-attention"></a>

## Рекуррентная модель с attention

В учебной модели вместо перевода нужно развернуть порядок токенов:
`[3,4,5]` → `[5,4,3,EOS]`. Так можно проверить, что двунаправленный
энкодер, внимание и пошаговый декодер работают вместе.

Это упрощённый вариант attention до рекуррентного шага, а не дословная
реализация RNNsearch. Энкодер — двунаправленная GRU с четырьмя
координатами на направление, декодер — GRUCell с состоянием размерности 8
и линейным выходом. Порядок шагов и маска источника соответствуют реализации
[D2L, Defining the Decoder with Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html).
Maxout, многослойные сети и dropout здесь опущены.

Память имеет форму `[B,S,8]`, запрос — `[B,8]`.
После проекций и broadcasting складываются `[B,1,8]` и `[B,S,8]`.
Выходные оценки и веса имеют форму `[B,S]`, контекст — `[B,8]`.
Текущий контекст зависит от состояния декодера, поэтому его нужно
вычислять внутри цикла, а саму память энкодер вычисляет только один раз.

<details>
<summary>Полный код: обучение, генерация и вывод весов</summary>

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

torch.manual_seed(22)
torch.set_num_threads(1)
PAD, BOS, EOS = 0, 1, 2
rows = [[3, 4, 5], [3, 4], [4, 5], [5], [4, 3], [3, 5]]
lengths = torch.tensor([len(row) for row in rows])
src = torch.tensor([row + [PAD] * (3 - len(row)) for row in rows])
target_rows = [[BOS] + row[::-1] + [EOS] for row in rows]
tgt = torch.tensor([row + [PAD] * (5-len(row)) for row in target_rows])
dec_input, targets = tgt[:, :-1], tgt[:, 1:]

class AdditiveAttention(nn.Module):
    def __init__(self, query_size, memory_size, score_size):
        super().__init__()
        self.Wq = nn.Linear(query_size, score_size, bias=False)
        self.Wh = nn.Linear(memory_size, score_size, bias=False)
        self.v = nn.Linear(score_size, 1, bias=False)

    def forward(self, query, memory, valid):
        if valid.dtype != torch.bool or valid.shape != memory.shape[:2]:
            raise ValueError("valid must be a Boolean [B, S] mask")
        if not valid.any(dim=-1).all():
            raise ValueError("every source must have at least one real token")
        features = torch.tanh(
            self.Wq(query)[:, None, :] + self.Wh(memory))
        scores = self.v(features).squeeze(-1)
        weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        context = torch.bmm(weights[:, None, :], memory).squeeze(1)
        return context, weights

class AttentiveSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(6, 6, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(6, 6, padding_idx=PAD)
        self.encoder = nn.GRU(6, 4, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(8, 8)
        self.attention = AdditiveAttention(8, 8, 8)
        self.decoder = nn.GRUCell(6 + 8, 8)
        self.output = nn.Linear(8, 6)

    def encode(self, ids, valid_lengths):
        packed = pack_padded_sequence(
            self.src_emb(ids), valid_lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        packed_outputs, final = self.encoder(packed)
        memory, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=ids.size(1))
        # final[0] is forward; final[1] is backward.
        state = torch.tanh(self.bridge(torch.cat([final[0], final[1]], -1)))
        positions = torch.arange(ids.size(1), device=ids.device)
        valid = positions[None, :] < valid_lengths.to(ids.device)[:, None]
        return memory, state, valid

    def step(self, previous, state, memory, valid):
        context, weights = self.attention(state, memory, valid)
        state = self.decoder(
            torch.cat([self.tgt_emb(previous), context], -1), state)
        return self.output(state), state, weights

    def forward(self, ids, valid_lengths, target_input):
        memory, state, valid = self.encode(ids, valid_lengths)
        logits, attention = [], []
        for previous in target_input.unbind(1):
            out, state, weights = self.step(previous, state, memory, valid)
            logits.append(out)
            attention.append(weights)
        return torch.stack(logits, 1), torch.stack(attention, 1)

    @torch.no_grad()
    def greedy(self, ids, valid_lengths, max_steps=6):
        assert ids.size(0) == 1
        memory, state, valid = self.encode(ids, valid_lengths)
        previous = torch.tensor([BOS], device=ids.device)
        result, attention = [], []
        for _ in range(max_steps):
            out, state, weights = self.step(previous, state, memory, valid)
            attention.append(weights[0])
            previous = out.argmax(-1)
            if previous.item() == EOS:
                return result, True, torch.stack(attention)
            result.append(previous.item())
        matrix = torch.stack(attention) if attention else memory.new_empty((0, ids.size(1)))
        return result, False, matrix

model = AttentiveSeq2Seq()
optimizer = torch.optim.Adam(model.parameters(), lr=.03)

def objective():
    logits, _ = model(src, lengths, dec_input)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                           ignore_index=PAD)

before = objective().item()
for _ in range(400):
    optimizer.zero_grad()
    loss = objective()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
after = objective().item()
model.eval()
print(f"NLL: {before:.6f} -> {after:.6f}")
for row, length in zip(src, lengths):
    result, ended, _ = model.greedy(row[None], length[None])
    print(row[:length].tolist(), "->", result, "EOS:", ended)
_, _, alignment = model.greedy(src[:1], lengths[:1])
print("attention, including the EOS step:")
print(alignment.round(decimals=3))
```

</details>

За 400 обновлений средняя потеря на обучающих данных снизилась
с $1.955285$ до $0.000560$. Декодер развернул все шесть обучающих
последовательностей и для каждой выбрал EOS. На новых входах модель
не проверяли.

### Маска источника и маска потерь

Булева маска `valid` исключает PAD до softmax. Поэтому ни большой
случайный score, ни ненулевой PAD-вектор не должны попадать в сумму.
Пустой разрешённый набор отклоняется: softmax строки из одних
$-\infty$ не определён.

Эта маска не исправляет представления, которые энкодер уже вычислил
с учётом padding. Поэтому bidirectional GRU получает packed sequences,
а после распаковки маска определяет, какие выходы действительны.
Начальное состояние декодера строится из конечных состояний обоих
направлений; обратное конечное состояние нельзя получить как
последний элемент обычного выходного массива.

`ignore_index=PAD` решает другую задачу — исключает дополненные
целевые позиции из cross-entropy. EOS остаётся целью. Чтение
источника и предсказание EOS — разные вычисления.

В этой реализации dropout весов attention отсутствует, поэтому каждая
строка разрешённых весов суммируется в единицу и на обучении.
Если применять dropout после softmax, как в некоторых библиотеках,
сумма конкретной случайной строки после dropout уже не обязана быть единицей.

<a id="краткие-итоги"></a>

### Правильный ответ и веса attention

Для первого обучающего входа `[3,4,5]` модель выдала `[5,4,3]`.
Однако при предсказании первого токена `5` примерно $0.969$ веса пришлось
на **вторую** позицию источника, а не на третью. Это не нарушает
вычисление: двунаправленное представление второй позиции уже содержит
сведения о токене `5`, как и начальное состояние декодера.

Следовательно, правильный выход не требует точного совпадения
тепловой карты с перестановкой исходных ID. Для анализа решения
нужно учитывать не только коэффициенты чтения, но и то, как были
построены читаемые представления и какие ещё каналы состояния
имеет декодер.

<a id="4-bahdanau-и-luong--похожий-принцип-разные-сборки"></a>

## Внимание до и после шага RNN

При выборе «кошка» Bahdanau вычисляет веса из $s_1$, до того как ячейка
получит эмбеддинг «чёрная». В схеме Luong эмбеддинг «чёрная» сначала
поступает в рекуррентную ячейку, а затем новое состояние служит запросом.
Место чтения и формулу оценки можно менять отдельно.

В схеме Bahdanau порядок имеет вид

$$
\begin{gathered}
s_{t-1}\longrightarrow\alpha_t\longrightarrow v_t\\
\longrightarrow s_t\longrightarrow p(y_t\mid y_{<t},x).
\end{gathered}
$$

В схеме Luong сначала выполняется рекуррентный шаг, затем из его
**текущего** состояния читается источник:

$$
\begin{gathered}
s_t\longrightarrow\alpha_t\longrightarrow v_t\\
\longrightarrow\widetilde s_t\longrightarrow p(y_t\mid y_{<t},x).
\end{gathered}
$$

Результат чтения объединяется с текущим состоянием:

$$
\begin{gathered}
\widetilde s_t=\tanh(W_c[v_t;s_t]),\\
p(y_t=j\mid y_{<t},x)=
[\operatorname{softmax}(W_o\widetilde s_t)]_j.
\end{gathered}
$$

В рисунках Luong обычное состояние декодера обозначено $h_t$,
объединённое — $\widetilde h_t$, контекст — $c_t$.
Это наши $s_t,\widetilde s_t,v_t$ соответственно.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-global.png|Глобальное внимание Luong: запрос из текущего состояния декодера ко всему источнику]]

*Global attention: текущее состояние декодера сравнивается со всеми
выходами энкодера, после чего контекст объединяется с состоянием.
Luong, Pham, Manning,
[Effective Approaches to Attention-based NMT, рис. 2](https://aclanthology.org/D15-1166/).
Фрагмент исходного PDF, CC BY-NC-SA 3.0.*

<details>
<summary>Скалярная, билинейная и аддитивная оценки</summary>

<h3 id="функции-совместимости">Функции совместимости</h3>

Для одинаковых размерностей можно использовать $a(q,h)=q^\top h$.
Если размерности различаются, билинейная форма имеет вид
$a(q,h)=q^\top Wh$, где $W$ имеет форму $[d_s,d_h]$.
При фиксированной памяти произведение $Wh_j$ можно подготовить заранее.

Аддитивная функция из предыдущего раздела также применима к запросу
$s_t$, а не только к $s_{t-1}$. Конкатенация с проекцией эквивалентна
сумме двух проекций до нелинейности:

$$
W[q;h]=W_qq+W_hh.
$$

Поэтому названия additive и concat могут описывать одну структуру
функции совместимости. В [arXiv v5 статьи Luong, §3.1](https://arxiv.org/abs/1508.04025v5)
concat записан с tanh и выходным вектором; в PDF из ACL Anthology запись
формулы (8) отличается. Здесь используется явно указанная
нелинейная форма, а не линейная сумма, теряющая зависимость от запроса.

Скалярное произведение дешевле маленькой MLP при одинаковых размерах,
но его масштаб зависит от норм векторов и размерности. Деление
на $\sqrt{d_k}$ в Transformer будет отдельным решением следующей
главы; его не следует молча добавлять к исходному варианту Luong.

</details>

<details>
<summary>Глобальное и локальное внимание: local-m и local-p</summary>

<h3 id="global-local-m-и-local-p">Окно внимания</h3>

Global attention сравнивает запрос со всем источником. Local attention
ограничивает прямое чтение небольшим окном около позиции $p_t$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-local.png|Локальное внимание Luong читает позиции в окне вокруг предсказанного центра]]

*Local attention: в контекст попадают только состояния внутри выбранного
окна. Отдельная ветвь предсказывает его центр $p_t$.
Luong, Pham, Manning,
[Effective Approaches to Attention-based NMT, рис. 3](https://aclanthology.org/D15-1166/),
CC BY-NC-SA 3.0; исходный рисунок без перерисовки.*

В **local-m** центр задаётся как $p_t=t$: предполагается приблизительно
монотонное соответствие. В **local-p** он предсказывается из состояния
декодера:

$$
p_t=n\,\sigma(w_p^\top\tanh(W_ps_t)).
$$

Затем выбирается окно около $p_t$ с радиусом $D$, а веса дополнительно
подавляются гауссовым множителем вдали от центра:

$$
\begin{gathered}
\beta_{tj}=\alpha_{tj}
\exp\left(-\frac{(j-p_t)^2}{2\sigma_p^2}\right),\\
\sigma_p=D/2.
\end{gathered}
$$

Центр вещественный, индексы памяти целочисленные. На границе источника
читаются только существующие позиции; выбор целых границ окна должен
быть определён в реализации.

Произведение в этой формуле само по себе уже не обязано суммироваться
в единицу. Дополнительное деление на $\sum_j\beta_{tj}$ даёт
нормированный вариант и меняет масштаб контекста. Формула (11)
печатной статьи Luong показывает произведение без такого деления.

Возьмём позиции 1–6, центр 3, радиус 1 и одинаковые исходные оценки
в окне. До гауссова множителя позиции 2, 3, 4 имеют веса по $1/3$.
После него остаются приблизительно $0.045112,0.333333,0.045112$,
а их сумма равна $0.423557$. Если читаемые значения равны номерам
позиций, результат суммы — $1.270671$; с дополнительной
нормировкой — $3.0$.

```python
import torch

dtype = torch.float64
positions = torch.arange(1, 7, dtype=dtype)
center = torch.tensor(3., dtype=dtype, requires_grad=True)
radius = 1.
sigma = radius / 2
window = (positions - center.detach()).abs() <= radius
scores = torch.zeros_like(positions)
base = scores.masked_fill(~window, -torch.inf).softmax(0)
gaussian = torch.exp(-.5 * ((positions-center) / sigma).square())
raw = base * gaussian
normalized = raw / raw.sum()
values = positions
context_raw = raw @ values
context_normalized = normalized @ values
context_raw.backward()
print("weights without renormalization:", raw.detach().tolist())
print("sum:", raw.sum().item())
print("contexts:", context_raw.item(), context_normalized.item())
print("center gradient in this fixed window:", center.grad.item())
```

Градиент по центру здесь вычисляется при фиксированном наборе индексов
окна. При пересечении целочисленной границы состав окна меняется;
это не всюду гладкая операция. Само прямое чтение локально,
но представления и состояние декодера могут уже содержать сведения
о других частях источника.

</details>

<details>
<summary>Как Luong передаёт контекст на следующий шаг</summary>

<h3 id="input-feeding-и-ранее-прочитанная-информация">Input feeding</h3>

В варианте Luong с **input feeding** объединённый вектор
$\widetilde s_t$ подаётся вместе с эмбеддингом следующего входного токена
на следующий рекуррентный шаг. Так предыдущий результат чтения
влияет на дальнейший запрос к памяти.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-input-feeding.png|Пунктир возвращает объединённый вектор предыдущего шага во вход следующей ячейки]]

*Пунктирные связи возвращают объединённый вектор предыдущего шага
в следующий вход рекуррентной сети. Это дополнительный канал состояния,
не teacher forcing и не возврат выбранного ID токена.
Luong, Pham, Manning,
[Effective Approaches to Attention-based NMT, рис. 4](https://aclanthology.org/D15-1166/),
CC BY-NC-SA 3.0; исходная схема из PDF.*

Без input feeding в базовом варианте Luong новые запросы не получают
предыдущий объединённый вектор напрямую. Они всё равно связаны
рекуррентным состоянием и историей токенов; «независимые решения
attention» не означают независимые предсказания текста.

Input feeding может помогать учитывать ранее прочитанное, но не
запрещает повторное чтение и не гарантирует покрытие каждого исходного
слова ровно один раз. У Bahdanau контекст уже участвует в обновлении
состояния, поэтому существует другой путь его влияния на будущие шаги.

</details>

<a id="5-что-именно-улучшил-attention"></a>

## Память, градиент и вычисления

<a id="динамический-доступ"></a>

Декодер читает $h_1,\ldots,h_n$. Если источник вдвое длиннее, в памяти
вдвое больше таких векторов. Размер контекста $v_t$ при этом остаётся
равен размеру одного $h_j$.

<a id="короткий-путь-для-информации-и-градиента"></a>

Для сигнала от текущей потери к $h_j$ появляется путь через операцию
взвешенной суммы. Ему не обязательно проходить через конечное
состояние энкодера. При этом вычисление самого $h_j$ всё ещё
использует рекуррентные цепочки.

За дополнительный доступ приходится платить вычислениями.
При аддитивной функции $W_hh_j$ можно вычислить для всех позиций
один раз, затратив порядка $n d_h d_a$ операций. Эта проекция не зависит
от целевого шага; её переиспользование описано и в
[приложении A.1.2 Bahdanau et al.](https://arxiv.org/html/1409.0473v7).
На каждом шаге остаются проекция нового запроса порядка $d_s d_a$,
нелинейность и оценки порядка $n d_a$, softmax по $n$ позициям
и сумма значений порядка $n d_h$.

Для памяти из трёх векторов с $d_h=d_a=8$ проекция даёт
$3\cdot8=24$ числа и требует 192 умножения с накоплением.
Если вычислять её заново для четырёх целевых шагов, только на эту
операцию уйдёт 768 умножений с накоплением вместо 192. Это подсчёт работы
линейной проекции, без оценки времени на CPU или GPU.
Переиспользование проекции не избавляет от расчёта весов $\alpha_t$
и контекста $v_t$: они зависят от очередного состояния декодера.

При обучении сохранённая проекция остаётся частью графа
автодифференцирования. Её использования на разных шагах дают суммарный
градиент по $W_h$ и памяти. Применение `detach()` к этой проекции
отрежет путь через оценки, а не просто сэкономит вычисления.
В учебной модели `self.Wh(memory)` вызывается внутри attention на каждом
шаге, поэтому проекция пересчитывается. Для переиспользования её
вычисляют до цикла декодирования и передают вместе с памятью.

<a id="переменная-длина-памяти"></a>

Общая память выходов энкодера имеет размер порядка $n d_h$.
Хранение всех весов для анализа требует ещё порядка $m n$ чисел.
Эти оценки не включают параметры сетей, их обучение и промежуточные
активации для backward. Local attention сокращает непосредственно
просматриваемую область, но может исключить нужную дальнюю позицию.

<a id="6-почему-этого-всё-ещё-недостаточно"></a>

## От внимания для перевода к self-attention

В рекуррентном энкодере следующее состояние зависит от предыдущего.
Двунаправленность добавляет вторую цепочку, а не делает позиции
одного направления независимыми. Обучение рекуррентного декодера
с teacher forcing тоже сохраняет зависимость $s_t$ от $s_{t-1}$.

Attention между энкодером и декодером не заменяет эти переходы.
Чтобы строить представления входа без рекуррентной цепочки, можно
применить внимание между позициями самого входа. В self-attention
каждая известная позиция формирует собственный запрос к той же
последовательности.

Именно это вычисление будет разобрано дальше. Параллельное вычисление
слоя на известном тексте не делает авторегрессивную генерацию
параллельной по ещё не выбранным токенам.

<a id="7-три-термина-которые-нельзя-смешивать"></a>

### Откуда берутся запросы и память

| Операция | Запрос | Память |
|---|---|---|
| Внимание в RNN-декодере | $s_{t-1}$ или $s_t$ | Выходы энкодера |
| Cross-attention Transformer | Целевая позиция | Выходы энкодера |
| Self-attention | Позиция текста | Позиции того же текста |

В первом случае запросом служит $s_{t-1}$ или $s_t$ в зависимости
от сборки декодера. Ключи используются для вычисления оценок,
значения — для итоговой суммы. В базовой рекуррентной модели обе
роли выполняют $h_j$; в следующей главе появятся отдельные
проекции Q, K и V.

В self-attention из следующей главы запросы, ключи и значения вычисляются
из одной последовательности. При причинной маске запрос каждой позиции
по-прежнему видит только разрешённую часть этой последовательности.

<a id="источники-и-хорошие-продолжения"></a>

## Источники

### Первичные

- Bahdanau, Cho, Bengio, [Neural Machine Translation by Jointly Learning to
  Align and Translate](https://arxiv.org/abs/1409.0473), §3 и приложение A:
  двунаправленные annotations, динамический контекст и совместное обучение.
- Luong, Pham, Manning, [Effective Approaches to Attention-based Neural
  Machine Translation, EMNLP 2015](https://aclanthology.org/D15-1166/),
  §3.1–3.3 и рисунки 2–4: global/local, input feeding.
  Для нелинейной concat-функции дополнительно сверена
  [arXiv v5](https://arxiv.org/abs/1508.04025v5).
  Схемы из ACL PDF воспроизведены по CC BY-NC-SA 3.0.
- Sutskever, Vinyals, Le, [Sequence to Sequence Learning with Neural
  Networks](https://arxiv.org/abs/1409.3215):
  рекуррентная модель без динамического чтения источника.

<a id="два-объяснения-которые-стоит-прочитать-в-оригинале"></a>

### Объяснения

- Zhang, Lipton, Li, Smola, **Dive into Deep Learning**:
  [Attention Scoring Functions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)
  и [The Bahdanau Attention Mechanism](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html).
  Формы, маски, additive attention и декодер с пошаговым чтением памяти.
  Рисунки — CC BY-SA 4.0; приведённый здесь код помечен как упрощённая
  реализация, а не полный код книги.
- Olah, Carter, [Distill — Attention and Augmented Recurrent Neural
  Networks](https://distill.pub/2016/augmented-rnns/):
  чтение памяти и attention как интерфейс между сетями. Рисунок Distill —
  CC BY 4.0; исходные объяснения и интерактивность доступны по ссылке.
- Lena Voita, [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html):
  путь от фиксированного представления к attention и Transformer.
- Jay Alammar, [Visualizing Neural Machine Translation](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/):
  дополнительное пошаговое объяснение перевода.
- [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 07 — Attention|CS224N: Attention]].

**Назад:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]
