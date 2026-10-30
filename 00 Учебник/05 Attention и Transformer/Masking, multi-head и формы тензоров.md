---
title: Маски, несколько голов и формы тензоров
type: textbook-chapter
status: canonical
last_updated: 2026-10-26
source_unit_id:
  - lecture-02-einops-named-axes
  - assignment-01-task-scaled-dot-product-attention
  - assignment-01-task-multihead-self-attention
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
  - https://jalammar.github.io/illustrated-transformer/
  - https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
---

# Маски, несколько голов и формы тензоров

Пусть в одном пакете находятся фразы из двух и четырёх токенов.
Короткая фраза дополнена до общей длины, но её настоящие токены
не должны читать заполнители. При обучении языковой модели нужно
дополнительно закрыть доступ к будущим позициям, хотя весь обучающий
текст уже находится в памяти.

Маска задаёт разрешённые пары запроса и ключа. Несколько голов
вычисляют для этих пар разные оценки и читают разные проекции значений.
В тензоре появятся отдельные оси пакета, головы, запроса и ключа.
Их порядок определяет, какие именно строки будут перемножаться.

## Маска меняет множество доступных ключей

Пусть $S=QK^\top/\sqrt{D_h}$ — матрица оценок. Маска добавляется к $S$ **до**
softmax:

$$
A=\operatorname{softmax}(S+M),\qquad Z=AV.
$$

В разрешённых ячейках $M_{ij}=0$, в запрещённых — $-\infty$. После
экспоненцирования запрещённая ячейка даёт ноль, поэтому она не участвует ни во
взвешенной сумме, ни в нормировке оставшихся весов. Обнулить уже готовую матрицу
$A$ недостаточно: сумма разрешённых весов перестанет равняться единице, если не
выполнить нормировку заново.

Например, оценки одного запроса равны $(2,1,0)$, а третий ключ запрещён.
Обычный softmax даёт примерно $(0.665241,0.244728,0.090031)$.
После удаления третьего веса остаётся сумма $0.909969$.
Если убрать третий ключ до softmax, знаменатель содержит только
$e^2+e^1$, а веса равны $(0.731059,0.268941,0)$.

Для значений $(10,20,100)$ правильный результат — примерно $12.689414$.
У обнулённой, но не перенормированной строки получается $11.546979$:
запрещённый ключ уже повлиял на доли разрешённых значений.

```python
import torch

scores = torch.tensor([2., 1., 0.], dtype=torch.float64)
allowed = torch.tensor([True, True, False])
values = torch.tensor([10., 20., 100.], dtype=torch.float64)
late = scores.softmax(0) * allowed
early = scores.masked_fill(~allowed, -torch.inf).softmax(0)
print("after softmax:", late.tolist(), "sum:", late.sum().item())
print("before softmax:", early.tolist(), "sum:", early.sum().item())
print("outputs:", (late @ values).item(), (early @ values).item())

# Запрещённая большая оценка может обнулить разрешённые веса в floating point.
extreme = torch.tensor([-1000., -1001., 1000.], dtype=torch.float64)
late_extreme = extreme.softmax(0) * allowed
early_extreme = extreme.masked_fill(~allowed, -torch.inf).softmax(0)
assert late_extreme.sum() == 0
torch.testing.assert_close(early_extreme, early)
```

Дополнительная нормировка после удаления весов математически
восстановила бы ответ при точных вычислениях и ненулевой сумме.
В примере `extreme` разрешённые экспоненты уже потеряны из-за underflow;
деление нулей на их нулевую сумму не исправляет результат.

### Причинная маска: не подсматривать ответ

При обучении языковой модели вся последовательность известна и все позиции
обрабатываются параллельно. Но представление позиции $i$, из которого модель
предсказывает следующий токен, должно зависеть только от позиций $j\le i$.
Иначе сеть увидит правильное продолжение во входе и обучающая задача потеряет
смысл. Эту зависимость задаёт нижнетреугольная маска:

$$
M^{\text{causal}}_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/causal-mask-detailed.png]]

*На схеме запрос второго токена (красный путь) сравнивается с ключами
первого и второго токенов. Будущие ключи в вычислении не участвуют.
Автор изображения — dvgodoy,
[оригинал и лицензия CC BY 4.0](https://commons.wikimedia.org/wiki/File:Decoder_self-attention_with_causal_masking,_detailed_diagram.png).*

На странице Lena Voita тот же переход показан как
[готовая анимация masked self-attention](https://lena-voita.github.io/resources/lectures/seq2seq/transformer/masked_self_attn.mp4): при движении по целевой последовательности доступная нижнетреугольная область расширяется на одну позицию.

Для последовательности A, B, EOS сдвинутый вход и цели выглядят так:

| Позиция запроса | Известный вход | Предсказываемая цель | Разрешённые входы |
| --- | --- | --- | --- |
| 0 | BOS | A | BOS |
| 1 | A | B | BOS, A |
| 2 | B | EOS | BOS, A, B |

Диагональ разрешена: токен A на входе позиции 1 уже известен,
а цель этой позиции — ещё не прочитанный B. Маска и сдвиг целей
работают совместно. Если вместо сдвинутого входа подать сами цели,
нижнетреугольная маска не устранит утечку по диагонали.

Маска не делает обучение последовательным. Все строки $S$ по-прежнему можно
вычислить одним матричным умножением; запрещённые связи удаляются перед
softmax. Последовательной остаётся генерация, потому что следующий входной
токен ещё не известен.

### Padding mask: не читать заполнители

Предложения разной длины обычно дополняют специальным токеном до общей длины
$T$. Эти позиции нужны форме тензора, но не содержат текста. Если из семи
позиций настоящими являются только первые пять, маска ключей имеет вид
`[1, 1, 1, 1, 1, 0, 0]`: любой настоящий запрос может читать первые пять
ключей, но не два заполнителя.

Причинная маска зависит от пары `(query position, key position)` и имеет форму
`[T, T]`. Маска заполнителей зависит от примера и ключевой позиции, поэтому её
естественная форма — `[B, T]`. Для decoder-only модели нужны обе:

$$
M_{b,i,j}=M^{\text{causal}}_{i,j}+M^{\text{padding}}_{b,j}.
$$

Важно маскировать именно **ключевые** позиции. Запрос из padding-позиции тоже
можно занулить позже, однако это другая операция: она не защищает настоящие
токены от чтения заполнителей.

## Зачем одной позиции несколько голов

Обозначим через $X_Q,X_K,X_V$ **входы до проекций**. В self-attention это один и тот же $X$, а в cross-attention $X_Q$ приходит из decoder, $X_K=X_V$ — из последнего выхода encoder. Уже спроецированные $Q,K,V$ ниже сохраняют смысл обозначений предыдущей главы.

Один softmax создаёт одно распределение по ключам. Между тем одному слову могут
быть одновременно нужны разные отношения: ближайший сосед, согласуемое
подлежащее и слово, уточняющее значение. Multi-head attention предоставляет
несколько отдельных наборов проекций и несколько распределений
attention. Такие роли не назначаются головам заранее, и несколько голов
могут научиться похожим распределениям. Параметры всех голов обучаются
совместно по общей потере:

$$
\begin{aligned}
Q_r&=X_QW_r^Q,\\
K_r&=X_KW_r^K,\\
V_r&=X_VW_r^V,\\
\operatorname{head}_r&=\operatorname{Attention}(Q_r,K_r,V_r).
\end{aligned}
$$

$$
\begin{gathered}
\operatorname{MHA}(X_Q,X_K,X_V)\\
=\operatorname{Concat}(\operatorname{head}_1,\ldots,
\operatorname{head}_H)W^O.
\end{gathered}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/qkv-for-heads.png]]

*Каждая голова получает собственные проекции запросов, ключей и значений.
Источник: Lena Voita, раздел
[Multi-Head Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#multi_head_attention). На исходной странице рядом доступна анимация всего процесса.*

В классическом Transformer выбирают $d_k=d_v=D_h=D_{model}/H$.
Тогда каждая проекция головы уже общей ширины модели,
а конкатенация имеет ширину $H D_h=D_{model}$.
Это выбор размерностей, а не обязательное свойство любой MHA:
в общей записи $W^O$ имеет форму $[H d_v,D_{model}]$.

Выходная проекция смешивает признаки разных голов для одной позиции.
Она не смешивает строки разных запросов и не заменяет attention между ними.
В формулах здесь используются векторы-строки. PyTorch `Linear` хранит
вес так, что вычисляется $xW^\top$; при загрузке матриц из этих формул
веса нужно транспонировать.

<a id="named-attention-shapes"></a>

## Split и merge: одна операция, две записи

Оси входа называются `batch`, `sequence` и `model_dim`.
Последняя ось после проекции содержит подряд признаки всех голов.
Её нужно разделить на `heads` и `head_dim`, а затем поставить ось голов
перед последовательностью: последние две оси станут матричными.

В [Lecture 2 CS336](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py#L202-L276)
показаны выделение оси голов и обратное объединение через `rearrange`,
а также свёртки по именованным осям через `einsum`.
Сначала запишем преобразование через `reshape` и `transpose`, затем
сопоставим его с записью через именованные оси.

Условия $D_{model}=H D_h$ и равенства соответствующих осей проверяются до
матричного умножения. Broadcasting не создаёт новую семантику: ось размера 1
можно повторить, но нельзя молча решить, является ли она batch, head или query.

Чтобы вычислить запросы всех голов одним умножением, их матрицы $W_r^Q$
объединяют по выходной оси в общую $W^Q$. Результат имеет форму
`[B,T,Dmodel]`: признаки первой головы стоят перед признаками второй
и так далее. Аналогично вычисляют ключи и значения через $W^K$ и $W^V$.
Затем последнюю ось каждой проекции разбивают на номер головы и признаки
внутри головы.

В self-attention можно объединить ещё и три вида проекций:
одно умножение на общую матрицу даст `[B,T,3*Dmodel]`,
после чего результат разделяют на Q, K и V.
В коде ниже оставлены три отдельных `Linear`, чтобы их параметры
и пути вычисления были видны по отдельности.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_qkv.png]]

*Jay Alammar показывает те же отдельные проекции, из которых получаются
запросы, ключи и значения каждой головы. Источник:
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

Пусть $B=2,T=3,D_{model}=4,H=2,D_h=2$.
У первого примера три строки проекции:
$(0,1,2,3)$, $(4,5,6,7)$ и $(8,9,10,11)$.
Первая голова должна получить пары $(0,1),(4,5),(8,9)$,
вторая — $(2,3),(6,7),(10,11)$.

```python
import torch

B, T, Dmodel, H = 2, 3, 4, 2
Dh = Dmodel // H
q = torch.arange(B*T*Dmodel).reshape(B, T, Dmodel)
split = q.reshape(B, T, H, Dh).transpose(1, 2)
wrong = q.reshape(B, H, T, Dh)
print("head 0:", split[0, 0].tolist())
print("wrong head 0:", wrong[0, 0].tolist())
assert split.shape == wrong.shape
assert not torch.equal(split, wrong)

merged = split.transpose(1, 2).contiguous().view(B, T, Dmodel)
torch.testing.assert_close(merged, q)
```

Прямой `reshape(B,H,T,Dh)` даёт правильные размеры, но неправильные пары:
его первая голова получила бы $(0,1),(2,3),(4,5)$.
Проверки только формы этого не заметят.

`transpose` меняет шаги обращения к памяти, а не переставляет
элементы в новое непрерывное хранилище. Поэтому после обратной перестановки
используется `contiguous().view(...)`. `reshape(...)` при необходимости
сам создаёт копию. Для особых размеров `view` может быть допустим
и без копии; общий код не должен полагаться на это совпадение.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_z.png]]

*Результаты голов остаются отдельными до конкатенации. Источник: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_weight_matrix_o.png]]

*После конкатенации выходная матрица $W^O$ смешивает результаты голов и
возвращает ширину $D_{model}$. Источник: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

<a id="einops-attention-shapes"></a>

### Именованные оси: rearrange, einsum и reduce

В выражении `q @ k.transpose(-2, -1)` смысл осей приходится восстанавливать
по окружающему коду. Библиотека `einops` позволяет записать его в самой
операции. Для разделения голов используется шаблон
`"batch token (head coord) -> batch head token coord"`.
Скобки слева означают, что последняя ось объединяет номер головы и координату
внутри неё. Параметр `head=2` определяет длину первой из этих осей;
вторая имеет длину `Dmodel / 2`. Справа указано новое расположение осей.
Поэтому один `rearrange` выражает оба шага предыдущего примера:
разделение последней оси и перестановку токенов с головами.

Порядок внутри скобок задаёт порядок координат. Для `(head coord)` сначала
меняется координата, затем номер головы: признаки первой головы идут подряд.
Шаблон с `(coord head)` выбрал бы чередующиеся признаки. Это другое разбиение
тех же чисел, даже если размер результата совпал.
После вычислений обратный шаблон
`"batch head token coord -> batch token (head coord)"` объединяет головы
в прежнем порядке. Как и `reshape`, `rearrange` при необходимости может
копировать данные; запись не обещает ни отдельного GPU-ядра, ни ускорения.
Семантика порядка и представления без копии описана в
[документации `rearrange`](https://einops.rocks/api/rearrange/).

При вычислении оценок нужно сохранить оси примера и головы, перебрать все
пары «запрос, ключ» и просуммировать произведения координат:

$$
S_{b,h,i,j}=\sum_{c=0}^{D_h-1}Q_{b,h,i,c}K_{b,h,j,c}.
$$

Та же операция в `einops.einsum` задаётся шаблоном
`"batch head query coord, batch head key coord -> batch head query key"`.
Имена по разные стороны запятой описывают два входа. Общая ось `coord`
отсутствует справа, поэтому по ней выполняется суммирование. `batch` и `head`
остаются справа: они выбирают пару матриц, а не суммируются. Разные имена
`query` и `key` создают все пары позиций; их длины могут различаться.
Порядок имён справа определяет порядок осей результата.
У `einops.einsum` сначала передаются тензоры, последним аргументом — шаблон;
это порядок из [API библиотеки](https://einops.rocks/api/einsum/).

Для одного входа `reduce` задаёт свёртку явно. Например,
`"batch head token coord -> batch head token"` с операцией `"sum"`
складывает координаты каждой строки, сохраняя отдельный результат для
каждого примера, головы и токена. Если передать квадраты координат,
получатся квадраты евклидовых длин. Для головы с рядами `(0,1)`, `(4,5)`,
`(8,9)` это `1`, `41` и `145`. Такая сумма не смешивает головы или позиции
и не является нормировкой attention. Для других задач
[`reduce`](https://einops.rocks/api/reduce/) принимает, например, `"mean"`
или `"max"`; выбор операции меняет результат.

Следующий блок можно выполнить независимо от предыдущих. Нужны PyTorch
и `einops` (проверено с версиями 2.14.1 и 0.8.1). Числа 0–23 позволяют
проследить каждый элемент, а не только проверить размеры:

```python
import torch
from einops import rearrange, einsum, reduce

projected = torch.arange(24., dtype=torch.float64).reshape(2, 3, 4)
q_heads = rearrange(
    projected, "batch token (head coord) -> batch head token coord", head=2)
k_heads = rearrange(
    projected.flip(1),
    "batch token (head coord) -> batch head token coord", head=2)
torch.testing.assert_close(
    q_heads, projected.reshape(2, 3, 2, 2).transpose(1, 2))

named_scores = einsum(
    q_heads, k_heads,
    "batch head query coord, batch head key coord -> batch head query key")
torch.testing.assert_close(
    named_scores, q_heads @ k_heads.transpose(-2, -1))
assert named_scores[0, 0, 0].tolist() == [9., 5., 1.]

norm_sq = reduce(
    q_heads.square(), "batch head token coord -> batch head token", "sum")
torch.testing.assert_close(norm_sq, q_heads.square().sum(-1))
assert norm_sq[0, 0].tolist() == [1., 41., 145.]

restored = rearrange(
    q_heads, "batch head token coord -> batch token (head coord)")
torch.testing.assert_close(restored, projected)
print("head 0:", q_heads[0, 0].tolist())
print("scores of first query:", named_scores[0, 0, 0].tolist())
print("squared row norms:", norm_sq[0, 0].tolist())
```

В примере порядок ключевых позиций обращён: первая строка головы ключей
равна `(8,9)`, затем идут `(4,5)` и `(0,1)`. Поэтому запрос `(0,1)`
получает оценки `0·8 + 1·9 = 9`, `5` и `1`. Это ещё не веса attention:
масштабирование, маска и softmax выполняются после этого произведения.

Имена существуют в шаблоне операции, а не прикреплены к тензору.
Если заранее перепутать оси одинаковой длины, библиотека не узнает,
что данные обозначают другое. Например, при `B=H=2` перестановка примеров
и голов сохранит размеры. Проверки по конкретным элементам остаются нужны.
Многоточие `...` в шаблонах может обозначать несколько ведущих осей,
как в `"... query coord, ... key coord -> ... query key"`, но смысл
скрытых за ним осей по-прежнему задаёт вызывающий код.

Последовательность `einsum` → `reduce` → `rearrange` разобрана в
[CS336, Lecture 2, функции `einops_einsum`, `einops_reduce` и `einops_rearrange`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py#L202-L276).

### Две головы на числовом примере

Пусть значения первой головы равны $(0,8)$, второй — $(4,12)$.
Для одного запроса первая голова имеет веса $(0.75,0.25)$,
вторая — $(0.25,0.75)$. Тогда

$$
\begin{gathered}
z_1=0.75\cdot0+0.25\cdot8=2,\\
z_2=0.25\cdot4+0.75\cdot12=10.
\end{gathered}
$$

Такие веса получаются из оценок $(\log3,0)$ и $(0,\log3)$
при скалярных запросах, равных единице. Ниже код задаст проекции,
которые дают именно эти запросы, ключи и значения.

После конкатенации имеем строку $(2,10)$. Выберем выходную матрицу:

$$
W^O=\begin{bmatrix}1&2\\3&-1\end{bmatrix}.
$$

Тогда $(2,10)W^O=(32,-6)$.
Усреднение голов дало бы одно число 6 и потеряло различие между
$(2,10)$ и $(6,6)$. Между тем $(6,6)W^O=(24,6)$.
По одному среднему восстановить оба результата этой проекции нельзя.

## Полная трассировка форм

Пусть пакет содержит $B$ последовательностей длины $T$, ширина модели равна
$D_{model}$, число голов — $H$, а ширина головы — $D_h=D_{model}/H$.

| Шаг | Общая форма | При $B=2,T=3,H=2,D_h=2$ |
| --- | --- | --- |
| вход $X$ | `[B,T,Dmodel]` | `[2,3,4]` |
| три проекции $Q,K,V$ | по `[B,T,Dmodel]` | по `[2,3,4]` |
| split + transpose | по `[B,H,T,Dh]` | по `[2,2,3,2]` |
| $K^\top$ | `[B,H,Dh,T]` | `[2,2,2,3]` |
| оценки $S$ и веса $A$ | по `[B,H,T,T]` | по `[2,2,3,3]` |
| значения $Z=AV$ | `[B,H,T,Dh]` | `[2,2,3,2]` |
| merge и выходная проекция | `[B,T,Dmodel]` | `[2,3,4]` |

В матрице оценок этого примера 36 чисел:
для двух примеров, двух голов и девяти пар позиций.
Если при той же ширине $D_{model}=4$ взять четыре головы ширины 1,
форма оценок станет `[2,4,3,3]`: теперь в ней 72 числа.
Каждая голова хранит собственное распределение по тем же парам позиций,
даже когда у её запросов и ключей меньше признаков.

Четыре проекции при этом по-прежнему содержат
$4D_{model}^2=64$ веса без bias. Для произведения $QK^\top$
нужно $BHT^2D_h=BT^2D_{model}=72$ операций умножения с накоплением
и при двух, и при четырёх головах: число голов компенсируется их шириной.
Для $AV$ получается столько же операций при принятом $d_v=D_h$.
Но число элементов, к которым применяется softmax, выросло с 36 до 72.
Поэтому равенство числа параметров и этих матричных операций
не означает одинакового времени выполнения.

Подсчёт 36 и 72 относится к каждой явно сохранённой матрице $S$ или $A$.
Слитые алгоритмы, например FlashAttention, могут вычислять результат
блоками, не сохраняя целую матрицу. Их память нельзя оценивать
только по форме из этой таблицы.

Softmax выполняется по **последней** оси: для каждого `(b, h, i)` сумма по всем
ключам $j$ равна единице для непустой разрешённой строки. Для пустых строк
ниже задана отдельная политика. При нормировке по оси запросов единица
получалась бы в столбце: распределялись бы доли разных запросов для одного
ключа. Строка запроса уже не задавала бы нормированную смесь значений.

## Broadcasting масок

После split оценки имеют форму `[B, H, Tq, Tk]`. Маске не обязательно физически
хранить все эти элементы: оси размера 1 распространяются по правилам
broadcasting.

| Назначение | Удобная форма | По каким осям распространяется |
|---|---|---|
| одна causal mask для всех | `[1,1,Tq,Tk]` | batch и heads |
| padding ключей каждого примера | `[B,1,1,Tk]` | heads и queries |
| исключение padding-запросов | `[B,1,Tq,1]` | heads и keys |
| своя маска каждого примера | `[B,1,Tq,Tk]` | heads |

Например, позиция 3 второго примера может читать ключ 1, только если
это допускает причинное правило, ключ 1 настоящий и сам запрос
не относится к padding. Для self-attention длины запросов и ключей
совпадают; для cross-attention их нужно задавать отдельно.

### Пустые строки и направление булевой маски

При left padding `[0,0,1,1]` первые два запроса могут не иметь
ни одного разрешённого ключа. Softmax строки из одних $-\infty$
не определён. Подстановка нуля вместо уже получившегося `NaN`
не гарантирует корректный backward через предшествующий softmax.

В реализации ниже пустым строкам перед softmax подставляются конечные
вспомогательные оценки. После softmax их веса заменяются нулями.
Так на выходе получается нулевой $AV$, а запрещённые пути не дают
градиент. Для непустых строк softmax остаётся обычным masked softmax.
Предполагаются конечные входы Q/K/V; маска не исправляет произвольный
`NaN` внутри значений.

Полученная сумма весов для `[0,0,1,1]` равна `[0,0,1,1]`.
Первый настоящий токен читает себя, второй — оба настоящих токена.
Выходная проекция с bias или residual могут снова сделать padding-состояния
ненулевыми; маска целей в функции потерь решает отдельную задачу.

Одно и то же `True` в разных интерфейсах имеет противоположный смысл:

| Интерфейс PyTorch 2.14 | Что означает `True` |
| --- | --- |
| `F.scaled_dot_product_attention(attn_mask=...)` | пара участвует во внимании |
| `nn.MultiheadAttention(attn_mask=...)` | пара запрещена |
| `nn.MultiheadAttention(key_padding_mask=...)` | ключ относится к padding |

Это текущие интерфейсы, а не разделение на «новый» и «старый» PyTorch.
Семантика зафиксирована в документации
[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
и [MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html).
У SDPA вещественная маска, в отличие от булевой, прибавляется к оценкам.

### Прямоугольная маска и позиции в KV-cache

Если запросов два, а исходных ключей пять, оценки имеют форму
`[B,H,2,5]`. Такая ситуация обычна для cross-attention.
Для него причинная маска по индексам перевода и источника обычно
не требуется: весь источник уже известен.

При генерации с KV-cache прямоугольность означает другое.
Пусть один запрос относится к абсолютной позиции 3, а ключи —
к позициям 0, 1, 2, 3. Запросу разрешены все четыре ключа.
Обычная `tril` размера `[1,4]` разрешила бы только первый:
она сравнивает локальные индексы строки и столбца.

Поэтому правило доступа здесь строится по абсолютным позициям.
В PyTorch 2.14 `is_causal=True` для прямоугольного SDPA использует
верхнее левое выравнивание. Оно не сообщает функции, на какой позиции
находится единственный запрос. Это видно на значениях $(1,2,3,4)$
и одинаковых оценках:

```python
import torch
import torch.nn.functional as F

q = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
k = torch.zeros(1, 1, 4, 2, dtype=torch.float64)
v = torch.arange(1., 5., dtype=torch.float64).reshape(1, 1, 4, 1)
query_positions = torch.tensor([3])
key_positions = torch.arange(4)
allowed = key_positions[None, :] <= query_positions[:, None]

cached = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                        dropout_p=0.)
upper_left = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=0.)
print("absolute-position mask:", cached.item())
print("upper-left causal mask:", upper_left.item())
torch.testing.assert_close(cached, torch.full_like(cached, 2.5))
torch.testing.assert_close(upper_left, torch.ones_like(upper_left))
```

Здесь `2.5` — среднее четырёх разрешённых значений,
а `1.0` — чтение только первого.

Теперь в cache уже есть позиции 0–2, а за один вызов обрабатываются
два новых токена, на позициях 3 и 4. Ключи этих токенов добавлены
к трём прежним ключам. У первого запроса ещё нет доступа к позиции 4,
а у второго доступны все пять позиций. Булева маска должна быть такой:

$$
\operatorname{allowed}=\begin{bmatrix}
1&1&1&1&0\\
1&1&1&1&1
\end{bmatrix}.
$$

При нулевых Q и K все разрешённые оценки одинаковы.
Для значений $(1,2,3,4,5)$ первая строка даёт
$(1+2+3+4)/4=2.5$, вторая — $(1+2+3+4+5)/5=3$.
Верхний левый треугольник разрешил бы первой строке только ключ 0,
второй — ключи 0 и 1: результаты были бы 1 и 1.5.
Если вовсе убрать маску, обе строки дадут 3, и первый запрос
прочитает ещё не разрешённое ему будущее значение 5.
Следующий пример проверяет все три варианта:

```python
import torch
import torch.nn.functional as F

q_block = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
k_block = torch.zeros(1, 1, 5, 2, dtype=torch.float64)
v_block = torch.arange(1., 6., dtype=torch.float64).reshape(1, 1, 5, 1)
query_positions = torch.tensor([3, 4])
key_positions = torch.arange(5)
block_allowed = key_positions[None, :] <= query_positions[:, None]
block_cached = F.scaled_dot_product_attention(
    q_block, k_block, v_block, attn_mask=block_allowed, dropout_p=0.)
block_upper_left = F.scaled_dot_product_attention(
    q_block, k_block, v_block, is_causal=True, dropout_p=0.)
block_unmasked = F.scaled_dot_product_attention(
    q_block, k_block, v_block, dropout_p=0.)
print("two-query absolute mask:", block_allowed.int().tolist())
print("two-query outputs:", block_cached.flatten().tolist())
torch.testing.assert_close(
    block_cached.flatten(), torch.tensor([2.5, 3.], dtype=torch.float64))
torch.testing.assert_close(
    block_upper_left.flatten(), torch.tensor([1., 1.5], dtype=torch.float64))
torch.testing.assert_close(
    block_unmasked.flatten(), torch.tensor([3., 3.], dtype=torch.float64))
```

В этом случае запросы занимают последние две позиции непрерывного
набора ключей. Маску можно также получить как нижний треугольник
со сдвигом диагонали $T_k-T_q=3$; такой вариант задаёт
[`causal_lower_right`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.bias.causal_lower_right.html).
При произвольных позициях доступ нужно определять по их значениям,
а для padding и упакованных независимых текстов учитывать ещё
действительность токенов и принадлежность одному тексту.
Подробности хранения KV-cache рассматриваются в разделе инференса.

SDPA применяет dropout согласно переданному `dropout_p`, даже если
вызывающий модуль переведён в `eval()`. При оценивании в него нужно
явно передавать ноль. Все числовые проверки ниже выполняются без dropout.

## Ошибки, которые выглядят правдоподобно

- **Softmax по неверной оси.** Проверка: `weights.sum(-1)` должна состоять из
  единиц для строк, где есть хотя бы один разрешённый ключ.
- **Маска после softmax.** Запрещённые веса станут нулевыми, но оставшиеся не
  будут перенормированы.
- **Перепутаны $T_q$ и $T_k$.** В cross-attention они вообще могут различаться;
  оценки всегда имеют форму `[B, H, Tq, Tk]`.
- **Тихое неверное broadcasting.** Маска `[B,T]` выравнивается с последними двумя осями
  `[B,H,T,T]`. При `B=T` это может пройти без ошибки, но означать не маску
  ключей каждого примера. Для этого нужны `[B,1,1,T]`.
- **Строка целиком замаскирована.** Прямой `torch.softmax` от одних $-\infty$ даёт `NaN`.
  Такая строка требует отдельной политики, особенно при left padding.
- **Забыто объединение голов.** Просто усреднить головы — не то же самое, что
  конкатенация и обучаемая $W^O$.
- **Неверный масштаб.** Делить нужно на $\sqrt{D_h}$, а не на
  $\sqrt{D_{model}}$.

<a id="shape-evidence"></a>

## Минимальная проверка реализации

```python
import math
import torch
from torch import nn

def masked_attention(q, k, v, allowed):
    # Последние оси: [Tq,dk], [Tk,dk], [Tk,dv].
    if allowed.dtype != torch.bool:
        raise ValueError("allowed must be Boolean; True means allowed")
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    allowed = torch.broadcast_to(allowed, scores.shape)
    nonempty = allowed.any(-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(nonempty, masked, torch.zeros_like(masked))
    weights = safe.softmax(-1)
    weights = torch.where(nonempty, weights, torch.zeros_like(weights))
    return weights @ v, weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("positive d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, valid=None, causal=True):
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,T,d_model]")
        B, T, _ = x.shape
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
        if valid.dtype != torch.bool or valid.shape != x.shape[:2]:
            raise ValueError("valid must be a Boolean [B,T] mask")
        valid = valid.to(x.device)
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        allowed = valid[:, None, None, :] & valid[:, None, :, None]
        if causal:
            positions = torch.arange(T, device=x.device)
            causal_mask = positions[None, :] <= positions[:, None]
            allowed = allowed & causal_mask[None, None, :, :]
        z, weights = masked_attention(q, k, v, allowed)
        merged = z.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(merged), weights

torch.set_num_threads(1)
dtype = torch.float64
toy = MultiHeadSelfAttention(2, 2).to(dtype)
WQ = torch.ones(2, 2, dtype=dtype)
WK = torch.eye(2, dtype=dtype) * math.log(3)
WV = torch.tensor([[0., 4.], [8., 12.]], dtype=dtype)
WO = torch.tensor([[1., 2.], [3., -1.]], dtype=dtype)
with torch.no_grad():
    for layer, matrix in [(toy.q_proj, WQ), (toy.k_proj, WK),
                          (toy.v_proj, WV), (toy.out_proj, WO)]:
        layer.weight.copy_(matrix.T)

toy_out, toy_weights = toy(torch.eye(2, dtype=dtype)[None], causal=False)
print("two head rows:", toy_weights[0, :, 0].detach().tolist())
print("projected output:", toy_out[0, 0].detach().tolist())
torch.testing.assert_close(
    toy_weights[0, :, 0],
    torch.tensor([[.75, .25], [.25, .75]], dtype=dtype))
torch.testing.assert_close(
    toy_out, torch.tensor([[[32., -6.], [32., -6.]]], dtype=dtype))

torch.manual_seed(11)
model = MultiHeadSelfAttention(4, 2).to(dtype)
x = torch.randn(3, 4, 4, dtype=dtype, requires_grad=True)
valid = torch.tensor([[False, False, True, True],
                      [True, True, True, True],
                      [False, False, False, False]])
out, weights = model(x, valid)
assert out.shape == (3, 4, 4)
assert weights.shape == (3, 2, 4, 4)
expected_sums = valid[:, None, :].expand_as(weights.sum(-1)).to(dtype)
torch.testing.assert_close(weights.sum(-1), expected_sums)
assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
assert torch.count_nonzero(out[~valid]) == 0
assert torch.isfinite(out).all()
print("left-padding row sums:", weights[0, 0].sum(-1).detach().tolist())

loss = out[valid].square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
assert torch.count_nonzero(x.grad[~valid]) == 0
for parameter in model.parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
```

Две строки весов игрушечного примера равны $(0.75,0.25)$ и
$(0.25,0.75)$, выход первой позиции — $(32,-6)$.
В этом запуске причинная маска отключена: обе головы могут читать
обе известные позиции.

Второй запуск проверяет причинное внимание в пакете с левым padding,
обычным текстом и полностью пустым примером.
У первого примера суммы строк равны $(0,0,1,1)$.
Квадратичная функция от настоящих выходов нужна здесь для проверки
обратного прохода, а не для обучения языковой модели.

Модуль не содержит bias, dropout, RoPE, residual или нормализацию.
Поэтому нулевой $AV$ padding-запроса остаётся нулевым и после $W^O$.
В полном Transformer остальные операции нужно учитывать отдельно.
Маскированный softmax, три проекции и merge следуют конструкции
[D2L](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html)
и [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/);
обработка пустых строк — явно заданное здесь дополнение.

Проверка одной формы не подтверждает порядок данных: пример с `wrong`
имеет правильные размеры. Для сравнения реализаций нужны также
числовые выходы и производные, отдельный проход по головам,
прямоугольные Q/K и проверка независимости от запрещённых токенов.
Допустимая численная погрешность зависит от dtype и вычислительного backend.

В [Assignment 1 CS336](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf),
§3.4.4–3.4.5, заданы контракты для трёх- и четырёхмерного attention
и для полного MHA. В задании отдельно применяется RoPE;
повороты запросов и ключей разбираются в следующей позиционной главе.
Для выполнения этого задания к модулю требуется добавить RoPE.

Полный Transformer добавит к attention FFN, residual и нормализацию.
Связь форм с параметрами, FLOPs и памятью продолжена в
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость#Сначала имена осей, затем операции|главе о стоимости Transformer]],
а реализация — в [[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля#Этап 2. Модель от базовых операций к логитам|Capstone 1]].

## Источники и визуальные продолжения

- [Stanford CS336 Spring 2026, Lecture 2 — зафиксированный исходник](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py) — именованные оси, `einsum` и связь форм с подсчётом ресурсов.
- [Stanford CS336 Spring 2026, Assignment 1 — зафиксированное задание](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf) — проверяемые интерфейсы attention и Transformer.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  разделы 3.2.2–3.2.3 — исходные определения multi-head и masked attention.
- Lena Voita, [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — анимации причинной маски и вычислений отдельных голов.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — пошаговое визуальное объяснение split, concat и $W^O$.
- Dive into Deep Learning, [Multi-Head Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html) — реализация с явными преобразованиями форм.
- Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — реализация маски последующих позиций и multi-head attention рядом с формулами статьи.
- PyTorch, [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) — точная семантика `attn_mask`, `is_causal` и dropout.

**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]]
