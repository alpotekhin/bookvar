---
title: Self-Attention — Q, K, V
type: textbook-chapter
status: canonical
last_updated: 2026-11-21
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
---

# Self-Attention — Q, K, V

<a id="зачем-слову-смотреть-на-другие-слова"></a>

## Представление токена в контексте

В выражениях `American shrew mole`, `one mole of carbon dioxide` и
`a biopsy of the mole` слово `mole` обозначает крота, моль вещества и родинку.
Если токенизатор выдал для него один и тот же ID, таблица эмбеддингов
вернёт один и тот же вектор. В этой операции сведения о соседних словах
ещё не используются. Позиционные добавки могут различаться, но сами
по себе они тоже не объясняют, какое значение слова нужно в предложении.

Такой пример используется в
[лекции 3Blue1Brown об attention](https://www.3blue1brown.com/lessons/attention).
Реальный токенизатор может разбить это слово на несколько токенов.

В предыдущей главе состояние декодера запрашивало представления
исходного предложения. В self-attention каждая позиция самой
последовательности получает свой запрос и читает представления
разрешённых позиций. Обычно среди них есть и сама позиция. На выходе
получается последовательность той же длины.

В предложении «Кот, который весь день бегал, устал» представление
сказуемого в энкодере можно вычислить с учётом подлежащего, хотя слова
не соседствуют. В задаче предсказания следующего токена вход и цель
сдвинуты: чтобы предсказать «устал», модель использует состояние
предыдущей позиции. Если разрешить ей читать само «устал», цель
окажется доступна во входе.

<a id="1-самый-простой-канал-среднее-прошлого"></a>

## Усреднение текущей и предыдущих позиций

Пусть $x_i$ — строка признаков позиции $i$. Для начала соберём
контекст как среднее текущей и предшествующих строк:

$$
\bar x_i=\frac1i\sum_{j\le i}x_j.
$$

Возьмём четыре вектора $(1,0)$, $(0,2)$, $(3,1)$ и $(-1,3)$.
Первый результат равен $(1,0)$, второй — $(0.5,1)$,
третий — $(4/3,1)$, четвёртый — $(0.75,1.5)$.
Например, третья строка складывает первые три вектора и делит сумму на три.

Все четыре результата можно получить одним умножением:

$$
\begin{bmatrix}
1&0&0&0\\
\frac12&\frac12&0&0\\
\frac13&\frac13&\frac13&0\\
\frac14&\frac14&\frac14&\frac14
\end{bmatrix}
\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}.
$$

Строка матрицы соответствует выходной позиции, столбец — входной. Нули
выше диагонали исключают будущие входы. Для энкодера можно разрешить
чтение всей известной последовательности.

В третьей строке доля каждого из трёх доступных входов равна $1/3$,
какими бы ни были их векторы. В self-attention эти доли меняются
вместе с входными векторами.
[В лекции Карпати](https://www.youtube.com/watch?v=kCc8FmEb1nY)
такое усреднение предшествует обучаемому attention.

```python
import torch

X = torch.tensor([[1., 0.], [0., 2.], [3., 1.], [-1., 3.]],
                 dtype=torch.float64)
allowed = torch.ones(4, 4, dtype=torch.bool).tril()
A_mean = allowed.to(X.dtype)
A_mean = A_mean / A_mean.sum(-1, keepdim=True)
out = A_mean @ X

loop = torch.stack([X[:i+1].mean(0) for i in range(len(X))])
torch.testing.assert_close(out, loop)

# Одинаковые оценки на разрешённых позициях дают то же среднее.
scores = torch.zeros(4, 4, dtype=X.dtype).masked_fill(~allowed, -torch.inf)
torch.testing.assert_close(scores.softmax(-1), A_mean)
print(out)
```

Пока все входные $x_i$ известны, строки этого умножения вычисляются
одновременно. Для вычисления $\bar x_3$ не нужен результат $\bar x_2$.
Это отличается от рекуррентного перехода, принимающего предыдущее
скрытое состояние.

<a id="2-сделаем-веса-зависимыми-от-содержимого"></a>

## Запросы, ключи и значения

Сохраним схему взвешенной суммы. Для позиции $i$ нужно определить,
по каким признакам сравнивать её с позицией $j$ и какой вектор
из $j$ войдёт в сумму. Для этого из каждой входной строки $x_i$
вычисляются три проекции:

$$
\begin{gathered}
q_i=x_iW_Q,\qquad k_i=x_iW_K,\\
v_i=x_iW_V.
\end{gathered}
$$

Запрос (query) $q_i$ строится для позиции, чей выход вычисляем.
Его сравнивают с ключом (key) $k_j$ каждой доступной позиции.
Из этих сравнений рассчитывают веса для векторов значений (value) $v_j$.
Три матрицы — параметры головы; одни и те же матрицы применяются ко всем
позициям. Если в последовательности больше токенов, число параметров
в этих проекциях не меняется.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png|Вектор Thinking умножается на три матрицы и даёт отдельные q₁, k₁ и v₁]]

*Один входной вектор умножается на три разные матрицы.
Для остальных позиций повторяются те же проекции.
Jay Alammar, [The Illustrated Transformer, Self-Attention in Detail](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

<a id="4-численный-пример-который-можно-проверить-вручную"></a>

## Выход для третьего токена

Возьмём три позиции: `a`, `cat` и `sleeps`. Для ручного расчёта
считаем каждое слово одним токеном и разрешаем читать все три позиции.
Пусть входные строки образуют единичную матрицу $X=I_3$.
Все три матрицы проекций имеют форму $[3,2]$; в $W_Q$ и $W_K$
для этого примера стоят одинаковые числа:

$$
\begin{gathered}
Q=K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\\
V=
\begin{bmatrix}
1&0\\
0&2\\
3&1
\end{bmatrix}.
\end{gathered}
$$

При $X=I_3$ умножение на $x_i$ выбирает $i$-ю строку каждой матрицы
проекции. Так $x_3=(0,0,1)$ даёт $q_3=k_3=(1,1)$ и $v_3=(3,1)$.
Равенство $q_3=k_3$ здесь следует из выбранных чисел: параметр $W_Q$
и параметр $W_K$ могут меняться независимо. Если бы входы удовлетворяли
$x_3=x_1+x_2$, линейность потребовала бы $v_3=v_1+v_2$. Строки $I_3$
позволяют задать все три значения независимо.

<a id="шаг-a-dot-products"></a>

<a id="шаг-a-скалярные-произведения"></a>

### Сравнение запроса с ключами

Оценка ключа $k_j$ для запроса $q_i$ равна скалярному произведению
$s_{ij}=q_ik_j^\top$. Запрос и ключ имеют одинаковое число координат
$d_k$, иначе это произведение нельзя вычислить. Для третьего запроса:

$$
q_3K^\top=(1,1,2).
$$

Первый ключ даёт $1\cdot1+1\cdot0=1$, второй — $1\cdot0+1\cdot1=1$,
третий — $1\cdot1+1\cdot1=2$. При фиксированном запросе изменение
одного ключа меняет только его оценку до softmax. Изменение запроса
может изменить все оценки строки. Направления $i\to j$ и $j\to i$
обычно имеют разные оценки: в них запрос и ключ берутся из разных позиций.

<a id="шаг-b-scaling"></a>

<a id="шаг-b-масштабирование"></a>

### Масштабирование и softmax

Делим каждое произведение на $\sqrt{d_k}=\sqrt2$:

$$
\begin{gathered}
\frac{q_3K^\top}{\sqrt2}
\\
\approx(0.707107,0.707107,1.414214).
\end{gathered}
$$

<a id="шаг-c-softmax"></a>

Чтобы получить веса, берём экспоненты трёх оценок и делим каждую
на их сумму. Вычитание одного и того же числа из всех оценок
сокращается в дробях. Вычтем $1/\sqrt2$; тогда экспоненты равны
$(1,1,e^{1/\sqrt2})$, а их сумма — $2+e^{1/\sqrt2}$:

$$
a_3\approx(0.248255,0.248255,0.503490).
$$

Если какой-либо ключ запрещён маской, его оценку перед softmax
заменяют на $-\infty$. Тогда $\exp(-\infty)=0$: этот ключ не участвует
в знаменателе и получает нулевой вес, если в строке есть разрешённый ключ.

<a id="шаг-d-weighted-sum-values"></a>

<a id="шаг-d-взвешенная-сумма-значений"></a>

### Сумма векторов значений

Вес $a_{3j}$ умножается на $v_j$, а три вклада складываются:

$$
\begin{aligned}
z_3&=0.248255(1,0)\\
&\quad+0.248255(0,2)\\
&\quad+0.503490(3,1)\\
&\approx(1.758725,1).
\end{aligned}
$$

Третий вектор получил примерно половину веса, но остальные два
тоже вошли в сумму. Вторую координату можно проверить отдельно:
$2\cdot0.248255+0.503490=1$ с учётом округления.

<a id="3-почему-нужен-отдельный-value"></a>

Если при тех же ключах и запросах заменить только $v_3=(3,1)$ на
$(3,3)$, оценки и веса останутся прежними. Вторая координата $z_3$
увеличится на $2a_{33}\approx1.006980$; новый выход —
$(1.758725,2.006980)$. При $X=I_3$ достаточно изменить вторую координату
третьей строки $W_V$. Так одна проекция определяет, какой вектор войдёт в
сумму, а две другие — с каким весом.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/self-attention-output.png|Для Thinking запрос сравнивают с двумя ключами, оценки делят на 8 и используют веса 0,88 и 0,12 для суммы значений]]

*На рисунке Джея Аламмара запрос для `Thinking` сравнивается с ключами
`Thinking` и `Machines`. В примере $d_k=64$, поэтому оценки делятся
на 8. Вес 0,88 умножает первое значение, 0,12 — второе.
Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/#self-attention-in-detail),
CC BY-NC-SA 4.0.*

До dropout веса разрешённых ключей неотрицательны и суммируются в
единицу, так что $z_i$ лежит в выпуклой оболочке их значений. В этой
сумме softmax нормирует оценки по позициям, а выходной softmax языковой
модели — по словарю. Поэтому $z_i$ остаётся вектором признаков.

Каждая строка таблицы показывает веса $a_{ij}$ одного запроса.
Столбцы подписаны векторами, на которые умножаются эти веса.
Числа в таблице округлены до четырёх знаков.

| Запрос | $v_1$ | $v_2$ | $v_3$ |
| --- | ---: | ---: | ---: |
| $q_1$ | 0.4011 | 0.1978 | 0.4011 |
| $q_2$ | 0.1978 | 0.4011 | 0.4011 |
| $q_3$ | 0.2483 | 0.2483 | 0.5035 |

Даже при $Q=K$ веса $a_{13}=0.401112$ и $a_{31}=0.248255$ различаются.
Оценка третьего ключа для первого запроса та же, что оценка первого
ключа для третьего запроса. Но в первой строке исходные оценки —
$(1,0,1)$, в третьей — $(1,1,2)$. Поэтому разные суммы экспонент дают
разные доли.

В строке `cat` на `sleeps` приходится вес $0.401112$. Для энкодера,
которому дана вся фраза, такая связь допустима. Для модели, которая
должна по `a cat` предсказать `sleeps`, это подсказка из будущего.
[[00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров#Что видит второй токен в a cat sleeps|Пересчитаем ту же строку с причинной маской]].

<a id="всё-вычисление-на-двух-рисунках"></a>

## Матричная запись и формы

Соберём $T$ входных векторов в строки $X$. Умножения $Q=XW_Q$,
$K=XW_K$ и $V=XW_V$ сразу дают все три набора проекций. Если ширина
входа $d_{\mathrm{model}}$, то $W_Q,W_K$ имеют форму
$[d_{\mathrm{model}},d_k]$, а $W_V$ — $[d_{\mathrm{model}},d_v]$.
Для взвешенной суммы ширина $d_v$ может отличаться от $d_k$:
каждый вес умножается на целый вектор значения.

| Величина | Форма | Смысл строки |
| --- | --- | --- |
| $X$ | $[T,d_{\mathrm{model}}]$ | вход позиции |
| $Q,K$ | $[T,d_k]$ | запрос или ключ |
| $V$ | $[T,d_v]$ | значение позиции |
| $S=QK^\top/\sqrt{d_k}$ | $[T,T]$ | оценки одного запроса ко всем ключам |
| $A=\operatorname{softmax}(S)$ | $[T,T]$ | веса по ключам одного запроса |
| $Z=AV$ | $[T,d_v]$ | результат для одного запроса |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png|Матрицы запросов и ключей дают таблицу оценок; после масштабирования и softmax веса умножаются на матрицу значений]]

*Матричная запись: сначала $QK^\top$, затем масштабирование и построчный
softmax, затем умножение на $V$. Каждому запросу соответствует одна
строка результата. Jay Alammar,
[The Illustrated Transformer, Matrix Calculation of Self-Attention](https://jalammar.github.io/illustrated-transformer/#matrix-calculation-of-self-attention),
CC BY-NC-SA 4.0.*

Без маски вычисление записывается так:

$$
\begin{gathered}
\operatorname{Attention}(Q,K,V)=\\
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\end{gathered}
$$

Softmax применяют к каждой строке $S$ отдельно, по оси ключей, а не
ко всей матрице. Суммы столбцов $A$ не обязаны быть равны единице.
Умножение $AV$ повторяет взвешенную сумму для каждой строки $A$.

<a id="5-зачем-делить-на-dksqrtd_kdk"></a>

<a id="5-масштабирование-оценок"></a>

## Масштаб оценок и производная softmax

Предположим, что все компоненты $q$ и $k$ независимы, имеют среднее
ноль и дисперсию один. Тогда $\mathbb E[q_ak_a]=0$, а по независимости
$\mathbb E[q_a^2k_a^2]=\mathbb E[q_a^2]\mathbb E[k_a^2]=1$.
Каждое произведение имеет дисперсию один. Дисперсии независимых
слагаемых складываются:

$$
\begin{gathered}
\operatorname{Var}(qk^\top)
=\sum_{a=1}^{d_k}\operatorname{Var}(q_ak_a)=d_k,\\
\operatorname{Var}\left(
qk^\top/\sqrt{d_k}\right)=1.
\end{gathered}
$$

Это модель масштаба при указанных предположениях, приведённая
в [§3.2.1 исходной статьи](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1).
В обученной сети координаты могут коррелировать.
В предыдущем числовом примере $Q=K$, поэтому предположение
о независимых случайных запросах и ключах к нему не относится.

Softmax чувствителен к разностям оценок. У пары $(0,1)$
больший вес примерно $0.731059$, у $(0,8)$ — уже $0.999665$.
Производная этого веса по собственной оценке равна $p(1-p)$:
она уменьшается примерно с $0.196612$ до $0.000335$.
Через такой почти насыщенный softmax изменение оценок слабо
меняет смесь значений.

```python
import torch

for divisor in [1., 8.]:
    scores = torch.tensor([0., 8.], dtype=torch.float64) / divisor
    weights = scores.softmax(0)
    p = weights[1]
    print("divisor:", divisor,
          "weights:", weights.tolist(),
          "softmax diagonal:", (p * (1-p)).item())

# Большая общая константа не создаёт острого распределения.
torch.testing.assert_close(
    torch.tensor([100., 100.]).softmax(0),
    torch.tensor([.5, .5]))
```

Масштабирование на $\sqrt{d_k}$ сдерживает рост оценок с размерностью.
Оно не гарантирует умеренные веса для любых обученных матриц.
В формулах используется размерность ключа одной головы, не автоматически
$d_{\mathrm{model}}$.

<a id="6-минимальная-реализация-одной-головы"></a>

## Обучение проекций

Функция `attention` в коде ниже принимает уже вычисленные $Q,K,V$.
Последние две оси — `[Tq,dk]`, `[Tk,dk]` и `[Tk,dv]`.
`Tq` и `Tk` могут различаться; в нашем примере они равны трём.

### Обратный проход на том же примере

Градиент по $X$ складывается из трёх вкладов: через запросы, ключи
и значения. Начнём с производной по выходу $G=\partial L/\partial Z$:

$$
\begin{gathered}
D_V=A^\top G,\qquad D_A=GV^\top,\\
b_i=\sum_\ell a_{i\ell}(D_A)_{i\ell},\\
(D_S)_{ij}=a_{ij}((D_A)_{ij}-b_i),\\
D_Q=D_SK/\sqrt{d_k},\\
D_K=D_S^\top Q/\sqrt{d_k}.
\end{gathered}
$$

Здесь $S=QK^\top/\sqrt{d_k}$ уже содержит масштабирование;
деление на $\sqrt{d_k}$ появляется в производных по Q и K.
Величина $b_i$ — взвешенное среднее входящего сигнала одной строки.
Обратный проход через softmax вычитает это среднее из производной
по каждому весу и умножает результат на сам вес.
Матрица $D_S$ имеет ту же форму $[T,T]$, что и оценки.
Далее правило цепочки даёт, например,
$\partial L/\partial W_Q=X^\top D_Q$.
Поскольку один $X$ использован в трёх проекциях, его градиент
содержит сумму трёх вкладов.

Для проверки выберем желаемый выход третьей позиции $u=(0,2)$
и потерю $\frac12\|z_3-u\|^2$. Первые две строки не входят в эту
вспомогательную потерю, но их ключи и значения влияют на третью.
Это проверка механизма, не обучение языковой модели.

Третья строка $G$ равна $(1.758725,-1)$, остальные две — нулю.
Поэтому производная по каждому значению — это $G_3$, умноженный
на вес, с которым значение вошло в $z_3$. Например:

$$
\begin{gathered}
(D_V)_1=a_{31}G_3\\
\approx0.248255\,(1.758725,-1)\\
\approx(0.436612,-0.248255).
\end{gathered}
$$

Все три значения повлияли на $z_3$ и получают ненулевые производные,
хотя потеря вычислена только для последнего выхода.

Чтобы изменить выбор значений, нужно пройти от ошибки к весам,
а затем через softmax к оценкам. Производная по весу $a_{3j}$
равна скалярному произведению $G_3v_j^\top$:

$$
\begin{aligned}
(D_A)_{31}&\approx1.758725,\\
(D_A)_{32}&=-2,\\
(D_A)_{33}&\approx4.276174.
\end{aligned}
$$

Эти производные ещё не учитывают, что увеличение одной оценки
меняет через softmax все веса строки. Взвешенное среднее трёх
полученных чисел равно $b_3\approx2.093112$.
Для оценки второго ключа получаем:

$$
\begin{aligned}
(D_S)_{32}
&\approx0.248255\,(-2-2.093112)\\
&\approx-1.016136.
\end{aligned}
$$

Для всей третьей строки $D_S$ получается примерно
$(-0.083013,-1.016136,1.099149)$. Сумма равна нулю:
одинаковая добавка ко всем оценкам не меняет softmax.
При спуске по самим оценкам первые две увеличились бы,
а третья уменьшилась. В модели оценки не являются отдельными
параметрами: этот сигнал проходит дальше к Q и K.

Здесь ненулевой градиент получает только запрос $q_3$:
примерно $(0.718517,0.058699)$. Первые два запроса участвуют
только в выходах, не вошедших в потерю. Напротив, все три ключа
использовались третьим запросом. Например, градиент $k_1$ равен
примерно $(-0.058699,-0.058699)$. Градиенты по ключам и значениям
передадут ошибку также входам первых двух позиций.

Наконец, умножение на $X^\top$ переносит производные на общие матрицы
проекций. В этом примере $X=I_3$, поэтому $\partial L/\partial W_Q=D_Q$;
при других входах в каждой строке матрицы складываются вклады
тех позиций, где использовался соответствующий входной признак.

### Код прямого и обратного прохода

```python
import math
import torch

torch.set_num_threads(1)
dtype = torch.float64

def attention(q, k, v):
    # q: [..., Tq, dk], k: [..., Tk, dk], v: [..., Tk, dv]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights

X = torch.eye(3, dtype=dtype, requires_grad=True)
WQ = torch.tensor([[1., 0.], [0., 1.], [1., 1.]],
                  dtype=dtype, requires_grad=True)
WK = WQ.detach().clone().requires_grad_()
WV = torch.tensor([[1., 0.], [0., 2.], [3., 1.]],
                  dtype=dtype, requires_grad=True)
Q, K, V = X @ WQ, X @ WK, X @ WV
Z, A = attention(Q, K, V)
target = torch.tensor([0., 2.], dtype=dtype)
loss = .5 * (Z[-1] - target).square().sum()
loss.backward()

with torch.no_grad():
    G = torch.zeros_like(Z)
    G[-1] = Z[-1] - target
    dV = A.T @ G
    dA = G @ V.T
    dS = A * (dA - (A * dA).sum(-1, keepdim=True))
    dQ = dS @ K / math.sqrt(Q.size(-1))
    dK = dS.T @ Q / math.sqrt(Q.size(-1))
    torch.testing.assert_close(WQ.grad, X.T @ dQ)
    torch.testing.assert_close(WK.grad, X.T @ dK)
    torch.testing.assert_close(WV.grad, X.T @ dV)
    torch.testing.assert_close(
        X.grad, dQ @ WQ.T + dK @ WK.T + dV @ WV.T)
    torch.testing.assert_close(A.sum(-1), torch.ones(3, dtype=dtype))

print("weights:", A.detach().tolist())
print("outputs:", Z.detach().tolist())
before = loss.item()
with torch.no_grad():
    for parameter in [WQ, WK, WV]:
        parameter -= .05 * parameter.grad
    updated, _ = attention(X @ WQ, X @ WK, X @ WV)
    after = (.5 * (updated[-1] - target).square().sum()).item()
print("one SGD step:", before, "->", after)
assert after < before
```

Ручные производные проверяются для всех трёх матриц и общего входа.
Это позволяет обнаружить ошибку транспонирования или softmax по другой
оси даже тогда, когда формы выхода случайно совпали.
Обновляются только матрицы; $X$ остаётся заданным входом.
При шаге 0.05 эта вспомогательная потеря уменьшается
с 2.046556 до 1.838140. Один такой шаг не означает, что голова
уже обучена решать какую-либо языковую задачу.

Маски, dropout, несколько голов и выходная проекция здесь опущены.
Из-за $d_v=2$ выход имеет две координаты, хотя $X$ имеет три.
Чтобы вернуть размер для residual-соединения, требуется проекция;
её и объединение нескольких голов разбирает
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|следующая глава]].

## Параллельное вычисление и порядок позиций

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/cnn-rnn-self-attention.svg|Свёртка читает соседние входы, RNN передаёт состояние вдоль последовательности, полное self-attention соединяет каждый выход со всеми входами]]

*Связи между позициями в CNN, RNN и полном self-attention.
У RNN переход к следующему состоянию использует предыдущее;
у self-attention каждая выходная позиция непосредственно читает входные.
Zhang, Lipton, Li, Smola,
[Dive into Deep Learning, §11.6.2](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html),
CC BY-SA 4.0.*

Все строки Q/K/V строятся из входа текущего слоя. Строке $z_3$
не нужен уже вычисленный $z_2$ того же слоя.
Поэтому матричные операции можно выполнять параллельно по известным
позициям. При генерации неизвестного продолжения после выбора очередного
токена меняется вход следующего шага; эта зависимость не исчезает.

У полного внимания цена такого доступа квадратична по длине:
$QK^\top$ требует порядка $T^2d_k$ операций, умножение на $V$ —
порядка $T^2d_v$. Проекции добавляют работу порядка
$T d_{\mathrm{model}}(2d_k+d_v)$.
Обычная реализация также хранит $T^2$ весов на голову.
Это описание прямой матричной реализации; FlashAttention позднее
позволит вычислять тот же результат без сохранения всей матрицы в HBM.

Само немаскированное внимание без позиционных признаков не узнаёт
порядок слов. Если переставить строки $X$, одинаково переставятся
строки Q/K/V, обе оси весов и строки выхода.
Для перестановочной матрицы $P$:

$$
\begin{gathered}
A(PX)=P A(X)P^\top,\\
Z(PX)=P Z(X).
\end{gathered}
$$

Например, перестановка $(3,1,2)$ строк нашего примера даёт
прежние выходы в порядке $(z_3,z_1,z_2)$.
Содержимое представления одного и того же токена от этой перестановки
не меняется, меняется только место его строки.
Так нельзя различать роли в `Alice follows Bob` и `Bob follows Alice`,
используя порядок как признак.

Позиционные признаки и маски меняют условия этого утверждения.
Для причинного усреднения из начала главы произвольная перестановка
уже не сохраняет правило доступа. Позиционные представления подробно
разобраны в [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|отдельной главе]]
перед полной архитектурой Transformer.

<a id="что-легко-перепутать"></a>

## Что показывают веса внимания

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_self-attention_visualization.png|В энкодере линии от позиции it ведут к словам того же предложения, в том числе к the animal]]

*Визуализация внимания к позиции `it`: линии соединяют её с позициями
источника, а их толщина показывает веса внимания. Это encoder self-attention,
поэтому доступны и правые соседи. Jay Alammar,
[The Illustrated Transformer, Self-Attention at a High Level](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

По такой карте видно, с какими коэффициентами голова смешивает
векторы значений. Толстая линия сама по себе не доказывает, что
удаление соответствующего токена изменит предсказание модели.
Даже при неизменных значениях разные наборы коэффициентов могут
дать один и тот же выход.

<!-- attention-identifiability 2026-10-26 -->
### Different weights, the same output

Let the values be $v_1=(1,0)$, $v_2=(0,1)$ and
$v_3=(0.5,0.5)$. The third value is the midpoint of the first two.
For scalar keys $K=(0,0,1)^{\top}$ and a scalar query $q$, the scores
are $(0,0,q)$, so

$$
a(q)=\frac{(1,1,e^q)}{2+e^q}.
$$

| Query | Attention weights | Output $a(q)V$ |
|---|---|---|
| $\ln2$ | $(0.25,0.25,0.50)$ | $(0.5,0.5)$ |
| $-\ln2$ | $(0.40,0.40,0.20)$ | $(0.5,0.5)$ |

Both rows are strictly positive softmax distributions from finite Q/K
scores. Moving weight from $v_3$ equally to $v_1$ and $v_2$ preserves
the mixture. Their difference $\delta=(0.15,0.15,-0.30)$ satisfies

$$
\delta V=0,\qquad \sum_j\delta_j=0.
$$

The code constructs both cases with shared projections from $X=I_3$;
only the third query changes. Here $d_k=1$, $d_v=2$.

```python
import math
import torch

dtype = torch.float64
X = torch.eye(3, dtype=dtype)
WK = torch.tensor([[0.], [0.], [1.]], dtype=dtype)
WV = torch.tensor([[1., 0.], [0., 1.], [.5, .5]], dtype=dtype)
K, V = X @ WK, X @ WV
rows, outputs = [], []
for query in [math.log(2), -math.log(2)]:
    WQ = torch.tensor([[0.], [0.], [query]], dtype=dtype)
    Q = X @ WQ
    A = (Q @ K.T).softmax(-1)  # sqrt(dk) = 1
    rows.append(A[-1])
    outputs.append((A @ V)[-1])

torch.testing.assert_close(torch.stack(rows), torch.tensor(
    [[.25, .25, .50], [.40, .40, .20]], dtype=dtype))
torch.testing.assert_close(torch.stack(outputs),
                           torch.full((2, 2), .5, dtype=dtype))
delta = rows[1] - rows[0]
torch.testing.assert_close(delta @ V, torch.zeros(2, dtype=dtype),
                           atol=1e-15, rtol=0)
print("different weights:", torch.stack(rows).tolist())
print("same output:", torch.stack(outputs).tolist())
```

The output alone cannot distinguish these weights. For fixed inputs and
parameters the head remains deterministic. This example does not establish
that every head admits arbitrary replacements of its attention matrix.
Brunner et al. analyse the corresponding null-space question in
[On Identifiability in Transformers, §3.1–3.2](https://arxiv.org/html/1908.04211v3#S3).
An attention map records mixing coefficients; interpreting it as causal
token importance requires additional evidence.

<!-- /attention-identifiability -->

<details>
<summary>Реализация головы в лекции Карпати</summary>

<a id="лаборатория-по-карпати"></a>

В [коде лекции, класс Head](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L61-L90)
видны три `Linear`, $QK^\top$, деление на корень размерности ключа,
причинная маска и умножение весов на значения.
В отличие от кода этой главы там есть dropout после softmax:
сумма конкретной строки после dropout уже не обязана быть единицей.

Практический маршрут лекции:

1. [bigram baseline](https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py);
2. равномерное усреднение прошлого и нижнетреугольная матрица;
3. обучаемые Q/K и изменение весов по содержимому;
4. значения V и выход одной головы;
5. маски, несколько голов и выходная проекция;
6. [полный минимальный GPT](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py).

В [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
можно проследить те же стадии внутри GPT-2. Эта визуализация уже
содержит операции полного блока.

</details>

## Источники

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  §3.2.1–3.2.3: scaled dot-product, размерности,
  разные источники Q/K/V и причинное ограничение.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
  Self-Attention in Detail и Matrix Calculation of Self-Attention.
  Четыре рисунка — CC BY-NC-SA 4.0.
- Zhang, Lipton, Li, Smola, [D2L §11.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html):
  self-attention, сравнение с RNN/CNN и необходимость позиционной информации.
  Рисунок — CC BY-SA 4.0.
- [Karpathy — Let’s build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  и [код лекции](https://github.com/karpathy/ng-video-lecture).
- [PyTorch — scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).
- [3Blue1Brown — Attention in transformers](https://www.3blue1brown.com/lessons/attention).
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/12_attention|TinyTorch 12 — Attention]]:
  scaled dot-product, multi-head attention, causal masking и проверка стоимости.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
  и [BertViz](https://github.com/jessevig/bertviz).

**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров]]
