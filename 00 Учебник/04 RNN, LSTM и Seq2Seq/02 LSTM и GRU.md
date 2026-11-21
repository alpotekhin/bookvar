---
title: LSTM и GRU
type: textbook-chapter
status: canonical
last_updated: 2026-11-21
source_language: mixed
reference_terms:
  - label: "Память c и скрытое состояние h"
    label_en: "Cell memory c and hidden state h"
    anchor: lstm-two-states
  - label: "Вентиль забывания"
    label_en: "Forget gate"
    anchor: lstm-forget
  - label: "Входной вентиль и кандидат"
    label_en: "Input gate and candidate"
    anchor: lstm-write
  - label: "Выходной вентиль и чтение памяти"
    label_en: "Output gate and memory readout"
    anchor: lstm-output
  - label: "Численный проход LSTM"
    label_en: "Scalar LSTM forward pass"
    anchor: lstm-scalar
  - label: "Производные вентилей и полный BPTT"
    label_en: "Gate derivatives and full BPTT"
    anchor: lstm-gate-gradients
  - label: "Прямой путь, забывание и полный якобиан"
    label_en: "Direct path, retention and full Jacobian"
    anchor: lstm-gradient-path
  - label: "Совмещённая матрица и порядок i/f/g/o"
    label_en: "Fused weights and i/f/g/o ordering"
    anchor: lstm-cell-weights
  - label: "Исходная LSTM и добавление forget gate"
    label_en: "Original LSTM and the added forget gate"
    anchor: lstm-history
  - label: "Связанные вентили и peephole"
    label_en: "Coupled gates and peepholes"
    anchor: lstm-variants
  - label: "Единое состояние GRU"
    label_en: "Single GRU state"
    anchor: gru-state
  - label: "Reset gate и кандидат GRU"
    label_en: "GRU reset gate and candidate"
    anchor: gru-reset
  - label: "Update gate и выпуклая смесь"
    label_en: "Update gate and convex mixing"
    anchor: gru-update
  - label: "Соглашения z и порядок reset"
    label_en: "z conventions and reset placement"
    anchor: gru-conventions
  - label: "Почему смещения кандидата GRU не складываются"
    label_en: "Why GRU candidate biases do not merge"
    anchor: gru-candidate-biases
  - label: "Состояния и число параметров"
    label_en: "Recurrent states and parameter counts"
    anchor: recurrent-parameter-counts
  - label: "Слои, направления и формы тензоров"
    label_en: "Layers, directions and tensor shapes"
    anchor: recurrent-sequence-shapes
  - label: "Два направления и конечные состояния"
    label_en: "Two directions and final states"
    anchor: recurrent-bidirectional
  - label: "Проекция внутри рекуррентного состояния"
    label_en: "Projection inside recurrent state"
    anchor: lstm-projection
  - label: "Реальные длины LSTM/GRU и padding"
    label_en: "True LSTM/GRU lengths and padding"
    anchor: gated-packed-sequences
  - label: "Перенос и отсоединение обоих состояний"
    label_en: "Carrying and detaching both states"
    anchor: gated-state-carry
previous: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT]]"
next: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]"
primary_sources:
  - https://d2l.ai/chapter_recurrent-modern/bi-rnn.html
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html
  - https://www.bioinf.jku.at/publications/older/2604.pdf
  - https://doi.org/10.1162/089976600300015015
  - https://aclanthology.org/D14-1179/
  - https://d2l.ai/chapter_recurrent-modern/lstm.html
  - https://d2l.ai/chapter_recurrent-modern/gru.html
---

# LSTM и GRU

В предложении «Самолёты, которые компания купила, были новыми» форма «были»
согласуется с «самолёты», хотя «компания» стоит ближе. Пока модель
обрабатывает придаточное, ей нужно сохранить число подлежащего
главной части. Такой пример с согласованием разбирают
[Jurafsky и Martin, §14.5](https://web.stanford.edu/~jurafsky/slp3/14.pdf).

[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|Простая RNN]]
использует одно скрытое состояние и для текущего предсказания, и для передачи
информации следующим шагам. Каждый шаг заново преобразует всё состояние
через матрицу и нелинейность. При обучении производная проходит через цепочку
этих преобразований и может затухать.

LSTM, long short-term memory, переносит между шагами два вектора.
В состоянии ячейки $c_t$ можно сохранить часть прошлого значения
и прибавить к ней новую запись. Скрытое состояние $h_t$ вычисляют
из $c_t$ отдельно: его получают выходной слой и вентили следующего
шага. У координат памяти нет заранее назначенного смысла вроде
«число подлежащего».

<a id="lstm-two-states"></a>

<a id="lstm-память-и-три-решения"></a>

## Состояние ячейки LSTM

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-hidden-state.svg|Один шаг LSTM: обновление памяти C и чтение в скрытое состояние H]]

*Zhang, Lipton, Li, Smola, [D2L, Hidden State, рис. 10.1.4](https://en.d2l.ai/chapter_recurrent-modern/lstm.html#hidden-state).
Прежняя память $C_{t-1}$ входит сверху слева. В верхней ветви она
обновляется до $C_t$; справа из неё получают $H_t$ через tanh и выходной
вентиль $O_t$. Заглавные буквы у D2L обозначают пакеты векторов.
Полный исходный SVG; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Одна и та же ячейка с теми же весами работает на каждом шаге.
Пара $(h_t,c_t)$ становится входным состоянием для шага $t+1$,
который также получает новый $x_{t+1}$. На
[развёрнутой схеме Chris Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
это две отдельные линии между соседними ячейками.

Пусть $x_t\in\mathbb R^d$, а $h_{t-1},c_{t-1}\in\mathbb R^m$.
Состояния в формулах считаются столбцами. Обозначим объединённый вход
$u_t=[h_{t-1};x_t]\in\mathbb R^{m+d}$. Описываем обычную LSTM
без проекций и без прямых связей от $c$ к вентилям.
У каждой из четырёх обучаемых ветвей матрица имеет форму
$m\times(m+d)$, смещение имеет длину $m$.
Все три вентиля и кандидат вычисляются из одного $u_t$;
для их расчёта не нужно ждать результатов друг друга.

<a id="lstm-forget"></a>

### Сколько прежней памяти сохранить

Вентиль забывания, forget gate, вычисляет

$$
a_t^f=W_fu_t+b_f,\qquad f_t=\sigma(a_t^f).
$$

Затем прежнее состояние умножается на $f_t$ покоординатно.
Например, коэффициент 0.9 превращает старую координату 2 в 1.8,
а коэффициент 0.1 превращает её в 0.2.
В разных координатах коэффициенты могут быть разными.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-gates.svg|Три вентиля LSTM получают прежнее скрытое состояние и текущий вход]]

*Zhang et al., [D2L, Input Gate, Forget Gate, and Output Gate, рис. 10.1.1](https://en.d2l.ai/chapter_recurrent-modern/lstm.html#input-gate-forget-gate-and-output-gate).
Из одних и тех же $X_t,H_{t-1}$ три слоя с разными весами вычисляют
$I_t,F_t,O_t$. Само умножение $F_t\odot C_{t-1}$ происходит позже,
на верхней ветви полной схемы. Исходный SVG;
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Если после «самолёты» вводится новое подлежащее, старое число может
перестать помогать в предсказании. Вес сохранения зависит от текущего
входа и прежнего $h$. Размечать смену подлежащего отдельно не нужно:
вентиль получает градиент от ошибки предсказания.

Сигмоида вычисляется отдельно для каждой координаты. Её выходы не являются
распределением вероятностей и не должны суммироваться в единицу.
При конечных вещественных аргументах $\sigma$ принимает значения строго
между 0 и 1. Значения 0 и 1 ниже используются как предельные режимы;
в вычислениях с конечной точностью насыщение также может дать эти значения.

<a id="lstm-write"></a>

### Что добавить к памяти

Кандидат $g_t$ содержит возможную новую запись, а входной вентиль
$i_t$ определяет её масштаб:

$$
a_t^i=W_iu_t+b_i,\qquad i_t=\sigma(a_t^i),
$$

$$
a_t^g=W_gu_t+b_g,\qquad g_t=\tanh(a_t^g).
$$

На рисунках D2L кандидат обозначен $\tilde C_t$.
Его координаты находятся между $-1$ и 1 и могут быть отрицательными.
Вентиль $i_t$ не меняет знак кандидата, а уменьшает или почти сохраняет
его величину. При $i_t=0.2$ кандидат $-0.5$ даёт запись $-0.1$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/d2l-lstm-1.svg|Кандидат LSTM вычисляется отдельным слоем с tanh]]

*Zhang et al., [D2L, Input Node, рис. 10.1.2](https://en.d2l.ai/chapter_recurrent-modern/lstm.html#input-node).
Четыре слоя получают $X_t,H_{t-1}$. Три используют сигмоиду,
а ветвь с tanh даёт кандидат $\tilde C_t$, то есть наш $g_t$.
На этом рисунке вентиль ещё не умножил кандидат. Исходный SVG;
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Новое состояние получается сложением двух вкладов:

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t.
$$

Символ $\odot$ обозначает поэлементное умножение.
В одной координате старая память может почти сохраниться, а новая запись
почти отсутствовать. В другой координате прежняя память может уменьшиться,
а кандидат заменить её. Входной вентиль и вентиль забывания вычисляют
с разными весами; равенство $i_t+f_t=1$ не требуется. Ячейка может
сохранить старое значение и прибавить к нему новое.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/d2l-lstm-2.svg|Память LSTM складывается из двух поэлементных произведений]]

*Zhang et al., [D2L, Memory Cell Internal State, рис. 10.1.3](https://en.d2l.ai/chapter_recurrent-modern/lstm.html#memory-cell-internal-state).
К верхнему плюсу сходятся $F_t\odot C_{t-1}$ и $I_t\odot\tilde C_t$.
Выходной вентиль $O_t$ не входит в это сложение. Исходный SVG;
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

<a id="lstm-output"></a>

### Что передать в скрытое состояние

Выходной вентиль, output gate, управляет чтением:

$$
a_t^o=W_ou_t+b_o,\qquad o_t=\sigma(a_t^o),
$$

$$
h_t=o_t\odot\tanh(c_t).
$$

Состояние $c_t$ переносится на следующий шаг независимо от того,
насколько открыт $o_t$. Если выходной вентиль близок к нулю,
текущий $h_t$ почти не показывает содержимое памяти, но сама память
не обнуляется. Позже вентиль может открыться.

В обычной LSTM вентили следующего шага получают $h_t$. Если $o_t$
скрыл координату памяти от текущего выхода, она будет слабо влиять
и на расчёт следующих вентилей через $h_t$. При этом прежнее $c_t$
по-прежнему поступает в следующее обновление состояния ячейки.

Для языкового моделирования поверх $h_t$ ставится тот же классификатор
слов, что и в предыдущей главе: логиты $Ah_t+b$, затем softmax.
Вектор $h_t$ сам по себе не является распределением следующего токена.

<a id="lstm-scalar"></a>

<a id="вычислимый-пример-одной-координаты"></a>

## Два шага LSTM на одной координате

Пусть в одной координате $c_{t-1}=2$, $f_t=0.9$, $i_t=0.2$,
$g_t=-0.5$, $o_t=0.7$. Последовательность вычислений такова:

| Операция | Вычисление | Результат |
|---|---|---:|
| Сохранить часть старого | $0.9\cdot2$ | 1.8 |
| Подготовить запись | $0.2\cdot(-0.5)$ | −0.1 |
| Обновить память | $1.8-0.1$ | 1.7 |
| Преобразовать для чтения | $\tanh(1.7)$ | 0.935409 |
| Получить скрытое состояние | $0.7\cdot0.935409$ | 0.654786 |

Все коэффициенты вентилей здесь заранее выбраны. В сети они получились бы
из аффинных преобразований $u_t$. Чтобы получить ровно эти значения
в исполняемом примере, можно задать аргументы сигмоид
$\operatorname{logit}(p)=\ln(p/(1-p))$, а для кандидата задать
$\operatorname{atanh}(-0.5)$.

Теперь изменим только $o_t$ с $0.7$ на $0.01$. Память $c_t=1.7$
останется прежней, хотя $h_t$ уменьшится до $0.009354$. На следующем
шаге возьмём $f_{t+1}=0.99$, $g_{t+1}=0$, $o_{t+1}=0.7$.
Кандидат равен нулю, поэтому новой записи нет:

$$
\begin{aligned}
c_{t+1}&=0.99\cdot1.7=1.683,\\
h_{t+1}&=0.7\tanh(1.683)\approx0.653275.
\end{aligned}
$$

Второй выход снова заметно отличается от нуля. Он получился из
сохранённого $c_t$, а не из почти закрытого $h_t$. Эти значения
вентилей заданы для примера. В сети они могут различаться между
шагами при тех же весах: меняются $x_t$ и предыдущее $h$.

<a id="lstm-gradient-path"></a>

<a id="как-далеко-проходит-производная"></a>

### Градиент вдоль состояния ячейки

На [схеме прямой ветви у Олаха](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#the-core-idea-behind-lstms)
выделен путь от $c_{t-1}$ через умножение и сложение к $c_t$.
Он не проходит через tanh, который нужен для чтения $h_t$.

Если $h_{t-1}$ считать независимым входом ячейки и зафиксировать его,
в обычной LSTM вентили не зависят от $c_{t-1}$. Частная производная
по прямой ветви равна

$$
\left.\frac{\partial c_t}{\partial c_{t-1}}
\right|_{h_{t-1},x_t}
=\operatorname{diag}(f_t).
$$

В произведении вдоль этой ветви между шагами $k$ и $T$ остаются
поэлементные множители $f_{k+1},\ldots,f_T$.
В примере из двух шагов при фиксированных вентилях изменение
$c_{t-1}$ на малую величину меняет $c_t$ с коэффициентом $0.9$,
а $c_{t+1}$ с коэффициентом $0.9\cdot0.99=0.891$. Этот путь проходит
мимо $h_t$, поэтому закрытый на первом шаге $o_t=0.01$ не даёт
дополнительного множителя $0.01$. Если потеря зависит от $h_{t+1}$,
вклад этого пути ещё проходит через чтение памяти на втором шаге:
$o_{t+1}(1-\tanh^2(c_{t+1}))$.

При постоянном коэффициенте $f$ чувствительность равна $f^{T-k}$.
Здесь нет обязательного умножения на одну и ту же плотную рекуррентную
матрицу на каждом шаге, как было в простой RNN.

| Постоянный $f$ | Коэффициент через 100 шагов | Шагов до уменьшения вдвое |
|---:|---:|---:|
| 0.9 | 0.00002656 | 6.58 |
| 0.99 | 0.366032 | 68.97 |
| 0.999 | 0.904792 | 692.80 |

Через сто шагов при $f=0.9$ от исходного изменения остаётся меньше
трёх стотысячных, при $f=0.99$ — около 37%. В обученной ячейке
у разных координат могут быть разные $f$, а новый вход меняет их
на каждом шаге.

Чтобы при нулевом вкладе входа получить $f=0.99$, нужен аргумент
сигмоиды $\ln(99)\approx4.595$. При аргументе 1 получается
$f\approx0.731$, а через сто шагов $f^{100}\approx2.48\cdot10^{-14}$.
Смещение вентиля может изменить начальную склонность к сохранению,
но конкретное число не задаёт универсальную длину памяти.
Кроме того, у насыщенной сигмоиды мала производная $f(1-f)$,
поэтому обучение самого вентиля тоже может замедлиться.

Полная производная по более раннему состоянию включает и другие пути.
Например, если $h_{t-1}=0.7\tanh(c_{t-1})$, а аргумент $f_t$
содержит $0.4h_{t-1}$, изменение прежнего $c$ изменит ещё и вентиль.
При $c_{t-1}=2$ и $f_t=0.9$ суммарная производная нового $c_t$
в таком скалярном примере равна примерно 0.903561, а не ровно 0.9.
Поэтому произведение вентилей описывает выделенную прямую ветвь,
а не весь якобиан LSTM.

```python
import math
import torch

dtype = torch.float64
for f in [.9, .99, .999]:
    c0 = torch.tensor(1., dtype=dtype, requires_grad=True)
    c = c0
    for _ in range(100):
        c = f*c  # фиксированный вентиль, без новой записи
    derivative, = torch.autograd.grad(c, c0)
    assert abs(derivative.item()-f**100) < 1e-12
    print(f, "after 100:", derivative.item(),
          "half-life:", math.log(.5)/math.log(f))

c_prev = torch.tensor(2., dtype=dtype, requires_grad=True)
h_prev = .7*c_prev.tanh()
bias = math.log(.9/.1)-.4*(.7*math.tanh(2.))
f = torch.sigmoid(.4*h_prev+bias)
c = f*c_prev-.1
full_derivative, = torch.autograd.grad(c, c_prev)
assert abs(f.item()-.9) < 1e-12
assert abs(full_derivative.item()-.9035608015725995) < 1e-12
print("direct branch:", f.item(), "all paths:", full_derivative.item())
```

Выходной слой тоже может уменьшить сигнал: при большом $|c_t|$
производная $\tanh(c_t)$ мала. Само $c_t$ не обязано лежать в $[-1,1]$,
хотя кандидат и обычное $h_t$ ограничены активациями.
Например, в предельном режиме $f=i=1$ и при $g=0.5$
память прибавляла бы по 0.5 за шаг. Её чтение через tanh
при достаточно большом значении почти насытилось бы.
LSTM облегчает перенос по одной ветви, но не устраняет все
причины малых или больших градиентов.

<details>
<summary>Производные вентилей и полный обратный проход</summary>

<a id="lstm-gate-gradients"></a>

<h3 id="как-обучаются-вентили">Производные по аргументам вентилей</h3>

Сначала рассмотрим производные скрытого выхода $h_t$ в этом примере.
Производная по новому состоянию ячейки равна

$$
\lambda=\frac{\partial h_t}{\partial c_t}
=o_t(1-\tanh^2(c_t))
\approx0.087507.
$$

Умножение $f_tc_{t-1}$ даёт производную по $f_t$, равную $c_{t-1}=2$.
Сигмоида добавляет множитель $f_t(1-f_t)=0.09$. Поэтому

$$
\frac{\partial h_t}{\partial a_t^f}
=\lambda c_{t-1}f_t(1-f_t)
\approx0.015751.
$$

Аналогично для входного вентиля производная умножения равна кандидату
$-0.5$. Увеличение аргумента входной сигмоиды увеличит отрицательную запись,
поэтому производная выхода по $a_t^i$ будет отрицательной.

| Аргумент обучаемой ветви | Производная $h_t$ по этому аргументу |
|---|---:|
| $a_t^i$ | −0.007001 |
| $a_t^f$ | 0.015751 |
| $a_t^g$ | 0.013126 |
| $a_t^o$ | 0.196436 |

У аргумента выходной сигмоиды производная содержит $\tanh(c_t)o_t(1-o_t)$.
У аргумента кандидата она содержит $\lambda i_t(1-g_t^2)$.

```python
import math
import torch

dtype = torch.float64
a = torch.tensor([
    math.log(.2/.8), math.log(.9/.1),
    math.atanh(-.5), math.log(.7/.3),
], dtype=dtype, requires_grad=True)
c_prev = torch.tensor(2., dtype=dtype, requires_grad=True)
i, f, g, o = a[0].sigmoid(), a[1].sigmoid(), a[2].tanh(), a[3].sigmoid()
c = f*c_prev + i*g
h = o*c.tanh()
h.backward()  # производная самого скалярного выхода
lam = o.detach()*(1-c.detach().tanh().square())
expected = torch.stack([
    lam*g.detach()*i.detach()*(1-i.detach()),
    lam*c_prev.detach()*f.detach()*(1-f.detach()),
    lam*i.detach()*(1-g.detach().square()),
    c.detach().tanh()*o.detach()*(1-o.detach()),
])
torch.testing.assert_close(a.grad, expected)
torch.testing.assert_close(c_prev.grad, lam*f.detach())
print("c, h:", c.item(), h.item())
print("dh/da [i,f,g,o]:", a.grad.tolist())
print("dh/dc_prev:", c_prev.grad.item())
```

В развёрнутой сети $h_t$ влияет на текущий выход и на вентили следующего
шага. Обозначим суммарную производную потери по этим двум ветвям через
$\bar h_t$. Пусть $q_t$ — градиент, пришедший в $c_t$ из $c_{t+1}$
по прямой ветви. Тогда к $q_t$ нужно прибавить вклад чтения памяти через
$h_t$:

$$
\lambda_t=q_t+
\bar h_t\odot o_t\odot(1-\tanh^2(c_t)).
$$

Отсюда получаются производные по аргументам четырёх ветвей:

$$
\delta_t^f=\lambda_t\odot c_{t-1}\odot f_t\odot(1-f_t),
$$

$$
\delta_t^i=\lambda_t\odot g_t\odot i_t\odot(1-i_t),
$$

$$
\delta_t^g=\lambda_t\odot i_t\odot(1-g_t^2),
$$

$$
\delta_t^o=
\bar h_t\odot\tanh(c_t)\odot o_t\odot(1-o_t).
$$

По прямой ветви к предыдущей памяти передаётся
$\lambda_t\odot f_t$. К предыдущему $h_{t-1}$ производная приходит
через четыре аффинные ветви. Если собрать их аргументы в
$a_t=Wu_t+b$ и соответствующие $\delta$ в один столбец, то
производная по $u_t$ равна $W^\top\delta_t$.
Первые $m$ координат добавляются к производной по $h_{t-1}$,
последние $d$ относятся к текущему $x_t$.

Как и в обычной RNN, общие веса обучаются по сумме вкладов всех шагов:
$dL/dW=\sum_t\delta_tu_t^\top$ и $dL/db=\sum_t\delta_t$.
LSTM по-прежнему обучается BPTT; в развёрнутом графе стало больше
ветвей, но правило цепочки не изменилось.

</details>

<details>
<summary>Исходная LSTM, связанные вентили и peephole</summary>

<a id="lstm-history"></a>

Историческая LSTM 1997 года и показанная современная ячейка различаются.
В исходной конструкции использовался постоянный прямой перенос памяти;
вентиль забывания добавили Gers, Schmidhuber и Cummins в
[Learning to Forget](https://doi.org/10.1162/089976600300015015), 2000.

<a id="lstm-variants"></a>

Есть и варианты современной ячейки. В coupled input–forget gate
полагают $i_t=1-f_t$. Тогда при $c_{t-1}=2$, $f_t=0.9$, $g_t=-0.5$
получается $c_t=1.75$, а не 1.7 из нашего примера с независимым $i_t=0.2$.
Сохранять старое значение и одновременно полностью открывать новую запись
в этом варианте уже нельзя.

Peephole connections добавляют в расчёт вентилей прямые связи
от состояния ячейки. Тогда вентиль может различать две разные
памяти даже при одинаковом закрытом $h_{t-1}$.
Например, добавка $c_{t-1}$ в аргумент сигмоиды для состояний 2 и −2
при остальных нулевых слагаемых даёт примерно 0.881 и 0.119.
У обычной ячейки с одинаковыми $h_{t-1}$ и $x_t$ эти значения
вентиля совпали бы. У вариантов с такими связями производная
по $c_{t-1}$ уже не ограничивается одной прямой ветвью.
Иллюстрации вариантов приведены в
[разделе Olah «Variants on Long Short Term Memory»](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

</details>

<a id="gru-state"></a>

<a id="gru-объединить-память-и-выход"></a>

## Скрытое состояние GRU

GRU, gated recurrent unit, хранит один вектор $h_t$ и использует
два вентиля. Reset gate $r_t$ управляет участием прежнего состояния
в кандидате, update gate $z_t$ — смешиванием прежнего состояния
с кандидатом. Отдельного $c_t$ и отдельного выходного вентиля нет.

Запишем уравнения из [D2L, §10.2](https://d2l.ai/chapter_recurrent-modern/gru.html).
Порядок вычислений кандидата соответствует исходной работе
[Cho et al., §2.3](https://aclanthology.org/D14-1179/).
Пусть $x_t\in\mathbb R^d$ и $h_{t-1}\in\mathbb R^m$ — столбцы. Тогда

$$
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r),
$$

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/gru.svg|Расчёт reset и update gate в GRU]]

*Расчёт только reset gate $R_t$ и update gate $Z_t$ из $X_t,H_{t-1}$.
Кандидат и смешивание состояний на этой схеме не показаны.
Источник: Zhang, Lipton, Li, Smola,
[D2L, §10.2.1](https://d2l.ai/chapter_recurrent-modern/gru.html),
CC BY-SA 4.0.*

<a id="gru-reset"></a>

<a id="reset-gate-и-кандидат"></a>

### Прежнее состояние в расчёте кандидата

Прежнее состояние сначала умножается на $r_t$, затем преобразуется
рекуррентной матрицей:

$$
\tilde h_t=\tanh\!\left(
W_hx_t+U_h(r_t\odot h_{t-1})+b_h
\right).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/d2l-gru-2.svg|Reset gate масштабирует старое состояние до рекуррентной матрицы кандидата GRU]]

*Zhang et al., [D2L, Candidate Hidden State, рис. 10.2.2](https://en.d2l.ai/chapter_recurrent-modern/gru.html#candidate-hidden-state).
Следите за ветвью $H_{t-1}$: сначала поэлементное умножение на $R_t$,
затем линейное преобразование в слое кандидата. $Z_t$ уже вычислен,
но применяется только при смешивании, на следующем рисунке.
Исходный SVG; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Если $r_t$ близок к нулю во всех координатах, кандидат почти не использует
прежнее состояние. Но это ещё не означает, что новое $h_t$ станет нулём:
кандидат зависит от текущего входа, а окончательный результат зависит
также от update gate.

<a id="gru-update"></a>

<a id="update-gate-и-новое-состояние"></a>

### Смешивание кандидата с прежним состоянием

Используем соглашение, в котором $z_t$ — доля сохранённого прошлого:

$$
h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t.
$$

При $z_t\approx1$ состояние почти копируется. При $z_t\approx0$
оно почти заменяется кандидатом. Каждая координата имеет свой
коэффициент. Это поэлементная выпуклая смесь: в отличие от LSTM
здесь коэффициенты старого состояния и кандидата суммируются в единицу.

Например, пусть $h_{t-1}=0.6$, $z_t=0.75$, а кандидат равен $-0.2$.
Тогда

$$
h_t=0.75\cdot0.6+0.25\cdot(-0.2)=0.4.
$$

Этот результат возможен и при $r_t=0$, если текущий вход и смещение
дали кандидата $-0.2$. Reset gate исключил прежнее состояние
из расчёта кандидата, но update gate сохранил его вклад 0.45
в итоговый результат. Поэтому reset gate нельзя отождествлять
со сбросом всего $h$ в ноль.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm-gru-2026-10-21/d2l-gru-3.svg|Кандидат GRU и смешивание старого и нового состояний]]

*Полный граф GRU. Reset gate действует перед вычислением кандидата.
Верхняя ветвь умножает прежнее состояние на $Z_t$, другая умножает
кандидат на $1-Z_t$, после чего результаты складываются.
Источник: Zhang, Lipton, Li, Smola,
[D2L, §10.2.3](https://d2l.ai/chapter_recurrent-modern/gru.html),
CC BY-SA 4.0.*

Если зафиксировать $z_t$ и кандидат, производная по прежнему состоянию
в нашем скалярном примере равна $0.75$. Это вклад прямой ветви
$z_t h_{t-1}$. Когда $z_t$ и кандидат сами зависят от $h_{t-1}$,
по правилу произведения получаем

$$
\begin{aligned}
\frac{d h_t}{d h_{t-1}}
&=z_t+(h_{t-1}-\tilde h_t)\frac{d z_t}{d h_{t-1}}\,\\
&\quad+(1-z_t)\frac{d \tilde h_t}{d h_{t-1}}.
\end{aligned}
$$

Например, оставим кандидат постоянным, $\tilde h_t=-0.2$, и возьмём
$z_t=\sigma(\ln 3+h_{t-1}-0.6)$. При $h_{t-1}=0.6$ всё ещё
получаются $z_t=0.75$ и $h_t=0.4$. Но теперь изменение старого
состояния меняет и коэффициент смеси. Производная всего перехода
равна $0.75+0.8\cdot0.75\cdot0.25=0.9$. Как и в LSTM, прямой
путь объясняет лишь одну часть градиента.

<a id="recurrent-parameter-counts"></a>

<a id="что-выбирать"></a>

## Состояния и число параметров

Сравним обновления на тех же числах. Возьмём старую координату
$0.6$, кандидата $-0.2$ и коэффициент сохранения $0.75$. У GRU
коэффициент кандидата уже задан: $1-0.75=0.25$. У LSTM входной
вентиль может принимать другое значение при том же вентиле
забывания. Во всех трёх случаях от старого состояния остаётся
$0.75\cdot0.6=0.45$. Коэффициент $0.25$ даёт запись $-0.05$,
а коэффициент $0.5$ даёт $-0.1$.

| Состояние | Вклад кандидата | Результат |
|---|---|---:|
| $h_t$ в GRU | $-0.05$ | $0.4$ |
| $c_t$ в LSTM при $i_t=0.25$ | $-0.05$ | $0.4$ |
| $c_t$ в LSTM при $i_t=0.5$ | $-0.1$ | $0.35$ |

При одинаковой сумме $0.4$ GRU передаёт её дальше как $h_t$.
В LSTM это значение $c_t$. При $o_t=0.7$ ячейка выдаёт
$h_t=0.7\tanh(0.4)\approx0.266$ и передаёт на следующий шаг
оба числа: $c_t=0.4$ и $h_t\approx0.266$.

У стандартных однослойных ячеек без проекций различаются число состояний
и число обучаемых аффинных преобразований:

| Ячейка | Передаваемое состояние | Аффинных ветвей | Параметров с одним bias на ветвь |
|---|---|---:|---|
| Простая RNN | $h$ | 1 | $m(d+m+1)$ |
| GRU | $h$ | 3 | $3m(d+m+1)$ |
| LSTM | $(h,c)$ | 4 | $4m(d+m+1)$ |

Счётчик относится к самой ячейке. Embeddings и выходной классификатор
в него не включены. При $d=2,m=3$ получаются 18, 54 и 72 параметра.
PyTorch хранит два bias-вектора на каждую ветвь, поэтому его
однослойные RNN, GRU и LSTM имеют соответственно 21, 63 и 84 параметра
при `bias=True`. При `bias=False` оба bias-вектора отсутствуют.

У GRU меньше весов при одинаковой размерности состояния, но из этого
не следует превосходство в качестве или фиксированное ускорение.
Для сравнения нужно указать размерности, число слоёв, бюджет параметров,
длину последовательностей и режим исполнения. LSTM может хранить
информацию в $c$, слабо показывая её через $h$; GRU переносит один
вектор и использует его и для выхода, и для следующего обновления.

## Реализация в PyTorch

<details>
<summary>LSTM: объединение матриц и порядок вентилей</summary>

<a id="lstm-cell-weights"></a>

<h3 id="полная-ячейка-и-порядок-весов">Полная ячейка и порядок весов</h3>

Четыре аффинных преобразования удобно объединить в одно:

$$
\begin{aligned}
a_t&=W[h_{t-1};x_t]+b,\\
W&\in\mathbb R^{4m\times(m+d)}.
\end{aligned}
$$

В следующем коде блоки идут в порядке $i,f,g,o$, как в PyTorch.
Затем для $i,f,o$ применяется сигмоида, для $g$ — tanh.
В пакете объекты хранятся строками, поэтому используется умножение
на $W^\top$.

<details>
<summary>Ячейка с нуля, сверка с nn.LSTMCell и шаг обучения</summary>

```python
import torch
from torch import nn

torch.manual_seed(17)
dtype = torch.float64
B, T, d, m = 2, 3, 2, 3

class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(.1*torch.randn(4*m, m+d, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(4*m, dtype=dtype))

    def forward(self, x, state):
        h, c = state
        a = torch.cat([h, x], -1) @ self.W.T + self.b
        ai, af, ag, ao = a.chunk(4, -1)
        i, f, g, o = ai.sigmoid(), af.sigmoid(), ag.tanh(), ao.sigmoid()
        c = f*c + i*g
        h = o*c.tanh()
        return h, c

cell = Cell()
reference = nn.LSTMCell(d, m, dtype=dtype)
with torch.no_grad():
    reference.weight_hh.copy_(cell.W[:, :m])
    reference.weight_ih.copy_(cell.W[:, m:])
    reference.bias_ih.copy_(cell.b)
    reference.bias_hh.zero_()

X = torch.randn(B, T, d, dtype=dtype)
zero = torch.zeros(B, m, dtype=dtype)
h, c = zero.clone(), zero.clone()
rh, rc = zero.clone(), zero.clone()
for t in range(T):
    h, c = cell(X[:, t], (h, c))
    rh, rc = reference(X[:, t], (rh, rc))
    torch.testing.assert_close(h, rh)
    torch.testing.assert_close(c, rc)

def objective():
    h, c = zero.clone(), zero.clone()
    for t in range(T):
        h, c = cell(X[:, t], (h, c))
    return .5*(h.square().mean()+c.square().mean())

optimizer = torch.optim.SGD(cell.parameters(), lr=.1)
before = objective().item()
optimizer.zero_grad(set_to_none=True)
loss = objective()
loss.backward()
nn.utils.clip_grad_norm_(cell.parameters(), 1., error_if_nonfinite=True)
optimizer.step()
after = objective().item()
assert after < before
print("parameters:", sum(p.numel() for p in cell.parameters()))
print("loss:", before, "->", after)
```

</details>

В ячейке 72 параметра. У `nn.LSTMCell(2,3)` их 84:
библиотека хранит отдельные смещения для входа и рекуррентного состояния.
В нашем варианте их сумма заменена одним вектором $b$.
Совпадение прямых расчётов проверяется до шага оптимизатора.
После независимого обновления двух bias-векторов совпадение
с обновлением одного bias-вектора не следует автоматически.

</details>

<details>
<summary>GRU: обозначение z, положение reset и смещения кандидата</summary>

<a id="gru-conventions"></a>

<h3 id="почему-схемы-могут-выглядеть-противоречиво">Обозначения z и положение reset</h3>

В [формуле (7) Cho et al.](https://aclanthology.org/D14-1179.pdf#page=3)
$z_t$ умножается на прежнее состояние.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/cho-gru-figure2.png|Ранняя схема GRU: reset r на ветви к кандидату и update z на ветви к новому состоянию]]

*Cho et al., [Learning Phrase Representations, §2.3, рис. 2](https://aclanthology.org/D14-1179.pdf#page=3).
Внизу $r$ стоит на ветви от $h$ к кандидату $\tilde h$; вход $x$
поступает в кандидат отдельно. Вверху $z$ смешивает прежнее
состояние с кандидатом. Переключатели на рисунке условны: по формулам
(5)–(8) это непрерывные коэффициенты, причём $z$ взвешивает прошлое.
Оригинал извлечён из PDF; [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).*

В [схеме GRU у Олаха](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#variants-on-long-short-term-memory)
той же буквой обозначена доля нового кандидата:

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
$$

Чтобы получить наш результат 0.4, в этом обозначении нужно взять
$z_t=0.25$, а не 0.75. Если заменить аргумент сигмоиды на противоположный,
$\sigma(-a)=1-\sigma(a)$ даст тот же переход. Здесь изменилось
обозначение коэффициента; следующий пример меняет сами вычисления.

Есть и содержательное различие реализаций.
В `torch.nn.GRU` reset gate применяется после рекуррентного
матричного преобразования:

$$
\begin{aligned}
v_t&=U_{hn}h_{t-1}+b_{hn},\\
n_t&=\tanh(W_{in}x_t+b_{in}+r_t\odot v_t).
\end{aligned}
$$

В D2L было $U_h(r_t\odot h_{t-1})$, а здесь
$r_t\odot(U_{hn}h_{t-1}+b_{hn})$.
В общем случае поэлементное умножение и умножение на матрицу
нельзя менять местами.

Возьмём $U=\left[\begin{smallmatrix}0.1&0.2\\0.3&0.4\end{smallmatrix}\right]$,
$h=[1,2]^\top$, $r=[0.2,0.8]^\top$, без остальных слагаемых.
Вариант до матрицы даст $U(r\odot h)=[0.34,0.7]^\top$,
вариант после матрицы даст $r\odot(Uh)=[0.1,0.88]^\top$.
После tanh получатся соответственно примерно
$[0.327477,0.604368]^\top$ и $[0.099668,0.706419]^\top$.

| Источник | Какой коэффициент сохраняет старое $h$ | Где стоит reset |
|---|---|---|
| Cho et al., D2L | $z_t$ | До рекуррентной матрицы |
| Olah | $1-z_t$ | До рекуррентной матрицы |
| PyTorch 2.14 | $z_t$ | После рекуррентной матрицы, вместе с recurrent bias |
| CS224N Notes 5, 2019 | $z_t$ | В показанной формуле после матрицы; biases опущены |

Отличие PyTorch от исходного порядка reset описано в
[официальной документации GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).
Число матриц может совпадать с D2L, но перенос одинаковых численных
весов не гарантирует одинаковый результат.

<details>
<summary>Два порядка reset на числах и ручной forward nn.GRUCell</summary>

```python
import torch
from torch import nn
import torch.nn.functional as F

dtype = torch.float64
U = torch.tensor([[.1, .2], [.3, .4]], dtype=dtype)
h = torch.tensor([1., 2.], dtype=dtype)
r = torch.tensor([.2, .8], dtype=dtype)
before_matrix = U @ (r*h)
after_matrix = r * (U@h)
torch.testing.assert_close(before_matrix, torch.tensor([.34, .7], dtype=dtype))
torch.testing.assert_close(after_matrix, torch.tensor([.1, .88], dtype=dtype))
assert not torch.allclose(before_matrix.tanh(), after_matrix.tanh())
print("reset before:", before_matrix.tanh().tolist())
print("reset after:", after_matrix.tanh().tolist())

torch.manual_seed(9)
B, d, m = 2, 2, 3
cell = nn.GRUCell(d, m, dtype=dtype)
x = torch.randn(B, d, dtype=dtype)
h_prev = torch.randn(B, m, dtype=dtype)
ir, iz, inn = F.linear(x, cell.weight_ih, cell.bias_ih).chunk(3, -1)
hr, hz, hn = F.linear(h_prev, cell.weight_hh, cell.bias_hh).chunk(3, -1)
r, z = (ir+hr).sigmoid(), (iz+hz).sigmoid()
n = (inn+r*hn).tanh()
manual = (1-z)*n + z*h_prev
torch.testing.assert_close(manual, cell(x, h_prev))

# Иной порядок reset при тех же численных весах.
candidate_before = (
    inn + F.linear(r*h_prev, cell.weight_hh[2*m:], cell.bias_hh[2*m:])
).tanh()
other = (1-z)*candidate_before + z*h_prev
assert not torch.allclose(manual, other)
print("manual equals nn.GRUCell; max variant difference:",
      (manual-other).abs().max().item())
```

</details>

<a id="gru-candidate-biases"></a>

<h4 id="why-the-candidate-biases-cannot-be-merged">Why the candidate biases cannot be merged</h4>

In the PyTorch candidate, the input bias is outside the reset product
and the recurrent bias is inside it. Their contribution before tanh is
$b_{in}+r_t\odot b_{hn}$. In the RNN and LSTM branches above, both biases
are added with coefficient one and the forward pass depends on their
sum. The GRU candidate has a different dependence on the two biases.

Take one coordinate with $b_{in}=0.1$ and $b_{hn}=0.8$. At reset values
0.25 and 0.75, their contributions are 0.3 and 0.7. Replacing the pair
with $b_{in}=0.9$, $b_{hn}=0$ preserves the sum of the parameters, but
gives 0.9 in both cases. A single constant bias cannot reproduce both
values. The reset and update branches do allow the two biases to be
summed; this obstruction concerns the candidate branch.

The example isolates those offsets. Only the input weight of the reset
gate is nonzero. It turns the two supplied log-odds into reset values
0.25 and 0.75; the update gate is 0.5 and the previous state is zero.

```python
from copy import deepcopy
import torch

dtype = torch.float64
biased_gru = torch.nn.GRUCell(1, 1, dtype=dtype)
with torch.no_grad():
    for parameter in biased_gru.parameters():
        parameter.zero_()
    biased_gru.weight_ih[0, 0] = 1.  # PyTorch gate order: r, z, n
    biased_gru.bias_ih[2] = .1
    biased_gru.bias_hh[2] = .8

reset_values = torch.tensor([[.25], [.75]], dtype=dtype)
inputs = torch.logit(reset_values)
initial = torch.zeros_like(inputs)
separate_output = biased_gru(inputs, initial)
expected_output = .5 * (.1 + .8 * reset_values).tanh()
torch.testing.assert_close(separate_output, expected_output)

merged_gru = deepcopy(biased_gru)
with torch.no_grad():
    merged_gru.bias_ih[2] = .9
    merged_gru.bias_hh[2] = 0.
merged_output = merged_gru(inputs, initial)
torch.testing.assert_close(merged_output, .5 * torch.full_like(inputs, .9).tanh())
assert not torch.allclose(separate_output, merged_output)
print("separate candidate biases:", separate_output[:, 0].tolist())
print("same bias sum, different output:", merged_output[:, 0].tolist())
```

This follows from the candidate equation in
[PyTorch 2.14, GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).
It also limits weight conversion between implementations: the reset
placement and recurrent candidate bias must both be matched.

</details>

<a id="recurrent-sequence-shapes"></a>

### Последовательности, слои и формы тензоров

Одна ячейка обрабатывает один шаг. `nn.LSTM` и `nn.GRU`
разворачивают ячейку по всей последовательности.
Для `batch_first=True` вход имеет форму `[B,T,d]`,
а выход одного однонаправленного слоя — `[B,T,m]`.
У LSTM возвращаются последовательность скрытых состояний
и пара конечных состояний `(h_n,c_n)`; у GRU возвращается
последовательность и одно `h_n`. Конечное $c$ не заменяет конечное $h$.

При $L$ слоях и $D$ направлениях форма конечных состояний без
проекций равна `[L*D,B,m]` даже при `batch_first=True`.
Верхний слой получает на каждом шаге скрытый выход нижнего.
Значение `dropout` в этих API относится к выходам между рекуррентными
слоями, кроме последнего. Оно не означает автоматического dropout
на всей временной ветви $c$; при одном слое межслойного перехода нет.

<details>
<summary>Два направления и конечные состояния</summary>

<a id="recurrent-bidirectional"></a>

<h3 id="two-directions-and-two-sequence-endpoints">Two directions and two sequence endpoints</h3>

Двунаправленная сеть обрабатывает текст и слева направо, и справа налево.
Её выход имеет $2m$ признаков на позицию и использует будущие токены.
Такой слой пригоден для кодирования полностью доступного исходного текста,
но не для причинного предсказания следующего токена с теми же входами.
Конечное обратное состояние нельзя получить, взяв просто последний
элемент двунаправленного `output`: обратный проход заканчивается
у начала исходной последовательности.


<!-- source-excerpt: d2l-bidirectional-order -->
> For the first RNN layer,
> the first input is $\mathbf{x}_1$
> and the last input is $\mathbf{x}_T$,
> but for the second RNN layer,
> the first input is $\mathbf{x}_T$
> and the last input is $\mathbf{x}_1$.
> To produce the output of this bidirectional RNN layer,
> we simply concatenate together the corresponding outputs
> of the two underlying unidirectional RNN layers.
<!-- /source-excerpt -->

*Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola,
[D2L, Bidirectional Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html),
original English excerpt, CC BY-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm-gru-contracts-2026-11-03/d2l-birnn.svg|Два независимых направления RNN и объединение состояний одной позиции]]

*Original D2L bidirectional RNN, Fig. 10.4.1. The two hidden-state rows
have opposite temporal arrows and separate parameters. Both states at
position $t$ feed that position’s output. The diagram shows generic
recurrent cells; using LSTM changes each cell’s internal operations and
adds its carried $c$, not the direction of these arrows.
Source: Zhang et al.,
[D2L](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html#fig-birnn),
CC BY-SA 4.0. Complete pinned SVG, without cropping or redrawing.*

For inputs $x_0,x_1,x_2$, the forward recurrence visits positions
0, 1, 2. The reverse recurrence visits 2, 1, 0. The library aligns both
sets of outputs to the original position axis:

| Original position | Forward state has read | Reverse state has read |
|---:|---|---|
| 0 | $x_0$ | $x_2,x_1,x_0$ |
| 1 | $x_0,x_1$ | $x_2,x_1$ |
| 2 | $x_0,x_1,x_2$ | $x_2$ |

Thus `output[:, -1]` combines a complete forward pass with the
first computed reverse state. The complete reverse pass appears at
`output[:, 0, m:]`. For one unprojected layer, `h_n[0]` and
`h_n[1]` are the final forward and reverse states. With $L$ layers,
`h_n.reshape(L, 2, B, m)[-1]` selects the top layer’s pair.

The following program copies each direction into a separate one-step
LSTM cell and visits the positions explicitly. This verifies every
aligned output and both final memory states, not just the tensor shapes.

```python
import torch
from torch import nn

torch.manual_seed(47)
dtype = torch.float64
B, T, d, m = 1, 3, 1, 2
sequence = torch.tensor([[[1.], [2.], [3.]]], dtype=dtype)
bidirectional_lstm = nn.LSTM(d, m, bidirectional=True,
                             batch_first=True, dtype=dtype)
combined, (final_h, final_c) = bidirectional_lstm(sequence)

def trace_direction(reverse):
    cell = nn.LSTMCell(d, m, dtype=dtype)
    suffix = "_reverse" if reverse else ""
    with torch.no_grad():
        for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
            getattr(cell, name).copy_(getattr(bidirectional_lstm, name + "_l0" + suffix))
    h = torch.zeros(B, m, dtype=dtype)
    c = torch.zeros_like(h)
    aligned = [None] * T
    positions = range(T-1, -1, -1) if reverse else range(T)
    for position in positions:
        h, c = cell(sequence[:, position], (h, c))
        aligned[position] = h
    return torch.stack(aligned, dim=1), h, c

forward, forward_h, forward_c = trace_direction(False)
reverse, reverse_h, reverse_c = trace_direction(True)
torch.testing.assert_close(combined, torch.cat([forward, reverse], dim=-1))
torch.testing.assert_close(final_h, torch.stack([forward_h, reverse_h]))
torch.testing.assert_close(final_c, torch.stack([forward_c, reverse_c]))
torch.testing.assert_close(combined[:, -1, :m], final_h[0])
torch.testing.assert_close(combined[:, 0, m:], final_h[1])
assert not torch.allclose(combined[:, -1, m:], final_h[1])
```

The reverse output at position 0 depends on all three inputs, so it
cannot be used as a causal state after reading only $x_0$. When
sequences have different lengths, the reversal must stop at each
sequence’s actual boundary. Running backward over right-padding first
would allow PAD transitions to affect the valid reverse states.
Packing supplies those boundaries; a zero loss on PAD does not.
The endpoint contract is also documented in
[PyTorch 2.14, LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html).

</details>

<details>
<summary>Проекция скрытого состояния LSTM</summary>

<a id="lstm-projection"></a>

<h3 id="projection-inside-the-recurrent-state">Projection inside the recurrent state</h3>

У LSTM с проекцией сначала вычисляется $m$-мерный выход ячейки,
затем он умножается на матрицу $P\in\mathbb R^{p\times m}$.
В API PyTorch ненулевая размерность проекции должна быть меньше $m$.
Передаваемое $h$ и выход имеют размерность $p$, а память $c$
сохраняет размерность $m$. Поэтому прежние формулы счётчика
и утверждение «$h$ и $c$ одной формы» к этому варианту не относятся.
Например, при $d=2,m=3,p=2$ один слой PyTorch имеет
$4m(d+p+2)+pm=78$ параметров.


Write the unprojected readout as $\widehat h_t=o_t\odot\tanh(c_t)$.
The projected state is $h_t=P\widehat h_t$. On the next step, the
four gates receive this $p$-dimensional $h_t$, so the recurrent matrix
has $4m$ rows and $p$ columns. The projection participates in the
recurrence; applying a smaller classifier only after an ordinary LSTM
would leave a different recurrent state.

For the $d=2,m=3,p=2$ example, the parameter count separates into

| Parameter group | Shape | Count |
|---|---|---:|
| Input weights | `[12,2]` | 24 |
| Recurrent weights | `[12,2]` | 24 |
| Two bias vectors | two `[12]` | 24 |
| Projection | `[2,3]` | 6 |

The sum is 78. In the corresponding unprojected library LSTM it is 84.
For general $d,m,p$, subtracting the unprojected one-layer count from
the projected count gives $m(5p-4m)$. Therefore $p<m$ alone does not
guarantee fewer parameters: $m=6,p=5,d=2$ gives 246 rather than 240.
This comparison concerns parameter counts, not execution time.

The full forward calculation below uses nonzero initial $h$ and $c$.
It checks the projection at every step and compares gradients with the
library implementation.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(53)
dtype = torch.float64
B, T, d, m, p = 2, 3, 2, 3, 2
projected_lstm = torch.nn.LSTM(d, m, proj_size=p, batch_first=True, dtype=dtype)
features = torch.randn(B, T, d, dtype=dtype, requires_grad=True)
h0 = torch.randn(1, B, p, dtype=dtype, requires_grad=True)
c0 = torch.randn(1, B, m, dtype=dtype, requires_grad=True)
library_output, (library_h, library_c) = projected_lstm(features, (h0, c0))

h, c = h0[0], c0[0]
history = []
for time in range(T):
    gates = (F.linear(features[:, time], projected_lstm.weight_ih_l0,
                      projected_lstm.bias_ih_l0)
             + F.linear(h, projected_lstm.weight_hh_l0, projected_lstm.bias_hh_l0))
    ai, af, ag, ao = gates.chunk(4, dim=-1)
    c = af.sigmoid() * c + ai.sigmoid() * ag.tanh()
    unprojected_h = ao.sigmoid() * c.tanh()
    h = F.linear(unprojected_h, projected_lstm.weight_hr_l0)
    history.append(h)
manual_output = torch.stack(history, dim=1)
torch.testing.assert_close(manual_output, library_output)
torch.testing.assert_close(h[None], library_h)
torch.testing.assert_close(c[None], library_c)
assert library_output.shape == (B, T, p)
assert library_h.shape == (1, B, p) and library_c.shape == (1, B, m)
assert sum(parameter.numel() for parameter in projected_lstm.parameters()) == 78

leaves = (features, h0, c0, *projected_lstm.parameters())
manual_loss = manual_output.square().sum() + .3 * c.square().sum()
library_loss = library_output.square().sum() + .3 * library_c.square().sum()
manual_gradients = torch.autograd.grad(manual_loss, leaves)
library_gradients = torch.autograd.grad(library_loss, leaves)
for manual, reference in zip(manual_gradients, library_gradients):
    torch.testing.assert_close(manual, reference)
```

The $c$-state still has three coordinates throughout this example.
A projection neither truncates the time graph nor resets the memory.
Its matrix receives gradients through both the outputs and later gate
calculations. Parameter names, shapes and the projection definition are
specified in
[PyTorch 2.14, LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html).

</details>

<a id="gated-packed-sequences"></a>

### Разная длина текстов и перенос состояния

Padding не должен становиться продолжением настоящего текста.
Как было показано для простой RNN, нулевой вход тоже меняет состояние:
у LSTM при отсутствии новой записи прежнее $c$ всё ещё умножается на $f$.
Нулевая потеря на PAD не останавливает этот прямой расчёт.

В PyTorch можно передать фактические длины в packed sequence.
Тогда конечные состояния соответствуют последним настоящим позициям,
а не концу дополненного пакета. В следующем примере второй объект
имеет длину 1 при общей длине пакета 3. Его два дополнительных
нулевых входа исключаются из рекуррентного расчёта.

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(23)
dtype = torch.float64
X = torch.randn(2, 3, 2, dtype=dtype)
X[1, 1:] = 0.
lengths = torch.tensor([3, 1])

for factory in [nn.LSTM, nn.GRU]:
    layer = factory(2, 3, batch_first=True, dtype=dtype)
    packed = pack_padded_sequence(
        X, lengths, batch_first=True, enforce_sorted=False
    )
    packed_output, state = layer(packed)
    output, restored_lengths = pad_packed_sequence(
        packed_output, batch_first=True, total_length=3
    )
    h = state[0] if isinstance(state, tuple) else state
    assert output.shape == (2, 3, 3)
    torch.testing.assert_close(restored_lengths, lengths)
    for row, length in enumerate(lengths.tolist()):
        single_output, single_state = layer(X[row:row+1, :length])
        single_h = single_state[0] if isinstance(single_state, tuple) else single_state
        torch.testing.assert_close(h[:, row:row+1], single_h)
        torch.testing.assert_close(output[row:row+1, :length], single_output)
        if isinstance(state, tuple):
            torch.testing.assert_close(state[1][:, row:row+1], single_state[1])
    _, plain_state = layer(X)
    plain_h = plain_state[0] if isinstance(plain_state, tuple) else plain_state
    assert not torch.allclose(h[:, 1], plain_h[:, 1])
    print(factory.__name__, "packed h:", tuple(h.shape),
          "padding difference:", (h[:, 1]-plain_h[:, 1]).abs().max().item())
```

<a id="gated-state-carry"></a>

Между независимыми текстами начальное состояние задаётся заново.
При продолжении одного потока передаётся конечное состояние:
у LSTM нужно перенести оба вектора $(h,c)$.
При усечённом BPTT отсоединяются оба вектора, а не только $h$.
Если отсоединить лишь $h$, путь через $c$ продолжит связывать
фрагменты вычислительного графа. У GRU отсоединяется единственное $h$.
Сброс в ноль, перенос численного состояния и отсоединение графа
решают разные задачи.

Чтобы перевести предложение, можно прочитать его одной рекуррентной
сетью и передать её конечное состояние другой сети, которая будет
писать перевод. В главе
[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора|Seq2Seq]]
разберём, что получает декодер и почему одного состояния фиксированного
размера может не хватить для длинной фразы.

## Источники и дальнейшее чтение

- Zhang, Lipton, Li, Smola, [D2L, §10.4: Bidirectional Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html) — original English explanation and complete source SVG, CC BY-SA 4.0. The endpoint trace here uses LSTM cells and checks both h and c.
- Zhang, Lipton, Li, Smola, [D2L, §10.1: Long Short-Term Memory](https://d2l.ai/chapter_recurrent-modern/lstm.html) — состояние ячейки, три вентиля, кандидат и реализация с нуля. Четыре исходных SVG, CC BY-SA 4.0.
- Те же авторы, [D2L, §10.2: Gated Recurrent Units](https://d2l.ai/chapter_recurrent-modern/gru.html) — reset, кандидат и update; $z$ означает сохранённое прошлое. Три исходных SVG, CC BY-SA 4.0.
- Chris Olah, [Understanding LSTM Networks, Step-by-Step Walk Through](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#step-by-step-lstm-walk-through), 2015 — четыре крупные схемы, от вентиля забывания до чтения памяти.
- Jurafsky, Martin, [SLP3, §14.5: The LSTM](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — разделение переноса контекста и текущего выхода; проверена редакция 19 августа 2026 года.
- [Stanford CS224N, Notes 5, §3–4](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf) — формулы и схема GRU/LSTM. В GRU этих заметок reset стоит после рекуррентной матрицы, а не как в D2L.
- Hochreiter, Schmidhuber, [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf), 1997 — исходная архитектура.
- Gers, Schmidhuber, Cummins, [Learning to Forget: Continual Prediction with LSTM](https://doi.org/10.1162/089976600300015015), 2000 — добавление вентиля забывания.
- Cho et al., [Learning Phrase Representations using RNN Encoder–Decoder](https://aclanthology.org/D14-1179/), 2014, §2.3 — исходный вариант GRU и формулы (5)–(8).
- [PyTorch 2.14: LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html), [GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) — формы, порядок весов, biases, проекции и отличие кандидата GRU.
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-recurrent-neural-networks.pdf|Stanford CS230 — Recurrent Neural Networks Cheatsheet]] — компактные схемы и уравнения после полного разбора.

**Назад:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]
