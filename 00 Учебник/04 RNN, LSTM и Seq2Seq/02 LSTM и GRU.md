---
title: LSTM и GRU
type: textbook-chapter
status: canonical
last_updated: 2026-10-21
previous: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT]]"
next: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]"
primary_sources:
  - https://www.bioinf.jku.at/publications/older/2604.pdf
  - https://doi.org/10.1162/089976600300015015
  - https://aclanthology.org/D14-1179/
  - https://d2l.ai/chapter_recurrent-modern/lstm.html
  - https://d2l.ai/chapter_recurrent-modern/gru.html
---

# LSTM и GRU

В предложении «Самолёты, которые компания купила, были новыми» форма «были»
согласуется с далёким подлежащим «самолёты», а не с более близким словом
«компания». Языковой модели нужно сохранить сведения о подлежащем,
одновременно обрабатывая слова внутри придаточного предложения.

[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|Простая RNN]]
использует одно скрытое состояние и для текущего предсказания, и для передачи
информации следующим шагам. Каждый шаг заново преобразует всё состояние
через матрицу и нелинейность. При обучении производная проходит через цепочку
этих преобразований и может затухать.

LSTM, long short-term memory, вводит два вектора: состояние ячейки $c_t$
и скрытое состояние $h_t$. Первое передаёт память между шагами, второе
поступает в выходной слой и в расчёт следующего шага. Вентили управляют
сохранением, новой записью и чтением памяти. Это непрерывные обучаемые
операции; отдельные координаты не обязаны иметь понятный человеку смысл.

## LSTM: память и три решения

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-chain.png]]

*Три последовательных шага LSTM у Chris Olah. Верхняя горизонтальная
ветвь передаёт состояние ячейки; нижняя передаёт скрытое состояние.
В середине раскрыты вычисления одного шага.
Источник: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

На схеме каждая линия переносит вектор. Жёлтый прямоугольник обозначает
обучаемое аффинное преобразование с указанной активацией; кружки обозначают
поэлементные операции. Разветвление копирует значение, слияние входных
линий перед слоем означает конкатенацию. Один блок повторяется с теми же
параметрами на всех шагах.

Пусть $x_t\in\mathbb R^d$, а $h_{t-1},c_{t-1}\in\mathbb R^m$.
Состояния в формулах считаются столбцами. Обозначим объединённый вход
$u_t=[h_{t-1};x_t]\in\mathbb R^{m+d}$. Описываем обычную LSTM
без проекций и без прямых связей от $c$ к вентилям.
У каждой из четырёх обучаемых ветвей матрица имеет форму
$m\times(m+d)$, смещение имеет длину $m$.
Все три вентиля и кандидат вычисляются из одного $u_t$;
для их расчёта не нужно ждать результатов друг друга.

### Сколько прежней памяти сохранить

Вентиль забывания, forget gate, вычисляет

$$
a_t^f=W_fu_t+b_f,\qquad f_t=\sigma(a_t^f).
$$

Затем прежнее состояние умножается на $f_t$ покоординатно.
Например, коэффициент 0.9 превращает старую координату 2 в 1.8,
а коэффициент 0.1 превращает её в 0.2.
В разных координатах коэффициенты могут быть разными.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-f.png]]

*Выделены расчёт $f_t$ из $[h_{t-1},x_t]$ и умножение прежнего $C_{t-1}$
на этот вектор. Заглавное $C$ на рисунке соответствует нашему $c$.
Источник: Chris Olah,
[Understanding LSTM Networks, Step-by-Step LSTM Walk Through](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

После смены подлежащего прежнее число и род могут перестать быть
полезными для дальнейшего согласования. Вентиль позволяет уменьшить
сохраняемую часть состояния в зависимости от текущего входа и прежнего
$h$. Но ему не сообщают отдельной меткой «сейчас сменилось подлежащее»:
его веса обучаются через ошибку той задачи, для которой используется сеть.

Сигмоида вычисляется отдельно для каждой координаты. Её выходы не являются
распределением вероятностей и не должны суммироваться в единицу.
При конечных вещественных аргументах $\sigma$ принимает значения строго
между 0 и 1. Значения 0 и 1 ниже используются как предельные режимы;
в вычислениях с конечной точностью насыщение также может дать эти значения.

### Что добавить к памяти

Кандидат $g_t$ содержит возможную новую запись, а входной вентиль
$i_t$ определяет её масштаб:

$$
a_t^i=W_iu_t+b_i,\qquad i_t=\sigma(a_t^i),
$$

$$
a_t^g=W_gu_t+b_g,\qquad g_t=\tanh(a_t^g).
$$

На рисунках Олаха кандидат обозначен $\tilde C_t$.
Его координаты находятся между $-1$ и 1 и могут быть отрицательными.
Вентиль $i_t$ не меняет знак кандидата, а уменьшает или почти сохраняет
его величину. При $i_t=0.2$ кандидат $-0.5$ даёт запись $-0.1$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-i.png]]

*Две разные обучаемые ветви: сигмоида вычисляет масштаб записи $i_t$,
а tanh вычисляет сам кандидат $\tilde C_t$. Их произведение попадёт
в состояние ячейки.
Источник: Chris Olah,
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

Новое состояние получается сложением двух вкладов:

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t.
$$

Символ $\odot$ обозначает поэлементное умножение.
В одной координате старая память может почти сохраниться, а новая запись
почти отсутствовать. В другой координате прежняя память может уменьшиться,
а кандидат заменить её. Входной вентиль и вентиль забывания независимы:
условия $i_t+f_t=1$ в стандартной LSTM нет. Поэтому сеть может
и сохранять старое значение, и добавлять к нему новое.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-C.png]]

*Обновление $C_t$: слева умножение старого состояния на $f_t$, затем
сложение с $i_t\odot\tilde C_t$. Знаки «×» и «+» обозначают
поэлементные операции над векторами.
Источник: Chris Olah,
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-o.png]]

*Память $C_t$ проходит через tanh и умножается на $o_t$.
Верхняя ветвь $C_t$ продолжается независимо от полученного $h_t$.
Источник: Chris Olah,
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

Например, для одного $c_t=1.7$ выходные коэффициенты 0.01 и 0.7
дадут $h_t\approx0.009354$ и $h_t\approx0.654786$ соответственно.
Память при этом остаётся равной 1.7. В обычной LSTM вентили следующего
шага видят именно $h_t$, поэтому закрытие выходного вентиля также
уменьшает доступ следующего расчёта вентилей к этой памяти.

Для языкового моделирования поверх $h_t$ ставится тот же классификатор
слов, что и в предыдущей главе: логиты $Ah_t+b$, затем softmax.
Вектор $h_t$ сам по себе не является распределением следующего токена.

<details>
<summary>Те же операции в обозначениях D2L</summary>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-gates.svg]]

*Только расчёт трёх вентилей $F_t,I_t,O_t$ из текущего входа $X_t$
и прежнего скрытого состояния $H_{t-1}$. Кандидат и обновление памяти
здесь ещё не показаны. Источник: Zhang, Lipton, Li, Smola,
[D2L, §10.1.1.2](https://d2l.ai/chapter_recurrent-modern/lstm.html),
CC BY-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-hidden-state.svg]]

*Полная ячейка D2L: вверху обновление $C$, справа чтение через tanh
и выходной вентиль. Кандидат обозначен $\tilde C_t$.
Заглавные буквы в D2L соответствуют пакету объектов, а не одному столбцу.
Источник: те же авторы,
[D2L, §10.1.1.5](https://d2l.ai/chapter_recurrent-modern/lstm.html),
CC BY-SA 4.0.*

</details>

## Вычислимый пример одной координаты

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

### Как обучаются вентили

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

При обучении последовательности у $h_t$ и $c_t$ есть входящие производные
от последующих шагов. Обозначим через $\bar h_t$ уже накопленную
производную потери по $h_t$, а через $q_t$ производную, пришедшую
в $c_t$ по ветви следующего состояния ячейки. Тогда

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

### Как далеко проходит производная

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-C-line.png]]

*Выделена прямая ветвь $C_{t-1}\to C_t$, включающая умножение
и сложение. Нелинейные ветви расчёта вентилей затенены, но остаются
частью полной сети. Источник: Chris Olah,
[Understanding LSTM Networks, The Core Idea Behind LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

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
При постоянном коэффициенте $f$ чувствительность равна $f^{T-k}$.
Здесь нет обязательного умножения на одну и ту же плотную рекуррентную
матрицу на каждом шаге, как было в простой RNN.

| Постоянный $f$ | Коэффициент через 100 шагов | Шагов до уменьшения вдвое |
|---:|---:|---:|
| 0.9 | 0.00002656 | 6.58 |
| 0.99 | 0.366032 | 68.97 |
| 0.999 | 0.904792 | 692.80 |

Таким образом, 0.9 недостаточно для почти неизменного переноса
на сотню шагов. Даже 0.99 оставляет около 37% первоначального сигнала.
Разные координаты и разные входы могут создавать разные интервалы
сохранения; сеть не обязана использовать одно $f$ для всего текста.

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

### Полная ячейка и порядок весов

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

Историческая LSTM 1997 года и показанная современная ячейка различаются.
В исходной конструкции использовался постоянный прямой перенос памяти;
вентиль забывания добавили Gers, Schmidhuber и Cummins в
[Learning to Forget](https://doi.org/10.1162/089976600300015015), 2000.

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

## GRU: объединить память и выход

GRU, gated recurrent unit, хранит один вектор $h_t$ и использует
два вентиля. Reset gate $r_t$ управляет участием прежнего состояния
в кандидате, update gate $z_t$ — смешиванием прежнего состояния
с кандидатом. Отдельного $c_t$ и отдельного выходного вентиля нет.

Сначала запишем вариант [Cho et al., §2.3](https://aclanthology.org/D14-1179/)
и D2L. Для столбцов $x_t\in\mathbb R^d$, $h_{t-1}\in\mathbb R^m$

$$
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r),
$$

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/gru.svg]]

*Расчёт только reset gate $R_t$ и update gate $Z_t$ из $X_t,H_{t-1}$.
Кандидат и смешивание состояний на этой схеме не показаны.
Источник: Zhang, Lipton, Li, Smola,
[D2L, §10.2.1](https://d2l.ai/chapter_recurrent-modern/gru.html),
CC BY-SA 4.0.*

### Reset gate и кандидат

Прежнее состояние сначала умножается на $r_t$, затем преобразуется
рекуррентной матрицей:

$$
\tilde h_t=\tanh\!\left(
W_hx_t+U_h(r_t\odot h_{t-1})+b_h
\right).
$$

Если $r_t$ близок к нулю во всех координатах, кандидат почти не использует
прежнее состояние. Но это ещё не означает, что новое $h_t$ станет нулём:
кандидат зависит от текущего входа, а окончательный результат зависит
также от update gate.

### Update gate и новое состояние

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm-gru-2026-10-21/d2l-gru-3.svg]]

*Полный граф GRU. Reset gate действует перед вычислением кандидата.
Верхняя ветвь умножает прежнее состояние на $Z_t$, другая умножает
кандидат на $1-Z_t$, после чего результаты складываются.
Источник: Zhang, Lipton, Li, Smola,
[D2L, §10.2.3](https://d2l.ai/chapter_recurrent-modern/gru.html),
CC BY-SA 4.0.*

В прямой ветви производная по прежнему состоянию содержит
$\operatorname{diag}(z_t)$. При фиксированных вентиле и кандидате
в нашем скалярном примере она равна 0.75.
В полной сети и $z_t$, и кандидат также зависят от $h_{t-1}$,
поэтому полный якобиан включает их производные.
Как и у LSTM, короткая прямая ветвь помогает переносу, но не делает
весь обратный проход тождественным копированием.

### Почему схемы могут выглядеть противоречиво

У Олаха в формуле смешивания $z_t$ обозначает долю нового кандидата:

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
$$

Чтобы получить наш результат 0.4, в этом обозначении нужно взять
$z_t=0.25$, а не 0.75. Если вместе с переименованием заменить
аргумент сигмоиды на противоположный, равенство
$\sigma(-a)=1-\sigma(a)$ даст тот же переход.
Это изменение соглашения, а не новая архитектура.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-var-GRU.png]]

*На этой схеме коэффициент $z_t$ относится к новой записи, а $1-z_t$
к старому состоянию. По сравнению с D2L смысл символа $z$ обратный.
Reset gate по-прежнему действует на прежнее состояние до линейного
преобразования кандидата.
Источник: Chris Olah,
[Understanding LSTM Networks, Variants on Long Short Term Memory](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*

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

## Что выбирать

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

Двунаправленная сеть обрабатывает текст и слева направо, и справа налево.
Её выход имеет $2m$ признаков на позицию и использует будущие токены.
Такой слой пригоден для кодирования полностью доступного исходного текста,
но не для причинного предсказания следующего токена с теми же входами.
Конечное обратное состояние нельзя получить, взяв просто последний
элемент двунаправленного `output`: обратный проход заканчивается
у начала исходной последовательности.

У LSTM с проекцией сначала вычисляется $m$-мерный выход ячейки,
затем он умножается на матрицу $P\in\mathbb R^{p\times m}$.
В API PyTorch ненулевая размерность проекции должна быть меньше $m$.
Передаваемое $h$ и выход имеют размерность $p$, а память $c$
сохраняет размерность $m$. Поэтому прежние формулы счётчика
и утверждение «$h$ и $c$ одной формы» к этому варианту не относятся.
Например, при $d=2,m=3,p=2$ один слой PyTorch имеет
$4m(d+p+2)+pm=78$ параметров.

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

Между независимыми текстами начальное состояние задаётся заново.
При продолжении одного потока передаётся конечное состояние:
у LSTM нужно перенести оба вектора $(h,c)$.
При усечённом BPTT отсоединяются оба вектора, а не только $h$.
Если отсоединить лишь $h$, путь через $c$ продолжит связывать
фрагменты вычислительного графа. У GRU отсоединяется единственное $h$.
Сброс в ноль, перенос численного состояния и отсоединение графа
решают разные задачи.

Обе ячейки сохраняют последовательную зависимость между шагами.
Чтобы кодировать исходное предложение для перевода, можно взять
конечные состояния LSTM или GRU и передать их декодеру.
В [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора|следующей главе]]
будет разобрано, как из этого получается encoder–decoder и почему
фиксированный вектор может терять сведения об исходной последовательности.

## Источники и дальнейшее чтение

- Zhang, Lipton, Li, Smola, [D2L, §10.1: Long Short-Term Memory](https://d2l.ai/chapter_recurrent-modern/lstm.html) — состояние ячейки, три вентиля, кандидат и реализация с нуля. Два исходных SVG сохранены с CC BY-SA 4.0.
- Те же авторы, [D2L, §10.2: Gated Recurrent Units](https://d2l.ai/chapter_recurrent-modern/gru.html) — reset, кандидат и update; $z$ означает сохранённое прошлое. Два исходных SVG в главе имеют CC BY-SA 4.0.
- Chris Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), 2015 — последовательное визуальное объяснение LSTM и варианты. Сохранены семь оригинальных иллюстраций; в схеме GRU у автора $z$ означает новую запись.
- Jurafsky, Martin, [SLP3, §14.5: The LSTM](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — разделение переноса контекста и текущего выхода; проверена редакция 19 августа 2026 года.
- [Stanford CS224N, Notes 5, §3–4](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf) — формулы и схема GRU/LSTM. В GRU этих заметок reset стоит после рекуррентной матрицы, а не как в D2L.
- Hochreiter, Schmidhuber, [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf), 1997 — исходная архитектура.
- Gers, Schmidhuber, Cummins, [Learning to Forget: Continual Prediction with LSTM](https://doi.org/10.1162/089976600300015015), 2000 — добавление вентиля забывания.
- Cho et al., [Learning Phrase Representations using RNN Encoder–Decoder](https://aclanthology.org/D14-1179/), 2014, §2.3 — исходный вариант GRU и формулы (5)–(8).
- [PyTorch 2.14: LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html), [GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) — формы, порядок весов, biases, проекции и отличие кандидата GRU.
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-recurrent-neural-networks.pdf|Stanford CS230 — Recurrent Neural Networks Cheatsheet]] — компактные схемы и уравнения после полного разбора.

**Назад:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]
