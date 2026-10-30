---
title: Нейрон и многослойный перцептрон
type: textbook-chapter
status: canonical
last_updated: 2026-10-27
primary_sources:
  - https://d2l.ai/chapter_multilayer-perceptrons/mlp.html
  - https://cs231n.github.io/neural-networks-1/
  - https://github.com/karpathy/micrograd
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html
---

# Нейрон и многослойный перцептрон

У линейного классификатора каждому входному признаку соответствует постоянный
вес. Если два признака обозначают наличие двух сигналов, правило «ровно один
сигнал присутствует» уже нельзя выразить одной линейной границей. Сочетания
$(0,1)$ и $(1,0)$ должны относиться к одному классу, а $(0,0)$ и $(1,1)$ — к
другому. Это задача XOR; ниже разберём, почему линейная модель с ней не
справляется и как её решает небольшой скрытый слой.

Многослойный перцептрон (multilayer perceptron, MLP) сначала вычисляет
промежуточные признаки, затем строит по ним предсказание. Эти признаки тоже
зависят от обучаемых весов. Поэтому при обучении меняется и представление
объекта, и правило, которое использует это представление.

Для чтения нужны [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|матричное умножение]]
и [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация|линейная классификация]].
Производные понадобятся в разделе с одним шагом обучения; подробный обратный
проход разобран в [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|главе о backpropagation]].

## Один нейрон

Пусть объект имеет два признака, а нейрон вычисляет взвешенную сумму

$$
z=2x_1-x_2-0{,}5.
$$

При $x=(1,2)$ получаем $z=2-2-0{,}5=-0{,}5$.
При $x=(2,1)$ получаем $z=4-1-0{,}5=2{,}5$.
Взвешенная сумма с весами $(2,-1)$ растёт при увеличении первого признака и
уменьшается при увеличении второго. Смещение $-0{,}5$ добавляется независимо от входа.

После суммы применяется функция активации. Например, ReLU оставляет
положительное число без изменения, а отрицательное заменяет нулём:

$$
h=\operatorname{ReLU}(z)=\max(0,z).
$$

Два входа выше дают $h=0$ и $h=2{,}5$. Число $z$ называют
предактивацией, а $h$ — активацией нейрона. В общем случае

$$
z=\sum_{i=1}^{d}w_i x_i+b,
\qquad h=\phi(z).
$$

Веса $w_i$ и смещение $b$ обучаются; функция $\phi$ в обычном MLP выбирается
при задании архитектуры. ReLU не содержит собственных обучаемых параметров.

Выход скрытого нейрона не обязан быть вероятностью. ReLU может вернуть любое
неотрицательное число. Если вместо неё использовать сигмоиду
$\sigma(z)=1/(1+e^{-z})$ и интерпретировать результат как вероятность класса 1,
получится уже знакомая логистическая регрессия. При пороге $1/2$ решение
зависит от знака $z$, поэтому граница остаётся прямой $2x_1-x_2-0{,}5=0$.
Нелинейное преобразование вероятности само по себе не делает эту границу
нелинейной. Утверждение относится к такому правилу классификации с монотонной
сигмоидой, а не к произвольной функции $\phi$.

## Слой считает много нейронов одновременно

Один скрытый нейрон вычисляет один промежуточный признак. Чтобы получить
несколько признаков, применяют несколько разных наборов весов к одному входу.
Их удобно собрать в матрицу.

Будем записывать объект строкой $x$ размера $1\times d$, а пакет из $B$
объектов — матрицей $X\in\mathbb R^{B\times d}$. Для слоя из $m$ нейронов

$$
Z=XW_1+b_1,\qquad H=\phi(Z),
$$

где $W_1\in\mathbb R^{d\times m}$, $b_1\in\mathbb R^m$,
$Z,H\in\mathbb R^{B\times m}$. Каждый столбец $W_1$ содержит веса одного
нейрона. Смещение прибавляется к каждой строке; это и есть распространение
вектора по оси пакета, или broadcasting.

Рассмотрим пакет из двух объектов и слой из трёх нейронов:

$$
\begin{aligned}
X&=\begin{bmatrix}1&2\\2&-1\end{bmatrix},\\[4pt]
W_1&=\begin{bmatrix}1&-1&2\\2&1&-1\end{bmatrix},\\[4pt]
b_1&=(0,1,-1).
\end{aligned}
$$

Первая строка входа даёт три предактивации:

$$
\begin{aligned}
z_{11}&=1\cdot1+2\cdot2+0=5,\\
z_{12}&=1\cdot(-1)+2\cdot1+1=2,\\
z_{13}&=1\cdot2+2\cdot(-1)-1=-1.
\end{aligned}
$$

Для второй строки теми же весами получаем $(0,-2,4)$. Следовательно,

$$
\begin{aligned}
Z&=\begin{bmatrix}5&2&-1\\0&-2&4\end{bmatrix},\\[4pt]
H&=\operatorname{ReLU}(Z)\\
 &=\begin{bmatrix}5&2&0\\0&0&4\end{bmatrix}.
\end{aligned}
$$

У первого объекта активны первые два нейрона, у второго — третий. Выбор
активных нейронов зависит от входа, хотя веса у объектов общие. Внутри
полносвязного слоя нет отдельных параметров для каждой строки пакета.

| Величина | Размер в примере | Смысл |
|---|---|---|
| $X$ | $2\times2$ | Два объекта, два входных признака |
| $W_1$ | $2\times3$ | Три набора весов по два числа |
| $b_1$ | $3$ | Одно смещение на нейрон |
| $H$ | $2\times3$ | По три вычисленных признака на объект |

Размер пакета влияет на число вычислений и хранимых активаций, но не на число
параметров. При $B=100$ веса остаются теми же шестью числами, а смещения —
тремя числами. В обычном полносвязном слое результат для одного объекта
не зависит от остальных объектов пакета. Операции вроде BatchNorm могут
добавить такую зависимость, но в этом примере их нет.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-mlp-layered.jpeg]]

*Три входа, два скрытых слоя по четыре нейрона и один выход. Это трёхслойная
сеть: входные координаты не считаются обучаемым слоем. Каждая стрелка —
отдельный вес; смещения на рисунке не показаны. Источник: Stanford CS231n,
[Neural Networks Part 1, Layer-wise organization](https://cs231n.github.io/neural-networks-1/#layers),
[оригинальное изображение](https://cs231n.github.io/assets/nn1/neural_net2.jpeg),
без изменений.*

## Зачем нужна нелинейность

Если после взвешенной суммы сразу поставить ещё одну взвешенную сумму,
то новое семейство нелинейных функций не появится. В одномерном примере

$$
h=2x+1,\qquad o=3h-4=6x-1.
$$

Скрытую переменную $h$ можно исключить. Для матриц выполняется то же самое:

$$
\begin{aligned}
O&=(XW_1+b_1)W_2+b_2\\
 &=X(W_1W_2)+(b_1W_2+b_2)\\
 &=XW'+b'.
\end{aligned}
$$

Композиция аффинных преобразований остаётся аффинным преобразованием.
При узком промежуточном слое ранг $W'$ ещё и ограничен его шириной.
Факторизация может менять оптимизацию, но не превращает итоговую функцию в
нелинейную.

Теперь заменим $h$ на $\max(0,2x+1)$. При $x\leq-1/2$ получаем $o=-4$,
при $x>-1/2$ — $o=6x-1$. Вместо одной прямой появились два линейных участка,
соединённых в точке $x=-1/2$. Одним аффинным преобразованием эту функцию
на всей числовой прямой уже не представить.

В первом скрытом слое каждый ReLU переключается на гиперплоскости
$xw+b=0$. В последующих слоях его вход уже зависит от предыдущих
нелинейностей, поэтому граница переключения во всём пространстве может
состоять из нескольких участков. В области, где набор активных нейронов
всех слоёв фиксирован, сеть остаётся аффинной функцией входа.
При пересечении границы области меняется формула этой функции.
Поэтому ReLU-сеть задаёт кусочно-аффинное отображение.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/micrograd/moon-mlp.png]]

*Пример из Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd):
сеть с двумя скрытыми слоями по 16 ReLU-нейронов разделяет данные
`two moons`. В [исходной тетради](https://github.com/karpathy/micrograd/blob/7bc720e951fe422b8f8814aa5aa1b64121d26b4c/demo.ipynb)
обучается `MLP(2, [16, 16, 1])` с hinge loss, L2-штрафом и SGD.
Это три обучаемых линейных слоя при принятом здесь подсчёте.
Оригинальная картинка скопирована без изменений, MIT License.*

Разные участки границы соответствуют разным наборам активных ReLU.
Видимые ступени и толщина перехода зависят ещё и от отрисовки.
В тетради поле предсказаний вычисляется
на сетке с шагом $0{,}25$; по растру нельзя восстановить точное положение
всех изломов сети.

Сопоставление матричной записи и схемы слоёв дано в
[Dive into Deep Learning, §5.1.1.3 From Linear to Nonlinear](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#from-linear-to-nonlinear).
В примерах этой главы используется такое же соглашение: объекты лежат
в строках, веса умножаются справа.

## XOR как минимальный контрпример линейной модели

Обозначим линейный балл через $s=w_1x_1+w_2x_2+b$ и будем выбирать класс 1
при $s>0$. Для XOR положительны точки $(0,1)$ и $(1,0)$, отрицательны
$(0,0)$ и $(1,1)$. Из первых двух условий следует

$$
\begin{gathered}
w_2+b>0,\qquad w_1+b>0\\
\Longrightarrow\quad w_1+w_2+2b>0.
\end{gathered}
$$

Из отрицательных меток следует $b\leq0$ и $w_1+w_2+b\leq0$.
Сложение этих неравенств даёт $w_1+w_2+2b\leq0$, что противоречит
предыдущему результату. Никакой выбор трёх параметров линейной модели
не удовлетворяет всем четырём меткам.

Построим два новых признака:

$$
\begin{aligned}
h_1&=\max(0,x_1+x_2),\\
h_2&=\max(0,x_1+x_2-1).
\end{aligned}
$$

Первый считает сумму бинарных входов. Второй становится положительным,
только когда оба входа равны единице. Выход $o=h_1-2h_2$ вычитает этот
случай:

| Вход $x$ | $h_1$ | $h_2$ | $o=h_1-2h_2$ | Метка XOR |
|---|---:|---:|---:|---:|
| $(0,0)$ | 0 | 0 | 0 | 0 |
| $(0,1)$ | 1 | 0 | 1 | 1 |
| $(1,0)$ | 1 | 0 | 1 | 1 |
| $(1,1)$ | 2 | 1 | 0 | 0 |

В исходных координатах одной прямой было недостаточно. В координатах
$(h_1,h_2)$ достаточно правила $h_1-2h_2>1/2$. Это пример представления,
в котором линейное разделение стало возможным.

Полный прямой проход можно повторить в NumPy:

```python
import numpy as np

X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
W1_xor = np.array([[1., 1.], [1., 1.]])
b1_xor = np.array([0., -1.])
W2_xor = np.array([[1.], [-2.]])
H_xor = np.maximum(X_xor @ W1_xor + b1_xor, 0.)
o_xor = (H_xor @ W2_xor).ravel()
assert np.array_equal(o_xor, np.array([0., 1., 1., 0.]))
print(o_xor)
```

Веса здесь сконструированы вручную. Проверка доказывает, что сеть может
представить решение на четырёх бинарных входах; она не доказывает, что
градиентный спуск найдёт именно эти веса из любой начальной точки.
Выход $o$ в этом примере тоже не следует автоматически считать вероятностью
на произвольном входе: архитектура не ограничивает его диапазоном $[0,1]$.

У скрытого представления нет единственной обязательной системы координат.
Например, перестановка скрытых нейронов и соответствующая перестановка
весов выходного слоя не меняют предсказание. Поэтому смысл отдельной
координаты нельзя определить только по её номеру.

## Что именно обучается

Вернёмся к пакету $X$ и скрытым активациям $H$ из раздела о слоях.
Добавим выход для двух классов:

$$
\begin{aligned}
W_2&=\begin{bmatrix}
0{,}2&-0{,}2\\
0{,}1&0{,}1\\
-0{,}25&0{,}25
\end{bmatrix},\\
b_2&=(0,0).
\end{aligned}
$$

Выходной слой даёт

$$
O=HW_2+b_2=
\begin{bmatrix}
1{,}2&-0{,}8\\
-1&1
\end{bmatrix}.
$$

Эти числа называют логитами (logits). Они могут быть отрицательными,
не обязаны лежать в $[0,1]$ и не обязаны давать в сумме единицу.
В первой строке больше логит класса 0, во второй — класса 1.

Для вероятностей применяют softmax отдельно к каждой строке. У первого
объекта

$$
\begin{aligned}
p_0&=\frac{e^{1{,}2}}{e^{1{,}2}+e^{-0{,}8}}\\
   &=\frac{1}{1+e^{-2}}\approx0{,}8808,\\
p_1&=1-p_0\approx0{,}1192.
\end{aligned}
$$

У второго объекта вероятности меняются местами. Для меток $y=(0,1)$
средняя перекрёстная энтропия равна

$$
\begin{aligned}
L&=-\frac{\log p_{1,0}+\log p_{2,1}}{2}\\
 &=\log(1+e^{-2})\approx0{,}126928.
\end{aligned}
$$

Сеть уже выбирает правильные классы, но ошибка не нулевая: вероятность
правильной метки меньше единицы. Обучение по cross-entropy продолжает
менять веса и после достижения правильной классификации.

Параметры сети — $W_1,b_1,W_2,b_2$. Обратное распространение вычисляет
производную $L$ по каждому из них, затем оптимизатор делает шаг.
Для примера рассмотрим только элемент $(W_2)_{11}=0{,}2$.
У первого объекта первый скрытый признак равен 5. Производная средней
cross-entropy по первому логиту равна $(p_0-1)/2$, где деление на 2
появляется из усреднения двух объектов. У второго объекта этот скрытый
признак равен нулю, поэтому его вклад в производную данного веса нулевой:

$$
\begin{aligned}
\frac{\partial L}{\partial (W_2)_{11}}
&=5\frac{0{,}880797-1}{2}\\
&\approx-0{,}298007.
\end{aligned}
$$

При шаге SGD $\eta=0{,}1$ этот вес станет
$0{,}2-0{,}1(-0{,}298007)\approx0{,}229801$.
Ниже оптимизатор обновляет все параметры, а не только этот вес.
Полный вывод производных и накопление вкладов по разным путям будут
разобраны в главе об обратном распространении.

В PyTorch слой `Linear(d_in, d_out)` хранит матрицу в форме
`[d_out, d_in]` и вычисляет `x @ weight.T + bias`. Поэтому при переносе
наших матриц в `Linear.weight` их нужно транспонировать. Это соглашение
API, а не другая математическая операция:
[документация Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html).

```python
import torch

class MLP(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.fc2 = torch.nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP(2, 3, 2).double()
X = torch.tensor([[1., 2.], [2., -1.]], dtype=torch.float64)
y = torch.tensor([0, 1])

W1 = torch.tensor([[1., -1., 2.], [2., 1., -1.]],
                  dtype=torch.float64)
W2 = torch.tensor([[.2, -.2], [.1, .1], [-.25, .25]],
                  dtype=torch.float64)
with torch.no_grad():
    model.fc1.weight.copy_(W1.T)
    model.fc1.bias.copy_(torch.tensor([0., 1., -1.]))
    model.fc2.weight.copy_(W2.T)
    model.fc2.bias.zero_()

optimizer = torch.optim.SGD(model.parameters(), lr=.1)
optimizer.zero_grad()
logits = model(X)
loss = torch.nn.functional.cross_entropy(logits, y)
before = loss.item()
loss.backward()
grad = model.fc2.weight.grad[0, 0].item()
optimizer.step()
with torch.no_grad():
    after = torch.nn.functional.cross_entropy(model(X), y).item()

print(logits.detach())
print(f"loss: {before:.6f} -> {after:.6f}")
print(f"gradient: {grad:.6f}")
assert sum(p.numel() for p in model.parameters()) == 17
assert after < before
```

Здесь средняя ошибка уменьшается с $0{,}126928$ до $0{,}098018$.
Это результат одного шага на двух заданных объектах. Он не оценивает
обобщение сети и не воспроизводит опыт с `two moons` на рисунке выше.

Вызов `backward()` заполняет градиенты, но не меняет веса. Веса меняет
`optimizer.step()`. `zero_grad()` очищает предыдущие градиенты: по умолчанию
PyTorch накапливает их между обратными проходами.

В нашем примере одна предактивация равна ровно нулю. Обычной производной
ReLU в этой точке нет; PyTorch использует в обратном проходе значение 0.
Это соглашение влияет на обновление первого слоя. Посчитанная выше
производная по весу второго слоя от выбора производной ReLU в нуле
не зависит: при её вычислении используются уже полученные активации.

В `cross_entropy` передаются исходные логиты. Функция уже включает
устойчивое вычисление log-softmax; второй softmax перед ней изменил бы
вычисляемую функцию потерь. Для отображения вероятностей отдельно можно
использовать `torch.softmax(logits, dim=-1)`. В регрессии выходной слой
может возвращать предсказанное число без softmax; для бинарной
классификации есть и вариант с одним логитом и `BCEWithLogitsLoss`.
Выбор выхода определяется задачей, а не самим названием MLP.

<!-- BEGIN REVIEW: Module state 2026-10-27 -->

## Model state: parameters, buffers, and submodules

The training example passes `model.parameters()` to SGD. The optimizer receives
four tensors: `fc1.weight`, `fc1.bias`, `fc2.weight`, and `fc2.bias`. It does not
inspect `forward()` to find them. When `self.fc1 = nn.Linear(...)` executes,
`nn.Module` registers `fc1` as a child module. `parameters()` then visits that
child and finds its registered weight and bias. It does the same for `fc2`.

The printed model exposes this nesting:

```text
MLP(
  (fc1): Linear(in_features=2, out_features=3, bias=True)
  (fc2): Linear(in_features=3, out_features=2, bias=True)
)
```

Here `N` counts the scalar values in each tensor.

| Child | Name | Shape | N |
|---|---|---|---:|
| `fc1` | `fc1.weight` | `[3, 2]` | 6 |
| `fc1` | `fc1.bias` | `[3]` | 3 |
| `fc2` | `fc2.weight` | `[2, 3]` | 6 |
| `fc2` | `fc2.bias` | `[2]` | 2 |

The names describe ownership; the shapes describe the stored tensors. Calling
the same layer twice in `forward()` uses the same parameters twice. It creates
two uses in the computation graph, not a second set of weights. Conversely,
a registered child can remain unused in a particular forward pass. The module
hierarchy and the graph of executed operations answer different questions.

### Registration does not depend on a gradient

The following code is the complete **Parameter Registration** example from
[Harvard ML Systems, ML Frameworks — Automatic parameter discovery](https://mlsysbook.ai/contents/vol1/frameworks/frameworks.html#sec-ml-frameworks-nn-module-parameter-discovery-4a91),
Vijay Janapa Reddi and contributors. Code and comments are unchanged from
[commit `2bd97c5`](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd).
The source is distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); this surrounding explanation
adds checks and distinguishes registration from autograd.

```python
import torch
import torch.nn as nn


class CustomLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size)
        )
        self.bias = nn.Parameter(torch.randn(output_size))
        self.register_buffer("running_mean", torch.zeros(output_size))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


layer = CustomLayer(10, 20)
# Framework discovers both parameters automatically:
for name, param in layer.named_parameters():
    print(f"{name}: shape {param.shape}")
```

The output lists `weight: shape torch.Size([20, 10])` and
`bias: shape torch.Size([20])`: 220 parameter values. `running_mean` contains
another 20 values, but is returned by `named_buffers()`, not `named_parameters()`.
This example never reads or updates `running_mean` in `forward()`. The name alone
does not implement a running average or normalization.

Assigning an `nn.Parameter` to a module attribute triggers registration through
[`Module.__setattr__`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/module.py#L1976).
Assigning a child `nn.Module` registers that child. `super().__init__()` creates
the registries before either assignment. A plain tensor assigned to a new
attribute is not automatically a parameter, even with `requires_grad=True`.

| State | List | Save | Move |
|---|---|---|---|
| Parameter | Yes | Yes | Yes |
| Saved buffer | No | Yes | Yes |
| Unsaved buffer | No | No | Yes |
| Plain tensor | No | No | No |

Here “List” means returned by `parameters()`, “Save” means included in
`state_dict()`, and “Move” means handled by `.to()`. Saved buffers use
`persistent=True`. An unsaved buffer uses `persistent=False`: it is omitted
from the state dictionary, not cleared after each forward pass.

These entries refer to ordinary non-`None` tensors and the standard module
implementation. A floating-point buffer can also require gradients: registration
does not disable autograd. An optimizer constructed from `model.parameters()`
does not receive buffers or plain tensor attributes, regardless of their
`requires_grad` flags. A parameter with `requires_grad=False` remains registered;
`parameters()` does not filter it out.

The distinction can be observed with a two-input, two-output instance. Set its
weights to `[[2, -1], [1, 3]]` and bias to `[0.5, -0.5]`. On input `[1, 2]`,
the output is `[0.5, 6.5]`. Add a separate plain tensor `plain = 2` and use
`sum(output) + plain**2` as the loss. The derivative with respect to `plain`
is 4, but SGD will not update it unless it is explicitly given that tensor.

```python
state_probe = CustomLayer(2, 2)
state_probe.plain = torch.tensor(2., requires_grad=True)
state_probe.register_buffer("scratch", torch.ones(2), persistent=False)
state_probe.double()
assert state_probe.weight.dtype == state_probe.scratch.dtype == torch.float64
assert state_probe.plain.dtype == torch.float32  # not registered

with torch.no_grad():
    state_probe.weight.copy_(torch.tensor([[2., -1.], [1., 3.]]))
    state_probe.bias.copy_(torch.tensor([.5, -.5]))
state_probe.bias.requires_grad_(False)
probe_input = torch.tensor([[1., 2.]], dtype=torch.float64)
probe_optimizer = torch.optim.SGD(state_probe.parameters(), lr=.1)
probe_output = state_probe(probe_input)
torch.testing.assert_close(probe_output, torch.tensor([[.5, 6.5]], dtype=torch.float64))
probe_loss = probe_output.sum() + state_probe.plain.square()
probe_loss.backward()
probe_optimizer.step()

assert state_probe.plain.grad.item() == 4 and state_probe.plain.item() == 2
assert state_probe.bias.grad is None
assert "bias" in dict(state_probe.named_parameters())
torch.testing.assert_close(state_probe.bias, torch.tensor([.5, -.5], dtype=torch.float64))
torch.testing.assert_close(state_probe.weight,
    torch.tensor([[1.9, -1.2], [.9, 2.8]], dtype=torch.float64))
assert set(state_probe.state_dict()) == {"weight", "bias", "running_mean"}
```

Both output coordinates use the input `[1, 2]`, so both weight rows receive
the gradient `[1, 2]`. A step of size 0.1 subtracts `[0.1, 0.2]` from each row.
The frozen bias receives no gradient in this run. The buffer remains zero
because it takes no part in the computation. The plain tensor receives a
gradient, but is absent from the optimizer's parameter groups. Thus a missing
weight update can be a registration problem even when backward succeeds.

Freezing an existing parameter during training also requires clearing any old
gradient already stored in `.grad`. Setting `requires_grad=False` prevents new
gradient computation; it does not erase the previous tensor, and an optimizer
can still act on that tensor. In the example, the bias is frozen before its
first backward pass. Freezing and gradient accumulation are developed further
in the [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|backpropagation chapter]].

### A list of layers must register its contents

PyTorch's [Modules as Building Blocks](https://docs.pytorch.org/docs/2.14/notes/modules.html#modules-as-building-blocks)
describes the recursive rule:

> For any given module, its parameters consist of its direct parameters as well
> as the parameters of all submodules.

A Python list does not register the modules stored inside it. `nn.ModuleList`
does. It stores layers for use by your `forward()`; it does not define a forward
computation of its own. `nn.Sequential` additionally defines a computation
that passes each layer's output into the next layer. The following comparison
uses the same two `CustomLayer` objects in both containers, so differences
cannot be attributed to random initialization.

```python
class LayerStack(nn.Module):
    def __init__(self, blocks, register):
        super().__init__()
        self.blocks = nn.ModuleList(blocks) if register else list(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = torch.relu(block(x))
        return x

stack_blocks = [CustomLayer(2, 2), CustomLayer(2, 2)]
plain_stack = LayerStack(stack_blocks, register=False)
registered_stack = LayerStack(stack_blocks, register=True)
stack_input = torch.tensor([[1., 2.]])
torch.testing.assert_close(plain_stack(stack_input), registered_stack(stack_input))
assert list(plain_stack.parameters()) == []
assert plain_stack.state_dict() == {}
assert sum(p.numel() for p in registered_stack.parameters()) == 12
assert set(registered_stack.state_dict()) == {
    "blocks.0.weight", "blocks.0.bias", "blocks.0.running_mean",
    "blocks.1.weight", "blocks.1.bias", "blocks.1.running_mean",
}
```

Both forward calls work: Python can call a layer from an ordinary list, and
autograd can differentiate its tensor operations. Only `registered_stack`
exposes those layers to recursive module operations. For `plain_stack`, an
optimizer built from `parameters()` would receive an empty list. A model with
other correctly registered layers could instead train only that subset,
leaving the omitted layers unchanged without an empty-list error.

The registered stack has 12 parameter values: two copies of a 2×2 weight and
a length-2 bias. Its six state entries also include two length-2 buffers.
Parameter count and number of state entries count different objects.
The test deliberately shares the two layer objects between containers;
updating them through one container also affects the other.

### Saving values and rebuilding the computation

`state_dict()` returns named parameters and persistent buffers, including those
of registered children. The dotted key `blocks.0.weight` identifies the weight
of the first layer in `blocks`. It does not encode the ReLU in `forward()`,
the Python loop, or a constructor that can rebuild `LayerStack`.

Construct the destination model first, then load its tensor values. This
example serializes the modified `state_probe` in memory rather than creating
an extra file:

```python
import io

checkpoint_bytes = io.BytesIO()
torch.save(state_probe.state_dict(), checkpoint_bytes)
checkpoint_bytes.seek(0)
loaded_state = torch.load(checkpoint_bytes, weights_only=True)
restored_layer = CustomLayer(2, 2).double()
restored_layer.load_state_dict(loaded_state)
torch.testing.assert_close(restored_layer(probe_input), state_probe(probe_input))
torch.testing.assert_close(restored_layer(probe_input),
    torch.tensor([[0., 6.]], dtype=torch.float64))
assert not hasattr(restored_layer, "plain")
assert not hasattr(restored_layer, "scratch")
assert restored_layer.bias.requires_grad  # constructor default, not saved freeze state
```

`load_state_dict()` fills a compatible module that already exists. It does not
reconstruct its hierarchy from key names. A shape mismatch remains an error
even with `strict=False`; that option permits missing or unexpected keys,
not arbitrary conversion between architectures. The frozen bias's *value*
was restored, but its `requires_grad` flag came from the destination object.
The plain attribute and non-persistent buffer were not saved.

The returned state dictionary also holds references to the module's current
state. Keeping `best_state = model.state_dict()` in memory does not freeze a
historical copy; subsequent in-place updates can change the referenced values.
Use a deep copy when retaining a best model in memory. A resumable training
checkpoint needs optimizer state and other run state as well; the
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|optimization chapter]]
contains a complete save/resume example.

### Evaluation mode and gradient recording

Calling `.eval()` recursively sets the registered modules' `training` flags
to `False`. Layers that use this flag can then select evaluation behavior;
the `Linear` layers and functional ReLU in our MLP do not change their
formula. Dropout and BatchNorm will use it in later chapters. `.eval()` does
not disable gradient recording or freeze parameters.

```python
mode_probe = MLP(2, 3, 2).double().eval()
mode_input = torch.ones(1, 2, dtype=torch.float64)
assert all(not module.training for module in mode_probe.modules())
assert mode_probe(mode_input).requires_grad
with torch.no_grad():
    assert not mode_probe(mode_input).requires_grad
assert all(p.requires_grad for p in mode_probe.parameters())
```

Use `model(x)` rather than calling `model.forward(x)` directly: the module
call also runs registered hooks. The full API contracts and original English
examples are in PyTorch's [Modules](https://docs.pytorch.org/docs/2.14/notes/modules.html)
and [Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
documentation. The PyTorch excerpt above is reproduced under its
[BSD-style license](/licenses/pytorch-v2.14.1-LICENSE.txt).

<!-- END REVIEW: Module state 2026-10-27 -->

<!-- BEGIN REVIEW: Module hooks 2026-10-27 -->
## Observing a layer: forward and backward hooks

A forward hook receives a layer's inputs and computed output. A full backward
hook receives derivatives with respect to its inputs and outputs. The weights
and bias of `Linear` are attributes of the module, not positional inputs to
its `forward(x)`: their gradients are not entries in `grad_input`.

The complete original example below is from Vijay Janapa Reddi and Harvard
ML Systems contributors, [ML Frameworks, Module Hooks](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L3871),
CC BY-NC-SA 4.0. Code and English comments are unchanged.

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


# Forward hook to inspect activations
def forward_hook(module, input, output):
    print(
        f"Layer: {module.__class__.__name__}, "
        f"Output shape: {output.shape}, "
        f"mean={output.mean():.3f}, "
        f"std={output.std():.3f}"
    )


# Backward hook to inspect gradients
def backward_hook(module, grad_input, grad_output):
    print(f"Gradient norm: {grad_output[0].norm():.3f}")


# Register hooks on specific layer
handle_fwd = model[0].register_forward_hook(forward_hook)
handle_bwd = model[0].register_full_backward_hook(backward_hook)

# Execute forward and backward pass
x = torch.randn(32, 10)
y = model(x)
loss = y.sum()
loss.backward()

# Remove hooks when done
handle_fwd.remove()
handle_bwd.remove()
```

Both hooks are attached to the first `Linear`. Its output has shape `[32,20]`;
the final model output has shape `[32,5]`. The sum seeds the derivative of
those final outputs with ones. Before reaching the observed layer, that
derivative passes through the second linear map and the ReLU mask. Thus the
printed norm describes a `[32,20]` derivative, not an all-ones array or the
first layer's weight gradient. The mean and standard deviation likewise
summarize all entries of that layer's output, mixing batch and feature axes.
Those two statistics cannot establish how each individual feature behaves.

### One layer, three different derivatives

Use the same two-input, two-output weights as the state example above,
before its SGD update. PyTorch stores the weight rows as
`[[2,-1],[1,3]]`, with bias `[0.5,-0.5]`. On input `[1,2]` the output is
`[0.5,6.5]`. Choose the scalar loss $L=z_1+2z_2=13.5$.
The output derivative is `[1,2]`: increasing the second output by a small
amount changes this loss twice as much as increasing the first.

The input derivatives combine both output paths. For the first input,
$1\cdot2+2\cdot1=4$; for the second, $1\cdot(-1)+2\cdot3=5$.
Each weight derivative instead multiplies its output's incoming derivative
by the corresponding input value. The vector norms below are Euclidean;
the matrix norm is Frobenius.

| Quantity | Values | Norm |
|---|---|---|
| `grad_output[0]` | `[1,2]` | $\sqrt5$ |
| `grad_input[0]` | `[4,5]` | $\sqrt{41}$ |
| `weight.grad` | `[1,2]`<br>`[2,4]` | $5$ |

The bias derivative is `[1,2]`. Its numerical equality with the output
derivative comes from adding one bias vector to this one example; with a
batch, the bias gradient sums the output derivatives over examples.
The [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|backpropagation chapter]]
derives these matrix rules.

The following observation function keeps detached copies, so the records do
not retain the training graph. Neither callback returns a replacement tensor.
Handles are removed even if the computation raises an exception.

```python
def inspect_linear(input_requires_grad, parameters_require_grad, evaluation=False):
    observed_layer = nn.Linear(2, 2).double().train(not evaluation)
    with torch.no_grad():
        observed_layer.weight.copy_(torch.tensor([[2., -1.], [1., 3.]]))
        observed_layer.bias.copy_(torch.tensor([.5, -.5]))
    observed_layer.requires_grad_(parameters_require_grad)
    records = []

    def capture_forward(module, args, output):
        records.append({"stage": "forward", "output": output.detach().clone()})

    def capture_backward(module, grad_input, grad_output):
        records.append({
            "stage": "backward",
            "input_gradient": None if grad_input[0] is None else grad_input[0].detach().clone(),
            "output_gradient": grad_output[0].detach().clone(),
        })

    forward_handle = observed_layer.register_forward_hook(capture_forward)
    backward_handle = observed_layer.register_full_backward_hook(capture_backward)
    observed_input = torch.tensor([[1., 2.]], dtype=torch.float64,
                                  requires_grad=input_requires_grad)
    try:
        observed_output = observed_layer(observed_input)
        coefficients = observed_output.new_tensor([[1., 2.]])
        observed_loss = (observed_output * coefficients).sum()
        if observed_loss.requires_grad:
            observed_loss.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    count = len(records)
    with torch.no_grad():
        observed_layer(observed_input)
    assert len(records) == count  # removed hooks no longer observe the layer
    return observed_layer, observed_input, records, observed_loss.detach()

observed_layer, observed_input, observed_records, observed_loss = inspect_linear(True, True)
forward_record, backward_record = observed_records
torch.testing.assert_close(forward_record["output"], observed_input.new_tensor([[.5, 6.5]]))
torch.testing.assert_close(observed_loss, observed_input.new_tensor(13.5))
torch.testing.assert_close(backward_record["output_gradient"], observed_input.new_tensor([[1., 2.]]))
torch.testing.assert_close(backward_record["input_gradient"], observed_input.new_tensor([[4., 5.]]))
torch.testing.assert_close(observed_layer.weight.grad, observed_input.new_tensor([[1., 2.], [2., 4.]]))
torch.testing.assert_close(observed_layer.bias.grad, observed_input.new_tensor([1., 2.]))
torch.testing.assert_close(observed_input.grad, backward_record["input_gradient"])
```

### When the backward hook fires

In the original Harvard example the input does not require gradients, but
the layer's parameters do. Its output therefore participates in autograd,
and the full backward hook observes the output derivative. It has no input
derivative to report. Freezing parameters while requesting an input derivative
produces the opposite parameter-storage result: the hook still runs and the
input receives a derivative, but `weight.grad` remains `None`.
PyTorch 2.14 emits a warning for output-side firing when no inputs require
gradients. This follows the documented second firing rule; it does not mean
that the parameter gradient is missing.

| Input flag | Params flag | Backward record | Weight `.grad` |
|---|---|---|---|
| Yes | Yes | Input `[4,5]`, output `[1,2]` | Present |
| No | Yes | Input `None`, output `[1,2]` | Present |
| Yes | No | Input `[4,5]`, output `[1,2]` | `None` |
| No | No | No backward record | `None` |

```python
for input_grad in (False, True):
    for parameter_grad in (False, True):
        case_layer, case_input, case_records, _ = inspect_linear(input_grad, parameter_grad)
        assert len(case_records) == 1 + int(input_grad or parameter_grad)
        assert (case_layer.weight.grad is not None) == parameter_grad
        assert (case_input.grad is not None) == input_grad
        if input_grad or parameter_grad:
            observed = case_records[1]
            assert (observed["input_gradient"] is not None) == input_grad
            torch.testing.assert_close(observed["output_gradient"], case_input.new_tensor([[1., 2.]]))

# eval changes layer behavior, not whether autograd records these operations.
_, _, eval_records, _ = inspect_linear(True, True, evaluation=True)
assert [record["stage"] for record in eval_records] == ["forward", "backward"]
with torch.no_grad():
    _, _, no_grad_records, _ = inspect_linear(True, True)
assert [record["stage"] for record in no_grad_records] == ["forward"]
```

For this one-input `Linear`, `grad_input` and `grad_output` are one-element
tuples. More general module hooks receive derivatives for positional inputs;
keyword arguments are not included in `grad_input`. PyTorch's
[full-backward-hook contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)
specifies the firing rules and the restriction on in-place modification of
inputs and outputs. Full backward hooks introduce views around those tensors.

Calling `forward` directly bypasses the module-call machinery that installs
the hooks. The numerical layer operation still runs and autograd can still
differentiate it:

```python
bypass_layer = nn.Linear(2, 2).double()
bypass_input = torch.ones(1, 2, dtype=torch.float64, requires_grad=True)
bypass_records = []
bypass_fwd = bypass_layer.register_forward_hook(lambda *args: bypass_records.append("forward"))
bypass_bwd = bypass_layer.register_full_backward_hook(lambda *args: bypass_records.append("backward"))
try:
    bypass_layer.forward(bypass_input).sum().backward()
    assert bypass_records == [] and bypass_input.grad is not None
    bypass_layer.zero_grad(set_to_none=True)
    bypass_input.grad = None
    bypass_layer(bypass_input).sum().backward()
    assert bypass_records == ["forward", "backward"]
finally:
    bypass_fwd.remove()
    bypass_bwd.remove()
```

Hooks can also change outputs or replace input derivatives when they return
replacement values. The callbacks here return `None` and only record values.
Recording every activation, computing reductions and printing scalars adds
work and can change memory use or timing. These CPU examples establish what
is observed; they are not a low-overhead profiler or a compiler/GPU benchmark.

<!-- END REVIEW: Module hooks 2026-10-27 -->

## Ширина, глубина и число параметров

Сеть из примера имеет два обучаемых слоя: скрытый и выходной. Входные
координаты не выполняют преобразование и в этот счёт не входят.
Ширина скрытого слоя равна 3. Увеличить ширину означает вычислять больше
признаков на одном уровне; увеличить глубину — добавить ещё одно
последовательное преобразование.

У слоя из $d$ входов и $m$ выходов $dm$ весов и $m$ смещений.
Для нашего MLP:

| Слой | Веса | Смещения | Всего |
|---|---:|---:|---:|
| $2\to3$ | $2\cdot3=6$ | 3 | 9 |
| $3\to2$ | $3\cdot2=6$ | 2 | 8 |
| Сеть | 12 | 5 | 17 |

В общем случае одного скрытого слоя

$$
N_{\text{params}}=d_{\text{in}}m+m+md_{\text{out}}+d_{\text{out}}.
$$

На первой схеме CS231n архитектура $3\to4\to4\to1$ содержит
$(12+4)+(16+4)+(4+1)=41$ параметр. Схема показывает рёбра с весами;
если посчитать только рёбра, смещения потеряются.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-depth-width.jpeg]]

*Три сети с одним скрытым слоем ширины 3, 6 и 20. Глубина одинакова;
сравнивается именно ширина. Точки — обучающие объекты, заливка —
предсказанные классы. У широкой сети видны небольшие отдельные области
вокруг некоторых точек. Источник: Stanford CS231n,
[Setting number of layers and their sizes](https://cs231n.github.io/neural-networks-1/#arch),
[оригинальное изображение](https://cs231n.github.io/assets/nn1/layer_sizes.jpeg),
без изменений.*

Сеть с большим числом нейронов может провести более сложную границу,
но этот рисунок не содержит тестовой выборки. По нему нельзя заключить,
какая из трёх сетей лучше предсказывает новые объекты. Отдельные области
могут соответствовать как закономерности, так и шуму.

При фиксированной ширине поведение зависит и от регуляризации:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-decision-regions.jpeg]]

*Во всех трёх сетях по 20 скрытых нейронов, а коэффициент L2-штрафа
последовательно равен $0{,}001$, $0{,}01$ и $0{,}1$.
В этом опыте при большем штрафе граница становится менее извилистой.
Источник: Stanford CS231n,
[Setting number of layers and their sizes](https://cs231n.github.io/neural-networks-1/#arch),
[оригинальное изображение](https://cs231n.github.io/assets/nn1/reg_strengths.jpeg),
без изменений.*

L2-штраф добавляет к функции ошибки слагаемое, зависящее от квадратов
весов. Коэффициент определяет его вклад в оптимизацию. Сглаживание
границы на этих трёх картинках — результат конкретного опыта, а не
теорема о монотонном изменении формы границы при любом запуске обучения.
Ширину, глубину и силу регуляризации выбирают по валидации; обучение
и проверка должны использовать одно и то же преобразование признаков.

Теоремы об универсальной аппроксимации описывают выразительность сети.
Например, непрерывную функцию на компактной области можно с любой заданной
точностью приблизить сетью с одним достаточно широким скрытым слоем
и подходящей нелинейностью. Теорема не задаёт доступную на практике ширину,
не выбирает веса и не гарантирует их нахождение градиентным спуском.
Тем более она не гарантирует качество на неизвестном распределении данных.
Наглядное объяснение условий и конструкции:
[Michael Nielsen, Chapter 4](http://neuralnetworksanddeeplearning.com/chap4.html).

## MLP внутри Transformer

В Transformer у объекта есть ещё одна ось: позиция токена.
Пакет представлений имеет форму $[B,T,d_{\text{model}}]$.
Обычный FFN, или feed-forward-подслой, применяет один MLP к последней оси:

$$
\begin{aligned}
h_{b,t}&=\phi(x_{b,t}W_{\text{up}}+b_{\text{up}}),\\
\operatorname{FFN}(x_{b,t})&=h_{b,t}W_{\text{down}}+b_{\text{down}}.
\end{aligned}
$$

Для каждой пары $(b,t)$ берётся отдельный входной вектор, но одни и те же
матрицы $W_{\text{up}},W_{\text{down}}$. При $B=2$, $T=3$,
$d_{\text{model}}=2$ и $d_{\text{ff}}=3$ формы будут

$$
[2,3,2]\ \longrightarrow\ [2,3,3]\ \longrightarrow\ [2,3,2].
$$

Наш скрытый слой $2\to3$ уже подходит для первого перехода; выход $3\to2$
возвращает прежнюю ширину представления. Для шести позиций остаются те же
17 параметров, а не шесть копий по 17. Если изменить вход только одной
позиции, этот FFN изменит результат только в ней. Сам по себе он не читает
соседние позиции. Связь между токенами появляется в attention, который
изучается далее. Вход FFN после attention уже может содержать сведения
о доступном контексте.

Для одной последовательности из трёх векторов исходные веса примера
(до шага SGD) дадут:

| Позиция | Вход FFN | Выход FFN |
|---|---|---|
| 1 | $(1,2)$ | $(1{,}2;\,-0{,}8)$ |
| 2 | $(2,-1)$ | $(-1,1)$ |
| 3 | $(1,2)$ | $(1{,}2;\,-0{,}8)$ |

Одинаковые входы в позициях 1 и 3 дают одинаковые выходы. Если заменить
только второй вход на $(1,2)$, изменится только вторая строка результата:
она тоже станет $(1{,}2;\,-0{,}8)$. Номер позиции не входит в формулу FFN.

Такое применение одной сети ко всем позициям показано в
[Dive into Deep Learning, §11.7.2 Positionwise Feed-Forward Networks](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html#positionwise-feed-forward-networks).

Выход FFN в таком блоке — новый вектор признаков, а не вероятности токенов
словаря. Поэтому softmax из примера классификации внутрь FFN не добавляется.
Предсказание следующего токена использует отдельный выходной слой языковой
модели.

В современных декодерах, включая LLaMA, встречается
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|SwiGLU]]:
две входные проекции образуют ветви, их результаты перемножаются,
а третья проекция возвращает размер $d_{\text{model}}$.
Это уже другая параметризация FFN; к ней вернёмся после attention,
residual-соединений и нормализации.

<span id="краткие-итоги"></span>

## Источники

- Dive into Deep Learning, [§5.1 Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html):
  скрытые представления и матричная запись в §5.1.1; выбор активации — в §5.1.2.
- Stanford CS231n, [Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/):
  устройство слоёв, подсчёт параметров и три исходных рисунка в этой главе.
- Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd) и
  [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html):
  скалярные вычисления, обратное распространение и обучение небольшой сети.
  Пример `two moons` и его рисунок взяты из репозитория micrograd.
- PyTorch, [Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)
  и [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html):
  формы параметров, последние оси входа и вычисление ошибки по логитам.
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/03_layers|TinyTorch 03 — Layers]] —
  реализация полносвязного слоя и учёт параметров.
- [[05 Источники/Courses/Machine Learning Visualized/book/main.pdf|Machine Learning Visualized — Complete Book]]
  и [[05 Источники/Courses/Machine Learning Visualized/chapter3/interactive_perceptron|Interactive Perceptron]] —
  дополнительные объяснения и интерактивное исследование линейного правила.

Далее: [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации|функции активации]].
