---
title: Вычислительный граф и backpropagation
type: textbook-chapter
status: canonical
last_updated: 2026-10-27
primary_sources:
  - https://cs231n.github.io/optimization-2/
  - https://github.com/karpathy/micrograd
  - https://colah.github.io/posts/2015-08-Backprop/
  - https://docs.pytorch.org/docs/2.14/notes/autograd.html
---

# Вычислительный граф и обратное распространение ошибки

В [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|главе об MLP]]
мы вычислили предсказания сети и сделали шаг обучения. Между вызовами
`loss.backward()` и `optimizer.step()` появились градиенты параметров.
Чтобы получить их, библиотека разбирает выполненные вычисления на операции
и применяет правило цепочки в обратном порядке.

## 1. Что говорит производная

Для $f(x)=x^2$ в точке $x=3$ производная равна шести. Если увеличить $x$
на $0{,}001$, функция вырастет на $0{,}006001$:

$$
f(3.001)-f(3)=0.006001\approx 6\cdot0.001.
$$

Производная задаёт линейное приближение в окрестности точки. При большом
шаге оно может оказаться неточным: увеличение $x$ с трёх до четырёх
увеличивает квадрат на семь, а не на шесть.

Для функции нескольких аргументов частная производная описывает изменение
одного аргумента при фиксированных остальных. Градиент объединяет эти
производные в вектор. При обучении входной пример обычно фиксирован,
а изменять мы будем веса и смещения. При этом производные по входам
промежуточного слоя тоже нужны: через них градиент передаётся предыдущему слою.

Ниже $\bar v$ обозначает $\partial L/\partial v$, производную конечной
скалярной величины $L$ по значению узла $v$. Это не то же самое, что
локальная производная отдельной операции.

## 2. Один вычислительный граф

В примере Stanford CS231n вычисляется $L=(x+y)z$. Положим
$x=-2$, $y=5$, $z=-4$ и введём промежуточное значение $q=x+y$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/backpropagation-2026-10-21/cs231n-chain.svg]]

*Исходная схема Stanford CS231n,
[Compound expressions with chain rule](https://cs231n.github.io/optimization-2/#backprop).
Зелёные числа над линиями обозначают значения прямого прохода, красные
под линиями — производные конечного выхода по этим значениям.
На рисунке конечный выход назван $f$, в тексте — $L$.
Автор: Andrej Karpathy. [MIT License](/licenses/cs231n-MIT.txt).
Содержимое схемы сохранено; добавлены только XML-атрибуты для отдельного SVG.*

Прямой проход выполняет операции в порядке зависимостей:
сначала $q=-2+5=3$, затем $L=3\cdot(-4)=-12$.
Отрицательное значение здесь допустимо: это учебное выражение,
а не выбранная для обучения функция потерь.

Обратный проход начинается с $\bar L=1$, потому что
$\partial L/\partial L=1$. Последняя операция умножает $q$ на $z$.
Если менять $q$, оставляя $z$ фиксированным, её производная равна $z$;
если менять $z$, производная равна $q$. Поэтому

$$
\bar q=\bar L\cdot z=-4,\qquad
\bar z=\bar L\cdot q=3.
$$

Теперь можно пройти через сложение $q=x+y$. Его локальная производная
по каждому входу равна единице:

$$
\bar x=\bar q\cdot1=-4,\qquad
\bar y=\bar q\cdot1=-4.
$$

| Узел | Значение прямого прохода | Производная конечного $L$ |
| --- | ---: | ---: |
| $L$ | $-12$ | $1$ |
| $q$ | $3$ | $-4$ |
| $z$ | $-4$ | $3$ |
| $x$ | $-2$ | $-4$ |
| $y$ | $5$ | $-4$ |

Локальная производная $\partial q/\partial x$ равна единице, но конечная
$\partial L/\partial x$ равна $-4$. Множитель $-4$ учитывает то, как результат
сложения используется дальше. Если увеличить $x$ на $0{,}001$,
$q$ увеличится на столько же, а $L$ уменьшится на $0{,}004$.

Каждой операции для обратного прохода достаточно её входных значений
и производной по её выходу. Умножению нужны $q$ и $z$, а сложению
численные значения входов вообще не нужны для вычисления производных.

## 3. Почему не считать производную каждого параметра отдельно

Можно оценить производную по параметру $\theta_i$, выполнив ещё один
прямой проход с небольшим изменением этого параметра:

$$
\frac{\partial L}{\partial\theta_i}
\approx
\frac{L(\theta+\varepsilon e_i)-L(\theta)}{\varepsilon}.
$$

Здесь $e_i$ изменяет только координату $i$. Для $N$ параметров такая
проверка требует $N$ дополнительных вычислений функции.
При центральной разности потребуются два вычисления на координату.

Обратный проход использует уже вычисленные промежуточные результаты.
В скалярном графе каждая операция обрабатывается после того, как собраны
вклады всех её потребителей. Число действий зависит от размера графа,
а не от числа отдельных путей между входом и выходом.
Обратный проход матричного слоя включает умножения матриц;
их стоимость зависит от размерностей слоя и батча.

В [[#4. Правило цепочки и разветвлённый граф|разветвлённом примере ниже]]
видно, почему это экономит вычисления: производная по общему узлу сначала
суммируется, после чего его предшественники обрабатываются один раз.
Такое объединение путей разбирает Chris Olah в
[Factoring Paths](https://colah.github.io/posts/2015-08-Backprop/#factoring-paths).

Экономия требует памяти. Например, для производной произведения нужны
его множители, а для производной $\tanh$ достаточно сохранить выход.
Можно пересчитывать часть значений вместо хранения; к этому компромиссу
возвращается глава о
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload|gradient checkpointing]].

<span id="4-правило-цепочки-в-одной-строке"></span>

## 4. Правило цепочки и разветвлённый граф

Для $L=f(g(x))$ правило цепочки даёт

$$
\frac{dL}{dx}=\frac{dL}{dg}\frac{dg}{dx}.
$$

Если значение используется несколько раз, нужно учесть каждый путь.
Возьмём пример из статьи Chris Olah:
$c=a+b$, $d=b+1$, $L=cd$. При $a=2$ и $b=1$ получаем
$c=3$, $d=2$, $L=6$.

Из последнего умножения приходят $\bar c=d=2$ и $\bar d=c=3$.
Через $c=a+b$ в производную по $b$ входит $2\cdot1=2$.
Через $d=b+1$ входит ещё $3\cdot1=3$:

$$
\bar b=2+3=5,\qquad \bar a=2.
$$

Оба вклада относятся к одному $b$. Если сохранить только последний,
получится неверная производная. Проверка изменением аргумента подтверждает сумму:
при $b=1+h$

$$
L=(3+h)(2+h)=6+5h+h^2.
$$

Коэффициент линейного приращения равен пяти. В общем случае для узла $v$
с несколькими непосредственными потребителями $u_k$

$$
\bar v=\sum_k\bar u_k\frac{\partial u_k}{\partial v}.
$$

Особый случай — $L=x\cdot x$. При $x=3$ в операцию умножения дважды
передан один и тот же объект. Первый вход даёт вклад $3$, второй тоже $3$;
производная равна шести. Удалять повторный узел из списка посещённых
при обходе можно; удалять второй вклад операции нельзя.

### Сложение, умножение и выбор максимума

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/backpropagation-2026-10-21/cs231n-gates.svg]]

*Stanford CS231n,
[Patterns in backward flow](https://cs231n.github.io/optimization-2/#patterns).
Исходная схема Andrej Karpathy; [MIT License](/licenses/cs231n-MIT.txt).
Числа и геометрия не изменены; исправлена запись XML-атрибутов.*

На рисунке вычисляется $L=2(xy+\max(z,w))$. При
$x=3$, $y=-4$, $z=2$, $w=-1$ внутреннее сложение даёт $-10$,
а результат равен $-20$. Множитель два передаёт производную $2$
обоим слагаемым. Из произведения получаем
$\bar x=2y=-8$ и $\bar y=2x=6$.
В максимуме выиграл $z$, поэтому $\bar z=2$, а $\bar w=0$:
малое изменение $w$ пока не меняет результат.

Сложение передаёт всю входящую производную каждому аргументу;
оно не делит её между ними. Умножение дополнительно масштабирует
производную значением другого множителя. Выбор максимума при разных
аргументах передаёт производную только выбранному входу.
При равенстве аргументов обычной производной нет; правило библиотеки
нужно уточнять отдельно. Например, PyTorch для `relu(0)` выбирает ноль.

### Общий параметр и residual-путь

Пусть один вес $w$ применяется к двум входам:
$L=(wx_1)^2+(wx_2)^2$. При $w=2$, $x_1=1$, $x_2=3$
два слагаемых равны $4$ и $36$. Производные по общему весу равны
$2wx_1^2=4$ и $2wx_2^2=36$, их сумма — $40$.

Матрица FFN в Transformer применяется ко всем позициям с общими параметрами.
Поэтому её градиент складывается из вкладов позиций, а при усреднённой
функции потерь содержит ещё и соответствующий нормирующий множитель.
У позиций разные активации, но отдельные копии параметров не создаются.

В residual-блоке $y=x+F(x)$ значение $x$ тоже используется двумя путями.
В скалярном случае

$$
\frac{dL}{dx}=\frac{dL}{dy}\left(1+F'(x)\right).
$$

Например, для $F(x)=2x$, $x=3$, $L=y^2/2$ получаем $y=9$,
$\bar y=9$ и $\bar x=9+18=27$. Если $F'(x)=0$, остаётся прямой
вклад $\bar y$. Однако при $F'(x)=-1$ два вклада могут взаимно
уничтожиться: residual-соединение не гарантирует ненулевой градиент.

<span id="5-от-числа-к-нейрону"></span>

## 5. Обратный проход нейрона и матричного слоя

Возьмём тот же нейрон, который ниже обучим в PyTorch:
$x=(1,2)$, $w=(2,-1)$, $b=-0{,}5$, целевой ответ $t=1$.
Прямой проход состоит из трёх этапов:

$$
\begin{aligned}
z&=w^\top x+b=-0.5,\\
p&=\tanh(z)\approx-0.462117.
\end{aligned}
$$

$$
L=\frac12(p-t)^2\approx1.068893.
$$

Выход нейрона слишком мал относительно цели. Производная потерь по
предсказанию равна $\bar p=p-t\approx-1{,}462117$.
Для $\tanh$ локальная производная равна $1-p^2\approx0{,}786448$.
После умножения

$$
\bar z=(p-t)(1-p^2)\approx-1.149879.
$$

При небольшом увеличении $z$ потери уменьшаются. Теперь остаётся пройти
через взвешенную сумму:

$$
\bar w_i=\bar z\,x_i,\qquad
\bar b=\bar z,\qquad
\bar x_i=\bar z\,w_i.
$$

| Величина | Производная $L$ |
| --- | ---: |
| $w_1$ | $-1.149879$ |
| $w_2$ | $-2.299757$ |
| $b$ | $-1.149879$ |
| $x_1$ | $-2.299757$ |
| $x_2$ | $1.149879$ |

Вход $x_2$ вдвое больше $x_1$, поэтому производная по второму весу
вдвое больше по модулю. Веса задают другой масштаб для производных по входам.
Если $x$ — выход предыдущего слоя, именно $\bar x$ позволяет продолжить
обратный проход вглубь сети.

### Одну функцию можно считать одной операцией

Для вычисления $\tanh$ не обязательно раскрывать её в экспоненты
и дифференцировать каждую из них. Достаточно знать производную целой
функции. CS231n показывает это для сигмоидального нейрона.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/backpropagation-2026-10-21/cs231n-sigmoid.svg]]

*Stanford CS231n,
[Modularity: Sigmoid example](https://cs231n.github.io/optimization-2/#sigmoid).
Исходная схема Andrej Karpathy; [MIT License](/licenses/cs231n-MIT.txt).
Зелёные числа — значения, красные — производные конечного выхода.
Это другой нейрон: его входы $(-1,-2)$, веса $(2,-3)$ и смещение $-3$.
Здесь дифференцируется сам выход сигмоиды, без функции потерь.
Для мелких подписей доступно увеличение по нажатию на рисунок.*

У этого нейрона взвешенная сумма равна $1$, выход
$\sigma(1)\approx0{,}731059$. Производная по сумме равна
$p(1-p)\approx0{,}196612$. На рисунке она получена через отрицание,
экспоненту, прибавление единицы и обратную величину; ту же производную
даёт одна операция `sigmoid`. Поэтому производные по двум весам
равны примерно $-0{,}196612$ и $-0{,}393224$, а по смещению — $0{,}196612$.
Если добавить функцию потерь, все эти производные нужно домножить
на производную потерь по выходу нейрона.

### Обратный проход матричного слоя

В главе об MLP примеры записаны строками:
$X$ имеет форму $B\times d$, $W$ — $d\times m$, смещение $b$ — длину $m$.
Прямой проход вычисляет $Z=XW+b$.
Пусть следующий слой уже вернул $G=\partial L/\partial Z$
формы $B\times m$.

Для одного элемента $Z_{ij}=\sum_k X_{ik}W_{kj}+b_j$.
Вес $W_{kj}$ влияет на все строки $i$, а смещение $b_j$ прибавляется
ко всем строкам. Складывая эти вклады, получаем

$$
\begin{aligned}
\frac{\partial L}{\partial W_{kj}}
&=\sum_i X_{ik}G_{ij},\\
\frac{\partial L}{\partial b_j}&=\sum_i G_{ij}.
\end{aligned}
$$

Входной элемент $X_{ik}$ влияет на все выходные координаты $j$ своей строки:

$$
\frac{\partial L}{\partial X_{ik}}=\sum_j G_{ij}W_{kj}.
$$

В матричной форме это три операции:

$$
\begin{aligned}
\bar W&=X^\top G,\\
\bar X&=GW^\top,\\
\bar b&=\sum_i G_{i,:}.
\end{aligned}
$$

Транспонирование следует из индексов, а не из условного правила
«при обратном проходе перевернуть матрицу». Формы позволяют проверить
вывод: $\bar W$ должна иметь форму $d\times m$, $\bar X$ — $B\times d$,
$\bar b$ — $m$. Здесь используется соглашение $XW$; PyTorch
`nn.Linear` хранит вес формы $m\times d$ и вычисляет $XW^\top+b$.

Посчитаем на двух строках и двух выходах:

$$
X=\begin{pmatrix}1&2\\3&-1\end{pmatrix},\qquad
W=\begin{pmatrix}2&-1\\1&3\end{pmatrix},
$$

$$
\begin{aligned}
b&=(0.5,-0.5),\\[4pt]
Z&=\begin{pmatrix}4.5&4.5\\5.5&-6.5\end{pmatrix}.
\end{aligned}
$$

Выберем $L=\sum_{ij}G_{ij}Z_{ij}$ с фиксированной матрицей
$G=\begin{pmatrix}1&2\\-1&3\end{pmatrix}$.
Тогда производная по $Z$ в точности равна $G$, а $L=-11{,}5$.
Для $W_{11}$ вклад первой строки равен $1\cdot1=1$,
второй — $3\cdot(-1)=-3$, в сумме $-2$.
Остальные элементы дают

$$
\begin{aligned}
\bar W&=\begin{pmatrix}-2&11\\3&1\end{pmatrix},\\[4pt]
\bar X&=\begin{pmatrix}0&7\\-5&8\end{pmatrix},\\[4pt]
\bar b&=(0,5).
\end{aligned}
$$

Суммирование для $b$ обращает broadcasting прямого прохода:
один вектор использовался дважды, значит, нужно сложить два вклада.
При добавлении смещения к тензору $B\times T\times m$
складываются вклады и по батчу, и по позициям $T$.

Если за $Z$ идёт поэлементная активация $H=\phi(Z)$, то сначала
$G_Z=G_H\odot\phi'(Z)$, а затем применяются формулы матричного слоя.
Знак $\odot$ означает поэлементное умножение.
Так обратный проход MLP чередует умножения матриц и производные активаций.

```python
import torch

dtype = torch.float64
X = torch.tensor([[1., 2.], [3., -1.]], dtype=dtype, requires_grad=True)
W = torch.tensor([[2., -1.], [1., 3.]], dtype=dtype, requires_grad=True)
b = torch.tensor([.5, -.5], dtype=dtype, requires_grad=True)
G = torch.tensor([[1., 2.], [-1., 3.]], dtype=dtype)
Z = X @ W + b
loss = (Z * G).sum()
loss.backward()

torch.testing.assert_close(W.grad, X.detach().T @ G)
torch.testing.assert_close(X.grad, G @ W.detach().T)
torch.testing.assert_close(b.grad, G.sum(dim=0))
print(Z.detach(), W.grad, X.grad, b.grad, sep="\n")
```

## 6. Что хранит micrograd

`micrograd` Андрея Карпати реализует эти правила для скалярных чисел.
Объект `Value` хранит значение `data`, накопленную производную `grad`,
ссылки на предшественников `_prev` и функцию `_backward` для своей операции.
Ниже приведены исходные методы умножения и обратного прохода из
[engine.py](https://github.com/karpathy/micrograd/blob/7bc720e951fe422b8f8814aa5aa1b64121d26b4c/micrograd/engine.py),
без перевода комментариев. Это фрагменты класса, а не отдельная программа;
полный файл распространяется по [MIT License](/licenses/micrograd-MIT.txt).

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
```

Замыкание `_backward` запоминает множители `self` и `other` и созданный
результат `out`. К моменту его вызова производная по результату уже собрана
в `out.grad`. Две строки с `+=` добавляют соответствующие вклады входам.
Для `x*x` переменные `self` и `other` ссылаются на один объект,
поэтому обе строки изменяют один и тот же `grad`.

Топологический порядок ставит каждый узел после всех его предшественников.
В обратном порядке узел обрабатывается после своих потребителей.
В примере $c=a+b$, $d=b+1$, $L=cd$ допустим прямой порядок
$a,b,c,d,L$; обратный — $L,d,c,b,a$. К моменту обработки $b$
уже учтены обе ветви. `visited` не даёт обходить общий узел повторно,
но локальные операции всё равно добавляют каждый свой вклад.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/micrograd/computational-graph.svg]]

*Исходный граф из [micrograd](https://github.com/karpathy/micrograd/blob/7bc720e951fe422b8f8814aa5aa1b64121d26b4c/gout.svg),
Andrej Karpathy, [MIT License](/licenses/micrograd-MIT.txt).
В каждой записи `data` — значение узла, `grad` — производная конечного
выхода по нему. Это ReLU-нейрон с входами $1$ и $-2$, а не tanh-пример
из текста. Конечный выход положителен, поэтому производная ReLU здесь равна
единице. Числа в графе округлены до четырёх знаков. Рисунок можно увеличить.*

PyTorch применяет обратный режим к тензорным операциям и сохраняет нужные
для них данные. В обычном eager-исполнении граф строится по фактически
выполненным операциям заново при каждом прямом проходе.
Параметры являются листовыми тензорами, в чьи `.grad` по умолчанию
накапливается результат `backward()`. Производные промежуточных тензоров
участвуют в расчёте, но для их сохранения в `.grad` нужен `retain_grad()`.

Повторный `backward()` по тому же графу обычно невозможен после освобождения
сохранённых значений. Если нужен ещё один проход по прежнему графу,
используют `retain_graph=True`; для нового шага обучения обычно заново
вычисляют предсказание и потери. У учебного `micrograd` нет такого механизма
освобождения, и повторный вызов на том же графе со старыми промежуточными
`grad` не следует считать эквивалентом нового расчёта.

<span id="7-из-градиента-получается-обучение"></span>

## 7. Один шаг градиентного спуска

Вернёмся к tanh-нейрону. При скорости обучения $\eta=0{,}1$
градиентный спуск обновляет веса по правилу
$\theta_{\text{new}}=\theta-\eta\bar\theta$:

$$
\begin{aligned}
w_{\text{new}}&\approx(2.114988,-0.770024),\\
b_{\text{new}}&\approx-0.385012.
\end{aligned}
$$

Обе производные по весам были отрицательными, поэтому оба веса выросли.
На том же входе новая взвешенная сумма равна $0{,}189927$,
предсказание — $0{,}187676$, потери — $0{,}329935$ вместо $1{,}068893$.
При ненулевом градиенте и достаточно малом шаге движение против него
уменьшает дифференцируемую функцию. Произвольно большой шаг может увеличить потери.
Размер шага и более сложные обновления разбираются в следующей главе.

```python
import torch

x = torch.tensor([1., 2.], dtype=torch.float64)
w = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
b = torch.nn.Parameter(torch.tensor(-.5, dtype=torch.float64))
optimizer = torch.optim.SGD([w, b], lr=.1)

optimizer.zero_grad()
prediction = torch.tanh(w @ x + b)
loss = .5 * (prediction - 1).square()
before = loss.item()
loss.backward()
expected = (prediction.detach() - 1) * (1 - prediction.detach().square())
torch.testing.assert_close(w.grad, expected * x)
torch.testing.assert_close(b.grad, expected)
# backward computes gradients; it has not changed these parameters.
torch.testing.assert_close(w.detach(), torch.tensor([2., -1.], dtype=w.dtype))
optimizer.step()
with torch.no_grad():
    after = (.5 * (torch.tanh(w @ x + b) - 1).square()).item()
assert after < before
print(f"{before:.6f} -> {after:.6f}")
```

`backward()` вычисляет градиенты; `step()` меняет параметры.
Блок `no_grad()` в конце исключает контрольную оценку из нового
обучающего графа. Ни один из этих вызовов не выбирает правильную функцию
потерь за пользователя: производная вычисляется именно для записанной программы.

## 8. Почему градиенты нужно обнулять

Нужно различать сложение вкладов внутри одного графа и сохранение результата
между вызовами `backward()`. Внутри графа сумма обязательна по правилу цепочки.
Между отдельными вычислениями PyTorch тоже добавляет градиенты в прежние
`.grad`. Это позволяет накапливать несколько порций данных перед одним
обновлением, но требует явной очистки перед следующим шагом.

Например, при $w=2$ функция $L_1=w^2/2$ даёт производную $2$.
Ещё один прямой проход для $L_2=3w$ даёт производную $3$.
Без очистки в `w.grad` будет $5$; после очистки перед вторым проходом — $3$.
Сам `optimizer.step()` накопленные градиенты не обнуляет.

```python
import torch

w_acc = torch.tensor(2., requires_grad=True)
(.5 * w_acc.square()).backward()
assert w_acc.grad.item() == 2
(3 * w_acc).backward()  # a new graph, the same leaf
assert w_acc.grad.item() == 5
w_acc.grad = None
(3 * w_acc).backward()
assert w_acc.grad.item() == 3
```

`optimizer.zero_grad(set_to_none=True)` очищает градиенты параметров
оптимизатора перед новой группой вычислений. `None` означает отсутствие
сохранённого градиента и отличается от тензора нулей: оптимизатор может
пропустить параметр с `grad=None`, тогда как нулевой градиент не обязательно
означает отсутствие обновления, например при momentum или weight decay.

### Накопление по нескольким мини-батчам

Пусть средняя функция потерь задана по пяти независимым примерам,
а в память помещаются порции из двух и трёх примеров.
Для средних потерь $L_A$ и $L_B$ общий результат равен

$$
L=\frac25 L_A+\frac35 L_B.
$$

Следовательно, можно вызвать `(.4 * loss_A).backward()`, затем
`(.6 * loss_B).backward()` и сделать один `step()`.
Параметры между этими проходами не меняются.
Деление каждой средней потери на два дало бы одинаковый вес двум порциям,
хотя у них разное число примеров. При усреднении по токенам аналогично
учитывается число незамаскированных целевых токенов, а не число последовательностей.

Равенство с большим батчем предполагает ту же функцию для каждого примера.
BatchNorm меняет статистики при разбиении батча; стохастические операции
могут дать другие реализации случайности. Совпадение накопленной суммы
градиентов само по себе не устраняет эти различия.

## 9. Автоматическое дифференцирование в обратном режиме

До сих пор конечный результат был скаляром. Если функция возвращает вектор,
нужно уточнить, какая комбинация его координат нас интересует.
Для $u=2$ возьмём $v=(u^2,3u)=(4,6)$.
Производные координат по $u$ равны $4$ и $3$.
Для $L=v_1+2v_2$ получаем $dL/du=4+2\cdot3=10$.

```python
import torch

u = torch.tensor(2., requires_grad=True)
v = torch.stack((u.square(), 3 * u))
v.backward(torch.tensor([1., 2.]))
assert u.grad.item() == 10
```

Переданный в `backward` вектор задаёт производную по выходу.
Для отображения $v=f(\theta)$ с $N$ входами и $M$ выходами
матрица Якоби $J$ имеет размер $M\times N$.
При входящей производной $g$ обратный проход вычисляет $J^\top g$,
не создавая всю матрицу $J$. Транспонированная запись результата
имеет вид $g^\top J$; это vector–Jacobian product, VJP.
При одном скалярном выходе $g=1$ получаем весь градиент по $N$ параметрам.

Это объясняет пригодность обратного режима для обучения: параметров много,
а скалярная функция потерь одна. Прямой режим передаёт возмущение входа
в направлении $h$ и получает $Jh$. Он удобен для другого соотношения
размерностей. Для полной матрицы Якоби в общем случае нужно несколько
направлений: $M$ базисных выходных в обратном режиме или $N$ входных
в прямом, с возможностью группировать вычисления.
Один `backward` векторного выхода не возвращает сразу все её элементы.

В матричном примере роль входящей производной играла $G$.
У слоя могли быть тысячи выходных элементов; следующий слой уже объединил
их влияние на скалярные потери. Поэтому ему достаточно передать $G$,
а не полную матрицу Якоби.

<!-- BEGIN REVIEW: forward mode 2026-10-27 -->
### Forward mode: carrying a value and its derivative

Let the input move along $x(t)=x+t h$. Each intermediate value $v$ then has
a tangent $\dot v=dv/dt$ at $t=0$. For multiplication, the pair of rules is

$$
\begin{aligned}
c&=ab,\\
\dot c&=\dot a\,b+a\,\dot b.
\end{aligned}
$$

The tangent uses the same values $a,b$ as the original multiplication.
It describes an infinitesimal change along the chosen input direction;
there is no finite-difference step size. In the dual-number notation,
$(a+\dot a\epsilon)(b+\dot b\epsilon)$ has this tangent as the coefficient
of $\epsilon$, with $\epsilon^2=0$. The symbol $\epsilon$ here is an
algebraic device, not a small floating-point number.

The following manually expanded trace is the complete `Forward Mode AD`
example from Vijay Janapa Reddi and Harvard ML Systems contributors,
[ML Frameworks, forward mode](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L1977),
CC BY-NC-SA 4.0. Code and comments are unchanged; the math imports are added
so the example runs on its own.

```python
from math import sin, cos

def f(x):  # Computing both value and derivative
    # Step 1: x -> x²
    a = x * x  # Value: x²
    da = 2 * x  # Derivative: 2x

    # Step 2: x -> sin(x)
    b = sin(x)  # Value: sin(x)
    db = cos(x)  # Derivative: cos(x)

    # Step 3: Combine using product rule
    result = a * b  # Value: x² * sin(x)
    dresult = a * db + b * da  # Derivative: x²*cos(x) + sin(x)*2x

    return result, dresult
```

At $x=2$ with seed $\dot x=1$, the square has value 4 and tangent 4.
The sine has value $0.909297\ldots$ and tangent $-0.416147\ldots$.
The final multiplication combines both contributions:

$$
\begin{aligned}
f(2)&=4\sin 2\approx3.637190,\\
\dot f&=4\cos2+4\sin2\approx1.972602.
\end{aligned}
$$

The example hardcodes the unit seed in `da` and `db`. For an input
direction $h=3$, the output tangent is three times as large; the function
value is unchanged. PyTorch propagates the selected tangent through the
operations in [`torch.func.jvp`](https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html):

```python
import torch

ad_x = torch.tensor(2., dtype=torch.float64)
def square_sine(t):
    return t.square() * t.sin()

for direction in (1., 3., -2.):
    seed = torch.tensor(direction, dtype=ad_x.dtype)
    primal, tangent = torch.func.jvp(square_sine, (ad_x,), (seed,))
    manual_value, manual_derivative = f(2.)
    torch.testing.assert_close(primal, ad_x.new_tensor(manual_value))
    torch.testing.assert_close(tangent, ad_x.new_tensor(manual_derivative * direction))
```

### The same graph in forward and reverse mode

Baydin, Pearlmutter, Radul and Siskind use a two-input example in
[Automatic Differentiation in Machine Learning: a Survey, §§3.1–3.2](https://jmlr.org/papers/volume18/17-468/17-468.pdf#page=9):

$$
F(x_1,x_2)=\ln x_1+x_1x_2-\sin x_2.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/autodiff-2026-10-27/baydin-figure4-graph.png]]

*Baydin et al., Figure 4, p. 9, CC BY 4.0. Complete original graph and
English caption, rendered from the vector PDF. The node names are defined
by the two original trace tables below.*

At $(x_1,x_2)=(2,5)$, the primal values are $\ln2$, $10$ and $\sin5$.
Because $\sin5$ is negative, the final subtraction adds approximately
$0.958924$; the result is $11.652071\ldots$. Trigonometric arguments are
in radians. In the forward trace, seed $(1,0)$ changes $x_1$ while holding
$x_2$ fixed. The logarithm contributes $1/2$, the product contributes 5,
and the sine contributes zero. The output tangent is $5.5$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/autodiff-2026-10-27/baydin-table2-forward.png]]

*Baydin et al., Table 2, p. 10, CC BY 4.0. Complete original table and
English caption. Both columns run downward; the right column carries
tangents for seed $(1,0)$ alongside the primal values.*

For the reverse trace, start at the output with adjoint 1 and read the
right column upward. The subtraction sends 1 to $v_4$ and $-1$ to $v_3$.
The product contributes 5 to $\bar x_1$ and 2 to $\bar x_2$; the logarithm
adds $1/2$ to $\bar x_1$, and the sine path adds $-\cos5$ to $\bar x_2$.
Both input derivatives are obtained in this traversal:

$$
\nabla F(2,5)=\bigl(5.5,\ 2-\cos5\bigr).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/autodiff-2026-10-27/baydin-table3-reverse.png]]

*Baydin et al., Table 3, p. 13, CC BY 4.0. Complete original table and
English caption. The primal column runs down and the adjoint column runs
up; its $1.716$ is the rounded value of $2-\cos5$.*

Forward seed $(0,1)$ gives the partial derivative with respect to $x_2$
in another pass.
Seed $(2,-1)$ instead gives $2\,\partial F/\partial x_1-\partial F/\partial x_2$,
one directional derivative. A reverse seed $g=-3$ returns the gradient of
$-3F$. The identity $g(Jh)=(J^\top g)^\top h$ checks the two products against
each other without constructing a Jacobian:

```python
def baydin(t):
    return t[0].log() + t[0] * t[1] - t[1].sin()

ad_point = torch.tensor([2., 5.], dtype=torch.float64)
e1 = ad_point.new_tensor([1., 0.])
e2 = ad_point.new_tensor([0., 1.])
baydin_value, d1 = torch.func.jvp(baydin, (ad_point,), (e1,))
_, d2 = torch.func.jvp(baydin, (ad_point,), (e2,))
value_again, pullback = torch.func.vjp(baydin, ad_point)
full_gradient, = pullback(torch.ones_like(value_again))
expected_gradient = torch.stack((ad_point.new_tensor(5.5), 2 - ad_point[1].cos()))
torch.testing.assert_close(full_gradient, expected_gradient)
torch.testing.assert_close(torch.stack((d1, d2)), full_gradient)
torch.testing.assert_close(baydin_value, value_again)

h = ad_point.new_tensor([2., -1.])
g = ad_point.new_tensor(-3.)
_, jh = torch.func.jvp(baydin, (ad_point,), (h,))
jtg, = pullback(g)
torch.testing.assert_close(g * jh, jtg @ h)
```

For a general map with $N$ inputs and $M$ outputs, a single JVP yields
one direction through all outputs; a single VJP combines the outputs into
one derivative with respect to all inputs. Recovering a dense Jacobian
one basis direction at a time takes $N$ forward or $M$ reverse sweeps.
Batching directions, sparsity and implementation details can change the
cost; counting sweeps alone is not a runtime benchmark.

Forward mode carries tangents for live primal values. A program retaining
a large array still needs that array; carrying several directions needs
several tangents. Constant memory with respect to arbitrary program depth
does not follow from the definition. Nor is the tangent computation always
exactly twice the primal arithmetic: multiplication uses a product rule,
while other operations use different derivative formulas.

### A Hessian–vector product without the Hessian

The two modes can be composed. For
$\ell(a,b)=a^2b+\sin b$, the gradient is $(2ab,a^2+\cos b)$.
The Hessian is the Jacobian of this gradient: each entry differentiates
one gradient component with respect to one input. Here it is

$$
H_\ell(a,b)=\begin{pmatrix}2b&2a\\2a&-\sin b\end{pmatrix}.
$$

Applying a JVP to this gradient gives $H_\ell h$, a change of the gradient
along direction $h$. At $(a,b)=(2,3)$ and $h=(1,-2)$,

$$
\begin{aligned}
(H_\ell h)_1&=6\cdot1+4\cdot(-2)=-2,\\
(H_\ell h)_2&=4+2\sin3.
\end{aligned}
$$

```python
def curvature_loss(t):
    return t[0].square() * t[1] + t[1].sin()

curvature_point = torch.tensor([2., 3.], dtype=torch.float64)
curvature_direction = curvature_point.new_tensor([1., -2.])
gradient_value, hvp = torch.func.jvp(
    torch.func.grad(curvature_loss),
    (curvature_point,), (curvature_direction,),
)
expected_hvp = torch.stack((curvature_point.new_tensor(-2.),
                            4 + 2 * curvature_point[1].sin()))
torch.testing.assert_close(hvp, expected_hvp)
expected_gradient_value = torch.stack((curvature_point.new_tensor(12.),
                                       4 + curvature_point[1].cos()))
torch.testing.assert_close(gradient_value, expected_gradient_value)
```

This forward-over-reverse calculation stores the vectors needed by the
transforms, not a dense $N\times N$ Hessian. It does not make arbitrary
models constant-memory or guarantee that every operator has a forward-AD
implementation. The example runs on the stated PyTorch version; unsupported
operators require a supported decomposition or a different differentiation
strategy.

<!-- END REVIEW: forward mode 2026-10-27 -->

<!-- BEGIN REVIEW: JAX transforms 2026-10-27 -->
### Gradients as functions: grad, vmap and jit

A function can compute a derivative without accumulating it in a tensor's
`.grad`. In JAX, `grad(loss_fn)` returns a new callable: supplying parameters
and data to that callable returns the gradient with respect to the first
argument, `params`, by default. The loss must return a scalar; a pytree of
parameters can still contain many arrays. The parameters remain explicit
arguments. This makes it possible to apply another transformation
to the gradient function, for example to evaluate it for each example in a
batch.

The complete original `JAX Function Transformations` example below is from
Vijay Janapa Reddi and Harvard ML Systems contributors,
[ML Frameworks](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L4003),
CC BY-NC-SA 4.0. Code and English comments are unchanged. It defines the
functions; the following blocks supply data and inspect their results.

```python
import jax
import jax.numpy as jnp


def loss_fn(params, x, y):
    pred = jnp.dot(x, params["w"]) + params["b"]
    return jnp.mean((pred - y) ** 2)


# Transform: compute gradients
grad_fn = jax.grad(loss_fn)

# Transform: vectorize over batch dimension
batched_grad = jax.vmap(grad_fn, in_axes=(None, 0, 0))

# Transform: compile to XLA
fast_batched_grad = jax.jit(batched_grad)

# Compose all three: fast, batched gradient computation
```

Take $w=(1,-1)$ and $b=0.5$. The two inputs are $(1,2)$ and $(3,4)$, with
targets 1 and $-1$. Both predictions are $-0.5$. Their errors are $-1.5$ and
$0.5$, so the squared losses are 2.25 and 0.25. For a single example with
error $e=w^\top x+b-y$,

$$
\nabla_w\ell=2ex,\qquad \frac{\partial\ell}{\partial b}=2e.
$$

The first example gives weight gradient $(-3,-6)$ and bias gradient $-3$;
the second gives $(3,4)$ and 1. `vmap` retains both contributions.
In `in_axes=(None,0,0)`, `None` shares the same parameter dictionary across
examples, while the zeros map the leading axes of `x` and `y`. It does not
create two independently trainable copies of the parameters.

The dictionary is a *pytree*: a container whose numerical leaves here are
`w` and `b`. Differentiation preserves its keys. Vectorizing the gradient
adds an example axis to each gradient leaf, not to the input parameters:

| Result | Shape of `w` | Shape of `b` |
|---|---|---|
| Parameters | `(2,)` | `()` |
| One example's gradient | `(2,)` | `()` |
| Both gradients from `vmap` | `(2,2)` | `(2,)` |
| Gradient of the mean loss | `(2,)` | `()` |

```python
import numpy as np

jax.config.update("jax_enable_x64", True)
params = {"w": jnp.array([1., -1.]), "b": jnp.array(0.5)}
jax_x = jnp.array([[1., 2.], [3., 4.]])
jax_y = jnp.array([1., -1.])
per_example = batched_grad(params, jax_x, jax_y)
compiled_per_example = jax.block_until_ready(fast_batched_grad(params, jax_x, jax_y))
np.testing.assert_allclose(per_example["w"], [[-3., -6.], [3., 4.]])
np.testing.assert_allclose(per_example["b"], [-3., 1.])
for name in ("w", "b"):
    np.testing.assert_allclose(compiled_per_example[name], per_example[name])
```

The mean loss is $(2.25+0.25)/2=1.25$. Its weight gradient is the row mean
$(0,-1)$, and its bias gradient is $-1$. Calling `grad_fn` on the whole batch
computes that aggregate directly. Calling `batched_grad` returns the stack
from which the same aggregate can be computed. These results have different
shapes and serve different purposes.

An SGD step with learning rate 0.05 gives $w=(1,-0.95)$ and $b=0.55$.
The predictions become $-0.35$ and $-0.25$; their mean squared error is
$(1.35^2+0.75^2)/2=1.1925$. A functional update returns a new parameter tree:

```python
mean_gradient = grad_fn(params, jax_x, jax_y)
np.testing.assert_allclose(mean_gradient["w"], [0., -1.], atol=1e-14)
np.testing.assert_allclose(mean_gradient["b"], -1.)
for name in ("w", "b"):
    np.testing.assert_allclose(per_example[name].mean(axis=0), mean_gradient[name])

updated_params = jax.tree.map(lambda p, g: p - 0.05 * g, params, mean_gradient)
np.testing.assert_allclose(loss_fn(params, jax_x, jax_y), 1.25)
np.testing.assert_allclose(loss_fn(updated_params, jax_x, jax_y), 1.1925)
np.testing.assert_allclose(params["w"], [1., -1.])
np.testing.assert_allclose(params["b"], 0.5)
```

The equality between the mean of individual gradients and the gradient of
the mean uses this loss's decomposition into independent examples. A training
operation that couples examples, such as BatchNorm using current batch
statistics, cannot be replaced by independent single-example calls without
changing the function. Here `jit` compiles the already-vectorized gradient
function; it does not add the missing reduction or distribute it across
devices. The code checks compiled CPU results, not a speedup or a multi-GPU run.
The underlying transformation and tracing rules are explained in the
[JAX guide to grad and vmap](https://docs.jax.dev/en/latest/101/transformations.html).

#### A target shape can preserve the loss value but change its gradient

In this example, `pred` and `jax_y` both have shape `(2,)`. Changing the target
to `(2,1)` makes broadcasting produce a `(2,2)` error array: each prediction
is compared with both targets. At the chosen initial parameters the predictions
happen to coincide, so this wrong loss also equals 1.25. Its weight gradient,
however, is $(-2,-3)$ rather than $(0,-1)$:

```python
column_targets = jax_y[:, None]
np.testing.assert_allclose(loss_fn(params, jax_x, column_targets), 1.25)
wrong_gradient = grad_fn(params, jax_x, column_targets)
np.testing.assert_allclose(wrong_gradient["w"], [-2., -3.])
assert not np.allclose(wrong_gradient["w"], mean_gradient["w"])
```

The derivative is correct for the broadcast program. Matching one loss value
does not establish that the program represents the intended training objective.
The target shape is part of that objective's input contract.

#### The same transformation pattern in PyTorch

Function transforms are also available through
[`torch.func`](https://docs.pytorch.org/docs/2.14/func.html).
The following computes the same per-example and aggregate gradients from an
explicit parameter dictionary. `torch.func.grad` returns gradients as values;
these calls leave the original tensors' `.grad` fields empty.

```python
import torch

def torch_loss_fn(p, x, y):
    pred = x @ p["w"] + p["b"]
    return (pred - y).square().mean()

torch_params = {"w": torch.tensor([1., -1.], dtype=torch.float64),
                "b": torch.tensor(0.5, dtype=torch.float64)}
torch_x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
torch_y = torch.tensor([1., -1.], dtype=torch.float64)
torch_grad_fn = torch.func.grad(torch_loss_fn)
torch_per_example = torch.func.vmap(torch_grad_fn, in_dims=(None, 0, 0))(
    torch_params, torch_x, torch_y,
)
torch_mean_gradient = torch_grad_fn(torch_params, torch_x, torch_y)
for name in ("w", "b"):
    np.testing.assert_allclose(torch_per_example[name].detach().numpy(), per_example[name])
    np.testing.assert_allclose(torch_mean_gradient[name].detach().numpy(), mean_gradient[name], atol=1e-14)
    assert torch_params[name].grad is None
```

The numerical checks use JAX/JAXlib 0.12.0.1 and PyTorch 2.14.1 on CPU with
float64. They establish this shared mathematical result, not interchangeability
of arbitrary operators, random-number streams or compiler backends. The custom
`autograd.Function` interfaces later in the chapter show another boundary:
a backward rule does not automatically supply batching and forward-mode rules.

<!-- END REVIEW: JAX transforms 2026-10-27 -->

<span id="10-что-может-пойти-не-так"></span>

## 10. Ошибки и нестабильность обратного прохода

### Исчезающие градиенты

Для цепочки $h_{k+1}=0{,}5h_k$ производная после двадцати звеньев равна
$0{,}5^{20}\approx9{,}54\cdot10^{-7}$.
При единичной производной на конце до первого входа дойдёт именно это число.
В нелинейной сети вместо одного коэффициента появляются производные
активаций и матрицы весов. Насыщенная активация может уменьшить один из
множителей почти до нуля; в матричном случае разные направления
могут затухать с разной скоростью.

Поэтому ограниченность выхода активации сама по себе не означает
устойчивый градиент. Важны её производная в текущих точках, веса
и пути через граф. Подробные примеры насыщения есть в
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации|главе об активациях]].

### Взрывающиеся градиенты

В цепочке $h_{k+1}=2h_k$ за двадцать звеньев получается множитель
$2^{20}=1\,048\,576$. В многомерном случае большие нормы произведений
матриц Якоби могут усиливать некоторые направления.
При ограничении нормы градиент $g$ заменяют на

$$
g_{\text{clip}}=
g\min\left(1,\frac{c}{\lVert g\rVert_2}\right)
$$

для ненулевого $g$; нулевой оставляют нулевым.
Например, $(6,8)$ имеет норму $10$ и при пороге $c=5$
превращается в $(3,4)$. Направление сохраняется, длина ограничивается.
Это уменьшает возможный размер обновления SGD, но не исправляет
ошибку графа, переполнение или неподходящую инициализацию.

### NaN

Даже конечное значение потерь не гарантирует конечных производных.
В примере из документации PyTorch деление
$(1,1)/(0,1)$ даёт $(\infty,1)$.
Если после деления выбрать только второй элемент, потери равны единице,
но обратная формула для первого элемента всё ещё может вычислять
$0/0$ и вернуть NaN.

Безопасный вариант сначала выбирает допустимые элементы,
а затем делит: `(x[mask] / divisor[mask]).sum()`.
Недопустимое деление тогда вообще не попадает в граф.
Поиск первой нечисловой операции полезнее, чем замена всех NaN нулями;
`torch.autograd.detect_anomaly()` помогает связать ошибку обратного
прохода с операцией прямого.

### Неверный граф

`detach()` сохраняет численное значение, но обрывает путь производной.
При $x=2$ выражение $x\cdot x$ даёт производную $4$,
а `x * x.detach()` даёт $2$: второй множитель считается константой.
Значение обоих выражений равно четырём.

`torch.no_grad()` не записывает операции блока в обратный граф.
`model.eval()` решает другую задачу: переключает поведение слоёв
вроде Dropout и BatchNorm и не отключает autograd.
Замороженные веса с `requires_grad=False` тоже не обязательно обрывают
путь к входу. В $y=wx$ при фиксированном $w=3$ производная по
обучаемому $x$ по-прежнему равна трём.

Изменение сохранённого значения на месте может сделать обратный проход
неверным. Если для $y=x^2$ сохранён $x=2$, а до `backward` его заменить
на три, формула $2x$ уже не относится к выполненному прямому проходу.
PyTorch проверяет версии сохранённых тензоров и в таких случаях
выдаёт ошибку. Обходить проверку через `.data` не следует.

Эти случаи разобраны в разделах
[Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)
о сохранённых тензорах, делении на ноль, режимах градиентов и in-place
операциях.

## 11. Как проверить собственный обратный проход

Для небольшой гладкой функции сравнивают вычисленный градиент
с центральной разностью:

$$
\frac{\partial L}{\partial\theta_i}
\approx
\frac{L(\theta+\varepsilon e_i)-L(\theta-\varepsilon e_i)}
{2\varepsilon}.
$$

В примере $L=(a+b)(b+1)$ при $(a,b)=(2,1)$ ожидаются производные $(2,5)$.
Следующий код отдельно меняет каждую координату, а затем проверяет тот же
результат autograd.

```python
import torch

def scalar_loss(a, b):
    return (a + b) * (b + 1)

point = [2., 1.]
eps = 1e-6
numeric = []
for i in range(2):
    plus, minus = point.copy(), point.copy()
    plus[i] += eps
    minus[i] -= eps
    numeric.append((scalar_loss(*plus) - scalar_loss(*minus)) / (2 * eps))

a_check = torch.tensor(2., dtype=torch.float64, requires_grad=True)
b_check = torch.tensor(1., dtype=torch.float64, requires_grad=True)
scalar_loss(a_check, b_check).backward()
analytic = torch.stack((a_check.grad, b_check.grad))
torch.testing.assert_close(torch.tensor(numeric, dtype=torch.float64), analytic)
print(numeric, analytic.tolist())
```

Слишком большое $\varepsilon$ измеряет уже не локальное изменение,
а слишком маленькое усиливает ошибки округления при вычитании близких чисел.
Для новых операций удобно начинать с `float64` и нескольких разумных
приращений. Встроенная проверка `torch.autograd.gradcheck` сравнивает
производные численно и по обратной формуле; её стандартные допуски рассчитаны
на двойную точность.

Проверять нужно вдали от точек излома. Для ReLU в нуле центральная разность
равна $1/2$, а PyTorch выбирает производную $0$; это не доказывает ошибку
реализации. При проверке также фиксируют случайность и изменяемые состояния,
чтобы оба прямых прохода вычисляли одну функцию.
Если несколько индексов входа ссылаются на одну область памяти, например
после `expand`, перед проверкой нужен независимый клон: изменение одной
координаты иначе затрагивает несколько логических элементов.

<!-- BEGIN REVIEW: custom backward 2026-10-27 -->
### A custom backward returns an incoming derivative to each input

For $o=xy+z$, a later loss $L$ supplies $g=\partial L/\partial o$.
The custom operation must return $gy$, $gx$ and $g$, in the order of its
three inputs. It must not assume $g=1$. At $(x,y,z)=(2,3,1)$, the output
is 7. If $L=o$, the input derivatives are $(3,2,1)$; if $L=o^2/2$, then
$g=7$ and they are $(21,14,7)$.

The complete original `Custom Autograd Function` example follows, including
its usage and comments. Source: Vijay Janapa Reddi and Harvard ML Systems
contributors, [ML Frameworks](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L2324),
CC BY-NC-SA 4.0. Only the `torch` import is added.

```python
import torch

class MultiplyAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, z):
        # Save tensors needed for backward
        ctx.save_for_backward(x, y)
        return x * y + z

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, y = ctx.saved_tensors

        # Compute gradients using chain rule
        grad_x = grad_output * y  # ∂L/∂x = ∂L/∂out * ∂out/∂x
        grad_y = grad_output * x  # ∂L/∂y = ∂L/∂out * ∂out/∂y
        grad_z = grad_output  # ∂L/∂z = ∂L/∂out * 1

        return grad_x, grad_y, grad_z


# Usage
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
z = torch.tensor([1.0], requires_grad=True)

output = MultiplyAdd.apply(x, y, z)
output.backward()

print(
    x.grad, y.grad, z.grad
)  # tensor([3.]), tensor([2.]), tensor([1.])
```

`save_for_backward(x,y)` keeps the values required by the two product
derivatives. The numerical value of $z$ is unnecessary because
$\partial o/\partial z=1$. The engine calls `backward` with the derivative
from later operations and accumulates the returned contributions. It also
handles an input used twice: when the same tensor supplies $x$ and $y$,
both returned contributions reach that leaf.

```python
custom_inputs = tuple(torch.tensor([v], dtype=torch.float64, requires_grad=True)
                      for v in (2., 3., 1.))
custom_output = MultiplyAdd.apply(*custom_inputs)
(.5 * custom_output.square()).backward()
for tensor, expected in zip(custom_inputs, (21., 14., 7.)):
    torch.testing.assert_close(tensor.grad, tensor.new_tensor([expected]))

shared = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
offset = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
shared_output = MultiplyAdd.apply(shared, shared, offset)  # 2*2+1 = 5
shared_output.backward(shared.new_tensor([3.]))
torch.testing.assert_close(shared.grad, shared.new_tensor([12.]))
torch.testing.assert_close(offset.grad, offset.new_tensor([3.]))

check_inputs = tuple(t.detach().clone().requires_grad_() for t in custom_inputs)
assert torch.autograd.gradcheck(MultiplyAdd.apply, check_inputs)
assert torch.autograd.gradgradcheck(MultiplyAdd.apply, check_inputs)
```

The double-backward check succeeds here because the backward rule uses
differentiable tensor operations on the saved inputs. A correct first
derivative does not establish that property for every custom function.
For example, replacing both product-rule results with detached tensors
would preserve their numerical values in one backward pass but remove
the dependence needed to differentiate those results again.

An ordinary Python expression `x*y+z` already has autograd rules. The custom
class is useful here for exposing the contract; it is not needed to train
this expression. A custom kernel or an operation from another system needs
an appropriate differentiation rule and separate correctness checks.

### Adding a forward rule and torch.func support

The combined `forward(ctx,...)` interface above works with ordinary
autograd. PyTorch's [Extending torch.func](https://docs.pytorch.org/docs/2.14/notes/extending.func.html)
requires a separate `setup_context` for function transforms. A forward-mode
rule is another method, `jvp`: it receives one tangent per input and returns
one tangent per output. A `backward` method alone does not supply it.

The following editorial variant keeps the same operation and adds those
interfaces. The backward explicitly reverses broadcasting with
`sum_to_size`: an input reused across several output elements receives
the sum of their contributions. The JVP broadcasts the input tangents
in the same way as the primal multiplication and addition.

```python
class MultiplyAddAD(torch.autograd.Function):
    @staticmethod
    def forward(x, y, z):
        return x * y + z

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, z = inputs
        ctx.save_for_backward(x, y)
        ctx.save_for_forward(x, y)
        ctx.z_shape = z.shape

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = (grad_output * y).sum_to_size(x.shape)
        grad_y = (grad_output * x).sum_to_size(y.shape)
        grad_z = grad_output.sum_to_size(ctx.z_shape)
        return grad_x, grad_y, grad_z

    @staticmethod
    def jvp(ctx, dx, dy, dz):
        x, y = ctx.saved_tensors
        return dx * y + x * dy + dz
```

`save_for_backward` supplies the saved tensors to the reverse rule;
`save_for_forward` supplies them to the JVP. Here both rules need $x,y$,
while only the shape of $z$ is kept. At $(2,3,1)$ with input tangents
$(2,-1,0.5)$, the JVP is $2\cdot3+2\cdot(-1)+0.5=4.5$.
Reverse seed 7 instead gives $(21,14,7)$. These are different products
of the same local Jacobian, $[3,2,1]$.

```python
ad_inputs = tuple(torch.tensor([v], dtype=torch.float64) for v in (2., 3., 1.))
ad_tangents = tuple(torch.tensor([v], dtype=torch.float64) for v in (2., -1., .5))
ad_value, ad_tangent = torch.func.jvp(MultiplyAddAD.apply, ad_inputs, ad_tangents)
torch.testing.assert_close(ad_value, ad_value.new_tensor([7.]))
torch.testing.assert_close(ad_tangent, ad_tangent.new_tensor([4.5]))
_, custom_pullback = torch.func.vjp(MultiplyAddAD.apply, *ad_inputs)
for result, expected in zip(custom_pullback(ad_value.new_tensor([7.])), (21., 14., 7.)):
    torch.testing.assert_close(result, result.new_tensor([expected]))

ad_check = tuple(t.detach().clone().requires_grad_() for t in ad_inputs)
assert torch.autograd.gradcheck(MultiplyAddAD.apply, ad_check, check_forward_ad=True)
assert torch.autograd.gradgradcheck(MultiplyAddAD.apply, ad_check)

# The original class has not acquired these interfaces automatically.
try:
    torch.func.jvp(MultiplyAdd.apply, ad_inputs, ad_tangents)
except RuntimeError as error:
    assert "setup_context" in str(error)
else:
    raise AssertionError("Expected the original combined-forward interface to be rejected")
```

For a broadcasting example, let $x$ be a $2\times1$ column with entries
2 and 4, $y=(3,5,7)$ a $1\times3$ row, and $z=1$ a scalar.
The output is `[[7,11,15],[13,21,29]]`. With all output derivatives
equal to one, each row of $x$ receives $3+5+7=15$; each entry of $y$
receives $2+4=6$; $z$ receives six contributions. This is the same
broadcast-reduction rule derived earlier for a matrix-layer bias:

```python
broadcast_inputs = (
    torch.tensor([[2.], [4.]], dtype=torch.float64, requires_grad=True),
    torch.tensor([[3., 5., 7.]], dtype=torch.float64, requires_grad=True),
    torch.tensor(1., dtype=torch.float64, requires_grad=True),
)
broadcast_output = MultiplyAddAD.apply(*broadcast_inputs)
torch.testing.assert_close(broadcast_output, broadcast_output.new_tensor([[7.,11.,15.],[13.,21.,29.]]))
broadcast_output.sum().backward()
for tensor, value in zip(broadcast_inputs, (15., 6., 6.)):
    torch.testing.assert_close(tensor.grad, torch.full_like(tensor, value))
assert torch.autograd.gradcheck(MultiplyAddAD.apply, broadcast_inputs, check_forward_ad=True)
assert torch.autograd.gradgradcheck(MultiplyAddAD.apply, broadcast_inputs)
assert torch.autograd.gradcheck(MultiplyAdd.apply, broadcast_inputs)
assert torch.autograd.gradgradcheck(MultiplyAdd.apply, broadcast_inputs)
```

These checks establish the printed CPU cases, including first and second
reverse derivatives and direct JVPs. They do not establish a `vmap` rule,
support for every composition such as `jacfwd`, or CUDA-kernel correctness.
The examples use real tensors; a complex-valued operation needs the appropriate
conjugation rules and separate checks.
Both classes pass the shown broadcast reverse-gradient checks in PyTorch
2.14; the explicit reductions in `MultiplyAddAD` expose the sums derived above.

<!-- END REVIEW: custom backward 2026-10-27 -->

## Практический маршрут

В лекции Karpathy
[The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
вычисления проходят от численного приращения до обучения MLP.
Исходный [micrograd](https://github.com/karpathy/micrograd)
позволяет проследить `Value`, правила операций, топологический обход
и цикл обновления параметров без скрытого тензорного движка.

Для самостоятельной реализации есть
[[05 Источники/Courses/Harvard ML Systems/tinytorch/06_autograd|TinyTorch 06 — Autograd]].
Матричный вывод и исходные схемы также приведены в
[[05 Источники/Courses/Machine Learning Visualized/book/main.pdf|Machine Learning Visualized — Complete Book]].
Эти материалы дополняют разбор; прежде чем переходить к обучению большого MLP,
скалярный движок должен правильно обрабатывать общую ветвь и `x*x`.

<span id="краткие-итоги"></span>

## Источники

- [Stanford CS231n — Backpropagation](https://cs231n.github.io/optimization-2/):
  составные выражения, локальные производные, ветви, группировка операций
  и матричные градиенты. Три схемы выше взяты из оригинала, а числовые
  примеры матричного слоя и tanh-нейрона разобраны отдельно.
- [Andrej Karpathy — micrograd](https://github.com/karpathy/micrograd):
  исходные правила `Value` и обратный топологический обход,
  [закреплённая версия](https://github.com/karpathy/micrograd/tree/7bc720e951fe422b8f8814aa5aa1b64121d26b4c).
- [Chris Olah — Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/):
  разветвлённое выражение, сумма по путям и сравнение режимов дифференцирования.
- [PyTorch 2.14 — Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)
  и [torch.autograd.backward](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.backward.html):
  граф исполнения, сохранённые значения, накопление и входящая производная.
- [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 03 — Backpropagation and Neural Networks|Stanford CS224N — Backpropagation]].
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/super-cheatsheet-deep-learning.pdf|Stanford CS230 — Complete Deep Learning Cheatsheet]].

**Назад:** [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации|Функции активации]].
**Дальше:** [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|SGD, Momentum, Adam и schedules]].
