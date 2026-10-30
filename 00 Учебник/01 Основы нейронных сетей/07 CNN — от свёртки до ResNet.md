---
title: CNN — от свёртки до ResNet
type: textbook-chapter
status: canonical
last_updated: 2026-10-27
primary_sources:
  - https://d2l.ai/chapter_convolutional-neural-networks/index.html
  - https://cs231n.github.io/convolutional-networks/
  - https://d2l.ai/chapter_convolutional-modern/resnet.html
  - https://arxiv.org/abs/1512.03385
  - https://distill.pub/2019/computing-receptive-fields/
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
---

# CNN: от свёртки до ResNet

У изображения есть пространственная структура: соседние пиксели образуют края,
текстуры и части объектов. После разворачивания изображения в вектор обычный
полносвязный слой получает отдельный коэффициент для каждой позиции. Если
полезный край встретится в другом месте, те же коэффициенты к нему уже не
применятся.

Свёрточный слой использует небольшое окно и один набор весов во всех его
положениях. Например, фильтр $3\times3$ для RGB-изображения содержит 27 весов,
независимо от того, имеет изображение размер $32\times32$ или $224\times224$.
Он выдаёт карту откликов: в каждой позиции записан результат применения
фильтра к соответствующему окну. Несколько фильтров образуют несколько карт,
или каналов выхода. Все веса обучаются через ту же функцию потерь и обратное
распространение, что и в MLP.

## Двумерная cross-correlation

Возьмём пример из [D2L, Convolutions for Images](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html).
Вход содержит числа от 0 до 8, ядро — четыре коэффициента:

$$
X=\begin{bmatrix}0&1&2\\3&4&5\\6&7&8\end{bmatrix},
\qquad
K=\begin{bmatrix}0&1\\2&3\end{bmatrix}.
$$

В левом верхнем окне попарные произведения равны $0$, $1$, $6$ и $12$.
Их сумма — 19. Сдвиг окна вправо даёт $0+2+8+15=25$, вниз — 37,
в правый нижний угол — 43. Во всех четырёх положениях используются одни
и те же четыре веса.

![[00 Учебник/Assets/Figures/curated/source-first-7-14/d2l-cross-correlation.svg|Вычисление двумерной взаимной корреляции]]

*Выделенное окно входа и ядро дают первый элемент выхода 19. Полный исходный
рисунок Aston Zhang и соавторов, [D2L, The Cross-Correlation Operation](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#the-cross-correlation-operation),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Для одного входного и одного выходного канала, шага 1 и без дополнения краёв
операция записывается так:

$$
Y_{i,j}=b+\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}
X_{i+u,j+v}K_{u,v}.
$$

Здесь $b$ — одно обучаемое смещение, общее для всех положений этого фильтра.
У ядра $2\times2$ без дополнения есть четыре допустимых положения во входе
$3\times3$, поэтому выход имеет размер $2\times2$.

В [PyTorch Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)
эта операция названа свёрткой, хотя математически это взаимная корреляция
(cross-correlation). Для строгой свёртки ядро нужно перевернуть по обеим осям.
При обучаемом свободном ядре это различие можно учесть перенумерацией весов.
Но для уже заданного ядра переворот в общем случае изменяет результат.

Ниже сохранён поэлементный алгоритм D2L. Вместо заполнения буфера используется
`torch.stack`, чтобы результат сохранял тип, устройство и вычислительный граф
входных тензоров. Примеры главы выполняются по порядку на CPU.

```python
import torch
from torch import nn
from torch.nn import functional as F

def corr2d(X, K):
    h, w = K.shape
    rows = []
    for i in range(X.shape[0] - h + 1):
        row = []
        for j in range(X.shape[1] - w + 1):
            row.append((X[i:i+h, j:j+w] * K).sum())
        rows.append(torch.stack(row))
    return torch.stack(rows)

X = torch.arange(9, dtype=torch.float64).reshape(3, 3)
K = torch.tensor([[0., 1.], [2., 3.]], dtype=torch.float64)
Y = corr2d(X, K)
expected = torch.tensor([[19., 25.], [37., 43.]], dtype=X.dtype)
torch.testing.assert_close(Y, expected)
torch.testing.assert_close(F.conv2d(X[None, None], K[None, None])[0, 0], Y)

contrast = torch.tensor([[1., 0.], [0., -1.]], dtype=X.dtype)
torch.testing.assert_close(corr2d(X + 1, contrast), torch.full((2, 2), -4., dtype=X.dtype))
print(Y)
```

Последняя проверка использует другое ядро: оно вычитает правый нижний пиксель
окна из левого верхнего. На матрице от 1 до 9 все четыре разности равны $-4$.
Фильтр вычисляет один и тот же диагональный контраст в разных местах.

### Общие веса и обратный проход

Пусть $\Delta_{i,j}$ — производная потери по элементу выхода. Поскольку один
вес участвует во всех окнах, его производная складывается из вкладов всех
положений:

$$
\begin{aligned}
\frac{\partial L}{\partial K_{u,v}}
&=\sum_{i,j}\Delta_{i,j}X_{i+u,j+v},\\
\frac{\partial L}{\partial b}&=\sum_{i,j}\Delta_{i,j}.
\end{aligned}
$$

Для $L=\sum_{i,j}Y_{i,j}$ все $\Delta_{i,j}=1$. Производная по $K_{0,0}$
равна $0+1+3+4=8$, по остальным весам — 12, 20 и 24. Центральный входной
пиксель участвует сразу в четырёх окнах, поэтому получает сумму четырёх
соответствующих весов, $0+1+2+3=6$.

```python
x_grad = X.clone().requires_grad_()
k_grad = K.clone().requires_grad_()
corr2d(x_grad, k_grad).sum().backward()
torch.testing.assert_close(k_grad.grad, torch.tensor([[8., 12.], [20., 24.]], dtype=X.dtype))
torch.testing.assert_close(x_grad.grad, torch.tensor(
    [[0., 1., 1.], [2., 6., 4.], [2., 5., 3.]], dtype=X.dtype))
print(k_grad.grad, x_grad.grad)
```

### Обучение фильтра по желаемому отклику

В следующем примере D2L изображение состоит из единиц, но столбцы с индексами
2, 3, 4 и 5 заменены нулями; индексация начинается с нуля. Ядро $[1,-1]$
выдаёт 1 при переходе
с единицы на ноль, $-1$ при обратном переходе и 0 на постоянном участке.
Индексная запись здесь — $X_{i,j}-X_{i,j+1}$: сравниваются соседние столбцы.

Это ядро можно восстановить из пары «изображение — желаемая карта», не задавая
его веса заранее. Возьмём начальные веса $(a,b)=(0,0)$ и сумму квадратов
ошибок. Во всех шести строках есть по два ненулевых целевых отклика, поэтому
начальная потеря равна 12. Производные по двум весам равны $(-12,12)$.
При шаге 0,03 первые обновлённые веса — $(0{,}36,-0{,}36)$; потеря
снижается до $12(1-0{,}36)^2=4{,}9152$.

```python
edge_image = torch.ones(1, 1, 6, 8, dtype=torch.float64)
edge_image[:, :, :, 2:6] = 0
target_kernel = torch.tensor([[[[1., -1.]]]], dtype=torch.float64)
edge_target = F.conv2d(edge_image, target_kernel)

edge_filter = nn.Conv2d(1, 1, (1, 2), bias=False, dtype=torch.float64)
nn.init.zeros_(edge_filter.weight)
edge_optimizer = torch.optim.SGD(edge_filter.parameters(), lr=.03)
edge_losses = []
for step in range(40):
    edge_optimizer.zero_grad()
    edge_loss = (edge_filter(edge_image) - edge_target).square().sum()
    edge_losses.append(edge_loss.item())
    edge_loss.backward()
    if step == 0:
        torch.testing.assert_close(edge_filter.weight.grad, -12 * target_kernel)
    edge_optimizer.step()

assert edge_losses[0] == 12.
assert abs(edge_losses[1] - 4.9152) < 1e-12
torch.testing.assert_close(edge_filter.weight, target_kernel, atol=1e-6, rtol=0)
print(edge_losses[:2], edge_filter.weight.detach().flatten())
```

Здесь обучаются только два коэффициента на искусственной задаче. В сети
классификации нет готовой целевой карты для каждого слоя: производные проходят
к его фильтрам от ошибки классификатора. Поэтому фильтры обучаются совместно,
а не независимо под заранее назначенные названия вроде «детектор глаза».

## Входные и выходные каналы

Для цветного изображения один пространственный пиксель содержит три числа.
Фильтр $3\times3$ должен обработать все три канала: его форма —
$[3,3,3]$, где первая ось перечисляет каналы. Результаты по входным каналам
складываются, образуя одну выходную карту. Чтобы получить 16 выходных каналов,
нужны 16 разных таких фильтров.

[CS231n, Local Connectivity](https://cs231n.github.io/convolutional-networks/#conv)
различает пространственную локальность и полноту связей по каналам.
В обычной свёртке каждый выходной канал получает информацию от всех входных.
Ограничение по каналам появится ниже, в групповой свёртке.

![[00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-local-connectivity.jpeg|Пространственные оси и каналы карт признаков]]

*Слева изображение, далее объёмы промежуточных активаций, справа выходной
вектор. Height и width — пространственные оси, depth — число каналов,
а не число слоёв сети. Stanford CS231n,
[Convolutional Networks, Architecture Overview](https://cs231n.github.io/convolutional-networks/#overview).*

В примере [D2L, Multiple Input Channels](https://d2l.ai/chapter_convolutional-neural-networks/channels.html)
есть два входных канала. Первый даёт в левом верхнем окне 19, второй — 37.
Они складываются: элемент одного выходного канала равен 56.

![[00 Учебник/Assets/Figures/curated/cnn-2026-10-26/conv-multi-in.svg|Суммирование откликов двух входных каналов]]

*Два двумерных произведения и сумма по каналам дают один выход. Aston Zhang
и соавторы, [D2L, Multiple Input and Multiple Output Channels](https://d2l.ai/chapter_convolutional-neural-networks/channels.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

```python
multi_X = torch.stack([X, X + 1])
multi_K = torch.stack([K, K + 1])
single_map = sum(corr2d(x, k) for x, k in zip(multi_X, multi_K))
torch.testing.assert_close(single_map, torch.tensor(
    [[56., 72.], [104., 120.]], dtype=X.dtype))

all_kernels = torch.stack([multi_K, multi_K + 1, multi_K + 2])
multi_Y = torch.stack([
    sum(corr2d(x, k) for x, k in zip(multi_X, kernel))
    for kernel in all_kernels
])
torch.testing.assert_close(F.conv2d(multi_X[None], all_kernels)[0], multi_Y)
assert multi_Y.shape == (3, 2, 2)
print(multi_Y)
```

Порядок осей PyTorch с пакетом примеров:

```text
input   [B, C_in, H, W]
weight  [C_out, C_in, k_h, k_w]
bias    [C_out]
output  [B, C_out, H_out, W_out]
```

Без групп число весов равно $C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w$,
а со смещениями добавляется $C_{\mathrm{out}}$. Например,
`Conv2d(3,16,3)` содержит $16\cdot3\cdot9+16=448$ параметров.
Размер изображения меняет число применений этих параметров и объём активаций,
но не их количество. Одна карта признаков не обязана иметь самостоятельный
словесный смысл: обучению достаточно, чтобы совокупность каналов помогала
следующим слоям решать задачу.

### Свёртка 1×1

В окне $1\times1$ нет соседних пространственных позиций, но остаётся ось каналов.
Такой слой применяет одну и ту же матрицу к вектору каналов в каждом пикселе.
Для входа $(1,2,3)$ и строк весов $(1,0,-1)$, $(0,1,1)$ результат —
$(-2,5)$. Слой может менять число каналов, сохраняя высоту и ширину.

![[00 Учебник/Assets/Figures/curated/cnn-2026-10-26/conv-1x1.svg|Свёртка 1 на 1 как преобразование каналов]]

*В каждой позиции три входных значения преобразуются в два выходных.
Одна и та же матрица применяется во всех позициях. Aston Zhang и соавторы,
[D2L, 1×1 Convolutional Layer](https://d2l.ai/chapter_convolutional-neural-networks/channels.html#times-1-convolutional-layer),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

```python
point_weights = torch.tensor([[1., 0., -1.], [0., 1., 1.]], dtype=X.dtype)
point_value = torch.tensor([1., 2., 3.], dtype=X.dtype).reshape(1, 3, 1, 1)
torch.testing.assert_close(F.conv2d(point_value, point_weights[:, :, None, None]).flatten(),
                           torch.tensor([-2., 5.], dtype=X.dtype))

point_image = torch.arange(12, dtype=X.dtype).reshape(1, 3, 2, 2)
point_output = F.conv2d(point_image, point_weights[:, :, None, None])
matrix_output = (point_weights @ point_image.reshape(3, 4)).reshape(1, 2, 2, 2)
torch.testing.assert_close(point_output, matrix_output)
print(point_output)
```

Последовательность двух линейных преобразований каналов без промежуточной
нелинейности можно объединить в одну матрицу. ReLU между ними меняет семейство
реализуемых функций. Поэтому блок из нескольких свёрток $1\times1$ с
нелинейностями может выполнять больше, чем одна линейная проекция каналов.

## Padding, stride и размер выхода

Padding дополняет вход у границ, обычно нулями. Он позволяет вычислять окна,
которые без дополнения выходили бы за изображение. Stride задаёт шаг между
соседними положениями окна. Dilation задаёт расстояние между соседними
коэффициентами ядра при их наложении на вход: при $d=2$ ядро $3\times3$
использует девять точек в области $5\times5$, пропуская промежутки.

Для одной оси эффективный размер окна и размер выхода равны

$$
k_{\mathrm{eff}}=d(k-1)+1,
$$

$$
H_{\mathrm{out}}=
\left\lfloor\frac{H+2p-k_{\mathrm{eff}}}{s}\right\rfloor+1.
$$

Здесь $p$ — дополнение с каждой стороны оси, как в PyTorch. В D2L в некоторых
формулах указан суммарный padding обеих сторон; эти соглашения различаются
множителем 2. Для ширины применяется та же формула независимо.

Для $H=5$, $k=3$, $p=1$, $d=1$ шаг $s=1$ даёт пять выходных позиций;
шаг $s=2$ — три. Положения левого края окна в дополненном входе —
соответственно $(0,1,2,3,4)$ и $(0,2,4)$.

![[00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-stride-padding.jpeg|Шаг 1 и шаг 2 при дополнении границ]]

*Одинаковое ядро $[1,0,-1]$, вход длины 5 и padding 1. Слева stride 1,
справа stride 2. Stanford CS231n,
[Convolutional Networks, Spatial Arrangement](https://cs231n.github.io/convolutional-networks/#conv).*

Окна не обязаны делить длину входа без остатка. При $H=10$, $k=3$, $p=0$,
$s=2$ PyTorch выдаёт четыре позиции с началами $0,2,4,6$; последний элемент
входа не используется. Дробное значение до округления вниз не означает ошибку
API. Это уточняет обсуждение «неподходящих» шагов в старых заметках CS231n.

```python
def out_size(n, kernel, stride=1, padding=0, dilation=1):
    return (n + 2*padding - dilation*(kernel-1) - 1) // stride + 1

shape_input = torch.ones(1, 1, 10, 11, dtype=X.dtype)
shape_kernel = torch.ones(1, 1, 3, 3, dtype=X.dtype)
shape_output = F.conv2d(shape_input, shape_kernel, stride=2)
assert shape_output.shape == (1, 1, out_size(10, 3, 2), out_size(11, 3, 2))
assert shape_output.shape[-2:] == (4, 5)
dilated_output = F.conv2d(shape_input, shape_kernel, padding=2, dilation=2)
assert dilated_output.shape == shape_input.shape
print(shape_output.shape, dilated_output.shape)
```

Для нечётного ядра при $s=1$ дополнение $p=d(k-1)/2$ сохраняет размер.
При чётном эффективном окне для этого требуется несимметричное дополнение.
Строка `padding='same'` в PyTorch поддерживает только stride 1; в других
библиотеках соглашение same может включать округление и при большем шаге.

Дополнение не восстанавливает отсутствующие данные за границей. Нули, отражение
и повтор крайних значений задают разные предположения о продолжении изображения.
Общие веса обеспечивают согласованный сдвиг карты при сдвиге входа вдали от
границ, но нулевой padding и прореживание с шагом больше 1 ограничивают это
свойство. Свёртка сама по себе не делает классификатор инвариантным ко всем
переносам, поворотам и изменениям масштаба.

### Групповая и поканальная свёртки

У обычной свёртки каждый выход использует все входные каналы. При `groups=2`
входные и выходные каналы делятся на две группы, которые обрабатываются
независимо. Результаты соединяются по оси каналов. Числа входных и выходных
каналов должны делиться на число групп.

```python
group_input = torch.arange(200, dtype=X.dtype).reshape(2, 4, 5, 5) / 100
group_weight = torch.arange(108, dtype=X.dtype).reshape(6, 2, 3, 3) / 100
group_output = F.conv2d(group_input, group_weight, groups=2)
separate_output = torch.cat([
    F.conv2d(group_input[:, :2], group_weight[:3]),
    F.conv2d(group_input[:, 2:], group_weight[3:])
], dim=1)
torch.testing.assert_close(group_output, separate_output)
assert group_output.shape == (2, 6, 3, 3)
print(group_output.shape)
```

Форма весов теперь $[C_{\mathrm{out}},C_{\mathrm{in}}/g,k_h,k_w]$.
При $g=C_{\mathrm{in}}$ и одном выходе на входной канал каждый канал имеет
собственное пространственное ядро. Это depthwise convolution. Она не смешивает
каналы; последующая свёртка $1\times1$ может выполнить смешивание.

Без смещений переход от 3 каналов к 16 с ядром $3\times3$ требует 432 веса.
Композиция depthwise с одним фильтром на канал и $1\times1$ требует
$3\cdot9+3\cdot16=75$. Такая экономия ограничивает возможные ядра:
для каждого входного канала все выходы используют одно пространственное ядро,
умноженное на разные коэффициенты. Это не универсальная точная замена обычной
свёртки. [Документация Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)
задаёт также более общий depthwise multiplier, когда выходов на входной канал
несколько.

## Pooling

Max pooling выбирает наибольшее значение в окне; average pooling вычисляет
среднее. В отличие от свёртки, эти операции не содержат обучаемого ядра и
обрабатывают каналы независимо. Уменьшается число пространственных позиций,
но не число каналов.

В том же входе от 0 до 8 окно $2\times2$ с шагом 1 даёт для max pooling
матрицу $[[4,5],[7,8]]$, а для average pooling — $[[2,3],[5,6]]$.

![[00 Учебник/Assets/Figures/curated/cnn-2026-10-26/pooling.svg|Максимум в каждом окне 2 на 2]]

*Первый выход равен $\max(0,1,3,4)=4$. На рисунке шаг равен 1, окна
перекрываются. Aston Zhang и соавторы, [D2L, Maximum Pooling and Average Pooling](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html#maximum-pooling-and-average-pooling),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

```python
pool_input = X.clone().requires_grad_()
max_output = F.max_pool2d(pool_input[None, None], 2, stride=1)[0, 0]
avg_output = F.avg_pool2d(pool_input[None, None], 2, stride=1)[0, 0]
torch.testing.assert_close(max_output, torch.tensor([[4., 5.], [7., 8.]], dtype=X.dtype))
torch.testing.assert_close(avg_output, torch.tensor([[2., 3.], [5., 6.]], dtype=X.dtype))
max_gradient, = torch.autograd.grad(max_output.sum(), pool_input, retain_graph=True)
avg_gradient, = torch.autograd.grad(avg_output.sum(), pool_input)
torch.testing.assert_close(max_gradient, torch.tensor(
    [[0., 0., 0.], [0., 1., 1.], [0., 1., 1.]], dtype=X.dtype))
torch.testing.assert_close(avg_gradient, torch.tensor(
    [[.25, .5, .25], [.5, 1., .5], [.25, .5, .25]], dtype=X.dtype))
print(max_output, avg_output)
```

При единственном максимуме производная max pooling проходит к выбранному
элементу. У среднего она делится между всеми элементами окна; перекрывающиеся
окна складывают вклады. При равных максимумах функция недифференцируема,
и библиотека выбирает конкретное правило обратного прохода.

В `MaxPool2d(2)` шаг по умолчанию равен 2, а не 1, как на этом рисунке.
Для размера $32\times32$ это даёт $16\times16$: число пространственных
элементов уменьшается в четыре раза. Дополнение у max pooling в PyTorch
эквивалентно $-\infty$, а не нулю, который мог бы ошибочно победить
отрицательные значения. У average pooling нужно проверять, включаются ли
добавленные позиции в знаменатель.

Максимум может сохранить отклик при небольшом перемещении внутри окна.
Но если отклик пересечёт границу неперекрывающихся окон, выход изменится.
Например, $[0,1,0,0]$ при окне и шаге 2 даёт $[1,0]$, а сдвиг на один
элемент до $[0,0,1,0]$ даёт $[0,1]$. Общей гарантии инвариантности здесь нет.

Global average pooling усредняет каждую карту целиком. Из
$[B,C,H,W]$ получается $[B,C,1,1]$, после удаления единичных осей — $[B,C]$.
Классификатор после него не получает отдельных координат каждого пикселя.
Так можно заменить большую полносвязную часть, ценой потери явной
пространственной раскладки признаков.

<!-- group-equivariance 2026-10-26 -->
## Rotation-equivariant feature maps

A vertical-edge filter and its quarter-turn detect different orientations.
Rotating an image while keeping just one of those filters fixed will
generally change the response by more than a spatial rotation. Translation
weight sharing, introduced above, does not tie those orientations together.

The group-equivariance section of
[Harvard Network Architectures](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd#L1688)
extends this question to rotations and reflections. The original English
explanations and animations below come from
[escnn](https://github.com/QUVA-Lab/escnn/tree/9ad44cc37d694d9c805c4fef4d16722a254b3bf2),
by Gabriele Cesa, Leon Lang, Maurice Weiler and contributors. The discrete
worked calculation is a Bookvar example; it does not run that library.

### A transformation also acts on feature values

When a temperature image rotates, each temperature moves to a new position
but remains the same number. When an optical-flow field rotates, the
velocity vector at each position must rotate as well. Moving the arrows
without changing their directions would describe a different field.

![[00 Учебник/Assets/Figures/curated/group-equivariance-2026-10-26/escnn-feature-fields.png|Scalar and vector fields under rotation]]

*Cesa, Lang, Weiler and escnn contributors,
[feature-field explanation](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/README.md),
complete original PNG. Left: a scalar moves without changing its value.
Right: a vector moves and rotates. Copyright © 2021 Qualcomm Innovation
Center, Inc.; [BSD-3-Clause-Clear license](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/LICENSE).
No crop, relabeling or translation.*

<!-- source-excerpt: escnn-feature-fields -->
> The feature spaces of $\mathrm{E}(n)$-equivariant Steerable CNNs are defined as spaces of *feature fields*, being characterized by their transformation law under rotations and reflections.
> Typical examples are scalar fields (e.g. gray-scale images or temperature fields) or vector fields (e.g. optical flow or electromagnetic fields).
<!-- /source-excerpt -->

*Original escnn README, unchanged English. The full copyright notice and
license accompany the copied assets.*

For a network map $\Phi$, equivariance requires

$$
\Phi(T_g x)=T'_g\Phi(x).
$$

Here $T_g$ transforms the input, while $T'_g$ specifies the corresponding
transformation of the output representation. They need not perform the
same array operation. An invariant classifier has $T'_g=I$: the output
label does not change. A vector-valued intermediate representation needs
a nontrivial output action. This distinction is formalized in
[Cohen and Welling, §2](https://arxiv.org/html/1602.07576v3#S2).

### Four orientations from one learned filter

Consider only quarter-turns: 0°, 90°, 180° and 270°. Their composition adds
angles modulo 360°, forming the cyclic group $C_4$. Each quarter-turn maps
a square pixel lattice to itself, so it can be implemented by an exact
index permutation without interpolating pixel values.

Start with one learned kernel $K$. Apply its four rotated copies to the
same image and keep all four output maps. The result has an orientation
axis in addition to the spatial axes. If an input turns by 90°, its feature
maps turn spatially and cycle along that orientation axis. Keeping the
orientation indices fixed would compare the wrong detectors.

The following CPU example uses valid cross-correlation on square arrays,
stride 1, no padding and no bias. It is the first-layer lifting operation,
not a complete implementation of a multi-layer G-CNN.

```python
def lift_c4(image, base_kernel):
    return torch.stack([
        F.conv2d(image, torch.rot90(base_kernel, q, dims=(-2, -1)))
        for q in range(4)
    ], dim=2)

rotation_image = torch.arange(81, dtype=X.dtype).reshape(1, 1, 9, 9).square()
rotation_kernel = torch.tensor(
    [[[[1., 2., 0.], [-1., 0., 1.], [0., -2., 3.]]]], dtype=X.dtype)
orientation_maps = lift_c4(rotation_image, rotation_kernel)
assert orientation_maps.shape == (1, 1, 4, 7, 7)

turned_maps = lift_c4(torch.rot90(rotation_image, 1, (-2, -1)), rotation_kernel)
expected_maps = torch.rot90(orientation_maps, 1, (-2, -1)).roll(1, dims=2)
torch.testing.assert_close(turned_maps, expected_maps)
assert not torch.allclose(turned_maps, torch.rot90(orientation_maps, 1, (-2, -1)))

# ReLU commutes with this permutation of orientation and pixel indices.
torch.testing.assert_close(F.relu(turned_maps),
    torch.rot90(F.relu(orientation_maps), 1, (-2, -1)).roll(1, dims=2))
# Pooling over all orientations and spatial positions removes the pose.
torch.testing.assert_close(turned_maps.amax(dim=(2, 3, 4)),
                           orientation_maps.amax(dim=(2, 3, 4)))
print(orientation_maps.shape, (turned_maps - expected_maps).abs().max().item())
```

The four filters share nine independent kernel values rather than learning
36 unrelated values. Their output contains four times as many maps as a
single-filter planar convolution. Sharing parameters therefore does not
remove the extra activations or applications of rotated filters.

The one-step `roll(1)` follows the convention in this code: orientation
index $q$ uses the kernel turned by $q$ positive quarter-turns. Turning
the input changes that detector's response to the turned response of
detector $q-1$. A different orientation ordering needs a different index
permutation, not a different definition of equivariance.

Subsequent layers receive orientation-indexed maps. Their filters must
respect that transformation law too. With maps represented as functions
on a group, group correlation has the form

$$
[f\star\psi](g)=\sum_{h\in G}\sum_c f_c(h)\psi_c(g^{-1}h).
$$

For the discrete rotation-translation group $p4$, a pose combines an integer
translation and one of the four orientations. Here $g$ is the output pose,
$h$ ranges over input poses and $c$ is the input feature channel. The filter
uses their relative pose $g^{-1}h$. At the spatial origin, output orientation
1 and input orientation 3 have relative orientation $3-1=2$ modulo 4.
The first lifting layer sums over image positions; a later group layer
also sums over the four input orientations.

The argument $g^{-1}h$ ties the filter copies across poses. Replacing the
orientation axis with arbitrary unrelated channel weights would discard
that constraint. The construction and its equivariance proof are in
[Cohen and Welling, §§6–7](https://arxiv.org/html/1602.07576v3#S6).
This use of a symmetry group is different from `Conv2d(groups=...)` above,
which partitions channels into independent sets without specifying any
rotation action.

### Read the original steerable-CNN demonstration

![[00 Учебник/Assets/Figures/curated/group-equivariance-2026-10-26/escnn-vectorfield.gif|Rotation-equivariant scalar and vector responses]]

*Cesa, Lang, Weiler and escnn contributors,
[original demo](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/README.md#demo).
Complete original GIF, unchanged; BSD-3-Clause-Clear. This is the authors'
recorded demonstration, not a locally reproduced model run.*

<!-- source-excerpt: escnn-demo -->
> The visualization below demonstrates this claim by feeding rotated images into a randomly initialized $\mathrm{E}(2)$-steerable CNN (left).
> The middle plot shows the equivariant transformation of a feature space, consisting of one scalar field (color-coded) and one vector field (arrows), after a few layers.
> In the right plot we transform the feature space into a comoving reference frame by rotating the response fields back (stabilized view).
<!-- /source-excerpt -->

Read the right panel after the input has turned: undoing the input pose
also stabilizes the response. This is a test of the response's transformation
law, not proof that every intermediate feature stays unchanged in the
original coordinates. The source notes fluctuations from sampling and
interpolation on the pixel grid. Unlike the exact quarter-turn permutations
in the CPU example, arbitrary continuous rotations generally require
resampling a finite image.

![[00 Учебник/Assets/Figures/curated/group-equivariance-2026-10-26/escnn-conventional-cnn.gif|Conventional CNN response to the same rotating input]]

*The same authors and [source demo](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/README.md#demo),
complete original conventional-CNN comparison; BSD-3-Clause-Clear. The
response in the stabilized coordinates changes in this demonstration.*

The conventional network in the comparison has no imposed rotation
equivariance. Rotated training examples may help it learn a useful
orientation-insensitive classifier, but do not impose an exact equality
on every intermediate response. Conversely, rotation equivariance is
useful only when that symmetry matches the task. A pose-sensitive label
or a finite image's boundary conditions must be handled explicitly.

The full
[University of Amsterdam group-convolution tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html)
develops the discrete construction; its
[steerable-CNN companion](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html)
continues to typed scalar/vector fields.

<details>
<summary>Copyright and source license</summary>

>
> Copyright (c) 2021 Qualcomm Innovation Center, Inc.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted (subject to the limitations in the
> disclaimer below) provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright
> notice, this list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright
> notice, this list of conditions and the following disclaimer in the
> documentation and/or other materials provided with the distribution.
>
> * Neither the name of Qualcomm Innovation Center, Inc. nor the names of its
> contributors may be used to endorse or promote products derived from
> this software without specific prior written permission.
>
> NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
> GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
> HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
> WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
> MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
> BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
> OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
> IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

</details>
<!-- /group-equivariance -->

## Receptive field

Область восприятия (receptive field) выходного элемента содержит позиции входа,
которые могут повлиять на его значение. Одно ядро $3\times3$ использует
небольшое окно; два последовательных ядра с шагом 1 охватывают область
$5\times5$. Во втором случае выход зависит от нескольких перекрывающихся
окон первого слоя, а не только от трёх новых пикселей.

Для последовательности локальных операций обозначим через $r_l$ размер
теоретической области по одной оси, а через $j_l$ — расстояние между центрами
соседних выходов в координатах исходного входа. При $r_0=j_0=1$

$$
\begin{aligned}
j_l&=j_{l-1}s_l,\\
r_l&=r_{l-1}+d_l(k_l-1)j_{l-1}.
\end{aligned}
$$

Рекуррентный расчёт и границы применимости разобраны у Araujo, Norris и Sim,
[Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/).
Формулы применяются по высоте и ширине отдельно. Для входа $[3,32,32]$
получается такая последовательность; в свёртках dilation 1:

| Операция | Выход без B | $(j,r)$ |
|---|---|---|
| Вход | $3\times32\times32$ | $(1,1)$ |
| Conv 3, s=1, p=1 | $8\times32\times32$ | $(1,3)$ |
| Pool 2, s=2 | $8\times16\times16$ | $(2,4)$ |
| Conv 3, s=1, p=1 | $16\times16\times16$ | $(2,8)$ |

Последнее ядро охватывает три позиции с шагом 2 в предыдущей карте.
Оно добавляет $2\cdot2=4$ к уже доступному окну размера 4, получая 8.
Проверить геометрию можно обратным проходом. Возьмём положительные единичные
ядра и average pooling, чтобы все доступные пути давали ненулевой вклад.

```python
rf_input = torch.ones(1, 1, 32, 32, dtype=X.dtype, requires_grad=True)
rf_kernel = torch.ones(1, 1, 3, 3, dtype=X.dtype)
rf_output = F.conv2d(F.avg_pool2d(
    F.conv2d(rf_input, rf_kernel, padding=1), 2, stride=2),
    rf_kernel, padding=1)
rf_gradient, = torch.autograd.grad(rf_output[0, 0, 5, 5], rf_input)
positions = (rf_gradient[0, 0] != 0).nonzero()
assert positions.shape == (64, 2)
assert positions.amin(0).tolist() == [7, 7]
assert positions.amax(0).tolist() == [14, 14]
print(positions.amin(0), positions.amax(0))
```

Геометрическая область не равна распределению фактического градиента:
ReLU и max pooling выключают часть путей, а dilation может оставлять
промежутки внутри охватываемого прямоугольника. У границы часть номинальной
области приходится на padding, поэтому реальных пикселей в ней меньше.
Операции с глобальными
статистиками тоже требуют отдельного учёта: BatchNorm в обучающем режиме
использует значения во всём канале пакета. Приведённый расчёт относится
к последовательности локальных операций в примере, а не автоматически
к любому графу с нормализацией, ветвями и глобальным pooling.

## Свёртка как матричное умножение

В [CS231n, Implementation as Matrix Multiplication](https://cs231n.github.io/convolutional-networks/#conv)
каждое локальное окно разворачивается в столбец. Если есть
$N=H_{\mathrm{out}}W_{\mathrm{out}}$ положений, матрица окон имеет форму
$[C_{\mathrm{in}}k_hk_w,N]$. Матрица фильтров —
$[C_{\mathrm{out}},C_{\mathrm{in}}k_hk_w]$. Их произведение даёт все
отклики сразу; затем результат возвращается в пространственную форму.

```python
columns = F.unfold(multi_X[None], kernel_size=(2, 2))
flat_weights = all_kernels.reshape(3, -1)
matrix_maps = (flat_weights @ columns[0]).reshape(3, 2, 2)
torch.testing.assert_close(matrix_maps, multi_Y)
assert columns.numel() == 2 * 2 * 2 * 4
print(columns.shape, matrix_maps.shape)
```

<!-- BEGIN REVIEW: Harvard im2row 2026-10-27 -->
<a id="two-channels-window-rows-and-filter-vector"></a>

### Two channels, window rows and a filter vector

The Harvard diagram below uses a second example: the first input channel
contains 1 through 9, the second 10 through 18. A single output channel has
two $2\times2$ kernel slices, containing 1 through 4 and 5 through 8.
Stride and dilation are 1; there is no padding or bias. There are four output
positions, visited left to right and then top to bottom.

![[00 Учебник/Assets/Figures/curated/nn-native-2026-10-27/harvard-window-rows.svg|Two input channels flattened into four window rows and an eight-value filter vector]]

*Vijay Janapa Reddi and Harvard ML Systems contributors,
[Network Architectures, Core computational primitives](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd#L4042),
complete original archived SVG, unchanged;
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Each **row** of the $4\times8$ matrix contains both channels of one window;
the right-hand $8\times1$ vector contains both slices of one filter.
This explanatory caption corrects the source caption's reversed row/column
description. No labels or geometry in the figure have been changed.*

The upper red outline selects $[1,2,4,5]$ from the first channel.
The next four entries in that same matrix row, $[10,11,13,14]$, come from
the matching position in the second channel. The lower red outline instead
follows the second channel's top-right window into the **second** row.
It does not identify an additional output channel. The numbers and outlines
specify these correspondences without relying on the colors alone.

Call this window matrix $P$. Its first four columns form $P_0$ and the
remaining four form $P_1$:

$$
P_0=\begin{bmatrix}
1&2&4&5\\2&3&5&6\\4&5&7&8\\5&6&8&9
\end{bmatrix},
$$

$$
P_1=\begin{bmatrix}
10&11&13&14\\11&12&14&15\\13&14&16&17\\14&15&17&18
\end{bmatrix}.
$$

Flattening the filter in that same channel-first order gives
$k=[1,2,3,4,5,6,7,8]^\mathsf{T}$. The first dot product contains two
channel contributions:

$$
\begin{aligned}
y_{0,0}
&=(1\cdot1+2\cdot2+4\cdot3+5\cdot4)\\
&\quad +(10\cdot5+11\cdot6\\
&\qquad +13\cdot7+14\cdot8)\\
&=37+319=356.
\end{aligned}
$$

The other rows give $47+345=392$, $67+397=464$ and $77+423=500$.
Reshaping $Pk$ by the window order produces the map

$$
Y=\begin{bmatrix}356&392\\464&500\end{bmatrix}.
$$

The previous example and
[PyTorch Unfold](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Unfold.html)
place windows in columns. For this input, `F.unfold` returns $[1,8,4]$;
removing the batch axis and transposing gives Harvard's $P$ of shape $[4,8]$.
Thus `flat_weights @ columns` and `window_rows @ filter_vector` use the same
dot products. An implementation must keep the window, channel and kernel
orders consistent on both operands.

```python
harvard_input = torch.arange(1, 19, dtype=torch.float64).reshape(1, 2, 3, 3)
harvard_filter = torch.arange(1, 9, dtype=torch.float64).reshape(1, 2, 2, 2)
harvard_rows = torch.tensor([
    [1., 2., 4., 5., 10., 11., 13., 14.],
    [2., 3., 5., 6., 11., 12., 14., 15.],
    [4., 5., 7., 8., 13., 14., 16., 17.],
    [5., 6., 8., 9., 14., 15., 17., 18.]
], dtype=torch.float64)
harvard_columns = F.unfold(harvard_input, kernel_size=2)
assert harvard_columns.shape == (1, 8, 4)
torch.testing.assert_close(harvard_columns[0].T, harvard_rows)
harvard_values = harvard_rows @ harvard_filter.flatten()
torch.testing.assert_close(harvard_values,
                           torch.tensor([356., 392., 464., 500.], dtype=torch.float64))
torch.testing.assert_close(harvard_values.reshape(1, 1, 2, 2),
                           F.conv2d(harvard_input, harvard_filter))
print(harvard_rows.shape, harvard_values.reshape(2, 2))
```

There are 32 stored window entries instead of 18 input values. Each channel's
center appears in four rows; a corner appears in only one. This duplication
explains the extra storage of a fully materialized window matrix. The
identity above does not prescribe a convolution implementation or establish
a speedup: a kernel can generate tiles as needed instead of storing the
entire matrix. No timing measurement is made in this example.
<!-- END REVIEW: Harvard im2row 2026-10-27 -->

Число умножений и сложений с накоплением (MAC) для обычной свёртки без смещений:

$$
\mathrm{MAC}
=B\,H_{\mathrm{out}}W_{\mathrm{out}}
C_{\mathrm{out}}C_{\mathrm{in}}k_hk_w.
$$

Если умножение и сложение считают двумя FLOP, арифметический объём примерно
вдвое больше. Это соглашение нужно указывать при сравнении оценок.

Общие веса сокращают число параметров, но не число применений ядра.
Кроме того, явная матрица окон дублирует пиксели из перекрывающихся областей.
В нашем примере 18 входных чисел превращаются в 32 числа матрицы окон.
Поэтому тождество с матричным умножением само по себе не означает, что
материализация im2col — самый быстрый или экономный по памяти алгоритм.

<a id="lenet-5"></a>

## LeNet: полный проход изображения через сеть

После двух блоков «свёртка — нелинейность — pooling» карта разворачивается
в вектор и передаётся классификатору. В учебной реализации
[D2L, LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)
первый фильтр $5\times5$ с padding 2 сохраняет размер $28\times28$ и даёт
шесть каналов. Pooling уменьшает его до $14\times14$. Вторая свёртка без
padding даёт $16\times10\times10$, второй pooling — $16\times5\times5$.
После flatten на каждый пример приходится 400 чисел.

![[00 Учебник/Assets/Figures/curated/d2l/lenet.svg|Последовательность слоёв учебной LeNet]]

*Учебная LeNet из D2L: две свёртки, pooling и классификатор. Это адаптация
исторической сети, а не точное воспроизведение всех деталей LeNet-5 1998 года.
Aston Zhang и соавторы, [D2L, LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Ниже воспроизведена последовательность D2L с явно заданными входными размерами
вместо Lazy-модулей. Последний слой возвращает десять logits.
`cross_entropy` сама вычисляет log-softmax; дополнительная softmax перед
потерей не нужна. Для получения вероятностей при прогнозе можно применить
softmax к logits отдельно. D2L заменяет этим способом исторический
гауссовский выходной слой.

```python
torch.manual_seed(26)
lenet = nn.Sequential(
    nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, 5), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2), nn.Flatten(),
    nn.Linear(400, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
).double()
for layer in lenet:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
lenet_input = torch.randn(4, 1, 28, 28, dtype=torch.float64)
with torch.no_grad():
    state = lenet_input
    lenet_shapes = []
    for layer in lenet:
        state = layer(state)
        lenet_shapes.append(tuple(state.shape))
assert lenet_shapes[5] == (4, 16, 5, 5)
assert lenet_shapes[6] == (4, 400) and lenet_shapes[-1] == (4, 10)
assert sum(p.numel() for p in lenet.parameters()) == 61706

lenet_optimizer = torch.optim.SGD(lenet.parameters(), lr=.1)
before_first_filter = lenet[0].weight.detach().clone()
lenet_optimizer.zero_grad()
classification_loss = F.cross_entropy(lenet(lenet_input), torch.tensor([0, 1, 2, 3]))
classification_loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lenet.parameters())
lenet_optimizer.step()
assert not torch.equal(before_first_filter, lenet[0].weight)
print(lenet_shapes, classification_loss.item())
```

Метки в этом запуске назначены четырём случайным входам. Проверка показывает
передачу форм, вычисление потери и обновление первого фильтра, но не качество
распознавания и не результат обучения на MNIST. Проверочную точность получают
после обучения на настоящих данных с отдельной проверочной выборкой.

## AlexNet, VGG и переход к глубине

В AlexNet, победившей в ImageNet 2012, пять свёрточных слоёв соединены с тремя
полносвязными. Вместо sigmoid используются ReLU, в большой полносвязной части —
dropout, а при обучении — преобразования изображений. Сеть обучалась на GPU.
[D2L, AlexNet](https://d2l.ai/chapter_convolutional-modern/alexnet.html)
сопоставляет этот переход с LeNet и отдельно описывает данные, нелинейности
и архитектуру.

Большая полносвязная часть может содержать основную долю параметров.
Например, соединение карты $[256,6,6]$ со слоем ширины 4096 требует
$9216\cdot4096=37\,748\,736$ весов, не считая смещений.
Это около 144 MiB только для весов float32; градиенты, состояния оптимизатора
и активации занимают дополнительную память.

VGG группирует несколько свёрток $3\times3$ с ReLU и padding 1 перед одним
pooling. Пространственное разрешение сохраняется внутри группы и уменьшается
на её выходе. Это позволяет добавлять нелинейные слои, не уменьшая карту после
каждой свёртки.

![[00 Учебник/Assets/Figures/curated/cnn-2026-10-26/vgg.svg|Сравнение AlexNet и повторяемых блоков VGG]]

*Слева отдельные слои учебной AlexNet, справа повторяемые блоки VGG. Обозначение
повтора относится к группе операций, а не к одному общему набору весов.
Aston Zhang и соавторы, [D2L, Networks Using Blocks](https://d2l.ai/chapter_convolutional-modern/vgg.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

В левой части исходного рисунка последняя свёртка AlexNet подписана как
384 канала, тогда как код AlexNet в D2L задаёт для неё 256. Для расчёта
полносвязной части выше использована форма из кода; рисунок сохранён без
изменения авторских подписей.

При постоянном числе каналов $C$ два слоя $3\times3$ содержат $18C^2$ весов
против $25C^2$ у одного слоя $5\times5$, без смещений.
Теоретическая область восприятия в обоих случаях — $5\times5$.
Но два слоя с промежуточной ReLU не эквивалентны одному линейному фильтру:
их ответ зависит от того, какие промежуточные значения прошли через ReLU.

Функция ниже — исходный PyTorch-блок D2L; английский код сохранён.
Он создаёт новые параметры для каждого слоя, а не применяет один слой повторно.

```python
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

small_vgg_block = vgg_block(2, 8).double()
assert small_vgg_block(torch.ones(1, 3, 32, 32, dtype=X.dtype)).shape == (1, 8, 16, 16)
assert sum(p.numel() for p in small_vgg_block.parameters()) == 808
print(sum(p.numel() for p in small_vgg_block.parameters()))
```

### Почему глубина могла ухудшать обучение

В экспериментах He и соавторов более глубокие обычные сети давали более высокую
ошибку уже на обучающей выборке. Это наблюдение нельзя объяснить только
переобучением, при котором обучение успешно, а перенос на новые данные плох.

Если дополнительные слои реализуют тождественное преобразование, более глубокая
сеть может повторить решение мелкой. Существование таких весов, однако, не
обеспечивает их нахождение оптимизатором. ResNet меняет параметризацию блока:
обучаемая ветвь предсказывает поправку к входу. Постановка и результаты
сравнения приведены в [Deep Residual Learning, §§1 и 3](https://arxiv.org/abs/1512.03385).

## ResNet

Пусть желаемое преобразование блока — $H(x)$. При одинаковых формах входа и
выхода вместо прямого приближения $H$ можно обучать
$F(x)=H(x)-x$ и складывать результат со входом:

$$
z=x+F(x).
$$

Для $H(x)=x$ ветвь должна давать ноль. Она не обязана сама обучиться копировать
все координаты входа. Это мотивирует residual-параметризацию, но не доказывает
универсального преимущества для любого набора данных и режима обучения.

В basic block исходной post-activation ResNet за сложением следует ReLU:
$y=\operatorname{ReLU}(z)$. Ветвь имеет порядок
`Conv 3×3 → BatchNorm → ReLU → Conv 3×3 → BatchNorm`.
Между второй BatchNorm и сложением ReLU нет.

![[00 Учебник/Assets/Figures/curated/initialization-normalization-2026-10-21/d2l-resnet-block.svg|Residual-блок без проекции и с проекцией]]

*Слева shortcut сохраняет вход, справа свёртка $1\times1$ согласует его форму
с основной ветвью. ReLU применяется после сложения. Aston Zhang и соавторы,
[D2L, Residual Blocks](https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

### Формы на двух ветвях

При входе $[B,64,56,56]$ две свёртки с 64 выходными каналами, шагом 1 и
padding 1 сохраняют форму. Shortcut передаёт вход без изменения.
Складываются два тензора $[B,64,56,56]$.

В первом блоке следующей группы первая свёртка имеет 128 выходных каналов
и шаг 2: основная ветвь выдаёт $[B,128,28,28]$. Прежний вход прибавить к нему
нельзя. В выбранной реализации D2L shortcut использует свёртку $1\times1$
с 128 каналами и шагом 2. Она даёт ту же форму, но уже имеет обучаемые веса.
Это вариант проекции; не все варианты исторической ResNet используют именно
такое согласование.

Ниже сохранён исходный класс D2L, включая Lazy-модули и английскую docstring.
Его shortcut не содержит дополнительной BatchNorm. Сравнивая с другой
библиотекой, нужно проверять эту деталь, bias и место шага в блоке.

```python
class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

same_block = Residual(64).double().eval()
projected_block = Residual(128, use_1x1conv=True, strides=2).double().eval()
block_input = torch.randn(1, 64, 56, 56, dtype=X.dtype)
with torch.no_grad():
    assert same_block(block_input).shape == (1, 64, 56, 56)
    assert projected_block(block_input).shape == (1, 128, 28, 28)
print(same_block.conv3, projected_block.conv3.weight.shape)
```

### Нулевой residual и градиент

До последней ReLU якобиан блока равен $I+J_F(x)$. Слагаемое $I$ даёт
отдельный путь производной, но не гарантирует ненулевой итог: например,
для $F(x)=-x$ сумма равна нулю. После ReLU добавляется её маска.
Поэтому утверждение «в residual-блоке градиент всегда проходит без изменений»
неверно.

Даже при $F=0$ весь post-activation блок возвращает $\operatorname{ReLU}(x)$,
а не $x$ для произвольных отрицательных координат. Проверим это настоящим
классом: обнулим коэффициенты и смещения последней BatchNorm ветви.
Нормализация переведена в eval, чтобы использовать фиксированное состояние.

```python
zero_block = Residual(3).double().eval()
zero_input = torch.linspace(-1, 1, 108, dtype=X.dtype).reshape(1, 3, 6, 6).requires_grad_()
zero_block(zero_input)  # Initialize lazy parameters.
with torch.no_grad():
    zero_block.bn2.weight.zero_()
    zero_block.bn2.bias.zero_()
zero_output = zero_block(zero_input)
torch.testing.assert_close(zero_output, F.relu(zero_input))
zero_gradient, = torch.autograd.grad(zero_output.sum(), zero_input)
torch.testing.assert_close(zero_gradient, (zero_input > 0).to(X.dtype))
print(zero_gradient.unique())
```

Pre-activation ResNet меняет порядок нормализации и нелинейностей.
В более глубоких ResNet используются также bottleneck-блоки
$1\times1 \to 3\times3 \to 1\times1$: сначала уменьшается число каналов для
пространственной свёртки, затем восстанавливается. Эти конструкции нельзя
считать тем же basic block только потому, что у них есть shortcut.

### Сборка ResNet-18

У ResNet-18 четыре группы по два basic block. Первый слой — свёртка
$7\times7$ с шагом 2; затем max pooling с шагом 2.
В следующих группах число каналов растёт от 64 до 512, а разрешение
уменьшается. В конце global average pooling даёт по одному числу на канал,
и полносвязный слой выдаёт logits классов.

![[00 Учебник/Assets/Figures/curated/d2l/resnet18.svg|Повторение блоков в ResNet-18]]

*Общая структура ResNet-18 из D2L. Два блока первой группы показаны отдельно;
переход к новой группе с проекцией и следующий блок повторяются для трёх
групп с растущим числом каналов. Aston Zhang и соавторы,
[D2L, ResNet Model](https://d2l.ai/chapter_convolutional-modern/resnet.html#resnet-model),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Код собирает те же группы с классом D2L. Здесь явно выбраны RGB-вход
$224\times224$ и 1000 выходов, а не параметры задачи Fashion-MNIST из курса.
Число 18 считает один начальный слой, 16 свёрток внутри восьми блоков
и классификатор; дополнительные shortcut-проекции не включены в это название.

```python
def make_resnet18(num_classes=1000):
    stages = [nn.Sequential(
        nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]
    for stage_index, channels in enumerate([64, 128, 256, 512]):
        blocks = []
        for block_index in range(2):
            transition = stage_index > 0 and block_index == 0
            blocks.append(Residual(channels, use_1x1conv=transition,
                                   strides=2 if transition else 1))
        stages.append(nn.Sequential(*blocks))
    stages.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(), nn.LazyLinear(num_classes)))
    return nn.Sequential(*stages)

resnet18 = make_resnet18().eval()
resnet_shapes = []
with torch.no_grad():
    state = torch.zeros(1, 3, 224, 224)
    for stage in resnet18:
        state = stage(state)
        resnet_shapes.append(tuple(state.shape))
assert resnet_shapes == [
    (1, 64, 56, 56), (1, 64, 56, 56), (1, 128, 28, 28),
    (1, 256, 14, 14), (1, 512, 7, 7), (1, 1000)]
print(resnet_shapes)
```

Проверка выполняет один прямой проход случайно инициализированной сети.
Она не воспроизводит обучение ImageNet и не измеряет производительность.

<!-- BEGIN REVIEW: Winograd 2026-10-27 -->
## Winograd: fewer multiplications for overlapping outputs

Consider two adjacent outputs of a length-three filter:

$$
\begin{aligned}
y_0&=d_0g_0+d_1g_1+d_2g_2,\\
y_1&=d_1g_0+d_2g_1+d_3g_2.
\end{aligned}
$$

Evaluating the dot products separately requires six products between input
values and filter values. The windows overlap, but the repeated input
values multiply different filter coefficients. Reusing a loaded input value
does not by itself eliminate those products. Winograd minimal filtering
instead transforms the input and filter, multiplies their transformed
values, and combines the products to recover both outputs.

The notation $F(m,r)$ means $m$ consecutive outputs from a filter of length
$r$. It consumes $t=m+r-1$ input values. Here $m=2$, $r=3$ and $t=4$.
As throughout this chapter, the CNN operation is valid cross-correlation:
the filter coefficients are not reversed.

### Four products recover two outputs

Andrew Lavin's [wincnn, Example: F(2,3)](https://github.com/andravin/wincnn/blob/c2bf6305cc75866b19e204628b95baed4d13065c/README.md#example-f23)
constructs the following transforms:

$$
B^\mathsf{T}=\begin{bmatrix}
1&0&-1&0\\0&1&1&0\\0&-1&1&0\\0&-1&0&1
\end{bmatrix},
$$

$$
G=\begin{bmatrix}
1&0&0\\\frac12&\frac12&\frac12\\
\frac12&-\frac12&\frac12\\0&0&1
\end{bmatrix},
$$

$$
A^\mathsf{T}=\begin{bmatrix}
1&1&1&0\\0&1&-1&1
\end{bmatrix}.
$$

The input transform forms four sums or differences, $v=B^\mathsf{T}d$.
The filter transform forms four values, $u=Gg$. Only the next stage pairs
input-dependent and filter-dependent values: $z=u\odot v$, where $\odot$
is element-wise multiplication. Finally, $y=A^\mathsf{T}z$ adds or subtracts
those four products. Multiplication by the fixed coefficient $1/2$ in $G$
is part of the transform, not one of the four variable products being counted.

For $d=[1,2,3,4]^\mathsf{T}$ and $g=[2,-1,3]^\mathsf{T}$, the intermediate
values are

$$
\begin{aligned}
v&=[-2,5,1,2]^\mathsf{T},\\
u&=[2,2,3,3]^\mathsf{T},\\
z&=[-4,10,3,6]^\mathsf{T}.
\end{aligned}
$$

The output transform gives $y_0=-4+10+3=9$ and $y_1=10-3+6=13$.
Direct evaluation gives the same results:
$1\cdot2+2(-1)+3\cdot3=9$ and $2\cdot2+3(-1)+4\cdot3=13$.

This agreement holds for arbitrary values, not only these numbers. In the
first output, $z_1+z_2=d_1g_1+d_2(g_0+g_2)$. Adding
$z_0=(d_0-d_2)g_0$ cancels $d_2g_0$. In the second output,
$z_1-z_2=d_1(g_0+g_2)+d_2g_1$; adding $z_3=(d_3-d_1)g_2$ cancels
$d_1g_2$. Both original dot products are recovered.

```python
win_BT = torch.tensor([
    [1., 0., -1., 0.], [0., 1., 1., 0.],
    [0., -1., 1., 0.], [0., -1., 0., 1.]
], dtype=torch.float64)
win_G = torch.tensor([
    [1., 0., 0.], [.5, .5, .5], [.5, -.5, .5], [0., 0., 1.]
], dtype=torch.float64)
win_AT = torch.tensor([[1., 1., 1., 0.], [0., 1., -1., 1.]], dtype=torch.float64)
win_d = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
win_g = torch.tensor([2., -1., 3.], dtype=torch.float64)
win_v, win_u = win_BT @ win_d, win_G @ win_g
win_z = win_u * win_v
torch.testing.assert_close(win_v, torch.tensor([-2., 5., 1., 2.], dtype=win_d.dtype))
torch.testing.assert_close(win_u, torch.tensor([2., 2., 3., 3.], dtype=win_d.dtype))
torch.testing.assert_close(win_z, torch.tensor([-4., 10., 3., 6.], dtype=win_d.dtype))
torch.testing.assert_close(win_AT @ win_z,
    F.conv1d(win_d[None, None], win_g[None, None]).flatten())
print(win_AT @ win_z)  # tensor([9., 13.], dtype=torch.float64)
```

The matrices above retain the signs used by the pinned wincnn generator.
Other valid presentations negate the last row of $B^\mathsf{T}$ together
with the last column of $A^\mathsf{T}$. Either pair works; mixing one
matrix from each convention does not. The author's symbolic verification
and the other tile sizes remain available in the
[original Python module](/examples/wincnn-c2bf6305.py), unchanged,
under its [Apache-2.0 license](/licenses/wincnn-c2bf6305-Apache-2.0.txt).

<!-- source-excerpt: wincnn-f23-verification -->
> The last matrix is the 1D convolution F(2,3) computed using the
> transforms AT, G, and BT, on 4 element signal d[0..3] and 3 element
> filter g[0..2], and serves to verify the correctness of the
> transforms. This is a symbolic computation, so the result should be
> exact.
<!-- /source-excerpt -->

*Andrew Lavin, wincnn README, unchanged English; Apache-2.0. The calculation
above uses the same transforms with explicit numerical inputs.*

### A two-dimensional tile

Applying the transforms along both axes gives $F(2\times2,3\times3)$.
A $4\times4$ input tile $D$ and a $3\times3$ filter $g$ produce a
$2\times2$ output. The four stages are

$$
U=GgG^\mathsf{T},\qquad V=B^\mathsf{T}DB,
$$

$$
M=U\odot V,\qquad Y=A^\mathsf{T}MA.
$$

Both $U$ and $V$ have shape $4\times4$, so their element-wise product
requires 16 variable multiplications. Directly evaluating four outputs
with nine coefficients each requires 36. Thus the variable-product count
falls by $36/16=2.25$, or from nine to four per output. Neither count
includes all transform arithmetic, and 2.25 is not a measured speedup.

For a complete tile calculation, take

$$
D=\begin{bmatrix}
1&2&3&4\\5&6&7&8\\9&10&11&12\\13&14&15&16
\end{bmatrix},
$$

$$
g=\begin{bmatrix}1&0&-1\\2&1&0\\-1&0&1\end{bmatrix}.
$$

The transformed input and element-wise product are

$$
V=\begin{bmatrix}
0&-16&0&0\\-4&34&2&4\\0&8&0&0\\0&16&0&0
\end{bmatrix},
$$

$$
M=\begin{bmatrix}
0&0&0&0\\-4&25.5&0.5&0\\0&-6&0&0\\0&0&0&0
\end{bmatrix}.
$$

For example, $U_{1,1}=0.75$, so $M_{1,1}=0.75\cdot34=25.5$.
The first output sums the upper-left $3\times3$ part of $M$:
$-4+25.5+0.5-6=16$. The other three entries give
$Y=[[16,19],[28,31]]$. The code retains all four stages for inspection:

```python
win_D = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4)
win_kernel = torch.tensor([[1., 0., -1.], [2., 1., 0.], [-1., 0., 1.]],
                          dtype=torch.float64)
win_U = win_G @ win_kernel @ win_G.T
win_V = win_BT @ win_D @ win_BT.T
win_M = win_U * win_V
win_Y = win_AT @ win_M @ win_AT.T
torch.testing.assert_close(win_V, torch.tensor([
    [0., -16., 0., 0.], [-4., 34., 2., 4.],
    [0., 8., 0., 0.], [0., 16., 0., 0.]], dtype=win_D.dtype))
torch.testing.assert_close(win_M, torch.tensor([
    [0., 0., 0., 0.], [-4., 25.5, .5, 0.],
    [0., -6., 0., 0.], [0., 0., 0., 0.]], dtype=win_D.dtype))
torch.testing.assert_close(win_Y, torch.tensor([[16., 19.], [28., 31.]], dtype=win_D.dtype))
torch.testing.assert_close(win_Y,
    F.conv2d(win_D[None, None], win_kernel[None, None])[0, 0])
print(win_U, win_V, win_M, win_Y, sep="\n")
```

Some entries happen to vanish for these inputs. The generic algorithm still
has 16 product positions; the 36-to-16 comparison does not assume sparse
inputs or skip products that become zero in this example.

### Channels become sixteen matrix multiplications

![[00 Учебник/Assets/Figures/curated/winograd-2026-10-27/dolz-workflow.png|Winograd filter and input transforms, batched GEMM, and output inverse transform]]

*“Workflow for the Winograd algorithm.” Manuel F. Dolz, Héctor Martínez,
Adrián Castelló, Pedro Alonso-Jordá and Enrique S. Quintana-Ortí,
[Figure 1](https://link.springer.com/article/10.1007/s11227-023-05088-4/figures/1),
2023, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Complete original figure rendered from the vector PDF; page text and the
caption are outside the crop. Labels are unchanged. Here $t=4$, $m=2$ and
$r=3$. The $k$ and $c$ axes count output and input channels.*

In step 1, each of the $KC$ channel-specific filters is transformed.
In step 2, each input channel
is split into overlapping $4\times4$ tiles and transformed. Corresponding
coordinates from those tiles are then collected into matrices. For one
coordinate $(a,b)$ of the transformed $4\times4$ tile, the filter matrix
has shape $[K,C]$ and the input matrix has shape $[C,P]$, where $P$ counts
tiles across all images. Their product has shape $[K,P]$. The sixteen
coordinates therefore produce sixteen independent matrix multiplications.
This is the Batched GEMM in step 3. Its batch axis indexes transform
coordinates; the model's image batch is included in $P$.

Each dot product sums the input-channel contributions before the output
transform. This is valid because that transform is linear:

$$
M_{k,p}=\sum_{c=0}^{C-1}U_{k,c}\odot V_{c,p},
$$

$$
Y_{k,p}=A^\mathsf{T}M_{k,p}A.
$$

Step 4 gathers sixteen coordinates for one output channel and one tile,
applies the output transform, and writes four neighboring output pixels.
The source develops this arrangement in
[Dolz et al., §2](https://link.springer.com/article/10.1007/s11227-023-05088-4#Sec2).

Tile counts must follow the output geometry. With valid stride-one
convolution, $H_o=H-2$ and $W_o=W-2$. The required count per image is
$\lceil H_o/2\rceil\lceil W_o/2\rceil$. The original figure instead
labels its tile axis using the input sizes $h_i,w_i$; those labels are
retained but are not the count used in the valid-convolution example below.
An input $7\times8$ produces a $5\times6$ output: three tile rows and
three tile columns. One extra bottom row completes the last input tile.
The resulting sixth output row is discarded; the five valid rows do not
depend on those extra zeros. Neighboring input tiles overlap by two rows
or columns, while the corresponding output tiles do not overlap.

```python
def winograd_valid_3x3(image, weight):
    # NCHW, dense 3x3 filters, stride=dilation=1, no padding or bias.
    n, c, h, w = image.shape
    k = weight.shape[0]
    assert weight.shape == (k, c, 3, 3) and min(h, w) >= 3
    bt, gg, at = (matrix.to(image) for matrix in (win_BT, win_G, win_AT))
    ho, wo = h - 2, w - 2
    th, tw = (ho + 1) // 2, (wo + 1) // 2
    p = th * tw
    padded = F.pad(image, (0, wo % 2, 0, ho % 2))
    tiles = F.unfold(padded, kernel_size=4, stride=2).reshape(n, c, 4, 4, p)
    u = torch.einsum('ai,kcij,bj->kcab', gg, weight, gg)
    v = torch.einsum('ai,ncijp,bj->ncabp', bt, tiles, bt)
    u_matrices = u.permute(2, 3, 0, 1).reshape(16, k, c)
    v_matrices = v.permute(2, 3, 1, 0, 4).reshape(16, c, n*p)
    m_matrices = torch.bmm(u_matrices, v_matrices)
    m = m_matrices.reshape(4, 4, k, n, p).permute(3, 2, 0, 1, 4)
    output_tiles = torch.einsum('ai,nkijp,bj->nkabp', at, m, at)
    output = output_tiles.permute(0, 1, 4, 2, 3).reshape(n, k, th, tw, 2, 2)
    output = output.permute(0, 1, 2, 4, 3, 5).reshape(n, k, 2*th, 2*tw)
    return output[:, :, :ho, :wo]

win_images = torch.arange(2*3*7*8, dtype=torch.float64).reshape(2, 3, 7, 8) / 100
win_weights = torch.arange(4*3*3*3, dtype=torch.float64).reshape(4, 3, 3, 3) / 100
win_maps = winograd_valid_3x3(win_images, win_weights)
assert win_maps.shape == (2, 4, 5, 6)
torch.testing.assert_close(win_maps, F.conv2d(win_images, win_weights))
print(win_maps.shape)
```

Here `torch.bmm` receives sixteen matrix pairs of shapes $[4,3]$ and
$[3,18]$: four output channels, three input channels, and nine tiles for
each of two images. The code materializes all intermediate tensors to
expose the correspondence with the diagram. It is a reference calculation,
not an optimized convolution kernel.

### Transform cost, storage and rounding

The transformed filter has 16 values per input/output-channel pair rather
than nine. During inference it can be reused for every image until the
weights change. Input transforms can be reused across output channels;
output transforms operate after summing input channels. With few channels
or very small images, the transforms and rearrangements may consume much
of the time saved in the products. Materializing all transformed tiles
also requires workspace. A tiled implementation can change its size and
lifetime; neither the 16-value filter nor the 16-value input tile alone
specifies peak memory.

The cancellations used in the proof also expose rounding errors. In binary16,
2048 is representable but 2049 is not. Take $d=[1,2048,1,1]$ and
$g=[0,0,1]$. The direct outputs are both 1. With exact arithmetic the
transformed input contains 2049, and the products contain 1024.5. If each
stage is stored in binary16, those values round to 2048 and 1024.
The final cancellation then gives 0.5 instead of 1:

```python
rounding_outputs = {}
for dtype in [torch.float64, torch.float32, torch.float16]:
    d = torch.tensor([1., 2048., 1., 1.], dtype=dtype)
    g = torch.tensor([0., 0., 1.], dtype=dtype)
    v = win_BT.to(dtype) @ d
    u = win_G.to(dtype) @ g
    rounding_outputs[dtype] = win_AT.to(dtype) @ (u * v)
    print(dtype, v.tolist(), rounding_outputs[dtype].tolist())
torch.testing.assert_close(rounding_outputs[torch.float64], torch.ones(2, dtype=torch.float64))
torch.testing.assert_close(rounding_outputs[torch.float32], torch.ones(2))
torch.testing.assert_close(rounding_outputs[torch.float16], torch.full((2,), .5, dtype=torch.float16))
```

This CPU example fixes the dtype of every intermediate. It does not test
a GPU library's accumulation rules or establish that every low-precision
Winograd implementation fails. Larger output tiles can reduce the number
of variable products further, but require different transforms with their
own arithmetic and numerical behavior. Choosing an implementation requires
measuring the complete operator on the target shapes and hardware, and
checking its error against a higher-precision reference. The matrices and
symbolic derivation are in [wincnn](https://github.com/andravin/wincnn/tree/c2bf6305cc75866b19e204628b95baed4d13065c);
the original algorithm and numerical discussion are in
[Lavin and Gray, §§3–4](https://arxiv.org/abs/1509.09308v2).
<!-- END REVIEW: Winograd 2026-10-27 -->

<a id="связь-cnn-с-текстом-и-современными-моделями"></a>

## Свёртки в последовательностях и связь с Transformer

В одномерной свёртке пространственная ось заменяется временем или позицией
токена. Для каждого положения применяется одно и то же ядро.
При причинной свёртке окно не включает будущее; дополнение делают слева,
а не симметрично. Dilation позволяет увеличивать охватываемый контекст без
пропорционального увеличения числа весов; такой подход используется в
[WaveNet](https://arxiv.org/abs/1609.03499).

У свёртки коэффициенты смешивания внутри окна являются параметрами,
одинаковыми для всех примеров. В self-attention веса между позициями
вычисляются из текущих представлений. Общая residual-структура встречается
в обоих случаях, но она не делает операции смешивания одинаковыми.
Переход к patch embeddings и vision-энкодерам разобран отдельно в разделе VLM.

<a id="краткие-итоги"></a>

## Проверка собственной реализации

При переносе блока сначала проверяют порядок осей, число входных и выходных
каналов, правило padding, stride и фактические формы на обеих ветвях.
Для одной маленькой матрицы рассчитывают выход вручную и сравнивают с API.
Затем проверяют производные: общие веса должны собирать вклады всех окон,
а перекрывающиеся окна — складывать производные по входу.

Сравнение архитектур требует отдельно фиксировать число параметров,
размеры активаций и арифметический объём. Сокращение одного из них не
гарантирует сокращение остальных или ускорение на конкретном устройстве.
Качество классификации проверяют после обучения на отдельной выборке,
а не по успешному прохождению теста форм.

## Источники

- Aston Zhang, Zachary Lipton, Mu Li, Alexander Smola, [Dive into Deep Learning: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html):
  cross-correlation, обучение фильтра, padding/stride, каналы, pooling и LeNet.
  [AlexNet](https://d2l.ai/chapter_convolutional-modern/alexnet.html),
  [VGG](https://d2l.ai/chapter_convolutional-modern/vgg.html),
  [ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html).
  Рисунки и явно отмеченные фрагменты кода D2L сохранены в оригинале,
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- Stanford [CS231n, Convolutional Networks](https://cs231n.github.io/convolutional-networks/):
  локальные связи, общие веса и im2col.
- Araujo, Norris, Sim, [Computing Receptive Fields](https://distill.pub/2019/computing-receptive-fields/):
  расчёт области восприятия для последовательных операций и графов.
- LeCun et al., [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
- Krizhevsky, Sutskever, Hinton, [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
- Simonyan, Zisserman, [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556).
- He et al., [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
  и [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).
- PyTorch [Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html),
  [MaxPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html),
  [AvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html).
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/09_convolutions|TinyTorch 09 — полный исходный модуль о свёртках]]:
  самостоятельная реализация и анализ пространственных операций.
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-convolutional-neural-networks.pdf|Stanford CS230 — Convolutional Neural Networks Cheatsheet]].

[[00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|Глава об автоэнкодерах]]
в отдельном модуле меняет обучающую цель: сеть восстанавливает вход вместо
предсказания класса. Основной курс продолжается
[[00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|представлением текста числами]]:
как получить числовой вход для модели, если исходные данные — слова.
