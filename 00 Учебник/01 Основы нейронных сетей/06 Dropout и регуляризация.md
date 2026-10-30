---
title: Dropout и регуляризация нейросетей
type: textbook-chapter
status: canonical
last_updated: 2026-10-30
source_language: mixed
reference_terms:
  - label: DropConnect
    label_en: DropConnect
    anchor: dropconnect
  - label: Stochastic depth / DropPath
    label_en: Stochastic depth / DropPath
    anchor: stochastic-depth
primary_sources:
  - https://d2l.ai/chapter_multilayer-perceptrons/dropout.html
  - https://cs231n.github.io/neural-networks-2/
  - https://jmlr.org/papers/v15/srivastava14a.html
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html
  - https://arxiv.org/abs/1711.05101
  - https://proceedings.mlr.press/v28/wan13.html
  - https://arxiv.org/abs/1603.09382
  - https://jmlr.org/papers/v26/20-1396.html
---

# Dropout и регуляризация нейросетей

Нейронная сеть может уменьшать ошибку на обучающих примерах, подстраиваясь под
их случайные особенности. Например, несколько скрытых признаков вместе хорошо
описывают обучающую выборку, но один из них отражает фон изображения или шум
измерений. При изменении этого признака на новых данных предсказание ухудшается.
Регуляризация меняет условия обучения так, чтобы уменьшить такую зависимость.
Она может штрафовать веса, добавлять возмущения, преобразовывать примеры или
ограничивать продолжительность обучения.

Dropout добавляет шум к скрытым активациям. Часть значений временно заменяется
нулями, и следующий слой получает разные подмножества признаков для разных
проходов. Веса сети остаются общими. На новых данных dropout обычно отключают;
чтобы этот переход не менял средний масштаб входа следующего слоя, во время
обучения сохранённые значения масштабируют.

## Inverted dropout

Пусть $h_j$ — одна координата скрытого представления, $p$ — вероятность её
удаления, $q=1-p$ — вероятность сохранения. Для $0\le p<1$ выбирается маска:

$$
\begin{aligned}
m_j&\sim\operatorname{Bernoulli}(q),\\
\widetilde h_j&=\frac{m_j}{q}h_j.
\end{aligned}
$$

С вероятностью $p$ координата равна нулю, с вероятностью $q$ — $h_j/q$.
При фиксированном $h_j$ среднее по маскам равно исходному значению:

$$
\mathbb E_m[\widetilde h_j\mid h_j]
=p\cdot0+q\cdot\frac{h_j}{q}=h_j.
$$

Например, при $p=0{,}2$ сохранённые активации делят на 0,8, при $p=0{,}5$ —
удваивают. Масштабирование происходит при обучении; на инференсе слой становится
тождественным. Такая реализация называется inverted dropout.
В [конспекте CS231n](https://cs231n.github.io/neural-networks-2/#reg) символом
`p` обозначена вероятность сохранения, а в D2L и PyTorch — удаления. Поэтому
`p=0.8` из первого соглашения соответствует `p=0.2` во втором.

![[00 Учебник/Assets/Figures/curated/d2l/dropout.svg]]

*В левой сети пять скрытых координат. В правом проходе выключены $h_2$ и $h_5$,
поэтому показанный выход от них не зависит. В следующем проходе маска может
измениться; размер матриц параметров не меняется. Источник:
Zhang et al., [D2L: Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Для $H$ независимо маскируемых координат существует $2^H$ масок. Это не
$2^H$ независимо обученных сетей: все варианты используют одни и те же веса.
Выпавший признак перестаёт влиять на ошибку текущего примера через этот путь.
Другой пример в том же пакете может сохранить его и дать ненулевой градиент.

![[00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-dropout-train.jpeg]]

*Полная сеть и один вариант прореживания; зачёркнутые узлы исключены из
этого прохода. Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov,
[Dropout, Figure 1](https://jmlr.org/papers/v15/srivastava14a.html).
Рисунок из статьи воспроизведён в
[Stanford CS231n](https://cs231n.github.io/neural-networks-2/#reg).*

### Реализация и границы вероятности

Для получения маски достаточно независимых равномерных случайных чисел:
сохраняются элементы, для которых число не меньше $p$. Учебная функция ниже
применяется к вещественным тензорам и сознательно требует $p<1$:

```python
import torch

def dropout(x, p, training):
    # A deliberately restricted teaching contract: floating tensors, p < 1.
    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1")
    if not training or p == 0:
        return x
    keep = (torch.rand_like(x) >= p)
    return keep * x / (1 - p)

torch.manual_seed(26)
x_example = torch.tensor([2., -1., 3., 4.], dtype=torch.float64)
sample = dropout(x_example, .5, training=True)
assert torch.all((sample == 0) | (sample == 2 * x_example))
assert torch.equal(dropout(x_example, 0., True), x_example)
assert torch.equal(dropout(x_example, .5, False), x_example)
print(sample)
```

При $p=1$ она выдаст `ValueError` даже в режиме применения. Это ограничение
примера, а не PyTorch: `nn.Dropout(1)` поддерживается, на обучении даёт нули,
на инференсе возвращает вход. Случай $p=1$ обрабатывается отдельно от формулы
с делением; он уже не сохраняет среднее ненулевой активации.
При $p=0$ dropout тождественен в обоих режимах.

### Один прямой проход в числах

Пусть скрытый слой вернул $h=(2,-1,3,4)$, а $p=0.5$. Для маски
$m=(1,0,1,0)$ на обучении получим

$$
h'=\frac{m\odot h}{1-p}=(4,0,6,0).
$$

Сумма активаций изменилась с 8 до 10. Сохранение среднего по маскам не означает
сохранение суммы в каждом проходе. Для фиксированной маски производная по
исходной координате равна $m_j/q$. Следовательно, градиент последующего слоя
умножается на тот же коэффициент: выпавший путь даёт ноль, сохранённый —
масштабируется. Это относится к градиенту ошибки данных через этот путь;
отдельный штраф на параметры может продолжать изменять веса.

```python
from itertools import product

h = torch.tensor([2., -1., 3., 4.], dtype=torch.float64, requires_grad=True)
mask = torch.tensor([1., 0., 1., 0.], dtype=h.dtype)
masked = mask * h / .5
assert torch.equal(masked, torch.tensor([4., 0., 6., 0.], dtype=h.dtype))
masked.sum().backward()
assert torch.equal(h.grad, torch.tensor([2., 0., 2., 0.], dtype=h.dtype))

all_masks = torch.tensor(list(product([0., 1.], repeat=4)), dtype=h.dtype)
all_hidden = all_masks * h.detach() / .5
torch.testing.assert_close(all_hidden.mean(0), h.detach())
torch.testing.assert_close(all_hidden.var(0, correction=0), h.detach().square())
print(all_hidden.mean(0), all_hidden.var(0, correction=0))
```

Вместо приближённого среднего по случайной выборке здесь перебраны все 16 масок.
При $p=0{,}5$ они равновероятны. Дисперсии координат составляют $(4,1,9,16)$.
В общем случае:

$$
\begin{aligned}
\mathbb E_m[\widetilde h_j^2\mid h_j]&=\frac{h_j^2}{q},\\
\operatorname{Var}_m(\widetilde h_j\mid h_j)&=\frac{p}{q}h_j^2.
\end{aligned}
$$

Для ненулевой активации при $p\to1$ дисперсия растёт без ограничения, хотя при каждом $p<1$ среднее
сохраняется. Увеличение вероятности удаления поэтому не даёт монотонного
улучшения качества: следующий слой получает всё более редкие и крупные значения.

### Среднее представление и среднее предсказание

Для линейного следующего слоя усреднение можно перенести через матрицу весов:
$\mathbb E[W\widetilde h+b]=Wh+b$. После нелинейности это в общем случае
неверно. Пусть $h=1$, $p=0{,}5$, а следующий блок вычисляет
$f(z)=\max(0,z-1)$. На обучении $z$ равен 0 или 2 с одинаковой вероятностью;
выход блока равен соответственно 0 или 1. Средний выход — 0,5. Без dropout
блок получает $z=1$ и возвращает 0:

```python
possible_hidden = torch.tensor([0., 2.], dtype=torch.float64)
mean_training_prediction = torch.relu(possible_hidden - 1).mean()
deterministic_prediction = torch.relu(torch.tensor(1.) - 1)
assert mean_training_prediction.item() == .5
assert deterministic_prediction.item() == 0.
print(mean_training_prediction.item(), deterministic_prediction.item())
```

Таким образом, детерминированный проход полной глубокой сети — приближение
эффекта ансамбля, а не точное арифметическое среднее его предсказаний.
Отключение dropout не восстанавливает все возможные маски и не выполняет
усреднение явно. В исходной статье обсуждается приближённое объединение
подсетей; точность этого приближения зависит от модели.

## Режим применения и вычисление градиентов

`model.eval()` рекурсивно меняет флаг `training` у модулей. `nn.Dropout`
проверяет этот флаг и перестаёт вносить шум. `torch.no_grad()` отключает
построение графа градиентов, но не меняет режим модулей. Поэтому один
`no_grad()` не отключит dropout, а один `eval()` не запретит дифференцирование.

```python
from torch import nn

model = nn.Sequential(nn.Dropout(.5), nn.Linear(4, 1, bias=False)).double()
with torch.no_grad():
    model[1].weight.fill_(1.)
inputs = torch.tensor([[2., -1., 3., 4.]], dtype=torch.float64, requires_grad=True)

model.train()
torch.manual_seed(26)
with torch.no_grad():
    train_outputs = model(inputs.expand(32, -1))
assert not train_outputs.requires_grad
assert train_outputs.unique().numel() > 1

model.eval()
eval_output = model(inputs)
assert eval_output.item() == 8. and eval_output.requires_grad
eval_output.sum().backward()
torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
assert torch.equal(model(inputs), model(inputs))
print(train_outputs[:5].flatten(), eval_output.item())
```

Функциональный `torch.nn.functional.dropout` не принадлежит модулю и сам не
узнает, что был вызван `model.eval()`. В собственном `forward` ему передают
`training=self.training`; его аргумент `training` по умолчанию равен `True`.
Повторная установка одного seed перед каждым проходом тоже меняет процедуру:
маска начинает повторяться вместо независимой генерации.

Если нужен стохастический прогноз для оценки неопределённости, dropout можно
оставить включённым и усреднить несколько проходов. Это отдельная процедура,
а не обычный детерминированный инференс. Включать для неё `train()` у всей сети
с BatchNorm без разбора нельзя: одновременно изменится режим нормализации.

### Поэлементная и поканальная маска

Для тензора `[N,C,H,W]` обычный `Dropout` маскирует отдельные элементы.
`Dropout2d` маскирует целый канал внутри примера: все $H\times W$ значений
канала получают один коэффициент. Это различная структура шума:

```python
torch.manual_seed(26)
maps = torch.ones(2, 3, 4, 4)
elementwise = nn.Dropout(.5)(maps)
channelwise = nn.Dropout2d(.5)(maps)
channel_values = channelwise[:, :, 0, 0]
assert torch.equal(channelwise, channel_values[:, :, None, None].expand_as(maps))
assert torch.all((channel_values == 0) | (channel_values == 2))
assert not torch.equal(elementwise, elementwise[:, :, :1, :1].expand_as(maps))
print(channel_values)
```

Независимая поэлементная маска не совпадает с удалением связного фрагмента
изображения или целой ветки residual-блока. Формулы для независимых координат
нельзя без проверки переносить на коррелированные маски.

<!-- BEGIN REVIEW: structured dropout 2026-10-29 -->

<a id="dropconnect"></a>

## DropConnect: independent masks on connections

Suppose two output units receive the same two inputs. Dropping one input
removes that input from both output units. DropConnect can remove its
connection to the first unit while keeping its connection to the second.
The two procedures therefore produce different random networks, even when
they retain the same fraction of scalar values.

For a linear layer with weights $W\in\mathbb R^{o\times i}$ and input
$x\in\mathbb R^i$, inverted input dropout and inverted DropConnect give

$$
y_{\mathrm{input}}=W\left(\frac{m\odot x}{q}\right),\qquad
y_{\mathrm{connect}}=\left(\frac{M\odot W}{q}\right)x.
$$

Here $q$ is the keep probability, $m$ has $i$ independent Bernoulli entries,
and $M$ has $oi$ independent entries. Input dropout uses one mask value
for an entire column of $W$; DropConnect samples the entries of that column
independently. In both formulas the mean linear output is $Wx$.

With $x=(1,1)^T$, $W=\left[\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\right]$
and $q=0.5$, retaining only the first input gives $(2,6)^T$.
Keeping only the diagonal connections gives $(2,8)^T$, which input dropout
cannot produce. Enumerating all masks also reveals how the noise differs:

| Masked quantity | Possible masks | Mean output | Output covariance |
|---|---:|---|---|
| Two input coordinates | $2^2=4$ | $(3,7)^T$ | $\left[\begin{smallmatrix}5&11\\11&25\end{smallmatrix}\right]$ |
| Four connections | $2^4=16$ | $(3,7)^T$ | $\left[\begin{smallmatrix}5&0\\0&25\end{smallmatrix}\right]$ |

The variances of the two outputs agree, but input dropout correlates their
fluctuations because both outputs lose the same input. Independent connection
masks remove that covariance for this fixed linear layer.

```python
from itertools import product
import math
import torch

dc_W = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
dc_x = torch.ones(2, dtype=dc_W.dtype)
dc_q = .5
dc_inputs = torch.tensor(list(product([0., 1.], repeat=2)), dtype=dc_W.dtype)
dc_edges = torch.tensor(list(product([0., 1.], repeat=4)),
                        dtype=dc_W.dtype).reshape(-1, 2, 2)
dc_y_input = (dc_inputs * dc_x / dc_q) @ dc_W.T
dc_y_edges = ((dc_edges * dc_W / dc_q) @ dc_x)

def dc_covariance(outputs):
    centered = outputs - outputs.mean(0)
    return centered.T @ centered / len(outputs)

torch.testing.assert_close(dc_y_input.mean(0), dc_W @ dc_x)
torch.testing.assert_close(dc_y_edges.mean(0), dc_W @ dc_x)
torch.testing.assert_close(dc_covariance(dc_y_input),
                           torch.tensor([[5., 11.], [11., 25.]], dtype=dc_W.dtype))
torch.testing.assert_close(dc_covariance(dc_y_edges),
                           torch.diag(torch.tensor([5., 25.], dtype=dc_W.dtype)))

dc_parameter = dc_W.clone().requires_grad_()
dc_diagonal = torch.eye(2, dtype=dc_W.dtype)
((dc_diagonal * dc_parameter / dc_q) @ dc_x).sum().backward()
torch.testing.assert_close(dc_parameter.grad, 2 * dc_diagonal)
torch.testing.assert_close(dc_parameter.detach(), dc_W)

# Original, non-inverted convention: four equiprobable connection masks.
dc_z = dc_inputs @ torch.tensor([1., -1.], dtype=dc_W.dtype)
assert torch.relu(dc_z).mean().item() == .25
assert torch.relu(dc_z.mean()).item() == 0.
dc_gaussian_relu_mean = math.sqrt(.5) / math.sqrt(2 * math.pi)
assert abs(dc_gaussian_relu_mean - .28209479177387814) < 1e-14
print(dc_covariance(dc_y_input), dc_covariance(dc_y_edges))
print("Exact ReLU mean:", .25, "Gaussian approximation:", dc_gaussian_relu_mean)
```

The backward mask in this example is also explicit: the diagonal weights
receive gradient $2$, while the two removed connections receive zero
data-loss gradient. Their stored parameters have not been erased. A later
example can retain them; an optimizer's separate weight decay can still
change them. DropConnect is not permanent pruning.

Wan et al., [Sections 2.2–3.2 and Algorithms 1–2](https://proceedings.mlr.press/v28/wan13.pdf),
use unscaled masks during training, not the inverted convention of the
first example. They sample a different mask for each training example.
For a fixed input, a preactivation $z_i=\sum_j M_{ij}W_{ij}x_j$ has

$$
\mu_i=q\sum_jW_{ij}x_j,\qquad
\sigma_i^2=q(1-q)\sum_jW_{ij}^2x_j^2.
$$

At test time their inference algorithm approximates this sum by a Gaussian
with those moments, samples preactivations, applies the activation, and
averages the resulting activations before the classifier. This accounts for
the fact that applying a nonlinearity to the mean is not the same as averaging
the nonlinear outputs. It is still an approximation. In the code above,
$z=M_1-M_2$ takes values $-1,0,1$; its exact mean ReLU output is $0.25$.
A Gaussian with the same mean $0$ and variance $0.5$ gives approximately
$0.2821$, while ReLU of the mean gives $0$.

![[00 Учебник/Assets/Figures/curated/structured-dropout-2026-10-29/dropconnect-original.png|Original DropConnect graph: full network, sampled connections and an arborescence]]

*Albert Senen–Cerda and Jaron Sanders,
[Almost Sure Convergence of Dropout Algorithms for Neural Networks, Figure 1](https://jmlr.org/papers/volume26/20-1396/20-1396.pdf#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The full original figure and caption are retained. Panels (a–b) compare the
full graph with a sampled connection mask; panel (c) is the tree-shaped
network used in the paper's theoretical analysis. The original caption says
“removing” with probability $p$, but the panels and the following paragraph
use $p$ as the keep probability. Our equations use $q$ for keeping.*

Random zeros do not by themselves reduce the work of a dense matrix
multiplication. Per-example connection masks also need a representation
and an implementation; a single masked weight matrix shared across the
whole batch is a different noise process from the original algorithm.

<a id="stochastic-depth"></a>

## Stochastic depth: one gate for a residual branch

For a residual block $y=x+F(x)$, stochastic depth samples one gate for the
whole branch $F$. If the gate is zero, every coordinate of that branch
vanishes together. Elementwise dropout on $F(x)$ can retain one coordinate
and remove another; a single branch gate cannot.

The original method of Huang et al.,
[Section 3, Eqs. 2–5](https://arxiv.org/pdf/1603.09382v3),
uses a Bernoulli gate $b_l$ with survival probability $q_l$ for each block
and each mini-batch. Its post-activation ResNet block is
$\operatorname{ReLU}(x+b_lF(x))$. At test time all branches are active and
their outputs are multiplied by $q_l$.
When $b_l=0$, the block is an identity under the paper's nonnegative-input
assumption: $\operatorname{ReLU}(x)=x$. That statement does not hold for
an arbitrary negative input.

Modern DropPath implementations often use inverted scaling instead.
For a fixed $x$ and fixed branch output $F(x)$, the two conventions are:

| Convention | Training, before any final activation | Evaluation |
|---|---|---|
| Original stochastic depth | $x+bF(x)$ | $x+qF(x)$ |
| Inverted branch dropout | $x+bF(x)/q$ | $x+F(x)$ |

For $x=(1,2)$, $F(x)=(3,4)$ and $q=0.5$, the original training outputs are
$(1,2)$ or $(4,6)$; their mean is $(2.5,4)$.
Inverted scaling instead gives $(1,2)$ or $(7,10)$, whose mean is $(4,6)$.
The formulas preserve different reference outputs for the same branch
weights. After a nonlinear activation, even equality of these preactivation
means does not establish equality of mean predictions.

```python
import torch

sd_x = torch.tensor([1., 2.], dtype=torch.float64)
sd_branch = torch.tensor([3., 4.], dtype=sd_x.dtype)
sd_gates = torch.tensor([0., 1.], dtype=sd_x.dtype)[:, None]
sd_original = sd_x + sd_gates * sd_branch
sd_inverted = sd_x + sd_gates * sd_branch / .5
torch.testing.assert_close(sd_original.mean(0), sd_x + .5 * sd_branch)
torch.testing.assert_close(sd_inverted.mean(0), sd_x + sd_branch)
sd_coordinate_dropout = sd_x + torch.tensor([1., 0.]) * sd_branch / .5
assert sd_coordinate_dropout.tolist() == [7., 2.]
assert not torch.any(torch.all(sd_inverted == sd_coordinate_dropout, dim=1))
print(sd_original, sd_inverted, sd_coordinate_dropout)
```

The original function below is from Ross Wightman's
[timm implementation, pinned revision beabdc3a](https://github.com/huggingface/pytorch-image-models/blob/beabdc3a6833d39814c895b2bc84cbc61a05834d/timm/layers/drop.py#L157-L175).
It uses inverted scaling by default. Its name also avoids confusing branch
dropping with the independent connection masks of DropConnect.
The function is reproduced unchanged under
[Apache 2.0](/licenses/timm-beabdc3a-Apache-2.0.txt); the calls after it
are executable examples.

```python
import torch
from itertools import product

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

torch.manual_seed(29)
sd_values = torch.ones(16, 3, 4)  # [batch, positions, features]
sd_masked = drop_path(sd_values, .5, training=True)
assert torch.equal(sd_masked, sd_masked[:, :1, :1].expand_as(sd_values))
assert torch.all((sd_masked == 0) | (sd_masked == 2))
assert set(sd_masked[:, 0, 0].tolist()) == {0., 2.}
assert drop_path(sd_values, .5, training=False) is sd_values
assert torch.count_nonzero(drop_path(sd_values, 1., training=True)) == 0

# Original paper's depth schedule, with independent gates across four blocks.
sd_keep = 1 - torch.arange(1, 5, dtype=torch.float64) / 4 * (1 - .5)
sd_masks = torch.tensor(list(product([0., 1.], repeat=4)), dtype=sd_keep.dtype)
sd_probabilities = torch.where(sd_masks.bool(), sd_keep, 1 - sd_keep).prod(1)
sd_depths = sd_masks.sum(1)
sd_expected_depth = sd_probabilities @ sd_depths
torch.testing.assert_close(sd_probabilities.sum(), torch.tensor(1., dtype=sd_keep.dtype))
assert sd_expected_depth.item() == 2.75
print("Keep probabilities:", sd_keep, "Expected active branches:", sd_expected_depth)
```

For a tensor shaped $[B,T,d]$, the mask has shape $[B,1,1]$: each batch
example gets its own gate, shared across all its positions and features.
The original paper instead shares a block gate across the mini-batch.
Neither is an independent gate per token. With `scale_by_keep=False`,
this function returns the unscaled training branch, but its evaluation
path still returns the input; that flag alone does not implement the
original paper's test-time factor $q$.

The depth schedule in the example follows the paper:
$q_l=1-\frac lL(1-q_L)$. Linearity of expectation gives
$\mathbb E[\text{active branches}]=\sum_{l=1}^Lq_l$.
For four branches and $q_L=0.5$, the survival probabilities are
$(0.875,0.75,0.625,0.5)$, so the mean active depth is $2.75$.
For the paper's $L=54$ residual blocks, it is $40.25$.

Active depth is not automatically a runtime reduction. The original
mini-batch gate can be sampled before evaluating $F$, allowing the
whole branch to be skipped. In `x + drop_path(F(x), ...)`, the argument
`F(x)` has already been computed before `drop_path` runs. Multiplying it
by zero saves no branch forward computation. Per-example masks require
additional conditional execution to skip work for individual examples;
normalization statistics can also differ from skipping the whole batch.

The attention and embedding variants occur later on the reading route,
in [[00 Учебник/05 Attention и Transformer/03 Полный Transformer#transformer-dropout|the complete Transformer]].
There the attention mask forbids connections structurally, while dropout
randomly removes some allowed attention weights during training.

<!-- END REVIEW: structured dropout 2026-10-29 -->

## Как шум меняет обучающую цель

Сохранение среднего входа не означает сохранение средней функции потерь.
Рассмотрим линейный прогноз $\hat y=w^T\widetilde h$, истинный ответ $y=1$,
признаки $h=(2,-1)$ и веса $w=(0{,}5,2)$. Без dropout прогноз равен $-1$,
квадратичная ошибка — 4. При $p=0{,}5$ четыре маски дают:

| Маска | Прогноз | Ошибка |
|---|---|---|
| $(0,0)$ | 0 | 1 |
| $(1,0)$ | 2 | 1 |
| $(0,1)$ | −4 | 25 |
| $(1,1)$ | −2 | 9 |

Средний прогноз по-прежнему равен −1, но средняя ошибка равна
$(1+1+25+9)/4=9$. Ещё пять единиц ошибки возникли из-за разброса предсказаний.
Если обозначить шум прогноза через $\varepsilon$, то
$\hat y=w^Th+\varepsilon$ и $\mathbb E[\varepsilon]=0$. При раскрытии квадрата
смешанное слагаемое исчезает после усреднения. Для независимых масок:

$$
\begin{aligned}
\mathbb E_m[(y-w^T\widetilde h)^2]
&=(y-w^Th)^2\\
&\quad+\frac{p}{1-p}\sum_j w_j^2h_j^2.
\end{aligned}
$$

В примере добавка равна $0{,}5^2\cdot2^2+2^2\cdot(-1)^2=5$.
Минимизируется сумма исходной ошибки и штрафа за чувствительность к удалению
признаков; уменьшение одного слагаемого может компенсировать рост другого.
Коэффициент при $w_j^2$ зависит от величины признака $h_j$, поэтому это
взвешенный квадратичный штраф, а не один и тот же L2-штраф для всех координат.
После усреднения по обучающим примерам в коэффициенты входят средние $h_j^2$.
Для нелинейной глубокой сети такой точной формулы общего вида уже нет.

```python
linear_h = torch.tensor([2., -1.], dtype=torch.float64)
linear_w = torch.tensor([.5, 2.], dtype=torch.float64)
linear_masks = torch.tensor(list(product([0., 1.], repeat=2)), dtype=torch.float64)
linear_predictions = (linear_masks * linear_h / .5) @ linear_w
mean_dropout_loss = (1 - linear_predictions).square().mean()
clean_loss = (1 - linear_h @ linear_w).square()
noise_penalty = (linear_w * linear_h).square().sum()
assert clean_loss.item() == 4.
assert noise_penalty.item() == 5.
assert mean_dropout_loss.item() == 9.
torch.testing.assert_close(mean_dropout_loss, clean_loss + noise_penalty)
print(linear_predictions, clean_loss, noise_penalty, mean_dropout_loss)
```

Этот частный случай разобран также в §9.1
[исходной статьи о Dropout](https://jmlr.org/papers/v15/srivastava14a.html).
У авторов вероятность сохранения обозначена $p$, а масштабирование выполняется
при тестировании; здесь используется вероятность удаления и inverted dropout.

<a id="weight-decay"></a>

## Штраф на веса и weight decay

L2-регуляризация добавляет к ошибке данных штраф на величину параметров:

$$
J(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2.
$$

Для обычного SGD без momentum градиент L2-штрафа равен $\lambda\theta$, поэтому обновление
можно переписать как

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\nabla L(\theta_t).
$$

Здесь добавление L2-штрафа совпадает с умножением каждого веса на
$1-\eta\lambda$ и отдельным шагом по ошибке данных. При $0\le\eta\lambda\le1$
этот множитель уменьшает модуль веса или оставляет его прежним. В Adam добавка
$\lambda\theta$ при связанном L2 входит в оценки первого и второго моментов
вместе с остальным градиентом. Она тоже подвергается адаптивному масштабированию.
AdamW выносит уменьшение параметра за пределы вычисления этих моментов:

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

В этой записи $\hat m_t$ и $\hat v_t$ в AdamW получены только из градиентов
ошибки данных. Коэффициент $\lambda$ используется в соглашении PyTorch:
уменьшение за шаг равно $\eta\lambda\theta$. В Eq. 1 статьи
[Loshchilov и Hutter](https://arxiv.org/abs/1711.05101) коэффициент уменьшения
за шаг обозначен отдельно от learning rate; сравнивать численные значения
коэффициентов без их формулы нельзя.

Рассмотрим один шаг из $\theta=(1,2)$ с градиентом данных $(0{,}2,2)$,
$\eta=0{,}1$ и $\lambda=0{,}1$. SGD с L2 даёт $(0{,}97,1{,}78)$.
Для сравнения первых адаптивных шагов положим $\beta_1=\beta_2=0$:
Adam с L2 нормирует градиент $(0{,}3,2{,}2)$ и получает примерно $(0{,}9,1{,}9)$.
AdamW уменьшает исходные веса до $(0{,}99,1{,}98)$ и отдельно вычитает
адаптивный шаг примерно $(0{,}1,0{,}1)$, получая $(0{,}89,1{,}88)$.

```python
initial_weights = torch.tensor([1., 2.], dtype=torch.float64)
w_sgd, w_adam, w_adamw = [nn.Parameter(initial_weights.clone()) for _ in range(3)]
optimizers = [
    torch.optim.SGD([w_sgd], lr=.1, weight_decay=.1),
    torch.optim.Adam([w_adam], lr=.1, betas=(0., 0.), eps=1e-8,
                     weight_decay=.1, decoupled_weight_decay=False),
    torch.optim.AdamW([w_adamw], lr=.1, betas=(0., 0.), eps=1e-8, weight_decay=.1),
]
for parameter, optimizer in zip([w_sgd, w_adam, w_adamw], optimizers):
    parameter.grad = torch.tensor([.2, 2.], dtype=torch.float64)
    optimizer.step()

for parameter, expected in zip([w_sgd, w_adam, w_adamw], [[.97, 1.78], [.9, 1.9], [.89, 1.88]]):
    torch.testing.assert_close(parameter, torch.tensor(expected, dtype=torch.float64), atol=1e-7, rtol=0)
print(w_sgd.detach(), w_adam.detach(), w_adamw.detach())
```

Это проверка одного шага, не сравнение качества трёх оптимизаторов после
обучения. У PyTorch различаются также нулевой градиент и `grad=None`:
во втором случае оптимизатор пропускает параметр, включая weight decay.
Решение о штрафе для смещений и параметров нормализации задают явно группами
параметров. Сам выбор AdamW не исключает их из штрафа автоматически.

<a id="data-augmentation"></a>

## Преобразования обучающих данных

Аугментация добавляет разные представления одного примера. Для классификации
выбирают преобразование $T$, при котором правильная метка $y$ сохраняется:
модель обучается и на $(x,y)$, и на $(T(x),y)$. Например, небольшое изменение
яркости фотографии может оставить изображённый объект тем же. Если модель
использовала яркость фона как случайную подсказку, такая вариация уменьшает
надёжность этой подсказки при обучении.

Свойство сохранения метки — предположение о задаче, которое нужно проверить.
При распознавании дорожных знаков отражение способно поменять направление,
при распознавании цифр большой поворот или зеркальное отражение — уничтожить
исходный символ. В детекции и сегментации геометрическое преобразование
изображения требует преобразовать также координаты или маску разметки.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/data_augmentation.png|Отражение, обрезка и поворот цифры «3»]]

*Слева цифра «3», далее её отражение, обрезка и поворот. Рисунок показывает
геометрические преобразования; он не доказывает, что все полученные изображения
остаются корректными примерами класса «3». Vijay Janapa Reddi и соавторы,
[Harvard ML Systems, Robust AI, Figure 22](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html#fig-data-augmentation),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

В тексте удаление одного слова может изменить целевую метку: «мне понравился
фильм» и «мне не понравился фильм» выражают противоположную оценку. Поэтому
случайное удаление слов или перестановку нельзя считать безопасными только
потому, что они увеличивают число примеров. Для перевода, извлечения фактов
и классификации допустимы разные преобразования.

Аугментации применяют после разделения данных: производные одной исходной
фотографии не должны попасть одновременно в train и test. Случайные
преобразования обучающего режима также не должны незаметно менять процедуру
валидации. Если несколько преобразованных версий используются при инференсе,
их усреднение задаёт отдельный алгоритм test-time augmentation.

<a id="early-stopping"></a>

## Ранняя остановка и сохранение лучшей версии

Число шагов обучения тоже влияет на подгонку выборки. При ранней остановке
после каждой выбранной точки измеряют проверочную метрику и сохраняют параметры
лучшей версии. Когда улучшений нет достаточно долго, обучение прекращают и
возвращают сохранённые веса. Обучающая потеря может при этом продолжать снижаться.
Test set не участвует в выборе момента остановки.

Пусть проверочная ошибка по эпохам равна $0{,}50$, $0{,}40$, $0{,}35$,
$0{,}36$, $0{,}37$. При минимизации и `patience=2` первые три эпохи обновляют
лучшую версию. На четвёртой счётчик отсутствия улучшений становится равен 1,
на пятой — 2. Обучение останавливается на пятой, но вернуть нужно веса третьей.
В примере ошибка берётся из заданного журнала, а в параметр записывается номер
эпохи. Так можно проверить, что восстановилась именно нужная копия весов.

```python
checkpoint_model = nn.Linear(1, 1, bias=False).double()
validation_losses = [.50, .40, .35, .36, .37, .34]
patience, wait = 2, 0
best_value, best_epoch, best_state = float("inf"), None, None

for epoch, validation_loss in enumerate(validation_losses, 1):
    with torch.no_grad():
        checkpoint_model.weight.fill_(epoch)
    if validation_loss < best_value:
        best_value, best_epoch, wait = validation_loss, epoch, 0
        best_state = {name: tensor.detach().clone()
                      for name, tensor in checkpoint_model.state_dict().items()}
    else:
        wait += 1
    if wait >= patience:
        stopped_epoch = epoch
        break

assert stopped_epoch == 5 and best_epoch == 3
assert checkpoint_model.weight.item() == 5
assert best_state["weight"].item() == 3
checkpoint_model.load_state_dict(best_state)
assert checkpoint_model.weight.item() == 3
print(stopped_epoch, best_epoch, checkpoint_model.weight.item())
```

Тензоры из `state_dict()` разделяют память с текущим состоянием модели. Одного присваивания
`best_state = model.state_dict()` недостаточно для неизменяемой копии:
дальнейшие обновления параметров изменят сохранённые значения. В примере
каждый тензор скопирован; для реального продолжения обучения сохраняют также
состояние оптимизатора, scheduler и генераторов случайных чисел.

В журнал добавлено значение 0,34 на шестой эпохе. До неё алгоритм не доходит:
patience не гарантирует остановку в глобальном минимуме будущей кривой.
Частоту проверок, допустимое число ухудшений и минимальное значимое улучшение
`min_delta` задают заранее. Многократный ручной подбор этих настроек по одной
проверочной выборке тоже подстраивает процедуру под неё.

В [Keras EarlyStopping](https://keras.io/api/callbacks/early_stopping/)
восстановление лучшей версии управляется `restore_best_weights` и по умолчанию
выключено. Остановка и восстановление — два разных действия, которые нужно
проверять в используемом коде.

## Кривые обучения при включённом dropout

Обучающая и проверочная потери могут вычисляться в разных режимах сети.
На обучении dropout вносит шум; при валидации он отключён. Кроме того,
обучающая потеря часто усредняется по пакетам, обработанным разными версиями
весов внутри эпохи, а проверочная вычисляется после её окончания.
Поэтому проверочная потеря ниже обучающей сама по себе не доказывает ошибку
разбиения данных.

![[00 Учебник/Assets/Figures/curated/dropout-2026-10-26/d2l-training.svg|Кривые обучения MLP с dropout из D2L]]

*Aston Zhang и соавторы, [D2L, Dropout — Training](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html#training),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Полный авторский график для MLP с двумя скрытыми слоями по 256 координат,
$p=0{,}5$ в обоих слоях, Fashion-MNIST, десять эпох. Синяя линия — train_loss,
оранжевая — val_loss, зелёная — val_acc. Accuracy и loss имеют разный смысл,
хотя здесь показаны на одной оси. Это сохранённый запуск курса, не результат
локального обучения в Bookvar.*

Такой график показывает ход одного запуска, но не отделяет пользу dropout
от влияния архитектуры и остальных настроек. Для сопоставимого разрыва между
обучением и проверкой можно дополнительно пересчитать ошибку на обеих выборках
одними и теми же весами в `eval()` и без добавления штрафа к одной из метрик.
Для оценки самого метода нужен запуск без dropout с тем же протоколом.

## Как выбрать ограничение по наблюдаемой ошибке

Хороший результат на обучающей выборке и плохой на проверочной совместимы
с переобучением. До усиления регуляризации проверяют, что метрики измерены
сопоставимо, данные разделены по единице задачи и между выборками нет
непредусмотренного сдвига. Затем сравнивают дополнительные данные, допустимые
аугментации, штраф на веса, dropout и раннюю остановку.

Если результат плох и на обучающих данных, и на проверочных, сильнее удалять
признаки может оказаться бесполезно. Причиной бывают недостаточная модель,
ошибка в признаках, короткое обучение или проблема оптимизации. Dropout не
исправит неверную разметку и не гарантирует устойчивость к новому распределению
данных. Например, удаление случайных активаций само по себе не обучает модель
распознавать ранее не встречавшийся язык.

## Регуляризация должна соответствовать режиму

| Режим | Ограничение и проверка |
|---|---|
| Мало размеченных данных | Аугментации, перенос представлений и штрафы; проверить метки и разбиение. |
| Предобучение | Состав данных, дубликаты и штрафы; проверить запоминание и пересечение с тестами. |
| SFT | Длительность обучения и смесь данных; проверить качество задачи и сохранение других навыков. |
| LoRA/PEFT | Ранг и регуляризация адаптера; проверить недообучение и переобучение. |
| RL | Отклонение от опорной модели; проверить надёжность награды и её переоптимизацию. |

Таблица перечисляет проверяемые варианты, а не готовые рецепты. Вероятность
dropout, коэффициент штрафа и длительность обучения выбирают для конкретных
данных и модели. Значение $p=0{,}5$ из учебного MLP не является универсальной
настройкой для Transformer, адаптера или свёрточного слоя.

<a id="краткие-итоги"></a>

## Контрольный эксперимент

Чтобы отделить действие dropout и weight decay, сравнивают четыре комбинации:
оба выключены, включён только dropout, включён только штраф и включены оба.
Начальные веса, разбиение данных, порядок пакетов и бюджет обучения фиксируют.
Генератор порядка данных отделяют от генератора масок: иначе разное число
случайных вызовов внутри модели может изменить и порядок примеров.

Для каждой комбинации сохраняют проверочную кривую, лучшую эпоху, конечную
обучающую ошибку в согласованном режиме и разброс результатов по нескольким
начальным состояниям. Аугментации и ранняя остановка либо одинаковы во всех
вариантах, либо являются отдельными факторами эксперимента. Тестовая выборка
используется после выбора настроек, а не для поиска наиболее удачного seed.

## Источники

- D2L, [Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html).
- Srivastava et al., [Dropout](https://jmlr.org/papers/v15/srivastava14a.html).
- Stanford CS231n, [Regularization](https://cs231n.github.io/neural-networks-2/#reg).
- Loshchilov, Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- PyTorch, [Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html),
  [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
- Keras, [EarlyStopping](https://keras.io/api/callbacks/early_stopping/).
- Wan et al., [Regularization of Neural Networks using DropConnect](https://proceedings.mlr.press/v28/wan13.html), Sections 2.2–3.2, Algorithms 1–2.
- Huang et al., [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382), Section 3.
- Senen–Cerda, Sanders, [Almost Sure Convergence of Dropout Algorithms for Neural Networks](https://jmlr.org/papers/v26/20-1396.html), Introduction and Figure 1.

[[00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований|Следующая глава]]
разбирает преобразования обучающих примеров, смешивание изображений и меток,
а также выбор и проверку политик аугментации.
