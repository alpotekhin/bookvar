---
title: CNN — от свёртки до ResNet
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
primary_sources:
  - https://d2l.ai/chapter_convolutional-neural-networks/index.html
  - https://cs231n.github.io/convolutional-networks/
  - https://arxiv.org/abs/1512.03385
---

# CNN: от свёртки до ResNet

Полносвязный слой не знает, что соседние пиксели связаны, а один и тот же объект
может находиться в разных частях изображения. Convolutional neural network
встраивает два предположения: локальные признаки собираются из ближайших
позиций, а один детектор применяется ко всему изображению с общими весами.

## Двумерная cross-correlation

То, что библиотеки называют convolution, обычно является cross-correlation:

$$
Y_{i,j}=\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}
X_{i+u,j+v}K_{u,v}.
$$

Небольшое ядро $K$ скользит по входу и в каждой позиции вычисляет взвешенную
сумму локального окна. Поворот ядра, требуемый строгой математической свёрткой,
не нужен: weights обучаются и могут принять нужную ориентацию.

Для одного канала возьмём
$X=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}$ и
$K=\begin{bmatrix}1&0\\0&-1\end{bmatrix}$, без padding и bias,
со stride 1. В левом верхнем окне получаем
$Y_{0,0}=1\cdot1+2\cdot0+4\cdot0+5\cdot(-1)=-4$.
Сдвиг вправо даёт $2-6=-4$, вниз — $4-8=-4$, в правый нижний
угол — $5-9=-4$. Весь выход — матрица $2\times2$ из чисел $-4$.
Один и тот же контраст диагональных пикселей вычислен в четырёх местах,
но обучаемых коэффициентов ядра по-прежнему четыре.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/d2l-cross-correlation.svg]]

*Первый элемент выхода получается из выделенного окна входа и ядра: элементы
перемножаются попарно и складываются. Затем то же ядро сдвигается к следующей
позиции. Источник: Dive into Deep Learning,
[The Convolution Operation](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html),
прямая [ссылка](https://d2l.ai/_images/correlation.svg), CC BY-SA 4.0.*

Для RGB-входа ядро имеет форму `[C_in, k_h, k_w]`; слой содержит
$C_{out}$ таких ядер:

```text
X [B, C_in, H, W]
K [C_out, C_in, k_h, k_w]
────────────────────────────
Y [B, C_out, H_out, W_out]
```

Число параметров $C_{out}C_{in}k_hk_w$ не зависит от размера изображения. В
полносвязном слое оно росло бы вместе с $H\cdot W$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-local-connectivity.jpeg]]

*Один столбец выходных нейронов смотрит на одну локальную область по ширине и
высоте, но на всю глубину входа. Разные каналы выхода используют разные ядра.
Источник: Stanford CS231n,
[Convolutional Networks](https://cs231n.github.io/convolutional-networks/),
прямая [ссылка](https://cs231n.github.io/assets/cnn/cnn.jpeg).*

## Padding, stride и размер выхода

Для одной пространственной оси

$$
H_{out}=\left\lfloor
\frac{H+2p-d(k-1)-1}{s}+1
\right\rfloor,
$$

где $p$ — padding, $s$ — stride, $d$ — dilation. Padding сохраняет границы,
stride уменьшает разрешение, dilation расширяет receptive field без увеличения
числа параметров.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-stride-padding.jpeg]]

*Одно и то же ядро размера 3 при padding 1: stride 1 сохраняет пять выходных
позиций, stride 2 оставляет три. Источник: Stanford CS231n,
[Convolutional Networks](https://cs231n.github.io/convolutional-networks/),
прямая [ссылка](https://cs231n.github.io/assets/cnn/stride.jpeg).*

## Receptive field

Один нейрон первого слоя видит маленькое окно. После нескольких convolution
его receptive field охватывает всё большую область исходного изображения.
Поэтому ранние слои могут реагировать на края и текстуры, а поздние — собирать
их в части объектов. Реальное распределение признаков не обязано следовать этой
удобной истории, но геометрия доступной информации действительно иерархична.

Для последовательности слоёв обозначим через $r_l$ размер теоретического
receptive field по одной оси, а через $j_l$ — шаг между соседними выходными
позициями в координатах исходного входа. При $r_0=j_0=1$

$$
j_l=j_{l-1}s_l,\qquad
r_l=r_{l-1}+d_l(k_l-1)j_{l-1}.
$$

Например, для входа $32\times32$ с тремя каналами и dilation 1:

| Операция | $k,s,p$ | Выход $H\times W\times C$ | $j_l$ | $r_l$ |
|---|---|---|---:|---:|
| Вход | — | $32\times32\times3$ | 1 | 1 |
| Conv, 8 каналов | 3,1,1 | $32\times32\times8$ | 1 | 3 |
| MaxPool | 2,2,0 | $16\times16\times8$ | 2 | 4 |
| Conv, 16 каналов | 3,1,1 | $16\times16\times16$ | 2 | 8 |

Последнее ядро охватывает три позиции с шагом 2 в предыдущей карте:
оно добавляет $2\cdot2=4$ к уже доступному окну размера 4. Это
теоретическая область зависимости, включая padding у границы; реальный
градиент может распределяться внутри неё неравномерно или обращаться в ноль.

Pooling или convolution со stride агрегируют соседние позиции и снижают
разрешение. Max pooling сохраняет максимум в окне; global average pooling
усредняет каждую карту признаков до одного числа и часто заменяет большую
полносвязную «голову».

## LeNet-5

LeNet показала полный шаблон ранней CNN: convolution → nonlinearity → pooling,
повторение блока и несколько полносвязных слоёв для классификации.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l/lenet.svg]]

*Формы тензоров в LeNet-5. Источник: Zhang et al.,
[D2L: LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html),
CC BY-SA 4.0.*

Главное наследие LeNet — не конкретные размеры, а alternating hierarchy:
пространственный экстрактор признаков постепенно увеличивает число каналов и
уменьшает разрешение.

## AlexNet, VGG и переход к глубине

AlexNet продемонстрировала масштабируемость CNN на ImageNet, используя ReLU,
dropout, data augmentation и GPU. VGG сделала архитектуру регулярной: несколько
$3\times3$ convolutions перед downsampling. Два слоя $3\times3$ дают receptive
field, сопоставимый с $5\times5$, но добавляют нелинейность между операциями.

Простое добавление слоёв, однако, привело к degradation problem: более глубокая
сеть могла иметь более высокую ошибку на обучающей выборке, хотя теоретически могла реализовать
identity в лишних слоях.

Эта последовательность важна как история снятия ограничений. LeNet показала,
что общие локальные фильтры обучаются вместе с классификатором; AlexNet — что
такая система масштабируется на большой набор изображений и GPU; VGG — что
архитектуру можно собрать из повторяющихся малых блоков; ResNet — что глубине
нужен короткий тождественный путь. Более выразительная модель ещё не означает,
что её удастся оптимизировать.

## ResNet

ResNet заменяет прямое обучение отображения на остаточное обновление:

$$
y=x+F(x).
$$

Если пространственный размер или число каналов меняются, shortcut использует
$1\times1$ convolution, чтобы формы совпали.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l/resnet18.svg]]

*Архитектура ResNet-18: разрешение уменьшается, число каналов растёт, residual
blocks повторяются. Источник: Zhang et al.,
[D2L: ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html),
CC BY-SA 4.0.*

Разберём вычисление внутри одного блока приведённой ResNet-18, а не только
историческое название. В basic block с входом $[B,64,56,56]$ ветвь идёт как
`Conv 3×3 → BatchNorm → ReLU → Conv 3×3 → BatchNorm`; обе свёртки имеют
stride 1 и padding 1, поэтому форма остаётся $[B,64,56,56]$.
Shortcut передаёт $x$ без изменения. Только после сложения двух тензоров
применяется завершающая ReLU: $y=\operatorname{ReLU}(x+F(x))$.
Формула $x+F(x)$ выше выделяет сам residual-механизм, а не весь этот блок.

В первом блоке следующей группы основная ветвь меняет форму на
$[B,128,28,28]$: первая свёртка имеет 128 каналов и stride 2,
вторая — 128 каналов и stride 1. Identity такого размера уже нет:
в показанной реализации D2L shortcut — $1\times1$ convolution со stride 2,
также выдающая $[B,128,28,28]$. Складываются именно эти совпадающие формы.
Это описание выбранного basic block, не bottleneck и не более позднего
pre-activation ResNet: порядок normalization/activation и способ проекции
shortcut нужно сверять с конкретной архитектурой.

Вычислительный урок ResNet состоит в наличии отдельного пути и согласовании
форм в точке сложения. Transformer
тоже строит глубокое представление как последовательность поправок к residual
stream.

## Связь CNN с текстом и современными моделями

Одномерные convolution использовались для character- и word-level NLP, а
dilated causal convolution лежит в WaveNet. CNN вычисляет локальное смешивание с
фиксированными weights ядра; self-attention строит зависящие от входа веса между
позициями. Современные vision-language системы часто используют ViT вместо CNN,
но convolution остаётся важным inductive bias и компонентом гибридных моделей.

## Краткие итоги

- Convolution использует locality и weight sharing.
- Channels — разные обучаемые карты признаков; spatial axes сохраняют положение.
- Stride, padding и dilation определяют размер выхода и receptive field.
- LeNet задала базовый конвейер, VGG систематизировала блоки, ResNet открыла путь
  к очень глубоким сетям.
- Residual learning является общим принципом CNN и Transformer.

## Источники

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/09_convolutions|TinyTorch 09 — Convolutions]] — исполняемые `Conv2d` и pooling с явными формами тензоров.
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-convolutional-neural-networks.pdf|Stanford CS230 — Convolutional Neural Networks Cheatsheet]].
- D2L, [Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html).
- Stanford CS231n, [Convolutional Networks](https://cs231n.github.io/convolutional-networks/).
- LeCun et al., [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
- He et al., [Deep Residual Learning](https://arxiv.org/abs/1512.03385).

**Дальше:** autoencoder меняет цель — вместо label сеть восстанавливает собственный
вход и учит latent representation через bottleneck.
