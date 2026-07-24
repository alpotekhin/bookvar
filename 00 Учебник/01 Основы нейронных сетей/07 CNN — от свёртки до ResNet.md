---
title: CNN — от свёртки до ResNet
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://d2l.ai/chapter_convolutional-neural-networks/index.html
  - https://cs231n.github.io/convolutional-networks/
  - https://arxiv.org/abs/1512.03385
---

# CNN: от свёртки до ResNet

**Полный исполняемый модуль:** [[05 Источники/Courses/Harvard ML Systems/tinytorch/09_convolutions|TinyTorch 09 — Convolutions]]. Модуль реализует `Conv2d`, `MaxPool2d` и `AvgPool2d` явными операциями, делая видимыми формы тензоров и вычислительную стоимость пространственной свёртки.

Полная оригинальная англоязычная памятка Stanford — от свёртки и padding до
типичных архитектур и transfer learning — сохранена в
[[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-convolutional-neural-networks.pdf|Convolutional Neural Networks Cheatsheet]].

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
fully connected layer оно росло бы вместе с $H\cdot W$.

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

Pooling или convolution со stride агрегируют соседние позиции и снижают
разрешение. Max pooling сохраняет максимум в окне; global average pooling
усредняет каждую карту признаков до одного числа и часто заменяет большую
полносвязную «голову».

## LeNet-5

LeNet показала полный шаблон ранней CNN: convolution → nonlinearity → pooling,
повторение блока и несколько fully connected layers для классификации.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l/lenet.svg]]

*Формы тензоров в LeNet-5. Источник: Zhang et al.,
[D2L: LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html),
CC BY-SA 4.0.*

Главное наследие LeNet — не конкретные размеры, а alternating hierarchy:
пространственный feature extractor постепенно увеличивает число каналов и
уменьшает разрешение.

## AlexNet, VGG и переход к глубине

AlexNet продемонстрировала масштабируемость CNN на ImageNet, используя ReLU,
dropout, data augmentation и GPU. VGG сделала архитектуру регулярной: несколько
$3\times3$ convolutions перед downsampling. Два слоя $3\times3$ дают receptive
field, сопоставимый с $5\times5$, но добавляют нелинейность между операциями.

Простое добавление слоёв, однако, привело к degradation problem: более глубокая
сеть могла иметь выше training error, хотя теоретически могла реализовать
identity в лишних слоях.

Эта последовательность важна как история снятия ограничений. LeNet показала,
что общие локальные фильтры обучаются вместе с классификатором; AlexNet — что
такая система масштабируется на большой набор изображений и GPU; VGG — что
архитектуру можно собрать из повторяющихся малых блоков; ResNet — что глубине
нужен короткий тождественный путь. Более выразительная модель ещё не означает,
что её удастся оптимизировать.

## ResNet

ResNet заменяет прямое обучение отображения на residual update:

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

Residual connection стала общим приёмом далеко за пределами vision. Transformer
тоже строит глубокое представление как последовательность поправок к residual
stream.

## Связь CNN с текстом и современными моделями

Одномерные convolution использовались для character- и word-level NLP, а
dilated causal convolution лежит в WaveNet. CNN вычисляет локальное смешивание с
фиксированными weights ядра; self-attention строит зависящие от входа веса между
позициями. Современные vision-language системы часто используют ViT вместо CNN,
но convolution остаётся важным inductive bias и компонентом гибридных моделей.

## Что нужно унести из главы

- Convolution использует locality и weight sharing.
- Channels — разные обучаемые карты признаков; spatial axes сохраняют положение.
- Stride, padding и dilation определяют размер выхода и receptive field.
- LeNet задала базовый pipeline, VGG систематизировала блоки, ResNet открыла путь
  к очень глубоким сетям.
- Residual learning является общим принципом CNN и Transformer.

## Источники

- D2L, [Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html).
- Stanford CS231n, [Convolutional Networks](https://cs231n.github.io/convolutional-networks/).
- LeCun et al., [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
- He et al., [Deep Residual Learning](https://arxiv.org/abs/1512.03385).

**Дальше:** autoencoder меняет цель — вместо label сеть восстанавливает собственный
вход и учит latent representation через bottleneck.
