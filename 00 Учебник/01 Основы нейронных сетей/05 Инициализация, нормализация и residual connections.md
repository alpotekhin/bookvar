---
title: Инициализация, нормализация и residual connections
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html
  - https://d2l.ai/chapter_convolutional-modern/batch-norm.html
  - https://d2l.ai/chapter_convolutional-modern/resnet.html
---

# Инициализация, нормализация и residual connections

Глубокая сеть должна переносить информацию и градиент через десятки или сотни
слоёв. Инициализация управляет масштабом в самом начале обучения, нормализация —
масштабом текущих представлений, residual connection — доступностью короткого
пути через глубину. Это разные механизмы, хотя все три улучшают trainability.

## Почему одинаковые веса не работают

Если все нейроны слоя начать с одинаковых весов, они получат одинаковые выходы
и одинаковые градиенты. Обновления не нарушат симметрию: множество нейронов
останется копиями одного. Случайная инициализация нужна прежде всего для
разрушения этой симметрии.

Но масштаб случайных чисел нельзя выбирать произвольно. Для
$z_j=\sum_{i=1}^{n}w_{ij}x_i$ при независимых нулевых средних

$$
\operatorname{Var}(z_j)
\approx n\operatorname{Var}(w)\operatorname{Var}(x).
$$

Если $\operatorname{Var}(w)$ постоянна, дисперсия растёт с шириной слоя. Чтобы
сохранить масштаб, веса делают порядка $1/\sqrt n$.

## Xavier и He initialization

Xavier/Glorot выбирает масштаб с учётом fan-in и fan-out:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{in}+n_{out}}.
$$

Для ReLU примерно половина значений обнуляется. He/Kaiming initialization
компенсирует это:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{in}}.
$$

Это не универсальные константы. Правильный gain зависит от активации,
параметризации residual branches и направления, в котором нужно сохранить
дисперсию. Проверка гистограмм активаций полезнее слепого копирования initializer.

## Что делает normalization

Для вектора признаков токена LayerNorm вычисляет

$$
\mu=\frac1d\sum_i x_i,
\qquad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{LayerNorm}(x)=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

$\gamma$ и $\beta$ обучаемы: нормализация не запрещает модели восстановить
нужный масштаб и сдвиг. Она задаёт более предсказуемую систему координат для
следующего подслоя.

BatchNorm усредняет статистики по пакету и пространственным позициям отдельно для
каждого канала. Она хорошо согласуется с CNN, но использует running statistics
при применении модели. LayerNorm нормализует признаки каждого примера независимо и
поэтому естественнее для последовательностей переменной длины.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/layer-normalization/batch-normalization.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/layer-normalization/layer-normalization.png]]

*Синие элементы участвуют в вычислении одних статистик. BatchNorm собирает их
по примерам батча для фиксированного признака, LayerNorm — по признакам одного
примера. Источник: Ba, Kiros, Hinton,
[Layer Normalization](https://arxiv.org/abs/1607.06450), Figure 2.*

Различие осей определяет поведение на инференсе. BatchNorm должен заменить
статистики текущего батча накопленными средними; результат может зависеть от
режима `train`/`eval`. LayerNorm вычисляет статистики непосредственно для
текущего токена и не нуждается в накопленном состоянии.

RMSNorm не вычитает среднее:

$$
\operatorname{RMSNorm}(x)=
\gamma\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}.
$$

Она контролирует среднеквадратичный масштаб и требует одну редукцию вместо
вычисления mean и variance. Большинство LLaMA-подобных моделей используют
RMSNorm, но это не доказательство универсального превосходства.

## Residual connection учит поправку

Обычный блок должен приблизить $F(x)$. Residual block возвращает

$$
y=x+G(x).
$$

Если преобразование не нужно, достаточно сделать $G(x)\approx0$; identity path
остаётся доступным напрямую.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-1-18/resnet-figure-2.png]]

*В исходной схеме ResNet преобразование $F(x)$ состоит из двух весовых слоёв,
а неизменённый $x$ обходит их и складывается с результатом. Фрагмент Figure 2
из статьи Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,
[Deep Residual Learning for Image Recognition, p. 2](https://arxiv.org/pdf/1512.03385#page=2).*

Для градиента:

$$
\frac{\partial y}{\partial x}=I+\frac{\partial G}{\partial x}.
$$

Identity term создаёт короткий путь, который не требует прохождения через все
операции residual branch. Это не гарантирует отсутствия любых проблем с
градиентами, но резко облегчает обучение глубоких сетей.

## Pre-norm и post-norm

Оригинальный Transformer использовал post-norm:

$$
y=\operatorname{Norm}(x+F(x)).
$$

Современные decoder-only LLM чаще используют pre-norm:

$$
y=x+F(\operatorname{Norm}(x)).
$$

В pre-norm identity path residual stream не проходит через normalization на
каждом блоке. Это обычно облегчает оптимизацию глубоких стеков. Цена — иная
динамика масштаба residual stream и необходимость финальной normalization перед
выходную голову.

## Диагностика

До большого запуска полезно построить по слоям:

- mean и standard deviation предактиваций;
- долю насыщенных sigmoid/tanh и нулевых ReLU;
- RMS скрытых состояний до и после residual addition;
- нормы градиентов и обновлений;
- отношение $\|\Delta W\|/\|W\|$.

Плоская кривая функции потерь не сообщает, где исчез сигнал. Профиль по глубине показывает слой,
после которого масштабы начали систематически расти или затухать.

## Краткие итоги

- Random initialization разрушает симметрию, а её масштаб сохраняет дисперсию.
- Xavier и He init соответствуют разным предположениям об активации.
- BatchNorm, LayerNorm и RMSNorm усредняют по разным осям и не взаимозаменяемы.
- Residual connection оставляет identity path и обучает поправку к нему.
- Pre-norm описывает положение normalization; RMSNorm — её конкретную формулу.

## Источники

- D2L, [Numerical Stability and Initialization](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html).
- D2L, [Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html).
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Zhang, Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).

**Дальше:** dropout и другие regularizers ограничивают зависимость модели от
конкретной обучающей выборки.
