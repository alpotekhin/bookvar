---
title: Нейрон, градиент и backpropagation
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://github.com/karpathy/micrograd
  - https://colah.github.io/posts/2015-08-Backprop/
---

# Нейрон, градиент и backpropagation

> [!abstract] Идея главы
> Нейросеть — это большая математическая функция. Обучение начинается с простого
> вопроса: как слегка изменить каждый параметр, чтобы ответ функции стал лучше?
> Backpropagation отвечает на него, проходя вычисления в обратном порядке.

Лучший практический вход в тему — первая лекция
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). Карпати
не начинает с матриц и слоёв: он берёт одно число, считает производную, а затем
постепенно собирает маленький autograd engine `micrograd`.

## 1. Что говорит производная

Возьмём:

$$
f(x)=x^2.
$$

При $x=3$ небольшое увеличение $x$ примерно в шесть раз сильнее изменяет $f$:

$$
f(3.001)-f(3)\approx 0.006.
$$

Именно это сообщает производная:

$$
\frac{df}{dx}=2x,\qquad \frac{df}{dx}\bigg|_{x=3}=6.
$$

Производная — локальная чувствительность результата к изменению входа. Для
обучения нас интересует чувствительность loss к каждому параметру сети.

## 2. Один вычислительный граф

Пусть:

$$
a=2,\qquad b=-3,\qquad c=10,
$$

$$
d=a\cdot b,\qquad L=d+c.
$$

Forward pass:

$$
d=-6,\qquad L=4.
$$

Теперь спросим: как изменится $L$, если немного изменить каждый исходный input?

Для сложения:

$$
\frac{\partial L}{\partial d}=1,\qquad
\frac{\partial L}{\partial c}=1.
$$

Для умножения:

$$
\frac{\partial d}{\partial a}=b=-3,\qquad
\frac{\partial d}{\partial b}=a=2.
$$

По chain rule:

$$
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial d}
\frac{\partial d}{\partial a}
=1\cdot(-3)=-3,
$$

$$
\frac{\partial L}{\partial b}=1\cdot2=2.
$$

Мы прошли граф назад: каждый узел взял пришедший градиент, умножил его на свою
локальную производную и передал дальше.

## 3. Почему не считать производную каждого параметра отдельно

Численную производную можно оценить так:

$$
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta_i+\varepsilon)-L(\theta_i)}{\varepsilon}.
$$

Но для миллиарда параметров понадобятся примерно миллиард дополнительных
forward passes. Backprop повторно использует общие части вычислений и получает
все производные за один обратный проход.

В статье
[Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)
Chris Olah показывает backprop именно как dynamic programming на графе:
промежуточный градиент вычисляется один раз и переиспользуется всеми ветвями,
которые через него проходят.

## 4. Chain rule в одной строке

Если:

$$
L=f(g(x)),
$$

то:

$$
\frac{dL}{dx}
=
\frac{dL}{dg}
\frac{dg}{dx}.
$$

Для длинной цепочки локальные производные перемножаются. Если один tensor
используется в нескольких ветвях, вклады складываются:

$$
\frac{\partial L}{\partial x}
=
\sum_k
\frac{\partial L}{\partial y_k}
\frac{\partial y_k}{\partial x}.
$$

Это объясняет важный баг, который Карпати специально разбирает в micrograd:
градиент нельзя перезаписывать, если значение использовалось дважды, — его нужно
накапливать.

## 5. От числа к нейрону

Нейрон сначала считает взвешенную сумму:

$$
z=w^\top x+b,
$$

а затем применяет нелинейность:

$$
y=\tanh(z)
$$

или ReLU, GELU, SiLU.

Если убрать нелинейности между слоями:

$$
W_2(W_1x)=W'x.
$$

Сколько бы линейных слоёв мы ни поставили, они свернутся в одну матрицу.
Нелинейности позволяют сети строить функции, которые нельзя описать одним
линейным преобразованием.

## 6. Что хранит micrograd

Минимальный объект `Value` содержит:

- числовое значение `data`;
- накопленный градиент `grad`;
- ссылки на значения, из которых он получен;
- операцию, создавшую узел;
- функцию локального backward.

Упрощённая идея:

```python
class Value:
    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(children)
        self._backward = lambda: None

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
```

После forward граф сортируется топологически. Backward начинается с:

```python
loss.grad = 1.0
```

и вызывает локальные `_backward()` в обратном порядке.

Полный [micrograd](https://github.com/karpathy/micrograd) — около сотни строк
существенного кода. PyTorch делает то же для tensors, GPU kernels и тысяч
операций.

## 7. Из градиента получается обучение

Пусть $\theta$ — все параметры, а $L(\theta)$ — loss. Gradient descent:

$$
\theta\leftarrow\theta-\eta\nabla_\theta L.
$$

Если производная положительна, уменьшение параметра локально уменьшит loss. Если
отрицательна — параметр нужно увеличить. Learning rate $\eta$ задаёт размер
шага.

Минимальный training loop:

```python
for x, y in data:
    prediction = model(x)
    loss = criterion(prediction, y)

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= learning_rate * p.grad
```

`backward()` ничего не «обучает» сам по себе: он только считает градиенты.
Параметры меняет optimizer step.

## 8. Почему градиенты нужно обнулять

PyTorch и micrograd складывают новые значения с уже лежащими в `.grad`. Это
нужно, когда параметр участвует в нескольких ветвях графа или когда намеренно
накапливаются несколько mini-batches.

Но в обычном training loop забытый `zero_grad()` смешивает градиенты разных
шагов:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Это одна из тех ошибок, при которых программа работает, но обучает не ту модель,
которую вы думаете.

## 9. Reverse-mode autodiff

Нейросеть отображает огромное число параметров в один scalar loss:

$$
f:\mathbb{R}^{N}\rightarrow\mathbb{R}.
$$

Reverse-mode autodiff особенно удобен именно здесь: один backward pass даёт
градиент scalar output по всем $N$ inputs. Backpropagation — применение этого
режима к вычислительному графу нейросети.

## 10. Что может пойти не так

### Vanishing gradients

В длинной цепочке перемножаются числа меньше единицы, и ранние слои почти не
получают сигнала.

### Exploding gradients

Произведение больших Jacobians создаёт огромные значения. Gradient clipping
ограничивает норму, но не исправляет причину нестабильности.

### NaN

Частые причины: слишком большой learning rate, overflow в низкой точности,
деление на ноль и неверные маски.

### Неверный граф

`detach()` или `torch.no_grad()` могут случайно разорвать нужный путь.
И наоборот, отсутствие `no_grad()` на inference зря хранит активации.

## 11. Как проверить собственный backward

Сравните аналитический градиент с центральной разностью:

$$
\frac{\partial f}{\partial x_i}
\approx
\frac{f(x+\varepsilon e_i)-f(x-\varepsilon e_i)}
{2\varepsilon}.
$$

Такой gradient check медленный, но очень полезен для небольшой новой операции.
Другой простой тест — сравнить свой scalar engine с PyTorch на том же выражении.

## Практический маршрут

1. Посмотреть первые 90 минут
   [лекции micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).
2. Самостоятельно реализовать `Value`, `+`, `*`, `tanh` и topological backward.
3. Воспроизвести тот же граф в PyTorch.
4. Обучить маленький MLP на нескольких двумерных точках.
5. Намеренно убрать `zero_grad()` и увидеть ошибку.

## Что должно остаться после главы

- Производная показывает локальную чувствительность результата к входу.
- Нейросеть можно представить вычислительным графом.
- Backprop идёт по графу назад и применяет chain rule.
- Градиенты ветвей складываются.
- `backward()` считает градиенты; optimizer изменяет параметры.
- Autograd скрывает рутину, но не отменяет необходимость понимать граф.

## Источники

- [Andrej Karpathy — The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Chris Olah — Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)
- [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 03 — Backpropagation and Neural Networks|CS224N: Backpropagation]]
- [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

