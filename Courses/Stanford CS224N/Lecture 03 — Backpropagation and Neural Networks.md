---
title: "CS224N — Lecture 3: Neural Net Learning: Gradients by Hand and Algorithmically"
course: "Stanford CS224N"
lecture: 3
type: course-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture03-neuralnets]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Training/Backpropagation|Backpropagation]]", "[[02 Areas/ML & DL/Concepts/Training/Chain Rule|Chain Rule]]", "[[02 Areas/ML & DL/Concepts/Training/Jacobian|Jacobian]]", "[[02 Areas/ML & DL/Concepts/NLP/Neural Network|Neural Network]]", "[[02 Areas/ML & DL/Concepts/NLP/Activation Functions|Activation Functions]]"]
---

# Lecture 3: Backpropagation and Neural Networks

> *"Assignment 2 makes sure you really understand the math of neural networks... then we'll let the software do it!"* -- Christopher Manning

Лектор: Christopher Manning.

## Нейросеть как многослойный классификатор

### Один слой нейросети

Нейросеть -- это цепочка логистических регрессий с нелинейными активациями:

$$z = Wx + b$$
$$a = f(z)$$

где $f$ -- нелинейная функция активации, применяемая **поэлементно**:

$$f([z_1, z_2, z_3]) = [f(z_1), f(z_2), f(z_3)]$$

### Зачем нужны нелинейности?

**Без нелинейностей** глубокая сеть сводится к одному линейному преобразованию:

$$W_1 W_2 x = Wx$$

Любое количество линейных слоёв "компилируется" в одну матрицу. **С нелинейностями** нейросеть может аппроксимировать **любую** непрерывную функцию (Universal Approximation Theorem).

## [[02 Areas/ML & DL/Concepts/NLP/Activation Functions|Функции активации]]

### Классические

| Функция | Формула | Диапазон | Особенности |
|---------|---------|----------|-------------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $[0, 1]$ | Для вероятностей, vanishing gradient |
| **tanh** | $\tanh(z) = 2\sigma(2z) - 1$ | $[-1, 1]$ | В 2x круче sigmoid, zero-centered |
| **ReLU** | $\max(z, 0)$ | $[0, \infty)$ | Быстрое обучение, хороший gradient flow |

### Современные

| Функция | Формула | Где используется |
|---------|---------|-----------------|
| **Leaky ReLU** | $\max(z, 0.01z)$ | Решает "dead neuron" проблему ReLU |
| **Swish** | $x \cdot \sigma(x)$ | Гладкая альтернатива ReLU |
| **GELU** | $x \cdot P(X \leq x)$, $X \sim N(0,1)$ | BERT, GPT, современные [[02 Areas/ML & DL/Concepts/Architectures/Transformer\|Transformer]]-ы |

GELU $\approx x \cdot \sigma(1.702x)$ -- гладкая аппроксимация ReLU, ставшая стандартом для Transformer-архитектур.

**Практическое правило**: для глубоких сетей начинайте с **ReLU**. Для Transformers -- **GELU/Swish**.

## Matrix Calculus: вычисление градиентов

### Градиент функции одной переменной

Для $f(x) = x^3$: производная $\frac{df}{dx} = 3x^2$. Интерпретация: "на сколько изменится выход при малом изменении входа?" При $x = 1$: $1.01^3 \approx 1.03$ (изменение в $\sim 3$ раза больше).

### Градиент вектор-функции

Для функции $f: \mathbb{R}^n \to \mathbb{R}$, градиент -- **вектор** частных производных:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$

### [[02 Areas/ML & DL/Concepts/Training/Jacobian|Jacobian]]: обобщение градиента

Для функции $f: \mathbb{R}^n \to \mathbb{R}^m$, Jacobian -- матрица $m \times n$:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

### Jacobian element-wise функции

Для $h = f(z)$, где $f$ применяется поэлементно:

$$\frac{\partial h}{\partial z} = \text{diag}(f'(z))$$

Jacobian -- **диагональная** матрица производных $f'(z_i)$. Это ключевой факт для эффективного backprop.

## [[02 Areas/ML & DL/Concepts/Training/Chain Rule|Chain Rule]] для матриц

### Одномерный случай

Для композиции: $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$ -- умножаем производные.

### Многомерный случай

Для $f: \mathbb{R}^n \to \mathbb{R}^m$ и $g: \mathbb{R}^m \to \mathbb{R}^p$:

$$\frac{\partial}{\partial x} g(f(x)) = \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}$$

Произведение Jacobian-ов. Размерности: $(p \times m) \cdot (m \times n) = p \times n$.

## Пример: градиенты NER-классификатора

### Постановка задачи

Бинарная классификация "является ли центральное слово location":

$$x = [x_{museums}, x_{in}, x_{Paris}, x_{are}, x_{amazing}] \in \mathbb{R}^{5d}$$
$$z = Wx + b$$
$$h = f(z)$$
$$s = u^T h$$

### Пошаговый backprop

Разбиваем на простые шаги и применяем chain rule:

**Шаг 1**: $\frac{\partial s}{\partial u} = h$ (прямо из определения $s = u^T h$)

**Шаг 2**: $\frac{\partial s}{\partial h} = u^T$ (Jacobian линейной функции)

**Шаг 3**: $\frac{\partial s}{\partial z} = \frac{\partial s}{\partial h} \cdot \frac{\partial h}{\partial z} = u^T \cdot \text{diag}(f'(z))$ -- для element-wise $f$

**Шаг 4**: $\frac{\partial s}{\partial W}$ -- здесь возникает **shape convention**

### Shape Convention vs Jacobian Form

Два подхода:
1. **Jacobian form**: строгая математика, chain rule через умножение матриц. Удобно для вычислений.
2. **Shape convention**: градиент имеет **ту же форму**, что и параметры. Удобно для SGD update.

Результат для $W$:

$$\frac{\partial s}{\partial W} = \delta \cdot x^T$$

где $\delta = u \odot f'(z)$ -- upstream gradient ("error signal"), $x$ -- local input. **Внешнее произведение** error signal и input.

### Производная по bias

$$\frac{\partial s}{\partial b} = \delta$$

Upstream gradient передаётся напрямую (Jacobian bias -- единичная матрица).

## [[02 Areas/ML & DL/Concepts/Training/Backpropagation|Backpropagation]]

### Ключевая идея

Backpropagation -- это **алгоритмическое** применение chain rule с **повторным использованием промежуточных вычислений** (dynamic programming).

При вычислении $\frac{\partial s}{\partial W}$ и $\frac{\partial s}{\partial x}$ общая часть ($\delta$) вычисляется **один раз**:

$$\delta = \frac{\partial s}{\partial z} = u \odot f'(z)$$

Это "error signal", который **передаётся назад** от выхода ко входу.

### Computation Graph

Нейросеть представляется как **направленный ациклический граф** (DAG):
- **Source nodes**: входы ($x$, $W$, $b$)
- **Interior nodes**: операции ($+$, $\times$, $f$)
- **Рёбра**: передают результат операции (forward) и градиент (backward)

### Forward propagation

Проход от входов к выходу -- вычисление функции.

### Backward propagation (backprop)

Проход от выхода к входам:

1. Для каждого узла: принять **upstream gradient** $\delta$
2. Вычислить **local gradient** (Jacobian операции)
3. Передать вниз: **downstream gradient** = upstream $\times$ local

```
Forward:  x → [Wx + b] → z → [f(z)] → h → [u^T h] → s
Backward: ∂s/∂x ← δ_z ← δ_h ← 1
```

### Узлы с несколькими выходами

Если узел используется в нескольких местах, upstream градиенты **суммируются** (multivariate chain rule):

$$\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x}$$

## Практические советы

### Проверка размерностей

**Трюк**: если $W \in \mathbb{R}^{n \times m}$, то $\frac{\partial L}{\partial W} \in \mathbb{R}^{n \times m}$ (shape convention). Используйте это для проверки правильности вычислений -- если размерности не сходятся, ошибка в формуле.

### Gradient checking

Численная проверка: $\frac{\partial f}{\partial x_i} \approx \frac{f(x + h \cdot e_i) - f(x - h \cdot e_i)}{2h}$. Сравните с аналитическим градиентом -- если относительная разница $> 10^{-5}$, есть ошибка.

### Автоматическое дифференцирование

На практике PyTorch/JAX вычисляют gradients **автоматически** через computation graph. Но понимание ручного backprop критично для отладки и архитектурного дизайна.

## NER: практический пример

### Задача

Классификация слов в тексте: *"Last night, **Paris Hilton** wowed in a sequin gown."* vs *"Samuel Quinn was arrested in the **Hilton Hotel** in **Paris**."*

### Приложения NER
- Tracking mentions в документах
- [[02 Areas/ML & DL/Concepts/NLP/Question Answering|Question Answering]]: ответы часто -- именованные сущности
- Sentiment analysis по отношению к конкретной сущности
- Entity Linking в Knowledge Base (Wikidata)

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Training/Backpropagation|Backpropagation]] -- алгоритм обратного распространения ошибки через computation graph
- [[02 Areas/ML & DL/Concepts/Training/Chain Rule|Chain Rule]] -- цепное правило для композиции функций, умножение Jacobian-ов
- [[02 Areas/ML & DL/Concepts/Training/Jacobian|Jacobian]] -- матрица частных производных $m \times n$
- [[02 Areas/ML & DL/Concepts/NLP/Neural Network|Neural Network]] -- последовательность линейных слоёв с нелинейностями
- [[02 Areas/ML & DL/Concepts/NLP/Activation Functions|Activation Functions]] -- sigmoid, tanh, ReLU, GELU, Swish
