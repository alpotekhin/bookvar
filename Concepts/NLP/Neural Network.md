---
title: "Neural Network"
aliases: [Neural Network, MLP, Multilayer Perceptron, нейросеть]
type: concept
category: NLP
papers: []
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 02 — Word Vectors|CS224N L02]]"
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 03 — Backprop|CS224N L03]]"
sources:
  - "[Rumelhart, Hinton, Williams — Learning representations by back-propagating errors (1986)](https://www.nature.com/articles/323533a0)"
  - "[Goodfellow, Bengio, Courville — Deep Learning Book (Chapter 6)](https://www.deeplearningbook.org/contents/mlp.html)"
  - "[Nielsen — Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)"
---

# Neural Network — многослойный перцептрон

## Зачем это нужно: нелинейные модели

Линейные модели ($y = Wx + b$) ограничены: они не могут выучить XOR, нелинейные границы, иерархические признаки. Нейронная сеть решает это композицией линейных слоёв с **нелинейной активацией** между ними.

Базовый блок — **многослойный перцептрон** (MLP, multilayer perceptron) или **feed-forward network**:

$$h_1 = \sigma(W_1 x + b_1)$$
$$h_2 = \sigma(W_2 h_1 + b_2)$$
$$y = W_L h_{L-1} + b_L$$

где $\sigma$ — **нелинейная** функция активации. Без неё композиция линейных слоёв оставалась бы линейной ($W_2(W_1 x) = (W_2 W_1) x$) — ничего бы не выиграли.

## Структура одного слоя

```
         W, b
x ─► [linear] ─► z ─► [activation σ] ─► h
    (Wx + b)
```

- **x** — вход размерности $d_{in}$
- **W** — матрица весов $\mathbb{R}^{d_{out} \times d_{in}}$
- **b** — bias $\mathbb{R}^{d_{out}}$
- **σ** — поэлементная нелинейность
- **h** — выход размерности $d_{out}$, передаётся на следующий слой

Число параметров слоя — $d_{in} \cdot d_{out} + d_{out}$. Сеть — последовательность таких слоёв.

## Функции активации

Нелинейность между слоями определяет выразительность и свойства обучения.

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Диапазон $(0, 1)$ — исторически использовалась в классификации
- **Проблемы:** saturation (градиент стремится к 0 при больших $|x|$), not zero-centered
- Сегодня: только на выходе для binary classification

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- Диапазон $(-1, 1)$, zero-centered
- Всё ещё saturates для больших $|x|$
- Используется в LSTM/GRU, классических RNN

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- Простая, быстрая, нет saturation для $x > 0$
- **De-facto стандарт** для CNN и MLP с 2012 (AlexNet)
- **Проблема:** "dying ReLU" — если нейрон получает отрицательный вход, градиент = 0 навсегда. Решения: Leaky ReLU ($\max(0.01x, x)$), PReLU, ELU

### GELU

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715 x^3)\right)\right)$$

где $\Phi$ — CDF стандартного нормального распределения. Гладкая версия ReLU — **стандарт в [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]** (BERT, GPT-2/3, T5).

### SwiGLU

$$\text{SwiGLU}(x) = (xW_1) \odot \sigma(xW_2)$$

Gated linear unit с Swish-активацией. Используется в современных LLM — LLaMA, Mistral, PaLM. Даёт 1-2% улучшение качества при тех же параметрах.

| Activation | Диапазон | Использование сегодня |
|-----------|----------|----------------------|
| Sigmoid | (0, 1) | Output binary classification |
| Tanh | (-1, 1) | LSTM/GRU |
| ReLU | [0, ∞) | CNN, ResNet |
| GELU | ≈ (-0.17, ∞) | Transformer (BERT, GPT) |
| SwiGLU | вся ось | LLaMA, Mistral (FFN) |

## Universal Approximation Theorem

**Теорема** (Cybenko 1989, Hornik 1991): MLP с **одним скрытым слоем** достаточной ширины и нелинейной активацией может приблизить **любую непрерывную функцию** на компакте с произвольной точностью.

Формально: для любой непрерывной $f: [0,1]^n \to \mathbb{R}$ и любого $\epsilon > 0$ существуют $N, W_1, b_1, W_2$ такие что:

$$\sup_{x \in [0,1]^n} \left| f(x) - W_2 \sigma(W_1 x + b_1) \right| < \epsilon$$

### Что это значит и не значит

**Значит:** MLP — универсальный аппроксиматор. В теории достаточно одного скрытого слоя.

**Не значит:**
- Что обучение такой сети простое — теорема не говорит о learnability, только о существовании весов
- Что узкая сеть справится — ширина может быть экспоненциальной
- Что глубина не нужна — на практике **глубокие** сети учатся эффективнее

Именно последний пункт объясняет переход от shallow MLP к deep learning.

## Почему глубокие сети: иерархические признаки

Теоретические результаты (Telgarsky 2016, Eldan & Shamir 2016) показывают: есть функции, для аппроксимации которых **глубокая** сеть требует **экспоненциально меньше** параметров, чем мелкая. Глубина = экспоненциальный compositional gain.

### Иерархия признаков

Эмпирическое наблюдение: каждый следующий слой выучивает более абстрактные признаки:

- **CNN на ImageNet:** слой 1 — границы, слой 2 — текстуры, слой 3 — части объектов, слой 4 — целые объекты
- **BERT:** нижние слои — синтаксис (POS, зависимости), верхние — семантика (coreference, reasoning) (Tenney et al., 2019)
- **LLM:** нижние — токены и grammar, средние — смысл, верхние — task-specific композиции

Глубина позволяет сети строить **композиционные представления**: `пиксели → границы → текстуры → части → объекты`. Эта иерархия сама появляется в ходе обучения без explicit supervision.

### Практические глубины

| Архитектура | Типичная глубина |
|-------------|------------------|
| Классический MLP | 2-5 слоёв |
| AlexNet (2012) | 8 |
| VGG (2014) | 16-19 |
| ResNet (2015) | 50-152 |
| BERT-large | 24 |
| GPT-3 | 96 |
| LLaMA-3 70B | 80 |

Увеличение глубины без архитектурных трюков (residual connections, layer norm) наталкивается на проблемы оптимизации — vanishing gradients, плохая обусловленность.

## Обучение: loss + backprop + gradient descent

Стандартный pipeline:

1. **Loss function:** сравнение выхода с target
   - Classification: cross-entropy $L = -\sum y_i \log \hat{y}_i$
   - Regression: MSE $L = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$
2. **[[02 Areas/ML & DL/Concepts/Training/Backpropagation|Backpropagation]]**: chain rule для $\nabla_\theta L$
3. **Optimizer**: градиентный спуск или его варианты (SGD, Adam, AdamW)

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

### Практические трюки

- **Initialization**: Xavier (tanh), He (ReLU) — контролируют распределение активаций
- **Batch Normalization** / [[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]] — стабилизируют обучение
- **[[02 Areas/ML & DL/Concepts/Training/Dropout|Dropout]]** — регуляризация
- **Residual connections** (ResNet, Transformer) — позволяют обучать очень глубокие сети

## Feed-Forward в Transformer

В [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] каждый блок содержит **FFN** — двухслойный MLP:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

С размерностями $d_{\text{model}} \to d_{ff} \to d_{\text{model}}$, обычно $d_{ff} = 4 \cdot d_{\text{model}}$. FFN применяется **independently** к каждой позиции. Это ~2/3 параметров Transformer — там живёт большая часть знаний модели (Geva et al., 2021 показали, что FFN — key-value memory).

Так что MLP — не реликт прошлого, а основной вычислительный блок современных LLM. Changed только то, что он стоит между self-attention слоями.

## Foundation для современных архитектур

Любая современная архитектура — композиция MLP с другими блоками:

- **CNN** — convolution + MLP head
- **RNN/LSTM** — MLP с shared weights по времени
- **Transformer** — attention + MLP (FFN) + residuals
- **Diffusion models** — U-Net = стек CNN + MLP
- **Mixture of Experts** — несколько MLP с learned routing

MLP остаётся базовым блоком — понимание его свойств (активации, инициализация, backprop) необходимо для работы с любой нейросетью.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Backpropagation|Backpropagation]] — алгоритм обучения нейросетей
- [[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]] — MLP внутри Transformer
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура на базе MLP + attention
- [[02 Areas/ML & DL/Concepts/Training/Dropout|Dropout]] — регуляризация для нейросетей
- [[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]] — нормализация активаций
