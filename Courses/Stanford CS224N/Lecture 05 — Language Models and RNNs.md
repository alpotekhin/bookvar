---
title: "CS224N — Lecture 5: Language Models and Recurrent Neural Networks"
course: "Stanford CS224N"
lecture: 5
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture05-rnnlm]]"
concepts: ["[[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]]", "[[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]]", "[[02 Areas/ML & DL/Concepts/NLP/N-gram|N-gram]]", "[[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]]"]
---

# Lecture 5: Language Models and Recurrent Neural Networks

> *"This is the most important concept in the class! It leads to BERT, GPT-3 and ChatGPT!"* -- слайд о Language Modeling

## Language Modeling: самый важный концепт курса

### Определение

**Language Model** -- система, предсказывающая следующее слово по предыдущему контексту:

$$P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$$

где $x^{(t+1)}$ -- любое слово из словаря $V = \{w_1, \ldots, w_{|V|}\}$.

Эквивалентно: LM присваивает **вероятность тексту**:

$$P(x^{(1)}, \ldots, x^{(T)}) = \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)})$$

### Где используется

LM -- основа почти всех современных NLP систем:
- **Autocompletion** (клавиатура телефона, Google Search suggestions)
- **Machine Translation** (P(target | source))
- **Speech Recognition** (P(text | audio))
- **Text Generation** (GPT, ChatGPT)
- **Summarization**, **QA**, **Code Generation**

## N-gram Language Models

### Markov assumption

N-gram LM: следующее слово зависит **только от $n-1$ предыдущих**:

$$P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)}) \approx P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(t-n+2)})$$

Вероятности оцениваются **подсчётом** в корпусе:

$$P(w \mid \text{students opened their}) = \frac{\text{count(students opened their } w)}{\text{count(students opened their)}}$$

Пример из слайдов: если в корпусе "students opened their books" встречается 400 раз, а "students opened their" -- 1000 раз, то P(books | students opened their) = 0.4.

### Проблемы n-gram моделей

**Sparsity Problem 1**: что если "students opened their *w*" никогда не встречается? Тогда $P(w) = 0$. Решение: **smoothing** -- добавить малое $\delta$ к каждому count.

**Sparsity Problem 2**: что если "students opened their" никогда не встречается? Нельзя посчитать *никакую* вероятность. Решение: **backoff** -- использовать более короткий контекст ("opened their").

**Фундаментальная проблема**: увеличение $n$ ухудшает sparsity. На практике $n > 5$ невозможно. Но контекст из 4 слов -- это катастрофически мало для понимания языка. Пример: "As the proctor started the clock, the students opened their ___" -- 4-gram модель игнорирует "proctor", который подсказывает "exams", а не "books".

### Storage

N-gram модель хранит **все наблюдённые n-gramы**. С ростом $n$ количество комбинаций растёт экспоненциально.

## Recurrent Neural Networks (RNN)

### Мотивация: преодоление фиксированного окна

| Подход | Проблема |
|--------|----------|
| N-gram | Фиксированное окно $n$, sparsity |
| Fixed-window NN | Фиксированное окно, не разделяет веса |
| **RNN** | **Произвольная длина контекста, shared weights** |

### Архитектура

На каждом шаге $t$:

$$h_t = \sigma(W_h \cdot h_{t-1} + W_x \cdot x_t + b_1)$$
$$\hat{y}_t = \text{softmax}(U \cdot h_t + b_2)$$

где:
- $x_t$ -- word embedding текущего слова
- $h_t$ -- hidden state (хранит "память" обо всём предыдущем контексте)
- $W_h, W_x, U$ -- **одни и те же веса на каждом шаге** (weight sharing)
- $\sigma$ -- нелинейность (tanh или ReLU)

### Преимущества RNN

1. **Произвольная длина контекста**: $h_t$ теоретически зависит от всех предыдущих $x_1, \ldots, x_t$
2. **Размер модели не растёт** с длиной входа (те же $W_h, W_x$)
3. **Симметрия по шагам**: каждый шаг использует одну и ту же "функцию перехода"

### Недостатки RNN

1. **Последовательное вычисление**: $h_t$ зависит от $h_{t-1}$ -- **нельзя параллелизовать**
2. **На практике контекст забывается**: информация затухает через длинные последовательности (vanishing gradients)
3. **Медленное обучение**: Backpropagation Through Time (BPTT)

## Обучение RNN

### Teacher Forcing

На каждом шаге подаём **ground truth** предыдущего слова (а не то, что модель предсказала). Loss: cross-entropy на каждом шаге:

$$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J^{(t)} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)}; \theta)$$

### Backpropagation Through Time (BPTT)

Стандартный backprop, но "развёрнутый" по всем временным шагам. На практике: **truncated BPTT** -- градиент вычисляется по окну из $k$ шагов, а не по всему корпусу.

### Проблема исчезающих/взрывающихся градиентов

При BPTT градиент проходит через умножение на $W_h$ на каждом шаге:

$$\frac{\partial J^{(T)}}{\partial h^{(1)}} = \prod_{t=2}^{T} \frac{\partial h^{(t)}}{\partial h^{(t-1)}}$$

Если собственные значения $W_h$ < 1 -- градиент **затухает** (vanishing). Если > 1 -- **взрывается** (exploding).

Решения:
- **Gradient clipping**: ограничить норму градиента (для exploding)
- **LSTM / GRU**: специальные архитектуры с gating (для vanishing) -- Lecture 6
- **[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]**: полный отказ от рекуррентности -- Lecture 8

## [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]]

### Определение

$$\text{perplexity} = \exp(J(\theta)) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(x^{(t)})\right)$$

Интуиция: средняя "разветвлённость" при предсказании. Perplexity = 10 означает, что в среднем модель "выбирает" из 10 равновероятных вариантов.

**Чем ниже perplexity -- тем лучше модель**. Идеальная модель: perplexity = 1 (всегда предсказывает правильный токен с вероятностью 1). Random uniform: perplexity = |V| (размер словаря).

Perplexity -- стандартная метрика для оценки language models (до GPT-4 era, когда перешли на downstream benchmarks).

## Применения RNN

RNN Language Model -- это не только генерация текста. Архитектуру можно применять к:
- **Part-of-speech tagging**: $x_t$ -> tag_t на каждом шаге
- **Sentiment analysis**: последний $h_T$ как representation предложения
- **Machine translation** (encoder-decoder): encoder-RNN -> decoder-RNN
- **Speech recognition**: аудио фреймы -> текст

## От RNN к Transformers (preview)

RNN были **доминирующей архитектурой** для NLP в 2013-2017. Три ключевых проблемы мотивировали переход к [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]:

1. **Последовательность**: нельзя параллелизовать -> медленное обучение на GPU
2. **Длинные зависимости**: vanishing gradients -> теряет контекст
3. **Interaction distance**: O(n) шагов для связи далёких слов

Transformer решает все три: O(1) interaction distance, полная параллелизация, self-attention вместо рекуррентности.

## Concepts covered

- [[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]] -- $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$, основа всего современного NLP
- [[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]] -- рекуррентная нейросеть с shared weights
- [[02 Areas/ML & DL/Concepts/NLP/N-gram|N-gram]] -- простейшая LM, подсчёт в корпусе
- [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]] -- exp(cross-entropy), метрика LM
