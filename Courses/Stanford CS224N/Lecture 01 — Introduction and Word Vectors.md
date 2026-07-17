---
title: "CS224N — Lecture 1: Introduction and Word Vectors"
course: "Stanford CS224N"
lecture: 1
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture01-wordvecs1]]"
concepts: ["[[02 Areas/ML & DL/Concepts/NLP/Word Embeddings|Word Embeddings]]", "[[02 Areas/ML & DL/Concepts/NLP/Word2Vec|Word2Vec]]", "[[02 Areas/ML & DL/Concepts/NLP/Distributional Semantics|Distributional Semantics]]"]
---

# Lecture 1: Introduction and Word Vectors

> *"The (astounding!) result that word meaning can be represented rather well by a high-dimensional vector of real numbers"* -- Christopher Manning

## О курсе

Stanford CS224N (Spring 2024), Christopher Manning. Курс охватывает:
1. Основы: word vectors, feed-forward networks, RNNs, attention
2. Современные методы: [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformers]], pretraining, RLHF, SFT, efficient adaptation, agents

Assignments: 4 штуки (PyTorch), финальный проект (implement BERT и fine-tune, или custom research project).

## Человеческий язык и задача представления смысла

### Почему NLP важен

Язык -- это discrete/symbolic/categorical signal system, в отличие от vision или speech (continuous). Именно дискретность делает NLP уникально сложным.

Современные приложения: machine translation (Google Translate), free-text QA (YONO, Lee et al. 2021 -- T5-Large 3 раза для retrieval/reranking/reading), ChatGPT/GPT-4 (generation, vision), image generation (DALL-E, Sora).

### Denotational vs Distributional semantics

**Denotational semantics** -- слово как символ, указывающий на объект или идею: `tree <=> {деревья}`. Формализация: signifier (symbol) <=> signified (idea or thing).

**Distributional semantics** -- *"You shall know a word by the company it keeps"* (J.R. Firth, 1957). Значение слова определяется его **контекстом**. Это фундамент для word vectors.

## Проблема дискретного представления слов

### One-hot vectors

В классическом NLP слова -- дискретные символы:
```
motel = [0 0 0 0 0 0 0 0 1 0 0 0]
hotel = [0 0 0 0 0 1 0 0 0 0 0 0]
```

Размерность = размер словаря (500,000+). Проблема: **все вектора ортогональны** друг другу. `motel · hotel = 0` -- нет понятия сходства.

### WordNet

Ручной тезаурус с synonym sets и hypernym/hyponym ("is-a") связями. Код:
```python
from nltk.corpus import wordnet as wn
panda = wn.synset("panda.n.01")
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))
# [Synset('procyonid.n.01'), Synset('carnivore.n.01'), ...]
```

Проблемы WordNet:
- **Нет нюансов**: "proficient" как синоним "good" -- только в некоторых контекстах
- **Нет новых слов**: wicked, badass, nifty, wizard, ninja -- невозможно поддерживать актуальность
- **Субъективность**: offensive synonyms без учёта connotation
- **Ручной труд**: дорого создавать и поддерживать
- **Нет word similarity**: нельзя вычислить, насколько "good" похоже на "great"

## Word Vectors: распределённые представления

### Идея

Представить каждое слово плотным вещественным вектором (обычно 50-300 размерностей), так что **похожие слова имеют близкие вектора**.

Каждая компонента вектора -- не интерпретируемый атрибут (как в one-hot), а часть распределённого представления. Семантика "размазана" (distributed) по всем компонентам.

### [[02 Areas/ML & DL/Concepts/NLP/Word2Vec|Word2Vec]] (Mikolov et al., 2013)

**Skip-gram модель**: для каждой позиции $t$ в корпусе, предсказать слова в контекстном окне размера $m$ по центральному слову.

**Целевая функция** (максимизируем likelihood):

$$L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \leq j \leq m \\ j \neq 0}} P(w_{t+j} \mid w_t; \theta)$$

Что эквивалентно минимизации negative log-likelihood:

$$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P(w_{t+j} \mid w_t; \theta)$$

**Вероятность контекстного слова** через softmax:

$$P(o \mid c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

где $v_c$ -- center word vector, $u_o$ -- outside (context) word vector. **Два набора векторов** для каждого слова -- после обучения обычно усредняются.

### Интуиция

Что делает обучение: если слова "hotel" и "motel" часто появляются в **похожих контекстах** ("located near the...", "booked a room at the..."), их вектора будут **близкими**. Модель вынуждена сжимать всю информацию о контексте в плотный вектор.

### Знаменитые аналогии

Word vectors захватывают семантические направления:
- `king - man + woman ≈ queen`
- `Paris - France + Italy ≈ Rome`

Это не запрограммировано -- это **эмерджентное свойство** обучения на предсказание контекста.

## Gradient Descent и оптимизация

### Gradient Descent

Итеративное обновление параметров:

$$\theta^{new} = \theta^{old} - \alpha \nabla_\theta J(\theta)$$

где $\alpha$ -- learning rate.

### Stochastic Gradient Descent (SGD)

Вычисление градиента по **всему корпусу** слишком дорого (корпус может быть миллиарды слов). SGD: обновление на каждом **mini-batch** (подмножестве данных). Шумный, но гораздо быстрее и на практике работает хорошо.

## Практические аспекты DL (из слайдов)

### Regularization

L2 regularization:

$$J(\theta) = \frac{1}{N} \sum_{i=1}^{N} -\log\left(\frac{e^{f_{y_i}}}{\sum_{c=1}^{C} e^{f_c}}\right) + \lambda \sum_k \theta_k^2$$

Классический взгляд: борьба с overfitting. Современный взгляд: produces models that generalize well -- мы **не боимся** overfitting на train data, если модель достаточно большая.

### Dropout (Srivastava et al., 2012/2014)

- **Training**: случайно обнуляем каждый вход с вероятностью $p$ (обычно 0.5, для input layer ~0.15)
- **Test**: умножаем все веса на $(1-p)$
- Prevents feature co-adaptation, works as ensemble model (model bagging)

### Xavier Initialization

$$\text{Var}(W_i) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

Инициализация весов так, чтобы дисперсия сигнала не взрывалась и не затухала при прохождении через слои. Позже заменено layer normalization в Transformers.

### Optimizers

Adam -- хорошая отправная точка (lr ~ 0.001). Семейство: Adagrad -> RMSprop -> Adam -> AdamW -> NAdamW. Adaptive per-parameter learning rates.

### Vectorization

Матричные операции вместо циклов: numpy матричное умножение в 12x быстрее поэлементного цикла на CPU, на GPU разница 1-2 порядка.

## Concepts covered

- [[02 Areas/ML & DL/Concepts/NLP/Word Embeddings|Word Embeddings]] -- плотные векторы для слов вместо one-hot encoding
- [[02 Areas/ML & DL/Concepts/NLP/Word2Vec|Word2Vec]] -- Skip-gram и CBOW модели (Mikolov et al., 2013)
- [[02 Areas/ML & DL/Concepts/NLP/Distributional Semantics|Distributional Semantics]] -- "a word is characterized by the company it keeps"
- [[02 Areas/ML & DL/Concepts/NLP/WordNet|WordNet]] -- ручной тезаурус, проблемы rule-based ресурсов
