---
title: "CS224N — Lecture 2: Word Vectors, Word Senses, and Neural Classifiers"
course: "Stanford CS224N"
lecture: 2
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture02-wordvecs2]]"
concepts: ["[[02 Areas/ML & DL/Concepts/NLP/Word2Vec|Word2Vec]]", "[[02 Areas/ML & DL/Concepts/NLP/GloVe|GloVe]]", "[[02 Areas/ML & DL/Concepts/NLP/Negative Sampling|Negative Sampling]]", "[[02 Areas/ML & DL/Concepts/NLP/Co-occurrence Matrix|Co-occurrence Matrix]]", "[[02 Areas/ML & DL/Concepts/NLP/SVD|SVD]]"]
---

# Lecture 2: Word Vectors, Word Senses, and Neural Classifiers

> *"Doing no more than this, this algorithm learns word vectors that capture well word similarity and meaningful directions in a word space!"* -- Christopher Manning

Лектор: Christopher Manning.

## Оптимизация: Gradient Descent и SGD

### Gradient Descent

Имеем функцию потерь $J(\theta)$, которую хотим минимизировать. Алгоритм:

$$\theta^{new} = \theta^{old} - \alpha \nabla_\theta J(\theta)$$

где $\alpha$ -- learning rate (шаг обучения).

### Stochastic Gradient Descent (SGD)

Проблема: $J(\theta)$ зависит от **всех** окон в корпусе (миллиарды!). Вычислять полный градиент слишком дорого. Решение -- **SGD**: сэмплируем mini-batch, обновляем параметры после каждого.

## Recap: Word2Vec

### Основная идея

1. Начинаем со случайных word vectors
2. Итерируемся по каждой позиции в корпусе
3. Предсказываем окружающие слова через softmax:

$$P(o \mid c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

4. Обновляем вектора, чтобы лучше предсказывать реальные контекстные слова

Две матрицы параметров: $U$ (outside/context vectors) и $V$ (center vectors). Модель "bag of words" -- делает одинаковые предсказания в каждой позиции.

### Два варианта модели

1. **Skip-gram (SG)**: предсказываем context words по center word
2. **Continuous Bag of Words (CBOW)**: предсказываем center word по context words

## [[02 Areas/ML & DL/Concepts/NLP/Negative Sampling|Negative Sampling]] (Mikolov et al. 2013)

### Проблема softmax

Знаменатель $\sum_{w \in V} \exp(u_w^T v_c)$ требует суммирования по **всему** словарю ($|V|$ может быть 100K+). Это слишком дорого.

### Решение: бинарная логистическая регрессия

Вместо мультиклассовой классификации обучаем **бинарный** классификатор -- отличать настоящую пару (center, context) от "шумовых" пар:

$$J_{neg-sample}(u_o, v_c, U) = -\log \sigma(u_o^T v_c) - \sum_{k \in \{K\ negative\ samples\}} \log \sigma(-u_k^T v_c)$$

где $\sigma(x) = \frac{1}{1 + e^{-x}}$ -- sigmoid function.

### Сэмплирование негативных примеров

Негативные примеры сэмплируются с вероятностью:

$$P(w) = \frac{U(w)^{3/4}}{Z}$$

Степень $3/4$ делает редкие слова чуть более частыми в сэмплах, а частые -- чуть менее. Это эмпирически работает лучше, чем uniform или unigram distribution.

### Sparse updates

С negative sampling в каждом окне обновляем только $2m + 1$ слов (окно) плюс $2km$ негативных примеров. Обновления **разреженные** -- важно для distributed computing (не нужно отправлять гигантские обновления по сети).

## Count-based методы: Co-occurrence Matrix

### Идея

Зачем итерироваться по корпусу много раз? Можно **напрямую** собрать статистику совместной встречаемости слов.

### Два типа матриц

1. **Window-based**: аналогично word2vec, считаем co-occurrence в окне вокруг слова. Захватывает синтаксическую и семантическую информацию ("word space")
2. **Word-document**: матрица word $\times$ document. Даёт общие темы (все спортивные термины будут похожи) -- "Latent Semantic Analysis" ("document space")

### Проблемы с raw counts

Сырые вектора co-occurrence:
- **Высокая размерность**: растёт с размером словаря
- **Разреженность**: проблемы для downstream моделей
- **Function words** (the, he, has) доминируют -- синтаксис затмевает семантику

### [[02 Areas/ML & DL/Concepts/NLP/SVD|SVD]]: снижение размерности

**Singular Value Decomposition**: $X = U\Sigma V^T$, где $U$ и $V$ -- ортонормальные матрицы. Берём первые $k$ сингулярных значений:

$$X_k \approx U_k \Sigma_k V_k^T$$

$X_k$ -- наилучшее rank-$k$ приближение к $X$ по метрике наименьших квадратов. Результат: **плотные** вектора размерности 25-1000.

### Хаки для улучшения SVD (Rohde et al. 2005, COALS)

Сырой SVD на raw counts работает **плохо**! Нужны трансформации:
- **Log** частот
- **Clipping**: $\min(X, t)$, обычно $t \approx 100$
- Игнорирование function words
- **Ramped windows**: ближние слова весят больше
- **Pearson correlations** вместо counts (отрицательные значения -> 0)

## [[02 Areas/ML & DL/Concepts/NLP/GloVe|GloVe]] (Pennington, Socher, Manning, EMNLP 2014)

### Ключевой insight: отношения co-occurrence вероятностей

Отношения $P(x|ice) / P(x|steam)$ кодируют семантические компоненты:

| $x =$ | solid | gas | water | random |
|--------|-------|-----|-------|--------|
| $P(x \mid ice)$ | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(x \mid steam)$ | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| **Ratio** | **8.9** | **$8.5 \times 10^{-2}$** | **1.36** | **0.96** |

Для "solid": ratio >> 1 (ассоциация с ice). Для "gas": ratio << 1 (ассоциация с steam). Для нейтральных слов: ratio $\approx$ 1.

### Формулировка GloVe

Log-bilinear модель с vector differences:

$$w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log X_{ij}$$

Функция потерь:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

где $f(X_{ij})$ -- весовая функция, подавляющая слишком частые пары. GloVe объединяет **преимущества count-based и prediction-based** методов: быстрое обучение, масштабируемость на огромные корпуса.

## Оценка Word Vectors

### Intrinsic evaluation

**Аналогии**: $a : b :: c : ?$ -- ищем $d = \arg\max_{x} \cos(x, b - a + c)$

Пример: man : woman :: king : ? -> **queen**

Результаты GloVe на аналогиях показывают, что **семантические компоненты** кодируются как **линейные направления** в пространстве.

**Word Similarity**: корреляция cosine distances с человеческими оценками (WordSim353, SimLex-999).

| Model | Size | WS353 | MC | RG |
|-------|------|-------|----|----|
| SVD | 6B | 35.3 | 35.1 | 42.5 |
| SVD-L | 6B | 65.7 | 72.7 | 75.1 |
| GloVe | 6B | 65.8 | 72.7 | 77.8 |
| GloVe | 42B | **75.9** | **83.6** | **82.9** |

### Extrinsic evaluation

Оценка на реальной задаче -- например, **Named Entity Recognition** (NER). GloVe vectors дают F1 = 93.2 на dev set (лучше всех альтернатив).

## Word Senses и полисемия

### Проблема: одно слово -- много значений

Пример: **pike** -- острый наконечник, рыба, шоссе, поза в прыжках в воду, в австралийском английском "отказаться"...

Word2vec создаёт **один** вектор на слово -- это суперпозиция всех значений.

### Кластеризация контекстов (Huang et al. 2012)

Идея: кластеризовать окна вокруг слова, переобучить модель с отдельными кластерами: bank$_1$, bank$_2$ и т.д.

### Линейная суперпозиция значений (Arora et al., TACL 2018)

Удивительный результат: разные значения слова живут в **линейной суперпозиции** в стандартных word embeddings:

$$v_{pike} = \alpha_1 v_{pike_1} + \alpha_2 v_{pike_2} + \alpha_3 v_{pike_3}$$

где $\alpha_i = \frac{f_i}{f_1 + f_2 + f_3}$ пропорционально частоте значения. Благодаря идеям из sparse coding, отдельные значения **можно выделить** (если они достаточно частые).

## Neural Network Classifiers для NLP

### NER как бинарная классификация

Задача: для каждого слова определить, является ли оно **location** (или другой именованной сущностью). Вход -- конкатенация word vectors в окне:

$$x_{window} = [x_{museums}, x_{in}, x_{Paris}, x_{are}, x_{amazing}]^T \in \mathbb{R}^{5d}$$

### Отличие нейросетевого классификатора от линейного

Обычный softmax: $P(y \mid x) = \text{softmax}(Wx)$ -- **линейная** граница решения.

Нейросеть: обучаем **и** $W$, **и** distributed representations для слов. Слова перемещаются в промежуточном пространстве, где линейный softmax может их разделить. С глубокими сетями -- границы решения **нелинейные**.

## Concepts covered

- [[02 Areas/ML & DL/Concepts/NLP/Word2Vec|Word2Vec]] -- Skip-gram и CBOW, предсказание контекста
- [[02 Areas/ML & DL/Concepts/NLP/Negative Sampling|Negative Sampling]] -- эффективная альтернатива softmax, бинарная логистическая регрессия
- [[02 Areas/ML & DL/Concepts/NLP/GloVe|GloVe]] -- Global Vectors, комбинация count-based и prediction-based
- [[02 Areas/ML & DL/Concepts/NLP/SVD|SVD]] -- снижение размерности co-occurrence матрицы
- [[02 Areas/ML & DL/Concepts/NLP/Co-occurrence Matrix|Co-occurrence Matrix]] -- матрица совместной встречаемости слов
- [[02 Areas/ML & DL/Concepts/NLP/Word Senses|Word Senses]] -- полисемия и линейная суперпозиция значений
