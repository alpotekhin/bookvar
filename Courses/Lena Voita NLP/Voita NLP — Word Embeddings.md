---
title: "Voita NLP — Word Embeddings"
type: course-note
course: "Lena Voita NLP"
---

# Voita NLP — Word Embeddings

> Как представить слова числами так, чтобы сохранить их смысл. От count-based методов до Word2Vec и GloVe.

**Курс:** [[Lena Voita NLP/_index|Lena Voita NLP Course]]
**Страница:** https://lena-voita.github.io/nlp_course/word_embeddings.html
**Связанные концепты:** [[Tokenization]], [[Transfer Learning]], [[Self-Attention]]

---

## Distributional Hypothesis

Фундаментальная идея, на которой построены все word embeddings:

> **"Слова, которые часто встречаются в похожих контекстах, имеют похожий смысл."**
> — J.R. Firth, 1957

Из этого следует: чтобы понять значение слова, достаточно посмотреть на его окружение. Не нужен словарь, не нужна экспертная разметка — **контекст определяет семантику**.

```
"Кошка сидела на ___"     → коврике, диване, подоконнике
"Собака лежала на ___"    → коврике, диване, полу

→ "кошка" и "собака" семантически близки (похожие контексты)
```

---

## Count-Based Methods

### Co-occurrence Matrix

Самый простой способ: считаем, сколько раз слова встречаются рядом.

**Шаг 1:** Определяем окно контекста (window size L = 2)

**Шаг 2:** Для каждой пары (word, context) считаем частоту

```
Корпус: "I like deep learning. I like NLP."

           I    like   deep   learning   NLP
I        [ 0     2      0       0        0  ]
like     [ 2     0      1       0        1  ]
deep     [ 0     1      0       1        0  ]
learning [ 0     0      1       0        0  ]
NLP      [ 0     1      0       0        0  ]
```

Каждая строка — вектор слова. Проблема: размерность = размер словаря (огромная), много нулей.

### PPMI (Positive Pointwise Mutual Information)

Сырые частоты — плохая мера ассоциации. Слово "the" встречается со всем, но это не значит, что оно семантически связано со всем.

**PMI** измеряет силу ассоциации:

```
PMI(w, c) = log₂ [P(w,c) / (P(w) · P(c))]
```

- PMI > 0 — слова встречаются вместе чаще, чем случайно
- PMI = 0 — независимы
- PMI < 0 — встречаются вместе реже, чем случайно

**PPMI** — обрезаем отрицательные значения (они шумные):

```
PPMI(w, c) = max(0, PMI(w, c))
```

PPMI-матрица — state-of-the-art среди count-based методов до появления Word2Vec.

### LSA (Latent Semantic Analysis)

SVD-разложение term-document матрицы:

```
M = U · Σ · V^T
```

Берём первые k сингулярных значений → получаем k-мерные векторы для слов. Работает для document similarity и topic modeling.

---

## Word2Vec

### Ключевая идея

Вместо подсчёта статистик, **обучаем** нейронную сеть предсказывать контекст по слову (или наоборот). Побочный продукт обучения — word vectors.

### Skip-Gram

Предсказываем контекстные слова по центральному:

```
Вход: "learning"
Цель: предсказать "deep", "is", "fun", "and"  (окно = 2)
```

**Модель:**

```
center word → lookup embedding v_w → predict context words
P(context | center) = exp(u_c · v_w) / Σ_w' exp(u_w' · v_w)
```

Два набора векторов:
- **v_w** — vector для центрального слова (input embeddings)
- **u_c** — vector для контекста (output embeddings)

После обучения используем v_w (контекстные u_c отбрасываются).

**Loss:**

```
L = -Σ log P(context | center) = -Σ log [exp(u_c · v_w) / Σ_w' exp(u_w' · v_w)]
```

### CBOW (Continuous Bag of Words)

Обратная задача: предсказываем центральное слово по сумме контекстных:

```
Вход: v_"deep" + v_"is" + v_"fun" + v_"and"
Цель: "learning"
```

Skip-gram работает лучше на маленьких данных и для редких слов. CBOW — быстрее.

### Negative Sampling

Полный softmax по всему словарю (|V| = 100K+) — слишком дорого. Negative sampling аппроксимирует:

```
L ≈ log σ(u_positive · v_w) + Σ_{k=1}^{K} log σ(-u_negative_k · v_w)
```

Вместо нормализации по всем словам, **сэмплируем K негативных примеров** (типично K = 5-20). Негативные слова выбираются пропорционально U(w)^(3/4) — это даёт редким словам больше шансов быть негативными примерами.

---

## GloVe (Global Vectors)

### Мотивация

Word2Vec работает с **локальными** окнами. GloVe (Pennington et al., 2014) комбинирует:
- **Глобальную** статистику co-occurrence матрицы
- **Обучение** оптимизацией loss function

### Формула

```
L = Σ_{i,j} f(X_ij) · (w_i^T · w̃_j + b_i + b̃_j - log X_ij)²
```

где:
- X_ij — co-occurrence count для пары (i, j)
- f(X_ij) — весовая функция, ограничивающая влияние частых пар
- w_i, w̃_j — обучаемые векторы
- b_i, b̃_j — bias-термы

### f(x) — весовая функция

```
f(x) = (x/x_max)^α    если x < x_max
f(x) = 1               если x ≥ x_max
```

Типично: x_max = 100, α = 0.75. Частые пары (the, of) не доминируют в loss.

### GloVe vs Word2Vec

На практике качество сопоставимо. GloVe эффективнее для больших корпусов (одна итерация по всей матрице вместо стохастических обновлений).

---

## Свойства Word Embeddings

### Семантическое сходство

Семантически близкие слова группируются:

```
cos(v_"king", v_"queen") ≈ 0.75
cos(v_"king", v_"car")   ≈ 0.15
```

### Линейные отношения (аналогии)

Знаменитое свойство: семантические отношения закодированы как **линейные сдвиги**:

```
v_"king" - v_"man" + v_"woman" ≈ v_"queen"
v_"Paris" - v_"France" + v_"Germany" ≈ v_"Berlin"
v_"walking" - v_"walk" + v_"swim" ≈ v_"swimming"
```

Это работает, потому что отношение "мужчина→женщина" имеет **одинаковый вектор-смещение** для разных пар.

### Cross-lingual Mapping

Embedding-пространства разных языков можно выровнять **линейным преобразованием**:

```
W · v_english ≈ v_russian
```

Достаточно небольшого словаря переводов для обучения W. Это позволяет zero-shot перевод и cross-lingual retrieval.

---

## Evaluation

### Intrinsic Evaluation

Оцениваем **качество самих embeddings**:

**Word Similarity:**
- Человеческие оценки похожести пар слов (SimLex-999, MEN)
- Корреляция Spearman между cosine similarity и human ratings

**Word Analogy:**
- "king : queen :: man : ?" → "woman"
- Accuracy на стандартных бенчмарках (Google Analogy, BATS)

### Extrinsic Evaluation

Оцениваем **полезность для downstream tasks:**
- Text classification
- Named Entity Recognition
- Sentiment analysis
- Coreference resolution

Лучшие intrinsic results не гарантируют лучшие extrinsic results — но обычно коррелируют.

---

## Ограничения Static Embeddings

1. **Одно значение на слово:** "bank" (берег) и "bank" (банк) имеют один вектор
2. **Нет контекста:** каждое слово — фиксированный вектор, независимо от окружения
3. **OOV проблема:** слова, не встреченные при обучении, не имеют вектора

**Решения:**
- Subword embeddings (FastText) — частично решают OOV
- Contextual embeddings (ELMo, BERT) — полностью решают 1 и 2
- [[Transfer Learning]] — переход к pretrained language models

---

## Research Thinking (Voita)

Важная секция курса — **как критически читать papers**:

1. Чем отличается evaluation от claims автора?
2. Какие baseline-ы выбраны — честные или слабые?
3. Reproducibility: предоставлен ли код и данные?
4. Ablation studies: что именно даёт improvement?

Этот навык важнее любого конкретного метода.

---

## Ключевые выводы

1. **Distributional hypothesis** — смысл слова определяется контекстом
2. **Count-based** (PPMI) и **prediction-based** (Word2Vec) дают сопоставимое качество
3. **Word2Vec Skip-Gram + Negative Sampling** — стандартный метод обучения
4. **GloVe** комбинирует глобальную статистику с обучением
5. **Линейные аналогии** — ключевое свойство, позволяющее reasoning
6. **Static embeddings ограничены** — контекстуальные модели (ELMo, BERT) их заменили

---

## Источники

- Lena Voita, NLP Course: Word Embeddings — https://lena-voita.github.io/nlp_course/word_embeddings.html
- Mikolov et al., "Efficient Estimation of Word Representations" (Word2Vec, 2013)
- Pennington et al., "GloVe: Global Vectors for Word Representation" (2014)
- Levy & Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (2014)

---

**См. также:** [[Tokenization]], [[Transfer Learning]], [[Voita NLP — Seq2Seq and Attention]], [[Voita NLP — Transfer Learning]]
