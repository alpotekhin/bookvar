---
title: "CS224N — Lecture 4: Dependency Parsing"
course: "Stanford CS224N"
lecture: 4
type: course-note
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture04-dep-parsing]]"
concepts: ["[[02 Areas/ML & DL/Concepts/NLP/Dependency Parsing|Dependency Parsing]]", "[[02 Areas/ML & DL/Concepts/NLP/Constituency Parsing|Constituency Parsing]]", "[[02 Areas/ML & DL/Concepts/NLP/Treebank|Treebank]]", "[[02 Areas/ML & DL/Concepts/NLP/Neural Network|Neural Network]]"]
---

# Lecture 4: Dependency Parsing

> *"Explicit linguistic structure and how a neural net can decide it."* -- Christopher Manning

Лектор: Christopher Manning.

## Два взгляда на синтаксическую структуру

### [[02 Areas/ML & DL/Concepts/NLP/Constituency Parsing|Constituency (Phrase Structure)]]

Phrase structure grammar (= context-free grammars, CFGs). Слова объединяются во **фразы**, фразы -- в более крупные фразы:

```
Starting unit: words    → the, cat, cuddly, by, door
Words → phrases         → the cuddly cat, by the door
Phrases → bigger phrases → the cuddly cat by the door
```

Формализуется через правила: $S \to NP\ VP$, $NP \to Det\ (Adj)\ N\ (PP)$, и т.д. Результат -- **дерево составляющих** с нетерминальными узлами (NP, VP, S).

### [[02 Areas/ML & DL/Concepts/NLP/Dependency Parsing|Dependency Structure]]

Бинарные **асимметричные отношения** между словами: стрелки (dependencies) от **head** (governor) к **dependent** (modifier). Стрелки типизированы:

```
             submitted
    nsubj:pass     aux        obl
       Bills     were     Brownback
    nmod                        appos
       ports          case         flat
  case   cc    conj   by  Senator Republican
  on   and immigration              nmod
                                    Kansas
```

Типы зависимостей: `nsubj` (subject), `obj` (object), `det` (determiner), `amod` (adjective modifier), `nmod` (nominal modifier), `case` (preposition), `conj` (conjunction), и т.д.

Dependencies обычно формируют **дерево** -- связный ациклический граф с одним корнем (фиктивный ROOT, чтобы каждое слово было dependent ровно одного другого узла).

## Синтаксическая неоднозначность

### PP Attachment

*"Scientists count whales from space"*:
- Из космоса **считают** (from space → count) -- правильно
- Киты **из космоса** (from space → whales) -- неправильно

### Coordination Scope

*"Shuttle veteran and longtime NASA executive Fred Gregory appointed to board"*:
- [Shuttle veteran] and [longtime NASA executive Fred Gregory] -- один человек
- [Shuttle veteran] and [longtime NASA executive] [Fred Gregory] -- два человека

### Экспоненциальный рост парсов

Число возможных парсов растёт по **числам Каталана**: $C_n = \frac{(2n)!}{(n+1)!n!}$ -- экспоненциальная серия. Это делает exhaustive search непрактичным.

## Dependency Grammar: историческая справка

Идея зависимостей восходит к **Панини** (V в. до н.э., Индия). Современная традиция -- **Lucien Tesniere** (1959, "Elements de syntaxe structurale").

| Традиция | Доминирование | Языки |
|----------|---------------|-------|
| Dependency | "Восток", XX век | Русский, Китайский, Латынь -- свободный порядок слов |
| Constituency (CFG) | "Запад", с 1950-х | Английский -- R.S. Wells (1947), Chomsky (1953) |

Первый dependency parser в NLP: David Hays (1962) -- один из основателей American computational linguistics.

## [[02 Areas/ML & DL/Concepts/NLP/Treebank|Treebanks]]: аннотированные данные

### Ключевые treebanks

- **Penn Treebank** (Marcus et al. 1993) -- constituency trees для Wall Street Journal
- **Universal Dependencies** (universaldependencies.org) -- dependency trees для 100+ языков

### Преимущества treebanks

Построение treebank кажется медленнее, чем написание грамматики вручную, но:
- **Переиспользование**: на одном treebank можно обучить парсеры, POS-taggers, и т.д.
- **Broad coverage**: не только интуиции лингвиста, но реальные данные
- **Частотные распределения**: важны для статистических моделей
- **Evaluation**: объективный способ оценки NLP-систем

## Источники информации для dependency parsing

1. **Bilexical affinities**: зависимость [discussion → issues] правдоподобна
2. **Dependency distance**: большинство зависимостей -- между соседними словами
3. **Intervening material**: зависимости редко пересекают глаголы или знаки препинания
4. **Valency**: сколько dependents и с какой стороны типично для head

## Transition-based Parsing

### Идея: парсинг как последовательность действий

Парсер имеет три структуры данных:
- **Stack** $\sigma$ (top справа), инициализируется `[ROOT]`
- **Buffer** $\beta$ (top слева), инициализируется словами предложения $w_1, \ldots, w_n$
- **Arcs** $A$, инициализируется $\emptyset$

### Три операции (Arc-standard system)

| Действие | Эффект | Результат |
|----------|--------|-----------|
| **Shift** | $\sigma, w_i \mid \beta \Rightarrow \sigma \mid w_i, \beta$ | Слово из буфера в стек |
| **Left-Arc$_r$** | $\sigma \mid w_i \mid w_j, \beta \Rightarrow \sigma \mid w_j, \beta$ | $A \cup \{r(w_j, w_i)\}$ |
| **Right-Arc$_r$** | $\sigma \mid w_i \mid w_j, \beta \Rightarrow \sigma \mid w_i, \beta$ | $A \cup \{r(w_i, w_j)\}$ |

Завершение: $\sigma = [w]$, $\beta = \emptyset$.

### Пример: "I ate fish"

```
Start:     σ=[root]           β=[I, ate, fish]    A=∅
Shift:     σ=[root, I]        β=[ate, fish]
Shift:     σ=[root, I, ate]   β=[fish]
Left-Arc:  σ=[root, ate]      β=[fish]            A += nsubj(ate→I)
Shift:     σ=[root, ate, fish] β=[]
Right-Arc: σ=[root, ate]      β=[]                A += obj(ate→fish)
Right-Arc: σ=[root]           β=[]                A += root(root→ate)
```

### MaltParser (Nivre & Hall, 2005)

Выбор действия предсказывается **дискриминативным классификатором** (softmax) по features из конфигурации (стек, буфер). Максимум $|R| \times 2 + 1$ возможных действий.

**Greedy**: нет поиска (в простейшей форме). Можно добавить **beam search** для лучшего качества.

Линейное время: $O(n)$ -- одно действие на шаг. Скорость: 500+ предложений/сек.

## Neural Dependency Parser (Chen & Manning, 2014)

### Проблемы traditional features

Indicator features:
- **Разреженные**: бинарные, размерность $10^6 - 10^7$
- **Неполные**: не все комбинации наблюдались
- **Дорогие**: >95% времени парсинга уходит на feature computation

### Dense representations

Вместо hand-crafted features -- **конкатенация embeddings**:
- Слова из стека и буфера (top-$k$)
- POS-tags как отдельные embeddings
- Dependency labels как embeddings

Размерность: ~1000 (вместо миллионов для sparse features).

### Архитектура

```
Input layer:    x = lookup + concat (word, POS, dep embeddings)
Hidden layer:   h = ReLU(Wx + b1)
Output layer:   y = softmax(Uh + b2)    → {Shift, Left-Arc_r, Right-Arc_r}
```

### Два ключевых преимущества

1. **Distributed representations**: похожие слова/POS/labels имеют близкие вектора. Обобщение на unseen конфигурации.
2. **Non-linear classifier**: нелинейные decision boundaries через ReLU. Традиционные ML-классификаторы (Naive Bayes, SVM, logistic regression) дают только **линейные** boundaries.

### Результаты

| Parser | UAS | LAS | Скорость (sent/s) |
|--------|-----|-----|-------------------|
| MaltParser | 89.8 | 87.2 | 469 |
| MSTParser | 91.4 | 88.1 | 10 |
| TurboParser | 92.3 | 89.6 | 8 |
| **Chen & Manning 2014** | **92.0** | **89.7** | **654** |

Нейросетевой парсер: **лучшая точность и в 65 раз быстрее** graph-based парсеров.

### Оценка dependency parsing

- **UAS** (Unlabeled Attachment Score): доля слов с правильным head
- **LAS** (Labeled Attachment Score): доля слов с правильным head И правильным label

## Projectivity и non-projective parsing

**Projective parse**: нет пересекающихся дуг при линейном расположении слов. CFG-деревья всегда projective.

Пример non-projective: *"Who did Bill buy the coffee from yesterday?"* -- зависимость Who → from пересекает другие дуги.

Подходы к non-projectivity:
1. Игнорировать (для большинства конструкций это ОК)
2. Добавить **SWAP** transition (аналог bubble sort)
3. Использовать **graph-based** парсеры (MST, Dozat & Manning 2017)

## Дальнейшее развитие

Работа Chen & Manning (2014) -- **первый успешный нейросетевой dependency parser**. Дальнейшие улучшения (Google, 2016+):
- Более глубокие сети
- Лучшая настройка гиперпараметров
- Graph-based neural parsing (Dozat & Manning 2017) -- SOTA

## Concepts covered

- [[02 Areas/ML & DL/Concepts/NLP/Dependency Parsing|Dependency Parsing]] -- построение дерева зависимостей, transition-based и graph-based
- [[02 Areas/ML & DL/Concepts/NLP/Constituency Parsing|Constituency Parsing]] -- phrase structure grammar, CFG, деревья составляющих
- [[02 Areas/ML & DL/Concepts/NLP/Treebank|Treebank]] -- размеченные корпуса деревьев (Penn Treebank, Universal Dependencies)
- [[02 Areas/ML & DL/Concepts/NLP/Neural Network|Neural Network]] -- dense representations + non-linear classifier для парсинга
