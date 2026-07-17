---
title: "Tokenization"
aliases: [BPE, Byte Pair Encoding, subword tokenization, WordPiece, SentencePiece, tokenizer, Unigram tokenizer, tiktoken]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA|LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]]"
  - "[[02 Areas/ML & DL/Papers/T5|T5]]"
courses: []
sources:
  - "[Hugging Face — Tokenization algorithms](https://huggingface.co/docs/transformers/en/tokenizer_summary)"
  - "[Hugging Face — BPE tokenization (LLM Course)](https://huggingface.co/learn/llm-course/en/chapter6/5)"
  - "[Karpathy — minbpe](https://github.com/karpathy/minbpe)"
  - "[Karpathy — Let's build the GPT Tokenizer (видео)](https://www.youtube.com/watch?v=zduSFxRajkE)"
  - "[OpenAI — tiktoken](https://github.com/openai/tiktoken)"
  - "[Sebastian Raschka — BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)"
---

# Tokenization

## Зачем это нужно: от текста к числам

Языковая модель не «видит» текст — она работает с числами. **Tokenization** — это процесс разбиения входного текста на дискретные единицы (**токены**), каждая из которых получает числовой ID из словаря. Это первый и последний шаг в любом NLP-пайплайне: текст → токены → эмбеддинги → модель → токены → текст.

Выбор токенизатора критически влияет на всё: от размера контекстного окна (больше токенов на текст = меньше «полезного» контекста) до стоимости инференса (API тарифицируются по токенам) и даже способности модели к арифметике.

## Триллемма токенизации: три уровня

### Word-level (пословная)

Разбиение по пробелам и пунктуации: `"I love transformers"` → `["I", "love", "transformers"]`.

**Проблемы:**
- Словарь раздувается: каждая словоформа — отдельный токен (`"love"`, `"loved"`, `"loving"`, `"lovingly"` — четыре записи)
- Невиданные слова → `<UNK>`: модель не может работать с новыми словами
- Для агглютинативных языков (турецкий, финский) словарь вырастает до миллионов

### Character-level (посимвольная)

Каждый символ — токен: `"love"` → `["l", "o", "v", "e"]`.

**Проблемы:**
- Словарь маленький (~256 для ASCII), но последовательности огромные
- Одиночный символ `"l"` несёт значительно меньше смысла, чем `"love"`
- Модели нужно «собирать» значение из символов — это очень дорого

### Subword-level (субсловная) — золотая середина

Компромисс: частые слова остаются целыми, редкие разбиваются на осмысленные части. Например: `"unhappiness"` → `["un", "happi", "ness"]`. Словарь компактный (30-100K), `<UNK>` практически не встречается, семантика сохраняется.

**Все современные LLM используют subword tokenization.** Разница — в алгоритме обучения словаря.

## Byte Pair Encoding (BPE)

![[02 Areas/ML & DL/raw/papers/tokenization/images/bpe-overview.jpg]]
*BPE tokenization: итеративное объединение наиболее частых пар символов/подслов до достижения целевого размера словаря (источник: Sebastian Raschka)*

### История

BPE был создан в 1994 году Филипом Гейджем как алгоритм **сжатия данных**. В 2015 году Sennrich et al. адаптировали его для NLP — вместо сжатия байтов алгоритм объединяет частые пары символов/подслов в тексте.

### Алгоритм обучения (пошагово)

**Вход:** корпус текстов, целевой размер словаря $V$.

1. **Инициализация** — начинаем с символьного словаря (все уникальные символы корпуса)
2. **Считаем частоту пар** — для каждой пары соседних токенов в корпусе считаем, сколько раз она встречается
3. **Объединяем самую частую пару** — создаём новый токен, добавляем в словарь
4. **Обновляем корпус** — заменяем все вхождения пары новым токеном
5. **Повторяем** шаги 2-4 до достижения $|V|$

**Конкретный пример** (из Hugging Face Course):

Корпус со словами и частотами:
```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

Начальный словарь: `["b", "g", "h", "n", "p", "s", "u"]`

| Шаг | Самая частая пара | Новый токен | Словарь |
|-----|------------------|-------------|---------|
| 1 | `("u","g")` — 20 раз | `"ug"` | + `"ug"` |
| 2 | `("u","n")` — 16 раз | `"un"` | + `"un"` |
| 3 | `("h","ug")` — 15 раз | `"hug"` | + `"hug"` |

После 3 merge: `("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)`

### Алгоритм токенизации (инференс)

При токенизации нового текста применяются **выученные merge-правила в том порядке, в каком они были изучены**:

1. Разбить текст на символы
2. Применить merge 1: `("u","g")` → `"ug"` (везде, где встречается)
3. Применить merge 2: `("u","n")` → `"un"`
4. Применить merge 3: `("h","ug")` → `"hug"`
5. ...и так далее

Слово `"bug"` → `["b", "u", "g"]` → `["b", "ug"]`. Слово `"hug"` → `["h", "u", "g"]` → `["h", "ug"]` → `["hug"]`.

### Byte-level BPE (GPT-2/3/4)

Проблема: если в корпусе не было какого-то Unicode-символа, при инференсе он станет `<UNK>`. Решение GPT-2: **начать не с символов, а с байтов** (256 базовых токенов). Любой текст — это последовательность байтов, поэтому `<UNK>` невозможен в принципе.

GPT-2 BPE: 256 byte-токенов + 50,000 merges + 1 special token = **50,257 токенов**.

## Пример: как «unhappiness» токенизируется разными методами

| Метод | Результат | Комментарий |
|-------|-----------|-------------|
| Word-level | `["unhappiness"]` | Если слова нет в словаре → `<UNK>` |
| Character-level | `["u","n","h","a","p","p","i","n","e","s","s"]` | 11 токенов, мало семантики |
| **BPE** (GPT-2) | `["un", "h", "app", "iness"]` | Субслова с частичным смыслом |
| **BPE** (GPT-4/cl100k) | `["unh", "appiness"]` | Другой словарь — другая разбивка |
| **WordPiece** (BERT) | `["un", "##happ", "##iness"]` | `##` = продолжение слова |
| **SentencePiece** (T5) | `["▁un", "happin", "ess"]` | `▁` = начало нового слова |

Ключевой insight: **одно и то же слово разбивается по-разному в зависимости от алгоритма и корпуса обучения**. Нет «правильной» токенизации — есть trade-off между размером словаря и средней длиной последовательности.

## WordPiece (BERT)

WordPiece (Schuster & Nakamura, 2012) — вариант BPE, используемый в BERT, DistilBERT, Electra.

### Ключевое отличие от BPE

BPE объединяет самую **частую** пару. WordPiece объединяет пару, которая максимизирует **likelihood обучающего корпуса**:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

Это означает, что WordPiece предпочитает объединять токены, которые встречаются вместе **чаще, чем можно ожидать по отдельным частотам**. Пара `("g", "s")` с score 0.050 может быть объединена раньше `("u", "g")` со score 0.028, хотя `"ug"` встречается абсолютно чаще.

### Маркер продолжения `##`

WordPiece маркирует продолжение слова префиксом `##`:
```
"playing" → ["play", "##ing"]
"unhappiness" → ["un", "##happ", "##iness"]
```

### Параметры BERT

| Вариант | Словарь | Особенность |
|---------|---------|-------------|
| BERT-uncased | 30,522 | Lowercase весь текст перед токенизацией |
| BERT-cased | 28,996 | Сохраняет регистр |

## Unigram (T5, XLNet)

Unigram (Kudo, 2018) — принципиально другой подход: **top-down** вместо bottom-up.

### Алгоритм

1. **Начать с большого словаря** — все слова и подслова из корпуса
2. **Для каждого токена посчитать loss** — насколько вырастет общий loss (log-likelihood), если убрать этот токен
3. **Удалить 10-20% токенов с наименьшим impact**
4. **Повторять** до целевого размера

### Ключевые особенности

- **Вероятностный**: каждый токен имеет вероятность, при инференсе выбирается разбивка с максимальной суммарной вероятностью
- **Не детерминистический при обучении**: может сэмплировать разные токенизации (полезно для регуляризации)
- **Лучшая компрессия**: в среднем Unigram требует меньше токенов на текст, чем BPE

## SentencePiece (T5, LLaMA)

SentencePiece (Kudo & Richardson, 2018) — не отдельный алгоритм, а **библиотека**, которая может использовать BPE или Unigram, но с важным отличием: работает **напрямую с сырым текстом** без предварительного разделения по пробелам.

### Зачем это нужно

Стандартный BPE/WordPiece предполагает, что слова разделены пробелами. Но в китайском, японском, тайском нет пробелов. SentencePiece трактирует пробел как обычный символ `▁` (U+2581):

```
"Hello world" → ["▁Hello", "▁world"]
```

Пробел становится частью токена — это делает токенизацию полностью обратимой (detokenization тривиальна: просто конкатенация с заменой `▁` на пробел).

### LLaMA SentencePiece BPE

LLaMA использует SentencePiece с BPE-алгоритмом:
- Словарь: 32,000 токенов
- Цифры разбиваются поодиночке: `"123"` → `["1", "2", "3"]` (улучшает арифметику)
- Байтовые fallback-токены для неизвестных символов

## tiktoken (GPT-3.5/4/4o)

[tiktoken](https://github.com/openai/tiktoken) — библиотека OpenAI для BPE-токенизации, написанная на Rust для скорости (3-6x быстрее аналогов).

### Кодировки

| Кодировка | Словарь | Модели |
|-----------|---------|--------|
| `r50k_base` | ~50K | GPT-3 (davinci) |
| `p50k_base` | ~50K | Codex |
| `cl100k_base` | 100,256 | GPT-3.5-turbo, GPT-4 |
| `o200k_base` | ~200K | GPT-4o |

Рост словаря = лучшая компрессия (меньше токенов на текст), особенно для не-английских языков. GPT-4o с `o200k_base` значительно эффективнее на русском и других языках, чем GPT-3.

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("unhappiness")  # → [359, 438, 9949]
[enc.decode([t]) for t in tokens]   # → ["unh", "app", "iness"]
```

## minbpe — образовательная реализация от Karpathy

[minbpe](https://github.com/karpathy/minbpe) — минимальная, чистая реализация byte-level BPE. Лучший способ **разобраться** в BPE на практике.

### Три уровня сложности

| Класс | Описание |
|-------|----------|
| `BasicTokenizer` | Простейший BPE напрямую на тексте |
| `RegexTokenizer` | BPE с regex-препроцессингом (как GPT-2/4): split по категориям (буквы, цифры, пунктуация) перед merge |
| `GPT4Tokenizer` | Обёртка, воспроизводящая tiktoken `cl100k_base` 1-в-1 |

### Пример работы

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.train("aaabdaaabac", 256 + 3)  # 256 byte tokens + 3 merges
tokenizer.encode("aaabdaaabac")  # [258, 100, 258, 97, 99]
```

Merges: `aa` → 256 (Z), `ab` → 257 (Y), `ZY` → 258 (X). Результат: `"XdXac"`.

### Ключевые детали

- **Byte-level**: работает с UTF-8 bytes, не символами → никаких `<UNK>`
- **Regex split** (GPT-2+): текст сначала разбивается regex-паттерном по категориям → merges не пересекают границы категорий (цифра не склеится с буквой)
- **Special tokens**: регистрируются отдельно после обучения, ID начинается после последнего merge token
- Сопровождается [видео-лекцией](https://www.youtube.com/watch?v=zduSFxRajkE) (2ч 13мин)

## Сравнительная таблица токенизаторов

| Метод | Подход | Выбор merge | Модели | Словарь |
|-------|--------|-------------|--------|---------|
| **BPE** | Bottom-up | Самая частая пара | GPT-2/3/4, LLaMA, Gemma | 32-200K |
| **WordPiece** | Bottom-up | Максимизация likelihood | BERT, DistilBERT, Electra | ~30K |
| **Unigram** | Top-down | Минимизация loss при удалении | T5, XLNet, mBART | ~32K |
| **SentencePiece** | Библиотека | BPE или Unigram | T5, LLaMA, XLNet | varies |

## Практические последствия выбора токенизатора

### Мультиязычность и стоимость

Токенизаторы, обученные преимущественно на английском, плохо сжимают другие языки. Русский текст в GPT-2 занимает **2-4x больше токенов**, чем эквивалентный английский. Это означает:
- Контекстное окно «короче» для русского текста
- API-вызовы дороже
- Модель видит меньше «смысла» за то же количество токенов

GPT-4o (`o200k_base`) значительно улучшил ситуацию за счёт увеличения словаря и более мультиязычного корпуса обучения.

### Арифметика и цифры

Способ токенизации чисел влияет на математические способности модели. LLaMA разбивает числа на отдельные цифры (`"123"` → `["1", "2", "3"]`), что помогает при арифметике. GPT-2 может закодировать `"123"` как один токен, затрудняя поцифровые операции.

### Формальное определение

Из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]] (§3.1): словарь $V$ = конечное множество токенов. Token embedding: $W_e \in \mathbb{R}^{d_e \times |V|}$. Для токена $x$: embedding = $W_e[:, x]$. Embedding layer — это lookup table, превращающая дискретные ID в непрерывные векторы.

## Хронология

| Год | Milestone | Значение |
|-----|-----------|----------|
| 1994 | BPE (Gage) | Алгоритм сжатия данных |
| 2012 | WordPiece (Schuster & Nakamura) | Для Google Speech |
| 2013 | Word2Vec (Mikolov et al.) | Слова как векторы, но word-level |
| 2015 | BPE для NMT (Sennrich et al.) | Subword tokenization для NLP |
| 2018 | SentencePiece (Kudo & Richardson) | Language-agnostic tokenization |
| 2018 | BERT + WordPiece | 30K vocab, стандарт для NLU |
| 2019 | GPT-2 + byte-level BPE | 50K vocab, нет `<UNK>` |
| 2022 | tiktoken (OpenAI) | Быстрая Rust-реализация BPE |
| 2023 | GPT-4 + cl100k_base | 100K vocab |
| 2024 | GPT-4o + o200k_base | 200K vocab, лучшая мультиязычность |
| 2024 | minbpe (Karpathy) | Образовательная реализация |

## Key papers

- Sennrich et al. 2015 — *Neural Machine Translation of Rare Words with Subword Units* — оригинальная адаптация BPE для NLP
- [[02 Areas/ML & DL/Papers/BERT|BERT]] — WordPiece, 30K vocab
- [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]] — byte-level BPE, 50K vocab, reversible tokenizer
- [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]] — SentencePiece BPE, digit-by-digit numbers
- Kudo 2018 — *Subword Regularization* — Unigram алгоритм
- Kudo & Richardson 2018 — *SentencePiece* — language-agnostic tokenization

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — потребитель токенов
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — после токенизации каждый токен получает позицию
- [[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]] — словарь токенизатора = часть pre-trained модели
- [[02 Areas/ML & DL/Concepts/Training/nanoGPT|nanoGPT]] — использует tiktoken для BPE tokenization

## Дополнительные ресурсы

- [Hugging Face — Tokenization algorithms](https://huggingface.co/docs/transformers/en/tokenizer_summary) — лучшее сравнение BPE / WordPiece / Unigram
- [Hugging Face — BPE tokenization (LLM Course)](https://huggingface.co/learn/llm-course/en/chapter6/5) — пошаговая реализация с кодом
- [Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — 2-часовое видео, строим токенизатор с нуля
- [Sebastian Raschka — BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — ещё одна реализация с объяснениями
- [OpenAI tiktoken](https://github.com/openai/tiktoken) — production-grade BPE на Rust
