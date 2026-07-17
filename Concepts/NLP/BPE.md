---
title: "BPE"
aliases: [BPE, Byte Pair Encoding, Byte-Pair Encoding]
type: concept
status: legacy
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA|LLaMA]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 12 — Subword Models|CS224N L12]]"
sources:
  - "[Neural Machine Translation of Rare Words with Subword Units (2016)](https://arxiv.org/abs/1508.07909)"
  - "[Hugging Face — Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)"
  - "[Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)"
---

# BPE — Byte Pair Encoding

## Зачем нужна subword-токенизация

[[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]] — первый этап обработки текста в любой языковой модели. Выбор уровня токенизации критически влияет на качество и эффективность модели.

### Word-level токенизация

Словарь из целых слов (`"playing"`, `"played"`, `"playground"`). Проблемы:

- **OOV (Out-of-Vocabulary):** новые слова, опечатки, морфологически богатые языки (немецкий, турецкий, русский) порождают огромное количество форм, которых нет в словаре
- **Огромный словарь:** для покрытия хотя бы 95% текста на английском нужно ~100K+ токенов; для мультиязычных моделей — миллионы
- **Нет sharing морфем:** `"play"`, `"playing"`, `"replay"` — три разных токена, хотя семантически связаны

### Character-level токенизация

Словарь из отдельных символов (~256 для ASCII). Решает OOV, но:

- **Слишком длинные последовательности:** слово из 10 букв = 10 токенов. Контекстное окно тратится впустую
- **Потеря семантики:** один символ `"p"` не несёт смысловой нагрузки, модель должна "учить" значения с нуля на каждом уровне

### Subword: лучшее из двух миров

Subword-токенизация (BPE, WordPiece, Unigram) разбивает текст на **части слов переменной длины:** частые слова остаются целыми (`"the"`, `"is"`), а редкие разбиваются на подслова (`"tokenization"` → `"token"` + `"ization"`). Это даёт:

- Компактный словарь (32K-128K токенов)
- Нет OOV — любое слово разбивается до символов в худшем случае
- Sharing морфем: `"play"` переиспользуется в `"playing"`, `"replay"`, `"playful"`

## Алгоритм BPE: шаг за шагом

BPE (Sennrich et al., 2016) — адаптация алгоритма сжатия данных (Gage, 1994) для [[02 Areas/ML & DL/Concepts/NLP/Tokenization|токенизации]]. Алгоритм обучения:

**Шаг 1.** Начать с символьного словаря — каждый уникальный символ в корпусе = один токен.

**Шаг 2.** Посчитать частоту всех смежных пар токенов в корпусе.

**Шаг 3.** Объединить (merge) самую частую пару в один новый токен. Добавить merge rule в таблицу.

**Шаг 4.** Повторить шаги 2-3 ровно N раз (N — гиперпараметр, определяющий итоговый размер словаря).

### Пример

Корпус (с частотами слов): `"hug" (10), "pug" (5), "hugs" (12), "bugs" (4)`

```
Итерация 0 — символьный словарь:
  h u g p s b  (+ специальный end-of-word символ)

  h u g      ×10
  p u g      ×5
  h u g s    ×12
  b u g s    ×4

Итерация 1 — самая частая пара: (u, g) = 10+5+12+4 = 31
  Merge: u g → ug
  h ug       ×10
  p ug       ×5
  h ug s     ×12
  b ug s     ×4

Итерация 2 — самая частая пара: (h, ug) = 10+12 = 22
  Merge: h ug → hug
  hug        ×10
  p ug       ×5
  hug s      ×12
  b ug s     ×4

Итерация 3 — самая частая пара: (hug, s) = 12
  Merge: hug s → hugs
  hug        ×10
  p ug       ×5
  hugs       ×12
  b ug s     ×4
```

При инференсе merge rules применяются в том же порядке к новому тексту. Слово `"mugs"` будет разбито как `m` + `ug` + `s` — merge (u, g) применится, но `m` останется отдельным символом.

## Варианты subword-токенизации

### Byte-level BPE (GPT-2, GPT-3, GPT-4)

Классический BPE работает на уровне Unicode-символов. [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]] предложил работать на уровне **сырых байтов:**

- Базовый словарь = 256 байтов (0x00-0xFF), а не символы Unicode
- **Никогда не бывает UNK:** любой Unicode-символ — это последовательность байтов, каждый из которых есть в словаре
- Обрабатывает любой язык, эмодзи, бинарные данные без pre-processing
- Недостаток: для не-ASCII символов (кириллица, CJK) один символ = 2-4 байта = 2-4 базовых токена до мержей

GPT-2 использует pre-tokenization через regex, чтобы merges не пересекали границы слов и пунктуации.

### WordPiece (BERT)

Используется в [[02 Areas/ML & DL/Papers/BERT|BERT]] и его вариантах. Похож на BPE, но критерий merge — не частота пары, а **максимизация likelihood** корпуса:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

Пары, которые встречаются вместе чаще, чем ожидалось бы при независимости, мержатся первыми. Подслова обозначаются префиксом `##`: `"tokenization"` → `["token", "##ization"]`.

### SentencePiece (LLaMA, T5, mBART)

Используется в [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]], T5, mBART. Ключевые отличия:

- **Language-agnostic:** трактует вход как сырой поток символов (включая пробелы как `▁`), без pre-tokenization
- Не зависит от правил конкретного языка (нет word boundary heuristics)
- Поддерживает два алгоритма: BPE и Unigram
- Результат токенизации **однозначно обратим** в исходный текст

### Unigram (T5, mBART, XLNet)

Противоположный подход к BPE:

- **Начинает с большого словаря** (все подстроки до определённой длины)
- Итеративно **удаляет** токены, наименее влияющие на likelihood корпуса
- Оптимизирует unigram language model: $P(x) = \prod_i P(x_i)$
- Может давать **несколько разбиений** с разными вероятностями — полезно для регуляризации (subword regularization)

## Сравнение вариантов

| Метод | Модели | Базовый словарь | Критерий merge/prune | UNK | Детерминированность |
|-------|--------|-----------------|---------------------|-----|---------------------|
| BPE (char) | Оригинальный NMT | Символы Unicode | Частота пар | Возможен | Да |
| Byte-level BPE | GPT-2/3/4 | 256 байтов | Частота пар | Нет | Да |
| WordPiece | BERT | Символы Unicode | Likelihood ratio | Возможен | Да |
| Unigram | T5, mBART | Все подстроки | Prune по likelihood | Возможен | Нет (вероятностный) |
| SentencePiece | LLaMA, T5 | Байты/символы | BPE или Unigram | Нет | Зависит от алгоритма |

## Vocab size: trade-offs

Размер словаря — ключевой гиперпараметр, влияющий на всю архитектуру модели:

**Маленький словарь (~4K-8K):**
- Длинные последовательности (больше токенов на текст) — дороже compute (attention = $O(n^2)$)
- Embedding matrix маленькая — меньше параметров
- Лучшая генерализация на редкие слова (больше sharing подслов)

**Большой словарь (~100K-128K):**
- Короткие последовательности — дешевле inference
- Embedding matrix большая — больше параметров (128K × hidden_dim)
- Отдельные токены для частых фраз, чисел, code snippets
- GPT-4 использует ~100K, [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]] 3 — 128K (включая мультиязычные и code токены)

Типичный диапазон для современных LLM: **32K-128K токенов.**

## Tiktoken: быстрая BPE-имплементация

OpenAI выпустили **tiktoken** — BPE-токенизатор на Rust с Python-биндингами. В 3-6 раз быстрее, чем HuggingFace tokenizers на чистом Python. Используется для GPT-3.5/4 (`cl100k_base`, `o200k_base`).

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, world!")  # [13225, 11, 2375, 0]
text = enc.decode(tokens)             # "Hello, world!"
print(len(enc.encode("Токенизация"))) # ~5-7 (кириллица = больше токенов)
```

## Практика: HuggingFace tokenizers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

text = "Tokenization is surprisingly important"
tokens = tokenizer.encode(text)           # [int, int, ...]
decoded = tokenizer.decode(tokens)        # исходный текст
token_strs = tokenizer.tokenize(text)     # ["Token", "ization", " is", ...]

# Специальные токены
print(tokenizer.bos_token)  # <s> или <|begin_of_text|>
print(tokenizer.eos_token)  # </s> или <|end_of_text|>
print(tokenizer.vocab_size) # 128256
```

Специальные токены (`<bos>`, `<eos>`, `<pad>`, `<unk>`) добавляются поверх BPE-словаря и не участвуют в merge-процессе. Они управляют поведением модели: начало/конец текста, padding для батчей, разделение turns в чатах.

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]] — общее понятие токенизации текста
- [[02 Areas/ML & DL/Concepts/NLP/Embeddings|Embeddings]] — следующий шаг: преобразование токенов в векторы
- [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]] — ввёл byte-level BPE
- [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]] — использует SentencePiece BPE
- [[02 Areas/ML & DL/Papers/BERT|BERT]] — использует WordPiece
