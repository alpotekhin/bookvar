---
title: BPE, WordPiece и Unigram
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1508.07909
  - https://arxiv.org/abs/1808.06226
---

# Токенизация: BPE, WordPiece и Unigram

> [!abstract] Идея главы
> Модель не получает строку текста напрямую. Tokenizer сначала решает, какие
> куски текста считать элементами словаря. От этого решения зависит длина
> контекста, стоимость вычислений и то, насколько удобно модели работать с
> языками, числами и кодом.

## 1. У текста нет единственного правильного разбиения

Строку:

```text
Токенизация surprisingly важна
```

можно представить:

- словами;
- Unicode-символами;
- UTF-8 bytes;
- частыми фрагментами переменной длины.

Word-level tokenizer сохраняет много смысла в одном token, но требует огромный
словарь и не знает новых слов. Character- или byte-level tokenizer способен
представить любой текст, но создаёт длинные sequences.

Subword tokenization ищет компромисс: частые фрагменты получает целиком, редкие
слова собирает из меньших частей.

![[02 Areas/ML & DL/raw/papers/bpe/images/tokenizer-comparison.png]]

*Сравнение способов разбиения текста. Важно смотреть одновременно на две цены:
размер vocabulary и число tokens в последовательности.*

## 2. Tokenizer — часть модели

После обучения LLM нельзя безболезненно заменить tokenizer:

- изменятся token ids;
- старая embedding matrix будет относиться к другому словарю;
- изменится разбиение training data;
- специальные токены потеряют назначение.

Tokenizer определяет интерфейс между текстом и нейросетью так же жёстко, как
архитектура входного слоя.

## 3. Построим BPE вручную

Карпати в лекции
[Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
начинает с короткой последовательности:

```text
aaabdaaabac
```

Сначала каждый byte — отдельный token. Затем повторяем:

1. посчитать соседние пары;
2. найти самую частую;
3. заменить её новым token id;
4. сохранить merge rule.

Например:

```text
(a, a) → Z
(Z, b) → Y
(Y, d) → X
```

С каждым merge vocabulary увеличивается на один элемент, а training text обычно
становится короче.

Минимальная функция подсчёта:

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

Замена пары:

```python
def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
```

Именно из этих операций вырастает
[minBPE](https://github.com/karpathy/minbpe).

## 4. Почему современные BPE начинаются с bytes

Если базовый vocabulary состоит только из встреченных Unicode-символов, новый
символ может оказаться неизвестным. UTF-8 представляет любую строку bytes, а
возможных byte values всего 256.

Byte-level BPE начинает с этих 256 элементов и поверх них учит merges. Поэтому
любой текст можно закодировать без `<UNK>`, включая редкие scripts и emoji.

Но «представим любой текст» не означает «представим одинаково эффективно».
Один кириллический символ занимает несколько UTF-8 bytes; если merges плохо
покрывают русский корпус, строка распадётся на большее число tokens.

## 5. Зачем GPT использует предварительное regex-разбиение

Если разрешить BPE объединять любые соседние bytes во всём корпусе, он может
создавать странные merges через границы слов, пробелов и пунктуации.

GPT-style tokenizers сначала разбивают строку regex-паттерном на категории вроде:

- буквы;
- числа;
- пробельные фрагменты;
- пунктуация.

BPE работает внутри этих частей. Это не косметическая деталь: pre-tokenization
ограничивает, какие токены вообще способны появиться.

## 6. Как применяется обученный BPE

При encoding нового текста:

1. строка превращается в bytes;
2. применяется pre-tokenization;
3. доступные пары объединяются в порядке приоритета выученных merges;
4. получившиеся ids передаются модели.

Decode выполняет обратное:

1. каждый id заменяется byte sequence;
2. bytes соединяются;
3. результат декодируется как UTF-8.

Хороший tokenizer должен корректно выполнять round trip:

```python
assert decode(encode(text)) == text
```

## 7. WordPiece

WordPiece известен прежде всего по BERT. Внешне он похож на BPE, но выбирает
фрагменты по вероятностному критерию, а не просто по абсолютной частоте пары.

Продолжение слова часто отмечается `##`:

```text
playing → play, ##ing
```

Это отображение токенов, а не символы, которые модель потом должна удалить.

Классический WordPiece может вернуть `[UNK]`, если не способен разложить слово
на известные pieces. Byte-level schemes избегают этой проблемы по построению.

## 8. Unigram

BPE начинает с маленького vocabulary и добавляет merges. Unigram идёт с другой
стороны:

1. создаёт большой набор возможных fragments;
2. назначает им вероятности;
3. оценивает, насколько ухудшится corpus likelihood при удалении каждого;
4. постепенно сокращает vocabulary до нужного размера.

Для одной строки может существовать несколько допустимых segmentations. Во
время training их можно сэмплировать, чтобы модель не привязывалась к одному
единственному разбиению.

## 9. SentencePiece — библиотека, а не один алгоритм

SentencePiece умеет обучать BPE или Unigram непосредственно на raw text. Пробел
становится обычным символом, который часто отображается как `▁`:

```text
Hello world → ▁Hello, ▁world
```

Это удобно для языков без пробелов между словами и делает detokenization
обратимой без отдельного внешнего word splitter.

Нельзя говорить «SentencePiece против BPE»: SentencePiece может использовать
BPE внутри.

## 10. Размер словаря: две противоположные цены

Маленький vocabulary:

- маленькая embedding matrix и LM head;
- длиннее sequences;
- больше работы attention;
- больше переиспользования fragments.

Большой vocabulary:

- короче sequences;
- крупнее embedding и output matrices;
- больше редких специализированных tokens;
- больше памяти и параметров на словарь.

![[02 Areas/ML & DL/raw/papers/bpe/images/tokenizer-overview.png]]

*Fertility — среднее число tokens на слово. На графике больший vocabulary
сокращает разбиение, а multilingual tokenizer заметно эффективнее для
мультиязычных данных.*

## 11. Почему токенизация влияет на языки

Сравнивать нужно не «сколько символов знает tokenizer», а сколько tokens он
тратит на сопоставимый текст.

Если русский абзац занимает вдвое больше tokens, чем английский:

- в context window помещается меньше текста;
- attention обрабатывает более длинную sequence;
- API usage может стоить дороже;
- модель получает меньше примеров русских слов на один training token budget.

Больший мультиязычный vocabulary часто улучшает compression, но не заменяет
качественные training data на этих языках.

## 12. Числа, пробелы и код

Разбиение чисел может быть нестабильным:

```text
123456 → 123, 456
123457 → 12, 345, 7
```

Модель должна изучить арифметические отношения поверх не всегда согласованных
кусочков. Аналогично пробелы и indentation в коде могут входить в токены,
влияя на длину и регулярность representations.

Это одна из причин, почему tokenizer нельзя оценивать только общей compression
ratio: нужны отдельные наборы для языков, кода, чисел и специальных форматов.

## 13. Специальные токены и chat template

Помимо обычного текста vocabulary содержит управляющие элементы:

- начало и конец документа;
- padding;
- границы user/assistant messages;
- начало tool call;
- изображения или audio placeholders.

Chat template превращает список сообщений в одну token sequence. Ошибка в нём
может повлиять сильнее, чем decoding parameters: модель увидит другой формат,
чем во время post-training.

## 14. Как сравнивать tokenizers

Для одного и того же набора текстов измерьте:

| Метрика | Что показывает |
|---|---|
| bytes per token | compression |
| tokens per word | fertility |
| доля byte fallback | нехватка подходящих pieces |
| vocabulary size | цена embedding/LM head |
| round-trip errors | обратимость |
| разбивка чисел и кода | структурная регулярность |
| языковой разрыв | fairness и стоимость |

Одного среднего числа по английской Википедии недостаточно.

## Практика по Карпати

1. Реализовать `get_stats` и `merge`.
2. Обучить 20–100 merges на маленьком тексте.
3. Добавить encode/decode.
4. Перейти с Unicode code points на UTF-8 bytes.
5. Добавить regex pre-tokenization.
6. Сравнить результат с `tiktoken`.
7. Проверить русский, emoji, числа и Python-код.

## Что должно остаться после главы

- Tokenizer выбирает элементы vocabulary до входа в нейросеть.
- Subword methods балансируют размер словаря и длину sequence.
- BPE добавляет наиболее полезные частые merges.
- Byte-level BPE способен представить любой UTF-8 text.
- WordPiece и Unigram используют другие критерии построения vocabulary.
- SentencePiece — framework, поддерживающий несколько алгоритмов.
- Tokenizer влияет на стоимость, мультиязычность, числа, код и chat format.

## Источники

- [Andrej Karpathy — Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)
- [Stanford CS336 — Tokenization](https://stanford-cs336.github.io/spring2025/)
- [Sennrich et al. — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Kudo — Subword Regularization](https://arxiv.org/abs/1804.10959)
- [Kudo & Richardson — SentencePiece](https://arxiv.org/abs/1808.06226)
- [Hugging Face — Summary of tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary)

