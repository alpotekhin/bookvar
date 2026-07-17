---
title: "N-gram"
aliases: [N-gram, N-grams, N-gram Language Model]
type: concept
status: legacy
category: NLP
papers: []
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 05 — Language Models and RNNs|CS224N L05]]"
sources:
  - "[Jurafsky & Martin — SLP3, Ch. 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)"
  - "[Chen & Goodman — An Empirical Study of Smoothing Techniques (1998)](https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf)"
---

# N-gram — статистическая языковая модель

## Что это

**N-gram** — это последовательность из $n$ подряд идущих токенов. N-gram **языковая модель** оценивает вероятность текста, предполагая, что следующий токен зависит только от $n-1$ предыдущих.

Терминология:
- $n=1$ — **unigram** (отдельные слова)
- $n=2$ — **bigram** (пары)
- $n=3$ — **trigram** (тройки)
- $n=4, 5$ — 4-gram, 5-gram

Исторически — первый и **доминировавший подход к language modeling до 2010-х**. Сейчас почти полностью вытеснен нейронными LM, но остаётся в ряде прикладных ниш.

## Марковское предположение

Точная факторизация вероятности последовательности:

$$P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})$$

требует оценки вероятности при **растущем** (до длины $T-1$) контексте, что статистически неосуществимо — большинство длинных контекстов никогда не встречались в корпусе.

**Марковское приближение порядка $n-1$:**

$$P(w_t \mid w_1, \ldots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})$$

Для trigram ($n=3$):

$$P(w_t \mid w_{<t}) \approx P(w_t \mid w_{t-2}, w_{t-1})$$

Контекст становится **конечным** — можно считать по корпусу.

## Оценка вероятностей: MLE

Максимум правдоподобия — просто отношение счётчиков:

$$P_{\text{MLE}}(w_t \mid w_{t-n+1:t-1}) = \frac{C(w_{t-n+1:t-1}, w_t)}{C(w_{t-n+1:t-1})}$$

где $C(\cdot)$ — число вхождений n-граммы в корпусе. Для bigram:

$$P(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}$$

Реализация — хеш-таблицы счётчиков. На практике используются префиксные деревья (trie) для экономии памяти.

## Главная проблема: sparsity

Даже на огромных корпусах большинство валидных n-грамм **никогда не встречались**. Пример:

- $|V| = 50{,}000$ слов → потенциальных trigram: $1.25 \times 10^{14}$
- Реально встречается на 1B-токенном корпусе: $\sim 10^9$
- **Покрытие менее 0.001%**

Если фраза содержит невстреченную n-грамму — $P = 0$ → произведение всей вероятности обнуляется. Плюс: делить на ноль, если весь контекст не встречался.

Решение — **сглаживание** (smoothing): резервируем часть вероятностной массы для невидимых n-грамм.

## Техники сглаживания

### 1. Laplace (Add-1) Smoothing

Добавить 1 ко всем счётчикам:

$$P_{\text{Laplace}}(w_t \mid w_{<}) = \frac{C(w_{<}, w_t) + 1}{C(w_{<}) + |V|}$$

**Простейшее**, но даёт слишком много массы невидимым событиям — редко используется в production.

### 2. Add-k Smoothing

Обобщение: $+k$ вместо $+1$, где $k < 1$. Лучше Laplace, но всё ещё не идеально.

### 3. Backoff и Interpolation

**Интуиция:** если trigram не встречался, используй bigram; если bigram — unigram.

**Interpolation** — взвешенная сумма:

$$P(w_t|w_{t-2},w_{t-1}) = \lambda_3 P(w_t|w_{t-2},w_{t-1}) + \lambda_2 P(w_t|w_{t-1}) + \lambda_1 P(w_t)$$

с $\sum \lambda_i = 1$. Веса оцениваются на валидации.

**Backoff** (Katz) — использовать только более короткую n-грамму, если длинная не встречалась, с discount-коэффициентами.

### 4. Kneser-Ney Smoothing

**SOTA среди классических методов.** Ключевая идея — разделять **частоту** и **универсальность** слова.

Пример: «Francisco» очень частое (из-за «San Francisco»), но как unigram — плохой fallback, потому что встречается почти только после «San». Обычный backoff этого не учитывает.

Kneser-Ney использует **continuation probability**:

$$P_{\text{cont}}(w) \propto |\{w' : C(w', w) > 0\}|$$

То есть вероятность слова в low-order fallback = в **скольки разных контекстах** оно появлялось, а не сколько раз. «Francisco» имеет низкий $P_{\text{cont}}$ (мало разных предшественников) → bigram backoff снижает его вероятность в новых контекстах.

**Modified Kneser-Ney** с разными discount-ами для n-грамм с count 1, 2, 3+ — де-факто стандарт. Все академические n-gram системы 2000-х используют его.

## Размер модели и выбор $n$

| $n$ | Плюсы | Минусы |
|-----|-------|--------|
| 1 (unigram) | Простой, плотный | Игнорирует контекст → плохое качество |
| 2 (bigram) | Компромисс скорость/качество | Не ловит структуру фразы |
| 3 (trigram) | Исторический стандарт | Sparsity уже значительна |
| 4-5 | Лучшее качество на больших корпусах | Очень разреженно, нужны GB памяти |
| 6+ | Незначительный прирост | Сильно страдает от sparsity |

Google на 2007 опубликовал 5-gram counts из 1 триллиона токенов web-текста — до сих пор референсный артефакт.

## Генерация через n-gram

Простейший генератор:
1. Стартовать с `<s>` (start-of-sentence)
2. Сэмплировать $w_t \sim P(w_t | w_{t-n+1:t-1})$
3. Повторять до `</s>`

Результат — синтаксически неплохо, семантически бессвязно, особенно для малых $n$. Знаменитый пример Shannon (1951) показал, что даже character-level n-gram уже генерирует похожие на английский цепочки.

## Почему neural LM вытеснили n-gram

1. **Обобщение через distributed representations.** Neural LM знает, что "dog" и "cat" семантически близки, и может обобщить trigram "feed the dog" на "feed the cat" даже если последний не встречался. N-gram такого не умеет — каждая n-грамма независима.

2. **Неограниченный контекст** (теоретически). RNN/Transformer LM могут использовать контекст в сотни/тысячи токенов; n-gram — только $n-1$.

3. **Качество.** На Penn Treebank типичная 5-gram Kneser-Ney PPL ~140, LSTM ~60, Transformer ~20.

4. **Масштабируемость.** Neural LM получает пропорциональный прирост от большего compute; n-gram упирается в sparsity.

См. [[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]] для эволюции архитектур.

## Где n-gram живы до сих пор

### Speech recognition

ASR-системы часто используют hybrid decoder: акустическая модель + n-gram LM. Классические системы (Kaldi) и даже современные гибриды (WFST-based) применяют 3-4-gram KenLM как fast rescorer. Neural LM дороги в онлайн-декодировании.

### Machine Translation evaluation

**[[02 Areas/ML & DL/Concepts/Evaluation/BLEU Score|BLEU]]** — стандартная метрика оценки MT — основана на **precision по 1-gram, 2-gram, 3-gram, 4-gram** совпадениях с reference-переводом. Геометрическое среднее четырёх precision-ов + brevity penalty.

Также n-grams лежат в основе **ROUGE** (summarization), **METEOR**, **chrF** (character n-grams) и других метрик генерации.

### Feature engineering

В классических ML-моделях (SVM, logistic regression для классификации текста) n-gram features (bag-of-n-grams, TF-IDF) остаются быстрым и сильным baseline. В ряде доменов (коротких текстов, spam detection) до сих пор конкурентоспособны с transformers.

### Компрессия и индексация

Поисковые движки используют 1-3 gram индексы. KenLM — популярная библиотека n-gram LM, упакованная для быстрого lookup через memory-mapped файлы.

## Сравнение: n-gram vs neural LM

| Свойство | N-gram | Neural LM (Transformer) |
|----------|--------|--------------------------|
| Представление | Счётчики n-грамм | Dense distributed |
| Контекст | Фиксированный, $n-1$ | Длинный ($10^3$-$10^6$ токенов) |
| Обобщение | По точному совпадению | По семантической близости |
| Обучение | $O$(проход по корпусу) | SGD, часы-недели на GPU |
| Инференс | lookup, микросекунды | Forward pass, миллисекунды+ |
| Память | GB для 3-5gram на больших корпусах | GB-TB весов |
| PPL (WikiText-103) | ~140 | <20 |
| Интерпретируемость | Высокая | Низкая |

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]] — общая парадигма, частью которой являются n-gram
- [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]] — стандартная метрика, впервые применённая к n-gram LM
- [[02 Areas/ML & DL/Concepts/Evaluation/BLEU Score|BLEU]] — n-gram-based метрика для MT
- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]] — что считать «токеном» для n-gram
- [[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]] — первая нейронная LM, вытеснившая n-gram
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — современная замена n-gram
