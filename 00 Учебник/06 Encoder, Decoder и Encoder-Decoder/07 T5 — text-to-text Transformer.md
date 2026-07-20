---
title: T5 — text-to-text Transformer
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources: [https://jmlr.org/papers/v21/20-074.html]
---

# T5: text-to-text Transformer

Классификация, перевод, ответы на вопросы и краткое изложение обычно имеют
разные входы, выходные головы и код оценки. T5 приводит их к одному интерфейсу:
на входе всегда текст, на выходе тоже текст. Название задачи становится частью
входной строки — например, `translate English to German:` или `summarize:`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/text-to-text-framework.png]]

*Единый text-to-text интерфейс из статьи T5: имя задачи входит в source text, а
ответ любой задачи представлен target text. Источник: [Raffel et al., Figure 1](https://jmlr.org/papers/v21/20-074.html),
авторы Colin Raffel et al.; статья опубликована JMLR с открытым доступом.*

Рисунок читается по строкам. Для перевода префикс сообщает задачу, остальная
часть source содержит предложение, а target — перевод. Для классификации target
всё равно является текстовой меткой. Архитектура не получает новой головы:
различие задач выражено токенами и обучающими наборами.

## Предобучение: восстановление удалённых фрагментов

T5 предобучается не на названиях downstream-задач, а на span corruption.
Несколько последовательных фрагментов заменяются уникальными
sentinel-токенами, а целевая последовательность перечисляет удалённые фрагменты
в исходном порядке.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/t5-span-corruption-figure2.png]]

*Исходная строка, повреждённый вход encoder и цель decoder для span corruption.
Каждый удалённый фрагмент обозначен собственным sentinel-токеном. Источник:
Raffel et al., [T5, Figure 2](https://jmlr.org/papers/v21/20-074.html), 2020.*

Encoder читает повреждённый вход целиком, decoder авторегрессионно
восстанавливает target. Функция потерь считается только по target-токенам. Для
`source [B,S]` encoder memory имеет `[B,S,d]`, состояния decoder — `[B,T,d]`,
а cross-attention scores — `[B,h,T,S]`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/encoder-decoder-architecture.png]]

*Encoder читает условие целиком, а decoder создаёт ответ слева направо и на
каждом слое обращается к encoder memory. Схема из материалов к T5: Colin Raffel
et al., [Exploring the Limits of Transfer Learning](https://jmlr.org/papers/v21/20-074.html).*

T5 применяет относительный position bias, pre-norm-подобное расположение normalization и общий embedding/softmax. Важно отделять эту архитектуру от результатов масштабного C4-рецепта.

Span corruption не превращает T5 в causal language model: условие содержит весь
повреждённый вход, а вероятность задаётся для отдельной целевой
последовательности. Encoder–decoder также хранит source memory и выполняет
cross-attention при генерации.

## C4, objective и controlled study

T5 обучался на Colossal Clean Crawled Corpus (C4), полученном фильтрацией Common Crawl. Фильтры убирали многие короткие, шаблонные и нежелательные страницы, но одновременно задавали английский веб-домен и могли удалять полезные разновидности языка. Paper ценен не только большой моделью: авторы в единой установке сравнили objectives, architectures, unlabeled datasets, transfer methods и scale.

Span corruption выбирает примерно 15% токенов и объединяет соседние выбранные позиции в spans. Каждый span во входе заменяется уникальным sentinel, а target содержит sentinel и удалённый текст. По сравнению с предсказанием каждого masked token отдельно decoder восстанавливает переменную длину, а target существенно короче исходного документа. Это экономит вычисления decoder и обучает generation.

Текстовая метка должна представляться токенизатором, а её формулировка влияет на
функцию потерь. Единый интерфейс упрощает обучение и программный код, но не
устраняет различий между задачами.

## Архитектурные детали и перенос

T5 использует encoder с полной видимостью и causal decoder с cross-attention. Relative position bias добавляется к attention logits и разделяется между слоями внутри блока одного типа. Нормализация не вычитает среднее и расположена до подслоя; исходный FFN использует ReLU, тогда как более поздний T5.1.1 меняет recipe — эти версии нельзя смешивать.

После pretraining все задачи fine-tune-ятся той же maximum-likelihood целью. Multi-task setting смешивает примеры нескольких datasets; sampling rate определяет, не будет ли крупный набор подавлять малый. В evaluation строка декодируется и затем преобразуется в metric нужной задачи.

## Ключевой результат, ограничения и наследие

Controlled comparisons показали, что denoising objectives хорошо подходят encoder–decoder transfer, а масштаб повышает качество, но выбор данных и mixture также существенен. Модель 11B достигла сильных результатов на GLUE, SuperGLUE, summarization и QA, однако не каждое улучшение переносилось одинаково.

Text-to-text объединяет программный интерфейс, но не устраняет семантическую разницу задач. Exact-match может наказать правильный paraphrase; generation label тратит последовательные decode steps; closed-set classification иногда эффективнее линейной головой. C4 несёт веб-смещения, а полный attention ограничивает длину.

Наследие T5 — task prefixes, span corruption и единый seq2seq API. FLAN и instruction tuning развивают идею текстового описания задачи, но используют другие смеси и objectives; приписывать их свойства исходному T5 нельзя.

## Источники

- [Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)
- [Official T5 repository](https://github.com/google-research/text-to-text-transfer-transformer)
- [Hugging Face T5 documentation](https://huggingface.co/docs/transformers/model_doc/t5)
