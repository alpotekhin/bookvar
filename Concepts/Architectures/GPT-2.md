---
title: "GPT-2"
aliases: [GPT 2, GPT2]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
courses: []
sources:
  - "[OpenAI Blog — Better Language Models](https://openai.com/research/better-language-models)"
  - "[Jay Alammar — The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)"
  - "[Radford et al. — Language Models are Unsupervised Multitask Learners (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)"
  - "[GPT-2 — Wikipedia](https://en.wikipedia.org/wiki/GPT-2)"
---

# GPT-2

## Зачем эта модель появилась

До GPT-2 (февраль 2019) в NLP доминировала парадигма **pretrain + fine-tune**: BERT и подобные модели обучались на generic корпусе, а потом дообучались отдельно на каждую задачу с labeled данными. Radford et al. задали другой вопрос: *может ли достаточно большая языковая модель решать задачи без fine-tuning вообще?*

Ответ — да. GPT-2 стала **первой моделью, продемонстрировавшей zero-shot multi-task learning** через simple text conditioning. Это заложило фундамент для GPT-3 и всей парадигмы in-context learning.

## Ключевая идея: Language Models are Unsupervised Multitask Learners

Название статьи — это и есть тезис. Авторы утверждают, что **любая NLP задача может быть сформулирована как предсказание следующего токена** при правильном conditioning:

```
Summarization:  article text + "TL;DR:" → модель генерирует summary
Translation:    english text + "In French:" → модель генерирует перевод
QA:             context + "Q: question A:" → модель генерирует ответ
Reading Comp:   document + question → модель генерирует ответ
```

Формально, вместо моделирования $p(\text{output} | \text{input})$ мы моделируем $p(\text{output} | \text{input}, \text{task})$, где task описан **естественным языком** в самом промпте. Не нужны отдельные головы, не нужен fine-tuning — только достаточно мощная языковая модель.

## Архитектура

GPT-2 — **decoder-only Transformer** с несколькими важными модификациями относительно оригинального GPT:

### Pre-normalization (Layer Norm перед sublayer)

В оригинальном Transformer (Vaswani et al., 2017) LayerNorm применяется **после** sublayer:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

GPT-2 переносит LayerNorm **перед** sublayer:

$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

**Почему это важно:** pre-normalization стабилизирует градиенты при глубоких стеках (48 слоёв в largest модели). Без этого обучение больших моделей нестабильно — gradient magnitudes растут непредсказуемо. Эта модификация стала **де-факто стандартом** для всех последующих LLM (GPT-3, LLaMA, PaLM).

### Четыре размера моделей

| Модель | Параметры | Layers | d_model | Heads | Context |
|--------|-----------|--------|---------|-------|---------|
| GPT-2 Small | 117M | 12 | 768 | 12 | 1024 |
| GPT-2 Medium | 345M | 24 | 1024 | 16 | 1024 |
| GPT-2 Large | 762M | 36 | 1280 | 20 | 1024 |
| **GPT-2 XL** | **1.5B** | **48** | **1600** | **25** | **1024** |

### Другие детали

- **Modified initialization**: residual layer weights масштабируются на $1/\sqrt{N}$, где $N$ — число residual layers. Предотвращает рост активаций при глубоких стеках.
- **Vocabulary**: 50,257 токенов. BPE (Byte Pair Encoding) на уровне байтов — **reversible tokenizer**, способный закодировать любой Unicode текст без unknown tokens.
- **Context window**: 1024 токена (скромно по современным стандартам, но достаточно для демонстрации zero-shot).
- **Final layer norm:** дополнительный LayerNorm после последнего Transformer блока (перед output projection).

## WebText: данные, которые решают всё

### Методология сбора

Вместо стандартных корпусов (Wikipedia, BookCorpus) авторы создали **WebText** — новый датасет:

- **Источник:** все внешние ссылки с Reddit с **≥3 karma** (upvotes). Reddit karma работает как crowd-sourced фильтр качества — люди голосуют за контент, который они считают интересным, информативным или полезным.
- **Результат:** **8 миллионов документов**, ~40GB текста.
- **Wikipedia исключена** — сознательно, чтобы избежать data leakage в downstream QA бенчмарках (многие QA тесты используют Wikipedia как source).

### Топ домены WebText

| Позиция | Домен | Контент |
|---------|-------|---------|
| 1 | Google (Docs, Sites) | Разнообразный UGC |
| 2 | Archive.org | Книги, статьи, исторические документы |
| 3 | Blogspot | Блоги |
| 4 | GitHub | Код, документация, README |
| 5 | NYTimes | Новости |
| 6 | WordPress | Блоги, статьи |
| 7 | Washington Post | Новости |
| 8 | Wikia/Fandom | Fan-created wikis |
| 9-15 | BBC, Guardian, eBay, Pastebin, CNN, Yahoo, HuffPost | Смесь |

**Разнообразие:** новости, блоги, туториалы, форумы, код — всё, что пользователи Reddit считают достаточно интересным, чтобы поделиться и проголосовать.

### Влияние подхода

Этот подход к сбору данных через **social filtering** стал прообразом:
- **OpenWebText** — open-source reproduction WebText
- **The Pile** — расширенный корпус от EleutherAI
- **RefinedWeb** (Falcon) — масштабирование идеи web-only данных
- **FineWeb** (HuggingFace) — современный очищенный web корпус

## Zero-Shot результаты: детальный разбор

GPT-2 (1.5B) установила **zero-shot SOTA** на нескольких бенчмарках — без единого примера из training set этих бенчмарков:

| Бенчмарк | Задача | GPT-2 (zero-shot) | Previous SOTA | Комментарий |
|-----------|--------|-------------------|---------------|-------------|
| Penn Treebank | Language Modeling | **35.76 PPL** | 47.69 (fine-tuned) | Огромный отрыв без fine-tuning |
| WikiText-103 | Language Modeling | **18.34 PPL** | 18.3 (fine-tuned) | Практически идентичен fine-tuned SOTA |
| LAMBADA | Last word prediction | **70.70% Acc** | 59.23% (fine-tuned) | +11.5% zero-shot vs fine-tuned |
| CBT-NE | Named Entity prediction | **89.1% Acc** | 85.3% (fine-tuned) | Точное понимание контекста |
| CNN/DM | Summarization | 21.8 ROUGE-L | — | Reasonable, но не SOTA |
| WMT En→Fr | Translation | 11.5 BLEU | 35.0+ (supervised) | Значительно хуже supervised |

### Где zero-shot работает плохо

- **WinoGrad Schema:** GPT-2 не показала значимого улучшения — задача требует commonsense reasoning, который появляется только при большем масштабе
- **NLI (Natural Language Inference):** слабый результат — задачи на логический вывод требуют более сильного reasoning
- **Translation:** 11.5 BLEU vs 35+ supervised — zero-shot translation работает, но далеко от practical quality

**Вывод:** zero-shot learning работает для задач, которые **естественно встречаются** в web тексте (summarization, QA), но не для задач, требующих специфического формата (NLI) или глубокого reasoning (WinoGrad). GPT-3 (175B) частично решил эту проблему через масштаб + few-shot prompting.

### Scaling behavior

Критически важное наблюдение: zero-shot performance **монотонно растёт** с размером модели:

| Модель | LAMBADA Acc | CBT-NE Acc | PPL (PTB) |
|--------|------------|------------|-----------|
| GPT-2 Small (117M) | 46.3% | 80.9% | 65.85 |
| GPT-2 Medium (345M) | 55.7% | 84.6% | 47.33 |
| GPT-2 Large (762M) | 60.1% | 86.5% | 40.31 |
| GPT-2 XL (1.5B) | **70.7%** | **89.1%** | **35.76** |

Каждый 2x increase in parameters даёт ~5-10% improvement. Это **первое наблюдение scaling laws для zero-shot capabilities**, которое мотивировало GPT-3 (100x scale-up).

## Контроверсия: staged release

GPT-2 стала **первой AI-моделью с ограниченным релизом** из соображений безопасности. OpenAI изначально выпустила только 117M версию, аргументируя это рисками:

### Заявленные риски

- **Генерация fake news:** модель могла создавать убедительные тексты, неотличимые от написанных человеком
- **Spam и manipulation:** автоматическая генерация убедительного контента в масштабе
- **Impersonation:** имитация стиля конкретных авторов и изданий
- **Targeted harassment:** генерация персонализированных оскорбительных текстов

### Хронология релиза

| Дата | Модель | Параметры | Событие |
|------|--------|-----------|---------|
| Feb 2019 | GPT-2 Small | 117M | Анонс + ограниченный release |
| May 2019 | GPT-2 Medium | 345M | Второй этап |
| Aug 2019 | GPT-2 Large | 762M | Третий этап + отчёт по безопасности |
| Nov 2019 | **GPT-2 XL** | **1.5B** | Полный release всех весов |

### Дискуссия в сообществе

**Критики** (большинство исследователей):
- Модель не настолько опасна, чтобы ограничивать доступ — generated text часто содержит ошибки
- Закрытость тормозит исследования и создаёт информационную асимметрию
- Safety concern переоценён — аналогичные модели появятся через месяцы
- OpenAI использует controversy как PR (модель «слишком опасна» → больше внимания)

**Сторонники:**
- Прецедент ответственного release важен для будущих, более мощных моделей
- Время для исследования detection methods
- Norms-setting для индустрии

**Итог:** к ноябрю 2019 OpenAI открыла все веса, заявив, что **не наблюдала сильного misuse**. Но прецедент был создан — и дебаты о staged release продолжаются до сих пор (GPT-4 полностью закрыт, Llama — открыт).

## Наследие: архитектурный шаблон для LLM

GPT-2 заложила **три принципа**, которые определили дальнейшее развитие LLM:

1. **Decoder-only + causal LM = универсальный формат** для любых задач через text conditioning
2. **Pre-normalization** вместо post-normalization — стандарт для всех последующих моделей
3. **Scale → emergent capabilities**: больше параметров → качественно новые способности (zero-shot learning появляется только при достаточном масштабе)

Из [[02 Areas/ML & DL/Papers/GPT 3.0]] (SS2.1): GPT-3 использует **ту же архитектуру**, что и GPT-2, включая pre-normalization, modified initialization, и reversible tokenization. Единственное отличие — масштаб (175B vs 1.5B) и alternating dense/sparse attention layers.

## Сравнение с другими моделями эпохи

| | GPT-2 (Feb 2019) | BERT (Oct 2018) | XLNet (Jun 2019) |
|--|---|---|---|
| Архитектура | Decoder-only | Encoder-only | Decoder-only (permutation) |
| Objective | Causal LM | MLM + NSP | Permutation LM |
| Размер | 1.5B | 340M | 340M |
| Данные | WebText (40GB) | Wiki+Books (16GB) | Wiki+Books+CC (126GB) |
| Downstream | Zero-shot | Fine-tune | Fine-tune |
| Генерация | Да | Нет | Ограниченно |

## Key papers

- [[02 Areas/ML & DL/Papers/GPT 2.0]] — оригинал (Radford et al., 2019)
- [[02 Areas/ML & DL/Papers/GPT 3.0]] — successor; масштабирование до 175B, few-shot learning

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — прямой наследник
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-4|GPT-4]] — multimodal эволюция
- [[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]] — главная демонстрация GPT-2
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]] — training objective
- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]] — BPE tokenizer
- [[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]] — pre-norm архитектура

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — лучшая визуализация архитектуры и генерации
- [OpenAI Blog — Better Language Models](https://openai.com/research/better-language-models) — оригинальный анонс с обсуждением staged release
- [Gwern — GPT-2 as Step Toward General Intelligence](https://www.gwern.net/GPT-2) — глубокий анализ значимости
- [GitHub — openai/gpt-2](https://github.com/openai/gpt-2) — оригинальный код
