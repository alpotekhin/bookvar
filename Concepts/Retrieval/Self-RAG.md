---
title: "Self-RAG"
aliases: [Self-RAG, Adaptive RAG, Self-Reflective RAG, Self-Reflective Retrieval-Augmented Generation]
type: concept
status: legacy
category: Retrieval
papers: ["[[02 Areas/ML & DL/Papers/Self-RAG|Self-RAG]]"]
courses: []
sources:
  - "[Asai et al. — Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (2023)](https://arxiv.org/abs/2310.11511)"
  - "[Self-RAG project page](https://selfrag.github.io/)"
---

# Self-RAG

## Зачем это нужно: проблемы классического RAG

Классический [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]] работает просто: на каждый запрос извлекай $K$ документов → добавь к промпту → генерируй ответ. Но у этого подхода три фундаментальные проблемы:

1. **Слепое извлечение**: RAG **всегда** извлекает документы, даже когда это не нужно. Вопрос «напиши эссе о летних каникулах» не требует внешних фактов — но RAG всё равно добавит нерелевантные пассажи, которые могут **ухудшить** генерацию
2. **Нет контроля качества**: модель не проверяет, поддержан ли её ответ извлечёнными документами. Она может проигнорировать релевантный пассаж или выдумать факты, не подтверждённые источниками
3. **Отсутствие атрибуции**: нельзя проверить, какой документ подтверждает какое утверждение

Asai et al. (2023, UW / Allen AI / IBM) предложили **Self-RAG** — фреймворк, в котором одна LLM (7B/13B) **сама решает**, нужен ли retrieval, **сама оценивает** релевантность документов и **сама критикует** качество своих ответов.

## Ключевая инновация: Reflection Tokens

Self-RAG расширяет словарь модели **четырьмя типами специальных токенов**, которые модель генерирует вместе с обычным текстом:

| Токен | Input | Output | Что решает |
|-------|-------|--------|------------|
| **Retrieve** | $x$ или $x, y$ | yes / no / continue | Нужен ли retrieval на этом шаге? |
| **IsRel** | $x, d$ | relevant / irrelevant | Релевантен ли документ $d$ для запроса $x$? |
| **IsSup** | $x, d, y$ | fully / partially / no support | Подтверждается ли ответ $y$ документом $d$? |
| **IsUse** | $x, y$ | 5, 4, 3, 2, 1 | Насколько полезен ответ $y$ для запроса $x$? |

Критический момент: модель **сама генерирует** эти токены как часть output — это не отдельная reward model или внешний классификатор. Reflection tokens — часть расширенного словаря, предсказываемые standard next-token prediction.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-rag/fig1.png]]
*Сравнение классического RAG (слева) и Self-RAG (справа). Self-RAG адаптивно решает, когда извлекать, оценивает релевантность и поддержку каждого документа (источник: Asai et al., 2023)*

## Как это работает: inference pipeline

### Пошаговый процесс (Algorithm 1 из статьи)

Для каждого сегмента генерации $y_t$:

**Шаг 1: Решение о retrieval.** Модель генерирует токен `Retrieve`. Если `Retrieve = Yes`:

**Шаг 2: Параллельная генерация.** Retriever $R$ извлекает $K$ документов. Модель **параллельно** обрабатывает каждый документ:
- Генерирует `IsRel` (релевантен ли документ?)
- Генерирует continuation $y_t$ на основе каждого документа
- Генерирует `IsSup` (подтверждён ли ответ?)
- Генерирует `IsUse` (полезен ли ответ?)

**Шаг 3: Выбор лучшего.** Segment-level beam search ранжирует $K$ кандидатов по взвешенной сумме reflection token scores.

Если `Retrieve = No` — модель генерирует ответ без retrieval (как обычная LM) и оценивает его через `IsUse`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-rag/fig2.png]]
*Примеры training data для Self-RAG. Слева: задача без retrieval (эссе). Справа: задача с retrieval — документы, reflection tokens и оценки вставлены в training corpus (источник: Asai et al., 2023)*

## Обучение: critic model + generator model

### Шаг 1: Обучение Critic Model $C$

**Сбор данных**: для каждого типа reflection tokens промптим GPT-4 с type-specific инструкцией и few-shot примерами. Например, для `Retrieve`:

> *«Given an instruction, make a judgment on whether finding some external documents from the web helps to generate a better response.»*

Собрано **4k-20k примеров** на каждый тип токена. Ручная проверка показала **>90% agreement** с GPT-4.

**Обучение**: стандартный conditional LM objective на Llama 2-7B:

$$\max_C \, \mathbb{E}_{(x,y),r \sim D_{\text{critic}}} \log p_C(r | x, y)$$

### Шаг 2: Аугментация training corpus

Для каждой пары (input, output) в training data:
1. Critic $C$ решает, нужен ли retrieval для каждого сегмента
2. Если да — Retriever $R$ извлекает top-$K$ документов
3. Critic оценивает каждый документ (IsRel, IsSup) и общее качество (IsUse)
4. Reflection tokens и retrieved passages **вставляются в training corpus**

### Шаг 3: Обучение Generator $M$

Стандартный next-token prediction на расширенном словаре $V \cup \{\text{Critique}, \text{Retrieve}\}$:

$$\max_M \, \mathbb{E}_{(x,y,r) \sim D_{\text{gen}}} \log p_M(y, r | x)$$

Ключевой момент: **retrieved passages маскируются** в loss — модель учится использовать контекст, но не запоминать чужие тексты.

**Преимущество перед RLHF**: critique добавлен **offline** в training corpus. Не нужен PPO, не нужна reward model во время training. Значительно дешевле.

## Кастомизация на inference: один чекпоинт, разные задачи

Уникальная feature Self-RAG — **настройка поведения без переобучения** через веса reflection tokens:

| Задача | Настройка | Зачем |
|--------|-----------|-------|
| Fact verification | Высокий вес `IsSup` | Каждое утверждение должно быть подтверждено |
| Open-ended writing | Низкий порог `Retrieve` | Меньше retrieval, больше creativity |
| Citation-heavy QA | Высокий вес `IsSup` + `IsRel` | Точные цитаты с проверкой |
| General QA | Balanced | Стандартный режим |

Threshold для `Retrieve`: если нормализованная вероятность `Retrieve=Yes` превышает порог — триггерим retrieval. Порог задаётся вручную.

## Результаты

Self-RAG 7B/13B (на базе Llama 2) **превосходит ChatGPT** и retrieval-augmented Llama2-chat:

| Задача | Self-RAG 7B | ChatGPT | Llama2-chat 13B + RAG |
|--------|-------------|---------|----------------------|
| **PopQA** (QA) | **54.9** | 45.7 | 45.7 |
| **PubHealth** (fact verif.) | **72.4** | 70.0 | 67.0 |
| **Bio** (long-form) | **81.2** | — | — |
| **Citation precision** | **70.3** | 65.1 | — |

Self-RAG 7B также превосходит **CoVE 65B** (модель в 9x больше) и **Llama2-chat 13B** на всех задачах.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-rag/fig3.png]]
*Ablation study: вклад каждого компонента Self-RAG. Удаление любого типа reflection token ухудшает результат (источник: Asai et al., 2023)*

## Почему это важно

### Решение проблемы «всегда vs никогда»

Классический RAG — это бинарный выбор: всегда извлекать или никогда. Self-RAG предлагает **третий путь**: модель сама решает, когда retrieval полезен. Для фактологических вопросов — да. Для творческих задач — нет.

### Offline critique вместо RLHF

Self-RAG обходится без PPO/RLHF: critique tokens вставляются в training corpus offline, и модель обучается стандартным LM objective. Это **значительно дешевле** и стабильнее, чем RL-based подходы.

### Модель как собственный критик

Self-RAG — первый шаг к LLM, которая **систематически оценивает** качество собственных ответов в RAG pipeline. Каждый сегмент ответа сопровождается meta-информацией: «нужен ли был retrieval?», «подтверждён ли ответ?», «насколько полезен?».

### Interpretability и verifiability

Reflection tokens создают **аудиторский след**: для каждого утверждения в ответе можно проверить, какой документ его подтверждает и насколько сильно. Это критически важно для production (медицина, юриспруденция, финансы).

## Ограничения

- **Computational cost**: $K$ параллельных генераций на каждый retrieval step → $K$x inference cost
- **Critic quality ceiling**: reflection tokens ограничены качеством GPT-4 оценок при обучении critic'а
- **Segment granularity**: фиксированная гранулярность (предложение) может быть неоптимальной
- **Training data dependency**: требует quality training data с reflection tokens

## Key papers

- [[02 Areas/ML & DL/Papers/Self-RAG|Self-RAG]] — adaptive retrieval + reflection tokens, Self-RAG 7B > ChatGPT на QA и fact verification

## Related concepts

- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]] — базовый фреймворк, который Self-RAG улучшает
- [[02 Areas/ML & DL/Concepts/Retrieval/REALM|REALM]] — retrieval-augmented pre-training (предшественник)
- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]] — retriever, используемый в Self-RAG
- [[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]] — agent framework с рассуждением и действиями
- [[02 Areas/ML & DL/Concepts/Retrieval/Knowledge-Intensive NLP|Knowledge-Intensive NLP]] — класс задач, для которых Self-RAG наиболее полезен

## Дополнительные ресурсы

- [Asai et al. — Self-RAG (2023)](https://arxiv.org/abs/2310.11511) — оригинальная статья
- [Self-RAG project page](https://selfrag.github.io/) — код и обученные модели
- [Lilian Weng — RAG (2024)](https://lilianweng.github.io/posts/2024-07-07-rest-of-rag/) — обзор, включающий Self-RAG в контексте RAG evolution
