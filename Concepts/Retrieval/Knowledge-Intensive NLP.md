---
title: "Knowledge-Intensive NLP"
aliases: [knowledge-intensive tasks, KI-NLP, knowledge-intensive NLP tasks]
type: concept
status: legacy
category: Retrieval
papers:
  - "[[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]]"
  - "[[02 Areas/ML & DL/Papers/REALM]]"
  - "[[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]"
courses: []
sources:
  - "[Lewis et al. — RAG for Knowledge-Intensive NLP Tasks (NeurIPS 2020)](https://arxiv.org/abs/2005.11401)"
  - "[Petroni et al. — How Much Knowledge Can You Pack Into the Parameters of a Language Model? (2020)](https://arxiv.org/abs/2002.08910)"
---

# Knowledge-Intensive NLP

## Зачем это нужно: задачи, которые нельзя решить «из головы»

Представь, что тебя спрашивают: «Какова столица Буркина-Фасо?» или «В каком году была основана компания Nvidia?». Ты либо знаешь ответ, либо тебе нужно **посмотреть** в справочнике. Ни один человек не может хранить в памяти все факты мира.

**Knowledge-Intensive NLP** — категория задач, которые «cannot be reasonably expected to perform without access to an external knowledge source» (Lewis et al., 2020). Это задачи, для решения которых нужны **конкретные факты**, которые невозможно вывести из контекста или общих знаний.

Термин введён в [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]] и стал центральным для обоснования RAG-подхода.

## Что делает задачу knowledge-intensive

### Ключевые характеристики

1. **Parametric knowledge недостаточно**: модель не может запомнить все факты в своих весах. Даже T5-11B с 11 миллиардами параметров помнит лишь малую долю фактов из Wikipedia
2. **Знания устаревают**: мир меняется — новые президенты, новые открытия, обновлённая статистика. LLM имеет knowledge cutoff и не знает о событиях после обучения
3. **Требуется точность**: ответ должен быть конкретным, верифицируемым фактом, а не правдоподобным звуком
4. **Long-tail knowledge**: редкие факты (малоизвестные персоналии, нишевые термины) плохо запоминаются в параметрах, потому что встречаются в training data слишком редко

### Отличие от «обычных» NLP задач

| Knowledge-Intensive | «Обычная» NLP задача |
|---------------------|---------------------|
| «Когда родился Эйнштейн?» — нужен факт | «Это предложение позитивное или негативное?» — достаточно понимания языка |
| «Какая валюта в Дании?» — нужен конкретный ответ | «Переведи 'hello' на немецкий» — нужно знание языка |
| «Что говорится о COVID-19 в статье X?» — нужен доступ к документу | «Суммаризуй данный текст» — текст уже предоставлен |

## Типичные задачи

| Задача | Что требуется | Пример | Бенчмарк |
|--------|---------------|--------|----------|
| **Open-domain QA** | Найти ответ на любой вопрос о мире | «What is the birthplace of Einstein?» → «Ulm» | NaturalQuestions, TriviaQA |
| **Fact Verification** | Проверить утверждение по базе знаний | «Einstein was born in Munich» → REFUTES | FEVER |
| **Entity Linking** | Связать упоминание с конкретной сущностью | «Amazon» → Amazon Inc. или Amazon River | AIDA, TAC-KBP |
| **Slot Filling** | Заполнить атрибут сущности | «Apple, CEO: ???» → «Tim Cook» | TAC-KBP |
| **Knowledge Base QA** | Ответить на вопрос по KB (Wikidata, Freebase) | Structured queries | WebQuestions |
| **Multi-hop Reasoning** | Несколько шагов по фактам | «Who was president when Einstein immigrated to the US?» | HotpotQA |
| **Abstractive QA** | Генерировать ответ по найденным документам | Не extract span, а generate | MS-MARCO NLG |

## Два подхода к решению

### 1. Параметрический (Closed-Book)

Все знания **в параметрах** модели. Модель отвечает «из головы», без обращения к внешним источникам.

**Пример**: T5-11B closed-book QA (Roberts et al., 2020):
- NaturalQuestions: 34.5 EM
- TriviaQA: 50.1 EM

**Проблемы**:
- **Hallucination**: модель уверенно генерирует правдоподобные, но неверные факты
- **Outdated knowledge**: знания ограничены training data cutoff
- **Непрозрачность**: невозможно определить, откуда модель «знает» факт
- **Scaling ceiling**: чтобы запомнить больше фактов, нужна модель побольше — expensive и diminishing returns

### 2. Гибридный (Retrieval-Augmented)

Параметрическая модель + **non-parametric retrieval** из внешнего корпуса.

Из [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]]:
- **RAG** (NQ EM: 44.5%) значительно превосходит **T5-11B closed-book** (34.5%)
- TriviaQA: RAG 56.8% vs T5-11B 50.1%
- Модель в десятки раз меньше, но с доступом к Wikipedia

Из [[02 Areas/ML & DL/Papers/REALM]]:
- **REALM 330M** бьёт **T5-11B** на NaturalQuestions (40.4 vs 36.6 EM)
- Retrieval-augmented pre-training ещё сильнее: retriever обучен вместе с моделью

### Почему гибридный подход побеждает

| Аспект | Parametric | Retrieval-Augmented |
|--------|-----------|---------------------|
| **Знания** | В весах (fixed) | В корпусе (updatable) |
| **Актуальность** | Knowledge cutoff | Обновляй corpus |
| **Long-tail facts** | Плохо запоминает | Хорошо находит |
| **Interpretability** | Чёрный ящик | Можно проверить source |
| **Hallucination** | Высокий риск | Grounding в документах |
| **Scaling** | Нужна модель побольше | Нужен corpus побольше |

## Knowledge-Intensive задачи как тест для LLM

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] (§4.3):

**Knowledge-intensive QA — одна из немногих областей, где LLM (GPT-3/4) реально превосходят fine-tuned encoder-only модели:**

- MMLU: GPT-4 86.4% (5-shot) — **SOTA**, задачи требуют обширных знаний
- NQ: GPT-3 few-shot конкурирует с fine-tuned retrieval models
- TriviaQA zero-shot LLM: comparable с many fine-tuned approaches

Это **оправдывает** стоимость крупных LLM: для knowledge-intensive задач scaling действительно помогает (модель запоминает больше фактов). Но RAG всё равно полезен — он добавляет актуальность и interpretability.

## Paradigm shift: от «energy memorization» к retrieval

Осознание того, что языковые модели не могут быть «универсальными энциклопедиями», привело к paradigm shift:

```
2018: BERT → fine-tune на всё → надеяться, что «помнит» факты
2019: T5 → scale up → запомнит больше фактов (closed-book QA)
2020: REALM/RAG → retrieve + read → не нужно всё запоминать
2023: ChatGPT + Browse → LLM + live internet search
2024: RAG as default → production LLM всегда augmented with retrieval
```

REALM и RAG показали, что **внешний retrieval component необходим** для knowledge-intensive задач. Это не костыль — это фундаментальный архитектурный выбор.

## Связь с hallucination

Knowledge-intensive задачи — главное «поле битвы» с hallucination:

- **Factual hallucination**: модель генерирует FactuallyWrong утверждение («Эйнштейн родился в Мюнхене»)
- **Faithfulness hallucination**: ответ не соответствует retrieved documents
- **Intrinsic hallucination**: модель противоречит input

Retrieval → **factual grounding** → меньше hallucinations. Но retrieval не решает проблему полностью: модель может проигнорировать retrieved evidence или неправильно его интерпретировать. Именно это мотивирует [[02 Areas/ML & DL/Concepts/Retrieval/Self-RAG|Self-RAG]] с reflection tokens для проверки faithfulness.

## Evaluation

### Метрики

| Метрика | Задача | Как работает |
|---------|--------|-------------|
| **Exact Match (EM)** | Open-domain QA | Строгое string matching (с нормализацией) |
| **F1** | QA | Overlap между predicted и gold answer |
| **Accuracy** | Fact verification | Correct/Total |
| **KILT benchmark** | Multiple KI tasks | Unified evaluation framework |

**Проблема EM**: слишком строгая метрика. «Albert Einstein» vs «Einstein» — EM = 0, хотя ответ правильный. Модели лучше, чем кажется по EM (paraphrase ответов).

### KILT: unified benchmark

KILT (Knowledge Intensive Language Tasks, Petroni et al., 2021) объединяет 11 KI-NLP задач в единый формат с общим knowledge source (Wikipedia):
- Fact checking, entity linking, slot filling, QA, dialogue
- Единая evaluation pipeline
- Позволяет оценивать retrieval quality отдельно от generation quality

## Key papers

- [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]] — RAG framework для KI-NLP, введение термина (Lewis et al., 2020)
- [[02 Areas/ML & DL/Papers/REALM]] — retrieval-augmented pre-training (Guu et al., 2020)
- [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] — LLM vs fine-tuned на KI tasks

## Related concepts

- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|Retrieval-Augmented Generation]] — основной подход к KI-NLP
- [[02 Areas/ML & DL/Concepts/Retrieval/Open-domain QA|Open-domain QA]] — каноническая KI задача
- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]] — retriever для KI задач
- [[02 Areas/ML & DL/Concepts/Retrieval/REALM|REALM]] — retrieval-augmented pre-training
- [[02 Areas/ML & DL/Concepts/Retrieval/Self-RAG|Self-RAG]] — adaptive retrieval с самокритикой

## Дополнительные ресурсы

- [Lewis et al. — RAG (NeurIPS 2020)](https://arxiv.org/abs/2005.11401) — оригинальная статья, определяющая KI-NLP
- [Petroni et al. — KILT (NAACL 2021)](https://arxiv.org/abs/2009.02252) — unified benchmark для KI tasks
- [Petroni et al. — How Much Knowledge? (2020)](https://arxiv.org/abs/2002.08910) — исследование знаний в параметрах
