---
title: Вопросы по Retrieval и RAG
type: question-index
status: active
last_updated: 2026-07-16
last_verified: 2026-07-16
---

# Вопросы по Retrieval и RAG

## Retrieval

- Чем sparse lexical retrieval отличается от dense retrieval?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]
- Как dual encoder превращает запрос и документ в векторы?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]
- Почему in-batch negatives важны для обучения retriever?
  → [[02 Areas/ML & DL/Courses/MIPT NLP/ANCE — Dense Retrieval Training|ANCE — Dense Retrieval Training]]
- Чем late interaction ColBERT отличается от одного вектора на документ?
  → [[02 Areas/ML & DL/Concepts/Retrieval/ColBERT|ColBERT]]
- Что такое open-domain QA?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Open-domain QA|Open-domain QA]]

## RAG

- Из каких компонентов состоит базовый RAG pipeline?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|Retrieval-Augmented Generation]]
- Чем RAG отличается от помещения всего корпуса в длинный context?
  → [[02 Areas/ML & DL/03 Исследовательские линии/Длинный контекст|Длинный контекст]]
- Какие ошибки retriever передаются generator-у?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]
- Что значит knowledge-intensive NLP?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Knowledge-Intensive NLP|Knowledge-Intensive NLP]]
- Как REALM совместно обучает retrieval и language model?
  → [[02 Areas/ML & DL/Concepts/Retrieval/REALM|REALM]]
- Как Self-RAG решает, когда искать и критиковать ответ?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Self-RAG|Self-RAG]]

## Проектирование и оценка

- Что выбирать единицей индексации: документ, passage или chunk?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]
- Зачем нужен reranker после быстрого retrieval?
  → [[02 Areas/ML & DL/Concepts/Retrieval/ColBERT|ColBERT]]
- Почему faithfulness, answer quality и retrieval recall нужно измерять отдельно?
  → [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]
- Как предотвращать prompt injection из найденных документов?
  → [[02 Areas/ML & DL/Concepts/NLP/Adversarial Promting|Adversarial Prompting]]
