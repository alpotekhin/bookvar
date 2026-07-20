---
title: Вопросы по Retrieval и RAG
type: question-index
status: active
last_updated: 2026-07-20
last_verified: 2026-07-20
---

# Вопросы по Retrieval и RAG

## Embeddings и обучение пространства

- Как получить один embedding последовательности из токенных представлений и
  почему pooling является частью модели? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning#1. От токенных представлений к представлению последовательности|От токенов к embedding]]
- Почему hidden state обычного BERT не становится качественным sentence
  embedding без специальной задачи обучения? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning#2. Почему обычный BERT не является готовой моделью предложений|BERT и sentence embeddings]]
- Как positive pairs, hard negatives и in-batch negatives определяют геометрию
  embedding space? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning#4. Пары определяют геометрию|Пары и геометрия]]
- Какие offline- и downstream-проверки нужны embedding-модели помимо средней
  cosine similarity? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning#7. Как проверять эмбеддинг|Оценивание embeddings]]

## Retrieval и reranking

- Чем BM25, dense retrieval и hybrid retrieval различаются по сигналу
  соответствия запроса документу? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid#6. Гибридная выдача|Гибридная выдача]]
- Почему документ, passage и chunk создают разные retrieval-задачи? →
  [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid#4. Что именно индексировать|Единица индексирования]]
- Как query rewriting, metadata filters и decomposition меняют recall и риск
  потери исходного намерения? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid#5. Query processing|Обработка запроса]]
- Почему retrieval нужно диагностировать по candidate recall до оценки
  generator? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid#8. Диагностика retrieval|Диагностика retrieval]]
- Чем bi-encoder и cross-encoder различаются по доступному взаимодействию query
  и document и по стоимости? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction#1. Bi-encoder и cross-encoder решают разные вычислительные задачи|Bi-encoder и cross-encoder]]
- Как late interaction ColBERT сохраняет токенные совпадения без полного
  cross-encoder прохода для всего корпуса? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction#4. Late interaction: промежуточная точка ColBERT|Late interaction]]
- Как построить cascade retriever → reranker при фиксированном latency budget? →
  [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction#5. Cascade и бюджет вычислений|Retrieval cascade]]

## RAG как система

- Какие компоненты добавляет исходная архитектура RAG к parametric language
  model? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#1. Что добавила исходная архитектура RAG|Исходная архитектура RAG]]
- Почему chunking следует выбирать по реальным запросам, а не по универсальному
  числу символов? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#4. Chunking — гипотеза, проверяемая на запросах|Chunking как гипотеза]]
- Как собрать контекст, сохранив provenance, diversity и доступный token budget?
  → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#8. Сборка контекста|Сборка контекста]]
- Что требуется от генератора, чтобы цитата подтверждала конкретное утверждение,
  а не просто указывала на найденный документ? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#9. Генерация и цитаты|Генерация и цитаты]]
- Почему retrieval recall, context relevance, faithfulness и answer quality
  нужно измерять отдельно? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#10. Оценивание по этапам|Оценивание RAG по этапам]]
- Как отличить отсутствие нужного документа, ошибку поиска, потерю при
  reranking и игнорирование контекста генератором? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#11. Failure modes|Failure modes RAG]]
- Какие версии индекса, embeddings, prompts и источников нужны для
  воспроизводимого обновления RAG? → [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#12. Эксплуатация|Эксплуатация RAG]]
- Когда длинный prompt, fine-tuning или обычная база данных лучше RAG? →
  [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#13. Когда RAG не нужен|Когда RAG не нужен]]
