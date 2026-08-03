---
title: "RAG: от источника до проверяемого ответа"
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://arxiv.org/abs/2005.11401
  - https://web.stanford.edu/~jurafsky/slp3/11.pdf
  - https://github.com/danqi/acl2020-openqa-tutorial
  - https://huggingface.co/learn/cookbook/en/rag_evaluation
  - https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/
---

# RAG: от источника до проверяемого ответа

Retrieval-augmented generation (RAG) полезна тогда, когда ответ должен опираться на
данные, доступные отдельно от параметров модели: внутренние документы,
обновляемые факты, большие коллекции или источники, которые требуется
процитировать. Сам факт добавления найденных фрагментов в запрос к модели ничего не
гарантирует. Доказательство может исчезнуть при разборе файла, разбиении,
поиске, переранжировании, укладке контекста или генерации. RAG поэтому следует
рассматривать как последовательность проверяемых преобразований.

## 1. Что добавила исходная архитектура RAG

Lewis et al. формализовали документ $z$ как скрытую переменную. Retriever
$p_\eta(z\mid x)$ выбирает документы, generator $p_\theta(y\mid x,z)$ создаёт
ответ, а вероятность маргинализуется по верхним документам:

$$
p(y\mid x)\approx\sum_{z\in\operatorname{top-k}(x)}
p_\eta(z\mid x)p_\theta(y\mid x,z).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/rag-original-architecture.png]]

*Рисунок 1 из [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401),
Lewis et al., NeurIPS 2020. Пунктиром показано обучение retriever через итоговую
вероятность ответа; варианты RAG-Sequence и RAG-Token различаются тем,
фиксируется ли документ для всей выходной последовательности.*

Современное приложение часто не обучает поисковую и генеративную модели совместно: оно
индексирует собственный корпус, извлекает фрагменты и помещает их в контекст
готовой LLM. Это инженерно проще, но термин скрывает другую систему. Формула
из статьи объясняет происхождение идеи, а производственный конвейер требует отдельных
контрактов для данных, retrieval, context и ответа.

Практический конвейер целиком хорошо собран в HF cookbook: верхняя половина
рисунка — офлайн-подготовка корпуса, нижняя — обработка запроса в production.
Подписанные синим вопросы на рисунке — не декоративные комментарии, а параметры,
которые авторы затем меняют по одному в оценочном эксперименте.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/hf-rag-evaluation-workflow.png]]

*Схема из [Hugging Face RAG Evaluation cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation), Aymeric Roucher. Она одновременно показывает chunking, embedding и построение vector store до production; затем query embedding, retrieval top-k, агрегацию context и generation. В нашей главе к этой исходной схеме добавлены обязательные sparse channel, ACL, provenance и stage metrics, описанные в тексте.*

## 2. Сначала — задача и набор проверки

До выбора базы собирают набор запросов, представляющий будущую нагрузку. Для каждого фиксируют:

- допустимые источники и момент времени;
- source spans, достаточные для ответа;
- варианты правильного ответа и условия отказа;
- требование к цитатам, полноте и задержке;
- права пользователя.

Без этого невозможно обоснованно выбрать chunk size или embedding model.
Проверка одного красивого demo создаёт ложную уверенность: система может
отвечать благодаря памяти LLM, даже если retrieval сломан. Контрольные примеры
должны включать отсутствующий ответ, конфликт версий, редкие идентификаторы,
несколько необходимых документов и prompt injection внутри источника.

## 3. Ingestion: сохранить структуру и происхождение

Ingestion начинается с обнаружения документов и заканчивается не «текстом», а
структурированными блоками с происхождением. PDF parsing должен различать
колонки, заголовки, подписи, таблицы, сноски и повторяющиеся колонтитулы. Для
HTML нужны canonical URL и дата получения; для репозитория — commit; для
облачного документа — revision и ACL.

У каждого блока сохраняют document ID, версию, путь заголовков, страницу или
character offsets, язык, время действия, автора и права. Контроль качества
измеряет долю успешно разобранных документов, пустые страницы, потерянные
таблицы, дубликаты и расхождение с исходным текстом. Если parser переставил
ячейки таблицы, никакой retriever это не исправит.

## 4. Chunking — гипотеза, проверяемая на запросах

Фрагмент должен быть достаточно мал, чтобы иметь одну поисковую тему, и
достаточно велик, чтобы сохранить доказательство. Fixed windows просты и
воспроизводимы; структурные chunks уважают разделы; sentence windows возвращают
окружение найденного предложения; child-parent retrieval ищет по малому child,
но передаёт LLM более крупный parent.

Overlap помогает на границах, но увеличивает индекс и создаёт почти одинаковые
кандидаты. Семантическое разбиение не автоматически лучше: embedding-based
граница может разделить определение и исключение или дать нестабильные версии
после смены encoder. Стратегии сравнивают на одном evaluation set, измеряя
retrieval recall, достаточность контекста, дубликаты и число токенов.

Таблицы, код и изображения требуют собственных units. Таблицу часто
индексируют вместе с заголовками строк и столбцов; код — по функции или классу;
изображение — по подписи, OCR и мультимодальному embedding, сохраняя ссылку на
оригинал.

## 5. Индексирование

Обычно полезны как минимум два канала: BM25 для точных терминов и dense index
для перефразировок. Metadata index обеспечивает фильтры по дате, продукту,
языку и ACL. Индекс версионируют по corpus snapshot, parser, chunker, encoder и
index settings.

Перед approximate index измеряют точный Flat baseline. Затем подбирают HNSW,
IVF или PQ по кривой task recall–latency–memory. Быстрый ANN, потерявший
доказательство, создаёт downstream hallucination, которую легко ошибочно
приписать generator.

## 6. Обработка запроса и retrieval

Запрос классифицируют лишь там, где ветвление действительно помогает: нужен ли
поиск; какие коллекции и фильтры допустимы; требуется ли декомпозиция. Исходный
запрос всегда сохраняют. Query rewriting проверяют на semantic drift;
multi-query и HyDE повышают recall ценой шума и задержки.

Sparse и dense lists объединяют RRF или обученным fusion. Каждый кандидат
сохраняет канал, raw score, rank, chunk ID и причину фильтрации. Retrieval
оценивают recall@k и nDCG до генерации. Если evidence не найден, хороший prompt
не восстановит его надёжно.

## 7. Переранжирование

Cross-encoder перечитывает запрос вместе с каждым кандидатом и улучшает точный
порядок; ColBERT сохраняет token-level late interaction. Reranker обучают на
ошибках реального retriever, а не на случайных negatives. Его $k$ выбирают по
candidate recall и бюджету задержки.

После reranking полезен deduplication по source span, а не только по cosine:
overlapping chunks одного абзаца не должны вытеснять независимое доказательство.
Для сложного ответа можно обеспечить diversity по документам или подзапросам.

## 8. Сборка контекста

Top-k — ещё не prompt. Context builder решает, какие части войдут в токенный
бюджет и в каком порядке. Он может расширить child до parent, вернуть соседние
предложения, удалить повторы и сгруппировать фрагменты по подзадачам.

Каждый фрагмент получает стабильный идентификатор, заголовок и источник.
Инструкции отделяют данные от управляющего текста: содержимое retrieved
documents нельзя исполнять как команды. «Lost in the middle» проверяют
перестановкой одного и того же evidence; критический ответ не должен зависеть от
случайного места fragment.

Context compression допустим только при сохранении отображения summary claim →
source span. Иначе система экономит токены ценой непроверяемого нового текста.

## 9. Генерация и цитаты

Generator получает вопрос, явно отделённые источники и контракт ответа. Он
должен уметь: ответить по evidence; сообщить о недостаточности; представить
конфликт версий; отказаться от вывода, которого источники не поддерживают.

Цитата проверяется на двух уровнях. Citation correctness спрашивает,
подтверждает ли указанный span утверждение. Citation completeness — все ли
проверяемые утверждения имеют поддержку. Ссылка только на документ без точного
span затрудняет аудит и скрывает ошибку chunking.

Параметрическая память модели не следует считать источником. Если ответ верен,
но отсутствует в retrieved context, grounded RAG должен либо найти evidence,
либо обозначить ответ как неподтверждённый.

## 10. Оценивание по этапам

End-to-end score не локализует ошибку. Минимальная таблица:

| Этап | Проверяемый вопрос | Артефакт или метрика |
|---|---|---|
| Ingestion | доказательство сохранилось? | parse coverage, визуальная сверка |
| Chunking | один fragment достаточен? | evidence containment, дубликаты |
| Retrieval | evidence попало в кандидаты? | recall@k, hit@k, nDCG |
| Reranking | evidence вошло в context budget? | nDCG, oracle gap |
| Context | evidence не потеряно и не искажено? | context recall, source mapping |
| Answer | claims поддержаны? | faithfulness, citation precision/recall |
| Product | задача решена вовремя? | success rate, abstention, p95, cost |

[Hugging Face RAG Evaluation cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation)
показывает практический evaluation loop; RAGAS предлагает автоматические
context precision/recall, faithfulness и answer relevance. LLM-as-judge удобен
для масштаба, но калибруется на человеческой выборке и проверяется на bias к
длине, стилю и модели-судье.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/hf-rag-settings-accuracy.png]]

*Результат последовательных ablations из того же [HF cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation). На конкретном учебном наборе автор сначала подбирает chunk size, затем embedding model, reranker и reader. Числа нельзя переносить на другой корпус; ценность рисунка — в экспериментальном порядке: менять один компонент и повторно измерять один и тот же набор вопросов.*

Нужны ablations: generator без retrieval, oracle context, retrieved context и
retrieval без generation. Oracle context показывает потолок generator;
разница между oracle и retrieved — ущерб retrieval/context stages.

## 11. Failure modes

1. **Нет документа:** ingestion coverage или freshness, а не prompting.
2. **Документ есть, chunk не содержит evidence:** изменить parsing/chunking.
3. **Evidence не найдено:** sparse/dense/query/ANN/filters.
4. **Найдено, но отброшено:** fusion или reranker.
5. **В context есть, модель игнорирует:** порядок, шум, instruction, capacity.
6. **Ответ поддержан неверной цитатой:** claim–span alignment.
7. **Старая версия выше новой:** temporal metadata и deduplication.
8. **Инструкция из документа управляет моделью:** retrieval prompt injection.
9. **Ответ верен только из parametric memory:** grounding failure.
10. **Средняя метрика хороша, редкие запросы провалены:** slice evaluation.

## 12. Эксплуатация

Работающая RAG-система журналирует версию запроса после переформулирования, идентификаторы кандидатов и
оценки, собранный контекст, ответ, цитаты, модель и задержку каждого этапа.
Чувствительный текст маскируют и ограничивают retention, но без stage traces
невозможно разбирать ошибки.

ACL применяют до передачи текста generator и повторно проверяют после cache.
Ключ кеша включает права и corpus version. Обновления используют blue-green
index; удаление распространяется на chunks, embeddings, caches и trace stores.

Наблюдаемость разделяет p95 parsing/index freshness, retrieval, reranking и
generation. Деградация может перейти в безопасный режим: lexical-only search,
меньший reranker, ответ со ссылками без генерации или явный отказ. Режим
выбирается заранее, а не после аварии.

## 13. Когда RAG не нужен

Если небольшой набор документов целиком помещается в контекст и редко
меняется, full-context baseline может быть проще и точнее. Если задача требует
строгого вычисления над таблицами, лучше SQL или tool call. Если ответ зависит
от устойчивого нового поведения, а не от внешнего знания, нужен fine-tuning.
RAG оправдан, когда retrieval даёт измеримый выигрыш в качестве, проверяемости
или стоимости.

## Источники и курсы

- [Lewis et al., RAG](https://arxiv.org/abs/2005.11401) — исходная архитектура
  с latent documents.
- [Jurafsky & Martin, SLP3 chapter 11](https://web.stanford.edu/~jurafsky/slp3/11.pdf) —
  IR, dense retrieval и RAG в академической последовательности.
- [ACL OpenQA Tutorial](https://github.com/danqi/acl2020-openqa-tutorial) —
  retriever–reader, evaluation и dense retrieval.
- [DeepLearning.AI, Retrieval Augmented Generation](https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/) —
  ingestion-to-evaluation как практический курс; архитектурные утверждения
  сверяются с первичными статьями.
- [Hugging Face RAG Evaluation cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation) —
  воспроизводимый evaluation workflow.
- [RAGAS](https://arxiv.org/abs/2309.15217),
  [ARES](https://github.com/stanford-futuredata/ARES) и
  [CRAG benchmark](https://github.com/facebookresearch/CRAG) — автоматическая и
  benchmark-оценка с разными ограничениями.
