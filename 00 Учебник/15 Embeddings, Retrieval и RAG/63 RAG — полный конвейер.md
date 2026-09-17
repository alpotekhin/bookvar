---
title: "RAG: от источника до проверяемого ответа"
type: textbook-chapter
status: canonical
last_updated: 2026-09-07
primary_sources:
  - https://arxiv.org/abs/2005.11401
  - https://arxiv.org/abs/2405.14831
  - https://arxiv.org/abs/2411.14199
  - https://web.stanford.edu/~jurafsky/slp3/11.pdf
  - https://github.com/danqi/acl2020-openqa-tutorial
  - https://huggingface.co/learn/cookbook/en/rag_evaluation
  - https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/
source_unit_id:
  - meeting-03-slides-long-term-memory-rag-problem
  - meeting-03-slides-yu-su-hipporag-memory-sequence
  - meeting-03-reading-02-catalogue-record
  - meeting-04-reading-03-catalogue-record
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

*Схема из [Hugging Face RAG Evaluation cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation), Aymeric Roucher. Она одновременно показывает chunking, embedding и построение vector store до production; затем query embedding, retrieval top-k, агрегацию context и generation. В нашей главе к этой исходной схеме добавлены sparse channel, ACL, provenance и stage metrics, описанные в тексте.*

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

<span id="почему-похожий-фрагмент-не-всегда-является-нужной-памятью"></span>
### Почему похожий фрагмент не всегда является нужной памятью

<!-- source_unit_id: meeting-03-slides-long-term-memory-rag-problem -->

Dense retrieval хорошо находит перефразировку одного утверждения, но
многошаговый вопрос может не быть похож ни на один passage, который нужен для
ответа. Рассмотрим запрос: «Какой профессор Stanford занимается neuroscience
of Alzheimer’s?» Один документ связывает исследователя со Stanford, другой —
того же человека с neuroscience, третий уточняет Alzheimer’s. Эмбеддинг всего
вопроса может поднять общие страницы о Stanford или болезни, не восстановив
цепочку через общую сущность.

Это не просто недостаток top-k. У retriever-а нет явной операции «активируй
сущность из первого факта и пройди к связанному второму факту». Итеративный RAG
может сначала извлечь один passage, переписать запрос и повторить поиск, но
каждый дополнительный шаг умножает стоимость и риск semantic drift. В лекции 3
Berkeley, стр. 18–24, этот разрыв формулируется как различие поверхностной
схожести и associative recall.

Графовый retriever хранит сущности, связи и обратные ссылки на исходные
фрагменты, а запрос запускает распространение активации от найденных сущностей.
Это отдельный retrieval channel, а не замена provenance: итоговый контекст всё
равно должен состоять из проверяемых passages.

<span id="hipporag-ассоциативное-извлечение-по-графу"></span>
### HippoRAG: ассоциативное извлечение по графу

<!-- source_unit_id: meeting-03-slides-yu-su-hipporag-memory-sequence -->
<!-- source_unit_id: meeting-03-reading-02-catalogue-record -->

В [HippoRAG](https://arxiv.org/abs/2405.14831) offline-этап извлекает из
passages сущности и отношения, связывает близкие сущности и сохраняет обратные
ссылки на тексты. Online-этап превращает сущности запроса в начальное
распределение Personalized PageRank, распространяет активацию по графу и
возвращает passages найденных узлов. При сравнении с dense baseline фиксируют
corpus snapshot, passage recall, стоимость построения и обновления графа, а
также downstream answer quality; ошибки entity linking проверяют отдельно.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-posttraining-memory/m03-p31-hipporag-index-and-retrieval.png]]

*Верхняя половина слайда показывает offline indexing: извлечение triples,
сопоставление сущностей и построение графа. Нижняя — online retrieval: сущности
запроса задают personalization vector, PageRank распространяет активацию, а
passage links возвращают исходный текст. Источник: Yu Su,
[Berkeley Advanced LLM Agents, meeting 3](https://rdi.berkeley.edu/adv-llm-agents/sp25),
слайд 31; первичный источник: Gutiérrez et al., HippoRAG.*

Вернёмся к вопросу о профессоре. Запрос активирует узлы «Stanford» и
«Alzheimer’s»; граф распространяет вес к сущностям, связанным с обоими
понятиями; passage links возвращают два фрагмента, из которых можно восстановить
ответ. В терминах аналогии авторов entity encoder выполняет pattern separation,
а графовое распространение — pattern completion. Аналогия поясняет механизм,
но не превращает систему в модель человеческой памяти.

Граф не устраняет ошибки retrieval, а переносит их. Неверное слияние тёзок
создаёт ложный путь, пропущенная сущность разрывает нужный, а популярный hub
может собрать PageRank-массу лишь из-за степени вершины. Поэтому графовый канал
сравнивают с BM25, dense и итеративным retrieval на одинаковом corpus snapshot;
измеряют passage recall и стоимость отдельно от качества финального ответа.

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

<span id="openscholar-литературный-поиск-как-итеративный-retrieval-and-synthesis-loop"></span>
### OpenScholar: литературный поиск как итеративный retrieval-and-synthesis loop

<!-- source_unit_id: meeting-04-reading-03-catalogue-record -->

Научный обзор редко строится одним top-k запросом. Формулировка уточняется после
чтения, разные утверждения требуют разных источников, а итоговые citations нужно
сверить с passages. [OpenScholar](https://arxiv.org/abs/2411.14199) оформляет
это как итеративный цикл над коллекцией open-access papers: система извлекает
кандидатные passages, пишет черновой synthesis, с помощью self-feedback
обнаруживает неподдержанные или неполные места, формирует дополнительные
запросы и пересобирает ответ с цитатами.

Представим вопрос о том, помогает ли graph retrieval в multi-hop QA. Первый
поиск возвращает HippoRAG и несколько обзоров. Черновик утверждает, что граф
«всегда быстрее». Citation checker не находит такого общего доказательства и
локализует утверждение. Следующий запрос ищет protocol, corpus и стоимость
baseline; новая редакция ограничивает вывод экспериментами исходной работы.
Такой цикл улучшает не красоту текста, а соответствие «claim → passage → paper».

OpenScholar не следует называть законченным «научным агентом» только из-за
итераций. Его проверяемый контракт — литературный поиск и synthesis. Авторы
оценивают систему на ScholarQABench и отдельно измеряют корректность и качество
цитат; перенос на новый домен требует нового corpus snapshot, expert queries и
проверки coverage. Новая итерация не отменяет versioned corpus, точные source
spans и независимую проверку citation correctness.

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
retrieval без generation. Oracle context даёт контроль качества generator при заранее проверенном evidence, а не математический потолок: другой порядок или формулировка могут изменить результат. Разница с retrieved context помогает диагностировать вклад retrieval/context stages при прочих равных.

## 11. Failure modes

### Как исключение теряется после успешного retrieval

Рассмотрим учебный документ `policy-17/v2`: «Журналы доступа хранятся три
года. Исключение: отладочные журналы хранятся 30 дней». Вопрос пользователя:
«Сколько хранить отладочные журналы?» После разбора обе фразы сохранены,
но chunker разделил правило и исключение:

| артефакт | наблюдаемое содержимое |
|---|---|
| `p4/c1` | «Журналы доступа хранятся три года» |
| `p4/c2` | «Исключение: отладочные журналы хранятся 30 дней» |
| BM25 top-3 | `c2, c1, unrelated` |
| dense top-3 | `c1, unrelated, c2` |
| после reranker | `c1: 0.91, c2: 0.88, unrelated: 0.10` |
| context при лимите одного chunk | только `c1` |
| generated claim | «Отладочные журналы хранятся три года [c1]» |

Цитата существует и первый retrieval нашёл исключение, однако evidence для
данного типа журнала исчезло при сборке контекста. Сначала возвращаем обе фразы
как родительский раздел и повторяем **тот же** запрос генератору. Если ответ
становится «30 дней», этот oracle-context контроль поддерживает гипотезу об
ошибке упаковки. Если ответ всё ещё «три года», нужно исследовать и генератор,
и формулировку запроса; один успешный контроль не доказывает причинность на всём
распределении. Артефакты и ранги здесь заданы для учебного разбора, а не получены
из измеренного production-запуска.

### Классификация по первому месту потери

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
