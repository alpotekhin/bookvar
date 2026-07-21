---
title: SGLang и RadixAttention
type: concept
status: active
last_updated: 2026-07-21
last_verified: 2026-07-21
---

# SGLang и RadixAttention

SGLang возник из наблюдения, что LLM-приложение редко состоит из одного
независимого вызова. Multi-turn chat повторяет историю, few-shot evaluation —
одни и те же demonstrations, agent loop — system prompt и предыдущие шаги,
self-consistency — общий вопрос с несколькими продолжениями. Если runtime видит
только плоский список запросов, он заново вычисляет одинаковые prefixes.

Исходная работа SGLang соединила две части: язык для описания generation
programs и специализированный runtime SRT. Современный проект значительно шире
первой статьи и является полноценным inference/serving framework. Однако
RadixAttention остаётся удобной точкой входа, потому что показывает, как
структура workload влияет на scheduler и KV-cache.

## Программа содержит больше информации, чем HTTP-запрос

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/sglang-program-figure2.png]]

*Один SGLang program ветвится, запускает генерации параллельно, объединяет
результаты и требует JSON. Красным в оригинале отмечены language primitives, а
справа — возможности runtime, которые используют структуру программы.
Источник: Zheng et al., [SGLang paper](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf), Figure 2. Изображение — crop оригинальной страницы без перерисовки.*

Примитив `fork` явно сообщает, что несколько ветвей имеют общий prefix. `gen`
обозначает место генерации, `select` — выбор из вариантов, constraint — форму
выхода. Frontend может исполнять программу как асинхронный stream или как graph,
а backend получает возможность планировать общие фрагменты вместе.

Для понимания современного SGLang необязательно использовать старый frontend
DSL. Те же runtime-механизмы работают через OpenAI-compatible server. Но paper
важен исторически: он показывает, почему co-design приложения и runtime может
дать больше, чем оптимизация одного model call.

## Radix tree как индекс KV-cache

KV-cache prefix зависит только от token IDs этого prefix. SGLang хранит
соответствие «последовательность токенов → позиции KV-cache» в radix tree.
Ребро дерева может содержать не один token, а целую общую подпоследовательность;
ветвление появляется в месте расхождения запросов.

При поступлении запроса runtime:

1. ищет самый длинный prefix в дереве;
2. повторно использует соответствующие cache slots;
3. выполняет prefill только для несовпавшего suffix;
4. после генерации добавляет новый путь;
5. при нехватке памяти удаляет неиспользуемые листья по LRU-политике.

## Девять состояний RadixAttention

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/sglang-radixattention-figure3.png]]

*Дерево последовательно принимает chat sessions, few-shot batch и
self-consistency branches; общие prefixes разделяются, листья добавляются и
вытесняются. Источник: Zheng et al.,
[SGLang paper](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf), Figure 3. Изображение — crop оригинальной страницы без перерисовки.*

Рисунок нужно читать слева направо по номерам:

- (1–3) первая беседа создаёт путь; следующий turn находит общий prefix;
- (4–5) вторая беседа делит system prompt с первой, а старый лист может быть
  вытеснен;
- (6–7) few-shot requests разделяют demonstrations и расходятся на вопросах;
- (8–9) self-consistency создаёт несколько continuations одного вопроса, а LRU
  освобождает место под новые ветви.

Зелёные узлы в исходном рисунке только что добавлены, синие использованы
повторно, пунктирные оранжевые удаляются. Цвет дополняет, но не заменяет
структуру: reuse виден по общим рёбрам, eviction — по помеченным листьям.

## Cache-aware scheduling

Если waiting queue содержит несколько запросов, порядок влияет на cache hit
rate. Запуск запросов с длинным общим prefix подряд позволяет сохранить нужные
ветви горячими. Случайное чередование несвязанных запросов вызывает cache
thrashing.

Однако максимизация reuse может конфликтовать с fairness. Длинно совпадающие
prompts нельзя бесконечно ставить впереди коротких независимых запросов.
Production scheduler балансирует:

- cache locality;
- время ожидания и приоритет;
- token budget текущего batch;
- доступную KV-память;
- prefill/decode mix;
- SLO по TTFT и inter-token latency.

## RadixAttention и PagedAttention — не конкуренты

[[02 Areas/ML & DL/01 Справочник/Inference/PagedAttention|PagedAttention]] отвечает
на вопрос, где физически хранить KV blocks и как адресовать несмежную память.
RadixAttention отвечает, какие token prefixes уже вычислены и кому их можно
переиспользовать. В исходной статье radix tree указывает на paged cache slots;
continuous batching и tensor parallelism также остаются совместимыми.

Коротко:

| Механизм | Основной объект | Главный выигрыш |
|---|---|---|
| PagedAttention | физические KV blocks | меньше фрагментации и резервов |
| Prefix hash cache | полные хешированные blocks | повторный prefill точного prefix |
| RadixAttention | дерево token prefixes | longest-prefix reuse и cache-aware scheduling |

## Structured output

Вторая центральная идея первой работы — constrained decoding. Обычный FSM
маскирует недопустимые tokens после каждого шага. Если из состояния допустим
только один длинный фрагмент JSON, последовательное декодирование тратит forward
pass на каждый token. Compressed FSM объединяет цепочки однозначных переходов и
позволяет runtime обработать несколько обязательных tokens эффективнее.

Современные serving engines используют разные grammar backends, поэтому
конкретные цифры paper нельзя переносить на любую версию. Устойчивый вывод иной:
structured output — часть decode runtime, а не только post-processing строки.

## Современный SGLang Runtime

Сегодня SGLang включает continuous batching, chunked prefill, paged KV-cache,
speculative decoding, quantization, structured output, tensor/pipeline/data/
expert parallelism, prefill-decode disaggregation и несколько attention/kernel
backends. Поддержка конкретной модели зависит от сочетания architecture,
hardware и backend; список возможностей меняется быстрее, чем paper.

Поэтому страницу следует читать в двух слоях:

1. paper объясняет RadixAttention и co-design программы с runtime;
2. официальная документация фиксирует, что поддерживает текущий release.

## Когда SGLang особенно интересен

- длинные общие system prompts;
- multi-turn chat;
- few-shot evaluation;
- agent/RAG pipelines с повторным контекстом;
- parallel sampling и self-consistency;
- structured JSON/tool calls;
- serving больших MoE/MLA-моделей с современными backends.

При независимых случайных prompts с длинным decode prefix reuse может почти не
помочь: время будет определяться чтением весов/KV, kernels и batching. Это одна
из причин не сравнивать engines по одной рекламной цифре throughput.

## Связанные страницы

- [[02 Areas/ML & DL/01 Справочник/Inference/vLLM — анатомия inference engine]]
- [[02 Areas/ML & DL/01 Справочник/Inference/PagedAttention]]
- [[02 Areas/ML & DL/01 Справочник/Inference/Triton и GPU kernels]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/66 Agent harness и context engineering]]

## Источники

- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), NeurIPS 2024.
- [SGLang documentation](https://docs.sglang.ai/).
- [SGLang repository](https://github.com/sgl-project/sglang).
- [SGLang attention backend selection](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md).
