---
title: Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer
type: textbook-chapter
status: active
last_updated: 2026-07-22
last_verified: 2026-07-22
---

# Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer

После загрузки весов и токенизатора модель ещё не становится сервисом. Один
вызов `generate` может последовательно обработать prompt и выпустить ответ, но
производственная система одновременно принимает запросы разной длины, меняет
состав пакета после каждого шага, выделяет память под растущие KV-cache,
передаёт токены клиентам и освобождает ресурсы завершённых последовательностей.
Serving engine — это программа, которая согласует все эти операции с работой
GPU.

Названия vLLM, SGLang, TensorRT-LLM и FlashInfer часто оказываются в одной
таблице, хотя они описывают не вполне одинаковые уровни системы. vLLM и SGLang
предоставляют готовые runtimes и серверы. TensorRT-LLM соединяет компиляцию и
оптимизированный runtime, тесно связанный со стеком NVIDIA. FlashInfer прежде
всего является библиотекой ядер для attention, sampling и MoE; её может
использовать другой runtime. Чтобы сравнивать эти проекты осмысленно, сначала
нужно разложить inference-систему на уровни.

## Пять уровней одной системы

Удобно начать не с продуктов, а с вопросов, на которые должна отвечать система.

| Уровень | Главный вопрос | Примеры решений |
|---|---|---|
| API и frontend | как принять запрос, проверить параметры и вернуть поток токенов | OpenAI-compatible HTTP API, tokenizer pool |
| Scheduler и cache manager | какие запросы исполнять сейчас и где хранить их состояние | continuous batching, PagedAttention, RadixAttention |
| Model executor | как разложить модель и запустить один шаг на устройствах | workers, CUDA Graphs, tensor/pipeline parallelism |
| Attention и operator backends | как выполнить attention, GEMM, norms, sampling | FlashAttention, FlashInfer, CUTLASS, Triton kernels |
| Hardware runtime | как управлять streams, памятью, коллективными обменами и графами | CUDA, NCCL, TensorRT |

Эти уровни взаимодействуют, но не заменяют друг друга. Быстрое attention-ядро
не решает проблему очереди запросов. Умный scheduler не компенсирует kernel,
который несколько раз переносит один и тот же тензор между HBM и SRAM. Высокий
prefix-cache hit rate не поможет, если frontend блокирует engine loop медленной
токенизацией.

## Что происходит с одним запросом

Рассмотрим запрос, который уже прошёл HTTP-валидацию. Processor применяет chat
template, токенизирует текст и создаёт внутреннее описание sampling parameters.
Scheduler помещает запрос в очередь и на одном из следующих шагов выделяет ему
KV-блоки. Model executor выполняет prefill, sampler выбирает первый токен, после
чего запрос возвращается в scheduler уже как decode-задача. Цикл повторяется до
EOS, stop condition или лимита длины.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-engine_loop.png]]

*Engine loop в разборе vLLM V1: scheduler формирует работу, model executor
выполняет шаг, output processor обновляет состояние запросов. Источник: Aleksa
Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходное изображение](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/engine_loop.png).*

Системная сложность появляется из-за того, что в каждом цикле находятся разные
запросы. Одни ещё читают длинный prompt, другие выпускают следующий токен,
третьи завершились, четвёртые ожидают свободных KV-блоков. Поэтому основной
объект runtime — не неизменный batch, а изменяющееся множество запросов и их
состояний.

## vLLM: scheduler и блочная память в центре архитектуры

Исторически vLLM вырос из работы о PagedAttention. Её исходная постановка была
системной: непрерывное размещение KV-cache приводило к внутренней и внешней
фрагментации, уменьшало возможный batch и тем самым снижало пропускную
способность. Block table отделила логическую последовательность токенов от
физического расположения cache. Scheduler получил возможность добавлять и
удалять запросы, не перемещая большие непрерывные буферы.

Современный vLLM включает больше компонентов, но эта связь по-прежнему важна.
Engine core содержит scheduler и cache manager; model executor управляет одним
или несколькими workers; model runner подготавливает входные тензоры и metadata
для attention backend. Frontend сервера живёт отдельно от горячего GPU-цикла.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-engine_constructor.png]]

*Компоненты, создаваемые вокруг `LLM`: конфигурация, processor, engine core,
model executor, scheduler и KV-cache manager. Источник: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходное изображение](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/engine_constructor.png).*

Перед первым запросом worker загружает веса, выбирает attention backend,
оценивает занятую память и отдаёт оставшийся бюджет KV-cache. Затем engine может
захватить CUDA Graphs для часто встречающихся размеров исполнения. Захваченный
граф уменьшает накладные расходы запуска CPU → GPU, но требует заранее
подготовленных буферов и набора допустимых shapes. Поэтому eager execution и
graph replay являются разными режимами исполнения одного model runner, а не
разными моделями.

На каждом шаге scheduler задаёт token budget. Для decode-запроса обычно нужен
один новый token position; длинный prefill можно разделить на chunks. Cache
manager проверяет, хватает ли блоков, и создаёт slot mapping: соответствие
логических позиций физическим участкам KV-cache. Model runner упаковывает токены
разных запросов в общий вход, а attention metadata сохраняет границы
последовательностей.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-fwd_pass.png]]

*Один forward pass для запросов с различной историей. Последовательности
упакованы вместе, а metadata связывает позиции с их KV-блоками. Источник: Aleksa
Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходное изображение](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/fwd_pass.png).*

Такой дизайн особенно удобен для обычного completion/chat workload: много
независимых запросов, меняющийся batch, длинная decode-фаза и необходимость
предсказуемо управлять памятью. Подробное чтение конкретных классов вынесено в
[[02 Areas/ML & DL/01 Справочник/Inference/vLLM — анатомия inference engine|анатомию vLLM]],
потому что имена модулей и API меняются быстрее принципов scheduler и cache.

## SGLang: структура программы становится частью расписания

Не вся LLM-нагрузка состоит из независимых prompts. Multi-turn chat повторяет
system prompt и историю. Few-shot evaluation многократно использует одинаковые
demonstrations. Tree search создаёт несколько продолжений общего prefix. Agent
может вернуться к прежнему состоянию после вызова инструмента. Если runtime
видит только плоскую очередь запросов, он узнаёт об этих связях слишком поздно.

Исходная работа SGLang предложила язык для описания language-model programs и
runtime, который использует их структуру. Главным механизмом повторного
использования стал RadixAttention: KV-cache хранится как radix tree по token
prefixes. Рёбра дерева соответствуют последовательностям токенов, узлы хранят
ссылки на cache, а longest-prefix match находит уже вычисленную часть нового
запроса.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/sglang-program-figure2.png]]

*Language-model program с ветвлением и возможности runtime, показанные в
Figure 2 статьи SGLang. Источник: Lianmin Zheng et al.,
[SGLang: Efficient Execution of Structured Language Model Programs](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf).*

При вставке нового запроса дерево может разделить существующее ребро в месте,
где prefixes расходятся. Завершение запроса не обязательно немедленно удаляет
его cache: узлы остаются кандидатами для повторного использования. Когда памяти
не хватает, runtime вытесняет листья по LRU-политике. Cache-aware scheduler
предпочитает запросы с большим совпавшим prefix, но должен учитывать starvation
и общую справедливость очереди.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/sglang-radixattention-figure3.png]]

*Девять состояний radix tree при последовательных запросах, вставках и
вытеснении. Источник: Zheng et al., SGLang paper, Figure 3.*

RadixAttention и PagedAttention решают разные задачи. Paging отвечает за
физическое размещение блоков и уменьшение фрагментации. Radix tree индексирует
содержимое префиксов и помогает находить общие участки. Один runtime может
использовать блочный allocator для памяти и radix tree для prefix reuse.
Подробный пошаговый разбор находится на странице
[[02 Areas/ML & DL/01 Справочник/Inference/SGLang и RadixAttention|SGLang и RadixAttention]].

## TensorRT-LLM: оптимизация графа и runtime как единый стек

TensorRT-LLM решает ту же конечную задачу обслуживания LLM, но делает больший
акцент на компиляции и специализированных реализациях для NVIDIA GPU. Модель
проходит этап подготовки или построения engine: выбираются precision,
quantization, parallelism и набор оптимизированных plugins. Runtime исполняет
полученный engine, управляет in-flight batching и paged KV-cache, а executor API
организует запросы и распределённое исполнение.

Важное следствие: сравнение «vLLM против TensorRT-LLM» нельзя сводить к одной
цифре tokens/s. Результат зависит от того, поддерживается ли конкретная модель
оптимизированным plugin, сколько времени допустимо потратить на build, нужна ли
динамичность PyTorch-экосистемы, какие precision и parallelism разрешены, какова
реальная смесь длин prompts и outputs. TensorRT-LLM способен особенно хорошо
использовать известную заранее конфигурацию NVIDIA, тогда как vLLM часто удобнее
как быстро меняющийся открытый runtime с широким модельным интерфейсом. Это
инженерные профили, а не универсальный рейтинг.

Документация TensorRT-LLM обновляется часто. Поэтому перечень поддерживаемых
plugins, флаги build и server options не фиксируются в основном тексте. Их нужно
проверять в актуальных разделах
[Architecture Overview](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html)
и [Executor](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html) для
конкретной версии.

## FlashInfer: библиотека ядер, а не ещё один scheduler

FlashInfer предоставляет оптимизированные kernels и API для prefill, decode,
append, sampling, cascade attention, sparse/paged KV-cache и операций MoE. Он
не обязан владеть HTTP-сервером или глобальной очередью запросов. Serving runtime
может выбрать FlashInfer как attention backend и продолжить самостоятельно
решать, какие запросы войдут в следующий batch.

Это разделение хорошо показывает границу ответственности. Scheduler создаёт
ragged или paged представление активных последовательностей; kernel library
получает data pointers, page tables, lengths и исполняет attention. В
FlashInfer встречаются layouts NHD и HND: порядок осей page, token/head и head
dimension влияет на адресацию и пригодность конкретного kernel. Математическая
формула attention остаётся той же, но layout определяет, какие участки памяти
читаются совместно.

FlashInfer следует изучать после FlashAttention и PagedAttention. Тогда видно,
что первое объясняет IO-aware алгоритм, второе — управление динамической
памятью запросов, а FlashInfer упаковывает специализированные реализации для
serving workloads. Актуальные layouts и сигнатуры собраны в официальном
[KV-cache layout tutorial](https://docs.flashinfer.ai/tutorials/kv_layout.html).

## Где находится Triton

Triton расположен ещё на один уровень ниже. Это язык и компилятор для написания
GPU kernels через tiles и program instances. С его помощью можно реализовать
fused softmax, quantization, sampling или специализированный attention backend.
Однако Triton сам по себе не ведёт очередь пользовательских запросов, не
принимает HTTP и не решает, чей KV-cache вытеснить.

Поэтому выражение «перейти с vLLM на Triton» смешивает уровни: runtime может
использовать kernels, написанные на Triton. И отдельно существует NVIDIA Triton
Inference Server — другой продукт, не совпадающий с языком Triton. Подробный
маршрут от memory traffic до tiled matmul находится в справочной главе
[[02 Areas/ML & DL/01 Справочник/Inference/Triton и GPU kernels|Triton и GPU kernels]].

## Как выбирать runtime

Выбор начинается с workload, а не с названия проекта.

1. **Модель и hardware.** Помещаются ли веса? Какие dtype, quantization и
   attention backend реально поддерживаются на целевой GPU?
2. **Форма запросов.** Преобладают длинные prompts или длинный decode? Есть ли
   общие prefixes, branching, multi-turn chat или много LoRA adapters?
3. **SLO.** Важнее TTFT, равномерный TPOT, максимальный throughput или goodput
   при заданных хвостовых задержках?
4. **Масштабирование.** Нужны ли tensor/expert parallelism, несколько replicas,
   disaggregated prefill/decode или heterogeneous hardware?
5. **Операционная цена.** Насколько важны OpenAI-compatible API, observability,
   rolling upgrades, стабильность версии и простота отладки?

После этого кандидаты сравниваются на одной модели, одинаковой precision,
одинаковых ограничениях sampling и одном trace распределения длин. Peak
throughput на синтетическом фиксированном prompt не отвечает на вопрос, какой
runtime выдержит производственный SLO.

## Что остаётся стабильным, а что нужно перепроверять

Стабильная часть этой главы — декомпозиция уровней, engine loop, связь scheduler
с KV-cache, различие paging и prefix indexing, граница runtime/kernel library.
Она переживает переименование классов.

Быстро меняются:

- внутренние классы vLLM и SGLang;
- server arguments и CLI;
- список attention backends;
- поддерживаемые quantization formats;
- совместимость CUDA Graphs и compilation;
- конкретные benchmark-лидеры.

Эти сведения должны жить в датированных справочных карточках и лабораторных
работах. Основной учебник объясняет, какие вопросы задать новой версии runtime и
как интерпретировать её устройство.

## Источники и дальнейшее чтение

- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/).
- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm).
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104).
- [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/).
- [FlashInfer documentation](https://docs.flashinfer.ai/).
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache, пакетирование и PagedAttention]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]]

