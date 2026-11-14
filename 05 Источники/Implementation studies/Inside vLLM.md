---
title: vLLM — анатомия inference engine
type: source-note
status: active
last_updated: 2026-11-11
last_verified: 2026-11-11
---

# vLLM — анатомия inference engine

> [!SOURCE] Источник и границы версии
> Статья Алексы Гордича *Inside vLLM* целиком на английском: [сайт автора](https://www.aleksagordic.com/blog/vllm) и [блог vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm).
> В статье разобран [commit `42172ad`](https://github.com/vllm-project/vllm/tree/42172ad)
> от 9 августа 2025 года. Здесь к этому разбору добавлены учебное расписание
> и расчёт KV-блоков. Ссылки на рисунки Гордича ведут к оригиналам;
> у встроенных схем из других источников указана версия.

`transformers.generate()` отвечает на вопрос, как получить продолжение одной
последовательности. Inference engine решает другую задачу: как одновременно
обслуживать множество запросов разной длины, не оставлять GPU без работы, не
терять память на фрагментации и при этом контролировать задержку каждого
пользователя. vLLM соединяет model execution, scheduler, управление KV-cache,
sampling и сетевой слой в один runtime.

## 1. От вызова `generate` к engine core

Минимальный офлайн-вызов скрывает несколько разных подсистем. Processor
проверяет запрос и выполняет токенизацию. Engine core превращает запрос в
последовательность шагов исполнения. Model executor запускает forward pass, а
scheduler перед каждым шагом решает, какие токены каких запросов войдут в
следующий batch.

На [схеме конструктора `LLM` у Гордича](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/engine_constructor.png)
видно, какие компоненты создаёт клиент и что инициализирует Engine Core.

В однопроцессной конфигурации executor управляет одним worker на одной GPU.
Даже здесь engine уже должен заранее оценить память весов и служебных буферов,
выделить оставшуюся память под KV-cache, подготовить attention backend и при
необходимости захватить CUDA Graphs. Распределённая версия размножает workers и
добавляет обмены, но не отменяет эту базовую декомпозицию.

## 2. Engine loop

Каждый незавершённый запрос многократно проходит один цикл:

1. scheduler выбирает работу для следующего шага;
2. KV-cache manager резервирует или возвращает блоки памяти;
3. model executor выполняет forward pass;
4. sampler выбирает новые токены;
5. output processor обновляет состояние, проверяет EOS, stop tokens и лимиты;
6. для завершённых запросов менеджер освобождает блоки; остальные могут попасть
   в следующий шаг. Активный запрос остаётся в наборе `running`, а вытесненный
   планировщик возвращает в `waiting`
   ([реализация `42172ad`](https://github.com/vllm-project/vllm/blob/42172ad/vllm/v1/core/sched/scheduler.py#L204-L274)).

[Схема одного шага Engine Core у Гордича](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/engine_loop.png).

Цикл важнее конкретного API. Offline inference получает все запросы заранее;
асинхронный server принимает новые запросы между шагами. Именно возможность
пересобирать batch после каждого шага лежит в основе continuous batching.

## 3. Prefill и decode требуют разного расписания

Во время **prefill** модель обрабатывает токены запроса и записывает их
ключи и значения в KV-кеш. Если запрос не разбит на части, prefill обрабатывает
весь запрос за один проход. Большие матричные операции обычно дают высокую
арифметическую интенсивность.
Во время **decode** каждый запрос добавляет один токен, но для этого снова читает
веса модели и весь релевантный KV-cache. Поэтому decode часто ограничен
пропускной способностью памяти.

Scheduler не может относиться к ним одинаково:

- длинный prefill хорошо загружает GPU, но способен надолго увеличить waiting
  time остальных запросов;
- decode даёт мало токенов работы на запрос, зато его нужно быстро объединять с
  другими decode-запросами;
- chunked prefill делит длинный prompt на части и позволяет перемежать его с
  decode;
- при нехватке cache blocks runtime откладывает, вытесняет или пересчитывает
  работу в зависимости от принятой политики.

## 4. Continuous batching и paged KV-cache работают вместе

Статический batch фиксирует состав запросов до завершения самого длинного.
Continuous batching после каждого шага удаляет завершившиеся запросы и добавляет
новые. Последовательности при этом не обязаны иметь одинаковую длину: runtime
собирает активные токены в общую «суперпоследовательность», а metadata указывает
attention kernel, к каким KV-блокам относится каждая позиция.

[Подготовка метаданных для нескольких запросов на схеме Гордича](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/fwd_pass.png).

Связь двух механизмов принципиальна. Continuous batching постоянно меняет
состав batch; непрерывный KV-буфер потребовал бы дорогих перемещений. [[02 Areas/ML & DL/01 Справочник/Inference/PagedAttention|PagedAttention]]
разделяет cache на блоки и позволяет менять логическое размещение без
копирования всей последовательности.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-blocks.png|Таблица блоков связывает логический порядок токенов с разнесёнными физическими блоками, к которым обращается запрос для forth.]]

*Woosuk Kwon et al., [PagedAttention, рис. 5](https://arxiv.org/html/2309.06180v1#S4.F5),
[оригинал из материалов CS336](https://raw.githubusercontent.com/stanford-cs336/lectures/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/paged-attention-blocks.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Первый блок последовательности, `Block 0`, находится в нижней строке.
Для вычисления внимания к `forth` ядро читает `Block 0`, `Block 1`
и `Block 2` там, где они лежат в памяти.*

### Один цикл на конкретных запросах

Рассмотрим учебный engine с бюджетом не более четырёх **обрабатываемых входных
токенов** на шаг, блоком cache на два токена и пулом из пяти блоков. Запрос A
имеет три токена входа и лимит три токена ответа, B — два токена входа и лимит
один токен ответа. C с тремя входными токенами приходит только перед шагом 3.
Нет speculative decoding, prefix sharing, предварительного резервирования
будущих токенов или EOS до указанных лимитов.

| Шаг | Токены, поданные в модель | Выданный результат | Cache перед освобождением завершённых запросов |
|---|---|---|---|
| 1 | A: три токена prefill; B: первый токен prefill | A: первый токен ответа; B ещё не закончил prefill | A: 3 позиции/2 блока; B: 1 позиция/1 блок |
| 2 | A: первый токен ответа; B: второй токен prefill | A: второй токен ответа; B: единственный токен ответа, запрос завершён | A: 4 позиции/2 блока; B: 2 позиции/1 блок |
| 3 | A: второй токен ответа; C: три токена prefill | A: третий токен ответа, запрос завершён; C: первый токен ответа | A: 5 позиций/3 блока; C: 3 позиции/2 блока |

На шаге 1 четыре сохранённые позиции занимают шесть слотов: два слота в
незаполненных хвостах пока свободны. После шага 2 блок B возвращается в пул;
бюджет вычисления был использован лишь наполовину, поскольку C ещё не пришёл.
Шаг 3 использует все пять блоков, а после завершения A освобождаются три.
Это две разные проверки допустимости шага: четыре входных токена укладываются
в вычислительный бюджет, а восемь сохранённых позиций из-за округления требуют
десяти физических слотов.

Выданный sampler-ом токен ещё не имеет собственного KV: он появится в cache,
когда этот токен будет подан на следующем шаге. Поэтому последнее выданное
продолжение A не обязано записываться в cache после завершения запроса.
Пример показывает переходы состояний, но не предсказывает задержки и не
утверждает, что каждая версия vLLM выберет именно это расписание.

## 5. Prefix caching

Общий точный префикс допускает reuse только при совпадении модели, адаптера,
позиционных условий и всех влияющих на состояния входов. Полный контракт дан
в [[02 Areas/ML & DL/01 Справочник/Inference/KV-cache#Условия безопасного повторного использования префикса|условиях безопасного повторного использования KV-cache]].
Например, одинаковые token IDs с разными изображениями не задают один cache.
В [документации vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/)
ключ блока учитывает предыдущий prefix, токены блока и дополнительные признаки,
включая LoRA ID, хэши мультимодальных входов и cache salt. Совпадение в
совместимом пространстве ключей позволяет пропустить prefill полных блоков.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-scheduling-review-2026-10/vllm-prefix-cache-overview.png|Пул физических блоков, очередь свободных блоков, индекс хешей и списки блоков по запросам.]]

*Команда vLLM, [Automatic Prefix Caching](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/design/prefix_caching.md),
[исходный PNG](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/assets/design/prefix_caching/overview.png),
[Apache-2.0](/licenses/vllm-23fbc2ec-Apache-2.0.txt).
Индекс находит блок по ключу, а список запроса хранит ссылку на уже
выделенный физический блок.*

У Гордича этот процесс показан в трёх состояниях:
[вычисление хешей](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/prefix_pt1.png),
[заполнение блоков KV](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/prefix_pt2.png),
[повторное использование](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/prefix_pt3.png).

Prefix caching не делает attention дешёвым для новых decode-токенов: новый
query всё равно должен обратиться к сохранённым keys/values. Выигрыш приходится
на повторный prefill. Поэтому cache hit rate зависит от workload: общие system
prompts, few-shot examples и повторно используемые документы помогают, случайные
независимые prompts — почти нет.

## 6. Расширения базового цикла

После понимания scheduler и cache остальные возможности занимают своё место:

- **chunked prefill** ограничивает число prompt-токенов в одном engine step;
- **guided decoding** маскирует недопустимые токены по grammar/FSM;
- **speculative decoding** добавляет draft/verify/accept цикл;
- **disaggregated prefill/decode** исполняет две фазы на разных workers;
- **quantization** меняет формат весов, activations или KV-cache;
- **LoRA serving** добавляет выбор адаптера и batching запросов с разными LoRA;
- **expert parallelism** распределяет experts MoE-модели между устройствами.

Это не независимые флаги. Например, speculative decoding добавляет cache
состояния draft-модели; chunked prefill меняет fairness scheduler; structured
output влияет на sampling; disaggregated serving добавляет передачу KV между
prefill- и decode-workers.

## 7. От одного worker к нескольким GPU

[Переход от одного worker к нескольким процессам на схеме Гордича](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/multiprocexecutor.png).

Tensor parallelism делит матричные операции слоя и добавляет коллективные
обмены. Pipeline parallelism распределяет группы слоёв, но усложняет
микропакетирование. Data parallelism создаёт несколько replicas и требует
маршрутизации запросов. Expert parallelism особенно важен для MoE. Выбор нельзя
делать только по числу GPU: нужно учитывать межсоединение, размер весов, KV-cache,
длину prompt, batch и целевую задержку.

## 8. Serving layer

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-engines-review-2026-10/vllm-v1-processes-tp4.png|Один процесс API-сервера передаёт запросы одному Engine Core, который управляет четырьмя процессами GPU Worker.]]

*Команда vLLM, [V1 Process Architecture, TP=4](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/arch_overview.md),
[исходный PNG](https://raw.githubusercontent.com/vllm-project/vllm/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/assets/design/arch_overview/v1_process_architecture_tp4.png),
[Apache-2.0](/licenses/vllm-db9527a4-Apache-2.0.txt).
При четырёх GPU один запрос проходит через API-сервер и Engine Core,
а один шаг модели вычисляют четыре процесса GPU Worker. Схема показывает
конфигурацию из документации, а не все способы запуска vLLM.*

[Сетевой слой и Engine Core на схеме Гордича](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/server_setup.png).

HTTP streaming, токенизация и клиентские соединения не должны блокировать GPU
loop. Поэтому production server отделяет frontend от engine core и связывает их
асинхронными очередями или IPC. На нескольких replicas появляется ещё одна
задача: load balancer должен учитывать не только число запросов, но и ожидаемое
число токенов, cache locality и состояние workers.

## 9. Как измерять результат

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/first-token-accounting.svg|Сверху этапы обработки на сервере. Снизу клиент получает первый токен отдельно, остальные токены приходят одним фрагментом.]]

*Vijay Janapa Reddi et al., [Harvard ML Systems, т. III, рис. 5: First-Token Accounting](https://harvard-edge.github.io/cs249r_book_dev/vol3/02_foundation_model/02_foundation_model.html#fig-vol3-first-token-accounting),
[исходный SVG](https://raw.githubusercontent.com/harvard-edge/cs249r_book_dev/a32512942fe30f4fa1edb51486f7de53c1318ac7/vol3/02_foundation_model/images/svg/first_token_accounting.svg),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), без изменений.
Скобка Client TTFT заканчивается на Token 1. Остальные токены приходят
одним фрагментом; по нему нельзя восстановить интервалы между шагами декодирования.*

[Диаграмма задержки у Гордича](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/latency_diagram.png).

Прежде чем подбирать scheduler и размер batch, полезно определить, чем
ограничен конкретный этап вычисления. На roofline diagram операция находится
между двумя пределами: слева производительность ограничивает пропускная
способность памяти, справа — вычислительная мощность GPU.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/prefill-vs-decode-roofline.svg|На наклонной части roofline расположен decode с одним запросом; prefill длинного запроса находится ближе к вычислительному потолку.]]

*Vijay Janapa Reddi et al., [Harvard ML Systems, Performance Engineering](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/prefill-vs-decode-roofline.svg),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Точки на графике схематические. При увеличении decode-пакета стоимость
чтения весов распределяется между запросами, поэтому положение точки меняется.*

[Roofline из статьи Гордича](https://github.com/vllm-project/vllm-project.github.io/blob/05f81b0527e2d484c48721f9dc1c49d06849c9f3/assets/figures/2025-vllm-anatomy/roofline.png).

Одного throughput недостаточно. Минимальный отчёт различает:

- **TTFT** — наблюдаемое клиентом время до первого токена, включая очередь,
  обработку запроса, prefill, выбор токена и доставку; перекрывающиеся этапы
  нельзя механически суммировать;
- **inter-token latency** — интервалы между соседними выданными токенами;
  **TPOT** часто агрегирует время после первого токена на число оставшихся
  выходных токенов и требует отдельной оговорки для ответа из одного токена;
- **end-to-end latency**;
- input и output throughput;
- p50/p95/p99, а не только среднее;
- SLO attainment — долю запросов, уложившихся в ограничения задержки;
- goodput — число успешно обслуженных в рамках SLO запросов в единицу времени.

Например, если за 10 секунд завершились 100 запросов и 90 из них выполнили
условия SLO, доля выполнения SLO равна 90%, а goodput за этот интервал —
9 запросов/с. В [DistServe](https://arxiv.org/html/2401.09670v1) используется
связанный критерий: максимальная поддерживаемая интенсивность входящих запросов
при заданной доле выполнения SLO, в пересчёте на GPU. Определение и единицы
нужно указывать рядом с результатом.

Benchmark должен фиксировать модель, dtype, quantization, hardware, распределение
длин prompts/outputs, concurrency, sampling и warm-up. Иначе сравниваются не
engines, а разные нагрузки.

## Что читать дальше

- [[02 Areas/ML & DL/01 Справочник/Inference/PagedAttention|PagedAttention: логические и физические KV-блоки]]
- [[02 Areas/ML & DL/01 Справочник/Inference/SGLang и RadixAttention|SGLang и RadixAttention]]
- [[02 Areas/ML & DL/01 Справочник/Inference/Triton и GPU kernels|Triton и GPU kernels]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention]]

## Источники

- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm), 2025.
- vLLM team, [официальная перепубликация](https://vllm-project.github.io/2025/09/05/anatomy-of-vllm.html).
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
- [vLLM documentation](https://docs.vllm.ai/en/stable/).
- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/).
