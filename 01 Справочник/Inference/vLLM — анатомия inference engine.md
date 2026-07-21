---
title: vLLM — анатомия inference engine
type: concept
status: active
last_updated: 2026-07-21
last_verified: 2026-07-21
---

# vLLM — анатомия inference engine

> [!SOURCE] Основной материал
> Эта страница служит русскоязычным путеводителем по статье Алексы Гордича
> [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm).
> Оригинал следует читать целиком: автор постепенно поднимается от одного
> синхронного процесса на одной GPU до распределённого serving. Разбор привязан
> к [vLLM commit `42172ad`](https://github.com/vllm-project/vllm/tree/42172ad)
> от 9 августа 2025 года, поэтому имена классов могут меняться, а устройство
> scheduler, KV-cache и граница prefill/decode остаются главными объектами чтения.

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-engine_constructor.png]]

*Компоненты, создаваемые вокруг `LLM`: конфигурация, processor, engine core,
model executor, scheduler и KV-cache manager. Иллюстрация: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm), оригинал в
[официальном blog repository](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/engine_constructor.png).*

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
6. завершённые запросы освобождают блоки, а незавершённые возвращаются в очередь.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-engine_loop.png]]

*Жизненный цикл запроса внутри engine loop. Иллюстрация: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходный файл](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/engine_loop.png).*

Цикл важнее конкретного API. Offline inference получает все запросы заранее;
асинхронный server принимает новые запросы между шагами. Именно возможность
пересобирать batch после каждого шага лежит в основе continuous batching.

## 3. Prefill и decode требуют разного расписания

Во время **prefill** модель обрабатывает сразу все токены prompt. Большая
матричная работа обычно даёт сравнительно высокую арифметическую интенсивность.
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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-fwd_pass.png]]

*Подготовка metadata и один forward pass для запросов разной длины. Иллюстрация:
Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходный файл](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/fwd_pass.png).*

Связь двух механизмов принципиальна. Continuous batching постоянно меняет
состав batch; непрерывный KV-буфер потребовал бы дорогих перемещений. [[02 Areas/ML & DL/01 Справочник/Inference/PagedAttention|PagedAttention]]
разделяет cache на блоки и позволяет менять логическое размещение без
копирования всей последовательности.

## 5. Prefix caching

KV-cache определяется последовательностью всех предыдущих токенов. Если два
запроса имеют общий точный prefix, вычисленные K/V этого prefix можно повторно
использовать. vLLM хеширует полные cache blocks с учётом предыдущего prefix;
совпадение позволяет пропустить часть prefill.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-prefix_pt1.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-prefix_pt2.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-prefix_pt3.png]]

*Пошаговый пример построения, завершения и повторного использования cache
blocks. Иллюстрации: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm), исходные файлы
[1](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/prefix_pt1.png),
[2](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/prefix_pt2.png),
[3](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/prefix_pt3.png).*

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-multiprocexecutor.png]]

*Переход от одного worker к multiprocess executor. Иллюстрация: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходный файл](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/multiprocexecutor.png).*

Tensor parallelism делит матричные операции слоя и добавляет коллективные
обмены. Pipeline parallelism распределяет группы слоёв, но усложняет
микропакетирование. Data parallelism создаёт несколько replicas и требует
маршрутизации запросов. Expert parallelism особенно важен для MoE. Выбор нельзя
делать только по числу GPU: нужно учитывать межсоединение, размер весов, KV-cache,
длину prompt, batch и целевую задержку.

## 8. Serving layer

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-server_setup.png]]

*Сетевой слой отделён от engine core: API server принимает и валидирует запросы,
а engine client передаёт их runtime. Иллюстрация: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходный файл](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/server_setup.png).*

HTTP streaming, токенизация и клиентские соединения не должны блокировать GPU
loop. Поэтому production server отделяет frontend от engine core и связывает их
асинхронными очередями или IPC. На нескольких replicas появляется ещё одна
задача: load balancer должен учитывать не только число запросов, но и ожидаемое
число токенов, cache locality и состояние workers.

## 9. Как измерять результат

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-latency_diagram.png]]

*Разложение пользовательской задержки. Иллюстрация: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходный файл](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/latency_diagram.png).*

Прежде чем подбирать scheduler и размер batch, полезно определить, чем
ограничен конкретный этап вычисления. На roofline diagram операция находится
между двумя пределами: слева производительность ограничивает пропускная
способность памяти, справа — вычислительная мощность GPU.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-roofline.png]]

*Roofline-модель связывает арифметическую интенсивность операции с достижимой
производительностью. В типичной LLM-нагрузке prefill ближе к
compute-bound-режиму, а decode с малым batch — к memory-bound. Иллюстрация:
Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходный файл](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/roofline.png).*

Одного throughput недостаточно. Минимальный отчёт различает:

- **TTFT** — время до первого токена: queueing + prefill;
- **inter-token latency / TPOT** — темп decode после первого токена;
- **end-to-end latency**;
- input и output throughput;
- p50/p95/p99, а не только среднее;
- goodput — долю запросов, уложившихся в SLO.

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
