---
title: PagedAttention — управление KV-cache блоками
type: concept
status: canonical
last_updated: 2026-07-21
last_verified: 2026-07-21
---

# PagedAttention — управление KV-cache блоками

Во время авторегрессионной генерации модель сохраняет keys и values всех уже
обработанных токенов. Для одного запроса cache растёт линейно с длиной, но server
одновременно обслуживает запросы разной и заранее неизвестной длительности.
Проблема PagedAttention начинается не с новой формулы attention, а с вопроса:
как выделять и возвращать эту динамическую GPU-память без крупных резервов,
фрагментации и копирования последовательностей.

## Сколько памяти занимает KV-cache

Для стандартного Transformer приблизительный объём cache одного запроса равен

$$
M_{KV}=2LTH_{KV}D_hb,
$$

где $L$ — число слоёв, $T$ — число сохранённых токенов, $H_{KV}$ — число KV-heads,
$D_h$ — размер головы, $b$ — число байт элемента. Множитель два соответствует K
и V. MQA, GQA и MLA уменьшают отдельные множители, но не устраняют динамический
характер памяти.

Если runtime заранее резервирует память под максимальную длину каждого запроса,
большая часть блоков остаётся пустой. Если хранит каждый cache одним непрерывным
участком, рост последовательности требует свободного места рядом или
перемещения. Оба варианта снижают число запросов, которые помещаются в batch.

## Почему непрерывное размещение теряет память

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-fragmentation.png]]

*В исходной постановке vLLM память теряется из-за заранее зарезервированного
пространства и внутренней/внешней фрагментации. Источник: Kwon et al.,
[PagedAttention paper](https://arxiv.org/abs/2309.06180), Figure 2.*

Внутренняя фрагментация — неиспользованный хвост внутри выделенного участка.
Внешняя — свободная память между участками, которую трудно отдать большому
непрерывному запросу. Случайная длина outputs делает точное предварительное
резервирование невозможным.

## Логические и физические блоки

PagedAttention переносит идею виртуальной памяти: логически соседние токены не
обязаны лежать в соседних физических адресах. KV-cache делится на блоки
фиксированного числа токенов. У каждого запроса есть block table, сопоставляющая
номер логического блока с физическим блоком GPU-пула.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-blocks.png]]

*Последовательность разбита на KV blocks; заполняется только последний блок.
Источник: Kwon et al., [PagedAttention paper](https://arxiv.org/abs/2309.06180).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-logical.png]]

*Логические блоки одного запроса отображаются на несмежные физические блоки.
Block table восстанавливает порядок при чтении. Источник: Kwon et al.,
[PagedAttention paper](https://arxiv.org/abs/2309.06180), Figure 3.*

Когда появляется новый токен, manager пишет K/V в свободную позицию последнего
блока. После заполнения он берёт любой свободный физический блок и добавляет его
в table. Уход запроса возвращает все блоки в общий pool. Потеря ограничена
незаполненным хвостом одного последнего блока, а не разницей между фактической и
максимальной длиной.

## Как attention читает разбросанный cache

Обычный kernel предполагает, что cache позиции $0\ldots T-1$ расположен
непрерывно. Paged kernel получает block table и для каждого диапазона логических
позиций вычисляет физический block number и offset внутри него. Затем warps
загружают соответствующие K/V, вычисляют scores и выполняют online softmax или
другую принятую редукцию.

Важно различать:

- **KV block** — единица управления памятью vLLM;
- **CUDA thread block** — группа GPU threads;
- **attention tile** — фрагмент матрицы, обрабатываемый kernel.

Совпадение слова *block* не означает, что эти размеры одинаковы.

## Sharing и copy-on-write

Параллельные samples, beam search и общий prefix могут ссылаться на одни и те же
физические KV blocks. Пока содержимое prefix не меняется, копирование не нужно.
Если ветвь должна изменить разделяемый блок, runtime создаёт копию — аналог
copy-on-write. Это особенно полезно, когда несколько continuations имеют длинный
общий prompt.

Prefix caching расширяет идею во времени: cache blocks могут пережить завершение
конкретного запроса и быть найдены последующим запросом с тем же prefix. Сам
PagedAttention отвечает за адресацию и размещение; политика поиска, хеширования,
LRU и scheduler — отдельные части runtime.

## Связь с continuous batching

После каждого decode step одни запросы завершаются, другие продолжаются, третьи
только поступают. Continuous batching меняет состав batch, а block pool позволяет
одновременно менять распределение памяти без compaction. Scheduler должен
совместно учитывать token budget и свободные blocks: batch, который помещается
по FLOPs, может не помещаться по KV-cache.

## Чего PagedAttention не делает

- Он не уменьшает число операций полного attention по контексту.
- Он не заменяет FlashAttention: FlashAttention сокращает HBM traffic внутри
  attention kernel, а paging управляет долговременным KV-cache между шагами.
- Он не гарантирует prefix reuse сам по себе.
- Он не устраняет чтение cache во время decode.
- Он не определяет fairness и admission control scheduler.

Практический runtime сочетает paging с FlashAttention/FlashInfer/Triton kernels,
continuous batching, chunked prefill, quantized KV и prefix caching. Поэтому
benchmark одной техники нельзя автоматически приписывать всему engine.

## Что смотреть в реализации

При чтении vLLM полезно проследить один токен по четырём структурам:

1. request хранит token IDs и состояние выполнения;
2. scheduler решает, сколько новых tokens запустить;
3. KV-cache manager назначает blocks и строит slot mapping;
4. attention backend использует block table для чтения K/V.

После этого high-level рисунки paper превращаются в конкретный путь данных, а
не остаются аналогией с памятью ОС.

## Связанные страницы

- [[02 Areas/ML & DL/01 Справочник/Inference/vLLM — анатомия inference engine]]
- [[02 Areas/ML & DL/01 Справочник/Inference/SGLang и RadixAttention]]
- [[02 Areas/ML & DL/01 Справочник/Inference/FlashAttention]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention]]

## Источники

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
- vLLM, [Paged Attention design](https://docs.vllm.ai/en/latest/design/paged_attention/).
- Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm), 2025.
- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/).
