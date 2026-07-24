---
title: Network, storage и distributed checkpoints
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44g. Network, storage и distributed checkpoints

Collective не живёт в абстрактном «канале». Байты проходят HBM, accelerator interconnect, NIC, кабель, switches и обратный путь. Пять уровней Harvard — link, transport, topology, fabric behavior, cluster design — помогают найти место, где nominal bandwidth перестал быть effective.

## Что нужно знать и чему научимся

Нужны collectives из 44a и distributed state из 44e. После главы можно провести tensor по HBM→NIC→fabric→storage, рассчитать lower bound, выбрать метрики для RoCE/PFC и спроектировать атомарный checkpoint без rank-0 bottleneck.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/five-level-model.svg]]

*Источник: Harvard Edge ML Systems Book, [Network Fabrics, figure `fig-network-five-level-model`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd), CC BY-NC-SA 4.0.*

## RDMA, GPUDirect, InfiniBand и RoCE

RDMA позволяет NIC читать/писать remote registered memory без kernel TCP data path. GPUDirect RDMA даёт NIC прямой DMA к GPU memory: исчезают GPU→host и host→GPU staging copies и CPU packet processing.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/gpudirect-data-path.svg]]

*Источник: Harvard Edge ML Systems Book, [Network Fabrics, section `sec-network-fabrics-rdma`, figure `fig-gpudirect-data-path`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd), CC BY-NC-SA 4.0.*

InfiniBand предоставляет credit-based lossless fabric и RDMA как единый стек. RoCEv2 переносит RDMA поверх Ethernet/IP, но нуждается в согласованных PFC/ECN и congestion control: pause storms и head-of-line blocking могут сделать «lossless» сеть нестабильной. Выбор проверяют не названием протокола, а tail collective latency, retransmission/ECN/PFC counters и bisection bandwidth.

Топология задаёт oversubscription и число hops. Для TP с несколькими exchanges на слой crossing медленной spine связи умножается на число слоёв; DP all-reduce можно иерархически свернуть внутри узла и только затем пересечь spine. Topology-aware rank mapping должен совпадать с collective hierarchy.

### Расчёт

175B gradients в BF16 — 350 GB logical tensor. Ring на 1024 rank передаёт на каждом почти $2M=700$ GB. При 50 GB/s идеальный bandwidth lower bound — около 14 s на rank, если весь global tensor действительно реплицирован на каждом rank и нет sharding/overlap. Формула немедленно показывает, почему такой DDP режим непрактичен и нужны sharding, hierarchy и overlap.

## Storage path и checkpoint storm

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/complete-data-path.svg]]

*Источник: Harvard Edge ML Systems Book, [Data Storage, figure `fig-complete-data-path`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/data_storage/data_storage.qmd), CC BY-NC-SA 4.0.*

Training data обычно движутся object/parallel storage → local cache → host page cache/pinned buffers → GPU. Checkpoint идёт обратно. Если 1024 rank одновременно создают файлы и пишут shards, перегружаются не только data servers, но metadata service и сеть — checkpoint storm.

Для checkpoint $C=2$ TB и устойчивой aggregate bandwidth $B=100$ GB/s физический lower bound $C/B=20$ s. Если storage shared fair-share падает до 25 GB/s, pause становится 80 s. Async checkpointing не устраняет $C/B$: он переносит pause в staging и требует дополнительной DRAM/NVMe ёмкости. Два 2-TB in-flight checkpoint требуют 4 TB staging и backpressure, иначе обучение обгонит writer.

Distributed checkpoint должен:

1. записывать крупные shards без единого rank-0 bottleneck;
2. фиксировать atomic manifest только после durable completion;
3. содержать checksums, global shapes и placement-independent mapping;
4. ограничивать concurrency и разносить metadata operations;
5. проверяться restore на другой допустимой topology.

```text
freeze logical step metadata
for each rank in parallel:
    write_large_shard(temp_generation, checksum, global_tensor_metadata)
barrier_and_validate_all_shards()
single_committer.write_atomic_manifest(generation, durable=true)
garbage_collect_only_generations_older_than_last_known_good()
```

Неполный каталог нельзя считать последним checkpoint. Temp generation + atomic commit marker отделяет завершённую версию от оборванной.

## Наблюдаемость

Для сети нужны per-link bandwidth, congestion, errors и p50/p99 collective duration; для storage — bytes/s, queue depth, metadata ops, dirty/staging bytes и checkpoint age. Средняя bandwidth скрывает один деградировавший link, который синхронно задерживает весь world.

## Источники

- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Network Fabrics, `sec-network-fabrics-rdma`, `sec-network-fabrics-roce`, `sec-network-fabrics-pfc` and `sec-network-fabrics-topology`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd).
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Data Storage, `sec-data-storage-training-data-path` and `sec-data-storage-checkpoint-storms`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/data_storage/data_storage.qmd).

← [[44f Expert и hybrid parallelism]] · Далее: [[44h Fault tolerance и fleet orchestration]]
