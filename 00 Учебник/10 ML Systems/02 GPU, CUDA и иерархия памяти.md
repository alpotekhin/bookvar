---
title: GPU, CUDA и иерархия памяти
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# GPU, CUDA и иерархия памяти

GPU ускоряет DL не магически: он исполняет много однотипной арифметики
параллельно. Реальная скорость появляется, когда работа регулярна, данные
переиспользуются близко к вычислителям, а host не заставляет device ждать.

## Host, device, kernel

> In CUDA, we launch kernels from the host that are executed in parallel on the
> device. Kernels are executed by threads grouped in thread blocks of limited
> size. Multiple thread blocks are arranged in grids.

CPU-host формирует аргументы и помещает kernel launch в очередь. GPU-device
распределяет blocks по Streaming Multiprocessors (SM). Block живёт на одном SM,
может синхронизировать свои threads и совместно использовать shared memory.
Grid может быть 1D/2D/3D — это способ адресовать задачу, а не новая физика.

## SIMT, warp и divergence

> **SIMT (Single Instruction, Multiple Thread).** On a physical level, threads
> are executed in groups of 32 called warps. A warp executes one instruction at
> a time: in case of branching, all paths need to be taken.

Если половина warp идёт по `if`, а половина по `else`, hardware маскирует
неактивные lanes и последовательно проходит обе ветки. Результат корректен, но
полезная ширина исполнения падает. Нерегулярность также создают:

- **tile quantization** — размеры матрицы не кратны tile;
- **wave quantization** — число tiles плохо делится на число SM;
- нехватка registers/shared memory, уменьшающая occupancy.

## Иерархия памяти

Путь становится медленнее и вместительнее по мере удаления:

```text
registers → shared memory / L1 → L2 → HBM (device)
                                  ↕ PCIe / NVLink
                              RAM (host) → SSD/network
```

Registers принадлежат thread, shared memory — block, caches — GPU, HBM хранит
тензоры. Harvard формулирует главный вывод так: специализация ускорителей
увеличила compute быстрее, чем bandwidth, поэтому data movement всё чаще
ограничивает производительность.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/hw_acceleration_energy_ladder.svg]]

*Источник: Harvard CS249r, Hardware Acceleration,
`hw_acceleration_energy_ladder.svg`, CC BY-NC-SA 4.0; величины
technology-specific.*

## PCIe и pinned memory

> GPU has a separate memory unit (called device memory). Need to copy from host
> memory and back (PCIe 4.0 x16 — 32 GB/s peak). Memory transfer is often a
> bottleneck. Pinned (page-locked) memory access is much faster.

32 GB/s — контекст слайда EDLS, не характеристика любой системы. HBM обычно на
порядки быстрее PCIe, поэтому копирование маленькими порциями разрушает
throughput. Page-locked host buffers позволяют DMA и асинхронное копирование,
но pinning расходует дефицитную RAM — используют разумный pool, а не pin всего
датасета.

## Асинхронный запуск

> By default, CUDA kernel calls and device transfers are asynchronous. You can
> send several kernels and wait for results.

Python может закончить enqueue раньше GPU. `tensor.item()`, печать CUDA tensor,
копирование device→host и явный `synchronize()` создают границу ожидания.
Streams позволяют перекрывать независимые kernels и transfer; CUDA Graphs
уменьшают launch overhead для повторяемой последовательности.

### Пример: почему batching важен

100 копирований по 1 MB несут тот же payload, что одно 100 MB, но оплачивают
latency запуска 100 раз. Если latency операции $\ell=10\,\mu s$, только overhead
равен 1 ms против 0,01 ms. Это один из источников $L_{\text{lat}}$ Iron Law.

## Практическая проверка

1. Тензоры действительно на нужном device?
2. Shapes кратны эффективным GEMM tiles?
3. В hot path нет `.item()` и скрытых D2H?
4. H2D идёт из pinned memory с `non_blocking=True`?
5. Trace показывает overlap, а не только надежду на него?

## Источники

- [EDLS week 1 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf)
- [Harvard CS249r, Hardware Acceleration](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/01 Модель как часть системы|Модель как часть системы]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Измерение производительности и roofline]] →
