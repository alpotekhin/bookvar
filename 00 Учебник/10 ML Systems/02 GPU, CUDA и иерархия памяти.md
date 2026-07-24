---
title: GPU, CUDA и иерархия памяти
type: textbook-chapter
status: draft
last_verified: 2026-07-24
source_language: mixed
---

# GPU, CUDA и иерархия памяти

GPU быстр, когда много одинаковой работы можно запланировать одновременно, а
данные переиспользуются рядом с ALU.

## От grid до SM: исполнение и планирование

Host ставит kernel launch в очередь device. Grid делится на thread blocks; block
назначается одному Streaming Multiprocessor (SM), где threads группируются в
warps по 32 lanes. Warp scheduler выбирает готовый warp каждый цикл. Если warp
ждёт память, другой скрывает latency — при условии, что он resident и готов.

**Occupancy** — resident warps / аппаратный максимум. Её ограничивают threads per
block, shared memory и registers. Максимальная occupancy не равна максимальной
скорости: kernel с большим reuse может осознанно расходовать registers/shared
memory и выигрывать при меньшей occupancy.

SIMT исполняет одну инструкцию для активных lanes. При divergent `if/else` warp
проходит пути с масками последовательно. Tail tiles и число blocks, не кратное
числу SM, создают tile/wave quantization.

## Оригинальная схема: где живут work и bytes

```text
HOST RAM ── PCIe/NVLink ──► DEVICE HBM ─► L2 (общий)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
                  SM 0                          SM N
          warp schedulers                warp schedulers
          registers/thread               registers/thread
          shared memory/block            shared memory/block
          L1/cache                       L1/cache
                    └── Tensor/Core pipelines ──┘
```

*Оригинальная учебная схема Bookvar; синтез EDLS Week 1, PDF pp. 6–10, и
Harvard CS249r Hardware Acceleration § “Evolution from SIMD to SIMT
architectures”, locator
`sec-hardware-acceleration-evolution-simd-simt-architectures-e1fd`, и
§ “Memory hierarchy”, locator `sec-hardware-acceleration-memory-hierarchy-1839`.
Не является копией исходной фигуры.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/gpu-memory-hierarchy.svg]]

*Оригинальная иллюстрация Harvard CS249r Vol. II, Performance Engineering,
§ “Memory Hierarchy”, `gpu-memory-hierarchy.svg`, CC BY-NC-SA 4.0.*

## Memory access: coalescing, banks, spills

- **Coalescing**: соседние lanes должны обращаться к соседним адресам, чтобы
  warp обслуживался минимальным числом memory transactions. Strided access
  превращает один полезный запрос в несколько cache-line/sector transactions.
- **Shared-memory bank conflict**: разные адреса одного bank сериализуются
  (broadcast одного адреса — особый быстрый случай). Padding tile, например
  `[32][33]` вместо `[32][32]`, часто убирает конфликт при transpose.
- **Register spill**: если компилятору не хватает registers, значения уходят в
  local memory, которая физически находится в device memory и кэшируется.
  “Local” означает scope thread, а не близость.

### Численный пример транзакций

Warp читает 32 FP32 = 128 B. При выровненном contiguous access полезные 128 B
укладываются в минимальное число секторов. При stride 32 elements lanes
затрагивают 32 разнесённые области: payload тот же, но transferred bytes могут
вырасти примерно до $32\times128=4096$ B, то есть полезность линии около 3%.
Точное число зависит от архитектуры и cache state; важно считать transactions,
а не только payload.

## Tiling и Tensor Cores

Наивный GEMM многократно читает $A$ и $B$ из HBM. Tiled kernel загружает их
фрагменты в shared memory, синхронизирует block и переиспользует элементы для
многих FMA. Tile больше — reuse выше, но растут shared memory/register pressure
и риск tail waste. Tensor Cores выполняют matrix multiply-accumulate над
фиксированными фрагментами и форматами; выигрыш требует подходящих dtype,
alignment/layout и размеров, а accumulation precision надо выбирать осознанно.

## PCIe, NVLink и overlap

EDLS Week 1 приводит PCIe 4.0 x16 ≈ 32 GB/s peak (PDF pp. 11–12): это контекст,
не универсальная характеристика. NVLink даёт более быстрые device links, но
топология и поколение определяют реальный путь. Page-locked host memory
позволяет DMA и async H2D; pinning всей RAM вредно.

CUDA calls обычно enqueue asynchronous work. **Stream** сохраняет порядок внутри
себя; разные streams могут перекрываться при отсутствии зависимостей и наличии
resources/copy engines. **Event** отмечает точку device timeline: им измеряют
GPU interval или задают `stream.wait_event`, не блокируя весь device. Default
stream semantics и скрытые `.item()`, D2H, printing могут сериализовать путь.

### Pipeline example

Для трёх batches с copy 3 ms и compute 7 ms последовательность занимает 30 ms.
Double buffering в двух streams даёт в идеале $3+3\cdot7=24$ ms: copy batch
$i+1$ перекрывается с compute batch $i$. Проверять overlap следует timeline:
асинхронность API сама его не гарантирует.

## Практический разбор kernel

1. Достаточно ли blocks/warps для всех SM?
2. Что ограничивает occupancy: registers, shared memory или block size?
3. Coalesced ли global loads, нет ли bank conflicts и spills?
4. Сколько HBM transactions устраняет tile?
5. Используется ли Tensor Core path?
6. Видны ли overlap и зависимости streams/events в trace?

## Источники

- [EDLS Week 1 lecture, pinned e632aa8](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf) — PDF pp. 6–20
- [Harvard CS249r, Hardware Acceleration, pinned 45ecc8d](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd) — §§ “GPU Architecture”, “Memory Hierarchy”, “Computation Scheduling”

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/01 Модель как часть системы|Модель как часть системы]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Измерение производительности и roofline]] →
