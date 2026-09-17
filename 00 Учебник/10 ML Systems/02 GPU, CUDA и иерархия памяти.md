---
title: GPU, CUDA и иерархия памяти
type: textbook-chapter
status: draft
last_verified: 2026-09-06
source_language: mixed
source_unit_id:
  - lecture-05-gpu-anatomy
  - lecture-05-coalescing-failures
  - lecture-06-section-6-hardware
  - lecture-06-figure-step-7-rendering-1
  - lecture-06-accelerator-memory-hierarchy-table
  - lecture-06-section-22-programming-model
  - lecture-06-figure-step-23-rendering-1
  - lecture-06-section-36-interaction-between-programming-model-and-hardware
---

<a id="cs336-systems-gpu"></a>

# GPU, CUDA и иерархия памяти

GPU быстр, когда много одинаковой работы можно запланировать одновременно, а
данные переиспользуются рядом с ALU. Само число ядер почти ничего не говорит о
скорости программы: один kernel загружает вычислительные блоки полезной работой,
а другой проводит большую часть времени в ожидании памяти, простаивает из-за
ветвлений или запускает слишком мало блоков. Чтобы отличать эти случаи, нужно
понимать сразу две вещи — как GPU планирует threads и как данные проходят через
его иерархию памяти.

## От сложения векторов к исполнению на GPU

Пусть нужно сложить два вектора из 1000 чисел FP32. Один поток выполняет одну
операцию `c[i] = a[i] + b[i]`. В CUDA C++ адрес позиции можно задать так:

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// a, b, c уже находятся в памяти GPU; запуск с CPU:
add<<<4, 256>>>(a, b, c, 1000);
```

CPU здесь называется host, а GPU — device. Запуск задаёт сетку (grid) из
четырёх блоков по 256 потоков: всего 1024 логические позиции. Последние 24
потока ничего не записывают благодаря проверке `i < n`. В каждом блоке восемь
групп по 32 потока, называемых warp; номер потока внутри такой группы — lane.
Для первых восьми потоков первого блока индексы равны 0…7, а смещения в каждом
FP32-массиве —0,4,8,12,16,20,24,28 байт. Соседние потоки поэтому читают
соседние адреса, что позволяет объединять обращения к памяти.

Блок назначается одному потоковому мультипроцессору (Streaming Multiprocessor,
SM). Несколько блоков могут одновременно размещаться на одном SM, если хватает
регистров и общей памяти. Планировщик выбирает готовые warp для исполнения;
когда один ждёт память, другой может занять вычислитель. Это скрывает задержку,
но только если другая группа уже размещена на SM и готова работать. Подробная
спецификация: [NVIDIA CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).

**Occupancy** — resident warps / аппаратный максимум. Её ограничивают threads per
block, shared memory и registers. Максимальная occupancy не равна максимальной
скорости: kernel с большим reuse может осознанно расходовать registers/shared
memory и выигрывать при меньшей occupancy.

SIMT исполняет одну инструкцию для активных lanes. При divergent `if/else` warp
проходит пути с масками последовательно. Tail tiles и число blocks, не кратное
числу SM, создают tile/wave quantization.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/cuda-grid-cta.png]]

*CUDA grid состоит из Cooperative Thread Arrays (в CUDA API — thread blocks),
а каждый CTA — из threads. Оригинальный рисунок NVIDIA PTX ISA, встроенный в
[Stanford CS336 Lecture 6, step 23](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py);
изображение перенесено без перерисовки. Эта схема задаёт программную иерархию;
она не утверждает, что один CTA навсегда закреплён за отдельным физическим
вычислительным блоком.*

## Где исполняется работа и где находятся данные

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/gpu-hardware.png]]

*Иерархия GPU в [Stanford CS336 Lecture 6, step 7](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py):
несколько SM разделяют L2 и HBM, а внутри SM находятся scheduler, cores,
register file и shared memory. Рисунок — схема уровней, а не масштабная карта
конкретного кристалла.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/gpu-memory-hierarchy.svg]]

*Оригинальная иллюстрация Harvard CS249r Vol. II, Performance Engineering,
§ “Memory Hierarchy”; [исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/gpu-memory-hierarchy.svg),
CC BY-NC-SA 4.0.*

Схему следует читать снизу вверх. Registers принадлежат отдельному thread,
shared memory — thread block, L2 и HBM разделяются большим числом SM. Чем дальше
от ALU находится уровень, тем больше обычно его ёмкость и тем дороже повторное
получение данных. Поэтому tile — не декоративная оптимизация: он превращает
несколько обращений к HBM в одну загрузку и серию обращений к более близкой
памяти.

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
затрагивают 32 разнесённые области. В иллюстративной модели, где каждый запрос
lane вынуждает получить отдельную 128-байтовую cache line, transferred bytes
могут вырасти до $32\times128=4096$ B при payload 128 B. Это не универсальное
число: современные GPU обслуживают доступ секторами, а итог зависит от
архитектуры, ширины операции, выравнивания, cache policy и состояния кэша.
Поэтому в профиле проверяют число и размер фактических transactions, а не
переносят коэффициент 32 на любое устройство.

## GPU и TPU: одинаковая задача, разные уровни управления

GPU и TPU оба ускоряют плотную линейную алгебру, но давать им одну и ту же
мысленную модель опасно. В GPU программист или компилятор распределяет множество
небольших программ по SM, а latency скрывается готовыми warps. Локальность
выражается через registers, shared memory, L2 и HBM. TPU строится вокруг более
крупных матричных блоков и явно управляемого обмена между accelerator memory и
матричным вычислителем; распределённая программа сильнее зависит от формы
mesh и коллективных операций.

Из этого не следует, что один тип ускорителя «быстрее вообще». Для обоих нужно
ответить на одни и те же вопросы: какой tensor layout получает матричный блок,
сколько байтов пересекает каждый уровень памяти, достаточно ли параллельной
работы и где проходит межчиповый обмен. Паспортные FLOP/s — потолок для
конкретного dtype и набора инструкций; training throughput устанавливает только
измерение полной программы.

Stanford CS336 Lecture 5, pp. 2–19, использует это сравнение как вход в
проектирование kernels. Дальше учебник следует GPU-маршруту, потому что именно
его программная модель понадобится для Triton и FlashAttention; TPU возвращается
в распределённой главе только там, где различие topology меняет коллективы.

## Tiling и Tensor Cores

Наивный GEMM многократно читает $A$ и $B$ из HBM. Tiled kernel загружает их
фрагменты в shared memory, синхронизирует block и переиспользует элементы для
многих FMA. Tile больше — reuse выше, но растут shared memory/register pressure
и риск tail waste. Tensor Cores выполняют matrix multiply-accumulate над
фиксированными фрагментами и форматами; выигрыш требует подходящих dtype,
alignment/layout и размеров, а accumulation precision надо выбирать осознанно.

Повторное использование повышает арифметическую интенсивность
$I=F/Q_{\mathrm{HBM}}$: те же FLOP выполняются при меньшем трафике памяти.
В модели roofline пропускная способность ограничена меньшей из двух величин:
$BW_{\mathrm{HBM}}I$ и пиковыми FLOP/s вычислителя. Точка их равенства
$I^*=P_{\mathrm{peak}}/BW_{\mathrm{HBM}}$ называется точкой излома (ridge).
Слева доминирует доставка байтов, справа — арифметический предел. Следующая
глава выводит эту модель и показывает численные примеры.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/hw_acceleration_roofline_elbow.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. I, Hardware Acceleration,
§ “Roofline Model”, locator `sec-hardware-acceleration-roofline-model-42ff`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/hw_acceleration/images/svg/hw_acceleration_roofline_elbow.svg),
CC BY-NC-SA 4.0; файл не изменён. Точка слева от ridge
показывает режим, где дополнительная арифметика не заменяет доставку байтов.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/operator-fusion.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Performance Engineering,
§ “Operator Fusion”, locator `sec-performance-engineering-operator-fusion`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/operator-fusion.svg),
CC BY-NC-SA 4.0; файл не изменён. Сопоставление отдельных
kernels и fused kernel делает видимыми устранённые промежуточные записи в HBM.*

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

Следующая глава превращает эти вопросы в измерительный контракт, а
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|глава о Triton]]
показывает полный цикл от PyTorch reference до собственного tiled kernel.

## Практика и первоисточники

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/17_acceleration|TinyTorch 17 — Acceleration]] — исполняемое сравнение базовых, векторизованных и fused operations.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/hw_acceleration|Harvard CS249r — Hardware Acceleration]] — SIMD/SIMT, устройство GPU, memory hierarchy и специализированные ускорители.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/lecture.pdf|EDLS Week 1 — lecture]] — оригинальные слайды курса.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/seminar.ipynb|EDLS Week 1 — seminar notebook]] — измерения CPU/GPU, матричного умножения и памяти.
- [EDLS Week 1 lecture, pinned e632aa8](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf) — PDF pp. 6–20
- [Harvard CS249r, Hardware Acceleration, pinned 45ecc8d](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd) —
  §§ “Compute Units and Execution Models”
  (`sec-hardware-acceleration-compute-units-execution-models-f406`),
  “Evolution from SIMD to SIMT architectures”
  (`sec-hardware-acceleration-evolution-simd-simt-architectures-e1fd`),
  “Memory hierarchy”
  (`sec-hardware-acceleration-memory-hierarchy-1839`) и “Hardware Mapping”
  (`sec-hardware-acceleration-hardware-mapping-fundamentals-neural-networks-f9a9`)
- [Stanford CS336 Lecture 5, pinned `8b59b507`, pp. 2–19](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf) — GPU/TPU, execution units и memory hierarchy.
- [Stanford CS336 Lecture 6, pinned `8b59b507`, steps 6–92](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py) — hardware, CUDA programming model и его отображение на GPU.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/01 Модель как часть системы|Модель как часть системы]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Измерение производительности и roofline]] →
