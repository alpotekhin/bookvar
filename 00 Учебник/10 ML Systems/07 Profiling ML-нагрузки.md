---
title: Profiling ML-нагрузки
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# Profiling ML-нагрузки

Benchmarking говорит, что программа медленная; profiling показывает, где
исчезает время и память. Начинают с самого дешёвого уровня и углубляются только
после локализации bottleneck.

## Иерархия инструментов

| Вопрос | Инструмент |
|---|---|
| CPU или GPU простаивает? | system metrics, DCGM, timeline |
| Какая Python-функция занята? | `py-spy`, `cProfile`, Scalene |
| Какие PyTorch operators дороги? | PyTorch Profiler |
| Почему растёт CUDA memory? | PyTorch Memory Snapshot |
| Где gaps, copies и synchronization? | Nsight Systems |
| Почему конкретный kernel медленный? | Nsight Compute |

`nvidia-smi utilization=100%` означает лишь, что GPU исполнял хоть что-то в
sampling interval: dummy wait kernel тоже может дать 100%. Это не MFU.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/diagnostic-flow.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Performance Engineering,
§ “Iron Law Diagnostic Flowchart”, locator
`sec-performance-engineering-iron-law`, commit `45ecc8d…`,
CC BY-NC-SA 4.0; файл не изменён. Дерево фиксирует порядок исключения I/O, CPU
и communication stalls до перехода к kernel-level диагнозу.*

## CPU: py-spy

Sampling profiler периодически снимает stacks и почти не меняет программу:

```bash
py-spy top --pid PID
py-spy record -o profile.svg --pid PID
```

Возможность attach к уже запущенному процессу особенно полезна для stalls.
Sampling может пропустить очень короткие функции; instrumenting profiler точнее
по calls, но сильнее perturb execution.

## PyTorch Profiler

Профилируют ограниченное окно после warmup:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for batch in loader:
        step(batch)
        prof.step()
```

Смотрят CPU self time, CUDA time, shapes, memory и trace. `record_shapes` и
stacks имеют overhead, поэтому profiler-run не используют как финальный
benchmark.

## Memory Snapshot

Snapshot отвечает «кто аллоцировал live/reserved blocks?»:

```python
torch.cuda.memory._record_memory_history()
run_steps()
torch.cuda.memory._dump_snapshot("snapshot.pickle")
```

Это PyTorch-private API на момент курса; имя и viewer зависят от версии.
Snapshot видит allocator PyTorch, но может не видеть всю память CUDA libraries.
Различают allocated live tensors, reserved caching allocator и non-PyTorch
allocations. Снимок до/после помогает найти утечку ссылок, fragmentation и
временный peak.

## Nsight Systems

Systems trace связывает CPU threads, CUDA API, kernels, streams, memcpy и
collectives. Типовые паттерны:

- большие плотные kernels без gaps — оптимизировать kernels/precision;
- множество крошечных kernels и CPU launch gaps — fusion/compile/CUDA Graphs;
- частые D2H и synchronization — убрать `.item()` и зависимости;
- GPU ждёт DataLoader — workers, decode, prefetch, pinned transfer;
- communication не перекрыта — изменить schedule/stream dependencies.

Nsight Compute нужен после Systems, когда выбран конкретный kernel: occupancy,
memory throughput, Tensor Core instructions, stalls и roofline position.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/profiling-hierarchy.svg]]

*Источник: Harvard CS249r, Vol. II preview, Performance Engineering,
`profiling-hierarchy.svg`, commit `45ecc8d…`, CC BY-NC-SA 4.0.*

## End-to-end кейс

Baseline после warmup: step median/p95 420/470 ms. `py-spy` показывает ожидание
`next(loader)`. PyTorch trace раскладывает median: 110 ms loader wait, 18 ms
H2D, 275 ms CUDA kernels, 17 ms launch gaps. Nsight Systems подтверждает:
pageable H2D не перекрывается с compute.

После `pin_memory`, persistent workers и bounded prefetch loader wait почти
целиком перекрыт, critical path — 304 ms. Затем trace показывает сотни мелких
elementwise kernels; `torch.compile` даёт 286 ms. Непрофилированная повторная
серия даёт median/p95 288/315 ms при тех же tokens и loss. Так разделяются
причина, эффект изменения и perturbation самого profiler.

## Воспроизводимый цикл

1. Зафиксировать workload, hardware/software versions и baseline distribution.
2. Найти доминирующий interval на end-to-end timeline.
3. Сформулировать одну гипотезу в терминах compute/data/latency.
4. Изменить одну вещь.
5. Повторить benchmark без profiler overhead.
6. Проверить correctness и memory peak.

Вместе с trace сохраняют git commit, command/config, input shapes и token count,
GPU/driver/CUDA/cuDNN/NCCL, PyTorch/compiler versions, clocks/power mode,
warmup/active schedule, rank/host и before-after benchmark. PyTorch export
делают через `tensorboard_trace_handler`; Nsight сохраняют в `.nsys-rep`,
memory snapshot — в versioned `.pickle`. Trace может содержать stack paths,
shapes и NVTX labels с пользовательскими данными — перед публикацией его
очищают. Сравнение артефактов разных версий без manifest ненадёжно: fusion,
operator names и private snapshot API меняются.

Framework layer также платит dispatch tax: eager graph удобно отлаживать, но
Python и operator dispatch заметны при мелких операциях. Compilation/fusion
помогают только если trace действительно показывает этот режим.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/optimization-decision-tree.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Performance Engineering,
§ “Optimization decision tree”, locator
`sec-performance-engineering-optimization-decision-tree`, commit `45ecc8d…`,
CC BY-NC-SA 4.0; файл не изменён. Она превращает установленный bottleneck в
выбор класса вмешательства и удерживает profiling перед optimization.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/frameworks_dispatch_tax_divergence.svg]]

*Источник: Harvard CS249r, Frameworks,
`frameworks_dispatch_tax_divergence.svg`, CC BY-NC-SA 4.0.*

## Источники

- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf) — “Profiling: what and why”, “How to profile Python/GPU/PyTorch code?”, “PyTorch Profiler + trace viewer”, “Nsight Systems/Nsight Compute”, “Profiling: typical patterns”; title locators used because incremental slides repeat in the PDF.
- [EDLS week 2 profiler practice](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/seminar/practice.ipynb)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch Memory Snapshot](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [Harvard CS249r, Frameworks](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/frameworks/frameworks.qmd)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]] ·
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Сбор, очистка и смеси данных]] →
