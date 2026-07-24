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

## Воспроизводимый цикл

1. Зафиксировать workload, hardware/software versions и baseline distribution.
2. Найти доминирующий interval на end-to-end timeline.
3. Сформулировать одну гипотезу в терминах compute/data/latency.
4. Изменить одну вещь.
5. Повторить benchmark без profiler overhead.
6. Проверить correctness и memory peak.

Framework layer также платит dispatch tax: eager graph удобно отлаживать, но
Python и operator dispatch заметны при мелких операциях. Compilation/fusion
помогают только если trace действительно показывает этот режим.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/frameworks_dispatch_tax_divergence.svg]]

*Источник: Harvard CS249r, Frameworks,
`frameworks_dispatch_tax_divergence.svg`, CC BY-NC-SA 4.0.*

## Источники

- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf)
- [EDLS week 2 profiler practice](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/seminar/practice.ipynb)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch Memory Snapshot](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [Harvard CS249r, Frameworks](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/frameworks/frameworks.qmd)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]] ·
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Сбор, очистка и смеси данных]] →
