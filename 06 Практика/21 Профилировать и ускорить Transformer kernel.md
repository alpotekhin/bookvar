---
title: Профилировать и ускорить Transformer kernel
type: practice
status: canonical
last_updated: 2026-09-06
source_unit_id:
  - lecture-06-benchmark-profile-experiment
  - lecture-07-distributed-benchmark-experiment
  - assignment-02-task-benchmarking-script
  - assignment-02-benchmark-synchronization
  - assignment-02-deliverable-benchmarking-script-001
  - assignment-02-deliverable-benchmarking-script-002
  - assignment-02-deliverable-benchmarking-script-003
  - assignment-02-nsight-systems-trace
  - assignment-02-task-nsys-profile
  - assignment-02-deliverable-nsys-profile-004
  - assignment-02-deliverable-nsys-profile-005
  - assignment-02-deliverable-nsys-profile-006
  - assignment-02-deliverable-nsys-profile-007
  - assignment-02-deliverable-nsys-profile-008
  - assignment-02-task-mixed-precision-accumulation
  - assignment-02-deliverable-mixed-precision-accumulation-009
  - assignment-02-task-benchmarking-mixed-precision
  - assignment-02-deliverable-benchmarking-mixed-precision-010
  - assignment-02-deliverable-benchmarking-mixed-precision-011
  - assignment-02-deliverable-benchmarking-mixed-precision-012
  - assignment-02-task-memory-profiling
  - assignment-02-deliverable-memory-profiling-013
  - assignment-02-deliverable-memory-profiling-014
  - assignment-02-deliverable-memory-profiling-015
  - assignment-02-deliverable-memory-profiling-016
  - assignment-02-deliverable-memory-profiling-017
  - assignment-02-deliverable-memory-profiling-018
  - assignment-02-task-gradient-checkpointing
  - assignment-02-deliverable-gradient-checkpointing-019
  - assignment-02-deliverable-gradient-checkpointing-020
  - assignment-02-task-pytorch-attention
  - assignment-02-deliverable-pytorch-attention-021
  - assignment-02-task-torch-compile
  - assignment-02-deliverable-torch-compile-022
  - assignment-02-deliverable-torch-compile-023
  - assignment-02-flashattention-online-softmax
  - assignment-02-task-flash-forward
  - assignment-02-deliverable-flash-forward-024
  - assignment-02-deliverable-flash-forward-025
  - assignment-02-deliverable-flash-forward-026
  - assignment-02-flashattention-tests
  - assignment-02-task-flash-backward
  - assignment-02-deliverable-flash-backward-027
  - assignment-02-task-flash-benchmarking
  - assignment-02-deliverable-flash-benchmarking-028
  - assignment-02-distributed-benchmark-controls
  - assignment-02-distributed-rank-topology
  - assignment-02-task-distributed-communication-single-node
  - assignment-02-deliverable-distributed-communication-single-node-029
  - assignment-02-task-naive-ddp
  - assignment-02-deliverable-naive-ddp-030
  - assignment-02-task-naive-ddp-benchmarking
  - assignment-02-deliverable-naive-ddp-benchmarking-031
  - assignment-02-task-minimal-ddp-flat-benchmarking
  - assignment-02-deliverable-minimal-ddp-flat-benchmarking-032
  - assignment-02-task-ddp-overlap-individual-parameters
  - assignment-02-deliverable-ddp-overlap-individual-parameters-033
  - assignment-02-task-ddp-overlap-individual-parameters-benchmarking
  - assignment-02-deliverable-ddp-overlap-individual-parameters-benchmarking-034
  - assignment-02-deliverable-ddp-overlap-individual-parameters-benchmarking-035
  - assignment-02-task-optimizer-state-sharding
  - assignment-02-deliverable-optimizer-state-sharding-036
  - assignment-02-task-optimizer-state-sharding-accounting
  - assignment-02-deliverable-optimizer-state-sharding-accounting-037
  - assignment-02-deliverable-optimizer-state-sharding-accounting-038
  - assignment-02-deliverable-optimizer-state-sharding-accounting-039
  - assignment-02-task-fsdp
  - assignment-02-deliverable-fsdp-040
  - assignment-02-task-fsdp-accounting
  - assignment-02-deliverable-fsdp-accounting-041
  - assignment-02-deliverable-fsdp-accounting-042
  - assignment-02-task-alternate-ring-all-reduce
  - assignment-02-deliverable-alternate-ring-all-reduce-043
  - assignment-02-task-data-parallel-calcs
  - assignment-02-deliverable-data-parallel-calcs-044
  - assignment-02-deliverable-data-parallel-calcs-045
  - assignment-02-deliverable-data-parallel-calcs-046
  - assignment-02-task-fsdp-calcs
  - assignment-02-deliverable-fsdp-calcs-047
  - assignment-02-deliverable-fsdp-calcs-048
  - assignment-02-deliverable-fsdp-calcs-049
  - assignment-02-task-tp-calcs
  - assignment-02-deliverable-tp-calcs-050
  - assignment-02-deliverable-tp-calcs-051
  - assignment-02-deliverable-tp-calcs-052
  - assignment-02-deliverable-tp-calcs-053
  - assignment-02-task-fsdp-tp-calcs
  - assignment-02-deliverable-fsdp-tp-calcs-054
  - assignment-02-deliverable-fsdp-tp-calcs-055
  - assignment-02-deliverable-fsdp-tp-calcs-056
  - assignment-02-deliverable-fsdp-tp-calcs-057
  - assignment-02-leaderboard-timing
  - assignment-02-task-leaderboard
  - assignment-02-deliverable-leaderboard-058
  - assignment-02-test-tests-test-sharded-optimizer-py-19-test-sharded-optimizer
  - assignment-02-test-tests-test-ddp-py-28-test-distributeddataparallel
  - assignment-02-test-tests-test-attention-py-62-test-flash-forward-pass-pytorch
  - assignment-02-test-tests-test-attention-py-71-test-flash-forward-pass-triton
  - assignment-02-test-tests-test-attention-py-81-test-flash-backward-pytorch
  - assignment-02-test-tests-test-attention-py-97-test-flash-backward-triton
  - assignment-02-test-tests-test-fsdp-py-107-test-fsdp-correctness
  - assignment-02-test-tests-test-fsdp-py-192-test-fsdp-gradient-sync
primary_sources:
  - https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736
  - https://github.com/stanford-cs336/lectures/tree/8b59b50730766695c2ffedd1a79c50cd09b9eb91
contract: "[[02 Areas/ML & DL/06 Практика/Contracts/stanford-cs336-a2.yml]]"
---

<a id="cs336-a2-capstone"></a>

# Профилировать и ускорить Transformer kernel

Эта работа связывает отдельные оптимизации в один воспроизводимый эксперимент:
сначала доказать корректность Transformer step, затем найти bottleneck, изменить
один механизм и подтвердить результат тестом, trace и измерением. За основу взят
[Stanford CS336 Spring 2026 Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736).
Готового solution code здесь нет; точные adapters, тесты и режимы закреплены в
[[02 Areas/ML & DL/06 Практика/Contracts/stanford-cs336-a2.yml|контракте Assignment 2]].

## Что должно быть известно заранее

- [[06 Измерить CUDA без самообмана]] — warmup, synchronization и raw samples;
- [[06a Проверить mixed precision и loss scaling]] — роли dtype;
- [[07 Построить roofline и найти bottleneck]] — compute- и memory-bound regimes;
- [[09 Реализовать ring all-reduce]] — collectives и alpha–beta model;
- [[10 Измерить checkpointing и offload]] — saved tensors и peak memory;
- [[11 Разрезать Transformer по TP и SP]] — layouts и communication boundaries;
- [[12 Собрать и проверить FSDP]] — parameter lifecycle;
- [[13 Оптимизировать один Transformer step]] — end-to-end acceptance.

Теоретический маршрут: [[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|GPU kernels и Triton]] →
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]] →
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision|distributed training]].

## Нулевой gate: зафиксировать эксперимент

Выберите один reference Transformer block и неизменяемую матрицу shapes:
`batch`, `sequence`, `d_model`, `n_heads`, `head_dim`, `d_ff`, layers. Запишите
device/count/topology, CPU, driver, CUDA, PyTorch, Triton, backend, commit,
рабочее состояние git, seed и dtype policy. Для каждой серии заранее задайте
warmup, число итераций, места GPU synchronization и statistic: median вместе с
raw samples; p95 — если исследуется tail latency.

До первого таймера должны пройти reference forward/backward checks. Для
приближённых dtype запишите `atol` и `rtol`; для distributed run сравните
результат с unwrapped model на том же global batch.

## Gate 1. Baseline и profiler

1. Измерьте eager forward, backward и полный training step отдельно.
2. Снимите PyTorch profiler trace и memory snapshot одного репрезентативного
   шага после warmup.
3. Добавьте NVTX ranges для attention, MLP, normalization, backward и optimizer;
   снимите Nsight Systems trace.
4. Свяжите каждую гипотезу с наблюдением: kernel launches, HBM traffic,
   collective на critical path либо transient memory peak.

Не делайте вывод по числу kernels: fusion должна снизить end-to-end latency на
той же shape matrix. Profiler run хранится отдельно от timing run, поскольку
инструментирование само добавляет overhead.

## Gate 2. Precision и memory ledger

Повторите baseline для FP32 и выбранной mixed-precision policy. Проследите dtype
inputs, weights, products, accumulators, loss, gradients и optimizer state.
Зафиксируйте peak allocated/reserved HBM и таблицу крупнейших saved tensors из
`saved_tensors_hooks`. NaN/Inf либо неизвестный accumulator dtype останавливают
работу до benchmark.

## Gate 3. Первый Triton kernel: fused RMSNorm

Сначала напишите PyTorch oracle и тесты для contiguous/non-contiguous inputs,
нескольких widths, tail block, очень малой variance и двух dtypes. Затем
реализуйте fused normalization-style kernel: одна программа обрабатывает строку,
mask закрывает tail, reduction накапливается в оговорённом dtype. Сравните
forward и gradients с oracle.

После correctness измерьте eager, builtin/compiled и Triton варианты на matrix
shapes. Trace должен подтвердить ожидаемую fusion, а roofline — объяснить, была
ли устранена memory traffic либо только launch overhead.

## Gate 4. FlashAttention forward и backward

1. Реализуйте reference `torch.autograd.Function` без Triton и сохраните ровно
   тот state, который требует официальный adapter contract.
2. Проверьте output и log-sum-exp против dense stable attention.
3. Реализуйте tiled Triton forward с causal и non-causal tail tiles.
4. Выведите backward из сохранённых `O` и `L`: сначала `D=rowsum(O*dO)`, затем
   tile-local `dV`, `dS`, `dQ`, `dK` с reconstruction probabilities.
5. Запустите все параметризованные official forward/backward tests и лишь затем
   сравните time/memory с PyTorch attention и `torch.compile`.

Regression matrix должна менять sequence length, head dimension, causal flag и
размер, не кратный block. Один квадратный случай не проверяет mask discipline.

## Gate 5. Collective и DDP progression

На двух и более process ranks измерьте payload sweep для all-reduce и
reduce-scatter. Запишите launcher, backend, rank mapping, intra/inter-node
контекст, barriers и формулу effective bandwidth.

Затем сохраните одну семантику DDP в трёх вариантах:

1. синхронный all-reduce каждого gradient после backward;
2. flat/bucketed gradient buffer;
3. asynchronous per-parameter или per-bucket overlap через autograd hooks.

Для каждого варианта официальный CPU/Gloo test сравнивает обе
параметризованные model classes с unwrapped model. GPU trace должен показать,
где начинается communication и какая его часть остаётся на critical path.

## Gate 6. Optimizer sharding и FSDP

Сначала разделите optimizer state при replicated weights/gradients и составьте
bytes-per-parameter ledger. Затем реализуйте custom FSDP lifecycle:
parameter shard → pre-forward all-gather → compute → reshard → pre-backward
all-gather → reduce-scatter gradient → local optimizer update.

Официальные `fp32` и `fp16` test variants должны проверить reconstructed full
parameters и gradient shapes/dtypes. После этого измерьте memory и trace для
разных unit boundaries/prefetch policies. Отдельно покажите transient peak,
когда соседние all-gather overlap.

## Gate 7. Checkpointing и полный step

Сравните baseline, uniform segment checkpointing и одну selective/nested policy
при одинаковом memory budget. Запишите recompute time, saved tensors,
allocated/reserved peak и RNG equivalence. Если добавляется offload, отдельно
запишите D2H/H2D bytes и exposed transfer time.

В финале выберите одну измеренную bottleneck и внесите одну оптимизацию в полный
forward-and-backward step. Повторите всю regression matrix. Leaderboard timing —
не обязательный результат: pass condition задают корректность и полнота
evidence, а не место в таблице.

## Evidence bundle

Сдаётся каталог с:

- `environment.json`: commits, git state, hardware/topology и версии software;
- `correctness.jsonl`: command, parametrized case, reference, tolerances, status;
- `benchmarks.jsonl`: shapes, dtype, warmup, sync policy, raw samples и summary;
- `memory/`: snapshots и таблица saved tensors с интерпретацией пика;
- `profiles/`: PyTorch/Nsight traces, NVTX labels и одно проверяемое наблюдение
  на trace;
- `distributed.jsonl`: launcher, ranks, backend, payload, collectives и topology;
- `change-log.md`: hypothesis → одна правка → before/after → regression result.

Итог считается воспроизводимым, если другой человек может проверить source и
student commits, выполнить указанную команду и получить тот же qualitative
вывод. Скриншот без raw samples или trace без сформулированного наблюдения этим
условиям не соответствует.

## Первоисточники

- Stanford CS336 Spring 2026, [Assignment 2 handout and tests](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736), pinned commit `ca8bc81a…`, MIT.
- Stanford CS336 Spring 2026, [Lecture 5](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf), GPU performance and FlashAttention.
- Stanford CS336 Spring 2026, [Lecture 6](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py), benchmarking, profiling and Triton.
- Stanford CS336 Spring 2026, [Lectures 7–8](https://github.com/stanford-cs336/lectures/tree/8b59b50730766695c2ffedd1a79c50cd09b9eb91), distributed and parallel training.
