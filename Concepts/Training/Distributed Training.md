---
title: "Distributed Training"
aliases: [Distributed Training, Распределённое обучение, DDP, FSDP, ZeRO]
type: concept
category: Training
papers: []
courses: []
sources:
  - "[Rajbhandari et al. — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2019)](https://arxiv.org/abs/1910.02054)"
  - "[PyTorch FSDP docs](https://pytorch.org/docs/stable/fsdp.html)"
  - "[Megatron-LM (NVIDIA)](https://github.com/NVIDIA/Megatron-LM)"
  - "[DeepSpeed (Microsoft)](https://www.deepspeed.ai/)"
---

# Distributed Training — распределённое обучение

## Зачем это нужно

Одна GPU не вмещает большие модели и не даёт разумного времени обучения. LLaMA-70B требует ~140 GB только на веса в FP16 — это не помещается даже на H100 (80 GB). GPT-4 обучался на десятках тысяч GPU. Distributed training — необходимое условие для современных LLM.

Ключевые проблемы, которые решает:
1. **Memory** — модель не помещается на одну GPU.
2. **Speed** — батч можно параллелить между GPU.
3. **Scale** — обучение за разумное время (дни, а не годы).

Классифицируется по **тому, что параллелится**: данные, веса модели, слои, эксперты.

## Data Parallelism (DP)

### Основная идея

Каждая GPU имеет **полную копию модели**, но обрабатывает **разные подбатчи данных**. После forward/backward градиенты усредняются между GPU.

```
GPU 0: full model | batch [0:N/4]   → grads_0
GPU 1: full model | batch [N/4:N/2] → grads_1   } all-reduce → avg grads
GPU 2: full model | batch [N/2:3N/4]→ grads_2
GPU 3: full model | batch [3N/4:N]  → grads_3
```

### DP (PyTorch legacy)

`torch.nn.DataParallel` — single-process, multi-threaded. Python GIL убивает производительность. **Не используйте.**

### DDP (Distributed Data Parallel)

`torch.nn.parallel.DistributedDataParallel` — multi-process, один процесс на GPU. Градиенты синхронизируются через **all-reduce** (NCCL). Overlap вычислений и коммуникаций: пока backward идёт по нижним слоям, градиенты верхних уже отправляются.

**Стандарт для малых-средних моделей**, которые помещаются на одну GPU. Почти линейное ускорение до ~100 GPU.

**Ограничение:** каждая GPU хранит полную копию весов, градиентов и optimizer states. Для 7B модели в BF16 + Adam это ~140 GB на GPU — не помещается.

## Model Parallelism

Когда модель не помещается на одну GPU, её **разрезают между GPU**.

### Tensor Parallelism (intra-layer)

Разрезаются **отдельные слои** между GPU. Например, matmul $Y = XW$ с $W \in \mathbb{R}^{d \times 4d}$:

- GPU 0 хранит $W[:, :d]$, вычисляет $Y[:, :d]$
- GPU 1 хранит $W[:, d:2d]$, вычисляет $Y[:, d:2d]$
- ... и т. д., результат конкатенируется через all-gather.

В attention разрезаются головы: каждая GPU считает свою группу heads.

**Плюсы:** работает внутри одного слоя, ровная нагрузка.
**Минусы:** требует частых коммуникаций (all-reduce/all-gather на каждый слой) → хорошо работает только **внутри узла** (NVLink ~600 GB/s). Через Ethernet/InfiniBand это медленно.

Реализации: Megatron-LM, DeepSpeed Tensor Parallelism.

### Pipeline Parallelism (inter-layer)

Разные **слои** на разных GPU. Модель из 64 слоёв на 4 GPU: слои 0-15 → GPU0, 16-31 → GPU1, и т. д.

Forward течёт GPU0 → GPU1 → ... → GPUN, backward в обратном порядке. Проблема: **pipeline bubble** — пока один слой работает, остальные ждут.

Решение — **microbatching** (GPipe, PipeDream): батч делится на микробатчи, которые идут по конвейеру.

**Плюсы:** низкие требования к коммуникации (точки разреза между слоями).
**Минусы:** bubble снижает утилизацию, сложность реализации.

Обычно Pipeline Parallelism **комбинируют** с TP и DP (3D parallelism в Megatron-LM).

### Expert Parallelism (для MoE)

В [[MoE]] моделях (Mixtral, DeepSeek-V3, Qwen3-MoE) разные **эксперты** живут на разных GPU. Router отправляет каждый токен только на те GPU, где находятся выбранные эксперты — это **all-to-all** коммуникация.

Масштабируется до сотен-тысяч GPU: каждая GPU хранит лишь подмножество экспертов.

## FSDP / ZeRO

Революция в распределённом обучении: вместо полной копии модели на каждой GPU, **шардируем** параметры, градиенты и optimizer states.

### ZeRO стадии (DeepSpeed)

| Стадия | Что шардируется | Память на GPU |
|--------|-----------------|---------------|
| ZeRO-1 | Optimizer states | ~4x экономия |
| ZeRO-2 | Optimizer states + Gradients | ~8x экономия |
| ZeRO-3 | Optimizer states + Gradients + **Parameters** | Nx экономия (N — число GPU) |

**ZeRO-3** = каждая GPU хранит лишь 1/N весов. Перед forward нужного слоя делается **all-gather** — временно собираются полные веса → вычисление → освобождение.

**Trade-off:** больше коммуникации, но позволяет обучать огромные модели на доступном железе.

### FSDP (Fully Sharded Data Parallel)

PyTorch-native реализация ZeRO-3 (`torch.distributed.fsdp`). Стал стандартом в open-source экосистеме вытеснив DeepSpeed в PyTorch-only проектах.

Ключевые параметры:
- **sharding strategy:** `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `NO_SHARD` (обычный DDP).
- **auto_wrap_policy:** как группировать слои в шарды (обычно по transformer block).
- **CPU offload:** выгружать optimizer states на CPU для экономии VRAM.

### ZeRO-Infinity

Расширение: offload на CPU RAM и NVMe. Позволяет обучать модели больше суммарной VRAM кластера (но медленнее).

## 3D Parallelism

Для самых больших моделей комбинируются **все три**:

- **DP** между группами узлов (через Ethernet/InfiniBand)
- **PP** между узлами внутри группы
- **TP** внутри узла (через NVLink)

Пример GPT-4 класса: 4096 GPU = 8 TP × 16 PP × 32 DP.

Плюс [[MoE]] — Expert Parallelism как четвёртое измерение. См. [[MoE]] для деталей expert parallelism в DeepSeek-V3/Qwen-Next.

## Gradient Accumulation

Псевдо-параллелизм: если не хватает памяти на большой батч, делаем $K$ forward/backward без `optimizer.step()`, накапливая градиенты. Эффективный батч = `micro_batch × K × num_gpus`.

Стандартная практика в pre-training LLM — effective batch size 4M-8M токенов при micro batch всего 1-4 на GPU.

## Коммуникационные примитивы (NCCL)

| Примитив | Описание | Где используется |
|----------|----------|------------------|
| `all-reduce` | Сумма от всех GPU, результат у всех | DDP gradient sync |
| `all-gather` | Конкатенация от всех GPU у всех | FSDP params gather, TP output |
| `reduce-scatter` | Сумма + шардирование результата | FSDP backward |
| `all-to-all` | Каждый отправляет каждому разные данные | MoE expert routing |
| `broadcast` | Один → всем | Init весов |

Скорость коммуникации критична: NVLink (300-900 GB/s) внутри узла, InfiniBand (200-400 Gb/s) между узлами, Ethernet (25-100 Gb/s) — топология определяет, что куда класть.

## Фреймворки

- **PyTorch DDP + FSDP** — стандарт для open-source.
- **DeepSpeed** (Microsoft) — ZeRO, pipeline, CPU offload.
- **Megatron-LM** (NVIDIA) — лучшая реализация 3D parallelism для NVIDIA GPU.
- **Megatron-DeepSpeed** — комбинация.
- **Colossal-AI** — альтернатива с упором на доступность.
- **JAX + pjit/shmap** — альтернативная экосистема (Google, Anthropic).

## Related concepts

- [[MoE]] — expert parallelism как отдельное измерение параллелизма
- [[Mixed Precision Training]] — обычно используется вместе с распределёнными режимами
- [[Gradient Clipping]] — работает глобально через all-reduce норм
- [[Scaling Laws]] — distributed training сделал реальным обучение 100B+ моделей
- [[Pre-training]] — основной use-case distributed training
