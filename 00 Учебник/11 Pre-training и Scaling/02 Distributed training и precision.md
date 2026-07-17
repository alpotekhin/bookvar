---
title: Distributed training и precision
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Distributed training и precision

> [!abstract] Идея главы
> Одна GPU не вмещает большие weights, activations, gradients и optimizer state.
> Distributed training делит эти объекты между устройствами, но каждое деление
> превращается в communication.

Модель масштабируется по нескольким измерениям:

| Parallelism | Что делим | Главная цена |
|---|---|---|
| Data | batches | gradient all-reduce |
| Tensor | матрицы слоя | частые collectives |
| Pipeline | layers | bubbles |
| Expert | MoE experts | all-to-all routing |
| Sequence/context | sequence dimension | communication attention |

Data parallel replicas обрабатывают разные batches и синхронизируют gradients.
Tensor parallel делит большие matmul внутри layer. Pipeline parallel размещает
последовательные группы layers на разных devices. Expert parallel распределяет
MoE experts и требует all-to-all для tokens.

ZeRO/FSDP sharding делит parameters, gradients и optimizer states между data
parallel workers, чтобы не хранить полную копию всего на каждом accelerator.

BF16 сохраняет широкий exponent и обычно стабильнее FP16. FP8 уменьшает memory
traffic, но требует scaling, высокоточной аккумуляции и тщательной валидации.
DeepSeek-V3 показал крупномасштабный FP8 training; это engineering achievement,
а не изменение функции attention.

## Pipeline bubbles

Первый stage ждёт, пока поздние stages закончат microbatch. Interleaving и
большее число microbatches уменьшают idle time, но увеличивают scheduling
complexity и activation memory.

## Главное правило

Parallelism выбирают по bottleneck конкретной модели и сети. Конфигурация с
наименьшей памятью не обязана иметь лучший throughput: слишком частые
collectives могут оставить accelerators без работы.

- [[02 Areas/ML & DL/Concepts/Training/Distributed Training]]
- [[02 Areas/ML & DL/Concepts/Training/Mixed Precision Training]]
- [[02 Areas/ML & DL/Papers/DeepSeek-V3 Technical Report]]
