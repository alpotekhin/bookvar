---
title: Pipeline parallelism
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44d. Pipeline parallelism

Pipeline parallelism помещает последовательные группы слоёв на $p$ стадиях. Микропакет идёт forward через стадии и возвращается backward; между соседями передаются активации и их gradients, а не параметры.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/pipeline-parallelism.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-pipeline-parallelism`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

## GPipe и bubble

GPipe выполняет forward всех $m$ микропакетов, затем backward. При равных стадиях один односторонний проход занимает $m+p-1$ slots, из которых полезны $m$:

$$\mathrm{bubble}_{\mathrm{one\ way}}=\frac{p-1}{m+p-1}.$$

Для $p=4,m=8$ это $3/11=27.3\%$ на фазу; для $m=32$ — $8.6\%$. Но GPipe хранит активации многих микропакетов до backward.

```text
slot:  1 2 3 4 5 6
S0 F0 F1 F2 F3 .  .
S1 .  F0 F1 F2 F3 .
S2 .  .  F0 F1 F2 F3
S3 .  .  .  F0 F1 F2
```

1F1B после warm-up чередует forward нового микропакета и backward старого. Bubble не исчезает, но число живых activation уменьшается примерно до глубины pipeline, а не $m$. Interleaved 1F1B даёт rank несколько virtual stages и улучшает баланс ценой большего числа сообщений.

```text
S0: F0 F1 F2 F3 B0 F4 B1 F5 B2 ...
S1:  . F0 F1 F2 B0 F3 B1 F4 B2 ...
```

## Баланс важнее числа слоёв

Время slot задаёт медленнейшая стадия:

$$t_{\mathrm{slot}}=\max_i(t_i^{compute}+t_i^{exposed\ comm}).$$

Если времена стадий 80, 82, 130 и 78 ms, pipeline работает по 130 ms: третья стадия создаёт 48–52 ms простоя у остальных каждый steady-state slot. Делить нужно по измеренному времени и activation size; embedding, loss, shared weights и неодинаковая стоимость attention нарушают деление «поровну по слоям».

Передача BF16 activation $B_\mu S D$ содержит $2B_\mu SD$ байт в каждую сторону границы. Для $B_\mu=2,S=4096,D=8192$ — 128 MiB. На 25 GB/s минимум 5.4 ms на transfer; при 80-ms compute его можно скрыть, при десятках мелких sends latency возрастает.

Weight stashing/versioning нужен расписаниям, где backward использует не ту версию весов, что forward. Синхронные GPipe/1F1B обычно обеспечивают согласованную семантику шага; более агрессивные schedules должны явно определить её.

## Верификация

На малой модели сначала проверяют dense и pipeline forward logits после сборки output, затем gradients каждой стадии. После этого тестируют накопление $m$ микропакетов: normalization loss должна соответствовать тому же global batch. Отдельно проверяют последний неполный batch, tied embeddings и restart внутри schedule.

## Источники

- EDLS, [week 4](https://github.com/mryab/efficient-dl-systems), pipeline parallelism.
- Huang et al., [GPipe](https://arxiv.org/abs/1811.06965), 2019.
- Narayanan et al., [PipeDream-Flush/1F1B](https://arxiv.org/abs/2104.04473), 2021.
- Harvard Edge ML Systems Book, [Distributed Training](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), section `sec-distributed-training-model-parallelism`.

← [[44c Tensor и sequence parallelism]] · Далее: [[44e ZeRO, FSDP2, DeviceMesh и DTensor]]
