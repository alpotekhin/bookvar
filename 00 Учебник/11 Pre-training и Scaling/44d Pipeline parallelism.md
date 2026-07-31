---
title: Pipeline parallelism
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44d. Pipeline parallelism

Pipeline parallelism делит не матрицу, а последовательность слоёв. Если модель из 48 Transformer-блоков не помещается на одном GPU, четыре стадии могут хранить по 12 блоков. Цена такого размещения — зависимости между стадиями: стадия $i+1$ не начнёт forward микропакета, пока не получит activation от $i$, а стадия $i$ не начнёт backward, пока не получит gradient activation от $i+1$.

Используя forward/backward, gradient accumulation и point-to-point `send/recv` из [[44a Processes, collectives и DDP]], разберём четыре задачи:

1. построить dependency-valid GPipe и 1F1B timeline;
2. вывести bubble из числа пустых slot, а activation memory — из числа незавершённых forward;
3. рассчитать формы и объём межстадийных тензоров;
4. проверить pipeline против непараллельного запуска.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/pipeline-parallelism.svg]]

*Источник: Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-pipeline-parallelism-8748`, figure `fig-pipeline-parallelism`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

## Состояние и формы на границе

Для микропакета $b_\mu$, длины $S$ и hidden size $D$ forward-сообщение имеет форму

$$A_i^{(\mu)}\in\mathbb R^{b_\mu\times S\times D},$$

а backward возвращает тензор той же формы $\partial L/\partial A_i^{(\mu)}$. При BF16 объём одного направления равен $2b_\mu SD$ байт. Для $b_\mu=2,S=4096,D=8192$ это $134\,217\,728$ B = 128 MiB. На эффективных 25 GB/s нижняя граница передачи — 5.37 ms; sender и receiver должны согласовать dtype, shape, microbatch id и stream dependency.

На стадии одновременно живут:

- её параметры, gradients и optimizer state;
- входная activation каждого forward, backward которого ещё не завершён;
- communication buffers текущего send/recv;
- временные tensors локальных operators.

## GPipe: complete p=4, m=4 schedule

Обозначим `Fμ` forward и `Bμ` backward микропакета $\mu$. Один slot равен времени медленнейшей стадии в одном направлении. GPipe сначала полностью заполняет и опустошает forward, затем выполняет backward в обратном порядке микропакетов:

```text
slot  0  1  2  3  4  5  6  7  8  9 10 11 12 13
S0   F0 F1 F2 F3  .  .  .  .  .  . B3 B2 B1 B0
S1    . F0 F1 F2 F3  .  .  .  . B3 B2 B1 B0  .
S2    .  . F0 F1 F2 F3  .  . B3 B2 B1 B0  .  .
S3    .  .  . F0 F1 F2 F3 B3 B2 B1 B0  .  .  .
```

Проверка зависимостей выполняется для каждой клетки:

- `Fμ@Si` позже `Fμ@S(i-1)` минимум на один slot;
- `Bμ@Si` позже `Bμ@S(i+1)`;
- `Bμ@Si` позже собственного `Fμ@Si`;
- одна стадия выполняет не более одной операции за slot.

Для одного forward fill+drain требуется $m+p-1$ slot, из которых стадия делает $m$ полезных операций:

$$u_{\mathrm{phase}}=\frac{m}{m+p-1},\qquad b_{\mathrm{phase}}=\frac{p-1}{m+p-1}.$$

При $p=m=4$ utilization фазы $4/7=57.1\%$, bubble $3/7=42.9\%$. В полном flush-цикле 14 slot каждая стадия выполняет 8 операций, поэтому тот же utilization $8/14=57.1\%$. После `F3` стадия S0 хранит четыре незавершённых activation boundary: activation memory растёт как $O(m)$.

### Псевдокод GPipe

```text
for μ in 0 .. m-1:
    x = recv_forward(μ) if stage > 0 else next_microbatch(μ)
    y = local_forward(x)
    save_for_backward(μ, x, y)
    send_forward(μ, y) if stage < p-1

for μ in reverse(0 .. m-1):
    dy = recv_backward(μ) if stage < p-1 else loss_grad(μ)
    dx, dθ = local_backward(μ, dy)
    accumulate(dθ)
    send_backward(μ, dx) if stage > 0

optimizer_step_once_after_all_microbatches()
```

## 1F1B: warm-up, steady state и drain

1F1B начинает backward раньше. Ниже корректное flush-расписание для тех же $p=4,m=4$; пустые slot сохранены, поэтому dependency chain можно проверить буквально:

```text
slot  0  1  2  3  4  5  6  7  8  9 10 11 12 13
S0   F0 F1 F2 F3  .  .  . B0  . B1  . B2  . B3
S1    . F0 F1 F2 F3  . B0  . B1  . B2  . B3  .
S2    .  . F0 F1 F2  B0 F3 B1  . B2  . B3  .  .
S3    .  .  . F0 B0 F1 B1 F2 B2 F3 B3  .  .  .
```

- **Warm-up:** S0 выпускает четыре `F`; S3 достигает `F0` в slot 3.
- **Steady state:** S3 чередует `B0,F1,B1,F2,…`; upstream stages чередуют направления, когда gradient дошёл до них.
- **Drain:** после последнего `F3` gradients `B3` проходят S3→S0 в slot 10–13.

Для короткого $m=p=4$ 1F1B не уменьшает число slot: фундаментальные fill/drain dependencies остаются. Зато peak незавершённых forward зависит от стадии. В показанном schedule после соответствующих slots максимумы равны S0=4, S1=4, S2=3, S3=1: поздняя стадия начинает backward раньше, а первая всё ещё должна прогреть весь pipeline. При $m\gg p$ число live microbatches ограничивается warm-up и локальным порядком операций, а не $m$ как в GPipe; конкретный peak нужно считать из timeline, а не подставлять одно $p-i$ для всех стадий.

### Псевдокод 1F1B

```text
warmup = min(p - stage - 1, m)
run warmup forward operations

for each remaining forward μ:
    run Fμ and send activation
    receive oldest ready output-gradient
    run its backward and send input-gradient

drain all outstanding backward operations
optimizer_step_once_after_flush()
```

Конкретная runtime должна избегать взаимной блокировки: заранее согласовать порядок nonblocking `irecv/isend`, microbatch tags и streams. Простой обмен двух blocking `send` на соседях может повиснуть.

## Stage balance, bubble и память

Slot задаёт медленнейшая стадия:

$$t_{\mathrm{slot}}=\max_i(t_i^{compute}+t_i^{exposed\ communication}).$$

Времена 80, 82, 130 и 78 ms означают 130-ms slot: третья стадия оставляет 48–52 ms простоя у соседей. Делить «по 12 слоёв» недостаточно — embedding, loss, shared weights, attention sequence length и recomputation различаются.

Activation ledger для стадии $i$:

$$M_{\mathrm{act},i}=n_{\mathrm{live},i}\cdot M_{\mathrm{saved\ per\ microbatch},i}.$$

Если local stage сохраняет 900 MiB на микропакет, GPipe с $m=16$ требует до 14.1 GiB. Для **показанного** 1F1B schedule S2 достигает $n_{\mathrm{live}}=3$, поэтому $3\cdot900=2700$ MiB = 2.64 GiB; S3 при peak 1 требует 900 MiB, а S0 при peak 4 — 3.52 GiB. Это stage-dependent ledger, не универсальная граница: при другом $m$, interleaving или checkpointing saved set и live counts меняются.

## Полная конфигурация

Рассмотрим 48-layer, $D=8192$, $S=4096$ на 32 GPU:

- `PP=4`: по 12 блоков на stage;
- `DP=8`: восемь replicas pipeline;
- global batch 256, значит на replica 32 sequences;
- $m=16$, $b_\mu=2$, gradient accumulation охватывает все 16;
- BF16 activation transfer 128 MiB на boundary;
- 1F1B flush, optimizer update только после drain;
- tied embedding размещён на S0/S3 и его gradient синхронизируется отдельной группой.

Ожидаемый односторонний bubble $3/(16+3)=15.8\%$. Если stage compute 90 ms, а transfer 5.4 ms полностью скрывается, теоретический cycle порядка $2(m+p-1)90=3.42$ s. Если одна stage занимает 120 ms, нижняя граница растёт до 4.56 s: сначала балансируют partition, затем увеличивают $m$.

## Trade-offs и верификация

Большое $m$ уменьшает bubble, но увеличивает GPipe activation memory и уменьшает microbatch, ухудшая GEMM utilization. 1F1B снижает memory, но усложняет communication order. Interleaving virtual stages улучшает баланс, однако увеличивает число boundary messages. Weight stashing нужен только расписаниям, где optimizer меняет weight до backward соответствующего forward; flush-схемы выше обновляют веса один раз и сохраняют синхронную семантику.

Проверка:

1. На маленькой модели собрать pipeline output и сравнить logits с dense baseline.
2. Сравнить input-gradient каждой boundary и weight-gradient каждого stage.
3. Убедиться, что сумма loss нормализована по тому же global batch.
4. Валидатор schedule проверяет четыре dependency rules для каждой операции и ровно по одному `Fμ/Bμ` на stage.
5. Профиль подтверждает ожидаемые live activation counts, bubble slots и отсутствие blocking gaps.
6. Повторить для неполного последнего batch, tied weights и restart между optimizer steps.

## Материалы для практики

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/lecture.pdf|EDLS Week 4 — полная лекция]];
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/distributed_training|Harvard CS249r — Distributed Training]].

В EDLS pipeline parallelism рассматривается рядом с memory pressure больших моделей, а в Harvard — рядом с data и tensor parallelism. Первая версия помогает посчитать bubble для конкретного schedule, вторая — понять, когда pipeline становится подходящей осью разбиения всей системы.

## Источники

- EDLS, pinned commit `e632aa89…`, [`week04_large_models/lecture.pdf`, PDF pp. 23–27 model-parallel/GPipe, pp. 28–33 bubble/1F1B/ZeroBubble/PipeDream, pp. 34–38 “Pipelining Recap”](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/lecture.pdf), and [`week04_large_models/practice_part2.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/practice_part2.ipynb).
- Huang et al., [GPipe](https://arxiv.org/abs/1811.06965), Algorithm 1 and pipeline partitioning, 2019.
- Narayanan et al., [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473), §3.2, PipeDream-Flush/1F1B, 2021.
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-pipeline-parallelism-8748` and `sec-distributed-training-systems-systems-model-parallelism-tradeoffs`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).

← [[44c Tensor и sequence parallelism]] · Далее: [[44e ZeRO, FSDP2, DeviceMesh и DTensor]]
