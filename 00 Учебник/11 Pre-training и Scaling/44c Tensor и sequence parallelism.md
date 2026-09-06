---
title: Tensor, sequence и context parallelism
type: textbook-chapter
status: canonical
last_updated: 2026-09-06
source_unit_id:
  - lecture-07-parallelism-strategy-comparison
  - lecture-08-width-parallelism
  - lecture-08-activation-memory
  - assignment-02-task-tp-calcs
  - assignment-02-task-fsdp-tp-calcs
---

<a id="cs336-distributed-tp-sp-cp"></a>

# 44c. Tensor, sequence и context parallelism

Три похожих названия скрывают разные layouts. Tensor parallelism (TP) делит hidden dimensions и веса внутри слоя. Megatron sequence parallelism (SP) делит **только sequence-local activation** вокруг LayerNorm/dropout и работает вместе с TP. Context/attention parallelism делит сам длинный контекст и меняет способ вычисления attention — например, Ulysses all-to-all или Ring Attention.

Для каждого тензора нужно явно задать layout, коллективную операцию на границе
и проверку распределённого слоя относительно плотного. Здесь используются формы
Transformer из глав об attention, collectives из [[44a Processes, collectives и DDP]]
и memory ledger из [[44b Gradient checkpointing и offload]].

## Tensor parallelism для MLP

Пусть

$$X\in\mathbb R^{T\times D},\quad A\in\mathbb R^{D\times H},\quad B\in\mathbb R^{H\times D},\quad Y=\phi(XA)B,$$

где $T=B_{\mathrm{micro}}S$. При $p$ TP-rank:

1. Column-parallel $A=[A_0,\ldots,A_{p-1}]$, $A_r\in\mathbb R^{D\times H/p}$.
2. Каждый rank вычисляет $U_r=XA_r\in\mathbb R^{T\times H/p}$.
3. Row-parallel $B_r\in\mathbb R^{H/p\times D}$ даёт partial $Y_r=U_rB_r\in\mathbb R^{T\times D}$.
4. AllReduce суммирует $\sum_rY_r$ или ReduceScatter сразу оставляет sequence shard для SP.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/tensor-parallel-split.svg]]

*Источник: Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-tensor-parallelism-d76e`, figure `fig-tensor-parallel-split`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

Backward проходит границы в обратном порядке: градиент реплицированного выхода умножается на локальный $B_r^\top$ и даёт принадлежащий rank тензор $dU_r:[T,H/p]$; объединять его не нужно. Затем каждый rank вычисляет свой вклад $dX_r=dU_rA_r^\top:[T,D]$, и уже эти вклады суммируются между rank. Градиенты весов остаются рядом с соответствующими shards параметров.

### Shape ledger для SwiGLU

Для SwiGLU удобнее сразу записать обе входные проекции:

$$U=XW_u,\quad G=XW_g,\quad H=\operatorname{SiLU}(G)\odot U,
\quad Y=HW_o,$$

где $X:[T,D]$, $W_u,W_g:[D,F]$, $W_o:[F,D]$. При column-sharding на
$p$ rank каждый держит $W_{u,r},W_{g,r}:[D,F/p]$ и получает
$U_r,G_r,H_r:[T,F/p]$. Row-sharded $W_{o,r}:[F/p,D]$ даёт partial
$Y_r:[T,D]$, после чего нужен AllReduce либо ReduceScatter.

Backward сохраняет тот же ownership: из `dY:[T,D]` каждый rank получает
`dH_r:[T,F/p]`; локально вычисляет `dU_r`, `dG_r`, `dW_{u,r}`, `dW_{g,r}` и
`dW_{o,r}`; contributions `dX_r:[T,D]` суммируются. Эта ведомость форм полезнее
слов «разрезать MLP»: по ней видны границы обмена и тензоры, которые остаются
реплицированными.

Из неё следуют bottleneck inequalities. TP полезен по памяти, лишь если
уменьшение sharded weights/activations больше transient collective buffers; по
времени — если локальные GEMM и выигрыш памяти перекрывают exposed collective
time. Assignment 2 предлагает вывести эти неравенства символически для forward
и backward, а затем проверить profiler trace; одно фиксированное значение
`TP=8` из чужого измерения ответом не является.

Для activation $T\times D$ ring AllReduce передаёт на rank $2(p-1)TD/p$ элементов. При $T=8192,D=8192,p=8$ исходный BF16 tensor содержит $8192^2\cdot2=134\,217\,728$ B = 128 MiB. Ring factor $2(8-1)/8=1.75$ даёт 224 MiB на rank на одну редукцию. Это именно переданный объём, а не размер локального tensor; повторение в каждом блоке объясняет, почему TP обычно остаётся внутри NVLink-domain.

## Attention TP: heads, QKV и output

Исходный layout $X:[B,S,D]$. QKV projections column-shard heads:

$$Q_r,K_r,V_r:[B,h/p,S,d_h].$$

Attention локален, если heads независимы. После concatenation local context имеет `[B,S,D/p]`; row-parallel output projection создаёт partial `[B,S,D]`, затем AllReduce/ReduceScatter. Для GQA число KV heads может быть меньше $p$: implementation либо ограничивает TP degree, либо реплицирует KV heads. Неявное uneven sharding даёт разный compute и неверный layout.

## Megatron sequence parallelism — не attention parallelism

Без SP TP-region обычно держит replicated `[B,S,D]` activation вокруг LayerNorm, dropout и residual. Эти операции **не требуют соседних sequence positions**: каждый token нормируется по hidden dimension $D$, dropout применяется elementwise. Значит sequence axis можно разделить:

```text
после row-parallel linear:
Partial[B,S,D] --ReduceScatter(sequence)--> Shard[B,S/p,D]

LayerNorm / dropout / residual:
Shard[B,S/p,D] --локально, без межпозиционной зависимости-->

перед column-parallel linear:
Shard[B,S/p,D] --AllGather(sequence)--> Replicate[B,S,D]
```

Корректное утверждение не в том, что LayerNorm «нужен полный hidden view»: hidden $D$ остаётся локально полным. SP экономит replicated activation, разделяя независимые token rows; AllGather нужен следующему column-parallel matmul, ожидающему полный $T$ при принятом Megatron layout.

Для $B=2,S=32768,D=8192$ BF16 full activation — 1 GiB, SP при $p=8$ — 128 MiB persistent на rank. Но AllGather materializes logical 1 GiB input; current shard, gathered buffer и output могут перекрыться, поэтому peak измеряют timeline.

В обозначениях следующего слайда $s=S$, $b=B_{micro}$, $h=D$, $t$ — TP
degree, $a$ — число attention heads. Для принятой в лекции политики сохранения
активаций оценка одного слоя записана как

$$M_{act}=sbh\left(10+\frac{24}{t}+\frac{5as}{ht}\right).$$

Слагаемые с $1/t$ уменьшаются при TP; постоянное `10` соответствует
реплицированным LayerNorm/dropout и входам attention/MLP. Коэффициенты относятся
к конкретному saved-tensor ledger лекции, а не ко всякой реализации: fused
kernels, FlashAttention и checkpointing меняют их. Формула полезна именно тем,
что показывает, почему один TP не устраняет replicated activation term и зачем
к нему добавляют SP.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l08-p47.png]]

*Подчёркнутое слагаемое остаётся replicated без sequence parallelism, тогда как
остальные уменьшаются с ростом TP degree. Источник: Stanford
CS336 Spring 2026, Lecture 8, PDF p. 47, pinned commit
[`8b59b507`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf).
Все символы и границы применимости формулы определены непосредственно перед
рисунком.*

## Context parallelism: Ulysses

Context parallelism сохраняет sequence shards и распределяет **attention**, которому нужны связи между всеми positions. Ulysses начинает с

```text
Q,K,V local layout: [B, S/p, h, d_h]
```

AllToAll переставляет его в

```text
[B, S, h/p, d_h]
```

то есть каждый rank получает полный контекст, но только часть heads. После local attention обратный AllToAll возвращает `[B,S/p,h,d_h]`. До обмена на одном rank находится $3BShd_h/p$ элементов QKV; доля $(p-1)/p$ уходит другим участникам. Поэтому объём отправки одного QKV-перехода на rank равен

$$
3BShd_h\frac{p-1}{p^2},
$$

а сумма по всей группе — $3BShd_h(p-1)/p$. Ограничение: $h$ должен делиться на Ulysses degree или нужен uneven/replicated head layout.

## Context parallelism: Ring Attention

Ring Attention не собирает весь $S$. Rank $r$ хранит:

$$Q_r,K_r,V_r:[B,S/p,h,d_h].$$

На раунде $j$ он вычисляет scores local $Q_r$ с текущим KV-block, обновляет online-softmax state `(row_max, row_sum, weighted_value)` и посылает KV следующему rank. После $p$ раундов каждый query увидел все keys:

```text
state = empty_online_softmax()
kv = local_kv
for round in 0 .. p-1:
    state = attention_update(local_q, kv, causal_block_mask)
    kv = ring_send_recv(kv)
output = finalize(state)
```

На rank проходит примерно $(p-1)/p$ полного KV tensor. Нет полного attention matrix `[B,h,S,S]`, но latency имеет $p-1$ rounds. Causal triangle создаёт неодинаковую полезную работу по block pairs; load-balanced ring schedules переставляют blocks.

## Полная конфигурация

Модель: $D=8192,H=28672,h=64,d_h=128$, $B_\mu=2,S=32768$.
На восьми NVLink GPU можно сравнить две конфигурации, но нельзя приписать одной
и той же группе оба выигрыша одновременно:

- **TP/SP-вариант:** `TP=8` даёт MLP shard $A_r:[8192,3584]$ и 8 query
  heads/rank; `SP=8` оставляет LayerNorm/dropout/residual activation
  `[2,4096,8192]`, то есть 128 MiB BF16/rank.
- **CP-вариант на тех же восьми GPU:** взять `TP=1, CP=8`; Ulysses degree 8
  допустим, потому что 64 heads делятся на 8.
- **Совместный `TP=8, CP=8`:** это две независимые оси 2D mesh и потому 64
  ranks до учёта DP/PP. Восемь физических ranks не следует повторно считать
  сразу по обеим осям.

Пошаговый слой TP/SP-варианта:

1. SP shard проходит local LayerNorm.
2. AllGather sequence формирует input column-parallel QKV/MLP.
3. TP attention/MLP вычисляет hidden shards.
4. Row-parallel partial output проходит ReduceScatter sequence.
5. Local residual/dropout возвращает SP shard.

В совместной 2D-конфигурации attention дополнительно выполняет Ulysses/Ring
transitions внутри ортогональной CP-group; принадлежность каждого rank обеим
группам должна быть задана явно.

## Trade-offs и проверка

TP делит weights и крупную GEMM, но вызывает collectives каждый layer. SP уменьшает non-TP activation почти без изменения FLOPs, но добавляет/переиспользует AllGather/ReduceScatter и transient buffers. Ulysses имеет два AllToAll и head divisibility; Ring Attention имеет много rounds и сложный causal balance, зато избегает full-context materialization.

Проверка на маленьком слое:

1. Инициализировать одинаковые dense weights и разрезать их по documented dimensions.
2. Сравнить Q/K/V, logits attention, MLP intermediate и final logits после сборки.
3. Сравнить $\partial L/\partial X$ и каждый weight-gradient после обратного layout transform.
4. Для SP проверить LayerNorm mean/variance по hidden $D$, не по sequence shard.
5. Для Ring сравнить online-softmax `(max,sum)` с dense stable softmax и causal mask.
6. Профиль должен показать именно ожидаемые collectives и bytes; лишний implicit redistribution — ошибка layout design.

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/lecture.pdf|EDLS Week 4 — лекция]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/practice_part2.ipynb|EDLS Week 4 — practice part 2]];
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/distributed_training|Harvard CS249r — Distributed Training]];
- [[02 Areas/ML & DL/06 Практика/11 Разрезать Transformer по TP и SP|практика TP/SP]].

Практику удобно начинать с двух linear layers: сначала для каждого rank подписать локальные shapes и collectives и только затем перенести тот же разрез на attention и MLP. Harvard-глава помещает этот разрез в общую систему осей параллелизма.

- EDLS, pinned commit `e632aa89…`, [`week04_large_models/lecture.pdf`, PDF pp. 40–45 “Tensor-parallel training”, pp. 46–47 “Sequence Parallelism”](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/lecture.pdf), and [`week04_large_models/practice_part2.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/practice_part2.ipynb).
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-tensor-parallelism-d76e` and `sec-distributed-training-parallelism-infrastructure`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), §4 sequence parallelism, 2022.
- Jacobs et al., [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509), §3, 2023.
- Liu et al., [Ring Attention](https://arxiv.org/abs/2310.01889), Algorithm 1, 2023.
- Stanford CS336 Spring 2026, [Lecture 8](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf), PDF pp. 39–54; p. 47 activation accounting.
- Stanford CS336 Spring 2026, [Assignment 2](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736), TP and 2D FSDP×TP arithmetic tasks.

← [[44b Gradient checkpointing и offload]] · Далее: [[44d Pipeline parallelism]]
