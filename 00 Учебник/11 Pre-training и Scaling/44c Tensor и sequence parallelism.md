---
title: Tensor, sequence и context parallelism
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44c. Tensor, sequence и context parallelism

Три похожих названия скрывают разные layouts. Tensor parallelism (TP) делит hidden dimensions и веса внутри слоя. Megatron sequence parallelism (SP) делит **только sequence-local activation** вокруг LayerNorm/dropout и работает вместе с TP. Context/attention parallelism делит сам длинный контекст и меняет способ вычисления attention — например, Ulysses all-to-all или Ring Attention.

## Предпосылки и цели

Нужны формы Transformer из глав об attention, collectives из [[44a Processes, collectives и DDP]] и memory ledger из [[44b Gradient checkpointing и offload]]. После главы можно проследить layout каждого tensor, назвать collective на каждой границе и проверить distributed layer против dense.

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

Backward выполняет обратные transitions: gradient replicated output проходит локальный $B_r^\top$, shards hidden-gradient объединяются, а weight-gradients остаются при соответствующих weight shards.

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

## Context parallelism: Ulysses

Context parallelism сохраняет sequence shards и распределяет **attention**, которому нужны связи между всеми positions. Ulysses начинает с

```text
Q,K,V local layout: [B, S/p, h, d_h]
```

AllToAll переставляет его в

```text
[B, S, h/p, d_h]
```

то есть каждый rank получает полный контекст, но только часть heads. После local attention обратный AllToAll возвращает `[B,S/p,h,d_h]`. Logical send volume одного QKV transition порядка $3BShd_h(p-1)/p$ элементов на rank. Ограничение: $h$ должен делиться на Ulysses degree или нужен uneven/replicated head layout.

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

Модель: $D=8192,H=28672,h=64,d_h=128$, $B_\mu=2,S=32768$, восемь NVLink GPU.

- `TP=8`: MLP shard $A_r:[8192,3584]$, attention по 8 query heads/rank.
- `SP=8`: LayerNorm/dropout/residual activation `[2,4096,8192]` = 128 MiB BF16/rank.
- Если памяти attention всё ещё мало, выбрать **отдельную** CP-group: Ulysses degree 8 допустим, потому что 64 heads делятся на 8.
- Не следует одновременно считать TP и CP одной осью без явного 2D mesh: иначе один и тот же rank-count дважды «экономит» память на бумаге.

Пошаговый layer:

1. SP shard проходит local LayerNorm.
2. AllGather sequence формирует input column-parallel QKV/MLP.
3. TP attention/MLP вычисляет hidden shards.
4. Row-parallel partial output проходит ReduceScatter sequence.
5. Local residual/dropout возвращает SP shard.
6. При CP вместо полного attention применяют Ulysses/Ring transitions внутри CP group.

## Trade-offs и проверка

TP делит weights и крупную GEMM, но вызывает collectives каждый layer. SP уменьшает non-TP activation почти без изменения FLOPs, но добавляет/переиспользует AllGather/ReduceScatter и transient buffers. Ulysses имеет два AllToAll и head divisibility; Ring Attention имеет много rounds и сложный causal balance, зато избегает full-context materialization.

Проверка на маленьком слое:

1. Инициализировать одинаковые dense weights и разрезать их по documented dimensions.
2. Сравнить Q/K/V, logits attention, MLP intermediate и final logits после сборки.
3. Сравнить $\partial L/\partial X$ и каждый weight-gradient после обратного layout transform.
4. Для SP проверить LayerNorm mean/variance по hidden $D$, не по sequence shard.
5. Для Ring сравнить online-softmax `(max,sum)` с dense stable softmax и causal mask.
6. Профиль должен показать именно ожидаемые collectives и bytes; лишний implicit redistribution — ошибка layout design.

## Источники

- EDLS, pinned commit `e632aa89…`, [`week04_large_models/lecture.pdf`, PDF pp. 40–45 “Tensor-parallel training”, pp. 46–47 “Sequence Parallelism”](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/lecture.pdf), and [`week04_large_models/practice_part2.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/practice_part2.ipynb).
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-tensor-parallelism-d76e` and `sec-distributed-training-parallelism-infrastructure`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), §4 sequence parallelism, 2022.
- Jacobs et al., [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509), §3, 2023.
- Liu et al., [Ring Attention](https://arxiv.org/abs/2310.01889), Algorithm 1, 2023.

← [[44b Gradient checkpointing и offload]] · Далее: [[44d Pipeline parallelism]]
