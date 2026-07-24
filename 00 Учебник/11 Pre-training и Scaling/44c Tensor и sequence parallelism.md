---
title: Tensor и sequence parallelism
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44c. Tensor и sequence parallelism

Если одна матрица или её вычисление не помещается на одном GPU, слой делят между rank. В MLP $Y=\phi(XA)B$ удобно разделить столбцы $A=[A_1,\ldots,A_p]$ и соответствующие строки $B^\top=[B_1^\top,\ldots,B_p^\top]$: каждый rank получает промежуточный hidden shard, затем частичные $Y_i$ суммируются.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/tensor-parallel-split.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-tensor-parallel-split`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

## MLP и attention

Для hidden $D$, expansion $H$ и $p$ rank local weights занимают $(DH+HD)/p$, но activation collectives происходят в каждом Transformer block. При batch-sequence $T=BS$ all-reduce тензора $T\times D$ стоит на ring порядка $2(p-1)TD/p$ элементов. Поэтому TP чувствителен к latency и topology и обычно ограничивается NVLink-доменом.

Multi-head attention естественно делит heads: rank хранит часть $Q,K,V$ projections и голов. Output projection снова требует объединения частичных результатов. При grouped-query attention число KV heads должно делиться или реплицироваться осмысленно; иначе «равное» разбиение создаёт дисбаланс.

Sequence parallelism оставляет elementwise/norm/dropout activation только для $S/p$ позиций, выполняя all-gather перед операцией, которой нужен полный hidden/sequence view, и reduce-scatter после неё. Это уменьшает activation memory, но не отменяет коммуникацию TP.

## Длинный контекст

Ulysses делит sequence и перед attention выполняет all-to-all, меняя раскладку «локальные позиции, все heads» на «все позиции, локальные heads». Ограничение: число heads должно поддерживать степень разбиения; all-to-all создаёт крупную перестановку.

Ring Attention оставляет KV blocks распределёнными и циркулирует их по кольцу. На каждом из $p$ раундов локальные queries обрабатывают очередной KV block; online softmax объединяет частичные максимумы и суммы без полного attention matrix. Каждый rank принимает примерно $(p-1)/p$ полного KV объёма, но передачу можно перекрыть с attention следующего блока. Causal masking меняет полезную работу блоков и требует load balancing.

### Численный пример

$B=2,S=32768,D=8192$, BF16 activation имеет $2BSD=1$ GiB. При sequence parallel $p=8$ постоянный shard — 128 MiB на rank. Но одна all-gather полного activation снова требует до 1 GiB logical payload; если implementation держит input, output и communication buffer одновременно, кратковременный пик может уничтожить ожидаемую экономию.

## Верификация

TP/sequence реализация проверяется на маленьком детерминированном слое: собрать shards и сравнить logits с dense baseline, затем сравнить input gradients и каждый weight-gradient после обратного разбиения. Проверка только loss недостаточна: взаимно компенсирующие ошибки layout могут дать близкий scalar.

## Источники

- EDLS, [week 4](https://github.com/mryab/efficient-dl-systems), tensor and sequence parallelism.
- Harvard Edge ML Systems Book, [Distributed Training](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), section `sec-distributed-training-model-parallelism`.
- Jacobs et al., [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509), 2023.
- Liu et al., [Ring Attention](https://arxiv.org/abs/2310.01889), 2023.

← [[44b Gradient checkpointing и offload]] · Далее: [[44d Pipeline parallelism]]
