---
title: Арифметика Transformer и MoE
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# Арифметика Transformer и MoE

Перед выбором parallelism полезно оценить пять бюджетов: параметры, FLOP,
активации, optimizer state и коммуникации. Ни один из них не выводится только
из надписи «70B».

Обозначения: batch tokens $N=BS$, hidden width $H$, FFN width $I$, слоёв $L$,
attention heads $n_h$, KV-heads $n_{kv}$, head width $d$, experts $E$, top-$k$
активных experts.

## Dense Transformer: параметры

Для GQA attention:

$$P_{\text{attn}}=H(n_hd)+2H(n_{kv}d)+H^2.
$$

При MHA, $n_hd=n_{kv}d=H$, получаем $4H^2$. SwiGLU FFN имеет три матрицы:

$$P_{\text{ffn}}=3HI.
$$

На слой $P_\ell\approx P_{\text{attn}}+3HI$ (без малых norm/bias). Embeddings
добавляют $VH$, если output head tied, и ещё $VH$ иначе.

## FLOP forward/backward

Linear $[N,a]\times[a,b]$ стоит примерно $2Nab$ FLOP. Поэтому projections:

$$F_{\text{attn-proj}}\approx2N P_{\text{attn}},\qquad
F_{\text{ffn}}\approx6NHI.
$$

Attention scores и применение probabilities:

$$F_{\text{quadratic}}\approx4BS^2(n_hd).
$$

Forward слоя — сумма этих величин. Для dense matmul backward по input и weight
обычно добавляет примерно удвоенный forward, поэтому training step
$F_{\text{train}}\approx3F_{\text{forward}}$; известная оценка всей dense LM —
порядка $6P$ FLOP на token, когда quadratic attention не доминирует.

## Memory ledger

В BF16 веса занимают $2P$ bytes. Adam mixed precision часто хранит:

| Состояние | Bytes/parameter |
|---|---:|
| BF16 weight | 2 |
| FP32 master weight | 4 |
| gradient | 2 или 4 |
| FP32 first/second moments | 8 |

Итого обычно 16–18 bytes/parameter до временных buffers. Для 7B это
112–126 GB: одной 80-GB GPU мало, даже если сами BF16 weights — 14 GB.

Активации растут примерно как $O(LNSH)$ для сохранённых states и как
$O(Bn_hS^2)$ при материализации attention matrix. FlashAttention убирает
материализацию $S^2$, activation checkpointing обменивает память на recompute.

## MoE: параметры не равны compute

MoE FFN хранит $E$ experts:

$$P_{\text{MoE}}=3EHI,
\qquad
F_{\text{MoE/token}}\approx6kHI.
$$

Параметры растут с $E$, вычисления — с $k$. Отсюда низкое отношение compute к
перемещаемым weights: слой легче становится memory/communication-bound.

Router создаёт scores, выбирает top-$k$, затем:

```text
tokens → route → dispatch/all-to-all → permute by expert
       → GroupedGEMM → unpermute → all-to-all → weighted combine
```

GroupedGEMM объединяет разные expert matmuls в один эффективный запуск. Но
неравномерный routing создаёт load imbalance; capacity factor и auxiliary
balancing loss меняют и качество, и systems cost.

## TP против EP

**Tensor parallelism** режет матрицу каждого expert между ranks и требует
AllGather/ReduceScatter activations. **Expert parallelism** размещает целые
experts на разных ranks и пересылает токены All-to-All. При $S=8192$, $E=256$,
$k=8$, $H=7168$, $I=2048$, как в EDLS:

$$F=2SkHI\approx1{,}92\cdot10^{12}\text{ FLOP},
$$

что при 800 TFLOP/s даёт идеальные 2,4 ms. Реальное время — максимум compute,
memory и dispatch, а не только эта дробь. EP уменьшает объём expert weights на
rank, но добавляет чувствительность к network topology и imbalance.

## Pipeline parallelism

PP делит слои по stages; между ними идут только activations/gradients. Для
microbatch activation payload порядка $2NSH$ bytes в BF16 — часто дешевле,
чем собирать гигантский MoE layer.

- **GPipe**: все forwards, затем backwards; большой activation memory и bubble.
- **1F1B**: после warmup каждый stage чередует один forward и один backward,
  уменьшая число живых microbatches.
- **ZeroBubble**: дробит backward на градиенты по input и weights и размещает
  работу в idle slots, стремясь убрать pipeline bubble.
- **DualPipeV**: двунаправленный V-shaped schedule для сочетания PP и EP;
  вычисления двух потоков помогают перекрыть All-to-All, но schedule и memory
  planning сложнее.

При $p$ stages и $m$ microbatches грубая bubble fraction обычного pipeline
порядка $(p-1)/(m+p-1)$: больше microbatches повышает utilization, но увеличивает
activation pressure и latency шага.

## Как выбрать разбиение

1. Посчитать bytes всех состояний и peak activations.
2. Для каждого оператора оценить compute, HBM и network time.
3. TP держать внутри быстрого fabric, если возможно.
4. EP выбирать, когда expert weights не помещаются или FSDP traffic доминирует.
5. PP использовать для дешёвой межхостовой передачи, принимая scheduling cost.
6. Проверить расчёт trace-профилированием.

## Источники

- [EDLS week 6 complete lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf)
- [Harvard CS249r, Neural Network Computation](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/nn_computation/nn_computation.qmd)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Измерение производительности и roofline]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Численные форматы и mixed precision]] →
