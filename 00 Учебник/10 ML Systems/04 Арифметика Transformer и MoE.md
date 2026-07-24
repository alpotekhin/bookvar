---
title: Арифметика Transformer и MoE
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# Арифметика Transformer и MoE

Эта глава превращает архитектуру в ведомость ресурсов. Мы считаем не только
параметры и FLOP, но и живые активации, состояния оптимизатора, байты HBM и
коммуникации. Именно такой порядок использует полная лекция EDLS week 6:
**data logistics → local GPU logistics → elementwise/fusion → model-state
logistics → activation logistics → MoE logistics** (slides 4–146).

Обозначения: $B$ — число sequences, $S$ — длина, $N=BS$ — число токенов
microbatch, $H$ — hidden width, $I$ — FFN width, $L$ — слои, $V$ — vocabulary,
$n_h,n_{kv}$ — query- и KV-heads, $d$ — head width, $E$ — experts, $k$ —
top-$k$. Один BF16-элемент занимает $b=2$ bytes.

## Нулевая стадия: данные тоже входят в step time

EDLS начинает не с GEMM, а с простоя: CPU читает и готовит batch, пока дорогой
GPU ждёт (slides 4–7). Синхронный шаг

$$T_{\text{step}}=T_{\text{load}}+T_{\text{compute}}$$

после prefetch из отдельного процесса в устойчивом режиме приближается к
$\max(T_{\text{load}},T_{\text{compute}})$. Если compute занимает 180 ms, а
подготовка — 70 ms, overlap меняет потолок с 250 до 180 ms, то есть с 4 до
5,56 steps/s. Но очередь должна держать хотя бы один готовый batch, pinned
host memory — не исчерпываться, а worker seeds — оставаться воспроизводимыми.

## Параметры dense Transformer

Для GQA attention четыре проекции дают

$$
P_{\text{attn}}
=H(n_hd)+2H(n_{kv}d)+H(n_hd).
$$

При MHA $n_hd=n_{kv}d=H$, поэтому $P_{\text{attn}}=4H^2$. Для SwiGLU три
матрицы — gate, up и down:

$$P_{\text{ffn}}=3HI.$$

Без малых norm/bias один слой содержит $P_\ell=P_{\text{attn}}+3HI$;
embeddings добавляют $VH$, если output head tied, и ещё $VH$ иначе.

### Проверка на Llama 7B

Для $H=4096$, $I=11008$, MHA и $L=32$:

$$
P_\ell=4\cdot4096^2+3\cdot4096\cdot11008
\approx202{,}4\text{ M},
$$

то есть $\approx6{,}48$ B параметров блоков. Tied embedding
$32000\cdot4096\approx131$ M и нормы доводят порядок до заявленных 7B. Такой
расчёт служит sanity check: ошибка в коэффициенте SwiGLU на один projection
сразу даёт сотни миллионов параметров.

## FLOP: projection, attention и backward

GEMM $[N,a]\times[a,b]$ стоит $2Nab$ FLOP. Поэтому

$$F_{\text{proj}}=2NP_{\text{attn}},\qquad
F_{\text{SwiGLU}}=6NHI.
$$

Attention scores $QK^\top$ и умножение probabilities на $V$ вместе дают

$$F_{\text{quadratic}}\approx4BS^2(n_hd).$$

Backward linear-слоя вычисляет градиенты по input и weight и обычно добавляет
примерно два forward. Поэтому для GEMM-heavy dense LM
$F_{\text{train}}\approx3F_{\text{forward}}\approx6P$ FLOP/token, пока
quadratic attention и vocabulary projection не доминируют.

Для Llama 7B при $N=8192$:

$$F_{\ell,\text{linear}}\approx2N\cdot202{,}4\text{ M}=3{,}32\text{ TFLOP}.$$

На 800 TFLOP/s идеальный forward слоя — 4,15 ms. EDLS slides 86–88 специально
сопоставляют эту нижнюю границу с H100 memory/compute time: фактическое время
выше из-за elementwise kernels, launch gaps и неполной эффективности.

## Activation ledger: где появляется $O(LNH)$

Поскольку $N=BS$, базовое состояние одного residual stream имеет форму
$[B,S,H]$ и занимает $bNH$ bytes. Для $L$ слоёв минимальный порядок сохранённых
границ блоков — **$O(LNH)=O(LBSH)$**, а не $O(LNSH)$: множитель $S$ уже входит
в $N$.

Но реальный ledger шире:

| Что требуется backward | Форма/порядок на слой | BF16 bytes |
|---|---:|---:|
| вход блока/residual | $[B,S,H]$ | $bNH$ |
| Q | $[B,S,n_h,d]$ | $bNH$ |
| K,V при GQA | по $[B,S,n_{kv},d]$ | $2bNSn_{kv}d/S=2bNn_{kv}d$ |
| logits/probabilities обычного attention | $[B,n_h,S,S]$ | $bBn_hS^2$ каждое |
| MLP gate и up intermediates | два $[N,I]$ | $2bNI$ |
| SwiGLU product / dropout masks | $[N,I]$ и implementation-dependent masks | $\gtrsim bNI$ |
| norm statistics | $[N]$ плюс, иногда, normalized input | $O(N)+O(NH)$ |

Здесь запись для K,V намеренно упрощается до $[B,S,n_{kv},d]$: в bytes это
$2bBn_{kv}Sd=2bNn_{kv}d$.

### Что меняют FlashAttention и checkpointing

- Обычный attention может сохранить $O(Bn_hS^2)$ logits/probabilities.
  FlashAttention вычисляет softmax плитками, хранит row statistics и не
  материализует полную $S^2$ матрицу; Q/K/V и block boundaries всё равно нужны.
- Full-block checkpointing сохраняет только вход блока $O(NH)$, а QKV и MLP
  intermediates пересчитывает в backward. Для $L$ блоков это уменьшает главный
  ledger до порядка $LbNH$ плюс временные активации одного пересчитываемого
  блока.
- Selective checkpointing оставляет дорогие GEMM outputs и пересчитывает
  дешёвые norm/activation операции либо наоборот в зависимости от memory
  budget. EDLS slides 93–104 показывают, почему «не все активации одинаково
  дорого вычислять».

Для $B=1,S=8192,H=4096,I=11008$ один residual — 67,1 MB, а два MLP projections
— около 360,7 MB. Полная attention matrix при $n_h=32$ — 4,29 GB на один
тензор. Это объясняет, почему FlashAttention меняет сам порядок памяти, а
checkpointing — множитель и recompute time.

## Model-state ledger и FSDP

Типичный Adam mixed-precision ledger:

| Состояние | bytes/parameter |
|---|---:|
| BF16 working weight | 2 |
| FP32 master weight | 4 |
| gradient | 2 или 4 |
| FP32 first и second moments | 8 |

Итого 16–18 bytes/parameter до buffers. 7B требует 112–126 GB, 70B —
1,12–1,26 TB.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/training_optimizer_memory.svg]]

*Источник: Harvard CS249r, Vol. I, Model Training, section “Memory
Architecture”/optimizer-state accounting, figure `training_optimizer_memory.svg`,
commit `45ecc8d…`, CC BY-NC-SA 4.0. Рисунок импортирован без изменений.*

FSDP шарит параметры, gradients и optimizer state между $p$ ranks. Идеальный
resident ledger становится примерно $(16\text{-}18)P/p$, но перед вычислением
слоя нужен AllGather weights, после backward — ReduceScatter gradients.
EDLS slides 82–91 рассматривают FSDP как **логистику**, а не бесплатное
сокращение памяти.

Для BF16 слоя с $P_\ell$ параметрами каждый rank должен получить порядка
$2P_\ell(p-1)/p$ bytes на AllGather и отправить сопоставимый объём на
ReduceScatter. Если слой Llama 7B занимает 405 MB BF16 и fabric даёт
200 GB/s effective, одна такая передача имеет нижнюю границу около 2 ms.
Она скрывается только когда prefetch следующего слоя перекрывается compute
текущего и одновременно не раздувает peak memory.

```text
time →
layer i:      [all-gather i][ forward i ][free full weights]
layer i + 1:                [all-gather i+1][ forward i+1 ]
                              ^ полезный overlap ^
```

Слишком ранний prefetch держит несколько full-parameter buffers; слишком
поздний оставляет GPU ждать NCCL. Slides 89–90 также предупреждают, что
коммуникации конкурируют за topology links и требуют корректных NCCL streams
и dependencies.

## Локальная логистика: fusion и Liger

После больших GEMM остаются RMSNorm, RoPE, SwiGLU, dropout, cross-entropy и
optimizer updates. У каждого может быть малая arithmetic intensity:
прочитать tensor, сделать несколько операций, записать обратно. Несколько
отдельных kernels многократно проходят HBM.

Fusion выполняет цепочку в одном kernel и держит промежуточные значения в
register/shared memory. Если три elementwise стадии читают и пишут tensor
размера $M$, независимое исполнение перемещает примерно $6M$ bytes, fused —
около $2M$; теоретическое bandwidth speedup до $3\times$, но register pressure
и occupancy могут его уменьшить. EDLS slides 39–81 разбирают locality,
torch.compile/Triton и Liger Kernel; slide 74 рекомендует Liger как источник
готовых kernels, включая memory-efficient loss. Это **адаптация содержания
лекции**, не утверждение о любой версии библиотеки: kernel eligibility нужно
проверять trace и tests для конкретных shapes/dtypes.

## Dense worked maps: 7B и 70B

EDLS slides 24–30 дают слой-за-слоем memory maps для 100M, 1B, 7B, Llama 70B,
Qwen 30B-A3B и Qwen 235B-A32B. Их назначение — различать:

1. persistent weights/optimizer states;
2. layer-local full weights после FSDP gather;
3. saved activations;
4. temporary communication и GEMM workspaces.

Для Llama 70B ($H=8192,I=28672,L=80$, GQA $n_{kv}=8$) FFN одного слоя:

$$P_{\text{ffn}}=3\cdot8192\cdot28672\approx704{,}6\text{ M},$$

то есть 1,41 GB BF16 — значение того же порядка, что slide 28. Даже если
persistent state разделён на 1024 GPU, один gathered layer всё ещё требует
гигабайты; поэтому peak определяется schedule gather/free, а не средним
$P/p$. Slides 109–113 показывают вариант TP=2: матрицы и compute делятся, но
появляются activation collectives.

## MoE: stored parameters ≠ active FLOP

MoE хранит $E$ SwiGLU experts:

$$P_{\text{experts}}=3EHI,\qquad
F_{\text{experts/token}}=6kHI.
$$

Коэффициент **6** — это все три projections и правило 2 FLOP на MAC. Если
считается только один expert projection, формула была бы $2kHI$; смешивать эти
две оценки нельзя.

Router создаёт scores и top-$k$, затем:

```text
tokens
  │ route + capacity accounting
  ▼
dispatch ── all-to-all ──► permute/group by local expert
                              │
                              ▼
                         GroupedGEMM
                              │
combine ◄── all-to-all ◄── unpermute + router weights
```

GroupedGEMM запускает матрицы experts совместно, но не превращает ragged
workload в плотный: число tokens на expert задаёт разные $M$ dimensions.
Load imbalance, padding до capacity и stragglers уменьшают utilization.

### Qwen 235B-A32B: почему FSDP становится дорогим

EDLS slides 122–123 приводят для Qwen 3 235B слой около 5 GB BF16 и сравнивают
идеальное forward time с FSDP communication. Смысл расчёта: sparse compute
использует лишь $k$ experts, но обычный FSDP AllGather перемещает все expert
weights. При 200 GB/s 5 GB имеют нижнюю границу 25 ms только на чтение по
fabric, прежде чем считать protocol/topology overhead. EP размещает experts
постоянно и вместо weights пересылает token activations.

### DeepSeek-V3 671B

Slides 124–126 повторяют карту для DeepSeek-V3: около 21 GB BF16 на слой в
приведённой конфигурации. Это не универсальная характеристика всех
implementations, а оценка конкретного slide. Даже 1 TB/s дал бы 21 ms на
полный перенос слоя; sparse active compute может быть короче. Отсюда
необходимость EP/PP и topology-aware placement.

### TP=8 и EP=8: исправленный worked example

Пусть $S=8192,E=256,k=8,H=7168,I=2048$, как на EDLS slides 131–132.
Полный SwiGLU compute на один token batch:

$$
F_{\mathrm{SwiGLU}}=6SkHI
\approx 6\cdot8192\cdot8\cdot7168\cdot2048
\approx 5.77\,\mathrm{TFLOP}.
$$

На 800 TFLOP/s идеал — 7,2 ms. Число около 2,4–2,5 ms в исходном slide
относится к **одному GroupedGEMM projection**:

$$
2SkHI\approx1.92\,\mathrm{TFLOP}\Rightarrow2.4\,\mathrm{ms}.
$$

TP режет каждую expert matrix и требует collectives вокруг projections; EP
оставляет experts целыми и делает dispatch/combine All-to-All. Выбор зависит
от того, что дороже на реальной topology: repeated activation collectives,
weight traffic или token All-to-All.

## PP schedules: memory, bubble и коммуникации

PP делит $L$ слоёв между $p$ stages. Payload границы microbatch —
приблизительно $bNH$ bytes forward и столько же gradients backward.

### GPipe и 1F1B

```text
GPipe, p=2, m=4
stage 0: F1 F2 F3 F4 .. .. B4 B3 B2 B1
stage 1: .. F1 F2 F3 F4 B4 B3 B2 B1 ..

1F1B после warmup
stage 0: F1 F2 B1 F3 B2 F4 B3 .. B4
stage 1: .. F1 B1 F2 B2 F3 B3 F4 B4
```

GPipe держит активации всех $m$ microbatches; 1F1B ограничивает число живых
microbatches и раньше освобождает память. Грубая bubble fraction
$(p-1)/(m+p-1)$: при $p=8,m=32$ это $7/39\approx18\%$.

### ZeroBubble

Backward содержит input-gradient ($B_x$), нужный предыдущему stage, и
weight-gradient ($B_w$), который можно отложить. ZeroBubble (EDLS slide 142)
ставит $B_w$ в idle slots:

```text
обычно:     [ F ][idle][ Bx+Bw ][idle]
ZeroBubble: [ F ][ Bx ][ Bw from another microbatch ]
```

Это meaningful schedule, а не обещание нулевого overhead: дробление меняет
dependencies, peak gradients и число kernel launches.

### DualPipeV

DualPipeV (slide 143) ведёт microbatches по V-shaped placement в двух
направлениях. Пока один поток ждёт MoE All-to-All, другой выполняет local
compute:

```text
stream A: stage 0 → 1 → 2 → 3
                              ↘
stream B: stage 0 ← 1 ← 2 ← 3
compute : [GEMM A][GEMM B][GEMM A]
network :    [A2A A]   [A2A B]
```

Польза появляется, если streams действительно используют разные engine/link
ресурсы и memory хватает на два набора in-flight activations.

## Как собирать hardware/topology map

Перед выбором FSDP/TP/EP/PP заполняют карту:

| Уровень | Измеряем | Обычно размещаем |
|---|---|---|
| внутри SM/GPU | Tensor Core FLOP/s, HBM BW, workspace | fusion, GroupedGEMM |
| GPU↔GPU одного узла | NVLink/NVSwitch BW и contention | TP, иногда EP |
| узел↔узел | NIC/rail BW, latency, oversubscription | PP, FSDP/EP с overlap |
| CPU/storage→GPU | decode/pin/copy time | asynchronous data pipeline |

Затем для каждого кандидата оценивают
$T\ge\max(F/R_{\text{eff}},D_{\text{HBM}}/BW_{\text{HBM}},
D_{\text{net}}/BW_{\text{net}})+L$ и проверяют trace. Среднее
$P/p$ не заменяет peak ledger, а паспортный bandwidth — effective collective
bandwidth.

## Контрольный расчёт перед запуском

1. Сверить parameter count с model config.
2. Выписать persistent, gathered, activation и workspace peaks по времени.
3. Разделить FLOP на projections, $S^2$ attention и elementwise work.
4. Для MoE отдельно считать stored experts, active $k$, dispatch bytes и
   imbalance.
5. Для каждого collective записать tensor shape, participants, topology и
   возможность overlap.
6. Нарисовать microbatch schedule и посчитать bubble/live activations.
7. Проверить расчёт profiler trace и allocator snapshot.

## Источники и статус переноса

- [EDLS week 6, complete lecture, slides 4–146](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf) —
  основной источник структуры, model maps и численных примеров. Формулы выше
  адаптированы и дополнительно выведены; slide-specific числа явно помечены.
- [Harvard CS249r, Neural Computation, “Matrix Operations”](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/nn_computation/nn_computation.qmd) —
  связь MAC/GEMM и системного bottleneck.
- [Harvard CS249r, Model Training, “Memory Architecture”](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/training/training.qmd) —
  optimizer-state ledger и зарегистрированный рисунок.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Измерение производительности и roofline]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Численные форматы и mixed precision]] →
