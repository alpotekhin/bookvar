---
title: Арифметика Transformer и MoE
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# Арифметика Transformer и MoE

Запись «модель содержит 70 миллиардов параметров» почти ничего не говорит о
том, можно ли её обучить на данном кластере. Параметры нужно хранить, перед
вычислением слоя — доставлять к ускорителю, для обратного прохода — сохранять
или пересчитывать активации, а градиенты — передавать между устройствами. У
разреженной MoE-модели к этому добавляется пересылка представлений токенов между
экспертами. Поэтому архитектуру полезно сразу переводить в четыре отдельные
величины: число операций, объём памяти, сетевой обмен и время жизни тензоров.

Такой расчёт ниже выполняется последовательно: сначала для одного плотного слоя,
затем для полной модели и состояний оптимизатора, после чего добавляются
шардинг, конвейерный параллелизм и маршрутизация MoE.

Обозначения: $B$ — число sequences, $S$ — длина, $N=BS$ — число токенов
microbatch, $H$ — hidden width, $I$ — FFN width, $L$ — слои, $V$ — vocabulary,
$n_h,n_{kv}$ — query- и KV-heads, $d$ — head width, $E$ — experts, $k$ —
top-$k$. Один BF16-элемент занимает $b=2$ bytes.

## Нулевая стадия: данные тоже входят в step time

До первого GEMM система уже может терять время: CPU читает и готовит batch, пока
GPU ждёт. Для синхронного шага

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

На 800 TFLOP/s идеальный прямой проход через слой занял бы 4,15 мс. Это лишь
нижняя граница: обращения к памяти, поэлементные ядра, паузы между их запусками
и неполная загрузка вычислительных блоков увеличивают фактическое время.

## Учёт активаций: где появляется $O(LNH)$

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
- Selective checkpointing выбирает, какие активации хранить, с учётом стоимости
  их повторного вычисления. Результаты дорогих GEMM обычно выгоднее сохранить,
  а выходы сравнительно дешёвых нормализаций и функций активации — вычислить
  заново. Выбор зависит от доступной памяти и профиля конкретной модели.

Для $B=1,S=8192,H=4096,I=11008$ один residual — 67,1 MB, а два MLP projections
— около 360,7 MB. Полная attention matrix при $n_h=32$ — 4,29 GB на один
тензор. Это объясняет, почему FlashAttention меняет сам порядок памяти, а
checkpointing — множитель и recompute time.

## Состояния модели и FSDP

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
Architecture”/optimizer-state accounting; [исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/training/images/svg/training_optimizer_memory.svg),
CC BY-NC-SA 4.0. Рисунок импортирован без изменений.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/distributed_training_memory_budget.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Distributed Training,
§ “Hybrid parallelism memory budget”, locator
`sec-distributed-training-systems-systems-hybrid-parallelism`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/images/svg/distributed_training_memory_budget.svg),
CC BY-NC-SA 4.0; файл не изменён. Общий ledger разделён на
weights, optimizer и activations, чтобы sharding не смешивался с peak buffers.*

FSDP шарит параметры, gradients и optimizer state между $p$ ranks. Идеальный
resident ledger становится примерно $(16\text{-}18)P/p$, но перед вычислением
слоя нужен AllGather weights, после backward — ReduceScatter gradients.
Поэтому FSDP следует понимать как управление перемещением и временем жизни
состояний модели, а не как бесплатное сокращение памяти.

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

Слишком ранняя предварительная загрузка удерживает несколько буферов с полными
параметрами; слишком поздняя заставляет GPU ждать NCCL. Коллективные операции
к тому же конкурируют за одни и те же линии связи, поэтому их потоки и
зависимости должны быть согласованы с вычислениями.

## Объединение операций и Liger Kernel

После больших GEMM остаются RMSNorm, RoPE, SwiGLU, dropout, cross-entropy и
optimizer updates. У каждого может быть малая arithmetic intensity:
прочитать tensor, сделать несколько операций, записать обратно. Несколько
отдельных kernels многократно проходят HBM.

Fusion выполняет цепочку в одном kernel и держит промежуточные значения в
register/shared memory. Если три elementwise стадии читают и пишут tensor
размера $M$, независимое исполнение перемещает примерно $6M$ bytes, fused —
около $2M$; теоретическое bandwidth speedup до $3\times$, но register pressure
и occupancy могут его уменьшить. Готовые реализации таких операций, включая
экономную по памяти функцию потерь, есть в Liger Kernel. Возможность применить
конкретное ядро зависит от версии библиотеки, формы тензоров и типов данных;
это проверяют по трассировке и тестам, а не предполагают заранее.

## Расчёт для плотных моделей 7B и 70B

При расчёте памяти для моделей от 100M до Llama 70B и Qwen 235B-A32B важно
различать четыре категории:

1. persistent weights/optimizer states;
2. layer-local full weights после FSDP gather;
3. saved activations;
4. temporary communication и GEMM workspaces.

Для Llama 70B ($H=8192,I=28672,L=80$, GQA $n_{kv}=8$) FFN одного слоя:

$$P_{\text{ffn}}=3\cdot8192\cdot28672\approx704{,}6\text{ M},$$

то есть 1,41 GB в BF16. Даже если
persistent state разделён на 1024 GPU, один gathered layer всё ещё требует
гигабайты; поэтому peak определяется schedule gather/free, а не средним
$P/p$. При TP=2 матрицы и вычисления делятся пополам, но появляются
коллективные обмены активациями.

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/moe-all-to-all-routing.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Distributed Training,
§ “Expert parallelism (mixture of experts)”, locator
`sec-distributed-training-systems-systems-expert-parallelism-mixture-experts-bc45`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/images/svg/moe-all-to-all-routing.svg),
CC BY-NC-SA 4.0; файл не изменён. Dispatch и combine
показаны как две отдельные All-to-All фазы вокруг локального expert compute.*

### Qwen 235B-A32B: почему FSDP становится дорогим

В использованном в курсе расчёте для Qwen 3 235B один слой занимает около
5 GB в BF16. Разреженные вычисления используют лишь $k$ экспертов, но обычный
FSDP AllGather перемещает веса всех экспертов. При 200 GB/s перенос 5 GB имеет
нижнюю границу 25 ms ещё до учёта накладных расходов протокола и топологии. EP
размещает экспертов постоянно и вместо весов пересылает представления токенов.

### DeepSeek-V3 671B

Для приведённой конфигурации DeepSeek-V3 аналогичная оценка даёт около 21 GB
в BF16 на слой. Это не универсальная характеристика любой реализации. Даже
пропускная способность 1 TB/s дала бы нижнюю границу 21 ms на
полный перенос слоя; вычисления только по активным экспертам могут завершиться
быстрее. Отсюда необходимость EP/PP и размещения с учётом топологии сети.

### TP=8 и EP=8: численный пример

Пусть $S=8192,E=256,k=8,H=7168,I=2048$.
Полный объём вычислений SwiGLU для одного пакета токенов:

$$
F_{\mathrm{SwiGLU}}=6SkHI
\approx 6\cdot8192\cdot8\cdot7168\cdot2048
\approx 5.77\,\mathrm{TFLOP}.
$$

На 800 TFLOP/s идеал — 7,2 ms. Важно не спутать эту оценку с 2,4–2,5 ms для
**одной проекции GroupedGEMM**:

$$
2SkHI\approx1.92\,\mathrm{TFLOP}\Rightarrow2.4\,\mathrm{ms}.
$$

TP режет каждую expert matrix и требует collectives вокруг projections; EP
оставляет experts целыми и делает dispatch/combine All-to-All. Выбор зависит
от того, что дороже на реальной topology: repeated activation collectives,
weight traffic или token All-to-All.

## Расписания конвейера: память, простой и коммуникации

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/pipeline-parallelism.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Distributed Training,
§ “Pipeline parallelism”, locator
`sec-distributed-training-systems-systems-pipeline-parallelism-8748`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/images/svg/pipeline-parallelism.svg),
CC BY-NC-SA 4.0; файл не изменён. Пространственное деление
слоёв по stages связывает расписание microbatches с передачей activations.*

### ZeroBubble

Backward содержит input-gradient ($B_x$), нужный предыдущему stage, и
weight-gradient ($B_w$), который можно отложить. ZeroBubble помещает $B_w$ в
периоды простоя:

```text
обычно:     [ F ][idle][ Bx+Bw ][idle]
ZeroBubble: [ F ][ Bx ][ Bw from another microbatch ]
```

Название ZeroBubble не означает отсутствия любых накладных расходов: разделение
обратного прохода меняет зависимости, пиковый объём градиентов и число запусков
ядер.

### DualPipeV

DualPipeV ведёт микробатчи по V-образному размещению в двух направлениях. Пока
один поток ждёт MoE All-to-All, другой выполняет локальные вычисления:

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

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/lecture.pdf|EDLS Week 6 — лекция по арифметике глубокого обучения]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/seminar/practice.ipynb|EDLS Week 6 — семинарская тетрадь]]: профилирование, объединение операций и память на исполняемых примерах.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/homework/README|EDLS Week 6 — практическое задание]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/nn_computation|Harvard CS249r — Neural Computation]] и [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/model_compression|Model Compression]].

- [EDLS week 6, slides 4–146](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf) —
  источник структуры, model maps и численных примеров; числа, относящиеся к
  конкретным слайдам, явно помечены.
- [Harvard CS249r, Neural Computation, “Matrix Operations”](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/nn_computation/nn_computation.qmd) —
  связь MAC/GEMM и системного bottleneck.
- [Harvard CS249r, Model Training, “Memory Architecture”](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/training/training.qmd) —
  optimizer-state ledger и зарегистрированный рисунок.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Измерение производительности и roofline]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Численные форматы и mixed precision]] →
