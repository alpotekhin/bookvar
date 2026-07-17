---
title: От SwiGLU к Mixture of Experts
type: textbook-chapter
status: canonical
last_updated: 2026-07-17
previous: "[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache]]"
next: "[[02 Areas/ML & DL/00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока]]"
primary_sources:
  - https://arxiv.org/abs/2101.03961
  - https://arxiv.org/abs/2401.04088
  - https://arxiv.org/abs/2401.06066
  - https://arxiv.org/abs/2412.19437
---

# От одного FFN к Mixture of Experts

> [!info] Карта углублённого модуля
> План отдельных уроков, практики, визуалов и первичных источников:
> [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/00 Карта модуля и источники]].

> [!abstract] Что нужно понять
> В dense Transformer каждый token проходит через один и тот же FFN. Sparse MoE
> хранит много FFN и выбирает для каждого token только $k$ из них. Это позволяет
> увеличивать общее число параметров быстрее, чем FLOPs на token. Но веса всё
> равно нужно хранить, tokens нужно пересылать к нужным experts, а router —
> балансировать.

В Switch Transformer attention остаётся dense, а обычный FFN заменяется набором
experts и router. Switch выбирает top-1; другие MoE часто используют top-2 или
больше. Ниже это показано на более читаемых схемах Mixtral и DeepSeekMoE.

## Почему MoE находится именно на месте FFN

Transformer block чередует:

1. attention — коммуникацию между positions;
2. FFN — независимое преобразование каждого position state.

Для SwiGLU:

$$
E(x)=
\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_d.
$$

Эта функция применяется к каждому token отдельно. Значительная доля параметров
LLM находится именно в FFN. Поэтому естественный способ добавить capacity —
создать несколько вариантов $E_1,\ldots,E_N$, не дублируя attention.

```text
dense block:
token → attention → one shared FFN → residual

MoE block:
token → attention → router → selected FFN experts → combine → residual
```

## Dense FFN: все токены используют все веса

Пусть FFN содержит приблизительно $P_E$ параметров. Для batch из $M$ token
states одна и та же function вычисляется $M$ раз:

$$
y_t=E(x_t).
$$

Увеличение hidden width повышает и total parameters, и compute каждого token.
В dense model эти величины связаны.

MoE разрывает связь: total capacity растёт с числом experts $N$, а active
compute — с числом выбранных experts $k$.

## Router: маленькая сеть принимает большое решение

Для token state $x_t$ router вычисляет logits:

$$
z_t=W_rx_t,\qquad z_t\in\mathbb R^N.
$$

Затем:

$$
p_t=\operatorname{softmax}(z_t).
$$

Выбираются $k$ наибольших scores:

$$
G_t=\operatorname{TopK}(p_t,k).
$$

Output:

$$
y_t=\sum_{i\in G_t}\alpha_{t,i}E_i(x_t),
$$

где $\alpha$ — выбранные routing weights, часто перенормированные внутри top-k.

### Пример top-2

```text
router scores: [0.04, 0.61, 0.08, 0.27]
selected:          E₂             E₄

y = 0.69 · E₂(x) + 0.31 · E₄(x)   # после renormalization
```

На следующем layer тот же token может выбрать другую пару. Routing — решение
для token × layer, а не постоянное назначение всего предложения одному expert.

![[02 Areas/ML & DL/raw/papers/mixtral-of-experts/images/sparse-moe-routing.png]]

*Визуализация sparse routing в Mixtral: разные tokens направляются разным
подмножествам FFN, после чего возвращаются в исходный sequence order.*

## Total parameters и active parameters

Если shared dense часть содержит $P_D$, а каждый из $N$ experts — $P_E$:

$$
P_{\text{total}}\approx P_D+NP_E.
$$

При top-$k$ для одного token:

$$
P_{\text{active}}\approx P_D+kP_E.
$$

Это объясняет формулировки вида «671B total, 37B activated». Но active parameters
не равны ни точному FLOPs, ни VRAM:

- все weights обычно должны быть размещены в aggregate device memory;
- attention остаётся active для всех tokens;
- router, dispatch, combine и communication имеют стоимость;
- часть parameters может быть embeddings или shared experts.

> [!warning] Нельзя делить total на active
> Модель 8×7B не является «7B, которая иногда меняется». Общие параметры не
> умножаются на восемь, experts имеют свою структуру, а top-2 активирует больше
> одного FFN.

## Mixtral: понятный top-2 пример

Mixtral 8×7B заменяет каждый FFN восемью experts и выбирает два на token.

![[02 Areas/ML & DL/raw/papers/moe/images/mixtral-smoe.png]]

*Схема sparse MoE layer Mixtral: attention общий, router выбирает два expert
FFN. На каждом layer выбор выполняется заново.*

Paper сообщает около 47B total parameters и около 13B active parameters на
token. Название «8×7B» не следует интерпретировать как простую сумму восьми
полных независимых 7B models.

## Почему experts не обязаны быть «математиком» и «переводчиком»

Expert — обычный FFN. Ему заранее не присваивается semantic role. Router и expert
weights совместно обучаются по language-model loss.

После обучения можно измерять:

- какие tokens чаще идут к expert;
- preference по языкам, syntax или domains;
- routing entropy;
- overlap выбранных experts;
- стабильность routing между layers.

![[02 Areas/ML & DL/raw/papers/mixtral-of-experts/images/expert-routing-analysis.png]]

*Пример анализа routing из материалов Mixtral. Наблюдаемая специализация —
эмпирический результат, а не гарантированная человечески понятная профессия.*

Часто нижние layers маршрутизируют по поверхностным признакам, а более высокие —
по сложным combinations. Некоторые experts остаются generalists.

## Главная training problem: collapse routing

Без ограничений router может предпочесть несколько experts:

```text
E1: ████████████████████  72%
E2: █████                18%
E3: ██                    7%
E4: █                     3%
```

Тогда:

- популярный device перегружен;
- остальные experts почти не обучаются;
- batch ждёт самый медленный route;
- effective capacity оказывается намного меньше total.

Это одновременно optimization и distributed-systems problem.

## Auxiliary load-balancing loss

Один из подходов учитывает:

- $f_i$ — долю tokens, отправленных expert $i$;
- $P_i$ — среднюю routing probability expert $i$.

Упрощённая форма Switch loss:

$$
\mathcal L_{\text{aux}}
=\alpha N\sum_{i=1}^{N}f_iP_i.
$$

Loss поощряет более равномерное использование. Коэффициент $\alpha$ задаёт
trade-off: слишком слабый не предотвращает collapse, слишком сильный может
заставлять router выбирать expert ради баланса, а не качества token.

Итоговый objective:

$$
\mathcal L=\mathcal L_{\text{LM}}+\mathcal L_{\text{aux}}.
$$

## Capacity factor и token dropping

Expert не может принять неограниченное число tokens в статически выделенном
buffer. При $M$ tokens, $N$ experts и top-1 средняя нагрузка:

$$
\frac{M}{N}.
$$

Capacity задают как:

$$
C=\left\lceil
\text{capacity factor}\cdot\frac{Mk}{N}
\right\rceil.
$$

Если expert получает больше $C$ assignments, overflow tokens приходится:

- отбросить из expert path;
- отправить резервному expert;
- обработать позднее;
- допустить dynamic shapes и дополнительную communication.

Большой capacity factor уменьшает dropping, но увеличивает padding и
неиспользуемую память.

## Expert parallelism: где возникает all-to-all

Experts распределяются по devices:

```text
GPU 0: E0 E1       GPU 1: E2 E3
GPU 2: E4 E5       GPU 3: E6 E7
```

После router tokens физически находятся не там, где их expert weights:

1. tokens группируются по destination expert;
2. all-to-all пересылает activations;
3. local experts выполняют FFN;
4. второй all-to-all возвращает outputs;
5. outputs восстанавливаются в исходный order и смешиваются.

```text
original batch
 [a b c d e f]
       ↓ dispatch / all-to-all
 GPU0 [a e]   GPU1 [c]   GPU2 [b f]   GPU3 [d]
       ↓ expert FFN
       ↓ combine / all-to-all
 [a' b' c' d' e' f']
```

Поэтому MoE выгоден, когда большие expert GEMMs перекрывают communication.
При маленьком batch или низкой latency overhead может доминировать.

## Switch, Mixtral и DeepSeekMoE — разные designs

### Switch Transformer

- top-1 routing;
- цель — упростить routing и communication;
- auxiliary balancing и capacity;
- показал scaling до очень большого числа parameters.

### Mixtral

- восемь experts;
- top-2 на каждом token/layer;
- простой и наглядный sparse MoE decoder.

### DeepSeekMoE

DeepSeek предлагает две идеи:

1. fine-grained experts: крупные experts дробятся на более мелкие, выбирается
   большее число; комбинации становятся гибче;
2. shared experts: часть FFN всегда active и хранит общие знания, routed experts
   меньше дублируют general patterns.

```text
token ─┬→ shared expert(s) ─────────────┐
       └→ router → top-k routed experts ├→ sum
                                       ┘
```

![[02 Areas/ML & DL/raw/papers/deepseek-v2/images/x3.png]]

*Схема DeepSeekMoE из DeepSeek-V2: shared experts идут по постоянному пути,
routed experts выбираются router.*

## Auxiliary-loss-free balancing в DeepSeek-V3

DeepSeek-V3 корректирует expert selection с помощью per-expert bias,
обновляемого по наблюдаемой нагрузке. Bias участвует в выборе route, но не
должен искажать основную gating weight при смешивании outputs.

Интуитивно:

```text
expert перегружен  → selection bias немного уменьшается
expert недогружен  → selection bias немного увеличивается
```

Это позволяет балансировать devices без сильного auxiliary objective,
конфликтующего с language-model loss. В report всё же используется небольшой
sequence-wise balance loss для ограничения экстремальных случаев; поэтому
«совсем без balancing loss» — слишком грубое описание.

## Qwen и отсутствие единого канона

Современные Qwen MoE variants показывают, что choices продолжают меняться:
число experts, top-k, shared experts и способ balancing нельзя переносить с
DeepSeek автоматически.

MoE следует описывать набором полей:

```yaml
num_experts:
experts_per_token:
shared_experts:
expert_hidden_size:
routing_score:
normalization:
capacity_or_dropping:
load_balancing:
expert_parallel_layout:
```

Название семейства без этих полей недостаточно.

## Что происходит с gradients через top-k

Top-k selection дискретна: experts вне выбранного множества не получают gradient
от данного token. Выбранные experts и их routing weights обучаются обычным
backpropagation. Router logits получают signal через weights выбранных routes и
balancing terms.

Это создаёт feedback loop:

```text
expert немного лучше для token
→ router выбирает его чаще
→ expert получает больше training examples
→ может стать ещё сильнее и популярнее
```

Balancing и routing noise помогают не дать ранней случайности захватить систему.

## Training и serving видят разные bottlenecks

### Training

- all-to-all между expert-parallel ranks;
- imbalance внутри global batch;
- activation memory;
- numerical stability router;
- достаточный token batch для каждого expert.

### Serving

- все expert weights должны поместиться в cluster memory;
- request batch может быть мал и несбалансирован;
- latency определяется самым загруженным expert/device;
- weight loading и quantization experts важнее raw FLOPs;
- tensor/expert/data parallel strategies взаимодействуют.

Sparse compute не означает sparse storage.

## Dense или MoE

| Критерий | Dense | Sparse MoE |
|---|---|---|
| parameters на token | все | часть expert parameters |
| implementation | проще | router + dispatch + combine |
| memory weights | ниже при равном active compute | выше |
| communication | обычный parallelism | дополнительный all-to-all |
| small-batch latency | предсказуемее | overhead может мешать |
| capacity scaling | вместе с FLOPs | быстрее FLOPs |
| training stability | проще | требует balancing |

MoE особенно привлекателен для крупного training/serving cluster и высокого
throughput. Dense model может быть удобнее на одном устройстве или при жёсткой
tail latency.

## Ошибки, которые нужно видеть

> [!danger] «64 experts, top-2 — значит compute в 32 раза меньше»
> Сравнивать нужно с dense baseline того же FFN width; attention и shared paths
> остаются, communication добавляется.

> [!danger] «Expert специализируется по теме»
> Это гипотеза для измерения, а не архитектурное обещание.

> [!danger] «Неактивные weights не занимают VRAM»
> Они не участвуют в текущем token compute, но должны быть доступны runtime.

> [!danger] «Balanced token counts означают одинаковое время»
> Devices могут получать разные shapes, network routes и kernel efficiency;
> важен end-to-end step time.

> [!danger] «DeepSeek — это просто MoE»
> MLA, precision, pipeline schedule, objectives и post-training — отдельные
> части системы.

## Практика

### 1. Toy router

```python
scores = torch.softmax(x @ router_weight, dim=-1)
weights, experts = torch.topk(scores, k=2, dim=-1)
weights = weights / weights.sum(dim=-1, keepdim=True)
```

Посчитайте histogram assignments. Добавьте небольшой bias одному expert и
наблюдайте, как быстро возникает imbalance.

### 2. Capacity

Для 4096 tokens, 8 experts и top-2 вычислите mean assignments. Сравните capacity
factors 1.0, 1.25 и 2.0: сколько slots выделяется и сколько padding останется
при идеально равном routing.

### 3. Отличить total от active

Возьмите config MoE model и разложите parameters на:

- embeddings;
- attention;
- norms;
- routed experts;
- shared experts;
- output head.

Не используйте marketing name как арифметику.

### 4. Наблюдать routing

Для маленькой открытой MoE сохраните top-k expert ids по token и layer.
Постройте:

- load histogram;
- entropy;
- heatmap layer × expert;
- examples tokens с максимальной preference.

Не называйте pattern специализацией, пока он не воспроизводится на независимой
выборке.

## После главы нужно уметь

- вывести router и weighted top-k output;
- различать total, active parameters и FLOPs;
- объяснить capacity factor и token overflow;
- нарисовать два all-to-all этапа;
- сравнить Switch, Mixtral и DeepSeekMoE;
- объяснить shared/fine-grained experts;
- отделить load balancing от semantic specialization.

## Материалы, на которых построена глава

### Основные papers

- [Fedus et al. — Switch Transformers](https://arxiv.org/abs/2101.03961).
- [Jiang et al. — Mixtral of Experts](https://arxiv.org/abs/2401.04088).
- [Dai et al. — DeepSeekMoE](https://arxiv.org/abs/2401.06066).
- [DeepSeek-AI — DeepSeek-V2](https://arxiv.org/abs/2405.04434).
- [DeepSeek-AI — DeepSeek-V3](https://arxiv.org/abs/2412.19437).

### Figures и связанные заметки

- Локальные figures извлечены из Switch, Mixtral и DeepSeek reports; captions
  выше указывают смысл и происхождение, а papers остаются источником контекста.
- [[02 Areas/ML & DL/01 Справочник/FFN и MoE/Mixture of Experts]]
- [[02 Areas/ML & DL/01 Справочник/FFN и MoE/SwiGLU]]

**Дальше:** [[02 Areas/ML & DL/00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока|соберём Llama, Qwen, DeepSeek, GLM и Kimi в единый атлас без смешения architecture, training и post-training.]]
