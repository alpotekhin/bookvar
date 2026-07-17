---
title: SSM, recurrent и hybrid alternatives
type: textbook-chapter
status: canonical
last_updated: 2026-07-17
previous: "[[02 Areas/ML & DL/00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока]]"
next: "[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training]]"
primary_sources:
  - https://arxiv.org/abs/2312.00752
  - https://arxiv.org/abs/2305.13048
  - https://arxiv.org/abs/2307.08621
  - https://arxiv.org/abs/2403.19887
---

# Что вместо полного attention: Mamba, RWKV, RetNet и гибриды

> [!abstract] Что нужно понять
> Transformer хранит K/V каждого прошлого token и может адресоваться к ним по
> содержанию. Recurrent и state-space модели обновляют состояние фиксированного
> размера. Их decode дешевле по памяти, но история проходит через bottleneck.
> Mamba делает state update зависимым от input, RWKV и RetNet дают параллельную
> training form и recurrent inference form, а hybrid models оставляют часть
> attention layers для точного recall.

## Два способа помнить прошлое

### Attention: архив с произвольным доступом

Для нового query:

$$
y_t=\sum_{j\le t}
\operatorname{softmax}_j(q_t^\top k_j)v_j.
$$

Прошлые $k_j,v_j$ хранятся явно. Query текущего token решает, какую запись
прочитать.

### Recurrence: постоянно переписываемое резюме

$$
h_t=f(h_{t-1},x_t),\qquad y_t=g(h_t).
$$

После обработки token $t$ всё прошлое доступно только через $h_t$. Память decode
не растёт с context length, но потерянную из state деталь нельзя прочитать
заново.

```text
attention:  x₁ x₂ x₃ ... xₜ  → хранить отдельные K/V
recurrent:  h₀ → h₁ → h₂ → ... → hₜ  → хранить hₜ
```

Это главный trade-off. «Linear time» само по себе не говорит, насколько хорошо
модель выполняет associative recall.

## От динамической системы к SSM

Непрерывная linear state-space model:

$$
\frac{dh(t)}{dt}=Ah(t)+Bx(t),
$$

$$
y(t)=Ch(t)+Dx(t).
$$

После discretization:

$$
h_t=\bar Ah_{t-1}+\bar Bx_t,
$$

$$
y_t=Ch_t+Dx_t.
$$

$A$ определяет, как прошлое затухает/распространяется; $B$ записывает input в
state; $C$ читает state; $D$ даёт прямой skip от input.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mamba/ssm-three-views.png]]

*Figure из Mamba materials: один linear time-invariant SSM можно вычислять как
recurrence, convolution или structured scan. Формы математически связаны, но
удобны в разных режимах.*

## Почему convolution позволяла parallel training

Разворачивая recurrence:

$$
h_t=\bar A^t h_0+
\sum_{j=1}^{t}\bar A^{t-j}\bar Bx_j.
$$

При $h_0=0$:

$$
y_t=\sum_{j=1}^{t}C\bar A^{t-j}\bar Bx_j.
$$

Ядро:

$$
K_\tau=C\bar A^\tau\bar B
$$

зависит от расстояния $\tau$, а не content. Тогда весь sequence можно обработать
convolution. Это объясняет, как SSM совмещает recurrent interpretation с
parallel sequence computation.

Но content-independent kernel плохо решает задачу «запомни значение после
специального marker и игнорируй шум». Нужна selectivity.

## Mamba: параметры записи и чтения зависят от input

Mamba делает $\Delta$, $B$ и $C$ функциями текущего token:

$$
\Delta_t=s_\Delta(x_t),\qquad
B_t=s_B(x_t),\qquad
C_t=s_C(x_t).
$$

Теперь модель может:

- увеличить update для важного token;
- почти не менять state на filler token;
- читать разные компоненты state в зависимости от input.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mamba/selective-ssm.png]]

*Ключевая иллюстрация Mamba paper: selection mechanism делает SSM
input-dependent. Цветом показано, что разные tokens по-разному проходят через
state.*

### Сквозной пример

```text
Код доступа: 7319. ... много текста ... Какой код доступа?
```

Идеальное поведение:

```text
«Код доступа:» → открыть запись
7319            → сильно обновить state
filler tokens   → сохранить нужную компоненту
«Какой код?»    → прочитать её
```

Attention может найти `7319` прямым similarity lookup. Selective SSM должна
записать число в state и защитить его от последующих обновлений. Это разные
вычислительные пути к ответу.

## Почему Mamba не вычисляется простой convolution

Когда $B_t,C_t,\Delta_t$ зависят от input, kernel уже не time-invariant. Нельзя
заранее построить одну convolution для всех examples.

Mamba использует hardware-aware selective scan:

- сохраняет линейную работу по sequence;
- параллелит associative structure scan;
- минимизирует перенос большого expanded state в HBM;
- fusion/recomputation становятся частью алгоритма, а не косметической
  оптимизацией.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mamba/mamba-block.png]]

*Mamba block из paper: input projection разветвляется, short convolution
обрабатывает локальный pattern, selective SSM — длинное состояние, gate смешивает
результат.*

## Shapes и стоимость

У causal Transformer:

- prefill attention scores имеют $O(T^2)$ pairs;
- decode хранит KV-cache $O(T)$;
- один новый token читает весь cache.

У recurrent SSM:

- work по sequence $O(T)$;
- decode state не растёт с $T$;
- один новый token выполняет fixed-size update.

Но constants важны: state expansion, scan kernel, convolution и projections
могут сделать короткие sequences медленнее хорошо оптимизированного attention.

## RWKV: time-mixing как recurrent linear attention

RWKV расшифровывается как Receptance Weighted Key Value. Architecture сочетает:

- time-mixing между positions;
- channel-mixing внутри token;
- parallel training form;
- recurrent inference form.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/rwkv-figure3-hq.png]]

*RWKV architecture из paper. Важно видеть не «Transformer без attention», а
двойную форму одного computation: sequence-parallel при training и stateful при
decode.*

Упрощённая идея weighted memory:

$$
s_t=\lambda\odot s_{t-1}+\phi(k_t,v_t),
$$

$$
y_t=r_t\odot \operatorname{read}(s_t),
$$

где receptance $r_t$ играет роль gate. Реальные версии RWKV различаются, поэтому
эта формула — intuition, а не checkpoint specification.

## RetNet: retention в трёх формах

RetNet строит retention mechanism, который можно вычислять:

1. parallel — для training;
2. recurrent — для token-by-token decode;
3. chunkwise recurrent — компромисс для длинных sequences.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/retnet-figure3-hq.png]]

*Dual form RetNet: один mechanism имеет параллельное и recurrent представления.*

Chunkwise form особенно полезна как system idea:

```text
внутри chunk → parallel matrix operations
между chunks → recurrent state
```

Это снимает ложную дихотомию «либо Transformer parallelism, либо RNN
recurrence».

## Mamba-2 и state-space duality

Mamba-2/SSD показывает более глубокую связь structured state-space models и
некоторых attention-like matrix transformations. Для учебной траектории важно:

- Mamba-1 вводит selectivity и hardware-aware scan;
- Mamba-2 упрощает/реорганизует block через state-space duality;
- поздние Mamba versions нельзя считать тем же точным block с новой цифрой.

Актуальный официальный index:
[state-spaces/mamba](https://github.com/state-spaces/mamba). На 2026 год он уже
содержит ссылки и на последующие поколения; каждое требует отдельной release
card в атласе.

## Hybrid models: не выбирать крайность

Pure SSM эффективна, но attention особенно хорошо выполняет content-addressable
recall. Гибрид чередует layers:

```text
Mamba → Mamba → Attention → Mamba → Mamba → Attention → ...
```

### Jamba

Jamba объединяет:

- Transformer attention layers;
- Mamba layers;
- MoE в части layers.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/jamba-figure1-hq.png]]

*Jamba architecture: три независимые оси — sequence mixer, MoE и layer ratio.
Нельзя описать её одним словом «Mamba».*

Attention layers дают прямой retrieval path, Mamba layers снижают среднюю цену
длинного context, MoE увеличивает parameter capacity.

### Что нужно указывать для hybrid

```yaml
attention_layer_ratio:
attention_type:
ssm_type:
state_size:
local_convolution:
moe_layer_ratio:
experts_per_token:
cache:
  kv_per_attention_layer:
  recurrent_state_per_ssm_layer:
```

Два hybrids с одинаковым названием classes могут иметь совершенно разный memory
profile.

## Честное сравнение памяти

| Model type | Persistent decode memory |
|---|---|
| full Transformer | KV всех layers × context |
| GQA Transformer | уменьшенный KV × context |
| pure recurrent/SSM | fixed state всех layers |
| hybrid | KV attention layers × context + fixed SSM state |

У hybrid KV-cache всё равно растёт, но только для attention layers. Это
количественная, а не бинарная экономия.

## Где recurrent compression проигрывает

Типичные stress tests:

- exact copying;
- needle-in-a-haystack;
- induction heads / повтор pattern;
- retrieval нескольких далёких facts;
- восстановление редкой детали после длинного distractor.

Высокий language-model benchmark не гарантирует точный long-context recall.
Нужны tests, соответствующие application.

## Где SSM особенно естественна

- бесконечные/длинные streams;
- audio и signals;
- genomics;
- edge inference с жёсткой memory;
- high-throughput generation;
- задачи, где полезное прошлое действительно можно сжать в state.

Mamba paper оценивает несколько modalities именно потому, что inductive bias
последовательной динамики шире текста.

## Multimodality — отдельная ось

SSM, attention и recurrence отвечают за sequence mixing. Multimodal model
добавляет способ представить image/audio/video:

- внешний encoder + projector;
- cross-attention;
- early-fusion modality tokens;
- modality-specific output decoder.

Можно построить multimodal Transformer, Mamba или hybrid. Поэтому подробный
разбор вынесен в
[[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/01 Vision-language и omni models]],
а в названии этой главы multimodality больше не используется как синоним
«альтернативы Transformer».

## Ошибки, которые нужно видеть

> [!danger] «SSM помнит бесконечный context»
> State имеет fixed size. Обработать stream можно, но сохранить в нём все детали
> без потерь нельзя.

> [!danger] «Linear time означает быстрее всегда»
> Для коротких sequences constants и зрелость kernels могут победить asymptotic
> advantage.

> [!danger] «Parallel training и recurrent inference — две модели»
> Это две эквивалентные или согласованные формы одного набора weights.

> [!danger] «Mamba — просто RNN»
> Recurrent inference — часть картины; structured parameterization, selectivity
> и hardware-aware scan определяют практическую architecture.

> [!danger] «Hybrid устраняет KV-cache»
> Cache остаётся у attention layers.

## Практика

### 1. Развернуть recurrence

Для scalar:

$$
h_t=0.9h_{t-1}+x_t,\qquad y_t=h_t
$$

вычислите первые четыре шага вручную и покажите convolution kernel:

$$
[1,\ 0.9,\ 0.9^2,\ 0.9^3].
$$

### 2. Добавить selectivity

Пусть gate $g_t\in[0,1]$:

$$
h_t=(1-g_t)h_{t-1}+g_tx_t.
$$

Для sequence с marker задайте $g_t$ вручную и покажите, как state запоминает
только выбранный token.

### 3. Сравнить memory

Для 32-layer Transformer с 8 KV heads и hybrid, где attention стоит в каждом
четвёртом layer, посчитайте cache при 128K context. Добавьте fixed SSM state.

### 4. Прочитать figures, затем код

Сопоставьте Mamba paper figure с modules официального repository:
`in_proj`, short convolution, selective scan, gate, `out_proj`. Не начинайте с
CUDA kernel — сначала восстановите dataflow.

## После главы нужно уметь

- вывести convolution form linear SSM;
- объяснить, что именно делает input-dependent selection;
- сравнить persistent memory attention и recurrence;
- различать Mamba, RWKV, RetNet и Jamba;
- объяснить parallel/recurrent/chunkwise forms;
- выбрать stress tests для exact recall;
- не смешивать sequence mixer и multimodality.

## Материалы, на которых построена глава

### Основные papers

- [Gu & Dao — Mamba](https://arxiv.org/abs/2312.00752).
- [Dao & Gu — Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060).
- [Peng et al. — RWKV](https://arxiv.org/abs/2305.13048).
- [Sun et al. — RetNet](https://arxiv.org/abs/2307.08621).
- [Lieber et al. — Jamba](https://arxiv.org/abs/2403.19887).

### Визуальные и code companions

- [Maarten Grootendorst — A Visual Guide to Mamba and State Space Models](https://www.maartengrootendorst.com/blog/mamba/)
  — более 50 последовательных визуализаций от SSM к selective scan.
- [Official Mamba repository](https://github.com/state-spaces/mamba).
- [[02 Areas/ML & DL/Papers/Mamba]]
- [[02 Areas/ML & DL/Papers/RWKV]]
- [[02 Areas/ML & DL/Papers/RetNet]]
- [[02 Areas/ML & DL/Papers/Jamba SSM-Transformer Hybrid]]

**Дальше:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|из чего на практике складывается pre-training recipe: данные, objective, tokens и scaling.]]
