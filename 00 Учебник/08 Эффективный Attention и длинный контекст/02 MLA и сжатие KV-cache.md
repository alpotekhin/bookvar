---
title: MLA и сжатие KV-cache
type: textbook-chapter
status: canonical
last_updated: 2026-07-18
previous: "[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA]]"
next: "[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 От SwiGLU к MoE]]"
primary_sources:
  - https://arxiv.org/abs/2405.04434
  - https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
---

# Multi-head Latent Attention: хранить компактный источник K/V

GQA уменьшает число наборов ключей и значений, но всё ещё сохраняет их в готовом
виде. Multi-head Latent Attention (MLA) предлагает хранить более раннее и
компактное представление: каждый токен сжимается в низкоразмерный латентный
вектор, из которого линейными преобразованиями получаются ключи и значения
разных голов.

Одного сжатия недостаточно. Если на каждом шаге полностью восстанавливать все
головы, экономия памяти может превратиться в дополнительные вычисления и
пересылки. Поэтому важнейшая часть MLA — алгебраический перенос матриц: текущий
запрос сравнивается непосредственно с сохранённым латентным представлением, а
значения разворачиваются после взвешенного суммирования. Позиционная часть RoPE
мешает такому переносу и потому хранится отдельно. Глава последовательно выводит
все три решения, а не сводит MLA к фразе «сжатый KV-кэш».

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/deepseek-v2-figure2-hq.png]]

*Оригинальная схема MLA из DeepSeek-V2. На первом чтении она перегружена, поэтому
ниже разберём её в три прохода: обычный MHA, latent compression, затем RoPE.*

## Начнём с того, что именно дорого в MHA

Для скрытого состояния токена $h_t\in\mathbb R^d$ обычное внимание создаёт:

$$
q_t=W_Qh_t,\qquad k_t=W_Kh_t,\qquad v_t=W_Vh_t.
$$

После разделения по головам:

$$
k_t\in\mathbb R^{n_h d_h},\qquad
v_t\in\mathbb R^{n_h d_v}.
$$

Для прошлого токена запрос больше не нужен: новый запрос появится на текущем
шаге. Но $k_t$ и $v_t$ будут нужны каждому будущему токену, поэтому они попадают
в кэш.

```text
h_t ─→ W_K ─→ full multi-head k_t ─┐
    └→ W_V ─→ full multi-head v_t ─┴→ KV-cache
```

На один слой и токен ширина кэша равна:

$$
n_h(d_h+d_v).
$$

MLA ставит вопрос: обязаны ли мы хранить уже развёрнутые векторы отдельных голов,
если все они получены линейным преобразованием одного $h_t$?

## Шаг 1. Низкоранговое сжатие K и V

Сначала скрытое состояние сжимается:

$$
c_t^{KV}=W^{DKV}h_t,
$$

где:

$$
W^{DKV}\in\mathbb R^{d_c\times d},
\qquad d_c\ll n_h(d_h+d_v).
$$

Буква $D$ обозначает понижающую проекцию. Из латентного вектора
восстанавливаются содержательные части:

$$
k_t^C=W^{UK}c_t^{KV},
$$

$$
v_t^C=W^{UV}c_t^{KV}.
$$

```text
                           ┌→ W^UK → content keys for all heads
h_t → W^DKV → c_t^KV ─────┤
                           └→ W^UV → values for all heads
              ↑
          хранить это
```

Это факторизация исходных проекций:

$$
W_K^C=W^{UK}W^{DKV},\qquad
W_V=W^{UV}W^{DKV}.
$$

Если промежуточный ранг $d_c$ мал, кэш хранит $c_t^{KV}$ вместо полных K/V.

## Численный масштаб экономии

Упрощённый пример:

$$
n_h=32,\quad d_h=d_v=128,\quad d_c=512.
$$

Обычный MHA хранит на токен и слой:

$$
32(128+128)=8192
$$

элементов.

Латентный вектор занимает:

$$
512
$$

элементов до учёта небольшой позиционной части. В этом учебном примере
коэффициент сжатия равен $16\times$.

Фактические размеры конкретной модели DeepSeek нужно брать из конфигурации, а не из
этого примера. Отчёт DeepSeek-V2 сообщает уменьшение KV-кэша на 93,3% относительно
сравниваемой архитектуры; это результат конкретной конфигурации, не универсальная
константа MLA.

## Шаг 2. Запросы тоже можно факторизовать

В полной MLA проекция запроса также может проходить через низкоранговое сжатие:

$$
c_t^Q=W^{DQ}h_t,
$$

$$
q_t^C=W^{UQ}c_t^Q.
$$

На кэш это напрямую не влияет: запросы прошлых токенов не сохраняются.
Сжатие запросов уменьшает параметры и вычисления проекции и является отдельной
частью архитектуры.

В официальной эталонной реализации DeepSeek-V3 это видно по полям:

```python
q_lora_rank
kv_lora_rank
```

и веткам `wq_a → q_norm → wq_b`, `wkv_a → kv_norm → wkv_b`.
Название LoRA здесь описывает низкоранговую факторизацию внутри базовой
архитектуры; это не адаптеры для параметрически эффективного дообучения.

## Проблема: RoPE нельзя бездумно спрятать в latent

RoPE — position-dependent операция:

$$
k_{t}^{R}=R_t\,\tilde k_t.
$$

Если full key равен:

$$
k_t=R_tW^{UK}c_t,
$$

то $R_t$ различается для каждого cached token. Нельзя один раз объединить
$W^{UK}$ с query projection и забыть о реконструкции: между matrices стоит
зависящее от позиции вращение.

Для content projection без RoPE работает matrix absorption. Attention score:

$$
(q_i^C)^\top k_j^C
=(q_i^C)^\top W^{UK}c_j^{KV}
=\left((W^{UK})^\top q_i^C\right)^\top c_j^{KV}.
$$

Можно преобразовать текущий query и сравнивать его прямо с cached latent.
Полный $k_j^C$ материализовывать не обязательно.

Но если добавить $R_j$ между $W^{UK}$ и $c_j$, это равенство уже нельзя
использовать тем же способом для всех positions.

## Шаг 3. Decoupled RoPE

DeepSeek разделяет query/key на:

- content component без RoPE;
- небольшую positional component с RoPE.

$$
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],
$$

$$
k_{t,i}=[k_{t,i}^{C};k_t^{R}].
$$

Здесь positional key $k_t^R$ shared между heads, а content key остаётся
head-specific через up-projection.

Attention score decomposes:

$$
q_{i}^{\top}k_j
=(q_i^C)^\top k_j^C+(q_i^R)^\top k_j^R.
$$

Первое слагаемое можно считать через latent absorption, второе — через
небольшой cached RoPE vector.

```text
cache token j

content:     c_j^KV  ───────────────┐
                                     ├→ attention score
position:    k_j^R = RoPE(position) ─┘
```

Именно поэтому cache MLA — не только один latent. Обычно нужно хранить:

$$
[c_j^{KV};k_j^R].
$$

## Два способа inference

Официальный DeepSeek-V3 reference implementation показывает полезное различие.

### Naive path

Latent разворачивается в полные head-specific K/V, и cache выглядит почти как
обычный:

```python
k_cache: [batch, seq, heads, qk_head_dim]
v_cache: [batch, seq, heads, v_head_dim]
```

Так проще понять correctness, но исчезает главная memory benefit.

### Absorbed path

Хранятся:

```python
kv_cache: [batch, seq, kv_lora_rank]
pe_cache: [batch, seq, qk_rope_head_dim]
```

Up-projection weights алгебраически переносятся на query и output sides.
Attention работает с compressed latent, не создавая полный cache.

> [!warning] Архитектура и kernel неразделимы
> Если runtime формально поддерживает checkpoint, но материализует full K/V,
> модель выдаст правильные tokens, однако не получит обещанную efficiency.
> MLA требует специализированного inference path.

## Matrix absorption для values

Обычный output одной head:

$$
o_i=\sum_j a_{ij}v_j
=\sum_j a_{ij}W^{UV}c_j.
$$

По линейности:

$$
o_i=W^{UV}\left(\sum_j a_{ij}c_j\right).
$$

Сначала attention суммирует компактные latents, затем up-projection
восстанавливает value-space result. Если дальше стоит output projection $W_O$,
matrices можно сгруппировать:

$$
W_O\,W^{UV}
\left(\sum_j a_{ij}c_j\right).
$$

Это вторая половина трюка: не только keys можно не разворачивать до сравнения,
values можно не разворачивать до weighted sum.

## MLA, MHA, GQA: сравнение на одной оси

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/deepseek-v2/x1.png]]

*Схема DeepSeek-V2 показывает место MLA рядом с экспертным полносвязным слоем.
Здесь важно смотреть только на ветвь внимания; устройство DeepSeekMoE
разбирается в следующем модуле.*

| Метод | Что хранится на прошлый token | Как экономит |
|---|---|---|
| MHA | K/V каждой head | не экономит |
| MQA | один общий K и V | sharing всех heads |
| GQA | K/V каждой группы | sharing внутри groups |
| MLA | low-rank KV latent + RoPE key | compression + matrix absorption |

GQA выбирает меньше явных memory banks. MLA пытается представить множество
head-specific banks через общий компактный basis.

## MLA — не «обычный autoencoder»

Down/up projections обучаются end-to-end вместе со всей language model. Нет
отдельной reconstruction loss, заставляющей восстановить исходные K/V некоторой
готовой MHA model. Критерий — next-token loss и итоговое поведение attention.

Latent должен сохранять не все детали потенциальных K/V, а только информацию,
полезную для модели.

## Что относится к DeepSeek, но не к MLA

DeepSeek-V2/V3 объединяют несколько innovations:

- MLA — attention и KV-cache;
- DeepSeekMoE — sparse FFN;
- auxiliary-loss-free balancing — routing;
- Multi-Token Prediction — training objective;
- FP8 — training/inference precision;
- DualPipe — pipeline scheduling;
- RL/RLVR — post-training.

Фраза «DeepSeek быстрее благодаря MLA» может быть частью ответа, но не полным
объяснением system result. Каждое решение влияет на свой bottleneck.

## Trade-offs

### Плюсы

- существенно меньший KV-cache;
- ниже bandwidth при decode;
- head-specific content можно сохранить через learned up-projections;
- особенно привлекательно для длинного context и больших batches.

### Цена

- сложнее algebra и implementation;
- стандартный MHA kernel недостаточен для полной выгоды;
- decoupled RoPE добавляет отдельные shapes;
- tensor parallelism и quantized cache требуют специальной поддержки;
- ошибка в absorption может дать правильные shapes, но неверную математику.

## Как прочитать официальный код

В
[DeepSeek-V3 `inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
найдите:

1. `wq_a`, `q_norm`, `wq_b` — query low-rank path;
2. `wkv_a` — совместно создаёт KV latent и RoPE key;
3. `wkv_b` — восстанавливает content keys и values;
4. `qk_nope_head_dim` — content dimension;
5. `qk_rope_head_dim` — positional dimension;
6. `kv_cache` и `pe_cache` — compressed inference path;
7. ветвление `attn_impl == "naive"` — наглядное сравнение двух реализаций.

Это один из редких случаев, когда names в reference code сами образуют хорошую
аннотированную схему paper.

## Ошибки, которые нужно видеть

> [!danger] «MLA хранит один общий K/V head»
> Это MQA. MLA хранит latent, из которого определяются head-specific content
> projections.

> [!danger] «Low-rank значит LoRA fine-tuning»
> Здесь low-rank matrices являются постоянной частью model architecture и
> обучаются с pre-training.

> [!danger] «RoPE тоже полностью находится в latent»
> Positional component специально отделён, потому что position-dependent
> rotation мешает absorption.

> [!danger] «Любой runtime с DeepSeek weights использует compressed cache»
> Correctness support и optimized MLA kernels — разные уровни поддержки.

> [!danger] «93.3% — свойство любого MLA»
> Это reported comparison DeepSeek-V2 с конкретными dimensions и baseline.

## Практика

### 1. Проверить absorption алгеброй

Создайте random $q$, $c$, $W$ и сравните:

```python
left = q @ (W @ c)
right = (W.T @ q) @ c
```

После совпадения добавьте position-dependent rotation между $W$ и $c$ и
объясните, почему один absorbed query больше нельзя переиспользовать одинаково
для всех cached positions.

### 2. Сравнить cache shapes

Для chosen dimensions посчитайте elements/token/layer:

```text
MHA:  heads × (key_dim + value_dim)
GQA:  kv_heads × (key_dim + value_dim)
MLA:  kv_lora_rank + rope_head_dim
```

Затем умножьте на layers, context, batch и bytes. Не сравнивайте models только
по total parameters — serving memory определяется другими dimensions.

### 3. Проследить naive и absorbed paths

В reference code выпишите, какие tensors кешируются в каждом режиме. Нарисуйте
граф matrix operations и отметьте, где материализуются head dimensions.

## После главы нужно уметь

- объяснить MLA как factorization, а не как магическую compression;
- вывести, почему content key projection можно поглотить в query;
- вывести перенос value up-projection после weighted sum;
- объяснить необходимость decoupled RoPE;
- различать naive и absorbed inference;
- отделить MLA от MoE, FP8 и RLVR в DeepSeek system.

## Материалы, на которых построена глава

### Основные источники

- [DeepSeek-AI — DeepSeek-V2](https://arxiv.org/abs/2405.04434) — введение MLA,
  архитектурная схема и reported KV-cache reduction.
- [DeepSeek-V2 official repository](https://github.com/deepseek-ai/DeepSeek-V2)
  — report, figures и model context.
- [DeepSeek-V3 official inference code](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
  — наиболее полезный code companion для naive/absorbed cache.
- [DeepSeek — FlashMLA](https://github.com/deepseek-ai/FlashMLA) — пример того,
  что архитектурная экономия требует специализированных kernels.

### Внутри базы

- [[02 Areas/ML & DL/Papers/DeepSeek-V2]]
- [[02 Areas/ML & DL/Concepts/Training/Multi-head Latent Attention]]
- [[02 Areas/ML & DL/01 Справочник/Attention/MLA]]

**Дальше:** [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 От SwiGLU к MoE|как sparse MoE увеличивает число параметров, не активируя их все для каждого токена.]]
