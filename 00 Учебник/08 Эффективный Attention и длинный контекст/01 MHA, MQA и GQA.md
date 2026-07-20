---
title: MHA, MQA и GQA
type: textbook-chapter
status: canonical
last_updated: 2026-07-18
previous: "[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура]]"
next: "[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1911.02150
  - https://arxiv.org/abs/2305.13245
---

# MHA, MQA и GQA: почему экономят ключи и значения

Во время авторегрессионной генерации новый запрос сравнивается с ключами всех
предыдущих токенов, а затем смешивает соответствующие значения. Ключи и
значения прошлого не меняются, поэтому их сохраняют в KV-кэше. С ростом
контекста и батча этот кэш занимает всё больше памяти и должен заново читаться
на каждом шаге генерации.

MHA, MQA и GQA используют одну формулу внимания, но по-разному организуют
хранимую память. В многоголовом внимании каждая голова запроса имеет собственные
ключи и значения. MQA оставляет всем запросам один общий набор, а GQA объединяет
головы запросов в несколько групп. Чтобы увидеть цену каждого варианта, мы
сначала выведем размер KV-кэша, а затем проследим формы тензоров и реальное
чтение памяти при генерации.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png]]

*Иллюстрация из материалов GQA: query heads сохраняются, а число независимых
K/V heads уменьшается от MHA к MQA. GQA — промежуточная точка.*

## Проблема появляется не на обучении, а при генерации

Рассмотрим prompt из $S$ токенов и попросим модель сгенерировать ещё один.
Новый query должен сравниваться со всеми прошлыми keys:

$$
\operatorname{Attention}(q_{S+1},K_{1:S},V_{1:S})
=\operatorname{softmax}\left(
\frac{q_{S+1}K_{1:S}^{\top}}{\sqrt{d_h}}
\right)V_{1:S}.
$$

Старые $K_{1:S}$ и $V_{1:S}$ зависят только от уже обработанных токенов.
Пересчитывать их на каждом шаге бессмысленно — они сохраняются в KV-cache.

```text
prefill:
tokens 1 ... S ─→ K₁...Kₛ, V₁...Vₛ ─→ cache

decode step S+1:
new token ─→ qₛ₊₁, kₛ₊₁, vₛ₊₁
             │         └────────────→ append to cache
             └→ read ALL cached K/V → attention output
```

При следующем токене кэш снова читается. Для длинного контекста и большого батча
декодирование становится ограничено пропускной способностью памяти: ускоритель
тратит время не только на арифметику, но и на перенос K/V из HBM.

## Сколько памяти занимает KV-cache

Для одного sequence:

$$
\text{KV bytes}
=2\,L\,S\,n_{kv}\,d_h\,b,
$$

где:

- $2$ — отдельно keys и values;
- $L$ — число layers;
- $S$ — число cached tokens;
- $n_{kv}$ — число K/V heads;
- $d_h$ — head dimension;
- $b$ — bytes на element.

### Численный пример

Пусть:

$$
L=32,\quad S=8192,\quad n_q=32,\quad d_h=128,\quad b=2
$$

для BF16.

При MHA $n_{kv}=32$:

$$
2\cdot32\cdot8192\cdot32\cdot128\cdot2
=4\,294\,967\,296\text{ bytes}\approx4\text{ GiB}.
$$

Это кэш **одной** последовательности. Батч из 16 последовательностей потребовал бы примерно 64 GiB,
не считая weights, activations и allocator overhead.

Если $n_{kv}=8$, cache уменьшается до 1 GiB. Если $n_{kv}=1$ — до 128 MiB.
Именно эту ось меняют GQA и MQA.

## MHA: отдельная память для каждой головы

В Multi-Head Attention:

$$
n_q=n_{kv}=h.
$$

Для каждой головы $r$:

$$
Q^{(r)}=XW_Q^{(r)},\quad
K^{(r)}=XW_K^{(r)},\quad
V^{(r)}=XW_V^{(r)}.
$$

Голова может выучить собственное пространство для поиска и собственное
представление переносимого содержимого. Это наиболее свободный вариант.

```text
q₁ ↔ k₁,v₁
q₂ ↔ k₂,v₂
q₃ ↔ k₃,v₃
q₄ ↔ k₄,v₄
```

Но все $k_r,v_r$ прошлого приходится хранить. При пошаговом декодировании стоимость
такой свободы платится на каждом шаге.

## MQA: все запросы читают одну память K/V

Multi-Query Attention оставляет много голов запросов, но использует общие K и V:

$$
n_q=h,\qquad n_{kv}=1.
$$

```text
q₁ ─┐
q₂ ─┼→ shared k,v
q₃ ─┤
q₄ ─┘
```

Проекции запросов всё ещё различаются, поэтому головы могут задавать разные
вопросы. Но они задают их одной общей системе адресов K и одному набору значений.

Ноам Шазир предложил MQA именно как способ уменьшить требования к пропускной
способности памяти при пошаговом декодировании. Статья показывает практическое ускорение с небольшим, но возможным
ухудшением качества.

> [!note] Почему название multi-query
> «Multi» относится к множеству голов запросов. Голова ключей и значений при
> этом одна.

## GQA: несколько групп памяти

Grouped-Query Attention выбирает:

$$
1<n_{kv}<n_q.
$$

Головы запросов делятся на группы. Все головы внутри группы используют общие K/V.

Для $n_q=8$ и $n_{kv}=2$:

```text
group 1: q₁ q₂ q₃ q₄ ─→ k₁,v₁
group 2: q₅ q₆ q₇ q₈ ─→ k₂,v₂
```

Число repetitions:

$$
r=\frac{n_q}{n_{kv}}.
$$

Если $n_q=32$ и $n_{kv}=8$, $r=4$: каждые четыре головы запросов читают одну
голову KV. Кэш в четыре раза меньше MHA.

## Формы тензоров: место, где перестаёт работать интуиция

Пусть:

$$
X\in\mathbb R^{B\times T\times d},\quad
n_q=32,\quad n_{kv}=8,\quad d_h=128.
$$

Тогда:

$$
Q\in\mathbb R^{B\times32\times T\times128},
$$

$$
K,V\in\mathbb R^{B\times8\times T\times128}.
$$

Чтобы записать формулу как обычный batched attention, implementation может
логически повторить K/V по query groups:

$$
K'\in\mathbb R^{B\times32\times T\times128}.
$$

Но физически материализовывать четыре копии кэша не нужно. Вычислительное ядро может
переиспользовать одну голову KV для нескольких голов запросов. Иначе смысл экономии
частично потеряется.

Упрощённая функция shape transformation:

```python
def repeat_kv(x, n_rep):
    # x: [batch, tokens, kv_heads, head_dim]
    b, t, kv, d = x.shape
    return (
        x[:, :, :, None, :]
        .expand(b, t, kv, n_rep, d)
        .reshape(b, t, kv * n_rep, d)
    )
```

Такая функция есть в официальном Llama code. `expand` описывает shared view;
оптимизированный attention backend может обойтись без реальной полной копии.

## GQA меняет не только cache, но и projections

В MHA:

$$
W_Q,W_K,W_V\in\mathbb R^{d\times d}
$$

при $h\,d_h=d$.

В GQA:

$$
W_Q\in\mathbb R^{d\times(n_qd_h)},
$$

$$
W_K,W_V\in\mathbb R^{d\times(n_{kv}d_h)}.
$$

K/V projections становятся уже. Поэтому уменьшаются:

- параметры $W_K,W_V$;
- compute этих projections;
- объём K/V, записываемый на token;
- bandwidth при чтении cache.

Query и output projections обычно остаются прежней ширины.

## Как превратить готовую MHA model в GQA

GQA paper рассматривает uptraining существующего checkpoint. Для уменьшения
числа KV heads исходные MHA heads объединяются по группам, например усреднением
их projection weights:

$$
\bar W_K^{(g)}
=\frac1{|G_g|}\sum_{r\in G_g}W_K^{(r)},
$$

и аналогично для $W_V$. После такого surgery модель дополнительно обучают.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gqa/uptrain-recycling.png]]

*Схема из GQA paper: checkpoint не выбрасывают; K/V heads агрегируют и проводят
короткое uptraining. Авторы исследовали бюджет около 5% исходного pre-training
compute.*

Простое усреднение без обучения не обязано сохранять качество: разные heads
могли выучить несовместимые пространства.

## Prefill и decode — два разных режима

### Prefill

Все prompt tokens обрабатываются параллельно. Attention строит большую матрицу
$T\times T$, и compute может быть главным ограничением.

### Decode

Обычно обрабатывается один новый token на sequence. Query length равна 1, но K/V
длина равна всему context. Arithmetic intensity ниже, cache читается снова и
снова, поэтому bandwidth особенно важен.

GQA часто даёт наиболее очевидную пользу именно на decode. Ускорение end-to-end
зависит от:

- длины prompt и output;
- batch size;
- конкретного GPU;
- cache precision;
- kernel support;
- scheduler и paged memory;
- tensor parallel layout.

Архитектурное уменьшение bytes не равно автоматическому ускорению во столько же
раз.

## Что GQA не решает

GQA не:

- убирает линейный рост KV-cache с context length;
- делает prefill attention линейным по sequence;
- уменьшает количество query heads;
- заменяет FlashAttention;
- гарантирует качество MHA при любом числе groups;
- само по себе расширяет training context.

FlashAttention оптимизирует порядок вычислений и memory IO attention. GQA
уменьшает сами K/V tensors. Эти методы совместимы и решают разные части системы.

## Как выбирать число KV heads

Крайние случаи:

$$
n_{kv}=n_q\Rightarrow\text{MHA},
$$

$$
n_{kv}=1\Rightarrow\text{MQA}.
$$

Между ними — trade-off:

| Больше $n_{kv}$ | Меньше $n_{kv}$ |
|---|---|
| больше capacity K/V | меньше cache |
| потенциально лучше quality | выше decode throughput |
| больше bandwidth | больше sharing между queries |

Нет универсально лучшего значения. Его выбирают вместе с model size, quality
target, serving hardware и kernels.

## Где встречается

- LLaMA 1 использовала MHA.
- Llama 2 70B использовала GQA, меньшие Llama 2 — MHA.
- Llama 3 использует GQA в основных опубликованных размерах.
- Mistral и многие Qwen models также используют GQA.
- Некоторые модели выбирают MQA.
- DeepSeek-V2/V3 идут дальше и используют MLA — следующая глава.

Всегда проверяйте config конкретного checkpoint:

```json
{
  "num_attention_heads": 32,
  "num_key_value_heads": 8
}
```

Отношение этих полей сразу показывает MHA/MQA/GQA.

## Ошибки, которые нужно видеть

> [!danger] «GQA уменьшает число attention heads»
> Query heads остаются. Уменьшается число независимых K/V heads.

> [!danger] «K/V shared, значит heads одинаковые»
> Queries различаются, поэтому attention weights и output каждой query head
> различаются даже при общем K/V.

> [!danger] «Cache — это attention matrix»
> В cache хранятся projected keys и values прошлых tokens, не матрица softmax
> scores.

> [!danger] «В четыре раза меньше cache — в четыре раза быстрее»
> Это верхнеуровневая экономия bytes. Реальная latency включает projections,
> FFN, communication, scheduling и kernel overhead.

## Практика

### 1. Калькулятор KV-cache

Реализуйте:

```python
def kv_cache_gib(layers, tokens, kv_heads, head_dim, bytes_per_element, batch=1):
    total = 2 * layers * tokens * kv_heads * head_dim
    return total * bytes_per_element * batch / 2**30
```

Сравните MHA/GQA/MQA для context 8K, 32K и 128K. Затем измените BF16 на FP8
cache и отделите эффект architecture от quantization.

### 2. Проверка shapes

Создайте random Q с 8 heads и K/V с 2 heads. Повторите K/V по groups, выполните
обычный scaled dot-product attention и проверьте output shape.

### 3. Прочитать config

Для нескольких checkpoints выпишите:

- `hidden_size`;
- `num_attention_heads`;
- `num_key_value_heads`;
- `head_dim`;
- число layers и context.

По формуле оцените cache одного 32K sequence. Это полезнее списка названий
архитектур: появляется количественная интуиция.

## После главы нужно уметь

- вывести формулу KV-cache из tensor shapes;
- объяснить, почему decode memory-bound сильнее prefill;
- нарисовать MHA, MQA и GQA без смешения Q и KV heads;
- прочитать тип attention по model config;
- объяснить, почему меньше cache не гарантирует пропорциональную latency;
- отделить GQA от FlashAttention, quantization и context extension.

## Материалы, на которых построена глава

### Основные papers

- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  — исходная MHA.
- [Shazeer — One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150)
  — MQA и bandwidth motivation.
- [Ainslie et al. — GQA](https://arxiv.org/abs/2305.13245) — grouped queries и
  uptraining MHA checkpoints.

### Код и связанные объяснения

- [Meta — Llama 3 `model.py`](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
  — `n_kv_heads`, `repeat_kv` и cache shapes в небольшом reference code.
- [Llama 2 paper](https://arxiv.org/abs/2307.09288) — практическое применение
  GQA в 70B model.
- [[02 Areas/ML & DL/01 Справочник/Attention/MHA]]
- [[02 Areas/ML & DL/01 Справочник/Attention/MQA]]
- [[02 Areas/ML & DL/01 Справочник/Attention/GQA]]

**Дальше:** [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA: вместо нескольких явных K/V heads хранить компактный latent.]]
