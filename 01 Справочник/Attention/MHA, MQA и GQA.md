---
title: MHA, MQA и GQA
aliases: [Multi-Head Attention, Multi-Query Attention, Grouped-Query Attention]
type: concept
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1911.02150
  - https://arxiv.org/abs/2305.13245
---

# MHA, MQA и GQA

MHA, MQA и GQA выполняют одно и то же scaled dot-product attention. Различие
между ними состоит в числе проекций key и value и в том, какие query-heads
пользуются общей парой K/V. Это почти не меняет вычисления над queries, зато
напрямую определяет размер KV-cache и объём данных, читаемых при генерации.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png]]

*В MHA каждой query-head соответствует собственная пара K/V; в GQA несколько
query-heads образуют группу вокруг одной пары; в MQA все query-heads используют
единственные K и V. Joshua Ainslie et al., “GQA: Training Generalized
Multi-Query Transformer Models from Multi-Head Checkpoints”, Figure 2,
[с. 4](https://arxiv.org/pdf/2305.13245#page=4).*

## MHA: отдельная память у каждой головы

Пусть после проекций получены $h_q$ query-heads и $h_{kv}$ пар key/value-heads
размерности $d_h$. В обычном Multi-Head Attention

$$h_{kv}=h_q.$$

Каждая голова вычисляет

$$
H_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^T}{\sqrt{d_h}}+M\right)V_i,
\qquad
\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_{h_q})W_O.
$$

Собственные $K_i$ и $V_i$ дают каждой голове независимое представление памяти.
Во время autoregressive decode эти тензоры уже вычислены для прошлых токенов и
лежат в KV-cache. Поэтому свобода MHA оплачивается хранением $h_q$ наборов K/V.

## MQA: одна K/V-пара для всех queries

Multi-Query Attention сохраняет независимые query-проекции, но задаёт

$$h_{kv}=1.$$

Для каждой query-head меняется $Q_i$, тогда как $K$ и $V$ общие:

$$
H_i=\operatorname{softmax}\!\left(\frac{Q_iK^T}{\sqrt{d_h}}+M\right)V.
$$

Query-heads не исчезают и их результаты не усредняются. Экономия возникает
только на стороне K/V: каждый новый токен добавляет в cache одну пару векторов,
а не $h_q$ пар. Shazeer предложил MQA именно как способ уменьшить объём памяти и
memory bandwidth при incremental decoding. Цена такого решения — одна общая
K/V-память должна обслуживать все способы чтения, что иногда ухудшает качество.

## GQA: несколько групп K/V

Grouped-Query Attention выбирает промежуточное число голов:

$$1<h_{kv}<h_q, \qquad g=\frac{h_q}{h_{kv}}.$$

Каждые $g$ query-heads читают одну K/V-пару. Например, при $h_q=32$ и
$h_{kv}=8$ в каждой группе четыре query-heads. В сравнении с MHA такой cache
примерно в четыре раза меньше, но модель сохраняет восемь независимых
представлений памяти вместо одного у MQA.

## Что именно экономится

Для batch size $B$, длины уже обработанного контекста $T$, размерности головы
$d_h$ и размера одного элемента $s$ байт объём cache одного attention-слоя равен

$$
M_{KV}=2BT h_{kv}d_hs.
$$

Множитель 2 отвечает за key и value. Поэтому переход с 32 KV-heads на 8
сокращает этот компонент в четыре раза, а переход на MQA — в 32 раза. Отношение
не зависит от precision; абсолютный объём, конечно, зависит от `bf16`, `fp8` или
квантизации cache.

| Вариант | $h_{kv}$ | Кто делит K/V | KV-cache относительно MHA |
|---|---:|---|---:|
| MHA | $h_q$ | никто | $1$ |
| GQA | от 2 до $h_q-1$ | query-heads одной группы | $h_{kv}/h_q$ |
| MQA | 1 | все query-heads | $1/h_q$ |

Ускорение wall-clock не обязано совпадать с этим отношением: prefill по-прежнему
выполняет большие матричные умножения, а decode зависит также от batch size,
реализации kernels и пропускной способности памяти. Наиболее заметен выигрыш в
memory-bound decode с длинным контекстом или большим continuous batch.

## Как выбирать

- MHA сохраняет максимальную независимость K/V-heads и служит исходной точкой.
- MQA минимизирует cache, но сильнее всего ограничивает K/V-представление.
- GQA позволяет подобрать число KV-heads как архитектурный компромисс.

В статье GQA также описано преобразование MHA-checkpoint: K/V-heads внутри
будущей группы усредняются, после чего модель недолго дообучается. Это не
означает, что любая pretrained MHA-модель автоматически становится GQA без
потери качества; uptraining является частью метода.

См. также [[02 Areas/ML & DL/01 Справочник/Inference/KV-cache|KV-cache]] и
[[02 Areas/ML & DL/01 Справочник/Attention/MLA|MLA]].

## Подробнее

Связь числа query- и KV-heads с формами тензоров, объёмом cache и скоростью
decode разобрана в главе [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|MHA, MQA и GQA]].

## Источники

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Noam Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150).
- Joshua Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245).
