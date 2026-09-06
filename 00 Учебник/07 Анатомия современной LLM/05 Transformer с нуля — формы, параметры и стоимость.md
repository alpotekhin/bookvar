---
title: Transformer с нуля — формы, параметры и стоимость
type: textbook-chapter
status: reviewed
last_updated: 2026-09-06
source_unit_id:
  - lecture-01-figure-step-177-rendering-1
  - lecture-01-figure-step-214-rendering-1
  - lecture-02-tensor-memory-accounting
  - lecture-02-section-189-intuitions
  - lecture-02-section-196-linear-model
  - lecture-02-section-251-model-flops-utilization-mfu
  - lecture-02-arithmetic-intensity-derivation
  - lecture-02-section-440-zoom-in-on-one-layer
  - lecture-02-section-454-consider-all-layers
  - lecture-02-section-515-memory
  - lecture-02-section-526-compute-for-one-training-step
  - lecture-02-section-529-transformers
  - lecture-02-gradient-accumulation-example
  - assignment-01-transformer-construction-sequence
  - assignment-01-task-linear
  - assignment-01-task-embedding
  - assignment-01-task-rope
  - assignment-01-task-transformer-accounting
  - assignment-01-task-adamw-accounting
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
  - https://arxiv.org/abs/1706.03762
previous: "[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE и позиционная информация]]"
next: "[[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля]]"
---

# Transformer с нуля: формы, параметры и стоимость

Transformer легко описать цепочкой названий — embedding, attention, SwiGLU,
residual, logits. Этого описания недостаточно, чтобы собрать работающую модель:
две формулы могут быть по отдельности верны и всё же не состыковываться по
осям, а архитектура с правильным числом параметров — не помещаться в память.
Поэтому здесь модель читается сразу в трёх системах координат:

1. **форма** отвечает, какие оси входят в тензор и где они меняются;
2. **параметры** отвечают, какие обучаемые матрицы существуют постоянно;
3. **стоимость шага** отвечает, сколько арифметики и движения байтов вызовет
   конкретный пакет токенов.

Такой разбор не заменяет профилирование. Он нужен, чтобы до запуска получить
проверяемую гипотезу, а после запуска понять, какой именно член формулы измерил
профилировщик.

<a id="notation-and-shapes"></a>

## Сначала имена осей, затем операции

Зафиксируем обозначения до первой строки кода.

| Символ | Смысл |
|---|---|
| $B$ | число последовательностей в пакете |
| $T$ | число токенов в каждой последовательности |
| $V$ | размер словаря |
| $D$ | ширина residual stream, `d_model` |
| $H$ | число attention-голов |
| $K=D/H$ | ширина одной головы |
| $F$ | внутренняя ширина SwiGLU, `d_ff` |
| $L$ | число Transformer-блоков |
| $N=BT$ | число токен-позиций в пакете |

Делимость $D$ на $H$ — обязательное условие архитектуры: без неё тензор нельзя
разложить на одинаковые головы. Входные token IDs имеют форму $[B,T]$.
Embedding lookup добавляет ось признаков:

$$[B,T]\longrightarrow[B,T,D].$$

Каждый блок обязан вернуть ту же внешнюю форму $[B,T,D]$; иначе следующий блок
не сможет принять результат. Внутри attention одна матрица признаков временно
становится четырёхмерной:

$$
Q,K,V:[B,T,D]\longrightarrow[B,T,H,K]
\longrightarrow[B,H,T,K].
$$

Последняя перестановка не меняет данные, но делает головы отдельной осью. Теперь

$$QK^\top:[B,H,T,K]\times[B,H,K,T]\to[B,H,T,T].$$

Softmax нормализуется по последней оси — по ключам, доступным одному query.
Причинная маска также должна broadcast-иться к $[B,H,T,T]$. Умножение весов на
$V$ возвращает $[B,H,T,K]$, после чего головы переставляются и объединяются в
$[B,T,D]$. SwiGLU меняет только последнюю ось:

$$[B,T,D]\to[B,T,F]\to[B,T,D].$$

Наконец LM head создаёт логиты $[B,T,V]$. Эта форма нужна функции потерь;
softmax не обязан материализоваться внутри модели, потому что устойчивую
cross-entropy можно вычислить непосредственно по логитам.

Имена осей полезнее запоминания конкретного `reshape`. Запись вроде
`batch sequence (heads head_dim)` заставляет проверить равенство $D=HK$ в том
месте, где оно действительно нужно. Анонимные `view(B, T, H, -1)` и `transpose`
могут дать тензор правильного размера, но перепутать семантику осей — такой баг
часто не вызывает исключения.

<a id="parameter-accounting"></a>

## Параметры: считаем матрицы, а не строки класса

Пусть bias отключён, как в базовой конфигурации Stanford CS336 Assignment 1.
Token embedding содержит $VD$ параметров. Если LM head не разделяет с ним веса,
он добавляет ещё $DV$; при weight tying это одна и та же матрица, а не две.
RoPE обучаемых параметров не добавляет.

В self-attention четыре проекции ширины $D$: $W_Q,W_K,W_V,W_O$. При одинаковой
ширине query и key/value это

$$P_{\text{attn}}=4D^2.$$

SwiGLU имеет две входные матрицы и одну выходную:

$$P_{\text{SwiGLU}}=DF+DF+FD=3DF.$$

Два RMSNorm внутри pre-norm-блока дают ещё $2D$ масштабов. Поэтому без bias и
экзотических attention-вариантов

$$P_{\text{block}}=4D^2+3DF+2D.$$

Для всей decoder-only LM надо отдельно назвать соглашения:

$$
P_{\text{model}}=VD+L P_{\text{block}}+D
+\begin{cases}
0,&\text{tied embeddings},\\
DV,&\text{untied LM head}.
\end{cases}
$$

Последний $D$ — финальный RMSNorm. Формула ценнее одной цифры: она показывает,
почему изменение $F$, weight tying или числа слоёв немедленно меняет бюджет.
Она также не переносится без поправок на GQA, MoE, bias и дополнительные нормы.

На учебной конфигурации A1 ($V=10\,000$, $D=512$, $F=1344$, $H=16$, $L=4$)
один блок содержит примерно $3.11$ млн параметров, четыре блока — $12.45$ млн.
Untied LM head добавляет ещё $5.12$ млн; поэтому формулировка handout
«около 17 млн non-embedding parameters» согласуется с блоками плюс выходной
проекцией, а не означает 17 млн параметров вообще.

<a id="compute-accounting"></a>

## FLOPs: сначала одна матрица, потом весь граф

Для произведения $[m,k]\times[k,n]$ обычно считают примерно $2mkn$ FLOPs:
каждый элемент результата требует $k$ умножений и почти столько же сложений.
Это соглашение, а не физический закон; прежде чем сравнивать числа из разных
источников, надо проверить, считает ли каждый fused multiply-add за одну или две
операции.

Для $N=BT$ token-позиций четыре attention-проекции требуют примерно
$8ND^2$ FLOPs. Три матрицы SwiGLU — $6NDF$. Два матричных произведения внутри
обычного multi-head attention дают ещё

$$4BT^2D.$$

Именно этот член квадратичен по длине контекста. При коротком $T$ и широкой
модели проекции и FFN могут доминировать; при росте $T$ attention-score/value
становятся всё заметнее. LM head стоит примерно $2NDV$, что особенно существенно
при большом словаре и untied выходной матрице.

Для dense Transformer часто используют грубое правило обучения
$C\approx6P_{\text{active}}\,N_{\text{tokens}}$: forward линейных слоёв даёт
порядок $2PN$, backward — ещё примерно вдвое больше. Это полезная оценка порядка,
но не точный чек: attention-член, embeddings, recomputation, sparsity, padding и
конкретное определение FLOP нарушают её предпосылки.

<a id="memory-accounting"></a>

## Память: что живёт долго, а что зависит от пакета

К числу параметров нельзя просто приписать один dtype. Во время обучения могут
одновременно существовать low-precision weights, gradients, FP32 master weights
и два момента AdamW. Частая оценка mixed-precision AdamW — около 16 байт на
параметр, но она верна лишь для конкретного размещения этих копий. Fused
optimizer, sharding и отказ от master copy меняют коэффициент.

Активации имеют другую природу. Их объём растёт с $B$, $T$, $D$, $F$ и $L$;
часть промежуточных значений нужна для обратного прохода. Пересчёт активаций
(activation checkpointing) уменьшает число сохранённых значений, но требует
повторного прямого прохода. Накопление градиента увеличивает эффективный пакет,
не заставляя одновременно держать все его микропакеты, однако не уменьшает
состояние модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/compute-memory.png]]

*Смотрите не на размеры прямоугольников, а на узкий канал между памятью и
вычислителем: операция может ждать байты даже при свободных арифметических
блоках. Оригинальный кадр Stanford CS336 Lecture 1/2 из зафиксированной версии
`8b59b50730766695c2ffedd1a79c50cd09b9eb91`.*

Отсюда следует практический порядок: отдельно оценить model state, отдельно
пиковые активации и только затем добавить служебную память allocator и временные
буферы. Проверка — измеренный `max_memory_allocated` в том же dtype, с теми
же $B,T$ и режимом пересчёта активаций, которые указаны в расчёте.

<a id="arithmetic-intensity"></a>

## Почему одинаковое число FLOPs выполняется с разной скоростью

Время ограничивается не только пиковыми FLOP/s, но и bandwidth. Арифметическая
интенсивность операции

$$I=\frac{\text{FLOPs}}{\text{bytes moved}}$$

сравнивает полезную арифметику с трафиком памяти. Ниже — четыре состояния одного
официального edtrace-вывода Stanford CS336. Это не четыре независимые картинки,
а один аргумент: от поэлементной операции к большому matrix multiplication.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/arithmetic-intensity-01-relu.png]]

*Кадр 1. Для ReLU в принятой в лекции модели $I\approx1/4$: байты доминируют над
арифметикой. Сверьте числитель и знаменатель, а не слово `memory-bound`.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/arithmetic-intensity-02-matvec.png]]

*Кадр 2. Даже более дорогая поэлементная функция и первый matrix-vector пример
остаются далеко ниже интенсивности ускорителя.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/arithmetic-intensity-03-square-matvec.png]]

*Кадр 3. У square matrix-vector product $I\approx1$: веса читаются ради одного
вектора. Это объясняет, почему decode с маленьким batch часто memory-bound.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/arithmetic-intensity-04-matmul.png]]

*Кадр 4. Matrix multiplication переиспользует элементы для многих dot products;
в примере $I\approx n/3$ и операция пересекает условную границу compute-bound.
Вывод зависит от dtype, формы и конкретного ускорителя.*

Roofline объединяет эти два потолка, но аналитическая классификация остаётся
гипотезой. Низкая реальная производительность может происходить из-за launch
latency, unfused kernels, плохого tiling, communication или padding — причин,
которых в отношении FLOPs/bytes нет.

<a id="a1-construction"></a>

## От базовых операций к языковой модели: условия сборки A1

Официальный Assignment 1 строит модель в порядке зависимостей: linear и
embedding; RMSNorm и SiLU/SwiGLU; RoPE; стабильный softmax и scaled dot-product
attention; causal multi-head attention; блок; полная LM; затем подсчёт
ресурсов. Порядок важен: на каждом шаге можно зафиксировать форму входа, выхода
и имя проверяющей функции-переходника до того, как ошибка спрячется внутри всей
модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/transformer-architecture.png]]

*Сначала проследите две внешние формы $[B,T,D]$ на правой схеме, затем residual
стрелки и лишь после этого левый путь от token IDs к logits. Оригинальный кадр
Stanford CS336 Lecture 1 из зафиксированной версии `8b59b50730766695c2ffedd1a79c50cd09b9eb91`.*

Полная десятикадровая последовательность ниже сохраняет ход исходного задания,
но не содержит готового кода решения. В каждом кадре ищите новое условие и его
стык с предыдущим.

<details>
<summary>Открыть 10 исходных кадров Assignment 1, страницы 18–27</summary>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-01-linear-embedding.png]]

*1/10: linear задаёт ориентацию $W$, embedding превращает IDs $[B,T]$ в
$[B,T,D]$.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-02-embedding-rmsnorm.png]]

*2/10: выход embedding становится входом следующего шага, затем вводится pre-norm-путь.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-03-rmsnorm.png]]

*3/10: RMSNorm сохраняет форму и нормализует по последней оси.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-04-silu.png]]

*4/10: SiLU вводится до gated FFN, поэтому нелинейность можно проверить отдельно.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-05-swiglu.png]]

*5/10: SwiGLU добавляет третью матрицу и временную ось $F$.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-06-rope.png]]

*6/10: RoPE вращает пары координат Q/K и не меняет форму тензора.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-07-softmax-attention.png]]

*7/10: стабильный softmax предшествует scaled dot-product attention.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-08-multihead-attention.png]]

*8/10: causal mask, разделение голов и выходная проекция образуют единое MHA.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-09-block-lm.png]]

*9/10: два pre-norm residual sublayers складываются в блок, блоки — в LM.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/a1-sequence-10-resource-accounting.png]]

*10/10: только после фиксации архитектуры считаются параметры и FLOPs.*

</details>

<a id="evidence-boundary"></a>

## Где заканчивается формула и начинается эксперимент

**Model FLOPs Utilization (MFU)** сопоставляет скорость полезной работы,
следующей из формулы модели, с паспортным пиком ускорителей:

$$
\operatorname{MFU}
=\frac{F_{model/step}/t_{step}}
{N_{device}\,P_{peak}}.
$$

Здесь $F_{model/step}$ — оценённые FLOPs модели на шаг, $t_{step}$ — измеренное
время шага, а $P_{peak}$ — паспортные FLOP/s одного ускорителя.

Числитель здесь не обязан быть показанием аппаратного счётчика. Обычно сначала
выводят FLOPs forward/backward для выбранной архитектуры, умножают на реально
обработанное число токенов и делят на wall-clock time после прогрева. Знаменатель
берут из спецификации устройства для **той же** точности и режима: dense и
sparse Tensor Core peak, например, различаются. В примере Lecture 2 CS336 MFU
записан как `actual FLOP/s / promised FLOP/s`; перед этой формулой на матричном
умножении отдельно показывается, как получить обе величины.

MFU отвечает на узкий вопрос: какая доля теоретического пика превращается в
арифметику самой модели. Время обменов, запуска kernels, чтения памяти и простоя
уменьшает отношение, хотя эти операции необходимы для запуска. Это отличает
MFU от **hardware FLOPs utilization (HFU)**: HFU может учитывать фактически
исполненные аппаратные операции, включая повторные вычисления при activation
checkpointing, тогда как MFU оставляет в числителе аналитическую работу модели.
Поэтому checkpointing способен увеличить HFU, не меняя полезные model FLOPs.

Высокий MFU сам по себе не означает меньшего времени ответа или лучшей модели.
Можно повысить долю матричных операций, но ухудшить latency, использовать больше
устройств или обработать меньше полезных токенов. Вместе с MFU нужно сообщать
tokens/s, форму batch и sequence, число устройств, dtype, режим sparsity,
checkpointing и правило подсчёта FLOPs.

Три типа утверждений требуют разного подтверждения.

- **Тождество форм.** Проверяется размерностями, unit-тестом и сравнением с
  эталонной функцией на маленьком тензоре.
- **Аналитическая оценка.** Проверяется явным соглашением о FLOP, параметрах,
  dtype и сохранённых активациях.
- **Системный результат.** Требует описания оборудования, версий программ,
  форм тензоров, прогрева, числа повторов и исходных измерений. Чужой MFU или
  runtime не является свойством архитектуры.

Числа из Assignment 1 — критерии прохождения конкретного курса, а не обещание
для любого ноутбука или GPU. В частности, 20–30 минут на одном B200 и наблюдения
для M4 Max относятся к реализации авторов курса и заданной конфигурации. Их
можно использовать как ориентир воспроизводимости, но нельзя подменять ими
собственные измерения времени и пиковой памяти.

<a id="exercise-transition"></a>

## Упражнение: предсказать измерение до запуска

Для выбранных $B,T,V,D,H,F,L$ составьте таблицу форм каждого промежуточного
тензора, выведите число параметров с явным решением о weight tying и оцените
FLOPs forward. Затем измените только $T$ вдвое: какие члены выросли вдвое, а
какие — вчетверо? Наконец сравните прогноз с profiler trace и объясните хотя бы
одно расхождение.

Следующий шаг — не писать ещё одну формулу, а реализовать те же условия в
проверяемом порядке. Для этого предназначен
[[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля|Capstone 1]].
Он закрепляет точную версию Assignment 1, официальные функции-переходники и
тесты, локальный и полный профили, схему отчёта и критерии остановки.

## Источники

- [Stanford CS336 Spring 2026, Lecture 2 — pinned source](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py).
- [Stanford CS336 Spring 2026, Assignment 1 — pinned handout](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf).
- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
