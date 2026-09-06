---
title: "FlashAttention: точное внимание с меньшим обменом памятью"
type: textbook-chapter
status: canonical
last_updated: 2026-09-06
source_unit_id:
  - lecture-05-performance-recap-transition
  - lecture-05-flashattention-derivation
  - lecture-05-recap
  - assignment-02-task-pytorch-attention
  - assignment-02-flashattention-online-softmax
  - assignment-02-task-flash-forward
  - assignment-02-flashattention-tests
  - assignment-02-task-flash-backward
primary_sources:
  - https://arxiv.org/abs/2205.14135
  - https://crfm.stanford.edu/2023/07/17/flash2.html
  - https://github.com/Dao-AILab/flash-attention
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
---

<a id="cs336-flashattention"></a>

# FlashAttention: точное внимание с меньшим обменом памятью

Формула внимания содержит два матричных умножения и softmax:

$$
S=\frac{QK^\top}{\sqrt d},\qquad P=\operatorname{softmax}(S),\qquad O=PV.
$$

На бумаге естественно сначала получить $S$, затем $P$, а потом умножить $P$ на
$V$. На GPU такая последовательность заставляет записать квадратные матрицы
$S$ и $P$ в основную память устройства и вскоре прочитать их обратно. Для
длинного контекста время тратится не только на арифметику, но и на перенос этих
промежуточных данных. FlashAttention сохраняет ту же математическую операцию,
но меняет порядок вычисления: полные $S$ и $P$ вообще не материализуются.
На-chip storage содержит только временную плитку и компактное состояние
online-softmax, после чего плитка заменяется следующей.

## Вычислительная сложность не объясняет время работы

Стандартное внимание требует $O(N^2d)$ операций для последовательности длины
$N$ и головы размера $d$. По этой записи нельзя узнать, сколько раз данные
переедут между уровнями памяти.

У ускорителя есть большая, но относительно медленная HBM и гораздо меньшая
on-chip SRAM. Матричные ядра могут считать быстрее, чем HBM подаёт им данные.
Если алгоритм постоянно записывает и читает промежуточные тензоры, он становится
ограничен обменом памятью даже при неизменном числе FLOPs.

Исходная работа FlashAttention называет такой анализ **IO-aware**: алгоритм
проектируется с учётом числа чтений и записей между HBM и SRAM, а не только
асимптотики арифметических операций.

## Обычная реализация материализует квадратные тензоры

Пусть $Q,K,V\in\mathbb R^{N\times d}$. Реализация из отдельных операторов
обычно выполняет:

1. прочитать $Q$ и $K$, записать $S=QK^\top$ размером $N^2$;
2. прочитать $S$, вычислить построчный softmax, записать $P$ размером $N^2$;
3. прочитать $P$ и $V$, записать $O$ размером $Nd$.

При $N=8192$ одна матрица FP16 размером $N^2$ занимает 128 МиБ **на одну
голову и один пример**, ещё до служебных буферов и обратного прохода. Даже если
фреймворк объединяет часть операций, материализация $P$ создаёт квадратичный
след памяти.

FlashAttention не делает внимание разреженным и не удаляет пары токенов. Он
избегает записи полной $S$ и $P$ в HBM.

## Блочный проход через SRAM

Переход от обычного attention к FlashAttention удобнее видеть в двух кадрах.
Первый фиксирует проблему: отдельные operators записывают attention matrix в
HBM. Второй вводит состояние online softmax, которое позволяет завершать output
по tiles.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p52.png]]

*Stanford CS336 Lecture 5, p. 52, по материалам FlashAttention: обычный путь
между HBM и SRAM материализует крупные промежуточные matrices. [Pinned PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf).
Рисунок задаёт IO-проблему; он не означает, что HBM — единственный возможный
bottleneck.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p53.png]]

*Stanford CS336 Lecture 5, p. 53: online softmax переносит между key tiles
running maximum, denominator и накопленный числитель. Эти состояния заменяют
полную probability matrix, но не устраняют попарные Q–K вычисления.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p54.png]]

*Stanford CS336 Lecture 5, p. 54: последовательность замыкается полным
tile-wise forward — score tile, локальная fusion и online-softmax state. Задача
кадра — связать две предыдущие идеи до перехода к точному алгоритму Assignment
2. [Pinned PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/gpu-memory-hierarchy.svg]]

*Harvard ML Systems, Vol. II, `performance_engineering.qmd`: on-chip SRAM мала, но существенно быстрее HBM; [оригинальный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/gpu-memory-hierarchy.svg), CC BY-NC-SA 4.0.*

Большая схема Tri Dao на странице Stanford CRFM показывает весь forward pass.
Блоки $K_j,V_j$ поочерёдно загружаются из HBM в SRAM. Для каждого блока $Q_i$
вычисляется небольшая плитка $Q_iK_j^\top$, сразу применяется локальный softmax
и обновляется соответствующий блок выхода. В HBM остаются $Q,K,V,O$ и несколько
построчных статистик, но не полная матрица внимания.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/flashattention-tiling.png]]

*Forward pass FlashAttention: внешние циклы по блокам $K,V$ и $Q$, локальная
матрица $S_{ij}$ в SRAM и обновление статистик softmax. Оригинальная схема Tri
Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work
Partitioning](https://crfm.stanford.edu/2023/07/17/flash2.html).*

Размер плитки выбирают так, чтобы $Q_i,K_j,V_j$, локальные scores и выход
помещались в SRAM. Одни блоки $Q$ приходится читать несколько раз, но это дешевле
записи и повторного чтения двух полных $N\times N$ матриц.

## Как softmax считается по частям

Softmax одной строки требует знать сумму экспонент всех её элементов. Наивно это
мешает завершить блок до просмотра остальных ключей. Решение — онлайн-softmax.

Для уже обработанной части строки хранят максимум $m$ и сумму

$$
\ell=\sum_j e^{s_j-m}.
$$

Пусть новый блок имеет максимум $m_b$ и нормировочную сумму $\ell_b$. Общие
статистики равны

$$
m'=\max(m,m_b),
$$

$$
\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b.
$$

Накопленный числитель выхода масштабируется теми же коэффициентами:

$$
u'=e^{m-m'}u+e^{m_b-m'}u_b,
\qquad o=\frac{u'}{\ell'}.
$$

Таким образом, каждый новый блок корректирует старый вклад, как если бы максимум
и знаменатель были известны с самого начала. Это не приближение softmax; меняется
ассоциативный порядок операций. Из-за конечной точности результат может
отличаться последними битами, как отличаются две допустимые редукции на GPU.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-p25-flash-forward.png]]

*Полный forward schedule из [Stanford CS336 Assignment 2, Algorithm 1, p. 25](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf),
commit `ca8bc81`. Внешний цикл фиксирует query tile, внутренний перебирает key/value
tiles; строки 10–13 обновляют running maximum, denominator и ненормированный
output. Crop сохраняет исходный алгоритм без перерисовки.*

Финальный forward записывает $O_i$ и log-sum-exp

$$L_i=m_i+\log \ell_i.$$

$L$ понадобится backward: по нему любая вероятность восстанавливается из
локально пересчитанного score без сохранения всей строки softmax.

## Почему память становится линейной

Квадратичная **арифметика** остаётся: каждая строка $Q$ по-прежнему
взаимодействует с каждой строкой $K$. Но квадратная матрица scores не хранится в
HBM. Для batch $B$, heads $H$, sequence $N$ и head dimension $d$ входы
$Q,K,V$ и output $O$ занимают $O(BHNd)$ элементов, а статистики softmax —
$O(BHN)$. При фиксированных $B,H,d$ это линейно по $N$, но полная запись
важна: фраза “$O(N)$ памяти” не означает, что head dimension или batch исчезли.

В обратном проходе стандартная реализация могла бы сохранить $P$ с forward.
FlashAttention вместо этого сохраняет компактные нормировочные статистики и
повторно вычисляет локальные плитки. FLOPs становится немного больше, зато
исчезает дорогое чтение сохранённой квадратной матрицы. На GPU дополнительная
арифметика может быть дешевле дополнительного трафика HBM.

## Backward: что требуется пересчитать

Сначала полезно выписать обычный backward. Пусть $dO$ пришёл из следующей части
графа. Тогда

$$
dV=P^\top dO,
\qquad dP=dOV^\top,
$$

$$
dS_i=P_i\odot\left(dP_i-\sum_j P_{ij}dP_{ij}\right),
$$

$$
dQ=\frac{dSK}{\sqrt d},
\qquad dK=\frac{dS^\top Q}{\sqrt d}.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-p23-attention-equations.png]]

*Stanford CS336 Assignment 2, p. 23, equations 4–11: обычный forward/backward
явно требует $P$; эта dependency и заставляет стандартную реализацию сохранять
квадратный tensor.*

FlashAttention заменяет row-wise сумму в Jacobian softmax на компактную
статистику

$$
D_i=\sum_k O_{ik}\,dO_{ik}
=\sum_j P_{ij}\,dP_{ij}.
$$

Равенство следует из $O=PV$ и $dP=dOV^\top$. Теперь для каждой плитки можно
пересчитать

$$
S_{ij}=\frac{Q_iK_j^\top}{\sqrt d},
\qquad P_{ij}=\exp(S_{ij}-L_i),
$$

а затем локально получить

$$
dV_j\mathrel{+}=P_{ij}^\top dO_i,
\qquad dP_{ij}=dO_iV_j^\top,
$$

$$
dS_{ij}=P_{ij}\odot(dP_{ij}-D_i),
$$

$$
dK_j\mathrel{+}=\frac{dS_{ij}^\top Q_i}{\sqrt d},
\qquad
dQ_i\mathrel{+}=\frac{dS_{ij}K_j}{\sqrt d}.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-p24-backward-recompute.png]]

*Stanford CS336 Assignment 2, p. 24, equations 13–19: $L$ восстанавливает
probability tile, а $D=\operatorname{rowsum}(O\odot dO)$ устраняет повторную
полную softmax reduction. Важно читать этот кадр после обычного backward выше.*

### Почему в алгоритме два tiled прохода

$dK_j$ и $dV_j$ получают вклады от всех query tiles. Поэтому первый проход
фиксирует key/value tile $j$, проходит все $i$ и накапливает его gradients до
одной записи. $dQ_i$, напротив, получает вклады от всех key tiles; второй
проход фиксирует query tile и накапливает $dQ_i$. Такая перестановка циклов
уменьшает число глобальных записей и избегает atomics для основных
accumulators, но повторяет score/probability tiles.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-p29-flash-backward.png]]

*Stanford CS336 Assignment 2, Algorithm 2, p. 29: первый nested loop собирает
$dK,dV$, второй — $dQ$. Показан исходный алгоритм целиком; это спецификация
dependencies, а не готовое решение задания.*

### Causal mask в неполной плитке

Для query position $q$ значения с key position $k>q$ должны давать нулевую
вероятность. В forward соответствующие scores заменяют на $-\infty$ до
online-softmax. В backward та же block/element mask применяется до
восстановления $P_{ij}$; иначе запрещённые позиции получают gradient. Полностью
будущие key tiles можно пропустить, диагональные tiles требуют elementwise
mask, а tail по реальной длине — отдельной bounds mask.

Сравнение с reference должно охватывать causal/non-causal режимы, не кратные
tile sizes lengths, разные head dimensions и forward/backward по всем входам.
Отдельно измеряют forward и backward: одинаковая скорость forward не доказывает
эффективность recomputation schedule.

## Что изменил FlashAttention-2

Первая версия уже сокращала обмен с HBM, но не всегда равномерно занимала
потоковые мультипроцессоры и требовала обмена между warp внутри блока.
FlashAttention-2 улучшил разбиение работы в двух масштабах.

1. Помимо batch и heads, работа делится по длине последовательности. Это важно
   при длинном контексте, когда batch мал и прежнего числа thread blocks не
   хватает для всех SM.
2. Внутри thread block схема `sliced-K` заменена на `sliced-Q`: каждый warp
   получает собственную часть $Q$, а $K,V$ доступны всем. Warp может получить
   свой кусок выхода без редукции промежуточных результатов через shared memory.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/flashattention2-partitioning.png]]

*Слева FlashAttention делит $K,V$ между warp и затем объединяет результаты;
справа FlashAttention-2 делит $Q$, поэтому каждому warp принадлежит независимый
кусок выхода. Оригинальная схема Tri Dao,
[Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html).*

FlashAttention-3 специализирует вычисление под Hopper и использует асинхронность
и низкую точность новых тензорных ядер; FlashAttention-4 в официальном
репозитории реализован на CuTeDSL для Hopper и Blackwell. Номер версии сам по
себе не обещает выигрыш на любом GPU: нужны поддерживаемая архитектура,
подходящие размеры головы, тип данных, маска и фактически выбранное ядро.

## Обучение, prefill и decode

На обучении и длинном prefill много строк $Q$, поэтому блочный алгоритм хорошо
использует матричные ядра и устраняет крупные промежуточные тензоры. При decode
обычно имеется одна новая строка $Q$, а $K,V$ читаются из растущего KV-cache.
Это другой геометрический режим. Для него используются FlashDecoding и
специализированные kernels, которые параллельно делят длинный KV-cache и
объединяют частичные результаты.

Следовательно, фраза «модель использует FlashAttention» недостаточна для
описания serving. Нужно уточнить kernel для prefill, kernel для decode, формат и
размещение KV-cache, GQA/MQA и scheduler.

## FlashAttention не следует путать с архитектурой внимания

| механизм | что меняет | результат относительно dense attention |
|---|---|---|
| FlashAttention | порядок вычислений и движение данных | то же dense attention |
| sliding-window attention | множество видимых ключей | другая маска и меньшая работа |
| block-sparse attention | набор вычисляемых блоков | приближённое/структурно иное внимание |
| linear attention | саму формулу агрегации | другая архитектура |
| PagedAttention | физическое размещение KV-cache | те же логические ключи и значения |

FlashAttention может реализовывать causal или sliding-window mask, но
эффективность kernel и архитектурное ограничение контекста остаются разными
решениями.

## Как понять, что kernel действительно используется

FlashAttention — частный случай слияния операций: промежуточный тензор выгодно
использовать сразу, не записывая его в HBM. Тот же принцип применяется к QKV,
RoPE, cross-entropy, RMSNorm и SwiGLU. Компилятор может объединить поэлементные
операции, Triton — выразить специализированное ядро, а библиотека — предоставить
вручную настроенную реализацию. Во всех случаях фактический путь нужно
подтвердить профилировщиком; методика такой проверки разобрана в
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|главе о профилировании]].

На prefill много строк Q и крупные tiles загружают tensor cores. На decode новая Q обычно одна на sequence, а paged KV читается из HBM; нужен decode/paged-attention kernel. Ускорение prefill не доказывает улучшение TPOT.

В PyTorch функция
[`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
может выбрать flash, memory-efficient или math backend. Выбор зависит от
устройства, dtype, формы и параметров. Отсутствие ошибки не доказывает, что
сработал flash backend: реализация может незаметно перейти на fallback.

Проверка включает:

- profiler с именем фактически вызванного kernel;
- пиковую выделенную память при нескольких $N$;
- время forward и backward отдельно;
- одинаковые mask, dropout, dtype и размеры головы;
- сравнение выхода и градиента с reference в допустимой численной погрешности.

Бенчмарк одной attention-операции не равен ускорению всей модели. Если большую
часть времени занимают MLP, коммуникации или чтение KV-cache на decode, общий
выигрыш будет меньше локального.

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/lecture.pdf|Efficient DL Systems — обмен активациями и fused kernels]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/seminar/practice.ipynb|Практикум Efficient DL Systems по FlashAttention]].
- [[02 Areas/ML & DL/Papers/Flash Attention|Карточка FlashAttention]].
- [[02 Areas/ML & DL/Papers/Flash Attention 2|Карточка FlashAttention-2]].

- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — IO-aware постановка, алгоритм и доказательство точности.
- Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691) — work partitioning and parallelism improvements.
- Tri Dao, [FlashAttention-2 at Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html) — наиболее наглядные схемы tiling и разбиения warp.
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference implementation, ограничения и тесты численной корректности.
- PyTorch, [Scaled Dot Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — выбор backend в практическом API.
- [Stanford CS336 Lecture 5, pinned `8b59b507`, pp. 50–54](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf) — IO transition and online-softmax visual sequence.
- [Stanford CS336 Assignment 2, pinned `ca8bc81`, pp. 23–31](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf) — forward/backward equations, Algorithms 1–2, official tests and benchmark contract.

← [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines]] ·
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Квантизация языковых моделей]] →
