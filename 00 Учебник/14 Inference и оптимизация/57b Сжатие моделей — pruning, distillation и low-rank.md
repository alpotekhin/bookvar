---
title: Сжатие моделей — pruning, distillation и low-rank
type: textbook-chapter
status: canonical
last_updated: 2026-10-29
last_verified: 2026-10-29
source_language: mixed
source_unit_id:
  - lecture-10-model-pruning-distillation
  - lecture-10-figure-step-325-rendering-1
  - lecture-10-figure-step-331-rendering-1
primary_sources:
  - https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/model_compression/model_compression.qmd
  - https://arxiv.org/abs/2407.14679v2
  - https://arxiv.org/abs/2306.11695v1
  - https://arxiv.org/abs/1503.02531
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
---

# Сжатие моделей: pruning, distillation и low-rank

Плотная матрица $4096\times4096$ содержит 16 777 216 весов и занимает
32 MiB в BF16. Можно занулить половину элементов, но обычное плотное
умножение по-прежнему прочитает все 32 MiB. Если вместо этого удалить
половину входных каналов и согласованно изменить соседний слой,
поменяются и размер матрицы, и число выполняемых умножений.

Есть и другие варианты: приблизить матрицу произведением двух меньших
или обучить небольшую модель по ответам большой.
[[57 Квантизация языковых моделей|Квантизация]] уменьшает число битов
на элемент; эти методы меняют набор параметров, структуру вычислений
или саму обучаемую модель. Уменьшение файла, объёма памяти и задержки
нужно оценивать раздельно.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/5b8ca5d44a1556b84569cdc9156a94c4b1500e92.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems — Optimization Framework](https://mlsysbook.ai/vol1/model_compression/model_compression.html#sec-model-compression-optimization-framework-9e21),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Исходная схема разделяет изменение представления модели, точности чисел
и аппаратного исполнения. SVG сохранён без изменения.*

## Четыре разных преобразования

| Метод | Что меняется | Как может уменьшиться работа |
|---|---|---|
| Pruning | Отдельные веса или целые нейроны, головы, каналы, слои | Разреженное ядро пропускает нули либо плотные матрицы становятся меньше |
| Low-rank factorization | Одна матрица заменяется несколькими факторами | Умножения выполняются через узкое промежуточное представление |
| Дистилляция | Обучается модель, имитирующая учителя | На запросе исполняется только выбранный ученик |
| Выбор архитектуры | Глубина, ширина, операции и связи | Под ограничения устройства подбирается другая структура |

У методов разная стоимость подготовки. Маску по величине весов можно получить
без обучающих примеров; оценка по активациям требует калибровочных входов;
дистилляция требует обучения ученика и получения сигнала от учителя.
При этом pruning и дистилляция могут входить в один процесс:
обрезанная модель получает начальные веса от исходной, затем восстанавливает
качество по её распределениям. Так построен Minitron.

<a id="pruning-параметр-равен-нулю--но-исчезла-ли-работа"></a>

## Pruning: маска и физическое удаление параметров

Рассмотрим матрицу из восьми весов:

$$
W=\begin{pmatrix}
0.1&-0.4&0.02&2\\
0.8&-0.03&1&0.2
\end{pmatrix}.
$$

При magnitude pruning сохраняют веса с наибольшим модулем.
Чтобы оставить четыре из восьми, удаляем $0.02$, $-0.03$, $0.1$ и $0.2$.
Получается

$$
\hat W=\begin{pmatrix}
0&-0.4&0&2\\
0.8&0&1&0
\end{pmatrix}.
$$

Доля нулей равна 50%, но форма осталась $[2,4]$.
Это *неструктурированное* разреживание: удалены отдельные связи,
а не целые строки или столбцы.

<!-- checked-example: compression-magnitude -->
```python
import numpy as np

weights = np.array([[0.1, -0.4, 0.02, 2.0],
                    [0.8, -0.03, 1.0, 0.2]])
order = np.argsort(np.abs(weights).ravel(), kind="stable")
mask = np.ones(weights.size, dtype=bool)
mask[order[:4]] = False
mask = mask.reshape(weights.shape)
pruned = weights * mask

np.testing.assert_allclose(pruned, [[0, -0.4, 0, 2], [0.8, 0, 1, 0]])
assert np.count_nonzero(pruned) == 4
assert pruned.shape == weights.shape
```

В примере явно выбираются четыре позиции. Один порог по процентилю
при одинаковых величинах весов может оставить не ровно заданную долю:
нужно определить правило обработки равенств. Глобальный порог на всю модель
также может почти полностью удалить слой с небольшими по модулю весами,
сохранив другой слой. Порог на слой или на выходной нейрон задаёт
другое ограничение.

В общей записи используют маску $M$:

$$
\hat W=M\odot W,\qquad M_{ij}\in\{0,1\}.
$$

Хотелось бы выбрать её с минимальным ухудшением функции потерь при заданном
числе ненулевых весов:

$$
\min_{\hat W}\mathcal L(\hat W)
\quad\text{при}\quad \|\hat W\|_0\le k.
$$

Здесь $\|\hat W\|_0$ — количество ненулевых элементов, не обычная норма.
Перебор всех масок непрактичен. Сортировка по модулю — дешёвая эвристика,
которая не вычисляет, как изменится ответ модели.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/fc4813524b57d9439c3a0b9101f2c1f2a79382c3.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems — Unstructured pruning](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-sparse-matrix),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Зануление отдельных связей создаёт разреженную матрицу прежней формы.
Источник изображения не изменён.*

Если обрезанную модель дообучают с фиксированной маской, её продолжают
применять в прямом проходе или явно поддерживают зануление параметров
и соответствующего состояния оптимизатора. Однократное присваивание нуля
не запрещает следующему обновлению снова сделать вес ненулевым.

### Формат хранения и 2:4 sparsity

Плотное хранение исходной и обрезанной матрицы в BF16 занимает одинаковые
$8\cdot2=16$ байт. У разреженного представления появляются индексы.
Например, CSR с четырьмя ненулевыми BF16-значениями и 32-битными индексами
содержит 8 байт значений, 16 байт индексов столбцов и 12 байт указателей строк:
всего 36 байт без прочих накладных расходов.

Для матрицы $m\times n$ такая упаковка занимает

$$
M_{\mathrm{CSR}}=2\,nnz+4\,nnz+4(m+1),
$$

тогда как плотная — $2mn$ байт.
При большом числе элементов даже 50% нулей недостаточно для экономии
в этом конкретном формате. Другие размеры индексов, блочная упаковка
и шаблонная разреженность дают другой результат.

При *2:4 sparsity* в каждой согласованной группе из четырёх весов
оставляют не более двух ненулевых. Например, группа
$(0.1,-0.4,0.02,2)$ может стать $(0,-0.4,0,2)$.
Это сильнее ограничивает выбор маски, чем условие «половина всей матрицы
равна нулю». Зато позиционные метаданные регулярны, и поддерживаемые
разреженные матричные инструкции могут пропускать работу.

Ось группировки, типы данных, размеры и раскладка должны отвечать требованиям
конкретного ядра. Обычный dense GEMM не станет разреженным от появления
такой маски. Даже двукратная пиковая пропускная способность sparse-инструкции
не означает двукратного ускорения всей модели.

### Важность веса зависит от входа: Wanda

Уменьшим линейный слой до одного выхода и двух весов:
$w=(0.05,1)^\top$. На калибровочных входах

$$
X=\begin{pmatrix}100&0\\0&0.1\end{pmatrix},\qquad
Xw=\begin{pmatrix}5\\0.1\end{pmatrix}.
$$

Magnitude pruning удалит первый, меньший вес. Тогда выход станет
$(0,0.1)^\top$, сумма квадратов ошибок равна 25.
Если удалить второй вес, выход будет $(5,0)^\top$,
а ошибка — 0.01. Малый вес оказался связан с крупной активацией.

[Wanda, §3](https://arxiv.org/html/2306.11695v1#S3) учитывает оба множителя.
Для записи $Y=XW$, где $W\in\mathbb R^{d_{\mathrm{in}}\times d_{\mathrm{out}}}$,
оценка связи входа $j$ с выходом $k$ равна

$$
s_{jk}=|W_{jk}|\ \|X_{:,j}\|_2.
$$

В статье матрица весов транспонирована относительно нашей записи:
её строки соответствуют выходам. Поэтому отбор «по строке» из статьи
у нас означает отбор внутри столбца W — среди связей одного выходного нейрона.
Wanda сравнивает оценки локально для каждого выхода, а не одним общим порогом
для всей модели. Для 2:4 сравнение ограничивают соответствующими четвёрками.

<!-- checked-example: compression-wanda -->
```python
calibration = np.array([[100.0, 0.0], [0.0, 0.1]])
w = np.array([0.05, 1.0])
scores = np.abs(w) * np.linalg.norm(calibration, axis=0)
np.testing.assert_allclose(scores, [5.0, 0.1])

by_magnitude = w.copy()
by_magnitude[np.argmin(np.abs(w))] = 0
by_activation = w.copy()
by_activation[np.argmin(scores)] = 0
target = calibration @ w
np.testing.assert_allclose(np.sum((calibration @ by_magnitude - target)**2), 25)
np.testing.assert_allclose(np.sum((calibration @ by_activation - target)**2), 0.01)
```

После разреживания предыдущего слоя в Wanda собираются уже изменившиеся входные
активации следующего слоя. Оставшиеся веса не компенсируются и не дообучаются
самим алгоритмом. Это отличает его от методов послойной реконструкции,
таких как SparseGPT, которые дополнительно корректируют оставшиеся веса;
идея компенсации через квадратичную задачу разобрана на
[[57 Квантизация языковых моделей#GPTQ|примере GPTQ]].

Калибровочные входы не описывают все будущие запросы. В нашем маленьком примере
при входе $(0,1000)$ удаление второго веса даст большую ошибку.
Оценка по активациям объясняет выбор на известных данных, а не гарантирует
сохранение любого поведения модели.

### Неструктурированное и структурное разреживание

При удалении нейрона нужно согласовать два соседних преобразования.
Пусть

$$
\begin{aligned}
H&=\operatorname{ReLU}(XW_1+b_1),\\
Y&=HW_2+b_2,
\end{aligned}
$$

где $W_1:[2,4]$, $b_1:[4]$, $W_2:[4,2]$, $b_2:[2]$.
Четыре компоненты H — четыре скрытых нейрона.
Если оставить нейроны 0 и 2, из $W_1$ удаляются столбцы 1 и 3,
из $b_1$ — те же элементы, из $W_2$ — строки 1 и 3.
Ширина скрытого представления уменьшается с четырёх до двух.

| Параметр | До удаления | После удаления |
|---|---:|---:|
| $W_1$ | $[2,4]$: 8 весов | $[2,2]$: 4 веса |
| $b_1$ | 4 | 2 |
| $W_2$ | $[4,2]$: 8 весов | $[2,2]$: 4 веса |
| $b_2$ | 2 | 2 |
| Всего параметров | 22 | 12 |

Для входа $(1,2)$ в коде ниже $H$ равно $(2,2.5,1,3.8)$.
После удаления остаются $(2,1)$, а выход равен $(2.1,0.8)$.
Физически уменьшенная модель должна совпасть с моделью, в которой эти
скрытые активации занулены маской. Совпадение с исходной полной моделью
из этого не следует.

<!-- checked-example: compression-structured -->
```python
x = np.array([[1.0, 2.0], [-1.0, 1.0]])
w1 = np.array([[1.0, 0.1, -1.0, 0.2],
               [0.5, 0.2, 1.0, 0.3]])
b1 = np.array([0.0, 2.0, 0.0, 3.0])
w2 = np.array([[1., 0.], [1., 1.], [0., 1.], [2., -1.]])
b2 = np.array([0.1, -0.2])
keep = [0, 2]
hidden = np.maximum(x @ w1 + b1, 0)
hidden_mask = np.zeros(4)
hidden_mask[keep] = 1
masked_output = (hidden * hidden_mask) @ w2 + b2
compact_output = np.maximum(x @ w1[:, keep] + b1[keep], 0) @ w2[keep, :] + b2

np.testing.assert_allclose(hidden[0], [2., 2.5, 1., 3.8])
np.testing.assert_allclose(compact_output, masked_output)
np.testing.assert_allclose(compact_output[0], [2.1, 0.8])
assert w1.size + b1.size + w2.size + b2.size == 22
assert w1[:, keep].size + b1[keep].size + w2[keep, :].size + b2.size == 12
```

Занулить только столбцы $W_1$ недостаточно: в удаляемом канале может остаться
ненулевой bias. В многослойной сети согласуют все потребители удаляемого
представления. Для SwiGLU это обе входные проекции и выходная проекция;
для attention — соответствующие проекции голов и выходная матрица.
Изменение общей ширины остаточного потока затрагивает ещё больше матриц,
эмбеддинги и нормализации.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/4000264baf2311c2b61ca0ddd3fcedaa9d047ed3.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems — Channel vs. Layer Pruning](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-channel-layer-pruning),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
На исходной схеме CNN слева после удаления канала меняются формы соседних
свёрток; справа после удаления слоя заново соединяются оставшиеся блоки.
Это тот же принцип согласования потребителей, что и у двух матриц MLP.*

При pruning целых блоков трансформера меняется и глубина преобразования.
Даже если остаточные соединения позволяют соединить соседние блоки
без ошибки размеров, их состояния могут иметь другое распределение.
Обычно качество восстанавливают дообучением, иногда после нескольких
небольших этапов удаления. Порядок удаления и калибровочные данные
становятся частью метода.

<!-- systems-completion: pruning:start -->
### Prune, recover, and reset: three different training procedures

The mask specifies which connections remain. It does not specify how to obtain
the values on those connections. One-shot pruning, iterative fine-tuning and
lottery-ticket experiments make different choices here.

In one-shot pruning, the target mask is selected once. The surviving weights
may then be fine-tuned; “one-shot” describes mask selection, not necessarily
the absence of recovery training. Iterative pruning alternates smaller cuts
with recovery. After recovery, importance is measured again because the
surviving weights and activations have changed. This costs additional training
and does not guarantee that every iterative schedule beats every one-shot method.

Removing 20% of the **remaining** connections on each round is not the same as
adding 20 percentage points of sparsity. Starting from 100 connections gives
80, then 64, then approximately 51: about 49% sparsity after three rounds.
Ignoring integer rounding, after $r$ rounds with removal fraction $p$:

$$
N_r=N_0(1-p)^r,\qquad s_r=1-(1-p)^r.
$$

To reach 90% sparsity in ten equal multiplicative rounds, use
$p=1-0.1^{1/10}\approx0.2057$, not 9% per round. An implementation must
also specify rounding and whether the budget is global or per layer.

The lottery-ticket procedure asks a different question: can the selected sparse
network train from its **original initialization**, rather than continue from
its already trained weights? Save $W_0$, train to obtain $W_T$, and derive a
mask $M$ from $W_T$. Fine-tuning starts from $M\odot W_T$; resetting starts
from $M\odot W_0$. Reinitializing with a new random seed is a third experiment.
The mask, parameter values and optimizer state must not be conflated.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/28c3ef541fd4ed95724389fad083bb720b229e40.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems, Lottery Ticket Iteration Cycle](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-winning-ticket),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Complete original SVG. Follow the left-hand loop: train, prune, reset surviving
weights, train again. The highlighted network on the right is a candidate
subnetwork; the drawing is not evidence that an arbitrary mask is a winning ticket.*

For example, suppose $W_0=(0.1,-0.2,0.3,-0.4)$ and training produces
$W_T=(2,0.01,-3,0.02)$. Keeping the two largest magnitudes gives mask
$(1,0,1,0)$. Fine-tuning starts with $(2,0,-3,0)$, while resetting starts
with $(0.1,0,0.3,0)$. These have the same sparsity and different functions.

<!-- checked-example: compression-pruning-schedule -->
```python
import numpy as np

remaining = [100]
for _ in range(3):
    remaining.append(remaining[-1] - round(0.2 * remaining[-1]))
assert remaining == [100, 80, 64, 51]
p = 1 - 0.1**(1/10)
np.testing.assert_allclose((1-p)**10, 0.1)
initial = np.array([0.1, -0.2, 0.3, -0.4])
trained = np.array([2., 0.01, -3., 0.02])
mask = np.array([True, False, True, False])
np.testing.assert_allclose(mask * trained, [2, 0, -3, 0])
np.testing.assert_allclose(mask * initial, [0.1, 0, 0.3, 0])
assert not np.array_equal(mask * initial, mask * trained)
```

The original [Lottery Ticket Hypothesis paper](https://arxiv.org/abs/1803.03635)
tests trainability and accuracy under specified training conditions. Finding
a useful mask after dense training does not recover the cost of that dense
training. Nor does resetting establish a universal result for arbitrary LLMs,
sparsity levels or optimizers. *Weight rewinding* restores an early training
checkpoint rather than step zero; it must name that checkpoint and the learning
rate schedule used when retraining.

There are also two distinct uses of “dynamic sparsity.” During sparse training,
a method may remove connections and grow others; the mask changes while learning.
At inference, input-dependent routing may activate different channels or blocks
for different examples. A gradually increasing pruning target alone does not
imply regrowth, and zeroing a computed activation does not save the work already
spent computing it. Saving that work requires a gate available **before** the
skipped operation and an execution path that actually omits it.

Harvard's deployment distinction is retained in its original language below;
source: [Pruning in practice, paragraph at line 1821](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/model_compression/model_compression.qmd#L1821),
CC BY-NC-SA 4.0. The numerical examples above are Bookvar additions.

<!-- completion-excerpt: pruning-artifact:start -->
> These trade-offs become concrete when examining real-world deployments. Some model families reduce deployment cost through architecture rather than post-hoc pruning: MobileNet uses depthwise separable convolutions for mobile and embedded vision, while EfficientNet uses compound scaling to improve the accuracy-efficiency trade-off under resource constraints. Pruning remains a separate optimization lever.
<!-- completion-excerpt: pruning-artifact:end -->

This excerpt omits the source's citation markup and stops before its BERT example.
The original references are [MobileNet](https://arxiv.org/abs/1704.04861) and
[EfficientNet](https://arxiv.org/abs/1905.11946). A model built with depthwise
convolutions is not, for that reason alone, a pruned dense convolutional model.

<!-- systems-completion: pruning:end -->

## Low-rank factorization: заменить одну матрицу двумя

Матрица

$$
W=\begin{pmatrix}3&1\\1&3\end{pmatrix}
$$

действует по-разному на два направления.
Вектор $(1,1)$ она умножает на 4, а $(1,-1)$ — на 2.
Если сохранить только первое направление, получится матрица ранга 1:

$$
W_1=\begin{pmatrix}2&2\\2&2\end{pmatrix}.
$$

Теперь $(1,1)W_1=(4,4)$, как и раньше, но
$(1,-1)W_1=(0,0)$ вместо $(2,-2)$.
Усечение не удаляет отдельные маленькие веса: оно исключает целое
направление линейного преобразования.

Для $W\in\mathbb R^{m\times n}$ сингулярное разложение имеет вид

$$
W=U\Sigma V^\top,\qquad
\sigma_1\ge\sigma_2\ge\ldots\ge0.
$$

Оставив $r$ старших компонент, получают

$$
W_r=U_r\Sigma_rV_r^\top=AB,
$$

где $A:[m,r]$, $B:[r,n]$.
Например, можно включить $\sqrt{\Sigma_r}$ в оба фактора.
Теорема Эккарта — Янга задаёт оптимальность усечения по ошибке матрицы
в норме Фробениуса; для неё

$$
\|W-W_r\|_F^2=\sum_{i>r}\sigma_i^2.
$$

В нашем примере $\sigma=(4,2)$, поэтому квадрат ошибки равен 4.
Доля сохранённой суммы квадратов сингулярных чисел равна
$16/(16+4)=80\%$, но на входе $(1,-1)$ потерян весь исходный выход.
«Сохранено 80% энергии» не означает «сохранено 80% качества модели».

<!-- checked-example: compression-svd -->
```python
w = np.array([[3.0, 1.0], [1.0, 3.0]])
u, singular, vt = np.linalg.svd(w, full_matrices=False)
rank = 1
a = u[:, :rank] * np.sqrt(singular[:rank])
b = np.sqrt(singular[:rank])[:, None] * vt[:rank, :]
approximation = a @ b
np.testing.assert_allclose(singular, [4, 2])
np.testing.assert_allclose(approximation, [[2, 2], [2, 2]])
np.testing.assert_allclose(np.sum((w - approximation)**2), 4)
np.testing.assert_allclose(np.array([1., -1.]) @ approximation, 0, atol=1e-14)
np.testing.assert_allclose(np.array([1., -1.]) @ w, [2, -2])
```

Если цель — сохранить выходы слоя на известных входах $X$,
задача $\|XW-XAB\|_F^2$ отличается от $\|W-AB\|_F^2$.
Обычный SVD не учитывает частоту входных направлений и последующие
нелинейности. После факторизации можно дообучить факторы по данным
или по сигналу учителя; итоговую потерю качества проверяют на модели.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/dcbd6535b09a28528559354f928838baaae7644b.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems — Low-Rank Factorization](https://mlsysbook.ai/vol1/model_compression/model_compression.html#sec-model-compression-lowrank-factorization-955e),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Исходная матрица заменяется двумя факторами. На рисунке $M=W$,
$k=r$, $L_k=A$, $R_k^\top=B$. Между факторами нет
новой функции активации: иначе получается другая модель, а не просто
запись линейной операции в факторизованном виде.*

### Число параметров, промежуточный тензор и LoRA

Число хранимых весов меняется с $mn$ на $r(m+n)$.
Для квадратной матрицы размера $d$ экономия появляется только при $r<d/2$.
У маленького примера $2\times2$, $r=1$, оба представления содержат
по четыре числа: он объясняет ошибку усечения, но не даёт экономии весов.

Для исходной матрицы $4096\times4096$ и входа $X:[32,4096]$
факторы применяются последовательно:
$Z=XA:[32,r]$, затем $Y=ZB:[32,4096]$.

| Ранг | Формы $A$ и $B$ | Параметров | BF16 веса | BF16 $Z$ |
|---|---|---:|---:|---:|
| Исходная матрица | $[4096,4096]$ | 16 777 216 | 32 MiB | Не нужен |
| 512 | $[4096,512]$ и $[512,4096]$ | 4 194 304 | 8 MiB | 32 KiB |
| 2048 | $[4096,2048]$ и $[2048,4096]$ | 16 777 216 | 32 MiB | 128 KiB |
| 3072 | $[4096,3072]$ и $[3072,4096]$ | 25 165 824 | 48 MiB | 192 KiB |

Матричных умножений по числу операций при $r=512$ вчетверо меньше,
а при $r=3072$ — в 1.5 раза больше исходного.
Дополнительно появляются запись и чтение $Z$, второй запуск и другие формы
задач для матричных блоков GPU.
Восстанавливать полное $AB$ перед каждым запросом бессмысленно:
тогда стоимость разложения не используется для ускорения вычисления.

[[Papers/LoRA|LoRA]]
использует низкоранговую обучаемую поправку к исходной матрице:
$W+\Delta W$, $\Delta W=AB$. Базовая $W$ при этом никуда не исчезает.
При слиянии получается снова полная плотная матрица $W+AB$;
при отдельном исполнении нужны и $W$, и факторы.
Низкоранговое дообучение не равно низкоранговому сжатию самой $W$.

<!-- tensor-factorization-addition: start -->
## Tensor factorization: CP, Tucker and convolution kernels

A convolution with 64 input channels, 128 output channels and a $3\times3$
kernel stores $128\cdot64\cdot3\cdot3=73\,728$ weights. Its weight tensor has
four axes: output channel, input channel, vertical offset and horizontal
offset. Flattening it into a matrix allows SVD compression, but fixes which
axes are grouped together. Tensor decompositions keep those axes separate and
place different restrictions on their interactions.

The following original-language excerpts are from the TensorLy user guide,
[“Tensor decomposition”](https://github.com/tensorly/tensorly/blob/91837a6570947d232b3bdb5206af619d1a9d26e3/doc/user_guide/tensor_decomposition.rst),
by the TensorLy developers, under
[BSD-3-Clause](https://github.com/tensorly/tensorly/blob/91837a6570947d232b3bdb5206af619d1a9d26e3/LICENSE.txt).
The worked examples and convolution calculations are Bookvar additions.
The corresponding course section is
[Harvard ML Systems, “Tensor decomposition”](https://mlsysbook.ai/vol1/model_compression/model_compression.html#sec-model-compression-tensor-decomposition-5e9e).

### CP: one component is an outer product

<!-- tensorly-excerpt: cp:start -->
> The idea is to express the tensor as a sum of rank one tensors. That is, a sum of outer products of vectors.
> Such representation can be obtained by applying Canonical Polyadic Decomposition (also known as CANDECOMP-PARAFAC, CP, or PARAFAC decomposition).
<!-- tensorly-excerpt: cp:end -->

For a three-dimensional tensor, take vectors $a=(1,2)$, $b=(3,4)$ and
$c=(5,6)$. Their outer product is the tensor $X_{ijk}=a_i b_j c_k$.
There are two slices along its last axis:

$$
X_{:,:,0}=\begin{pmatrix}15&20\\30&40\end{pmatrix},
\qquad
X_{:,:,1}=\begin{pmatrix}18&24\\36&48\end{pmatrix}.
$$

Both slices are scaled copies of the same matrix $ab^T$. Six stored numbers
generate eight entries, but the entries cannot vary independently. Doubling
$a_0$ doubles every entry in the first row of both slices. This coupling is
the restriction imposed by a rank-one tensor.

CP adds $R$ such components:

$$
\widehat X_{ijk}=\sum_{r=1}^{R} A_{ir}B_{jr}C_{kr}.
$$

For dimensions $I\times J\times K$, the factors contain
$R(I+J+K)$ parameters instead of $IJK$. Here component weights are absorbed
into a factor; implementations may store a separate length-$R$ vector.
Compression occurs only if the chosen rank makes this count smaller.
The CP rank is the smallest number of rank-one components needed for an
exact representation, not the number of dimensions of the tensor.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tensor-factorization-2026-10-29/harvard-cp.svg]]

*Vijay Janapa Reddi et al., Harvard ML Systems,
[Figure 11, Tensor Decomposition](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-tensor-decomposition),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The complete original SVG is unchanged. The highlighted entry is obtained
by multiplying the corresponding entries of the factor rows and summing
over their shared component index. The source uses $M,N,T$ for the three
dimensions, but labels the depth axis $K$ in the drawing; both denote the
third dimension here. The letter $R$ denotes the number of components.*

For a measured tensor, its factors are unknown. Alternating least squares
(ALS), used by
[TensorLy's `parafac`](https://tensorly.org/stable/modules/generated/tensorly.decomposition.parafac.html),
holds two factors fixed and solves for the third. With $B$ and $C$ fixed,
define $Z_{(j,k),r}=B_{jr}C_{kr}$. The update of $A$ is ordinary least
squares, $X_{(1)}\approx AZ^T$, where $X_{(1)}$ is $X$ reshaped into
$I$ rows and $JK$ columns. The same operation then updates $B$ and $C$.
Each exact subproblem can only decrease or preserve the squared residual,
but the joint optimization is nonconvex. A small change between iterations
does not establish a globally best approximation or a suitable rank.

<!-- checked-example: tensor-cp-als -->
```python
import numpy as np

a, b, c = np.array([1., 2.]), np.array([3., 4.]), np.array([5., 6.])
X = np.einsum("i,j,k->ijk", a, b, c)
assert X[1, 0, 1] == 2 * 3 * 6 == 36
np.testing.assert_array_equal(X[:, :, 0], [[15, 20], [30, 40]])

# A single ALS update: B and C are known, A starts at zero.
B, C = b[:, None], c[:, None]
Z = np.einsum("jr,kr->jkr", B, C).reshape(4, 1)
before = np.linalg.norm(X)**2
A = np.linalg.lstsq(Z, X.reshape(2, 4).T, rcond=None)[0].T
reconstructed = np.einsum("ir,jr,kr->ijk", A, B, C)
np.testing.assert_allclose(A[:, 0], a)
np.testing.assert_allclose(reconstructed, X)
assert np.linalg.norm(X - reconstructed)**2 < before
```

This example isolates one update, not a complete CP fitting run. In a full
run, initialization and stopping criteria matter, and changing a component's
scale in one factor can be compensated in another. Comparing raw factor
values from two runs is therefore less useful than comparing reconstructions
and the downstream model's outputs.

### Tucker: the core mixes different components

<!-- tensorly-excerpt: tucker:start -->
> The Tucker decomposition can be seen as a generalisation of the CP decomposition: it decomposes the tensor into a small core tensor and factor matrices. CP can be seen as a Tucker decomposition with a super-diagonal core.
>
> A tensor in its decomposed Tucker form is therefore nothing more than a core tensor with the same order as the original tensor and a list of projection matrices, one for each mode of the core tensor.
<!-- tensorly-excerpt: tucker:end -->

For three axes, Tucker uses a core $G$ with shape $r_1\times r_2\times r_3$:

$$
\widehat X_{ijk}
=\sum_{p=1}^{r_1}\sum_{q=1}^{r_2}\sum_{s=1}^{r_3}
G_{pqs}A_{ip}B_{jq}C_{ks}.
$$

CP pairs column $r$ of every factor with column $r$ of the others. Tucker
allows column $p$ of $A$ to interact with a different column $q$ of $B$ and
column $s$ of $C$, with strength $G_{pqs}$. A super-diagonal core has nonzero
entries only where $p=q=s$; substituting it into this equation gives CP.
The Tucker ranks can differ between axes. Storage is
$Ir_1+Jr_2+Kr_3+r_1r_2r_3$; the core can become expensive when many ranks
are retained.

The guide's example uses a $12\times12$ array shaped like a cross. A Tucker
core of shape $(2,3)$ and factors of shapes $(12,2)$ and $(12,3)$ store
$6+24+36=66$ values instead of 144. The thin factors reconstruct the original
cross because it lies in the retained row and column subspaces. Arbitrary
noise added to its pixels need not lie in those subspaces and will generally
leave a nonzero reconstruction residual. The runnable example below keeps
the guide's data but uses the current `(core, factors)` representation.

<!-- checked-example: tensor-tucker-cross -->
```python
import numpy as np

cross = np.zeros((12, 12))
cross[1:11, 4:8] = 1
cross[4:8, 1:11] = 1

# For a matrix, orthogonal Tucker factors can be obtained from its SVD.
U, s, Vt = np.linalg.svd(cross, full_matrices=False)
A, B = U[:, :2], Vt[:3].T
G = A.T @ cross @ B
assert G.shape == (2, 3)
assert A.size + G.size + B.size == 66
np.testing.assert_allclose(A @ G @ B.T, cross, atol=1e-14)
assert np.linalg.matrix_rank(cross) == 2

# Three-dimensional Tucker includes off-diagonal interactions in its core.
core = np.zeros((2, 2, 2))
core[0, 1, 0] = 7
I = np.eye(2)
X = np.einsum("pqs,ip,jq,ks->ijk", core, I, I, I)
assert X[0, 1, 0] == 7
assert np.count_nonzero(X) == 1
```

For higher-order tensors, one can initialize the factors with singular
vectors of each mode's unfolding, then refine one factor at a time while
the others are fixed. TensorLy's
[`partial_tucker`](https://tensorly.org/stable/modules/generated/tensorly.decomposition.partial_tucker.html)
uses higher-order orthogonal iteration and permits selecting which axes to
factorize. A truncated matrix SVD's global optimality result does not carry
over to an arbitrary choice of tensor ranks and an iterative Tucker fit.

### Tucker-2 turns one convolution into three

For the $128\times64\times3\times3$ kernel, compress only its two channel
axes. Keep the spatial axes in the core. Let $V$ have shape $64\times16$,
$G$ shape $32\times16\times3\times3$, and $U$ shape $128\times32$:

$$
\widehat W_{o,i,u,v}
=\sum_{p=1}^{32}\sum_{q=1}^{16} U_{o,p}G_{p,q,u,v}V_{i,q}.
$$

Substitute this expression into the convolution and group the sums by
$i$, then $(q,u,v)$, then $p$. This produces three ordinary convolutions,
the sequence implemented by
[TensorLy-Torch's `tucker_conv`](https://github.com/tensorly/torch/blob/25c915d554476d62dfc80c91de8e50fcd3d13542/tltorch/functional/convolution.py).

| Stage | Weights | Operation | Weight count |
|---|---|---|---:|
| Input projection | $V^T$, shape $[16,64,1,1]$ | Mix 64 input channels into 16 channels at each pixel | 1 024 |
| Spatial convolution | $G$, shape $[32,16,3,3]$ | Apply $3\times3$ kernels to the reduced channels | 4 608 |
| Output projection | $U$, shape $[128,32,1,1]$ | Mix 32 channels into the original 128 output channels | 4 096 |
| Total | Three factors | Same external input and output shapes | 9 728 |

The factorized layer uses about $7.58$ times fewer weights than 73 728.
With stride 1 and padding that preserves spatial size, it also performs
9 728 multiply-accumulates per spatial location instead of 73 728. With
stride greater than 1, the first projection still runs at input resolution,
so a single per-output-pixel weight count no longer gives the total MACs.

There is no activation between these three operations. Adding ReLU would
change their function and break equivalence to a single convolution with
$\widehat W$. Add the original bias only after the final projection.
The following CPU check tests that equivalence with unequal channel ranks,
nonzero bias, stride and dilation. It constructs a factorized kernel exactly;
it does not claim to fit an arbitrary trained kernel without error.

<!-- checked-example: tensor-factorized-conv -->
```python
import torch
import torch.nn.functional as F

torch.manual_seed(57)
dtype = torch.float64
x = torch.randn(2, 4, 9, 10, dtype=dtype)
V = torch.randn(4, 2, dtype=dtype)
U = torch.randn(6, 3, dtype=dtype)
G = torch.randn(3, 2, 3, 3, dtype=dtype)
bias = torch.randn(6, dtype=dtype)
W = torch.einsum("op,pquv,iq->oiuv", U, G, V)
for stride, dilation in [(1, 1), (2, 1), (1, 2)]:
    padding = dilation
    dense = F.conv2d(x, W, bias, stride=stride,
                     padding=padding, dilation=dilation)
    z = F.conv2d(x, V.T[:, :, None, None])
    z = F.conv2d(z, G, stride=stride, padding=padding, dilation=dilation)
    factorized = F.conv2d(z, U[:, :, None, None], bias)
    torch.testing.assert_close(factorized, dense, rtol=1e-12, atol=1e-12)

assert 128 * 64 * 3 * 3 == 73728
assert 64 * 16 + 32 * 16 * 3 * 3 + 128 * 32 == 9728

```

CP factorization also gives a sequence of smaller convolutions. For a 2D
kernel it uses an input $1\times1$ projection, a $3\times1$ depthwise
convolution, a $1\times3$ depthwise convolution, and an output $1\times1$
projection. Each of the $R$ intermediate channels is one CP component;
the two spatial filters act independently within that channel. Its parameter
count is $R(64+3+3+128)=198R$. Choosing $R=32$ stores 6 336 weights, but
imposes separability of each component's spatial kernel, unlike Tucker-2's
unrestricted $3\times3$ core filters. The two rank choices do not promise
the same reconstruction error.

<!-- checked-example: tensor-cp-conv -->
```python
import torch
import torch.nn.functional as F

torch.manual_seed(58)
dtype = torch.float64
x = torch.randn(2, 4, 9, 10, dtype=dtype)
bias = torch.randn(6, dtype=dtype)

# CP: the two spatial stages are depthwise (one filter per component).
rank = 2
A = torch.randn(6, rank, dtype=dtype)
B = torch.randn(4, rank, dtype=dtype)
C = torch.randn(3, rank, dtype=dtype)
D = torch.randn(3, rank, dtype=dtype)
Wcp = torch.einsum("or,ir,ur,vr->oiuv", A, B, C, D)
z = F.conv2d(x, B.T[:, :, None, None])
z = F.conv2d(z, C.T[:, None, :, None], padding=(1, 0), groups=rank)
z = F.conv2d(z, D.T[:, None, None, :], padding=(0, 1), groups=rank)
z = F.conv2d(z, A[:, :, None, None], bias)
torch.testing.assert_close(z, F.conv2d(x, Wcp, bias, padding=1),
                           rtol=1e-12, atol=1e-12)
```

TensorLy-Torch distinguishes `implementation='factorized'`, which executes
the contractions, from `implementation='reconstructed'`, which forms the
dense kernel before convolution. Both can have identical outputs, but the
second does not remove the dense convolution's work. Intermediate feature
maps, kernel launches and the device's efficiency for small convolutions
can also consume the arithmetic savings. Measure the factorized implementation
at the deployment batch size and resolution; counting parameters alone is
insufficient.

### Tensor-train: a chain of small cores

Tucker stores one central core whose size is the product of all retained
ranks. For a tensor with many axes, this product can be too large.
Tensor-train (TT), also called matrix product state, instead connects a
sequence of three-dimensional cores. For three axes:

$$
\widehat X_{ijk}
=\sum_{a=1}^{r_1}\sum_{b=1}^{r_2}
G_1[1,i,a]G_2[a,j,b]G_3[b,k,1].
$$

Fixing indices $(i,j,k)$ leaves a row vector, a matrix and a column vector.
Multiplying them gives that one entry. For example, $(1,2)$, the identity
matrix, and $(3,4)^T$ give $1\cdot3+2\cdot4=11$. The internal ranks
$r_1,r_2$ connect neighboring cores; the two outer ranks equal 1. For $d$
axes, storage is $\sum_{k=1}^{d}r_{k-1}n_k r_k$, with $r_0=r_d=1$.
It grows linearly in $d$ only while the mode sizes and internal ranks remain
bounded. A harder tensor may require growing ranks, and reconstructing all
entries still requires producing $\prod_k n_k$ outputs.

<!-- checked-example: tensor-train-entry -->
```python
import numpy as np

G1 = np.array([[[1., 2.], [0., 1.]]])       # (1, 2, 2)
G2 = np.stack([np.eye(2), 2 * np.eye(2)], axis=1)  # (2, 2, 2)
G3 = np.array([[[3.], [5.]], [[4.], [6.]]]) # (2, 2, 1)
X = np.einsum("ia,ajb,bk->ijk", G1[0], G2, G3[:, :, 0])
entry = (G1[0, 0, :] @ G2[:, 0, :] @ G3[:, 0, 0]).item()
assert entry == X[0, 0, 0] == 11
assert G1.size + G2.size + G3.size == 16 > X.size == 8

# A larger shape with bounded ranks has a different storage balance.
shape, ranks = [10] * 6, [1, 4, 4, 4, 4, 4, 1]
stored = sum(ranks[k] * shape[k] * ranks[k+1] for k in range(6))
assert stored == 720
assert np.prod(shape) == 1_000_000
```

The small example deliberately stores more values than its dense tensor.
The million-entry example counts storage for the specified ranks; it does
not show that any million-entry tensor can be approximated accurately with
720 parameters. TensorLy's
[`tensor_train`](https://tensorly.org/stable/modules/generated/tensorly.decomposition.tensor_train.html)
obtains cores by successive truncated SVDs of reshaped remainders. The axis
order determines these unfoldings and can change the ranks needed for a
given approximation error.

For model compression, choose the representation and ranks, fit the factors,
then measure the layer's output error on calibration inputs. A small weight
residual alone does not guarantee a small output residual: inputs may
concentrate on the discarded directions, as in the SVD example above.
Fine-tuning or distillation can then recover task quality. The next section
explains the teacher signal used for that recovery.

<!-- tensor-factorization-addition: end -->

<a id="knowledge-distillation-передать-функцию-а-не-веса"></a>

## Дистилляция: обучение по распределению учителя

Пусть классификатор различает три класса, а правильный класс — первый.
Разметка даёт цель $(1,0,0)$.
Учитель с логитами $z_T=(4,2,0)$ предсказывает приблизительно
$(0.867,0.117,0.016)$.
Кроме правильного класса, он сообщает, что второй считает правдоподобнее
третьего. Ученик с логитами $z_S=(3,0,1)$ согласен с первым местом,
но расставляет два остальных класса в обратном порядке.

Дистилляция использует распределение учителя как дополнительную цель.
Учитель в обычном таком обучении фиксирован; обновляются параметры ученика.
Ученик может иметь меньше слоёв, меньшую ширину или другую архитектуру.
Его не обязательно получать удалением параметров учителя.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/708770dcb60d00ea7f382f39d15b3206ba6ccb1f.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems — Soft Target Distribution](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-kd-targets),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Учитель отдаёт 85% вероятности кошке, 10% собаке и 5% лисе.
Разметка «кошка» не задаёт соотношения двух других вероятностей,
а мягкая цель передаёт и это соотношение. Числа иллюстративные;
это отдельный пример из источника, не распределение из логитов $(4,2,0)$.*

### Температура, KL и один шаг обновления

В [Hinton, Vinyals, Dean, §2](https://arxiv.org/abs/1503.02531)
вероятности вычисляются с температурой $\tau$:

$$
p_i^{(\tau)}=
\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

При $\tau>1$ различия вероятностей уменьшаются.
Оба распределения для мягкой цели вычисляют при одной температуре;
обычную ошибку по меткам — при температуре 1.

| Распределение | Класс 0 | Класс 1 | Класс 2 |
|---|---:|---:|---:|
| Правильная метка | 1 | 0 | 0 |
| Учитель, $\tau=1$ | 0.8668 | 0.1173 | 0.0159 |
| Учитель, $\tau=2$ | 0.6652 | 0.2447 | 0.0900 |
| Ученик, $\tau=2$ | 0.6285 | 0.1402 | 0.2312 |

Ученик слишком мало вероятности отводит второму классу и слишком много —
третьему. Это различие удобно записать через forward KL:

$$
D_{\mathrm{KL}}(p_T\|p_S)=
\sum_i p_{T,i}\log\frac{p_{T,i}}{p_{S,i}}.
$$

При фиксированном учителе его энтропия не влияет на градиент ученика:
минимизировать такой KL или cross-entropy по мягким целям эквивалентно
для обновления ученика. Значения двух функций потерь отличаются константой.
Порядок аргументов существенен; обратный KL — другая задача.

Смесь с размеченными целями можно задать так:

$$
\begin{aligned}
\mathcal L={}&\alpha\,\operatorname{CE}(y,p_S^{(1)})\\
&+(1-\alpha)\tau^2
D_{\mathrm{KL}}(p_T^{(\tau)}\|p_S^{(\tau)}).
\end{aligned}
$$

Здесь $\alpha$ — вес обычной ошибки по меткам.
Для мягкой части без коэффициента $1-\alpha$:

$$
\frac{\partial[\tau^2D_{\mathrm{KL}}]}{\partial z_{S,i}}
=\tau\bigl(p_{S,i}^{(\tau)}-p_{T,i}^{(\tau)}\bigr).
$$

В нашем примере при $\tau=2$ градиент равен приблизительно
$(-0.0734,-0.2090,0.2824)$. Шаг градиентного спуска поднимает первый
и второй логиты, опускает третий.
Ошибка по жёсткой метке, напротив, сама по себе стремится уменьшать
вероятности обоих неправильных классов. Здесь виден дополнительный
сигнал о соотношении классов 1 и 2.

<!-- checked-example: compression-distillation -->
```python
def softmax_temperature(logits, temperature):
    scaled = np.asarray(logits, dtype=float) / temperature
    exponentials = np.exp(scaled - scaled.max())
    return exponentials / exponentials.sum()

teacher_logits = np.array([4., 2., 0.])
student_logits = np.array([3., 0., 1.])
temperature = 2.0
teacher_probability = softmax_temperature(teacher_logits, temperature)
student_probability = softmax_temperature(student_logits, temperature)

def soft_loss(logits):
    p = softmax_temperature(logits, temperature)
    return temperature**2 * np.sum(
        teacher_probability * (np.log(teacher_probability) - np.log(p))
    )

gradient = temperature * (student_probability - teacher_probability)
np.testing.assert_allclose(gradient, [-0.07341847, -0.20896818, 0.28238665], atol=1e-8)
np.testing.assert_allclose(soft_loss(student_logits), 0.3563859280823313)
updated_logits = student_logits - 0.1 * gradient
assert soft_loss(updated_logits) < soft_loss(student_logits)
```

Множитель $\tau^2$ компенсирует уменьшение масштаба градиента мягкой цели
при росте температуры, особенно в высокотемпературном приближении.
Он не делает распределения и градиенты одинаковыми при всех $\tau$.
После дистилляции модель можно использовать с обычной температурой
или с выбранным для генерации режимом; обучающая температура не становится
обязательной настройкой инференса.

### Какие сигналы доступны у языковой модели

При token-level дистилляции учитель и ученик получают одинаковый префикс,
а мягкая цель задаётся на следующем токене. KL усредняют по нужным
непустым позициям; маска определяет, какие части примера входят в обучение.
Для непосредственного сравнения распределений нужны согласованные словарь
и значения индексов токенов. При разных токенизаторах поэлементный KL
между векторами логитов нельзя считать без отдельного согласования.

Если доступны только ответы API, можно обучать ученика на сгенерированном
тексте. Это последовательностная дистилляция: у каждого следующего токена
обычная жёсткая цель из ответа учителя. Полное мягкое распределение
учителя при этом не передаётся. В рассуждениях, коде и tool calls нужно
также проверять правильность сгенерированной последовательности.

Промежуточные состояния дают ещё один сигнал. Если учитель шире ученика,
обучаемая проекция $P$ может сопоставлять их представления:
$\|H_SP-H_T\|^2$. Дополнительно выбирают соответствие слоёв.
Одинаковый номер слоя у моделей разной глубины не гарантирует одинакового
назначения представления.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/model-compression-review-2026-10/minitron-distillation-vector.png]]

*Muralidharan et al.,
[Minitron, Figure 4](https://arxiv.org/html/2407.14679v2#S3.F4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Авторская схема показывает возможные сопоставления эмбеддингов,
промежуточных блоков и логитов. Это перечень исследованных сигналов,
а не требование включить все потери одновременно.
Полный рендер исходного векторного PDF, без обрезки.*

В Harvard ML Systems отдельно описаны решения перед запуском дистилляции.
Абзац сохранён на английском; источник —
[закреплённая глава Knowledge distillation](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/model_compression/model_compression.qmd#L1829),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

<!-- source-excerpt: compression-harvard-workflow:start -->
> Operationally, the workflow has four decisions before training begins: choose a teacher with the desired behavior, choose a student whose dense architecture fits the deployment target, run the teacher on calibration or task data to produce soft targets, and select the temperature and loss weight that balance teacher imitation against the hard labels. The validation step then checks more than accuracy. A successful distilled model must preserve calibration, subgroup behavior, and latency on the target hardware, because the student can inherit the teacher's errors as easily as its useful uncertainty.
<!-- source-excerpt: compression-harvard-workflow:end -->

<a id="minitron-структурное-pruning-как-измеряемая-кампания"></a>

### Minitron: уменьшение модели и восстановление качества

В [Compact Language Models via Pruning and Knowledge Distillation, §2–4](https://arxiv.org/html/2407.14679v2#S2)
новая модель сначала наследует оставшуюся часть параметров исходной,
а затем обучается по её сигналу.
Веса не просто зануляют: архитектуру перестраивают по выбранным размерам.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/inference-evaluation/minitron-pruning-distillation-loop.png]]

*Muralidharan et al.,
[Minitron, Figure 2](https://arxiv.org/html/2407.14679v2#S2.F2).
Предобученная модель проходит оценку важности, ранжирование, обрезку
и восстановление качества дистилляцией. Полученная модель может стать
отправной точкой следующего уменьшения. Этот рисунок также используется в
[Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L490-L502).*

Для ширины авторы оценивают активации голов внимания, MLP и нормализаций
на 1024 калибровочных примерах, без обратного распространения.
Оценки агрегируют по токенам и примерам; выбор агрегации проверяют
экспериментально. Для удаления слоёв используют другие оценки:
изменение perplexity после исключения слоя или Block Importance —
косинусное расстояние между входом и выходом блока.
Это не один универсальный показатель для любой структурной единицы.

Далее ранжируют компоненты и физически уменьшают соответствующие матрицы.
Для общей ширины меняются эмбеддинги, проекции внимания и MLP, нормализации.
Из допустимых комбинаций размеров выбирают архитектуры под бюджет,
недолго дообучают кандидатов, затем продолжают обучение выбранных.
В статье короткий этап нужен потому, что качество сразу после обрезки
не даёт устойчивого ранжирования кандидатов.

В Table 5 статьи итоговые размеры таковы:

| Модель | Слоёв | Ширина | Query-голов | KV-групп | Ширина MLP |
|---|---:|---:|---:|---:|---:|
| Nemotron-4 15B | 32 | 6144 | 48 | 8 | 24576 |
| Minitron 8B | 32 | 4096 | 48 | 8 | 16384 |
| Minitron 4B | 32 | 3072 | 24 | 8 | 9216 |

Обе выбранные модели сохраняют 32 слоя. Результат «меньше параметров»
здесь не означает «меньше слоёв».
Для этих моделей авторы используют только KL по логитам.
Обычная ошибка по исходным меткам исследовалась, но не вошла в лучший
вариант. При значительном уменьшении глубины промежуточные сигналы
могут быть полезны; в экспериментах без сильного удаления слоёв
они улучшения не дали.

Исходные рекомендации 5–7 из §4.1 оставлены на английском,
[Minitron](https://arxiv.org/html/2407.14679v2#S4.SS1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

<!-- source-excerpt: compression-minitron-losses:start -->
> Retrain exclusively with distillation loss using KLD instead of conventional training.
>
> Use (logit+intermediate state+embedding) distillation when depth is reduced significantly.
>
> Use logit-only distillation when depth isn’t reduced significantly.
<!-- source-excerpt: compression-minitron-losses:end -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/inference-evaluation/minitron-training-cost-quality.png]]

*Muralidharan et al., [Minitron, Figure 1](https://arxiv.org/html/2407.14679v2#S1.F1),
также в [Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L490-L502).
На графике сопоставляются MMLU и обучающие токены конкретных моделей.
«40× cheaper, 9% better» относится к сравнению Minitron 8B
с Nemotron-3 8B, а не ко всем вариантам pruning.*

Уменьшение числа токенов для дополнительной производной модели не равно
экономии обучения всего семейства: большая исходная модель уже должна
быть обучена. В расчёте §4.1 стоимость шага моделей 15B, 8B, 4B пропорциональна
4.4, 2.5, 1.2. При одинаковом полном числе шагов обучение трёх моделей
с нуля стоит 8.1 условной единицы. Если две младшие требуют в 40 раз
меньше шагов, итоговая стоимость равна
$4.4+2.5/40+1.2/40=4.4925$:
экономия всего семейства около 1.80 раза.
Это расчёт из допущений статьи, не универсальная оценка затрат на дистилляцию.

<a id="architecture-optimization-и-graph-optimization--разные-уровни"></a>

## Выбор архитектуры и оптимизация вычислительного графа

Архитектура задаёт операции и связи между ними: число слоёв, ширину,
вид блока, размеры attention и MLP.
При поиске архитектуры выбирают из допустимых вариантов и сравнивают
их качество и стоимость на целевом устройстве.
Даже одинаковое число параметров не гарантирует одинаковую задержку:
формы матриц и последовательность операций различаются.

В [Harvard ML Systems — NAS](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/model_compression/model_compression.qmd#L2567)
задача включает два уровня: обучить веса кандидата и оценить архитектуру
на валидационных данных с ограничением стоимости.
Полное обучение каждого кандидата дорого, поэтому используют ограниченные
пространства поиска, короткое обучение, общие веса или предсказатели стоимости.
Короткое обучение кандидатов Minitron — пример такой экономии поиска.

Пусть есть четыре учебных кандидата. Значения ниже заданы для объяснения
выбора, это не результаты измерения реальных LLM:

| Кандидат | Задержка, ms | Оценка качества |
|---|---:|---:|
| A | 11 | 78 |
| B | 15 | 81 |
| C | 18 | 83 |
| D | 27 | 86 |

При бюджете 20 ms выбирается C. Лучший по качеству D не проходит ограничение.
Если бюджет уменьшается до 15 ms, допустимым лучшим становится B.
При реальном выборе заранее фиксируют оборудование, длины, пакет,
квантизацию и процедуру измерения: стоимость кандидата зависит от них.

Оптимизация графа решает другую задачу.
Например, линейное преобразование, прибавление bias и GELU можно выполнить
одним слитым ядром вместо трёх. Матрица весов остаётся прежней.
Если между тремя ядрами записываются и читаются два тензора $[32,4096]$
в BF16, их устранение экономит до
$4\cdot32\cdot4096\cdot2=1$ MiB промежуточного трафика
в такой модели исполнения.
Это не сжатие файла весов и не уменьшение числа параметров.

<!-- checked-example: compression-budget-and-fusion -->
```python
candidates = [
    ("A", 11, 78), ("B", 15, 81), ("C", 18, 83), ("D", 27, 86)
]
def choose_under_budget(budget):
    feasible = [c for c in candidates if c[1] <= budget]
    return max(feasible, key=lambda c: c[2])[0]

assert choose_under_budget(20) == "C"
assert choose_under_budget(15) == "B"
assert 4 * 32 * 4096 * 2 == 2**20
np.testing.assert_allclose((4.4 + 2.5 + 1.2) / (4.4 + 2.5/40 + 1.2/40), 1.8030050083472453)
```

Слияние должно сохранять требуемую математическую операцию.
Побитовое совпадение не следует из алгебраической эквивалентности:
порядок округлений и вариант аппроксимации GELU могут отличаться.
Проверка точности остаётся необходимой и для оптимизации графа.

<!-- systems-completion: nas-exits:start -->
### Search space, search strategy, and performance estimation

In the four-candidate example, every candidate already had a quality score.
NAS must obtain those scores, which is often the expensive part. An architecture
$a$ selects operations and connections; its parameters $\theta$ still need
training. The idealized nested problem is

$$
\begin{aligned}
\theta^*(a)&=\arg\min_\theta L_{\rm train}(\theta,a),\\
a^*&=\arg\min_{a\in\mathcal A}
 L_{\rm val}(\theta^*(a),a),\\
&\text{subject to }C(a)\le C_{\max}.
\end{aligned}
$$

Here $C$ must specify a deployment configuration. FLOPs, peak memory and measured
latency are different costs. Repeated architecture selection also uses the
validation set, so a held-out test set is still needed for the final assessment.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/755a01d61fa04f057cd25ccea73187a7b8a8d769.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems, Neural Architecture Search Flow](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-nas-flow),
CC BY-NC-SA 4.0. Complete original SVG. A search strategy proposes a candidate;
the performance estimator returns a score. The enclosing one-shot box describes
shared training of candidate operations, not one-shot pruning.*

For three layers with choices `3×3 convolution`, `5×5 convolution`, and `skip`,
there are $3^3=27$ operation sequences before widths and connections are varied.
Restricting the search to these choices cannot discover an operation outside
the set. Reusing one searched cell reduces the number of decisions, but also
imposes repeated structure on the result.

The original course explanation of an evolutionary search is reproduced below
from [Search strategies](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/model_compression/model_compression.qmd#L2606),
CC BY-NC-SA 4.0; indexing markup is removed and the historical cost example is omitted.

<!-- completion-excerpt: nas-evolution:start -->
> Evolutionary algorithms maintain a population of candidate architectures and iteratively apply mutations (changing operations, adding connections) and crossover (combining parent architectures) to generate offspring. Fitness-based selection retains high-performing architectures for the next generation, so useful components such as skip connections or depthwise separable convolutions can be recombined rather than rediscovered from scratch.
<!-- completion-excerpt: nas-evolution:end -->

Random search supplies a baseline: sample legal candidates and evaluate them
under the same budget. An RL controller instead learns a distribution over
architecture decisions from their rewards. Neither strategy avoids the cost of
scoring a candidate. Early stopping, smaller proxy tasks and shared weights reduce
that cost but can change the ranking. A candidate that trains quickly for a few
steps need not have the best converged validation loss.

[DARTS](https://arxiv.org/abs/1806.09055) relaxes an edge's discrete choice into
a weighted mixture. For operations $o_1,o_2$ and architecture logits $\alpha$:

$$
\begin{gathered}
\bar o(x)=\pi_1o_1(x)+\pi_2o_2(x),\\
\pi=\operatorname{softmax}(\alpha).
\end{gathered}
$$

Model weights and architecture logits are optimized using their respective
training and validation objectives. After search, the mixture is converted into
a discrete architecture. That conversion changes the function: at $x=2$,
$o_1(x)=x$, $o_2(x)=3x$ and $\pi=(0.75,0.25)$, the mixture returns 3,
while selecting $o_1$ returns 2. The selected architecture must therefore be
trained and evaluated in its discrete form. Search-time mixture quality and
latency are not the deployment result.

<!-- checked-example: compression-nas-relaxation -->
```python
from itertools import product
choices = list(product(["conv3", "conv5", "skip"], repeat=3))
assert len(choices) == 27
x = 2.
probability = np.array([0.75, 0.25])
outputs = np.array([x, 3*x])
assert probability @ outputs == 3.
assert outputs[probability.argmax()] == 2.
```

### Early exit: average work and the requests that remain

An intermediate classifier can produce a prediction before the final layer.
The controller evaluates its confidence and either returns that prediction or
continues. This reduces average computation only when enough requests exit
early to offset the extra heads, decisions and routing work. A confident wrong
answer is still wrong; the confidence threshold needs calibration and evaluation
on deployment-like data, including difficult and rare cases.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/c6d3e77bf90984ccd1a0aa0e731ead3ded0c3e64.svg]]

*Vijay Janapa Reddi et al.,
[Harvard ML Systems, Early Exit Architecture](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-early-exit-transformers),
CC BY-NC-SA 4.0; original course drawing based on Xin et al.
After each classifier, the lower branch returns a prediction; `Continue`
resumes the deeper path. `LTE` is the source's learned exit estimator label.
The diagram describes a classification decision, not a guarantee that a
decoder can skip arbitrary layers while preserving its KV-cache.*

Let a shallow stage cost 2 ms, its exit head 0.2 ms, and the additional deep
stage 6 ms. Assume 70% of requests exit after the shallow stage. Sequential
per-request mean cost is

$$
E[T]=2+0.2+(1-0.7)\cdot6=4\text{ ms}.
$$

The full-depth baseline costs 8 ms. A request that continues costs 8.2 ms,
which is **higher** than the baseline. If the routing head costs $h$ and saves
the remaining work $d$ for fraction $p$, the mean saving is $pd-h$ before
batching and queueing overhead. The controller is worthwhile under this model
only when $pd>h$.

Suppose early-exit accuracy on the exiting subset is 96%, and deep-path
accuracy on the continuing subset is 90%. The routed system's accuracy is
$0.7\cdot0.96+0.3\cdot0.90=94.2\%$. To compare against the original model,
run the original model on **both** subsets. Reporting only the deep path's
overall accuracy does not measure the loss introduced by early exits.

<!-- checked-example: compression-early-exit -->
```python
p, shallow, head, deeper = 0.7, 2., 0.2, 6.
mean = shallow + head + (1-p)*deeper
np.testing.assert_allclose(mean, 4.)
assert shallow + head + deeper == 8.2
np.testing.assert_allclose(p*0.96 + (1-p)*0.90, 0.942)
assert p*deeper > head
# Waiting for every item in a static batch changes the cost model.
all_exit = p**8
batch_mean = shallow + head + (1-all_exit)*deeper
np.testing.assert_allclose(all_exit, 0.05764801)
assert batch_mean > mean
```

In that final calculation, a batch of eight keeps its dense shape whenever
at least one request continues. Under an illustrative independence assumption,
all eight exit with probability $0.7^8\approx5.8\%$. The batch therefore
executes the deep stage about 94.2% of the time. Compacting the remaining
requests can save work, but changes matrix sizes and adds gather/scatter and
scheduling costs. This is why per-request FLOPs do not establish batched latency.

For a decoder, later tokens may need keys and values from layers skipped by an
earlier token. Methods for layer skipping must specify how those states are
constructed or approximated. This is distinct from
[[58 Спекулятивное декодирование|speculative decoding]], which validates proposals
with a target model, and from
[[00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|MoE]], which routes tokens
among experts within a layer. The source's related examples remain in its
[complete adaptive-computation section](https://mlsysbook.ai/vol1/model_compression/model_compression.html#sec-model-compression-adaptive-computation-methods-d164);
their routing mechanisms should not be treated as interchangeable guarantees
of less computation.

<!-- systems-completion: nas-exits:end -->

<!-- systems-completion: architecture-details:start -->
### Scaling a convolutional network under a compute budget

Harvard's [Scaling optimization](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/model_compression/model_compression.qmd#L5017)
explains why parameter count alone cannot describe the cost of an architecture.
The following paragraph retains the original English wording, CC BY-NC-SA 4.0:

<!-- completion-excerpt: compound-scaling:start -->
> The first architectural response is global: when every stage of the profile is over budget, the problem is not one bad layer; the model is scaled incorrectly for the deployment envelope. The design task is then to distribute capacity across depth, width, and input resolution rather than choose a single parameter count. Depth increases sequential work and activation storage. Width exposes more parallel work but raises memory use. Resolution improves spatial detail while increasing the number of positions each convolution must process. The right balance depends on the machine: a highly parallel accelerator can often exploit width, while a small edge device may be dominated by memory capacity and energy per access.
<!-- completion-excerpt: compound-scaling:end -->

For a sequence of similarly shaped dense convolutions, let $d$, $w$ and $r$
be multipliers of depth, channel width and spatial resolution. The approximate
ratios are

$$
\frac{\operatorname{MAC}_{\rm new}}{\operatorname{MAC}_{\rm old}}
\approx dw^2r^2,\qquad
\frac{P_{\rm new}}{P_{\rm old}}\approx dw^2.
$$

Both input and output channel counts grow with $w$, hence the square.
Resolution changes the number of output positions, not the number of weights.
Depth adds sequential layers. These formulas omit boundary layers, integer
rounding, strides and operator-specific differences; depthwise convolution
has a different width dependence.

[EfficientNet](https://proceedings.mlr.press/v97/tan19a.html) couples the three
multipliers through a coefficient $\phi$:
$d=\alpha^\phi$, $w=\beta^\phi$, $r=\gamma^\phi$.
Choosing $\alpha\beta^2\gamma^2\approx2$ makes each increment of $\phi$
approximately double the dense-convolution budget. This is a scaling rule,
not a claim that doubling compute doubles accuracy.

The following illustrative coefficients are chosen for arithmetic, not taken
from an EfficientNet checkpoint: $\alpha=1.2$, $\beta=1.1$ and
$\gamma=\sqrt{2/(1.2\cdot1.1^2)}$. At $\phi=1$, work doubles,
parameters grow by 1.452, and one feature map grows by
$\beta\gamma^2\approx1.515$. At $\phi=2$, work quadruples but
parameters grow by about 2.108. The activation-memory change also depends on
which tensors must stay live; it is not determined by parameter count.

<!-- checked-example: compression-compound-scaling -->
```python
alpha, beta = 1.2, 1.1
gamma = np.sqrt(2/(alpha*beta**2))
for phi in [0, 1, 2, 3]:
    depth, width, resolution = alpha**phi, beta**phi, gamma**phi
    work_ratio = depth*width**2*resolution**2
    np.testing.assert_allclose(work_ratio, 2**phi)
np.testing.assert_allclose(alpha*beta**2, 1.452)
np.testing.assert_allclose((alpha*beta**2)**2, 2.108304)
```

Changing the operator can be more useful than scaling every layer.
[[00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet#Групповая и поканальная свёртки|The depthwise/pointwise example]]
replaces 432 weights with 75 and explains which kernels it can no longer
represent. Grouped convolutions and $1\times1$ bottlenecks make similar,
explicit restrictions. Feature reuse is another choice, but keeping earlier
feature maps alive can increase memory: DenseNet connectivity alone is not
evidence of lower peak activation storage.

### Fold BatchNorm into weights before deciding whether to fuse kernels

The source's Conv–BatchNorm–ReLU example combines two different transformations.
At inference, BatchNorm with fixed running statistics is affine in each output
channel. It can be absorbed into the preceding convolution's parameters.
Kernel fusion instead changes how the remaining operations execute.
The [PyTorch tutorial by Horace He and Will Feng](https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html#fusing-convolution-with-batch-norm)
implements the first transformation before matching the graph pattern.

For one output channel, let $u=Wx+b$ and
$y=\gamma(u-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$.
With $a=\gamma/\sqrt{\sigma^2+\epsilon}$, substitution gives

$$
\begin{gathered}
W'=aW,\qquad b'=a(b-\mu)+\beta,\\
y=W'x+b'.
\end{gathered}
$$

For multiple output channels, scale each filter and its bias independently.
ReLU remains a nonlinear operation: it cannot generally be absorbed into
fixed linear weights, but a kernel may apply it before writing its output.

<!-- checked-example: compression-bn-folding -->
```python
W = np.array([[1., -2.], [3., 4.]])
b = np.array([.5, -1.])
mean, variance = np.array([1., 2.]), np.array([4., 9.])
gamma, beta, eps = np.array([2., -3.]), np.array([.1, .2]), 1e-5
x = np.array([[1., 2.], [-1., 3.], [0., 0.]])
scale = gamma/np.sqrt(variance+eps)
unfolded = ((x @ W.T + b)-mean)*scale + beta
folded_W = scale[:, None]*W
folded_b = scale*(b-mean)+beta
folded = x @ folded_W.T + folded_b
np.testing.assert_allclose(folded, unfolded)
np.testing.assert_allclose(np.maximum(folded, 0), np.maximum(unfolded, 0))
```

This identity uses fixed statistics. Training-mode BatchNorm depends on the
current batch and cannot use these fixed folded weights. Folding also changes
weight ranges, so calibration must describe the graph that will be deployed.
Equality in real arithmetic does not imply bitwise equality after floating-point
reassociation or quantization. The example checks algebra on CPU; it neither
compiles a GPU kernel nor measures its speed.

### Binary and ternary values need a matching execution path

The source's [Extreme quantization](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/model_compression/model_compression.qmd#L4918)
extends precision reduction to two or three values. With binary operands
$x_i,w_i\in\{-1,+1\}$, each product is +1 when their signs agree and -1
when they differ. If $e$ of $n$ positions agree, then

$$
x^Tw=e-(n-e)=2e-n.
$$

Encode +1 as bit 1 and -1 as bit 0. XNOR marks agreements; a population count
computes $e$. This is the bitwise dot-product mechanism used by
[XNOR-Net, §3.2](https://arxiv.org/html/1603.05279v4#S3.SS2).
It requires both operands to be binary. Binarizing only the weights while
keeping real activations instead turns products into signed additions.

| Position | $x_i$ | $w_i$ | Agreement bit | Product |
|---:|---:|---:|---:|---:|
| 0 | +1 | +1 | 1 | +1 |
| 1 | -1 | +1 | 0 | -1 |
| 2 | +1 | -1 | 0 | -1 |
| 3 | -1 | -1 | 1 | +1 |
| 4 | +1 | +1 | 1 | +1 |

Three agreements give a dot product of $2\cdot3-5=1$.
The bit mask in the program removes complement bits outside the logical vector.
Padding in a packed machine word needs the same treatment.

<!-- checked-example: compression-binary-dot -->
```python
x = [1, -1, 1, -1, 1]
w = [1, 1, -1, -1, 1]
pack = lambda values: sum((v == 1) << i for i, v in enumerate(values))
mask = (1 << len(x))-1
agreements = (~(pack(x)^pack(w)) & mask).bit_count()
assert 2*agreements-len(x) == sum(a*b for a, b in zip(x, w)) == 1
ternary_w = [1, 0, -1, 0, 1]
assert sum(a*b for a, b in zip(x, ternary_w)) == x[0]-x[2]+x[4]
```

Ternary weights add zero, which allows omission of a contribution, but the
zero positions must still be represented. Three values require two bits under
simple fixed-width coding; $\log_2 3\approx1.585$ bits per value is not the
size of an ordinary two-bit packed array. Scales, padding, non-quantized layers
and runtime state add further storage. Training also needs a rule for gradients
through discretization, such as the
[[57 Квантизация языковых моделей#QAT: модель видит ошибку во время обучения|STE discussed in the quantization chapter]].
Neither this CPU identity nor a nominal bit width measures the speed or quality
of a trained binary network.

### Diagnose damage separately from measuring compression

An error histogram answers which values moved during quantization. It does not
show whether a rare class lost accuracy. A layerwise sparsity plot answers where
weights were removed; it does not show that those zeros are skipped by the
runtime. Compare three records on the same validation inputs: numerical changes
in tensors, changes in predictions, and the execution trace of the exported
artifact. Record tensor scales and saturation counts alongside error norms.
For classification, include per-class quality and calibration; for language
models, include the relevant languages, lengths and tasks.

One useful isolation test changes only one layer at a time, measuring the
quality change before combining transformations. The combined model still needs
its own evaluation because the errors interact. A visually blank filter is
not sufficient evidence to remove it: downstream scaling and rare inputs may
make its contribution significant.

<!-- systems-completion: architecture-details:end -->

## Как выбирать метод

Начинают с профиля исходной модели и воспроизводимой проверки качества.
Если не помещаются веса, уменьшают их физическое представление;
если мешает задержка, проверяют операции и данные на критическом пути.
Уменьшение FLOP само по себе не показывает, что на устройстве исчезла
ограничивающая часть работы.

После преобразования сохраняют сам артефакт, его формат и настройки
исполнения. Для разреженной модели нужны не только процент нулей,
но и фактическая упаковка и выбранное ядро. Для факторизованной —
формы обоих умножений и промежуточный тензор. Для ученика —
архитектура, данные дистилляции, сигналы учителя и процедура обучения.

Качество проверяют на задачах исходной модели, включая редкие случаи,
длинные контексты и ограничения безопасности, если они входили
в требования. Хорошее совпадение логитов на обучающих примерах
не гарантирует сохранение поведения вне этих примеров.
Скорость измеряют при одинаковом качестве обслуживания, а не сравнивают
короткий одиночный запрос после сжатия с большой нагрузкой до него.

[[58b Benchmarking, SLO и эксплуатация inference|Итоговое сравнение]]
содержит размер артефакта, фактически занятую память, задержки,
пропускную способность и качество. При выборе из нескольких вариантов
отбрасывают доминируемые: вариант не нужен, если другой не хуже него
по каждому критерию и строго лучше хотя бы по одному. Среди оставшихся
один может быть быстрее, другой — точнее или экономнее по памяти.

## Практика и первоисточники

- [[05 Источники/Courses/Harvard ML Systems/vol1/model_compression|Harvard ML Systems — полная оригинальная глава Model Compression]].
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/16_compression|TinyTorch Module 16]] — исходные реализации magnitude pruning и SVD; примеры выше исполняются независимо на NumPy.
- [Muralidharan et al., Minitron, §2–4](https://arxiv.org/html/2407.14679v2#S2).
- [Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L490-L502).
- [Sun et al., Wanda, §3](https://arxiv.org/html/2306.11695v1#S3).
- [Hinton, Vinyals, Dean — Distilling the Knowledge in a Neural Network, §2](https://arxiv.org/abs/1503.02531).
- [Han et al., Deep Compression](https://arxiv.org/abs/1510.00149).
- [Blalock et al. — What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html).

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload|KV-cache compression и offload]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Спекулятивное декодирование]]
