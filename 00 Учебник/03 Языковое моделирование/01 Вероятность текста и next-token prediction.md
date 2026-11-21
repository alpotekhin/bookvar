---
title: Вероятность текста и предсказание следующего токена
type: textbook-chapter
status: canonical
last_updated: 2026-11-03
source_language: mixed
reference_terms:
  - label: Авторегрессионная NLL и общие параметры
    label_en: Autoregressive NLL and shared parameters
    anchor: nlm-sequence-nll
  - label: Вектор контекста, выходная проекция и logits
    label_en: Context vector, output projection and logits
    anchor: nlm-head
  - label: Конкатенация embeddings и фиксированное окно
    label_en: Concatenated embeddings and a fixed context window
    anchor: nlm-window
  - label: Сдвиг входов и следующих целей
    label_en: Input-target shift
    anchor: nlm-target-shift
  - label: Причинность и параллельные позиции
    label_en: Causality and parallel positions
    anchor: nlm-causality
  - label: Teacher forcing и собственный префикс
    label_en: Teacher forcing and generated prefixes
    anchor: nlm-teacher-forcing
  - label: Cross-entropy с целевым ID токена
    label_en: Cross-entropy for a target token ID
    anchor: nlm-cross-entropy
  - label: Градиент softmax и общая выходная матрица
    label_en: Softmax gradient and shared output weights
    anchor: nlm-vocabulary-gradient
  - label: Устойчивый log-softmax и контракт CE
    label_en: Stable log-softmax and the CE API contract
    anchor: nlm-stability
  - label: Perplexity, accuracy и протокол сравнения
    label_en: Perplexity, accuracy and comparison protocol
    anchor: nlm-perplexity
  - label: Усреднение потерь по токенам и документам
    label_en: Token-wise and document-wise loss averaging
    anchor: nlm-loss-reduction
  - label: Повторяющиеся контексты и нижняя граница NLL
    label_en: Repeated contexts and the conditional NLL floor
    anchor: nlm-entropy-floor
  - label: Частота продолжения и истинность ответа
    label_en: Continuation frequency and factual correctness
    anchor: nlm-objective-limits
  - label: In-context применение и post-training
    label_en: In-context use and post-training
    anchor: nlm-base-assistant
  - label: Маска позиций при общем ID EOS и PAD
    label_en: Positional masking when EOS and PAD share an ID
    anchor: nlm-eos-pad-mask
  - label: Полное обучение WindowLM на малом корпусе
    label_en: Complete WindowLM training on a small corpus
    anchor: nlm-window-training
  - label: Нейронная LM Bengio — схема и обозначения
    label_en: Bengio neural LM diagram and notation
    anchor: nlm-bengio-architecture
  - label: Порядок окна и общий градиент embeddings
    label_en: Context order and accumulated embedding gradients
    anchor: nlm-context-permutation
source_unit_id:
  - assignment-01-task-softmax
  - assignment-01-task-cross-entropy
primary_sources:
  - https://www.jmlr.org/papers/v3/bengio03a.html
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
  - https://d2l.ai/chapter_recurrent-neural-networks/language-model.html
  - https://d2l.ai/chapter_linear-classification/softmax-regression.html
  - https://github.com/karpathy/nn-zero-to-hero/blob/73c3fcc741f0ec104ca850b1fb0df90e7e8d4cde/lectures/makemore/makemore_part1_bigrams.ipynb
previous: "[[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель]]"
next: "[[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings]]"
---

# Вероятность текста и предсказание следующего токена

В строке `кот спит бежит` слово `спит` уже записано, но для
модели оно становится ответом на отдельную задачу: увидев `кот`,
она должна оценить, какое слово идёт следом. В предыдущей главе
мы брали эту оценку из таблицы: после `кот` было три перехода,
два из них вели к `спит`. Теперь вероятность вычислит нейронная
сеть. Если она даёт `спит` вероятность $1/3$, эта позиция
добавляет к потере $-\log(1/3)=\log 3$.

На входе по-прежнему три текста: D1 — `кот кот спит`,
D2 — `пёс спит`, D3 — `кот спит бежит`. В конце главы
из них соберём один батч и обучим модель с окном в два слова.
Для этого достаточно таблицы векторов токенов и MLP.

<a id="autoregressive-objective"></a>

<a id="nlm-sequence-nll"></a>

## Вероятность последовательности

Для оценки законченного текста перемножают вероятности его токенов
и маркера конца EOS. Заданный маркер начала BOS не является
дополнительной предсказываемой целью.
Цепное правило и различие между префиксом и законченным текстом
разобраны в
[[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель#Цепное правило|предыдущей главе]].

У D3 четыре цели: `кот`, `спит`, `бежит` и EOS.
Отрицательный логарифм вероятности текста — сумма потерь
для этих четырёх предсказаний. Не будем повторять BOS в записи
условия после того, как появились обычные токены:

$$
\begin{gathered}
-\log P_\theta(\text{кот спит бежит},\mathrm{EOS})\\
=-\log p_\theta(\text{кот}\mid\mathrm{BOS})\\
{}-\log p_\theta(\text{спит}\mid\text{кот})\\
{}-\log p_\theta(\text{бежит}\mid\text{кот спит})\\
{}-\log p_\theta(\mathrm{EOS}\mid\text{кот спит бежит}).
\end{gathered}
$$

Одна и та же модель с параметрами $\theta$ вычисляет все четыре
вероятности. При обратном проходе вклады четырёх слагаемых
в градиент общих весов складываются.

Ответ для каждой позиции уже находится в исходном тексте.
Такую постановку называют обучением предсказанию следующего токена,
next-token prediction (NTP), и относят к self-supervised learning:
для получения этих меток не нужен отдельный разметчик.
Но сбор, фильтрация и выбор частоты текстов всё равно влияют
на то, какие ответы будет учить модель.

<a id="от-биграммы-к-длинному-контексту"></a>

## Логиты и окно из нескольких токенов

<a id="nlm-head"></a>

### Представление контекста и выходная проекция

Сохраним прежние ID слов и добавим BOS/EOS:

| ID | Токен |
|---:|---|
| 0 | `<pad>` |
| 1 | `<unk>` |
| 2 | кот |
| 3 | спит |
| 4 | пёс |
| 5 | бежит |
| 6 | `<s>`, BOS |
| 7 | `</s>`, EOS |

Входной и выходной словари совпадают: $V=8$. У BOS и PAD
есть выходные логиты, хотя среди правильных целей этих токенов
нет. Пока эти ответы могут получить ненулевую вероятность:
мы не исключаем их из знаменателя softmax.

В простейшей нейронной биграмме вектор контекста —
embedding последнего входного токена:

$$
h=E[i]\in\mathbb R^d.
$$

Линейная выходная проекция превращает его в логиты:

$$
z=Wh+b,
\qquad W\in\mathbb R^{V\times d}.
$$

У каждого ответа $v$ своя строка весов $W_v$.
Логит $z_v=W_vh+b_v$ может быть отрицательным или больше единицы.
Чтобы получить вероятности, возводим $e$ в степень каждого логита,
а затем делим на сумму полученных чисел.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/neural-language-model-2026-10-21/d2l-softmaxreg.svg|Полносвязная выходная проекция из четырёх признаков в три оценки]]

*Zhang, Lipton, Li, Smola,
[D2L, Softmax Regression — Linear Model](https://d2l.ai/chapter_linear-classification/softmax-regression.html#linear-model).
На исходной схеме четыре входных признака и три линейные оценки.
Каждый выход использует все признаки, но имеет собственные веса.
В языковой модели на месте входных признаков стоит вектор контекста,
а выходов столько, сколько токенов в словаре.
Softmax применяется после этих оценок и на схеме не показан.
Оригинальный SVG без изменений; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Softmax вычисляет распределение по всем ответам:

$$
p_v=\frac{\exp z_v}{\sum_{u=0}^{V-1}\exp z_u}.
$$

Знаменатель общий для всей строки, поэтому увеличение
одного логита меняет не только вероятность его токена.
Для конечных логитов математически все вероятности положительны;
численное округление малых вероятностей обсудим ниже.

Эта выходная проекция называется языковой головой
(language-modeling head). В нашем примере таблица входных
векторов $E$ и матрица выхода $W$ обучаются отдельно.

<a id="nlm-window"></a>

### Несколько предыдущих токенов

Чтобы использовать два токена вместо одного, можно соединить
их embeddings и передать результат в MLP:

$$
\begin{gathered}
c_t=[E[x_{t-1}];E[x_t]]\in\mathbb R^{2d},\\
h_t=\tanh(Ac_t+a),\qquad z_t=Wh_t+b.
\end{gathered}
$$

При $d=2$ контекст `кот пёс` с векторами $(1,0)$ и $(1,1)$
превращается в $(1,0,1,1)$.
Обратный порядок `пёс кот` даёт $(1,1,1,0)$.
Конкатенация сохраняет, какой вектор стоял на каком месте;
обычное среднее этих двух векторов различия бы не сохранило.

Одна и та же сеть применяется ко всем позициям. Для первого
предсказания в документе её окно состоит из двух векторов BOS.
Чтобы предсказать `бежит` в D3, она получит векторы
`кот` и `спит`. Более ранние токены в окно не попадают.

Линейный выход и его связь с MLP описаны в
[D2L, Neural Networks without Hidden States](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html#neural-networks-without-hidden-states).
Переход от счётчиков к обучаемым логитам подробно прослеживается
в [makemore, Part 1](https://github.com/karpathy/nn-zero-to-hero/blob/73c3fcc741f0ec104ca850b1fb0df90e7e8d4cde/lectures/makemore/makemore_part1_bigrams.ipynb).

<details>
<summary>Модель Бенжио: оригинальная схема и порядок векторов в окне</summary>

<a id="nlm-bengio-architecture"></a>

### The original feed-forward neural language model

![[00 Учебник/Assets/Figures/curated/neural-lm-bengio-2026-11-03/bengio-original-figure1.png|Original Bengio neural language model: shared word lookup, tanh, softmax and optional direct connections]]

*Yoshua Bengio, Réjean Ducharme, Pascal Vincent and Christian Jauvin,
[Figure 1, A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf#page=6),
JMLR 3 (2003), p.1142. Complete original diagram and caption, rendered from
the publisher PDF; no labels redrawn. Author copyright ©2003.
[JMLR publication policy and reuse note](/licenses/bengio03a-figure1-jmlr-policy.txt).*

Read the figure from the word indices at the bottom. Each index selects
a row of the same matrix $C$; the three lookup positions do not own
three separate embedding tables. The selected vectors form the input to
a tanh hidden layer. The output layer produces one score per vocabulary
word, then softmax normalizes the scores.

The dashed green paths bypass the hidden layer. In the paper's notation,
the unnormalized output vector is

$$
y=b+Wx+U\tanh(d+Hx).
$$

Here $x$ concatenates the selected word vectors. The direct matrix $W$ is
optional: setting it to zero leaves the hidden-layer route used by
`WindowLM` below. The paper's $y$ corresponds to this chapter's
logits $z$, not to a target token ID. Its $W$ is also not the output
matrix called $W$ in our earlier bigram calculation.

For the `WindowLM` dimensions used at the end of the chapter, the
correspondence is:

| Paper | `WindowLM` parameter | Shape |
|---|---|---|
| $C$ | `embedding.weight` | $8\times2$ |
| $H,d$ | `network[0].weight`, `.bias` | $8\times4$, $8$ |
| $U,b$ | `network[2].weight`, `.bias` | $8\times8$, $8$ |
| Optional $W$ | No direct path in this implementation | $8\times4$ if added |

The hidden-layer-only version has 128 parameters, as counted in the
training example. Adding the direct $W$ would add 32 weights and change
the model. The source annotation “most computation here” concerns its
large-vocabulary output layer; the diagram alone does not measure the
runtime of our eight-token example.

<a id="nlm-context-permutation"></a>

### Context order and repeated lookups

The paper writes its concatenation from the newest context word to the
oldest. `WindowLM` constructs windows in chronological order.
Either fixed convention works, but the columns of the first projection
must follow the same ordering as the input blocks.

For the vectors already used above, `кот` is $(1,0)$ and `пёс`
is $(1,1)$. Chronological concatenation gives $(1,0,1,1)$.
With projection row $(1,2,3,4)$, its pre-activation is 8.
Reversing the two vector blocks gives $(1,1,1,0)$; the corresponding
projection row must become $(3,4,1,2)$ to keep 8. Keeping the old row
instead gives 6.

When the same word occurs twice, both lookups select one parameter row.
For the context `кот кот`, the projection is $(1+3)C_{\rm cat,0}
+(2+4)C_{\rm cat,1}$. The gradient of this pre-activation with respect
to the shared cat vector is therefore $(4,6)$, not two independently
updated vectors.

```python
import torch

source_vectors = torch.zeros(8, 2, dtype=torch.float64)
source_vectors[2] = torch.tensor([1., 0.])  # кот
source_vectors[4] = torch.tensor([1., 1.])  # пёс
source_vectors.requires_grad_()
source_row = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
chronological = source_vectors[[2, 4]].flatten()
newest_first = source_vectors[[4, 2]].flatten()
permuted_row = source_row.reshape(2, 2).flip(0).flatten()
assert (chronological @ source_row).item() == 8
assert (newest_first @ permuted_row).item() == 8
assert (newest_first @ source_row).item() == 6

repeated = source_vectors[[2, 2]].flatten()
lookup_gradient, = torch.autograd.grad(repeated @ source_row, source_vectors)
expected_gradient = torch.zeros_like(source_vectors)
expected_gradient[2] = torch.tensor([4., 6.])
torch.testing.assert_close(lookup_gradient, expected_gradient)
```

This calculation checks lookup sharing and block order. The complete
nonlinear model, loss and optimizer are implemented below; it is not
a reproduction of the paper's corpus-scale experiment.

</details>

<a id="один-текст-даёт-много-обучающих-примеров"></a>
<a id="nlm-target-shift"></a>

## Сдвиг входов и целей

Логиты $z_t$ в позиции $t$ относятся к следующему токену $x_{t+1}$.
Для D3 выпишем вход и цель каждой позиции:

| $t$ | Вход | Цель |
|---:|---|---|
| 0 | BOS | кот |
| 1 | кот | спит |
| 2 | спит | бежит |
| 3 | бежит | EOS |

Входная последовательность и цели имеют одинаковую длину,
но сдвинуты на одну позицию:

```text
input:  BOS  кот   спит   бежит
target: кот  спит  бежит  EOS
```

В коде это срезы `tokens[:, :-1]` и `tokens[:, 1:]`.
Столбец «Вход» показывает токен текущей позиции, а не весь
доступный контекст. Например, в позиции 2 модель может
использовать и предыдущий `кот`. Если подать в сеть всю строку
включая EOS, у последнего выхода не будет цели. В этом случае
убирают последний выход, сохраняя тот же сдвиг.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ntp-review-2026-10/d2l-language-model-data.png|Посимвольные входы и цели со сдвигом на одну позицию]]

*Zhang, Lipton, Li, Smola,
[D2L, Language Models — Partitioning Sequences](https://d2l.ai/chapter_recurrent-neural-networks/language-model.html#fig-lang-model-data).
В оригинале показаны посимвольные отрезки длины пять:
например, входу e tim соответствует цель « time», включая ведущий пробел.
Сдвиг равен одному символу, а каждая пара отрезков даёт пять целей.
Это не разбиение по словам. Полная исходная схема без обрезки;
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Часть библиотечных классов выполняет сдвиг внутри функции потерь.
Например, это соглашение показано для GPT-2 в
[Hugging Face, Initializing a new model](https://huggingface.co/learn/llm-course/en/chapter7/6#initializing-a-new-model):
collator копирует входные ID в labels, а модель выравнивает их сама.
В нашем коде ниже вход и цели сдвигаются явно ровно один раз.
Эти два контракта нельзя смешивать.

<a id="nlm-causality"></a>

### Причинность и параллельное вычисление

При обучении `спит` находится во входе D3 в позиции 2.
При расчёте логитов для позиции 1 модель не должна видеть
эту позицию: иначе она могла бы прочитать готовый ответ.
Сдвиг определяет, с чем сравнивать выход, но сам по себе
не ограничивает доступ модели к будущим входам.

В Transformer каузальная маска разрешает выходу позиции $t$
использовать входы $0,\ldots,t$.
Вход $x_t$ доступен, а предсказываемый $x_{t+1}$ — нет.
Это не запрет смотреть на текущий входной токен.
Слои выполняются последовательно, но позиции внутри одного слоя
можно обрабатывать параллельно.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ntp-review-2026-10/d2l-gpt-decoder.png|Сдвинутые цели GPT и доступные причинные зависимости]]

*Zhang, Lipton, Li, Smola,
[D2L, GPT and GPT-2](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#gpt-and-gpt-2).
Слева входы и сдвинутые ответы, справа — разрешённые зависимости.
Вертикальная ось справа подписана предсказываемыми токенами
и читается снизу вверх. Поэтому строка like имеет доступ
к BOS и входному I, но не к входному like.
Перевёрнутая относительно привычной картинки треугольная форма
связана с порядком строк. Полная схема без изменений;
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

Проверить причинность можно и без attention.
Пусть три входных вектора равны $(0,0),(1,0),(0,1)$,
а состояние позиции — среднее доступного префикса.
Одновременный расчёт всех трёх состояний даёт
$(0,0),(1/2,0),(1/3,1/3)$.
Если изменить третий вход, первые два состояния не изменятся.

```python
import torch

x = torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]], dtype=torch.float64)
lengths = torch.arange(1, 4, dtype=x.dtype)[None, :, None]
parallel = x.cumsum(dim=1) / lengths
serial = torch.stack([x[:, :t+1].mean(dim=1) for t in range(3)], dim=1)
torch.testing.assert_close(parallel, serial)

changed = x.clone()
changed[:, 2] = torch.tensor([-1., 1.])
other = changed.cumsum(dim=1) / lengths
torch.testing.assert_close(parallel[:, :2], other[:, :2])
assert not torch.allclose(parallel[:, 2], other[:, 2])

# Неверный вариант: общее среднее раскрывает будущее ранним позициям.
wrong = x.mean(dim=1, keepdim=True).expand_as(x)
wrong_other = changed.mean(dim=1, keepdim=True).expand_as(changed)
assert not torch.allclose(wrong[:, :2], wrong_other[:, :2])
```

Этот усреднитель — проверочный пример, не реализация attention.
В MLP с окном в два токена будущие входы исключены
самим построением окна.
Для Transformer запрет обеспечивают маски в слоях attention;
одного правильного сдвига labels недостаточно.

Маска потерь отвечает за другую операцию:
какие ответы входят в сумму ошибок.
Она не убирает токен из контекста.
Например, при обучении отвечать на вопрос сам вопрос
может быть виден модели, но потери начисляются только на ответе.
Маска входных заполнителей и границы упакованных документов
также задаются отдельно.
Их реализация разобрана в
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|главе об обработке пакетов]].

<a id="nlm-teacher-forcing"></a>

### Истинный префикс и сгенерированный префикс

При обычном NTP следующая обучающая позиция получает
предыдущие токены из корпуса, независимо от текущих предсказаний.
Этот режим называют teacher forcing.
Если после BOS модель предпочла `пёс`, а в тексте стоит `кот`,
следующая обучающая позиция всё равно получает `BOS кот`.
Метка для неё известна именно из этого текста.

При генерации выбранный `пёс` действительно становится новым входом.
Следующее распределение вычисляется для `BOS пёс`.
Подставить собственное предсказание при обучении и оставить
старую цель без изменения означало бы задать другую задачу.

Для вычисления стандартной NLL не нужно сэмплировать продолжения
или дифференцировать через `argmax`.
Вероятности правильных следующих токенов вычисляют
на наблюдаемых префиксах.
Один вызов модели может вернуть все выходы `[B,T,V]`,
но сам по себе этот интерфейс ещё не доказывает
параллельность временных шагов: внутри RNN они связаны рекуррентно.

<a id="stable-cross-entropy"></a>

<a id="cross-entropy-считается-из-logits"></a>
<a id="nlm-cross-entropy"></a>

## Кросс-энтропия из логитов

В позиции 1 строки D3 цель — `спит` с ID 3. Для расчёта
возьмём модель с контекстом из одного токена. Пусть после `кот`
она получает $h=(1,0)$. Выберем веса выходной проекции без
смещения так, чтобы $z_2=\log 2$ для `кот`, $z_3=\log 4$ для
`спит`, а остальные шесть логитов равнялись нулю.

После экспоненты значения равны $2$, $4$ и шести единицам.
Их сумма $12$, поэтому

$$
\begin{gathered}
p(\text{спит})=\frac4{12}=\frac13,\\
p(\text{кот})=\frac2{12}=\frac16.
\end{gathered}
$$

Каждый из остальных шести токенов получил $1/12$.
Для потери берём вероятность цели `спит` и получаем
$-\log(1/3)=\log 3$.

Для общего вектора логитов и правильного ID $y$:

$$
\ell(z,y)
=-\log p_y
=\log\sum_{v=0}^{V-1}\exp z_v-z_y.
$$

Это кросс-энтропия между предсказанным распределением и
целевым, в котором наблюдаемый токен имеет вероятность $1$.

<a id="nlm-vocabulary-gradient"></a>

### Градиент по всему словарю

Производная по логиту $z_v$ равна

$$
g_v=\frac{\partial\ell}{\partial z_v}
=p_v-\mathbf1[v=y].
$$

Производная первого слагаемого $\log\sum_u\exp z_u$
даёт $\exp z_v/\sum_u\exp z_u=p_v$ по цепному правилу.
Производная второго, $-z_y$, равна $-1$ для правильного класса
и нулю для остальных. Поэтому градиент затрагивает весь словарь,
хотя правильная цель в позиции одна.

В нашем примере для `спит` получаем $1/3-1=-2/3$,
для `кот` — $1/6$, для остальных — $1/12$.
Локальный сигнал повышает правильную оценку
и понижает остальные, сохраняя связь через общий знаменатель.
Их градиенты суммируются в ноль,
как и должно быть для функции, не зависящей
от общей добавленной к логитам константы.

При $z=Wh$ градиенты параметров и входного вектора равны

$$
\frac{\partial\ell}{\partial W}=gh^\top,
\qquad
\frac{\partial\ell}{\partial h}=W^\top g.
$$

Так как $h=(1,0)$, в этом шаге меняются только первые
координаты строк выходной матрицы.
У строки `спит` градиент $(-2/3,0)$,
у `кот` — $(1/6,0)$.
Для входного embedding `кот` получаем
$(-7\log2/6,0)\approx(-0{,}808672,0)$.

После SGD с шагом $0{,}3$:

- первая координата выходной строки `спит` увеличится на $0{,}2$;
- первая координата строки `кот` уменьшится на $0{,}05$;
- входной вектор `кот` станет примерно $(1{,}242602,0)$.

Пересчёт выхода с обеими обновлёнными матрицами
даёт $p(\text{спит})\approx0{,}471700$ и
$\ell\approx0{,}751411$ вместо $1{,}098612$.
Здесь потеря уменьшилась после одного шага на одном примере.
При слишком большом шаге SGD она могла бы вырасти.

```python
from math import log
import torch
import torch.nn.functional as F

embeddings = torch.zeros(8, 2, dtype=torch.float64)
embeddings[2] = torch.tensor([1., 0.])  # кот
embeddings[4] = torch.tensor([1., 1.])  # пёс
E = torch.nn.Parameter(embeddings)
weights = torch.zeros(8, 2, dtype=torch.float64)
weights[2, 0], weights[3, 0] = log(2), log(4)
W = torch.nn.Parameter(weights)

logits = E[2] @ W.T
before = F.cross_entropy(logits[None, :], torch.tensor([3]))
before.backward()
torch.testing.assert_close(W.grad[3], torch.tensor([-2/3, 0.], dtype=W.dtype))
torch.testing.assert_close(E.grad[2],
                           torch.tensor([-7*log(2)/6, 0.], dtype=E.dtype))
old_dog = E[4].detach().clone()

optimizer = torch.optim.SGD([E, W], lr=.3)
optimizer.step()
new_logits = E[2] @ W.T
after = F.cross_entropy(new_logits[None, :], torch.tensor([3]))
torch.testing.assert_close(after.detach(),
                           torch.tensor(.7514114860758478, dtype=after.dtype))
torch.testing.assert_close(E[4], old_dog)
assert after < before
print(float(before.detach()), float(after.detach()))
```

Строка входного embedding `пёс` в этом шаге не участвовала
и не изменилась. Но общая выходная матрица изменилась,
поэтому для её вектора $(1,1)$ вероятность `спит`
тоже увеличилась: с $1/3$ до примерно $0{,}386522$.
Так проявляется общность параметров.
В другом примере изменение общего слоя может ухудшить
чужое предсказание; оценивать нужно и другие контексты.

Вывод производной и её связь с one-hot целью приведены в
[D2L, Softmax and Cross-Entropy Loss](https://d2l.ai/chapter_linear-classification/softmax-regression.html#softmax-and-cross-entropy-loss).
У Карпати в makemore тот же переход разобран
от явного one-hot входа до обратного прохода и обновления матрицы.

<a id="nlm-stability"></a>

### Численная устойчивость и библиотечный контракт

Логиты $z$ и $z+c$ задают одно распределение:
множитель $e^c$ сокращается в числителе и знаменателе softmax.
При непосредственном вычислении это сокращение может не помочь:
`exp(1000)` переполняет float64 ещё до деления.

Пусть $m=\max_v z_v$.
Вычтем максимум до вычисления экспонент:

$$
\ell(z,y)
=\log\sum_v\exp(z_v-m)-(z_y-m).
$$

Все аргументы экспоненты теперь неположительны.
Но вычислять отдельно даже устойчивый softmax,
а затем логарифм его результата тоже не всегда достаточно.
При логитах $(0,-1000)$ малая вероятность округлится до нуля,
хотя потеря для второго ответа должна оставаться конечной — $1000$.
Log-softmax или прямая формула cross-entropy
не требуют сначала представлять эту малую вероятность.

```python
from math import log
import torch
import torch.nn.functional as F

z = torch.zeros(8, dtype=torch.float64)
z[2], z[3] = log(2), log(4)
target = torch.tensor([3])
m = z.max()
manual = torch.logsumexp(z - m, dim=0) - (z[3] - m)
reference = F.cross_entropy(z[None, :], target)
torch.testing.assert_close(manual, reference)
torch.testing.assert_close(reference,
                           F.cross_entropy((z + 1000)[None, :], target))
assert torch.isinf((z + 1000).exp()).all()

extreme = torch.tensor([[0., -1000.]], dtype=torch.float64)
assert extreme.softmax(dim=-1)[0, 1] == 0
assert torch.isinf(-extreme.softmax(dim=-1)[0, 1].log())
torch.testing.assert_close(F.cross_entropy(extreme, torch.tensor([1])),
                           torch.tensor(1000., dtype=extreme.dtype))
```

В `F.cross_entropy` передают ненормированные логиты,
а не результат softmax.
При входе `[B,T,V]` размер классов находится в последней оси,
тогда как эта функция ожидает его во второй оси для многомерного входа.
Поэтому используют `reshape(B*T,V)` и цели `[B*T]`,
перестановку осей либо выбор целевых позиций,
как в законченном коде ниже.

Здесь используются целочисленные ID целей,
без весов классов и label smoothing.
Иные настройки меняют формулу или усреднение и задаются отдельно.
Вектор из одних $-\infty$ не образует нормированного
распределения: вычитание максимума не исправляет такой вход.

Требования к устойчивому softmax и cross-entropy разобраны в
[CS336 Assignment 1, §3.4.4 и §4.1](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=28).
Формы, типы целей и `ignore_index` описаны в
[PyTorch 2.14, cross_entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html).

<a id="nlm-perplexity"></a>

## Perplexity

Perplexity равна $\exp L$, если $L$ — средняя NLL
с натуральным логарифмом на выбранных целевых позициях.
Для равномерного предсказания по нашему словарю из восьми токенов
$L=\log8$, а PPL равна $8$.
Для единственного предсказания `спит` в числовом примере
до шага SGD она равнялась $3$.

Два распределения могут иметь одинаковый правильный `argmax`,
но разную потерю: при вероятности правильного ответа $0{,}4$
получаем $-\log0{,}4\approx0{,}9163$,
а при $0{,}9$ — около $0{,}1054$.
Accuracy не различит такие два правильных ответа,
но NLL и PPL различат.

При сравнении сохраняют одинаковые цели, токенизацию,
доступный контекст и правила маскирования.
Подробный расчёт с EOS, агрегирование по корпусу,
различие логарифмических оснований и численный underflow
уже разобраны
[[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель#Логарифмы и perplexity|в главе о n-граммах]].
Протоколы длинного контекста и нормировка на байты
появятся в разделе о предобучении.

<a id="objective-evidence"></a>

<a id="что-показывает-функция-потерь--и-чего-не-показывает"></a>
<a id="nlm-loss-reduction"></a>

## Средняя потеря по токенам

Пусть $\mathcal M$ — множество позиций, по которым
считается ошибка. При $M$ допустимых целях в батче

$$
L=\frac1M\sum_{(b,t)\in\mathcal M}\ell_{b,t},
\qquad M=|\mathcal M|>0.
$$

Если включены все токены текста и EOS, сумма потерь —
его NLL при заданном BOS.
Если часть целей исключена, остаётся выбранная сумма
условных потерь. Её нельзя автоматически выдавать
за отрицательный логарифм вероятности всего документа.

У трёх наших текстов $4$, $3$ и $4$ цели.
Предположим, средние потери текстов равны $1$, $3$ и $1$.
Потеря по всем токенам равна $(4+9+4)/11=17/11$,
а среднее трёх средних — $5/3$.
Во втором случае короткий текст получил больший вес
на один токен. Оба правила возможны как разные задачи,
но в отчёте должно быть указано выбранное усреднение.

<a id="почему-не-всегда-можно-получить-ноль-на-одном-пакете"></a>
<a id="nlm-entropy-floor"></a>

### Нижняя граница потери при одинаковых контекстах

Для модели с окном в два токена в нашем корпусе
есть повторяющиеся контексты с разными целями:

| Контекст | Цели | Наблюдений |
|---|---|---:|
| BOS BOS | кот: 2; пёс: 1 | 3 |
| BOS кот | кот: 1; спит: 1 | 2 |
| кот спит | EOS: 1; бежит: 1 | 2 |

Как и в n-граммной оценке MLE, на каждой строке
лучшее распределение воспроизводит частоты целей.
После двух BOS получаем вероятности $2/3$ и $1/3$,
поэтому средняя потеря трёх наблюдений для этой строки
равна их энтропии $H(2/3)$. Для двух остальных строк
таблицы цели равновероятны, и каждое из четырёх предсказаний
стоит $-\log(1/2)=\log 2$. В четырёх не вошедших в таблицу
предсказаниях у каждого контекста одна цель, так что их
потери в пределе равны нулю.

Если $H(q)=-q\log q-(1-q)\log(1-q)$,
нижний предел средней потери здесь равен

$$
L_{\min}
=\frac{3H(2/3)+4\log2}{11}
\approx0{,}425648.
$$

При конечных логитах softmax к этому пределу приближаются;
для точного нуля вероятности отдельного класса потребовался бы предел логитов.
Увеличение окна иногда разделяет совпавшие контексты.
В этой выборке даже с полной историей первый контекст
одинаков у всех трёх текстов, а ответы различаются.

При проверке обучения на этом батче ориентиром служит
$0{,}425648$, а не ноль. Для батча без конфликтующих целей
достаточно выразительная модель могла бы подойти к нулю.

<a id="минимальная-реализация-функции-потерь"></a>

## Функция потерь и обучение модели

<a id="nlm-eos-pad-mask"></a>

### Маска позиций и совпадающие ID EOS/PAD

Разрешённые цели зададим булевой маской позиций.
Сравнение ID с `pad_id` не всегда эквивалентно этой маске.
В отдельном примере ниже EOS и padding используют один ID $7$:

```text
tokens:      BOS кот EOS PAD
token_valid:  1   1   1   0
targets:         кот EOS PAD
target_valid:     1   1   0
```

Настоящий EOS остаётся целью.
Если удалить все ID $7$, модель не получит ошибки
за завершение этого текста.
Выбор позиций до вычисления cross-entropy также не допускает
попадания `NaN` из исключённой строки в среднее.

```python
from math import log
import torch
import torch.nn.functional as F

tokens = torch.tensor([[6, 2, 7, 7]])
token_valid = torch.tensor([[True, True, True, False]])
targets = tokens[:, 1:]
keep = token_valid[:, 1:]
logits = torch.zeros(1, 3, 8, dtype=torch.float64)
logits[0, 1, 7] = log(7)  # У настоящего EOS вероятность 1/2.
logits[0, 2] = float("nan")  # Эта позиция исключена, но не настоящий EOS.
logits.requires_grad_()
loss = F.cross_entropy(logits[keep], targets[keep])
torch.testing.assert_close(loss.detach(), torch.tensor(log(4), dtype=logits.dtype))
loss.backward()
torch.testing.assert_close(logits.grad[0, 2], torch.zeros(8, dtype=logits.dtype))
assert logits.grad[0, 1, 7] < 0

wrong_keep = targets != 7
wrong = F.cross_entropy(logits.detach()[wrong_keep], targets[wrong_keep])
torch.testing.assert_close(wrong, torch.tensor(log(8), dtype=logits.dtype))
assert wrong != loss
```

Маска потерь не передаёт модели правила внимания.
В Transformer дополнительно задаются каузальность,
входной padding и, при необходимости, границы документов.
Наш следующий MLP-пример использует только правый padding:
он не может повлиять на окна предыдущих допустимых позиций.

<a id="законченное-обучение-модели-с-окном-в-два-токена"></a>
<a id="nlm-window-training"></a>

### Один батч и модель с окном в два токена

Дополним D2 справа одним PAD, чтобы все три строки были
одинаковой длины. После сдвига у D2 получаются:

```text
Вход:   BOS  пёс   спит  EOS
Цель:   пёс  спит  EOS   PAD
Маска:    1     1    1     0
```

Для третьей цели сеть видит окно `пёс спит` и вычисляет
восемь логитов; для потери нужен токен EOS, то есть ID 7.
Следующая строка логитов соответствует цели PAD, поэтому
её исключает маска. В батче из трёх документов остаётся
$4+3+4=11$ целей. Выражения `logits[keep]` и `targets[keep]`
выбирают одни и те же позиции: из массива `[3, 4, 8]`
получаются 11 строк по 8 логитов и 11 соответствующих ID.
`F.cross_entropy` вычисляет среднюю потерю этих 11 предсказаний.

Для каждой позиции соединяем два вектора токенов длины $2$,
получая вектор длины $4$. Первый линейный слой MLP вычисляет
$8$ скрытых признаков, после `tanh` выходной слой преобразует
их в $8$ логитов словаря.

В модели $128$ параметров:
$8\cdot2$ в embedding, $4\cdot8+8$ в первом линейном слое
и $8\cdot8+8$ в выходном. Окно совпадает с контекстом триграммы,
поэтому результат можно сравнить с частотами продолжений
из предыдущей главы.

```python
from math import log
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(42)
# ID: PAD=0, UNK=1, кот=2, спит=3, пёс=4, бежит=5, BOS=6, EOS=7.
tokens = torch.tensor([[6, 2, 2, 3, 7],
                       [6, 4, 3, 7, 0],
                       [6, 2, 3, 5, 7]])
token_valid = torch.tensor([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1]], dtype=torch.bool)
inputs, targets = tokens[:, :-1], tokens[:, 1:]
keep = token_valid[:, 1:]
assert inputs.shape == targets.shape == (3, 4)
assert keep.sum() == 11

class WindowLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 2, padding_idx=0)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8)
        )

    def forward(self, input_ids):
        windows = F.pad(input_ids, (1, 0), value=6).unfold(1, 2, 1)
        # [B,T,2] → [B,T,2,2] → [B,T,4]; порядок двух токенов сохранён.
        context = self.embedding(windows).flatten(-2)
        return self.network(context)

model = WindowLM().double()
assert sum(p.numel() for p in model.parameters()) == 128

def batch_loss():
    if not keep.any():
        raise ValueError("Нет целевых позиций")
    logits = model(inputs)
    assert logits.shape == (3, 4, 8)
    return F.cross_entropy(logits[keep], targets[keep])

initial = batch_loss().detach().item()
optimizer = torch.optim.Adam(model.parameters(), lr=.03)
for _ in range(800):
    optimizer.zero_grad(set_to_none=True)
    loss = batch_loss()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    final = batch_loss().item()
    left = model(torch.tensor([[6, 2, 3, 5]]))
    right = model(torch.tensor([[6, 2, 4, 2]]))
    torch.testing.assert_close(left[:, :2], right[:, :2])
binary_entropy = -(2/3)*log(2/3) - (1/3)*log(1/3)
floor = (3*binary_entropy + 4*log(2)) / 11
assert floor <= final + 1e-8 and final < floor + .01
assert final < initial
print("initial =", initial, "final =", final, "lower bound =", floor)
```

На проверенном запуске с PyTorch 2.14 CPU потеря снизилась
примерно с $2{,}148789$ до $0{,}425851$.
Нижний предел $0{,}425648$ вычислен по частотам контекстов,
а не подобран под результат кода.
Распределение первого токена приблизилось к `кот: 2/3; пёс: 1/3`,
а после `кот спит` модель распределила почти всю массу
поровну между EOS и `бежит`.

Проверка общего префикса в коде меняет будущее продолжение,
оставляя первые два входа одинаковыми.
Совпадение первых двух выходов подтверждает причинность
этого конкретного MLP. Оно не заменяет такую же проверку
будущей реализации Transformer.

Потерю до и после обучения посчитали на одних и тех же
трёх строках. Этот опыт проверяет, что модель приблизилась
к лучшим оценкам на батче; для проверки на новых текстах
нужен отдельный корпус.

<a id="что-модель-приобретает-через-простую-задачу"></a>
<a id="nlm-objective-limits"></a>

## Частота продолжения и истинность

Если после одной и той же записи `2 + 2 =` в корпусе
трижды стоит `4` и один раз `5`, максимальное правдоподобие
для этой строки даёт вероятности $3/4$ и $1/4$. В целевой
функции нет условия, запрещающего неверное равенство.
На отложенных текстах малая NLL говорит о вероятности наблюдавшихся
продолжений. Для проверки арифметики пришлось бы отдельно сравнить
ответы с верными результатами.

<a id="базовая-модель-и-assistant"></a>
<a id="nlm-base-assistant"></a>

### От продолжения текста к ответу на инструкцию

Базовой модели можно дать начало диалога:

```text
User: вопрос
Assistant:
```

После `Assistant:` она предскажет продолжение так же, как после
любого другого префикса. В опытах [Brown et al.](https://arxiv.org/abs/2005.14165v4)
инструкцию и примеры помещали во входной текст, не меняя веса для
новой задачи. Такой способ использования модели называется
in-context learning.

При дообучении веса меняются. В работе
[Ouyang et al.](https://arxiv.org/abs/2203.02155) GPT-3 сначала
дообучали на ответах разметчиков, а затем использовали сравнения
ответов для RLHF. После этих стадий ответ всё равно строится
последовательно, по одному токену. Как создают обучающие ответы,
разобрано в главе
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|о SFT]].

Для переноса того же расчёта на Transformer см. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|главу о цели предобучения]].
В ней также разобраны длинный контекст и дополнительные цели MTP.

<details>
<summary>Задание с проверкой обучения на малом батче</summary>

Сдвиг, причинность и совпадение с библиотечной функцией потерь
проверяют до обучения на полном корпусе. Эти проверки входят в
[[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля#Этап 3. Функция потерь, оптимизатор, данные и контрольная точка|Capstone 1]].

</details>

## Источники

- Bengio, Ducharme, Vincent, Jauvin,
  [A Neural Probabilistic Language Model, §2 and Figure 1](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf#page=5):
  shared word representations, fixed context window and optional direct connections.
- [CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf),
  softmax в §3.4.4 и cross-entropy/PPL в §4.1.
- Zhang, Lipton, Li, Smola, D2L:
  [Partitioning Sequences](https://d2l.ai/chapter_recurrent-neural-networks/language-model.html#partitioning-sequences),
  [Linear Model и Softmax](https://d2l.ai/chapter_linear-classification/softmax-regression.html#linear-model),
  [Softmax and Cross-Entropy Loss](https://d2l.ai/chapter_linear-classification/softmax-regression.html#softmax-and-cross-entropy-loss).
  Исходники и рисунки закреплены на `23d7a5aecceee57d1292c56e90cce307f183bb0a`.
- Andrej Karpathy, [makemore Part 1](https://github.com/karpathy/nn-zero-to-hero/blob/73c3fcc741f0ec104ca850b1fb0df90e7e8d4cde/lectures/makemore/makemore_part1_bigrams.ipynb),
  переход от таблицы к логитам, NLL, backward и обновлению весов.
- [PyTorch 2.14, cross_entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html):
  ненормированные входы, формы, целочисленные цели и `ignore_index`.
- [Hugging Face, Training a causal language model from scratch](https://huggingface.co/learn/llm-course/en/chapter7/6):
  соглашение о labels и внутреннем сдвиге GPT-2.
- [Brown et al., Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165v4);
  [Ouyang et al., Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling]];
  [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity]].

**Назад:** [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель|N-граммная языковая модель]].

**Дальше:** [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings|Word2Vec, GloVe и распределительные представления]].
