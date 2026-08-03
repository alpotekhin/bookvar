---
title: Dense FFN — token-wise вычисление, expansion и gating
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/2002.05202
  - https://arxiv.org/abs/2002.12327
---

# Dense FFN: вычисление внутри каждого токена

Attention смешивает информацию **между позициями**; feed-forward network (FFN,
MLP) преобразует каждый полученный вектор **внутри позиции**. Attention отвечает
«откуда прочитать», FFN — «как нелинейно переработать прочитанное». Поскольку
одна и та же FFN применяется ко всем токенам, её называют position-wise.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-dense-ffn/transformer-block-ffn.png]]

*FFN — отдельная ветвь после self-attention в decoder block. Автор: Jay Alammar,
страница [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
прямой файл [gpt2-transformer-block-vectors-2.png](https://jalammar.github.io/images/gpt2/gpt2-transformer-block-vectors-2.png).
На сайте указана лицензия текста/иллюстраций [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).* 

## Position-wise не значит «без контекста»

Пусть вход блока $X\in\mathbb R^{B\times T\times d}$. Обычная FFN:

$$
H=\phi(XW_1+b_1),\qquad Y=HW_2+b_2,
$$

где $W_1\in\mathbb R^{d\times m}$, $W_2\in\mathbb R^{m\times d}$. Для каждой
позиции $t$:

$$
y_t=W_2^\top\phi(W_1^\top x_t+b_1)+b_2.
$$

Нет суммы по $t$: токены можно сложить в большую матрицу и выполнить GEMM, но
они не перемешиваются. Однако $x_t$ уже содержит результат attention, то есть
контекст соседей. FFN локальна по оси sequence, не по смыслу.

```python
# x: [batch, tokens, d_model]
h = activation(x @ W1)  # [batch, tokens, d_ff]
y = h @ W2              # [batch, tokens, d_model]
```

Одинаковые веса для всех позиций дают translation-like equivariance по sequence:
перестановка токенов переставит выходы FFN так же. Позиционность и связь между
токенами приходят из embeddings и attention.

## Зачем сначала расширять размерность

Если бы обе проекции были линейны, $W_2W_1$ можно было бы свернуть в одну
матрицу. Нелинейность между ними создаёт набор признаков в пространстве ширины
$m=d_{ff}$. В исходном Transformer $m=4d$ и ReLU. BERT/GPT используют GELU;
современные decoder-only модели часто используют SwiGLU.

Коэффициент расширения $r=m/d$ — не универсальная константа. Широкий
промежуточный слой увеличивает число нелинейных признаков и параметров, но также
расход памяти на активации и число FLOP. При фиксированном бюджете вентильная
архитектура обычно требует уменьшить $m$.

| Вариант | Формула без bias | Число матриц | Параметры |
|---|---|---:|---:|
| ReLU/GELU | $\phi(XW_1)W_2$ | 2 | $2dm$ |
| GLU | $(XW_a\odot\sigma(XW_g))W_o$ | 3 | $3dm$ |
| GEGLU | $(XW_a\odot\operatorname{GELU}(XW_g))W_o$ | 3 | $3dm$ |
| SwiGLU | $(XW_a\odot\operatorname{SiLU}(XW_g))W_o$ | 3 | $3dm$ |

Размерности проще всего проверять по весам. Первая матрица имеет форму
$[d_{model},d_{ff}]$ и превращает каждый вектор остаточного потока в более
широкий вектор. Вторая имеет форму $[d_{ff},d_{model}]$ и возвращает результат
к ширине residual stream. На рисунке ниже эти же матрицы показаны в разбиении
по двум GPU. Пока не нужно следить за коммуникациями: сначала найдите подписи
`d_model × d_ff/2` слева и `d_ff/2 × d_model` справа. Две половины вместе
восстанавливают обычный путь $d_{model}\rightarrow d_{ff}\rightarrow d_{model}$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/tensor-parallel-split.svg]]

*Формы матриц расширяющей и сжимающей проекций на примере Megatron-style tensor
parallelism. Слева столбцы первой матрицы делятся между GPU, справа строки второй
матрицы делятся согласованным образом. Источник: Harvard Edge ML Systems Book,
[Distributed Training, Figure 14](https://mlsysbook.ai/vol2/distributed_training/distributed_training.html#fig-tensor-parallel-split),
[исходный файл курса](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd),
CC BY-NC-SA 4.0.*

## Gating: значение и пропуск в разных проекциях

В обычной GELU-FFN нелинейность применяется к единственной расширяющей
проекции. В gated-варианте вход проецируется дважды. Одна ветвь создаёт
значения, которые могут попасть в выход; другая вычисляет для каждой координаты
плавный множитель. Поэтому слово *gate* здесь означает не бинарный выключатель,
а обучаемое поэлементное масштабирование.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png]]

*SiLU, ReLU и тождественная функция. Смотрите на отрицательную область и
окрестность нуля: ReLU полностью обнуляет отрицательные значения, тогда как
SiLU оставляет небольшой отрицательный выход и плавно переходит к почти
линейному режиму. Сам рисунок не является схемой SwiGLU, но две его кривые
помогают прочитать формулу ниже: value-ветвь остаётся линейной, а к gate-ветви
применяется синяя функция SiLU перед умножением. Источник: Stanford
CS336, Assignment 1: Basics, Figure 3,
[с. 21](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf#page=21),
[репозиторий задания](https://github.com/stanford-cs336/assignment1-basics).*

SwiGLU вычисляет:

$$
U=XW_u,\qquad G=\operatorname{SiLU}(XW_g),\qquad
Y=(U\odot G)W_d.
$$

$U$ несёт candidate features, $G$ непрерывно регулирует, какие координаты и с
каким знаком пропустить. Это не маршрутизатор MoE: SwiGLU не выбирает отдельную
сеть и не создаёт разреженной пересылки; все три плотные проекции выполняются для каждого
токена.

Эту формулу полезно читать буквально слева направо:

1. `up_proj` строит $U=XW_u$ формы $[B,T,d_{ff}]$ — value-ветвь;
2. `gate_proj` независимо строит $XW_g$ той же формы;
3. SiLU преобразует только gate-ветвь;
4. поэлементное произведение $U\odot G$ не меняет форму и не смешивает токены;
5. `down_proj` возвращает результат из $d_{ff}$ в $d_{model}$.

Именно совпадение форм двух ветвей делает умножение корректным. Это также
объясняет, почему в реализации `gate_proj` и `up_proj` иногда объединяют в одну
матрицу с выходной шириной $2d_{ff}$: математически ветви остаются разными, но
один крупный GEMM и последующее разбиение могут быть выгоднее двух запусков
ядра.

Чтобы сравнить gated и ungated FFN при похожем числе параметров, приравняем:

$$
2d(4d)=3dm_{\text{SwiGLU}}
\quad\Rightarrow\quad
m_{\text{SwiGLU}}\approx\frac{8}{3}d.
$$

Отсюда типичные промежуточные ширины около $2.67d$, часто округлённые под
эффективный размер ядра. Сравнивать «4d GELU против 4d SwiGLU» как архитектурно
равные варианты нельзя: второй имеет в 1.5 раза больше весов FFN и матричных умножений.

## Точный parameter accounting

Для dense GELU-FFN с bias:

$$
P_{FFN}=dm+m+md+d=2dm+m+d.
$$

Для bias-free SwiGLU:

$$
P_{SwiGLU}=3dm.
$$

При $d=4096$, $m=11008$ один SwiGLU слой содержит:

$$
3\cdot4096\cdot11008=135{,}266{,}304
$$

параметра — примерно 135.3M. Для 32 слоёв это 4.33B параметров только FFN.
Именно поэтому FFN часто содержит большую часть весов декодерной модели.

## FLOPs и MACs: сначала договоримся о единицах

Матрица $[N,d]$ на $[d,m]$ требует $Ndm$ multiply-accumulate operations (MAC).
Если одно умножение и одно сложение считать как 2 FLOPs, то:

$$
\text{FLOPs}_{GELU}\approx4Ndm,
$$

$$
\text{FLOPs}_{SwiGLU}\approx6Ndm,
$$

где $N=B\cdot T$ — число строк токенов. Поэлементная активация и вентиль добавляют
$O(Nm)$ операций, обычно меньше матричных умножений, но не бесплатно. Обратный
проход требует хранить или пересчитывать промежуточные активации и выполняет
дополнительные матричные умножения, поэтому число FLOP при обучении нельзя
выдавать за стоимость прямого прохода.

Для одного токена FFN без смещений использует все веса слоя. Для $N$ токенов
веса переиспользуются $N$ раз: число параметров не растёт, а объём вычислений
растёт линейно с $N$.

## Почему FFN дорогая, даже когда attention квадратична

Сравним leading terms одного слоя:

$$
\text{attention projections}\sim 4Td^2,
\qquad
\text{attention mixing}\sim 2T^2d,
\qquad
\text{SwiGLU}\sim 3Tdm\;\text{MAC}.
$$

При умеренном $T$ и $m\approx 2.7d$ плотная FFN может доминировать по параметрам
и вычислениям. Квадратичный член attention начинает доминировать лишь при
достаточно большом $T$. Реальное время определяется не одной асимптотикой, но и
размером пакета, вычислительной интенсивностью, объединением ядер и пропускной
способностью памяти.

## Что может хранить FFN

Иногда отдельные neurons FFN интерпретируют как key–value memories: строки первой
матрицы обнаруживают некоторый шаблон в контекстном состоянии, а столбцы второй
записывают связанное направление обратно в residual stream. Эта картина полезна,
потому что объясняет, как одинаковая token-wise функция реализует множество
условных преобразований. Но это не буквальная база фактов: признаки распределены
между координатами и слоями, активация зависит от контекста, а один нейрон редко
имеет устойчивое человеческое значение. Работа Geva et al. предлагает
эмпирический анализ такого представления FFN как памяти; использовать его нужно как измеряемую
гипотезу, а не как доказательство, что «каждый факт лежит в одном нейроне».

Практический эксперимент: сохранить промежуточные activations $H$ для корпуса,
найти токены с максимальной активацией выбранной координаты и проверить
устойчивость паттерна на другой выборке. Затем обнулить координату и измерить
изменение logits. Корреляция примеров без causal intervention недостаточна.

## Тензорный параллелизм и память активаций

FFN удобно делить по промежуточной размерности. При разбиении первой проекции по
столбцам каждый GPU строит часть $H$; вторая проекция, разбитая по строкам, суммирует частичные
выходы. Для SwiGLU парные $W_u$ и $W_g$ должны иметь согласованное разбиение,
чтобы elementwise product был локальным.

Промежуточный тензор $[B,T,m]$ крупнее residual stream $[B,T,d]$. Activation
Сохранение только контрольных активаций уменьшает расход памяти ценой повторного
прямого прохода. Объединённое ядро SwiGLU
может не записывать все промежуточные тензоры в HBM.

Теперь можно вернуться к схеме Harvard выше. Для обычной FFN column-parallel
разбиение применяется к расширяющей матрице. В SwiGLU одинаково делят обе
матрицы $W_u$ и $W_g$: один rank должен получить соответствующие части $U$ и
$G$, чтобы вычислить локальное $U\odot G$ без обмена. Сжимающая $W_d$ делится
по строкам, а частичные результаты суммируются после неё. Ветвление SwiGLU
добавляет матрицу, но не требует отдельного collective между gate и value.

## Как читать конфигурацию модели

Поля `hidden_size`, `intermediate_size`, `hidden_act`, `mlp_bias` и число слоёв
позволяют восстановить FFN budget. Но нужно открыть код: названия `gate_proj`,
`up_proj`, `down_proj` показывают SwiGLU-подобную тройку, а fused checkpoint
может хранить несколько проекций в одном tensor.

> [!danger] Частые ошибки
> FFN не смешивает токены; $4d$ — исторический выбор, не закон; SwiGLU gate не
> делает вычисление sparse; «параметры» и «FLOPs на последовательность» имеют
> разные единицы; маркетинговое число active parameters не заменяет layer-wise
> accounting.

## От одной dense FFN к нескольким экспертам

Dense FFN выполняет один и тот же набор матриц для каждого токена. Естественный
следующий вопрос — можно ли увеличить число таких преобразований, но для
конкретного токена выполнять лишь несколько. Именно эту замену делает sparse
Mixture of Experts: вместо одной FFN появляется набор FFN-экспертов и отдельный
router.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-moe/mixtral-smoe-layer.png]]

*Переход от одной FFN к sparse MoE. Router выбирает для представления токена
несколько expert FFN, после чего их выходы складываются с весами маршрутизации.
Сравните эту схему с SwiGLU: SwiGLU умножает две координатные ветви **внутри
каждой** FFN и вычисляет их для всех токенов; MoE-router выбирает **между целыми
FFN** и создаёт разреженное выполнение. Авторы иллюстрации: Omar Sanseviero,
Lewis Tunstall, Philipp Schmid, Sourab Mangrulkar, Younes Belkada и Pedro
Cuenca; источник: Hugging Face,
[Mixture of Experts Explained](https://huggingface.co/blog/moe),
[прямой файл](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_smoe-layer.png).*

Следующая глава разбирает то, чего нет у dense FFN: top-$k$ routing, capacity,
балансировку нагрузки, dispatch токенов между устройствами и стоимость serving.

## После главы нужно уметь

- указать ось, по которой FFN независима;
- вывести формы всех тензоров ReLU-FFN и SwiGLU;
- посчитать параметры, MACs и FLOPs с явно указанной конвенцией;
- подобрать ширину SwiGLU при бюджете обычной $4d$ FFN;
- объяснить, почему FFN содержит контекст, хотя не смешивает позиции.

## Материалы для дальнейшего чтения

- [Stanford CS336](https://cs336.stanford.edu/) — tensor shapes, parameter/FLOP
  accounting; [CS25](https://web.stanford.edu/class/cs25/) — связь Transformer
  block с современными архитектурами.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/) и
  [Transformers Llama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
  — реальные `gate_proj/up_proj/down_proj`.
- [D2L Transformer, PositionwiseFFN](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
  — формы тензоров и применение одной и той же MLP к каждой позиции.
- [Full Stack Deep Learning LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)
  и [Chip Huyen on LLM engineering](https://huyenchip.com/2023/04/11/llm-engineering.html)
  — serving и hardware-aware accounting.
- Первичные работы: [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202),
  [GLU](https://arxiv.org/abs/2002.12327),
  [Transformer FFN Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
  и [Megatron-LM](https://arxiv.org/abs/1909.08053).
