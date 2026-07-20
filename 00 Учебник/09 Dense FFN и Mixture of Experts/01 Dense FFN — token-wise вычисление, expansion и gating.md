---
title: Dense FFN — token-wise вычисление, expansion и gating
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
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

Expansion ratio $r=m/d$ — не универсальная константа. Широкий intermediate
слой увеличивает число нелинейных признаков и параметры, но также activation
memory и FLOPs. При фиксированном бюджете gating обычно требует уменьшить $m$.

| Вариант | Формула без bias | Число матриц | Параметры |
|---|---|---:|---:|
| ReLU/GELU | $\phi(XW_1)W_2$ | 2 | $2dm$ |
| GLU | $(XW_a\odot\sigma(XW_g))W_o$ | 3 | $3dm$ |
| GEGLU | $(XW_a\odot\operatorname{GELU}(XW_g))W_o$ | 3 | $3dm$ |
| SwiGLU | $(XW_a\odot\operatorname{SiLU}(XW_g))W_o$ | 3 | $3dm$ |

## Gating: значение и пропуск в разных проекциях

SwiGLU вычисляет:

$$
U=XW_u,\qquad G=\operatorname{SiLU}(XW_g),\qquad
Y=(U\odot G)W_d.
$$

$U$ несёт candidate features, $G$ непрерывно регулирует, какие координаты и с
каким знаком пропустить. Это не MoE-router: SwiGLU не выбирает отдельную сеть и
не создаёт sparse dispatch; все три dense-проекции выполняются для каждого
токена.

Чтобы сравнить gated и ungated FFN при похожем числе параметров, приравняем:

$$
2d(4d)=3dm_{\text{SwiGLU}}
\quad\Rightarrow\quad
m_{\text{SwiGLU}}\approx\frac{8}{3}d.
$$

Отсюда типичные промежуточные ширины около $2.67d$, часто округлённые под
эффективный размер kernel. Сравнивать «4d GELU против 4d SwiGLU» как архитектурно
равные варианты нельзя: второй имеет в 1.5 раза больше FFN weights и matmul.

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
Именно поэтому FFN часто хранит большую часть weights decoder model.

## FLOPs и MACs: сначала договоримся о единицах

Матрица $[N,d]$ на $[d,m]$ требует $Ndm$ multiply-accumulate operations (MAC).
Если одно умножение и одно сложение считать как 2 FLOPs, то:

$$
\text{FLOPs}_{GELU}\approx4Ndm,
$$

$$
\text{FLOPs}_{SwiGLU}\approx6Ndm,
$$

где $N=B\cdot T$ — число token rows. Elementwise activation и gating добавляют
$O(Nm)$, обычно меньше matmul, но не бесплатны. Backprop требует хранить или
пересчитывать промежуточные активации и выполняет дополнительные matmul, поэтому
training FLOPs нельзя выдавать за forward FLOPs.

Для одного токена bias-free FFN число активных weights равно числу параметров
слоя. Для $N$ токенов веса переиспользуются $N$ раз: parameter count не растёт,
compute растёт линейно с $N$.

## Почему FFN дорогая, даже когда attention квадратична

Сравним leading terms одного слоя:

$$
\text{attention projections}\sim 4Td^2,
\qquad
\text{attention mixing}\sim 2T^2d,
\qquad
\text{SwiGLU}\sim 3Tdm\;\text{MAC}.
$$

При умеренном $T$ и $m\approx 2.7d$ dense FFN может доминировать по параметрам и
вычислениям. Квадратичный член attention начинает доминировать лишь при достаточно
большом $T$. Chip Huyen и Full Stack Deep Learning добавляют production-вывод:
реальное время зависит от batch, arithmetic intensity, kernel fusion и memory
bandwidth, а не от одной асимптотики.

## Что может хранить FFN

Иногда отдельные neurons FFN интерпретируют как key–value memories: строки первой
матрицы детектируют некоторый шаблон в contextualized state, а столбцы второй
записывают связанное направление обратно в residual stream. Эта картина полезна,
потому что объясняет, как одинаковая token-wise функция реализует множество
условных преобразований. Но это не буквальная база фактов: признаки распределены
между координатами и слоями, activation зависит от контекста, а один neuron редко
имеет устойчивое человеческое значение. Работа Geva et al. предлагает
эмпирический анализ такого memory view; использовать его нужно как измеряемую
гипотезу, а не как доказательство, что «каждый факт лежит в одном нейроне».

Практический эксперимент: сохранить промежуточные activations $H$ для корпуса,
найти токены с максимальной активацией выбранной координаты и проверить
устойчивость паттерна на другой выборке. Затем обнулить координату и измерить
изменение logits. Корреляция примеров без causal intervention недостаточна.

## Tensor parallelism и activation memory

FFN удобно делить по intermediate dimension. В column-parallel первой проекции
каждая GPU строит часть $H$; вторая проекция row-parallel суммирует частичные
выходы. Для SwiGLU парные $W_u$ и $W_g$ должны иметь согласованное разбиение,
чтобы elementwise product был локальным.

Промежуточный тензор $[B,T,m]$ крупнее residual stream $[B,T,d]$. Activation
checkpointing уменьшает сохранение ценой повторного forward. Fused SwiGLU kernel
может не записывать все промежуточные тензоры в HBM.

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

## После главы нужно уметь

- указать ось, по которой FFN независима;
- вывести формы всех тензоров ReLU-FFN и SwiGLU;
- посчитать параметры, MACs и FLOPs с явно указанной конвенцией;
- подобрать ширину SwiGLU при бюджете обычной $4d$ FFN;
- объяснить, почему FFN содержит контекст, хотя не смешивает позиции.

## Сопоставление учебных объяснений и источники

| Подраздел | Выбранная основа объяснения | Почему она сильнее сравненных альтернатив | Готовый визуал |
|---|---|---|---|
| position-wise computation | [D2L Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) | D2L показывает формы `[batch,time,features]` и одинаковый MLP по позициям; HF course даёт больше API, чем механизма | Jay Alammar block выше |
| expansion | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | задаёт исходный $d\to4d\to d$ baseline; CS336 сильнее для accounting, но не для исторической конструкции | формулы и таблица variants |
| gating | [GLU Variants](https://arxiv.org/abs/2002.05202) | напрямую сравнивает GLU/GEGLU/SwiGLU при контроле бюджета; model docs лишь фиксируют готовую конфигурацию | таблица формул из paper, адаптированная выше |
| parameters/FLOPs | [Stanford CS336](https://cs336.stanford.edu/) | требует явной MAC/FLOP convention и layer-wise accounting; FSDL сосредоточен на end-to-end serving | численный пример $4096\times11008$ |
| memory interpretation | [Geva et al.](https://arxiv.org/abs/2012.14913) | даёт измеряемый key-value-memory анализ; общие лекции Karpathy/D2L не проверяют эту интерпретацию | готовые эксперименты paper перенесены как протокол, без новой схемы |
| parallel execution | [Megatron-LM](https://arxiv.org/abs/1909.08053) | конкретно выводит column/row split MLP; HF implementation скрывает collective | tensor-shape description, без авторской схемы |

- [Stanford CS336](https://cs336.stanford.edu/) — tensor shapes, parameter/FLOP
  accounting; [CS25](https://web.stanford.edu/class/cs25/) — связь Transformer
  block с современными архитектурами.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/) и
  [Transformers Llama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
  — реальные `gate_proj/up_proj/down_proj`.
- [D2L Transformer, PositionwiseFFN](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
  — наиболее ясное объяснение shared MLP по позициям.
- [Full Stack Deep Learning LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)
  и [Chip Huyen on LLM engineering](https://huyenchip.com/2023/04/11/llm-engineering.html)
  — serving и hardware-aware accounting.
- Первичные работы: [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202),
  [GLU](https://arxiv.org/abs/2002.12327),
  [Transformer FFN Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
