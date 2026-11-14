---
title: Современный decoder block
type: textbook-chapter
status: canonical
last_updated: 2026-11-13
previous: "[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура]]"
next: "[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual]]"
reference_terms:
  - label: Прямоугольная причинная маска с KV-кэшем
    label_en: Rectangular causal masking with a KV cache
    anchor: decoder-rectangular-mask
source_unit_id:
  - lecture-03-hyperparameter-evidence
  - lecture-03-kv-cache-attention
  - lecture-03-sparse-attention
primary_sources:
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf
  - https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py
  - https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/generation.py#L160-L186
  - https://arxiv.org/abs/2204.02311v5
  - https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
  - https://arxiv.org/html/2310.06825v1#S2.F3
  - https://arxiv.org/abs/2001.08361v1
  - https://arxiv.org/abs/2109.10686v2
  - https://arxiv.org/abs/2310.04415v2
  - https://proceedings.mlr.press/v235/liu24ce.html
  - https://proceedings.mlr.press/v306/han26b.html
  - https://arxiv.org/abs/2002.05202v1
  - https://arxiv.org/html/2503.19786v1#S5.SS2
  - https://arxiv.org/html/2505.09388v1#S2
  - https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json
  - https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/config.json
---

# Современный decoder block

<a id="modern-decoder-architecture"></a>

Пусть один блок получил три вектора ширины 4. Их можно обработать
одновременно, а можно сначала обработать первый вектор, сохранить
его ключ и значение, затем добавить оставшиеся два. При тех же
весах, позициях и маске, без dropout оба расчёта дают одинаковые
состояния трёх позиций с точностью до округления.

В [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|предыдущей главе]]
прослежен полный путь LLaMA от токенов до словарных вероятностей.
На том же блоке можно разобрать две детали современных моделей:
[[#qwen3-checkpoint-shapes|почему у Qwen3 ширина Q больше ширины состояния]]
и [[#gemma3-mixed-kv-cache|почему у Gemma 3 слои хранят разное число позиций]].

В [[02 Атлас моделей/Сравнения/Сравнение семейств#llama-qwen-deepseek-worked-comparison|сравнении Llama, Qwen и DeepSeek]]
тот же подсчёт применён к четырём базовым моделям: одинаковое число
позиций в префиксе, разные объёмы кэша и разные матрицы FFN на токен.

<span id="два-последовательных-уточнения"></span>

## Два остаточных сложения

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/cs336-prenorm-block.png|Внизу вход раздваивается: одна ветвь идёт через норму и причинное внимание, другая прямо к сложению. Результат снова раздваивается перед нормой и FFN.]]

*Stanford CS336 Staff, Assignment 1, Version 26.0.3,
[Figure 2, с. 13](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=13),
[MIT](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/LICENSE).
Схема pre-norm-блока из задания, без перерисовки.*

Блок получает $X$ формы $[B,T,d]$, где $B$ — число последовательностей,
$T$ — число обрабатываемых сейчас позиций, $d$ — ширина состояния.
RMSNorm нормализует каждую строку по её $d$ координатам.
Она не усредняет разные позиции и не меняет форму тензора.

В нижнем подслое внимание использует нормализованные строки
для вычисления Q/K/V, но его результат прибавляется к исходному,
ненормализованному $X$. Во втором подслое FFN получает уже новое
состояние $H$:

$$
\begin{aligned}
H&=X+\operatorname{Attn}(R_1(X)),\\
Y&=H+\operatorname{FFN}(R_2(H)).
\end{aligned}
$$

Матрицы двух подслоёв и масштабы $R_1,R_2$ обучаются отдельно.
Оба слагаемых каждого сложения должны иметь форму $[B,T,d]$.
Внутренние размеры могут отличаться: FFN расширяет строку, а
attention разбивает проекции на головы и затем собирает их обратно.

<details>
<summary>Original CS336 description, §3.5 (English)</summary>

> A Transformer block contains two ‘sub-layers’, one for the multihead self
> attention, and another for the SwiGLU feed-forward network. In each
> sub-layer, we first perform RMSNorm, then the main operation (MHA/FF),
> finally adding in the residual connection.

</details>

Такая последовательность используется в LLaMA и в задании CS336,
но не определяет все decoder-only архитектуры. Например,
[PaLM, §2](https://arxiv.org/html/2204.02311v5#S2) вычисляет
attention и MLP параллельно по одному нормализованному входу:

$$
\begin{aligned}
Z&=\operatorname{LayerNorm}(X),\\
Y&=X+\operatorname{Attn}(Z)+\operatorname{MLP}(Z).
\end{aligned}
$$

Разницу зависимостей можно проверить отдельно от нормализации.
Если временно положить $N(x)=x$, $A(x)=x$, $F(x)=x^2$ и взять $x=1$,
последовательная запись даёт $H=1+1=2$, затем $Y=2+2^2=6$.
Параллельная даёт $Y=1+1+1^2=3$.
Это пример композиции функций, а не числовая конфигурация PaLM.

В последовательном блоке FFN этой же глубины уже видит результат
attention; в параллельном — ещё нет. В PaLM параллельная запись
позволила объединить входные матричные умножения двух ветвей.
Авторы получили около 15% ускорения обучения на своём большом
масштабе; при 8B качество немного ухудшилось, при 62B ухудшения
не наблюдалось. Изменение связей между подслоями меняет функцию
модели даже при одинаковых размерах матриц.

## Что происходит внутри attention

Рассмотрим $B=1$, $T=3$, $d=4$, две query-головы ширины 2
и одну общую голову K/V. Входные строки равны

$$
X=\begin{bmatrix}
1&0&1&0\\
0&1&0&2\\
1&1&2&0
\end{bmatrix}.
$$

Пусть $R_1$ использует единичный масштаб и $\epsilon=10^{-5}$.
У последней строки средний квадрат координат равен
$(1^2+1^2+2^2+0^2)/4=1.5$. Деление каждой координаты на
$\sqrt{1.5+\epsilon}$ даёт $(0.8165,0.8165,1.6330,0)$.
Проекция Q в этом примере единичная: первые две координаты
образуют первую query-голову, последние две — вторую.
Проекция K выбирает первые две координаты нормализованной строки,
проекция V — последние две. Все матрицы будут заданы в коде ниже.

Формы проекций и изменение порядка осей:

| Тензор | Форма |
| --- | --- |
| Нормализованный вход | `[1,3,4]` |
| Q | `[1,3,2,2]` |
| K и V | `[1,3,1,2]` |
| Q с осью голов перед позициями | `[1,2,3,2]` |
| K и V с осью голов перед позициями | `[1,1,3,2]` |

RoPE поворачивает Q и K на угол, соответствующий позиции.
Для головы ширины 2 и основания 10000 единственная частота равна 1.
На позиции 2 первая query после поворота равна
$(-1.0822,0.4027)$, вторая — $(-0.6796,1.4849)$.
Векторы V не поворачиваются.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png|В MHA у каждого запроса своя пара K/V, в GQA пары разделяют группы запросов, в MQA одна пара общая для всех запросов.]]

*Ainslie et al., [GQA, Figure 2](https://aclanthology.org/2023.emnlp-main.298/),
[оригинальный PNG](https://arxiv.org/html/2305.13245v3/images/gmq_architecture.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Внизу — query-головы; одна общая пара K/V
может обслуживать одну, несколько или все query-головы.*

В общем GQA число query-голов $h_q$ кратно числу KV-голов $h_{kv}$.
Для $h_q=4$, $h_{kv}=2$ соответствие имеет вид
`Q0,Q1 → KV0` и `Q2,Q3 → KV1`.
Это не чередование `KV0,KV1,KV0,KV1`.
В нашем малом примере $h_q=2$, $h_{kv}=1$: обе query-головы
читают одну и ту же пару K/V. Крайний случай с одной KV-головой
также называется MQA.

Для каждой головы отдельно вычисляются оценки, причинная маска
и нормировка по ключам:

$$
\begin{aligned}
S^{(a)}&=Q^{(a)}(K^{(g(a))})^\top/\sqrt{d_h},\\
A^{(a)}&=\operatorname{softmax}(S^{(a)}+M).
\end{aligned}
$$

Здесь $g(a)$ выбирает KV-группу query-головы, $d_h=2$.
$M_{ij}$ равна нулю при $j\le i$ и $-\infty$ в остальных случаях.
Для последней позиции допустимы все три ключа. Первый ключ
$(1.4142,0)$ не повернулся, поскольку его позиция равна нулю.
Первая голова последней позиции сравнивает свой повёрнутый
запрос с этим ключом и получает
$((-1.0822)\cdot1.4142+0.4027\cdot0)/\sqrt2\approx-1.0822$.
Сравнения со всеми ключами дают

$$
S_{\rm last}\approx
\begin{bmatrix}
-1.0822&0.7135&0.9428\\
-0.6796&0.8691&0.9428
\end{bmatrix},
$$

а веса после softmax —

$$
A_{\rm last}\approx
\begin{bmatrix}
0.0685&0.4126&0.5189\\
0.0929&0.4369&0.4703
\end{bmatrix}.
$$

Несмотря на общие K/V, веса голов различаются: различны их query.
В каждой строке сумма равна единице.
Для первой позиции обе головы получают веса $(1,0,0)$;
для второй третий вес равен нулю.

Три общих V равны примерно $(1.4142,0)$, $(0,1.7888)$
и $(1.6330,0)$. Их взвешенные суммы для последней позиции:

$$
\begin{bmatrix}
0.9442&0.7381\\
0.8993&0.7815
\end{bmatrix}.
$$

Головы сначала возвращают в порядок `[B,T,h_q,d_h]`,
затем объединяют последние две оси.
Получается $(0.9442,0.7381,0.8993,0.7815)$.
Проекция $W_O$ переводит $h_qd_h$ координат обратно в $d$;
в этом примере она единичная.
Прибавление исходного $(1,1,2,0)$ даёт
$H_{\rm last}=(1.9442,1.7381,2.8993,0.7815)$.

Перестановка осей перед объединением обязательна.
Память тензора `[B,h_q,T,d_h]` нельзя просто трактовать как
`[B,T,h_q*d_h]`: у разных позиций перемешаются части голов,
хотя итоговый размер останется правильным.

## Что происходит внутри SwiGLU

Вторая RMSNorm применяется к $H$, а не повторно к $X$.
Для последней позиции нормализованный вход FFN равен примерно
$(0.9777,0.8740,1.4580,0.3930)$.
Две проекции из $d=4$ в $f=6$ вычисляют разные ветви:

$$
\begin{aligned}
G&=R_2(H)W_g,\\
U&=R_2(H)W_u,\\
F&=(\operatorname{SiLU}(G)\odot U)W_d.
\end{aligned}
$$

У $W_g,W_u$ форма $[4,6]$, у $W_d$ — $[6,4]$.
Здесь матрицы записаны для векторов-строк; `nn.Linear` хранит
веса транспонированными.
SiLU применяется только к ветви $G$, затем шесть пар чисел
перемножаются поэлементно. Выходная проекция снова даёт четыре
координаты, которые можно прибавить к $H$.

Для последней позиции значения двух ветвей и их произведения таковы:

| $j$ | $G_j$ | $U_j$ | $\operatorname{SiLU}(G_j)U_j$ |
| --- | ---: | ---: | ---: |
| 1 | −1.5672 | 1.8936 | −0.5123 |
| 2 | −0.9501 | 1.3999 | −0.3709 |
| 3 | −0.3329 | 0.9062 | −0.1260 |
| 4 | 0.2842 | 0.4125 | 0.0669 |
| 5 | 0.9013 | −0.0812 | −0.0521 |
| 6 | 1.5184 | −0.5749 | −0.7161 |

Например, в первой координате

$$
\begin{aligned}
\operatorname{SiLU}(-1.5672)&\approx-0.2705,\\
(-0.2705)\cdot1.8936&\approx-0.5123.
\end{aligned}
$$

Полученные шесть чисел поступают в $W_d$; это ещё не выход FFN
ширины 4. В шестой координате множитель SiLU равен примерно 1.2456,
а в первой он отрицательный. Поэтому слово *gate* здесь не означает
вероятность пропуска из интервала $[0,1]$: множитель может усиливать
координату или менять её знак. В исходной GLU используется sigmoid,
в SwiGLU её заменяют на $x\sigma(x)$
([Shazeer, §2, формулы 4–6](https://arxiv.org/abs/2002.05202v1)).

С полностью заданными ниже весами последние состояния таковы.
В каждой ячейке сначала приведены координаты 1–2, затем 3–4.

| Стадия | Две пары координат |
| --- | --- |
| $H$ | $(1.9442,1.7381)$<br>$(2.8993,0.7815)$ |
| $R_2(H)$ | $(0.9777,0.8740)$<br>$(1.4580,0.3930)$ |
| $F$ | $(0.1864,0.0153)$<br>$(-0.1557,-0.3267)$ |
| $Y=H+F$ | $(2.1306,1.7534)$<br>$(2.7436,0.4547)$ |

FFN не читает другие строки. Но её вход уже содержит результат
attention, поэтому она преобразует информацию контекста,
собранную первой половиной блока.
Матрицы FFN общие для всех позиций этого слоя; у следующего слоя
свои параметры.

## Прямой проход без скрытых шагов

Этот CPU-пример реализует описанный блок и задаёт все 128 его
обучаемых параметров. В нём нет токенизатора, embeddings и словарной
головы: входом служат уже готовые скрытые векторы, выходом — новые
векторы той же ширины. Полная языковая модель разобрана в главе LLaMA.

Используется FP64 для сверки арифметики. В рабочем mixed-precision
коде статистики RMSNorm и softmax обычно вычисляются в FP32;
[реализация Llama 3](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py)
также распределяет матрицы по устройствам и заранее выделяет cache.
Здесь K/V явно копируются для query-групп и сохраняются все
промежуточные состояния — ради доступного расчёта, без обещаний
экономии памяти или скорости.

~~~python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.set_printoptions(precision=6, sci_mode=False)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(
            x.square().mean(-1, keepdim=True) + self.eps
        ) * self.gain

class DecoderBlock(nn.Module):
    def __init__(self, d=4, nq=2, nkv=1, dh=2, f=6, theta=10000.):
        super().__init__()
        assert nq % nkv == 0 and dh % 2 == 0
        self.nq, self.nkv, self.dh = nq, nkv, dh
        self.theta = theta
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, nq * dh, bias=False)
        self.k = nn.Linear(d, nkv * dh, bias=False)
        self.v = nn.Linear(d, nkv * dh, bias=False)
        self.o = nn.Linear(nq * dh, d, bias=False)
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def rope(self, x, positions):
        freq = self.theta ** (
            -torch.arange(0, self.dh, 2, dtype=x.dtype) / self.dh
        )
        angle = positions[:, None] * freq[None, :]
        c, s = angle.cos(), angle.sin()
        pairs = x.reshape(*x.shape[:-1], self.dh // 2, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        return torch.stack((a*c - b*s, a*s + b*c), -1).flatten(-2)

    def forward(self, x, start_pos=0, cache=None, window=None):
        b, t, _ = x.shape
        assert t > 0
        assert (cache is None and start_pos == 0) or (
            cache is not None and cache[0].shape[2] == start_pos)
        positions = start_pos + torch.arange(t)
        r = self.norm1(x)
        q = self.q(r).reshape(b, t, self.nq, self.dh).transpose(1, 2)
        k = self.k(r).reshape(b, t, self.nkv, self.dh).transpose(1, 2)
        v = self.v(r).reshape(b, t, self.nkv, self.dh).transpose(1, 2)
        q, k = self.rope(q, positions), self.rope(k, positions)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = (k, v)
        # Explicit copies simplify this CPU example, not a serving kernel.
        k_heads = k.repeat_interleave(self.nq // self.nkv, dim=1)
        v_heads = v.repeat_interleave(self.nq // self.nkv, dim=1)
        key_positions = torch.arange(k.shape[2])
        allowed = key_positions[None, :] <= positions[:, None]
        if window is not None:
            assert window >= 1
            allowed &= key_positions[None, :] > positions[:, None] - window
        scores = q @ k_heads.transpose(-1, -2) / math.sqrt(self.dh)
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        mixed = weights @ v_heads
        joined = mixed.transpose(1, 2).contiguous().reshape(b, t, -1)
        h = x + self.o(joined)
        z = self.norm2(h)
        gate, up = self.gate(z), self.up(z)
        activated = F.silu(gate)
        product = activated * up
        ff = self.down(product)
        y = h + ff
        return y, new_cache, {
            "q": q, "k": k, "v": v, "scores": scores,
            "weights": weights, "mixed": mixed, "h": h, "z": z,
            "gate": gate, "up": up, "activated": activated,
            "product": product, "ff": ff, "y": y
        }

model = DecoderBlock().double()
with torch.no_grad():
    model.q.weight.copy_(torch.eye(4))
    model.k.weight.copy_(torch.tensor([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.]]))
    model.v.weight.copy_(torch.tensor([[0., 0., 1., 0.],
                                     [0., 0., 0., 1.]]))
    model.o.weight.copy_(torch.eye(4))
    grid = torch.arange(24, dtype=torch.float64).reshape(6, 4)
    model.gate.weight.copy_((grid - 11.5) / 24)
    model.up.weight.copy_(grid.flip(0) / 30 - 0.2)
    model.down.weight.copy_(
        torch.arange(24, dtype=torch.float64).reshape(4, 6) / 60 - 0.15)
x = torch.tensor([[[1., 0., 1., 0.],
                   [0., 1., 0., 2.],
                   [1., 1., 2., 0.]]], dtype=torch.float64)
with torch.no_grad():
    y, cache, trace = model(x)
    for name in ("q", "k", "v", "scores", "weights", "mixed", "h", "z",
                 "gate", "up", "activated", "product", "ff", "y"):
        print(name, trace[name])
    first, prefix_cache, _ = model(x[:, :1])
    rest, _, _ = model(x[:, 1:], start_pos=1, cache=prefix_cache)
    torch.testing.assert_close(y, torch.cat((first, rest), dim=1),
                               rtol=1e-12, atol=1e-12)
    print("stored K/V:", cache[0].shape, cache[1].shape)
    local, _, local_trace = model(x, window=2)
    print("local weights:", local_trace["weights"][0, :, -1])
    print("local last output:", local[0, -1])
~~~

Вызов с двумя новыми позициями использует `start_pos=1`, поэтому RoPE
получает индексы 1 и 2. Проверка в конце сравнивает все строки,
не только последний ответ. `cache` содержит K и V формы `[1,1,3,2]`,
без размножения на две query-головы.

В примере нет padding, packed documents, dropout, удаления старого
cache или разбиения между GPU. Все последовательности в пакете
должны иметь одинаковую действительную длину. Для этих дополнительных
режимов нужны соответствующие маски и учёт позиций; форма входа сама
по себе их не задаёт.

<span id="обучение-prefill-и-decode--три-разных-режима-чтения-блока"></span>

## Полный проход, prefill и продолжение из cache

При обучении вход уже содержит все токены отрезка.
Причинная маска позволяет вычислить состояния одновременно.
Для позиции $i$ результат зависит только от позиций $0,\ldots,i$;
вход следующего блока сохраняет это свойство.
FFN и нормализация работают построчно и не открывают доступ к будущему.
Так причинность сохраняется через весь стек.

При генерации первый проход обрабатывает известный prompt — это prefill.
Для каждого слоя сохраняются его собственные K/V. Они получены
из входа именно этого слоя: ключи первого слоя не заменяют ключи второго.
Когда добавляется токен, прежние скрытые состояния не меняются,
поэтому прежние K/V можно использовать повторно.
Новый токен вычисляет Q, K и V; внимание читает cache вместе с
новыми K/V, после чего FFN преобразует только новые строки.
Прошлые Q и выходы FFN для этого не нужны.

Cache допустим при тех же весах, входном префиксе, позиционных
преобразованиях и видимости. Например, после обновления весов
сохранённые K/V уже не соответствуют новой модели.
На обучении промежуточные состояния также нужны для backward;
inference-cache не заменяет вычислительный граф.

<a id="decoder-rectangular-mask"></a>

### Прямоугольная маска

После одной сохранённой позиции добавим сразу две.
Новые query относятся к глобальным позициям 1 и 2;
ключи — к позициям 0, 1 и 2.
Матрица разрешённых пар равна

$$
\begin{bmatrix}
1&1&0\\
1&1&1
\end{bmatrix}.
$$

Первая новая строка читает прежний префикс и себя, но не следующую
новую строку. Нижнетреугольная матрица `tril(ones(2,3))`
дала бы другой результат:
первая query увидела бы только ключ 0, вторая — ключи 0 и 1.
Нужны индексы позиций, а не только размер прямоугольника.
В [`Transformer.forward` Llama 3](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py)
к треугольнику новых позиций слева добавляются столбцы старого префикса.
В учебном коде тот же результат даёт сравнение
`key_positions <= positions`.

Сохранённые K уже содержат RoPE. Если при продолжении снова
вращать query с позиции 0, она сравнивается с ключами в неверных
относительных фазах. Размеры тензоров при этом останутся правильными.

У [PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
флаг `is_causal=True` строит треугольник от левого верхнего угла.
Это верно для полного прохода, где первая строка Q и первый столбец K
относятся к одной позиции. При продолжении первая строка Q относится
к позиции 1, а первый столбец K всё ещё к позиции 0. Флаг оставит
первой голове только значение позиции 0, $(1.4142,0)$, вместо смеси
значений позиций 0 и 1, $(0.2985,1.4112)$. Форма результата не изменится.

Передадим SDPA явную булеву маску, где `True` разрешает пару Q/K.
Следующий код использует `model` и `x` из предыдущего примера.
Ключи и значения уже содержат префикс; RoPE применён в `model`.
Сравниваем выход внимания до проекции $W_O$ и остаточного сложения.

~~~python
with torch.no_grad():
    _, past, _ = model(x[:, :1])
    _, _, chunk = model(x[:, 1:], start_pos=1, cache=past)
    q, k, v = (chunk[name] for name in ("q", "k", "v"))
    repeat = model.nq // model.nkv
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    i = torch.arange(1, 3)[:, None]
    j = torch.arange(3)[None, :]
    visible = j <= i
    sdpa = F.scaled_dot_product_attention
    correct = sdpa(q, k, v, attn_mask=visible)
    wrong = sdpa(q, k, v, is_causal=True)
    torch.testing.assert_close(correct, chunk["mixed"])
    assert not torch.allclose(wrong, correct)
    print(wrong[0, 0, 0], correct[0, 0, 0])
    # tensor([1.414199, 0.000000]) tensor([0.298538, 1.411221])

    # Same keys and positions, but the window includes at most two tokens.
    local_visible = visible & (j > i - 2)
    assert local_visible.tolist() == [[True, True, False], [False, True, True]]
    _, _, local_chunk = model(x[:, 1:], start_pos=1, cache=past, window=2)
    torch.testing.assert_close(
        sdpa(q, k, v, attn_mask=local_visible), local_chunk["mixed"])
~~~

Для сплошного суффикса без padding и локального окна подойдёт также
[`causal_lower_right`](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.bias.causal_lower_right.html):
он смещает диагональ на разницу длин K и Q. У отдельной библиотеки
[FlashAttention начиная с 2.1](https://github.com/Dao-AILab/flash-attention/blob/v2.1.0/README.md#changes-in-v21-compared-to-v20)
флаг `causal=True` выровнен по правому нижнему углу. Поэтому при
замене вызова проверяют, какие позиции видит запрос, а не только
имя флага или размер выхода.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/long-context-2026-10-23/mistral-chunk-mask.png|Четыре строки новых запросов. В первой строке единицы стоят под тремя токенами из кэша и первым токеном новой порции; в последней разрешены все четыре новых токена и ни одного старого.]]

*Albert Q. Jiang et al., [Mistral 7B, рис. 3, с. 3](https://arxiv.org/pdf/2310.06825v1#page=3).
Исходная маска и английская подпись, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

На рисунке окно включает четыре позиции, считая текущую.
Первый новый `the` видит `mat`, `and`, `saw` и себя.
Последний новый `to` видит только четыре токена своей порции.
Это та же проверка $i-w<j\le i$, что в примере с окном 2.
Если перезаписать весь старый кэш новой порцией до вычисления первой
строки, `the` потеряет три нужных ключа. Как хранить хвост и обновлять
его после вычисления порции, разобрано в
[[00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание#Кольцевой кэш при обработке нескольких новых токенов|главе о длинном контексте]].

### Два блока и два кэша

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png|Векторы двух входных позиций проходят через первый блок. Его выход становится входом второго блока, а не возвращается в таблицу токенов.]]

*Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[оригинальный PNG](https://jalammar.github.io/images/gpt2/gpt2-transformer-block-vectors-2.png),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Нормализации, остаточные связи и KV-cache на этом упрощённом рисунке
не показаны; схема показывает последовательность блоков.*

Поставим подряд две отдельные копии учебного блока. У первой вход $X$,
у второй — весь выход первой $Y^{(1)}$. Начальные значения параметров
у копий одинаковы, но сами параметры не связаны: обновление одной
копии не меняет другую. После первой входной строки $(1,0,1,0)$
выход первого блока равен примерно $(2.5437,0.0155,2.3158,-0.2124)$.
Второй блок нормализует уже этот вектор и получает другие K/V:

| Слой | Ключ K и значение V позиции 0 |
| --- | --- |
| Первый | K: $(1.4142,0)$<br>V: $(1.4142,0)$ |
| Второй | K: $(1.4761,0.0090)$<br>V: $(1.3438,-0.1232)$ |

При продолжении каждый блок дописывает новые K/V к собственному
кэшу. Первый блок обрабатывает новые входные
векторы; второй получает только их новые выходы из первого блока.
Пересчитывать старые выходы первого блока для этого не требуется:
они уже участвовали в создании сохранённых K/V второго блока.

Следующий код выполняется после определения `model` и `x` выше:

~~~python
import copy

layers = nn.ModuleList([copy.deepcopy(model), copy.deepcopy(model)])

def through_stack(inp, caches=None, start_pos=0):
    if caches is None:
        caches = [None] * len(layers)
    assert len(caches) == len(layers)
    h, next_caches = inp, []
    for layer, past in zip(layers, caches):
        h, present, _ = layer(h, start_pos=start_pos, cache=past)
        next_caches.append(present)
    return h, next_caches

with torch.no_grad():
    full, full_caches = through_stack(x)
    prefix, prefix_caches = through_stack(x[:, :1])
    suffix, final_caches = through_stack(
        x[:, 1:], caches=prefix_caches, start_pos=1)
    torch.testing.assert_close(
        full, torch.cat((prefix, suffix), dim=1),
        rtol=1e-12, atol=1e-12)
    for expected, actual in zip(full_caches, final_caches):
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    print("stack last:", full[0, -1])
    for index, (keys, values) in enumerate(prefix_caches):
        print("layer", index, "prefix K/V:", keys, values)
~~~

Последняя строка выхода двух блоков равна примерно
$(3.1494,2.5612,3.3513,0.9441)$. В конце у каждого слоя
K и V формы `[1,1,3,2]`: всего 24 числа, или 192 байта в FP64.
Размеры кэшей совпадают, но их содержимое различно.
Подстановка K/V первого слоя во второй не вызовет ошибки формы
и всё же даст другой результат.

<a id="decoder-first-token"></a>

### Как из последней строки получается первый токен ответа

На схеме Hugging Face модель уже выдала `will`: его выбрали по
логитам последней позиции исходного запроса, `ject`. Сам `will`
в первом проходе через блоки ещё не участвовал. На следующем
проходе модель строит Q, K и V для `will`. К семи парам K/V
исходного запроса добавляется восьмая, и состояние позиции `will`
теперь позволяет предсказать `be`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/hf-kv-cache.png|Один запрос для will читает семь сохранённых пар K/V и новую пару для will. Стрелка справа ведёт к следующему токену be.]]

*Rémi Ouazan Reboul, Arthur Zucker и Luc Georges,
[Continuous batching, KV-cache](https://huggingface.co/blog/continuous_batching#kv-cache),
[оригинальный PNG](https://huggingface.co/datasets/huggingface/documentation-images/resolve/c8c3cf72b3b724ce734259e4a086d30618a9198b/blog/continuous_batching/kv_cache.png),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Рисунок без изменений. Головы опущены; $A$ обозначает ширину головы.*

Проверим этот порядок на тех же $X$ и `model`. Пусть первые две
строки $X$ соответствуют входу `a cat`. Дадим блоку словарь из
четырёх токенов в порядке `a`, `cat`, `sleeps`, `<eos>`.
После блока применим отдельную финальную RMSNorm и учебную выходную
проекцию. Её четыре строки выбирают из нормализованного вектора
координаты 1, 2, 4 и 3 соответственно. Это фиксированные веса
для проверки индексов, а не обученная языковая модель.

| Проход | Логиты | Токен и кэш |
| --- | --- | --- |
| Prefill<br>`a cat`, позиция 1 | 0.1996, 1.1409,<br>1.6305, 0.0027 | Выход: `sleeps`<br>Кэш: 0, 1 |
| Decode<br>`sleeps`, позиция 2 | 1.0877, 0.8951,<br>0.2321, 1.4006 | Выход: `<eos>`<br>Кэш: 0, 1, 2 |

В первой строке таблицы наибольший логит у `sleeps`. Для жадного
выбора вычислять softmax не нужно: он сохраняет порядок логитов.
Только на втором проходе берём представление выбранного токена
(третью строку $X$) и передаём его блоку с позицией 2.
После выбора `<eos>` останавливаемся, поэтому в этом примере
ключа и значения для `<eos>` в кэше нет.

~~~python
final_norm = RMSNorm(4).double()
vocab = ("a", "cat", "sleeps", "<eos>")
embeddings = dict(zip(vocab[:3], x.unbind(dim=1)))

def logits(states):
    return final_norm(states)[..., [0, 1, 3, 2]]

with torch.no_grad():
    prompt_states, past, _ = model(x[:, :2])
    prompt_logits = logits(prompt_states)[0, -1]
    first = int(prompt_logits.argmax())
    assert vocab[first] == "sleeps" and past[0].shape[2] == 2

    next_input = embeddings[vocab[first]].unsqueeze(1)
    state, past, _ = model(next_input, start_pos=2, cache=past)
    decode_logits = logits(state)[0, -1]
    assert vocab[int(decode_logits.argmax())] == "<eos>"
    assert past[0].shape[2] == 3
    full_states, _, _ = model(x)
    torch.testing.assert_close(
        logits(full_states)[0, 1:3], torch.stack((prompt_logits, decode_logits)))
~~~

В [`Transformer.forward` Llama 3](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L256-L279)
после всех блоков тоже идут финальная норма и выходная проекция.
[`generate`](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/generation.py#L160-L186)
берёт `logits[:, -1]`, записывает выбранный токен и только в
следующей итерации подаёт его на вход. Для пакета запросов разной
длины тот же цикл сохраняет исходный токен там, где запрос ещё
не закончился. В нашем расчёте последовательность одна.

### Почему кэш для генерации не заменяет граф обучения

В примерах `torch.no_grad()` отключает запись графа: веса уже заданы,
проверяется только прямой проход. В исходной реализации Llama 3
прямой проход `Transformer.forward` выполняется с декоратором
[`@torch.inference_mode()`](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py).

Для обучения возьмём искусственную скалярную цель
$\mathcal L=\lVert Y^{(2)}_{2}\rVert^2/4$: средний квадрат четырёх
координат последней позиции второго блока. Это не языковая потеря,
а способ проследить производные. При полном проходе градиент по
первой входной строке равен примерно
$(-0.3164,0.1146,0.3253,0.2356)$.
Последняя позиция читает прошлые K/V, поэтому её потеря зависит
и от более ранних входов.

Если разделить проход на префикс и суффикс, сохраняя граф у обоих
кэшей, выходы и градиенты совпадут с полным проходом. Если же
передать в суффикс `detach()`-копии K/V обоих слоёв, численные
значения внимания останутся прежними, но градиент по первой входной
строке станет нулевым. Её выход отдельно не входит в выбранную цель,
а все пути к последней позиции проходили через отсоединённые K/V.
Совпадение прямых результатов поэтому не доказывает эквивалентность
обучения. Обычный inference-cache сохраняет значения для продолжения;
backward требует ещё и зависимостей, по которым были получены эти значения.

### Почему время двух проходов различается

При prefill проекции и FFN получают сразу $T$ строк и выполняют
матричные умножения. При обычном decode поступает одна новая строка
на последовательность. Для неё нужно прочитать веса и накопленные
K/V, а следующий шаг нельзя завершить, пока не выбран предыдущий токен.

В прямой реализации полные scores имеют $Bh_qT^2$ элементов.
При однотокенном продолжении — $Bh_qT_{\rm cache}$,
где $T_{\rm cache}$ включает и ключ текущей позиции.
Это не означает, что любой prefill обязан хранить всю квадратную
матрицу: FlashAttention вычисляет тот же оператор блоками без её
полной записи. И наоборот, маленькая матрица scores не делает
decode дешёвым: остаётся чтение больших весов и cache.
Вклад каждого из них зависит от размера модели, пакета и длины истории.
Системные измерения разобраны в
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|главе о KV-cache и пакетировании]].

<a id="efficient-attention-boundary"></a>
<span id="что-именно-экономят-gqa-mqa-и-разрежённое-attention"></span>

## Параметры, KV-cache и разрежённая видимость

При общей ширине головы $d_h$ четыре проекции attention содержат

$$
P_{\rm attn}=2d(h_q+h_{kv})d_h
$$

весов: $d h_qd_h$ у Q, столько же у O,
по $d h_{kv}d_h$ у K и V.
У FFN ещё $3df$, у двух RMSNorm — $2d$.
Для малого блока получается $48+72+8=128$.
Замена одной KV-головы двумя при прежних $d,h_q,d_h,f$
дала бы 144 параметра: изменились бы только проекции K/V.

В $L$ слоях cache для $B$ последовательностей длины $T$ содержит

$$
N_{\rm KV}=2BLT h_{kv}d_h
$$

чисел. В нашем примере $2\cdot1\cdot1\cdot3\cdot1\cdot2=12$:
96 байт в FP64. С двумя KV-головами было бы 192 байта.
При 32 query- и 8 KV-головах cache вчетверо меньше, чем у MHA
с 32 KV-головами, при одинаковых остальных размерах.
Но каждая query-голова всё ещё вычисляет собственные scores
и взвешенную сумму V. Число этих голов не сократилось вчетверо.

В [CS336 Lecture 3, с. 58–63](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf#page=58)
MQA/GQA объясняются через уменьшение обменов с памятью при
инкрементальном выводе. Сокращение KV-storage не равно сокращению
всей памяти процесса или времени шага. В частности, учебный
`repeat_interleave` создаёт временные копии K/V для наглядного
расчёта; специализированное ядро может работать с группами без таких копий.

### Локальное окно меняет доступный контекст

GQA сохраняет множество разрешённых позиций и меняет число
независимых K/V-проекций.
Оконное внимание ограничивает сами позиции.
Пусть окно $w=2$ включает текущую позицию и одну предыдущую.
Тогда query позиции 2 читает ключи 1 и 2, но не 0.
В коде это проверяется условием $i-w<j\le i$.

У нашего блока последние строки весов становятся

$$
A_{\rm local}\approx
\begin{bmatrix}
0&0.4429&0.5571\\
0&0.4816&0.5184
\end{bmatrix}.
$$

Оставшиеся веса перенормированы; нельзя просто обнулить первый
столбец уже готового $A$ и оставить две другие вероятности прежними.
Последний выход блока теперь примерно
$(2.0984,1.8076,2.6884,0.5298)$ вместо
$(2.1306,1.7534,2.7436,0.4547)$.

За один такой слой позиция 2 не читает позицию 0 напрямую.
За два слоя информация может пройти по пути $0\to1\to2$.
В стеке из $L$ одинаковых causal-окон без глобальных связей
наибольшее расстояние такого пути — $L(w-1)$ позиций.
Доступность пути ещё не означает, что обученная модель сохранит
нужную информацию без потерь.

Полное внимание имеет порядка $T^2$ разрешённых пар;
фиксированное окно — порядка $Tw$. Для реальной экономии ядро
должно пропускать запрещённые пары. Учебный параметр `window`
меняет результат, но код всё ещё строит плотные scores и хранит
весь cache: ускорение или уменьшение памяти им не продемонстрировано.

[CS336, с. 64–66](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf#page=64)
показывает также чередование оконных и полных слоёв.
Полный causal-слой снова даёт каждой позиции прямой доступ ко всей
предшествующей истории. Поэтому число слоёв, порядок типов внимания
и реальное окно входят в описание модели наряду с размерами матриц.
Например, у закреплённой
[Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/27d67f1b5f57dc0953326b2601d68371d40ea8da/config.json)
есть отдельное поле `sliding_window=4096`; его нельзя восстановить
из одного `hidden_size`.

<a id="gemma3-mixed-kv-cache"></a>

### Сколько истории хранит каждый слой в Gemma 3

В [отчёте Gemma 3, §2](https://arxiv.org/html/2503.19786v1#S2)
после каждых пяти локальных слоёв с окном 1024 идёт один глобальный.
Если реализация при декодировании вытесняет старые K/V из локальных
слоёв, каждый из пяти хранит не более 1024 позиций. Шестой слой
хранит всю последовательность. При 32 768 токенах это
$5\cdot1024+32768=37\,888$ позиций, суммарно по шести слоям.
Если все шесть слоёв глобальные, получится $6\cdot32768=196\,608$.
При одинаковых размерах K/V и типе данных первая группа занимает
около 19,3% памяти второй. До 1024 токенов обе группы хранят
одинаковое число позиций.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma/gemma-3-kv-cache.svg|Память KV-cache двух моделей на 2 млрд параметров. При длинном контексте кэш варианта со всеми глобальными слоями растёт быстрее, чем у варианта с пятью локальными слоями на один глобальный.]]

*Gemma Team / Google DeepMind, [Gemma 3 Technical Report, рис. 6](https://arxiv.org/html/2503.19786v1#S5.F6),
[оригинальный SVG](https://arxiv.org/html/2503.19786v1/mem_lc.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Без изменений.*

На графике обе линии относятся к 2B-моделям из абляции §5.2.
При 128K жёлтая линия доходит примерно до 7000 МБ, а красная —
до 1100 МБ. Красная линия тоже растёт: глобальным слоям по-прежнему
нужны ключи и значения всех позиций. Эти величины относятся только
к кэшу авторского опыта. Они не описывают память весов, пик при
prefill или время ответа выпущенной Gemma 3. Само ограничение маски
не освобождает память: в учебном блоке выше `window` запрещает
обращение к старым позициям, но старые K/V всё ещё лежат в тензоре.

## От GPT-2 к LLaMA: что сохранилось, а что заменили

И GPT-2, и последовательный LLaMA-блок сначала выполняют
masked self-attention, затем FFN, сохраняя внешнюю ширину состояния.
GPT-2 уже использует pre-norm: LayerNorm стоит перед каждым подслоем.
Поэтому переход от GPT-2 к LLaMA не переносит нормализацию из
post-norm в pre-norm.

| Часть | GPT-2 | Первая LLaMA |
| --- | --- | --- |
| Норма | LayerNorm | RMSNorm |
| Позиции | Обучаемая абсолютная таблица | RoPE у Q/K |
| FFN | Две проекции, GELU | Три проекции, SwiGLU |
| Головы | MHA | MHA во всех четырёх размерах |
| Bias проекций | Есть | Нет |
| Таблицы словаря | Общие веса | Раздельные веса |

GQA — последующее изменение отдельных выпусков Llama 2
и всех апрельских Llama 3, а не свойство первой LLaMA.
Параллельные подслои PaLM — ещё одно независимое изменение.
Внешний вид decoder-only модели не сообщает эти детали;
их определяют формулы, конфигурация и реализация конкретного выпуска.

<a id="decoder-hyperparameter-evidence"></a>

## Как выбирают размеры блока

Увеличение $d$, $f$, числа голов или слоёв меняет одновременно
число весов и размеры матричных умножений.
При фиксированном бюджете эти изменения компенсируют друг друга.
Например, в MHA с $h_qd_h=d$ и SwiGLU при $f\approx8d/3$
один блок без малых членов содержит около $12d^2$ параметров.
Два слоя ширины $d$ и один слой ширины $\sqrt2d$ поэтому имеют
примерно одинаковое число весов. Но это разные вычислительные
графы, размеры активаций и последовательная глубина.

[CS336, с. 36–51](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf#page=36)
сопоставляет распространённые размеры с экспериментальными
основаниями для них. Привычное отношение служит отправной точкой;
доказательства оптимальности требуют заданных данных, бюджета
и метрики.

### Ширина FFN

Обычная FFN с двумя матрицами содержит $2df$ весов.
Для $f=4d$ это $8d^2$.
SwiGLU содержит три матрицы: тот же бюджет даёт
$3df=8d^2$, то есть $f=8d/3$.
Округление и дополнительные множители изменяют точное число.

У конкретных выпущенных моделей встречаются, например, такие размеры:

| Модель | $d$ | $f$ одной ветви | $f/d$ |
| --- | ---: | ---: | ---: |
| [T5 v1.1 Base](https://huggingface.co/google/t5-v1_1-base/blob/b5fc947a416ea3cb079532cb3c2bbadeb7f800fc/config.json) | 768 | 2048 | 2.667 |
| [Qwen1.5 14B](https://huggingface.co/Qwen/Qwen1.5-14B/blob/dce4b190d34470818e5bec2a92cb8233aaa02ca2/config.json) | 5120 | 13696 | 2.675 |
| [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/27d67f1b5f57dc0953326b2601d68371d40ea8da/config.json) | 4096 | 14336 | 3.5 |
| [Исходная T5-11B](https://github.com/google-research/text-to-text-transfer-transformer/blob/42909aa15d1fbd16c69333aa474a00ae8ed9fde4/t5/models/bi_v1_11B.gin) | 1024 | 65536 | 64 |

У T5 v1.1 используется GEGLU, у Qwen1.5 и Mistral — SwiGLU,
у исходной T5-11B — двухматричная ReLU-FFN.
Отношение 64 у последней не означает 64-кратное увеличение
всего блока: внимание и residual-путь имеют свои размеры.
Оно также не относится к T5 v1.1 XXL.

Название поля конфигурации не всегда совпадает с $f$.
В исходном
[Qwen-14B, `QWenMLP`](https://huggingface.co/Qwen/Qwen-14B/blob/680a3dab6f81d17abb140f57fe9a54c6b63cc988/modeling_qwen.py)
две входные проекции имеют ширину `intermediate_size // 2`.
При `intermediate_size=27392` ширина каждой — 13696.
В Qwen1.5 поле `intermediate_size=13696` уже задаёт ширину одной ветви.
Сравнение только одноимённых полей дало бы ложное различие в два раза.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/decoder-design-2026-11-10/mobilellm-design-figure3.png|Первые две строки MobileLLM: при 134.1 млн параметров средняя zero-shot accuracy растёт с 42.6% до 43.9% после перехода на SwiGLU.]]

*У первых двух цветных полос одинаковое число параметров: 134.1 млн.
Средняя точность по восьми zero-shot задачам выросла с 42.6% до 43.9%.
Штрихованные полосы относятся к более крупной группе, около 350 млн.
Zechun Liu et al., [MobileLLM, ICML 2024](https://proceedings.mlr.press/v235/liu24ce.html),
[рис. 3, с. 3](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24ce/liu24ce.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Рисунок и английская подпись из PDF.*

В обеих конфигурациях на 134.1 млн параметров по 12 слоёв ширины 768.
Авторы заменили двухматричную ReLU-FFN на SwiGLU, у которой три матрицы.
Сравнивать здесь нужно первые две строки: с третьей меняются уже глубина
и ширина всей сети. В группе около 350 млн параметров переход на SwiGLU
тоже повысил среднюю точность (47.4% → 48.7%), но число параметров выросло
с 376.8 до 386.7 млн. Точные размеры и результаты приведены в
[Table 10, с. 14](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24ce/liu24ce.pdf#page=14).

В [Kaplan et al., §2–3.1, Figure 5](https://arxiv.org/abs/2001.08361v1)
проверяли другой вопрос: как меняется потеря при изменении $f/d$ и
приблизительно постоянном числе параметров без embeddings. Вместе с $f$
меняли $d$, а небольшой остаток различий корректировали по зависимости
потерь от числа параметров. Для моделей около 50 млн параметров на WebText2
получилась область близких потерь от $f/d\approx1$ до нескольких единиц.
Крайние отношения давали худший результат. Этот опыт не измеряет, что
произойдёт, если расширить только FFN, оставив остальные размеры прежними.

### Число голов и размер головы

Проекция Q имеет форму $[d,h_qd_h]$, а O — $[h_qd_h,d]$.
Поэтому равенство $h_qd_h=d$ удобно, но не обязательно.
У исходной T5-11B $d=1024$, $h_q=128$, $d_h=128$:
внутренняя ширина внимания равна 16384, в 16 раз больше residual-ширины.
Именно 11B, а не любой T5, имеет эту конфигурацию.

Даже при фиксированном $h_qd_h$ разбиение на головы меняет функцию.
У двух голов два независимых softmax по ключам.
Их нельзя заменить одним softmax после склейки всех координат.
В числовом примере выше головы читают общие K/V,
но получают разные распределения весов.

Наконец, отдельно выбирается $h_{kv}$.
Сократить KV-головы можно без изменения $h_qd_h$ и выходной ширины.
При реализации GQA нужны допустимые группы, а при tensor parallelism —
согласованное разбиение голов. Требования делимости конкретного ядра
или библиотеки не являются математическим определением attention.

<a id="qwen3-checkpoint-shapes"></a>

#### Проверка по весам Qwen3

У [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json)
`hidden_size=1024`, `num_attention_heads=16`,
`num_key_value_heads=8` и `head_dim=128`. Если поделить
ширину состояния на число query-голов, выйдет 64. Это не размер
головы этой модели: Q расширяется до $16\cdot128=2048$ координат.
В таблице формы весов PyTorch записаны в порядке
`[out_features, in_features]`. Они сверены с заголовком
[`model.safetensors`](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/model.safetensors).

| Проекция | Форма веса первого слоя | Результат для 3 токенов |
| --- | --- | --- |
| Q | `[2048,1024]` | `[1,3,16,128]` после разбиения на головы |
| K | `[1024,1024]` | `[1,3,8,128]` после разбиения |
| V | `[1024,1024]` | `[1,3,8,128]` после разбиения |
| O | `[1024,2048]` | `[1,3,1024]` после объединения query-голов |

Две query-головы читают одну пару K/V, как в средней панели
[[#Что происходит внутри attention|рисунка Ainslie et al.]].
После проекций Qwen3 применяет ещё одну RMSNorm к каждому вектору
Q и K по его 128 координатам. Веса `q_norm` и `k_norm` имеют по
128 элементов: каждый вектор масштабов используется всеми головами
соответствующего типа. Затем RoPE поворачивает Q и K, а в кэш
записываются уже повёрнутые K и обычные V.
Порядок видно в
[`Qwen3Attention.forward`](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py#L214-L227).
Норма входа подслоя остаётся на месте; эффект дополнительной нормы
на оценки внимания разобран [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual#QK normalization|на числовом примере в следующей главе]].

Четыре проекции содержат вместе 6 291 456 весов, три матрицы SwiGLU —
9 437 184, а четыре нормы блока добавляют $2\cdot1024+2\cdot128=2304$.
Итого 15 730 944 параметра на слой. В стеке 28 слоёв. Добавив
одну общую таблицу входа и выхода размера $151936\times1024$
и финальную RMSNorm, получим 596 049 920 параметров. В заголовке
перечислено 310 тензоров. Если сложить произведения их размеров,
выходит то же число. Значения весов для этой проверки не нужны.

Сравним с [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/config.json).
В ней 32 query-головы, но по-прежнему восемь KV-голов по 128 координат.
Кэш одного слоя на токен при BF16 в обеих моделях занимает
$2\cdot8\cdot128\cdot2=4096$ байт. На 4096 токенах и с учётом
всех слоёв это 448 МиБ у 0.6B (28 слоёв) и 576 МиБ у 8B (36 слоёв).
Здесь считается только сохранённый K/V одной последовательности.
Размеры Q и FFN влияют на другие затраты. По одному числу
параметров модели этот кэш не вычислить.

### Глубина и ширина

При близком числе весов более глубокая сеть имеет больше
последовательных преобразований; более широкая выполняет более
крупные умножения и хранит более широкие активации.
Если один слой зависит от результата предыдущего,
удвоение числа слоёв нельзя устранить их параллельным запуском
для одного токена.

Для GPT-3 175B отношение $d/L$ равно $12288/96=128$,
для LLaMA-65B — $8192/80=102.4$.
У исходной T5-11B ширина 1024 и по 24 слоя в каждом из двух стеков,
то есть $1024/24\approx42.7$ при подсчёте одного стека.
Суммирование encoder и decoder дало бы другое число и не описывало
бы один decoder-only стек. Точные размеры поэтому задаются
вместе с архитектурой и выпуском модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/decoder-design-2026-11-10/mobilellm-depth-figure4.png|Шесть панелей MobileLLM сравнивают глубину в двух размерных группах на zero-shot reasoning, TQA и RACE.]]

*Слева, на панелях (a) и (b), каждая линия соответствует одной глубине.
Последняя отметка оси, Average, усредняет восемь предыдущих задач.
Zechun Liu et al., [MobileLLM, ICML 2024](https://proceedings.mlr.press/v235/liu24ce.html),
[рис. 4, с. 4](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24ce/liu24ce.pdf#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Все панели и английская подпись из PDF.*

Сравним две линии на панели (a). У 12 слоёв ширины 768 было 134.1 млн
параметров, у 30 слоёв ширины 512 — 135.0 млн. Средняя точность выросла
с 43.9% до 44.8%, но на ARC-Challenge снизилась с 28.9% до 26.1%.
При сужении сети авторы также сократили число голов с 12 до 8, сохранив
по 64 координаты на голову. Это сравнение двух форм сети близкого размера,
а не увеличение только числа слоёв.
Числа: [MobileLLM, Table 11, с. 15](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24ce/liu24ce.pdf#page=15).

В [Kaplan et al., Figure 5](https://arxiv.org/abs/2001.08361v1)
в широком диапазоне $d/L$ потери предобучения оставались близкими при
примерно одинаковом числе параметров.
В [Tay et al., §3–4.1, Table 3](https://arxiv.org/abs/2109.10686v2)
модель NL12-XXL с 3.6B параметров имеет лучший показатель
предобучения, чем NL32-XL с 3.8B: negative log-perplexity
$-1.46$ против $-1.49$ — здесь выше лучше.
Однако после fine-tuning SuperGLUE у неё 76.5 против 79.9.
Это encoder-decoder эксперимент со span-corruption на C4,
а не доказательство преимущества глубины для всех causal LM.

При выборе формы модели нужны и целевая метрика, и время обучения
или вывода. Два варианта с близким loss могут по-разному использовать
устройства: отличаются размеры матриц, число последовательных
операций и обмены между частями модели.

### Размер словаря

Словарь задаёт матрицы embeddings и выхода, но длина входа
зависит ещё и от правил токенизации.
При $d=4096$ две несвязанные матрицы с $V=32000$
содержат вместе 262 144 000 параметров.
При $V=128000$ их было бы вчетверо больше.
При связывании входных и выходных весов считается одна таблица.

Более дробная токенизация, напротив, увеличивает работу слоёв.
Если один и тот же текст представлен 100 токенами вместо 80,
построчные проекции и FFN получают в 1.25 раза больше строк,
а полная матрица attention — примерно в 1.5625 раза больше пар.
Увеличение словаря может сократить такую длину, но само по себе
не гарантирует этого для конкретного языка или корпуса.

Размеры словарных матриц в закреплённых конфигурациях выше:
32128 у T5 v1.1 Base, 152064 у Qwen1.5-14B и 32000 у Mistral-7B-v0.1.
Поле `vocab_size` может включать служебные и резервные строки
или округление для оборудования.
Его нельзя без проверки называть числом обычных текстовых токенов.
Редким токенам также достаётся меньше обучающих наблюдений;
длина последовательности — не единственный критерий словаря.

### Dropout и weight decay

Режим обучения задаётся отдельно от формулы блока.
Например, первая LLaMA не использует dropout, но обучается
с AdamW и weight decay 0.1.
У T5 v1.1 dropout выключен при предобучении, а
[описание выпущенных checkpoints](https://github.com/google-research/text-to-text-transfer-transformer/blob/42909aa15d1fbd16c69333aa474a00ae8ed9fde4/released_checkpoints.md)
рекомендует снова включать его при fine-tuning.
В сохранённом config T5 v1.1 Base стоит `dropout_rate=0.1`:
эта настройка не опровергает исторический режим предобучения.

В AdamW weight decay уменьшает веса при обновлении параметра.
Если градиентный вклад на данном шаге нулевой, при
$w=2$, $\eta=0.001$ и $\lambda=0.1$ получится
$(1-\eta\lambda)w=1.9998$.
Это изменение траектории оптимизации даже до обсуждения
разницы training и validation loss.

В [D'Angelo et al., §3](https://arxiv.org/html/2310.04415v2#S3)
рассматривается GPT-2-124M на OpenWebText, обычно с контекстом
256, AdamW, скоростью 0.0006 и cosine-спадом в десять раз.
При weight decay 0.1/0.3 улучшается конечный training loss,
хотя generalization gap близок к нулю.
Поэтому объяснения только через предотвращение переобучения
для этого эксперимента недостаточно.

Авторы связывают результат с изменением нормы весов и
эффективной скорости обучения. В FP32 им удалось подобрать
расписание без weight decay, повторяющее динамику loss.
В отдельных BF16-режимах такое сопоставление приводило к
скачкам потерь; при контексте 1024 и скорости 0.001 weight decay
помогал избежать расходимости.
Это результаты и объяснение конкретной работы, не универсальная
эквивалентность AdamW любому изменению learning rate.

Если после предобучения модель предстоит дообучать, проверочной потери
базы может оказаться недостаточно для выбора weight decay. Han et al.
предобучили несколько вариантов каждой архитектуры, меняя коэффициент
AdamW. Затем варианты одной группы дообучили на одних и тех же задачах
и сравнили точность после SFT.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/decoder-design-2026-11-10/han-weight-decay-figure3.png|По горизонтали проверочная потеря базы, по вертикали точность после SFT. В одной группе есть варианты с близкими потерями и разной точностью.]]

*Сравнивайте точки одного цвета: внутри группы совпадают архитектура,
размер и бюджет токенов, а меняется weight decay предобучения.
Tessa Han et al., [Weight Decay Improves Language Model Plasticity, ICML 2026](https://proceedings.mlr.press/v306/han26b.html),
[рис. 3, с. 6](https://raw.githubusercontent.com/mlresearch/v306/main/assets/han26b/han26b.pdf#page=6),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Все панели и английская подпись из PDF.*

В группе OLMo-2-1B-140x проверочная потеря базы при коэффициенте 0.3
равна 2.6208 против 2.6088 при 0.1. Если выбирать только по этой потере,
победил бы вариант 0.1. После SFT вариант 0.3 показал лучшую точность
на задачах рассуждения. В каждой группе было не более десяти точек;
результат даёт повод проверить качество после SFT на своих задачах,
но не переносить коэффициент 0.3 на любое предобучение.

<a id="decoder-evidence"></a>
<span id="что-считать-подтверждением-корректной-сборки"></span>
<span id="где-реализация-чаще-всего-расходится-с-формулой"></span>

## Проверка блока по промежуточным результатам

Совпадение $[B,T,d]$ на входе и выходе не обнаружит перепутанные
группы GQA или неверные позиции RoPE.
В нашем примере есть более точные проверки.
Обе последние строки внимания должны совпасть с приведёнными
числами, запрещённые вероятности — равняться нулю,
а $H$, $F$ и $Y$ — соответствовать последовательным сложениям.

Полный проход и продолжение из cache сравниваются для одних
входов и весов, без dropout. Изменение будущей строки не должно
менять уже вычисленные ранние выходы.
Отдельно сравниваются градиенты по входу и параметрам:
перестановка или detach может не испортить forward на частном
входе, но изменить обучение.

Более быстрая реализация обязана сохранять эти вычисления
в пределах ошибок выбранной точности.
Для измерений скорости затем убирают trace, выбирают подходящее
ядро attention и проверяют заданные длины, пакет и оборудование.
Наш CPU-пример устанавливает числовой результат блока;
он не измеряет производительность serving.

## Источники и реализации

- [Stanford CS336, Assignment 1, §3.4–3.5](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf): последовательная сборка pre-norm-блока, формы и подсчёт ресурсов; Figures 1–3.
- [Stanford CS336, Lecture 3](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf): с. 36–51 — гиперпараметры и основания их выбора, с. 57–66 — KV-cache и разрежённая видимость.
- [Закреплённая Llama 3](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py): отдельные Q/KV-головы, RoPE, cache, прямоугольная маска и два residual-сложения. CPU-код главы следует этим операциям, но не копирует CUDA/model-parallel инфраструктуру.
- [Qwen3 Technical Report, §2](https://arxiv.org/html/2505.09388v1#S2): настройки плотных моделей, переход от смещений QKV к QK-norm. Формы и размеры в примере взяты из конкретных конфигураций Base и заголовка тензоров.
- [Gemma 3 Technical Report, §§2 и 5.2](https://arxiv.org/html/2503.19786v1#S5.SS2): порядок локальных и глобальных слоёв, размер окна и абляция памяти кэша.
- [Shazeer, GLU Variants, §2](https://arxiv.org/abs/2002.05202v1): две ветви, произведение координат и замена sigmoid на Swish в SwiGLU.
- [PaLM, §2](https://arxiv.org/html/2204.02311v5#S2): параллельные attention/MLP и условия сравнения с последовательной записью.
- [Kaplan et al., §2–3.1, Figure 5](https://arxiv.org/abs/2001.08361v1): влияние формы при фиксированных non-embedding параметрах.
- [Tay et al., §3–4.1, Table 3](https://arxiv.org/abs/2109.10686v2): различие pre-training loss и downstream-результата после fine-tuning.
- [D'Angelo et al., §3](https://arxiv.org/abs/2310.04415v2): weight decay и динамика оптимизации малой LM.
- [Liu et al., MobileLLM, §2.2 и приложение](https://proceedings.mlr.press/v235/liu24ce.html): варианты FFN, глубины и ширины у моделей до миллиарда параметров.
- [Han et al., Weight Decay Improves Language Model Plasticity, §§3–4](https://proceedings.mlr.press/v306/han26b.html): выбор weight decay по потере базы и по результату после SFT.
- [[02 Areas/ML & DL/01 Справочник/Attention/MHA|Multi-head attention]], [[02 Areas/ML & DL/01 Справочник/FFN и MoE/SwiGLU|SwiGLU]] и [[02 Areas/ML & DL/02 Атлас моделей/Семейства/Llama|семейство Llama]].

**Назад:** [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA]].
**Далее:** [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|Нормализация и остаточные связи]].

Полный подсчёт стоимости продолжает
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|глава о формах, параметрах и FLOPs]];
реализация всей LM — [[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля|Capstone 1]].
