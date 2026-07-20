---
title: LLaMA как базовая архитектура
type: textbook-chapter
status: canonical
last_updated: 2026-07-18
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
next: "[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA]]"
primary_sources:
  - https://arxiv.org/abs/2302.13971
  - https://arxiv.org/abs/1910.07467
  - https://arxiv.org/abs/2104.09864
  - https://arxiv.org/abs/2002.05202
---

# LLaMA как базовая архитектура современной LLM

LLaMA не вводит новый класс нейронных сетей. Это причинный Transformer, в
котором несколько накопившихся к 2023 году решений собраны в простой и хорошо
описанный блок: нормализация стоит перед подслоем, LayerNorm заменена на RMSNorm,
позиции кодируются вращением запросов и ключей, а обычная полносвязная сеть —
вентильным SwiGLU. Именно эта комбинация стала удобной точкой отсчёта для
анализа Llama, Mistral, Qwen и многих последующих открытых моделей.

Важно не приписывать весь успех LLaMA форме блока. Токенизатор, состав данных,
число обучающих токенов и процедура дообучения менялись между поколениями не
меньше архитектуры. Поэтому глава сначала разбирает путь одного тензора через
блок LLaMA 1, а затем отдельно отмечает изменения Llama 2 и Llama 3.

## Сначала вся модель на одной странице

Для токенов $t_1,\ldots,t_T$:

```text
token ids
   ↓ embedding table
x₀ ∈ ℝ[T × d_model]
   ↓
┌──────────────── Transformer block × L ────────────────┐
│                                                       │
│  x ────────────────┐                                  │
│  ↓ RMSNorm         │ residual                         │
│  ↓ causal attention with RoPE                         │
│  └────────────── + ┘                                  │
│  h ────────────────┐                                  │
│  ↓ RMSNorm         │ residual                         │
│  ↓ SwiGLU FFN      │                                  │
│  └────────────── + ┘                                  │
│                                                       │
└───────────────────────────────────────────────────────┘
   ↓ final RMSNorm
   ↓ vocabulary projection
logits ∈ ℝ[T × |V|]
```

Каждый block выполняет две разные работы:

- attention переносит информацию между позициями;
- FFN независимо преобразует состояние каждой позиции.

Residual stream хранит общий результат, к которому оба подслоя добавляют
поправки.

Формулы одного pre-norm блока:

$$
h=x+\operatorname{Attention}(\operatorname{RMSNorm}(x)),
$$

$$
y=h+\operatorname{SwiGLU}(\operatorname{RMSNorm}(h)).
$$

Эти две строки полезно держать рядом с официальным
[Llama 3 `TransformerBlock`](https://github.com/meta-llama/llama3/blob/main/llama/model.py):

```python
h = x + self.attention(self.attention_norm(x), ...)
out = h + self.feed_forward(self.ffn_norm(h))
```

Код почти буквально повторяет математику.

## Что унаследовано от GPT

LLaMA остаётся causal language model:

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

От GPT-линии сохранены:

- token embeddings;
- stack одинаковых decoder blocks;
- causal self-attention;
- residual connections;
- position-wise FFN;
- vocabulary logits и next-token loss.

Поэтому «LLaMA architecture» — не альтернатива Transformer. Это конкретная
современная конфигурация decoder-only Transformer.

## Изменение 1: normalization переехала перед подслоем

### Post-norm в исходном Transformer

Упрощённо:

$$
y=\operatorname{Norm}(x+F(x)).
$$

Каждый residual path проходит через normalization. При большой глубине это
усложняет прямое распространение gradient.

### Pre-norm в LLaMA

$$
y=x+F(\operatorname{Norm}(x)).
$$

Теперь identity path от $x$ к $y$ остаётся прямым. Подслой получает
нормализованный вход, а residual stream не нормализуется после каждого
сложения.

```text
post-norm: x ─→ F ─→ + ─→ Norm ─→ y
             └──────↑

pre-norm:  x ─────────→ + ─→ y
             ↓ Norm → F ↑
```

Это не означает, что значения residual stream никогда не нормализуются:
перед attention и FFN стоят отдельные norms, а после последнего block —
финальная norm.

## Изменение 2: RMSNorm вместо LayerNorm

LayerNorm центрирует и масштабирует координаты:

$$
\operatorname{LayerNorm}(x)
=g\odot\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+b.
$$

RMSNorm не вычитает mean:

$$
\operatorname{RMSNorm}(x)
=g\odot
\frac{x}{\sqrt{\frac1d\sum_{i=1}^{d}x_i^2+\varepsilon}}.
$$

### Численный пример

Для $x=(1,2,2)$:

$$
\operatorname{RMS}(x)
=\sqrt{\frac{1^2+2^2+2^2}{3}}
=\sqrt3.
$$

До learnable scale:

$$
\hat x=\left(\frac1{\sqrt3},\frac2{\sqrt3},\frac2{\sqrt3}\right).
$$

Направление вектора сохраняется, а его root-mean-square magnitude становится
равной единице. RMSNorm управляет масштабом activation, но не делает среднее
нулём.

### Минимальная реализация

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms_inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms_inv).type_as(x) * self.weight
```

Именно такой короткий implementation находится в официальном Llama 3 code.
Вычисление нормы в `float()` — практическая деталь численной устойчивости, а
обратное преобразование `type_as(x)` возвращает исходную precision.

> [!warning] Не смешивать два решения
> `pre-norm` отвечает на вопрос **где** стоит normalization. `RMSNorm` — **какая**
> normalization используется. Можно построить pre-LayerNorm или post-RMSNorm;
> это независимые архитектурные оси.

## Изменение 3: RoPE вместо прибавления position embedding

Attention без позиции не различает перестановки токенов. В ранних GPT к token
embedding прибавлялся learned absolute position vector:

$$
x_m=e_{\text{token}}+p_m.
$$

LLaMA не добавляет $p_m$ в residual stream. Rotary Position Embedding вращает
пары координат query и key на угол, зависящий от позиции.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-32-34/roformer-rope-figure1.png]]

*Поворот query и key в RoPE: их скалярное произведение после вращения зависит
от относительного смещения позиций. Источник: Jianlin Su et al.,
[RoFormer](https://arxiv.org/pdf/2104.09864#page=5), Figure 1, p. 5.*

Для одной пары координат:

$$
R(m\theta)=
\begin{bmatrix}
\cos m\theta &-\sin m\theta\\
\sin m\theta & \cos m\theta
\end{bmatrix},
$$

$$
q_m^{\text{rope}}=R(m\theta)q_m,\qquad
k_n^{\text{rope}}=R(n\theta)k_n.
$$

Главное свойство появляется в dot product:

$$
\left(R(m\theta)q\right)^\top R(n\theta)k
=q^\top R((n-m)\theta)k.
$$

Attention score зависит от относительного смещения $n-m$, хотя каждый вектор
вращался по своей абсолютной позиции.

### Несколько частот

В реальном head пары координат вращаются с разными частотами:

$$
\theta_i=\text{base}^{-2i/d_{\text{head}}}.
$$

Быстрые вращения хорошо различают близкие позиции, медленные меняются на
дальних масштабах.

Геометрически положение кодируется фазой вращения. Поэтому RoPE полезнее
рассматривать как преобразование координат Q/K, а не как ещё один embedding,
прибавленный к токену.

### Чего RoPE не обещает

RoPE не делает контекст автоматически бесконечным. Если модель обучалась на
коротких sequences, новые углы и новые распределения расстояний могут ухудшить
качество. RoPE scaling, изменение base, interpolation и long-context continued
training — отдельные методы, которые нужно проверять экспериментально.

## Изменение 4: SwiGLU вместо обычного FFN

В исходном Transformer:

$$
\operatorname{FFN}(x)
=\operatorname{ReLU}(xW_1+b_1)W_2+b_2.
$$

В LLaMA есть две входные projections и gate:

$$
\operatorname{SwiGLU}(x)
=\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_d,
$$

где

$$
\operatorname{SiLU}(z)=z\sigma(z).
$$

```text
                    ┌→ Wg → SiLU ─┐
x [d_model] ────────┤              × → Wd → output [d_model]
                    └→ Wu ────────┘
```

Ветвь $W_u$ создаёт candidate features, ветвь $W_g$ решает, какие координаты и
в какой степени пропустить. Gate зависит от самого token state, поэтому FFN
становится условным преобразованием.

Официальный код снова совпадает с формулой:

```python
return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

Названия matrices различаются между papers и codebases. Надёжнее смотреть на
граф: две up projections, activation на одной ветви, elementwise product,
down projection.

### Почему hidden dimension выглядит необычно

Обычный FFN с шириной $4d$ имеет примерно $8d^2$ weights в двух matrices.
SwiGLU использует три matrices. Чтобы parameter count оставался сопоставимым,
базовую hidden width уменьшают примерно до $\frac83d$, а затем округляют до
удобного для hardware multiple.

Это объясняет строки официального кода:

```python
hidden_dim = int(2 * hidden_dim / 3)  # исходно hidden_dim = 4 * dim
hidden_dim = multiple_of * ceil(hidden_dim / multiple_of)
```

## Изменение 5: меньше необязательных параметров

В LLaMA linear projections обычно создаются без bias. Это не центральная
научная идея модели, но важная implementation detail: architecture config и
checkpoint shapes должны совпадать.

Нельзя восстановить LLaMA, просто заменив название класса GPT. Нужно согласовать:

- число layers, heads и KV heads;
- $d_{\text{model}}$ и FFN width;
- RMSNorm epsilon;
- RoPE base и maximum context;
- vocabulary и tokenizer;
- наличие biases;
- tied или untied output embeddings.

## Один токен проходит через block

Рассмотрим позицию слова «корм» в префиксе:

```text
Кошка не стала есть корм
```

1. `RMSNorm(x)` стабилизирует масштаб текущего residual state.
2. Q слова «корм» сравнивается с K доступных прошлых позиций.
3. RoPE сообщает attention относительные расстояния.
4. Weighted sum V приносит контекст: «Кошка», «не», «есть».
5. Attention output добавляется к прежнему state.
6. Вторая RMSNorm готовит state для FFN.
7. SwiGLU нелинейно активирует полезные признаки этой позиции.
8. FFN output снова добавляется в residual stream.

После десятков blocks состояние позиции содержит информацию, нужную для
предсказания следующего токена — например запятой или продолжения «потому».

## Проверка shapes по официальному коду

Пусть:

$$
B=2,\quad T=128,\quad d=4096,\quad h_q=32,\quad d_h=128.
$$

Тогда при обычном MHA:

$$
Q,K,V\in\mathbb R^{2\times32\times128\times128}.
$$

После attention heads соединяются:

$$
\operatorname{concat}(\text{heads})
\in\mathbb R^{2\times128\times4096}.
$$

Output projection возвращает тот же $d=4096$, иначе residual addition была бы
невозможна.

В более поздних Llama число KV heads может быть меньше query heads. Это GQA —
следующая глава. Важно не приписывать GQA исходной LLaMA 1 как универсальную
характеристику всего семейства.

## LLaMA 1, Llama 2 и Llama 3 — не одна конфигурация

| Версия | Что важно для архитектурного чтения |
|---|---|
| LLaMA 1 | MHA, RMSNorm, RoPE, SwiGLU; baseline этой главы |
| Llama 2 | recipe сохранён; GQA использована в 70B; контекст и post-training изменены |
| Llama 3 | GQA в основных размерах, vocabulary 128K, иной RoPE base и гораздо больше данных |

Большая часть улучшения между generations находится не только в block:

- tokenizer и vocabulary;
- data mixture, quality filters и число training tokens;
- optimization recipe;
- context extension;
- SFT, preference optimization и safety;
- inference kernels.

Поэтому benchmark gain нельзя честно объяснить одной стрелкой «MHA → GQA».

## Что LLaMA не изобрела

Оригинальный paper прямо называет источники:

- pre-normalization — использовалась в более ранних GPT-style моделях;
- RMSNorm — Zhang & Sennrich;
- RoPE — RoFormer;
- SwiGLU — работа Shazeer о GLU variants и PaLM recipe.

Вклад LLaMA — удачная сборка, масштабирование на публично описанных данных и
доступность весов исследовательскому сообществу. Архитектурная история здесь
кумулятивна.

## Ошибки при чтении схем

> [!danger] «RoPE применяется к embeddings»
> В стандартной LLaMA RoPE применяется к Q и K каждого attention layer, не
> прибавляется один раз к token embeddings.

> [!danger] «RMSNorm — это LayerNorm без bias»
> Главное отличие — отсутствие mean centering. Bias — отдельная деталь.

> [!danger] «SwiGLU — просто другая activation»
> Меняется топология FFN: появляется дополнительная projection и
> multiplicative gate.

> [!danger] «Llama 3 block объясняет всю LLaMA»
> Поздний код содержит GQA и другие configuration changes. Всегда фиксируйте
> generation и model size.

## Практика

### 1. Сопоставить формулу и код

Откройте официальный `model.py` и для каждой строки найдите:

- RMS normalization;
- residual addition;
- RoPE application;
- causal mask;
- SwiGLU multiplication;
- final vocabulary projection.

Цель — уметь восстановить block diagram из кода без model card.

### 2. Реализовать RMSNorm и проверить invariant

Для random tensor вычислите RMS последней dimension до и после normalization.
До learnable scale результат должен быть близок к единице. Затем умножьте input
на 10 и проверьте, насколько меняется normalized output.

### 3. Увидеть относительную позицию RoPE

Возьмите одну двумерную пару $q,k$, поверните обе на позиции $(m,n)$ и сравните
dot product с поворотом только одного вектора на $n-m$. Числа должны совпасть с
погрешностью floating point.

### 4. Посчитать параметры FFN

Сравните:

$$
P_{\text{GELU}}\approx2d\,d_{\text{ff}},
\qquad
P_{\text{SwiGLU}}\approx3d\,d_{\text{ff}}.
$$

Подберите $d_{\text{ff}}$ SwiGLU так, чтобы обе величины были близки.

## После главы нужно уметь

- написать две формулы pre-norm block;
- объяснить RMSNorm без фразы «упрощённый LayerNorm»;
- вывести зависимость RoPE score от относительного смещения;
- по схеме и формулам из первоисточника объяснить роль трёх проекций SwiGLU;
- отличить базовый LLaMA recipe от изменений Llama 2/3;
- отделить architecture block от tokenizer, data и post-training.

## Материалы, на которых построена глава

### Основные papers

- [Touvron et al. — LLaMA](https://arxiv.org/abs/2302.13971).
- [Zhang & Sennrich — Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
- [Su et al. — RoFormer / RoPE](https://arxiv.org/abs/2104.09864).
- [Shazeer — GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

### Код и хорошие разборы

- [Meta — Llama 3 `model.py`](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
  — компактный reference implementation; репозиторий архивирован, но код
  остаётся полезным учебным снимком.
- [Andrej Karpathy — build-nanoGPT](https://github.com/karpathy/build-nanogpt)
  — хороший предшествующий маршрут от GPT-2 block к современным деталям.
- [Michael Brenndoerfer — LLaMA Components](https://mbrenndoerfer.com/writing/llama-components-rmsnorm-swiglu-rope)
  — интерактивный разбор RMSNorm, SwiGLU и RoPE.

### Внутри базы

- [[02 Areas/ML & DL/Papers/LLaMA]]
- [[02 Areas/ML & DL/01 Справочник/Позиционные представления/RoPE]]
- [[02 Areas/ML & DL/01 Справочник/FFN и MoE/SwiGLU]]

**Дальше:** [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|почему при генерации именно K/V становятся узким местом и как GQA уменьшает cache.]]
