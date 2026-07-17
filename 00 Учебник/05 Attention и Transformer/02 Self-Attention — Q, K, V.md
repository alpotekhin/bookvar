---
title: Self-Attention изнутри — Q, K, V
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Self-Attention изнутри: Q, K, V

> [!abstract] Идея главы
> В начале работы Transformer каждому слову соответствует один и тот же исходный
> вектор — независимо от контекста. Attention позволяет окружающим словам
> изменить этот вектор. Поэтому одно и то же слово постепенно получает разные
> представления в разных предложениях.

## Зачем слову смотреть на другие слова

Это объяснение следует примеру
[3Blue1Brown](https://www.3blue1brown.com/lessons/attention), потому что он
показывает назначение механизма до матриц.

Сравним три фразы:

- `American shrew mole` — крот;
- `one mole of carbon dioxide` — моль вещества;
- `a biopsy of the mole` — родинка.

В таблице эмбеддингов слову `mole` сначала соответствует один и тот же вектор.
Таблица знает, какое это слово, но ещё не знает, в каком предложении оно
оказалось. Задача attention — уточнить исходный вектор с помощью контекста.

Для русскоязычного примера возьмём:

> **Кот, который весь день бегал, устал**

Чтобы предсказать «устал», модели полезно связать сказуемое с «кот», несмотря на
несколько слов между ними. Пока каждое слово обрабатывается отдельно, такой
связи в вычислении просто нет.

Карпати в
[Let’s build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
показывает, как построить эту связь шаг за шагом.

## 1. Самый простой канал: среднее прошлого

Пусть $x_i$ — вектор токена $i$. Создадим контекст позиции $i$ как среднее всех
доступных токенов:

$$
\bar{x}_i=\frac{1}{i}\sum_{j\le i}x_j.
$$

Для четырёх токенов это матричное умножение:

$$
\begin{bmatrix}
1&0&0&0\\
\frac12&\frac12&0&0\\
\frac13&\frac13&\frac13&0\\
\frac14&\frac14&\frac14&\frac14
\end{bmatrix}
\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}.
$$

Нижнетреугольная матрица задаёт правило доступа: строка $i$ использует только
текущую и предыдущие позиции $j\le i$.

Это уже контекст, но веса фиксированы. Для «устал» токены «кот» и «бегал» должны
быть важнее запятой, однако равномерное среднее этого не знает.

## 2. Сделаем веса зависимыми от содержимого

Позиции нужны две разные роли:

- описать, **что ей сейчас нужно**;
- объявить, **по какому признаку её можно выбрать**.

Из каждого входного вектора модель получает две обучаемые проекции:

$$
q_i=x_iW_Q,\qquad k_i=x_iW_K.
$$

Совместимость запроса позиции $i$ и ключа позиции $j$:

$$
s_{ij}=q_i^\top k_j.
$$

Большое значение $s_{ij}$ означает не «слова вообще похожи», а «в этом слое и
этой голове позиция $j$ подходит тому, что ищет позиция $i$».

> [!note] Аналогия и строгий смысл
> `Query = вопрос`, `Key = адрес` — полезная мнемоника. Но Q и K не содержат
> заранее заданных человеческих ролей. Это обучаемые линейные проекции, смысл
> которых определяется loss и всей остальной сетью.

## 3. Почему нужен отдельный Value

Key участвует в выборе, но передавать модель будет другой вектор:

$$
v_j=x_jW_V.
$$

Разделение полезно. Признаки, по которым позицию находят, не обязаны совпадать с
информацией, которую следует добавить к получателю. Аналогия с базой данных:
поиск идёт по индексу, а возвращается содержимое записи.

## Всё вычисление на двух рисунках

![[02 Areas/ML & DL/raw/papers/attention-is-all-you-need/images/transformer_self_attention_vectors.png]]

*Jay Alammar, «The Illustrated Transformer»: из каждого token vector получаются
Q, K и V. [Оригинальное объяснение](https://jalammar.github.io/illustrated-transformer/).*

![[02 Areas/ML & DL/raw/papers/attention-is-all-you-need/images/self-attention-matrix-calculation-2.png]]

*Jay Alammar: матричная запись всего вычисления. Эту картинку стоит читать справа
налево: Values определяют передаваемое содержание, а `softmax(QKᵀ/√dₖ)` —
коэффициенты его смешивания.*

Полная формула:

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V.
$$

Но теперь каждый символ имеет конкретную историю.

## 4. Численный пример, который можно проверить вручную

Возьмём три позиции и одну голову размерности $d_k=2$. Чтобы не прятать механику
за обучаемыми матрицами, в игрушечном примере положим $Q=K=X$:

$$
Q=K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
V=
\begin{bmatrix}
1&0\\
0&2\\
3&1
\end{bmatrix}.
$$

Рассмотрим query третьей позиции $q_3=[1,1]$.

### Шаг A. Dot products

$$
q_3K^\top=[1,1,2].
$$

### Шаг B. Scaling

$$
\frac{q_3K^\top}{\sqrt2}
\approx[0.707,0.707,1.414].
$$

### Шаг C. Softmax

$$
\alpha_3\approx[0.248,0.248,0.503].
$$

Погрешность округления объясняет, почему сумма показанных чисел равна 0.999.

### Шаг D. Weighted sum Values

$$
z_3=
0.248[1,0]+0.248[0,2]+0.503[3,1]
\approx[1.758,0.999].
$$

Третья позиция получила новый contextual vector. В нём смешано содержимое всех
трёх Values, но третье имело примерно половину веса.

## 5. Зачем делить на $\sqrt{d_k}$

Если компоненты $q$ и $k$ независимы, имеют среднее 0 и дисперсию 1, то

$$
\operatorname{Var}(q^\top k)=d_k.
$$

С ростом размерности dot products по модулю увеличиваются. Softmax на больших
logits становится почти one-hot: небольшое изменение score почти не меняет
распределение для большинства позиций, и оптимизация усложняется. Деление на
$\sqrt{d_k}$ возвращает типичный масштаб к порядку единицы.

Важно: делим на $\sqrt{d_k}$ — размерность key одной головы, а не автоматически
на $\sqrt{d_{\text{model}}}$.

## 6. Causal mask

При next-token prediction позиция $i$ не должна читать будущие $j>i$. Illegal
scores заменяются на $-\infty$ **до softmax**:

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

Для второй позиции нашего примера:

$$
\frac{q_2K^\top}{\sqrt2}+M
=[0,0.707,-\infty].
$$

После softmax:

$$
\alpha_2\approx[0.330,0.670,0].
$$

И output:

$$
z_2=0.330[1,0]+0.670[0,2]\approx[0.330,1.340].
$$

Будущее получило ровно нулевой вес. При training все строки матрицы всё равно
вычисляются параллельно; маска ограничивает доступ к данным, а не заставляет GPU
обрабатывать позиции по одной.

## 7. Tensor shapes без магии

Пусть:

- batch $B=2$;
- sequence length $T=128$;
- model width $d_{\text{model}}=768$;
- heads $h=12$;
- head width $d_k=d_v=64$.

| Tensor | Shape | Что означает |
|---|---|---|
| $X$ | `2 × 128 × 768` | residual stream |
| $Q,K,V$ до split | `2 × 128 × 768` | три projections |
| после split heads | `2 × 12 × 128 × 64` | независимые subspaces |
| $QK^\top$ | `2 × 12 × 128 × 128` | score каждой пары позиций |
| $A V$ | `2 × 12 × 128 × 64` | output каждой головы |
| concat | `2 × 128 × 768` | головы собраны обратно |

Квадратичный по $T$ объект — attention matrix. Это объясняет проблемы длинного
контекста. FlashAttention уменьшает memory traffic и не обязан материализовывать
всю матрицу в HBM, но математически вычисляет тот же exact attention.

## 8. Multi-Head Attention

Одна голова создаёт одно распределение weights для каждого query. Несколько голов
используют разные projections:

$$
\text{head}_r=
\operatorname{Attention}(XW_r^Q,XW_r^K,XW_r^V),
$$

$$
\operatorname{MHA}(X)=
\operatorname{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O.
$$

Почему не одна большая голова? Несколько heads позволяют одновременно строить
несколько различных communication patterns, не заставляя один softmax смешивать
их в одно распределение. Но утверждение «каждая голова обязательно отвечает за
понятную лингвистическую функцию» слишком сильное: головы могут быть
избыточными, распределёнными и трудными для интерпретации.

![[02 Areas/ML & DL/raw/papers/attention-is-all-you-need/images/transformer_multi-headed_self-attention-recap.png]]

*Jay Alammar: split в несколько heads, независимый attention, concat и output
projection. Изображение полезнее абстрактной формулы тем, что не теряет связь
между отдельными token vectors и матричной реализацией.*

### Посмотреть механизм в движении

- [3Blue1Brown — Attention in transformers](https://www.3blue1brown.com/lessons/attention):
  геометрическая интуиция query/key dot products и contextual update.
- [Karpathy — Let’s build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY):
  тот же механизм возникает постепенно из исполняемого PyTorch-кода.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/):
  можно выбрать token и пройти по слоям настоящего GPT-2.

## 9. Минимальный PyTorch

```python
import math
import torch

def attention(q, k, v, mask=None):
    # q: [..., Tq, dk], k: [..., Tk, dk], v: [..., Tk, dv]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    if mask is not None:
        # True = разрешено читать
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = scores.softmax(dim=-1)  # по keys
    output = weights @ v
    return output, weights
```

Production API:

```python
from torch.nn.functional import scaled_dot_product_attention

output = scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    dropout_p=0.0,  # явно 0 при evaluation
)
```

Официальная функция может выбрать оптимизированный backend, включая Flash
Attention. Её `dropout_p` применяется согласно переданному значению независимо
от `module.training`, поэтому evaluation-код должен передавать `0.0`.

## 10. Self-, cross- и masked attention — одна математика

| Вид | Q | K,V | Mask |
|---|---|---|---|
| full self-attention | та же sequence | та же sequence | padding, если нужен |
| causal self-attention | target sequence | target sequence | future positions |
| cross-attention | decoder states | encoder states | обычно padding |

Математический primitive один; различаются источники tensors и разрешённые
соединения.

## Частые ошибки

- Softmax идёт по keys отдельно для каждого query.
- Attention weights суммируются в 1 по строке, но не являются калиброванной
  вероятностью «истинной важности».
- Q/K/V — обучаемые проекции, а не исходные эмбеддинги с новыми названиями.
- Causal mask применяется к logits до softmax.
- Training может вычислять все target positions параллельно; generation новых
  positions остаётся последовательной.
- KV-cache не меняет attention: он не пересчитывает старые K/V при generation.

## Лаборатория по Карпати

Лучший практический маршрут:

1. [bigram baseline](https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py);
2. равномерное усреднение предыдущих tokens через lower-triangular matrix;
3. learned Q/K affinities;
4. Values и одна head;
5. multi-head + projection;
6. [полный минимальный GPT](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py).

После этого полезно открыть [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
и найти те же стадии внутри настоящего GPT-2.

## Источники

- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Karpathy — Let’s build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy — code from the lecture](https://github.com/karpathy/ng-video-lecture)
- [PyTorch — scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [3Blue1Brown — Attention in transformers](https://www.3blue1brown.com/lessons/attention)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [BertViz](https://github.com/jessevig/bertviz)

**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer]]
