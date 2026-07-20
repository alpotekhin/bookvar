---
title: Self-Attention изнутри — Q, K, V
type: textbook-chapter
status: canonical
last_updated: 2026-07-18
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Self-Attention изнутри: Q, K, V

## Зачем слову смотреть на другие слова

Входная таблица сопоставляет токену один и тот же исходный вектор во всех
предложениях. Поэтому до первого слоя слово `mole` представлено одинаково и в
разговоре о химическом количестве вещества, и в описании родинки. Контекстное
значение должно возникнуть внутри сети: позициям нужен способ получать
информацию от других позиций и при этом решать, какие из них важны именно сейчас.

Наглядный вариант этого примера используется в
[визуальной лекции 3Blue1Brown](https://www.3blue1brown.com/lessons/attention).
Здесь мы сначала восстановим назначение операции на простом усреднении, а затем
последовательно заменим фиксированные веса обучаемыми запросами, ключами и
значениями.

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
> которых определяется функцией потерь и всей остальной сетью.

## 3. Почему нужен отдельный Value

Key участвует в выборе, но передавать модель будет другой вектор:

$$
v_j=x_jW_V.
$$

Разделение полезно. Признаки, по которым позицию находят, не обязаны совпадать с
информацией, которую следует добавить к получателю. Аналогия с базой данных:
поиск идёт по индексу, а возвращается содержимое записи.

## Всё вычисление на двух рисунках

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png]]

*Jay Alammar, «The Illustrated Transformer»: из каждого вектора токена получаются
Q, K и V. [Оригинальное объяснение](https://jalammar.github.io/illustrated-transformer/).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png]]

*Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/):
матричная запись всего вычисления. Эту картинку стоит читать справа налево:
Values определяют передаваемое содержание, а `softmax(QKᵀ/√dₖ)` — коэффициенты
его смешивания.*

Полная формула:

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
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

Третья позиция получила новое контекстное представление. В нём смешано содержимое всех
трёх Values, но третье имело примерно половину веса.

## 5. Зачем делить на $\sqrt{d_k}$

Если компоненты $q$ и $k$ независимы, имеют среднее 0 и дисперсию 1, то

$$
\operatorname{Var}(q^\top k)=d_k.
$$

С ростом размерности dot products по модулю увеличиваются. Softmax на больших
оценок перед softmax становится почти one-hot: небольшое изменение оценки почти не меняет
распределение для большинства позиций, и оптимизация усложняется. Деление на
$\sqrt{d_k}$ возвращает типичный масштаб к порядку единицы.

Важно: делим на $\sqrt{d_k}$ — размерность key одной головы, а не автоматически
на $\sqrt{d_{\text{model}}}$.

## 6. Минимальная реализация одной головы

```python
import math
import torch

def attention(q, k, v):
    # q: [..., Tq, dk], k: [..., Tk, dk], v: [..., Tk, dv]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    weights = scores.softmax(dim=-1)  # по keys
    output = weights @ v
    return output, weights
```

Код намеренно не содержит масок и разбиения на головы. Запросы выбирают значения
через нормированные оценки совместимости с ключами. Следующая глава добавит к этой
операции ограничения доступа и четвёртое измерение — номер головы.

## Что легко перепутать

- Softmax идёт по keys отдельно для каждого query.
- Attention weights суммируются в 1 по строке, но не являются калиброванной
  вероятностью «истинной важности».
- Q/K/V — обучаемые проекции, а не исходные эмбеддинги с новыми названиями.
- Выход позиции — взвешенная сумма строк $V$, а не строк $Q$ или $K$.

## Лаборатория по Карпати

Лучший практический маршрут:

1. [bigram baseline](https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py);
2. равномерное усреднение предыдущих tokens через lower-triangular matrix;
3. learned Q/K affinities;
4. Values и одна head;
5. маска, несколько голов и выходная проекция;
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
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров]]
