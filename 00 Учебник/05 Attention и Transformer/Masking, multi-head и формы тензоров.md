---
title: Masking, multi-head и формы тензоров
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
  - https://jalammar.github.io/illustrated-transformer/
  - https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
---

# Маски, несколько голов и формы тензоров

Одна голова attention принимает матрицы $Q$, $K$ и $V$ и возвращает новую
матрицу представлений. В настоящей модели этого описания недостаточно. В одном
пакете могут находиться предложения разной длины, авторегрессионная модель не
имеет права читать будущие токены, а несколько голов должны вычисляться
одновременно без циклов Python. Поэтому формула превращается в надёжную
реализацию только после ответа на три вопроса: какие пары позиций разрешены,
где в тензоре находится номер головы и по какой оси нормализуется softmax.

## Маска меняет множество доступных ключей

Пусть $S=QK^\top/\sqrt{D_h}$ — матрица оценок. Маска добавляется к $S$ **до**
softmax:

$$
A=\operatorname{softmax}(S+M),\qquad Z=AV.
$$

В разрешённых ячейках $M_{ij}=0$, в запрещённых — $-\infty$. После
экспоненцирования запрещённая ячейка даёт ноль, поэтому она не участвует ни во
взвешенной сумме, ни в нормировке оставшихся весов. Обнулить уже готовую матрицу
$A$ недостаточно: сумма разрешённых весов перестанет равняться единице, если не
выполнить нормировку заново.

### Причинная маска: не подсматривать ответ

При обучении языковой модели вся последовательность известна и все позиции
обрабатываются параллельно. Но представление позиции $i$, из которого модель
предсказывает следующий токен, должно зависеть только от позиций $j\le i$.
Иначе сеть увидит правильное продолжение во входе и обучающая задача потеряет
смысл. Эту зависимость задаёт нижнетреугольная маска:

$$
M^{\text{causal}}_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/causal-mask-detailed.png]]

*Причинная маска в decoder self-attention: для каждой строки остаются только
текущая и предыдущие позиции. Автор изображения — Daniel Voigt Godoy,
[оригинал и лицензия CC BY 4.0](https://commons.wikimedia.org/wiki/File:Decoder_self-attention_with_causal_masking,_detailed_diagram.png).*

На странице Lena Voita тот же переход показан как
[готовая анимация masked self-attention](https://lena-voita.github.io/resources/lectures/seq2seq/transformer/masked_self_attn.mp4): при движении по целевой последовательности доступная нижнетреугольная область расширяется на одну позицию.

Маска не делает обучение последовательным. Все строки $S$ по-прежнему можно
вычислить одним матричным умножением; запрещённые связи удаляются перед
softmax. Последовательной остаётся генерация, потому что следующий входной
токен ещё не известен.

### Padding mask: не читать заполнители

Предложения разной длины обычно дополняют специальным токеном до общей длины
$T$. Эти позиции нужны форме тензора, но не содержат текста. Если во втором
примере настоящими являются только первые пять токенов, маска ключей имеет вид
`[1, 1, 1, 1, 1, 0, 0]`: любой настоящий запрос может читать первые пять
ключей, но не два заполнителя.

Причинная маска зависит от пары `(query position, key position)` и имеет форму
`[T, T]`. Маска заполнителей зависит от примера и ключевой позиции, поэтому её
естественная форма — `[B, T]`. Для decoder-only модели нужны обе:

$$
M_{b,i,j}=M^{\text{causal}}_{i,j}+M^{\text{padding}}_{b,j}.
$$

Важно маскировать именно **ключевые** позиции. Запрос из padding-позиции тоже
можно занулить позже, однако это другая операция: она не защищает настоящие
токены от чтения заполнителей.

## Зачем одной позиции несколько голов

Один softmax создаёт одно распределение по ключам. Между тем одному слову могут
быть одновременно нужны разные отношения: ближайший сосед, согласуемое
подлежащее и слово, уточняющее значение. Multi-head attention предоставляет
несколько независимо обучаемых наборов проекций и несколько распределений
attention:

$$
\operatorname{head}_r=
\operatorname{Attention}(QW_r^Q,KW_r^K,VW_r^V),
$$

$$
\operatorname{MHA}(Q,K,V)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,
\operatorname{head}_H)W^O.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/qkv-for-heads.png]]

*Каждая голова получает собственные проекции запросов, ключей и значений.
Источник: Lena Voita, раздел
[Multi-Head Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#multi_head_attention). На исходной странице рядом доступна анимация всего процесса.*

Несколько голов не означают несколько полных копий вектора ширины
$D_{model}$. В классическом Transformer общая ширина делится между $H$
головами: $D_h=D_{model}/H$. Поэтому конкатенация снова имеет ширину
$H D_h=D_{model}$, а $W^O$ смешивает признаки разных голов и возвращает их в
остаточный поток.

## Split и merge: одна операция, две записи

В учебной формуле у каждой головы свои $W_r^Q,W_r^K,W_r^V$. В библиотечной
реализации эти матрицы обычно склеены. Одно умножение даёт
$Q,K,V\in\mathbb{R}^{B\times T\times D_{model}}$, после чего последняя ось
разбивается на номер головы и признаки внутри головы.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_qkv.png]]

*Jay Alammar показывает те же отдельные проекции, из которых получаются
запросы, ключи и значения каждой головы. Источник:
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

Для входа $X$ формы `[B, T, Dmodel]` путь одной из матриц выглядит так:

```python
q = q_proj(x)                    # [B, T, Dmodel]
q = q.view(B, T, H, Dh)          # [B, T, H, Dh]
q = q.transpose(1, 2)            # [B, H, T, Dh]
```

Перестановка осей нужна не для математики модели, а для пакетного матричного
умножения: последние две оси должны быть `[T, Dh]`. После attention порядок
действий обращается:

```python
z = weights @ v                 # [B, H, T, Dh]
z = z.transpose(1, 2)           # [B, T, H, Dh]
z = z.contiguous().view(B, T, Dmodel)
z = out_proj(z)                  # [B, T, Dmodel]
```

Вызов `contiguous()` перед `view()` существенен в PyTorch: `transpose` меняет
шаги памяти, не переставляя сами данные. `reshape()` может создать нужную копию
самостоятельно, но понимание перестановки осей всё равно необходимо.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_z.png]]

*Результаты голов остаются отдельными до конкатенации. Источник: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_weight_matrix_o.png]]

*После конкатенации выходная матрица $W^O$ смешивает результаты голов и
возвращает ширину $D_{model}$. Источник: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

## Полная трассировка форм

Пусть пакет содержит $B$ последовательностей длины $T$, ширина модели равна
$D_{model}$, число голов — $H$, а ширина головы — $D_h=D_{model}/H$.

| Шаг | Тензор | Форма |
|---|---|---|
| вход | $X$ | `[B, T, Dmodel]` |
| три линейные проекции | $Q,K,V$ | по `[B, T, Dmodel]` |
| split + transpose | $Q,K,V$ | по `[B, H, T, Dh]` |
| транспонирование ключей | $K^\top$ | `[B, H, Dh, T]` |
| оценки всех пар | $S=QK^\top/\sqrt{D_h}$ | `[B, H, T, T]` |
| маска и softmax по ключам | $A$ | `[B, H, T, T]` |
| взвешенные значения | $Z=AV$ | `[B, H, T, Dh]` |
| transpose + concat | $Z_{cat}$ | `[B, T, Dmodel]` |
| выходная проекция | $Z_{cat}W^O$ | `[B, T, Dmodel]` |

Softmax выполняется по **последней** оси: для каждого `(b, h, i)` сумма по всем
ключам $j$ равна единице. Нормировка по запросам отвечала бы на другой вопрос и
изменила бы механизм.

## Broadcasting масок

После split оценки имеют форму `[B, H, Tq, Tk]`. Маске не обязательно физически
хранить все эти элементы: оси размера 1 распространяются по правилам
broadcasting.

| Назначение | Удобная форма | По каким осям распространяется |
|---|---|---|
| одна causal mask для всех | `[1, 1, T, T]` | batch и heads |
| padding ключей для каждого примера | `[B, 1, 1, T]` | heads и queries |
| своя произвольная маска примера | `[B, 1, T, T]` | heads |

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)  # [B,H,T,T]

causal = torch.ones(T, T, dtype=torch.bool).tril()
causal = causal[None, None, :, :]                   # [1,1,T,T]

key_ok = attention_mask[:, None, None, :].bool()   # [B,1,1,T]
allowed = causal & key_ok
scores = scores.masked_fill(~allowed, float("-inf"))
weights = scores.softmax(dim=-1)
```

В некоторых API булево значение `True` означает «разрешить», в других —
«скрыть». Это нельзя угадывать по имени аргумента: нужно проверить документацию
конкретной функции. У `torch.nn.functional.scaled_dot_product_attention`
булева маска использует `True` для разрешённого элемента, тогда как у ряда
старых интерфейсов PyTorch смысл противоположен.

## Ошибки, которые выглядят правдоподобно

- **Softmax по неверной оси.** Проверка: `weights.sum(-1)` должна состоять из
  единиц для строк, где есть хотя бы один разрешённый ключ.
- **Маска после softmax.** Запрещённые веса станут нулевыми, но оставшиеся не
  будут перенормированы.
- **Перепутаны $T_q$ и $T_k$.** В cross-attention они вообще могут различаться;
  оценки всегда имеют форму `[B, H, Tq, Tk]`.
- **Тихое неверное broadcasting.** Маска `[B, T]` не совпадает справа с
  `[B, H, T, T]`; перед применением ей нужны две одноэлементные оси.
- **Строка целиком замаскирована.** Softmax от одних $-\infty$ даёт `NaN`.
  Такая строка требует отдельной политики, особенно при left padding.
- **Забыто объединение голов.** Просто усреднить головы — не то же самое, что
  конкатенация и обучаемая $W^O$.
- **Неверный масштаб.** Делить нужно на $\sqrt{D_h}$, а не на
  $\sqrt{D_{model}}$.

## Минимальная проверка реализации

```python
assert Dmodel == H * Dh
assert q.shape == (B, H, T, Dh)
assert scores.shape == (B, H, T, T)
assert weights.shape == (B, H, T, T)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones_like(weights.sum(dim=-1)),
    atol=1e-5,
)

# Ни одна causal-голова не должна отдавать вес будущему.
future = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
assert torch.count_nonzero(weights[..., future]) == 0
```

Эти проверки связывают формулу с фактическими осями. После них полный
Transformer уже можно собирать из attention, feed-forward слоя, residual
connections и нормализации, не оставляя маски и головы неявной «магией
библиотеки».

## Источники и визуальные продолжения

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  разделы 3.2.2–3.2.3 — исходные определения multi-head и masked attention.
- Lena Voita, [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — наиболее ясная учебная анимация маски и независимых голов.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — пошаговое визуальное объяснение split, concat и $W^O$.
- Dive into Deep Learning, [Multi-Head Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html) — реализация с явными преобразованиями форм.
- Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — реализация маски последующих позиций и multi-head attention рядом с формулами статьи.
- PyTorch, [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) — точная семантика `attn_mask`, `is_causal` и dropout.

**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer]]
