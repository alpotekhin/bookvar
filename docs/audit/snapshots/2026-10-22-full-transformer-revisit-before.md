---
title: Собираем Transformer — от блока к BERT, GPT и LLaMA
type: textbook-chapter
status: canonical
last_updated: 2026-10-21
source_unit_id:
  - lecture-01-section-175-model-architecture
  - lecture-03-architecture-framing
  - lecture-03-recap
  - assignment-01-task-transformer-block
  - assignment-01-task-transformer-lm
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/2302.13971
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
---

# Собираем Transformer: от блока к BERT, GPT и LLaMA

Переводчик получает фразу `кот спит` и должен выдать `the cat sleeps`.
В учебном примере каждому слову сопоставим один токен; настоящая
токенизация может делить слова иначе.
Число входных и выходных токенов может различаться. Энкодер строит
представление каждой позиции исходной фразы. Декодер предсказывает перевод,
обращаясь и к этим представлениям, и к уже известному началу перевода.

В Transformer 2017 года обе части состоят из блоков attention и
полносвязных сетей с остаточными связями и нормализацией.
В отличие от рекуррентного переводчика, энкодер обрабатывает все известные
входные позиции параллельно. При обучении так же обрабатываются позиции
декодера; при генерации следующий токен ещё нужно получить от модели.

У BERT, GPT и LLaMA иные входы, связи между блоками и задачи обучения;
их отличия от переводчика сопоставлены в разделе 8.
Формы и вычисление Q/K/V предполагаются известными из
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|главы о self-attention]];
причинная маска и разделение голов — из
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|следующей главы]].

<a id="architecture-lineage"></a>

## 1. Главная декомпозиция

Пусть после первого слоя attention вектор токена `кот` уже зависит от
`спит`. Этот вектор содержит смесь признаков, извлечённых из всей фразы.
Полносвязная сеть, feed-forward network или FFN, преобразует его координаты:
расширяет представление, применяет нелинейность и возвращает исходную ширину.
Ту же сеть с теми же весами она применяет к состоянию `спит`.
У другого слоя Transformer будет своя FFN.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/Transformer_encoder.png]]

*Два подслоя энкодера. Self-attention соединяет позиции; FFN обрабатывает
каждый полученный вектор отдельно. Остаточные связи и нормализация
на этой упрощённой схеме не показаны.
Источник: Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

При входе `[B,T,d]` оба подслоя возвращают `[B,T,d]`.
Однако внутренние вычисления различаются: матрица весов внимания имеет две
оси позиций, а скрытое представление FFN — одну ось позиций и расширенную
ось признаков. FFN не обращается к соседним строкам напрямую. Она уже получает
контекстный вектор, и её результат влияет на соседей через внимание
следующего слоя.

Softmax делает attention нелинейной операцией. Поэтому утверждение
«FFN нужна, потому что attention линейно» неверно.
FFN добавляет обучаемое нелинейное отображение признаков в каждой позиции.
Если убрать активацию из двух линейных слоёв FFN, их можно свернуть
в один линейный слой; подробнее это видно в примере ниже.

## 2. Остаточный поток

Для строки $x$ attention или FFN вычисляет поправку $F(x)$ той же ширины.
Остаточная связь складывает её с входом. Например, для $x=(1,2,3,4)$
и $F(x)=(0,1,0,-1)$ сумма равна $(1,3,3,3)$.
Подслой может менять часть признаков, сохраняя остальные через прямое сложение.
Ему не нужно заново воспроизводить весь входной вектор.

В исходной статье после суммы применяется LayerNorm. Нормализация выполняется
по координатам **каждой строки отдельно**, без смешивания токенов или примеров:

$$
\begin{aligned}
\mu&=\frac1d\sum_{r=1}^d x_r,\\
\sigma^2&=\frac1d\sum_{r=1}^d(x_r-\mu)^2,\\
\operatorname{LN}(x)&=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
\end{aligned}
$$

При $\gamma=(1,1,1,1)$, $\beta=0$ и пренебрежимо малом $\epsilon$
вектор $(1,2,3,4)$ имеет среднее $2.5$, дисперсию $1.25$ и нормализуется
приблизительно в $(-1.3416,-0.4472,0.4472,1.3416)$.
Параметры $\gamma,\beta$ обучаются; нормализация не фиксирует навсегда
масштаб и сдвиг признаков.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_resideual_layer_norm_2.png]]

*Post-norm энкодер для двух позиций. Пунктирный путь переносит вход
в сложение с результатом подслоя; LayerNorm стоит после суммы.
Зелёные и розовые векторы внутри Add & Normalize обозначают слагаемые,
а не две разные нормализации.
Источник: Jay Alammar, [The Illustrated Transformer, The Residuals](https://jalammar.github.io/illustrated-transformer/#the-residuals),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Порядок операций различает post-norm и pre-norm:

$$
\begin{aligned}
\text{post-norm:}\quad&y=\operatorname{LN}(x+F(x)),\\
\text{pre-norm:}\quad&y=x+F(\operatorname{LN}(x)).
\end{aligned}
$$

Это разные функции даже с одинаковыми весами. При $F(x)=2x$ post-norm
приблизительно возвращает $\operatorname{LN}(x)$, поскольку нормализация
убирает общий положительный масштаб $3$.
Pre-norm возвращает $x+2\operatorname{LN}(x)$:
для прежнего вектора это примерно $(-1.6833,1.1056,3.8944,6.6833)$.
Исходный масштаб сохраняется на прямом остаточном пути.

При обратном проходе у $x+F(\operatorname{LN}(x))$ есть слагаемое с единичной
производной по $x$. В post-norm весь результат сложения проходит через
производную нормализации. Это объясняет различие путей градиента,
но не доказывает, что любой pre-norm стек обучится устойчиво:
слагаемые могут компенсироваться, а масштаб поправок зависит от инициализации
и глубины. В pre-norm моделях перед выходной проекцией обычно ставят
дополнительную финальную нормализацию.

В [Vaswani et al., §3.1](https://arxiv.org/html/1706.03762v7#S3.SS1)
использован post-norm. В [Annotated Transformer, Encoder and Decoder Stacks](https://nlp.seas.harvard.edu/annotated-transformer/#encoder-and-decoder-stacks)
поясняющий текст воспроизводит эту формулу, но класс `SublayerConnection`
вычисляет pre-norm; авторы отмечают это в docstring. У кода есть и другое
отличие: `std` со стандартной коррекцией и $\epsilon$ снаружи корня
не совпадают с приведённой выше формулой LayerNorm.
Поэтому ниже используется `nn.LayerNorm`, а схема статьи и код руководства
не объявляются тождественными.

При обучении исходного Transformer dropout применяется к выходу подслоя
**до остаточного сложения**. Сумма эмбеддинга и позиции также проходит через
dropout. В режиме `eval()` он отключён.
Подробности нормализации и распространения градиента разобраны в
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections|главе об инициализации и нормализации]].

## 3. Feed-Forward Network

В оригинальном Transformer FFN состоит из двух аффинных преобразований
с ReLU между ними:

$$
\begin{aligned}
u&=xW_1+b_1,& W_1&\in\mathbb R^{d\times d_{ff}},\\
a&=\max(0,u),\\
\operatorname{FFN}(x)&=aW_2+b_2,
& W_2&\in\mathbb R^{d_{ff}\times d}.
\end{aligned}
$$

Для Transformer Base $d=512$, $d_{ff}=2048$.
Расширение создаёт 2048 промежуточных признаков; ReLU оставляет положительные
координаты, после чего $W_2$ комбинирует их в 512 выходных координат.
У различных токенов активируются разные промежуточные координаты,
хотя матрицы общие.

Возьмём $d=2$, $d_{ff}=3$ и строки-векторы:

$$
\begin{aligned}
W_1&=\begin{pmatrix}1&-1&0\\0&1&1\end{pmatrix},
\quad b_1=(0,1,0),\\
W_2&=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix},
\quad b_2=0.
\end{aligned}
$$

Для $x=(1,-2)$ сначала получаем $u=(1,-2,-2)$.
После ReLU остаётся $(1,0,0)$, а выход равен $(1,2)$.
Для $x'=(-1,2)$ та же FFN даёт $u'=(-1,4,2)$,
затем $(0,4,2)$ и результат $(22,28)$.
Две позиции выбрали разные сочетания промежуточных признаков.
Если поменять только вторую строку входа, первая строка результата FFN
не изменится.

Без ReLU выражение раскрылось бы в
$x(W_1W_2)+(b_1W_2+b_2)$, то есть обычное аффинное отображение.
С ReLU эффективная матрица зависит от того, какие координаты $u$ положительны.
В первом примере производная не проходит через вторую и третью
промежуточные координаты; матрицы и входы обучаются через активную первую.

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1., -2.], [-1., 2.]], requires_grad=True)
w1 = torch.tensor([[1., -1., 0.], [0., 1., 1.]])
b1 = torch.tensor([0., 1., 0.])
w2 = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
u = x @ w1 + b1
y = u.relu() @ w2
torch.testing.assert_close(y, torch.tensor([[1., 2.], [22., 28.]]))
y[0].sum().backward()
torch.testing.assert_close(x.grad, torch.tensor([[3., 0.], [0., 0.]]))
print(u.detach(), y.detach(), x.grad)
```

Без учёта смещений FFN содержит $2dd_{ff}$ параметров.
При $d_{ff}=4d$ это $8d^2$ против $4d^2$ у обычного MHA с суммарной
шириной голов $d$. В одном энкодер-блоке FFN занимает две трети весов
этих матричных проекций. В декодер-блоке есть ещё cross-attention,
поэтому это уже половина, а не две трети.
Таблицы эмбеддингов в этот подсчёт не входят.

Варианты gated FFN меняют само отображение. В SwiGLU одна проекция
создаёт множитель для другой:

$$
\begin{aligned}
g&=\operatorname{SiLU}(xW_g),\\
u&=xW_u,\\
\operatorname{SwiGLU}(x)&=(g\odot u)W_d.
\end{aligned}
$$

Здесь $\operatorname{SiLU}(z)=z\,\sigma(z)$.
Вместо двух матриц появляются три. При сопоставлении числа параметров
с ReLU FFN ширины $4d$ получается $3dm\approx8d^2$,
то есть $m\approx8d/3$, до округления под реализацию.
Множитель SiLU не ограничен диапазоном $[0,1]$ и может быть отрицательным:
его нельзя интерпретировать как вероятность «открытого вентиля».
Здесь этот вариант нужен для сравнения с LLaMA; исходная модель 2017 года
использует ReLU. Постановка position-wise FFN и общие веса по позициям
показаны в [Vaswani et al., §3.3](https://arxiv.org/html/1706.03762v7#S3.SS3)
и [D2L, Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html#positionwise-feed-forward-networks).

## 4. Информация о позиции

Энкодер и декодер начинают с целочисленных идентификаторов.
Для общего словаря $V$ таблица $E\in\mathbb R^{|V|\times d}$ хранит
обучаемый вектор каждого токена. Выбор строк для входа `[B,T]`
даёт тензор `[B,T,d]`. В Transformer 2017 года к масштабированному
эмбеддингу прибавляется фиксированный синусоидальный вектор:

$$
h_i^{(0)}=\operatorname{Dropout}\big(\sqrt d\,E[t_i]+PE_i\big).
$$

При $d=4$ эмбеддинг $(1,0,0,0)$ в позиции 0 превращается в
$(2,1,0,1)$ до dropout: масштабированный токен равен $(2,0,0,0)$,
а синусоидальная кодировка позиции 0 равна $(0,1,0,1)$.
Входная и целевая последовательности имеют собственную нумерацию позиций.
PE прибавляется **один раз внизу каждого стека**, а не заново после
каждого блока.

В полном немаскированном self-attention без позиционных признаков
перестановка входных строк переставляет выходные строки тем же образом.
Причинная маска уже нарушает эту симметрию: позиции имеют разные префиксы.
Поэтому неверно утверждать, что *любая* модель без отдельной PE
совсем не может различать порядок. Проверка симметрии, обучаемые позиции,
Shaw, T5, RoPE и ALiBi разобраны в
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|предыдущей главе]].

У ранних GPT и BERT обучаемые абсолютные позиции также добавляются снизу.
У LLaMA RoPE вращает Q и K внутри каждого attention-слоя.
Перенос синусоидальной суммы из исходного Transformer в такой блок
добавил бы ещё один позиционный механизм и изменил бы модель.

## 5. Оригинальный encoder block

Пусть $H^{(0)}$ — эмбеддинги исходной фразы с позициями.
В слое $\ell$ энкодер сначала выполняет полное multi-head self-attention,
а затем FFN:

$$
\begin{aligned}
H&=H^{(\ell-1)},\\
A&=\operatorname{MHA}_{\ell}(H,H,H),\\
U&=\operatorname{LN}_{\ell,1}(H+\operatorname{Dropout}(A)),\\
F&=\operatorname{FFN}_{\ell}(U),\\
H^{(\ell)}&=\operatorname{LN}_{\ell,2}(U+\operatorname{Dropout}(F)).
\end{aligned}
$$

Здесь аргументы MHA обозначают источники Q, K и V до обучаемых проекций.
«Полное» разрешает каждой действительной позиции читать все действительные
позиции исходной фразы. PAD-ключи исключаются маской.
Причинная маска энкодеру переводчика не нужна: исходная фраза уже известна целиком.

После первого attention строка `кот` может зависеть от `спит`.
Следующая FFN преобразует эту контекстную строку.
Во втором слое attention получает уже преобразованные представления обеих
позиций, а не исходные словарные эмбеддинги. Так через глубину комбинируются
результаты разных подслоёв.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/The_transformer_encoder_decoder_stack.png]]

*Стеки энкодера и декодера. Между блоками передаются состояния позиций,
не идентификаторы токенов. Стрелки из верхнего блока энкодера ведут
к cross-attention каждого декодер-блока.
Источник: Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

У Transformer Base шесть энкодер-блоков и шесть декодер-блоков,
$d=512$, восемь голов ширины 64 и FFN ширины 2048.
Блоки одинаково устроены, но имеют **разные параметры**.
Повторное применение одного объекта `Block` разделило бы веса
между глубинами; обычный стек создаёт отдельные экземпляры.

Результат $M=H^{(N)}$ имеет форму `[B,T_src,d]`.
Длина исходной фразы сохраняется: энкодер не сжимает её в один вектор,
как фиксированный контекст раннего Seq2Seq. Все строки $M$ остаются
доступны вниманию декодера.

## 6. Оригинальный decoder block

Первое целевое состояние строится из `<BOS>`. Оно ещё не содержит
выбранного токена `the`. Masked self-attention обрабатывает известный
целевой префикс, затем cross-attention обращается к памяти $M$.
После него FFN преобразует каждую целевую строку.

Для состояния $D$ один post-norm блок выполняет три последовательных обновления:

$$
\begin{aligned}
A&=\operatorname{MHA}_{self}(D,D,D),\\
U&=\operatorname{LN}_1(D+\operatorname{Dropout}(A)),\\
B&=\operatorname{MHA}_{cross}(U,M,M),\\
C&=\operatorname{LN}_2(U+\operatorname{Dropout}(B)),\\
F&=\operatorname{FFN}(C),\\
D'&=\operatorname{LN}_3(C+\operatorname{Dropout}(F)).
\end{aligned}
$$

В self-attention применяется причинная маска целевых позиций.
В cross-attention разрешён весь действительный исходный текст.
Треугольная маска к нему не подходит: позиция перевода 1 может нуждаться
в последнем слове оригинала. Входная и выходная длины вообще могут различаться.

У одной cross-attention головы:

$$
\begin{aligned}
Q&=UW_Q\in\mathbb R^{T_{tgt}\times d_h},\\
K&=MW_K\in\mathbb R^{T_{src}\times d_h},\\
V&=MW_V\in\mathbb R^{T_{src}\times d_h}.
\end{aligned}
$$

Матрица оценок имеет форму $T_{tgt}\times T_{src}$.
Она отвечает на вопрос, какие исходные позиции участвуют в каждой
целевой строке. Каждый декодер-блок имеет собственные $W_K,W_V$:
память $M$ общая, её проекции не обязаны совпадать между слоями.

В числовом примере после проекций есть два запроса и три ключа,
причём последний ключ соответствует PAD:

$$
\begin{aligned}
Q&=\begin{pmatrix}1&0\\0&\sqrt2\ln3\end{pmatrix},\\
K&=\begin{pmatrix}\sqrt2\ln2&0\\0&1\\100&100\end{pmatrix},\\
V&=\begin{pmatrix}2&0\\0&3\\100&100\end{pmatrix}.
\end{aligned}
$$

После деления на $\sqrt2$ и исключения PAD оценки первой строки
равны $(\ln2,0,-\infty)$. Их экспоненты пропорциональны $(2,1,0)$,
веса равны $(2/3,1/3,0)$, контекст — $(4/3,1)$.
Для второй строки оценки $(0,\ln3,-\infty)$ дают
веса $(1/4,3/4,0)$ и контекст $(1/2,9/4)$.
Одни и те же исходные значения используются по-разному для двух запросов.
Даже очень большое число в PAD-строке не влияет на контекст.

```python
import math
import torch

q = torch.tensor([[1., 0.], [0., math.sqrt(2)*math.log(3)]],
                 dtype=torch.float64)
k = torch.tensor([[math.sqrt(2)*math.log(2), 0.],
                  [0., 1.], [100., 100.]], dtype=q.dtype)
v = torch.tensor([[2., 0.], [0., 3.], [100., 100.]], dtype=q.dtype)
scores = (q @ k.T / math.sqrt(2)).masked_fill(
    torch.tensor([[False, False, True]]), -torch.inf)
a = scores.softmax(-1)
context = a @ v
torch.testing.assert_close(a, torch.tensor(
    [[2/3, 1/3, 0.], [1/4, 3/4, 0.]], dtype=q.dtype))
torch.testing.assert_close(context, torch.tensor(
    [[4/3, 1.], [1/2, 9/4]], dtype=q.dtype))
print(a, context)
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/full-transformer-2026-10-21/transformer_resideual_layer_norm_3.png]]

*В каждом декодер-блоке три подслоя и три остаточные связи;
в энкодер-блоке их два. Слева показаны две исходные позиции, справа —
представления целевых позиций. Decoder self-attention должен быть причинным,
хотя в этой схеме он подписан общим названием Self-Attention.
Источник: Jay Alammar, [The Illustrated Transformer, The Residuals](https://jalammar.github.io/illustrated-transformer/#the-residuals),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Последний декодер-блок всё ещё возвращает векторы ширины $d$.
Для получения распределения по словарю нужна отдельная выходная проекция.

## 7. Обучение и генерация

### От состояния декодера к распределению

Пусть выход последнего блока $h_i\in\mathbb R^d$.
Проекция $W_{out}\in\mathbb R^{d\times|V|}$ вычисляет
$z_i=h_iW_{out}+b_{out}$. Координата $z_{iv}$ — логит токена $v$.
Softmax по словарю даёт $p(v\mid x,y_{<i})$.
Это ещё не выбранный токен: выбор argmax, сэмплирование и поиск по лучу
используют одно распределение разными способами.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoder_output_softmax.png]]

*Выходной вектор расширяется до размера словаря. На рисунке выбран индекс
с максимальной вероятностью; это greedy decoding.
Подпись log_probs соответствует log-softmax, тогда как блок назван Softmax:
для выбора argmax оба варианта дают один индекс, но вероятности и их
логарифмы различаются. Код ниже передаёт в cross-entropy исходные логиты.
Источник: Jay Alammar, [The Illustrated Transformer, The Final Linear and Softmax Layer](https://jalammar.github.io/illustrated-transformer/#the-final-linear-and-softmax-layer),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Для примера ограничим выходной словарь тремя токенами:
`cat`, `sleeps`, `<EOS>`. При состоянии $h=(1,2)$ и столбцах $W_{out}$,
равных $(1,0)$, $(0,1)$, $(-1,0)$, получаем логиты $(1,2,-1)$.
Нормирующая сумма равна $e^1+e^2+e^{-1}$; вероятности приблизительно
равны $(0.259496,0.705385,0.035119)$.
Для правильного `sleeps` потеря $-\ln p$ равна $0.349012$.

Производная по логитам равна $p-\mathrm{onehot}(\text{target})$:
$(0.259496,-0.294615,0.035119)$.
При шаге градиентного спуска логит правильного ответа увеличивается,
а остальных уменьшается, если оптимизировать непосредственно эти три числа.
В сети оптимизируются веса: градиент идёт через $W_{out}$ в последний блок,
cross-attention, память энкодера и таблицы эмбеддингов.

В исходной статье одна матрица разделяется между входной таблицей,
целевой таблицей и выходной проекцией при общем словаре.
Тогда $W_{out}=E^\top$; градиенты от всех использований складываются.
Совпадение размера матриц само по себе ещё не означает разделения:
в коде это должен быть один объект-параметр, а не две копии чисел.

### Сдвиг целевой последовательности и маски

При обучении правильный перевод известен:

```text
позиция:          0       1       2       3
вход декодера:  <BOS>    the     cat    sleeps
цель:            the    cat    sleeps   <EOS>
```

Строка 2 получает `cat` и должна предсказать `sleeps`.
В self-attention ей разрешены позиции 0, 1 и 2.
Токен `sleeps` уже присутствует во входной строке 3, но причинная маска
запрещает её читать. Без сдвига целевой токен лежал бы в той же строке;
одна треугольная маска не устранила бы такое подглядывание.

Во всех слоях декодера причинные связи сохраняются.
FFN и LayerNorm работают по одной позиции и не добавляют связей с будущим.
Cross-attention читает только исходный текст, поэтому тоже не открывает
будущие токены перевода. Все четыре распределения можно вычислить параллельно
и обучать по сумме или среднему их потерь.

Если примеры дополнены до общей длины, нужны три разные операции:
исключение PAD-ключей в энкодере, исключение PAD-ключей целевого префикса
в decoder self-attention и исключение PAD-целей из функции потерь.
Cross-attention повторно использует маску **исходных** PAD-ключей.
Маска ключей не обязана обнулять состояния PAD-запросов:
эти строки могут иметь ненулевые значения, но не должны влиять на
действительные позиции и потерю.

Для логитов $z_{bt}$ и целей $y_{bt}$ используем среднюю потерю
по $n$ действительным целевым токенам:

$$
\mathcal L=-\frac1n\sum_{(b,t):\,y_{bt}\ne PAD}
\log\operatorname{softmax}(z_{bt})_{y_{bt}}.
$$

EOS входит в эту сумму: модель учится останавливаться.
BOS служит началом префикса, но не дополнительной целью в этой раскладке.

Исходный Transformer использовал label smoothing с $\epsilon_{ls}=0.1$.
Для трёх классов вариант с уверенностью $0.9$ на правильном классе
и $0.05$ на двух остальных даёт в нашем примере потерю $0.549012$.
Вместо требования приблизить одну вероятность к единице оптимизируется
мягкое целевое распределение. В [Annotated Transformer, Label Smoothing](https://nlp.seas.harvard.edu/annotated-transformer/#label-smoothing)
дополнительно исключён класс PAD, а функция возвращает KL-дивергенцию.
Она отличается от cross-entropy на энтропию целевого распределения,
но при фиксированных целях имеет тот же градиент по логитам.
Параметр `label_smoothing` в PyTorch смешивает цель с равномерным распределением
**по всем** классам; для трёх классов это другие числа.
В полном примере ниже smoothing отключён, чтобы отдельно проверить
сдвиг, маски и обычную cross-entropy.

```python
import torch
import torch.nn.functional as F

z = torch.tensor([1., 2., -1.], dtype=torch.float64, requires_grad=True)
logp = z.log_softmax(-1)
p = logp.exp()
loss = -logp[1]
loss.backward()
expected_grad = p.detach() - torch.tensor([0., 1., 0.])
torch.testing.assert_close(z.grad, expected_grad)
soft_target = torch.tensor([0.05, 0.9, 0.05], dtype=z.dtype)
smooth_ce = -(soft_target * logp).sum()
smooth_kl = (soft_target * (soft_target.log() - logp)).sum()
torch.testing.assert_close(
    F.cross_entropy(z[None], torch.tensor([1]), label_smoothing=0.1),
    -(torch.tensor([1/30, 14/15, 1/30], dtype=z.dtype) * logp).sum())
print(p.detach(), float(loss.detach()), float(smooth_ce.detach()))
print("KL", float(smooth_kl.detach()), "gradient", z.grad)
```

### Полный проход и один шаг обучения

Следующая реализация сохраняет post-norm, три подслоя декодера,
масштабирование эмбеддинга, синусоидальные позиции и общий словарь.
Она показывает вычислительный граф, но не воспроизводит обучение
Transformer Base: ширина, число слоёв, инициализация, данные,
оптимизатор и функция потерь здесь учебные. Маски используют соглашение
`True = разрешённый ключ`.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2

class MHA(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.heads, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = [nn.Linear(d, d) for _ in range(4)]
        self.dropout = dropout

    def forward(self, query, key, value, allowed):
        b, tq, d = query.shape
        def split(x):
            return x.reshape(b, -1, self.heads, self.dh).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(value))
        if allowed.dtype != torch.bool or not allowed.any(-1).all():
            raise ValueError("each query needs an allowed key; use boolean masks")
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        self.last_weights = weights
        weights = F.dropout(weights, self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).contiguous().reshape(b, tq, d)
        return self.o(out)

class EncoderBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.attn = MHA(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, allowed):
        x = self.n1(x + self.drop(self.attn(x, x, x, allowed)))
        return self.n2(x + self.drop(self.ff(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.self_attn, self.cross_attn = MHA(d, heads, dropout), MHA(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = [nn.LayerNorm(d) for _ in range(3)]
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, self_allowed, source_allowed):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, self_allowed)))
        x = self.n2(x + self.drop(self.cross_attn(x, memory, memory, source_allowed)))
        return self.n3(x + self.drop(self.ff(x)))

class TinyTranslation(nn.Module):
    def __init__(self, vocab=10, d=8, heads=2, ff=16, layers=2, dropout=0.0):
        super().__init__()
        if d % 2 or layers < 1:
            raise ValueError("use even d and a nonempty stack")
        self.emb = nn.Embedding(vocab, d)
        self.enc = nn.ModuleList([EncoderBlock(d, heads, ff, dropout)
                                  for _ in range(layers)])
        self.dec = nn.ModuleList([DecoderBlock(d, heads, ff, dropout)
                                  for _ in range(layers)])
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.drop = nn.Dropout(dropout)

    def embed(self, ids):
        d = self.emb.embedding_dim
        pos = torch.arange(ids.shape[1], device=ids.device, dtype=self.emb.weight.dtype)
        freq = 10000.0 ** (-torch.arange(
            0, d, 2, device=ids.device, dtype=self.emb.weight.dtype) / d)
        angles = pos[:, None] * freq[None, :]
        pe = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        return self.drop(math.sqrt(d) * self.emb(ids) + pe)

    def encode(self, src):
        valid = src.ne(PAD)
        memory = self.embed(src)
        for layer in self.enc:
            memory = layer(memory, valid[:, None, None, :])
        return memory, valid

    def decode(self, ids, memory, source_valid):
        t = ids.shape[1]
        causal = torch.ones(t, t, dtype=torch.bool, device=ids.device).tril()
        allowed = causal[None, None] & ids.ne(PAD)[:, None, None, :]
        x = self.embed(ids)
        for layer in self.dec:
            x = layer(x, memory, allowed, source_valid[:, None, None, :])
        return self.head(x)

    def forward(self, src, decoder_ids):
        memory, valid = self.encode(src)
        return self.decode(decoder_ids, memory, valid)

torch.manual_seed(7)
model = TinyTranslation().double()
# Shared vocabulary: 3=кот, 4=спит, 5=the, 6=cat, 7=sleeps.
src = torch.tensor([[3, 4, EOS, PAD]])
target = torch.tensor([[BOS, 5, 6, 7, EOS, PAD]])
decoder_ids, labels = target[:, :-1], target[:, 1:]
logits = model(src, decoder_ids)
assert logits.shape == (1, 5, 10)
loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=PAD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
assert model.enc[0].attn.q.weight.grad is not None
assert model.head.weight is model.emb.weight
optimizer.step()
new_loss = F.cross_entropy(
    model(src, decoder_ids).flatten(0, 1), labels.flatten(), ignore_index=PAD)
print("before/after", float(loss.detach()), float(new_loss.detach()))
print("cross-attention", tuple(model.dec[-1].cross_attn.last_weights.shape))
```

Здесь память имеет форму `[1,4,8]`, вход декодера — `[1,5]`,
cross-attention — `[1,2,5,4]`, логиты — `[1,5,10]`.
В функции потерь участвуют четыре цели `the cat sleeps <EOS>`, но не последний PAD.
Отдельные экземпляры блоков не разделяют веса по глубине;
общая таблица эмбеддингов и выходной слой, напротив, намеренно разделяют их.

Градиент запроса первого энкодер-блока показывает путь от ошибки перевода
через декодер и cross-attention в энкодер. Проверять только
`head.weight.grad` недостаточно: выходной слой мог бы обучаться даже
при случайно отсоединённой памяти. Один уменьшающий потерю шаг
не означает, что модель научилась переводить или обобщать.

### Авторегрессионный вывод

На генерации правильных `the cat sleeps` во входе ещё нет.
Сначала декодер получает только BOS, выдаёт распределение первого токена,
выбранный токен добавляется к префиксу, и вычисление повторяется до EOS
или ограничения длины.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoding_2.gif]]

*Повторные шаги декодера с растущим целевым префиксом.
Исходная фраза уже закодирована; очередной выбранный токен возвращается
на вход снизу. Это анимация автора без изменения кадров.
Источник: Jay Alammar, [The Illustrated Transformer, The Decoder Side](https://jalammar.github.io/illustrated-transformer/#the-decoder-side),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

В реализации можно сохранять память энкодера и не запускать его заново.
K/V cross-attention также неизменны в пределах одного перевода, хотя
различаются между слоями декодера. Собственный KV-cache декодера
сохраняет K/V уже обработанных целевых позиций.
В post-norm Transformer без будущих связей эти старые состояния
не меняются при добавлении нового токена. Эквивалентность полному
пересчёту предполагает режим eval, те же позиции и маски.

Кэш сокращает повторные вычисления; следующий неизвестный токен всё равно
зависит от только что выбранного. Ни маска, ни параллельное обучение
не превращают обычную авторегрессионную генерацию в один проход.
В примере выше кэша нет: для прозрачности `decode` пересчитывает весь префикс.

## 8. Три архитектурные ветви

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt-2-transformer-xl-bert-3.png]]

*Слева GPT-2 с причинными блоками без cross-attention, в центре BERT
с двунаправленными блоками, справа Transformer-XL с памятью между сегментами.
Рисунок не сравнивает GPT-1 и LLaMA; их отличия перечислены в тексте ниже.
Источник: Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

В уже разобранном encoder–decoder исходный текст образует отдельную память,
а декодер генерирует условный выход. Два других варианта оставляют
только энкодер или только причинный декодер. LLaMA развивает последний
вариант, меняя нормализацию, FFN и позиционный механизм.

### Encoder-only: BERT

BERT оставляет стек энкодера:

- полное двунаправленное self-attention;
- представления токенов, сегментов и обучаемые позиционные представления;
- masked language modeling;
- в оригинале также next sentence prediction.

При MLM выбирается 15% WordPieces; из них 80% заменяются на `[MASK]`, 10% — на
случайный токен, 10% остаются неизменными. Функция потерь считается по выбранным позициям.
BERT не является авторегрессионным генератором: задача предобучения учит его
восстанавливать скрытые части, а не продолжать префикс слева направо
([Devlin et al., §3–3.1](https://arxiv.org/pdf/1810.04805v2#page=3)).

### Decoder-only: GPT

GPT использует причинный стек без энкодера и без cross-attention:

- masked self-attention;
- FFN;
- обучаемые позиционные представления в ранних GPT;
- предсказание следующего токена.

В GPT-1 причинное языковое предобучение используется как инициализация
для обучения на размеченных задачах NLP. GPT-2 затем масштабировал модель и данные
и сделал акцент на переносе без дополнительного градиентного обучения на
целевой задаче. В опытах GPT-2 с вопросами и ответами в контекст добавлялись
примеры пар, без изменения весов
([отчёт GPT-2, §3.8](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=7)).

### Modern decoder-only: LLaMA

LLaMA сохраняет causal decoder, но меняет детали:

- pre-normalization с RMSNorm;
- RoPE вместо абсолютных позиционных представлений;
- SwiGLU вместо ReLU FFN;
- отсутствие смещений в ряде линейных слоёв;
- grouped-query attention в поздних версиях Llama.

LLaMA 1 использовала multi-head attention. GQA в более поздних выпусках
не является свойством самой первой версии. Изменения описаны в
[Touvron et al., §2.2–2.4](https://arxiv.org/pdf/2302.13971v1#page=3);
отсутствие смещений и проекции Q/K/V с одинаковым числом голов видны в
[коде первого выпуска](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py).

## 9. Архитектурный diff

| Компонент | Transformer 2017 | BERT Base | GPT-1 | LLaMA 1 |
|---|---|---|---|---|
| архитектура | encoder-decoder | encoder | causal decoder | causal decoder |
| norm layout | Post-LN | Post-LN | Post-LN | Pre-RMSNorm |
| activation / FFN | ReLU | GELU | GELU | SwiGLU |
| позиции | sinusoidal | learned absolute | learned absolute | RoPE |
| cross-attention | decoder only | нет | нет | нет |
| задача обучения | translation | MLM + NSP | next token + fine-tune | next token |

В GPT-1 нормализация следует за суммой в обоих подслоях:
это видно в функции `block` исходного [OpenAI train.py](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/train.py#L140-L147).
GPT-2 переносит нормализацию перед подслоями и добавляет финальную LayerNorm
([отчёт GPT-2, §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
Поэтому строка GPT-1 в таблице не описывает все последующие GPT.
Обозначение LLaMA 1 также намеренно ограничено первой версией.

<a id="architecture-evidence-boundary"></a>

## 10. Архитектурное решение и эмпирический рецепт — не одно и то же

Расположение нормализации можно определить по коду.
Преимущество такой схемы по качеству требует сравнения обучения.
Замена LayerNorm на RMSNorm одновременно меняет центрирование,
число параметров и вычисления; сравнение двух готовых моделей не отделяет
эти причины от различий в данных, токенизации и числе обучающих токенов.

В [CS336 Lecture 3, слайды 2–9](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf#page=2)
исходный Transformer сопоставляется с современным учебным вариантом,
а затем — с набором выпущенных моделей. Этот набор показывает
распространённые решения, но не заменяет контролируемую ablation.

Например, у SwiGLU с тремя матрицами та же скрытая ширина, что у
двухматричной ReLU FFN, означает примерно в полтора раза больше весов
в этих проекциях. Чтобы сравнить активации при близком числе параметров,
ширину приходится менять. При фиксированной ширине можно изучать
другой вопрос, но тогда в эксперименте меняются сразу два свойства.

Сохранение числа параметров тоже не гарантирует равенства времени:
эффективность матричных ядер зависит от размеров и реализации.
Для утверждения о качестве нужны сопоставимые данные и обучение;
для утверждения о скорости — измерения на заданном оборудовании
и в заданном режиме.

## 11. Почему Transformer вытеснил рекуррентную основу

В сравнении оригинальной статьи:

| Layer | Complexity per layer | Sequential operations | Maximum path length |
|---|---:|---:|---:|
| self-attention | $O(T^2d)$ | $O(1)$ | $O(1)$ |
| recurrent | $O(Td^2)$ | $O(T)$ | $O(T)$ |
| dilated convolution | $O(kTd^2)$ | $O(1)$ | $O(\log_k T)$ |

Логарифмический путь в последней строке предполагает стек расширяющихся
dilated kernels ширины $k>1$. Для обычных contiguous kernels без dilation
оригинальная статья указывает путь $O(T/k)$: каждый слой увеличивает
доступную область лишь линейно. Стоимость дана для одного слоя,
а путь — через глубину, необходимую для связи удалённых позиций.

В таблице сравнивается взаимодействие позиций внутри слоя. Для полного
Transformer дополнительно считаются Q/K/V-проекции порядка $Td^2$
и FFN порядка $Tdd_{ff}$. Поэтому одна строка $O(T^2d)$ не описывает
всю стоимость блока и не доказывает ускорения при любой длине.

При обработке известной последовательности все запросы можно вычислять
параллельно; два удалённых токена связаны одним полным attention-слоем.
Для плотного внимания объём попарных вычислений растёт квадратично с длиной.
FlashAttention меняет обмен данными с памятью, не устраняя этот объём
арифметики; разреженное и линейное внимание меняют уже структуру
или вычисление взаимодействий.

> [!warning] Таблица зависит от режима
> $O(1)$ последовательных операций относится к обработке уже известной последовательности в
> слое, а не к генерации неизвестных будущих токенов. И self-attention не всегда
> дешевле рекуррентной сети: соотношение зависит от $T$, $d$, оборудования и реализации.

## 12. Минимальный современный causal block

В [CS336 Assignment 1, §3.5, стр. 26–27](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=26)
полный блок описан так, в оригинале:

> A Transformer block contains two ‘sub-layers’, one for the multihead self attention,
> and another for the SwiGLU feed-forward network. In each sub-layer, we first
> perform RMSNorm, then the main operation (MHA/FF), finally adding in the residual connection.

*Stanford CS336, Assignment 1, закреплённая версия a158843b.
Лицензия [MIT](/licenses/stanford-cs336-assignment1-MIT.txt).*

Вместо трёх подслоёв декодера переводчика остаются два: причинный
self-attention и FFN. Источник текста теперь тот же поток, который модель
продолжает. Входная таблица эмбеддингов, стек блоков, финальная RMSNorm
и выходная проекция вместе образуют языковую модель.

RMSNorm делит строку на корень из среднего квадрата координат,
не вычитая среднее:

$$
\operatorname{RMSNorm}(x)=
\gamma\odot\frac{x}{\sqrt{\frac1d\sum_r x_r^2+\epsilon}}.
$$

Для $(1,2,3,4)$ знаменатель без $\epsilon$ равен $\sqrt{7.5}$,
а выход при $\gamma=1$ приблизительно равен
$(0.365148,0.730297,1.095445,1.460593)$.
Это не результат LayerNorm из раздела 2: положительный сдвиг строки
не удалён. В таком блоке RMSNorm применяется перед attention и FFN;
RoPE — к Q/K после их проекций; V остаётся без поворота.

Следующий код самодостаточен. В нём обычное MHA, без GQA, KV-cache,
dropout, MoE и дополнительных механизмов конкретных выпусков.
Это небольшой современный декодер по указанной схеме, а не
загрузчик весов LLaMA или полный ответ для тестов CS336.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight

class RotarySelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        if d % heads or (d // heads) % 2:
            raise ValueError("use even head width and d divisible by heads")
        self.heads, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = [
            nn.Linear(d, d, bias=False) for _ in range(4)]

    def rotate(self, x):
        t = x.shape[-2]
        positions = torch.arange(t, device=x.device, dtype=x.dtype)
        freq = 10000.0 ** (-torch.arange(0, self.dh, 2,
                                        device=x.device, dtype=x.dtype) / self.dh)
        angles = positions[:, None] * freq[None, :]
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        a, b = pairs.unbind(-1)
        return torch.stack((a*angles.cos() - b*angles.sin(),
                            a*angles.sin() + b*angles.cos()), -1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        def split(y):
            return y.reshape(b, t, self.heads, self.dh).transpose(1, 2)
        q, k, v = self.rotate(split(self.q(x))), self.rotate(split(self.k(x))), split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        allowed = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        y = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        return self.o(y)

class SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate, self.up = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(d), RMSNorm(d)
        self.attn, self.ffn = RotarySelfAttention(d, heads), SwiGLU(d, ff)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

class TinyCausalLM(nn.Module):
    def __init__(self, vocab=10, d=8, heads=2, ff=24, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

torch.manual_seed(11)
lm = TinyCausalLM().double()
tokens = torch.tensor([[1, 5, 6, 7, 2]])
logits = lm(tokens[:, :-1])
loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
loss.backward()
print("logits", tuple(logits.shape), "loss", float(loss.detach()))
```

Слои применяются последовательно по глубине, позиции одного слоя —
параллельно. Вход `[1,4]` даёт логиты `[1,4,10]` для четырёх следующих
токенов. В примере нет padding; для пакета разной длины потребуется
добавить маски ключей и исключение PAD-целей, как в переводчике выше.
Выходная матрица здесь **не связана** с таблицей эмбеддингов.

Подробный разбор RMSNorm, SwiGLU и порядка операций продолжен в
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|анатомии современного блока]].
Официальные адаптеры и тесты заданий CS336 остаются в
[репозитории курса](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712/tests);
исполнение этого примера не считается их прохождением.

## 13. Как читать реальный код

Три уровня сложности:

1. [Karpathy `ng-video-lecture`](https://github.com/karpathy/ng-video-lecture) —
   последовательная сборка причинного декодера на PyTorch.
2. [Karpathy `build-nanogpt`](https://github.com/karpathy/build-nanogpt) —
   воспроизведение архитектуры GPT-2 124M, загрузка весов,
   конвейер данных, инициализация и обучение.
3. [Meta Llama 3 reference code](https://github.com/meta-llama/llama3/blob/main/llama/model.py) —
   RoPE, RMSNorm, SwiGLU и GQA в современной реализации.

У этих реализаций различаются цель и масштаб. Короткий учебный код показывает
связи операций; воспроизведение GPT-2 добавляет совместимость с весами и
процедуру обучения; reference code Llama задаёт конкретный формат
параметров и вычисления модели. Ни один из них не следует автоматически
считать точной реализацией Transformer 2017 года.

## 14. Проверка форм тензоров

Для causal LLM с обычным multi-head attention (MHA):

```text
token ids                 [B, T]
token embeddings          [B, T, d]
Q, K, V                   [B, h, T, dh]
attention scores          [B, h, T, T]
attention result          [B, T, d]
residual after attention  [B, T, d]
FFN hidden                [B, T, dff]
residual after FFN        [B, T, d]
vocabulary logits         [B, T, |Vocab|]
```

При GQA у Q число голов $h_q$, а у K/V — $h_{kv}<h_q$:
формы соответственно `[B,h_q,T,dh]` и `[B,h_kv,T,dh]`.
Приведённая таблица поэтому не является универсальной раскладкой KV-голов.

Для сложения $x+F(x)$ формы обоих слагаемых должны совпадать.
Совпадение форм ещё не проверяет порядок элементов после разделения голов,
правильность маски или используемую нормализацию.


Подсчёт параметров, матричных произведений и памяти для backward разобран в
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|главе о формах, параметрах и стоимости]],
после чего ту же последовательность можно реализовать в
[[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля|Capstone 1]].

## Краткие итоги

- Оригинальный Transformer — encoder-decoder, а не GPT-подобная башня.
- Блок энкодера состоит из полного self-attention и FFN.
- Блок оригинального декодера содержит causal self-attention, cross-attention и FFN.
- Attention переносит информацию между позициями; FFN нелинейно преобразует каждую позицию отдельно.
- Остаточный поток сохраняет форму `B × T × d_model`.
- BERT, GPT и LLaMA различаются не только маской, но и задачей обучения, позиционным механизмом,
  нормализацией и FFN.
- Схему современной LLM нельзя выдавать за точную схему статьи 2017 года.

## Источники и интерактивы

### Первичные

- [Stanford CS336 Spring 2026, Lecture 3 — pinned source](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf)
- [Stanford CS336 Spring 2026, Assignment 1 — pinned handout](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf)
- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Radford et al. — Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Devlin et al. — BERT](https://arxiv.org/abs/1810.04805)
- [Touvron et al. — LLaMA](https://arxiv.org/abs/2302.13971)

### Код и объяснения

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/13_transformers|TinyTorch 13 — Transformers]] — исполняемая сборка `TransformerBlock` из attention, MLP и layer normalization с авторегрессионной генерацией.
- [Harvard — The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy — Let’s build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy — Let’s reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder|Encoder–Decoder]]

**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]
