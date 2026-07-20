---
title: Dense FFN и gated activations
aliases: [Gated FFN, GLU, ReGLU, GEGLU, SwiGLU]
type: concept
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://arxiv.org/abs/1612.08083
  - https://arxiv.org/abs/2002.05202
  - https://web.stanford.edu/class/cs336/
---

# Dense FFN и gated activations

В Transformer attention переносит информацию между позициями, а feed-forward
network преобразует каждый токен отдельно. Если вход блока имеет форму
`[batch, sequence, d_model]`, одна и та же FFN применяется к последней оси во
всех позициях; токены внутри FFN друг с другом не смешиваются.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-dense-ffn/transformer-block-ffn.png]]

*FFN — самостоятельный подслой Transformer block после self-attention. Jay
Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[прямой файл](https://jalammar.github.io/images/gpt2/gpt2-transformer-block-vectors-2.png).
Текст и иллюстрации сайта опубликованы под
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

## Обычная двухслойная FFN

Классическая position-wise FFN из Transformer имеет вид

$$
\operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2,
$$

где $W_1\in\mathbb{R}^{d\times m}$ расширяет вектор из model dimension $d$ во
внутреннюю ширину $m$, а $W_2\in\mathbb{R}^{m\times d}$ возвращает его в
residual stream. При $m=4d$ две матрицы содержат примерно $8d^2$ параметров без
учёта bias. Столько же скалярных умножений приходится на один токен с точностью
до принятого способа считать multiply-add.

Эта FFN называется dense: каждый токен проходит через все её матрицы. В
[[02 Areas/ML & DL/01 Справочник/FFN и MoE/Mixture of Experts|Mixture of
Experts]] FFN заменяется несколькими экспертами, но router выбирает для токена
лишь часть из них.

## От GLU к gated FFN

В Gated Linear Unit одна ветвь создаёт значения, а другая решает, какие из них
пропустить:

$$
\operatorname{GLU}(x)=\sigma(xW_g+b_g)\odot(xW_u+b_u).
$$

Для Transformer Shazeer рассматривает семейство, в котором sigmoid заменяется
другой функцией $\phi$, а после произведения добавляется выходная проекция:

$$
\operatorname{FFN}_{\phi}(x)=
\bigl(\phi(xW_g)\odot xW_u\bigr)W_d.
$$

$W_g$ строит gate, $W_u$ — преобразуемое содержимое, $W_d$ возвращает результат
в residual dimension. Обе первые ветви получают один и тот же $x$, но имеют
разные обучаемые веса. Поэлементное произведение выполняется до $W_d$.

| Вариант | Gate $\phi(z)$ | Формула внутреннего представления |
|---|---|---|
| GLU | $\sigma(z)$ | $\sigma(xW_g)\odot xW_u$ |
| ReGLU | $\max(0,z)$ | $\operatorname{ReLU}(xW_g)\odot xW_u$ |
| GEGLU | $\operatorname{GELU}(z)$ | $\operatorname{GELU}(xW_g)\odot xW_u$ |
| SwiGLU | $\operatorname{Swish}_\beta(z)$ | $\operatorname{Swish}_\beta(xW_g)\odot xW_u$ |

В современных LLM под SwiGLU обычно понимают $\beta=1$, то есть
$\operatorname{Swish}_1(z)=z\sigma(z)$, также называемую SiLU.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png]]

*ReLU обнуляет всю отрицательную полуплоскость, тогда как SiLU/Swish оставляет
плавную отрицательную область и плавно переходит к почти линейному режиму.
Stanford CS336, Assignment 1, Figure 3,
[материалы курса](https://web.stanford.edu/class/cs336/).*

Рисунок показывает только нелинейность gate. Полная SwiGLU не равна обычной FFN
с SiLU: результат SiLU-ветви ещё умножается на независимую линейную ветвь
$xW_u$. Именно мультипликативное взаимодействие отличает gated FFN от простой
замены ReLU на другую activation function.

## Почему внутренняя ширина уменьшается

Обычная FFN использует две большие матрицы и содержит примерно

$$P_{plain}=2dm$$

параметров. Gated FFN использует три:

$$P_{gated}=3dm_g.$$

Если оставить ту же внутреннюю ширину, gated-вариант будет примерно в полтора
раза дороже по параметрам и основным matrix multiplications. Для честного
сравнения с классической FFN ширины $m=4d$ выбирают

$$
3dm_g\approx2d(4d)\quad\Rightarrow\quad m_g\approx\frac{8}{3}d.
$$

На практике ширину округляют под требования hardware kernels и tensor
parallelism. Поэтому в конфигурации конкретной модели встречается не буквально
$8d/3$, а ближайшее удобное число. Сравнивать модели только по `intermediate_size`
ошибочно: сначала нужно проверить, две или три проекции использует их FFN.

## Что даёт gating и чего он не делает

Gate делает преобразование входно-зависимым: координата линейной ветви может
усиливаться, ослабляться или менять знак в зависимости от второй проекции того
же токена. Эксперименты в “GLU Variants Improve Transformer” показали преимущество
нескольких gated-вариантов над обычными ReLU/GELU FFN при сопоставимом бюджете;
SwiGLU затем вошла, например, в PaLM и LLaMA.

При этом SwiGLU не выполняет routing между экспертами и не экономит active
parameters: все три матрицы используются для каждого токена. Она также не
смешивает позиции — эту работу по-прежнему выполняет attention.

## Подробнее

Пошаговый проход одного токена через FFN, подсчёт параметров и переход к
gated-активациям даны в главе [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating|Dense FFN — token-wise вычисление, expansion и gating]].

## Источники

- Dauphin et al., [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083) — исходная GLU.
- Noam Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — ReGLU, GEGLU, SwiGLU и сравнение при сопоставимом числе параметров.
- Stanford CS336, [Language Modeling from Scratch](https://web.stanford.edu/class/cs336/) — реализация SwiGLU и выбор внутренней ширины.
- Touvron et al., [LLaMA](https://arxiv.org/abs/2302.13971) — применение SwiGLU в современной decoder-only модели.
