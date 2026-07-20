---
title: Rotary Position Embedding
aliases: [RoPE]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2104.09864
---

# Rotary Position Embedding (RoPE)

**RoPE** кодирует позицию, поворачивая пары координат query и key на угол, зависящий от позиции. Оно не добавляет position vector к hidden state, а изменяет геометрию dot product.

Для пары координат:

$$
R_m(\theta)=
\begin{bmatrix}
\cos m\theta & -\sin m\theta\\
\sin m\theta & \cos m\theta
\end{bmatrix}.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-32-34/roformer-rope-figure1.png]]

*Верхняя часть рисунка показывает поворот одной пары координат. Внизу query или
key разбит на пары; каждая пара вращается со своей частотой, но угол каждой из
них определяется позицией токена. Jianlin Su et al., “RoFormer: Enhanced
Transformer with Rotary Position Embedding”, Figure 1,
[с. 5](https://arxiv.org/pdf/2104.09864#page=5). Фигура извлечена из статьи без
изменения содержания.*

Рисунок полезно читать слева направо: вектор не получает отдельную позиционную
добавку. Вместо этого его координаты попарно превращаются в точки на плоскости и
поворачиваются. Низкие частоты меняются медленно и различают далёкие позиции,
высокие — быстро и лучше различают близкие.

Ключевое свойство: внутреннее произведение повёрнутых Q и K зависит от относительного смещения позиций. Values обычно не вращаются.

## Длинный контекст

RoPE не гарантирует хорошую экстраполяцию далеко за training length. Расширения контекста меняют частоты или масштаб позиции: position interpolation, NTK-aware scaling, YaRN и другие методы. Их нельзя считать одним стандартным «RoPE scaling»: проверяйте рецепт конкретной модели.

## Trade-offs

RoPE прост, не требует learned position table и хорошо интегрируется с attention. Однако выбор base/frequencies и extrapolation recipe влияет на качество длинного контекста; неправильное scaling может ухудшить короткие и длинные зависимости.

## Подробнее

Геометрический вывод вращения, применение RoPE к парам координат и методы
расширения контекста разобраны в главе [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE|RoPE]].

## Источники

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
