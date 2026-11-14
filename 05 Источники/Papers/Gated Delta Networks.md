---
title: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
type: source-note
status: reviewed
source_type: paper
authors: [Songlin Yang, Jan Kautz, Ali Hatamizadeh]
published: 2024-12-09
last_verified: 2026-08-31
url: https://arxiv.org/abs/2412.06464
venue: ICLR 2025
concepts: [linear attention, delta rule, matrix state, hybrid architecture]
---

# Gated Delta Networks

## Проблема

Простая linear-attention memory добавляет outer products и плохо исправляет
устаревшие ассоциации. Независимые линии работ использовали gate для общего
стирания и delta rule для направленной коррекции. Статья объединяет их.

## Механизм

В записи авторов состояние $S_t$ имеет оси «значения × ключи».
Сначала прежняя память затухает с коэффициентом $\alpha_t$.
По ослабленному состоянию слой оценивает значение текущего ключа
и записывает поправку:

$$
S_t=\alpha_tS_{t-1}
+\beta_t(v_t-\alpha_t S_{t-1}k_t)k_t^\top.
$$

При единичном ключе и $\beta_t=1$ чтение по этому ключу после обновления
даёт $v_t$. Например, если старая оценка 10, новое значение 30, а
$\alpha_t=0.5$, ослабленная оценка равна 5. Слой добавляет $30-5=25$
в направлении ключа, поэтому новая оценка равна 30.
См. [§3.1, уравнение 10](https://arxiv.org/html/2412.06464v3#S3.SS1).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/sequence-architecture-figures-2026-11-11/gated-deltanet-figure1-complete.png|H1 чередует Gated DeltaNet и скользящее внимание, H2 добавляет Mamba-2. Справа показаны проекции одного блока.]]

*Songlin Yang, Jan Kautz и Ali Hatamizadeh,
[Gated Delta Networks, v3, рис. 1, с. 7](https://arxiv.org/pdf/2412.06464v3#page=7),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

## Граница интерпретации

Gated DeltaNet и Mamba обе используют recurrent state и hardware-aware
parallelism, но имеют разное происхождение правила памяти. Gated DeltaNet нельзя
называть просто новой Mamba. Современные модели могут сочетать его с attention,
локальной свёрткой, MoE и другими компонентами.

## Куда интегрировано

[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|linear attention и delta rule]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Qwen|Qwen3-Next]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Kimi|Kimi Linear]]
