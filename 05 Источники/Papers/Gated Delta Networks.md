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

Упрощённое обновление:

$$
S_t=\alpha_tS_{t-1}
+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
$$

$\alpha_t$ управляет общим сохранением состояния. Разность
$v_t-S_{t-1}k_t$ исправляет именно ошибку текущей ассоциации.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/gated-deltanet-figure1.png]]

*Figure 1 показывает сам блок и два гибрида: с sliding-window attention и с
Mamba-2.*

## Граница интерпретации

Gated DeltaNet и Mamba обе используют recurrent state и hardware-aware
parallelism, но имеют разное происхождение правила памяти. Gated DeltaNet нельзя
называть просто новой Mamba. Современные модели могут сочетать его с attention,
локальной свёрткой, MoE и другими компонентами.

## Куда интегрировано

[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|linear attention и delta rule]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Qwen|Qwen3-Next]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Kimi|Kimi Linear]]
