---
title: "End-to-End Test-Time Training for Long Context"
type: source-note
status: reviewed
source_type: paper
authors: [Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun]
published: 2025-12-29
last_verified: 2026-08-31
url: https://arxiv.org/abs/2512.23675
code: https://github.com/test-time-training/e2e
concepts: [Test-Time Training, continual learning, long context, sliding-window attention]
---

# End-to-End Test-Time Training for Long Context

## Почему это не TTT-layer

TTT-E2E сохраняет стандартный Transformer со sliding-window attention. При
чтении длинного контекста часть параметров продолжает обучаться на next-token
prediction. Начальная модель meta-learned так, чтобы online updates переносили
информацию из ранних фрагментов в веса.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/ttt-e2e-figure3.png]]

*Figure 3: градиент из последующего фрагмента обновляет часть MLP-блоков обычной
sliding-window модели.*

## Что показано

Авторы сравнивают scaling по контексту и сообщают результаты для моделей до 3B,
обученных на 164B токенов. Официальный JAX-repository публикует конфигурации и
часть checkpoints. Это более сильное свидетельство, чем одна формула, но всё ещё
исследовательская система.

## Production-границы

Request-local mutable weights требуют изоляции, reset, snapshot и rollback.
Continuous batching объединяет запросы с общей статической моделью, а TTT-E2E
создаёт отдельную траекторию параметров для каждого контекста. Prefix reuse,
speculative branches и worker migration требуют явной семантики копирования.

## Куда интегрировано

[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|глава об альтернативах полному вниманию]] ·
[[02 Areas/ML & DL/03 Исследовательские линии/Альтернативы полному attention и stateful sequence models|обзор направления]]
