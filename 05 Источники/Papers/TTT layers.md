---
title: "Learning to (Learn at Test Time): TTT layers"
type: source-note
status: reviewed
source_type: paper
authors: [Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin]
published: 2024-07-05
last_verified: 2026-08-31
url: https://arxiv.org/abs/2407.04620
concepts: [Test-Time Training, fast weights, recurrent memory, meta-learning]
---

# Learning to (Learn at Test Time): TTT layers

## Почему источник важен

Работа предлагает новый тип состояния sequence model. Вместо вектора или
матрицы слой хранит параметры маленькой модели. Каждый входной токен задаёт
самосупервизируемый шаг обучения этой внутренней модели, а её обновлённые
параметры используются для чтения контекста.

## Механизм

Для состояния $W_t$:

$$
W_t=W_{t-1}-\eta\nabla_W\ell(W_{t-1};x_t),
$$

$$
y_t=f(q_t;W_t).
$$

TTT-Linear использует линейную внутреннюю модель, TTT-MLP — двухслойный MLP.
Внешний цикл meta-learning подбирает инициализацию и задачу внутреннего обучения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/alternatives-transformer-2026/ttt-figure3.png]]

*Figure 3 из статьи сопоставляет vector state RNN, список K/V self-attention и
параметры внутренней модели TTT.*

## Что источник не доказывает

- что вся LLM должна обновляться на каждом пользовательском запросе;
- что TTT равен reasoning test-time compute;
- что результаты масштаба до 1.3B автоматически переносятся на frontier-модели;
- что mutable state уже имеет feature parity в mainstream serving engines.

## Куда интегрировано

[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|глава об альтернативах полному вниманию]] ·
[[02 Areas/ML & DL/03 Исследовательские линии/Альтернативы полному attention и stateful sequence models|обзор направления]]
