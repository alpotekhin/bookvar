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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/sequence-architecture-figures-2026-11-11/ttt-v1-figure4.png|Таблица сопоставляет три состояния и правила их обновления: вектор RNN, список K/V и параметры внутренней модели TTT.]]

*Yu Sun et al.,
[Learning to (Learn at Test Time), v1, рис. 4, с. 4](https://arxiv.org/pdf/2407.04620v1#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). В поздней версии
статьи это [рисунок 3](https://arxiv.org/html/2407.04620v4#S1.F3).*

## Что источник не доказывает

- что вся LLM должна обновляться на каждом пользовательском запросе;
- что TTT равен reasoning test-time compute;
- что результаты масштаба до 1.3B автоматически переносятся на frontier-модели;
- что mutable state уже имеет feature parity в mainstream serving engines.

## Куда интегрировано

[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|глава об альтернативах полному вниманию]] ·
[[02 Areas/ML & DL/03 Исследовательские линии/Альтернативы полному attention и stateful sequence models|обзор направления]]
