---
title: "Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks"
type: source-note
status: reviewed
last_verified: 2026-10-26
authors: Murtaza Dalal, Tarun Chiruvolu, Devendra Singh Chaplot, Ruslan Salakhutdinov
year: 2024
source_url: https://arxiv.org/abs/2405.01534v1
---

# Plan-Seq-Learn

ICLR 2024. [Paper, version 1](https://arxiv.org/html/2405.01534v1) ·
[Project, videos and code](https://mihdalal.github.io/planseqlearn/).

The method combines language plans, vision-based motion planning and learned
local control. A plan specifies target regions and stage termination conditions;
it does not assume an already trained library of manipulation skills. The
experiments use simulated environments, calibrated cameras and environment
reward functions. The language model itself is not trained by the low-level RL loop.

[[00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов#plan-seq-learn|The textbook section]]
reproduces the complete original English explanations in §§3.3–3.5 and D.3.1,
the prose paragraph of §3.2 and limitations in §B.3, with Figures 1, 2 and D.1.
The article is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
as recorded on its arXiv version page. Editorial comments are separated from quotations.
