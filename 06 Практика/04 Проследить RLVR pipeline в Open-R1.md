---
title: Проследить RLVR pipeline в Open-R1
type: practice
status: canonical
last_updated: 2026-07-16
---

# Проследить RLVR pipeline в Open-R1

По [Open-R1](https://github.com/huggingface/open-r1) проследите:

```text
dataset → prompt → rollout → parser → verifier → reward →
group advantages → policy update → evaluation
```

Отдельно выпишите:

- что является environment;
- что именно проверяет reward;
- как обрабатывается invalid format;
- какие shortcuts может найти policy;
- какой baseline доказывает пользу RL, а не только данных.

Связано: [[02 Areas/ML & DL/03 Исследовательские линии/RLHF → DPO → RLVR]].

