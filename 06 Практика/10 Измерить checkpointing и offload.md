---
title: Измерить activation checkpointing и offload
type: practice
status: canonical
last_updated: 2026-07-24
---

# Измерить activation checkpointing и offload

Используйте [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/practice_part1.ipynb|EDLS Week 4, practice part 1]] вместе с [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/lecture.pdf|полной лекцией]].

Для одного Transformer block и небольшой stack измерьте baseline, selective
activation checkpointing и offload. В каждом случае сохраните:

- peak allocated и reserved GPU memory;
- step time и tokens/s;
- число повторно выполненных forward operators;
- объём и длительность D2H/H2D copies;
- численное совпадение loss и gradients в заданном tolerance.

Постройте Pareto-график «peak memory — step time». Затем увеличивайте sequence
length, пока baseline не перестанет помещаться, и покажите, какой режим
расширяет допустимую длину. Не называйте offload бесплатным: на timeline должно
быть видно, перекрылись ли copies с compute или попали на критический путь.
