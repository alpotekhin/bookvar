---
title: Оптимизировать один Transformer step
type: practice
status: canonical
last_updated: 2026-07-24
---

# Оптимизировать один Transformer step

**Полный исполняемый модуль:** [[05 Источники/Courses/Harvard ML Systems/tinytorch/20_capstone|TinyTorch 20 — Capstone]]. Он объединяет profiling, quantization, compression, acceleration, memoization и benchmarking в воспроизводимое сравнение baseline и optimized model со schema-validated JSON-отчётом.

Это итоговая работа по training performance. Выполняйте [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/seminar/practice.ipynb|EDLS Week 6 seminar notebook]] и сверяйтесь с [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/lecture.pdf|полной 146-слайдовой лекцией]].

Зафиксируйте baseline Transformer step. До оптимизации составьте три ведомости:

1. parameters, gradients, optimizer states и peak activations в bytes;
2. FLOP по attention, MLP и vocabulary projection;
3. HBM и communication traffic, который можно оценить.

После этого по одному примените доступные оптимизации: fused optimizer,
compiled/fused elementwise path, FlashAttention, activation checkpointing,
mixed precision и улучшение data pipeline. После каждого изменения проверяйте
loss/gradients и повторяйте одинаковый benchmark.

Итоговая таблица должна содержать step time, tokens/s, MFU либо достижимую
roofline efficiency, peak memory и качество. Для отклонённой оптимизации тоже
сохраните результат: объяснение, почему она не помогла, является частью работы.
