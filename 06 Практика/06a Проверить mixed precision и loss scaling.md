---
title: Проверить mixed precision и loss scaling
type: practice
status: canonical
last_updated: 2026-08-06
---

# Проверить mixed precision и loss scaling

Смешанная точность полезна не потому, что «FP16 быстрее FP32», а когда выбранный
набор форматов уменьшает трафик и задействует быстрые аппаратные блоки, сохраняя
численно корректное обновление параметров. В этой работе нужно отдельно увидеть
ускорение, экономию памяти и потерю малых градиентов.

## Исходные материалы

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/lecture.pdf|Efficient DL Systems, Week 2 — полная оригинальная лекция]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/seminar/practice.ipynb|исходный seminar notebook]];
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision]].

## Экспериментальная матрица

На одной небольшой модели выполните одинаковый training step в режимах FP32,
TF32 (если применимо), FP16 с/без scaling и BF16. Зафиксируйте GPU, версии,
формат параметров, activations, gradients и accumulator.

Для каждого режима измерьте:

- loss и gradient norm по слоям;
- долю нулевых и неfinite gradients;
- peak allocated/reserved memory;
- step time после warmup и полезные samples/tokens per second;
- максимальную абсолютную и относительную ошибку против FP32 reference step.

## Сделать underflow наблюдаемым

Постройте синтетический loss, градиенты которого становятся меньше минимального
представимого нормального FP16-числа. Сравните три случая: прямой FP16 backward,
фиксированный scale и dynamic loss scaling. До optimizer step градиенты нужно
вернуть в исходный масштаб; clipping выполняется после `unscale`.

Добавьте проверки:

```python
assert all_finite(unscaled_gradients)
assert relative_parameter_error(fp32_step, amp_step) < tolerance
assert optimizer_step_was_skipped_when_overflow_detected
```

Порог ошибки задайте до запуска и объясните его относительно масштаба весов.

## Проверить реальное ускорение

Profiler trace должен показать dtype матричных операций и используемые kernels.
Если маленькая модель остаётся launch-bound или pipeline ждёт CPU, отсутствие
ускорения — корректный результат. Не выводите hardware throughput из одного
`dtype` тензора.

## Что сдать

Полную конфигурацию среды, raw timings, таблицу численной ошибки, memory
snapshots, trace каждого режима и короткое объяснение выбранной политики
scaling. Работа закончена, если повторный запуск воспроизводит и ускорение (либо
его отсутствие), и специально созданный underflow/overflow case.
