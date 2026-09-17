---
title: Реализовать W8A8 и SmoothQuant
type: practice
status: canonical
last_updated: 2026-09-15
---

# Реализовать W8A8 и SmoothQuant

Работайте по [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/seminar.ipynb|EDLS Week 9 seminar notebook]] и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/lecture.pdf|полной лекции]].

Реализуйте симметричное quantize/dequantize сначала для tensor-wise, затем для
channel-wise scales. Сравните FP baseline, weight-only INT8 и W8A8. Отдельно
измерьте ошибку матричного произведения и end-to-end качество модели.

До подбора масштабов разделите данные на калибровочную, валидационную и
тестовую части. Для SmoothQuant соберите статистику каналов активаций только
по калибровочной части, перенесите часть масштаба из активаций в веса и
сравните несколько значений $\alpha$ на валидации. После выбора $\alpha$
зафиксируйте масштабы и оцените качество на независимом тесте. Не подбирайте
параметры по тестовым ошибкам; сохраните идентификаторы и размеры всех трёх
частей. Обоснование преобразования — в
[исходной работе SmoothQuant](https://arxiv.org/abs/2211.10438).
Отчёт должен показать:

- распределение activation outliers до/после;
- reconstruction error по слоям;
- качество на фиксированном eval;
- latency, throughput и peak memory;
- использовался ли реальный INT8 kernel или только fake quantization.

Последний пункт проверяется profiler trace: меньший dtype tensor ещё не
доказывает ускорение.
