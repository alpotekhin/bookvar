---
title: Ускорить data pipeline
type: practice
status: canonical
last_updated: 2026-07-24
---

# Ускорить data pipeline

Полный исходник работы находится в [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/seminar/practice.ipynb|EDLS Week 2 seminar notebook]], а условия — в [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/homework/README|homework]]. Эта страница задаёт единый протокол сравнения.

Сначала измерьте baseline при `num_workers=0`, затем по одному включайте workers,
persistent workers, prefetch, pinned memory и asynchronous H2D. Для каждого
варианта разделите:

- время чтения и decode/preprocessing;
- ожидание следующего batch;
- H2D copy;
- GPU compute;
- end-to-end step time.

Во второй части соберите распределение длин и сравните static padding, dynamic
padding, bucketing и packing. Сообщайте не только samples/s, но и полезные
tokens/s:

$$\eta=\frac{\text{non-padding tokens}}{\text{processed token slots}}.$$

Для packing отдельно протестируйте attention mask, labels на границах документов
и position ids. Сдайте timeline baseline и лучшего варианта, utilization,
throughput, peak host/device memory и тест, доказывающий корректность границ.
