---
title: Измерить CUDA без самообмана
type: practice
status: canonical
last_updated: 2026-07-24
---

# Измерить CUDA без самообмана

Цель работы — получить измерение, которому можно доверять, и показать хотя бы
один случай, когда обычный Python timer даёт неверный вывод из-за асинхронного
исполнения CUDA.

## Исходный материал

Откройте [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/seminar.ipynb|оригинальный seminar notebook EDLS Week 1]] и держите рядом [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/lecture.pdf|полную лекцию]]. Не переписывайте notebook по этой странице: выполняйте исходные ячейки и сохраняйте результаты.

## Эксперимент

1. Выберите три операции: elementwise, reduction и GEMM.
2. Для каждой постройте sweep размеров, включая маленький launch-bound случай и
   размер, который устойчиво загружает GPU.
3. Сравните четыре метода: `time.perf_counter()` без синхронизации, тот же timer
   с `torch.cuda.synchronize()`, CUDA Events и `torch.utils.benchmark`.
4. Отделите первый запуск от steady state; зафиксируйте число warmup и repeats.
5. Проверьте результат операции, чтобы compiler или ошибка setup не сделали
   быстрое измерение бессмысленным.

## Что сдать

- таблицу raw samples и median/p95, а не одно лучшее число;
- версии GPU, driver, CUDA, PyTorch, dtype и shapes;
- график latency от размера;
- короткое объяснение границы каждого timer;
- один profiler trace, на котором видны CPU launch и CUDA kernel.

Работа считается законченной, если из артефактов можно независимо восстановить
эксперимент и объяснить расхождение timers. Следующий шаг —
[[02 Areas/ML & DL/06 Практика/07 Построить roofline и найти bottleneck|roofline]].
