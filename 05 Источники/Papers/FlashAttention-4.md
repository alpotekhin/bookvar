---
title: "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design"
type: source-note
status: reviewed
source_type: paper
authors: [Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao]
published: 2026-03-05
last_verified: 2026-11-11
url: https://arxiv.org/abs/2603.05451v1
concepts: [FlashAttention, online softmax, Blackwell, tensor memory]
---

# FlashAttention-4

Для Blackwell авторы сравнивают время матричных операций, чтений из shared
memory и экспонент softmax. В упрощённом расчёте для блока 128 × 128 и
размерности головы 128 две матричные операции и все экспоненты требуют
по 1024 такта на своих устройствах. Ускорение одних Tensor Cores
не уменьшит время softmax. [§3.1.1 и таблица 1](https://arxiv.org/html/2603.05451v1#S3.SS1.SSS1).

В forward два блока запросов обслуживаются одним CTA. Пока Tensor Cores
считают произведение для одного блока, другая группа потоков вычисляет
softmax для другого. Аккумуляторы Blackwell хранятся в tensor memory.
Группа коррекции отдельно масштабирует промежуточный выход. Авторы также
переносят часть вычислений экспонент на FMA и не меняют точку отсчёта
online softmax, пока рост максимума не превысит порог.
[§3.1.2–3.1.4 и рисунок 1](https://arxiv.org/html/2603.05451v1#S3.SS1).

В backward две соседние группы потоков (2-CTA) могут выполнить общую
матричную операцию, сохраняя отдельные части аккумулятора. Для вклада в
градиент запросов они обмениваются частями градиента scores через
distributed shared memory. Это не устраняет необходимость объединять
вклады от разных блоков ключей. [§3.2.3, рисунок 3](https://arxiv.org/html/2603.05451v1#S3.SS2.SSS3).

[Рисунок 1](https://arxiv.org/html/2603.05451v1#S3.F1) доступен под
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Разбор конвейера и числовой пример условного масштабирования:
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention#fa4-blackwell|глава FlashAttention]].
