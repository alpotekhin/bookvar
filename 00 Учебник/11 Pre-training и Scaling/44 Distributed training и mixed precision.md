---
title: Распределённое обучение — карта состояний и обменов
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44. Распределённое обучение: что реплицировать, делить, перемещать и сохранять

Распределённое обучение решает две разные задачи: разместить модель и все состояния обучения в памяти, а затем сократить время до результата. Число ускорителей само по себе не задаёт конфигурацию. Полезнее начать с пяти вопросов: какие состояния **реплицируются**, какие **разделяются**, какие тензоры **перемещаются**, какие активации **вычисляются повторно** и что необходимо **сохранить**, чтобы следующий шаг после сбоя совпал с непрерывным запуском.

## Бухгалтерская книга состояния

Для $P$ параметров Adam обычно хранит веса низкой точности, основную копию FP32, градиенты и два момента. В зависимости от реализаций это около $16$–$20P$ байт до активаций и временных буферов. Активации растут с размером микропакета, длиной последовательности и числом слоёв. Поэтому «модель весит 140 GB» ещё ничего не говорит о возможности её обучать.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/training-systems-courses/44-cs336-dp-memory.png]]

*Источник: Percy Liang, Tatsu Hashimoto и команда Stanford, CS336, лекция 8, слайд 17: [PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf).*

Пример: для $P=7\cdot10^9$ и условного ledger $2+4+4+8=18$ B/param состояния требуют около 126 GB. На восьми 80-GB GPU обычный DDP оставляет 126 GB **на каждом** устройстве и не помещается; полное разделение даёт нижнюю оценку $126/8=15{,}75$ GB на rank, но к ней добавятся собранные параметры текущего блока, активации и коммуникационные буферы.

## Оси параллелизма

Каждая ось отвечает на собственное ограничение, поэтому новая последовательность разбирает их отдельно:

- [[44a Processes, collectives и DDP]] — процессы, коллективные операции, цена синхронизации и сжатие градиентов.
- [[44b Gradient checkpointing и offload]] — память активаций, повторное вычисление и перенос состояния.
- [[44c Tensor и sequence parallelism]] — разбиение матриц, attention и длинной последовательности.
- [[44d Pipeline parallelism]] — стадии, микропакеты, GPipe, 1F1B и bubble.
- [[44e ZeRO, FSDP2, DeviceMesh и DTensor]] — время жизни шардов параметров, градиентов и optimizer state.
- [[44f Expert и hybrid parallelism]] — expert parallelism и согласование нескольких осей.
- [[44g Network, storage и distributed checkpoints]] — физический путь байтов от HBM до сети и хранилища.
- [[44h Fault tolerance и fleet orchestration]] — домены отказа, интервал checkpoint и размещение fleet.

DDP ускоряет при достаточной памяти; ZeRO/FSDP делят состояния; TP делит одну матрицу; PP — слои; sequence/context parallel — позиции; EP — экспертов. Размер мира должен согласовываться с произведением осей. Частые TP-обмены разумно оставлять внутри быстрого NVLink-домена, а межузловую сеть использовать для более редких DP/PP-обменов.

## Смешанная точность

FP16 ускоряет матричные операции, но имеет узкий диапазон порядков. Loss scaling умножает loss на $S$, выполняет backward, затем делит градиенты на $S$; clipping выполняют **после** unscale. При overflow шаг пропускают и уменьшают $S$. BF16 сохраняет диапазон FP32 и обычно не нуждается в scaling, но имеет короткую мантиссу. Редукции, нормировки и optimizer state часто оставляют в FP32. FP8 требует собственного масштабирования и проверки кривой обучения, а не включается как безусловная замена BF16.

Roofline объясняет, почему более быстрый dtype не обязательно ускоряет шаг:

$$I=\frac{F}{Q},\qquad P_{\max}=\min(P_{\mathrm{peak}},B_{\mathrm{mem}}I).$$

где $F$ — число операций, а $Q$ — число байтов, прочитанных из памяти.

BF16/FP8 повышают потолок tensor cores, но не ускоряют сеть, загрузчик данных или синхронную запись checkpoint. Поэтому профилируют вычисления, HBM, collectives, storage и простой по отдельности.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/training-systems-courses/44-cs336-parallelism-table.png]]

*Источник: Percy Liang, Tatsu Hashimoto и команда Stanford, CS336, лекция 8, слайд 55: [PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf).*

## Инвариант проверки

Сначала снимают single-GPU baseline на настоящем batch; затем включают BF16 и checkpointing; при нехватке памяти — FSDP/ZeRO, при слишком большой GEMM — TP, при большой глубине — PP. На каждом переходе сравнивают logits, loss, нормы и выбранные элементы градиентов с доверенным запуском. Системные метрики — useful tokens/s, MFU, peak memory, нескрытый communication, pipeline bubble, время записи и восстановления. Сильное масштабирование фиксирует задачу, слабое увеличивает глобальный batch вместе с числом устройств.

Типичные причины ложного успеха: разные числа batch у rank и зависание collective; TP через медленную межузловую связь; много padding при высокой номинальной throughput; clipping до unscale; checkpoint только локального shard без метаданных; изменившийся после restart порядок данных или RNG.

Контрольная точка — протокол, а не набор файлов. Она включает веса, optimizer и scheduler state, scaler, положение data loader, RNG каждого rank и topology-independent metadata. Проверка: восстановить её, выполнить следующий batch и сравнить IDs примеров, loss и обновление весов с непрерывным запуском.

## Источники

- Stanford CS336, [systems lecture 8](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf).
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740), 2018.
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).

← [[43 Scaling laws]] · Далее: [[44a Processes, collectives и DDP]]
