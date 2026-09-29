---
title: Stanford CS336 — Language Modeling from Scratch, Spring 2026
type: source-note
status: verified
last_verified: 2026-09-04
---

# Stanford CS336 — Language Modeling from Scratch, Spring 2026

[Официальная страница курса](https://cs336.stanford.edu/) ·
[[05 Источники/Курсы|Все курсы]]

Как обучить языковую модель с нуля и понять, на что уходят память, время
и вычисления? Курс Percy Liang и Tatsunori Hashimoto разбирает эту задачу
от токенизатора и устройства Transformer до подготовки корпуса,
распределённого обучения и проверки качества модели. Пять заданий
позволяют реализовать основные части этой системы самостоятельно.

Ниже — материалы Spring 2026 на английском языке: оригинальные лекции,
слайды и задания. Для каждой темы указаны соответствующие главы Bookvar
и лекции с оригинальным разбором.

## Где читать тему

| Тема | Главы Bookvar | Лекции курса |
|---|---|---|
| Токенизатор и реализация Transformer | [[00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece и Unigram]]; [[00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|Формы тензоров, параметры и стоимость]] | [1](#lecture-01), [2](#lecture-02), [3](#lecture-03) |
| Альтернативы attention и MoE | [[00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts]]; [[00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|Рекуррентные и гибридные модели]] | [4](#lecture-04) |
| GPU, память и быстрые ядра | [[00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU и иерархия памяти]]; [[00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|Ядра и Triton]] | [5](#lecture-05), [6](#lecture-06) |
| Распределённое обучение | [[00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision|Параллельное обучение]]; [[00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|Коллективные операции и DDP]] | [7](#lecture-07), [8](#lecture-08) |
| Размер модели, объём данных и бюджет обучения | [[00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Scaling laws]] | [9](#lecture-09), [11](#lecture-11) |
| Генерация и обслуживание запросов | [[00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|Prefill, decode и roofline]]; [[00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache и пакетирование]] | [10](#lecture-10) |
| Оценивание и подготовка данных | [[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей]]; [[00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Сбор, очистка и смеси данных]] | [12](#lecture-12), [13](#lecture-13), [14](#lecture-14) |
| Дообучение по примерам, предпочтениям и проверяемой награде | [[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT]]; [[00 Учебник/12 Post-training и Alignment/05 DPO|DPO]]; [[00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR]] | [15](#lecture-15), [16](#lecture-16) |
| Изображения и другие модальности | [[00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Мультимодальные модели]]; [[00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Обучение VLM]] | [17](#lecture-17) |

## 19 встреч курса

Открывайте название лекции для краткого описания и ссылок на материалы.
Видеозаписи доступны через [официальное расписание](https://cs336.stanford.edu/#schedule).
У двух гостевых встреч в сохранённой версии расписания нет ссылки на материалы.

| № | Дата | Тема | Преподаватель |
|---:|---|---|---|
| 1 | 2026-03-30 | [Overview, tokenization](#lecture-01) | Percy Liang |
| 2 | 2026-04-01 | [PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)](#lecture-02) | Percy Liang |
| 3 | 2026-04-06 | [Architectures, hyperparameters](#lecture-03) | Tatsunori Hashimoto |
| 4 | 2026-04-08 | [Attention alternatives and mixture of experts](#lecture-04) | Tatsunori Hashimoto |
| 5 | 2026-04-13 | [GPUs, TPUs](#lecture-05) | Tatsunori Hashimoto |
| 6 | 2026-04-15 | [Kernels, Triton](#lecture-06) | Percy Liang |
| 7 | 2026-04-20 | [Parallelism](#lecture-07) | Percy Liang |
| 8 | 2026-04-22 | [Parallelism](#lecture-08) | Tatsunori Hashimoto |
| 9 | 2026-04-27 | [Scaling laws](#lecture-09) | Tatsunori Hashimoto |
| 10 | 2026-04-29 | [Inference](#lecture-10) | Percy Liang |
| 11 | 2026-05-04 | [Scaling laws](#lecture-11) | Tatsunori Hashimoto |
| 12 | 2026-05-06 | [Evaluation](#lecture-12) | Percy Liang |
| 13 | 2026-05-11 | [Data (sources, datasets)](#lecture-13) | Percy Liang |
| 14 | 2026-05-13 | [Data (filtering, deduplication, mixing, synthetic data)](#lecture-14) | Percy Liang |
| 15 | 2026-05-18 | [Mid/post-training (SFT/RLHF)](#lecture-15) | Tatsunori Hashimoto |
| 16 | 2026-05-20 | [Post-training - RLVR](#lecture-16) | Tatsunori Hashimoto |
| 17 | 2026-05-27 | [Alignment - multimodality](#lecture-17) | Percy Liang |
| 18 | 2026-06-01 | Guest lecture: Daniel Selsam | Daniel Selsam |
| 19 | 2026-06-03 | Guest lecture: Dan Fu | Dan Fu |

## Лекции: что читать и смотреть

<a id="lecture-01-overview-tokenization"></a>
<a id="lecture-01"></a>
### 1. Overview, tokenization

От байтов и Unicode к словарю токенов: почему разбиение текста влияет на длину последовательности и стоимость модели. Сравниваются простые способы токенизации и BPE; обучение токенизатора отделено от применения готового словаря.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_01) · [сохранённый Python-файл](Lectures/repository/lecture_01.py)

<a id="lecture-02-pytorch-einops-resource-accounting-flops-memory-arithmetic-intensity"></a>
<a id="lecture-02"></a>
### 2. PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)

Формы тензоров, операции PyTorch и einops, градиенты и шаг оптимизатора. На этих операциях разбирается подсчёт параметров, FLOPs и памяти: важно учитывать не только веса, но и активации, градиенты и состояние оптимизатора.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_02) · [сохранённый Python-файл](Lectures/repository/lecture_02.py)

<a id="lecture-03-architectures-hyperparameters"></a>
<a id="lecture-03"></a>
### 3. Architectures, hyperparameters

Выбор компонентов Transformer: нормализация, позиционные представления, функции активации и устройство attention. Архитектурные решения рассматриваются вместе с гиперпараметрами обучения, а не как независимый список приёмов.

[Слайды PDF](Lectures/repository/lecture_03.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf)

<a id="lecture-04-attention-alternatives-and-mixture-of-experts"></a>
<a id="lecture-04"></a>
### 4. Attention alternatives and mixture of experts

Как уменьшить стоимость обработки длинных последовательностей и увеличить число параметров без пропорционального роста вычислений. Лекция сопоставляет альтернативы полному attention и разреженные MoE-модели с выбором экспертов для каждого токена.

[Слайды PDF](Lectures/repository/lecture_04.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_04.pdf)

<a id="lecture-05-gpus-tpus"></a>
<a id="lecture-05"></a>
### 5. GPUs, TPUs

Устройство GPU и TPU, матричные вычисления и движение данных между уровнями памяти. Пропускная способность памяти и вычислительная мощность ограничивают разные операции; это объясняет, почему число FLOPs само по себе не предсказывает время работы.

[Слайды PDF](Lectures/repository/lecture_05.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf)

<a id="lecture-06-kernels-triton"></a>
<a id="lecture-06"></a>
### 6. Kernels, Triton

От операции PyTorch к отдельному GPU-ядру: измерение времени, объединение операций и разбиение работы на блоки в Triton. Примеры связывают организацию вычислений с числом обращений к памяти и фактическим ускорением.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_06) · [сохранённый Python-файл](Lectures/repository/lecture_06.py)

<a id="lecture-07-parallelism"></a>
<a id="lecture-07"></a>
### 7. Parallelism

Процессы, обмен тензорами и коллективные операции, необходимые для обучения на нескольких GPU. Разбирается, какие данные нужно передавать между устройствами и как согласовать локальные вычисления с синхронизацией.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_07) · [сохранённый Python-файл](Lectures/repository/lecture_07.py)

<a id="lecture-08-parallelism"></a>
<a id="lecture-08"></a>
### 8. Parallelism

Способы распределить данные, параметры и слои модели между устройствами. Сравнение учитывает память, объём обменов и простой устройств: одного увеличения числа GPU недостаточно, чтобы обучение ускорилось пропорционально.

[Слайды PDF](Lectures/repository/lecture_08.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf)

<a id="lecture-09-scaling-laws"></a>
<a id="lecture-09"></a>
### 9. Scaling laws

Как по небольшим экспериментам оценить эффект увеличения модели и обучающего корпуса. Законы масштабирования связывают функцию потерь, число параметров, объём данных и вычислительный бюджет.

[Слайды PDF](Lectures/repository/lecture_09.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf)

<a id="lecture-10-inference"></a>
<a id="lecture-10"></a>
### 10. Inference

Чем обработка запроса отличается от последовательной генерации токенов и почему KV-cache меняет стоимость attention. Далее рассматриваются квантизация, сжатие, спекулятивное декодирование и обслуживание запросов разной длины.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_10) · [сохранённый Python-файл](Lectures/repository/lecture_10.py)

<a id="lecture-11-scaling-laws"></a>
<a id="lecture-11"></a>
### 11. Scaling laws

Продолжение темы масштабирования: выбор соотношения размера модели и числа обучающих токенов. Экстраполяция требует нескольких измерений и проверки предположений, особенно когда меняются данные или режим обучения.

[Слайды PDF](Lectures/repository/lecture_11.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf)

<a id="lecture-12-evaluation"></a>
<a id="lecture-12"></a>
### 12. Evaluation

Что измеряет тест языковой модели и когда его результат можно сравнивать с другими моделями. Обсуждаются наборы задач, метрики, загрязнение тестовых данных и ограничения оценок, полученных с помощью другой языковой модели.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_12) · [сохранённый Python-файл](Lectures/repository/lecture_12.py)

<a id="lecture-13-data-sources-datasets"></a>
<a id="lecture-13"></a>
### 13. Data (sources, datasets)

Откуда берутся данные для предобучения: веб, книги, научные тексты, код и специализированные наборы. Происхождение корпуса определяет его состав, доступность и ограничения использования.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_13) · [сохранённый Python-файл](Lectures/repository/lecture_13.py)

<a id="lecture-14-data-filtering-deduplication-mixing-synthetic-data"></a>
<a id="lecture-14"></a>
### 14. Data (filtering, deduplication, mixing, synthetic data)

Как превратить собранные документы в обучающий корпус: фильтрация, удаление повторов и выбор пропорций источников. Синтетические данные рассматриваются вместе с проверкой их качества, а не как автоматически полезное увеличение корпуса.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_14) · [сохранённый Python-файл](Lectures/repository/lecture_14.py)

<a id="lecture-15-midpost-training-sftrlhf"></a>
<a id="lecture-15"></a>
### 15. Mid/post-training (SFT/RLHF)

Как после предобучения научить модель выполнять инструкции и учитывать предпочтения. Обсуждаются обучающие примеры для SFT, сравнения ответов и оптимизация поведения модели после обучения модели награды.

[Слайды PDF](Lectures/repository/lecture_15.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_15.pdf)

<a id="lecture-16-post-training---rlvr"></a>
<a id="lecture-16"></a>
### 16. Post-training - RLVR

Обучение с наградой, которую можно вычислить по проверке результата, например ответа на математическую задачу. Разбирается связь между генерацией решений, проверяющей программой и обновлением модели в RLVR.

[Слайды PDF](Lectures/repository/lecture_16.pdf) · [оригинал в репозитории](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_16.pdf)

<a id="lecture-17-alignment---multimodality"></a>
<a id="lecture-17"></a>
### 17. Alignment - multimodality

Как связать текстовую модель с изображениями и другими модальностями. Разбираются представления входных данных, архитектурные способы их объединения и задачи обучения, согласующие разные представления.

[Оригинальная лекция с кодом и иллюстрациями](https://cs336.stanford.edu/lectures/?trace=lecture_17) · [сохранённый Python-файл](Lectures/repository/lecture_17.py)

<a id="lecture-18"></a>
### 18. Guest lecture: Daniel Selsam

В сохранённом расписании ссылка на материалы не опубликована.

<a id="lecture-19"></a>
### 19. Guest lecture: Dan Fu

В сохранённом расписании ссылка на материалы не опубликована.


## Assignments 1–5

<a id="assignment-01-assignment-1-basics"></a>
<a id="assignment-01"></a>
### Assignment 1: Basics

Реализовать BPE-токенизатор, Transformer и оптимизатор, затем обучить небольшую языковую модель. Задание связывает формулы с работающим кодом: готовую реализацию модели вместо собственной использовать не предполагается.

[Условия и запуск](Assignments/assignment1-basics/README.md) · [репозиторий задания](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712)

<a id="assignment-02-assignment-2-systems"></a>
<a id="assignment-02"></a>
### Assignment 2: Systems

Измерить узкие места модели из первого задания, реализовать FlashAttention-2 на Triton и распределённое обучение с экономией памяти. Результат проверяется не только корректностью, но и измерениями времени и памяти.

[Условия и запуск](Assignments/assignment2-systems/README.md) · [репозиторий задания](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736)

<a id="assignment-03-assignment-3-scaling"></a>
<a id="assignment-03"></a>
### Assignment 3: Scaling

Спланировать серию обучений, подобрать закон масштабирования и предсказать подходящий размер модели при заданном бюджете. В README отдельно описаны работа через учебный API и самостоятельный запуск для читателей вне Stanford.

[Условия и запуск](Assignments/assignment3-scaling/README.md) · [репозиторий задания](https://github.com/stanford-cs336/assignment3-scaling/tree/03e9372992e913061b9e78b5cfcb62ad8a87de35)

<a id="assignment-04-assignment-4-data"></a>
<a id="assignment-04"></a>
### Assignment 4: Data

Подготовить данные для предобучения из Common Crawl: извлечь текст, отфильтровать документы и удалить повторы. Эффект обработки корпуса проверяется по обученной модели, а не только по числу оставшихся документов.

[Условия и запуск](Assignments/assignment4-data/README.md) · [репозиторий задания](https://github.com/stanford-cs336/assignment4-data/tree/0555bea66369872d912652debf10b115ca0688c8)

<a id="assignment-05-assignment-5-alignment-and-reasoning-rl"></a>
<a id="assignment-05"></a>
### Assignment 5: Alignment and Reasoning RL

Дообучить модель на математических задачах с помощью SFT и обучения с подкреплением. Задание включает проверяемую награду и GRPO; в репозитории есть тесты, к которым нужно подключить собственную реализацию.

[Условия и запуск](Assignments/assignment5-alignment/README.md) · [репозиторий задания](https://github.com/stanford-cs336/assignment5-alignment/tree/c2734a26308710949fe13226960a1e8cece94b7e)

<a id="assignment-05-safety-supplement-assignment-5-optional-supplement-safety-instruction-tuning-and-rlhf"></a>
<a id="assignment-05-safety-supplement"></a>
### Assignment 5: optional safety supplement

Необязательное продолжение пятого задания: безопасность, выполнение инструкций и обучение по предпочтениям, в том числе DPO. Это дополнение к Assignment 5, а не отдельное шестое задание.

[Условия дополнения, PDF](Assignments/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

## Об оригиналах и заимствованиях

Тексты лекций и заданий сохранены на английском языке. Авторство и ссылки
на источники сохраняются при переносе материала в главы. Условия использования
кода, лекционных слайдов и рисунков из других публикаций могут различаться;
доступность файла в интернете сама по себе не устанавливает его лицензию.

<details>
<summary>Для редакторов: состав архива, версии и проверка переноса</summary>

## Реестры аудита

Архив материалов сохранён; тематическое дополнение и редактура глав продолжаются.
Технический статус: **inventory complete; editorial integration active**.
Наличие записи в реестре не заменяет проверку качества объяснения или рисунка.

- [source-manifest.yml](source-manifest.yml) — перечень оригинальных материалов;
- [semantic-review.json](semantic-review.json) — границы смысловых разделов;
- [editorial-map.yml](editorial-map.yml) — решения о переносе в главы;
- [source-units.yml](source-units.yml) и [coverage.yml](coverage.yml) — фрагменты источников и их назначение;
- [visuals.yml](visuals.yml) — иллюстрации и последовательности слайдов;
- [artifact-inventory.json](artifact-inventory.json) — контрольные суммы файлов;
- [snapshot-lock.json](snapshot-lock.json) — версии репозиториев и инструментов проверки.

### Сохранённые версии репозиториев

- `lectures`: [`8b59b50730766695c2ffedd1a79c50cd09b9eb91`](https://github.com/stanford-cs336/lectures/tree/8b59b50730766695c2ffedd1a79c50cd09b9eb91)
- `assignment1-basics`: [`a158843b20107949f1a8d7df1b05cd33b9166712`](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712)
- `assignment2-systems`: [`ca8bc81a59b70516f7ebb2da4808daade877c736`](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736)
- `assignment3-scaling`: [`03e9372992e913061b9e78b5cfcb62ad8a87de35`](https://github.com/stanford-cs336/assignment3-scaling/tree/03e9372992e913061b9e78b5cfcb62ad8a87de35)
- `assignment4-data`: [`0555bea66369872d912652debf10b115ca0688c8`](https://github.com/stanford-cs336/assignment4-data/tree/0555bea66369872d912652debf10b115ca0688c8)
- `assignment5-alignment`: [`c2734a26308710949fe13226960a1e8cece94b7e`](https://github.com/stanford-cs336/assignment5-alignment/tree/c2734a26308710949fe13226960a1e8cece94b7e)
- `stanford-cs336.github.io`: [`25d740fd9060cc6613163b5b88ca88a5f64138ff`](https://github.com/stanford-cs336/stanford-cs336.github.io/tree/25d740fd9060cc6613163b5b88ca88a5f64138ff)

### Права и атрибуция

Запись `rights_status: permission-recorded` фиксирует пользовательское
подтверждение образовательного переиспользования, а не название лицензии
правообладателя. Файлы LICENSE сохранены вместе с репозиториями. Условия
переноса сторонних рисунков проверяются отдельно для каждой иллюстрации.

</details>
