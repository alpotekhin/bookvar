---
title: Аудит 44b — checkpointing и offload
type: editorial
status: editorial
last_verified: 2026-10-13
---

# Границы работы

Каноническая русская глава:
`00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload.md`.

Прочитана целиком исходная версия и проверено связное изложение новой.
Источник сохраняется в Markdown; EN, общий registry, navigation, чужие главы,
общие tests, сборка и публикация не менялись этим агентом.
Регистрацию изображений, общую сборку и UI-проверку выполняет основной агент.
Самостоятельных commit/push/deploy не было.

Сохранены четыре source_unit_id, явный `cs336-distributed-checkpointing`,
существующие основные заголовки и ссылки предыдущей/следующей главы.
Для переименованных заголовков добавлены совместимые anchors:
`worked-configuration` и `offload--другая-граница`.

## Содержательная переработка

1. Сохранения backward объяснены через $dX=GW^T$, $dW=X^TG$, а не через
   набор названий тензоров. Отдельно объяснено, почему замороженный вес
   нужен для градиента по входу.
2. Hooks показывают события сохранения, а не автоматически producer/lifetime.
   Пример `x*x` отличает 64 байта суммарных событий от 32 байт общего storage.
   Отмечены повторное использование адресов и отсутствие рабочих буферов
   в таком списке.
3. Исправлена неполная оценка сегментов: 2 GiB границ не равны пику.
   Добавлены память восстановленного сегмента, временные буферы,
   явная модель $Lb/k+ka+M_{\rm temp}$ и допущения для правила $\sqrt L$.
4. Non-reentrant/early-stop объясняются через необходимость безопасного
   повторного вычисления. Указаны ограничения побочных эффектов,
   checkpoint_sequential и старого совета requires_grad на входе.
5. Dropout имеет контрпример: одинаковый исходный loss не гарантирует
   одинакового градиента. Уточнены границы RNG preservation и metadata-only
   determinism check.
6. SAC показан на настоящем `sin(mm(x,w))`: 3/4/3 матричных умножения
   за forward+backward, соответствующие сохранению/полной рекомпутации/SAC.
   Список mm-операций не выдается за универсальную политику Transformer.
7. Удалена универсальная таблица «LayerNorm сохраняет это, QKV пересчитывается
   так»: набор сохранений зависит от kernels/backend. Сохранён полезный
   разбор FlashAttention, MHA/GQA shapes и отсутствия полного $S^2$.
8. Пример CS336 с тремя сигмоидами сохранён как объяснение и таблица 8→5
   обращений к тензорам. Fusion, HBM traffic и Python checkpointing разведены.
9. Разделены activation/parameter/optimizer offload и recovery checkpoint.
   save_on_cpu не переносит вычисления слоя и не освобождает GPU-параметр,
   пока на него остаётся ссылка.
10. Передача рассчитана в согласованных GiB/GB; учтены оба направления,
    ограниченное окно перекрытия и CPU-память.
11. В пример двухбуферного расписания включены прямой проход, backpressure,
    незавершённая D2H, пиннинг, готовность буфера и ограничения prefetch.
    Числа 3 GiB CPU и 2.75 GiB GPU не смешиваются.
12. Для optimizer/parameter offload объяснены CPU bandwidth, NUMA, общая связь
    нескольких GPU, повторная загрузка весов и отдельная ступень NVMe.
    Отсутствуют измеренные GPU speedup claims.

## Первичные источники и проверенные фрагменты

- EDLS, commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`:
  [week04 lecture, pp.11–22](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/lecture.pdf)
  и [practice_part1](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/practice_part1.ipynb).
  Прочитаны механизмы checkpointing, offload и код/объяснение frozen weights.
  Старые API-советы не скопированы без проверки.
- CS336 lectures, commit `8b59b50730766695c2ffedd1a79c50cd09b9eb91`:
  [Lecture5, pp.34–36](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf#page=34)
  и [Lecture8](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf).
  Проверены memory traffic, активации вне ZeRO-state savings, оговорки kernels.
- CS336 Assignment2, commit `ca8bc81a59b70516f7ebb2da4808daade877c736`:
  [memory profiling, saved tensors, gradient checkpointing](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf).
  Пример 3651.31 MiB сохранений и вход 80 MiB отнесены именно к указанной
  реализации/формам/dtype, не к замеренному пику arbitrary XL-модели.
- [PyTorch2.14 checkpoint docs](https://docs.pytorch.org/docs/2.14/checkpoint.html),
  [autograd docs](https://docs.pytorch.org/docs/2.14/autograd.html),
  [saved hooks tutorial](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html),
  [pinning/nonblocking tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
  Сверены семантика non-reentrant, RNG, early_stop, nested contexts и асинхронная
  D2H в текущем save_on_cpu.
- **Code-derived + runtime-verified:** PyTorch2.14.1 исходник
  `5c4886908584029761b579af026dcfb627c84070` совпал с `torch.version.git_version`.
  [checkpoint.py, respect_saved_tensors_hooks, L554–568](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/utils/checkpoint.py#L554-L568).
  [SAC cache consumption, L1680–1695](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/utils/checkpoint.py#L1680-L1695).
  Нельзя утверждать, что эти параметры найдены на любой stable HTML-странице:
  точные условия взяты из version-pinned source и проверены исполнением.
- [Chen et al., 2016](https://arxiv.org/abs/1604.06174),
  [FlashAttention, §3.2–3.3](https://arxiv.org/abs/2205.14135).
  Использованы механизмы и явно условная арифметика, не перенесённые benchmark results.
- DeepSpeed [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/),
  [ZeRO-3](https://www.deepspeed.ai/2021/03/07/zero3-offload.html),
  [pinned offload++ article](https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/blogs/deepspeed-offloadpp/README.md).
  Для иллюстрации взята только исходная схема CPUAdam/FusedAdam, не цифры ускорения.

## Визуальный разбор

Лично просмотрены оба старых полных слайда CS33635/36; кандидаты EDLS checkpoint
pp.12–15 и offload pp.17–20; два новых PDF-кропа; два исходных рисунка PyTorch;
исходный PNG DeepSpeed. Основной агент также посмотрел кандидаты и дал замечания.

Активных изображений теперь три:

1. `curated/arithmetic-review-2026-09/pytorch-checkpoint.png`:
   обычные saved tensors и граница AC, BSD-3-Clause. Оригинал уже существует,
   без изменения байтов.
2. `curated/arithmetic-review-2026-09/pytorch-selective-checkpoint.png`:
   добавочное сохранение одной операции, BSD-3-Clause. Оригинал уже существует.
3. `curated/checkpoint-offload-review-2026-10/deepspeed-optimizer-offload.png`:
   546×364, Apache-2.0, точная upstream-копия. Native-resolution labels
   читаемы; файл не увеличивался искусственно. UI-размер проверяет основной агент.

Полные `stanford-cs336-2026/systems/l05-p35.png` и `l05-p36.png` удалены
только из embeds главы, физически сохранены. В p35 усечённая линия справа есть
уже внутри исходного рисунка. PDF-кроп не исправил её и выявил ограниченную
чёткость встроенной графики, поэтому оба новых кропа отвергнуты.
В Assets их больше нет; только временные копии в /tmp.
Смысл восьми и пяти обращений восстановлен текстом/таблицей с прямой ссылкой.

Новая схема DeepSpeed не имеет иной лицензии в своём каталоге; корневой
Apache-2.0 LICENSE и THIRD_PARTY_NOTICES.md проверены на pinned commit.
Полная лицензия лежит рядом, атрибуция сохранена.
Лицензия EDLS assignments не переносилась на сторонние рисунки лекций.
Старый user-permission record у CS336 не объявлен разрешением правообладателя.

Машинно-читаемый handoff для основного агента:
`00 Учебник/Assets/Figures/curated/checkpoint-offload-review-2026-10/assets.json`.
В нём `new_assets[1]` и `used_in_changes[4]`. Общий registry агентом не менялся.
Подробное происхождение и SHA — в соседнем NOTICE.md; полная Apache-лицензия —
DeepSpeed-LICENSE.txt.

## Воспроизводимая проверка

`publishing/tools/check_checkpoint_offload_examples.py` читает и исполняет
**шесть actual Python-блоков из Markdown**, а не их отдельные копии.
Таблицы, округлённые числа и указанные результаты также читаются из главы.

Команда в текущем окружении:

```sh
/opt/homebrew/bin/uv run --with torch --python /Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 python publishing/tools/check_checkpoint_offload_examples.py
```

Результат: **PASS**, PyTorch 2.14.1, CPU.

| Проверка | Наблюдение |
|---|---|
| Saved tensors | 2 события, 64 logical bytes, 32 unique storage bytes |
| Градиент `sum(x*x)` | `[2,4,6,8]` |
| Таблица сегментов | Все 6 строк совпали; минимум среди делителей48 при k=6 и8, 3.75 GiB |
| Недопустимые k | 7 значений отклонены:0,-1,5,49,1.5,True,None |
| CS336 input | 4×2048×2560×4 bytes =80 MiB |
| Offload accounting | 3.00 GiB CPU, 2.75 GiB GPU при заданных буферах |
| Передача16GiB | 0.7158278827s в одну сторону |
| Ожидание при окне0.4s | 0.3158278827s |
| Два направления16GiB | 1.4316557653s без перекрытия |
| Два направления2GiB границ | 0.1789569707s без перекрытия |
| Eager/plain/SAC | 3/4/3 вызова aten.mm за forward+backward |
| Loss/gradients plain и SAC | max error0 относительно eager; atol=rtol=1e-12 |
| Dropout RNG preserved | loss error0, grad error0 |
| Dropout RNG not preserved | loss error0, grad max error8.0 |
| save_on_cpu, CPU input | Градиент совпал; передачи GPU не было |
| Frozen W | dX совпал с GWᵀ; dW не нужен |
| Non-reentrant, input no-grad | Обучаемый closure-weight получает правильный gradient через autograd.grad |
| Early stop | Трасса entry/exit/entry против entry/exit/entry/exit без early-stop |
| respect_saved_tensors_hooks | Кэшированный mm-output(3,5) виден hook только при True |

Проверка производных не ограничена совпадением двух autograd-веток.
Для обычного AC выполнен `torch.autograd.gradcheck`.
Для SAC проведены центральные конечные разности по **всем 32 координатам X и W**
со свежими forward-графами для каждого возмущения:
eps=1e-6, atol=1e-5, rtol=1e-3;
наблюдаемый максимум ошибки `4.434184130985841e-10`.

Первый запуск стандартного gradcheck на SAC выявил реальное ограничение:
он повторно использует один граф для нескольких backward, а SAC-cache
в PyTorch 2.14.1 потребляется при первом обходе. Вместо ослабления теста
сделаны независимые fresh-forward finite differences; оговорка внесена
в учебную главу. Диагностический счётчик early-stop используется только
в проверке, не как пример допустимого побочного эффекта модели.

Допуски для печатных таблиц: половина последнего десятичного разряда
(0.005 GiB для двух знаков; 0.00005s для четырёх).
Арифметика, которая выражается точно, проверяется до 1e-12.

`git diff --check` для принадлежащих агенту файлов прошёл.
Изображения реально существуют и проверены отдельно от сборки.
Материальная проверка layout, ссылок публикации и лицензий на public-surface
остаётся за основным агентом. CUDA, pinned transfer throughput, overlap,
GPU allocator peaks и обучение Transformer в этом проходе **не запускались**.
