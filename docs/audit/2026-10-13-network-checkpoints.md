---
type: editorial
status: editorial
last_verified: 2026-10-13
---

# Редакционная проверка 44g: сеть, хранение и распределённые checkpoints

## Дополнение основного редактора, 2026-10-14

Запись ниже фиксирует первоначальную работу субагента. После полного
прочтения главы и просмотра страницы основной редактор заменил два рисунка:
`gpudirect-data-path.svg` и `complete-data-path.svg`. Числа «700 GB
eliminated» и «0.3 s» создавали неверное впечатление даже с поясняющими
подписями. Исходные SVG сохранены как исторические активы, но удалены
из русской 44g; центральный и прежний локальный реестры согласованы.

Вместо них встроены неизменённые оригиналы
`network-checkpoint-review-2026-10/openhackathons-gpudirect-rdma.png`
(OpenACC / NVIDIA, CC BY 4.0) и
`network-checkpoint-review-2026-10/pytorch-async-checkpoint-flow.png`
(PyTorch, BSD-3-Clause). Второй расположен непосредственно в разделе
асинхронной записи. Полные источники, commit и hashes — в соседнем NOTICE
и центральном registry. Оба изображения просмотрены основным редактором.

Также уточнена устойчивость детерминированной очереди:
при C/Δ = bandwidth очередь ограничена, но не имеет запаса; строгий
запас требует C/Δ < bandwidth. Checker дополнен проверкой равенства.
Основной редактор повторно выполнил его 2026-10-14: PASS, включая
настоящие CPU DCP save/load, удержание фонового писателя и 2→3 Gloo
resharding. Последующие изменения касались только текста и иллюстраций,
Python-блоки не менялись.

Итоговые hash, сборка и UI-проверка записаны отдельно в
`2026-10-14-distributed-detail-44e-44h-acceptance.md`.

## Объём работы

Каноническая глава:
`00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md`.

Исходная глава прочитана полностью. Вместо нескольких телеграфных абзацев
получилось последовательное объяснение физического пути байтов, цены
коллективной операции, размещения групп параллелизма и сохранения состояния.
Сохранены путь, заголовок, прежние заголовки разделов (включая «Расчёт»),
ссылки на соседние главы и все три оригинальные иллюстрации.
В исходнике не было `source_unit_id` и явных HTML anchors.

Область правок ограничена русской главой, её checker и этим audit.
EN, общие реестры, navigation, 44h и исходники курсов не изменены.
Сборка сайта, UI и публикация остаются у основного агента; этот отчёт
не утверждает, что они выполнены.

## Содержательные исправления

- RDMA объяснён через регистрацию области памяти, права, ключи и время жизни
  буфера. Отдельно названы путь данных и управляющие действия CPU.
- GPUDirect не приписывается устранение сетевой передачи или всей работы CPU.
  Разобраны PCIe/NUMA-топология, совместимость адресного отображения и порядок
  GPU/RDMA-операций.
- InfiniBand lossless ограничен предотвращением потерь от переполнения
  при нормальной работе, а не обещанием отсутствия аппаратных ошибок.
  PFC и ECN объяснены причинно; lossy RoCE с ECN подтверждён документацией.
- Различаются биты и байты, decimal GB/TB и binary KiB/MiB, односторонняя
  полоса и полнодуплексная передача. Ring volume — отправка одного rank,
  приём и сумма логических отправок; физические переходы учитываются отдельно.
- Иерархический all-reduce описан как local reduce-scatter →
  cross-node fragment all-reduce → local all-gather; восемь потоков
  не объявлены восемью NIC.
- Объяснена конкуренция ввода данных и checkpoint за PCIe, DRAM, NVMe,
  сеть и метаданные. DDP не обязан записывать все одинаковые реплики.
- Согласованность snapshot отделена от staging, завершения записи
  и пригодности после выбранного отказа.
- Добавлены полный состав training state, проблема alias в state_dict,
  граница optimizer step, локальные RNG/data cursors, ограничения
  mid-accumulation и prefetch.
- DCP разделён на state_dict adapters, planner и backend writer/reader.
  Указаны destination allocation, canonical FQN, resharding, ограничения
  архитектурной миграции, ключей участников и совместимости версий.
- Для async показаны два события завершения и backpressure. Объём очереди
  не отождествляется с объёмом целиком удерживаемых staging-буферов.
- Удалены подразумеваемые «встроенные» обязательные SHA-256, атомарный
  манифест всего поколения и универсальная durability.
  Поведение FileSystemWriter сверено по исходнику 2.14.1.
- Неизменяемые поколения и latest описаны как дополнительный протокол
  приложения. Для объекта S3 отличены атомарность одного ключа,
  условная запись указателя и отсутствие многоключевой транзакции.
- Восстановление проверяется следующим шагом, а не одним чтением весов.
  Модель отказов, Young/Daly и управление перезапуском остаются в 44h.

## Первичные источники и границы использования

Проверено 2026-10-13. Иллюстрации существующие; новый текст — связное
редакционное объяснение с вычисляемыми примерами, не перепечатка документов.

| Источник | Проверенные части и использование |
|---|---|
| [Harvard Network Fabrics, pinned 45ecc8d](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd) | RDMA/GPUDirect; alpha/bandwidth; topology/Clos/rails; PFC/ECN; наблюдаемость. Категоричные заявления курса о необходимости PFC и полном bypass CPU не перенесены |
| [Harvard Collective Communication, тот же commit](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd) | Ring scatter-reduce/all-gather, alpha term, логический объём; hierarchical all-reduce. Нет универсального обещания восьмикратного ускорения |
| [Harvard Data Storage, тот же commit](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/data_storage/data_storage.qmd) | Иерархия хранения, complete data path, checkpoint storage/staging, distributed coordination. Не перенесены смешение snapshot/durable completion и предположение о бесплатной DRAM-копии GPU-состояния |
| [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) | Registration/pinning; Supported Systems; Synchronization and Memory Ordering. Просмотренная версия документации 13.4 |
| [NVIDIA DOCA 3.5: RDMA Verbs](https://networking-docs.nvidia.com/doca/archive/3-5-0/doca-rdma-verbs) | Control/data path; MR, access permissions, local/remote keys |
| [Cumulus Linux 5.12 RoCE](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-512/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/) | Lossless PFC/ECN и lossy ECN; отсутствие универсальной необходимости PFC |
| [NCCL environment](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-algo) и [collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | Просмотрена документация 2.32.3: выбор алгоритмов по topology/architecture; CROSS_NIC/rails; определения коллективов |
| [DCP API 2.14](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) | save/load, same keys, destination allocation, stateful, FQN/get/set, version compatibility warning |
| [Module.state_dict 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict) | Shallow state dictionary / tensor references |
| [Async recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html) | CPU/pinned-memory footprint, one in-flight save, staging/upload futures |
| [PyTorch state_dict_saver.py, pinned 2.14.1](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/checkpoint/state_dict_saver.py#L299-L388) | **code-derived**: default blocking staging; Future vs AsyncSaveResponse; required CPU backend; lifecycle |
| [PyTorch staging.py, тот же commit](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/checkpoint/staging.py) | **code-derived**: DefaultStager и независимая копия, options и состояние буферов |
| [PyTorch filesystem.py, тот же commit](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/checkpoint/filesystem.py#L762-L802) | **code-derived**: flush/fsync, metadata.tmp, удаление прежней metadata перед rename; из кода не выводится whole-generation atomicity или обязательная checksum |
| [PyTorch воспроизводимость](https://docs.pytorch.org/docs/2.14/notes/randomness.html) | RNG coverage и ограничения побитовой повторяемости между конфигурациями |
| [Python os.replace](https://docs.python.org/3/library/os.html#os.replace) | Атомарность успешной локальной замены имени при поддерживаемых условиях; не доказательство power-failure durability |
| [S3 consistency](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel), [conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) | Atomic single-key updates / no atomic updates across keys; If-Match и If-None-Match |

## Иллюстрации и registry

Новых активов нет. Удалённых embeds нет. `used_in` существующих трёх
иллюстраций остаётся прежним; central registry менять для этой главы не требуется.

Все три SVG лично просмотрены через растровые preview Quick Look размером
1600 px; исходные SVG не изменялись и не кадрировались. Preview существуют
только в `/tmp` и не являются новыми активами книги.

| Существующий asset (общий каталог curated/ml-systems/harvard/distributed) | SHA-256 |
|---|---|
| five-level-model.svg | `6c3e1d7a7ed4bfe8e1d483b65300f83ee8b107a44424131682bc40ce0d8fe3c3` |
| gpudirect-data-path.svg | `484acf2bf4cbf17b1014d1771763045c06d51be3812c1a34bcbb8e4c756e43bf` |
| complete-data-path.svg | `b5aa33123e7e9312c8bbc82c246dde2dc79466d89fda6ebc9f154153c991e64a` |

Авторство сохранено: Vijay Janapa Reddi и соавторы / Harvard Edge ML Systems Book;
CC BY-NC-SA 4.0, pinned commit 45ecc8d82fcae70c149cdce550d3b3d3411df913.
Ссылки на точные QMD источники стоят под рисунками.
Фраза «700 GB eliminated» в оригинальной GPUDirect-схеме ограничена оценкой
дополнительных копий; не названа устранённым сетевым трафиком.
Числа скоростей и 0.3 s на complete-data-path явно названы условиями
иллюстрации, а не характеристиками любого хранилища.

## Воспроизводимая проверка

```sh
/opt/homebrew/bin/uv run --with torch==2.14.1 \
  --python /Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  python publishing/tools/check_network_checkpoint_examples.py
```

Checker извлекает и исполняет пять помеченных Python-блоков самой главы:
network-arithmetic, state-dict-alias, dcp-training-state, dcp-async-save,
checkpoint-queue. Он не подменяет их отдельной удобной демонстрацией.

Наблюдавшийся результат: **PASS**, macOS, Python 3.12, torch 2.14.1,
`torch.cuda.is_available() == False`. В тестах один CPU thread на процесс.
SHA-256 главы в последнем успешном запуске:
`06f3c1dfd22b79a6c79663de09003225db8498654227a32b8f76aae14dc91d64`.
SHA-256 checker:
`bf8c099dc737fc3d18e2c4e2cee14844c8814fb9e4e6e83b84d9ee8c416bdb07`.
Предупреждения DCP о неинициализированном distributed ожидаемы для
`no_dist=True` частей; отдельный resharding действительно создаёт Gloo-группы.

### Арифметика

| Величина | Проверенное значение |
|---|---|
| 400 Gb/s | 50,000,000,000 bytes/s в одном направлении |
| BF16 градиент 175B | 350,000,000,000 bytes |
| Ring 1024, отправка одного rank | 699,316,406,250 bytes |
| Только bandwidth | 13.986328125 s |
| С 2046 задержками по 2 мкс | 13.990420125 s |
| Сообщения 4 KiB / 4 MiB | 2.08192 / 85.88608 мкс |
| Сумма логических отправок | 716.1 TB |
| Checkpoint 2 TB при 100 / 25 GB/s | 20 / 80 s |
| Очередь после поступлений раз в 60 s | 2.0, 2.5, 3.0, 3.5, 4.0 TB |
| Два полных staging-буфера | 4 TB; при равном размещении на 64 узлах 62.5 GB/узел |
| Oversubscription 4×400 / 2×400 | 2:1; 200 Gb/s на сервер при равном внешнем обмене |
| Пересечения старых 6+6 и новых 4+4+4 частей | [4,0], [2,2], [0,4] элементов |

Для floating arithmetic: `rel_tol=1e-12, abs_tol=1e-12`.
Целые очереди и пересечения сравниваются точно.

### Настоящий CPU DCP, один процесс

После двух шагов Sequential Linear/Tanh/Dropout/Linear с Adam и ExponentialLR
сохраняется TrainingState. Третий шаг сравнивается с новым объектом, загруженным
через DCP. Loss: **0.11216250307017205** с обеих сторон, абсолютная ошибка **0**.
Модель, optimizer moments/groups/steps, scheduler, RNG, step и cursor
сравниваются рекурсивно с `rtol=atol=0`.

Четыре отрицательных контроля обнаружены:
изменённый RNG; заново созданный Adam без моментов; неверный cursor;
неверный scheduler.last_epoch. Подмена RNG и cursor меняет следующий loss,
Adam меняет обновление; проверка scheduler включает его состояние,
а не только совпадение одного обновления ExponentialLR.

Для async оригинальный метод FileSystemWriter.write_data временно
задержан threading.Event, пока живая модель делает ещё один шаг.
После снятия задержки выполняется настоящий писатель DCP; загруженное
состояние **точно совпадает со staged snapshot**, не с изменённым live state.
Это функциональная проверка независимости копии, не замер overlap.

### Настоящий CPU/Gloo resharding

Два отдельных процесса сохраняют DTensor[12] с Shard(0), по шесть чисел.
Затем новая группа из трёх процессов загружает по четыре числа:
[0,4), [4,8), [8,12). Проверены локальные части каждого rank и full_tensor.
Тот же checkpoint затем загружен в обычный полный CPU tensor одним процессом.
Все сравнения точные. Это **реальная многопроцессная DCP/Gloo-проверка**
на одной машине, не mock и не подтверждение GPU/FSDP/TP optimizer resharding.

### Локальный протокол поколений (отдельно от DCP)

Временный каталог, один писатель, fsync файлов/каталогов, immutable generation,
JSON manifest с длиной и SHA-256, os.replace указателя latest.
Незавершённое поколение и полное, но ещё не опубликованное поколение
не заменяют прежнее latest. Missing manifest, missing file, изменённая длина
и изменение байтов с сохранением длины отклоняются; прежний latest сохраняется.

Обрыв моделируется подготовкой частичных состояний файлов и отказом
до публикации; SIGKILL, отключение питания и реальный отказ носителя
не выполнялись. Это тест протокола приложения, не утверждение
о встроенных гарантиях DCP.

## Непроверенное и передача основному агенту

Не измерены NIC/RDMA/NCCL/GPU, ECN/PFC под нагрузкой, CUDA staging/overlap,
remote shared filesystem, отказ питания, долговечность и репликация хранилища,
объектные API и multi-writer publication, FSDP/TP optimizer resharding.
Числа сети и TB/s в главе — условия расчёта, не результаты бенчмарка.

`git diff --check` scoped chapter/checker: PASS.
Сайт не собирался, UI страницы не открывался этим subagent,
commit/push/deploy не выполнялись. Основной агент читает главу целиком
и проверяет итоговую страницу в общей сборке.
