# S6–S8: редакционная приёмка, 2026-10-08

## Область

Русские главы S6 (data pipeline), S7 (profiling), S8 (Triton), их иллюстрации
и проверочные примеры. Основной редактор прочитал главы целиком, проверил
результаты двух исполнителей, лично осмотрел изображения и страницы.
English, остальные главы, навигация и CSS не изменены. Исходные курсы не изменены.
Публикация, commit и push не выполнялись. Предсуществующие изменения
`AGENTS.md` и `.agents/` оставлены без вмешательства.

## Что изменилось

- S6: хранение и загрузка данных; очередь DataLoader и workers; pinned memory,
  передача и условия перекрытия; padding/bucketing с расчётами; упаковка,
  междокументная маска, границы labels; нормировка по токенам; шардирование,
  возобновление и измерение конвейера.
- S7: критический путь; исполняемый CPU/CUDA-aware пример Profiler; таблица
  операторов и trace; workers; NVTX и ограничение окна Nsight; GeLU и fusion;
  память аллокатора; выбор аппаратных счётчиков; проверка причинной гипотезы.
- S8: полный vector-add и softmax с host wrappers; адресация, маски и strides;
  длинные редукции; потоковая нормировка; блочное GEMM и reuse; точность,
  backward, autotuning и корректное измерение. GPU-запуск не заявлен.

Сейчас в главах 5 + 4 + 8 = 17 иллюстраций: оригинальные PNG или ранее
сохранённые страницы/фрагменты курсов. Добавлены 7 оригинальных PNG PyTorch
без обрезки и перерисовки. Удалены вставки нарисованных GeLU-таблиц и
неподходящих обобщённых SVG; сами старые файлы сохранены. Центральный
реестр отражает текущие вставки, прежние отмечены как historical_used_in.
В локальном legacy manifest исторические used_in снабжены review_note.

## Источники и происхождение S6

- EDLS week 2: закреплённая версия `e632aa89ca9e6638d52e1b686095e7442faffbb0`,
  лекция pp. 26–32 и batching seminar. В подписи и обоих реестрах исправлены
  автор Max Ryabinin и MIT. Оригинальная страница 31 лично сравнена с PNG.
  Правая часть рисунка показывает padded batches, а не packed attention.
- PyTorch Tutorials: `c80a43460c838fa28efa55e5680f3e97781e288e`,
  Vincent Moens, pinmem/nonblocking; Angel Li, variable-length attention.
  Четыре оригинальных файла, hashes, ссылки и BSD-3-Clause записаны в
  `05 Источники/asset-registry.yml` и NOTICE рядом с изображениями.
- PyTorch DataLoader и Hugging Face streaming documentation использованы
  для семантики workers/prefetch и ограниченного перемешивания.
- Источники S7/S8 и индивидуальные проверки описаны в соседних отчётах
  `2026-10-08-profiling.md` и `2026-10-08-triton.md`.

Полные notices сохранены вместе с ассетами. PyTorch Tutorials и EDLS notices
уже были в `site/public/licenses/`; дополнительно добавлен точный
`pytorch-v2.8.0-LICENSE.txt` для двух снимков аллокатора, SHA-256
`a8a2c6b67cf8a597ffaa5c82380a6bd1b82629c0417e00673d8013d662eb500a`.

## Исполненная проверка

Основной редактор самостоятельно выполнил:

- `publishing/tools/check_data_pipeline_examples.py`: настоящий EOS при
  PAD==EOS; shifted targets; расчёты padding/bucketing; CPU SDPA separate
  против packed; изоляция документов и отрицательный causal-only пример;
  границы labels; градиент при нормировке по числу токенов.
- `publishing/tools/check_profiling_examples.py`: AST 5 блоков; исполнение
  первого блока непосредственно из главы; JSON с 767 событиями и тремя
  forward/backward/optimizer ranges; GeLU outputs/gradients; cleanup на mock.
- `publishing/tools/check_triton_examples.py`: AST 3 блоков; source IDs;
  1026 вариантов индексов; 24 softmax, 144 потоковые нормировки, 4 GEMM,
  4 сравнения backward с autograd; расчёты интенсивности и количества блоков.

Среда примеров: macOS arm64, PyTorch 2.14.1 через отдельный uv environment,
CUDA unavailable. Скрипты запускаются из любой cwd, вычисляя корень по __file__.
Triton-check проверяет CPU-модели алгоритмов, а не исполняет Triton-код.
Profiler trace основного редактора:
`/var/folders/6c/j5sghwf51gx41g_055zlcs_m0000gq/T/s7-profiling-cpu-yd4ublbq/artifacts/profiler/step-5.json`.

Проверки проекта: 227 publisher unit tests, 64 site tests, 12 output tests —
pass. Astro check: 0 errors/warnings/hints. Build: 1961 pages.
Built link check: 0 broken internal routes, fragments or files.
`git diff --check`: pass.

## Визуальная проверка

Локальный preview: `http://127.0.0.1:4330/`.
Три страницы лично проверены в IAB при desktop 1280×960 и узкой ширине
390×844, дополнительно 512 px. Все 17 изображений загружены и сохраняют
исходные пропорции, выходов изображений за viewport и горизонтального
переполнения страницы нет; KaTeX errors отсутствуют.
Просмотрены текст, формулы, таблица bucketing, pinned/packing, trace,
Nsight, снимки памяти, softmax и grouped GEMM; щелчок по широкой временной
шкале действительно открывает оригинальный PNG. Для чтения мелких подписей
trace и слайдов требуется открытие оригинала; разрешение исходника не улучшалось
искусственно. Временная настройка viewport снята.

Первый снимок при смене viewport на 390 px был сделан до завершения изменения
размера: reported client width оказался 559. Он не засчитан. Повторный
осмотр подтвердил clientWidth = innerWidth = scrollWidth = 390.

## Границы готовности

Это локально проверенный редакционный блок, не новая опубликованная версия.
CUDA kernels, Nsight/NVTX, реальный CUDA Memory Snapshot, скорость GPU и
backward собственных Triton kernels не проверены из-за отсутствия NVIDIA GPU.
Английский перевод намеренно вне этой задачи, а не пропущенный пункт приёмки.

SHA-256 принятого текста:

- S6: `0dffc0e19f3cbff25950e4d04bf249e4ccc10cdb6284331d1d151df37dc19d39`
- S7: `649ea9212cb65dc800afbcaa50a346bfc631f03d6dcb095eff513c37ffb8de91`
- S8: `a3a7c358fe8a318316163288e1a992a909039ff8b523fb090d8dc640511707f6`
