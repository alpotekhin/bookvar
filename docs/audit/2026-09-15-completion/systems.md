# Task 2 — завершение системной редакторской проверки

Дата: 2026-09-15. Статус: исправления завершены; независимая приёмка и общая сборка — у основного агента.

## Итог

Полностью прочитаны все 60 назначенных русских страниц и 11 существующих английских версий. Изменены 50 RU и 9 EN; отсутствующие переводы не создавались. Все 88 исходных замечаний сопоставлены с состоянием начала этого прохода: **82 fixed, 6 already-fixed, 0 rejected-with-evidence, 0 open**. Это не означает, что прежние правки были перезаписаны: уже исправленные механизмы сохранены.

Подробные исходные формулировки, индивидуальные решения, доказательства, SHA256 до/после и сведения о перечитывании находятся в [systems.json](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/docs/audit/2026-09-15-completion/systems.json>). Точные снимки 59 изменённых файлов сохранены в `.superpowers/sdd/2026-09-15-close-editorial-findings/task-2-before/` с исходной структурой каталогов.

## Что проверялось

Чтение было постраничным и полным, не поиском отдельных слов. После изменений все изменённые страницы перечитаны полностью; ряд последних мелких уточнений появился при этом перечитывании и затем проверен в контексте своего заменённого блока. В JSON такие случаи честно помечены `complete-full-page-plus-local-correction`, а длинные страницы, прочитанные непрерывными частями, — `in-contiguous-parts`. Это не заявление о дополнительном полном проходе после каждой последней замены одного абзаца.

Все 392 исходных встраивания рисунков на 71 странице сохранены как мультимножества. В главе о decoder block исторический рисунок GPT-2 перенесён после современного сквозного разбора; остальные рисунки не переставлялись. Новые Mermaid/SVG/сгенерированные визуализации не создавались. `source_unit_id`, явные HTML/Markdown-якоря и парность fenced blocks проверены. Все 60 исходных RU-хешей согласованы с начальным учётом; для EN хеши снимка и результата записаны отдельно. `git diff --check` по системному диапазону завершился с кодом 0 без вывода.

## Основные содержательные результаты

- Архитектура: соглашения о формах и матрицах; реальные конфигурации FFN; финальная нормализация как свойство выбранного блока; нормализованный MLA latent; RWKV-4 в отличие от retention; точные предпосылки upcycling; смысл zero-centered RMSNorm.
- Вычисления и обучение: FLOPs против байтов, причинные против полных attention-пар, локальные против суммарных GPU FLOPs; AMP без выдуманной второй master-копии; token-weighted loss; корректные зависимости pipeline; offload ledger; контроль сохранения случайного состояния; checkpoint-интервалы Daly и условия применимости.
- Inference: единицы GB/GiB и вместимость 194/45; блочная фрагментация и copy-on-write; варианты первого токена при P/D handoff; точное тождество E2E; Poisson counts против интервалов; два backend-пути TensorRT-LLM; GPTQ-ориентация; автор SpinQuant; дерево speculative verification явно ограничено greedy.
- Evaluation/MLOps: один ранжирующий score и два порога при редком классе; цена ошибок и ограничение проверок; COMPAS как исторический аудит риска, не обвинений; PUE без двойного учёта; benefit/cost против чистого ROI; FedAvg с принятыми участниками; record/user DP, clipping и композиция.

## Выполненная численная проверка

Среда: `/private/tmp/bookvar-editorial-venv/bin/python`, PyTorch 2.8.0, NumPy 2.2.6, CPU. **22 проверки прошли.** Ниже — выполненные вычисления, а не измерения GPU.

| Проверка | Полученный результат |
|---|---|
| EDLS KV ledger | 3 272 704 000 байт = 3.272704 GB = 3.047943115 GiB; вместимость 194/45 |
| DSA training tokens | 2 097 152 000 и 943 718 400 000 |
| Daly | Young 3219.94 с; lowest-order 3099.94 с; three-term 3140.43 с |
| RWKV-4 | выходы 2, 3, 5.2; состояние 10.5 / 1.75 |
| Hybrid state | 16 GiB полного KV против 4 GiB KV + 6.75 MiB SSM/conv state |
| Masked NLL | 1.0397207708; perplexity 2.8284271247 |
| Paged blocks | 7 против 4 блоков, 14/16 MiB резерва; COW 2/4 MiB |
| Ring collectives | decode 28.57344 мкс; prefill 4.72562048 мс на одну условную редукцию |
| PD handoff | TTFT 120/180 мс, ITL 70/10 мс, одинаковый TTST 190 мс |
| Редкий класс | BA 0.7979798/0.9393939; цена 4040/1210; 100/300 тревог |
| FedAvg и clipping | (4,6); при одном принятом клиенте (1,3); clipping (1.2,1.6) |
| Экономика/очередь | ROI 0.25; 480 кг CO2e; 5 реплик; M/M/1 0.2/1 с |
| Low rank | r512: 4 194 304 параметра, Z 32 KiB; r3072: 25 165 824, Z 192 KiB |
| MLA absorption | максимальная ошибка 8.88e−16 |
| TP forward/backward | выход 1.33e−15; максимальная ошибка градиента 2.84e−14 |
| Online causal attention | выход 1.67e−16; градиент 6.66e−16; 28 допустимых пар из 49 |
| Speculative chain mass | принятая и корректирующая массы в сумме воспроизводят target (0.4,0.2,0.4) |
| Native CPU AMP | параметры и градиенты остаются FP32 |
| Две опубликованные PP-таблицы | 64 операции, зависимости соблюдены, пики live: 4/4/4/4 и 4/4/3/1 |
| Опубликованный RMSNorm | FP32-ошибка 2.38e−7; BF16-вход × FP32-gain даёт FP32, BF16-gain — BF16 |
| Unequal-token virtual DDP | 2 и 5 токенов; ошибка против общего среднего 5.55e−17 |
| Checkpoint с dropout | при `preserve_rng_state=True` выходы и градиенты совпадают точно |

В JSON сохранены полные численные значения. GPU CUDA/Triton, Nsight, реальный NCCL/сеть, многопроцессный DDP, serving engines, длинное обучение и полный sklearn/imblearn pipeline не запускались. Виртуальный DDP проверяет арифметику нормировки; chain-mass не доказывает точность произвольного sampling по дереву. Условные latency/energy/cost примеры явно отделены от экспериментальных результатов. Общая сборка и визуальный site QA остаются у основного агента.

## Первоисточники и границы актуальности

Спорные утверждения проверены по первичным материалам: Meta/DeepSeek reference code, точным конфигурациям T5/Qwen/Mistral, RWKV-4, DeepSeek LLM, pinned DeepSeek-V3.2-Exp report §2.1, Daly 2006, NVIDIA CUDA/Nsight/Cumulus/TensorRT-LLM, оригинальным GPTQ/AWQ/SpinQuant/Chen/EAGLE, vLLM CLI, ProPublica и Dwork–Roth/McMahan. Точные locators и ограничения записаны в `checked_sources` каждой страницы и в самих изменённых фрагментах. Для DSA, Daly и спорных figure captions просмотрены соответствующие оригинальные страницы/изображения. Даты `last_verified` не обновлялись массово: проверка одного источника не выдается за проверку всего современного продукта.

## Все страницы

В столбце «исходные замечания» указано решение каждого замечания исходного systems.json, а не число новых локальных правок.

| № | Страница | RU | Исходные замечания | EN |
|---:|---|---|---|---|
| 1 | [01 LLaMA как базовая архитектура](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 2 | [02 Современный decoder block](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 3 | [03 Pre-norm, RMSNorm, SwiGLU и residual](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual.md>) | изменена | 1: fixed | нет существующей пары |
| 4 | [04 RoPE](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/07 Анатомия современной LLM/04 RoPE.md>) | без правок | 1: already-fixed | нет существующей пары |
| 5 | [05 Transformer с нуля — формы, параметры и стоимость](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость.md>) | изменена | 1: fixed; 2: fixed; 3: fixed | нет существующей пары |
| 6 | [01 MHA, MQA и GQA](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md>) | без правок | 1: already-fixed | нет существующей пары |
| 7 | [02 MLA и сжатие KV-cache](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache.md>) | изменена | 1: fixed | нет существующей пары |
| 8 | [03 Длинный контекст — расширение, разреженность и оценивание](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 9 | [00 Карта модуля и источники](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/09 Dense FFN и Mixture of Experts/00 Карта модуля и источники.md>) | изменена | 1: fixed | нет существующей пары |
| 10 | [01 Dense FFN — token-wise вычисление, expansion и gating](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md>) | изменена | 1: fixed; 2: already-fixed; 3: fixed | нет существующей пары |
| 11 | [02 Mixture of Experts — routing, capacity и serving](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving.md>) | изменена | 1: fixed | нет существующей пары |
| 12 | [03 Mamba, RWKV, RetNet и гибридные архитектуры](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 13 | [01 Llama, Qwen и DeepSeek как эволюция блока](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока.md>) | изменена | 1: fixed; 2: fixed; 3: fixed; 4: fixed | нет существующей пары |
| 14 | [02 Альтернативы Transformer и multimodality](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 Атлас современных архитектур/02 Альтернативы Transformer и multimodality.md>) | изменена | 1: fixed; 2: fixed; 3: fixed | нет существующей пары |
| 15 | [01 Модель как часть системы](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/01 Модель как часть системы.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 16 | [02 GPU, CUDA и иерархия памяти](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 17 | [03 Измерение производительности и roofline](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md>) | без правок | замечаний не было; прочитана | нет существующей пары |
| 18 | [04 Арифметика Transformer и MoE](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 19 | [05 Численные форматы и mixed precision](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 20 | [06 Data pipeline, padding и packing](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md>) | изменена | 1: fixed | нет существующей пары |
| 21 | [07 Profiling ML-нагрузки](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 22 | [08 GPU kernels и Triton — от программы к измерению](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 23 | [01 Данные и pre-training](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training.md>) | без правок | замечаний не было; прочитана | нет существующей пары |
| 24 | [02 Distributed training и precision](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/02 Distributed training и precision.md>) | изменена | 1: fixed; 2: fixed; 3: fixed | нет существующей пары |
| 25 | [41 Сбор, очистка и смеси данных](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 26 | [41a Дедупликация, PII и контроль качества корпуса](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса.md>) | изменена | 1: fixed | нет существующей пары |
| 27 | [42 Next-token prediction](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md>) | изменена | 1: fixed | нет существующей пары |
| 28 | [43 Scaling laws](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md>) | изменена | 1: fixed | нет существующей пары |
| 29 | [44 Distributed training и mixed precision](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md>) | изменена | 1: fixed | нет существующей пары |
| 30 | [44a Processes, collectives и DDP](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md>) | изменена | 1: fixed | нет существующей пары |
| 31 | [44b Gradient checkpointing и offload](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 32 | [44c Tensor и sequence parallelism](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md>) | изменена | 1: already-fixed | нет существующей пары |
| 33 | [44d Pipeline parallelism](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44d Pipeline parallelism.md>) | изменена | 1: fixed | нет существующей пары |
| 34 | [44e ZeRO, FSDP2, DeviceMesh и DTensor](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor.md>) | без правок | замечаний не было; прочитана | нет существующей пары |
| 35 | [44f Expert и hybrid parallelism](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md>) | без правок | замечаний не было; прочитана | нет существующей пары |
| 36 | [44g Network, storage и distributed checkpoints](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md>) | изменена | 1: fixed | нет существующей пары |
| 37 | [44h Fault tolerance и fleet orchestration](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 38 | [53 Синтетические данные и учебные программы](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md>) | изменена | 1: fixed | нет существующей пары |
| 39 | [01 KV-cache, batching и FlashAttention](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/01 KV-cache, batching и FlashAttention.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 40 | [02 Quantization и deployment](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/02 Quantization и deployment.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 41 | [54 Декодирование и выбор следующего токена](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена.md>) | изменена | 1: fixed | изменена |
| 42 | [55 KV-cache, пакетирование и PagedAttention](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md>) | изменена | 1: fixed | изменена |
| 43 | [55a Физика LLM inference — prefill, decode и roofline](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md>) | изменена | 1: fixed | изменена |
| 44 | [55b Scheduling — continuous batching, chunked prefill и prefix caching](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md>) | без правок | 1: already-fixed | прочитана, без правок |
| 45 | [55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer.md>) | изменена | 1: fixed | изменена |
| 46 | [56 FlashAttention](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/56 FlashAttention.md>) | без правок | замечаний не было; прочитана | прочитана, без правок |
| 47 | [57 Квантизация языковых моделей](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md>) | изменена | 1: fixed | изменена |
| 48 | [57a KV-cache compression и offload](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 49 | [57b Сжатие моделей — pruning, distillation и low-rank](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank.md>) | изменена | 1: fixed | нет существующей пары |
| 50 | [58 Спекулятивное декодирование](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md>) | изменена | 1: already-fixed; 2: fixed | изменена |
| 51 | [58a Распределённый inference и disaggregated serving](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving.md>) | изменена | 1: fixed | изменена |
| 52 | [58a2 Раздельное обслуживание prefill и decode](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md>) | изменена | 1: fixed; 2: fixed | изменена |
| 53 | [58b Benchmarking, SLO и эксплуатация inference](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md>) | изменена | 1: fixed; 2: fixed | изменена |
| 54 | [58c Queueing и capacity planning](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 55 | [59 Оценивание моделей и контаминация](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md>) | без правок | замечаний не было; прочитана | нет существующей пары |
| 56 | [59a Несбалансированная классификация](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/18 Evaluation и методология/59a Несбалансированная классификация.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 57 | [59a Responsible systems](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/18 Evaluation и методология/59a Responsible systems.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 58 | [01 ML workflow](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md>) | без правок | замечаний не было; прочитана | нет существующей пары |
| 59 | [02 MLOps](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |
| 60 | [03 Edge и federated deployment](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md>) | изменена | 1: fixed; 2: fixed | нет существующей пары |

## Граница изменений

Правки ограничены страницами Task 2, существующими EN-парами, точными снимками и этими двумя отчётами. Navigation, source/asset registry, оригинальные курсы, raw, тестовая инфраструктура и чужие страницы не менялись. В частности, исправление реестра трёх Chen-иллюстраций выполнено основным агентом отдельно; этот отчёт не присваивает себе ту работу. Коммиты, push, merge, deploy и новая рабочая копия не создавались.
