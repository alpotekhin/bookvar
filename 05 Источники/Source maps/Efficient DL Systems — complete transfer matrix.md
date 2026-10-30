---
title: Efficient DL Systems — complete transfer matrix
type: source-note
status: editorial
last_updated: 2026-10-26
---

# Efficient DL Systems — complete transfer matrix

26 октября 2026: полностью прочитаны Week 8 seminar и homework, а также
исходные engine.py/config.py. В практику 14 добавлено полное исполняемое
KV/greedy-решение: реальная маленькая Qwen3 на CPU, раздельные prefill/decode,
правое padding KV, собственные кэши запросов и сравнение с uncached oracle.
Сохранён язык исходного контракта и исправлены переходы состояния при удалении
строк и лимите одного токена. Это не выполнение GPU-измерений Qwen3-4B,
асинхронного сервера и всего homework; соответствующие статусы остаются partial.

Затем целиком прочитаны scheduler/metric managers, API server, launcher и
benchmark. В ту же практику добавлены исходное задание на три политики,
исполняемый расчёт метрик по временным отметкам и ограничения исходного
non-streaming клиента. Времена в примере заданы явно; это не измерение сервера
и не проведённое сравнение политик. Проверка расчёта включает 200 переносов/
масштабирований часов, неопределённый TPOT и отказы/отмены.

Следующий проверенный слой — полный admission scheduler с единым внешним ID,
терминальными событиями и тремя политиками. Выполнены 54 конечных нейросетевых
CPU-запуска (два профиля, три лимита slots, три политики, три повтора) с отдельным
producer-процессом. Сохранены фактические временные отметки, график фаз и хеши;
648 ответов совпали с uncached oracle. Это ограниченный CPU-эксперимент,
а не воспроизведение HTTP/Qwen3-8B, GPU/SLO-capacity или вывод о лучшей политике.

В продолжение [[02 Areas/ML & DL/06 Практика/14a HTTP streaming и жизненный цикл запроса|14a]]
добавлены полный bounded token-ID HTTP/NDJSON слой и объяснение доставки,
отмены, deadline, readiness, terminal races и cleanup. Проверка использует
реальный loopback TCP, исходную маленькую Qwen3 и контрольные ошибки; серверы
останавливаются после проверки. Это отдельный учебный протокол, не wire-compatible
замена исходного строкового клиента EDLS и не OpenAI API. Tokenizer/checkpoint,
контейнеризация/нагрузочная проверка и GPU/SLO-capacity остаются открытыми.

В практике 17 разобран Week 7: девять полных исходных файлов, Docker cache,
readiness, контракт препроцессинга, Prometheus/Grafana, гистограммы, gRPC и
supervisord. Исходные код и лицензия сохранены. Проверены in-process
ASGI/Flask-сценарии и реальный exporter с заглушкой VGG; в исходном pytest
задокументирована одна ошибка ожидаемого значения. Контейнеры, реальные
scrapes, gRPC и нагрузочный опыт не запускались; строки E-07-P остаются partial.

Pinned source: [mryab/efficient-dl-systems](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0), commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`. Source language is English and remains English on import. License: MIT; third-party slide and image credits remain subject to figure-level review.

`audited` в последнем столбце означает аудит и полный импорт исходного объекта,
а не автоматическую готовность destination. Реальное состояние переноса
проверяется в
[[05 Источники/Source maps/ML systems — Bookvar gap matrix|ML systems — Bookvar gap matrix]]:
destination должен существовать, вести к полному оригиналу и содержать
перенесённую теорию, практический контракт или оба слоя.

| ID | Source path | Material | Adds to Bookvar | Decision | Destination | Assets | Status |
|---|---|---|---|---|---|---|---|
| E-01-L | `week01_intro/lecture.pdf` | GPU architecture, CUDA, and benchmarking | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | lecture.pdf | audited |
| E-01-S | `week01_intro/seminar.ipynb` | CUDA operations and timing experiments | yes | integrate | `06 Практика/06 Измерить CUDA без самообмана.md` | seminar.ipynb | audited |
| E-02-L | `week02_fast_pipelines/lecture.pdf` | mixed precision, pipelines, and profiling | yes | integrate | `00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md` | lecture.pdf | audited |
| E-02-S | `week02_fast_pipelines/seminar/practice.ipynb` | profilers, memory snapshots, and loading | yes | integrate | `06 Практика/08 Ускорить data pipeline.md` | practice.ipynb | audited |
| E-02-H1 | `week02_fast_pipelines/homework/task1/` | storage and data-loading assignment | yes | integrate | `06 Практика/08 Ускорить data pipeline.md` | dataset.py; train.py; unet.py | audited |
| E-02-H2 | `week02_fast_pipelines/homework/task2/` | dynamic padding and Transformer pipeline assignment | yes | integrate | `06 Практика/08 Ускорить data pipeline.md` | dataset.py; run_epoch.py; transformer.py | audited |
| E-02-H3 | `week02_fast_pipelines/homework/task3/` | profiling and ViT pipeline assignment | yes | integrate | `06 Практика/08 Ускорить data pipeline.md` | profiler.py; run_epoch.py; vit.py | audited |
| E-02-A1 | `week02_fast_pipelines/seminar/images/` | mixed-precision and profiler images | yes | source-only | `00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md` | FP8.png; MXFP8.png; loss_scaling.png | audited |
| E-02-A2 | `week02_fast_pipelines/seminar/pics/1/` | JPEG decoding benchmark inputs | yes | source-only | `00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md` | 1.jpg; 2.jpg; 3.jpg | audited |
| E-03-L | `week03_data_parallel/lecture.pdf` | data parallelism and collective communication | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | lecture.pdf | audited |
| E-03-S | `week03_data_parallel/practice.ipynb` | distributed-training practice | yes | integrate | `06 Практика/09 Реализовать ring all-reduce.md` | practice.ipynb | audited |
| E-03-H | `week03_data_parallel/homework/` | AllReduce, DDP, and SyncBN assignment | yes | integrate | `06 Практика/09 Реализовать ring all-reduce.md` | allreduce.py; ddp_cifar100.py; syncbn.py | audited |
| E-04-L | `week04_large_models/lecture.pdf` | tensor, pipeline, and sequence parallelism | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md` | lecture.pdf | audited |
| E-04-S1 | `week04_large_models/practice_part1.ipynb` | checkpointing and offload practice | yes | integrate | `06 Практика/10 Измерить checkpointing и offload.md` | practice_part1.ipynb | audited |
| E-04-S2 | `week04_large_models/practice_part2.ipynb` | tensor and pipeline parallelism practice | yes | integrate | `06 Практика/11 Разрезать Transformer по TP и SP.md` | practice_part2.ipynb | audited |
| E-05-L | `week05_fsdp/lecture.pdf` | FSDP, ZeRO, DeviceMesh, and DTensor | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor.md` | lecture.pdf | audited |
| E-05-S | `week05_fsdp/seminar.pdf` | FSDP2 and distributed checkpoints | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor.md` | seminar.pdf | audited |
| E-05-H | `week05_fsdp/homework/` | implementing FSDP from collectives | yes | integrate | `06 Практика/12 Собрать и проверить FSDP.md` | fsdp.py; test.py; train.py | audited |
| E-06-L | `week06_dl_arithmetic/lecture.pdf` | Transformer and MoE arithmetic | yes | integrate | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | lecture.pdf | audited |
| E-06-S | `week06_dl_arithmetic/seminar/practice.ipynb` | fusion, torch.compile, hierarchy, and Liger | yes | integrate | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | practice.ipynb | audited |
| E-06-H | `week06_dl_arithmetic/homework/` | efficient Transformer training assignment | yes | integrate | `06 Практика/13 Оптимизировать один Transformer step.md` | efficient_train.py; efficient_calculator.py; test_e2e.py | audited |
| E-06-A | `week06_dl_arithmetic/seminar/images/` | fusion and memory-hierarchy figures | yes | source-only | `00 Учебник/14 Inference и оптимизация/56 FlashAttention.md` | fused_kernels1.png; mem_hierarchy.png; liger_ce.png | audited |
| E-07-L | `week07_application_deployment/Effdl26-07.pdf` | web-service deployment foundations | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | Effdl26-07.pdf | audited |
| E-07-P1 | `week07_application_deployment/00_basics/non-smokers-repo/` | service baseline with readiness and reproducibility limits | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | Dockerfile.good; docker-compose.yml; test_health.py | partial-reviewed-2026-10-26; source explanation integrated; container/RPC/scrape/load reproduction remains open |
| E-07-P2 | `week07_application_deployment/00_basics/smokers-repo/` | service anti-pattern comparison | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | Dockerfile.bad; app.py; test_smoke.py | partial-reviewed-2026-10-26; source explanation integrated; container/RPC/scrape/load reproduction remains open |
| E-07-P3 | `week07_application_deployment/01_python_server/` | Python inference server | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | server.py; README.md; labels.json | partial-reviewed-2026-10-26; source explanation integrated; container/RPC/scrape/load reproduction remains open |
| E-07-P4 | `week07_application_deployment/02_docker/` | containerized inference server | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | Dockerfile; docker-compose.yaml; server.py | partial-reviewed-2026-10-26; source explanation integrated; container/RPC/scrape/load reproduction remains open |
| E-07-P5 | `week07_application_deployment/03_metrics/` | Prometheus, Grafana, and system metrics | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | server.py; prometheus.yml; telegraf.conf | partial-reviewed-2026-10-26; source explanation integrated; container/RPC/scrape/load reproduction remains open |
| E-07-P6 | `week07_application_deployment/04_microservices/` | REST and gRPC microservices | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | inference-api.py; grpc-client.py; inference.proto | partial-reviewed-2026-10-26; source explanation integrated; container/RPC/scrape/load reproduction remains open |
| E-07-P7 | `week07_application_deployment/supervisord/` | process supervision for inference service | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | supervisord.conf; server.py; docker-compose.yaml | partial-reviewed-2026-10-26; source explanation integrated; container/RPC/scrape/load reproduction remains open |
| E-07-A | `week07_application_deployment/dataset/` | demo inference images | no: illustrative inputs do not add instruction | exclude | `05 Источники/Source maps/Efficient DL Systems — complete transfer matrix.md` | 5.jpeg; 6.jpeg; 9.jpeg | audited |
| E-08-L | `week08_inference_software/lecture.pdf` | inference metrics, KV cache, batching, and serving frameworks | yes | integrate | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | lecture.pdf | audited |
| E-08-S | `week08_inference_software/seminar.ipynb` | KV shapes, padding, Qwen prefill/decode and measurements | yes | integrate-adapted | `06 Практика/14 Собрать минимальный inference engine.md` | complete source plus full original HF KV diagram | partial-reviewed-2026-10-26; complete seminar read and CPU neural KV/greedy reference verified; GPU measurements and pretrained Qwen3-4B not reproduced; 58b remains a related measurement destination |
| E-08-H | `week08_inference_software/homework/` | mini inference-engine assignment, scheduler, metrics, API, and benchmark | yes | integrate-adapted | `06 Практика/14 Собрать минимальный inference engine.md` | full neural/admission references and CPU traces; continuation 14a contains a tested token-ID HTTP/NDJSON lifecycle and original licensed diagram | partial-reviewed-2026-10-26; CPU core/policies and real loopback HTTP contract checked; original text-tokenizer/checkpoint interface, sampling bonus, container/observability, physical peak memory and GPU/SLO-capacity remain open |
| E-09-L | `week09_inference_algorithms/lecture.pdf` | quantization, speculative decoding, and KV compression | yes | integrate | `00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload.md` | lecture.pdf | audited |
| E-09-S | `week09_inference_algorithms/seminar.ipynb` | Triton, W8A8, and SmoothQuant practice | yes | integrate | `06 Практика/15 Реализовать W8A8 и SmoothQuant.md` | seminar.ipynb | audited |
| E-09-H | `week09_inference_algorithms/homework/homework.ipynb` | speculative decoding and KV rollback assignment | yes | integrate | `06 Практика/16 Проверить speculative decoding и rollback KV.md` | homework.ipynb | audited |

Literal rows: 37.
