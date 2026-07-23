---
title: Efficient DL Systems — complete transfer matrix
type: source-note
status: editorial
last_updated: 2026-07-23
---

# Efficient DL Systems — complete transfer matrix

Pinned source: [mryab/efficient-dl-systems](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0), commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`. Source prose/code remains English. License: MIT; asset credit is checked before publication.

| ID | Source path | Material | Adds to Bookvar | Decision | Destination | Assets | Status |
|---|---|---|---|---|---|---|---|
| E-01-L/S | `week01_intro/{lecture.pdf,seminar.ipynb}` | GPU, CUDA, benchmarking | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md`; `06 Практика/06 Измерить CUDA правильно.md` | PDF pages/notebook plots | audited |
| E-02-L/S | `week02_fast_pipelines/{lecture.pdf,seminar/practice.ipynb}` | pipeline, formats, profiler | yes | integrate | `00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md`; practice 08 | PDF/notebook | audited |
| E-02-H1…3 | `week02_fast_pipelines/homework/task{1,2,3}/` | loaders, padding, profiler | yes | integrate | `06 Практика/08 Padding, packing и profiler.md` | code | audited |
| E-02-A1…2 | `week02_fast_pipelines/seminar/{images,pics/1}/` | FP8 and profiler images | yes | source-only | `00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md` | credit required | audited |
| E-03-L/S/H | `week03_data_parallel/{lecture.pdf,practice.ipynb,homework/}` | DDP, AllReduce, SyncBN | yes | integrate | `44a Processes, collectives и DDP`; practice 09 | PDF/notebook/code | audited |
| E-04-L/S1…2 | `week04_large_models/{lecture.pdf,practice_part1.ipynb,practice_part2.ipynb}` | TP, PP, SP, offload | yes | integrate | `44b…d`; practice 10 | PDF/notebooks | audited |
| E-05-L/S/H | `week05_fsdp/{lecture.pdf,seminar.pdf,homework/}` | FSDP2, DeviceMesh, DCP | yes | integrate | `44e ZeRO, FSDP2, DeviceMesh и DTensor`; practice 12 | PDF/code | audited |
| E-06-L/S/H/A | `week06_dl_arithmetic/{lecture.pdf,seminar/,homework/,seminar/images/}` | arithmetic, fusion, compile, Liger | yes | integrate | chapters 04/07/56; practice 13 | asset credit required | audited |
| E-07-L | `week07_application_deployment/Effdl26-07.pdf` | service deployment | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | PDF pages | audited |
| E-07-P1…7 | `week07_application_deployment/{00_basics/*,01_python_server,02_docker,03_metrics,04_microservices,supervisord}/` | FastAPI, Docker, metrics, gRPC | yes | integrate | `06 Практика/17 Упаковать модель в наблюдаемый сервис.md` | code/configuration | audited |
| E-07-A | `week07_application_deployment/dataset/` | demo images | no — illustrative data only | exclude | — | no pedagogical transfer | audited |
| E-08-L/S | `week08_inference_software/{lecture.pdf,seminar.ipynb}` | metrics, KV, batching; Qwen measurements | yes | integrate | `55 KV-cache...`; `58b Benchmarking...` | PDF/notebook plots | audited |
| E-08-H/P | `week08_inference_software/homework/{homework_week8.ipynb,edlang/}` | mini inference engine | yes | integrate | `06 Практика/14 Собрать mini inference engine.md` | notebook/code | audited |
| E-09-L/S/H | `week09_inference_algorithms/{lecture.pdf,seminar.ipynb,homework/homework.ipynb}` | quantization, speculation, KV compression | yes | integrate | `57a`, `58`; practices 15–16 | PDF/notebooks | audited |

Inclusive suffix ranges expand to their separately audited source units.
