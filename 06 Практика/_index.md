---
title: Практика
type: practice
status: active
last_updated: 2026-07-18
---

# Практика

Практические работы превращают формулу или архитектурную схему в наблюдаемое
поведение. Для каждой работы указан конкретный результат: реализация,
измерение, трассировка чужого кода или набор проверяемых утверждений.

| Практика | Предпосылки | Время | Результат |
|---|---|---:|---|
| [[02 Areas/ML & DL/06 Практика/01 Собрать micrograd и bigram LM|Собрать micrograd и биграммную модель]] | Python, производная | 3–5 часов | Собственный скалярный autograd и минимальная языковая модель |
| [[02 Areas/ML & DL/06 Практика/02 Реализовать causal self-attention|Причинное self-attention]] | Матрицы, softmax | 45–90 минут | Интерактивная проверка маски и реализация на PyTorch |
| [[02 Areas/ML & DL/06 Практика/03 Прочитать современный decoder в LitGPT|Прочитать декодер LitGPT]] | Transformer block | 1–2 часа | Трасса тензоров от embedding до logits |
| [[02 Areas/ML & DL/06 Практика/04 Проследить RLVR pipeline в Open-R1|Проследить RLVR в Open-R1]] | SFT, policy gradient | 2–3 часа | Карта данных, генерации, verifier и обновления policy |
| [[02 Areas/ML & DL/06 Практика/05 Измерить KV-cache и quantization|Измерить KV-cache и квантование]] | Inference basics | 1–2 часа | Таблица памяти, скорости и ошибки квантования |
| [[02 Areas/ML & DL/06 Практика/06 Измерить CUDA без самообмана|Измерить CUDA без самообмана]] | Python, PyTorch, CUDA | 1–2 часа | Воспроизводимый benchmark и profiler trace |
| [[02 Areas/ML & DL/06 Практика/07 Построить roofline и найти bottleneck|Построить roofline]] | FLOP, bytes, CUDA timing | 2–3 часа | Roofline целевого GPU и проверенный диагноз bottleneck |
| [[02 Areas/ML & DL/06 Практика/08 Ускорить data pipeline|Ускорить data pipeline]] | DataLoader, tokenizer | 2–4 часа | Timeline и сравнение padding/bucketing/packing |
| [[02 Areas/ML & DL/06 Практика/09 Реализовать ring all-reduce|Реализовать ring all-reduce]] | Distributed PyTorch | 3–5 часов | Collective, тесты и bandwidth curve |
| [[02 Areas/ML & DL/06 Практика/10 Измерить checkpointing и offload|Checkpointing и offload]] | Transformer training | 2–4 часа | Pareto-график memory–time |
| [[02 Areas/ML & DL/06 Практика/11 Разрезать Transformer по TP и SP|Tensor и sequence parallelism]] | Collectives, tensor shapes | 4–6 часов | Sharded block и ledger коммуникаций |
| [[02 Areas/ML & DL/06 Практика/12 Собрать и проверить FSDP|Собрать FSDP]] | DDP, optimizer states | 4–6 часов | FSDP run, memory snapshots и переносимый checkpoint |
| [[02 Areas/ML & DL/06 Практика/13 Оптимизировать один Transformer step|Оптимизировать Transformer step]] | Profiling, AMP, attention | 4–8 часов | Проверенная серия оптимизаций и итоговая ведомость |
| [[02 Areas/ML & DL/06 Практика/14 Собрать минимальный inference engine|Минимальный inference engine]] | KV-cache, batching | 5–8 часов | Scheduler, prefill/decode и SLO-графики |
| [[02 Areas/ML & DL/06 Практика/15 Реализовать W8A8 и SmoothQuant|W8A8 и SmoothQuant]] | Quantization basics | 3–5 часов | Калибровка, accuracy и kernel-level benchmark |
| [[02 Areas/ML & DL/06 Практика/16 Проверить speculative decoding и rollback KV|Speculative decoding]] | Sampling, KV-cache | 4–6 часов | Корректный acceptance и rollback cache |
| [[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис|Наблюдаемый ML-сервис]] | HTTP, Docker, metrics | 4–8 часов | Контейнер, dashboard и load-test |

Первая интерактивная лаборатория уже доступна на сайте. Работы 06–17 используют
полностью импортированные оригинальные лекции, notebooks и homework EDLS: ссылки
внутри каждой работы ведут не на краткий пересказ, а на исходный учебный
материал. По мере развития сайта измерительные стенды можно переносить в
интерактивные демо, не меняя постановку эксперимента.
