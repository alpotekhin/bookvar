---
title: Данные и pre-training
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Данные и pre-training

> [!abstract] Идея главы
> «Обучили на интернете» скрывает главную часть работы. Корпус появляется после
> crawl, parsing, фильтрации, deduplication, выбора языков и смеси domains. Эти
> решения определяют модель не меньше, чем число layers.

## От страницы до training token

Типичный путь:

```text
raw web/books/code
→ extraction и language ID
→ quality/safety filters
→ exact и near deduplication
→ domain mixture
→ tokenization
→ document packing
→ next-token batches
```

Качество данных влияет на знания, язык, стиль, безопасность и memorization.
Deduplication уменьшает leakage и переобучение частых документов. Mixture
управляет долями code, math, multilingual и domain data.

Synthetic data полезны для редких skills, но могут усиливать ошибки teacher и
снижать разнообразие. Сильные семейства используют предыдущие модели как
annotators, OCR-системы и generators новых training instances.

## Deduplication

Повторяющиеся страницы заставляют модель тратить budget на один текст много раз,
усиливают memorization и загрязняют benchmarks. Exact hashes ловят копии;
MinHash/LSH и similarity methods — почти одинаковые документы.

## Mixture

Если просто sample документы пропорционально доступному объёму, web English
подавит math, code и малые языки. Mixture weights намеренно повышают долю
нужных domains. Поэтому количество raw bytes не равно количеству training tokens.

## Chinchilla и inference-optimal training

Compute-optimal scaling балансирует parameters и tokens при заданном training
budget. LLaMA сделала акцент на меньших моделях, обученных дольше, потому что
стоимость многократного inference может доминировать над однократным training.

Chinchilla отвечает на вопрос «как распределить фиксированный training compute».
Deployment-optimal model может обучаться дольше compute-optimal точки, если
после этого будет обслуживать огромное число запросов.

## Что измерять

- доли языков и domains;
- tokens после всех filters;
- contamination benchmarks;
- duplicate rate;
- токсичность и PII;
- качество held-out documents;
- memorization canaries;
- ablations смеси данных.

- [[02 Areas/ML & DL/Concepts/Training/Pre-training]]
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws]]
- [[02 Areas/ML & DL/Concepts/Training/Synthetic Data]]
