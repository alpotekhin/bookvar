---
title: "MIPT NLP — Topic 1: ANCE Dense Retrieval Training"
course: "MIPT NLP"
topic: 1
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/MIPT NLP/ANCE]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]"
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]"
---

# MIPT NLP — Topic 1: ANCE — Dense Retrieval Training

> Source: `raw/courses/MIPT NLP/ANCE.md`

## Key points

**Задача:** достать релевантные документы из текстового корпуса. Когда приходит запрос — нужно проверить релевантность относительно всех документов (все могут быть релевантными).

Задача решается через **[[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|контрастивное обучение (Contrastive Learning)]]**.

## Contrastive Learning для Dense Retrieval

- Обучение на **триплетах**: (запрос, позитивный документ, негативный документ)
- **Hard negatives** — наиболее похожие на позитив (но неправильные) примеры; самые информативные негативы
- **ANCE** = сэмплинг хард-негативов с наибольшим лоссом (наиболее трудные примеры)

## Pipeline обучения (3 шага)

| Шаг | Метод | Цель |
|---|---|---|
| 1 | **RetroMAE** — pretrain на тексте | Улучшить вектора токенов |
| 2 | **In-batch negative sampling** (large batch) | Улучшить вектора предложений через триплеты |
| 3 | **ANCE** — hard negative sampling | Дополнительное улучшение через мультитаскинг и трудные примеры |

## Метод ANCE

- ANCE = **Approximate Nearest Neighbor Negative Contrastive Estimation**
- На каждом шаге обучения динамически обновляем индекс всех документов
- Сэмплируем как негативы те документы, которые модель находит близкими к запросу (но которые не являются ответом) — это «хард негативы»
- Contrastive learning с хард негативом: **отдаляем сэмплы с наибольшим лоссом**

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]
- Contrastive Learning *(новый концепт, пока нет статьи в vault)*
- Hard Negatives *(новый концепт, пока нет статьи в vault)*
- RetroMAE *(новый концепт, пока нет статьи в vault)*
