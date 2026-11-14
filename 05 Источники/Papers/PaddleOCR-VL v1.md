---
title: "PaddleOCR-VL v1 — разметка страницы и порядок чтения"
type: source-note
status: reviewed
source_type: paper
last_verified: 2026-11-11
primary_sources:
  - https://arxiv.org/abs/2510.14528v1
---

# PaddleOCR-VL v1

Cheng Cui et al., PaddlePaddle Team, Baidu. [Отчёт v1, 16 октября 2025 года](https://arxiv.org/html/2510.14528v1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

В §2.1 и на [рисунке 2](https://arxiv.org/html/2510.14528v1#S2.F2)
видна граница между анализом страницы и распознаванием отдельных фрагментов.
PP-DocLayoutV2 находит области, их классы и порядок чтения. PaddleOCR-VL-0.9B
получает вырезанные области и инструкции по типу фрагмента. Постобработка
собирает результаты в Markdown и JSON.

- [§2.1.1, рисунок 3](https://arxiv.org/html/2510.14528v1#S2.F3):
  RT-DETR предлагает рамки и классы; сеть порядка оценивает пары областей.
- [§2.1.2, рисунок 4](https://arxiv.org/html/2510.14528v1#S2.F4):
  NaViT, проектор с объединением патчей и ERNIE-4.5-0.3B распознают фрагменты.
- [§4](https://arxiv.org/html/2510.14528v1#S4) разделяет оценку разбора целой
  страницы и распознавания отдельных элементов. Высокий балл на одном уровне
  не измеряет точность конкретного поля на другом.

[Код подсчёта порядка в Transformers v5.14.0](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/pp_doclayout_v2/image_processing_pp_doclayout_v2.py#L85-L111)
суммирует вероятности после сигмоиды, а не округлённые нули и единицы.
Числа на рисунке 3 удобны для проверки самого принципа сортировки.

[[00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding#doc-paddleocr-reading-order|В учебнике: разбор одного счёта по этой схеме]].
