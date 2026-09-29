---
title: Self-Rewarding Language Models
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://proceedings.mlr.press/v235/yuan24d.html
authors: [Weizhe Yuan et al.]
year: 2024
license: CC BY 4.0
---

# Self-Rewarding Language Models

Одна языковая модель выполняет две задачи: отвечает на запросы и оценивает
ответы по отдельной инструкции. Её оценки задают пары для DPO следующей версии.
Начальное обучение включает как ответы, так и примеры оценивания.

Механизм, ограничения общих ошибок и результаты итераций разобраны в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#одна-модель-как-actor-и-judge|главе о модели награды]].
Figure 1 перенесена из финальной статьи ICML, физическая страница 2,
без полей и текста вокруг схемы, без перерисовки.

Сохранённый PDF: `Source PDFs/self-rewarding-icml2024.pdf`.
Лицензия PMLR — CC BY 4.0; контрольные суммы и параметры извлечения при 240 DPI
находятся в `05 Источники/asset-registry.yml`.
