---
title: "Эксперименты с FFN, глубиной и weight decay"
type: source-note
status: reviewed
source_type: paper
last_verified: 2026-11-10
---

Материалы к [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block#decoder-hyperparameter-evidence|разбору гиперпараметров decoder block]]. Сохранены опубликованные PDF PMLR; в главе показаны только авторские рисунки с исходными подписями. Условия переноса: [PMLR Publication Agreement, §2](https://proceedings.mlr.press/pmlr-license-agreement.html) и [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## MobileLLM

Zechun Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, [ICML 2024, PMLR 235:32431–32454](https://proceedings.mlr.press/v235/liu24ce.html). [Опубликованный PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24ce/liu24ce.pdf).

- §2.2.1 и Figure 3 (PDF с. 3), Table 10 (с. 14): у 12-слойных конфигураций с шириной 768 и 134.1 млн параметров средняя точность восьми zero-shot задач составляет 42.6% для ReLU-FFN и 43.9% для SwiGLU. Это сравнение двух вариантов FFN, а не варьирование её ширины при неизменной активации.
- §2.2.2, Figure 4(a,b) (с. 4) и Table 11 (с. 15): у 30 слоёв ширины 512 и 135.0 млн параметров среднее 44.8%; у 12 слоёв ширины 768 и 134.1 млн — 43.9%. На ARC-Challenge порядок обратный (26.1% и 28.9%). Число голов меняется вместе с шириной при размере головы 64.
- В последней строке Figure 3 для малой модели напечатано 47.1%, а в соответствующей строке Table 10 — 47.0%. Глава сравнивает первые строки, где значения совпадают. В Figure 4(c,d) ось подписана *Exact match*, в Table 12 метрика TQA названа *F1 score*; количественный пример в главе опирается на Figure 4(a) и Table 11.

## Weight Decay Improves Language Model Plasticity

Tessa Han, Sebastian Bordt, Hanlin Zhang и Sham M. Kakade, [ICML 2026, PMLR 306:39571–39602](https://proceedings.mlr.press/v306/han26b.html). [Опубликованный PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/han26b/han26b.pdf).

- §§3, 4.1–4.3 и Figure 3 (PDF с. 6): авторы предобучают семейства с архитектурами Llama 2 и OLMo 2 при нескольких значениях AdamW weight decay, затем отдельно проводят SFT. В пределах одного цвета сравниваются варианты одной архитектуры, размера и бюджета токенов. Вертикальная ось показывает результаты на шести задачах CoT reasoning после SFT, а не разницу между обучающей и проверочной потерями базы.
- В OLMo-2-1B-140x базовая проверочная потеря при weight decay 0.3 равна 2.6208 против 2.6088 при 0.1, но у первой базы выше результат после SFT. Коэффициенты корреляции на рисунке описывают группы не более чем из десяти точек; из этого сравнения не следует универсальное значение weight decay.
