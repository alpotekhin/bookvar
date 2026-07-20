---
title: Supervised Fine-Tuning
aliases: [SFT, Instruction Tuning]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2109.01652
  - https://arxiv.org/abs/2203.02155
---

# Supervised Fine-Tuning (SFT)

**SFT** дообучает pretrained model на парах «вход → желаемый ответ» обычным maximum likelihood:

$$\mathcal L_{SFT}=-\sum_t \log\pi_\theta(y_t\mid x,y_{<t}).$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/post-training-courses/sft-prompt-masking.png]]

*System и user tokens участвуют в прямом проходе как контекст, но метки `-100`
исключают их из cross-entropy; loss вычисляется на assistant continuation.
Источник: Hugging Face,
[TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer).*

Такое маскирование объясняет, что означает «пара вход–ответ» на уровне токенов:
модель условится на всём диалоге, но градиент непосредственно учит продолжать
только размеченные участки assistant.

Loss часто маскируют на prompt tokens, чтобы оптимизировать именно assistant response; конкретный рецепт может обучать и другие части последовательности.

SFT учит формат, стиль, следование инструкциям, tool protocol и новые task patterns. Он не гарантирует factuality или предпочтительность и ограничен качеством demonstrations. Повторное обучение на узком наборе может вызвать forgetting или потерю разнообразия.

SFT обычно предшествует preference optimization / RL, создавая разумную reference policy. Но это практический pipeline, не обязательный закон: возможны direct alignment методы от base model.

## Подробнее

Подготовка instruction data, chat templates, loss masking и ошибки датасета
разобраны в главе [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT и instruction data]].

## Источники

- [FLAN](https://arxiv.org/abs/2109.01652)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
