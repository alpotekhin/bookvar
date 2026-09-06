---
title: "Перенос vision encoder между VLM"
type: concept
status: canonical
last_updated: 2026-08-19
primary_sources:
  - https://github.com/deepseek-ai/DeepSeek-VL2
  - https://arxiv.org/abs/2412.10302
  - https://github.com/QwenLM/Qwen3-VL
---

# Можно ли подключить vision encoder DeepSeek-VL к Qwen

Короткий ответ: **да, как новый обучаемый multimodal stack; нет, как механическую перестановку готовых деталей**. Vision encoder производит непрерывные patch features, а не универсальный язык, который любая LLM читает одинаково.

Практический путь — взять опубликованный vision backbone, например SigLIP-подобный encoder из DeepSeek-VL2, подключить новую linear/MLP projection к text-only Qwen и провести alignment на image–caption данных. Затем нужен multimodal instruction tuning на VQA, OCR, documents и grounding; при достаточном бюджете частично размораживают vision tower и LLM.

Готовый projector DeepSeek-VL2 переносить обычно нельзя. Он обучен под hidden-state geometry и sequence contract конкретного language backbone. Даже одинаковая размерность $d_{llm}$ не означает одинаковые направления признаков, нормы, positional conventions и chat template.

Замена vision encoder внутри уже готовой Qwen-VL ещё сложнее. Современный visual stack связан с dynamic resolution, patch merge, spatial/temporal position IDs, special tokens и иногда deep injection признаков в несколько слоёв. Новый encoder должен воспроизвести этот контракт либо потребует переобучения connector и зависимых частей.

## Минимальная программа эксперимента

1. Зафиксировать shapes: число patches, width encoder, patch size, pooling/merge и hidden size LLM.
2. Добавить новый connector; не загружать несовместимые projector weights только потому, что shapes совпали.
3. На alignment заморозить обе башни и обучить connector.
4. Проверить captioning и image–text dependence контрфактической заменой изображения.
5. Провести instruction SFT и добавить text-only mixture против забывания языка.
6. Отдельно измерить OCR, grounding, разрешение, visual token count и prefill cost.
7. Лишь затем сравнить partial/full unfreezing.

Наиболее важный baseline — родной vision stack выбранной VLM при одинаковой LLM, данных и token budget. Без него нельзя понять, дал ли DeepSeek/SigLIP encoder выигрыш или эксперимент просто изменил разрешение и число токенов.

## Источники

- [DeepSeek-VL2 repository](https://github.com/deepseek-ai/DeepSeek-VL2) и [paper](https://arxiv.org/abs/2412.10302).
- [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL).
- Механизм: [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64a Connectors и fusion|projector, resampler и fusion]].

← [[02 Areas/ML & DL/02 Атлас моделей/Сравнения/Сравнение семейств|К сравнениям]]
