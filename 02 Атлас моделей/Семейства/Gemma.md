---
title: "Gemma"
type: model-family
organization: Google DeepMind
first_release: 2024-02
latest_verified_release: Gemma 3 and specialized Gemma 3 variants
last_verified: 2026-07-20
architecture_base: compact decoder-only Transformer
modalities: [text, image]
status: active
---

# Gemma

## Место в истории

Gemma — открытая компактная линия Google DeepMind, связанная с исследованиями и инфраструктурой Gemini, но не открытая копия Gemini. Gemma 1 дала 2B/7B dense checkpoints; Gemma 2 показала distillation и чередование local/global attention; Gemma 3 добавила image input, 128K и широкую multilingual поддержку. PaliGemma, CodeGemma, RecurrentGemma, ShieldGemma и MedGemma — специализированные родственники, а не последовательные общие поколения.

## Таблица релизов

| Релиз | Architecture diff | Tokenizer/context/modalities | Training/post-training | Evidence |
|---|---|---|---|---|
| Gemma 1 (2024-02) | dense decoder; MHA у 2B, MQA у 7B; GeGLU/RMSNorm/RoPE | SentencePiece 256K; 8K; text | 2T/6T tokens; IT variants | **A** |
| Gemma 2 (2024-06) | GQA; alternating local 4K/global attention; logit soft-capping; 9B/27B | 8K; text | teacher distillation central to pre-training; SFT + RLHF | **A** |
| Gemma 3 (2025-03) | 5 local : 1 global pattern; SigLIP vision encoder for 4B+; QK norm | 1B text-only 32K; 4B/12B/27B multimodal 128K; 140+ languages | distillation; multimodal and long-context stages; IT/function calling | **A/B** |
| Gemma 3n / domain variants (2025+) | mobile-efficient or domain adapters/encoders | model-specific text/image/audio | specialized recipes; not a Gemma 4 claim | **B** |

## Неизменное ядро и diff

Основная линия остаётся decoder-only, pre-norm, RoPE и gated MLP. Gemma 2 экономит внимание: local layers видят окно, global layers связывают весь prompt. Soft-capping ограничивает logits через `tanh`, стабилизируя attention/final logits. Gemma 3 увеличивает долю local layers и добавляет vision encoder; изображение превращается в фиксированное число visual tokens. Мультимодальность относится к 4B/12B/27B, не к 1B.

## Tokenizer, данные и post-training

Большой 256K vocabulary сохраняется как важная multilingual особенность. Gemma reports называют объём и категории данных, фильтрацию безопасности и distillation, но не публикуют полный corpus manifest. В Gemma 2 logits teacher используются уже на pre-training, поэтому её успех нельзя приписать только архитектуре. Instruction-tuned варианты проходят SFT и RLHF/feedback stages; structured output и function calling — post-training/interface, не свойство SigLIP.

## Inference и serving

Local attention уменьшает attention work/cache только при корректной реализации alternating pattern. 128K существенно повышает KV memory; Google публикует quantized variants и интеграции с Keras, Transformers, Gemma.cpp и облачными runtimes. Vision добавляет encoder и image tokens, поэтому text-only latency нельзя переносить на multimodal запрос. Gemma license/terms отличаются от Apache/MIT и проверяются перед распространением.

## Визуальный первоисточник: где проявился data-centric выигрыш

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/gemma-capability-comparison.png]]

График Gemma 1 сравнивает 7B с Llama 2 и Mistral по агрегированным категориям:
наиболее заметен разрыв в math/science и coding, а не в обычном QA. Это полезный
контекст для последующего Gemma 2: distillation и data recipe развивали сильную
сторону семейства, а не просто меняли local/global attention. Автор: Gemma Team,
Google DeepMind. Источник: Figure 1,
[Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/pdf/2403.08295).
Файл перенесён без изменения из официального PDF extraction. Использование
весов регулируют Gemma Terms; лицензия самого рисунка отдельно не указана.
Проверено 2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** три technical reports, weights/configs, tokenizer, model cards и reference implementations. **Неизвестно полностью:** training documents, точные доли synthetic/distilled data, закрытый Gemini teacher и полный post-training mixture. Архитектуру Gemma нельзя дополнять деталями Gemini без прямого источника.

## Источники

- [Gemma technical report](https://arxiv.org/abs/2403.08295) — **A**.
- [Gemma 2 technical report](https://arxiv.org/abs/2408.00118) — **A**.
- [Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — **A**.
- [Google developer guide to Gemma 3](https://developers.googleblog.com/en/introducing-gemma3/) — **B**.
- [Google DeepMind Gemma repository](https://github.com/google-deepmind/gemma) и [Hugging Face docs](https://huggingface.co/docs/transformers/model_doc/gemma3) — code **A/B**, serving **C**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
