---
title: "Gemma"
type: model-family
organization: Google DeepMind
first_release: 2024-02
latest_verified_release: Gemma 3 and specialized Gemma 3 variants
last_verified: 2026-08-06
architecture_base: compact decoder-only Transformer
modalities: [text, image]
status: active
---

# Gemma

## Место в истории

Gemma — семейство Google DeepMind с опубликованными весами; его устройство
нельзя считать описанием закрытой Gemini. Первое поколение включает плотные
модели 2B/7B. Gemma 2 чередует локальное и полное внимание, а для части
размеров использует дистилляцию уже при предобучении. Gemma 3 добавляет
зрительный вход в старших размерах, контекст до 128K и многоязычное обучение.
PaliGemma, CodeGemma, RecurrentGemma, ShieldGemma и MedGemma — отдельные
специализированные ветви.

## Таблица релизов

| Релиз | Architecture diff | Tokenizer/context/modalities | Training/post-training | Evidence |
|---|---|---|---|---|
| Gemma 1 (2024-02) | dense decoder; MQA у 2B, MHA у 7B; GeGLU/RMSNorm/RoPE | SentencePiece 256K; 8K; text | 2T/6T tokens; IT variants | **A** |
| Gemma 2 (2024-06; 2B добавлена позднее) | GQA; alternating local 4K/global attention; logit soft-capping; 2B/9B/27B | 8K; text | distillation при предобучении 2B и 9B; 27B обучена без этой дистилляции; SFT + RLHF | **A** |
| Gemma 3 (2025-03) | 5 local : 1 global pattern; SigLIP vision encoder for 4B+; QK norm | 1B text-only 32K; 4B/12B/27B multimodal 128K; 140+ languages | distillation; multimodal and long-context stages; IT/function calling | **A/B** |
| Gemma 3n / domain variants (2025+) | mobile-efficient or domain adapters/encoders | model-specific text/image/audio | specialized recipes; not a Gemma 4 claim | **B** |

## Неизменное ядро и diff

Основная линия использует причинный декодер с pre-norm, RoPE и MLP с затвором. В Gemma 2 [[00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание|локальные слои читают ограниченное окно]], а слои полного внимания связывают все позиции запроса. Soft-capping ограничивает логиты с помощью `tanh`; его применяют к оценкам внимания и выходным логитам. Gemma 3 увеличивает долю локальных слоёв и добавляет зрительный кодировщик со сжатием до фиксированного числа визуальных токенов на фрагмент изображения. Зрительный вход есть у 4B/12B/27B, но отсутствует у 1B.

## Tokenizer, данные и post-training

Словарь содержит 256K токенов. Отчёты называют объём и категории данных, фильтры безопасности и применение дистилляции, но не дают полного списка документов. Меньшие Gemma 2 учатся по логитам модели-учителя уже при предобучении; их результаты зависят и от этой задачи обучения. Версии для выполнения инструкций проходят SFT и обучение по обратной связи. Следование заданному формату ответа и вызовы функций требуют соответствующего дообучения; зрительный кодировщик SigLIP сам по себе этим не обучает.

## Inference и serving

Локальное внимание уменьшает вычисления и кэш, если движок учитывает размер окна каждого слоя. Для контекста 128K отдельно рассчитывают память слоёв полного внимания. Google публикует квантизованные варианты и интеграции с Keras, Transformers, Gemma.cpp и облачными средами. Запрос с изображением дополнительно запускает зрительный кодировщик и вводит визуальные токены, поэтому его задержка отличается от текстового запроса. Условия Gemma отличаются от Apache/MIT и проверяются перед распространением.

<a id="визуальный-первоисточник-где-проявился-data-centric-выигрыш"></a>
## Визуальный первоисточник: сравнение результатов Gemma 1

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/gemma-capability-comparison.png]]

График Gemma 1 сравнивает 7B с Llama 2 и Mistral по агрегированным категориям:
наиболее заметен разрыв в math/science и coding, а не в обычном QA. Это наблюдение о результатах разных готовых моделей, а не абляция данных:
одновременно различаются архитектура, корпус и обучение. Поэтому график не
позволяет выделить причинный вклад качества данных или объяснить результаты
Gemma 2 дистилляцией. Режимы предобучения Gemma 2 (дистиллированные 2B/9B и
недистиллированная 27B) подтверждаются отдельными источниками в таблице выше. Автор: Gemma Team,
Google DeepMind. Источник: Figure 1,
[Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/pdf/2403.08295).
Файл перенесён без изменения из официального PDF extraction. Использование
весов регулируют Gemma Terms; лицензия самого рисунка отдельно не указана.
Проверено 2026-07-20.

## Опубликовано и неизвестно

Опубликованы три технических отчёта, веса, конфигурации, токенизатор, карточки моделей и эталонные реализации. Не раскрыты все обучающие документы, точные доли синтетических данных и данных дистилляции, устройство закрытой модели-учителя Gemini и полная смесь задач дообучения. Архитектуру Gemma нельзя дополнять деталями Gemini без прямого источника.

## Источники

- [Google: обзор архитектур Gemma](https://developers.googleblog.com/gemma-explained-overview-gemma-model-family-architectures/) — различия MQA/MHA в первом поколении и дистилляция Gemma 2.

- [Gemma technical report](https://arxiv.org/abs/2403.08295) — **A**.
- [Gemma 2 technical report](https://arxiv.org/abs/2408.00118) — **A**.
- [Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — **A**.
- [Google developer guide to Gemma 3](https://developers.googleblog.com/en/introducing-gemma3/) — **B**.
- [Google DeepMind Gemma repository](https://github.com/google-deepmind/gemma) и [Hugging Face docs](https://huggingface.co/docs/transformers/model_doc/gemma3) — code **A/B**, serving **C**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
