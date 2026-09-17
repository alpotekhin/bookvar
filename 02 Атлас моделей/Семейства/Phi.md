---
title: "Phi"
type: model-family
organization: Microsoft
first_release: 2023-06
latest_verified_release: Phi-4 family
last_verified: 2026-08-06
architecture_base: compact decoder-only Transformer
modalities: [text, image, audio]
status: active
---

# Phi

## Место в истории

Phi — наиболее чистый пример data-centric линии: Phi-1 показал, что маленькая модель может хорошо писать код после обучения на отфильтрованных и синтетических «учебниковых» данных; Phi-1.5 перенёс идею на common-sense; Phi-2 масштабировал до 2.7B; Phi-3 добавил семейство размеров и 128K variants; Phi-4 усилил reasoning через data mixture и post-training, а mini/multimodal ветви добавили tools, vision и speech. Рост здесь нельзя объяснять новой attention-архитектурой.

## Таблица релизов

| Релиз | Architecture | Context/modalities | Главный training diff | Evidence |
|---|---|---|---|---|
| Phi-1 / 1.5 (2023) | dense decoder, 1.3B | 2K; text/code | filtered web + synthetic textbooks/exercises | **A** |
| Phi-2 (2023-12) | dense 2.7B | 2K; text/code | 1.4T tokens, curriculum/data curation; base checkpoint | **B** |
| Phi-3 Mini/Small/Medium (2024) | dense 3.8B/7B/14B; GQA varies by size | 4K и 128K variants; text | 3.3T heavily filtered/synthetic tokens; SFT + DPO/safety | **A/B** |
| Phi-3 Vision (2024) | vision encoder/projector + Phi decoder | image+text, 128K | multimodal instruction tuning | **B** |
| Phi-4 (2024-12) | dense 14B | 16K; text | целевые синтетические данные; PTS-пары для DPO; SFT/DPO | **A** |
| Phi-4 mini (2025-02) | dense 3.8B, GQA | 128K; multilingual text | reasoning + function calling post-training | **B** |
| Phi-4 multimodal (2025-02) | 5.6B unified text/vision/speech model with modality encoders/adapters | 128K; image/audio+text | joint/specialized multimodal stages | **A/B** |

## Неизменное ядро и tokenizer

Основная text-линия — обычный causal Transformer с RoPE/RMSNorm или LayerNorm-вариантами и gated/non-gated MLP по поколению. Phi-3 использует vocabulary 32K, Phi-4 — 100K-class tokenizer; configs важнее семейного ярлыка. Long-context версии применяют отдельные RoPE/long-context настройки и не равны коротким checkpoints по памяти или качеству.

## Pre-training, post-training и reasoning

Ключевой метод — создание «textbook-quality» данных: фильтрация источников по образовательной ценности и синтетические учебники/задачи. Phi-3 смешивает web и synthetic corpora, а Phi-4 отдельно проектирует данные рассуждений и использует Pivotal Token Search (PTS) на стадии обучения предпочтениям. Это сильная гипотеза о качестве данных, но не доказательство, что масштаб параметров не нужен вообще. Phi-4 post-training использует SFT, preference optimization и safety data; function calling у mini — обученный output protocol.

PTS ищет не редкие токены, а решения, после которых резко меняется вероятность правильного ответа. Из префиксов сэмплируют продолжения и проверяют их тестами или эталонным ответом; затем рекурсивно делят участки последовательности, локализуя значимые изменения. В DPO-паре запросом становится исходный вопрос вместе с общим префиксом, а предпочтительным и отвергаемым продолжениями — два отдельных токена с более высокой и низкой вероятностью дальнейшего успеха. Например, условные оценки 0.8 и 0.3 для двух следующих токенов дают сигнал в пользу первого; это не их вероятности по модели. Поиск не гарантирует обнаружения всех важных позиций. Такой DPO дополняется отдельной стадией с полноразмерными ответами. См. [Phi-4, §4.3](https://arxiv.org/html/2412.08905v1#S4.SS3) и [[00 Учебник/12 Post-training и Alignment/05 DPO|вывод DPO]].

## Inference и serving

Размеры 1.3B–14B делают Phi удобным для edge, CPU/GPU quantization и domain fine-tuning, но 128K KV-cache может доминировать даже у 3.8B. Phi-4 multimodal добавляет modality encoders и заметно отличается от text-only latency/memory. Microsoft публикует checkpoints в Hugging Face/Azure/GitHub Models; конкретные лицензия, trust-remote-code, attention backend и chat template проверяются в model card.

## Визуальный первоисточник: data scaling вместо нового блока

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/phi-data-optimal-scaling.png]]

На Figure 3 по горизонтали отложено число параметров в миллиардах, по вертикали — процент ошибок MMLU; обе оси логарифмические. Синие точки соответствуют Phi-1.5, Phi-2, Phi-3-mini и Phi-3-small, зелёные — Llama 2 7B, 13B, 34B и 70B. Фраза оригинальной подписи об одинаковых фиксированных данных относится к моделям внутри сравниваемой линии Llama 2, а не к общему корпусу Phi и Llama 2. Для Phi авторы подбирали отфильтрованные веб-данные и синтетические данные с учётом масштаба модели. График сопоставляет результаты разных рецептов; это не контролируемая абляция, выделяющая отдельный причинный вклад фильтрации или curriculum. Автор: Microsoft Phi-3 Team. Источник: Figure 3,
[Phi-3 Technical Report](https://arxiv.org/pdf/2404.14219).
Файл перенесён без изменения из официального PDF extraction; лицензия рисунка
отдельно не указана. Проверено 2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** papers Phi-1, Phi-3 и Phi-4, model cards/configs большинства checkpoints, часть training mixture statistics. **Неизвестно:** полный synthetic generation pipeline/prompts, точные corpora и dedup lists, полный safety/RL mixture и вклад закрытых teacher models. «Textbook quality» — operational filtering recipe, а не доступный воспроизводимый dataset.

## Источники

- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) и [Textbooks Are All You Need II](https://arxiv.org/abs/2309.05463) — **A**.
- [Phi-2 model card](https://huggingface.co/microsoft/phi-2) — official card, **B**.
- [Phi-3 technical report](https://arxiv.org/abs/2404.14219) — **A**.
- [Phi-4 technical report](https://arxiv.org/abs/2412.08905) — **A**.
- [Microsoft Phi-4 mini and multimodal announcement](https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037) — **B**.
- [Hugging Face Phi-4 docs](https://huggingface.co/docs/transformers/model_doc/phi4_multimodal) — implementation/serving, **C**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
