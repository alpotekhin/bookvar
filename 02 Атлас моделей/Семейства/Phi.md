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

Работы Phi исследуют, какое качество можно получить у небольшой модели
при отборе и синтетическом создании обучающих данных. Phi-1 обучалась
программированию на отфильтрованных текстах и учебных примерах;
Phi-1.5 расширила набор задач, Phi-2 — размер до 2.7B. У Phi-3 появились
несколько размеров и версии с контекстом 128K. Phi-4 меняет состав данных
и дообучение рассуждениям; ветви mini и multimodal добавляют работу
с инструментами, зрением и речью.

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

Текстовая линия использует причинный Transformer. Выбор LayerNorm или RMSNorm, наличие затвора в MLP и параметры RoPE зависят от поколения. У Phi-3 словарь порядка 32K токенов, у Phi-4 — порядка 100K; точный размер берут из конфигурации выбранных весов. Версии для длинного контекста используют отдельные настройки RoPE и обучение. Их расход памяти и качество на длинных текстах нельзя выводить из характеристик короткоконтекстной версии.

## Pre-training, post-training и reasoning

Авторы отбирают источники по образовательной ценности и генерируют учебные тексты и задачи; в статьях это называется «textbook-quality» данными. Phi-3 смешивает веб-корпуса и синтетические примеры. Phi-4 дополнительно использует данные рассуждений и Pivotal Token Search (PTS) для построения предпочтений. При дообучении применяются SFT, оптимизация по предпочтениям и данные безопасности. Вызов функций у mini также требует обучения модели выдавать ответ в заданном формате.

PTS ищет не редкие токены, а решения, после которых резко меняется вероятность правильного ответа. Из префиксов сэмплируют продолжения и проверяют их тестами или эталонным ответом; затем рекурсивно делят участки последовательности, локализуя значимые изменения. В DPO-паре запросом становится исходный вопрос вместе с общим префиксом, а предпочтительным и отвергаемым продолжениями — два отдельных токена с более высокой и низкой вероятностью дальнейшего успеха. Например, условные оценки 0.8 и 0.3 для двух следующих токенов дают сигнал в пользу первого; это не их вероятности по модели. Поиск не гарантирует обнаружения всех важных позиций. Такой DPO дополняется отдельной стадией с полноразмерными ответами. См. [Phi-4, §4.3](https://arxiv.org/html/2412.08905v1#S4.SS3) и [[00 Учебник/12 Post-training и Alignment/05 DPO|вывод DPO]].

A worked account of the original Phi-1 datasets and the Phi-4 generation,
instruction-reversal and repetition experiments is in
[[00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы#phi-synthetic-recipes|Synthetic data: Phi recipes]].

## Inference и serving

Модели 1.3B–14B можно рассматривать для локального запуска, квантизации на CPU/GPU и предметного дообучения. Однако кэш контекста 128K может занимать больше памяти, чем веса компактной модели. Phi-4 multimodal дополнительно запускает кодировщики модальностей; её задержку и память измеряют отдельно от текстовой версии. Microsoft публикует веса и варианты доступа через Hugging Face, Azure и GitHub Models. Для выбранного варианта проверяют лицензию, необходимость `trust_remote_code`, реализацию внимания и шаблон диалога.

## Визуальный первоисточник: data scaling вместо нового блока

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/phi-data-optimal-scaling.png]]

На Figure 3 по горизонтали отложено число параметров в миллиардах, по вертикали — процент ошибок MMLU; обе оси логарифмические. Синие точки соответствуют Phi-1.5, Phi-2, Phi-3-mini и Phi-3-small, зелёные — Llama 2 7B, 13B, 34B и 70B. Фраза оригинальной подписи об одинаковых фиксированных данных относится к моделям внутри сравниваемой линии Llama 2, а не к общему корпусу Phi и Llama 2. Для Phi авторы подбирали отфильтрованные веб-данные и синтетические данные с учётом масштаба модели. График сопоставляет результаты разных рецептов; это не контролируемая абляция, выделяющая отдельный причинный вклад фильтрации или curriculum. Автор: Microsoft Phi-3 Team. Источник: Figure 3,
[Phi-3 Technical Report](https://arxiv.org/pdf/2404.14219).
Файл перенесён без изменения из официального PDF extraction; лицензия рисунка
отдельно не указана. Проверено 2026-07-20.

## Опубликовано и неизвестно

Опубликованы статьи Phi-1, Phi-3 и Phi-4, карточки и конфигурации большинства моделей, часть статистик смеси данных. Не раскрыты все этапы и запросы генерации синтетических примеров, полные корпуса и списки дубликатов, все данные безопасности и RL, вклад закрытых моделей-учителей. Название «textbook quality» описывает отбор данных, но не обозначает полностью доступный набор, на котором можно повторить обучение.

## Источники

- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) и [Textbooks Are All You Need II](https://arxiv.org/abs/2309.05463) — **A**.
- [Phi-2 model card](https://huggingface.co/microsoft/phi-2) — official card, **B**.
- [Phi-3 technical report](https://arxiv.org/abs/2404.14219) — **A**.
- [Phi-4 technical report](https://arxiv.org/abs/2412.08905) — **A**.
- [Microsoft Phi-4 mini and multimodal announcement](https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037) — **B**.
- [Hugging Face Phi-4 docs](https://huggingface.co/docs/transformers/model_doc/phi4_multimodal) — implementation/serving, **C**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
