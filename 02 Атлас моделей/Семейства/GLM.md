---
title: "GLM"
type: model-family
organization: Zhipu AI, Z.ai and Tsinghua KEG
first_release: 2021
latest_verified_release: GLM-5.1 (scope of this card)
last_verified: 2026-09-15
architecture_base: autoregressive blank infilling; later causal MoE
modalities: [text, image]
status: active
---

# GLM

## Место в истории

Исходный GLM 2021 года восстанавливает пропуски в тексте: видит
неповреждённую часть с обеих сторон и генерирует удалённые фрагменты
авторегрессионно. Два позиционных индекса различают место пропуска
в исходном тексте и положение внутри восстанавливаемого фрагмента.
GLM-130B масштабирует этот подход, а ChatGLM развивает двуязычное
диалоговое применение.

В GLM-4/4.5 расширяются контекст и работа с инструментами; GLM-4.5
публикуется как крупная MoE-модель. GLM-5 увеличивает её размер и добавляет
[[00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание#dsa-content-adaptive-selection|DeepSeek Sparse Attention]]. GLM-5.1 предназначен для длительных инженерных
задач с агентами. В официальной таблице у GLM-5 и 5.1 одинаковый размер
744B-A40B; из этого ещё нельзя заключить, что их архитектура и предобучение
полностью совпадают.

## Релизы как diff

| Релиз | Architecture | Context/modalities | Training/post-training | Evidence |
|---|---|---|---|---|
| GLM (2021) | autoregressive blank infilling, 2D positions | text | смешение целей NLU/generation | **A** |
| GLM-130B (2022) | 70-layer dense bidirectional GLM | 2K; bilingual text | 400B tokens; documented scaling/stability | **A** |
| ChatGLM-6B (2023) | GLM dialogue adaptation; INT4/8 support | 2K→32K variants; Chinese/English | instruction/dialogue tuning | **A/B** |
| ChatGLM2/3 (2023) | MQA, longer context; tool/agent formats | до 32K/128K by variant | stronger alignment and tool use | **B** |
| GLM-4 / 4V (2024) | proprietary text backbone; separate vision line | 128K and multimodal variants | All Tools, browsing/code interpreter | **A/B; часть architecture закрыта** |
| GLM-4.5 (2025-07) | 355B/32B active MoE; Air 106B/12B; GQA | 128K text | 23T tokens; thinking/non-thinking; agentic RL | **A/B** |
| GLM-5 (2026) | 744B/40B active; DeepSeek Sparse Attention | long-context text | 28.5T tokens; asynchronous RL via slime | **A/B** |
| GLM-5.1 (2026-04) | 744B-A40B по официальной таблице весов; равенство размеров не доказывает идентичность backbone | agentic text/coding | заявлены улучшения длительной инженерной работы; полный diff обучения здесь не установлен | **B** |

Проверка 15 сентября 2026 года относится к поколениям до GLM-5.1 включительно. В [официальном каталоге](https://github.com/zai-org/GLM-5) уже перечислены также 5.2, 5.3 и 5.3-Flash; они не входят в подробный архитектурный охват этой карточки. Поэтому поле `latest_verified_release` означает верхнюю границу рассмотрения, не последний доступный продукт.

## Неизменное ядро и перелом линии

Название GLM сохранилось при изменении моделей. Диалоговый интерфейс
ChatGLM/GLM-4.x/5 сам по себе не подтверждает сохранение исходной задачи
восстановления пропусков и двумерных позиций. У GLM-4.5 раскрыты
разреженная MoE и режимы ответа с рассуждением и без него.
В GLM-5 увеличены общее и активное число параметров; индексатор DSA
выбирает подмножество ключей для внимания к длинному контексту.

## Tokenizer, данные и post-training

Токенизаторы, служебные токены и шаблоны диалога менялись между поколениями
ChatGLM. Ранний код, загружаемый вместе с весами, нельзя применять
к современной версии без проверки совместимости. Отчёт GLM-4.5 указывает
23T токенов предобучения, GLM-5 — 28.5T, но полный состав корпусов
не опубликован. Режимы рассуждения формируются дообучением и выбираются
запросом. Обучение агентной работе в 5/5.1 использует среды с инструментами
и инфраструктуру RL `slime`; оценки на бенчмарках не раскрывают состав
этих обучающих задач.

## Inference и serving

Квантизованные версии раннего ChatGLM рассчитаны в том числе на
пользовательские GPU. Для GLM-4.5/5 нужно разместить 355B/744B весов,
хотя на токен активны 32B/40B. При распределении экспертов между GPU
требуются пересылки токенов. DSA сокращает работу внимания на длинном
контексте, но для этого нужны поддерживающие его ядра. Версии весов
в BF16 и FP8 различаются расходом памяти и требованиями к оборудованию.

## Визуальный первоисточник: семейство, а не одна линия checkpoints

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/glm-family-timeline.png]]

Хронология до GLM-4 разделена на четыре колонки: API, открытые языковые модели,
открытые мультимодальные модели и агенты. GLM-4V и All Tools относятся
к разным параллельно развивающимся ветвям.
Автор: GLM-4 Team / Zhipu AI / Tsinghua KEG. Источник: Figure 1, p. 2,
[ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools](https://arxiv.org/pdf/2406.12793).
Локальный файл — crop официального PDF; лицензия рисунка отдельно не указана.
Проверено 2026-07-20.

## Опубликовано и неизвестно

Опубликованы статьи GLM/130B/GLM-4/4.5/5, официальные репозитории, конфигурации и часть весов. Для некоторых закрытых версий GLM-4, доступных через API, точная архитектура неизвестна. Не раскрыты полные смеси данных и задач RL. В этой карточке также не установлено, изменилось ли предобучение 5.1 или основные отличия внесены при дообучении; результаты SWE-Bench этого не определяют (**C**).

## Источники

- [Конфигурация GLM-4.5, закреплённая ревизия](https://huggingface.co/zai-org/GLM-4.5/blob/8b91a96cb5e3a6dde04be29567b87b06f3dd61dc/config.json) — 96 query-голов и 8 KV-голов: каждая группа из 12 query-голов использует общие ключи и значения. Это GQA, не латентный KV-кэш MLA.

- [GLM: General Language Model Pretraining](https://arxiv.org/abs/2103.10360) — **A**.
- [GLM-130B](https://arxiv.org/abs/2210.02414) — **A**.
- [ChatGLM-6B repository](https://github.com/THUDM/ChatGLM-6B) — **A/B**.
- [GLM-4 report](https://arxiv.org/abs/2406.12793) — **A**.
- [GLM-4.5 official repository](https://github.com/zai-org/GLM-4.5) — report/weights, **A/B**.
- [GLM-5/5.1 official repository](https://github.com/zai-org/GLM-5) и [GLM-5 report](https://arxiv.org/abs/2602.15763) — **A/B**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
