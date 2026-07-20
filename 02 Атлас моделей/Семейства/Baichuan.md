---
title: "Baichuan"
type: model-family
organization: Baichuan Intelligence
first_release: 2023
latest_verified_release: Baichuan-M3-235B
last_verified: 2026-07-20
architecture_base: decoder-only Transformer and medical models
modalities: [text, image, audio]
status: active-specialized
---

# Baichuan

История Baichuan распадается на две фазы. Baichuan 1/2 — открытые китайско-английские универсальные decoder-only модели 2023 года. Начиная с Baichuan-Omni, M1/M2/M3 организация сместила центр серии к мультимодальным и медицинским моделям. Нельзя считать M3 просто «Baichuan 2 большего размера»: у неё другая область применения и существенно иная recipe.

## Линия релизов

| Релиз | Открытые сведения | Архитектурный/учебный смысл |
|---|---|---|
| Baichuan-7B / 13B (2023) | 1.2–1.4T токенов, 4K; 7B RoPE, 13B ALiBi | Уже внутри поколения positional scheme зависела от масштаба. |
| Baichuan 2 7B / 13B (2023) | 2.6T токенов, SentencePiece 125,696, SwiGLU | Более подробно раскрыты очистка данных, safety и промежуточные checkpoints. |
| Baichuan-Omni-1.5 / Audio (2025) | текст, изображение, видео и аудио | Отдельная мультимодальная ветка, не diff одного decoder-блока. |
| Baichuan-M1 14B (2025) | медицинские base/instruct веса | Специализация корпуса и post-training важнее масштаба. |
| Baichuan-M2 32B (2025) | медицинское reasoning | Продолжение клинической линии. |
| Baichuan-M3 235B (2026) | открытые base/quantized веса и technical report | Модель клинического опроса и принятия решений; latest verified release. |

## Что показал Baichuan 2

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/baichuan2-data-and-config.png]]

*Рисунок: Yang et al., [Baichuan 2 Technical Report](https://arxiv.org/abs/2309.10305), Figure 2 и Table 3; локальная копия. Верхняя Sankey-схема показывает, сколько корпуса отбрасывают дедупликация и фильтры. Таблица фиксирует различие RoPE/ALiBi между 7B и 13B.*

Baichuan 2 использует pre-norm decoder Transformer, SwiGLU и memory-efficient attention. SentencePiece обучен на том же многоязычном корпусе; числа разбиваются на цифры, а редкие символы имеют byte fallback. Это важнее абстрактного «поддерживает китайский»: разбиение определяет длину последовательности и представление редких иероглифов.

Данные проходят точную и MinHash-дедупликацию, эвристические, качественные и safety-фильтры. 2.6T токенов описывают объём после обработки, а не сырой web. Base и Chat — разные checkpoints: conversational behavior появляется после SFT и preference/safety alignment, не из decoder-only блока.

## Современная медицинская ветка

M1–M3 обучаются не только отвечать на медицинский вопрос, но и запрашивать недостающие сведения. В Baichuan-M3 акцент на *clinical inquiry*: модель должна выбирать следующий вопрос и обновлять решение по ответам пациента. Это требует специализированных траекторий и RL, поэтому сравнивать её лишь по общему MMLU с Baichuan 2 бессмысленно. На Hugging Face 9 февраля 2026 года опубликованы M3-235B, FP8, GPTQ и GGUF-варианты.

## Serving и границы знания

Для Baichuan 2 экосистема поддерживает Transformers и квантизацию, но нестандартно большой словарь увеличивает embedding/output head. В медицинской линии 235B весов определяют совсем иной класс развёртывания. Подробности M3 следует брать из её report/model card; архитектурные свойства Baichuan 2 нельзя переносить автоматически. Полные списки документов и все клинические данные не опубликованы, а benchmark не заменяет медицинскую валидацию.

## Источники

- [Baichuan 2](https://arxiv.org/abs/2309.10305) и [официальный репозиторий](https://github.com/baichuan-inc/Baichuan2).
- [Baichuan-M3 Technical Report](https://arxiv.org/abs/2602.06570).
- [Baichuan official model collection](https://huggingface.co/baichuan-inc) — проверка релизов M1–M3 и Omni.
