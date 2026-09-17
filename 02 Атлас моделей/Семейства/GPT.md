---
title: "GPT"
type: model-family
organization: OpenAI
first_release: 2018
latest_verified_release: GPT-5.6 Sol, Terra and Luna
last_verified: 2026-08-06
architecture_base: decoder-only Transformer
modalities: [text, image, audio]
status: active-partially-closed
---

# GPT

Название GPT объединяет две разные истории. Первая — документированная архитектурная линия GPT-1, GPT-2 и GPT-3, описанная papers. Вторая — закрытые продуктовые модели после GPT-3, для которых опубликованы возможности и API, но не полная архитектура, данные и training recipe. Атлас не должен заполнять этот разрыв догадками.

## От transfer learning к in-context learning

| Релиз | Масштаб | Главный сдвиг |
|---|---:|---|
| GPT-1 (2018) | 117M | Авторегрессионное предобучение на BooksCorpus, затем supervised fine-tuning для задач понимания. |
| GPT-2 (2019) | 1.5B | WebText, 50,257-token byte-level BPE, zero-shot постановка задач как продолжение текста. |
| GPT-3 (2020) | 175B | Масштаб и few-shot/in-context learning без обновления весов. |
| InstructGPT (2022) | 1.3B / 6B / 175B | Исследовательские модели: SFT, модель награды и PPO; размеры политик раскрыты в статье. |
| ChatGPT (с 2022) | зависит от версии продукта; точный размер не раскрыт | Диалоговый продукт нельзя отождествлять с одним из исследовательских checkpoint InstructGPT. |
| GPT-4 — GPT-5.6 | закрыто | Мультимодальные и reasoning-возможности известны по system cards/API; точный блок и корпус не опубликованы. GPT-5.6 Sol, Terra и Luna вышли 9 июля 2026 года. |

## Архитектурная основа

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/gpt-decoder-block.png]]

*Рисунок: Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), CC BY-NC-SA 4.0; локальная копия. Иллюстрация раскрывает paper-level decoder: токены и позиции превращаются в векторы, проходят повторяемые masked self-attention и MLP-блоки, затем проецируются в logits словаря.*

GPT использует только декодер Transformer. Causal mask запрещает позиции видеть будущие токены, поэтому одна и та же objective — предсказание следующего токена — подходит и обучению, и генерации. GPT-1 сохранил post-norm блок исходного Transformer; GPT-2 перенёс LayerNorm перед подслоями и добавил финальную нормализацию, что облегчило обучение глубокой модели. GPT-3 в основном масштабировал этот рецепт и применял alternating dense/local attention в части слоёв.

## Токенизация, данные и обучение

GPT-1 использовал BooksCorpus и словарь около 40 тысяч BPE-токенов. GPT-2 перешёл к byte-level BPE: любая строка представима байтами, а частые последовательности сжимаются в токены. WebText собирался по исходящим ссылкам популярных Reddit-публикаций — важный пример того, как правило отбора становится частью модели. GPT-3 обучался на смеси Common Crawl, WebText2, Books и Wikipedia; paper раскрывает источники и веса смеси, но не все документы.

Предобучение минимизирует cross-entropy следующего токена. Fine-tuning GPT-1 добавлял формат задачи и supervised loss. GPT-3 показал другой интерфейс: инструкция и примеры помещаются в контекст, а веса не меняются. Это свойство возникает из pre-training, но не означает, что модель была специально оптимизирована под диалог.

## Post-training и serving

InstructGPT отделил базовую модель от policy: демонстрации людей дают SFT, ранжирования обучают reward model, PPO оптимизирует ответы с KL-ограничением. Поздние GPT используют более развитые методы, детали которых раскрыты частично. Для вывода decoder-only модель хранит KV-cache каждого слоя; стоимость prefill растёт с длиной входа, decode — с числом сохранённых ключей и значений.

## Что опубликовано и что неизвестно

Для GPT-1/2/3 опубликованы статьи, размеры, описания данных и архитектурные параметры. Для GPT-4 и более поздних коммерческих систем OpenAI публикует system cards, API и оценки безопасности, но не полный состав архитектуры, число параметров, корпус и recipe. Продуктовая часть здесь ограничена GPT-5.6 (Sol, Terra и Luna) от 9 июля 2026 года; это не утверждение, что более новых моделей нет. Точные утверждения о MoE, числе слоёв или tokenizer без официального источника остаются неподтверждёнными.

Открытость артефактов неодинакова. Для [GPT-1](https://github.com/openai/finetune-transformer-lm) опубликованы код и предобученные веса, но сам репозиторий не распространяет весь BooksCorpus. Для [GPT-2](https://github.com/openai/gpt-2) доступны код выборки, tokenizer и веса четырёх размеров; исходный WebText целиком не опубликован. Репозиторий [GPT-3](https://github.com/openai/gpt-3) содержит примеры генерации, статистику корпуса и данные отдельных синтетических тестов, а не веса 175B и полный обучающий конвейер. Поэтому возможность реализовать описанный блок не равна воспроизведению оригинального обучения.

Механизм переноса поведения разобран в [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT и instruction data]], [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|reward modeling]] и [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|PPO для LLM]]. В [InstructGPT, §3.5](https://arxiv.org/html/2203.02155v1#S3.SS5) политики имеют 1.3B, 6B и 175B параметров, а модель награды — 6B; число параметров policy нельзя подставлять вместо размера проверяющей модели.

## Источники

- [GPT-1: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- [InstructGPT](https://arxiv.org/abs/2203.02155) — SFT/RLHF pipeline.
- [GPT-5.6 release](https://openai.com/index/gpt-5-6/) — официальный анонс рассматриваемого закрытого поколения; архитектурных деталей не раскрывает.
