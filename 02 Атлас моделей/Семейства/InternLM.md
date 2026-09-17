---
title: "InternLM"
type: model-family
organization: Shanghai AI Laboratory
first_release: 2023
latest_verified_release: InternLM3-8B-Instruct
last_verified: 2026-08-06
architecture_base: decoder-only Transformer
modalities: [text]
status: active-ecosystem
---

# InternLM

InternLM — одновременно линейка двуязычных моделей и открытая инженерная экосистема: InternEvo для обучения, XTuner для дообучения, LMDeploy для serving и AgentLego для инструментов. Эти компоненты важны для воспроизводимости, но не являются слоями нейросети. Архитектуру, данные и агентное поведение нужно разделять.

## Релизы

| Релиз | Что изменилось |
|---|---|
| InternLM 7B/20B (2023) | Базовые китайско-английские decoder-модели; открыты base/chat веса. |
| InternLM2 1.8B/7B/20B (2024) | 200K context, GQA, более сильное tool use через специальный формат и данные. |
| InternLM2.5 (2024) | 1.8B/7B/20B, отдельный 1M-context checkpoint, улучшены reasoning и code. |
| InternLM3-8B-Instruct (15.01.2025) | 4T токенов; normal и deep-thinking режимы в одном instruct checkpoint. |

Карточка охватывает основную текстовую линию до InternLM3-8B-Instruct; это ограничение рассмотрения, а не проверка отсутствия более новых релизов. InternLM-XComposer развивается отдельно как мультимодальное семейство.

## Данные как воспроизводимая часть модели

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/internlm-data-pipeline.png]]

*Рисунок: InternLM Team, [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297), Figure 3; локальная копия. Конвейер последовательно извлекает текст и язык, нормализует, дедуплицирует, отсекает unsafe domains/слова/токсичность и применяет классификаторы качества.*

Рисунок объясняет, почему «обучалась на web» недостаточно. MinHash удаляет почти дублирующиеся документы, разные фильтры нужны для книг, technical text и web, а safety применяется до генеративного alignment. InternLM2 раскрывает двуязычные источники и training system; InternLM3 сообщает 4T high-quality токенов, но не полный перечень документов.

## Архитектура и длинный контекст

InternLM2 — pre-norm decoder Transformer с RoPE, SwiGLU и GQA. Вместо порядка «все Q, затем все K, затем все V» проекция QKV укладывает рядом запросы одной KV-группы и соответствующие K/V; это важно при разбиении весов по выходным каналам. Long-context достигается не названием GQA: требуется изменение RoPE и продолжение обучения на длинных последовательностях. Поэтому 200K/1M относятся к конкретным checkpoints.

В [InternLM2-7B config](https://huggingface.co/internlm/internlm2-7b/blob/main/config.json) 32 головы Q, 8 голов K/V и скрытая ширина 4096, то есть размер головы 128 и четыре Q на KV-группу. [Реализация `InternLM2Attention`](https://huggingface.co/internlm/internlm2-7b/blob/main/modeling_internlm2.py) преобразует выход `wqkv` из [B,T,6144] в [B,T,8,6,128]: шесть позиций группы — четыре Q, затем K и V. После извлечения Q имеют форму [B,32,T,128], K/V — [B,8,T,128]. Число 6144 получается как 8×(4+2)×128, не 3×4096.

В маленьком примере с четырьмя Q и двумя KV порядок голов — [Q0,Q1,K0,V0,Q2,Q3,K1,V1]. Две равные непрерывные части содержат по целой группе, поэтому устройство получает нужные запросы вместе с их K/V. В порядке [Q0,Q1,Q2,Q3,K0,K1,V0,V1] такое же слепое разрезание пополам отделило бы запросы от ключей. Это объясняет удобство группового размещения для tensor parallelism; при смене самого числа KV-голов требуется корректное преобразование checkpoint, а не одна замена поля конфигурации.

Инструменты реализованы через формат, где модель генерирует структурированные вызовы и получает наблюдения обратно. Это post-training поверх LM. InternLM3 добавляет переключаемое «глубокое рассуждение»: контроль режима относится к обучению policy и шаблону, а не к новому attention-оператору.

## Serving

LMDeploy поддерживает tensor parallelism, KV-cache management и quantization для семейства. GQA сокращает KV-cache по сравнению с MHA. Для 1M-context checkpoint всё равно критичны prefill, память и реальные тесты retrieval: максимальная длина конфигурации не гарантирует одинаковую точность по всему окну.

## Известное и неизвестное

Опубликованы InternLM2 report, код платформы и большинство весов; архитектура InternLM3 доступна через config/model card, но полный новый technical report и corpus раскрыты слабее. Заявления об agent/tool skill не следует переносить на base checkpoint.

## Источники

- [InternLM official repository](https://github.com/InternLM/InternLM) — verified release log.
- [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297).
- [InternLM3-8B-Instruct model card](https://huggingface.co/internlm/internlm3-8b-instruct).
- [LMDeploy](https://github.com/InternLM/lmdeploy) — официальный serving stack.
