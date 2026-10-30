---
title: DeepSeek
type: model-family
organization: DeepSeek AI
first_release: 2023
latest_verified_release: DeepSeek-V4 (2026-04-24)
last_verified: 2026-10-29
architecture_base: decoder-only Transformer and sparse MoE
modalities: [text]
status: active
---

# DeepSeek

## Место в истории

В DeepSeek несколько изменений уменьшают разные затраты языковой модели.
DeepSeekMoE заменяет одну FFN набором небольших экспертов: общие работают
для каждого токена, остальные выбирает маршрутизатор. MLA сокращает объём
сохраняемых ключей и значений. V3 объединяет эти решения с обучением в FP8,
новой балансировкой экспертов и дополнительным предсказанием будущих токенов.

R1 сохраняет архитектуру V3, но меняет дообучение: модель учится рассуждать
по результатам проверки своих ответов. V3.2 добавляет разреженное внимание
и обучение агентным задачам с подкреплением. Ниже эти изменения разделены
по архитектуре, обучению и выполнению модели.

## Неизменное ядро

В поколениях V2–V3.2 модель использует причинный декодер с MLA и MoE из небольших
экспертов. Для одного токена вычисляется лишь часть экспертных FFN, а
внимание хранит сжатое представление контекста. R1, V3-0324 и V3.1/3.2
развивают эту основу через разные изменения дообучения, длинного контекста
и вывода.

## Что менялось между релизами

| Релиз | Architecture | Context / tokenizer | Pre/post-training | Evidence |
|---|---|---|---|---|
| DeepSeek LLM (2023) | dense decoder, 7B/67B | 4K, text | 2T tokens; base/chat | **A** |
| DeepSeekMoE (2024-01) | shared + fine-grained routed experts | text | специализация экспертов при меньших active FLOPs | **A** |
| DeepSeek-V2 (2024-05) | MLA + DeepSeekMoE; 236B/21B active | 128K | 8.1T tokens; SFT/RL chat | **A** |
| DeepSeek-V3 (2024-12) | 671B/37B active; loss-free balancing; MTP; FP8 recipe | 128K; 129K vocabulary | 14.8T tokens; SFT + RL | **A** |
| DeepSeek-R1 (2025-01) | V3 base architecture | 128K | R1-Zero: RL без SFT; R1: cold-start, RL, rejection sampling, второй RL; distillation | **A** |
| V3.2-Exp / V3.2 (2025) | DeepSeek Sparse Attention поверх MLA | long-context text | продолжение pre-training; V3.2 масштабирует reasoning и agentic RL | **A/B** |
| DeepSeek-V4 Pro / Flash (2026-04-24) | CSA/HCA вместо MLA; shared-KV MQA; mHC; MoE | 1 048 576 позиций в опубликованной конфигурации Pro | Muon и AdamW для разных параметров; SFT/RL специалистов и on-policy distillation | **A** |

## Механизмы

**MLA.** Ключи и значения восстанавливаются из общего низкоразмерного
представления. При выводе можно хранить это представление и отдельную
позиционную компоненту вместо полных K/V всех голов. Часть преобразований
объединяется с весами проекций — это называют weight absorption.
Для эффективного исполнения этой формы нужны поддерживающие её ядра;
сам механизм можно проверить и обычными тензорными операциями. В отличие от GQA,
MLA обучает само сжатие, а не просто использует общие K/V в группе голов.

**DeepSeekMoE.** Разделение FFN на небольших экспертов увеличивает число
доступных комбинаций. Общие эксперты предназначены для преобразований,
нужных разным токенам. При распределённом выполнении токены пересылаются
на устройства с выбранными экспертами; неравномерная загрузка задерживает
обработку пакета. У V3 на токен активны 37B параметров, но хранить нужно
все веса, включая экспертов, не выбранных этим токеном.

**Обучение V3.** Для балансировки меняются смещения оценок маршрутизатора,
которые влияют на выбор экспертов. Основной способ балансировки не добавляет
штраф в функцию потерь, хотя вспомогательный штраф на уровне последовательности
сохраняется. Multi-token prediction (MTP) добавляет задачи предсказания
дальнейших токенов; эти модули можно использовать и при спекулятивном
декодировании. FP8 определяет точность части вычислений обучения, а не новый
тип слоя.

**DSA.** В V3.2-Exp отдельный lightning indexer оценивает позиции истории
и выбирает подмножество для последующего внимания. Не все движки поддерживают
одни и те же варианты этого вычисления; наличие опубликованной модели
не подтверждает поддержку конкретным сервером.

Подробные выводы распределены по учебным главам, потому что эти механизмы решают разные задачи. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache#Что именно лежит в кэше MHA, GQA и MLA|MLA: содержимое кэша]] объясняет, какие тензоры сохраняются; [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|MoE: routing, capacity и serving]] — какие эксперты вычисляются; [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание#dsa-content-adaptive-selection|DSA: отбор позиций индексатором]] — какие записи истории читает полное внимание. [MTP (§2.2 отчёта V3)](https://arxiv.org/html/2412.19437v2#S2.SS2) — дополнительная цель предсказания будущих токенов; её использование как черновой модели относится к [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|speculative decoding]], но сама MTP не является гарантией ускорения. За проверяемыми наградами следует идти в [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR и verifiers]], а за групповой нормировкой и последовательностью стадий R1 — в [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|GRPO и DeepSeek-R1]].

<a id="deepseek-v4-csa-hca"></a>

## DeepSeek-V4: compression across token positions

V4 заменяет MLA из V2/V3 чередованием CSA и HCA. В
[закреплённой конфигурации Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json)
указаны одна общая KV-голова и коэффициенты сжатия 4 и 128.
Это особенности исходного V4; из них нельзя выводить устройство
последующих закрытых API-обновлений.

<a id="compressed-sparse-attention-compress-select-attend"></a>
<a id="heavily-compressed-attention-read-the-whole-smaller-pool"></a>
<a id="why-both-branches-retain-a-local-window"></a>

Полный разбор перенесён в
[[00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание#compressed-sparse-attention-csa-hca|главу о CSA и HCA]]:
там сохранены оригинальные Figures 3–4, английские source excerpts,
поэлементные веса компрессора, shared-KV attention, причинная граница
блока и численный пример. Здесь остаются конфигурация и отличия релиза.

Источник этих механизмов и чисел: DeepSeek-AI,
[DeepSeek-V4 technical report, §§2.3.1–2.3.3, pp. 9–13](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/89d501aed998d33fa4f4702102ec1bb2331e10f6/DeepSeek_V4.pdf).
Версия PDF закреплена по репозиторию от 6 мая 2026 года;
[MIT License](/licenses/deepseek-v4-89d501ae-MIT.txt) сохранена локально.

### A cache-count example, not a latency prediction

After a prefix of $n=1\,048\,576$ tokens at a completed-block boundary, the
next query has position $t=n$ in zero-based indexing. Its available CSA
pool has $n/4=262\,144$ entries; the HCA pool has $n/128=8192$. The published
Pro configuration has `index_topk=1024`, `sliding_window=128`,
`head_dim=512`, and `index_head_dim=128`. A saturated CSA query reads up to
1024 compressed entries plus its local window. An HCA query reads all 8192
compressed entries plus that window. Thus HCA can read more entries per
query while storing a much smaller long-range pool.

These are counts of logical main-attention entries for one layer, not full
cache bytes. The indexer, overlap/tail state, numeric formats, alignment and
layer schedule still matter. In particular, selecting 1024 entries does not
mean storing only 1024. The published Pro configuration lists 62 compression
rates for 61 decoder layers **plus one MTP layer**. The decoder has 30 CSA
layers and 31 HCA layers; the final zero rate belongs to the sliding-only MTP
layer. Counting that last entry as a decoder layer would give the wrong
architecture. The assignment of layer indices is explicit in the
[official implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py#L782-L793).

For this configuration, top-$k$ caps selected reads, not the total cache stored.
Partial RoPE covers the final 64 of 512 query/KV coordinates; query/KV RMSNorm,
the inverse output rotation, a learnable sink logit and the grouped output
projection are described in the linked mechanism section.

For the other V4 changes, §2.2 gives the mHC residual update
$X_{l+1}=B_lX_l+C_l\mathcal F_l(A_lX_l)$: $B_l$ is constrained to be doubly
stochastic, rather than an unrestricted mixing matrix. §2.4 applies Muon to
most matrix parameters while retaining AdamW for embeddings, the prediction
head, RMSNorm weights and selected mHC parameters. These are separate
changes to residual propagation and optimization, not effects of CSA/HCA.

The authors' relative FLOP/cache estimates depend on the baseline, context
length and storage formats. They are not measured latency on this machine.

## Post-training, reasoning и tools

R1-Zero обучали методом GRPO на проверяемых задачах без предварительного SFT;
в ходе обучения появились длинные рассуждения. Для полного R1 сначала
подготовили небольшую выборку образцовых решений, затем провели RL,
отобрали ответы для SFT на рассуждениях и общих задачах и выполнили
финальную стадию RL. Дистиллированные Qwen/Llama сохраняют собственную
плотную архитектуру и учатся на данных R1. Результаты V3.2 в работе
с инструментами зависят также от дообучения и внешней системы исполнения,
а не только от разреженного внимания.

![[00 Учебник/Assets/Figures/deepseek-r1-figure1-hq.png]]

*Рисунок: DeepSeek-AI, Figure 1 из [DeepSeek-R1](https://arxiv.org/pdf/2501.12948). Во время RL у R1-Zero одновременно растут точность на AIME и средняя длина ответа. Графики не доказывают, что длиннее всегда значит лучше: они показывают динамику конкретного эксперимента, в котором policy постепенно расходует больше вычислений на проверяемую задачу. Поэтому reasoning-релиз следует читать как изменение post-training, а не как новый тип decoder-блока.*

## Inference и serving

Официальный репозиторий V3 перечисляет SGLang, vLLM, TensorRT-LLM,
LMDeploy и реализации производителей оборудования; исходные веса
опубликованы в FP8. На скорость влияют распределение экспертов,
топология соединений, ядра MLA и квантизация. DSA сокращает число позиций,
участвующих во внимании, но расход памяти и качество извлечения сведений
из длинного контекста проверяются отдельно.

## MLA и MoE на схеме DeepSeek-V2

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/deepseek-v2-mla-moe.png]]

На схеме отдельно показаны эксперты FFN и сжатие K/V. Выбор экспертов
сокращает вычисления FFN на токен, MLA — память кэша внимания.
Ни один из этих механизмов не отменяет хранение всех экспертных весов.
Автор: DeepSeek-AI.
Источник: architecture figure,
[DeepSeek-V2](https://arxiv.org/pdf/2405.04434).
Файл перенесён без изменения из локальной выгрузки официального PDF; лицензия
рисунка отдельно не указана. Проверено 2026-07-20.

## Опубликовано и неизвестно

Опубликованы отчёты V2/V3/R1/V3.2, конфигурации, веса, демонстрационный код вывода и описание DualPipe/FP8. Полные обучающие корпуса, фильтры, смесь задач RL и устройство публичного API не раскрыты. Приведённые авторами затраты на обучение относятся к описанному запуску; независимое воспроизведение этих затрат здесь не проверялось (**C**).

## Источники

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — paper, **A**.
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) и [official repo](https://github.com/deepseek-ai/DeepSeek-V3) — **A**.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — paper, **A**.
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) и [V3.2-Exp repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) — **A/B**.
- [Hugging Face Transformers: DeepSeek-V3](https://huggingface.co/docs/transformers/model_doc/deepseek_v3) — serving-oriented secondary docs, **C**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
