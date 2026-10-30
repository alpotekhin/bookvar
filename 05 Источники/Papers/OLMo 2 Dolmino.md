---
title: "OLMo 2: Dolmino and mid-training"
type: source-note
status: canonical
last_verified: 2026-10-26
source_url: https://arxiv.org/abs/2501.00656
year: 2025
---

# OLMo 2: Dolmino и промежуточное обучение

В отчёте Team OLMo различаются исходные запасы данных и конкретные смеси,
использованные в поздних обучающих ветках. 832,6 млрд токенов общей
качественной части и 10,7 млрд токенов математической части — размеры
доступных корпусов, не фактическая длина одного запуска.

Ветки на 50/100/300 млрд токенов используют разные количества повторений.
Некоторые итоговые модели получают усреднением весов независимо обученных
веток. Складывать бюджеты веток и описывать результат как одну непрерывную
траекторию оптимизации нельзя.

В [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|главе о синтетических данных]]
таблица Dolmino иллюстрирует именно доступные источники. Точные доли из
Table 13 не пересчитываются в обучающие токены: произведение Source % на
исходный размер и столбец Mix % не во всех строках согласованы с номинальным
бюджетом. Для воспроизведения нужны конфигурации конкретного запуска.

Дополнительный первичный источник:
[карточка allenai/dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124).

## Two-stage recipe and original artifacts

[OLMo Team et al., 2 OLMo 2 Furious, v1](https://arxiv.org/html/2501.00656v1)
declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The archived HTML pins the three complete recipe paragraphs from §2.3 and
Tables 4, 9 and 14. The English paragraphs, a worked branch-budget example,
and all scores from Table 9 are included in
[[00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training#olmo-two-stage-recipe|the textbook's data and pretraining chapter]].

The main-stage checkpoint is reused as the starting point for several
independent mid-training runs. Their final weights are averaged; the token
budgets of those branches must not be presented as one sequential trajectory.
Table 14 also contains a counterexample to improvement on every task:
mix E's GSM* score decreases from 60.5 to 43.0 after averaging.

The corresponding Berkeley lecture is Hanna Hajishirzi's
[Open Training Recipes for Reasoning in Language Models](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf#page=143),
24 February 2025, physical pages 143–152. The textbook reproduces the complete
schedule slide at page 147. Its “<10B” warmup label fits the 7B batch, not
the doubled 13B batch. The next slide's 99%/1% split is schematic, not the
report's accounting of all mid-training branches. The bar chart at page 152
does not plot Qwen 2.5, so its headline is not used as evidence of a
Qwen comparison.

The paper's CC BY license applies to its text and tables. The lecture image
uses the separately recorded course reuse permission; no named Creative
Commons license has been verified for that slide. These two rights records
are not interchangeable.

## Complete source inventories

The following tables retain all rows and quantities from
[Table 1](https://arxiv.org/html/2501.00656v1#S2.T1) and
[Table 2 of the v1 report](https://arxiv.org/html/2501.00656v1#S2.T2),
CC BY 4.0. Source annotations are unwrapped into one cell; the mid-training
table is split into its two original groups. B, M, K and T mean billion,
million, thousand and trillion respectively. Bytes are source-reported
byte counts, not GPU memory requirements.

These are available inventories. Mixture sampling and repetition determine
how much of each inventory a training branch consumes. For example, the
752B-token high-quality web subset is much larger than a 50B-token branch;
it does not all pass through that branch. The token shares for that branch
are given separately in the textbook.

### OLMo 2 1124 Mix — Table 1

| Source | Type | Tokens | Words | Bytes | Docs |
|---|---|---:|---:|---:|---:|
| DCLM-Baseline | Web pages | 3.71T | 3.32T | 21.32T | 2.95B |
| StarCoder · filtered version · from OLMoE Mix | Code | 83.0B | 70.0B | 459B | 78.7M |
| peS2o · from Dolma 1.7 | Academic papers | 58.6B | 51.1B | 413B | 38.8M |
| arXiv | STEM papers | 20.8B | 19.3B | 77.2B | 3.95M |
| OpenWebMath | Math web pages | 12.2B | 11.1B | 47.2B | 2.89M |
| Algebraic Stack | Math proofs code | 11.8B | 10.8B | 44.0B | 2.83M |
| Wikipedia & Wikibooks · from Dolma 1.7 | Encyclopedic | 3.7B | 3.16B | 16.2B | 6.17M |
| Total | | 3.90T | 3.48T | 22.38T | 3.08B |

### Dolmino High Quality Subset — Table 2

| Source | Type | Tokens | Words | Bytes | Docs |
|---|---|---:|---:|---:|---:|
| DCLM-Baseline · FastText top 7% · FineWeb ≥ 2 | High quality web | 752B | 670B | 4.56T | 606M |
| FLAN · from Dolma 1.7 · decontaminated | Instruction data | 17.0B | 14.4B | 98.2B | 57.3M |
| peS2o · from Dolma 1.7 | Academic papers | 58.6B | 51.1B | 413B | 38.8M |
| Wikipedia & Wikibooks · from Dolma 1.7 | Encyclopedic | 3.7B | 3.16B | 16.2B | 6.17M |
| Stack Exchange · 09/30/2024 dump · curated Q&A data | Q&A | 1.26B | 1.14B | 7.72B | 2.48M |
| High quality total | | 832.6B | 739.8B | 5.09T | 710.8M |

### Dolmino Math Mix — Table 2

| Source | Type | Tokens | Words | Bytes | Docs |
|---|---|---:|---:|---:|---:|
| TuluMath | Synthetic math | 230M | 222M | 1.03B | 220K |
| Dolmino SynthMath | Synthetic math | 28.7M | 35.1M | 163M | 725K |
| TinyGSM-MIND | Synthetic math | 6.48B | 5.68B | 25.52B | 17M |
| MathCoder2 Synthetic · Ajibawa-2023 · M-A-P Matrix | Synthetic Math | 3.87B | 3.71B | 18.4B | 2.83M |
| Metamath · OWM-filtered | Math | 84.2M | 76.6M | 741M | 383K |
| CodeSearchNet · OWM-filtered | Code | 1.78M | 1.41M | 29.8M | 7.27K |
| GSM8K · Train split | Math | 2.74M | 3.00M | 25.3M | 17.6K |
| Math total | | 10.7B | 9.73B | 45.9B | 21.37M |

The MathCoder2 row groups the source names exactly as Table 2 does; the
3.87B figure is not assigned separately to each name. Quantities are copied
at the source's precision, including totals; they are not recomputed from
the displayed rounded rows. Dataset availability does not establish
permission to redistribute every constituent document: dataset cards and
source licenses remain separate records.
