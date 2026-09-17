---
title: Sparse- и MoE-модели
type: research-line
status: active
started: 2017
last_updated: 2026-08-06
last_verified: 2026-08-06
key_concepts: [MoE, Routing, Sparse Attention]
key_models: [Mixtral, DeepSeek-V3]
primary_sources:
  - https://arxiv.org/abs/1701.06538
  - https://arxiv.org/abs/2401.04088
  - https://arxiv.org/abs/2412.19437
---

# Sparse- и MoE-модели

## Зачем разреживать вычисление

Увеличение dense-модели связывает число параметров, вычисления на токен и
стоимость распределённого запуска. Conditional computation пытается разорвать
эту связь: хранить большую ёмкость, но выбирать для каждого токена лишь малую
часть графа. Transformer сделал естественной единицей маршрутизации токен, а
взаимозаменяемыми ветвями — несколько FFN.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-moe/mixtral-smoe-layer.png]]

*В sparse MoE-слое router вычисляет веса экспертов отдельно для каждого токена,
выбирает top-k FFN и смешивает их выходы. Иллюстрация из Omar Sanseviero et al.,
[Mixture of Experts Explained](https://huggingface.co/blog/moe); исходный файл
опубликован в наборе
[Hugging Face documentation-images](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png).*

[Sparsely-Gated Mixture-of-Experts (2017)](https://arxiv.org/abs/1701.06538)
показала MoE-слой с noisy top-k gating и одновременно сформулировала две
сохранившиеся проблемы: router должен обучаться через почти дискретный выбор, а
токены не должны стекаться к нескольким популярным experts. [GShard
(2020)](https://arxiv.org/abs/2006.16668) связал MoE с автоматическим шардированием,
а [Switch Transformer (2021)](https://arxiv.org/abs/2101.03961) упростил выбор до
top-1. Это уменьшило коммуникацию, но потребовало capacity factor, auxiliary
balancing loss и правил обработки переполнения.

<span id="тезис"></span>

## Что даёт MoE

MoE позволяет увеличить общее число параметров, ограничивая число активных
FFN на токен. Выигрыш качества при данном объёме вычислений — эмпирический
результат конкретного обучения, а не гарантия выбора разреженной архитектуры.
Распределение токенов между экспертами также не означает, что возникли
устойчивые «эксперт по математике» и «эксперт по литературе».

В [Mixtral, §4 и Figure 8](https://arxiv.org/html/2401.04088v1) маршрутизация
в исследованных слоях сильнее связана с синтаксическими признаками, чем с
тематическим доменом. Специализацию проверяют по токенам, слоям и сдвигам
данных; её нельзя выводить из одного названия Mixture of Experts.

Но total parameters всё равно нужно хранить и перемещать. MoE переносит нагрузку
из матричных вычислений в routing, коммуникацию и memory bandwidth.

## Современная ветвь

[Mixtral 8x7B (2024)](https://arxiv.org/abs/2401.04088) сделал sparse MoE
доступным в открытых весах: токен проходит через два из восьми FFN. Название
нельзя читать как восемь независимых моделей: attention общий, а active и total
parameters — разные величины. [DeepSeekMoE
(2024)](https://arxiv.org/abs/2401.06066) разделил experts на более мелкие и
добавил shared experts. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) развил
эту схему и использовал auxiliary-loss-free bias adjustment для балансировки.

Router создаёт положительную обратную связь: expert, случайно получивший больше
токенов, быстрее улучшается и становится ещё популярнее. Balancing loss, noise
и bias correction противодействуют коллапсу, но иногда посылают токен менее
подходящему expert. На нескольких GPU затем возникает all-to-all: токены
группируются по назначению, пересылаются к experts и возвращаются. Поэтому
малое число активных FLOPs не гарантирует высокой загрузки устройств:
маленькие матричные операции и слабая сеть способны сделать плотную модель
быстрее. Увеличение пакета иногда улучшает загрузку, но повышает требования
к памяти и допустимой задержке.

## Исследовательские ветви

| Ветка | Вопрос |
|---|---|
| routing | как выбирать экспертов устойчиво и дифференцируемо |
| load balancing | как не допустить перегрузки нескольких экспертов |
| shared experts | какие знания должны быть доступны каждому токену |
| expert granularity | много малых или мало крупных экспертов |
| systems | как разместить экспертов между устройствами |
| interpretability | действительно ли эксперты специализируются семантически |

## Ограничения доказательств

- Сравнение active parameters скрывает стоимость хранения total parameters.
- FLOPs не отражают all-to-all communication.
- «Эксперт по теме» часто является post-hoc интерпретацией, а не стабильным
  свойством.
- Dense и MoE checkpoints могут иметь разные данные и training budgets.

Сильное эмпирическое утверждение здесь узкое: при большом масштабе и подходящей
инфраструктуре MoE улучшает качество при сопоставимых FLOPs на токен. Из него не
следуют меньшая latency, меньшая память или семантически понятные experts. Для
сравнения нужны total/active parameters, training tokens, communication volume
и end-to-end throughput.

## Открытые вопросы

- Можно ли маршрутизировать не только FFN, но и память/attention?
- Как сделать routing устойчивым при domain shift?
- Как quantization влияет на router и редких экспертов?
- Когда MoE выгоден при малом batch и edge inference?

## Связанные страницы

[[02 Areas/ML & DL/01 Справочник/FFN и MoE/Mixture of Experts|MoE]] ·
[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Mistral и Mixtral|Mixtral]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/DeepSeek|DeepSeek-V3]]
