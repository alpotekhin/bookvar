---
title: "Nemotron"
type: model-family
organization: NVIDIA
first_release: 2023
latest_verified_release: Nemotron 3 Ultra and Nano Omni
last_verified: 2026-08-06
architecture_base: dense derivatives then hybrid Mamba-Transformer LatentMoE
modalities: [text, image, video, audio]
status: active
---

# Nemotron

Под названием Nemotron опубликованы модели разного происхождения.
Nemotron-4 — собственная плотная модель NVIDIA. Llama Nemotron использует
предобученную Llama, изменяет её размеры поиском архитектуры и удалением
компонентов, затем проходит дообучение. Nemotron 3 сочетает Mamba,
внимание и MoE. Эти линии отличаются не только размером, но и устройством
слоёв и способом получения исходных весов.

## Хронология

| Линия | Архитектура и назначение |
|---|---|
| Nemotron-3/4 (2023–2024) | Dense Transformer; Nemotron-4 340B включает Base, Instruct и Reward. |
| Llama Nemotron (2025) | Производные Llama 3.x после neural architecture search, distillation, SFT и RL. |
| Nemotron 3 Nano (15.12.2025) | 30B total / 3B active hybrid Mamba-2–attention MoE, 1M context. |
| Nemotron 3 Super (2026) | 120B / 12B active, LatentMoE, MTP и NVFP4 pre-training. |
| Nemotron 3 Ultra (04.06.2026) | 550B / 55B active, long-running agent reasoning. |
| Nano Omni (2026) | Text/image/video/audio encoders вокруг 30B-A3B unified decoder. |

## Внутри Nemotron 3 Super

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/nemotron3-layer-pattern.png]]

*Рисунок: NVIDIA, [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf), Figure 2; локальная копия. Повторяемый паттерн чередует Mamba-2, LatentMoE и редкие global-attention слои; цифры под группами показывают число повторов.*

Mamba-2 обновляет состояние фиксированного размера, а слои полного внимания сохраняют прямой доступ к позициям истории. [[00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving#LatentMoE: обмен в меньшей размерности|LatentMoE]] сначала проецирует представление токена в пространство меньшей размерности, затем выбирает 22 из 512 экспертов. Общий эксперт обрабатывает каждый токен. Сжатие уменьшает размеры экспертных матриц при большом числе возможных маршрутов. Две дополнительные головы предсказания будущих токенов могут использоваться при спекулятивном декодировании.

## Данные и post-training

Для Nemotron 3 описаны синтетические данные предобучения, веса, программы обучения и ПО NeMo. Указанные объёмы предобучения составляют почти 10T токенов для Nano и 25T для Super. Super использует NVFP4 при предобучении на соответствующем оборудовании; низкая точность входит в вычисления обучения, а не только в последующее сжатие весов. RL в нескольких средах NeMo Gym обучает рассуждениям, работе с инструментами и управлению длиной рассуждения. Ultra применяет дистилляцию от нескольких учителей на траекториях обучаемой модели.

![[00 Учебник/Assets/Figures/curated/topics-53-59-source-first/nemotron-generate-rank-filter.png]]

*Рисунок: NVIDIA, [Leverage Our Latest Open Models for Synthetic Data Generation with NVIDIA Nemotron-4-340B](https://developer.nvidia.com/blog/leverage-our-latest-open-models-for-synthetic-data-generation-with-nvidia-nemotron-4-340b/). Instruct-модель сначала генерирует несколько ответов, Reward-модель присваивает им оценки, после чего фильтр собирает отобранный синтетический набор. Это схема линии Nemotron-4, а не pre-training Nemotron 3; здесь она нужна, чтобы отделить происхождение обучающих примеров от архитектуры Mamba–Transformer. Локальная копия перенесена без визуальных изменений.*

Llama Nemotron начинает с предобученных весов Llama. Поиск архитектуры меняет глубину и ширину сети, после чего дистилляция и RL восстанавливают качество уменьшенной модели. Nemotron 3, напротив, обучается с нуля.

The original Nemotron-4 prompt, dialogue and preference-generation procedures
are explained in
[[00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы#nemotron4-synthetic-alignment|Synthetic data: Nemotron-4 alignment]].
This is a historical recipe, separate from the Nemotron 3 training recipe.

## Serving и мультимодальность

MoE требует хранить все веса, хотя active count меньше. Mamba сокращает число слоёв с растущим KV-кэшем. Применение NVFP4 при предобучении само по себе не уменьшает трафик памяти при выводе: нужны контрольная точка в соответствующем низкоразрядном формате и ядра, которые эффективно её исполняют. Для BF16-весов нельзя автоматически обещать четырёхбитную стоимость хранения или чтения. Даже при хранении весов в FP4 отдельно учитываются масштабы квантизации, активации, состояния и KV-кэш. MTP может ускорить декодирование при достаточно частом принятии предложенных токенов. Nano Omni добавляет Parakeet audio encoder, C-RADIO vision encoder, 3D convolution/video sampling и adaptors; reasoning происходит в общем decoder, но сырые модальности обрабатываются разными frontends.

Например, идеальная упаковка 120 млрд параметров занимает 240 GB при двух байтах на параметр и 60 GB при четырёх битах. Это только арифметика весов без накладных расходов, не замер Nemotron Super и не обещание четырёхкратного ускорения: достижимая скорость зависит от аппаратуры, ядра и размера пакета. Различие между форматом хранения и вычислением разобрано в [[00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|главе о квантизации]].

## Опубликовано и неизвестно

NVIDIA публикует технические отчёты, веса, описание подготовки данных и документацию. Часть результатов по скорости получена на Blackwell с NVFP4; для другого оборудования требуются отдельные измерения. Публикация описания данных не означает доступность всех исходных документов.

## Источники

- [Nemotron 3 overview](https://arxiv.org/abs/2512.20856).
- [Nemotron 3 Super report](https://arxiv.org/abs/2604.12374) и [Ultra announcement](https://developer.nvidia.com/blog/nvidia-nemotron-3-ultra-powers-faster-more-efficient-reasoning-for-long-running-agents/).
- [Nemotron official portal](https://developer.nvidia.com/topics/ai/nemotron).
- [Nano Omni architecture](https://docs.nvidia.com/nemotron/nightly/nemotron/omni3/architecture.html).
