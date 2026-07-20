---
title: "Nemotron"
type: model-family
organization: NVIDIA
first_release: 2023
latest_verified_release: Nemotron 3 Ultra and Nano Omni
last_verified: 2026-07-20
architecture_base: dense derivatives then hybrid Mamba-Transformer LatentMoE
modalities: [text, image, video, audio]
status: active
---

# Nemotron

Nemotron — несколько связанных линий, а не одна последовательность увеличивающихся Transformer. Nemotron-4 был собственной dense моделью NVIDIA; Llama Nemotron получался из Llama через pruning/NAS и post-training; Nemotron 3 перешёл к открытому hybrid Mamba–Transformer MoE. Сравнивать поколения нужно по происхождению backbone.

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

Mamba-2 несёт дешёвое состояние на каждом шаге, global attention периодически восстанавливает прямой доступ к прошлому. LatentMoE сначала проецирует token representation в меньшее latent space, маршрутизирует по 512 экспертам и активирует 22; shared expert обрабатывает каждый токен. Это уменьшает стоимость экспертных матриц по сравнению со стандартным MoE при большом числе маршрутов. Два multi-token prediction heads предсказывают будущие токены и могут использоваться для native speculative decoding.

## Данные и post-training

Nemotron 3 раскрывает необычно много компонентов: synthetic pretraining corpus почти 10T токенов для Nano, 25T для Super, weights, recipes и NeMo software. Super предобучался с NVFP4 на части оборудования; это формат вычислений обучения, а не только постфактум quantization. Multi-environment RL через NeMo Gym учит reasoning, tool use и регулируемый thinking budget. Ultra применяет multi-teacher on-policy distillation.

Ранние Llama Nemotron переиспользуют pretrained Llama: их улучшение нельзя приписывать новому pre-training. NAS меняет глубину/ширину выбранного backbone, затем distillation и RL возвращают качество. Это другой путь, чем обучение Nemotron 3 с нуля.

## Serving и мультимодальность

MoE требует хранить все веса, хотя active count меньше. Mamba сокращает долю растущего KV-cache, NVFP4 уменьшает bandwidth, MTP ускоряет decode при достаточном acceptance. Nano Omni добавляет Parakeet audio encoder, C-RADIO vision encoder, 3D convolution/video sampling и adaptors; reasoning происходит в общем decoder, но сырые модальности обрабатываются разными frontends.

## Опубликовано и неизвестно

NVIDIA публикует technical reports, модельные веса, data recipes и документацию. Некоторые benchmark и throughput claims зависят от Blackwell/NVFP4 и не переносятся на другое оборудование. Полнота лицензируемых исходных данных ограничена теми наборами, которые NVIDIA вправе распространять.

## Источники

- [Nemotron 3 overview](https://arxiv.org/abs/2512.20856).
- [Nemotron 3 Super report](https://arxiv.org/abs/2604.12374) и [Ultra announcement](https://developer.nvidia.com/blog/nvidia-nemotron-3-ultra-powers-faster-more-efficient-reasoning-for-long-running-agents/).
- [Nemotron official portal](https://developer.nvidia.com/topics/ai/nemotron).
- [Nano Omni architecture](https://docs.nvidia.com/nemotron/nightly/nemotron/omni3/architecture.html).
