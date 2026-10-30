---
title: Distributed Training
aliases:
  - Distributed Training
  - Распределённое обучение
  - DDP
  - FSDP
  - ZeRO
type: concept
status: redirect
category: Training
papers: []
courses: []
sources:
  - >-
    [Rajbhandari et al. — ZeRO: Memory Optimizations Toward Training Trillion
    Parameter Models (2019)](https://arxiv.org/abs/1910.02054)
  - '[PyTorch FSDP docs](https://pytorch.org/docs/stable/fsdp.html)'
  - '[Megatron-LM (NVIDIA)](https://github.com/NVIDIA/Megatron-LM)'
  - '[DeepSpeed (Microsoft)](https://www.deepspeed.ai/)'
last_updated: '2026-10-29'
canonical_topic: >-
  00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed
  precision.md
---
# Distributed Training

Полное объяснение: [[00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision|Распределённое обучение и смешанная точность]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="distributed-training--распределённое-обучение" aria-hidden="true"></span>
<span id="зачем-это-нужно" aria-hidden="true"></span>
<span id="data-parallelism-dp" aria-hidden="true"></span>
<span id="основная-идея" aria-hidden="true"></span>
<span id="dp-pytorch-legacy" aria-hidden="true"></span>
<span id="ddp-distributed-data-parallel" aria-hidden="true"></span>
<span id="model-parallelism" aria-hidden="true"></span>
<span id="tensor-parallelism-intra-layer" aria-hidden="true"></span>
<span id="pipeline-parallelism-inter-layer" aria-hidden="true"></span>
<span id="expert-parallelism-для-moe" aria-hidden="true"></span>
<span id="fsdp--zero" aria-hidden="true"></span>
<span id="zero-стадии-deepspeed" aria-hidden="true"></span>
<span id="fsdp-fully-sharded-data-parallel" aria-hidden="true"></span>
<span id="zero-infinity" aria-hidden="true"></span>
<span id="3d-parallelism" aria-hidden="true"></span>
<span id="gradient-accumulation" aria-hidden="true"></span>
<span id="коммуникационные-примитивы-nccl" aria-hidden="true"></span>
<span id="фреймворки" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
