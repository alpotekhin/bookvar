---
title: Матрица инноваций моделей
type: visual
status: canonical
last_updated: 2026-07-16
---

# Матрица: где именно находится инновация

| Модель/линия | Architecture | Pre-training | Post-training | Inference/system |
|---|---|---|---|---|
| LLaMA | RMSNorm, RoPE, SwiGLU | inference-optimal scaling | — | — |
| Llama 2/3 | GQA, dense block | больше данных/контекста | RLHF/DPO recipes | оптимизированный serving |
| Qwen3 | dense + MoE, QK-Norm | multilingual/data scaling | thinking/non-thinking, RL | thinking budget |
| DeepSeek-V2 | MLA, DeepSeekMoE | — | — | меньший KV-cache |
| DeepSeek-V3 | MLA + MoE, MTP | FP8-scale training | distillation/RL | DualPipe, MTP draft |
| DeepSeek-R1 | архитектура V3 | — | RLVR/GRPO reasoning | длинный reasoning |
| DeepSeek-V3.2 | sparse attention | long-context continuation | scaled agentic RL | дешевле long context |

Матрица предотвращает типичную ошибку: считать каждый новый способ обучения
новой нейросетевой архитектурой.

