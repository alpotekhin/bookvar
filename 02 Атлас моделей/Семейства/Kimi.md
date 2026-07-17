---
type: model-family
organization: Moonshot AI
first_release: 2023
latest_verified_release: Kimi K2.5
last_verified: 2026-07-16
architecture_base: decoder-only Transformer; later MoE
modalities: [text]
status: active
---

# Kimi

## Главный diff

Kimi сначала выделялся продуктовым long-context режимом, архитектура которого
публично описывалась ограниченно. Kimi Linear исследует hybrid linear attention.
Kimi K2 — открытая MoE-линия: большая общая ёмкость при малой активной части,
ориентированная на coding и agentic tool use. Kimi K2.5 продолжает K2-Base как
native multimodal agentic model с vision encoder, instant/thinking modes и
длинными tool workflows.

| Линия | Что известно | Уровень |
|---|---|---|
| Kimi long-context | возможности и context window; мало деталей блока | **B** |
| Kimi Linear | hybrid attention с Kimi Delta Attention | **A** |
| Kimi K2 | 1T total / 32B active, MLA + MoE; MuonClip optimizer | **A/B** |
| Kimi K2.5 | K2 MoE + MLA, MoonViT, 256K context | **A/B** |

- **Architecture:** K2 сочетает MLA с MoE (384 routed experts, 8 выбранных и
  1 shared); Kimi Linear чередует KDA и MLA.
- **Pre-training:** K2 связывает стабильное масштабирование с MuonClip.
- **Post-training:** agentic/coding-способности нельзя смешивать с MoE как
  архитектурной причиной.

## Primary sources

- [Kimi Linear paper](https://arxiv.org/abs/2510.26692) — **A**
- [Kimi K2 repository](https://github.com/MoonshotAI/Kimi-K2) — **A/B**
- [Kimi K2.5 official repository and report](https://github.com/MoonshotAI/Kimi-K2.5) — **A/B**
- [Moonshot AI](https://www.moonshot.ai/) — **B**
