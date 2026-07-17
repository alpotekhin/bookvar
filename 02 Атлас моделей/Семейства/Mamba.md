---
type: model-family
organization: researchers at CMU and Princeton; ecosystem contributors
first_release: 2023
latest_verified_release: Mamba-2
last_verified: 2026-07-16
architecture_base: selective state-space model
modalities: [sequence]
status: active-research
---

# Mamba

Mamba — не Transformer: вместо pairwise attention она обновляет компактное
состояние. **Selective SSM** делает параметры перехода зависимыми от входа,
позволяя выбирать, что запомнить или забыть. Mamba-2 связывает SSM и attention
через Structured State Space Duality.

| Свойство | Attention | Mamba |
|---|---|---|
| Training | попарное смешивание токенов | parallel scan |
| Decoding state | KV-cache растёт с длиной | фиксированное recurrent state |
| Retrieval | прямой доступ к прошлому токену | информация сжата в состоянии |

- **Architecture:** selective SSM (**A**).
- **Training/post-training:** архитектура не задаёт автоматически данные или
  alignment recipe.
- **Риск:** линейная сложность не гарантирует лучшее качество retrieval.

## Primary sources

- [Mamba paper](https://arxiv.org/abs/2312.00752) — **A**
- [Mamba-2 paper](https://arxiv.org/abs/2405.21060) — **A**
- [Official state-spaces/mamba code](https://github.com/state-spaces/mamba) — **A**
