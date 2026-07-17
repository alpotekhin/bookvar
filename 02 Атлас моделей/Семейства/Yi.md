---
type: model-family
organization: 01.AI
first_release: 2023
latest_verified_release: Yi-Lightning / Yi-1.5 line
last_verified: 2026-07-16
architecture_base: LLaMA-like decoder-only Transformer
modalities: [text, image]
status: active
---

# Yi

Yi — LLaMA-подобная bilingual-линия с акцентом на качество данных и длинный
контекст. Yi-1.5 обновил pre-training corpus и post-training; Yi-VL добавил
vision-ветку. Для закрытых Yi-Lightning архитектура публично раскрыта не
полностью.

- **Architecture:** dense decoder, RoPE, GQA/attention details зависят от
  checkpoint; проверять config (**A/C**).
- **Pre-training:** English/Chinese data и long-context extension (**B**).
- **Post-training:** chat checkpoints и preference alignment.
- **Evidence warning:** benchmark и API-description не доказывают новый блок.

## Primary sources

- [Yi official repository](https://github.com/01-ai/Yi) — **A/B**
- [Yi model collection](https://huggingface.co/01-ai) — **A/B**
- [Yi technical report](https://arxiv.org/abs/2403.04652) — **A**
