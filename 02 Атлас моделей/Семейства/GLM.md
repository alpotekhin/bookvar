---
title: "GLM"
type: model-family
organization: Zhipu AI, Z.ai and Tsinghua KEG
first_release: 2021
latest_verified_release: GLM-5.1
last_verified: 2026-07-20
architecture_base: autoregressive blank infilling; later causal MoE
modalities: [text, image]
status: active
---

# GLM

## Место в истории

GLM начался не как очередной causal decoder: paper 2021 обучал autoregressive blank infilling с двумерными positional IDs и объединял понимание, conditional generation и unconditional generation. GLM-130B масштабировал этот подход и подробно документировал engineering. ChatGLM сделал bilingual dialogue-модель практичной и открытой. В GLM-4/4.5 фокус сместился к long context, tools и agents, а GLM-4.5 впервые дал крупную открытую MoE-базу. GLM-5 масштабировал её и добавил DeepSeek Sparse Attention; GLM-5.1 — текущий post-trained agentic engineering release, но не отдельный опубликованный backbone.

## Релизы как diff

| Релиз | Architecture | Context/modalities | Training/post-training | Evidence |
|---|---|---|---|---|
| GLM (2021) | autoregressive blank infilling, 2D positions | text | смешение целей NLU/generation | **A** |
| GLM-130B (2022) | 70-layer dense bidirectional GLM | 2K; bilingual text | 400B tokens; documented scaling/stability | **A** |
| ChatGLM-6B (2023) | GLM dialogue adaptation; INT4/8 support | 2K→32K variants; Chinese/English | instruction/dialogue tuning | **A/B** |
| ChatGLM2/3 (2023) | MQA, longer context; tool/agent formats | до 32K/128K by variant | stronger alignment and tool use | **B** |
| GLM-4 / 4V (2024) | proprietary text backbone; separate vision line | 128K and multimodal variants | All Tools, browsing/code interpreter | **A/B; часть architecture закрыта** |
| GLM-4.5 (2025-07) | 355B/32B active MoE; Air 106B/12B; MLA-like efficient attention | 128K text | 23T tokens; thinking/non-thinking; agentic RL | **A/B** |
| GLM-5 (2026) | 744B/40B active; DeepSeek Sparse Attention | long-context text | 28.5T tokens; asynchronous RL via slime | **A/B** |
| GLM-5.1 (2026-04) | GLM-5-class weights/backbone per official repo | agentic text/coding | long-horizon engineering post-training | **B** |

## Неизменное ядро и перелом линии

Раннее ядро — blank infilling: модель получает маскированные spans и авторегрессивно восстанавливает их с учётом двунаправленного контекста. Название GLM сохранилось, но современные ChatGLM/GLM-4.x/5 serving interfaces выглядят как causal chat decoders; нельзя автоматически приписывать им тот же 2D objective без report. Главный подтверждённый перелом GLM-4.5 — sparse MoE и hybrid thinking. GLM-5 увеличивает total/active parameters и вводит DSA, где indexer выбирает подмножество ключей для длинного контекста.

## Tokenizer, данные и post-training

Tokenizers и special tokens менялись между ChatGLM поколениями; remote-code implementations ранних checkpoints несовместимы с современными templates. GLM-4.5 report сообщает 23T, GLM-5 — 28.5T pre-training tokens, но не полный corpus manifest. Thinking/non-thinking — режимы post-training и prompting. Agentic engineering у 5/5.1 опирается на tool environments и RL infrastructure `slime`; benchmark рост не раскрывает точные RL tasks.

## Inference и serving

Ранний ChatGLM был оптимизирован для consumer GPUs с quantization. GLM-4.5/5 — совсем иной класс: MoE требует хранения 355B/744B weights, expert parallelism и all-to-all, хотя active compute 32B/40B. DSA должно снижать long-context attention cost, но поддержка зависит от kernels в SGLang/vLLM/vendor stack. BF16 и FP8 checkpoints имеют разные объёмы и hardware paths.

## Визуальный первоисточник: семейство, а не одна линия checkpoints

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/glm-family-timeline.png]]

Официальная timeline до GLM-4 раскладывает releases по четырём колонкам: API,
open LLM, open VLM и agents. Она предотвращает частую ошибку «GLM-4V или
All Tools — следующее поколение одного backbone»: ветви развиваются параллельно.
Автор: GLM-4 Team / Zhipu AI / Tsinghua KEG. Источник: Figure 1, p. 2,
[ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools](https://arxiv.org/pdf/2406.12793).
Локальный файл — crop официального PDF; лицензия рисунка отдельно не указана.
Проверено 2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** papers GLM/130B/GLM-4/4.5/5, official repos/configs и часть weights. **Неизвестно:** точная архитектура некоторых закрытых GLM-4 API versions, полный data/RL mixture, является ли 5.1 новым pre-trained checkpoint или преимущественно post-training revision. Поэтому карточка не выводит архитектуру 5.1 из SWE-Bench (**C**).

## Источники

- [GLM: General Language Model Pretraining](https://arxiv.org/abs/2103.10360) — **A**.
- [GLM-130B](https://arxiv.org/abs/2210.02414) — **A**.
- [ChatGLM-6B repository](https://github.com/THUDM/ChatGLM-6B) — **A/B**.
- [GLM-4 report](https://arxiv.org/abs/2406.12793) — **A**.
- [GLM-4.5 official repository](https://github.com/zai-org/GLM-4.5) — report/weights, **A/B**.
- [GLM-5/5.1 official repository](https://github.com/zai-org/GLM-5) и [GLM-5 report](https://arxiv.org/abs/2602.15763) — **A/B**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
