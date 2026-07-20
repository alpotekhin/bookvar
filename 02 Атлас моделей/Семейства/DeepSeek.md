---
title: DeepSeek
type: model-family
organization: DeepSeek AI
first_release: 2023
latest_verified_release: DeepSeek-V3.2
last_verified: 2026-07-20
architecture_base: decoder-only Transformer and sparse MoE
modalities: [text]
status: active
---

# DeepSeek

## Место в истории

DeepSeek важен не одним benchmark-релизом, а последовательным co-design модели и системы. DeepSeekMoE дробит FFN на более мелкие routed experts и выделяет shared experts; MLA сжимает KV-представление; V3 связывает это с FP8 training, балансировкой без auxiliary loss и multi-token prediction; R1 показывает, что reasoning-линия — прежде всего post-training поверх V3, а не новый attention block. V3.2 добавляет sparse attention и отдельный масштабный agentic RL pipeline.

## Неизменное ядро

Начиная с V2, ядро — causal decoder с MLA и fine-grained MoE. На токен активна малая часть FFN-параметров, а attention сохраняет latent-представление вместо полного KV-cache. R1, V3-0324 и V3.1/3.2 следует читать как разные ветви post-training и long-context/inference изменений поверх этой базы.

## Релизы как diff

| Релиз | Architecture | Context / tokenizer | Pre/post-training | Evidence |
|---|---|---|---|---|
| DeepSeek LLM (2023) | dense decoder, 7B/67B | 4K, text | 2T tokens; base/chat | **A** |
| DeepSeekMoE (2024-01) | shared + fine-grained routed experts | text | специализация экспертов при меньших active FLOPs | **A** |
| DeepSeek-V2 (2024-05) | MLA + DeepSeekMoE; 236B/21B active | 128K | 8.1T tokens; SFT/RL chat | **A** |
| DeepSeek-V3 (2024-12) | 671B/37B active; loss-free balancing; MTP; FP8 recipe | 128K; 129K vocabulary | 14.8T tokens; SFT + RL | **A** |
| DeepSeek-R1 (2025-01) | V3 base architecture | 128K | R1-Zero: RL без SFT; R1: cold-start, RL, rejection sampling, второй RL; distillation | **A** |
| V3.2-Exp / V3.2 (2025) | DeepSeek Sparse Attention поверх MLA | long-context text | продолжение pre-training; V3.2 масштабирует reasoning и agentic RL | **A/B** |

## Механизмы

**MLA.** Вместо хранения отдельных полных K/V для каждого head модель сохраняет низкоразмерный latent. При serving это снижает KV-cache, но требует специализированных kernels и аккуратного weight absorption. Это не разновидность GQA: сжатие обучается как часть attention.

**DeepSeekMoE.** Fine-grained experts увеличивают число возможных комбинаций; shared experts должны удерживать общие знания. Цена — хранение сотен миллиардов параметров, all-to-all communication и сложная балансировка. 37B active у V3 описывает compute, не memory footprint.

**V3 training diff.** Auxiliary-loss-free balancing регулирует bias маршрутизатора, не добавляя основной balancing loss; sequence-level auxiliary term всё же сохраняется. Multi-token prediction служит дополнительной training objective и потенциально помогает speculative decoding. FP8 — свойство training system, а не архитектуры модели.

**DSA.** V3.2-Exp вводит lightning indexer, который выбирает релевантные позиции для sparse attention. Публичный experimental release полезен как архитектурная проверка, но не доказывает одинаковую поддержку во всех serving engines.

## Post-training, reasoning и tools

R1-Zero получил наблюдаемое long-chain reasoning через group-relative policy optimization на проверяемых задачах без предварительного SFT. Полный R1 добавил cold-start данные, reasoning-oriented RL, rejection sampling/SFT для общих задач и финальную RL стадию. Distilled Qwen/Llama checkpoints — не маленькие копии архитектуры V3: это dense student-модели, обученные на данных R1. Agentic/tool benchmarks V3.2 относятся к post-training и scaffold; они не следуют из sparse attention сами по себе.

## Inference и serving

Официальный V3 repo перечисляет SGLang, vLLM, TensorRT-LLM, LMDeploy и vendor backends; исходные checkpoints опубликованы в FP8. Реальная производительность зависит от expert parallelism, коммуникационной топологии, MLA kernels и quantization. Для длинных prompt DSA должен уменьшать attention work, но полный KV/activation budget и качество retrieval по позиции нужно проверять на конкретной реализации.

## Визуальный первоисточник: совместный diff MLA и MoE

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/deepseek-v2-mla-moe.png]]

Одна схема показывает оба устойчивых механизма линии V2→V3: routed/shared
experts заменяют dense FFN, а MLA сжимает кэшируемые K/V в latent. Важно, что
это независимые оси экономии — conditional FFN compute и KV-cache — и ни одна
не делает весь checkpoint «размером active parameters». Автор: DeepSeek-AI.
Источник: architecture figure,
[DeepSeek-V2](https://arxiv.org/pdf/2405.04434).
Файл перенесён без изменения из локальной выгрузки официального PDF; лицензия
рисунка отдельно не указана. Проверено 2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** reports V2/V3/R1/V3.2, configs, demo inference code, веса и системные детали DualPipe/FP8. **Неизвестно полностью:** точный training corpus и фильтры, весь RL task mixture, production kernels/API routing и стоимость, независимая воспроизводимость заявленного training budget. Последнее нельзя объявлять ложным или истинным без отдельной проверки (**C**).

## Источники

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — paper, **A**.
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) и [official repo](https://github.com/deepseek-ai/DeepSeek-V3) — **A**.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — paper, **A**.
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) и [V3.2-Exp repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) — **A/B**.
- [Hugging Face Transformers: DeepSeek-V3](https://huggingface.co/docs/transformers/model_doc/deepseek_v3) — serving-oriented secondary docs, **C**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
