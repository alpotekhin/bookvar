---
title: Библиотека источников
type: source-note
status: active
last_updated: 2026-11-10
---

# Библиотека источников

Здесь сохранены оригинальные курсы, книги, статьи, слайды и реализации.
[[05 Источники/Курсы|Каталог курсов]] сопоставляет их с темами учебника:
можно открыть конкретную лекцию или пройти оригинальный курс в порядке автора.
Основное объяснение каждой темы находится в соответствующей главе Bookvar.

- [[02 Areas/ML & DL/05 Источники/Registry]]
- [[02 Areas/ML & DL/05 Источники/Авторы и сайты]]
- [[02 Areas/ML & DL/05 Источники/Community repositories]]
- [[02 Areas/ML & DL/05 Источники/Курсы]]
- [[02 Areas/ML & DL/05 Источники/Визуальные материалы и лицензии]]
- [[02 Areas/ML & DL/05 Источники/LMCache/LMCache — карта материалов|LMCache: architecture, storage, transfer, and operations]]
- [[05 Источники/Papers/DeepSpeed-FastGen|DeepSpeed-FastGen — Dynamic SplitFuse]]
- [[02 Areas/ML & DL/05 Источники/Papers/TTT layers|TTT layers]]
- [[02 Areas/ML & DL/05 Источники/Papers/End-to-End Test-Time Training|End-to-End TTT]]
- [[02 Areas/ML & DL/05 Источники/Papers/Gated Delta Networks|Gated Delta Networks]]
- [[05 Источники/Papers/Qwen3.8-Flash-Next|Qwen3.8-Flash-Next — QSA, Gated Residual и таблицы n-грамм]]
- [[05 Источники/Papers/Kimi K3 и K3.1 Pro|Kimi K3 и K3.1 Pro — KDA, Attention Residuals, LatentMoE и разреженное MLA]]
- [[05 Источники/Papers/Gemma 3 и Gemma 4|Gemma 3 и Gemma 4 — локальные слои, общий KV и мультимодальный вход]]
- [[02 Areas/ML & DL/05 Источники/Papers/Recall in efficient sequence models|Recall в эффективных sequence models]]
- [[05 Источники/Papers/PaddleOCR-VL v1|PaddleOCR-VL — вёрстка, порядок чтения и распознавание областей]]
- [[05 Источники/Papers/Mathlib Euler products|Mathlib: Euler products — исходный модуль и условия теоремы]]
- [[05 Источники/Papers/AlphaProof|AlphaProof — поиск доказательств и обучение под задачу]]
- [[05 Источники/Papers/FunSearch|FunSearch — эволюция эвристик и проверка cap sets]]
- [[05 Источники/Papers/AlphaEvolve|AlphaEvolve — поиск программ и проверка разложений]]
- [[05 Источники/Papers/TinyBERT|TinyBERT — дистилляция слоёв Transformer и исходная реализация]]
- [[05 Источники/Papers/GradCache|GradCache — полный контрастивный пакет и повторные проходы энкодера]]
- [[05 Источники/Papers/ContAccum|ContAccum — наборы кандидатов и банк представлений]]
- [[05 Источники/Papers/Goujon et al. — Piecewise-linear regions|Goujon et al. — кусочно-линейные области Maxout]]
- [[05 Источники/Papers/Pan et al. 2023 — What in-context learning learns|What In-Context Learning “Learns” — выбор и подмена меток в демонстрациях]]
- [[05 Источники/Papers/Holtzman et al. — Surface Form Competition|Surface Form Competition — как формулировка ответа меняет оценку]]
- [[05 Источники/Papers/Gao et al. — How to Train Long-Context Language Models|How to Train Long-Context Language Models — ProLong и оценка до и после SFT]]
- [[05 Источники/Papers/LambdaRank and LambdaLoss|LambdaRank and LambdaLoss — nDCG pair weights and cutoff conventions]]
- [[05 Источники/Papers/Delayed conversion feedback|Delayed conversion feedback — conversion delay and observation time]]
- [[05 Источники/Papers/Implicit feedback weights and negative sampling|Implicit feedback — confidence weights and negative sampling]]
- [[05 Источники/Papers/Sequence windows and sampled evaluation|Sequential windows and sampled recommendation metrics]]
- [[05 Источники/Papers/The Complexity Trap|The Complexity Trap — observation masking and LLM summarization]]
- [[05 Источники/Papers/AppWorld|AppWorld — state-based evaluation of interactive agents]]
- [[05 Источники/Papers/Federated Learning with Buffered Asynchronous Aggregation|FedBuff — buffered asynchronous federated learning]]
- [[05 Источники/Papers/Membership inference and memorization audits|Membership inference and memorization audits — LiRA, RMIA, and canaries]]
- [[05 Источники/Papers/Label shift and certified robustness|Label shift and certified robustness — BBSE and randomized smoothing]]
- [[05 Источники/Papers/Gender Shades and selective labels|Gender Shades and selective labels — subgroup evaluation and missing outcomes]]

Первичные материалы хранятся в `raw`, а опубликованные обзоры и указатели —
в этом разделе источников.

## Разборы реализаций

Эти материалы показывают устройство конкретной системы или версии реализации.
Общие механизмы инференса объясняются в
[[00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|главе о средах исполнения]]
и связанных с ней главах.

- [[05 Источники/Implementation studies/Inside vLLM|Inside vLLM — подробный разбор engine loop, планировщика и памяти]]
- [[05 Источники/Implementation studies/SGLang and RadixAttention|SGLang — язык программ, RadixAttention и constrained decoding]]
- [[05 Источники/Implementation studies/Triton GPU kernels|Triton — разбор оригинальных GPU tutorials]]
- [[05 Источники/Implementation studies/Conditional actions and transactional outbox|Условные действия и transactional outbox — фиксация операции и её доставка]]
