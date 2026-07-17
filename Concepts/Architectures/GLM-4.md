---
title: "GLM-4"
aliases: [GLM-4, ChatGLM, ChatGLM-4, GLM, GLM-4-9B]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/GLM-4|GLM-4]]"
courses: []
sources:
  - "[Du et al. — GLM: Autoregressive Blank Infilling (ACL 2022)](https://arxiv.org/abs/2103.10360)"
  - "[ChatGLM: A Family of LLMs from GLM-130B to GLM-4 All Tools (2024)](https://arxiv.org/abs/2406.12793)"
  - "[GitHub — THUDM/GLM](https://github.com/THUDM/GLM)"
  - "[Sh-Tsang — Review of GLM Architecture](https://sh-tsang.medium.com/review-glm-general-language-model-pretraining-with-autoregressive-blank-infilling-c217bc91b7d5)"
---

# GLM-4

## Зачем эта модель появилась

К 2023-2024 году доминировали два подхода к pre-training: **masked language modeling** (BERT, T5 — хорошо для NLU) и **autoregressive** (GPT — хорошо для генерации). Каждый силён в своей нише, но слаб в чужой.

**GLM** (Tsinghua, THUDM) предложил **третий путь** — autoregressive blank infilling. Одна архитектура, один pre-training objective, но модель одинаково хорошо решает NLU, conditional generation и unconditional generation. Семейство эволюционировало: GLM-130B → ChatGLM → ChatGLM2 → ChatGLM3 → **GLM-4** (2024), конкурируя с GPT-4 в Chinese и English.

## Autoregressive Blank Infilling: ключевая идея

### Как это работает

Стандартные подходы:
- **MLM (BERT):** маскирует отдельные токены, предсказывает их независимо — хорошо для classification, плохо для генерации
- **Autoregressive (GPT):** генерирует слева направо — хорошо для генерации, неэффективно для NLU (unidirectional)

**GLM** маскирует не токены, а **spans** (непрерывные фрагменты текста), затем авторегрессивно заполняет их:

1. Текст разбивается на **Part A** (текст с масками `[MASK]`) и **Part B** (замаскированные spans)
2. Part A кодируется **bidirectional** — каждый токен видит все остальные в Part A
3. Part B генерируется **autoregressive** — каждый span заполняется последовательно, слева направо
4. Spans в Part B перемешиваются случайным образом (random permutation)

### Математика blank infilling

Формально, для входного текста $x = [x_1, \ldots, x_n]$ выбираются $m$ spans $\{s_1, \ldots, s_m\}$, которые заменяются на `[MASK]` токены. Оставшийся текст — $x_{\text{corrupt}}$ (Part A).

Цель обучения — максимизировать:

$$\mathcal{L} = \sum_{i=1}^{m} \log p_\theta(s_{z_i} | x_{\text{corrupt}}, s_{z_{<i}})$$

где $z$ — случайная перестановка (permutation) spans $\{1, \ldots, m\}$.

**Permutation важна:** без неё модель всегда заполняет spans в фиксированном порядке (слева направо), что создаёт bias. Случайный порядок учит модель генерировать spans в **любом порядке**, что улучшает обобщение.

Каждый span генерируется авторегрессивно:

$$\log p_\theta(s_i | x_{\text{corrupt}}, s_{z_{<i}}) = \sum_{j=1}^{l_i} \log p_\theta(s_{i,j} | x_{\text{corrupt}}, s_{z_{<i}}, s_{i,<j})$$

### Два режима маскирования

GLM использует два вида mask для унификации NLU и NLG задач:

| Режим | Длина span | Назначение |
|-------|-----------|------------|
| `[MASK]` | Короткие spans (~15% токенов) | NLU задачи (как BERT) |
| `[gMASK]` | Один длинный span (остаток текста) | NLG задачи (как GPT) |

При `[gMASK]` модель фактически работает как autoregressive LM — получает prefix и дописывает остальное. При коротких `[MASK]` — как MLM, но с авторегрессивной генерацией внутри каждого span.

### 2D Positional Encoding

Каждый токен кодируется двумя позиционными ID:
- **Position 1:** позиция в исходном (corrupted) тексте — для Part A tokens и для начала каждого span
- **Position 2:** intra-span позиция (0 для Part A, 1, 2, 3... для токенов внутри span)

$$\text{embedding} = \text{token\_emb} + \text{pos1\_emb}(\text{position}_1) + \text{pos2\_emb}(\text{position}_2)$$

**Зачем два позиционных кодирования:** модель должна знать и *где* в тексте находится пропуск (position 1), и *сколько* токенов уже сгенерировано внутри этого пропуска (position 2). Без position 2 модель путается при заполнении длинных spans.

### Attention mask: объединение bidirectional и autoregressive

Attention mask GLM нетривиален:
- Part A → Part A: **full attention** (bidirectional)
- Part B → Part A: **full attention** (span видит весь контекст)
- Part B → Part B (внутри span): **causal mask** (авторегрессивно)
- Part A → Part B: **no attention** (контекст не видит заполнение)
- Part B span_i → Part B span_j (i ≠ j): зависит от порядка permutation

Это позволяет **одной моделью** реализовать и encoder (Part A) и decoder (Part B).

## GLM-4: от исследования к production

### Эволюция семейства

| Модель | Год | Params | Ключевое |
|--------|-----|--------|----------|
| GLM-130B | 2022 | 130B | Первая крупная билингвальная модель EN/ZH |
| ChatGLM | 2023 | 6B | Первый китайский chatbot, 10M+ пользователей |
| ChatGLM2 | 2023 | 6B | Улучшенный pre-training, Flash Attention, 32K контекст |
| ChatGLM3 | 2023 | 6B | Tool use, code interpreter |
| **GLM-4** | 2024 | ~130B+ | Уровень GPT-4, All Tools, мультимодальность |
| GLM-4-9B | 2024 | 9B | Open-weight, бьёт LLaMA-3-8B в Chinese |

### GLM-4 All Tools: архитектура

Уникальная особенность GLM-4 — встроенная способность к **tool use** без отдельного fine-tuning:

**Code Interpreter:**
- Модель пишет и выполняет Python-код для math, data analysis, визуализации
- Sandbox execution с доступом к numpy, pandas, matplotlib
- Модель **самостоятельно решает**, когда написать код вместо рассуждений в тексте

**Web Browser:**
- Поиск в интернете, чтение страниц, суммаризация
- Multi-step browsing: поиск → чтение → уточнение запроса → повторный поиск

**File Operations:**
- Обработка документов (PDF, Excel, изображений)
- Extraction, transformation, analysis uploaded файлов

**Function Calling:**
- Вызов произвольных API по описанию
- JSON-structured output для интеграции с внешними системами

**Как обучено:** GLM-4 All Tools обучена на специальных **multi-turn tool-use диалогах**. Модель учится:
1. Определить, нужен ли инструмент для ответа
2. Выбрать правильный инструмент
3. Сформировать корректный запрос к инструменту
4. Интерпретировать результат и продолжить диалог

В отличие от подходов «tool use как afterthought» (добавление function calling через simple SFT), GLM-4 All Tools интегрирует инструменты **глубоко в reasoning pipeline**.

### Обучение GLM-4

Pre-training на **~10T токенов** (Chinese + English + 24 языка). Пост-обучение:
1. SFT на высококачественных instruction-following данных
2. RLHF для alignment (ChatGLM-RLHF pipeline)
3. Специализированный tool-use training
4. ChatGLM-Math: self-critique для улучшения mathematical reasoning

## Результаты GLM-4

| Бенчмарк | GLM-4 | GPT-4 (0125) | GPT-4 Turbo | Claude 3 Opus |
|-----------|-------|--------------|-------------|---------------|
| MMLU | ~85+ | 86.5 | 86.5 | 86.8 |
| GSM8K | ~90+ | 95.3 | — | — |
| MATH | ~55+ | 64.5 | — | — |
| C-Eval | **~90+** | 69.9 | — | — |
| CMMLU | **~88+** | — | — | — |
| IFEval | ~80+ | 83.6 | — | — |

**Разрыв в математике:** основной gap между GLM-4 и GPT-4 Turbo — на MATH и GPQA. Для закрытия разрыва TII применяет ChatGLM-Math (self-critique training).

GLM-4-9B — один из сильнейших 9B open-weight моделей, превосходящих LLaMA-3-8B на китайских задачах и сопоставимый на английских.

## Почему GLM важен для экосистемы

1. **Альтернативный pre-training:** autoregressive blank infilling — доказанная альтернатива pure autoregressive, объединяющая NLU и NLG. Хотя в итоге decoder-only победил по масштабу, идея blank infilling повлияла на design множества моделей (включая T5 span corruption)
2. **Chinese-first:** одна из немногих frontier-моделей, оптимизированных для Chinese с нуля. На C-Eval и CMMLU GLM-4 значительно превосходит западные модели
3. **Tool use как first-class citizen:** GLM-4 All Tools показал, что tool use можно интегрировать глубоко, а не добавлять как afterthought. Этот подход повлиял на Claude 3, GPT-4o и другие модели
4. **Open-weight ecosystem:** GLM-4-9B и ChatGLM серия стали стандартом для Chinese NLP community, с 10M+ пользователей

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — базовая архитектура
- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling]] — подход BERT, который GLM расширяет
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]] — подход GPT, с которым GLM объединяется
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — GLM использует 2D вариант
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — encoder-only подход к NLU

## Дополнительные ресурсы

- [GLM Paper (ACL 2022)](https://arxiv.org/abs/2103.10360) — оригинальная статья об autoregressive blank infilling
- [ChatGLM Family Paper](https://arxiv.org/abs/2406.12793) — эволюция от GLM-130B до GLM-4
- [GitHub — THUDM/GLM](https://github.com/THUDM/GLM) — код и модели
