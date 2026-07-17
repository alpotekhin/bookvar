---
title: "PaLM 2"
aliases: [PaLM2, Pathways Language Model 2]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/PaLM 2]]"
courses: []
sources:
  - "[Google — PaLM 2 Technical Report (2023)](https://arxiv.org/abs/2305.10403)"
  - "[Google AI Blog — PaLM 2](https://blog.google/technology/ai/google-palm-2-ai-large-language-model/)"
---

# PaLM 2

## Зачем эта модель появилась

PaLM (Chowdhery et al., 2022) был крупнейшей моделью Google — 540B параметров на 780B токенов. Но к маю 2023 года Chinchilla scaling laws (Hoffmann et al., 2022) показали, что PaLM был **катастрофически неоптимален**: слишком много параметров, слишком мало данных. PaLM 2 — переосмысление: **меньше параметров, больше данных, лучше архитектура**.

Ключевая идея: PaLM 2-L (largest) **значительно меньше** PaLM-540B, но **значительно лучше**. Это подтверждает: model size — не единственный путь к качеству. **Compute-optimal scaling** + **лучшие данные** + **архитектурные улучшения** = лучше, чем просто увеличение модели.

## Три столпа улучшений

### 1. Compute-Optimal Scaling

Chinchilla hypothesis: данные и параметры должны масштабироваться **примерно 1:1** по compute. PaLM нарушил это — модель в 3 раза больше, чем данные.

PaLM 2 валидирует Chinchilla на **ещё большем compute budget**:
- PaLM 2-L значительно меньше PaLM-540B
- Но обучен на **значительно большем количестве данных**
- Результат: лучше quality при **меньшей стоимости inference**

Практическое следствие: меньшая модель = **быстрее и дешевле serving**. Это позволило Google развернуть PaLM 2 в массовых продуктах (Bard, Workspace) с приемлемой латентностью.

### 2. Multilingual-First данные

PaLM использовал ~78% English данных (non-code). PaLM 2 радикально расширил multilingual корпус:
- Сотни языков (включая low-resource)
- Параллельные multilingual документы
- Математика и code на разных языках

Важное наблюдение: **большие модели лучше справляются** с disparate multilingual данными без деградации English performance. Маленькие модели при добавлении разнообразных языков теряют качество English; большие — нет.

### 3. Архитектурные и Objective улучшения

PaLM 2 обучен на **mixture of objectives** (не только causal LM):
- Autoregressive LM (next token prediction)
- Дополнительные objectives, вдохновлённые UL2 (Tay et al., 2023)

Точный состав mixture не раскрыт, но мотивация ясна: разные objectives учат модель понимать **разные аспекты языка** — one-directional prediction, bidirectional understanding, infilling, etc.

## Семейство моделей

Четыре размера с кодовыми именами:

| Кодовое имя | Масштаб | Deployment |
|-------------|---------|------------|
| **Gecko** | Smallest | Mobile, on-device |
| **Otter** | Small | Lightweight applications |
| **Bison** | Medium | General-purpose |
| **Unicorn** | Large | Complex reasoning tasks |

Точные размеры **не опубликованы**. Известно: PaLM 2-L (Unicorn) **значительно меньше** PaLM-540B.

## Ключевые результаты

### Multilingual Language Proficiency (Figure 1)

PaLM 2 сдаёт языковые экзамены на уровне, достаточном **для преподавания этого языка**:

| Язык | PaLM 2 | PaLM | Статус |
|------|--------|------|--------|
| Польский | **94%** | 62% | Pass (teacher-level) |
| Итальянский | **94%** | 83% | Pass |
| Испанский | **87%** | 81% | Pass |
| Японский | **82%** | 70% | Pass |
| Немецкий | **82%** | 77% | Pass |
| Французский | **83%** | 69% | Pass |

PaLM 2 passes экзамены **во всех** протестированных языках. PaLM fails некоторые. Разрыв особенно велик для low-resource языков.

### Reasoning

- **BIG-Bench**: значительное улучшение на hard reasoning задачах
- **MMLU**: улучшение на professional knowledge areas
- **GSM8K / MATH**: улучшение mathematical reasoning через CoT
- **WinoGrande**: close to human-level

### Coding (20+ языков!)

PaLM 2 генерирует код не только на Python/JS:
- **HumanEval**: превосходит PaLM
- **Prolog, Verilog, Fortran**: генерация на экзотических языках
- Code debugging и explanation
- Multilingual code generation (с комментариями на разных языках)

## Inference-Time Toxicity Control

Уникальная feature: PaLM 2 использует **control tokens** в pre-training для управления toxicity:

```
[SAFE] normal training text here
[UNSAFE] toxic text here
```

При inference модель conditioning'ится на `[SAFE]` token → значительно снижает toxicity **без** дополнительного overhead. Это эффективнее, чем post-hoc фильтрация или RLHF alignment для toxicity.

## Canary Tokens: измерение memorization

PaLM 2 вводит **canary tokens** — специальные уникальные последовательности, вставленные в pre-training данные. Затем проверяется, может ли модель воспроизвести эти последовательности → прямое измерение memorization.

Результаты:
- PaLM 2 имеет **более низкие rates memorization**, чем PaLM
- Для tail languages (мало данных) memorization растёт, когда данные повторяются несколько раз
- Нет случаев memorization sensitive personal data

## Deployment: от research к product

PaLM 2 стал основой для массовых Google продуктов:

| Продукт | Использование PaLM 2 |
|---------|---------------------|
| **Bard** (→ Gemini) | Основная модель chatbot |
| **Google Workspace** | Duet AI (Docs, Sheets, Gmail) |
| **MedPaLM 2** | Медицинский QA (medical-expert level) |
| **SecPaLM** | Security threat analysis |
| **Google Search** | SGE (Search Generative Experience) |

Меньший размер PaLM 2 → **значительно более быстрый inference** → возможность использования в real-time продуктах с миллиардами пользователей.

## Сравнение PaLM vs PaLM 2

| Аспект | PaLM | PaLM 2 |
|--------|------|--------|
| Размер (largest) | 540B | **Меньше** (точный — не раскрыт) |
| Training data | ~780B tokens, ~78% English | **Больше**, multilingual-first |
| Objectives | Causal LM only | **Mixture of objectives** |
| Multilingual | Побочный эффект | **Явный приоритет** |
| Inference cost | Очень высокая | **Значительно ниже** |
| Scaling | Over-parameterized | **Compute-optimal** |

## Почему PaLM 2 важна

### 1. Proof of compute-optimal scaling

PaLM 2 — первая **production-scale** валидация Chinchilla scaling laws. Не просто исследовательский эксперимент, а модель, развёрнутая в продуктах Google с миллиардами пользователей.

### 2. Multilingual как first-class priority

До PaLM 2 multilinguality в LLM была побочным эффектом англоцентричных данных. PaLM 2 показала, что **целенаправленная multilingual курация** даёт качественно иной уровень, не жертвуя English performance.

### 3. Efficiency > Scale

Сообщение индустрии: не нужно строить самую большую модель — нужно **самую эффективную**. Это повлияло на Mistral, Gemma, Phi, и другие «маленькие, но мощные» модели.

### 4. Control tokens для safety

Inference-time toxicity control через pre-training tokens — элегантное решение, не требующее дополнительного compute при inference. Альтернатива тяжёлому RLHF для safety.

## Key papers

- [[02 Areas/ML & DL/Papers/PaLM 2]] — оригинал (Google, May 2023)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — Chinchilla-optimal scaling
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain-of-Thought]] — reasoning improvements
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — instruction-tuned variants
- [[02 Areas/ML & DL/Concepts/Architectures/Flan-T5|Flan-T5]] — Google's instruction tuning research
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектурная основа

## Дополнительные ресурсы

- [PaLM 2 Technical Report (arXiv)](https://arxiv.org/abs/2305.10403) — полный отчёт
- [Google AI Blog — PaLM 2](https://blog.google/technology/ai/google-palm-2-ai-large-language-model/) — анонс с визуализациями
- [Chinchilla paper (Hoffmann et al.)](https://arxiv.org/abs/2203.15556) — scaling laws, мотивирующие PaLM 2
