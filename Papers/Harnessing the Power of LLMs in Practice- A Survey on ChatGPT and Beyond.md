---
title: "Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond"
url: https://arxiv.org/abs/2304.13712
authors: [Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, Xia Hu]
year: 2023
date_reviewed: 2026-04-06
type: source-note
status: legacy
category: paper
tags:
  - LLM
  - Review
Date: 2023-04-26
Organization: Amazon / Texas A&M University / Rice University
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/NLP/ChatGPT|ChatGPT]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompt Engineering|Prompt Engineering]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/harnessing-the-power-of-llms-in-practice--a-survey-on-chatgp/paper.txt]]"
---

# Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond

**Authors:** Jingfeng Yang et al. (Amazon + Texas A&M + Rice University)
**Published:** arXiv:2304.13712v2, Apr 2023 (ACM Computing Surveys)
**URL:** https://arxiv.org/abs/2304.13712

## TL;DR

Практический гайд: когда использовать LLM (GPT-style), а когда — fine-tuned модели (BERT-style). Систематизирует по трём измерениям — модели, данные, задачи — с конкретными рекомендациями (use cases / no-use cases) для каждой категории NLP задач. Дополнено обсуждением efficiency (стоимость, latency, PEFT) и trustworthiness (bias, hallucinations, safety).

## LLM vs Fine-tuned: Определения

Авторы предлагают операциональное разграничение:
- **LLM** = huge pretrained model (≥20B param), используется **без task-specific fine-tuning** (zero/few-shot via prompting)
- **Fine-tuned model** = обычно < 20B param, pretrained + fine-tuned на task-specific датасете

**Два архитектурных типа (Table 1):**

| Тип | Training | Примеры |
|-----|---------|---------|
| **BERT-style** (encoder/enc-dec) | Masked LM, предсказывает замаскированные слова | BERT, RoBERTa, T5, ELECTRA, DeBERTa |
| **GPT-style** (decoder-only) | Autoregressive LM, предсказывает следующее слово | GPT-3, PaLM, LLaMA, BLOOM, OPT, GPT-4 |

**Эволюция (Figure 1 — evolutionary tree):**
- До 2021: encoder-only и enc-decoder модели доминировали
- После GPT-3 (2020): decoder-only модели взяли лидерство
- 2023: BERT-style постепенно уходят в тень, но encoder-decoder (T5, Flan-T5) остаются живыми
- Тренд на closed-source: до 2020 большинство открытые, после GPT-3 — закрываются

## Practical Guide for Data (§3)

**Три ключевых тезиса:**
1. LLM лучше fine-tuned при OOD (out-of-distribution) данных или adversarial examples
2. LLM предпочтительнее при ограниченных аннотированных данных; при обильных — оба варианта OK
3. Выбирай LLM, pre-trained на данных, близких к вашей задаче

**По доступности данных:**

| Режим | Рекомендация |
|-------|-------------|
| **Zero-shot** | LLM; нет catastrophic forgetting, параметры не меняются |
| **Few-shot** | LLM (in-context learning); fine-tuned могут overfit на малых данных |
| **Abundant labels** | Оба варианта; fine-tuned обычно проще и дешевле |

**Про pre-training data:** PaLM и BLOOM имеют много multilingual данных → лучше для MT. GPT-3.5 код-версия обучалась на code data → лучший code generation.

## Task-by-Task Guide (§4)

### Traditional NLU Tasks

**Вывод:** Fine-tuned модели обычно лучше на GLUE/SuperGLUE при наличии аннотированных данных.

**Конкретно:**
- Sentiment (IMDB, SST): fine-tuned ≈ LLM
- Toxicity detection (CivilComments): LLM плохи, лучшая = BERT-based Perspective API
- NLI (RTE, SNLI): fine-tuned лучше; на CB — comparable
- QA (SQuAD v2, QuAC): fine-tuned лучше; на CoQA — comparable
- IR (MS MARCO): fine-tuned методы лучше (hard to formulate IR as few-shot)
- NER (CoNLL03): fine-tuned ~2× лучше LLM

**Use cases для LLM в NLU:** miscellaneous/diverse text classification; Adversarial NLI (ANLI, особенно R3/R2) — out-of-distribution обобщение.

### Generation Tasks

**Вывод:** LLM превосходят на большинстве generation задач.

**Конкретно:**
- Summarization: fine-tuned лучше по ROUGE, но **человек предпочитает LLM** (faithfulness, coherence, relevance) — значит ROUGE не та метрика
- Machine Translation: LLM чуть хуже по BLEU в среднем, но **лучше в low-resource** (e.g. Romanian→English). BLOOM хорош в multilingual.
- Code generation (HumanEval, MBPP): LLM хороши; GPT-4 решает 25% LeetCode задач
- Open-ended generation: GPT-3 новости почти неотличимы от человеческих

**No-use cases для LLM в generation:** rich-resource MT (DeltaLM+Zcode лучше); extremely low-resource MT (Kazakh).

### Knowledge-Intensive Tasks

**Вывод:** LLM превосходят за счёт массивных знаний.

**Use cases:** Closed-book QA (NaturalQuestions, WebQ, TriviaQA — даже zero-shot LLM лучше); MMLU (57 subjects, GPT-4: 86.5%); BIG-bench знаниевые задачи.

**No-use cases для LLM:** задачи, требующие контрфактуальных знаний (redefine-math: символ переопределяется); задачи, где достаточно контекстуальных знаний (MRC/reading comprehension — retrieval-augmented fine-tuned модели лучше при наличии корпуса).

### Scaling Abilities (§4.4)

**Три ключевых феномена:**

**1. Emergent Abilities** — способности, не экстраполируемые из малых моделей:
- Word manipulation (reversed words, sorting, unscrambling) — GPT-3
- ASCII word recognition, hyperbaton — PaLM
- Logical deduction, logic grid puzzles
- Advanced coding (auto debugging, code description)
→ Появляются внезапно при превышении порогового размера модели

**2. Inverse Scaling Phenomenon** — с ростом размера performance падает:
- Redefine-math (нужно игнорировать prior знания)
- Into-the-unknown, Memo-trap
→ Over-reliance на prior knowledge вместо контекста

**3. U-shaped Phenomenon** — perf падает, потом снова растёт:
- Hindsight-neglect, NegationQA, Quote-repetition
→ Маленькие и большие модели используют разные внутренние механизмы

### Miscellaneous Tasks

**LLM хороши:** Chatbot/mimicking human (ChatGPT); Data annotation (GPT-3.5 = human annotators в некоторых задачах); NLG evaluation (GPT-4 как judge > традиционные метрики); CoT interpretability.

**LLM плохи:** Regression (STS-B: ChatGPT < fine-tuned RoBERTa); Multimodal tasks (BEiT, PaLI доминируют в VQA).

### Real-World Tasks (§4.6)

Реальные задачи ≠ академические benchmarks. Три сложности:
1. Noisy/unstructured input (опечатки, смешанные языки)
2. Ill-defined tasks, multiple intents в одном запросе
3. Implicit instructions, follow-up questions needed

LLM лучше справляются благодаря diversity pre-training данных. Fine-tuned модели не обобщаются на распределения вне их training set.
→ **Instruction tuning** (FLAN, T0) и **RLHF** (InstructGPT, ChatGPT) — ключевые методы улучшения реального следования инструкциям.

## Other Considerations (§5)

### Efficiency

| Аспект | Данные |
|--------|--------|
| Training cost (GPT-3 175B) | ~$4.6M за один run |
| Training compute (GPT-3) | 3.14 × 10²³ FLOPs |
| Energy (PaLM) | ~3.4 GWh за ~2 месяца |
| Inference latency (GPT-J 6B, 32 tokens) | 0.707s; InstructGPT davinci: 1.969s |

**PEFT (Parameter-Efficient Tuning):** LoRA, Prefix Tuning, P-Tuning. Alpaca-LoRA = LLaMA + LoRA, обучение за несколько часов на RTX 4090.

### Trustworthiness

**Robustness & Calibration:** LLM точность коррелирует с robustness. Fine-tuning ухудшает calibration из-за over-parameterization. RLHF (InstructGPT) улучшает robustness.

**Fairness & Bias:** LLM чувствительны к demographic biases (dialect, religion, gender, race). InstructGPT показывает меньшие performance disparities.

**Spurious Biases (Shortcut Learning):** Fine-tuned модели страдают от shortcut learning (лексическое совпадение вместо понимания). LLM менее подвержены, но in-context learning → majority label bias, position bias, common token bias.

**Safety:**
- **Hallucinations:** LLM генерируют правдоподобный но ложный контент. RLHF снижает этот риск.
- **Harmful content:** hate speech, misinformation, dual-use risks. Safeguards + human feedback.
- **Privacy:** Samsung ChatGPT инцидент с утечкой исходного кода; Italy ban на ChatGPT.

## Decision Framework (Figure 2)

Дерево решений для выбора LLM vs fine-tuned:

```
NLP задача
├── Mimicking human (chatbot)? → LLM
├── Difficult task requiring scaling (reasoning, emergent)? → LLM
│   └── Multiple tasks? → LLM
│       └── Tasks far from language modeling? → Fine-tuned
│           ├── Little annotated data? → LLM
│           │   └── OOD data? → LLM / No clear winner
│           └── Creative/complex text generation? → LLM
│               └── Common NLU/NLG? → зависит
├── Knowledge inconsistent with real world? → осторожно
│   └── Context contains enough knowledge? → Fine-tuned
│       └── Knowledge-intensive? → LLM
```

## My notes

- Статья — **лучший практический guide** на ~2023 для выбора LLM vs fine-tuned. Хочешь быстро ответить "мне нужен LLM или BERT?" — читаешь сюда.
- **Главный инсайт:** метрики (ROUGE, BLEU) врут для generation задач — люди предпочитают LLM, хотя формальные метрики говорят обратное. Summarization — яркий пример.
- **Emergent abilities** здесь документированы конкретно (word sorting, logic puzzles), не абстрактно. Видно что они реальны и непредсказуемы.
- **Inverse scaling phenomenon** — важное предостережение против "bigger is always better". Существуют задачи где GPT-3.5 < GPT-2. GPT-4 частично восстанавливает performance на некоторых из них.
- Статья от 2023 — уже устарела по конкретным моделям (LLaMA-1, отсутствует GPT-4o, Claude, Gemini), но концептуальный framework остается актуальным.
- RLHF упомянут как ключевое улучшение, но не разобран детально — для деталей смотреть InstructGPT paper.
- **Decision flow (Figure 2)** — при прочих равных удобная шпаргалка. Рекомендую держать ссылку под рукой.
