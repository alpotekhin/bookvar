---
title: "GPT-4"
aliases: [GPT 4, GPT4]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 4.0]]"
courses: []
sources:
  - "[OpenAI — GPT-4 Technical Report (2023)](https://arxiv.org/abs/2303.08774)"
  - "[OpenAI — GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)"
  - "[Bubeck et al. — Sparks of Artificial General Intelligence (2023)](https://arxiv.org/abs/2303.12712)"
---

# GPT-4

## Зачем эта модель важна

GPT-4 (OpenAI, март 2023) — модель, которая **изменила восприятие возможностей AI** в обществе. Не потому что она была первой в чём-то (не первый multimodal LLM, не первая RLHF-модель), а потому что она показала **professional-level performance** на реальных задачах: сдала bar exam в топ-10% (адвокатский экзамен), набрала 86.4% на MMLU (выше большинства экспертов), и продемонстрировала reasoning уровня, который раньше считался недоступным для AI.

При этом OpenAI **намеренно скрыла** практически все технические детали: размер модели, архитектурные решения, состав данных, использованный compute. Это беспрецедентный уровень закрытости для научной публикации.

## Что известно об архитектуре

Технический отчёт содержит минимум деталей:

- **Transformer-based**, предобучен как decoder-only causal LM (next token prediction)
- **Multimodal**: принимает **text + images** на вход, генерирует только текст
- **RLHF alignment**: base model → SFT → Reward Model → PPO (как InstructGPT)
- Размер, hardware, dataset composition — **не раскрыты**

По неофициальным данным (утечки, интервью), GPT-4 — это **Mixture of Experts** модель с несколькими сотнями миллиардов параметров, но официально это не подтверждено.

## Predictable Scaling: ключевой инженерный вклад

Самый ценный научный вклад GPT-4 paper — не сама модель, а **методология предсказания capabilities**:

$$L(C) = aC^b + c$$

где $L$ — loss, $C$ — compute. Этот scaling law, fitted на моделях с **1000-10000x меньшим compute**, позволил:

1. **Предсказать финальный loss** GPT-4 до завершения обучения (отклонение <1%)
2. **Предсказать HumanEval pass rate** из моделей с 1000x меньшим compute
3. Принимать **engineering decisions** (размер модели, объём данных, compute budget) задолго до финального запуска

**Почему это революционно:** до GPT-4 обучение frontier модели было «прыжком в неизвестность» — огромные инвестиции без гарантий результата. Predictable scaling позволяет заранее знать, что получится, и планировать разработку как инженерный процесс.

На практике это означает: можно обучить 10 маленьких моделей, построить scaling curve, и точно предсказать performance модели, которая стоит $100M+ для обучения.

## RLHF: выравнивание с человеческими предпочтениями

GPT-4 использует тот же pipeline alignment, что и InstructGPT:

```
Base GPT-4 (pretrained) → SFT (supervised fine-tuning на демонстрациях)
                        → Reward Model (обучение на human comparisons)
                        → PPO (оптимизация policy через reward model)
```

Важное наблюдение из отчёта: **RLHF не улучшает академические бенчмарки** (multiple-choice MMLU и т.д.), но критически улучшает:
- **Factuality** — модель реже галлюцинирует
- **Adherence to instructions** — модель точнее следует инструкциям
- **Safety** — модель отказывается от вредных запросов

Это подтверждает гипотезу: RLHF учит модель **формату и стилю** ответа, а не фундаментальным знаниям.

## System Card: безопасность и риски

OpenAI опубликовала отдельный System Card — документ о рисках GPT-4:

- **Red teaming**: 50+ внешних экспертов тестировали модель на опасные use cases (биооружие, кибератаки, манипуляция)
- **CBRN risks**: GPT-4 даёт marginal uplift в доступе к информации о chemical/biological/radiological/nuclear оружии (но не больше, чем интернет-поиск)
- **Hallucinations**: GPT-4 галлюцинирует значительно реже GPT-3.5, но проблема не решена
- **Bias**: модель наследует предубеждения из training data

System Card стал **шаблоном для индустрии** — теперь большинство крупных лабораторий публикуют аналогичные документы.

## Ключевые результаты

### Профессиональные экзамены (Table 1)

| Экзамен | GPT-4 percentile | GPT-3.5 percentile |
|---------|------------------|---------------------|
| **Uniform Bar Exam** | **~90th** (298/400) | ~10th (213/400) |
| LSAT | ~88th | ~40th |
| SAT Evidence-Based R&W | ~93rd | ~87th |
| SAT Math | ~89th | ~70th |
| GRE Quantitative | ~80th | ~25th |
| **GRE Verbal** | **~99th** | ~63rd |
| AP Biology | 5 (max) | 4 |
| AMC 10 Math | 6-12th | 10-19th |

**Bar exam**: переход от bottom-10% к top-10% — это переход от «не сдал» к «квалифицированный адвокат». Один из самых наглядных прорывов.

### NLP бенчмарки (Table 2)

| Benchmark | GPT-4 | GPT-3.5 | Best open-source |
|-----------|-------|---------|------------------|
| **MMLU** | **86.4%** | 70.0% | 70.7% |
| HellaSwag | 95.3% | 85.5% | 84.2% |
| ARC Challenge | 96.3% | 85.2% | 85.2% |
| **HumanEval** | **67.0%** | 48.1% | 26.2% |
| **GSM-8K** | **92.0%** | 57.1% | 58.8% |
| WinoGrande | 87.5% | 81.6% | 85.1% |

MMLU 86.4% — первая модель, уверенно превзошедшая **expert-level** performance (~87%) на многих предметных областях. Разрыв с GPT-3.5 (+16.4%) — один из крупнейших скачков между поколениями моделей.

### Multimodal и многоязычность

- **Vision**: GPT-4 понимает изображения — графики, мемы, скриншоты, диаграммы. Не генерирует изображения, только анализирует.
- **Multilingual MMLU**: GPT-4 превосходит English-language SOTA в **24 из 26 протестированных языков**, включая языки с малым количеством training data.

## Почему GPT-4 победил

Четыре фактора успеха:

| Фактор | Описание |
|--------|----------|
| **Scale** | Предположительно >1T параметров (MoE), обученных на петабайтах данных |
| **Data quality** | Тщательная курация training data, фильтрация, дедупликация |
| **RLHF** | Многоитерационный alignment с human feedback |
| **Predictable scaling** | Инженерный процесс вместо «угадывания» — позволяет оптимизировать каждый аспект |

## Ограничения

Несмотря на впечатляющие результаты, GPT-4 имеет существенные проблемы:
- **Hallucinations**: уверенно генерирует правдоподобную, но ложную информацию
- **Reasoning brittleness**: на слегка модифицированных задачах performance может резко упасть
- **AMC 10 Math**: только 6-12th percentile — сложная математика остаётся проблемой
- **Knowledge cutoff**: модель не знает о событиях после даты обучения
- **Воспроизводимость**: закрытость деталей делает научную репликацию невозможной

## Влияние на индустрию

GPT-4 запустил несколько параллельных процессов:
1. **AI arms race**: Google (Gemini), Anthropic (Claude), Meta (Llama 3) ускорили разработку
2. **Коммерциализация AI**: GPT-4 стал основой ChatGPT Plus, Microsoft Copilot, и сотен продуктов
3. **AI regulation**: профессиональный уровень модели стимулировал регуляторные инициативы (EU AI Act, Executive Order Biden)
4. **Дебаты о AGI**: Bubeck et al. (Microsoft Research) опубликовали «Sparks of AGI» — спорную, но влиятельную работу о proto-AGI capabilities GPT-4

## Key papers

- [[02 Areas/ML & DL/Papers/GPT 4.0]] — оригинальный технический отчёт (OpenAI, 2023)
- [[02 Areas/ML & DL/Papers/GPT 3.0]] — predecessor

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — alignment метод
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — predictable scaling
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]] — capabilities, появляющиеся при масштабировании
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — predecessor
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-2|GPT-2]] — архитектурный прародитель

## Дополнительные ресурсы

- [OpenAI GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — оригинальный отчёт
- [GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf) — анализ рисков и безопасности
- [Sparks of AGI (Bubeck et al.)](https://arxiv.org/abs/2303.12712) — глубокий анализ capabilities GPT-4 от Microsoft Research
