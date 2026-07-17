---
title: "Emergent Abilities"
aliases: [emergent capabilities, emergent behavior, emergence in LLMs, phase transitions in LLMs]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 4.0|GPT-4]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]]"
  - "[[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond|Harnessing the Power of LLMs]]"
  - "[[02 Areas/ML & DL/Papers/COT|Chain of Thought]]"
  - "[[02 Areas/ML & DL/Papers/PaLM 2|PaLM 2]]"
courses: []
sources:
  - "[Wei et al. 2022 — Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)"
  - "[Schaeffer et al. 2023 — Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004)"
  - "[CSET Georgetown — Emergent Abilities Explainer](https://cset.georgetown.edu/article/emergent-abilities-in-large-language-models-an-explainer/)"
  - "[NeurIPS 2023 — Schaeffer et al. (Best Paper)](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf)"
---

# Emergent Abilities

## Зачем это важно: неожиданные способности

В физике **эмерджентность** — это появление сложных свойств системы, которые отсутствуют у её отдельных компонентов (пример: молекула воды не «мокрая», но совокупность молекул создаёт жидкость). В 2022 году Jason Wei et al. перенесли эту концепцию на LLM:

> **Emergent ability** — способность, которая **отсутствует у меньших моделей** и **появляется у больших**. Не является плавной экстраполяцией из меньшего масштаба.

Это утверждение вызвало жаркую дискуссию: если способности LLM появляются **непредсказуемо** при масштабировании, как обеспечить безопасность? Что ещё «проснётся» в GPT-5?

![[02 Areas/ML & DL/raw/papers/emergent-abilities/images/emergent-abilities-wei.png]]
*Примеры emergent abilities: near-zero performance у малых моделей → sudden jump у больших (источник: Wei et al., 2022)*

## Оригинальное определение (Wei et al., 2022)

### Паттерн emergence

Типичный scaling curve для большинства задач выглядит так: performance плавно растёт с размером модели (log-scale). Но для **некоторых задач** наблюдается другой паттерн:

1. Модели < $N$ параметров: **near-zero** или **random-chance** performance
2. Модели > $N$ параметров: **внезапный скачок** до нетривиального уровня

Нет постепенного улучшения — есть «порог», за которым способность «включается».

### Конкретные примеры

Wei et al. идентифицировали десятки задач с emergent pattern на BIG-Bench:

#### Word Manipulation (GPT-3)
- **Reversed words**: расположить буквы слова в обратном порядке
- **Sorting words**: алфавитная сортировка
- **Unscrambling letters**: расшифровать перемешанное слово

Маленькие модели (< 10B) дают случайные ответы. GPT-3 175B начинает решать.

#### Logic and Reasoning (PaLM)
- **ASCII word recognition**: распознать слово, написанное ASCII-art буквами
- **Hyperbaton**: определить необычный порядок слов
- **Logical deduction**: многошаговые логические задачи
- **Logic grid puzzles**: задачи типа «кто живёт в каком доме»

PaLM 8B: ~0%. PaLM 62B: ~5%. PaLM 540B: **~40-60%**.

#### Chain-of-Thought как emergent ability

Из [[02 Areas/ML & DL/Papers/COT|Chain of Thought]]: CoT prompting (цепочка рассуждений) — яркий пример emergence:

- Модели < ~100B: CoT **ухудшает** результаты или не помогает (модель генерирует «fluent but wrong» рассуждения)
- Модели ≥ ~100B: CoT → **значительные** улучшения (+2x на GSM8K для GPT-3 175B)

Из paper: *"chain-of-thought prompting does not positively impact performance for small models, and only yields performance gains when used with models of ~100B parameters."*

## Контраргумент: «Emergence = мираж» (Schaeffer et al., 2023)

В 2023 году Rylan Schaeffer, Brando Miranda и Sanmi Koyejo опубликовали статью, которая получила **NeurIPS 2023 Outstanding Paper Award** и перевернула дискуссию.

### Главный тезис

Emergent abilities — не фундаментальное свойство масштабирования LLM, а **артефакт выбора метрики**:

> *"Emergent abilities appear due to the researcher's choice of metric rather than due to fundamental changes in model behavior with scale."*

### Механизм: нелинейные vs линейные метрики

Schaeffer et al. показали, что один и тот же набор модельных предсказаний может выглядеть как «emergence» или как «плавное улучшение» — в зависимости от метрики:

#### Метрики, создающие emergence

- **Exact match accuracy**: ответ считается правильным **только** если совпадает целиком. При token-level error rate 5% модель может получить 0% exact match (одна ошибка = весь ответ неправильный). Когда per-token error падает ниже порога, exact match «внезапно» прыгает.
- **Multiple choice accuracy**: для задач с вариантами ответов — тоже дискретная метрика (либо верно, либо нет).

#### Метрики, уничтожающие emergence

- **Token-level error rate**: per-token точность — непрерывная метрика, показывающая **плавное** улучшение с масштабом.
- **Brier score**: вероятностная оценка, тоже непрерывная.
- **Log-likelihood**: плавно улучшается, как и scaling laws предсказывают.

### Математический аргумент

Если per-token error rate $p$ уменьшается как степенная функция масштаба (что следует из scaling laws), то exact-match accuracy на задаче из $L$ токенов:

$$\text{Acc}_{\text{exact}} = (1 - p)^L$$

При больших $L$ это **пороговая функция**: при $p = 0.05$ и $L = 20$ → $\text{Acc} = 0.36$, при $p = 0.10$ → $\text{Acc} = 0.12$, при $p = 0.15$ → $\text{Acc} = 0.04$. Маленькое изменение $p$ → резкое изменение accuracy.

### Эмпирическая проверка

Schaeffer et al. подтвердили гипотезу тремя способами:

1. **GPT-3/InstructGPT**: пересчитали задачи с «emergence» на непрерывных метриках — emergence исчезла, improvement плавный.

2. **Meta-анализ BIG-Bench**: из ~200 задач с «emergence» подавляющее большинство использовало exact match или multiple choice.

3. **Vision tasks**: искусственно создали «emergent abilities» в задачах компьютерного зрения, просто выбрав дискретную метрику. С непрерывной метрикой — плавный рост.

## Три феномена масштабирования

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond|Harnessing the Power of LLMs]]:

### 1. Emergent Abilities — внезапное появление

Паттерн: near-zero → sudden improvement.
Примеры: word sorting, logic grid puzzles, CoT reasoning.
Возможное объяснение (Schaeffer): артефакт метрики.

### 2. Inverse Scaling — с ростом хуже

Паттерн: **больше модель → хуже performance**.
Примеры:
- **Redefine-math**: «предположим, "+" означает умножение; 2 + 3 = ?» — большие модели сильнее привязаны к «2 + 3 = 5» (prior knowledge)
- **Memo-trap**: продолжить «Twinkle twinkle...» нестандартным способом — большие модели слишком хорошо запомнили оригинал
- **Into-the-unknown**: вопросы, на которые правильный ответ «я не знаю»

### 3. U-shaped Scaling — падение и восстановление

Паттерн: хорошо → плохо → снова хорошо.
Примеры:
- **Hindsight neglect**: маленькие модели угадывают, средние переусложняют, большие (GPT-4) понимают суть
- **NegationQA**: отрицание в вопросах запутывает средние модели

## Предсказуемость scaling (GPT-4 paper)

Из [[02 Areas/ML & DL/Papers/GPT 4.0|GPT-4]] §3: OpenAI разработал методологию **predictable scaling** для некоторых метрик:

$$L(C) = a \cdot C^b + c$$

Loss как степенная функция compute. Это позволяет предсказать loss модели по маленькому прокси-обучению (экономия миллионов долларов на валидацию).

Но: emergence (если она реальна) по определению **непредсказуема** из scaling curves — это нелинейный скачок, а не плавная экстраполяция.

## Текущий консенсус (2024-2025)

Сообщество **разделено**, но с уклоном:

### Позиция «emergence реальна, но reframed»

Многие исследователи согласны с Schaeffer в том, что **дискретные метрики преувеличивают** резкость перехода. Но утверждают, что существуют задачи, где качественный скачок **реален**, даже при непрерывных метриках:
- In-context learning с десятками примеров
- Multi-step reasoning chains длиной 5+
- Compositional generalization

### Позиция «emergence = артефакт»

Schaeffer et al. считают, что все наблюдаемые «emergence» объясняются метрикой + threshold. Модель плавно улучшается, просто мы этого не видим из-за грубой метрики.

### Практический consensus

Независимо от дебатов, сообщество согласно в нескольких вещах:
1. **Scaling laws для loss работают** — pre-training loss предсказуем
2. **Downstream task performance менее предсказуема** — даже если loss плавный, конкретные задачи могут «включаться» неожиданно
3. **Выбор метрики критичен** — нельзя делать выводы о «внезапности» по exact match
4. **Safety implications** остаются: даже если рост плавный, для dangerous capabilities (hacking, bioweapons) нет разницы между «плавным ростом до порога» и «внезапным скачком»

## Implications для AI Safety

Вопрос emergence имеет прямые policy implications (CSET Georgetown):

- Если capabilities emerge **непредсказуемо**: нужен мониторинг каждого нового масштаба, pre-deployment evals обязательны
- Если capabilities растут **плавно, но пороги — артефакт метрик**: можно предсказывать capabilities заранее по scaling curves, но нужно выбирать правильные метрики
- В обоих случаях: **eval infrastructure** (benchmarks, red-teaming) критически важна

## Хронология

| Год | Milestone | Значение |
|-----|-----------|----------|
| 2020 | Scaling Laws (Kaplan et al.) | Loss предсказуем как power law |
| 2020 | GPT-3 | Первые наблюдения «sudden abilities» (few-shot CoT) |
| 2022 | **Wei et al. — Emergent Abilities** | Формализация термина, 100+ примеров |
| 2022 | PaLM | Systematic documentation of emergence |
| 2022 | BIG-Bench | Benchmark, на котором измерялась emergence |
| 2023 | **Schaeffer et al. — Mirage** | Контраргумент: метрика как причина, **NeurIPS Best Paper** |
| 2023 | GPT-4 paper | Predictable scaling для loss, но не для emergence |
| 2024 | Chinchilla follow-ups | Уточнение scaling curves |
| 2025 | Ongoing debate | Consensus не достигнут |

## Key papers

- Wei et al. 2022 — *Emergent Abilities of Large Language Models* — оригинальное определение и каталог
- **Schaeffer et al. 2023 — *Are Emergent Abilities of Large Language Models a Mirage?*** — NeurIPS 2023 Outstanding Paper, контраргумент о метриках
- [[02 Areas/ML & DL/Papers/COT|Chain of Thought]] — CoT как emergent ability ≥ 100B
- [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond|Harnessing the Power of LLMs]] — три феномена: emergent / inverse / U-shaped
- [[02 Areas/ML & DL/Papers/GPT 4.0|GPT-4]] — predictable scaling vs unpredictable emergence
- Kaplan et al. 2020 — *Scaling Laws for Neural Language Models*

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — предсказуемый scaling loss vs непредсказуемые abilities
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — ключевой пример «emergent» ability
- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] — тоже наблюдалось как emergence
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — первая модель с масштабными emergent observations
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Inverse Scaling]] — противоположный феномен

## Дополнительные ресурсы

- [Wei et al. 2022 — Emergent Abilities (paper)](https://arxiv.org/abs/2206.07682) — оригинальная статья с полным каталогом
- [Schaeffer et al. 2023 — Mirage (paper)](https://arxiv.org/abs/2304.15004) — контраргумент, must-read
- [CSET Georgetown — Emergent Abilities Explainer](https://cset.georgetown.edu/article/emergent-abilities-in-large-language-models-an-explainer/) — доступное объяснение для policy audience
- [NeurIPS 2023 — Schaeffer presentation](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf) — полная версия best paper
