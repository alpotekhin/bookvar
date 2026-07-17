---
title: "Scaling Laws"
aliases: [neural scaling laws, Chinchilla scaling, compute-optimal training]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/PaLM 2]]"
  - "[[02 Areas/ML & DL/Papers/GPT 4.0]]"
courses: []
sources:
  - "[Cameron Wolfe — Scaling Laws for LLMs](https://cameronrwolfe.substack.com/p/llm-scaling-laws)"
  - "[Michael Brenndoerfer — Chinchilla Scaling Laws (Interactive)](https://mbrenndoerfer.com/writing/chinchilla-scaling-laws-compute-optimal-llm-training)"
  - "[Life Architect — Chinchilla in Plain English](https://lifearchitect.ai/chinchilla/)"
---

# Scaling Laws

## Зачем это нужно: предсказуемость стоимостью в миллионы долларов

Обучение GPT-4 стоило ~$100M. LLaMA 65B — ~$2M. Ошибка в выборе размера модели или объёма данных означает сожженные миллионы. Scaling laws дают возможность **предсказать performance до начала обучения** — экстраполировать от маленьких экспериментов к большим моделям.

Scaling laws — эмпирические степенные зависимости, описывающие как performance LLM улучшается с ростом:
- **N** — числа параметров
- **D** — объёма данных (в токенах)
- **C** — compute (в FLOP)

Два поворотных результата: **Kaplan et al. (2020, OpenAI)** и **Hoffmann et al. (2022, DeepMind / «Chinchilla»)**. Они дали противоположные рекомендации — и эта разница определила дизайн всех крупных моделей 2020-2024.

## Kaplan et al. (2020): больше параметров — важнее всего

### Power Laws

Kaplan и коллеги из OpenAI обнаружили, что test loss следует степенным законам по каждой из трёх переменных (при фиксации остальных):

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

Где $N_c, D_c, C_c$ — константы, $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$.

**Ключевое наблюдение:** power law не зависит от архитектурных деталей (ширина, глубина, число голов). Важен только **общий** размер модели $N$.

### Compute-Optimal по Kaplan

При фиксированном compute бюджете $C$:

$$N_{\text{opt}} \propto C^{0.73}$$

Параметры должны расти **быстрее**, чем данные. Практический вывод: при удвоении compute лучше увеличить модель, а не данные.

Это привело к стратегии GPT-3: **175B параметров, но всего 300B токенов**. Gopher (280B), Megatron-Turing NLG (530B) — та же логика: максимизируй N.

### Что Kaplan пропустил

Критическая ошибка: в экспериментах Kaplan данные были **недостаточно разнообразны по D**. Модели обучались до конвергенции, но на недостаточных объёмах данных. Это создало систематическое смещение: казалось, что N важнее D.

## Chinchilla (Hoffmann et al., 2022): баланс параметров и данных

### Исправление

DeepMind обучила **более 400 моделей** (от 70M до 16B параметров, на 5-500B токенов) и обнаружила, что Kaplan систематически переоценивал роль N:

**Chinchilla Rule:** при compute-optimal training N и D должны масштабироваться **одинаково**:

$$N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}$$

Грубое эмпирическое правило:

$$D_{\text{opt}} \approx 20 \cdot N$$

**~20 токенов на параметр** — compute-optimal training. Для модели с N параметров оптимально обучать на ~20N токенов.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scaling-laws/chinchilla-scaling.png]]
*Chinchilla scaling: compute-optimal allocation параметров и данных. Большинство моделей 2020-2021 были severely undertrained (источник: Alan D. Thompson / Life Architect)*

### Доказательство: Chinchilla 70B vs Gopher 280B

| Модель | Параметры | Данные | Compute | MMLU |
|--------|-----------|--------|---------|------|
| **Gopher** | 280B | 300B tokens | ~$5.76 \times 10^{23}$ FLOP | 60.0% |
| **Chinchilla** | 70B | 1.4T tokens | ~$5.76 \times 10^{23}$ FLOP | **67.6%** |

**Тот же compute, 4x меньше параметров, больше данных → лучший результат.** Gopher был catastrophically undertrained: 280B параметров требуют ~5.6T токенов по Chinchilla rule, а получил только 300B.

### Что это значило для индустрии

Chinchilla показала, что **почти все крупные модели 2020-2021 были undertrained**:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scaling-laws/chinchilla-dataset-sizes.png]]
*Размеры датасетов для различных моделей в контексте Chinchilla-optimal training (источник: Alan D. Thompson / Life Architect)*

| Модель | N | D (actual) | D (Chinchilla optimal) | Undertrained? |
|--------|---|-----------|----------------------|---------------|
| GPT-3 | 175B | 300B | 3.5T | **12x** |
| Gopher | 280B | 300B | 5.6T | **19x** |
| PaLM | 540B | 780B | 10.8T | **14x** |
| Megatron-NLG | 530B | 339B | 10.6T | **31x** |
| LLaMA 65B | 65B | 1.4T | 1.3T | Optimal |

## LLaMA: Inference-Budget Scaling

Meta (Touvron et al., 2023) внесла важную коррекцию: Chinchilla оптимизирует **training compute**, но в production основная стоимость — **inference**.

### Ключевой инсайт

> «Для достижения заданного performance уровня предпочтительна не та модель, которую быстрее обучить, а та, которая быстрее при инференсе.»

При inference: стоимость ∝ N (число параметров). Поэтому **меньшая модель, обученная дольше**, выгоднее в production:

| Модель | Параметры | Данные | Inference cost | Quality |
|--------|-----------|--------|---------------|---------|
| GPT-3 | 175B | 300B tokens | Очень высокий | Baseline |
| LLaMA-13B | 13B | 1T tokens | **13x дешевле** | **Лучше GPT-3** |

LLaMA-13B обучена на **77 токенов/параметр** (вместо Chinchilla-optimal ~20) — сильно «overtrained». Но это *целенаправленно*: больше данных → лучше quality/parameter → дешевле inference.

### Три режима масштабирования

| Парадигма | Стратегия | Формула | Примеры моделей |
|-----------|-----------|---------|-----------------|
| **Kaplan (2020)** | Максимизируй N | $N \propto C^{0.73}$ | GPT-3, Gopher |
| **Chinchilla (2022)** | Баланс N и D | $N \propto C^{0.50}, D \approx 20N$ | Chinchilla 70B |
| **LLaMA (2023)** | Оптимизируй inference | Overtrain small model | LLaMA 7B/13B/65B, Mistral 7B |

## GPT-4: Predictable Scaling

Из [[02 Areas/ML & DL/Papers/GPT 4.0]] Section 3 — OpenAI разработала метод предсказания capabilities **до завершения обучения**:

$$L(C) = a \cdot C^b + c$$

Loss как power law от compute. Fitted на моделях с **1000-10,000x меньшим compute** → точно предсказывает финальный loss GPT-4.

Практическое значение: можно оценить performance модели за $100K, а не за $100M. Это позволяет:
1. Планировать бюджет перед training run
2. Принимать go/no-go решения на ранней стадии
3. Предсказывать capabilities для safety evaluation

## PaLM 2: Data Quality > Scale

Из [[02 Areas/ML & DL/Papers/PaLM 2]] Section 2 — Google показала третий путь:

- PaLM 2-L **меньше** PaLM-540B по параметрам, но использует больше compute
- Meticulous data selection + efficient architecture → outperforms более крупные модели
- Масштабирование качества данных важнее масштабирования параметров

Это развитие Chinchilla insight: не только «больше данных», но и «лучше данные».

## Практические следствия для проектирования моделей

### Как выбрать размер модели для проекта

1. **Определи задачу и требуемый performance**
2. **Оцени inference budget** — сколько можно тратить на inference в production
3. **Выбери модель по inference cost** — LLaMA-логика: лучше маленькая, хорошо обученная
4. **Рассчитай D по Chinchilla:** $D \geq 20N$ для training-optimal, **$D \geq 50-100N$** для inference-optimal
5. **Оцени compute:** $C \approx 6ND$ (приблизительная формула для Transformer)

### Пример расчёта

Хочу модель уровня GPT-3 для production:
- GPT-3: 175B params, 300B tokens → $C \approx 3.1 \times 10^{23}$ FLOP
- По Chinchilla: 175B нужно 3.5T tokens (12x больше данных) → тот же compute, но лучше quality
- По LLaMA: 13B params, 1T tokens → $C \approx 7.8 \times 10^{22}$ FLOP → **4x меньше training compute, 13x меньше inference cost**, quality >= GPT-3

### Современные следствия (2024-2025)

Mistral 7B, Phi-2, Gemma 2B — все следуют LLaMA-логике:
- Маленькие модели (2B-7B), overtrained на 1-15T tokens
- Data curation важнее data volume
- Inference-optimal дизайн для edge deployment

## Ограничения scaling laws

1. **Только pre-training loss:** scaling laws предсказывают loss, не downstream performance. Emergent abilities могут появляться непредсказуемо.
2. **Архитектурная зависимость:** формулы откалиброваны на Transformer. Для Mamba, RWKV, SSM — нужны свои scaling laws.
3. **Data quality не учтена:** Chinchilla предполагает фиксированное качество данных. PaLM 2 показал, что лучшие данные меняют кривые.
4. **Saturation:** power law не может продолжаться бесконечно — должен быть предел.
5. **Post-training:** RLHF/DPO вносят нелинейные улучшения, не предсказуемые scaling laws.

## Хронология

| Год | Milestone | Следствие |
|-----|-----------|-----------|
| 2017 | Hestness et al. — первые scaling laws для DL | Power laws для разных задач |
| 2020 | **Kaplan et al.** — scaling laws для LLM | «Больше параметров — лучше» → GPT-3 |
| 2022 | **Hoffmann et al. (Chinchilla)** | «Баланс N и D» → конец era undertrained models |
| 2023 | **LLaMA** — inference-optimal scaling | «Overtrain small models» → open-source revolution |
| 2023 | **GPT-4** — predictable scaling | Предсказание capabilities до training |
| 2024 | Phi-2, Gemma — data-centric scaling | Data quality > data quantity |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — фаза, для которой определены scaling laws
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]] — capabilities, не предсказуемые scaling laws
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — inference-optimal training
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — Kaplan strategy
- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — overtrained small model

## Дополнительные ресурсы

- [Cameron Wolfe — Scaling Laws for LLMs](https://cameronrwolfe.substack.com/p/llm-scaling-laws) — лучший deep dive от GPT-3 до o3
- [Michael Brenndoerfer — Interactive Chinchilla](https://mbrenndoerfer.com/writing/chinchilla-scaling-laws-compute-optimal-llm-training) — интерактивный калькулятор
- [Life Architect — Chinchilla in Plain English](https://lifearchitect.ai/chinchilla/) — доступное объяснение без формул
- [Jon Vet — Scaling Laws for LLM Pretraining](https://www.jonvet.com/blog/llm-scaling-laws) — практический гайд
