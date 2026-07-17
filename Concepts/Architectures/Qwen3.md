---
title: "Qwen3"
aliases: [Qwen3, Qwen3-235B, Qwen3-235B-A22B, Qwen 3]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Qwen3|Qwen3]]"
courses: []
sources:
  - "[Qwen3 Technical Report (2025)](https://arxiv.org/abs/2505.09388)"
  - "[Qwen Blog — Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)"
  - "[HuggingFace — Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)"
  - "[GitHub — QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)"
  - "[MoE training in Qwen3 and Kimi-K2](https://medium.com/clickbait-programming/moe-training-in-qwen3-235b-a22b-and-kimi-k2-c51f582d2ef5)"
---

# Qwen3

## Зачем эта модель появилась

К 2025 году в LLM-индустрии обозначился раскол: «быстрые» chat-модели (GPT-4o, Claude 3.5) отдельно, «думающие» reasoning-модели (o1, DeepSeek-R1, QwQ) отдельно. Пользователь вынужден выбирать — или скорость, или глубина.

**Qwen3** (Alibaba, май 2025) — первая серия, которая **объединяет thinking и non-thinking в одной модели**. Flagship: 235B параметров (22B активных), MoE, 119 языков. Модель сама решает, когда «думать», а когда отвечать мгновенно.

## Архитектура: MoE с thinking budget

### Базовая конфигурация

| Параметр | Qwen3-235B-A22B | Qwen3-32B (dense) |
|----------|------------------|--------------------|
| Total Params | 235B | 32B |
| Active Params | 22B | 32B |
| Layers | 94 | 64 |
| Experts | 128 (top-8) | — |
| Hidden Size | 4,096 | 5,120 |
| Query Heads | 64 | 40 |
| KV Heads | 4 (GQA) | 8 (GQA) |
| Context | 128K | 128K |

### MoE: 128 экспертов, top-8 routing

Каждый FFN-слой (кроме первого и последнего) содержит **128 fine-grained экспертов**, из которых маршрутизатор выбирает **8** на каждый токен. Это даёт высокую model capacity (235B знаний) при низком inference cost (22B compute).

**Отличие от Qwen2.5-MoE:** Qwen3-MoE **не использует shared experts** (в отличие от DeepSeek-V3, где 1 shared expert всегда активен). Qwen3 полагается на более эффективную балансировку нагрузки через global-batch load balancing loss.

### MoE Routing: детали маршрутизации

**Router architecture:** линейный слой $W_r \in \mathbb{R}^{d \times E}$, где $d$ — hidden size, $E = 128$ — число экспертов. Для каждого токена:

$$g_i = \text{softmax}(W_r \cdot h_t)_i$$

$$\text{Top-}k: \quad \text{выбрать 8 экспертов с наибольшими } g_i$$

**Global-batch load balancing loss:** в отличие от per-batch auxiliary loss (как в Switch Transformer), Qwen3 вычисляет балансировку нагрузки **по всему global batch** (собирается статистика со всех GPU). Это даёт более стабильный gradient signal и лучшую специализацию экспертов.

$$\mathcal{L}_{balance} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

где $f_i$ — доля токенов, отправленных эксперту $i$ (по global batch), $p_i$ — средняя routing probability эксперта $i$, $\alpha$ — коэффициент balancing loss.

**Зачем global vs local:** при local (per-batch) balancing каждый GPU видит маленькую выборку, и шум в статистике загрузки портит gradient. Global balancing собирает статистику по **всему кластеру**, что даёт точную картину и более мягкую балансировку.

## Unified Thinking: главная инновация

### Две режима в одной модели

**Thinking mode** — модель генерирует развёрнутую цепочку рассуждений в теге `<think>...</think>`, затем выдаёт финальный ответ. Для сложных задач: math, code, logic, multi-step reasoning.

**Non-thinking mode** — мгновенный ответ без внутренних рассуждений. Для простых запросов: перевод, факты, casual conversation.

### Thinking Budget: механизм в деталях

Ключевой механизм — **thinking budget** — контроль глубины рассуждений:

1. **Явное управление:** `enable_thinking=True/False` — полное включение/отключение
2. **Budget в токенах:** максимальное число токенов на `<think>` блок. Модель адаптирует глубину:
   - `budget=0` → чистый non-thinking mode (0 latency overhead)
   - `budget=1024` → краткое размышление (проверка ключевых шагов)
   - `budget=8192` → глубокий reasoning (multi-step exploration)
   - Без ограничений → полноценный extended thinking (до 32K+ токенов)

3. **Scaling behavior:** увеличение thinking budget даёт **monotonic improvement** на reasoning-бенчмарках. Это подтверждает test-time compute scaling hypothesis.

**Зачем thinking budget:** на простом вопросе «Какая столица Франции?» тратить 2000 токенов на рассуждения — waste. На сложной олимпиадной задаче — необходимость. Budget позволяет оптимизировать баланс latency/quality **динамически**, в зависимости от конкретного запроса.

### Обучение unified режима: четыре стадии

Четырёхстадийный процесс пост-обучения (post-training):

**Stage 1: Long CoT Cold Start (SFT)**
Файнтюнинг на высококачественных CoT-данных (math, code, science). Модель учится генерировать `<think>` блоки. Данные: partially от reasoning-моделей (R1-style), partially human-annotated. Цель — задать **формат** рассуждений и базовый reasoning level.

**Stage 2: Reasoning RL**
GRPO на задачах с верифицируемыми ответами. Reward = правильность ответа. Этот этап **максимизирует reasoning capability** — модель учится «думать лучше» через trial-and-error. Только thinking mode.

**Stage 3: Thinking Mode Fusion (SFT)**
SFT с **mixed данными**: часть с `<think>` (reasoning tasks), часть без (general tasks). Модель учится **выбирать режим** — когда думать, когда отвечать сразу. Ключевой этап для unified behavior.

**Stage 4: General RL**
Финальный RL для alignment. Смешанные rewards:
- Rule-based rewards для reasoning-задач (правильность)
- Reward model для open-ended задач (helpfulness, safety)
- Балансировка thinking/non-thinking quality

## Pre-training: три стадии

### Stage 1: General Pre-training
- **~30T токенов** на 119 языках
- Контекст: 4K токенов
- Данные: web (filtered), code, books, academic papers, multilingual corpora
- Этот этап занимает ~80% training compute

### Stage 2: Reasoning Curriculum
- Дополнительные **~5T токенов** с акцентом на STEM и код
- Увеличенная доля высококачественных reasoning-данных (math proofs, code solutions, scientific reasoning)
- Контекст увеличивается до 8K-16K
- Более медленный learning rate decay

### Stage 3: Long-context Adaptation
- **Сотни миллиардов токенов** документов длиной до 32K
- Техники расширения контекста:
  - **ABF (Adjusted Base Frequency):** увеличение $\theta$ в RoPE для экстраполяции
  - **YARN:** yet another RoPE extension — NTK-aware interpolation
  - **DCA (Dynamic Context Adaptation):** адаптивная длина контекста
- После этого этапа модель расширяется до 128K через дополнительный fine-tuning

## Мультиязычность: 119 языков

Qwen3 обучена на данных из 119 языков — значительно больше, чем LLaMA 3 (8 основных) или DeepSeek-V3 (в основном EN/ZH).

| Модель | Языки | Мультиязычный reasoning |
|--------|-------|------------------------|
| LLaMA 3 | 8 основных | Слабый на non-EN |
| DeepSeek-V3 | EN/ZH + ограниченно | Средний |
| **Qwen3** | **119 языков** | **Сильный** |

**Зачем это важно практически:** для production-систем в non-English странах модель должна уметь reasoning на целевом языке, а не только переводить с английского. Thinking на родном языке даёт лучшие результаты, чем "think in English → translate".

## Серия моделей

| Модель | Params | Тип | Назначение |
|--------|--------|-----|------------|
| Qwen3-235B-A22B | 235B/22B | MoE | Flagship |
| Qwen3-30B-A3B | 30B/3B | MoE | Edge deployment |
| Qwen3-32B | 32B | Dense | Сильный reasoning |
| Qwen3-14B | 14B | Dense | Баланс |
| Qwen3-8B | 8B | Dense | Mainstream |
| Qwen3-4B | 4B | Dense | Mobile |
| Qwen3-1.7B | 1.7B | Dense | Embedded |
| Qwen3-0.6B | 0.6B | Dense | Ultra-light |

**Покрытие:** от 0.6B (edge/IoT) до 235B (cloud flagship). Все модели поддерживают unified thinking/non-thinking.

## Результаты

| Бенчмарк | Qwen3-235B | DeepSeek-R1 | o1 | GPT-4o |
|-----------|------------|-------------|-----|--------|
| AIME 2024 | ~80% | 79.8% | 79.2% | 9.3% |
| MATH-500 | ~97% | 97.3% | 96.4% | 74.6% |
| LiveCodeBench | ~65% | 65.9% | — | — |
| Arena-Hard | ~90% | — | — | ~85% |
| MMLU | ~89% | 90.8% | 91.8% | 87.2% |

Qwen3-32B (dense) — одна из сильнейших open-weight dense моделей, превосходящая многие 70B модели. При этом **inference cost как у 22B модели** для MoE flagship.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Qwen2|Qwen2]] — предыдущее поколение
- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-V3|DeepSeek-V3]] — другой подход к MoE
- [[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral of Experts]] — пионер open MoE
- [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] — RL-алгоритм для thinking mode
- [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute|Test-time Compute]] — парадигма extended thinking

## Дополнительные ресурсы

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — полный отчёт
- [Qwen Blog — Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/) — официальный анонс с примерами
- [MoE Training in Qwen3](https://medium.com/clickbait-programming/moe-training-in-qwen3-235b-a22b-and-kimi-k2-c51f582d2ef5) — анализ MoE routing
