---
title: "Sampling"
aliases: [decoding strategies, temperature sampling, nucleus sampling, top-p sampling, top-k sampling, greedy decoding, beam search]
type: concept
status: legacy
category: Inference
papers:
  - "[[02 Areas/ML & DL/Papers/Speculative Decoding]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[Holtzman et al. — The Curious Case of Neural Text Degeneration (2020)](https://arxiv.org/abs/1904.09751)"
  - "[Hugging Face — How to generate text (2020)](https://huggingface.co/blog/how-to-generate)"
---

# Sampling (Decoding Strategies)

## Зачем это нужно: проблема выбора следующего токена

Авторегрессивная языковая модель на каждом шаге выдаёт **распределение** над словарём: $P(x_t | x_{<t}; \theta)$. Но распределение — это не текст. Нужна **стратегия выбора** конкретного токена из этого распределения.

Представь, что ты пишешь рассказ и на каждом шаге у тебя есть 50 000 вариантов следующего слова с разными вероятностями. Взять самое вероятное? Бросить кубик? Выбрать из топ-10? Каждая стратегия даёт радикально разный текст — от скучного повторяющегося до хаотического бессмысленного.

Правильный выбор стратегии декодирования — **критически важен** для качества генерации.

## Greedy Decoding: самый простой, самый плохой

$$x_t = \arg\max P(x_t | x_{<t}; \theta)$$

Всегда выбираем наиболее вероятный токен. Детерминировано, быстро, но:

- **Repetition**: модель застревает в циклах («the the the» или повтор целых фраз)
- **Local optima**: жадный выбор на каждом шаге ≠ оптимальная последовательность. Пример: «The dog walked to the...» → greedy выбирает «store» (0.3), хотя «park» (0.25) + «bench» (0.4) дали бы лучшую последовательность
- **Generic outputs**: модель выбирает «безопасные» высокочастотные слова, игнорируя интересные продолжения

Holtzman et al. (2020) назвали это **neural text degeneration**: «the most boring possible continuation is often the most likely one».

## Temperature Sampling: регулятор «творчества»

Модифицируем логиты перед softmax:

$$P(x_t = w | x_{<t}) = \text{softmax}(z_w / T)$$

| Temperature $T$ | Эффект | Когда использовать |
|-----------------|--------|---------------------|
| $T \to 0$ | → greedy decoding (вся масса на одном токене) | Детерминированные задачи |
| $T = 1.0$ | Оригинальное распределение модели | Стандартный baseline |
| $T = 0.3{-}0.5$ | Sharper: меньше разнообразия, более предсказуемо | Code generation, factual QA |
| $T = 0.7{-}0.9$ | Слегка flatter: больше разнообразия | Dialogue, creative writing |
| $T > 1.0$ | Flatter: высокая энтропия, «творчество» | Brainstorming (осторожно!) |

**Интуиция**: temperature делит **логиты** (до softmax), а не вероятности. При $T < 1$ разница между вероятностями увеличивается (top token доминирует). При $T > 1$ — сглаживается (все токены более равновероятны).

**Проблема**: при высоком $T$ даже совсем невероятные токены (опечатки, бессмыслица) получают ненулевой шанс. Temperature alone не гарантирует качество.

## Top-k Sampling: отсечение хвоста

Оставляем только $k$ наиболее вероятных токенов, перераспределяем вероятности:

$$\mathcal{Z} = \{w : \text{rank}(P(w|x_{<t})) \leq k\}$$
$$P'(x_t = w) = \frac{P(w|x_{<t})}{\sum_{w' \in \mathcal{Z}} P(w'|x_{<t})} \quad \text{если } w \in \mathcal{Z}, \text{ иначе } 0$$

GPT-2 использовал $k = 40$ по умолчанию. Top-k отсекает «мусорный хвост» распределения — токены с ничтожной вероятностью, которые temperature sampling может выбрать.

**Проблема top-k**: фиксированный $k$ **не адаптируется** к форме распределения:
- Если модель уверена (вся масса на 2-3 токенах), $k = 40$ включает мусор
- Если модель не уверена (плоское распределение), $k = 40$ может отсечь хорошие варианты

## Nucleus Sampling (Top-p): адаптивный top-k

Holtzman et al. (2020) предложили элегантное решение:

$$V_p = \min\{V : \sum_{w \in V} P(w|x_{<t}) \geq p\}$$

Отбираем **минимальный** набор токенов с **суммарной вероятностью** $\geq p$.

**Адаптивность**: если модель уверена (90% на одном токене), при $p = 0.9$ выбираем 1-2 токена. Если не уверена (много токенов по 5-10%), выбираем 10-20 токенов. Число кандидатов **автоматически** подстраивается под confidence модели.

Типичные значения: $p = 0.9$ или $p = 0.95$.

**Пример**:
```
Распределение: [the: 0.35, a: 0.25, my: 0.15, his: 0.10, her: 0.08, ...]
Top-k (k=3): [the, a, my]                    — фиксированно 3 токена
Top-p (p=0.9): [the, a, my, his, her, ...]    — адаптивно, пока сумма < 0.9
```

## Beam Search: несколько гипотез параллельно

Держим $B$ лучших гипотез (beams). На каждом шаге:
1. Расширяем каждый beam на $|V|$ вариантов → $B \times |V|$ кандидатов
2. Оставляем top-$B$ по суммарной log-probability

**Свойства**:
- Детерминировано (при одинаковом seed)
- Приблизительный поиск оптимальной последовательности
- Стандарт для **структурированного output**: MT, summarization
- Обычно $B = 4{-}6$

**Проблемы** для open-ended generation:
- **Repetition**: ещё хуже чем greedy — beams «коллапсируют» в одинаковые последовательности
- **Generic outputs**: высоковероятные последовательности = скучные
- **Length bias**: короткие ответы имеют более высокую log-probability

**Модификации**:
- **Length penalty**: нормализация score на длину
- **No-repeat n-gram**: запрет повтора n-грамм (обычно $n = 3$)
- **Diverse beam search**: штраф за похожие beams

## Сравнительная таблица

| Стратегия | Deterministic | Diversity | Quality | Speed | Best for |
|-----------|--------------|-----------|---------|-------|----------|
| **Greedy** | Да | Низкая | Repetitive | Быстро | Debug, reproducibility |
| **Beam search** ($B=4$) | Да | Низкая | Structured | Средне | MT, summarization |
| **Temperature** ($T=0.7$) | Нет | Средняя | Хорошая | Быстро | General purpose |
| **Top-k** ($k=40$) | Нет | Средняя | Хорошая | Быстро | GPT-2 style |
| **Top-p** ($p=0.9$) | Нет | Высокая | Хорошая | Быстро | Modern default |
| **Top-p + temp** | Нет | Настраиваемая | Лучшая | Быстро | Production LLM |

## Практические рекомендации

| Задача | Параметры | Почему |
|--------|-----------|--------|
| **Machine Translation** | Beam search ($B=4{-}6$) | Структурированный, детерминированный output |
| **Summarization** | Beam search + length penalty | Контроль длины |
| **Code generation** | $T = 0.0{-}0.2$ | Точность > diversity; синтаксические ошибки критичны |
| **Dialogue** | $T = 0.7{-}0.9$, $p = 0.9$ | Разнообразие и естественность |
| **Creative writing** | $T = 0.8{-}1.0$, $p = 0.95$ | Максимальная свобода при базовом качестве |
| **Factual QA** | $T = 0.0$ или $p = 0.9$, $T = 0.3$ | Минимизация hallucinations |
| **Self-Consistency** | $T = 0.5{-}0.7$, $k = 40$, sample 40 paths | Diversity для majority voting |

## Комбинирование стратегий

В production обычно комбинируют несколько стратегий:

```python
# Типичные параметры ChatGPT-like API
temperature = 0.7     # Базовый уровень разнообразия
top_p = 0.9           # Отсечение совсем маловероятных
frequency_penalty = 0.5  # Штраф за повторы
presence_penalty = 0.3   # Поощрение новых тем
```

**Repetition penalties**: дополнительный механизм борьбы с повторами:
- **Frequency penalty**: штраф пропорционален числу появлений токена
- **Presence penalty**: фиксированный штраф за уже использованный токен

## Speculative Decoding: ускорение без изменения стратегии

Из [[02 Areas/ML & DL/Papers/Speculative Decoding]] (Leviathan et al., 2023):

Speculative decoding **ускоряет** любую стратегию sampling, не изменяя output:

1. Маленькая модель $M_q$ генерирует $\gamma$ draft-токенов
2. Большая модель $M_p$ верифицирует все параллельно
3. Принятые оставляем, отвергнутые корректируем
4. **Гарантия**: выходное распределение идентично $M_p$ при обычном декодировании

Speedup 2-3x на T5-XXL (11B) без изменения outputs. Подробнее: [[02 Areas/ML & DL/Concepts/Inference/Speculative Decoding|Speculative Decoding]].

## Self-Consistency: sampling для reasoning

Из [[02 Areas/ML & DL/Papers/Self-Consistency|Self-Consistency]] (Wang et al., 2023):

Sampling как инструмент **улучшения reasoning**: семплируем $K$ diverse CoT-путей → majority vote по финальным ответам. Интуиция: правильные рассуждения сходятся к одному ответу, неправильные — расходятся.

На GSM8K с PaLM-540B: CoT-greedy 56.5% → **Self-Consistency 74.4%** (+17.9%). Подробнее: [[02 Areas/ML & DL/Concepts/Reasoning/Self-Consistency|Self-Consistency]].

## Key papers

- [[02 Areas/ML & DL/Papers/Speculative Decoding]] — 2-3x ускорение inference без изменения output distribution
- Holtzman et al. (2020) — nucleus sampling (top-p), анализ neural text degeneration

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/Speculative Decoding|Speculative Decoding]] — ускорение sampling через draft-verify
- [[02 Areas/ML & DL/Concepts/Reasoning/Self-Consistency|Self-Consistency]] — majority vote по sampled reasoning paths
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектуры, для которых sampling актуален
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]] — задача, определяющая распределение
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — CoT + sampling = self-consistency

## Дополнительные ресурсы

- [Holtzman et al. — The Curious Case of Neural Text Degeneration (2020)](https://arxiv.org/abs/1904.09751) — nucleus sampling и анализ проблем greedy/beam search
- [Hugging Face — How to generate text](https://huggingface.co/blog/how-to-generate) — интерактивные визуализации стратегий декодирования
- [Fan et al. — Hierarchical Neural Story Generation (2018)](https://arxiv.org/abs/1805.04833) — top-k sampling
