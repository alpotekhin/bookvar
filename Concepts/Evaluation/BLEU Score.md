---
title: "BLEU Score"
aliases: [BLEU, BLEU Score, Bilingual Evaluation Understudy]
type: concept
category: Evaluation
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/T5|T5]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 06 — Seq2Seq|CS224N L06]]"
sources:
  - "[Papineni et al. — BLEU: a Method for Automatic Evaluation of Machine Translation (2002)](https://aclanthology.org/P02-1040/)"
  - "[Post — A Call for Clarity in Reporting BLEU Scores (2018)](https://aclanthology.org/W18-6319/)"
  - "[sacreBLEU GitHub](https://github.com/mjpost/sacrebleu)"
---

# BLEU Score — Bilingual Evaluation Understudy

## Зачем это нужно: автоматическая оценка перевода

До BLEU качество машинного перевода оценивалось только людьми: дорого, медленно, несовместимо между исследованиями. Papineni et al. (2002) предложили **BLEU** — автоматическую метрику, коррелирующую с human judgment на корпусном уровне. Это позволило быстро сравнивать MT-системы и стало стандартом на 20+ лет.

Идея простая: хороший перевод содержит те же **n-граммы**, что и референсный (эталонный) перевод человека. BLEU измеряет эту долю.

## Формула

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

где:
- $p_n$ — **precision для n-грамм** порядка $n$
- $w_n$ — веса, обычно $w_n = 1/N$ (равномерно)
- $N$ — максимальный порядок n-грамм, стандартно $N = 4$
- $BP$ — **brevity penalty** (штраф за короткие переводы)

### Modified n-gram precision

Наивная precision:

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \text{Count}(\text{n-gram})}{\sum_{\text{n-gram} \in \hat{y}} \text{TotalCount}(\text{n-gram})}$$

Проблема: перевод `"the the the the"` даст $p_1 = 1.0$ если `the` есть в референсе. **Modified precision** ограничивает счётчик каждой n-граммы её максимумом в референсе (**clipping**):

$$p_n = \frac{\sum_{\text{n-gram}} \min(\text{Count}_{\hat{y}}, \max_{\text{ref}} \text{Count}_{\text{ref}})}{\sum_{\text{n-gram}} \text{Count}_{\hat{y}}}$$

Для `"the the the the"` (референс `"the cat sat on the mat"`): clipped count = 2, total = 4, $p_1 = 0.5$.

### Brevity penalty

Без штрафа за длину MT-система могла бы выдавать одно-двухсловные переводы с высокой precision. **Brevity penalty**:

$$BP = \begin{cases} 1 & \text{если } c > r \\ \exp(1 - r/c) & \text{если } c \le r \end{cases}$$

где $c$ — длина кандидата, $r$ — длина ближайшей по длине референсной строки. Перевод короче референса экспоненциально штрафуется. Recall напрямую не входит в BLEU — его роль играет BP.

### Геометрическое среднее

Precision'ы $p_1, \ldots, p_N$ комбинируются через **геометрическое среднее** (сумма $\log p_n$). Это жёстче арифметического: если $p_4 = 0$, весь BLEU обнуляется. Поэтому на предложениях короче 4 слов BLEU плохо работает — нужна **smoothing** (например, `+1` к числителю и знаменателю, методы Chen & Cherry 2014).

## Corpus-level vs sentence-level

**Важный нюанс:** BLEU задумывался как **corpus-level** метрика. Числители и знаменатели агрегируются по всему тесту, затем берётся отношение. Sentence-level BLEU (для каждого предложения отдельно) работает плохо из-за геометрического среднего и нулевых $p_n$.

Для sentence-level обычно используют **chrF**, **BLEURT** или sentence-BLEU со smoothing.

## sacreBLEU — стандарт воспроизводимости

В 2018 Matt Post написал **"A Call for Clarity in Reporting BLEU Scores"**: оказалось, что разные реализации BLEU дают разные числа из-за tokenization, обработки пунктуации, регистра. Два paper'а с "BLEU=28" могли быть несравнимы.

**sacreBLEU** стандартизирует:
- Единая tokenization (`mteval-v13a`)
- Детерминированная обработка
- Signature string вида `BLEU+case.mixed+lang.en-de+...` для полной воспроизводимости

С 2018+ стандарт — репортить **sacreBLEU** с signature. Числа без signature некорректно сравнивать.

## Типичные значения

Шкала BLEU — от 0 до 100 (или 0 до 1). Интерпретация зависит от языковой пары и домена:

| Задача | BLEU |
|--------|------|
| EN-DE (news, WMT SOTA 2023) | 38-42 |
| EN-FR (news, WMT SOTA) | 42-45 |
| EN-ZH (news) | 22-28 |
| Low-resource pairs | 10-20 |
| Human translation (как baseline) | 60-70 |

**Правило большого пальца:**
- BLEU < 15 — перевод плохой, смысл часто теряется
- BLEU 25-35 — понятный перевод со стилистическими ошибками
- BLEU > 40 — качественный перевод, близкий к human
- Разница в 1 BLEU между сильными системами — значима (если подтверждена statistical significance test)

## BLEU в знаковых работах

[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]] (Vaswani et al., 2017) репортил:
- **EN-DE (newstest2014):** Transformer-Big — **28.4 BLEU** (SOTA на тот момент)
- **EN-FR (newstest2014):** Transformer-Big — **41.8 BLEU**

[[02 Areas/ML & DL/Papers/T5|T5]] (Raffel et al., 2020) репортил BLEU на WMT для EN-DE/FR/RO — на всех задачах T5-11B выжимал SOTA.

Google NMT (2016), ConvS2S (2017), Transformer (2017), Transformer-Big + back-translation (2018) — эта последовательность улучшений на EN-DE прошла от ~20 до ~35 BLEU и вывела MT на production-уровень качества.

## Ограничения BLEU

### Surface-level, не семантический

BLEU считает **лексические** совпадения n-грамм. Два проблемных сценария:

1. **Синонимы не засчитываются**: `"car"` vs `"automobile"` — разные n-граммы, даже если смысл идентичен
2. **Перефразирования штрафуются**: семантически эквивалентные, но структурно разные переводы получают низкий BLEU

Пример: референс `"The cat is on the mat"`, кандидат `"A feline sits upon the rug"` — смысл тот же, но $p_1 \approx 0.17$, $p_4 = 0$, BLEU $\approx 0$.

### Не коррелирует с human judgment на уровне предложения

Papineni показал корреляцию на **корпусном** уровне (десятки тысяч предложений). На отдельном предложении BLEU — шумная метрика, плохо предсказывающая качество.

### Плохо для nucleus generation (summarization, QA, dialogue)

В задачах с **несколькими правильными ответами** (саммаризация, вопросы-ответы, диалог) один референс не покрывает пространство приемлемых ответов. BLEU жёстко штрафует за отклонения от конкретной формулировки.

### Не улавливает fluency

Кандидат с правильными n-граммами, но синтаксически сломанный, получит высокий BLEU. Обратно — грамматически совершенный перефраз получит низкий.

## Альтернативы и преемники

| Метрика | Идея | Плюс к BLEU |
|---------|------|-------------|
| **chrF** | Character n-gram F1 | Лучше для морфологически богатых языков |
| **METEOR** | Учитывает синонимы (WordNet), stemming | Выше корреляция с human |
| **BERTScore** | Cosine similarity BERT embeddings | Семантическое сопоставление |
| **BLEURT** | Fine-tuned BERT на human ratings | Обученная метрика |
| **COMET** | Нейросеть на WMT human judgments | SOTA для MT (2020+) |

С 2020+ **COMET** стал де-факто стандартом для WMT. Но BLEU остаётся широко используемым как легковесный baseline и для сравнения с историческими работами.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]] — метрика language modeling, complementary к BLEU
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, оцениваемая через BLEU на WMT
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]] — парадигма MT, где BLEU — основная метрика
- [[02 Areas/ML & DL/Concepts/Evaluation/MMLU|MMLU]] — бенчмарк, где BLEU не применим
