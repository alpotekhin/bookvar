---
title: "Machine Translation"
aliases: [Machine Translation, MT, Машинный перевод, NMT]
type: concept
status: legacy
category: NLP
papers: []
courses: []
sources:
  - "[Bahdanau et al. — Neural MT by Jointly Learning to Align and Translate (2014)](https://arxiv.org/abs/1409.0473)"
  - "[Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)"
  - "[NLLB Team — No Language Left Behind (2022)](https://arxiv.org/abs/2207.04672)"
  - "[WMT Shared Tasks](https://www.statmt.org/wmt24/)"
---

# Machine Translation — машинный перевод

## Задача

Machine Translation (MT) — автоматический перевод текста с одного естественного языка на другой. Это **классическая seq2seq задача**: вход — последовательность токенов на исходном языке, выход — последовательность на целевом. MT исторически была главным драйвером NLP: на нём обкатывались почти все значимые архитектуры (encoder-decoder, attention, Transformer).

## Эволюция подходов

### Rule-based MT (1950-1990)

Вручную написанные правила трансформации: морфология → синтаксис → семантика → обратная генерация. SYSTRAN, METEO. Требовало огромный труд лингвистов на каждую языковую пару, плохо обобщалось, ломалось на идиомах.

### Statistical MT (SMT, 1990-2014)

Статистические модели из параллельных корпусов. Классика — **фразовая SMT**: Moses, Google Translate до 2016.

**Word alignment:** IBM-модели (1-5), выравнивание слов в параллельных предложениях.

**Phrase-based translation** (Koehn, 2003): перевод через таблицы «фраза → фраза» с вероятностями + n-gram языковая модель:

$$\arg\max_{t} P(s|t) \cdot P_{LM}(t)$$

SMT был **state of the art 15 лет**, но страдал от: ограниченного контекста (n-gram), отсутствия учёта дальних зависимостей, необходимости в feature engineering.

### Neural MT (NMT, 2014+)

**Seq2Seq с RNN** (Sutskever 2014, Cho 2014): encoder читает входное предложение в вектор, decoder генерирует перевод. Первая end-to-end нейронная модель, заметно превзошедшая SMT на коротких предложениях.

**Проблема:** всё предложение сжимается в один фиксированный вектор — bottleneck, плохо работает на длинных предложениях.

**Attention (Bahdanau 2014, Luong 2015):** decoder на каждом шаге «обращается» к релевантным позициям encoder. Это устранило bottleneck и стало прорывом — именно здесь родилась [[Attention Mechanism|механика внимания]], которая потом взорвала весь NLP.

**Transformer (Vaswani 2017):** полностью self-attention, без рекурренции. Параллельное обучение, лучше качество, лучше масштабируется. С тех пор **все production MT — на Transformer**.

### LLM-based MT (2022+)

Современный поворот: специализированные MT-модели уступают место **general-purpose LLM**. GPT-4, Claude, DeepSeek в zero-shot переводят лучше многих обученных на миллиардах параллельных пар NMT-моделей — особенно для редких языков и прагматики (идиомы, регистр, контекст).

Отдельные MT-модели сохраняются для:
- Низкой латентности / низкой стоимости inference.
- Специализированных доменов (медицина, право).
- Low-resource языков, где нужна точная настройка.

## Архитектуры

### Encoder-Decoder Transformer

Классика для MT: encoder кодирует исходный текст, decoder с cross-attention на encoder'а генерирует перевод. Примеры: Google's original Transformer, Marian, mBART, NLLB.

### Decoder-only LLM

Современные LLM (GPT, LLaMA) — decoder-only. Перевод выполняется как prompt-based generation:

```
Translate from English to French:
English: "Hello, how are you?"
French:
```

В zero-shot / few-shot режиме достигает SOTA на многих парах. Flores-200 benchmark демонстрирует, что Claude и GPT-4 превосходят NLLB на большинстве языков.

## Evaluation

### BLEU Score

[[BLEU Score]] — исторически главный метрик, введён Papineni et al. (2002). Измеряет совпадение n-grams гипотезы с reference translation (до 4-gram) плюс brevity penalty.

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

**Плюсы:** дешёвый, корпусный, хорошо коррелирует с human judgement на больших корпусах.

**Минусы:** не учитывает синонимы, грамматические перефразировки, семантику. Не работает на уровне предложения.

### Современные метрики

- **chrF / chrF++** — character n-gram F-score, лучше на морфологически богатых языках.
- **COMET** (Unbabel) — **обученная нейросетевая метрика**, fine-tuned на human ratings. Коррелирует с людьми существенно лучше BLEU.
- **BLEURT** (Google) — другая learned metric, на базе BERT.
- **MetricX**, **xCOMET** — современные варианты.

WMT shared tasks (ежегодные) с 2022 официально рекомендуют COMET/chrF как основные метрики, BLEU — как secondary. Но BLEU всё ещё везде для сравнимости с прошлыми работами.

### Human evaluation

Golden standard: MQM (Multidimensional Quality Metrics) — ассесоры помечают ошибки по категориям (accuracy, fluency, style). Дорого, но единственный надёжный источник ground truth.

## Low-resource MT

**Проблема:** из 7000+ языков мира параллельные корпуса доступны только для ~100. Для большинства языков <10K параллельных предложений — недостаточно для обучения NMT с нуля.

### Подходы

1. **Transfer learning** — обучить модель на high-resource pair (en-fr), fine-tune на low-resource (en-sw). mBART, mT5 используют этот подход.

2. **Multilingual models** — одна модель на много языков, с параметрами, общими между ними. **NLLB-200** (Meta, 2022) покрывает 200 языков, включая много low-resource африканских и индийских.

3. **Back-translation** (Sennrich 2016) — обучить обратную модель tgt→src, перевести monolingual tgt корпус в псевдо-src, использовать как дополнительные тренировочные данные. Стандартная техника увеличения данных.

4. **Pivot translation** — перевод через промежуточный язык (обычно английский): sw → en → ja. Работает, но накапливает ошибки.

5. **LLM zero-shot** — большие LLM переводят low-resource языки в zero-shot лучше специализированных моделей, потому что видели смесь monolingual данных на pre-training.

### Вызовы

- **Domain mismatch** — доступные корпуса часто религиозные (Библия) или правовые (ООН), а переводить надо диалоги или новости.
- **Script / orthography** — не все языки имеют стандартизированную письменность.
- **Морфология** — агглютинативные языки (финский, турецкий, суахили) требуют осторожной токенизации (BPE, SentencePiece).
- **Code-switching** — в реальности люди перемешивают языки, а параллельные корпуса — строго моноязычные.
- **Evaluation** — BLEU и COMET плохо работают на low-resource, некому создавать references.

## Context и document-level MT

Sentence-level MT игнорирует документный контекст: `pronouns`, `tense consistency`, `terminology`. Document-level MT использует окно из нескольких предложений или весь документ. Главный бенефициарий появления длинного контекста в LLM — именно MT: Claude 3.5/4 может держать весь документ и переводить с консистентной терминологией.

## Related concepts

- [[Seq2Seq]] — базовая архитектура NMT
- [[Attention Mechanism]] — родилась именно в MT
- [[Attention Is All You Need|Transformer]] — архитектура, сделавшая NMT SOTA
- [[BLEU Score]] — классическая метрика
- [[Cross-Attention]] — ключевой механизм в encoder-decoder
- [[BPE]] — стандартная токенизация для MT
