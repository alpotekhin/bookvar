---
title: "Cross-Attention"
aliases: [Cross-Attention, Encoder-Decoder Attention]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/T5|T5]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 08 — Transformers|CS224N L08]]"
sources:
  - "[Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)"
---

# Cross-Attention

## Определение

**Cross-Attention** (encoder-decoder attention) — механизм внимания, при котором queries приходят из одного представления (обычно decoder), а keys и values — из другого (обычно encoder). В отличие от [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]], где Q, K, V вычисляются из одной и той же последовательности, cross-attention **связывает два разных представления**:

$$\text{CrossAttn}(X_{dec}, X_{enc}) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) V$$

где:
- $Q = X_{dec} W^Q$ — queries из decoder hidden states
- $K = X_{enc} W^K$ — keys из encoder hidden states
- $V = X_{enc} W^V$ — values из encoder hidden states

## Роль в оригинальном Transformer

В архитектуре [[02 Areas/ML & DL/Papers/Attention Is All You Need|Transformer]] (Vaswani et al., 2017) каждый decoder layer содержит **три подслоя**:

1. **Masked Self-Attention** — decoder «смотрит» на предыдущие сгенерированные токены
2. **Cross-Attention** — decoder «смотрит» на выход encoder
3. **Feed-Forward Network** — поточечная нелинейная трансформация

Cross-attention — это механизм, через который decoder **получает доступ к входной последовательности**. Без него decoder не знал бы, что именно нужно перевести/суммаризировать/ответить.

### Как это работает на уровне токенов

При переводе предложения "Кошка сидела на коврике" → "The cat sat on the mat":

1. Encoder обрабатывает все входные токены и формирует контекстуализированные представления
2. При генерации слова "cat" decoder формирует query из текущего состояния
3. Cross-attention сопоставляет этот query с keys всех входных токенов
4. Максимальный вес внимания приходится на токен "Кошка"
5. Соответствующее value используется для генерации следующего слова

Это создаёт **мягкое выравнивание** (soft alignment) между входными и выходными позициями — аналог alignment table в классическом статистическом переводе, но обучаемый end-to-end.

## Encoder-decoder модели: T5 и BART

Encoder-decoder архитектуры активно используют cross-attention:

**[[02 Areas/ML & DL/Papers/T5|T5]]** (Text-to-Text Transfer Transformer):
- Encoder обрабатывает входной текст с bidirectional self-attention
- Decoder генерирует выход, используя cross-attention к encoder
- Все задачи NLP формулируются как text-to-text: вход → encoder → cross-attention → decoder → выход
- Cross-attention позволяет decoder выборочно фокусироваться на нужных частях входа

**BART:**
- Похожая архитектура, но с denoising pre-training
- Cross-attention особенно важен для задач суммаризации — decoder учится «выбирать» ключевую информацию из encoder

### Преимущество encoder-decoder

Cross-attention даёт encoder-decoder моделям структурное преимущество в задачах **с чётким разделением входа и выхода** (перевод, суммаризация, ответы на вопросы). Encoder обрабатывает вход bidirectional — каждый токен «видит» все остальные, что даёт более богатые представления. Decoder через cross-attention получает доступ к этим представлениям.

## Отсутствие cross-attention в decoder-only моделях

Современные LLM (GPT, LLaMA, Mistral) используют **decoder-only** архитектуру — чистый causal self-attention без encoder и, соответственно, без cross-attention. Как же они решают задачи, для которых cross-attention кажется необходимым?

**In-context learning заменяет cross-attention.** Вместо разделения вход/выход на уровне архитектуры, decoder-only модели получают всё в одной последовательности:

```
[Переведи на английский: Кошка сидела на коврике. Перевод:]
```

Causal self-attention на позиции генерируемого токена «видит» все предыдущие токены, включая входной текст. Это функционально аналогично cross-attention, но **без архитектурного разделения** — модель сама учится разделять «вход» и «выход» в единой последовательности.

**Почему decoder-only победили:**
- Проще архитектура — один тип блока вместо двух
- Лучше масштабируется — один поток вычислений
- Более гибкий — не нужно заранее определять, где вход, а где выход
- При достаточном масштабе in-context learning работает не хуже cross-attention

## Cross-Attention в мультимодальных моделях

В мультимодальных моделях cross-attention переживает **возрождение** — он связывает разные модальности:

### Image → Text (vision-language модели)

**Flamingo** (DeepMind, 2022):
- Замороженная LLM + замороженный vision encoder
- Cross-attention слои вставлены между self-attention слоями LLM
- Keys и values — из visual features (закодированные изображения)
- Queries — из текстовых hidden states LLM
- Модель «смотрит» на изображение при генерации текста

**LLaVA** (альтернативный подход):
- Вместо cross-attention использует projection layer — визуальные токены проецируются в текстовое пространство и конкатенируются с текстом
- Более простой подход, полагающийся на self-attention

### Audio → Text

**Whisper** (OpenAI):
- Encoder-decoder с cross-attention
- Audio encoder обрабатывает мел-спектрограмму
- Text decoder через cross-attention «читает» audio features
- Классический пример, где разделение encoder/decoder естественно

### Diffusion models

В Stable Diffusion cross-attention связывает **текстовый prompt** (закодированный через CLIP) с **изображением** (в UNet):
- Q из features изображения на каждом уровне UNet
- K, V из текстового embedding
- Каждый пиксель «смотрит» на релевантные слова prompt
- Это основной механизм контроля генерации через текст

## Сравнение с Self-Attention

| Свойство | Self-Attention | Cross-Attention |
|----------|---------------|-----------------|
| Источник Q, K, V | одна последовательность | Q из одной, K/V из другой |
| Назначение | контекстуализация внутри последовательности | связь между представлениями |
| В каких моделях | все Transformer-модели | encoder-decoder, мультимодальные |
| Decoder-only LLM | да (causal) | нет (заменён in-context learning) |
| Вычислительная сложность | $O(n^2)$ по длине | $O(n \cdot m)$, где $n$ — decoder, $m$ — encoder |

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] — attention внутри одной последовательности
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — общий механизм внимания
- [[02 Areas/ML & DL/Papers/Attention Is All You Need|Transformer]] — архитектура, где cross-attention используется в decoder
- [[02 Areas/ML & DL/Papers/T5|T5]] — encoder-decoder модель с cross-attention
