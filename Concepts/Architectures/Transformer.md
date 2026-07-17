---
title: "Transformer"
aliases: []
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]]"
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 4.0]]"
  - "[[02 Areas/ML & DL/Papers/T5]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2]]"
courses: []
sources:
  - "[Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)"
  - "[Lena Voita — Seq2Seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)"
  - "[Lilian Weng — The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)"
  - "[d2l.ai — The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)"
---

# Transformer

## Зачем это нужно: почему мир отказался от RNN

До 2017 года sequence-to-sequence задачи (перевод, суммаризация, генерация) решались через RNN/LSTM с attention. Проблемы были фундаментальные:

1. **Последовательная обработка**: скрытое состояние $h_t$ зависит от $h_{t-1}$, что делает невозможной параллелизацию по длине последовательности. Обучение на длинных текстах занимало недели.
2. **Затухание градиентов**: несмотря на гейты LSTM, информация между позицией 1 и позицией 100 проходит через 99 шагов — на каждом может потеряться.
3. **Бутылочное горлышко**: в базовом seq2seq энкодер сжимает всё в один вектор. Bahdanau attention (2014) смягчил проблему, но RNN-backbone остался.

**Vaswani et al. (2017)** предложили радикальное решение: выбросить рекуррентность полностью, оставив **только** механизм внимания. Результат — модель под названием **Transformer**, обученная за 3.5 дня на 8 GPU P100, побила SOTA по машинному переводу с BLEU 28.4 (EN-DE) и 41.8 (EN-FR).

Как пишут авторы: *"The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."*

## Архитектура: общий взгляд

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/The_transformer_encoder_decoder_stack.png]]
*Стек из N энкодеров и N декодеров. Каждый энкодер содержит self-attention + FFN, каждый декодер — masked self-attention + cross-attention + FFN (источник: Jay Alammar)*

Transformer следует классической encoder-decoder схеме, но вместо рекуррентных слоёв использует **стеки идентичных блоков**:

- **Encoder stack**: N = 6 одинаковых слоёв. Каждый слой — два подслоя: (1) multi-head self-attention, (2) position-wise FFN.
- **Decoder stack**: N = 6 одинаковых слоёв. Каждый слой — три подслоя: (1) masked multi-head self-attention, (2) multi-head cross-attention к выходу энкодера, (3) position-wise FFN.

Каждый подслой обёрнут в **residual connection + layer normalization**:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Все подслои и эмбеддинги имеют размерность $d_{\text{model}} = 512$ (для базовой модели).

## Входные представления: эмбеддинги + позиционное кодирование

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_positional_encoding_vectors.png]]
*Позиционные кодировки добавляются к токенным эмбеддингам. Синусоидальный паттерн позволяет модели обучиться относительным позициям (источник: Jay Alammar)*

Self-attention **инвариантен к порядку токенов**: если поменять местами слова в предложении, без дополнительной информации модель даст тот же результат. Поэтому необходимо **явно кодировать позицию**.

Оригинальный Transformer использует **синусоидальные позиционные кодировки**:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

**Зачем именно синусоиды?** Для любого фиксированного смещения $k$, $PE_{pos+k}$ можно представить как линейную функцию от $PE_{pos}$ — это позволяет модели легко обучиться относительным позициям. Кроме того, синусоидальные кодировки потенциально позволяют экстраполировать на длины, не встречавшиеся при обучении.

Авторы также тестировали **обучаемые** позиционные эмбеддинги — результаты почти идентичные (Table 3, row E). Однако последующие модели разделились:
- **BERT, GPT-2**: обучаемые позиционные эмбеддинги
- **LLaMA, Mistral**: RoPE (Rotary Positional Embeddings) — вращение в комплексном пространстве

Итоговый вход: $\text{input} = \text{TokenEmbedding}(x) + PE(pos)$, причём эмбеддинги умножаются на $\sqrt{d_{\text{model}}}$ для масштабирования.

## Self-Attention: пошаговый разбор

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/encoder_with_tensors_2.png]]
*Внутри блока энкодера: каждый токен проходит через self-attention (с доступом ко всем остальным токенам) и FFN (независимо) (источник: Jay Alammar)*

Self-attention — ключевая инновация Transformer. Каждый токен «задаёт вопрос» (Query) всем остальным токенам, которые «отвечают» через Key (насколько релевантен) и Value (какую информацию отдать).

### Шаг за шагом (по Jay Alammar)

**Шаг 1.** Каждый входной эмбеддинг (512-мерный) умножается на три обученные матрицы $W^Q$, $W^K$, $W^V$, создавая три 64-мерных вектора Query, Key, Value.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png]]
*Создание Q, K, V векторов: умножение эмбеддинга на три матрицы весов (источник: Jay Alammar)*

**Шаг 2.** Score = dot-product Query текущего токена с Key каждого другого токена — показывает «насколько сильно обращать внимание».

**Шаг 3.** Делим на $\sqrt{d_k} = \sqrt{64} = 8$ для стабилизации градиентов. Без этого при больших $d_k$ dot-product становится слишком большим, softmax уходит в насыщение, градиенты исчезают. Математически: если компоненты $q$ и $k$ — независимые случайные величины со средним 0 и дисперсией 1, то $q \cdot k$ имеет дисперсию $d_k$.

**Шаг 4.** Softmax нормализует скоры в вероятности (сумма = 1).

**Шаг 5.** Умножаем каждый Value на его softmax-вес.

**Шаг 6.** Суммируем — получаем выход self-attention для этой позиции.

В матричной форме:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png]]
*Матричная форма self-attention: одно выражение вместо шести шагов (источник: Jay Alammar)*

### Каноничный пример

*"The animal didn't cross the street because **it** was too tired."*

К чему относится «it»? Для человека очевидно — к «animal». Self-attention назначает высокие веса позициям «The» и «animal» при обработке «it».

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_self-attention_visualization.png]]
*Self-attention связывает «it» с «The animal» — модель обучается кореференции без прямого supervision (источник: Jay Alammar)*

## Multi-Head Attention: несколько точек зрения

Одна голова attention усредняет информацию и может пропустить нюансы. Решение — запустить **h параллельных голов**, каждая со своими весами:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

где $\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$

В оригинале: $h = 8$ голов, $d_k = d_v = d_{\text{model}} / h = 64$. Итоговая стоимость вычислений примерно та же, что у одной полноразмерной головы.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_multi-headed_self-attention-recap.png]]
*Multi-Head Attention: 8 голов работают параллельно, выходы конкатенируются и проецируются через $W^O$ (источник: Jay Alammar)*

**Что изучают разные головы?** Визуализации из Appendix оригинальной статьи (Figure 3-5) показывают специализацию:
- Одна голова отслеживает **анафору** (связь местоимений с антецедентами)
- Другая — **синтаксические зависимости** на дальних дистанциях
- Третья — **позиционные паттерны** (соседние токены)

## Три вида attention в Transformer

Архитектура использует attention тремя принципиально разными способами:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/the_transformer_3.png]]
*Полная архитектура: энкодер (слева) с bidirectional attention, декодер (справа) с masked attention + cross-attention (источник: Jay Alammar)*

| Вид | Q, K, V | Маска | Назначение |
|-----|---------|-------|-----------|
| **Encoder self-attention** | Все из предыдущего слоя энкодера | Нет (bidirectional) | Каждый токен видит все другие |
| **Decoder masked self-attention** | Все из предыдущего слоя декодера | Causal ($-\infty$ для будущих позиций) | Авторегрессивность — нет «подглядывания» |
| **Cross-attention** | Q из декодера, K и V из выхода энкодера | Нет | Декодер «смотрит» на входную последовательность |

Cross-attention — это по сути тот же Bahdanau attention, но с scaled dot-product вместо additive scoring и multi-head вместо одной головы.

## Feed-Forward Network (FFN)

Каждый слой содержит position-wise FFN — двухслойный MLP, применяемый к каждой позиции **независимо**:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Размерности: вход/выход = $d_{\text{model}} = 512$, скрытый слой = $d_{ff} = 2048$ (расширение 4x). Активация — ReLU.

**Зачем FFN после attention?** Self-attention хорошо агрегирует информацию из разных позиций, но плохо «обрабатывает» эту информацию. FFN играет роль нелинейного преобразования на каждой позиции — фактически это «thinking» слой модели.

## Residual Connections + Layer Normalization

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_resideual_layer_norm_2.png]]
*Каждый подслой обёрнут в residual connection и LayerNorm (источник: Jay Alammar)*

**Residual connections** (He et al., 2016) решают проблему затухающих градиентов в глубоких сетях: градиент может «пролететь» напрямую через skip-connection, минуя подслой.

**Layer Normalization** (Ba et al., 2016) нормализует активации по feature dimension, стабилизируя обучение. Оригинальный Transformer использует **Post-LN** (нормализация после residual addition). Последующие модели (GPT-2, GPT-3, LLaMA) переключились на **Pre-LN** (нормализация перед подслоем) — это стабильнее при большом масштабе.

## Декодирование: как генерируется выход

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoding_2.gif]]
*Авторегрессивное декодирование: каждый новый токен генерируется с attention ко всем предыдущим (источник: Jay Alammar)*

Финальный linear layer проецирует выход декодера в вектор размерности словаря, softmax превращает его в вероятности:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoder_output_softmax.png]]
*Linear + Softmax: проекция в пространство словаря и вычисление вероятностей (источник: Jay Alammar)*

Важная деталь: в оригинальном Transformer **weight tying** — одна и та же матрица весов используется для входных эмбеддингов, выходных эмбеддингов и pre-softmax линейного слоя (домноженная на $\sqrt{d_{\text{model}}}$ в эмбеддингах).

## Обучение: детали из оригинальной статьи

### Данные
- **WMT 2014 EN-DE**: ~4.5M пар предложений, BPE с общим словарём ~37K токенов
- **WMT 2014 EN-FR**: ~36M пар предложений, word-piece словарь 32K

### Hardware и время
- 8 NVIDIA P100 GPUs на одной машине
- Base model: 100K шагов = **12 часов** (0.4 сек/шаг)
- Big model: 300K шагов = **3.5 дня** (1.0 сек/шаг)

### Оптимизатор
Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$) с warmup learning rate schedule:

$$lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

Линейный рост LR первые 4000 шагов, затем затухание пропорционально $1/\sqrt{\text{step}}$.

### Регуляризация
- **Residual Dropout** $P_{drop} = 0.1$ — на выходе каждого подслоя и на сумме эмбеддингов с позиционными кодировками
- **Label Smoothing** $\epsilon_{ls} = 0.1$ — ухудшает perplexity, но улучшает BLEU

## Результаты: WMT Machine Translation

| Модель | EN-DE BLEU | EN-FR BLEU | Training FLOPs (EN-DE) |
|--------|-----------|-----------|----------------------|
| GNMT+RL Ensemble | 26.30 | 41.16 | $1.8 \times 10^{20}$ |
| ConvS2S Ensemble | 26.36 | 41.29 | $7.7 \times 10^{19}$ |
| **Transformer (base)** | **27.3** | 38.1 | $3.3 \times 10^{18}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \times 10^{19}$ |

Transformer (big) побил все предыдущие ансамбли на EN-DE (+2 BLEU) и установил SOTA на EN-FR — при **стоимости обучения на порядок меньше**.

## Model Variations (Table 3 из статьи)

Ablation studies на EN-DE development set:

| Конфигурация | Параметры | BLEU (dev) |
|-------------|----------|-----------|
| Base (h=8, $d_k$=64) | 65M | 25.8 |
| h=1, $d_k$=512 | 65M | 24.9 (одна голова хуже) |
| h=32, $d_k$=16 | 65M | 25.4 (слишком много голов тоже хуже) |
| N=2 layers | 36M | 23.7 |
| N=8 layers | 80M | 25.5 |
| $d_{\text{model}}$=1024, $d_{ff}$=4096 | 168M | 26.0 |
| Big (N=6, $d_{\text{model}}$=1024, h=16) | 213M | 26.4 |

**Ключевые выводы**: 8 голов — оптимум (одна голова -0.9 BLEU); масштабирование помогает; dropout критически важен.

## Почему self-attention побеждает RNN и CNN

Table 1 из статьи — главный теоретический аргумент:

| Слой | Сложность / слой | Последоват. операции | Макс. длина пути |
|------|-----------------|---------------------|-----------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | **$O(1)$** |
| Recurrent (RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

**Три killer features:**

1. **$O(1)$ максимальная длина пути** — любые две позиции связаны напрямую через один шаг (в RNN — через $n$ шагов).
2. **$O(1)$ последовательных операций** — полная параллелизация (в RNN $h_t$ зависит от $h_{t-1}$).
3. **Интерпретируемость** — attention weights показывают, на что модель «смотрит».

**Цена**: $O(n^2)$ по длине последовательности. При $n > 10000$ это проблема, мотивировавшая [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]], [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] и другие эффективные альтернативы.

## Три архитектурных варианта: потомки Transformer

Оригинальный Transformer — encoder-decoder для seq2seq. Но его наследники разделились на три ветви:

| Вариант | Attention | Pre-training | Примеры |
|---------|-----------|-------------|---------|
| **[[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder\|Encoder-Decoder]]** | Encoder: bidirectional; Decoder: causal + cross-attn | Seq2seq (span corruption) | [[02 Areas/ML & DL/Concepts/Architectures/T5\|T5]], BART, оригинальный Transformer |
| **[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only\|Encoder-only]]** | Bidirectional | Masked LM (MLM) | [[02 Areas/ML & DL/Concepts/Architectures/BERT\|BERT]], RoBERTa, DeBERTa |
| **[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only\|Decoder-only]]** | Causal (masked) | Autoregressive LM | [[02 Areas/ML & DL/Concepts/Architectures/GPT-3\|GPT-3]], [[02 Areas/ML & DL/Concepts/Architectures/LLaMA\|LLaMA]], Mistral |

**Encoder-only** (BERT) — лучше для NLU задач (classification, NER, QA). Видит весь контекст bidirectionally.

**Decoder-only** (GPT) — лучше для генерации. Масштабируется до сотен миллиардов параметров. Стал доминирующей парадигмой к 2023 году.

**Encoder-decoder** (T5) — хорошо для seq2seq задач (перевод, суммаризация). Унифицирует все задачи в text-to-text формат.

## Хронология: от Transformer к современным LLM

| Год | Milestone | Модель |
|-----|-----------|--------|
| 2017 | Оригинальный Transformer | Vaswani et al. |
| 2018 | Decoder-only LM (117M) | GPT-1 (Radford et al.) |
| 2018 | Encoder-only bidirectional | [[02 Areas/ML & DL/Concepts/Architectures/BERT\|BERT]] (Devlin et al.) |
| 2019 | Decoder-only scale-up (1.5B) | GPT-2 (Radford et al.) |
| 2019 | Encoder-decoder text-to-text (11B) | [[02 Areas/ML & DL/Concepts/Architectures/T5\|T5]] (Raffel et al.) |
| 2020 | In-context learning emergence (175B) | [[02 Areas/ML & DL/Concepts/Architectures/GPT-3\|GPT-3]] (Brown et al.) |
| 2022 | Chinchilla-optimal scaling | Chinchilla (Hoffmann et al.) |
| 2023 | Open-weight revolution | [[02 Areas/ML & DL/Concepts/Architectures/LLaMA\|LLaMA]] (Touvron et al.) |
| 2023 | Sliding window attention | [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B\|Mistral 7B]] |

## Ключевые числа

| Модель | Layers | $d_{\text{model}}$ | Heads | FFN | Параметры |
|--------|--------|---------------------|-------|-----|----------|
| Transformer (base) | 6 | 512 | 8 | 2048 | 65M |
| Transformer (big) | 6 | 1024 | 16 | 4096 | 213M |
| BERT_BASE | 12 | 768 | 12 | 3072 | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 4096 | 340M |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |
| LLaMA 65B | 80 | 8192 | 64 | ~22K | 65B |

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — фундамент архитектуры
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — без него attention инвариантен к порядку
- [[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]] — стабилизация обучения
- [[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]] — position-wise MLP
- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]] — BPE / WordPiece / SentencePiece
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]] — BERT-семейство
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — GPT-семейство
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]] — T5, BART
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — IO-aware ускорение attention
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — кэширование при инференсе

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — лучшие визуализации архитектуры
- [Lena Voita — Seq2Seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — исторический контекст
- [Lilian Weng — The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — обзор всех вариантов
- [d2l.ai — The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) — учебник с кодом
- [Harvard NLP — The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — построчная реализация на PyTorch
