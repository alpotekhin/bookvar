---
title: "Attention Mechanism"
aliases: [Attention, Self-Attention, Multi-Head Attention, MHA, MHSA, Scaled Dot-Product Attention]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]]"
  - "[[02 Areas/ML & DL/Papers/Flash Attention|Flash Attention]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/07 — Attention|CS224N Lecture 7]]"
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/08 — Transformers|CS224N Lecture 8]]"
sources:
  - "[Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)"
  - "[Lena Voita — Seq2Seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)"
  - "[Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)"
  - "[d2l.ai — Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/)"
---

# Attention Mechanism

## Зачем это нужно: проблема бутылочного горлышка

До 2014 года seq2seq модели (Sutskever et al.) работали так: энкодер-RNN читает входную последовательность и сжимает её в **один вектор фиксированной длины** (обычно 512 или 1024 размерности). Этот вектор передаётся декодеру, который из него генерирует выход.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bottleneck-min.png]]
*Проблема бутылочного горлышка: вся информация о входе сжимается в один вектор (источник: Lena Voita)*

Представь, что тебе нужно перевести абзац из 50 слов, но запомнить его можно только в виде одного предложения. На коротких фразах это работает, на длинных — катастрофически ломается. Как пишет Voita: *"the whole universe is compressed into a single vector of size 512"*.

Две конкретные проблемы:
- **Для энкодера:** невозможно сжать произвольно длинную последовательность в вектор фиксированной размерности без потерь.
- **Для декодера:** на каждом шаге генерации нужна *разная* информация из входа, но он получает один и тот же статичный вектор.

## Идея: пусть декодер *смотрит назад* на вход

**Bahdanau et al. (2014)** предложили решение: вместо одного вектора дать декодеру доступ ко **всем** скрытым состояниям энкодера. На каждом шаге генерации механизм внимания решает, какие части входа сейчас важнее.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bahdanau_examples-min.png]]
*Матрица выравнивания Bahdanau attention: яркие клетки показывают, на какие входные слова «смотрит» декодер при генерации каждого выходного слова. Модель автоматически учится выравнивать слова между языками без прямого supervision (источник: Lena Voita)*

Формально, контекстный вектор на шаге $t$:

$$c_t = \sum_{i} \alpha_{t,i} \cdot h_i$$

где $h_i$ — скрытое состояние энкодера для позиции $i$, а $\alpha_{t,i}$ — вес внимания:

$$\alpha_{t,i} = \text{softmax}(\text{score}(s_t, h_i))$$

Ключевой момент: **всё дифференцируемо**. Модель учится сама выбирать, куда смотреть — не нужно вручную задавать выравнивание между входом и выходом.

## Функции скоринга: как вычисляется «похожесть»

Разные авторы предложили разные способы вычислять score — насколько текущее состояние декодера $s_t$ «совместимо» с состоянием энкодера $h_i$:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/score_functions-min.png]]
*Сравнение функций скоринга (источник: Lena Voita)*

| Название | Формула | Откуда |
|----------|---------|--------|
| Dot-product | $s_t^T h_i$ | Luong, 2015 |
| Bilinear / General | $s_t^T W_a h_i$ | Luong, 2015 |
| Additive / MLP | $v_a^T \tanh(W_a[s_t; h_i])$ | Bahdanau, 2014 |
| **Scaled dot-product** | $(s_t^T h_i) / \sqrt{d_k}$ | **Vaswani, 2017** |
| Cosine | $\cos(s_t, h_i)$ | Graves, 2014 |

**Bahdanau (additive)** — MLP с одним скрытым слоем. Работает хорошо, но медленнее. Bidirectional RNN энкодер.

**Luong (bilinear)** — матричное умножение, быстрее. Unidirectional RNN энкодер. Предложил также **local attention** (окно вместо всей последовательности) как компромисс между скоростью и качеством.

**Scaled dot-product (Vaswani)** — стал стандартом. Деление на $\sqrt{d_k}$ критически важно: при больших размерностях dot-product растёт, softmax уходит в насыщение, градиенты становятся крошечными. Масштабирование решает проблему (если компоненты $q$ и $k$ независимы с нулевым средним и единичной дисперсией, то $q \cdot k$ имеет дисперсию $d_k$ — доказательство из оригинальной статьи).

## Self-Attention: внимание к самому себе

До 2017 attention использовался **между** энкодером и декодером. Революция Vaswani et al. — применить внимание **внутри** одной последовательности.

Каждый токен «задаёт вопрос» (Query) всем остальным токенам, «отвечающим» через Key и Value. Интуитивно (аналогия от Transformer Explainer, Georgia Tech): Query — это поисковый запрос, Key — заголовки страниц, Value — содержимое страниц.

### Пошаговый разбор (по Jay Alammar)

**Шаг 1.** Каждый входной эмбеддинг (512-мерный) умножается на три обученные матрицы $W^Q$, $W^K$, $W^V$, создавая три 64-мерных вектора: Query, Key, Value.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png]]
*Создание Q, K, V векторов для каждого токена (источник: Jay Alammar)*

**Шаг 2.** Score = dot-product Query текущего токена с Key каждого другого токена. Это показывает, насколько текущий токен должен «обращать внимание» на каждую другую позицию.

**Шаг 3.** Делим на $\sqrt{d_k} = \sqrt{64} = 8$ (стабилизация градиентов).

**Шаг 4.** Softmax нормализует скоры в вероятности (сумма = 1).

**Шаг 5.** Умножаем каждый Value на его softmax-вес.

**Шаг 6.** Суммируем — получаем выход self-attention для этой позиции.

В матричной форме — это одно выражение:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png]]
*Матричное вычисление self-attention (источник: Jay Alammar)*

### Каноничный пример

Предложение: *"The animal didn't cross the street because **it** was too tired."*

К чему относится «it»? Для человека очевидно — к «animal». Self-attention позволяет модели установить эту связь: при обработке «it» механизм внимания назначает высокие веса позициям «The» и «animal».

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_self-attention_visualization.png]]
*Визуализация self-attention: модель связывает «it» с «The animal» (источник: Jay Alammar)*

## Multi-Head Attention: несколько точек зрения одновременно

Одна голова внимания усредняет информацию и может упустить нюансы. Решение — **несколько параллельных голов**, каждая со своими $W^Q_i$, $W^K_i$, $W^V_i$:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

где $\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$

В оригинальном Transformer: $h = 8$ голов, каждая работает в 64-мерном подпространстве (512 / 8 = 64). Итоговая стоимость примерно та же, что у одной полноразмерной головы.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_multi-headed_self-attention-recap.png]]
*Multi-Head Attention: 8 голов, конкатенация, проекция через $W^O$ (источник: Jay Alammar)*

**Зачем это нужно на практике?** Визуализации из оригинальной статьи (Appendix, Figure 3-5) показывают, что разные головы специализируются на разных задачах:
- Одна голова отслеживает **анафору** (связь местоимений с антецедентами: «its» -> «Law»)
- Другая — **дальние синтаксические зависимости** (связь «making» с «more difficult» через много токенов)
- Третья — **позиционные паттерны** (соседние токены)

## Три вида attention в Transformer

В полной архитектуре Transformer attention используется тремя способами:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/the_transformer_3.png]]
*Общая архитектура Transformer: N энкодерных + N декодерных блоков (источник: Jay Alammar)*

1. **Encoder self-attention** — каждый токен видит все остальные (bidirectional). Q, K, V — все из предыдущего слоя энкодера. Mask = 1.

2. **Decoder masked self-attention** — токен видит только предшествующие позиции. Будущие позиции маскируются ($-\infty$ перед softmax). Это обеспечивает авторегрессивность: модель не «подглядывает» в будущее при генерации.

3. **Cross-attention (encoder-decoder)** — Query из декодера, Key и Value из выхода энкодера. Позволяет декодеру «смотреть» на входную последовательность (как классический Bahdanau attention, но с scaled dot-product).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoding_2.gif]]
*Авторегрессивное декодирование: каждый новый токен генерируется с attention ко всем предыдущим (источник: Jay Alammar)*

## Residual + Layer Norm

Каждый подслой (self-attention и FFN) обёрнут в residual connection + layer normalization:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_resideual_layer_norm_2.png]]
*Residual connection + LayerNorm внутри каждого блока (источник: Jay Alammar)*

Residual connections решают проблему затухающих градиентов при глубоких стеках (6+ слоёв). LayerNorm стабилизирует обучение.

## Почему self-attention побеждает RNN и CNN

Таблица 1 из оригинальной статьи — главный теоретический аргумент:

| Слой | Сложность/слой | Последоват. операции | Макс. длина пути |
|------|---------------|----------------------|-----------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | **$O(1)$** |
| Recurrent (RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

**$O(1)$ максимальная длина пути** — killer feature. В RNN сигнал между позицией 1 и позицией 100 должен пройти через 99 шагов (и на каждом может затухнуть). В self-attention любые две позиции связаны **напрямую** через один шаг.

**$O(1)$ последовательных операций** — self-attention полностью параллелизуется (в отличие от RNN, где $h_t$ зависит от $h_{t-1}$). Это позволило масштабировать обучение на кластерах GPU.

Цена — **$O(n^2)$ по длине последовательности** (матрица attention $n \times n$). При $n > 10000$ это становится проблемой, что мотивировало создание [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]], [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]], [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]] и других эффективных альтернатив.

## Attention как адресация памяти

Глубокий взгляд от Lilian Weng: attention можно понимать как **мягкую адресацию памяти**. Neural Turing Machine (Graves, 2014) использует аналогичный механизм для чтения из внешнего хранилища — content-based addressing через cosine similarity, location-based addressing через свёртки.

В этой перспективе:
- **Key** = адрес ячейки памяти
- **Value** = содержимое ячейки
- **Query** = запрос на чтение
- Attention weights = «какие ячейки прочитать и с каким весом»

Эта аналогия объясняет, почему attention работает: модель учится **структурированно обращаться к информации**, а не просто сжимать всё в один вектор.

## Виды attention (таксономия по Weng)

| Вид | Описание | Примечание |
|-----|----------|------------|
| **Soft / Global** | Веса по всей последовательности, дифференцируемо | Стандартный в Transformer |
| **Hard** | Выбирает одну позицию, не дифференцируемо | Требует REINFORCE |
| **Local** | Окно вокруг предсказанной позиции | Компромисс soft/hard (Luong, 2015) |
| **Self** | Внутри одной последовательности | Основа Transformer |
| **Cross** | Между двумя последовательностями | Encoder->Decoder |

## Хронология

| Год | Milestone | Статья |
|-----|-----------|--------|
| 2014 | Additive attention для NMT | Bahdanau et al. |
| 2014 | Neural Turing Machines (attention как память) | Graves et al. |
| 2015 | Multiplicative/local attention + image captioning | Luong; Xu et al. |
| 2015 | Pointer Networks (attention -> дискретный выбор) | Vinyals et al. |
| 2016 | Self-attention в LSTM | Cheng et al. |
| **2017** | **Transformer — Attention Is All You Need** | **Vaswani et al.** |
| 2018 | BERT, GPT — self-attention как основа LLM | Devlin; Radford |
| 2022 | Flash Attention — IO-aware efficient attention | Dao et al. |
| 2023 | Mamba, RWKV — linear alternatives to attention | Gu et al.; Peng et al. |

## Современные модификации

- **Grouped Query Attention (GQA)** — несколько query-голов разделяют одну KV-голову. Используется в [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral]] и LLaMA 2. Экономит память KV-cache при инференсе.
- **Multi-Query Attention (MQA)** — все query-головы разделяют одну KV-голову. Ещё быстрее, но может терять качество.
- **Sliding Window Attention (SWA)** — attention только к $w$ ближайшим токенам. [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] использует $w = 4096$, но за счёт стекирования слоёв теоретически покрывает ~131K токенов.
- **Flash Attention** — не меняет математику attention, но радикально ускоряет вычисления за счёт IO-aware тайлинга и kernel fusion. [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Подробнее]]

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — без него self-attention инвариантен к порядку токенов
- [[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]] — стабилизирует обучение в каждом блоке
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — полная архитектура, построенная на attention
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — эффективная реализация
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — кэширование Key/Value при инференсе
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — альтернатива без attention (SSM)

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — лучшие визуализации пошагового attention
- [Lena Voita — Seq2Seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — лучшее объяснение *зачем* нужен attention
- [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — самый полный обзор вариантов attention
- [Transformer Explainer (Georgia Tech)](https://poloclub.github.io/transformer-explainer/) — интерактивная визуализация в браузере
- [d2l.ai — Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/) — учебник с кодом
