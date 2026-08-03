---
title: Собираем Transformer — от блока к BERT, GPT и LLaMA
type: textbook-chapter
status: canonical
last_updated: 2026-07-18
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/2302.13971
---

# Собираем Transformer: от блока к BERT, GPT и LLaMA

Механизм внимания выполняет только одну часть работы: переносит информацию между
позициями. После такого обмена каждой позиции всё ещё нужно преобразовать
полученные признаки, а глубокой сети — провести сигнал через десятки слоёв без
разрушения исходного представления. Полный блок Transformer поэтому объединяет
внимание, позиционную полносвязную сеть, остаточные связи и нормализацию.

Порядок этих компонентов исторически менялся. Оригинальный Transformer был
энкодер-декодером с нормализацией после остаточного сложения. BERT оставил
энкодер, GPT — причинный декодер без cross-attention, а LLaMA изменила
нормализацию, позиционный механизм и FFN. Поэтому архитектура 2017 года служит
исходной точкой, а BERT, GPT и LLaMA определяются через конкретные изменения её
блоков и связей.

## 1. Главная декомпозиция

На каждом слое происходят два разных действия:

1. **Attention:** слово получает информацию от других слов.
2. **FFN:** небольшая нейросеть обрабатывает получившийся вектор.

Коротко:

> Attention переносит информацию между токенами; FFN преобразует признаки
> каждого токена по отдельности.

Если оставить только FFN, токены не смогут учитывать контекст. Если оставить
только attention, модели не хватит нелинейного преобразования собранной
информации.

## 2. Остаточный поток

Удобнее всего понимать Transformer не как башню из блоков, а как поток
$x\in\mathbb{R}^{B\times T\times d}$, к которому подслои добавляют свои изменения:

$$
x\leftarrow x+\operatorname{Attention}(\operatorname{Norm}(x)),
$$

$$
x\leftarrow x+\operatorname{FFN}(\operatorname{Norm}(x)).
$$

Это современная запись **pre-norm**. Остаточный путь сохраняет общий интерфейс
между слоями: каждый подслой принимает и возвращает `B × T × d_model`.

Оригинальный Transformer использовал **post-norm**:

$$
x\leftarrow\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
$$

Нельзя рисовать pre-norm-блок LLaMA и подписывать его «оригинальный
Transformer»: порядок операций влияет на обучение и является архитектурным
различием.

## 3. Feed-Forward Network

В оригинале:

$$
\operatorname{FFN}(x)=
\max(0,xW_1+b_1)W_2+b_2.
$$

Одна и та же FFN применяется к каждой позиции независимо. Для Transformer Base:

- $d_{\text{model}}=512$;
- hidden width $d_{\text{ff}}=2048$;
- activation ReLU.

FFN расширяет вектор, применяет нелинейность и сжимает обратно. В современных
LLM этот подслой часто содержит большую долю параметров. LLaMA заменяет обычный
ReLU-MLP на gated SwiGLU:

$$
\operatorname{SwiGLU}(x)=
\big(\operatorname{SiLU}(xW_g)\odot xW_u\big)W_d.
$$

## 4. Информация о позиции

Self-attention без позиционной информации не знает, какой токен первый:
перестановка строк входного тензора так же переставит строки результата. Поэтому модели нужен
позиционный сигнал.

Оригинальный Transformer прибавлял sinusoidal encoding:

$$
PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d}\right),
$$

$$
PE_{(pos,2i+1)}=\cos\left(pos/10000^{2i/d}\right).
$$

Но это не универсальное свойство Transformer:

- BERT и ранние GPT используют обучаемые абсолютные позиционные представления;
- LLaMA применяет RoPE к Q и K внутри каждого слоя;
- другие семейства используют relative bias, ALiBi и их варианты.

## 5. Оригинальный encoder block

Энкодер получает входные и позиционные представления. Каждый из $N$ блоков содержит:

1. полное multi-head self-attention;
2. residual + LayerNorm;
3. position-wise FFN;
4. residual + LayerNorm.

Полное внимание означает: каждая непустая входная позиция может читать каждую
другую. Энкодер возвращает контекстные представления всего входа.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/Transformer_encoder.png]]

*Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/):
энкодер как self-attention и FFN с остаточным путём. Эта схема позволяет сначала
проследить один блок, прежде чем переходить к полному стеку.*

## 6. Оригинальный decoder block

Decoder содержит три подслоя:

1. masked self-attention по уже известному префиксу целевой последовательности;
2. cross-attention к выходу энкодера;
3. FFN.

В cross-attention:

$$
Q=\mathrm{decoder\ states},\qquad K,V=\mathrm{encoder\ outputs}.
$$

Это развитие идеи Bahdanau: декодер читает память исходной последовательности. Отличаются функция совместимости,
организация голов внимания и отсутствие рекуррентного состояния декодера.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/The_transformer_encoder_decoder_stack.png]]

*Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/):
развёрнутый стек энкодера и декодера. Выходы энкодера становятся K и V для
cross-attention каждого блока декодера.*

## 7. Обучение и генерация

Во время обучения целевая последовательность сдвигается:

```text
вход декодера:  <BOS> Чёрный кот спит
метки:                 Чёрный кот спит <EOS>
```

Causal mask гарантирует, что позиция $t$ не использует метки справа.
Все позиции при этом можно обработать одной тензорной операцией.

Во время генерации следующий токен неизвестен. На каждом шаге модель:

1. вычисляет логиты следующего токена;
2. выбирает токен или сэмплирует его из распределения;
3. добавляет токен к префиксу;
4. переходит к следующему шагу.

KV-cache сохраняет K/V прошлых позиций и не вычисляет их заново. Но зависимость
токена $t+1$ от выбранного токена $t$ остаётся, поэтому авторегрессионная генерация
последовательна по времени.

## 8. Три архитектурные ветви

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt-2-transformer-xl-bert-3.png]]

*Jay Alammar, «The Illustrated GPT-2»: наглядное сравнение decoder-only GPT-2,
encoder-only BERT и рекуррентного расширения Transformer-XL.
[Оригинальная статья](https://jalammar.github.io/illustrated-gpt2/).*

От GPT-2 к LLaMA меняется не общий силуэт декодера, а устройство его частей:
абсолютные позиционные эмбеддинги уступают место RoPE, LayerNorm — RMSNorm, а
обычная двухслойная FFN — вентильной SwiGLU. Поэтому полезнее сопоставить эти
механизмы по пунктам, чем рисовать ещё один почти одинаковый набор блоков.

### Encoder-only: BERT

BERT оставляет стек энкодера:

- полное двунаправленное self-attention;
- представления токенов, сегментов и обучаемые позиционные представления;
- masked language modeling;
- в оригинале также next sentence prediction.

При MLM выбирается 15% WordPieces; из них 80% заменяются на `[MASK]`, 10% — на
случайный токен, 10% остаются неизменными. Функция потерь считается по выбранным позициям.
BERT не является авторегрессионным генератором: задача предобучения учит его
восстанавливать скрытые части, а не продолжать префикс слева направо.

### Decoder-only: GPT

GPT использует причинный стек без энкодера и без cross-attention:

- masked self-attention;
- FFN;
- обучаемые позиционные представления в ранних GPT;
- предсказание следующего токена.

GPT-1 не «изобрёл decoder-only Transformer», а показал, что генеративное
предобучение переносится на прикладные задачи NLP. GPT-2 затем масштабировал модель и данные
и сделал акцент на решении задач без дополнительных обучающих примеров.

### Modern decoder-only: LLaMA

LLaMA сохраняет causal decoder, но меняет детали:

- pre-normalization с RMSNorm;
- RoPE вместо абсолютных позиционных представлений;
- SwiGLU вместо ReLU FFN;
- отсутствие смещений в ряде линейных слоёв;
- grouped-query attention в поздних версиях Llama.

Важно: LLaMA 1 использовала multi-head attention. Нельзя задним числом приписать
GQA всему семейству или самой первой версии.

## 9. Архитектурный diff

| Компонент | Transformer 2017 | BERT Base | GPT-1 | LLaMA 1 |
|---|---|---|---|---|
| архитектура | encoder-decoder | encoder | causal decoder | causal decoder |
| norm layout | Post-LN | Post-LN | Post-LN-like | Pre-RMSNorm |
| activation / FFN | ReLU | GELU | GELU | SwiGLU |
| позиции | sinusoidal | learned absolute | learned absolute | RoPE |
| cross-attention | decoder only | нет | нет | нет |
| задача обучения | translation | MLM + NSP | next token + fine-tune | next token |

Эта таблица иллюстрирует правильный способ описывать новые модели: не повторять
весь Transformer, а фиксировать, что сохранено и что изменено.

## 10. Почему Transformer вытеснил рекуррентную основу

В сравнении оригинальной статьи:

| Layer | Complexity per layer | Sequential operations | Maximum path length |
|---|---:|---:|---:|
| self-attention | $O(T^2d)$ | $O(1)$ | $O(1)$ |
| recurrent | $O(Td^2)$ | $O(T)$ | $O(T)$ |
| convolution | $O(kTd^2)$ | $O(1)$ | $O(\log_k T)$ |

Ключевой выигрыш — параллельные вычисления при обучении и короткий путь между
любыми позициями. Цена — квадратичная матрица взаимодействий. При очень длинном
контексте $T^2$ становится главным ограничением и порождает отдельную линию:
FlashAttention, sparse/sliding-window attention, linear attention и SSM.

> [!warning] Таблица зависит от режима
> $O(1)$ последовательных операций относится к обработке уже известной последовательности в
> слое, а не к генерации неизвестных будущих токенов. И self-attention не всегда
> дешевле рекуррентной сети: соотношение зависит от $T$, $d$, оборудования и реализации.

## 11. Минимальный современный causal block

```python
class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

Это каркас блока, близкого к LLaMA, а не оригинальный Transformer. Для точного декодера 2017 года
нужно добавить энкодер, cross-attention, LayerNorm после остаточного сложения, ReLU FFN и
синусоидальные позиции.

## 12. Как читать реальный код

Три уровня сложности:

1. [Karpathy `ng-video-lecture`](https://github.com/karpathy/ng-video-lecture) —
   механизм виден целиком и помещается в голове.
2. [Karpathy `build-nanogpt`](https://github.com/karpathy/build-nanogpt) —
   точное воспроизведение GPT-2 124M, конвейер данных, инициализация и обучение.
3. [Meta Llama 3 reference code](https://github.com/meta-llama/llama3/blob/main/llama/model.py) —
   RoPE, RMSNorm, SwiGLU и GQA в современной реализации.

`nanoGPT` полезен как компактная историческая реализация, но сам автор теперь
направляет к более современному `nanochat`.

## 13. Проверка форм тензоров

Для causal LLM:

```text
token ids                 [B, T]
token embeddings          [B, T, d]
Q, K, V                   [B, h, T, dh]
attention scores          [B, h, T, T]
attention result          [B, T, d]
residual after attention  [B, T, d]
FFN hidden                [B, T, dff]
residual after FFN        [B, T, d]
vocabulary logits         [B, T, |Vocab|]
```

Если форма неожиданно меняется вдоль остаточного потока, это почти всегда означает
пропущенную проекцию или ошибочную конкатенацию голов.

## Краткие итоги

- Оригинальный Transformer — encoder-decoder, а не GPT-подобная башня.
- Блок энкодера состоит из полного self-attention и FFN.
- Блок оригинального декодера содержит causal self-attention, cross-attention и FFN.
- Attention переносит информацию между позициями; FFN нелинейно преобразует каждую позицию отдельно.
- Остаточный поток сохраняет форму `B × T × d_model`.
- BERT, GPT и LLaMA различаются не только маской, но и задачей обучения, позиционным механизмом,
  нормализацией и FFN.
- Схему современной LLM нельзя выдавать за точную схему статьи 2017 года.

## Источники и интерактивы

### Первичные

- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Radford et al. — Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Devlin et al. — BERT](https://arxiv.org/abs/1810.04805)
- [Touvron et al. — LLaMA](https://arxiv.org/abs/2302.13971)

### Код и объяснения

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/13_transformers|TinyTorch 13 — Transformers]] — исполняемая сборка `TransformerBlock` из attention, MLP и layer normalization с авторегрессионной генерацией.
- [Harvard — The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy — Let’s build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy — Let’s reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder|Encoder–Decoder]]

**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]
