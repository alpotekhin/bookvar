---
title: Собираем Transformer — от блока к BERT, GPT и LLaMA
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/2302.13971
---

# Собираем Transformer: от блока к BERT, GPT и LLaMA

> [!abstract] Идея главы
> Attention позволяет словам учитывать друг друга, но одного attention
> недостаточно. Между такими обменами каждый токен проходит через небольшую
> нейросеть, а исходный вектор сохраняется благодаря остаточной связи. BERT, GPT
> и LLaMA по-разному собирают и обучают эти блоки.

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

## 2. Residual stream

Удобнее всего понимать Transformer не как башню коробок, а как поток
$x\in\mathbb{R}^{B\times T\times d}$, к которому подслои добавляют updates:

$$
x\leftarrow x+\operatorname{Attention}(\operatorname{Norm}(x)),
$$

$$
x\leftarrow x+\operatorname{FFN}(\operatorname{Norm}(x)).
$$

Это современная **pre-norm** запись. Residual path сохраняет общий интерфейс
между слоями: каждый подслой принимает и возвращает `B × T × d_model`.

Оригинальный Transformer использовал **post-norm**:

$$
x\leftarrow\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
$$

Нельзя рисовать pre-norm LLaMA block и подписывать его «оригинальный
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

## 4. Порядок позиций

Self-attention без position information не знает, какой token первый:
перестановка входных rows переставит outputs тем же образом. Поэтому модели нужен
позиционный сигнал.

Оригинальный Transformer прибавлял sinusoidal encoding:

$$
PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d}\right),
$$

$$
PE_{(pos,2i+1)}=\cos\left(pos/10000^{2i/d}\right).
$$

Но это не универсальное свойство Transformer:

- BERT и ранние GPT используют learned absolute position embeddings;
- LLaMA применяет RoPE к Q и K внутри каждого слоя;
- другие семейства используют relative bias, ALiBi и их варианты.

## 5. Оригинальный encoder block

Encoder получает input embeddings + positions. Каждый из $N$ блоков содержит:

1. full multi-head self-attention;
2. residual + LayerNorm;
3. position-wise FFN;
4. residual + LayerNorm.

Full attention означает: каждая непустая input position может читать каждую
другую. Encoder выдаёт contextual representations всего входа.

![[02 Areas/ML & DL/raw/papers/attention-is-all-you-need/images/Transformer_encoder.png]]

*Jay Alammar, «The Illustrated Transformer»: encoder как self-attention и FFN с
residual path. Используем эту схему как промежуточную, а не начинаем сразу с
перегруженной Figure 1 оригинальной статьи.*

## 6. Оригинальный decoder block

Decoder содержит три подслоя:

1. masked self-attention по уже известному target prefix;
2. cross-attention к encoder output;
3. FFN.

В cross-attention:

$$
Q=\text{decoder states},\qquad K,V=\text{encoder outputs}.
$$

Это наследник идеи Bahdanau: decoder читает source memory. Отличаются scoring,
multi-head organization и отсутствие recurrent decoder state.

![[02 Areas/ML & DL/raw/papers/attention-is-all-you-need/images/The_transformer_encoder_decoder_stack.png]]

*Jay Alammar: развёрнутый encoder-decoder stack. На схеме особенно важно увидеть
единственную связь между башнями: encoder outputs становятся K и V для
cross-attention каждого decoder block.*

## 7. Training и inference

Во время training target сдвигается:

```text
decoder input:  <BOS> Чёрный кот спит
labels:          Чёрный кот    спит <EOS>
```

Causal mask гарантирует, что строка позиции $t$ не использует labels справа.
Все позиции можно обработать одним большим tensor operation.

Во время inference следующий token неизвестен. Цикл повторяется:

1. вычислить logits следующего token;
2. выбрать или sample token;
3. добавить его к prefix;
4. вычислить следующий step.

KV-cache сохраняет K/V прошлых positions и не вычисляет их заново. Но зависимость
`token t+1` от выбранного `token t` остаётся, поэтому autoregressive generation
последовательна по времени.

## 8. Три архитектурные ветви

![[02 Areas/ML & DL/raw/papers/gpt-30/images/gpt-2-transformer-xl-bert-3.png]]

*Jay Alammar, «The Illustrated GPT-2»: наглядное сравнение decoder-only GPT-2,
encoder-only BERT и recurrent extension Transformer-XL.
[Оригинальная статья](https://jalammar.github.io/illustrated-gpt2/).*

Ниже мы расширяем эту готовую картинку текстовым diff до LLaMA. Собственная
схема здесь не нужна: новое знание — не внешний вид прямоугольников, а точный
список изменившихся механизмов.

### Encoder-only: BERT

BERT оставляет encoder stack:

- full bidirectional self-attention;
- token + segment + learned position embeddings;
- masked language modeling;
- в оригинале также next sentence prediction.

При MLM выбирается 15% WordPieces; из них 80% заменяются на `[MASK]`, 10% — на
случайный token, 10% остаются неизменными. Loss считается по выбранным позициям.
BERT не является autoregressive generator: его pre-training objective учит
восстанавливать скрытые части, а не продолжать prefix слева направо.

### Decoder-only: GPT

GPT использует causal stack без encoder и без cross-attention:

- masked self-attention;
- FFN;
- learned position embeddings в ранних GPT;
- next-token objective.

GPT-1 не «изобрёл decoder-only Transformer», а показал перенос generative
pre-training на downstream NLP tasks. GPT-2 затем масштабировал модель и данные
и сделал акцент на zero-shot behavior.

### Modern decoder-only: LLaMA

LLaMA сохраняет causal decoder, но меняет детали:

- pre-normalization с RMSNorm;
- RoPE вместо absolute position embeddings;
- SwiGLU вместо ReLU FFN;
- no biases в ряде линейных слоёв;
- в поздних Llama variants — grouped-query attention.

Важно: LLaMA 1 использовала multi-head attention. Нельзя задним числом приписать
GQA всему семейству или самой первой версии.

## 9. Архитектурный diff

| Компонент | Transformer 2017 | BERT Base | GPT-1 | LLaMA 1 |
|---|---|---|---|---|
| stack | encoder-decoder | encoder | causal decoder | causal decoder |
| norm layout | Post-LN | Post-LN | Post-LN-like | Pre-RMSNorm |
| activation / FFN | ReLU | GELU | GELU | SwiGLU |
| positions | sinusoidal | learned absolute | learned absolute | RoPE |
| cross-attention | decoder only | нет | нет | нет |
| objective | translation | MLM + NSP | next token + fine-tune | next token |

Эта таблица иллюстрирует правильный способ описывать новые модели: не повторять
весь Transformer, а фиксировать, что сохранено и что изменено.

## 10. Почему Transformer вытеснил recurrent backbone

В сравнении оригинальной статьи:

| Layer | Complexity per layer | Sequential operations | Maximum path length |
|---|---:|---:|---:|
| self-attention | $O(T^2d)$ | $O(1)$ | $O(1)$ |
| recurrent | $O(Td^2)$ | $O(T)$ | $O(T)$ |
| convolution | $O(kTd^2)$ | $O(1)$ | $O(\log_k T)$ |

Ключевой выигрыш — параллельное training computation и короткий путь между
любыми positions. Цена — quadratic interaction matrix. При очень длинном
контексте $T^2$ становится главным ограничением и порождает отдельную линию:
FlashAttention, sparse/sliding-window attention, linear attention и SSM.

> [!warning] Таблица зависит от режима
> $O(1)$ sequential operations относится к обработке уже известной sequence в
> слое, а не к генерации неизвестных будущих tokens. И self-attention не всегда
> дешевле recurrence: соотношение зависит от $T$, $d$, hardware и реализации.

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

Это LLaMA-like skeleton, не оригинальный Transformer. Для точного 2017 decoder
нужно добавить encoder, cross-attention, LayerNorm post-residual, ReLU FFN и
sinusoidal positions.

## 12. Как читать реальный код

Три уровня сложности:

1. [Karpathy `ng-video-lecture`](https://github.com/karpathy/ng-video-lecture) —
   механизм виден целиком и помещается в голове.
2. [Karpathy `build-nanogpt`](https://github.com/karpathy/build-nanogpt) —
   точное воспроизведение GPT-2 124M, data pipeline, initialization и training.
3. [Meta Llama 3 reference code](https://github.com/meta-llama/llama3/blob/main/llama/model.py) —
   RoPE, RMSNorm, SwiGLU и GQA в современной реализации.

`nanoGPT` полезен как компактный исторический reference, но сам автор теперь
направляет к более современному `nanochat`.

## 13. Проверка понимания через трассировку shapes

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

Если shape неожиданно меняется вдоль residual stream, это почти всегда означает
пропущенную projection или ошибочную конкатенацию heads.

## Что должно остаться после главы

- Original Transformer — encoder-decoder, а не GPT-подобная башня.
- Encoder block: full self-attention + FFN.
- Original decoder block: causal self-attention + cross-attention + FFN.
- Attention отвечает за communication; FFN — за token-wise nonlinear compute.
- Residual stream сохраняет `B × T × d_model`.
- BERT, GPT и LLaMA различаются не только маской, но objective, positions,
  normalization и FFN.
- Modern LLM diagram нельзя выдавать за точную схему статьи 2017 года.

## Источники и интерактивы

### Первичные

- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Radford et al. — Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Devlin et al. — BERT](https://arxiv.org/abs/1810.04805)
- [Touvron et al. — LLaMA](https://arxiv.org/abs/2302.13971)

### Код и объяснения

- [Harvard — The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy — Let’s build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy — Let’s reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Legacy deep dive: Transformer]]

**Назад:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]
