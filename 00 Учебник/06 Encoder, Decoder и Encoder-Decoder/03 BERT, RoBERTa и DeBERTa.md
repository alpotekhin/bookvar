---
title: BERT, RoBERTa и DeBERTa
type: textbook-chapter
status: canonical
last_updated: 2026-10-29
previous: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение]]"
reference_terms:
  - label: Whole-word masking — выбор целых слов
    label_en: Whole-word masking — grouped targets
    anchor: whole-word-masking
  - label: SpanBERT и Span Boundary Objective
    label_en: SpanBERT and the span boundary objective
    anchor: spanbert-boundary-objective
  - label: Распределение длин фрагментов SpanBERT
    label_en: SpanBERT span-length distribution
    anchor: spanbert-span-sampling
  - label: ALBERT и Sentence Order Prediction
    label_en: ALBERT and sentence-order prediction
    anchor: sentence-order-prediction
primary_sources:
  - https://aclanthology.org/N19-1423/
  - https://arxiv.org/abs/1907.11692v1
  - https://arxiv.org/abs/2006.03654v6
  - https://aclanthology.org/2020.tacl-1.5/
  - https://arxiv.org/abs/1909.11942v6
---

# BERT, RoBERTa и DeBERTa

В BERT один предобученный энкодер становится основой для классификации,
разметки токенов и извлечения ответов. Сначала его обучают на обычных текстах,
затем добавляют выходной слой нужного формата и дообучают на разметке.
Например, для пары «вопрос — абзац» модель должна найти начало и конец ответа
в абзаце. Внимание связывает оба текста уже внутри энкодера.

[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Предыдущая глава]]
объясняет, почему двунаправленному энкодеру нужна иная задача, чем NTP.
Здесь разобран полный путь BERT: подготовка пары, входные представления,
MLM/NSP, обновление весов и применение к размеченной задаче.
RoBERTa изменяет этот рецепт обучения, а DeBERTa — вычисление внимания
и восстановление скрытых токенов.

## Архитектура и формы тензоров

`BERT_BASE` содержит 12 блоков, ширину $d=768$ и 12 голов.
Ширина одной головы равна $768/12=64$, промежуточная ширина FFN — 3072.
У `BERT_LARGE` 24 блока, $d=1024$, 16 голов и FFN ширины 4096.
Авторы указывают примерно 110 и 340 млн параметров соответственно
([статья, §3](https://aclanthology.org/N19-1423/)).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-base-bert-large-encoders.png]]

*Стеки из 12 и 24 энкодерных блоков. Источник: Jay Alammar,
[The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Исходный рисунок сохранён без изменений.*

Каждый блок применяет self-attention, затем FFN с GELU.
В исходном BERT нормализация стоит после прибавления остаточной связи:

$$
U=\operatorname{LN}(H+\operatorname{Dropout}(\operatorname{MHA}(H))),
$$

$$
H'=\operatorname{LN}(U+\operatorname{Dropout}(\operatorname{FFN}(U))).
$$

Это post-norm схема. Отдельного стека декодера и cross-attention в BERT нет.
При этом совместное self-attention над вопросом и абзацем разрешает
позициям одного текста обращаться к позициям другого.
Маска закрывает PAD-ключи, но не правый контекст.

### Как из двух фрагментов получается один вход

В статье «предложением» называется произвольный связный фрагмент:
он может содержать несколько предложений в лингвистическом смысле.
Сначала выбираются два таких фрагмента A/B, затем выполняется
WordPiece-токенизация. Пример из рис. 2 статьи:

```text
[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
```

Здесь 11 позиций. Токен `##ing` — продолжение слова, а не отдельное слово.
`[CLS]` начинает весь вход, первый `[SEP]` завершает A, второй завершает B.
Позиции нумеруются непрерывно от 0 до 10; начало B не сбрасывает счётчик.

Для каждой строки складываются три обучаемых вектора:

$$
\begin{aligned}
e_i&=E_{\mathrm{token}}[x_i]\\
&\quad+E_{\mathrm{position}}[i]+E_{\mathrm{segment}}[s_i].
\end{aligned}
$$

У `[CLS]` и первого `[SEP]` тип A, у токенов второго фрагмента
и последнего `[SEP]` — тип B. Segment embedding сообщает,
к какой части относится токен, но не запрещает внимание между частями.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure2.png]]

*Три слагаемых входного представления. Для `play` и `##ing`
используются разные токенные и позиционные строки, но одинаковый тип B.
Источник: Jacob Devlin et al., [BERT, рис. 2](https://aclanthology.org/N19-1423.pdf#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Схема и английская подпись не изменены.*

Официальный код применяет к этой сумме LayerNorm и dropout перед стеком
([`embedding_postprocessor`](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L428-L521)).
Поэтому сумма трёх таблиц описывает сборку признаков, но ещё не весь входной слой.

Для пакета с $B$ примерами, дополненного до длины $T$, получаются формы:

| Величина | Форма |
|---|---|
| ID токенов, типов и маска действительных позиций | `[B,T]` |
| Состояния до и после каждого блока | `[B,T,d]` |
| Q/K/V после разбиения на головы | `[B,h,T,d_h]` |
| Оценки внимания | `[B,h,T,T]` |
| Логиты MLM для $m$ выбранных позиций | `[m,V]` |

В примере `B=1,T=11`. Для BASE Q/K/V имеют форму `[1,12,11,64]`.
Добавление PAD увеличивает размер пакета, но не число действительных токенов
и не множество MLM-целей. Закрытие PAD-ключей само по себе не обнуляет
выходы PAD-запросов; эти строки исключают из дальнейшей потери или pooling.

## MLM: как получить двунаправленный обучающий сигнал

Из обычных подслов выбираются позиции для восстановления.
Исходный BERT использует долю около 15%; `[CLS]` и `[SEP]`
не входят в список кандидатов официального генератора.
Отбор выполняется после WordPiece-разбиения: в исходном рецепте
можно выбрать одну часть слова, не выбирая остальные
([BERT, §3.1, A.1–A.2](https://aclanthology.org/N19-1423/)).
Whole-word masking — более поздний вариант подготовки данных.

Пусть в примере выбрано `cute`. Вход может содержать `[MASK]`,
случайный токен или исходное `cute`; правильной целью во всех случаях
остаётся ID `cute`. Среди выбранных позиций эти варианты имеют вероятности
80%, 10% и 10%. Они не означают, что 80% всего текста скрыто.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png]]

*Состояние выбранной позиции преобразуется в распределение по словарю.
«All English words» на исходном рисунке — упрощение: настоящий выход
индексируется WordPiece-словарём, включая подслова и служебные токены.
Источник: Jay Alammar, [The Illustrated BERT, Masked Language Model](https://jalammar.github.io/illustrated-bert/#masked-language-model),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Двунаправленное внимание строит $h_{\mathrm{cute}}$ по всей испорченной паре.
MLM-голова исходной реализации выполняет ещё одно преобразование:

$$
\begin{aligned}
z_i&=\operatorname{LN}
\bigl(\operatorname{GELU}(h_iW_m+b_m)\bigr),\\
\ell_i&=z_iE_{\mathrm{token}}^\top+b_{\mathrm{vocab}}.
\end{aligned}
$$

Выходная матрица разделяет веса с входной токенной таблицей.
Дополнительные dense/GELU/LayerNorm и словарное смещение принадлежат
MLM-голове; при переходе к классификации они не нужны.
Источник: [официальный `get_masked_lm_output`](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_pretraining.py#L240-L282).

Пусть $\mathcal M$ — выбранные позиции, а $\tilde x$ — испорченный вход.
Средняя MLM-потеря равна

$$
L_{\mathrm{MLM}}=
-\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).
$$

В [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|вводной]]
уже проверено, почему выбранный неизменённый токен тоже даёт потерю.
Здесь используется тот же принцип: метки хранятся отдельно от испорченных ID.
У невыбранных строк нет прямой MLM-потери, но их состояния участвуют
в attention выбранной позиции и получают градиент через эти зависимости.

Смесь замен уменьшает несовпадение с применением, где `[MASK]` обычно нет.
В приложении C.2 авторы отдельно проверяют варианты 80/10/10 и 100% MASK.
При полном fine-tuning результаты сравнительно устойчивы, а в опыте
с фиксированными признаками для NER замена всех целей на MASK даёт
94.0 против 94.9 Dev F1 у 80/10/10. Это результат конкретного опыта,
а не доказательство оптимальности пропорции для любых моделей.

## Вторая цель BERT: отношения между сегментами

Для NSP половина пар содержит настоящий следующий фрагмент документа,
а вторая половина — случайный второй фрагмент.
Например, после `my dog is cute` продолжение `he likes playing`
и фрагмент про пингвинов дают разные метки.
Это примеры устройства цели, не данные для оценки качества.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-next-sentence-prediction.png]]

*Входная пара поступает в один двунаправленный энкодер,
а `[CLS]` используется для классификации пары.
Средний ряд Tokenized Input показывает только первый фрагмент;
настоящий вход содержит оба сегмента и разделители, как нижняя строка.
Источник: Jay Alammar, [The Illustrated BERT, Two-sentence Tasks](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

В официальной модели последнее состояние `[CLS]` проходит через pooler:
`dense → tanh`. Из результата бинарная голова получает два логита.
Код использует метку 0 для настоящего продолжения и 1 для случайного.
NSP не классифицирует каждую пару соседних слов и не выдаёт оценку
всей последовательности по правилу NTP.

За шаг складываются средняя потеря выбранных MLM-токенов
и средняя потеря NSP по примерам пакета:

$$
L=L_{\mathrm{MLM}}+L_{\mathrm{NSP}}.
$$

Суммировать необработанные потери всех MLM-позиций с одной NSP-потерей —
другая нормировка: соотношение целей тогда меняется вместе с длиной текста.
В исходном коде для MLM применяется сумма весов действительных целей
и добавляется $10^{-5}$ в знаменатель.
Это защищает деление, но не делает пакет без обучающих целей полезным.

### Один исполняемый шаг предобучения

Код ниже проводит испорченную пару через маленький post-norm энкодер,
MLM-голову с общими весами и NSP-голову, вычисляет обе потери и обновляет веса.
Он использует готовый attention PyTorch; операции внутри головы
разобраны в [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|главе о Transformer]].

Здесь 12 условных токенов словаря, два слоя ширины 16 и четыре головы.
Отбор одного `cute` задан заранее, а не получен реальным WordPiece-токенизатором.
Dropout отключён для воспроизводимой проверки, поэтому это вычислительный
пример, не воспроизведение обучения опубликованного BERT.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure1.png]]

*Слева общий энкодер получает пару A/B: MLM-выходы восстанавливают
выбранные позиции, а NSP-выход читает `[CLS]`. Справа те же предобученные
параметры служат начальной точкой отдельных моделей размеченных задач.
Стрелки между частями показывают перенос параметров, не передачу
текущих состояний из одной задачи в другую.
Источник: Jacob Devlin et al., [BERT, рис. 1](https://aclanthology.org/N19-1423.pdf#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Обе панели и исходная английская подпись сохранены без изменений.*

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(31)
torch.set_num_threads(1)

class TinyBertPretraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.word = nn.Embedding(12, 16)
        self.position = nn.Embedding(32, 16)
        self.segment = nn.Embedding(2, 16)
        self.input_norm = nn.LayerNorm(16, eps=1e-12)
        activation = lambda x: F.gelu(x, approximate="tanh")
        block = nn.TransformerEncoderLayer(
            16, 4, dim_feedforward=64, dropout=0.0,
            activation=activation, layer_norm_eps=1e-12,
            batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(
            block, 2, enable_nested_tensor=False)
        self.mlm_transform = nn.Sequential(
            nn.Linear(16, 16), nn.GELU(approximate="tanh"),
            nn.LayerNorm(16, eps=1e-12))
        self.vocab_bias = nn.Parameter(torch.zeros(12))
        self.pooler = nn.Linear(16, 16)
        self.nsp = nn.Linear(16, 2)

    def forward(self, ids, types, valid, chosen):
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.input_norm(
            self.word(ids)+self.position(pos)[None]+self.segment(types))
        h = self.encoder(h, src_key_padding_mask=~valid)
        z = self.mlm_transform(h[chosen])
        mlm_logits = F.linear(z, self.word.weight, self.vocab_bias)
        nsp_logits = self.nsp(torch.tanh(self.pooler(h[:, 0])))
        return h, mlm_logits, nsp_logits

# PAD=0, CLS=1, SEP=2, MASK=3, my=4, dog=5, is=6, cute=7,
# he=8, likes=9, play=10, ##ing=11.
original = torch.tensor([[1, 4, 5, 6, 7, 2, 8, 9, 10, 11, 2, 0]])
types = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]])
valid = original.ne(0)
chosen = torch.zeros_like(valid)
chosen[0, 4] = True
ids = original.clone()
ids[chosen] = 3
targets = original[chosen]         # cute, not MASK
nsp_target = torch.tensor([0])

model = TinyBertPretraining().double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
word_inputs = []
def save_word_inputs(_module, _args, output):
    word_inputs.append(output)
hook = model.word.register_forward_hook(save_word_inputs)
h, mlm_logits, nsp_logits = model(ids, types, valid, chosen)
hook.remove()
assert h.shape == (1, 12, 16) and mlm_logits.shape == (1, 12)
assert nsp_logits.shape == (1, 2)
mlm = F.cross_entropy(mlm_logits, targets, reduction="sum")/(chosen.sum()+1e-5)
nsp = F.cross_entropy(nsp_logits, nsp_target)
loss = mlm+nsp
# Differentiate MLM alone with respect to the input embedding vectors.
mlm_input_grad, = torch.autograd.grad(mlm, word_inputs[0], retain_graph=True)
assert mlm_input_grad[0, 2].norm() > 0  # unchosen dog is used as context
assert torch.count_nonzero(mlm_input_grad[~valid]) == 0
optimizer.zero_grad()
before = model.word.weight.detach().clone()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(model.word.weight.grad).all()
assert model.word.weight.grad[0].norm() > 0  # PAD remains an output class
optimizer.step()
assert not torch.equal(before, model.word.weight.detach())
print("MLM", float(mlm.detach()), "NSP", float(nsp.detach()))
print("dog input MLM gradient norm", float(mlm_input_grad[0, 2].norm()))
print("PAD input MLM gradient norm", float(mlm_input_grad[~valid].norm()))
print("hidden/MLM/NSP", tuple(h.shape),
      tuple(mlm_logits.shape), tuple(nsp_logits.shape))
```

Одни и те же веса энкодера получили два сигнала: восстановить `cute`
и классифицировать связь фрагментов. Словарная таблица получила градиент
через входные представления и через общую выходную проекцию.
Служебный PAD не выбран целью и закрыт как ключ.
Значения потерь случайно инициализированной сети не являются результатом
на языковой задаче.

Чтобы проверить участие невыбранного `dog` именно в MLM, код вычисляет
производную MLM-потери по промежуточным входным векторам.
Элемент `[0,2]` соответствует `dog`: он не является целью,
но его представление влияет на восстановление `cute` через attention.
NSP-потеря в этом измерении не участвует. Производная берётся по входному
вектору позиции, а не по строке весов общей входной/выходной таблицы.
Входной PAD-вектор, напротив, получает нулевой MLM-градиент:
его нельзя прочитать как ключ ни на одном слое.
При указанном seed на CPU с float64 норма градиента входного `dog`
равна примерно $0.118682$, а PAD — нулю.

У общей таблицы `word.weight` другая роль: её строки участвуют также
в словарном softmax. Поэтому ненулевой `word.weight.grad[5]` сам по себе
не доказывал бы использование входного `dog`.
Даже строка PAD получает градиент через выходной классификатор:
предсказывается `cute`, но вероятности нормируются по всем 12 классам,
включая PAD. Маска входного attention этот выходной класс не удаляет.

## Данные и рецепт BERT

В статье используются BooksCorpus, около 800 млн слов, и английская
Wikipedia, около 2.5 млрд слов. Из Wikipedia берут связные текстовые
фрагменты, исключая списки, заголовки и таблицы. Сохранённые границы
документов позволяют выбирать настоящие продолжения и длинный контекст.

Рецепт приложения A.2:

| Параметр | Исходное обучение |
|---|---|
| Обновления | 1 млн |
| Пакет | 256 последовательностей |
| Скорость обучения | разогрев 10 тыс. шагов до $10^{-4}$, затем линейное уменьшение |
| Adam | $\beta_1=0.9,\ \beta_2=0.999$ |
| Weight decay | 0.01 |
| Dropout | 0.1 |
| Длина | 128 первые 90% шагов, 512 последние 10% |

Переход к 512 даёт возможность обучить дальние абсолютные позиции,
не оплачивая длинное внимание на каждом раннем шаге.
Число обработанных последовательностей здесь нельзя сразу умножать
на 512 и называть фактическим числом содержательных токенов:
часть обучения имеет длину 128, примеры могут быть короче, встречается padding.

BASE обучали четыре дня на четырёх Cloud TPU в Pod-конфигурации
(16 TPU-чипов), LARGE — на 16 Cloud TPU (64 чипа).
Это описание аппаратуры эксперимента 2018 года, не обещание времени
обучения на современном ускорителе.

<a id="один-encoder--разные-способы-чтения-выхода"></a>

## Дообучение на размеченной задаче

После предобучения сохраняют токенизацию, входные таблицы и энкодер.
MLM/NSP-головы заменяются выходом нужного формата.
При полном fine-tuning градиент размеченной задачи меняет весь стек.
Для каждой задачи получается собственная копия дообученных весов;
один предобученный checkpoint не означает один общий готовый классификатор.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/architecture-patterns-2026-10-21/bert-figure4.png]]

*Сверху классификация одного текста и пары, снизу извлечение ответа и NER.
Источник: Jacob Devlin et al., [BERT, рис. 4](https://aclanthology.org/N19-1423.pdf#page=15),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Сохранены четыре панели и исходная подпись.*

Для классификации берут состояние `[CLS]` или соответствующий pooled output
и обучают выход по классам. В NER проекция применяется к состояниям токенов.
Если слово распалось на несколько WordPiece, нужна договорённость о разметке:
в опыте BERT используется состояние первого подслова.
Служебные позиции и PAD не должны незаметно добавлять обучающие метки.

### Извлечение ответа: две проекции и допустимый фрагмент

Для пары «вопрос — абзац» два обучаемых вектора $s,e$ дают
$\ell_i^{\mathrm{start}}=s^\top h_i$ и
$\ell_i^{\mathrm{end}}=e^\top h_i$.
Softmax нормируется по допустимым позициям, а не по словарю.
Правильные начало и конец задаются индексами в исходном абзаце.

Независимые максимумы начала и конца могут задать невозможный фрагмент.
Пусть абзац условно состоит из `Marie lives in Paris`.
Заданы логиты начала $(0,0,1,3)$ и конца $(4,0,0,2)$.
Независимый выбор даст начало `Paris` и конец `Marie`.
При ограничении $j\ge i$ максимальная сумма равна $3+2=5$:
модель выбирает `Paris` как однотокенный ответ.

```python
import torch
import torch.nn.functional as F

words = ["Marie", "lives", "in", "Paris"]
start = torch.tensor([0., 0., 1., 3.], requires_grad=True)
end = torch.tensor([4., 0., 0., 2.], requires_grad=True)
i, j = torch.arange(4)[:, None], torch.arange(4)[None, :]
scores = (start[:, None]+end[None, :]).masked_fill(j<i, -torch.inf)
flat = int(scores.argmax())
left, right = divmod(flat, 4)
assert (int(start.argmax()), int(end.argmax())) == (3, 0)
assert (left, right) == (3, 3)
loss = (F.cross_entropy(start[None], torch.tensor([3]))+
        F.cross_entropy(end[None], torch.tensor([3])))
loss.backward()
print("answer:", " ".join(words[left:right+1]), "loss:", float(loss.detach()))
```

В настоящем пакете нужно также исключить вопрос, разделители и padding,
а длинный документ разбивается на перекрывающиеся окна с отображением
индексов обратно в текст. Код выше проверяет только выбор допустимого
фрагмента по заданным логитам и его потерю; энкодер в нём не обучался.

В SQuAD 2.0 часть вопросов не имеет ответа. В исходном BERT для них
правильными началом и концом становится `[CLS]`.
При декодировании лучший непустой фрагмент сравнивается с суммой
оценок начала/конца `[CLS]` и порогом, подобранным на dev.
Это другая постановка, чем обязательный выбор некоторого слова абзаца
([BERT, §§4.2–4.3](https://aclanthology.org/N19-1423/)).

### Фиксированные признаки или полный fine-tuning

Можно заморозить BERT и обучать отдельную модель на его состояниях.
Тогда признаки для неизменного входа можно заранее вычислить.
Но выбирать последнее состояние каждого токена необязательно:
в статье сравниваются несколько слоёв и их комбинации.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png]]

*Сравнение способов извлечения признаков для CoNLL-2003 NER.
Источник: Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
по [BERT, §5.3, табл. 7](https://aclanthology.org/N19-1423/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

При фиксированном BERT конкатенация последних четырёх слоёв дала 96.1
Dev F1, последний слой — 94.9. Сверху в этом опыте обучалась двухслойная
BiLSTM и классификатор. Полный fine-tuning BASE дал 96.4 Dev F1.
По этим числам нельзя заключить, что последние четыре слоя всегда
лучше последнего: здесь сравнивались конкретные признаки, модель и NER-набор.

`[CLS]` также не становится поисковым вектором по определению.
Для независимых представлений документов нужна обученная геометрия близости;
совместный классификатор пары и bi-encoder решают разные задачи.

## Как читать результаты BERT

В табл. 5 статьи BASE получил 88.5 SQuAD 1.1 Dev F1.
Удаление NSP при сохранении MLM дало 87.9; переход к LTR без NSP — 77.8.
Прибавление BiLSTM к LTR повысило результат до 84.9.
Сравнение 88.5 и 77.8 меняет сразу две части предобучения;
для проверки LTR авторы сопоставляют вариант без NSP с LTR без NSP.
Это не численная оценка эффекта одной маски для всех возможных рецептов.

Лидербордовые результаты имеют другой протокол.
91.8 Test F1 на SQuAD 1.1 относится к одной LARGE-модели с предварительным
дообучением на TriviaQA; у LARGE без TriviaQA в таблице указан 90.9 Dev F1.
Для SQuAD 2.0 одна LARGE-модель получила 83.1 Test F1.
Нельзя поставить все эти значения в одну колонку без указания набора,
внешних данных и ансамблирования.

<a id="roberta-когда-новый-результат-даёт-не-новая-архитектура"></a>

## RoBERTa: изменения в обучении

RoBERTa сохраняет двунаправленный стек и MLM.
Liu et al. сначала исследуют подготовку примеров и оптимизацию при BASE-размере,
затем объединяют изменения в LARGE-модели.
Поэтому последовательность сравнений в статье помогает отделить эффект
отдельной настройки от результата всего рецепта
([§§3–5](https://arxiv.org/pdf/1907.11692v1)).

### Статические и динамические маски

В оригинальной подготовке BERT испорченный пример записывается заранее.
Чтобы одна строка не появлялась только с одной маской, данные дублируются
десять раз с разными вариантами. В описанном RoBERTa режиме из 40 эпох
каждый такой вариант встречается примерно четыре раза.

При динамическом маскировании позиции заново выбираются перед подачей входа.
Например, одна и та же токенизированная строка может сначала восстанавливать
`cute`, позже `dog`, затем `likes`. Меняется обучающий пример,
а не маска доступности двунаправленного attention.

| Вариант в репликации BASE | SQuAD 2.0 F1 | MNLI-m accuracy | SST-2 accuracy |
|---|---:|---:|---:|
| статические маски | 78.3 | 84.3 | 92.5 |
| динамические маски | 78.7 | 84.0 | 92.9 |

Источник чисел: [RoBERTa, табл. 1](https://arxiv.org/pdf/1907.11692v1#page=4).
Это медианы по пяти запускам. Эффект небольшой и не однонаправленный
для всех трёх задач; весь последующий выигрыш RoBERTa нельзя отнести
к одному динамическому маскированию.

### NSP и способ сборки текста

В §4.2 сопоставляются четыре формата:

| Формат | Какие тексты попадают в один вход | NSP |
|---|---|---|
| SEGMENT-PAIR | два фрагмента, каждый может содержать несколько предложений | да |
| SENTENCE-PAIR | два отдельных предложения | да |
| FULL-SENTENCES | целые предложения, заполняющие вход до 512; можно переходить к новому документу через разделитель | нет |
| DOC-SENTENCES | целые предложения только одного документа; конец документа может дать короткий вход | нет |

Короткая пара отдельных предложений даёт меньше дальних зависимостей,
чем длинные связные фрагменты. Авторы увеличивают пакет коротких примеров,
чтобы примерно сохранить число токенов, но в их опыте SENTENCE-PAIR
всё равно уступает SEGMENT-PAIR.

У FULL-SENTENCES SQuAD 2.0 Dev F1 равен 79.1,
у DOC-SENTENCES — 79.7; у репликации SEGMENT-PAIR с NSP — 78.7.
Для последующих опытов выбрали FULL-SENTENCES: при ограничении одним
документом длины и число примеров в пакете сильнее меняются.

Здесь одновременно меняются NSP и организация входа.
Из таблицы следует, что для полученного рецепта можно обойтись без NSP.
Она не доказывает, что добавление любого NSP к любому обучению обязательно вредит.
Сопоставление с абляцией BERT требует учитывать, какие входы остаются
после удаления функции потерь.

### Пакет, число шагов и словарь

Больше примеров за обновление позволяет эффективнее распределить вычисления.
Но 1 млн шагов с пакетом 256 и 1 млн шагов с пакетом 8192 —
разные объёмы предъявленных данных. В контролируемом опыте §4.3
число шагов уменьшают при увеличении пакета и отдельно подбирают
скорость обучения:

| Размер пакета | Шаги | Пиковая скорость обучения | MNLI-m |
|---|---:|---:|---:|
| 256 | 1 млн | $10^{-4}$ | 84.7 |
| 2K | 125 тыс. | $7\cdot10^{-4}$ | 85.2 |
| 8K | 31 тыс. | $10^{-3}$ | 84.6 |

Источник: [RoBERTa, табл. 3](https://arxiv.org/pdf/1907.11692v1#page=6).
Сходное число предъявлений не делает траектории оптимизации одинаковыми.
По MNLI здесь лучший вариант — 2K, а не самый большой пакет.
В итоговом рецепте используется 8K, удобный для масштабного обучения.

RoBERTa также переходит с WordPiece BERT к byte-level BPE
примерно на 50 тыс. единиц. Базовый байтовый алфавит позволяет
представлять вход без `[UNK]` из-за отсутствующего символа,
но не устраняет ограничение длины модели.
При увеличении словаря токенная таблица добавляет около 15 млн параметров
для BASE и 20 млн для LARGE при сохранении устройства основных блоков.
В §4.4 авторы отмечают небольшое ухудшение некоторых задач при новой
токенизации, а не объявляют её самостоятельной причиной всех улучшений.

### Итоговый рецепт и результаты

Исходные 16 ГБ BooksCorpus/Wikipedia дополняют CC-NEWS, OpenWebText
и STORIES; общий корпус превышает 160 ГБ.
В RoBERTa не используют ранний этап на длине 128:
входы собирают до полной длины 512.
Увеличиваются также длительность обучения и число предъявленных примеров.

| Данные и шаги LARGE, пакет 8K | SQuAD 2.0 Dev F1 | MNLI-m Dev accuracy |
|---|---:|---:|
| 16 ГБ, 100 тыс. | 87.3 | 89.0 |
| 160 ГБ, 100 тыс. | 87.7 | 89.3 |
| 160 ГБ, 300 тыс. | 88.7 | 90.0 |
| 160 ГБ, 500 тыс. | 89.4 | 90.2 |

Источник: [RoBERTa, табл. 4](https://arxiv.org/pdf/1907.11692v1#page=7).
Строки последовательно накапливают изменения.
Переход 16 → 160 ГБ меняет не только размер, но и состав корпуса;
авторы прямо оставляют отдельное исследование этих факторов на будущее.

Часто цитируемый GLUE 88.5 относится к тестовому ансамблю,
а не к одной модели из dev-таблицы. Для некоторых задач также меняется
дообучение: RTE/STS/MRPC стартуют с MNLI-модели, QNLI для leaderboard
переформулируется как ранжирование. SQuAD 2.0 Dev F1 89.4
и RACE Test accuracy 83.2 относятся к другим указанным в статье протоколам.

<a id="whole-word-masking"></a>
## Whole-word masking selects targets together

If a name is tokenized as `phil ##am ##mon`, predicting only
`##am` leaves the other parts visible. Whole-word masking groups those
three positions when selecting prediction targets. It keeps WordPiece
tokenization and the same vocabulary-sized prediction head; it does not
turn the name into one new vocabulary item.

The original BERT release explains the unchanged prediction task:

> The training is identical -- we still predict each masked WordPiece token
> independently. The improvement comes from the fact that the original prediction
> task was too 'easy' for words that had been split into multiple WordPieces.

— Google Research, [BERT README, Whole Word Masking Models, May 31, 2019](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/README.md#L74-L118),
Apache-2.0. This is the authors' motivation, not a claim that every
partially visible word has only one possible completion.

In the pinned generator, a piece beginning with `##` joins the preceding
candidate group. The groups are shuffled, then selected under the
prediction budget. A group that would exceed the remaining budget is
skipped rather than partly selected. Consequently, grouping can leave a
budget underfilled; selecting exactly 15% of the positions in every short
example is not a requirement.

The following reduced selector receives an explicit group order instead
of drawing a random shuffle. It isolates grouping and budget handling
from token replacement:

```python
def whole_word_groups(pieces):
    groups = []
    for index, piece in enumerate(pieces):
        if piece in ("[CLS]", "[SEP]"):
            continue
        if piece.startswith("##") and groups:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups

def choose_word_targets(pieces, group_order, budget):
    groups = whole_word_groups(pieces)
    chosen = []
    for group_index in group_order:
        candidate = groups[group_index]
        if len(chosen) + len(candidate) <= budget:
            chosen.extend(candidate)
    return sorted(chosen)

pieces = ["[CLS]", "the", "man", "likes", "play", "##ing", "[SEP]"]
assert whole_word_groups(pieces) == [[1], [2], [3], [4, 5]]
order = [3, 1, 0, 2]
assert choose_word_targets(pieces, order, budget=2) == [4, 5]
assert choose_word_targets(pieces, order, budget=1) == [2]
assert choose_word_targets(["[CLS]", "play", "##ing", "[SEP]"],
                           [0], budget=1) == []

targets = choose_word_targets(pieces, order, budget=2)
corrupted = pieces.copy()
corrupted[4] = "[MASK]"
# An unchanged draw is allowed for the other selected piece.
assert corrupted[5] == "##ing"
assert [pieces[i] for i in targets] == ["play", "##ing"]
```

Both selected positions retain their original labels, even though one
input piece is unchanged. The actual BERT generator applies its 80/10/10
replacement decision separately inside the selected group. Thus “select
the whole word” does not mean “all its input pieces must be literal
`[MASK]` tokens.” The README's all-MASK example illustrates one outcome,
while [the generator, lines 341–415](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L341-L415)
defines the replacement procedure.

This code assumes a valid WordPiece sequence and a permutation of its
candidate groups. Other tokenizers need their own word-boundary mapping;
a SentencePiece boundary marker is not the same convention as `##`.
The existing MLM head still predicts each selected subword separately
from its contextual state. Whole-word selection changes the corruption
distribution, not the encoder architecture or the attention visibility
mask.

The original release reports improved SQuAD and MultiNLI results for its
new BERT-Large checkpoints. That does not establish a universal gain:
[SpanBERT, Table 6](https://aclanthology.org/2020.tacl-1.5/)
reports 77.7 coreference F1 for subword masking and 76.6 for whole-word
masking in its particular bi-sequence ablation. Dataset, training budget,
and the other objectives belong with any such comparison.

<a id="spanbert-boundary-objective"></a>
## SpanBERT: predict a span from its boundaries

An extractive QA model locates a contiguous answer, and a coreference
model compares mentions such as `the champion` and `Denver Broncos`.
Their downstream classifiers often read a span's boundary states.
SpanBERT adds a pretraining loss that asks those states to retain enough
information to predict the words between them.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/spanbert-boundary-objective.png|Original SpanBERT span-boundary objective and token predictions]]

*Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer
and Omer Levy, [SpanBERT, Figure 1, printed p. 65](https://aclanthology.org/2020.tacl-1.5.pdf#page=2).
Original figure and complete caption, CC BY 4.0. Only outer page material
was cropped; the formula, all token states and all arrows are retained.*

The paper describes the auxiliary prediction directly:

> Given a masked span of tokens
> $(x_s,\ldots,x_e)\in Y$, where $(s,e)$ indicates its start
> and end positions, we represent each token $x_i$
> in the span using the output encodings of the
> external boundary tokens $\mathbf{x}_{s-1}$ and $\mathbf{x}_{e+1}$, as well
> as the position embedding of the target token
> $\mathbf{p}_{i-s+1}$:

$$
y_i=f(h_{s-1},h_{e+1},p_{i-s+1}).
$$

— [Joshi et al., section 3.2](https://aclanthology.org/2020.tacl-1.5/),
CC BY 4.0. Bold contextual vectors in the source equation are written
as $h$ here to distinguish them from input token IDs.

In Figure 1, positions 5–8 contain `an American football game`.
The external boundary states are $h_4$ for `was` and $h_9$ for `to`.
To predict `football` at position 7, the boundary head receives
$[h_4;h_9;p_3]$. It does not read $h_7$ directly. The ordinary MLM head
does read $h_7$, so the same target has two prediction paths.

The offset embedding distinguishes the first, second, third and fourth
outputs for one common pair of boundaries. Without an offset or another
position-dependent input, a deterministic head receiving the same two
vectors would return the same vocabulary distribution at every position
of the span. The offset specifies a location; the target word itself
remains a label outside the input.

The paper uses two dense transformations, each followed by GELU and
LayerNorm:

$$
\begin{aligned}
u_i&=[h_{s-1};h_{e+1};p_{i-s+1}],\\
a_i&=\operatorname{LN}(\operatorname{GELU}(W_1u_i)),\\
y_i&=\operatorname{LN}(\operatorname{GELU}(W_2a_i)).
\end{aligned}
$$

The input vocabulary embedding matrix supplies the tied output weights
for both predictions. For each selected token, the objective adds

$$
L_i=-\log P_{\mathrm{MLM}}(x_i\mid h_i)
    -\log P_{\mathrm{SBO}}(x_i\mid y_i).
$$

For illustrative probabilities 0.5 from MLM and 0.25 from SBO, this
token contributes $-\log0.5-\log0.25=\log8\approx2.0794$.
Dropping the second term would remove the training signal to the
boundary head. The predictions within the span are not fed to one
another as decoder inputs; this remains encoder pretraining, not
autoregressive generation of the missing phrase.

The small module below checks the two read paths. It starts with supplied
final encoder states rather than running another Transformer. Its widths
are 4 for a state and 2 for an offset, not the original model dimensions.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(131)
dtype = torch.float64
final_states = nn.Parameter(torch.randn(12, 4, dtype=dtype))
offsets = nn.Embedding(4, 2, dtype=dtype)
vocabulary = nn.Embedding(12, 4, dtype=dtype)
boundary_head = nn.Sequential(
    nn.Linear(10, 8, bias=False, dtype=dtype), nn.GELU(),
    nn.LayerNorm(8, dtype=dtype),
    nn.Linear(8, 4, bias=False, dtype=dtype), nn.GELU(),
    nn.LayerNorm(4, dtype=dtype),
)

# Zero-based rows 3 and 8 are h_4 and h_9 in the original figure.
boundary_inputs = torch.cat([
    final_states[3].expand(4, -1),
    final_states[8].expand(4, -1),
    offsets(torch.arange(4)),
], dim=-1)
sbo_states = boundary_head(boundary_inputs)
sbo_logits = F.linear(sbo_states, vocabulary.weight)
mlm_logits = F.linear(final_states[4:8], vocabulary.weight)
span_targets = torch.tensor([4, 5, 6, 7])
sbo_loss = F.cross_entropy(sbo_logits, span_targets)
mlm_loss = F.cross_entropy(mlm_logits, span_targets)
direct_sbo_gradient, = torch.autograd.grad(
    sbo_loss, final_states, retain_graph=True
)
assert direct_sbo_gradient[3].norm() > 0
assert direct_sbo_gradient[8].norm() > 0
assert torch.count_nonzero(direct_sbo_gradient[4:8]) == 0

same_offset_inputs = boundary_inputs.clone()
same_offset_inputs[:, -2:] = boundary_inputs[0, -2:]
same_offset_logits = F.linear(
    boundary_head(same_offset_inputs), vocabulary.weight
)
torch.testing.assert_close(
    same_offset_logits, same_offset_logits[0].expand_as(same_offset_logits)
)
(mlm_loss + sbo_loss).backward()
assert final_states.grad[4:8].norm() > 0
assert offsets.weight.grad.norm() > 0
assert vocabulary.weight.grad.norm() > 0
assert math.isclose(-math.log(.5) - math.log(.25), math.log(8))
```

The zero internal-state derivative above concerns direct inputs to
the SBO head at the final encoder output. In a complete encoder, the
boundary states themselves depend on other positions through attention.
SBO can therefore backpropagate through those earlier dependencies.
The check does not say that all internal-position computations receive
zero gradient throughout SpanBERT.

<a id="spanbert-span-sampling"></a>
### Sampling several words without introducing a decoder

Whole-word masking can select isolated words. SpanBERT instead starts
from a word boundary and extends a contiguous span. Its proposed
length distribution favors short spans but includes lengths up to ten
words; the total target budget is about 15% of subword positions.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/spanbert-span-lengths.png|Original SpanBERT distribution of sampled span lengths]]

*Joshi et al., [Figure 2, printed p. 66](https://aclanthology.org/2020.tacl-1.5.pdf#page=3),
CC BY 4.0. Complete original plot and caption. The horizontal axis counts
words, not WordPiece positions or decoder tokens.*

In the released implementation, weights $0.2(0.8)^{\ell-1}$ for
$\ell=1,\ldots,10$ are normalized over that finite range:

$$
P(\ell)=\frac{0.2(0.8)^{\ell-1}}{1-0.8^{10}},
\qquad \ell\in\{1,\ldots,10\}.
$$

This produces mean length about 3.8, as in the paper. Sampling an
unbounded geometric variable and replacing every value above ten
by ten would instead accumulate a tail probability at ten. That is a
different distribution from the plotted and implemented one.

```python
import math

weights = [.2 * .8**(length - 1) for length in range(1, 11)]
length_probabilities = [weight / sum(weights) for weight in weights]
mean_words = sum(length * probability for length, probability
                 in zip(range(1, 11), length_probabilities))
assert math.isclose(sum(length_probabilities), 1.)
assert 3.79 < mean_words < 3.80
clamped_geometric_mean = sum(.8**step for step in range(10))
assert 4.46 < clamped_geometric_mean < 4.47
assert not math.isclose(mean_words, clamped_geometric_mean)
```

See [the pinned sampling code, lines 74–132](https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/pretraining/fairseq/data/masking.py#L74-L132).
The paper describes spans of whole words; the released loop also stops
at the exact subword budget, so its final selected word can be cut
short. The sampled length distribution does not imply that every
realized span has that many complete words after budget and document
boundaries are applied.

The original span-level replacement setting draws the corruption type
once for a span: MASK, random replacements, or unchanged text.
This differs from BERT's per-piece replacement loop. Adjacent selected
intervals can also merge before SBO boundary targets are prepared.
External boundaries must exist; the implementation does not fabricate
a context token outside the document.

The model also uses a single contiguous training segment without NSP.
That input change and the auxiliary objective must be separated when
reading results. In the paper's 1.2M-step ablation, Table 7 reports:

| Span-masking setting | SQuAD 2.0 Dev F1 | NewsQA Dev F1 | Coreference F1 |
|---|---:|---:|---:|
| two segments + NSP | 85.4 | 73.0 | 76.4 |
| one segment | 86.7 | 73.4 | 76.3 |
| one segment + SBO | 86.8 | 74.1 | 79.0 |

The first transition changes both the segment construction and NSP.
The second keeps single-segment span masking and adds SBO.
Its coreference gain is much larger than its SQuAD gain in this ablation.
These are not the final 2.4M-step test results, and they do not establish
one fixed improvement from adding SBO to any model.

In [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5]],
a deleted span is replaced by one sentinel and a separate causal decoder
generates the missing text. SpanBERT retains the corrupted sequence's
positions and predicts its selected tokens through encoder heads.
The two methods share span-based corruption, but their target format,
conditional dependencies and computation are different.

<a id="sentence-order-prediction"></a>
## ALBERT: classify the order of two document segments

NSP's negative example can combine a paragraph about unlocking a door
with a paragraph about penguins. A classifier can use the change of topic
without learning which of two related events came first. ALBERT's
sentence-order prediction (SOP) changes how the negative pair is built.

Take consecutive segments from one document:

| Label | First segment | Second segment |
|---|---|---|
| ordered | She unlocked the door. | Then she entered. |
| swapped | Then she entered. | She unlocked the door. |

Both rows contain the same words. A topic histogram of the combined
text is identical, so that feature alone cannot distinguish these two
labels. Token order, positions and segment membership do change.
The segments need not be single linguistic sentences.

The released ALBERT generator implements the swapped case as follows:

```python title="Original ALBERT pair swap — Apache-2.0"
        elif not FLAGS.random_next_sentence and rng.random() < 0.5:
          is_random_next = True
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
          # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
          tokens_a, tokens_b = tokens_b, tokens_a
```

— Google Research, [create_pretraining_data.py, lines 338–343](https://github.com/google-research/albert/blob/b772393d3dae115b493258ce8e37c17b2cc62100/create_pretraining_data.py#L338-L343),
Copyright 2019 The Google Research Authors, Apache-2.0.
The historical variable name `is_random_next` also marks a swapped SOP
pair; it does not mean that this branch sampled another document.

Input assembly still produces `[CLS] A [SEP] B [SEP]`.
The first segment and its initial special tokens receive type 0,
the second segment and final separator type 1. Positional IDs run
through the assembled sequence. The binary head reads the pooled
`[CLS]` state, and its cross-entropy is added to MLM loss.

The following calculation uses supplied logits to check target labels,
mean cross-entropy and its gradient:

```python
from collections import Counter
import math
import torch
import torch.nn.functional as F

def assemble_sop(first, second):
    tokens = ["[CLS]", *first, "[SEP]", *second, "[SEP]"]
    types = [0] * (len(first) + 2) + [1] * (len(second) + 1)
    return tokens, types

first = ["she", "unlocked", "the", "door", "."]
second = ["then", "she", "entered", "."]
ordered, ordered_types = assemble_sop(first, second)
swapped, swapped_types = assemble_sop(second, first)
assert Counter(ordered) == Counter(swapped)
assert ordered != swapped and ordered_types != swapped_types
assert len(ordered) == len(ordered_types) == 12
assert len(swapped) == len(swapped_types) == 12
sentence_order_labels = torch.tensor([0, 1])
sop_logits = torch.tensor([[math.log(3), 0.],
                          [0., math.log(3)]], dtype=torch.float64,
                         requires_grad=True)
sop_loss = F.cross_entropy(sop_logits, sentence_order_labels)
assert math.isclose(sop_loss.item(), -math.log(.75))
sop_loss.backward()
torch.testing.assert_close(
    sop_logits.grad,
    torch.tensor([[-.125, .125], [.125, -.125]], dtype=torch.float64)
)
```

The supplied logits assign probability 0.75 to the correct label in
each row, giving mean loss about 0.28768. In training those scores
would come from the shared encoder and sentence head; the derivative
would reach both. SOP does not predict the next segment's words and
does not turn the encoder into a document generator.

For compatibility, the pinned preprocessor writes the label under the
feature name `next_sentence_labels` even for SOP. It maps ordered to 0
and swapped to 1. It also retains a random-document fallback when a
chunk contains only one segment. The ordered/swapped definition
describes the intended two-segment task; exact preprocessing must also
account for that fallback and truncation.

[Lan et al., ALBERT v6, sections 3.1 and 4.6, Table 5](https://arxiv.org/abs/1909.11942v6)
compare three losses in an ALBERT-base configuration. The following
are their reported measurements, not results of the small example:

| Sentence loss | NSP accuracy, % | SOP accuracy, % | SQuAD 2.0 Dev F1 | RACE accuracy, % |
|---|---:|---:|---:|---:|
| none | 52.4 | 53.3 | 78.1 | 61.7 |
| NSP | 90.5 | 52.0 | 77.2 | 62.3 |
| SOP | 78.9 | 86.5 | 80.0 | 64.0 |

High NSP accuracy did not transfer to the SOP diagnostic in this
experiment. The downstream columns support the authors' choice for
that setup; they do not prove that every NSP objective harms every
model, or that solving reversed pairs establishes general discourse
understanding. A useful diagnostic must test held-out documents and
control for punctuation, segment length and other ordering cues.

## DeBERTa: разделить содержание и относительное положение

В BERT абсолютная позиция складывается с токенным вектором на входе.
После проекций скалярное произведение Q/K смешивает оба сигнала.
В DeBERTa состояния содержания $H$ и таблица относительных позиций $P$
проецируются отдельно; относительный вектор выбирается для пары позиций,
а не прикрепляется к слову раз и навсегда.

Для одной головы:

$$
Q^c=HW_{q,c},\quad K^c=HW_{k,c},\quad V^c=HW_{v,c},
$$

$$
Q^r=PW_{q,r},\qquad K^r=PW_{k,r}.
$$

Векторы записаны строками. Если $H$ имеет форму $[T,d]$,
а $P$ — $[2k,d]$, пять матриц проекций имеют форму $[d,d_h]$.
Здесь $k$ ограничивает различимые расстояния, не длину attention.

### Три слагаемых одной оценки

Следуя соглашению [статьи, §3.1, формулы 3–4](https://arxiv.org/pdf/2006.03654v6#page=3),
индекс относительной строки равен

$$
\delta(i,j)=\operatorname{clip}(i-j+k,0,2k-1).
$$

Оценка связи запроса $i$ с ключом $j$:

$$
\begin{aligned}
\tilde A_{ij}
&=Q_i^cK_j^{c\top}\\
&\quad+Q_i^cK_{\delta(i,j)}^{r\top}\\
&\quad+K_j^cQ_{\delta(j,i)}^{r\top}.
\end{aligned}
$$

Первое произведение сопоставляет содержания.
Второе позволяет запросу выбирать относительное положение ключа:
запрос `learning` может по-разному оценивать слово рядом слева и далеко справа.
Третье зависит от содержания ключа и относительного положения запроса.
У него обратное расстояние $\delta(j,i)$.
Position-to-position слагаемое из общей декомпозиции авторы отбрасывают.

Возьмём $k=2$, $i=1$, $j=2$, $d_h=2$.
Тогда $\delta(i,j)=1$, $\delta(j,i)=3$.
Пусть $Q_i^c=(1,2)$, $K_j^c=(3,1)$,
$K_1^r=(0,1)$, $Q_3^r=(1,-1)$.
Три произведения равны $5$, $2$, $2$.
Перед softmax сумма масштабируется на $\sqrt{3d_h}$:
$9/\sqrt6\approx3.674235$.
Это ещё не вес внимания: для него нужны оценки остальных ключей строки.

Код дополняет пример до трёх ключей, выполняет softmax и получает
взвешенную сумму V. Здесь Q/K/V и относительные векторы уже спроецированы;
это вычисление одной головы, не полная реализация DeBERTa.

```python
import math
import torch

q = torch.tensor([[1., 0.], [1., 2.], [0., 1.]], dtype=torch.float64)
k = torch.tensor([[1., 0.], [0., 1.], [3., 1.]], dtype=torch.float64)
v = torch.tensor([[1., 0.], [0., 1.], [2., 2.]], dtype=torch.float64)
kr = torch.tensor([[2., 0.], [0., 1.], [1., 0.], [0., 2.]],
                  dtype=torch.float64)
qr = torch.tensor([[0., 1.], [1., 0.], [0., 2.], [1., -1.]],
                  dtype=torch.float64)

def disentangled(q, k, v, kr, qr):
    distance = len(kr)//2
    i, j = torch.arange(len(q))[:, None], torch.arange(len(k))[None, :]
    forward = (i-j+distance).clamp(0, 2*distance-1)
    reverse = (j-i+distance).clamp(0, 2*distance-1)
    c2c = q @ k.T
    c2p = (q[:, None, :]*kr[forward]).sum(-1)
    p2c = (k[None, :, :]*qr[reverse]).sum(-1)
    scores = (c2c+c2p+p2c)/math.sqrt(3*q.shape[-1])
    weights = scores.softmax(-1)
    return weights @ v, weights, (c2c, c2p, p2c)

output, weights, parts = disentangled(q, k, v, kr, qr)
assert [float(x[1, 2]) for x in parts] == [5., 2., 2.]
assert [float(x[1, 0]) for x in parts] == [1., 4., 1.]
torch.testing.assert_close(sum(x[1] for x in parts),
                           torch.tensor([6., 5., 9.], dtype=q.dtype))
print("scores:", sum(x[1] for x in parts)/math.sqrt(6))
print("weights:", weights[1], "output:", output[1])
```

Для строки $i=1$ суммы трёх слагаемых равны $(6,5,9)$.
После масштабирования и softmax веса примерно равны
$(0.1973,0.1312,0.6715)$. При значениях
$(1,0)$, $(0,1)$ и $(2,2)$ выход равен примерно $(1.5403,1.4742)$.
Третий ключ получил наибольший вес, но два других тоже входят в результат.

Если расстояние выходит за диапазон, используется крайняя строка таблицы.
Далёкий ключ остаётся доступен: clipping не является локальной маской.
При перестановке токенов значения $i-j$ могут измениться, даже если
сами токенные векторы те же.

Прямое создание отдельного относительного вектора для каждой пары
потребовало бы $O(T^2d_h)$ промежуточной памяти.
Алгоритм 1 статьи сначала вычисляет произведения с общей таблицей
размером $2k$, затем выбирает оценки по индексам расстояний.
Сами относительные таблицы требуют $O(kd_h)$ памяти,
но итоговая матрица оценок остаётся $T\times T$.
DeBERTa не превращает полное attention в линейное по длине.

### Enhanced Mask Decoder

EMD добавляет абсолютные позиции при восстановлении MLM-токенов,
после основной обработки с относительными позициями.
Авторы мотивируют это примером
`a new [MASK] opened beside the new [MASK]`:
оба пропуска следуют за `new`, но выполняют разные роли.
Это мотивация выбора дополнительного признака, не доказательство,
что полная относительная модель принципиально не может различить пропуски.

В [рис. 2 и приложении A.8](https://arxiv.org/pdf/2006.03654v6#page=20)
у EMD два входа. $H$ содержит контекстные состояния для K/V,
а $I$ задаёт состояния запросов. После первого прохода новое $I$
снова обращается к той же памяти $H$. В опыте статьи используются
два прохода с общими параметрами, затем MLM-голова.

В закреплённой официальной реализации, при `position_biased_input=False`,
$H$ берётся из предпоследнего выхода энкодера.
Начальные запросы равны сумме этого $H$ и абсолютных позиционных векторов.
Последний Transformer-блок вызывается дважды с одной памятью,
а его выход каждый раз заменяет запросы следующего прохода.

Ниже исходный фрагмент без перевода и изменения операторов
([Microsoft DeBERTa, `emd_context_layer`](https://github.com/microsoft/DeBERTa/blob/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098/DeBERTa/apps/models/masked_language_model.py#L51-L71)).
Copyright Microsoft Corporation; [MIT, полный текст](/licenses/microsoft-deberta-MIT.txt).

```python title="Original implementation — MIT"
    hidden_states = encoder_layers[-2]
    if not self.position_biased_input:
      layers = [encoder.layer[-1] for _ in range(2)]
      z_states +=  hidden_states
      query_states = z_states
      query_mask = attention_mask
      outputs = []
      rel_embeddings = encoder.get_rel_embedding()

      for layer in layers:
        # TODO: pass relative pos ids
        output = layer(hidden_states, query_mask, return_att=False, query_states = query_states, relative_pos=relative_pos, rel_embeddings = rel_embeddings)
        query_states = output
        outputs.append(query_states)
```

Например, два пропуска получают разные начальные позиционные векторы,
поэтому могут сформировать разные Q при общей контекстной памяти.
На втором проходе Q уже зависит и от результата первого чтения.
Это дополнительное attention-преобразование, а не прибавление двух
готовых словарных логитов или одного смещения перед softmax.

Два обращения к одной памяти можно проследить численно.
Для этого оставим обычное attention и остаточную связь: $Q=I$,
$K=V=H$, проекции единичные. Относительные слагаемые, LayerNorm и FFN
в этом небольшом примере отключены. Память содержит две строки:
$(1,0)$ и $(0,1)$. Позиционные векторы равны $(0,0)$ и $(1,0)$,
поэтому начальные запросы $I^{(0)}=H+P$ равны $(1,0)$ и $(1,1)$.
На каждом проходе выполняется одно и то же преобразование:

$$
\begin{aligned}
A^{(r)}&=\operatorname{softmax}
\left(\frac{I^{(r)}H^\top}{\sqrt2}\right),\\
I^{(r+1)}&=I^{(r)}+A^{(r)}H.
\end{aligned}
$$

Первый запрос даёт оценки $(1/\sqrt2,0)$ и веса
примерно $(0.669762,0.330238)$. Прибавление прочитанного вектора
к запросу даёт первую строку $I^{(1)}$ с координатами
$1.669762$ и $0.330238$.
При втором чтении меняется разность оценок: веса теперь равны
примерно $(0.720549,0.279451)$.
Первая строка $I^{(2)}$ получает координаты $2.390310$ и $0.609690$.
K/V в обоих проходах остались прежними.

```python
import math
import torch

memory = torch.eye(2, dtype=torch.float64)
positions = torch.tensor([[0., 0.], [1., 0.]], dtype=memory.dtype)
queries = memory+positions
history, attention_weights = [], []
for _ in range(2):
    weights = (queries @ memory.T/math.sqrt(2)).softmax(-1)
    queries = queries+weights @ memory
    attention_weights.append(weights)
    history.append(queries.clone())

torch.testing.assert_close(history[0][0], torch.tensor(
    [1.669761549, 0.330238451], dtype=memory.dtype))
torch.testing.assert_close(history[1][0], torch.tensor(
    [2.390310430, 0.609689570], dtype=memory.dtype))
torch.testing.assert_close(history[1][1], torch.tensor(
    [2., 2.], dtype=memory.dtype))
print("first read", history[0], "second read", history[1])
```

Это пример смены запросов при фиксированной памяти, а не численное
воспроизведение полного блока DeBERTa. В EMD сохранённый контекст
читается последним Transformer-блоком с его обученными проекциями,
относительными оценками, нормализацией и FFN.
Между двумя чтениями не вставляется новый токен; словарная голова
получает результат второго прохода.

Отдельной целевой последовательности и причинной генерации здесь нет.
После обучения для классификации или NER используется энкодер;
слово decoder в EMD относится к восстановлению MLM-целей.
Оригинальная схема также разобрана в
[слайдах авторов на ICLR 2021](https://iclr.cc/media/Slides/iclr/2021/virtual%2804-00-00%29-04-00-00UTC-2562-deberta_decodi.pdf).

### Дообучение, результаты и версии

SiFT из §4 — регуляризация при дообучении, не часть трёх слагаемых attention.
Модель сравнивает распределения ответов для исходного примера
и малого возмущения его входных представлений.
В SiFT возмущаются нормализованные векторы: это уменьшает зависимость
масштаба возмущения от нормы исходного эмбеддинга.
Авторы применяют SiFT к 1.5B-модели на SuperGLUE;
переносить это описание на каждый исходный DeBERTa checkpoint нельзя.

В основной LARGE-конфигурации 24 слоя, ширина 1024 и 16 голов.
После дедупликации используется около 78 ГБ Wikipedia, BooksCorpus,
OpenWebText и STORIES. В табл. 4 версии статьи v6 удаление EMD
меняет SQuAD 2.0 Dev F1 с 82.5 на 81.3, RACE accuracy с 71.7 на 70.3.
Это BASE-абляция, не результат LARGE и не измерение одного прохода EMD.

Для LARGE авторы указывают Avg 90.00 в
[табл. 1 GLUE dev](https://arxiv.org/pdf/2006.03654v6#page=6),
а также 90.7 SQuAD 2.0 F1 и 86.8 RACE accuracy в табл. 2.
При этом простое среднее восьми напечатанных GLUE-метрик равно
$719/8=89.875$, а не 90.00. Сообщённый авторами Avg не воспроизводится
этим расчётом; причину расхождения статья здесь не поясняет.
Тестовый GLUE 88.5 ансамбля RoBERTa относится к другому протоколу.
Результат 89.9 против human baseline 89.8 относится к конкретному
SuperGLUE-протоколу 1.5B-модели, а не ко всем языковым задачам.

Граница версий также существенна.
[Официальный репозиторий](https://github.com/microsoft/DeBERTa)
отдельно описывает V2: новый SentencePiece-словарь 128K, дополнительную
свёртку, общие проекции содержания/позиций и относительные buckets.
V3 меняет предобучение на ELECTRA-style распознавание заменённых токенов
и gradient-disentangled embedding sharing.
Приведённые выше формулы и шаг MLM описывают исходный вариант:
они не являются полным рецептом V3.

## Ограничения и наследие

Полное внимание обрабатывает известный текст, но MLM не задаёт обычное
авторегрессионное разложение его вероятности.
Свободная генерация требует отдельной процедуры или другого обучения.
При этом только выбранные позиции получают прямую MLM-потерю,
а полная матрица внимания всё равно вычисляется для всех действительных позиций.

Ограничение 512 относится к длине подсловного входа вместе со служебными
токенами, а не к 512 словам документа. Для длинного текста выбор окон,
перекрытие, объединение ответов и смещение индексов становятся частью метода.
Качество на коротких GLUE-примерах эти решения не проверяет.

На малых размеченных наборах результат fine-tuning зависит от начальной
инициализации головы, порядка данных и гиперпараметров.
BERT выбирает запуски по dev, RoBERTa в репликационных таблицах сообщает
медианы пяти запусков. Эти протоколы следует указывать вместе с числом,
а не выдавать лучший запуск за ожидаемое качество.

Предобучение также переносит свойства корпуса — включая ошибки
и социальные смещения. Средний результат benchmark не устанавливает
устойчивость к новому домену и не проверяет достоверность всех предсказаний.

Следующая [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|глава о GPT-1]]
разбирает причинное предобучение и дообучение на размеченных задачах.

## Источники и дальнейшее чтение

- Jacob Devlin et al., [BERT, §§3–5, приложения A.1–A.5 и C.2](https://aclanthology.org/N19-1423/).
- Yinhan Liu et al., [RoBERTa v1, §§3–5, таблицы 1–7](https://arxiv.org/abs/1907.11692v1).
- Pengcheng He et al., [DeBERTa v6, §§3–5, приложения A.7–A.8](https://arxiv.org/abs/2006.03654v6).
- Google Research, [BERT Whole Word Masking release and generator](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/README.md#L74-L118), Apache-2.0. The original explanation is quoted above; the reduced selector and budget examples are added. [Full license](/licenses/google-bert-eedf5716-Apache-2.0.txt).
- Mandar Joshi et al., [SpanBERT, sections 3, 4.2–4.3 and 6, Figures 1–2, Tables 6–7](https://aclanthology.org/2020.tacl-1.5/), CC BY 4.0. Original figures/captions and the boundary-input passage retain their language. The equations use $h$ for contextual states; the small PyTorch module and probability calculations are added.
- Zhenzhong Lan et al., [ALBERT v6, sections 3.1 and 4.6, Table 5](https://arxiv.org/abs/1909.11942v6). Pair construction and legacy feature naming are also checked against [the pinned original generator](https://github.com/google-research/albert/blob/b772393d3dae115b493258ce8e37c17b2cc62100/create_pretraining_data.py), Copyright 2019 The Google Research Authors, Apache-2.0. The swap excerpt is unchanged; the door example and supplied-logit calculation are added. [Full code license](/licenses/google-albert-b772393d-Apache-2.0.txt).
- Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/).
- Lena Voita, [Transfer Learning — BERT и fine-tuning](https://lena-voita.github.io/nlp_course/transfer_learning.html).
- [Официальный BERT: модель и подготовка данных](https://github.com/google-research/bert/tree/eedf5716ce1268e56f0a50264a88cafad334ac61).
- [Официальный DeBERTa: attention и MLM](https://github.com/microsoft/DeBERTa/tree/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098).
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-only|Encoder-only в справочнике]].
- [[02 Areas/ML & DL/04 Вопросы/100 вопросов по NLP|Вопросы по NLP]].

**Назад:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение]]
