---
title: RNN и обратное распространение во времени
type: textbook-chapter
status: canonical
last_updated: 2026-10-26
previous: "[[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram]]"
next: "[[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU]]"
primary_sources:
  - https://doi.org/10.1016/0364-0213(90)90002-E
  - https://doi.org/10.1038/323533a0
  - https://d2l.ai/chapter_recurrent-neural-networks/rnn.html
  - https://d2l.ai/chapter_recurrent-neural-networks/bptt.html
---

# RNN и обратное распространение во времени

В [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction|оконной языковой модели]]
перцептрон получает embeddings нескольких последних токенов. Если окно состоит
из двух слов, то после префиксов «кот кот спит» и «кот спит» на вход поступает
одинаковая пара «кот спит». Модель обязана выдать одинаковое распределение
следующего слова, хотя более ранняя история различается.

Рекуррентная нейронная сеть, RNN, читает по одному токену и после каждого шага
обновляет вектор состояния. При предсказании следующего слова она использует
этот вектор как представление уже прочитанного текста. Размер вектора не
меняется с длиной текста, поэтому число параметров тоже остаётся прежним.
Сохранится ли в состоянии нужная деталь из начала предложения, зависит от
параметров и обучения.

## Состояние вместо окна

Пусть сеть прочитала «кошка, которую вчера принесли домой,». Чтобы выбрать
форму «уснула», ей пригодится число и род подлежащего, отделённого несколькими
словами. Оконная модель с коротким окном уже не видит «кошка». RNN может
передать сведения о подлежащем через последовательность состояний. В ней нет
заранее назначенной координаты «род подлежащего»: представление формируется
при обучении на задаче предсказания.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/unfolded-rnn.svg]]

*Слева рекуррентная связь показана петлёй, справа тот же расчёт развёрнут
по времени. Hidden layers 1, 2 и T обозначают скрытые состояния на разных
шагах. Их значения различаются, но вычисляют их с общими весами.
Источник: Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola,
[Dive into Deep Learning, §9.4](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html),
CC BY-SA 4.0. Оригинальный SVG, без перерисовки.*

Петля на схеме не означает, что сеть должна вычислить неизвестное состояние
через него самого. На шаге $t$ состояние $h_{t-1}$ уже известно. Из него и
текущего входа вычисляется новое $h_t$. Для конечной последовательности
получается обычный ациклический вычислительный граф.

### Одна рекуррентная ячейка

Обозначим ID токена через $z_t$, а его embedding через $x_t=E[z_t]$.
В этой главе $E\in\mathbb R^{V\times d}$ хранит embeddings строками.
Состояние $h_t\in\mathbb R^m$ и вход $x_t\in\mathbb R^d$ в формулах
считаются столбцами. Простая RNN с активацией $\tanh$ вычисляет

$$
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h,
$$

$$
h_t=\tanh(a_t).
$$

Первое слагаемое преобразует текущий токен, второе преобразует прежнее
состояние, а смещение добавляется перед нелинейностью. Матрицы
$W_{xh}\in\mathbb R^{m\times d}$ и $W_{hh}\in\mathbb R^{m\times m}$
обучаются. Обычно перед новым независимым текстом задают $h_0=0$;
обучаемое начальное состояние тоже возможно, но здесь не используется.

Эту операцию можно записать как один полносвязный слой над объединённым
входом:

$$
a_t=
\begin{bmatrix}W_{xh}&W_{hh}\end{bmatrix}
\begin{bmatrix}x_t\\h_{t-1}\end{bmatrix}+b_h.
$$

Объединённый вектор имеет длину $d+m$, а объединённая матрица имеет
$m$ строк и $d+m$ столбцов. Это равенство объясняет обозначение
Concatenate на следующей схеме D2L.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rnn-bptt-2026-10-21/d2l-rnn.svg]]

*На каждом шаге вход $X_t$ объединяется с предыдущим состоянием и проходит
через полносвязный слой с активацией $\phi$. Полученное $H_t$ используется
и для выхода, и для следующего шага. Copy в легенде означает передачу
одного значения по двум ветвям, а не создание нового набора параметров.
В D2L заглавные $X,H$ описывают пакет объектов; наши $x,h$ описывают
один объект. Источник: Zhang et al.,
[D2L, §9.4.2](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html),
CC BY-SA 4.0. Оригинальный SVG.*

### Выход языковой модели

Из состояния вычисляются $V$ логитов и распределение следующего токена:

$$
o_t=W_{hy}h_t+b_y,\qquad
p_t=\operatorname{softmax}(o_t).
$$

Здесь $W_{hy}\in\mathbb R^{V\times m}$, $b_y\in\mathbb R^V$.
После чтения $z_t$ величина $p_t[j]$ означает вероятность того, что
следующим окажется токен $j$. Выход не заменяет состояние: в следующий
рекуррентный шаг передаётся $h_t$, а не вектор вероятностей $p_t$.

Вектор состояния зависит от конкретного префикса. Параметры $E,W_{xh},W_{hh}$
и $W_{hy}$ общие для всех текстов и позиций. При обучении изменяют параметры;
при чтении текста с зафиксированной моделью изменяются только состояния.

| Часть модели | Форма | При $V=8$, $d=3$, $m=4$ |
|---|---|---:|
| Embeddings $E$ | $V\times d$ | 24 |
| Входные веса $W_{xh}$ | $m\times d$ | 12 |
| Рекуррентные веса $W_{hh}$ | $m\times m$ | 16 |
| Смещение $b_h$ | $m$ | 4 |
| Выходные веса $W_{hy}$ | $V\times m$ | 32 |
| Смещение $b_y$ | $V$ | 8 |
| Всего | $Vd+md+m^2+m+Vm+V$ | 96 |

Та же сеть из 96 параметров может обработать три или триста токенов.
Увеличатся число операций и, при полном обратном проходе, объём сохранённых
активаций. Новых матриц весов для поздних позиций не появится.

В [SLP3, §14.2.3](https://web.stanford.edu/~jurafsky/slp3/14.pdf)
описано связывание входных и выходных embeddings, weight tying. В наших
обозначениях при $m=d$ можно положить $W_{hy}=E$ и вычислять $o_t=Eh_t+b_y$.
Тогда одна матрица используется как таблица входных embeddings и как
классификатор слов. В приведённой ниже модели $m=4$, $d=3$, поэтому
такое прямое связывание невозможно; выходная матрица отдельная.
Общие веса на разных шагах времени и weight tying входа с выходом
являются разными ограничениями на параметры.

Если в примере оставить $V=8$, но взять $m=d=3$, отдельная выходная
матрица содержала бы 24 параметра. Без связывания всего было бы
$24+9+9+3+24+8=77$ параметров, со связыванием $77-24=53$.
Прямой расчёт логитов сохранил бы форму $V$, но входные и выходные
представления слов больше нельзя было бы обучать независимо.

## Численный проход

Возьмём одномерную RNN без смещения и выходного классификатора:

$$
h_t=\tanh(0.5x_t+0.8h_{t-1}),\qquad h_0=0.
$$

Пусть входы равны $1,0,1$. Числа ниже округлены только для таблицы;
при дальнейших вычислениях используются неокруглённые состояния.

| Шаг | $x_t$ | $0.5x_t$ | $0.8h_{t-1}$ | $a_t$ | $h_t$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.500000 | 0 | 0.500000 | 0.462117 |
| 2 | 0 | 0 | 0.369694 | 0.369694 | 0.353724 |
| 3 | 1 | 0.500000 | 0.282979 | 0.782979 | 0.654413 |

Второй вход нулевой, но состояние осталось ненулевым: на него влияет первый
вход. Если перед третьим шагом сбросить состояние в ноль, получится
$\tanh(0.5)\approx0.462117$ вместо $0.654413$. Прежняя история
изменила результат последнего шага.

На каждом шаге прежнее состояние снова умножается на рекуррентный вес
и проходит через $\tanh$. Никакого отдельного неизменяемого хранилища
первого входа здесь нет. Если повторять нулевые входы с этим весом,
состояние будет приближаться к нулю.

### Предсказание следующего токена

В языковой модели к рекуррентной ячейке добавляется выходной классификатор.
Для строки «кот кот спит» обучающие вход и цель имеют вид

```text
вход:  <BOS>   кот    кот    спит
цель:   кот   кот   спит   <EOS>
```

На первой позиции сеть видит только начало текста. На второй видит
состояние после `<BOS>` и текущий «кот». На третьей её состояние уже
зависит от `<BOS>`, «кот», «кот». Цель на один токен опережает вход:
текущий токен не должен одновременно быть ответом для своего же выхода.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rnn-bptt-2026-10-21/d2l-rnn-train.svg]]

*Символьная языковая модель D2L: вход machin, цель achine. Например,
на третьем шаге прочитано mac и предсказывается h. Вход следующего
шага при обучении берётся из исходной строки, а не из предсказания сети.
Источник: Zhang et al.,
[D2L, §9.4.3](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html),
CC BY-SA 4.0. Оригинальный SVG.*

Если правильный следующий токен равен $y_t=z_{t+1}$, потеря на шаге
равна $\ell_t=-\ln p_t[y_t]$. Пока выводим градиенты, будем использовать
сумму потерь $L=\sum_t\ell_t$. Для обучения на корпусе удобно делить
её на число целевых токенов $N$. Все градиенты тогда делятся на то же $N$.
При натуральном логарифме perplexity равна $\exp(L/N)$.

Подача правильной истории на каждом шаге называется teacher forcing.
Она позволяет обучать модель, даже если её собственные предсказания пока
плохие: не нужно строить обучающий префикс из этих предсказаний.
При генерации исходного продолжения уже нет. Выбранный моделью токен
становится входом следующего шага. Поэтому ошибка генерации изменяет
дальнейшее состояние и последующие предсказания.
Постановка с этим сдвигом разобрана также в
[SLP3, §14.2.1–14.2.2](https://web.stanford.edu/~jurafsky/slp3/14.pdf).

### Полная маленькая модель

В коде объекты пакета хранятся строками. Поэтому формуле $W_{xh}x_t$
соответствует `x @ W_xh.T`. Вход `ids` имеет форму `[B, T]`, состояние
имеет форму `[B, m]`, логиты имеют форму `[B, T, V]`. Цикл проходит по
времени; объекты пакета обрабатываются одновременно.

Следующий пример содержит embeddings, рекуррентную ячейку и выходной слой,
а также обучение на тех же трёх текстах, что использовались
в главе о языковой модели. Чтобы не смешивать рекуррентность с padding,
тексты обрабатываются по одному; их потери суммируются и делятся
на общее число целей. Это один шаг оптимизатора по всему маленькому корпусу.

<details>
<summary>RNN на 96 параметров: полный код обучения и продолжения</summary>

```python
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(42)

class RNNLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Parameter(.1 * torch.randn(8, 3, dtype=torch.float64))
        self.W_xh = nn.Parameter(.1 * torch.randn(4, 3, dtype=torch.float64))
        self.W_hh = nn.Parameter(.1 * torch.randn(4, 4, dtype=torch.float64))
        self.b_h = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.W_hy = nn.Parameter(.1 * torch.randn(8, 4, dtype=torch.float64))
        self.b_y = nn.Parameter(torch.zeros(8, dtype=torch.float64))

    def forward(self, ids, h=None):
        B, T = ids.shape
        if h is None:
            h = self.E.new_zeros(B, 4)
        states = []
        for t in range(T):
            x = self.E[ids[:, t]]
            h = torch.tanh(x @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            states.append(h)
        H = torch.stack(states, dim=1)
        return H @ self.W_hy.T + self.b_y, h

vocab = ["<PAD>", "<UNK>", "кот", "спит", "пёс", "бежит", "<BOS>", "<EOS>"]
docs = [
    torch.tensor([[6, 2, 2, 3, 7]]),  # кот кот спит
    torch.tensor([[6, 4, 3, 7]]),     # пёс спит
    torch.tensor([[6, 2, 3, 5, 7]]),  # кот спит бежит
]
model = RNNLM()
optimizer = torch.optim.Adam(model.parameters(), lr=.03)
N = sum(z.shape[1] - 1 for z in docs)

def objective():
    total = model.E.new_zeros(())
    for z in docs:
        logits, _ = model(z[:, :-1])  # h=None: новый независимый текст
        total = total + F.cross_entropy(
            logits.reshape(-1, 8), z[:, 1:].reshape(-1), reduction="sum"
        )
    return total / N

initial = objective().item()
for step in range(1500):
    optimizer.zero_grad(set_to_none=True)
    loss = objective()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()

print("parameters:", sum(p.numel() for p in model.parameters()))
print("N:", N, "NLL:", initial, "->", objective().item())

@torch.no_grad()
def continue_text(prefix, max_new=8):
    ids = torch.tensor([[6] + prefix])
    logits, h = model(ids)
    result = []
    for _ in range(max_new):
        token = int(logits[0, -1].argmax())  # жадный выбор, не sampling
        result.append(token)
        if token == 7:
            break
        logits, h = model(torch.tensor([[token]]), h)
    return [vocab[i] for i in result]

print("кот спит ->", continue_text([2, 3]))
print("кот кот спит ->", continue_text([2, 2, 3]))
```

</details>

У примера 11 целей. В проверенном запуске средняя NLL уменьшилась
с $2.078633$ до $0.299976$. Это проверка обучения маленькой сети
на трёх строках, не оценка качества на новых текстах.

Околонулевая потеря здесь недостижима даже у более мощной модели:
один и тот же полный префикс встречается с разными продолжениями.
После `<BOS>` в двух текстах идёт «кот», в одном «пёс». Оптимальны
вероятности $2/3$ и $1/3$. После `<BOS>` «кот» в одном тексте идёт
«кот», в другом «спит», поэтому оптимальны $1/2$ и $1/2$.
Остальные полные префиксы имеют единственную цель. Нижняя граница
средней потери равна

$$
\begin{aligned}
L_{\min}&=\frac{3H(2/3,1/3)+2\ln2}{11}\\
&\approx0.299622,
\end{aligned}
$$

где $H(q)=-\sum_jq_j\ln q_j$. Обученная сеть приблизилась к этой границе.

При этом префиксы `<BOS>` «кот кот спит» и `<BOS>` «кот спит» она различила.
Первый заканчивается `<EOS>`, второй продолжается «бежит».
У оконной модели с окном два оба случая давали один вход «кот спит».
В этом корпусе рекуррентное состояние позволило сохранить сведения
о предшествующем «кот». Из такого опыта нельзя заключить, что та же
ячейка сохранит произвольный факт через сотню слов.

## BPTT: backpropagation по развёрнутой сети

Для обучения нужно вычислить производную потери по общим весам.
Состояние $h_t$ влияет на выход текущего шага и на следующее состояние.
Через это следующее состояние оно влияет и на более поздние выходы.
Поэтому при обратном проходе в $h_t$ складываются производные
от обеих ветвей.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l-recurrent/rnn-bptt.svg]]

*Вычислительный граф D2L для трёх шагов. Снизу вверх идут состояния
$h_1,h_2,h_3$, справа от них выходы и общая потеря $L$.
Кружки обозначают вычисления; синие прямоугольники внизу — общие
параметры $W_{hx},W_{hh},W_{qh}$. Все стрелки показывают зависимости
прямого расчёта. Штрихи у $x_2$ и $h_2$ сокращают изображение стрелок,
а не обозначают отдельный обратный проход.
В исходном выводе D2L ячейка линейная и без смещений; ниже для нашей
tanh-ячейки дополнительно учитывается производная активации.
$W_{hx}$ и $W_{qh}$ источника соответствуют нашим $W_{xh}$ и $W_{hy}$.
Источник: Zhang et al.,
[D2L, §9.7.2](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html),
CC BY-SA 4.0. Оригинальный SVG.*

При развёртке одна матрица $W_{hh}$ участвует во всех шагах.
Если мысленно сделать временные копии $W_{hh}^{(1)},\ldots,W_{hh}^{(T)}$,
можно вычислить производную по каждой копии обычным backpropagation.
Поскольку в настоящей модели все копии являются одним параметром,
их вклады складываются. Это обратное распространение во времени,
backpropagation through time, BPTT.
Развёртка и общий параметр разобраны в
[SLP3, §14.1.2](https://web.stanford.edu/~jurafsky/slp3/14.pdf).

### Обратный проход на тех же трёх шагах

Для одномерного примера выберем одну потерю на конце
$L=\tfrac12h_3^2\approx0.214128$. Цель этого расчёта — получить нулевое
финальное состояние; это отдельная учебная задача, а не
cross-entropy языковой модели.

Производная потери по $h_3$ равна $h_3$. Производная $\tanh(a_t)$
равна $1-h_t^2$. Обозначим $\delta_t=dL/da_t$. Тогда

$$
\delta_3=h_3(1-h_3^2)\approx0.374156.
$$

Чтобы получить производную по $h_2$, нужно пройти через множитель $0.8$
в формуле $a_3=0.5x_3+0.8h_2$. Затем производная по $a_2$ добавляет
производную второй активации:

$$
\delta_2=0.8\delta_3(1-h_2^2)\approx0.261873,
$$

$$
\delta_1=0.8\delta_2(1-h_1^2)\approx0.164760.
$$

У каждого использования рекуррентного веса $w$ локальная производная
$a_t=0.5x_t+wh_{t-1}$ по $w$ равна $h_{t-1}$. Поэтому

$$
\frac{dL}{dw}=\sum_{t=1}^3\delta_th_{t-1}.
$$

| Использование веса | Состояние на входе | Вклад в $dL/dw$ |
|---|---:|---:|
| На шаге 1 | $h_0=0$ | 0 |
| На шаге 2 | $h_1=0.462117$ | 0.121016 |
| На шаге 3 | $h_2=0.353724$ | 0.132348 |
| Сумма | | 0.253364 |

Нулевой первый вклад объясняется именно $h_0=0$. Первый вход всё равно
повлиял на $h_1$, затем на $h_2$ и на последнюю потерю.
Если вместо рекуррентного веса дифференцировать входной вес $v=0.5$,
получится $\sum_t\delta_tx_t=\delta_1+\delta_3\approx0.538916$.
Для общего смещения производная равнялась бы $\sum_t\delta_t$.

Проверим ручной обратный проход и один шаг SGD:

```python
import torch

dtype = torch.float64
v = torch.tensor(.5, dtype=dtype, requires_grad=True)
w = torch.tensor(.8, dtype=dtype, requires_grad=True)
xs = [1., 0., 1.]
h = torch.zeros((), dtype=dtype)
states = [h]
for x in xs:
    h = torch.tanh(v*x + w*h)
    states.append(h)
loss = .5*h*h
loss.backward()

hs = [float(s.detach()) for s in states]
delta = [0., 0., hs[3]*(1-hs[3]**2)]
for t in [1, 0]:
    delta[t] = .8 * delta[t+1] * (1-hs[t+1]**2)
manual_w = sum(delta[t]*hs[t] for t in range(3))
manual_v = sum(delta[t]*xs[t] for t in range(3))
assert abs(w.grad.item()-manual_w) < 1e-12
assert abs(v.grad.item()-manual_v) < 1e-12
print("states:", hs)
print("delta:", delta)
print("dL/dw, dL/dv:", manual_w, manual_v)

before = loss.item()
with torch.no_grad():
    w -= .01*w.grad
    v -= .01*v.grad
    h = torch.zeros((), dtype=dtype)
    for x in xs:
        h = torch.tanh(v*x + w*h)
    after = (.5*h*h).item()
assert after < before
print("loss after SGD:", before, "->", after)
```

### Общая формула для языковой модели

Введём $e_t=p_t-\operatorname{onehot}(y_t)$, производную суммы
cross-entropy по логитам $o_t$. Если потери усредняются, здесь также
нужно деление на число целей. Пусть $g_t=dL/dh_t$ и
$\delta_t=dL/da_t$ — столбцы длины $m$. При $\delta_{T+1}=0$

$$
g_t=W_{hy}^{\top}e_t+W_{hh}^{\top}\delta_{t+1},
$$

$$
\delta_t=g_t\odot(1-h_t^2).
$$

Первое слагаемое в $g_t$ приходит от текущего выхода, второе от будущих
потерь. Поэтому при потере только на последнем шаге локальное слагаемое
на предыдущих шагах нулевое, но производная через будущее сохраняется.
В языковой модели локальное слагаемое обычно есть на каждом шаге.

После обратной рекурсии градиенты параметров получаются суммированием:

$$
\begin{aligned}
\frac{dL}{dW_{hh}}&=\sum_t\delta_th_{t-1}^{\top},\\
\frac{dL}{dW_{xh}}&=\sum_t\delta_tx_t^{\top}.
\end{aligned}
$$

$$
\frac{dL}{db_h}=\sum_t\delta_t,
\qquad
\frac{dL}{dx_t}=W_{xh}^{\top}\delta_t,
$$

$$
\frac{dL}{dW_{hy}}=\sum_te_th_t^{\top},
\qquad
\frac{dL}{db_y}=\sum_te_t.
$$

Например, $\delta_th_{t-1}^{\top}$ имеет форму $m\times m$.
В этом локальном произведении $h_{t-1}$ считают фиксированным:
его зависимость от ранних использований параметров уже учтена
в обратной рекурсии. Дополнительное дифференцирование этой зависимости
здесь посчитало бы один путь дважды.

Производная по $x_t$ прибавляется в строку $E[z_t]$. Если один токен
встречался несколько раз, вклады в его embedding суммируются.
Так градиент проходит от ошибки следующего слова через состояние
к embeddings всех повлиявших позиций. Обычный autograd выполняет
эту же процедуру по развёрнутому графу.

### Почему далёкие зависимости трудно обучать

Даже при полном BPTT производная может стать очень маленькой задолго
до начала текста. Возьмём скалярную ячейку
$h_t=\tanh(wh_{t-1})$ и начальное состояние $h_0=0$.
Все состояния равны нулю. В нуле производная $\tanh$ равна единице,
поэтому чувствительность последнего состояния к начальному равна $w^T$.

| Вес $w$ | Через 20 шагов | Через 100 шагов |
|---:|---:|---:|
| $0.8$ | $1.15\cdot10^{-2}$ | $2.04\cdot10^{-10}$ |
| $1.0$ | $1$ | $1$ |
| $1.1$ | $6.73$ | $1.38\cdot10^4$ |

В первом случае изменение раннего состояния почти не изменяет позднее.
Градиент поздней потери плохо обучает операции в начале последовательности.
В третьем случае малое изменение раннего состояния может сильно
изменить позднее: обратный сигнал растёт. Значения самих состояний
в этом примере нулевые, так что нормальные по величине активации
не гарантируют нормальные производные.

```python
import torch

for w in [.8, 1., 1.1]:
    h0 = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    h = h0
    for _ in range(100):
        h = torch.tanh(w*h)
    gradient, = torch.autograd.grad(h, h0)
    assert abs(gradient.item()-w**100) <= 1e-10*max(1., abs(w**100))
    print(w, "h100:", h.item(), "dh100/dh0:", gradient.item())
```

В многомерной сети локальная производная является матрицей Якоби:

$$
J_t=\frac{\partial h_t}{\partial h_{t-1}}
=\operatorname{diag}(1-h_t^2)W_{hh}.
$$

Для состояний-столбцов более поздняя матрица стоит слева:

$$
\frac{\partial h_t}{\partial h_k}
=J_tJ_{t-1}\cdots J_{k+1}.
$$

В обратном проходе используются транспонированные матрицы в обратном
порядке. По субмультипликативности операторной нормы

$$
\|J_t\cdots J_{k+1}\|_2
\le\prod_{j=k+1}^{t}\|J_j\|_2.
$$

Если каждая норма не больше некоторого $\rho<1$, произведение
ограничено сверху $\rho^{t-k}$ и затухает с расстоянием.
Обратный вывод неверен: нормы отдельных множителей больше единицы
не гарантируют роста произведения или конкретного вектора.
Например, матрицы $\operatorname{diag}(2,0)$ и
$\operatorname{diag}(0,2)$ имеют норму 2, но их произведение нулевое.

Производные активации тоже участвуют в произведении. При $a=3$
получаем $1-\tanh^2(3)\approx0.009866$. Даже множитель веса $1.1$
даст локальную производную около $0.010853$, а не рост.
Поэтому одной нормы или собственных значений $W_{hh}$ недостаточно,
чтобы описать все траектории нелинейной сети.

[CS224N Notes 5, §2.3](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)
выводит эту цепочку производных для далёких слов.
У простой RNN одинаковый переход многократно применяется ко всем
шагам; параметрам приходится одновременно учиться обрабатывать новый
вход и сохранять полезные сведения через эту цепочку.

## Усечение и ограничение нормы

Полный BPTT по последовательности длины $T$ сохраняет промежуточные
активации всех шагов. Для скрытых состояний одного слоя это порядка
$BTm$ чисел; дополнительно нужны активации выходного слоя и данные
для обратных операций. В длинном непрерывном потоке хранить весь
граф с начала потока обычно непрактично.

### Передать состояние, отсоединить историю

При усечённом BPTT, truncated BPTT, поток делят на фрагменты.
Численное значение конечного состояния одного фрагмента передают
в следующий. На границе отсоединяют вычислительную историю этого
состояния: следующая потеря больше не дифференцирует расчёты
прошлого фрагмента.

В нашем примере можно после второго входа отсоединить $h_2$,
не изменив его значение. Третий шаг всё ещё выдаёт $h_3=0.654413$.
Но при производной по $w$ теперь считается только последнее
использование веса:

$$
\left.\frac{dL}{dw}\right|_{\text{detach }h_2}
=\delta_3h_2\approx0.132348.
$$

Полный градиент был $0.253364$. Потерянный вклад $0.121016$
приходил через обучение второго шага. Для этой последней потери
граница отключила именно этот путь; другие потери прошлого
фрагмента могли обучать те же веса раньше.

```python
import torch

def run(mode):
    w = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    h = torch.zeros((), dtype=torch.float64)
    for t, x in enumerate([1., 0., 1.]):
        if t == 2 and mode == "detach":
            h = h.detach()
        elif t == 2 and mode == "reset":
            h = torch.zeros_like(h)
        h = torch.tanh(.5*x + w*h)
    loss = .5*h*h
    loss.backward()
    return h.item(), loss.item(), w.grad.item()

full = run("full")
cut = run("detach")
reset = run("reset")
assert abs(full[0]-cut[0]) < 1e-12
assert abs(full[2]-.2533642027283495) < 1e-12
assert abs(cut[2]-.13234801386407055) < 1e-12
assert abs(reset[0]-full[0]) > .1
print("full:", full)
print("detach:", cut)
print("reset:", reset)
```

`detach()` не запрещает обучение $W_{hh}$ в новом фрагменте:
третий шаг по-прежнему использует этот обучаемый вес.
Он обрывает зависимость входящего состояния от прежних вычислений.
Сброс в ноль меняет и состояние, и предсказание, поэтому это другая операция.

При независимых текстах состояние сбрасывают, иначе первый токен
нового документа получит информацию из предыдущего. При непрерывном
потоке состояние можно переносить. Если поток разбит на пакет,
каждая строка пакета должна продолжать свой прежний поток.
Нельзя после перемешивания строк без согласования переносить
на них состояния старого пакета.

### Длина фрагмента и длина зависимости

В неперекрывающихся фрагментах длины $K$ потеря в конце фрагмента
может обучать до $K$ рекуррентных переходов. Потеря в начале того же
фрагмента видит более короткую обратную цепочку. За границу
фрагмента градиент не проходит, даже если нужный токен был совсем
рядом, но в предыдущем фрагменте.

При этом численное состояние может зависеть от гораздо более старых
входов. Поэтому $K$ ограничивает длину обратной цепочки данного
обновления, а не вводит жёсткое окно из $K$ токенов в прямом проходе.
Уменьшение $K$ экономит память, но меняет градиент обучения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rnn-bptt-2026-10-21/d2l-truncated-bptt.svg]]

*Три стратегии D2L на тексте The Time Machine: сверху случайные длины
усечения, посередине фиксированные фрагменты, снизу полный расчёт.
Горизонтальные стрелки обозначают протяжённость фрагментов;
разрывы не означают обязательный сброс численного состояния.
Источник: Zhang et al.,
[D2L, §9.7.1](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html),
CC BY-SA 4.0. Оригинальный SVG.*

Случайное усечение в D2L сопровождается поправкой весов, чтобы
оценка производной была правильной в среднем. Например, некоторый
обратный вклад можно с вероятностью $1/2$ отбросить, а с вероятностью
$1/2$ умножить на 2. Его ожидание сохранится. Для цепочки нужны
соответствующие поправки на каждом случайном продолжении.
Редкие длинные пути тогда получают большой вес, что увеличивает
разброс оценки. Простое случайное разбиение без этих поправок
несмещённого градиента не даёт. Это отдельная стратегия, не действие
обычного `detach()` на фиксированных границах.

### Ограничение нормы градиента

Пусть после `backward()` два параметра получили градиенты 3 и 4.
Общая евклидова норма равна 5. Если порог равен 1,
нужно умножить оба значения на $1/5$: получатся 0.6 и 0.8.
Направление сохранится, а норма станет равной 1.
Независимое обрезание каждой координаты до 1 дало бы
$(1,1)$ с другой нормой и другим направлением.

Для вектора $g$, собранного из градиентов всех обучаемых параметров,
ограничение нормы записывается как

$$
g_{\text{clip}}=
\begin{cases}
g,&\|g\|_2\le c,\\
\dfrac{c}{\|g\|_2}g,&\|g\|_2>c.
\end{cases}
$$

Нулевой градиент попадает в первую ветвь; деления на ноль не требуется.
Для обычного SGD без других слагаемых длина шага после такого ограничения
не больше $\eta c$. Для Adam это не граница длины обновления параметров:
он дополнительно преобразует градиент с помощью накопленных моментов.

Вызов `clip_grad_norm_` ставят после `backward()` и перед
`optimizer.step()`. При mixed precision сначала снимают масштабирование
градиента, иначе порог относится к искусственно увеличенным значениям.
Бесконечности и NaN нужно обнаруживать отдельно, а не пытаться
исправить умножением на коэффициент ограничения.

Следующий самостоятельный пример обучает скалярную ячейку на потоке
из шести чисел. Цели заданы явно. Он показывает границы графа и порядок
операций; это регрессионная задача, не языковая модель.

```python
import torch
from torch import nn

dtype = torch.float64
v = nn.Parameter(torch.tensor(.5, dtype=dtype))
w = nn.Parameter(torch.tensor(.8, dtype=dtype))
params = [v, w]
optimizer = torch.optim.SGD(params, lr=.01)
xs = torch.tensor([1., 0., 1., 0., 1., 0.], dtype=dtype)
targets = torch.tensor([0., 1., 0., 1., 0., 1.], dtype=dtype)
K = 3
h = torch.zeros((), dtype=dtype)

for start in range(0, len(xs), K):
    h = h.detach()  # значение из прошлого фрагмента, без его графа
    optimizer.zero_grad(set_to_none=True)
    outputs = []
    for x in xs[start:start+K]:
        h = torch.tanh(v*x + w*h)
        outputs.append(h)
    prediction = torch.stack(outputs)
    target = targets[start:start+K]
    loss = .5*(prediction-target).square().mean()
    loss.backward()
    norm_before = nn.utils.clip_grad_norm_(
        params, max_norm=1., error_if_nonfinite=True
    )
    optimizer.step()
    print("start:", start, "loss:", loss.item(),
          "norm before clipping:", norm_before.item())
```

Параметры обновляются после каждого фрагмента. Переданное состояние
вычислено ещё прежними весами; такой перенос состояния является
частью показанного алгоритма усечённого обучения.
Для сравнения полного и усечённого градиента при одинаковых параметрах
нужно отложить `step()`, как в предыдущем примере.

Clipping уменьшает большой градиент, но не восстанавливает затухший
и не возвращает пути, обрезанные при усечении.
[В D2L, §9.5.3](https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html#gradient-clipping)
норма также считается по всем параметрам вместе.

<!-- rnn-execution 2026-10-26 -->
## Parallel work inside a recurrent network

For the scalar example above, the three input contributions are already
known before the first state is computed: $0.5x_t$ gives
$[0.5,0,0.5]$. Computing those three products together does not require
$h_1$ or $h_2$. The recurrent contributions $0.8h_{t-1}$ still have to
wait for the preceding state. Separating these two parts exposes work
that a literal one-token-at-a-time implementation leaves inside its loop.

For a batch stored as rows, let $X$ have shape $[B,T,d]$. Flattening its
first two axes gives $X_{\mathrm{flat}}$ with shape $[BT,d]$. All input
projections for the first layer can be computed as one matrix product:

$$
U_{\mathrm{flat}}=X_{\mathrm{flat}}W_{xh}^{\top},
\qquad U_{\mathrm{flat}}\in\mathbb R^{BT\times m}.
$$

Reshape $U_{\mathrm{flat}}$ to $[B,T,m]$, then use its slices in the
ordinary recurrence:

$$
H_t=\tanh(U_t+H_{t-1}W_{hh}^{\top}+b_h).
$$

The same $X_t$ still reaches the same state and loss. Preparing a
projection for a later position does not feed it into an earlier state,
so this rearrangement preserves causality. It does not make the nonlinear
state transitions independent.

### Full sequence, chunks and a live stream

The entire training input is available under teacher forcing. An encoder
can also receive a complete source sequence. In either case the first
layer may project the available inputs together. During autoregressive
generation the next token is not available until it has been selected;
future input projections cannot be prepared in the same way. A known
prompt can still be processed before that generation loop begins.

The full projection buffer contains $BTm$ numbers. If $B=32$, $T=1000$
and $m=512$, it contains 16,384,000 numbers: 62.5 MiB in FP32. Projecting
only $F=20$ positions at a time reduces this particular buffer to
$BFm=327{,}680$ numbers, or 1.25 MiB. It does not remove the states and
other activations required by full BPTT.

With fixed weights, processing these projection chunks without detaching
the state preserves the full gradient. The projection chunk size $F$
and the truncated-BPTT length $K$ control different operations. Changing
$F$ only changes when the input products are evaluated; inserting
`detach()` at a boundary changes which paths the loss differentiates.

The following CPU example compares a literal recurrence with projections
prepared for one, two, three or all five positions. It checks every
state and the gradients of the inputs, both weight matrices, the bias
and the initial state. All versions retain the output states for the
comparison; this is not a constant-memory training implementation.

```python
import torch

torch.manual_seed(17)
dtype = torch.float64
B, T, d, m = 2, 5, 3, 4
X = torch.randn(B, T, d, dtype=dtype, requires_grad=True)
W_xh = (.2 * torch.randn(m, d, dtype=dtype)).requires_grad_()
W_hh = (.2 * torch.randn(m, m, dtype=dtype)).requires_grad_()
b = torch.randn(m, dtype=dtype, requires_grad=True)
h0 = torch.randn(B, m, dtype=dtype, requires_grad=True)
variables = (X, W_xh, W_hh, b, h0)

def recurrent(X, W_xh, W_hh, b, h0, chunk_size=None):
    h = h0
    states = []
    if chunk_size is None:
        for t in range(X.shape[1]):
            h = torch.tanh(X[:, t] @ W_xh.T + h @ W_hh.T + b)
            states.append(h)
    else:
        for start in range(0, X.shape[1], chunk_size):
            inputs = X[:, start:start + chunk_size]
            U = inputs @ W_xh.T  # independent of every hidden state
            for j in range(U.shape[1]):
                h = torch.tanh(U[:, j] + h @ W_hh.T + b)
                states.append(h)  # no detach at a projection boundary
    return torch.stack(states, dim=1)

reference = recurrent(*variables)
reference_grads = torch.autograd.grad(reference.square().sum(), variables)
for F in [1, 2, 3, T]:
    result = recurrent(*variables, chunk_size=F)
    grads = torch.autograd.grad(result.square().sum(), variables)
    torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)
    for got, expected in zip(grads, reference_grads):
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)
    print("projection chunk:", F, "states and all gradients match")
```

The number of multiply–accumulates in the input projection remains
$BTdm$. Combining positions changes matrix dimensions and the number of
calls, not the mathematical amount of work. A larger matrix operation
may use an accelerator better, but preparing a large buffer can delay
the first recurrent step. The best chunk size depends on the implementation
and workload; this correctness example makes no timing claim.

Jeremy Appleyard describes combining input GEMMs and overlapping layers
in [Optimizing Recurrent Neural Networks in cuDNN 5](https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/),
2016. NVIDIA's [Recurrent Layers User's Guide, §3](https://docs.nvidia.com/deeplearning/performance/dl-performance-recurrent/index.html#gemm-dim)
also separates the recurrent and layer-to-layer products. Its performance
examples use A100, CUDA 11.2 and cuDNN 8.1; their timings and selection
thresholds are not measurements of this example or promises for another GPU.

### Several layers form a dependency grid

An RNN can be deep along both time and layers. D2L describes the
layer-to-layer construction as follows:

<!-- source-excerpt: d2l-stacked-inputs -->
> Given a sequence of length $T$, the first RNN produces
> a sequence of outputs, also of length $T$.
> These, in turn, constitute the inputs to the next RNN layer.
<!-- /source-excerpt -->

*Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola,
[Dive into Deep Learning, §10.3: Deep Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-modern/deep-rnn.html).
Original English excerpt, unchanged apart from line wrapping; CC BY-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rnn-execution-2026-10-26/d2l-deep-rnn.svg]]

*The complete original D2L figure, CC BY-SA 4.0. Horizontal arrows carry
the previous state within one layer; vertical arrows carry the current
output of the layer below. Superscripts label layers and subscripts label
time. The ellipses omit intermediate layers or positions, not dependencies.
Source: Zhang et al.,
[D2L, Fig. 10.3.1](https://d2l.ai/chapter_recurrent-modern/deep-rnn.html#fig-deep-rnn).
No crop, redrawing or relabeling.*

Set $H_t^{(0)}=X_t$. Cell $(\ell,t)$ needs
$H_{t-1}^{(\ell)}$ from its left and $H_t^{(\ell-1)}$ from below:

$$
\begin{aligned}
H_t^{(\ell)}=\tanh\big(&H_t^{(\ell-1)}(W_{xh}^{(\ell)})^{\top}\\
&+H_{t-1}^{(\ell)}(W_{hh}^{(\ell)})^{\top}+b_h^{(\ell)}\big).
\end{aligned}
$$

Each layer has its own parameters. For the first layer,
$W_{xh}^{(1)}$ has shape $[m,d]$; with equal hidden widths the later
input matrices have shape $[m,m]$. Stacking layers does not mean sharing
one input matrix across the stack.

For three layers and four positions, assume that all inputs and initial
states are available. If every cell took one equal-duration slot and
independent cells could run concurrently, the dependency grid would allow
the following schedule:

| Slot | Ready cells $(\ell,t)$ |
|---:|---|
| 1 | $(1,1)$ |
| 2 | $(1,2)$ and $(2,1)$ |
| 3 | $(1,3)$, $(2,2)$ and $(3,1)$ |
| 4 | $(1,4)$, $(2,3)$ and $(3,2)$ |
| 5 | $(2,4)$ and $(3,3)$ |
| 6 | $(3,4)$ |

For example, $(2,2)$ waits for $(2,1)$ and $(1,2)$; both finish in
slot 2. It can run in slot 3 alongside $(1,3)$ and $(3,1)$.
Cells on one such diagonal have no dependency on each other. There are
still 12 cell evaluations, but the longest chain has $L+T-1=6$ cells.
These are ideal dependency slots, not a measured twofold GPU speedup:
the cells may compete for the same hardware, and their costs need not be
equal. A layer-by-layer implementation is also valid; it simply chooses
a different order for independent work.

Only the first layer knows all its inputs at the start. An upper layer
receives its input sequence as the layer below produces states. It can
project an available prefix or chunk; it cannot project lower-layer
states that have not been computed yet.

### Weight gradients can be reduced across time

The BPTT recurrence above computes $\delta_t=dL/da_t$ in reverse time.
Once those values are available, their contributions to a shared weight
are independent terms of a sum. Stack the row-batched $\delta_t$ and
$X_t$ in the same batch/time order as $\Delta_{\mathrm{flat}}$ and
$X_{\mathrm{flat}}$. Then

$$
\frac{dL}{dW_{xh}}
=\Delta_{\mathrm{flat}}^{\top}X_{\mathrm{flat}}.
$$

The shapes are $[m,BT]\,[BT,d]=[m,d]$. For the recurrent weights,
stack $H_0,H_1,\ldots,H_{T-1}$ in the matching order:

$$
\frac{dL}{dW_{hh}}
=\Delta_{\mathrm{flat}}^{\top}H_{\mathrm{prev,flat}}.
$$

This product sums the same outer products as the BPTT formula. The
reverse-time dependency is needed to obtain $\delta_t$; it does not
require accumulating the final weight gradient as $T$ separate matrix
operations. Faster matrix operations, fewer launches and less memory
traffic can therefore reduce RNN runtime without removing its recurrent
dependency. How much they reduce it has to be measured for the actual
model, batch and hardware.

<!-- /rnn-execution -->

## Что RNN умеет и где ломается

После обработки префикса для следующего шага RNN нужны текущее состояние
и новый токен. Хранить все прежние состояния для самого прямого
потокового вычисления не требуется. При обучении с полным BPTT они,
напротив, нужны обратному проходу. Поэтому постоянный размер состояния
при генерации совместим с растущей памятью на обучение длинной строки.

Однослойная ячейка требует на токен порядка $md+m^2$ операций
до выходного слоя. Полный классификатор слов добавляет порядка $Vm$.
Для $T$ шагов эта работа повторяется $T$ раз с теми же весами.
Матричные операции внутри шага и разные объекты пакета можно
распараллелить. Но обычный переход с $\tanh$ ждёт $h_{t-1}$,
поэтому все состояния нельзя независимо вычислить одной операцией
по оси времени.

Для классификации текста выход можно читать из конечного состояния,
а для разметки токенов вычислять отдельный выход на каждой позиции.
В обоих случаях положение потерь меняется, но суммирование градиентов
общих весов остаётся тем же.

При разной длине текстов нужно учитывать padding. Исключение PAD
из функции потерь убирает его прямой вклад в обучение, но само по себе
не запрещает рекуррентной ячейке обновлять состояние на PAD.
Если для классификации взять состояние после всего дополненного
пакета, оно может отличаться от состояния на последнем настоящем токене.
Даже при нулевом embedding PAD наша скалярная ячейка после $h_3=0.654413$
выдала бы $\tanh(0.8h_3)\approx0.480420$. Нулевая потеря на этом
добавленном шаге не вернёт состояние к прежнему значению.
Можно выбирать состояние по фактической длине, сохранять прежнее
состояние на неактивных шагах или применять packed sequences.
Для независимых текстов требуется также отдельный $h_0$;
маска потерь его не заменяет.

Остаются три разных ограничения. Вектор конечного размера может
потерять нужную информацию о префиксе. Длинная цепочка производных
может мешать обучению даже тогда, когда представление такой зависимости
в принципе возможно. Последовательный расчёт ограничивает параллелизм.
Усечение помогает с памятью обучения, clipping с чрезмерной нормой
градиента; ни один из них не устраняет все три ограничения.

В [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU|LSTM и GRU]]
появятся обучаемые вентили, управляющие сохранением и обновлением состояния.
У LSTM отдельный аддитивный переход состояния памяти изменяет путь
распространения производной. Он помогает обучать длительные зависимости,
но не отменяет последовательность шагов и не гарантирует сохранение
любой информации.

## Источники и дальнейшее чтение

- Zhang, Lipton, Li, Smola, [D2L, §9.4: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) — скрытое состояние, общие веса, объединение входов и символьная языковая модель. Три рисунка этого раздела перенесены в оригинале, CC BY-SA 4.0.
- Те же авторы, [§9.5: RNN Implementation from Scratch](https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html) — формы тензоров, последовательный прямой проход, общая норма градиента.
- Те же авторы, [§9.7: Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) — полный градиент, усечение и вычислительный граф. Оба рисунка перенесены в оригинале, CC BY-SA 4.0. В исходном подробном выводе ячейка линейная; в нашей обратной рекурсии сохранена производная $\tanh$.
- Jurafsky, Martin, [Speech and Language Processing, ch. 14, §14.1–14.2](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — развёртка, обучение языковой модели, teacher forcing и weight tying. Проверена редакция от 19 августа 2026 года; нумерация черновика может меняться.
- [Stanford CS224N, Notes 5, §2–2.4](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf) — RNN-языковые модели и произведения производных. В этой главе натуральные логарифмы, поэтому perplexity записывается как $\exp(\text{средняя NLL})$; степень двойки требует логарифмов по основанию 2.
- Elman, [Finding Structure in Time](https://doi.org/10.1016/0364-0213(90)90002-E), 1990 — простая рекуррентная сеть.
- Rumelhart, Hinton, Williams, [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0), 1986 — обратное распространение ошибки.
- [[05 Источники/Courses/Stanford CS230 Cheatsheets/en/cheatsheet-recurrent-neural-networks.pdf|Stanford CS230 — Recurrent Neural Networks Cheatsheet]] — краткие схемы и формулы после полного разбора.

**Назад:** [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram]] ·
**Дальше:** [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU]]
