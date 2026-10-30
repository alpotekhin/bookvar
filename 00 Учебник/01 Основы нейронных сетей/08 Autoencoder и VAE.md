---
title: Autoencoder и variational autoencoder
type: textbook-chapter
status: canonical
last_updated: 2026-10-26
primary_sources:
  - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
  - https://lilianweng.github.io/posts/2018-08-12-vae/
  - https://arxiv.org/abs/1312.6114
  - https://arxiv.org/abs/1606.05908
  - https://github.com/pytorch/examples/tree/main/vae
---

# Autoencoder и variational autoencoder

В классификаторе изображение превращается в вектор признаков, из которого
предсказывается метка. Автоэнкодер использует этот вектор для другой задачи:
по нему нужно восстановить само изображение. Данные для такого обучения
содержат входы, но не требуют меток классов. Целевым ответом служит исходный
объект.

Обычный автоэнкодер обучает два детерминированных преобразования. VAE начинает
с вероятностной модели: предполагается, что скрытая переменная $z$ порождает
наблюдение $x$. Энкодер VAE нужен, чтобы по известному $x$ приближённо
восстановить распределение возможных $z$. Для генерации нового объекта
направление вычислений обратное: сначала выбирают $z$, затем получают $x$
из декодера.

## Обычный autoencoder

Энкодер и декодер задают

$$
z=g_\phi(x),\qquad \hat x=f_\theta(z).
$$

При квадратичной ошибке параметры обучают совместно, минимизируя

$$
L_{\mathrm{rec}}=
\frac1B\sum_{i=1}^B\|x_i-f_\theta(g_\phi(x_i))\|_2^2.
$$

Потеря суммируется по координатам одного объекта и усредняется по $B$ объектам
пакета. Производные проходят от ошибки восстановления через декодер
к энкодеру. Никакой отдельной разметки «правильного кода» $z$ здесь нет.

![[00 Учебник/Assets/Figures/curated/source-first-7-14/weng-autoencoder.png|Энкодер, узкое место и восстановление входа]]

*Вход и восстановление сравниваются непосредственно; код в узком месте
обучается через эту ошибку. Lilian Weng,
[From Autoencoder to Beta-VAE, Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/#autoencoder).*

### Пример с одной скрытой координатой

Пусть данные лежат на прямой: $(-2,-2)$, $(-1,-1)$, $(1,1)$, $(2,2)$.
Двух чисел для их описания не требуется. Энкодер
$z=(x_1+x_2)/2$ возвращает одну координату; декодер $\hat x=(z,z)$
восстанавливает все четыре точки без ошибки. Но точка $(1,-1)$,
не лежащая на этой прямой, превращается в ноль и восстанавливается как $(0,0)$.

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

ae_dtype = torch.float64
line_data = torch.tensor([[-2., -2.], [-1., -1.], [1., 1.], [2., 2.]], dtype=ae_dtype)
projection_encoder = torch.tensor([[.5, .5]], dtype=ae_dtype)
projection_decoder = torch.tensor([[1.], [1.]], dtype=ae_dtype)
line_codes = line_data @ projection_encoder.T
line_reconstruction = line_codes @ projection_decoder.T
torch.testing.assert_close(line_reconstruction, line_data)
off_line = torch.tensor([[1., -1.]], dtype=ae_dtype)
torch.testing.assert_close(off_line @ projection_encoder.T @ projection_decoder.T,
                           torch.zeros_like(off_line))
print(line_codes.flatten())
```

Линейный автоэнкодер с кодом размерности $k$, квадратичной ошибкой и
центрированными данными может восстановить проекцию на главное $k$-мерное
подпространство, как PCA. Однако координаты самого кода не обязаны совпасть
с PCA scores: обратимое преобразование кода можно компенсировать в декодере,
не меняя восстановления. В примере выше код равен $x_1$, тогда как проекция
на единичное направление $(1,1)/\sqrt2$ даёт $\sqrt2x_1$.
Подробнее геометрия разобрана в
[[00 Учебник/01 Классическое машинное обучение/07 PCA, t-SNE и UMAP|главе о PCA]].

### Один шаг обучения

Теперь начнём с весов энкодера $(0{,}1,0{,}1)$, а декодера — $(1,1)$.
Восстановленная точка равна $0{,}2x$, и средняя сумма квадратов ошибок — 3,2.
Производные по каждому весу энкодера равны $-8$, декодера — $-0{,}8$.
После одновременного шага SGD с learning rate 0,05 веса становятся
$(0{,}5,0{,}5)$ и $(1{,}04,1{,}04)$. Теперь $\hat x=1{,}04x$,
а потеря равна 0,008.

```python
linear_encoder = nn.Linear(2, 1, bias=False, dtype=ae_dtype)
linear_decoder = nn.Linear(1, 2, bias=False, dtype=ae_dtype)
with torch.no_grad():
    linear_encoder.weight.fill_(.1)
    linear_decoder.weight.fill_(1.)
linear_optimizer = torch.optim.SGD(
    list(linear_encoder.parameters()) + list(linear_decoder.parameters()), lr=.05)
linear_optimizer.zero_grad()
linear_loss = (linear_decoder(linear_encoder(line_data)) - line_data).square().sum(-1).mean()
linear_loss.backward()
torch.testing.assert_close(linear_encoder.weight.grad, torch.full((1, 2), -8., dtype=ae_dtype))
torch.testing.assert_close(linear_decoder.weight.grad, torch.full((2, 1), -.8, dtype=ae_dtype))
linear_optimizer.step()
updated_linear_loss = (linear_decoder(linear_encoder(line_data)) - line_data).square().sum(-1).mean()
assert abs(linear_loss.item() - 3.2) < 1e-12
assert abs(updated_linear_loss.item() - .008) < 1e-12
print(linear_loss.item(), updated_linear_loss.item())
```

Низкая размерность кода ограничивает линейную модель, но сама по себе не
доказывает полезность нелинейного представления. Мощная сеть может запомнить
конечный обучающий набор. Проверять нужно восстановление на отложенных данных
и пригодность кода для нужной задачи, а не только его размер.

### Повреждённый вход и чистая цель

Denoising autoencoder получает повреждённый объект $\tilde x$, но сравнивает
восстановление с чистым $x$. Например, для $x=(1,1)$ можно занулить первую
координату и подать $(0,1)$. Копирование входа даст ошибку 1; восстановление
$(1,1)$ — ноль. Модель должна использовать зависимость между координатами,
если в данных есть информация для восстановления.

При случайном повреждении один и тот же чистый объект появляется в разных
вариантах. Шум и механизм маскирования выбирают под задачу. Они не гарантируют
устойчивости к произвольным искажениям. Эта постановка отличается от
[[00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация|dropout]]:
там случайная маска меняет внутренние активации обучаемой сети, а здесь
определена задача восстановления чистого объекта из испорченного.

Другие ограничения действуют на код непосредственно. Например, L1-штраф
$\lambda\sum_j|z_j|$ поощряет разреженные активации, а штраф на якобиан энкодера
ограничивает чувствительность к малым изменениям входа. Их цель и коэффициенты
нужно отличать от KL-члена VAE. Подробные варианты разобраны у
[Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/#sparse-autoencoder);
они не превращают обычный автоэнкодер в VAE автоматически.

## Свёрточный автоэнкодер

Для изображения энкодер может использовать свёртки с шагом 2, как в
предыдущей главе. Декодеру нужно увеличить разрешение обратно.
В [UvA, Deep Autoencoders](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html)
Phillip Lippe разбирает сеть для RGB-изображений $32\times32$:
энкодер уменьшает пространственный размер до $4\times4$, затем строит код,
а декодер восстанавливает карту и трижды увеличивает разрешение.

![[00 Учебник/Assets/Figures/curated/autoencoder-2026-10-26/uva-autoencoder.svg|Свёрточное сжатие и восстановление изображения из курса UvA]]

*Общая схема автоэнкодера. Phillip Lippe,
[UvA Deep Learning Tutorials, Building the autoencoder](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html#Building-the-autoencoder),
[MIT](https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/LICENSE.md).
Авторский SVG сохранён без изменений.*

### Transposed convolution не обращает свёртку

Свёртку при фиксированных весах можно записать как матрицу $C$.
Транспонированная свёртка применяет $C^\mathsf T$, а не $C^{-1}$.
Возьмём одномерное ядро $[1,2]$ и вход $(1,2,3)$.
Обычная операция даёт $(5,8)$. Транспонированная операция над этим результатом
даёт $(5,18,16)$, а не исходные числа: вклады от соседних выходов
складываются в перекрывающихся позициях.

```python
conv_matrix = torch.tensor([[1., 2., 0.], [0., 1., 2.]], dtype=ae_dtype)
conv_input = torch.tensor([1., 2., 3.], dtype=ae_dtype)
conv_kernel = torch.tensor([[[1., 2.]]], dtype=ae_dtype)
conv_result = F.conv1d(conv_input[None, None], conv_kernel).flatten()
transpose_result = F.conv_transpose1d(conv_result[None, None], conv_kernel).flatten()
torch.testing.assert_close(conv_result, conv_matrix @ conv_input)
torch.testing.assert_close(transpose_result, conv_matrix.T @ conv_result)
torch.testing.assert_close(transpose_result, torch.tensor([5., 18., 16.], dtype=ae_dtype))
print(conv_result, transpose_result)
```

При шаге больше 1 выходы располагаются дальше друг от друга, и размер
результата может вырасти. Важны эффективное окно и неоднозначность размера:

$$
\begin{aligned}
k_{\mathrm{eff}}&=d(k-1)+1,\\
H_{\mathrm{out}}&=(H_{\mathrm{in}}-1)s-2p\\
&\quad+k_{\mathrm{eff}}+\mathrm{output\_padding}.
\end{aligned}
$$

Для $k=3$, $s=2$, $p=1$, $d=1$ вход длины 2 даёт выход длины 3.
При `output_padding=1` длина становится равна 4. Этот параметр выбирает
одну из допустимых форм, а не просто добавляет нулевой столбец:
дополнительные значения тоже могут быть ненулевыми.

![[00 Учебник/Assets/Figures/curated/autoencoder-2026-10-26/transpose-convolution.gif|Анимация транспонированной свёртки с шагом и дополнением]]

*Синие карты — входы, голубые — выходы. Окна показывают перекрывающиеся вклады
транспонированной свёртки. Vincent Dumoulin и Francesco Visin,
[A guide to convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic#transposed-convolution-animations),
[MIT](https://github.com/vdumoulin/conv_arithmetic/blob/af6f818b0bb396c26da79899554682a8a499101d/LICENSE).
Полная исходная анимация Padding, strides, transposed.*

```python
short_code = torch.tensor([[[1., 2.]]], dtype=ae_dtype)
wide_kernel = torch.tensor([[[1., 2., 1.]]], dtype=ae_dtype)
length3 = F.conv_transpose1d(short_code, wide_kernel, stride=2, padding=1)
length4 = F.conv_transpose1d(short_code, wide_kernel, stride=2, padding=1, output_padding=1)
torch.testing.assert_close(length3.flatten(), torch.tensor([2., 3., 4.], dtype=ae_dtype))
torch.testing.assert_close(length4.flatten(), torch.tensor([2., 3., 4., 2.], dtype=ae_dtype))
print(length3, length4)
```

Формула и ограничения API приведены в
[PyTorch ConvTranspose2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html).
Изменение размера не возвращает утраченную информацию. Декодер обучается
восстанавливать характерные для данных структуры из сохранённого кода.
Для увеличения карты можно использовать и другие операции, например
интерполяцию с последующей обычной свёрткой.

### Исходные encoder и decoder UvA

Ниже сохранены классы курса, включая английские комментарии и docstrings.
В примере выбраны `base_channel_size=8` и `latent_dim=16`.
Энкодер получает $[B,3,32,32]$, перед flatten выдаёт $[B,16,4,4]$,
а затем — $[B,16]$. В декодере линейный слой восстанавливает
$16\cdot4\cdot4=256$ чисел, после чего идут переходы
$4\to8\to16\to32$.

```python
class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

image_encoder = Encoder(3, 8, 16).double()
image_decoder = Decoder(3, 8, 16).double()
image_batch = torch.zeros(2, 3, 32, 32, dtype=ae_dtype)
with torch.no_grad():
    image_codes = image_encoder(image_batch)
    image_reconstruction = image_decoder(image_codes)
assert image_codes.shape == (2, 16)
assert image_reconstruction.shape == image_batch.shape
assert image_reconstruction.abs().max() <= 1
print(image_codes.shape, image_reconstruction.shape)
```

Финальный tanh связан с подготовкой данных курса: пиксели масштабированы
в $[-1,1]$. Это соглашение о диапазоне выхода обычного автоэнкодера,
а не обязательная вероятностная модель наблюдений. Здесь выполнен только
прямой проход случайно инициализированной сети, не обучение на CIFAR10.

### Почему MSE может давать размытое восстановление

Если один код совместим с несколькими исходными изображениями, декодер
с квадратичной ошибкой предпочитает их условное среднее. Например, две
равновероятные цели $(1,0)$ и $(0,1)$ имеют среднее $(0{,}5,0{,}5)$.
Для любого из двух резких ответов средняя сумма квадратов ошибок равна 1,
для среднего — 0,5. Поэтому минимальная MSE не обязана соответствовать
чёткому изображению.

Пиксельная ошибка также чувствительна к сдвигу: перенос яркой точки
в соседний пиксель может дать большую ошибку, чем её ослабление в обоих
пикселях. UvA показывает это отдельным сравнением. Перцептуальная функция
потерь сравнивает признаки выбранной сети вместо исходных пикселей и тем
самым задаёт другой критерий сходства; её выбор не следует из MSE автоматически.

<a id="почему-обычный-latent-space-неудобен-для-генерации"></a>

## Почему реконструкции недостаточно для генерации

Обучение обычного автоэнкодера предъявляет декодеру только те коды,
которые возвращает энкодер на данных. Оно не задаёт известного распределения,
из которого можно независимо выбирать новые коды. Если энкодер размещает
объекты в узкой области пространства, произвольный вектор из $\mathcal N(0,I)$
может попасть далеко от неё.

Сам по себе энкодер не является вероятностным источником новых наблюдений.
Можно отдельно обучить распределение кодов, но это уже дополнительная модель.
Даже хорошая реконструкция отложенных объектов не доказывает, что случайный
код даст правдоподобное изображение. В UvA этот разрыв проверяется отдельно
от восстановления входов.

## Encoder VAE выдаёт распределение

Вероятностная модель VAE задаёт два распределения:

$$
p(z)=\mathcal N(0,I),
\qquad p_\theta(x\mid z).
$$

Для генерации выбирают $z$ из prior, а затем $x$ из условного распределения
декодера. Совместная плотность равна $p_\theta(x,z)=p(z)p_\theta(x\mid z)$.
Вероятность или плотность наблюдения получается суммированием вклада
всех возможных кодов:

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
$$

При нелинейном нейросетевом декодере этот интеграл обычно нельзя вычислить
точно. Отсюда возникает и трудность с истинным posterior:
$p_\theta(z\mid x)=p_\theta(x,z)/p_\theta(x)$.
Энкодер $q_\phi(z\mid x)$ приближает это распределение одной сетью для всех
объектов. Такое переиспользование процедуры вывода называют amortized inference.

![[00 Учебник/Assets/Figures/curated/source-first-7-14/weng-vae-graphical-model.png|Генеративное направление и приближённый вывод в одной схеме VAE]]

*Сплошная стрелка $z\to x$ относится к генеративному декодеру.
Пунктирная $x\to z$ показывает приближённый вывод $q_\phi(z\mid x)$.
Это одна схема с двумя направлениями вычислений. На рисунке использована
общая запись prior $p_\theta(z)$; в разбираемой модели $p(z)=\mathcal N(0,I)$
фиксирован. Lilian Weng,
[From Autoencoder to Beta-VAE, VAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder).*

Часто выбирают диагональное нормальное приближение:

$$
q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))).
$$

Энкодер выдаёт по $d$ средних и log-variance для $d$ латентных координат.
Логарифм дисперсии может принимать любое действительное значение;
после экспоненты дисперсия положительна. Для получения стандартного
отклонения нужна половина log-variance в показателе: $\sigma=\exp(\ell/2)$.
На разных объектах параметры распределения разные; параметры самой сети
$\phi$ общие. Декодер $p_\theta(x\mid z)$ задаётся отдельно.
Он может предсказывать вероятности бинарных признаков, средние нормального
распределения или параметры другого подходящего семейства.

![[00 Учебник/Assets/Figures/curated/vae/vae-blocks.png|Параметры нормального кода и вход декодера VAE]]

*Упрощённая блок-схема: энкодер получает $x$ и предсказывает $\mu,\sigma$,
выбранный код $z$ поступает в декодер. Независимый источник случайности
$\epsilon$ здесь не нарисован. wikipedia + sebastian nguyen,
[VAE blocks](https://commons.wikimedia.org/wiki/File:VAE_blocks.png),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

<a id="elbo-reconstruction-и-regularization-latent-space"></a>

## ELBO и два разных KL

Для одного наблюдения обозначим $q=q_\phi(z\mid x)$. Сначала запишем
ожидание разности совместной log-плотности и log-плотности приближения:

$$
\mathcal L(x)=\mathbb E_q[\log p_\theta(x,z)-\log q].
$$

По правилу Байеса $\log p_\theta(x,z)$ равно сумме
$\log p_\theta(z\mid x)+\log p_\theta(x)$.
Второе слагаемое не зависит от $z$ и выходит из ожидания. Оставшаяся
разность log-плотностей даёт KL между $q$ и истинным posterior:

$$
\log p_\theta(x)=\mathcal L(x)+G(x),
$$

$$
G(x)=D_{\mathrm{KL}}(q\|p_\theta(z\mid x))\ge0.
$$

Поэтому $\mathcal L$ — нижняя граница log-likelihood, evidence lower bound.
Зазор $G$ равен нулю, если приближение совпало с истинным posterior
при текущих параметрах модели.

Теперь разложим совместную плотность другим способом:
$p_\theta(x,z)=p_\theta(x\mid z)p(z)$. Получим

$$
\begin{aligned}
\mathcal L(x)&=R(x)-K(x),\\
R(x)&=\mathbb E_q[\log p_\theta(x\mid z)],\\
K(x)&=D_{\mathrm{KL}}(q\|p(z)).
\end{aligned}
$$

$R$ оценивает, насколько хорошо коды из энкодера объясняют наблюдение.
$K$ штрафует отличие распределения кода от prior. Оптимизация одновременно
обновляет генеративные параметры $\theta$ и параметры вывода $\phi$.
При фиксированном $\theta$ улучшение приближения posterior уменьшает зазор;
при обновлении $\theta$ меняются и сама модель, и её истинный posterior.

Здесь два KL с разными аргументами. $G$ — зазор до истинного posterior;
$K$ — вычисляемая часть обучающей цели, сравнивающая $q$ с prior.
Они не взаимозаменяемы. Вывод и алгоритм приведены у
[Kingma и Welling, §§2.2–2.3](https://arxiv.org/html/1312.6114v11#S2.SS2).

### Модель, где оба KL можно посчитать точно

Рассмотрим одномерную модель $z\sim\mathcal N(0,1)$,
$x\mid z\sim\mathcal N(z,1)$ и наблюдение $x=2$.
Сумма двух независимых нормальных переменных даёт
$p(x)=\mathcal N(0,2)$. По формуле Байеса истинный posterior —
$\mathcal N(1,1/2)$.

Это видно и без готовой формулы нормального posterior. В произведении
$p(z)p(x=2\mid z)$ показатель экспоненты равен
$-[z^2+(2-z)^2]/2$. Выделив полный квадрат,
$z^2+(2-z)^2=2(z-1)^2+2$, получаем плотность, пропорциональную
$\exp[-(z-1)^2]$. Её среднее равно 1, а дисперсия — $1/2$.

Выберем приближение $q=\mathcal N(0{,}5,1)$. Тогда

$$
K=\tfrac12(0{,}5^2+1-1)=0{,}125.
$$

Среднее $(x-z)^2$ равно $(2-0{,}5)^2+1=3{,}25$, поэтому

$$
R=-\tfrac12\log(2\pi)-\tfrac12\,3{,}25
\approx-2{,}543939.
$$

Граница $\mathcal L=R-K\approx-2{,}668939$, тогда как
$\log p(x)\approx-2{,}265512$. Разность равна
$G\approx0{,}403426$. Если подставить в $q$ точный posterior, зазор станет
нулём, но KL к prior останется положительным, примерно 0,596574.

```python
def normal_kl(mean_q, var_q, mean_p=0., var_p=1.):
    return .5 * (math.log(var_p / var_q)
                 + (var_q + (mean_q - mean_p)**2) / var_p - 1)

observed_value = 2.
log_evidence = -.5 * math.log(4 * math.pi) - observed_value**2 / 4
elbo_cases = []
for mean_q, var_q in [(.5, 1.), (1., .5)]:
    reconstruction_logprob = -.5 * math.log(2 * math.pi) - .5 * ((2 - mean_q)**2 + var_q)
    prior_kl = normal_kl(mean_q, var_q)
    elbo = reconstruction_logprob - prior_kl
    posterior_gap = normal_kl(mean_q, var_q, 1., .5)
    assert abs(log_evidence - elbo - posterior_gap) < 1e-12
    elbo_cases.append((elbo, prior_kl, posterior_gap))
assert abs(elbo_cases[0][1] - .125) < 1e-12
assert abs(elbo_cases[1][2]) < 1e-12
assert elbo_cases[1][1] > 0
print(log_evidence, elbo_cases)
```

Это вычислимый учебный пример, а не предположение, что posterior любой VAE
нормален. Для сложного декодера истинное распределение может быть существенно
сложнее диагонального Gaussian. Увеличение выразительности энкодера не отменяет
ограничений выбранного семейства $q$.

Приём связан с
[[00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM|EM для GMM]].
Но E-step GMM вычисляет точный posterior для текущих параметров, а VAE
обучает общую сеть приближённого вывода. Поэтому переносить равенство
границы и log-likelihood из точного E-step на произвольный VAE нельзя.

## Likelihood декодера и ошибка восстановления

Для бинарного $x_i$ декодер может задавать независимые при фиксированном $z$
распределения Бернулли с вероятностями $\pi_i(z)$. Отрицательная log-likelihood —
сумма binary cross-entropy по признакам:

$$
\begin{gathered}
-\log p_\theta(x\mid z)\\
=-\sum_i x_i\log\pi_i\\
\quad-\sum_i(1-x_i)\log(1-\pi_i).
\end{gathered}
$$

Если декодер задаёт нормальное распределение с фиксированной дисперсией
$\tau^2$, отрицательная log-likelihood содержит
$\|x-m_\theta(z)\|^2/(2\tau^2)$ и нормировочную константу.
MSE в таком случае соответствует конкретному соглашению о масштабе.
Когда дисперсия обучается, логарифм дисперсии тоже входит в цель:
его нельзя отбросить как константу.

Для диагонального нормального $q$ и prior $\mathcal N(0,I)$ KL вычисляется
аналитически, без сэмплирования. Если $\ell_j=\log\sigma_j^2$,

$$
K=\frac12\sum_j
\left(\mu_j^2+\exp(\ell_j)-1-\ell_j\right).
$$

В одной координате разность log-плотностей нормального $q$ и стандартного
нормального prior равна
$-\log\sigma-(z-\mu)^2/(2\sigma^2)+z^2/2$.
Под ожиданием по $q$ квадрат отклонения от среднего даёт $\sigma^2$,
а второй момент $z$ — $\mu^2+\sigma^2$.
Поэтому KL равен $(\mu^2+\sigma^2-1-\log\sigma^2)/2$.
Независимые координаты дают сумму в формуле выше.

Параметры $\mu=0$, $\ell=0$ дают prior и KL=0.
Увеличение среднего при той же дисперсии или сильное отклонение дисперсии
увеличивает штраф. Формула выведена в
[Appendix B Auto-Encoding Variational Bayes](https://arxiv.org/html/1312.6114v11#A2).

Для $x=(1,0)$ и logits $(\log3,0)$ вероятности равны $(0{,}75,0{,}5)$.
Отрицательная log-likelihood —
$-\log0{,}75-\log0{,}5\approx0{,}980829$.
При $\mu=0{,}5$, $\sigma^2=1$ KL равен 0,125, сумма — 1,105829.

```python
binary_observation = torch.tensor([[1., 0.]], dtype=ae_dtype)
binary_logits = torch.tensor([[math.log(3), 0.]], dtype=ae_dtype)
binary_rec = F.binary_cross_entropy_with_logits(
    binary_logits, binary_observation, reduction="none").sum(-1)
binary_kl = .5 * torch.tensor([.5**2], dtype=ae_dtype)
assert abs(binary_rec.item() - .9808292530117262) < 1e-12
assert abs((binary_rec + binary_kl).item() - 1.1058292530117262) < 1e-12
print(binary_rec, binary_kl)
```

Если усреднить BCE по двум признакам, а KL оставить прежним, изменится
относительный вес двух частей. Для изображений расхождение может составлять
сотни или тысячи раз. Согласованное правило — сначала суммировать по
координатам каждого объекта, сложить reconstruction и KL, затем усреднить
по объектам. Другие соглашения допустимы, но их коэффициенты нужно пересчитать.

## Reparameterization trick

В реконструкционный член входит ожидание по $q_\phi(z\mid x)$, которое само
зависит от параметров энкодера. Наивный вызов случайного генератора не даёт
обычного пути производной от выбранного числа к этим параметрам.
Репараметризация строит такой путь явно:

$$
\begin{aligned}
\epsilon&\sim\mathcal N(0,I),\\
z&=\mu+\exp(\ell/2)\odot\epsilon.
\end{aligned}
$$

Распределение полученного $z$ остаётся нужным нормальным распределением.
Источник случайности $\epsilon$ не зависит от параметров, а преобразование
$\epsilon\mapsto z$ дифференцируется по $\mu$ и $\ell$.
При фиксированном $\epsilon$ это обычный вычислительный граф.

![[00 Учебник/Assets/Figures/curated/source-first-7-14/weng-reparameterization.png|Источник случайности до и после репараметризации]]

*Справа случайным узлом остаётся $\epsilon$, а $z$ вычисляется через параметры.
По атрибуции Lilian Weng, схема — слайд 12 доклада Diederik P. Kingma на
NIPS 2015 workshop; [публикация Weng с источником](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick).
Ромбы обозначают детерминированные узлы, синие круги — случайные.
Здесь $f$ — функция выбранного $z$, а $g$ — преобразование шума в $z$.*

### Числовой путь производной

Возьмём $\mu=0{,}5$, $\ell=\log4$, $\epsilon=-0{,}25$.
Стандартное отклонение равно 2, и $z=0{,}5+2(-0{,}25)=0$.
Пусть декодер выдаёт logits $(z,-z)$ для цели $(1,0)$.
При $z=0$ обе вероятности равны 0,5; reconstruction loss равен $2\log2$,
а производная по $z$ равна $-1$.

Поскольку $\partial z/\partial\mu=1$, реконструкционный вклад в
$\partial L/\partial\mu$ равен $-1$. Вклад KL равен $\mu=0{,}5$,
сумма — $-0{,}5$.
Для log-variance $\partial z/\partial\ell=
\exp(\ell/2)\epsilon/2=-0{,}25$.
Реконструкционный вклад равен 0,25, вклад KL —
$(\exp\ell-1)/2=1{,}5$, сумма — 1,75.

```python
fixed_mu = torch.tensor([[.5]], dtype=ae_dtype, requires_grad=True)
fixed_logvar = torch.tensor([[math.log(4)]], dtype=ae_dtype, requires_grad=True)
fixed_epsilon = torch.tensor([[-.25]], dtype=ae_dtype)
fixed_z = fixed_mu + (.5 * fixed_logvar).exp() * fixed_epsilon
fixed_logits = torch.cat([fixed_z, -fixed_z], dim=-1)
fixed_rec = F.binary_cross_entropy_with_logits(fixed_logits, binary_observation, reduction="sum")
fixed_kl = .5 * (fixed_mu.square() + fixed_logvar.exp() - 1 - fixed_logvar).sum()
fixed_loss = fixed_rec + fixed_kl
fixed_gradients = torch.autograd.grad(fixed_loss, (fixed_mu, fixed_logvar))
torch.testing.assert_close(fixed_z, torch.zeros_like(fixed_z))
torch.testing.assert_close(fixed_gradients[0], torch.tensor([[-.5]], dtype=ae_dtype))
torch.testing.assert_close(fixed_gradients[1], torch.tensor([[1.75]], dtype=ae_dtype))
print(fixed_loss.item(), fixed_gradients)
```

Здесь $\epsilon$ выбран для ручного расчёта. При обучении его независимо
выбирают заново для объектов пакета. Один проход даёт случайную оценку
ожидаемой реконструкционной потери и её градиента. Проверять эту производную
центральной разностью нужно при одном и том же $\epsilon$ в обеих точках,
иначе сравниваются разные случайные функции.

В PyTorch `Normal.sample()` не строит этот путь, а `Normal.rsample()`
использует репараметризацию. Выражение `mu + std * torch.randn_like(std)`
строит его явно. Возможны и другие оценки градиента ожидания, поэтому
«любое сэмплирование делает обучение невозможным» — слишком сильное утверждение.

### Полный шаг маленького VAE

В следующем пакете два бинарных признака и одна латентная координата.
Энкодер выдаёт mean и log-variance, декодер — logits Бернулли.

```python
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)
x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
encoder = nn.Linear(2, 2)
decoder = nn.Linear(1, 2)  # Bernoulli logits, not probabilities
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.01)

optimizer.zero_grad()
mu, logvar = encoder(x).chunk(2, dim=-1)
z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
logits = decoder(z)
rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)
kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1)
loss = (rec + kl).mean()
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
optimizer.step()
print(rec.detach(), kl.detach(), loss.item())
```

При новом случайном $z$ следующая оценка потери не обязана уменьшиться даже
после корректного шага оптимизатора. Отдельно проверяют конечность градиентов,
изменение параметров и качество на отложенных данных. Для устойчивой
арифметики logits передаются в `binary_cross_entropy_with_logits` напрямую:
дополнительная sigmoid перед этой функцией не нужна.

## Исходный VAE из PyTorch examples

В [официальном примере](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/vae/main.py)
модель использует 784 входа, скрытый слой 400, код размерности 20 и
отдельные линейные головы для mean/log-variance.
Ниже класс и функция потерь сохранены без изменения английского кода.

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

torch.manual_seed(26)
original_vae = VAE().double()
original_binary_batch = (torch.rand(2, 1, 28, 28) > .5).to(ae_dtype)
original_optimizer = torch.optim.SGD(original_vae.parameters(), lr=1e-4)
original_optimizer.zero_grad()
original_reconstruction, original_mu, original_logvar = original_vae(original_binary_batch)
original_loss = loss_function(original_reconstruction, original_binary_batch, original_mu, original_logvar)
original_loss.backward()
assert original_reconstruction.shape == (2, 784)
assert original_mu.shape == original_logvar.shape == (2, 20)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in original_vae.parameters())
original_optimizer.step()
print(original_loss.item())
```

Источник использует sigmoid в декодере и `binary_cross_entropy`,
принимающую вероятности; предыдущий маленький пример использует logits.
Смешивать эти два API нельзя. Функция источника суммирует оба члена
по всему пакету. Деление всей суммы на размер пакета меняет масштаб
градиента, но сохраняет относительный вес reconstruction и KL.

В исходном MNIST-примере пиксели преобразуются в числа $[0,1]$.
BCE принимает такие мягкие цели, но распределение Бернулли буквально задаёт
массу на 0 и 1, а не плотность непрерывного оттенка.
В локальной проверке выше входы действительно бинарные.
Полный учебный запуск MNIST, загрузка данных и обучение исходного скрипта
здесь не воспроизводятся.

Код PyTorch examples распространяется под
[BSD 3-Clause](/licenses/pytorch-examples-acc295dc-BSD-3-Clause.txt).
Классы UvA — под [MIT](/licenses/uva-c7d9e18f-MIT.txt).

## Генерация и интерполяция

При реконструкции сначала используется энкодер данного объекта.
При генерации берут код из prior, поэтому входной объект и энкодер не нужны.
Для маленького Bernoulli-декодера выше:

```python
with torch.no_grad():
    z_prior = torch.randn(4, 1)
    probabilities = torch.sigmoid(decoder(z_prior))
    samples = torch.bernoulli(probabilities)
assert samples.shape == (4, 2)
```

`probabilities` содержит условные средние, а `samples` — бинарные наблюдения.
После одного учебного шага ни то ни другое ещё не доказывает хорошего
приближения распределения данных.

### Eval не отключает случайный код автоматически

В сохранённом классе PyTorch метод `reparameterize` вызывается и в train,
и в eval. Само `model.eval()` не отменяет случайное сэмплирование VAE.
Для детерминированной реконструкции можно подать в декодер posterior mean,
если именно такой протокол выбран для оценки.

```python
original_vae.eval()
with torch.no_grad():
    eval_mu, eval_logvar = original_vae.encode(original_binary_batch.view(-1, 784))
    mean_reconstruction = original_vae.decode(eval_mu)
    stochastic_reconstruction1, _, _ = original_vae(original_binary_batch)
    stochastic_reconstruction2, _, _ = original_vae(original_binary_batch)
assert not torch.equal(stochastic_reconstruction1, stochastic_reconstruction2)
assert mean_reconstruction.shape == (2, 784)
print(mean_reconstruction.shape)
```

Декодирование среднего кода $f_\theta(\mathbb E[z])$ в нелинейной сети не равно
в общем случае среднему восстановлению $\mathbb E[f_\theta(z)]$.

### Интерполяция между двумя кодами

Для двух кодов $z_a$ и $z_b$ линейная интерполяция задаёт

$$
\begin{gathered}
z_\alpha=(1-\alpha)z_a+\alpha z_b,\\
0\le\alpha\le1.
\end{gathered}
$$

При $\alpha=0$ получается первый код, при $\alpha=1$ — второй.
Например, между $(0,0)$ и $(2,4)$ середина равна $(1,2)$.
Декодер вычисляют заново для каждой промежуточной точки; усреднение двух
готовых изображений в общем случае даст другой результат из-за нелинейности.

Для двух объектов из предыдущего пакета возьмём средние их posterior.
Пять значений $\alpha$ дают пять кодов размерности 20. Сохранённый декодер
PyTorch возвращает для них 784 вероятности:

```python
interpolation_alpha = torch.linspace(0., 1., 5, dtype=ae_dtype)[:, None]
endpoint_a, endpoint_b = eval_mu[0], eval_mu[1]
interpolation_codes = (1 - interpolation_alpha) * endpoint_a + interpolation_alpha * endpoint_b
with torch.no_grad():
    interpolation_probabilities = original_vae.decode(interpolation_codes)
torch.testing.assert_close(interpolation_codes[0], endpoint_a)
torch.testing.assert_close(interpolation_codes[-1], endpoint_b)
torch.testing.assert_close(interpolation_codes[2], (endpoint_a + endpoint_b) / 2)
assert interpolation_probabilities.shape == (5, 784)
print(interpolation_codes.shape, interpolation_probabilities.shape)
```

Эта сеть прошла один учебный шаг, поэтому полученные вероятности не служат
демонстрацией качества перехода между цифрами. На обученной модели
интерполяции проверяют вместе с реконструкциями и независимыми выборками
из prior.

Гладкость декодера не гарантирует, что весь отрезок между кодами проходит
через области правдоподобных объектов. Даже распределение промежуточных кодов
отличается от распределения концов. Если два конца независимо выбраны из
$\mathcal N(0,I)$, их середина имеет распределение $\mathcal N(0,I/2)$.
При усреднении независимых величин дисперсии складываются с квадратами
коэффициентов: $1/4+1/4=1/2$. Поэтому интерполяция не заменяет проверку
генерации из исходного prior.

## Вес KL, β-VAE и потеря информации о входе

В β-VAE вес KL к prior меняется:

$$
L_\beta=-R+\beta K.
$$

Пусть два кандидата имеют пары (reconstruction loss, KL)
$(0{,}980829,0{,}125)$ и $(1{,}2,0{,}01)$.
При $\beta=1$ суммы равны примерно 1,106 и 1,21: первый лучше.
При $\beta=4$ получаются 1,481 и 1,24: выигрывает второй, хотя его
реконструкция хуже. Изменился критерий выбора параметров, а не только
скорость оптимизации.

```python
beta_candidates = [(.9808292530117262, .125), (1.2, .01)]
selected = [
    min(range(2), key=lambda i: beta_candidates[i][0] + beta * beta_candidates[i][1])
    for beta in [1., 4.]
]
assert selected == [0, 1]
print(selected)
```

Больший коэффициент сильнее ограничивает возможность кодировать каждый объект
отдельным распределением. Но он не гарантирует, что отдельные координаты
приобретут значения вроде «поворот», «цвет» или «размер». Для таких
интерпретаций нужны дополнительные предположения и проверка; ограничения
обучения disentangled representations обсуждают
[Locatello et al.](https://arxiv.org/abs/1811.12359).

Крайний случай — posterior collapse: энкодер выдаёт prior независимо от $x$,
а декодер не использует $z$. Тогда KL может быть нулём, но код не передаёт
информацию о конкретном объекте.

Для пакета $(1,0)$, $(0,1)$ можно построить такой случай явно.
Пусть энкодер для обоих объектов выдаёт $q=\mathcal N(0,1)$,
а декодер всегда возвращает две вероятности 0,5, независимо от $z$.
KL равен нулю, но оба входа восстанавливаются одинаково.
Для каждого отрицательная log-likelihood равна $2\log2$.
По коду невозможно определить, какой из двух объектов был подан на вход.

Сильный декодер способен моделировать данные и без этого канала;
низкий KL сам по себе не является признаком удачного
представления. Проверяют реконструкцию при изменении или замене $z$,
вклад KL по координатам и чувствительность декодера к коду.
При $\beta\ne1$ оптимизируется изменённая цель, не стандартная ELBO.

### Связь с последующими генеративными моделями

В VQ-VAE энкодер выбирает векторы из дискретного codebook, а обучение
квантования и распределения таких кодов требует отдельного разбора.
Связь с визуальными токенами и авторегрессивной генерацией раскрыта в
[[00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|главе о Chameleon]].
Этот дискретный механизм не следует подменять формулой нормального posterior
из текущего VAE.

В latent diffusion автоэнкодер переводит изображение в менее дорогой код,
отдельная диффузионная модель обучается генерировать коды, а декодер возвращает
изображение. Это другой способ моделировать распределение кодов;
весь процесс не сводится к прямому выбору из prior обычного VAE.
Первичный источник — [Rombach et al., Latent Diffusion Models](https://arxiv.org/abs/2112.10752).

<a id="краткие-итоги"></a>

## Что проверять в реализации

Сначала фиксируют модель наблюдений и диапазон входа. Затем проверяют,
что encoder действительно выдаёт mean и log-variance, стандартное отклонение
вычисляется как `exp(0.5*logvar)`, а источник шума не отсоединяет нужный путь
градиента. Формула KL и reduction проверяются отдельно от качества генерации.

Для оценки сохраняют реконструкции отложенных входов, выборки из prior,
распределения членов потери и протокол случайности при валидации.
Изображение декодированного среднего, один удачный пример или малая
обучающая MSE не заменяют эти разные проверки.

## Источники

- Phillip Lippe, [UvA Deep Learning Tutorials, Deep Autoencoders](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html):
  построение свёрточного автоэнкодера, transposed convolution, MSE и ограничения генерации.
- Lilian Weng, [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/):
  обычный/denoising автоэнкодер, вероятностная постановка VAE, ELBO и β-VAE.
- Kingma, Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114):
  §§2.1–2.4 и Appendix B.
- Carl Doersch, [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908).
- [PyTorch examples, VAE](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/vae/main.py),
  [полная BSD 3-Clause license](/licenses/pytorch-examples-acc295dc-BSD-3-Clause.txt).
- Dumoulin, Visin, [A guide to convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic),
  [полная MIT license](/licenses/conv-arithmetic-af6f818b-MIT.txt).
- [[05 Источники/Courses/Machine Learning Visualized/book/main.pdf|Machine Learning Visualized — Complete Book]]:
  дополнительный визуальный разбор autoencoder.

[[00 Учебник/01 Основы нейронных сетей/09 GAN и DCGAN — генератор и дискриминатор|Следующая глава]]
разбирает обучение генератора с помощью дискриминатора.
