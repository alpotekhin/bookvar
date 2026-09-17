---
title: Autoencoder и variational autoencoder
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
primary_sources:
  - https://lilianweng.github.io/posts/2018-08-12-vae/
  - https://arxiv.org/abs/1312.6114
  - https://arxiv.org/abs/1606.05908
---

# Autoencoder и variational autoencoder

Autoencoder получает объект $x$, сжимает его в код $z$ и пытается восстановить
$x$. Обычный autoencoder учит детерминированное представление; variational
autoencoder учит распределение латентных переменных и получает генеративную
модель, из которой можно осмысленно сэмплировать.

## Обычный autoencoder

Encoder и decoder задают

$$
z=f_\phi(x),
\qquad
\hat x=g_\theta(z).
$$

Параметры минимизируют ошибку реконструкции, например

$$
L_{rec}=\|x-\hat x\|_2^2.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/weng-autoencoder.png]]

*Encoder сжимает объект в код, decoder восстанавливает вход. Узкое место
ограничивает пропускную способность, но само по себе ещё не задаёт вероятностную
модель. Источник: Lilian Weng,
[From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/),
прямая [ссылка](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png).*

Если encoder и decoder слишком мощные и bottleneck ничего не ограничивает,
модель может приблизить identity function без полезного представления. Поэтому
используют низкую размерность $z$, sparse penalty, шум во входе или masking.
Denoising autoencoder восстанавливает чистый $x$ из повреждённого $\tilde x$ и
тем самым учится игнорировать допустимый шум.

Линейный autoencoder с MSE и подходящими ограничениями связан с PCA, но
нелинейный encoder способен описывать искривлённое многообразие данных.

## Почему обычный latent space неудобен для генерации

Encoder может разместить обучающие объекты отдельными островами. Точка между
двумя кодами не обязана декодироваться в правдоподобный объект, а случайная
точка из $\mathcal N(0,I)$ может вообще не попадать в область, которую decoder
видел при обучении.

VAE делает код вероятностным и согласует его с выбранным prior.

## Encoder VAE выдаёт распределение

Для каждого $x$ encoder предсказывает параметры приближённого posterior:

$$
q_\phi(z\mid x)=
\mathcal N\left(z;\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\right).
$$

Decoder задаёт likelihood $p_\theta(x\mid z)$, а prior обычно выбирают
$p(z)=\mathcal N(0,I)$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/weng-vae-graphical-model.png]]

*Слева генеративная модель $p_\theta(x,z)$, справа — приближённый вывод
$q_\phi(z\mid x)$. Сплошные стрелки принадлежат генеративному процессу,
пунктирные — encoder, который нужен для обучения и вывода. Источник: Lilian
Weng, [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/),
прямая [ссылка](https://lilianweng.github.io/posts/2018-08-12-vae/VAE-graphical-model.png).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vae/vae-blocks.png]]

*Encoder предсказывает $\mu$ и $\sigma$, latent sample поступает в decoder.
Автор схемы: Sebastian Nguyen / Wikimedia contributors,
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:VAE_blocks.png),
CC BY-SA 4.0.*

## ELBO: reconstruction и regularization latent space

Log-likelihood данных содержит трудно вычислимый интеграл по $z$.
Введём $q=q_\phi(z\mid x)$ и используем правило Байеса внутри ожидания:

$$
\begin{aligned}
D_{KL}(q\|p_\theta(z\mid x))
&=\mathbb E_q[\log q-\log p_\theta(x,z)+\log p_\theta(x)]\\
&=\log p_\theta(x)-\mathbb E_q[\log p_\theta(x,z)-\log q].
\end{aligned}
$$

Последнее ожидание обозначают ELBO. Поскольку
$p_\theta(x,z)=p_\theta(x\mid z)p(z)$, оно равно reconstruction term
минус KL к prior. Получено точное тождество
$\log p_\theta(x)=\mathrm{ELBO}+D_{KL}(q\|p_\theta(z\mid x))$.
Неотрицательность последнего KL даёт нижнюю границу:

$$
\log p_\theta(x)\ge
\underbrace{\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]}_{\text{reconstruction}}
-
\underbrace{D_{KL}(q_\phi(z\mid x)\|p(z))}_{\text{prior matching}}.
$$

Первое слагаемое требует сохранять информацию об объекте. Второе не позволяет
каждому примеру занять произвольный изолированный участок latent space. Слишком
сильный KL может привести к posterior collapse: decoder игнорирует $z$, а
$q(z\mid x)$ приближается к prior.

Важно не смешивать два KL. Расстояние до **истинного posterior** — зазор
между ELBO и log-likelihood; расстояние до **prior** — одно из слагаемых
самой ELBO. Это тот же вариационный приём, что в
[[00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM|EM для GMM]].
Но E-step GMM вычисляет точный posterior для текущих параметров, а VAE
приближает его одной обучаемой сетью сразу для многих объектов; зазор обычно
не равен нулю.

## Reparameterization trick

Прямое сэмплирование $z\sim\mathcal N(\mu,\sigma^2)$ выглядит как недифференцируемый
узел между энкодером и функцией потерь. Случайность выносят в независимую переменную:

$$
\epsilon\sim\mathcal N(0,I),
\qquad
z=\mu+\sigma\odot\epsilon.
$$

При фиксированном sample $\epsilon$ выражение детерминированно по $\mu$ и
$\sigma$, поэтому backpropagation проходит через них. Обычно encoder предсказывает
`logvar`, а standard deviation вычисляется как `exp(0.5 * logvar)`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/weng-reparameterization.png]]

*Слева случайный узел зависит от параметров распределения и разрывает обычную
цепочку производных. Справа случайность вынесена в независимый $\epsilon$, а
$z=\mu+\sigma\epsilon$ становится дифференцируемой функцией параметров.
Источник: Lilian Weng,
[From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/),
прямая [ссылка](https://lilianweng.github.io/posts/2018-08-12-vae/reparameterization-trick.png).*

Зафиксируем модель наблюдений: два бинарных признака независимы при заданном
$z$, а decoder возвращает их Bernoulli logits. Тогда отрицательный
reconstruction log-likelihood — сумма binary cross-entropy по двум признакам.
Для диагонального Gaussian posterior KL вычисляется аналитически:

$$
D_{KL}(q\|\mathcal N(0,I))=
\frac12\sum_j\left(\mu_j^2+\exp(\mathrm{logvar}_j)-1-\mathrm{logvar}_j\right).
$$

Например, для $x=(1,0)$, logits $(\log3,0)$, $\mu=0{,}5$,
$\sigma^2=1$ reconstruction loss равен
$-\log0{,}75-\log0{,}5\approx0{,}980829$, KL равен $0{,}125$,
а их сумма — $1{,}105829$. Усреднение BCE по признакам вместо суммы
уменьшило бы только reconstruction term вдвое и изменило относительный
вес KL. Поэтому reduction — часть модели обучения.

Следующий маленький батч иллюстрирует законченный шаг. Размер latent равен 1;
encoder выдаёт две координаты: mean и log-variance. Для каждого объекта берётся
один независимый sample, суммы считаются по признакам/latent-координатам,
а среднее — только по объектам батча.

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

Это один шаг, не обученная генеративная модель. Из-за нового случайного $z$
следующая оценка loss не обязана стать меньше. Проверять реализацию градиента
центральной разностью следует при фиксированном $\epsilon$, а проверять
качество обучения — по многим объектам и нескольким samples. Полный цикл на
изображениях приведён в [примере PyTorch](https://github.com/pytorch/examples/blob/main/vae/main.py).

## Генерация и интерполяция

После обучения можно взять $z\sim p(z)$ и декодировать новый объект. Плавность
latent space делает интерполяции более осмысленными, но качество зависит от
decoder likelihood и баланса ELBO. Для предыдущего Bernoulli decoder код
генерации отделён от реконструкции: входной объект и encoder не нужны.

```python
with torch.no_grad():
    z_prior = torch.randn(4, 1)
    probabilities = torch.sigmoid(decoder(z_prior))
    samples = torch.bernoulli(probabilities)
assert samples.shape == (4, 2)
```

`probabilities` — условные средние, а `samples` — собственно бинарные
наблюдения. После одного учебного шага они ещё не обязаны соответствовать
распределению данных. VAE часто даёт более размытые изображения,
чем adversarial или diffusion models, зато имеет явную вероятностную постановку
и удобную сеть приближённого вывода.

VAE остаётся практическим компонентом современных систем: latent diffusion
сжимает изображения encoder-ом, выполняет дорогой generative process в меньшем
latent space и декодирует результат обратно.

## Краткие итоги

- Autoencoder учит `x → z → x_hat`; bottleneck или noise не дают тривиально
  скопировать вход.
- VAE предсказывает распределение $q(z\mid x)$, а не одну точку.
- ELBO сочетает reconstruction likelihood и KL к prior.
- Reparameterization отделяет источник случайности от обучаемых параметров.
- Хорошая реконструкция сама по себе не гарантирует удобного пространства для
  sampling; в VAE за это отвечает probabilistic regularization.

## Источники

- [[05 Источники/Courses/Machine Learning Visualized/book/main.pdf|Machine Learning Visualized — Complete Book]] — исходный раздел об autoencoder и reconstruction loss.
- Lilian Weng, [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/).
- Kingma, Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- Carl Doersch, [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908).

**Дальше:** после общих принципов нейросетей учебник переходит к представлению
текста числами и обучению word embeddings.
