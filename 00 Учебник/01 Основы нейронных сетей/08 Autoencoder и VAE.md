---
title: Autoencoder и variational autoencoder
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://lilianweng.github.io/posts/2018-08-12-vae/
  - https://arxiv.org/abs/1312.6114
  - https://arxiv.org/abs/1606.05908
---

# Autoencoder и variational autoencoder

Полный исходный раздел об autoencoder, его функции потерь и обучении сохранён в
[[05 Источники/Courses/Machine Learning Visualized/book/main.pdf|Machine Learning Visualized — Complete Book]].

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

Параметры минимизируют reconstruction loss, например

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

Log-likelihood данных содержит трудно вычислимый интеграл по $z$. VAE
максимизирует evidence lower bound:

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

## Reparameterization trick

Прямое сэмплирование $z\sim\mathcal N(\mu,\sigma^2)$ выглядит как недифференцируемый
узел между encoder и loss. Случайность выносят в независимую переменную:

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

```python
mu, logvar = encoder(x).chunk(2, dim=-1)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
x_hat = decoder(z)
```

## Генерация и интерполяция

После обучения можно взять $z\sim p(z)$ и декодировать новый объект. Плавность
latent space делает интерполяции более осмысленными, но качество зависит от
decoder likelihood и баланса ELBO. VAE часто даёт более размытые изображения,
чем adversarial или diffusion models, зато имеет явную вероятностную постановку
и удобный inference network.

VAE остаётся практическим компонентом современных систем: latent diffusion
сжимает изображения encoder-ом, выполняет дорогой generative process в меньшем
latent space и декодирует результат обратно.

## Что нужно унести из главы

- Autoencoder учит `x → z → x_hat`; bottleneck или noise не дают тривиально
  скопировать вход.
- VAE предсказывает распределение $q(z\mid x)$, а не одну точку.
- ELBO сочетает reconstruction likelihood и KL к prior.
- Reparameterization отделяет источник случайности от обучаемых параметров.
- Хорошая реконструкция сама по себе не гарантирует удобного пространства для
  sampling; в VAE за это отвечает probabilistic regularization.

## Источники

- Lilian Weng, [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/).
- Kingma, Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- Carl Doersch, [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908).

**Дальше:** после общих принципов нейросетей учебник переходит к представлению
текста числами и обучению word embeddings.
