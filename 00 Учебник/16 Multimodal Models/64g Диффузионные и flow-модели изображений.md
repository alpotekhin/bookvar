---
title: "64.7. Как генерируются изображения: diffusion, latent diffusion и flow matching"
type: textbook-chapter
status: canonical
last_updated: 2026-09-07
primary_sources:
  - https://arxiv.org/abs/2006.11239
  - https://arxiv.org/abs/2112.10752
  - https://arxiv.org/abs/2212.09748
  - https://arxiv.org/abs/2210.02747
  - https://arxiv.org/abs/2403.03206
---

# Как diffusion-модель превращает шум в изображение

Diffusion-модель не получает один текст и не пытается угадать по нему заранее выбранный шум. На каждом шаге denoiser получает **три входа одновременно**: зашумлённое изображение или latent $x_t$, номер уровня шума $t$ и текстовое условие $c$. Его задача — по этой совокупности оценить, какая часть текущего тензора является шумом или в каком направлении его следует изменить.

Текст «красная машина у моря» не задаёт единственную картинку. Случайный начальный шум задаёт конкретную траекторию и разнообразие, а текст направляет её в область изображений, согласующихся с описанием.

## 1. Как создаётся учебный пример

Берётся реальное изображение $x_0$. Затем случайно выбираются timestep $t$ и Gaussian noise $\epsilon\sim\mathcal N(0,I)$. Зашумлённый вход строится сразу, без последовательного выполнения всех предыдущих шагов:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

При малом $t$ первый коэффициент велик и предметы хорошо видны. При большом $t$ доминирует шум. Поскольку именно обучающая программа сгенерировала $\epsilon$, правильный ответ известен бесплатно. Denoiser обучается минимизировать

$$
\mathcal L_\epsilon=
\mathbb E\left\|\epsilon-epsilon_\theta(x_t,t,c)\right\|_2^2.
$$

Критическая деталь: сеть предсказывает шум **не по тексту**, а по $(x_t,t,c)$. Если убрать $x_t$, невозможно понять, какая конкретная случайная реализация шума была добавлена. Если убрать $t$, одинаковый паттерн может означать почти чистую деталь на раннем шаге или остаток сигнала на позднем. Если убрать текст, восстановление останется возможным, но не будет управляться prompt.

Один training step выглядит так:

1. взять пару «изображение — подпись»;
2. закодировать подпись текстовым encoder;
3. сэмплировать $t$ и случайный $\epsilon$;
4. смешать чистый sample с этим шумом и получить $x_t$;
5. передать $x_t$, $t$ и text features в denoiser;
6. сравнить предсказание с тем самым $\epsilon$;
7. обновить веса denoiser backpropagation.

Модель не обязана во время обучения пройти всю обратную цепочку. Один случайный timestep даёт unbiased обучающий сигнал, поэтому разные batch постепенно покрывают все уровни шума.

![Алгоритмы обучения и генерации DDPM](https://hojonathanho.github.io/diffusion/assets/img/algorithms.png)

*Алгоритмы 1 и 2 из Ho, Jain, Abbeel, [официальная страница DDPM](https://hojonathanho.github.io/diffusion/). Слева за один training step выбирается случайный $t$ и известный $\epsilon$; справа sampling многократно применяет обученный denoiser, начиная с Gaussian noise.*

## 2. Почему локальное предсказание позволяет генерировать целую картинку

Во время inference чистого $x_0$ нет. Мы начинаем с $x_T\sim\mathcal N(0,I)$. На шаге $T$ сеть оценивает шум, scheduler вычисляет немного менее шумный $x_{T-1}$, затем операция повторяется. Сначала проявляется крупная композиция, позже уточняются границы, фактуры и мелкие детали.

Это похоже не на восстановление спрятанной конкретной фотографии, а на движение по выученному полю направлений. Denoiser видел множество уровней разрушения настоящих изображений и научился распознавать, в какую сторону лежит область правдоподобных данных. Разный initial seed приводит к разным допустимым изображениям одного prompt.

Scheduler не является самой нейросетью. Он задаёт формулу перехода и noise schedule; denoiser оценивает нужную величину. Число timesteps обучения также не равно обязательному числу inference steps: современные solvers могут приблизить траекторию значительно меньшим числом вызовов модели.

## 3. Latent diffusion: denoising не в RGB

Для изображения $512\times512$ пиксельный тензор велик. Latent Diffusion сначала обучает autoencoder:

$$
z_0=E(x_0),\qquad \hat x=D(z_0).
$$

После этого diffusion работает с компактным $z_t$, например формы $[4,64,64]$, а decoder VAE в конце возвращает RGB. Text encoder выдаёт последовательность embeddings, к которой пространственные признаки denoiser обращаются через cross-attention.

![Архитектура conditional latent diffusion](https://ommer-lab.com/wp-content/uploads/2022/08/article-Figure3-1-1024x508.png)

*Figure 3 из Rombach et al., [официальная страница Latent Diffusion Models](https://ommer-lab.com/research/latent-diffusion-models/). Слева encoder переводит RGB в latent, в центре работает denoiser с cross-attention к условию, справа decoder возвращает изображение.*

Экономия значительна, но VAE задаёт потолок: деталь, которую encoder не сохранил в $z_0$, diffusion-модель не сможет надёжно восстановить. VAE обычно заморожен при обучении denoiser; reconstruction losses кодека и diffusion loss не следует смешивать в одно описание.

## 4. Как текст влияет на пространственные признаки

Пусть text encoder вернул токены `красная`, `машина`, `у`, `моря`. В cross-attention позиции noisy latent формируют queries, а text tokens — keys и values. Одна область может сильнее читать слово `машина`, другая — `море`. Это не жёсткая маска объектов, но обучаемый канал, через который prompt меняет denoising на каждом слое.

Classifier-free guidance обучается благодаря случайному удалению условия. На inference модель вызывают в conditional и unconditional режимах и комбинируют предсказания:

$$
\hat\epsilon=\epsilon_{\varnothing}+s(\epsilon_c-\epsilon_{\varnothing}).
$$

Большой $s$ сильнее тянет sample к prompt, но может уменьшить разнообразие и создать пересыщенные детали. Guidance не добавляет модели новых знаний; он меняет силу уже выученного условного направления.

## 5. U-Net, DiT и objective — разные оси

DDPM исторически использовал time-conditioned U-Net. DiT разбивает noisy latent на patches и обрабатывает их Transformer-блоками. Это выбор backbone. Он не определяет автоматически, предсказывает ли модель $\epsilon$, clean sample, $v$-parameterization или velocity flow.

Поэтому фраза «это diffusion transformer» неполна. Для воспроизводимости нужны по крайней мере autoencoder, условные encoders, prediction target, timestep distribution, noise/path schedule, backbone и inference solver.

## 6. Flow matching: вместо шума — скорость

В простейшей rectified-flow записи выбираются noise $x_0$, data sample $x_1$ и промежуточная точка

$$
x_t=(1-t)x_0+tx_1.
$$

Целевая скорость этой прямой равна $u=x_1-x_0$. Сеть регрессирует vector field:

$$
\mathcal L_{FM}=\mathbb E\left\|v_\theta(x_t,t,c)-(x_1-x_0)\right\|_2^2.
$$

На inference ODE solver интегрирует $dx/dt=v_\theta(x,t,c)$ от noise к data. Интуитивно DDPM учит локально убирать шум, а flow matching — локально задавать скорость транспортировки распределения. В обоих случаях обучение использует случайную промежуточную точку, а генерация численно проходит траекторию от простого prior к изображению. Знак target зависит от направления времени в конкретной реализации, поэтому его нужно читать вместе со scheduler, а не запоминать изолированно.

Stable Diffusion 3 сочетает latent representation, rectified-flow objective и MM-DiT backbone. Image и text tokens имеют отдельные параметры, но взаимодействуют в joint attention. Этот пример хорошо показывает, почему `latent`, `flow` и `Transformer` отвечают на три разные архитектурные вопросы.

## 7. Быстрая проверка понимания

Если в training code есть `noise = randn_like(latents)`, `noisy = add_noise(latents, noise, t)` и `prediction = model(noisy, t, text)`, target — известен потому, что программа только что его сэмплировала. На inference строки с clean image и target исчезают: остаются initial noise, prompt, многократные вызовы model и solver step.

| Вопрос | Обучение | Генерация |
|---|---|---|
| Есть настоящее изображение? | да | нет |
| Известен добавленный шум? | да, мы его сэмплировали | нет, сеть его оценивает |
| Сколько timestep за один sample? | обычно один случайный | много последовательных solver steps |
| Зачем текст? | учит условному полю | направляет траекторию к prompt |
| Откуда разнообразие? | разные data/noise pairs | initial seed и stochastic solver |

## Источники и продолжение

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).
- Peebles and Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- Esser et al., [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).
- Назад: [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|64.6. Оценивание и serving]]. Далее: [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|64.8. Chameleon и единая мультимодальная последовательность]].
