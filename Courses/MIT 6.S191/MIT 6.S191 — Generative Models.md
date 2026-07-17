---
title: "MIT 6.S191 — Generative Models"
type: course-note
course: "MIT 6.S191"
---

# MIT 6.S191 — Deep Generative Modeling

> Лекция 4. VAE, GAN, Diffusion — как научить нейросеть создавать новые данные.

**Курс:** [[MIT 6.S191/_index|MIT 6.S191]]
**Лекторы:** Alexander Amini, Ava Amini
**Связанные концепты:** [[Causal Language Modeling]], [[Sampling]]

---

## Discriminative vs Generative

**Discriminative models** (классификаторы): учат p(y|x) — "это кот или собака?"

**Generative models:** учат p(x) — "как выглядят данные?" Позволяют:
- Генерировать новые данные (изображения, текст, музыку)
- Обнаруживать аномалии (p(x) низкий = outlier)
- Заполнять пропуски (inpainting, imputation)
- Учить хорошие представления данных (unsupervised)

```
Discriminative:  [image] → "cat" (label)
Generative:      [noise/latent] → [new image]
```

---

## Latent Variable Models

### Идея

Данные высокоразмерные (изображение 256x256 = 196K пикселей), но "живут" на низкоразмерном многообразии (manifold). Пример: все лица описываются небольшим числом факторов — пол, возраст, поза, освещение.

**Latent variable z** — сжатое представление этих факторов:

```
z ∈ R^d (d = 32-512)  →  Decoder  →  x ∈ R^D (D = 196,608)
```

---

## Autoencoders

### Vanilla Autoencoder

Учится **сжимать** и **восстанавливать** данные:

```
x → [Encoder] → z (bottleneck) → [Decoder] → x̂
Loss = ||x - x̂||²    (reconstruction loss)
```

**Проблема:** latent space неструктурирован. Нельзя сэмплировать z и получить осмысленное изображение.

---

## Variational Autoencoder (VAE)

### Идея

Вместо детерминированного z, encoder выдаёт **распределение** q(z|x):

```
x → [Encoder] → μ, σ² → z ~ N(μ, σ²) → [Decoder] → x̂
```

### Loss Function

```
L = E[||x - x̂||²]  +  D_KL(q(z|x) || p(z))
    ──────────────     ─────────────────────
    reconstruction       regularization
```

- **Reconstruction term** — точность восстановления
- **KL divergence** — "заставляет" q(z|x) быть похожим на N(0, I)

**Reparameterization trick:** чтобы backpropagation работал через sampling:

```
z = μ + σ ⊙ ε,    ε ~ N(0, I)
```

Градиенты текут через μ и σ, sampling ε не зависит от параметров.

### Свойства VAE

**Плюсы:**
- Структурированный latent space — интерполяция между точками осмысленна
- Стабильная тренировка
- Probabilistic framework — можно вычислить p(x)

**Минусы:**
- Blurry генерация (MSE loss размывает детали)
- Ограниченное качество по сравнению с GAN/Diffusion

---

## Generative Adversarial Networks (GAN)

### Идея — игра двух сетей

```
z ~ N(0,I) → [Generator G] → fake image
                                   │
real image ──────────────────┐     │
                             ▼     ▼
                        [Discriminator D] → real/fake?
```

**Generator** пытается создать изображения, неотличимые от настоящих.
**Discriminator** пытается отличить настоящие от поддельных.

### Loss Functions

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

- D максимизирует: правильно классифицировать real и fake
- G минимизирует: обмануть D

При оптимуме G генерирует данные из p_data, а D не может отличить — выдаёт 0.5.

### Эволюция GAN

| Модель | Год | Инновация |
|--------|-----|-----------|
| GAN | 2014 | Базовый фреймворк |
| DCGAN | 2015 | Свёрточный G и D |
| WGAN | 2017 | Wasserstein distance, стабильная тренировка |
| Progressive GAN | 2018 | Постепенное увеличение разрешения |
| StyleGAN | 2019 | Style-based G, контроль стиля по слоям |
| StyleGAN2/3 | 2020-21 | Alias-free, state-of-the-art лица |

### Проблемы GAN

1. **Mode collapse** — G генерирует только несколько "типов" изображений
2. **Training instability** — баланс G и D нестабилен
3. **No likelihood** — нельзя вычислить p(x)
4. **Evaluation** — FID/IS, но нет идеальной метрики

---

## Diffusion Models

### Идея

Два процесса:

**Forward process (diffusion):** постепенно добавляем шум к данным за T шагов:

```
x₀ → x₁ → x₂ → ... → x_T ≈ N(0, I)
           постепенно добавляем гауссов шум
```

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

**Reverse process (denoising):** обучаем нейросеть **убирать шум** на каждом шаге:

```
x_T ~ N(0,I) → x_{T-1} → ... → x₁ → x₀ (чистое изображение)
                 модель предсказывает шум на каждом шаге
```

### Обучение

Модель ε_θ учится предсказывать шум, добавленный на шаге t:

```
L = E[||ε - ε_θ(x_t, t)||²]
```

Простая MSE loss — но работает невероятно хорошо.

### Почему Diffusion > GAN

| Аспект | GAN | Diffusion |
|--------|-----|-----------|
| Тренировка | Нестабильная | Стабильная (простая loss) |
| Mode coverage | Mode collapse | Полное покрытие |
| Likelihood | Нет | Можно оценить |
| Качество | Отличное | State-of-the-art |
| Скорость генерации | Быстрая (1 pass) | Медленная (T шагов) |
| Контролируемость | Ограниченная | Classifier-free guidance |

### Ключевые модели

- **DDPM** (Ho et al., 2020) — базовый diffusion model
- **Stable Diffusion** (2022) — diffusion в latent space (быстрее)
- **DALL-E 2/3** — text-to-image через diffusion
- **Midjourney** — commercial art generation

---

## Autoregressive Models для генерации

Подход, используемый в LLM: генерируем по одному элементу, conditioning на предыдущих.

```
p(x) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... · p(x_n|x₁,...,x_{n-1})
```

Для изображений: PixelCNN, ImageGPT генерируют пиксели слева-направо, сверху-вниз.

Для текста: GPT, LLaMA — autoregressive LMs ([[Causal Language Modeling]]).

---

## Conditional Generation

Генерация с условием — класс, текст, другое изображение:

```
p(x | condition)
```

### Classifier-Free Guidance (CFG)

Стандарт для text-to-image diffusion:

```
ε_guided = ε_uncond + w · (ε_cond - ε_uncond)
```

w (guidance scale) контролирует "силу" условия:
- w = 1: следуем условию нормально
- w = 7-15: усиливаем соответствие тексту (стандарт для Stable Diffusion)
- w > 20: "перенасыщение", артефакты

---

## Метрики оценки

### FID (Frechet Inception Distance)

Сравнивает распределения real и generated изображений в feature space Inception-v3:

```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^(1/2))
```

**Ниже FID = лучше.** State-of-the-art: FID < 2 на ImageNet 256x256.

### IS (Inception Score)

Измеряет quality (уверенная классификация) и diversity (равномерное распределение по классам).

---

## Ключевые выводы

1. **VAE** — вероятностная модель с регуляризованным latent space, стабильная но blurry
2. **GAN** — adversarial training, sharp результаты, но нестабильная тренировка
3. **Diffusion** — итеративный denoising, state-of-the-art качество, медленная генерация
4. **Autoregressive** — по одному элементу, используется для текста (LLM)
5. Тренд: **Diffusion + Transformer** (DiT) = будущее image generation

---

## Источники

- MIT 6.S191, Lecture 4 — https://introtodeeplearning.com/
- Goodfellow et al., "Generative Adversarial Nets" (2014)
- Kingma & Welling, "Auto-Encoding Variational Bayes" (VAE, 2013)
- Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM, 2020)

---

**См. также:** [[Causal Language Modeling]], [[Sampling]], [[MIT 6.S191 — Intro to DL]], [[MIT 6.S191 — Computer Vision]]
