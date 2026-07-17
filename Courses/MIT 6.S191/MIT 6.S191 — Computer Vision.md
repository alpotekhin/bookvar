---
title: "MIT 6.S191 — Computer Vision"
type: course-note
course: "MIT 6.S191"
---

# MIT 6.S191 — Deep Computer Vision

> Лекция 3. CNN: от свёрточных фильтров до современных vision-архитектур.

**Курс:** [[MIT 6.S191/_index|MIT 6.S191]]
**Лекторы:** Alexander Amini, Ava Amini
**Связанные концепты:** [[Text Classification]], [[Attention Mechanism]]

---

## Зачем нужны CNN

MLP для изображений работает плохо:
- Изображение 224x224x3 = **150,528** входов
- Один fully-connected слой с 1000 нейронов = **150M параметров**
- Нет учёта пространственной структуры — пиксель (0,0) и (223,223) обрабатываются одинаково

**CNN решает обе проблемы:** weight sharing (свёрточные фильтры) + locality (каждый нейрон "видит" только локальную область).

---

## Свёртка (Convolution)

### Идея

Маленький фильтр (kernel) "скользит" по изображению, вычисляя скалярное произведение на каждой позиции:

```
Input (5x5)          Kernel (3x3)         Output (3x3)
┌─┬─┬─┬─┬─┐         ┌─┬─┬─┐
│1│0│1│0│1│         │1│0│1│              ┌─┬─┬─┐
├─┼─┼─┼─┼─┤         ├─┼─┼─┤              │4│3│4│
│0│1│0│1│0│         │0│1│0│              ├─┼─┼─┤
├─┼─┼─┼─┼─┤         ├─┼─┼─┤              │2│4│3│
│1│0│1│0│1│         │1│0│1│              ├─┼─┼─┤
├─┼─┼─┼─┼─┤         └─┴─┴─┘              │4│3│4│
│0│1│0│1│0│                               └─┴─┴─┘
├─┼─┼─┼─┼─┤
│1│0│1│0│1│
└─┴─┴─┴─┴─┘
```

**Формула:**

```
output(i, j) = Σ_m Σ_n input(i+m, j+n) · kernel(m, n)
```

### Параметры свёртки

- **Kernel size** — размер фильтра (3x3, 5x5, 7x7). Чем больше — тем шире receptive field
- **Stride** — шаг перемещения (1 = каждая позиция, 2 = через одну)
- **Padding** — добавление нулей по краям (same padding сохраняет размер)
- **Channels** — число фильтров = число выходных каналов

```
Output size = (Input - Kernel + 2·Padding) / Stride + 1
```

### Что выучивают фильтры

Каждый фильтр — детектор определённого паттерна:

```
Ранние слои:        Средние слои:       Глубокие слои:
┌─┬─┬─┐            Текстуры,           Объекты,
│-│-│-│  горизонт.  повторяющиеся       лица,
│+│+│+│  край       паттерны            колёса
│-│-│-│
└─┴─┴─┘
```

Сеть **сама выучивает** нужные фильтры через backpropagation — это ключевое отличие от классического CV, где фильтры проектировались вручную.

---

## Pooling

Уменьшение пространственного размера, сохраняя важную информацию:

### Max Pooling

```
┌──┬──┐      ┌──┐
│ 1│ 3│      │  │
├──┼──┤  →   │ 4│    (берём max из каждого 2x2 блока)
│ 2│ 4│      │  │
└──┴──┘      └──┘
```

### Average Pooling

Среднее вместо максимума. Используется реже.

### Global Average Pooling (GAP)

Усредняем каждый канал feature map до одного числа. Заменяет fully-connected слои в конце сети — меньше параметров, меньше overfitting.

---

## Архитектура CNN

Типичная CNN состоит из чередующихся свёрточных слоёв и pooling:

```
Input Image (224x224x3)
    │
    ▼
Conv 3x3, 64 filters → ReLU → MaxPool 2x2    (112x112x64)
    │
    ▼
Conv 3x3, 128 filters → ReLU → MaxPool 2x2   (56x56x128)
    │
    ▼
Conv 3x3, 256 filters → ReLU → MaxPool 2x2   (28x28x256)
    │
    ▼
Conv 3x3, 512 filters → ReLU → MaxPool 2x2   (14x14x512)
    │
    ▼
Global Average Pooling                         (1x1x512)
    │
    ▼
Fully Connected → Softmax                     (1000 classes)
```

**Паттерн:** пространственный размер уменьшается, число каналов растёт. От конкретных пикселей к абстрактным признакам.

---

## Эволюция архитектур

### LeNet-5 (LeCun, 1998)

Первая успешная CNN для распознавания рукописных цифр. 5 слоёв, ~60K параметров.

### AlexNet (Krizhevsky, 2012)

Прорыв на ImageNet. Показал, что deep CNN + GPU = state-of-the-art. 8 слоёв, 60M параметров.

### VGG (Simonyan, 2014)

Доказал: стек 3x3 свёрток лучше больших фильтров. VGG-16: 16 слоёв, 138M параметров.

### ResNet (He, 2015)

**Residual connections** — революция в глубоких сетях:

```
x → [Conv → BN → ReLU → Conv → BN] → + → ReLU
└────────────────────────────────────→┘
         skip connection
```

```
F(x) + x    вместо    F(x)
```

Позволил тренировать сети с 100+ слоями. ResNet-152: 152 слоя, 60M параметров, лучше человека на ImageNet.

**Почему работает:** если слой "не нужен", он может выучить F(x) = 0, и тогда output = x (identity). Градиенты текут напрямую через skip connection.

### Vision Transformer (ViT, 2020)

Применение Transformer к изображениям:

1. Разрезаем изображение на патчи (16x16)
2. Каждый патч — "токен" (линейная проекция)
3. Добавляем positional embeddings
4. Пропускаем через стандартный Transformer encoder

```
Image → [patch₁, patch₂, ..., patch_n] → Transformer → classification
```

ViT побеждает CNN при больших данных (>100M изображений). При малых данных CNN лучше (inductive bias помогает).

---

## Задачи Computer Vision

### Image Classification

Входное изображение → один из N классов.

```
[Photo of cat] → "cat" (p=0.92)
```

### Object Detection

Находим **все объекты** на изображении + их bounding boxes.

Архитектуры: YOLO, Faster R-CNN, DETR (Detection Transformer).

### Semantic Segmentation

Классифицируем **каждый пиксель**: дорога, машина, пешеход, небо.

Архитектуры: U-Net, DeepLab, Segment Anything (SAM).

### Instance Segmentation

Semantic segmentation + различаем отдельные экземпляры: "машина #1", "машина #2".

---

## Data Augmentation для Vision

Увеличение обучающей выборки через трансформации:

```
Original → [Random Crop, Horizontal Flip, Color Jitter, Rotation, Cutout]
```

Каждая эпоха — случайная комбинация трансформаций. Модель видит "разные" версии одних и тех же изображений → лучшая обобщающая способность.

### Современные подходы

- **MixUp** — линейная интерполяция двух изображений и их лейблов
- **CutMix** — вырезаем часть одного изображения и вставляем в другое
- **RandAugment** — автоматический выбор оптимальных аугментаций

---

## Transfer Learning в Vision

**Стандартный pipeline:**

1. Pre-train на ImageNet (1.2M изображений, 1000 классов)
2. Заменяем последний слой на свой (N_target классов)
3. Fine-tune на целевом датасете

```
[ImageNet pretrained] → freeze early layers → fine-tune last layers → target task
```

Работает, потому что ранние слои учат **универсальные** фильтры (края, текстуры), а поздние — **task-specific** (породы собак, типы опухолей).

### CLIP (OpenAI, 2021)

Обучает image encoder и text encoder совместно на 400M пар (изображение, подпись):

```
Image encoder: [photo] → vector
Text encoder:  "a photo of a cat" → vector
Loss: cosine similarity (matching pairs high, non-matching low)
```

Zero-shot classification: сравниваем image embedding с text embeddings всех классов. Без какого-либо fine-tuning.

---

## Практика: Lab 2 — Facial Detection

Студенты строят CNN для обнаружения лиц:
- Классификация: лицо / не лицо
- Регрессия: координаты bounding box
- Понимание bias в данных и fairness

---

## Ключевые выводы

1. **CNN** использует свёртки для извлечения пространственных признаков с weight sharing
2. **Pooling** уменьшает размерность, сохраняя ключевую информацию
3. **ResNet** (skip connections) позволил строить очень глубокие сети
4. **ViT** показал, что Transformer работает и для vision при достаточном количестве данных
5. **Transfer learning** (ImageNet → target) — стандарт для practical CV

---

## Источники

- MIT 6.S191, Lecture 3 — https://introtodeeplearning.com/
- He et al., "Deep Residual Learning" (2015)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020)

---

**См. также:** [[Attention Mechanism]], [[Transfer Learning]], [[MIT 6.S191 — Intro to DL]], [[MIT 6.S191 — Generative Models]]
