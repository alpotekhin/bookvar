---
title: "CS224N — Lecture 13: Speech Brain-Computer Interfaces"
course: "Stanford CS224N"
lecture: 13
type: course-note
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture13-speech-bci]]"
concepts: ["[[02 Areas/ML & DL/Concepts/NLP/Brain-Computer Interface|BCI]]", "[[02 Areas/ML & DL/Concepts/NLP/Speech Recognition|Speech Recognition]]", "[[02 Areas/ML & DL/Concepts/NLP/CTC|CTC]]", "[[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]]"]
---

# Lecture 13: Speech Brain-Computer Interfaces

> *"So many years of not being able to communicate and then suddenly the people in the room got what I said."* -- Участник клинического исследования, Stanford Magazine

Лектор: Chaofei Fan (Stanford NPTL Lab).

## Мотивация: потеря речи

Неврологические расстройства -- brainstem stroke, ALS (Amyotrophic Lateral Sclerosis, боковой амиотрофический склероз) -- могут привести к **полной потере речи** и моторики. Коммуникация с близкими и caregivers -- одна из самых острых потребностей таких пациентов.

Существующие assistive communication devices (eye-tracking, switch-based) медленные и утомительные. Brain-Computer Interfaces (BCI) обещают **прямое декодирование** речевых намерений из нейронной активности.

## История BCI

### Электричество в мозге

- **1875**: Richard Caton обнаружил электрические токи в мозге животных
- **1924**: Hans Berger (немецкий психиатр) записал **первую человеческую EEG** (электроэнцефалограмму)

### Эволюция записи нейронной активности

| Метод | Invasiveness | Разрешение | Применение |
|-------|-------------|-----------|-----------|
| **EEG** (электроды на скальпе) | Non-invasive | Низкое (миллионы нейронов) | P300 speller, BCI на волнах мозга |
| **ECoG** (электроды на коре) | Semi-invasive | Среднее | Speech BCI (Moses et al. 2021) |
| **Microelectrode arrays** (Utah arrays) | Invasive | Высокое (отдельные нейроны) | State-of-the-art speech BCI |

### Нейроны и спайки

Нейроны общаются через **спайки** (action potentials). Одиночные нейроны в моторной коре кодируют **направление движения**. Множество нейронов вместе позволяют точное декодирование -- "population coding".

### От движения к речи

Ранние BCI: управление 2D курсором, роботизированными руками, восстановление ходьбы. **Прорыв**: декодирование handwriting -- **90 символов/мин** с 95% точностью (Willett et al. 2021, Nature).

Следующий шаг: декодирование **речи** -- потенциально самый быстрый канал коммуникации.

## Нейробиология речи

### Языковые области мозга

Fedorenko et al. 2024: language processing распределён по нескольким областям коры (temporal, frontal). Моторная кора речи (ventral motor cortex, area 6v) и область Broca (area 44) -- ключевые для **production** речи.

### Артикуляторное и фонемное кодирование

Bouchard et al. 2013: моторная кора кодирует **артикуляторные** (движения языка, губ, челюсти) и **фонемные** параметры речи. Это делает декодирование возможным: нейронная активность → фонемы → слова.

## Speech BCI Pipeline

### Общая архитектура

```
Neural activity (microelectrode arrays)
        ↓
Neural-to-phonemes decoder (RNN/GRU + CTC)
        ↓
Phonemes-to-words decoder (beam search)
        ↓
Language Model rescoring (n-gram / Transformer LM)
        ↓
Text output
```

### Формализация задачи

- **Вход**: нейронные features $\{x_1, x_2, \ldots, x_n\}$, $x_i \in \mathbb{R}^{d \times 1}$ (d каналов записи, ~20ms time bins)
- **Выход**: слова $\{y_1, y_2, \ldots, y_m\}$, $y_i \in \mathbb{R}^{V \times 1}$ (one-hot из словаря V)

Это задача **seq2seq**, но с **монотонным alignment** (нейронная активность и фонемы идут в одном порядке) -- идеально для CTC.

### Выбор нейросети

| Архитектура | Когда лучше |
|------------|-------------|
| **[[02 Areas/ML & DL/Concepts/Architectures/Transformer\|Transformer]]** | Большие датасеты, long-range dependencies |
| **RNN/GRU** | Малые датасеты, short-range dependencies, efficient real-time processing |

Для BCI (малый объём данных, real-time требования) чаще используют **GRU** (Gated Recurrent Unit).

### [[02 Areas/ML & DL/Concepts/NLP/CTC|CTC]] (Connectionist Temporal Classification)

Hannun 2017: CTC решает проблему **неизвестного alignment** между входной последовательностью (нейронные features) и выходной (фонемы).

**Идея**: модель на каждом timestep предсказывает фонему или **blank** ($\varepsilon$). Множество путей (с разными расположениями blanks) маппятся на одну и ту же последовательность:

```
εhelεεllεo  →  hello
heεllεllεo  →  hello
hhellεεloo  →  hello
```

**CTC loss**: сумма вероятностей всех путей, дающих правильный output. Эффективно вычисляется через **forward-backward** алгоритм (аналог HMM).

### CTC inference с beam search

На inference: **beam search** по фонемным последовательностям. Ключевая формула:

$$Y^* = \arg\max_Y P(Y \mid X) \cdot P(Y)^\alpha \cdot L(Y)^\gamma$$

где:
- $P(Y \mid X)$ -- CTC-модель (neural decoder)
- $P(Y)^\alpha$ -- **language model** probability (n-gram или Transformer)
- $L(Y)^\gamma$ -- **word insertion bonus** (предотвращает слишком короткие гипотезы)

### Роль Language Model

**LM rescoring** критически важен: декодер часто путает фонемы (b/p, d/t), но LM знает, что "I **can** speak" гораздо вероятнее, чем "I **ban** speak".

Два подхода:
1. **Shallow fusion**: n-gram LM в реальном времени (внутри beam search)
2. **Transformer LM rescoring** (2nd pass): генерируем n-best hypotheses, rescoring более мощной LM

## Ключевые результаты: Stanford NPTL Lab

### Участник T12

- **Диагноз**: bulbar-onset ALS, ограниченная вокализация, но полная потеря intelligible speech
- **Имплантация**: четыре 64-канальных Utah arrays (два в area 6v, два в area 44)
- **Данные**: ~100 минут training + ~30 минут evaluation. Предложения из **Switchboard** corpus (телефонные разговоры, ~1000 предложений)

### Real-time brain-to-text BCI (Willett, Kunz, Fan et al. 2023)

- Декодирование **attempted speech** (пациент пытается говорить, но звуков не производит)
- Нейроны кодируют **orofacial movements** (движения лица/рта) и фонемную информацию
- **Результат**: >60 слов/мин с <25% Word Error Rate

### Оценка: Word Error Rate (WER)

$$WER(Y, \hat{Y}) = \frac{\text{distance}(Y, \hat{Y})}{\text{length}(Y)}$$

Нормализованное edit distance между предсказанными словами и ground truth.

### Brain-to-Text Benchmark '24

Стандартизированный бенчмарк для сравнения speech BCI систем. Позволяет оценить прогресс в декодировании.

## Будущее Speech BCI

### Multimodal Speech BCI

Metzger et al. 2023: декодирование не только текста, но и **параметров речи** (pitch, prosody) для синтеза voice output.

### BCI для личного использования

Card et al. 2024: движение к **accurate, practical BCI** для повседневного использования вне лаборатории.

### Декодирование inner speech

Frontier задача: декодирование **внутренней речи** (мысленной, без попытки артикуляции). Пока accuracy значительно ниже, чем для attempted speech, но результаты обнадёживающие (decoding accuracy существенно выше chance для some paradigms).

### Neuroethics

BCI поднимают **новые этические вопросы**: приватность мыслей, consent при тяжёлой инвалидности, ownership нейронных данных, dual-use potential.

## Concepts covered

- [[02 Areas/ML & DL/Concepts/NLP/Brain-Computer Interface|BCI]] -- интерфейс мозг-компьютер, от EEG до microelectrode arrays
- [[02 Areas/ML & DL/Concepts/NLP/Speech Recognition|Speech Recognition]] -- декодирование речи из нейронных сигналов
- [[02 Areas/ML & DL/Concepts/NLP/CTC|CTC]] -- Connectionist Temporal Classification для seq2seq с монотонным alignment
- [[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]] -- n-gram и Transformer LM для rescoring в BCI pipeline
