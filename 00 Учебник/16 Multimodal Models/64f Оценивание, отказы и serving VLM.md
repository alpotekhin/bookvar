---
title: "64.6. Оценивание, режимы отказа и serving VLM"
type: textbook-chapter
status: canonical
last_updated: 2026-07-23
primary_sources:
  - https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
  - https://arxiv.org/abs/2305.10355
  - https://arxiv.org/abs/2409.17146
  - https://arxiv.org/abs/2502.13923
---

# Оценивание, режимы отказа и serving VLM

Усреднённый multimodal score смешивает распознавание, OCR, знания, reasoning и качество формата. Рост одной способности способен скрыть деградацию другой. Хорошая оценка VLM поэтому начинается с разложения задачи и заканчивается проверкой того, что ответ действительно зависит от visual evidence.

## 1. Карта способностей

Perception benchmarks проверяют объекты, атрибуты и текст. Grounding — точки, boxes и spatial relations. Knowledge-and-reasoning datasets требуют совместить изображение с внешними знаниями или вычислением. Video benchmarks добавляют порядок и время. Hallucination tests проверяют утверждения об отсутствующих объектах.

MMMU охватывает университетские дисциплины, MathVista — визуальную математику, OCRBench/DocVQA/ChartQA — текст и структуру, Video-MME/LongVideoBench — временное понимание. Ни один набор не является полной мерой «мультимодального интеллекта».

## 2. Композиционность

CLIP-подобная модель может узнавать сущности, но путать отношения. Winoground строит минимальные пары изображений и подписей с одинаковыми словами, но разной композицией. Чтобы ответить, недостаточно знать, что на сцене есть лошадь и трава: нужно различить, кто что делает.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/cs231n-lecture16-winoground.png]]

*Stanford CS231n 2025, [Lecture 16: Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF p. 48. Слайд вводит Winoground как проверку композиционности CLIP-подобных representations; первичный источник указан на слайде.*

Минимальные пары полезнее случайного набора, потому что фиксируют объекты и меняют только отношение. Аналогично можно переставлять строки таблицы, кадры видео или подписи осей.

## 3. Галлюцинации и языковой prior

VLM часто называет объект, типичный для сцены, хотя его нет. POPE превращает это в polling: задаёт вопросы о наличии объектов, выбирая негативы случайно, по частоте или по co-occurrence. Precision показывает склонность отвечать «да» на правдоподобные, но отсутствующие объекты.

Однако POPE не исчерпывает галлюцинации. Модель может ошибиться в атрибуте, отношении, количестве или тексте. Новые сильные модели частично насыщают простой benchmark. Нужны adversarial distractors, открытые описания с CHAIR-подобной разметкой и human review.

Главная диагностика — удалить или заменить evidence. Если ответ сохраняется после закрытия нужной области, система опирается на вопрос или prior. При замене изображения на контрпример должна меняться не только формулировка, но и факт ответа.

## 4. LLM-as-a-judge

Judge удобен для открытых ответов, но предпочитает подробность, стиль и совпадение с reference. Он может не заметить маленькую visual error или сам не распознать изображение. Поэтому judge prompt, версия модели и доступ к изображению являются частью протокола.

Для проверяемых задач лучше deterministic metric: exact match числа, IoU, edit distance OCR, выполнение GUI action. Для описаний полезно сочетать rubric-based judge с выборочной человеческой проверкой и публиковать ответы, а не только score.

## 5. Контаминация

Изображение может отсутствовать в train set, но его caption, вопрос или решение быть опубликованы в сети. Перерисованный график сохраняет структуру benchmark. Проверка должна учитывать perceptual duplicates, OCR текста на изображениях и synthetic variants. Закрытые training data делают отрицательное утверждение о контаминации особенно трудным.

Новые тесты лучше строить после cutoff модели, хранить часть приватно и генерировать параметрические варианты. Но private benchmark снижает воспроизводимость; нужен прозрачный protocol и независимый аудит.

## 6. Открытость как часть воспроизводимости

Open weights недостаточны, если vision encoder, данные или synthetic teacher закрыты. Molmo предлагает полезную матрицу: отдельно проверяет weights и data+code у VLM, LLM backbone и vision encoder. Модель, обученная на ответах закрытой VLM, остаётся зависимой от неё даже при открытых весах.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure11-openness.png]]

*Рисунок 11 из Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 19. Таблица разделяет открытость итоговой VLM, языкового backbone, vision encoder и данных.*

Эта таблица относится к состоянию на дату статьи, а не к вечной характеристике семейств. В справочнике её следует сопровождать датой проверки.

## 7. Serving начинается с preprocessing

Для text-only LLM длина prompt известна после tokenization. В VLM стоимость зависит от декодирования изображения, resize, числа crops/frames и visual encoder. Сервер должен ограничивать pixels, images, frames и total visual tokens до постановки запроса в batch.

Этапы запроса:

1. загрузка и проверка media;
2. preprocessing и token-budget calculation;
3. vision encoder;
4. connector/resampler;
5. LLM prefill по visual+text tokens;
6. autoregressive decode.

TTFT включает первые пять стадий. Если логировать только LLM engine, задержка загрузки и vision encoder исчезает из наблюдаемости.

## 8. Кэширование и batching

Image embeddings можно кэшировать для повторных вопросов, но cache key обязан включать bytes/hash, preprocessing config, encoder и connector version. Изменение `max_pixels` меняет representation того же файла. Для видео key включает sampling policy.

Batching осложняется переменной длиной. Block-diagonal packing позволяет vision encoder обрабатывать разные размеры без взаимного attention. На LLM-этапе visual tokens конкурируют с текстом за prefill capacity и KV cache. Полезно разделять очереди по ожидаемой стоимости или вводить admission control.

## 9. Метрики системы

Наряду с accuracy нужны media preprocessing latency, vision encoder latency, LLM prefill, TTFT, inter-token latency, peak memory и tokens/pixels per second. Для streaming audio — time-to-first-audio и interruption latency. Quality–cost curve строят по pixel/frame/token budget.

Нагрузочный тест должен отражать смесь: короткие фотографии, многостраничные документы, несколько изображений и видео. Средний запрос скрывает head-of-line blocking от крупных media. SLO задают по классам запросов или вводят отдельные пределы.

## 10. Безопасность

Изображение может содержать prompt injection, невидимый мелкий текст, QR-код или персональные данные. OCR content нельзя автоматически считать доверенной системной инструкцией. Media parsers должны иметь ограничения размера и формата; удалённые URL — сетевую изоляцию. Логи с изображениями требуют более строгой политики хранения, чем обычные prompts.

Надёжный интерфейс показывает uncertainty и evidence. Для документов — crop и координаты, для видео — timestamp, для audio — segment. Возможность проверить источник ответа полезнее уверенного общего объяснения.

## Источники и продолжение

- Stanford CS231n, [Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf).
- Li et al., [Evaluating Object Hallucination in LVLMs (POPE)](https://arxiv.org/abs/2305.10355).
- Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Связано: [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|физика inference]] и [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|оценивание и контаминация]].
- Назад: [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|64.5]]. Далее: [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|65. Tool use]].
