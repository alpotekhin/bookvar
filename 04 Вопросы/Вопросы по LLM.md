---
title: Вопросы по LLM
type: question-index
status: active
last_updated: 2026-07-20
last_verified: 2026-07-20
---

# Вопросы по LLM

## Pre-training и масштабирование

- Как происхождение корпуса, очистка и веса источников меняют распределение,
  которое видит модель? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных#От сырого источника к обучающему потоку|От источника к обучающему потоку]]
- Почему почти точные дубликаты нельзя надёжно удалить одним хешированием? →
  [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных#Дедупликация|Дедупликация]]
- Как температура смеси меняет число проходов по малым доменам и риск
  переобучения? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных#Смеси данных|Смеси данных]]
- Как правило цепочки приводит к next-token loss и почему все позиции можно
  обучать параллельно? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction#От правила цепочки к функции потерь|Правило цепочки и NTP]]
- Чем teacher forcing на обучении отличается от генерации на собственном
  префиксе? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction#Истинный префикс на обучении и собственный префикс при генерации|Teacher forcing и генерация]]
- Как связаны cross-entropy, perplexity и выбранный токенизатор? →
  [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction#Перплексия|Перплексия]]
- Как из isoFLOP-экспериментов получают compute-optimal пару размера модели и
  числа токенов? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws#IsoFLOP-профили: наблюдаем минимум, а не угадываем его|IsoFLOP-профили]]
- Почему Chinchilla-optimal модель для обучения может быть неоптимальна с учётом
  многократного inference? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws#Где простая модель перестаёт работать|Границы scaling laws]]
- Какие состояния занимают память при обучении и что именно шардируют
  DDP, FSDP/ZeRO, TP и PP? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision#Оси параллелизма|Оси параллелизма]]
- Чем FP16, BF16 и FP8 различаются по диапазону, точности и требованиям к
  масштабированию? → [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision#Смешанная точность|Смешанная точность]]

## Post-training и alignment

- На каких позициях считается SFT loss и чем loss mask отличается от attention
  mask? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data#1. От сырой строки до обучающей последовательности|SFT: сообщения, токены и метки]]
- Почему chat template является частью обученной модели? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data#Шаблон диалога — часть модели|Шаблон диалога]]
- Что один SFT target сообщает о желаемом поведении и чего не сообщает о
  правдоподобных альтернативах? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data#5. Что SFT умеет и чего не обещает|Возможности и ограничения SFT]]
- Как ranking нескольких ответов превращают в пары, не создавая утечку между
  train и validation? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data#7. Преобразование ранжирования в пары без утечки|Ранжирование и зависимые пары]]
- Как position, verbosity и judge bias попадают в preference dataset? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data#5. Систематические смещения разметки|Смещения разметки]]
- Как из Bradley–Terry likelihood получается pairwise loss модели награды? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#2. Модель Брэдли—Терри|Модель Брэдли—Терри]]
- Чем reward model отличается от value model, verifier, PRM и LLM judge? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#5. Какие виды оценщиков существуют|Виды оценщиков]]
- Как sequence-level reward создаёт градиенты на отдельных token actions? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#2. Вывод REINFORCE без пропущенных шагов|Вывод REINFORCE]]
- Зачем PPO одновременно нужны current, old, reference, reward и value models? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#5. Пять разных ролей в PPO|Пять ролей в PPO]]
- Почему PPO clipping относительно old policy и KL к reference policy решают
  разные задачи? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#6. Почему простого градиента стратегии недостаточно|PPO clipping]]
- Как DPO выводится из KL-регуляризованной задачи RLHF и где сокращается
  partition function? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO#5. Выражаем скрытую награду через языковую модель|Вывод DPO]]
- Почему standard DPO остаётся offline-методом и зависит от reference model? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO#10. DPO и PPO: общая постановка, разные способы обучения|DPO и PPO]]
- Почему RLVR описывает источник награды, а не конкретный optimizer? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers#Главное различие|RLVR как источник награды]]
- Почему parser и verifier должны быть отдельными программами? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers#Разбор ответа и проверка — разные задачи|Parser и verifier]]
- Как GRPO вычисляет advantage внутри группы и что происходит при нулевой
  дисперсии наград? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1#Сквозной численный пример|Group-relative advantage]]
- Чем R1-Zero отличается от полного многоэтапного DeepSeek-R1? →
  [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1#R1-Zero и R1 — два разных эксперимента|R1-Zero и R1]]
- Чем sequence-level, logit и on-policy distillation отличаются по данным и
  функции потерь? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation#Три разных объекта под одним названием|Три вида дистилляции]]

## Генерация и inference

- Как temperature, top-k и top-p меняют множество и относительные вероятности
  кандидатов? → [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена#Случайная выборка|Случайная выборка]]
- Чем prefill отличается от decode по вычислениям и работе с памятью? →
  [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention#Prefill и decode используют устройство по-разному|Prefill и decode]]
- Из каких размерностей выводится объём KV-cache? →
  [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention#Сколько памяти занимает история|Размер KV-cache]]
- Как PagedAttention отделяет логическую последовательность от физических
  блоков памяти? → [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention#PagedAttention: логическая последовательность, физические блоки|PagedAttention]]
- Почему FlashAttention является точным tiled algorithm, а не новым видом
  attention? → [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention#FlashAttention не следует путать с архитектурой внимания|FlashAttention и архитектура]]
- Чем PTQ, QAT и QLoRA различаются по моменту внесения quantization error? →
  [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей#Что именно квантуется|Что квантуется]]
- Как speculative decoding сохраняет распределение target model и от чего
  зависит ускорение? → [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование#Откуда берётся возможность ускорения|Speculative decoding]]

## Reasoning и evaluation

- Чем pass@k, self-consistency и best-of-N различаются по способу выбора ответа?
  → [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#2. Две части любого метода: предложить и выбрать|Предложить и выбрать]]
- Почему увеличение числа reasoning tokens не гарантирует улучшения ответа? →
  [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#6. Длинное рассуждение и последовательное исправление|Длинное рассуждение]]
- Почему benchmark score нельзя интерпретировать без prompt, decoding protocol,
  uncertainty и проверки contamination? → [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация#Протокол является частью benchmark|Протокол benchmark]]
- Как искать contamination, когда обучающий корпус доступен и когда он закрыт? →
  [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация#Что такое контаминация|Контаминация]]
