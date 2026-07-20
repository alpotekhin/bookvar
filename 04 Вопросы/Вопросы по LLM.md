---
title: Вопросы по LLM
type: question-index
status: active
last_updated: 2026-07-17
last_verified: 2026-07-17
---

# Вопросы по LLM

## Обучение

- Из каких стадий обычно состоит создание современной LLM?
  → [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] ·
  [[02 Areas/ML & DL/Concepts/Training/SFT|SFT]] ·
  [[02 Areas/ML & DL/Concepts/Training/Alignment|Alignment]]
- Что предсказывает decoder-only модель при pre-training?
  → [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]]
- Что утверждают scaling laws и чего они не гарантируют?
  → [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]
- Чем fine-tuning отличается от instruction tuning?
  → [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] ·
  [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]
- Как LoRA уменьшает число обучаемых параметров?
  → [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]]
- Что такое mixed precision и distributed training?
  → [[02 Areas/ML & DL/Concepts/Training/Mixed Precision Training|Mixed Precision]] ·
  [[02 Areas/ML & DL/Concepts/Training/Distributed Training|Distributed Training]]

## Post-training и alignment

- На каких token positions считается SFT loss и чем loss mask отличается от
  attention mask? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data#1. От сырой строки до обучающей последовательности|SFT: сообщения → tokens → labels]]
- Почему chat template является частью обученной модели?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data#Шаблон диалога — часть модели|Шаблон диалога]]
- Что сообщает один SFT target и чего он не говорит об альтернативных ответах?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data#5. Что SFT умеет и чего не обещает|Ограничения SFT]]
- Как ranking четырёх ответов превращается в pairs и почему эти pairs зависимы?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data#Ранжирование $K$ ответов|Ранжирование → пары]]
- Как position, verbosity и judge bias попадают в preference dataset?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data#5. Систематические смещения разметки|Смещения разметки]]
- Почему scalar reward model не является абсолютной оценкой качества?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#2. Модель Брэдли—Терри|Модель Брэдли—Терри]]
- Чем reward model отличается от value model, verifier и LLM judge?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#5. Какие виды оценщиков существуют|Типы оценщиков]]
- Как policy может эксплуатировать reward model даже при высокой validation
  accuracy? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#7. Эксплуатация награды и сдвиг распределения|Эксплуатация награды]]
- Как sequence-level reward создаёт gradients на отдельных token actions?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#2. Вывод REINFORCE без пропущенных шагов|Вывод REINFORCE]]
- Зачем нужны baseline, value function и advantage?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#3. Базовый уровень и преимущество: сравнение с ожиданием|Базовый уровень и преимущество]]
- Чем old rollout policy отличается от frozen reference policy?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#5. Пять разных ролей в PPO|Роли моделей в PPO]]
- Почему PPO clipping и KL к reference не дублируют друг друга?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM#7. Полная целевая функция|Полная целевая функция PPO]]
- На каком шаге вывода DPO сокращается partition function $Z(x)$?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO#5. Выражаем скрытую награду через языковую модель|Вывод DPO]]
- Что означает DPO margin относительно reference и почему standard DPO всё ещё
  нужна reference policy? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO#6. Что именно сравнивает функция потерь DPO|Разность оценок DPO]]
- Почему DPO является offline-методом и чего он не исследует?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO#10. DPO и PPO: общая постановка, разные способы обучения|DPO против PPO]]
- Чем RLVR отличается от optimizer и почему verifier не обязательно является
  истиной? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers#Главное различие|RLVR как источник reward]]
- Почему parser и verifier должны быть разными программами?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers#Разбор ответа и проверка — разные задачи|Разбор и проверка]]
- Как вручную вычислить group-relative advantages и что происходит при нулевой
  variance группы? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1#Сквозной численный пример|Расчёт GRPO]]
- Что именно GRPO убирает по сравнению с PPO, а что остаётся?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1#Полная целевая функция|Полная целевая функция GRPO]]
- Чем R1-Zero отличается от полного многоэтапного DeepSeek-R1?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1#R1-Zero и R1 — два разных эксперимента|R1-Zero и R1]]
- Чем sequence-level, logit и on-policy distillation отличаются друг от друга?
  → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation#Три разных объекта под одним названием|Три вида distillation]]
- Почему успешная reasoning distillation не доказывает копирование внутреннего
  алгоритма teacher? → [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation#Дистилляция не копирует внутренний алгоритм|Граница вывода]]

## Генерация и inference

- Как temperature, top-k и top-p меняют sampling?
  → [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]]
- Что хранится в KV-cache и почему он растёт с длиной контекста?
  → [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-cache]]
- Как quantization уменьшает память и где теряется качество?
  → [[02 Areas/ML & DL/Concepts/Inference/Quantization|Quantization]]
- Как speculative decoding ускоряет генерацию?
  → [[02 Areas/ML & DL/Concepts/Inference/Speculative Decoding|Speculative Decoding]]
- Почему длинное context window не означает понимание всего входа?
  → [[02 Areas/ML & DL/03 Исследовательские линии/Длинный контекст|Длинный контекст]]

## Reasoning

- Что показывает Chain-of-Thought и является ли текст объяснением внутренних
  вычислений? → [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]
- Как self-consistency использует несколько решений?
  → [[02 Areas/ML & DL/Concepts/Reasoning/Self-Consistency|Self-Consistency]]
- Что такое test-time compute?
  → [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute|Test-time Compute]]
- Что именно показал DeepSeek-R1?
  → [[02 Areas/ML & DL/05 Источники/Papers/DeepSeek-R1|DeepSeek-R1 source note]]
- Чем reasoning distillation отличается от RL?
  → [[02 Areas/ML & DL/03 Исследовательские линии/Синтетические данные и дистилляция|Синтетические данные и дистилляция]]

## Evaluation

- Почему одной perplexity недостаточно?
  → [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]]
- Что проверяют MMLU и HumanEval?
  → [[02 Areas/ML & DL/Concepts/Evaluation/MMLU|MMLU]] ·
  [[02 Areas/ML & DL/Concepts/Evaluation/HumanEval|HumanEval]]
- Почему benchmark score не равен production quality?
  → [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 11 — Benchmarking and Evaluation|Benchmarking and Evaluation]]
