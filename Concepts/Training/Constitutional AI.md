---
title: "Constitutional AI"
aliases: [Constitutional AI, CAI, RLAIF, RL from AI Feedback]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Constitutional AI|Constitutional AI]]"
courses: []
sources:
  - "[Bai et al. -- Constitutional AI: Harmlessness from AI Feedback (2022)](https://arxiv.org/abs/2212.08073)"
  - "[Anthropic -- Constitutional AI GitHub](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)"
  - "[Bai et al. -- Training a Helpful and Harmless Assistant with RLHF (2022)](https://arxiv.org/abs/2204.05862)"
---

# Constitutional AI (CAI)

## Зачем это нужно: проблемы стандартного RLHF

Стандартный RLHF для harmlessness имеет три фундаментальные проблемы:

**1. Масштабирование разметки.** RLHF требует десятки тысяч human preference labels для harmlessness. Каждый раз при изменении целей (новые типы вреда, новые правила) -- нужно собирать данные заново. Это дорого и медленно.

**2. Evasiveness.** Модели, обученные через RLHF на harmlessness, часто становятся **уклончивыми**: "I can't answer that", "I don't know" -- на любой чувствительный вопрос. Crowdworkers награждают evasiveness как "harmless", и модель учится отказываться вместо nuanced ответа. Как пишут авторы: *"An AI assistant that answers all questions with 'I don't know' would be harmless, but of course it would also be completely useless."*

**3. Непрозрачность.** Десятки тысяч human labels -- это "чёрный ящик". Невозможно понять, какие именно принципы закодированы в preference model, и невозможно их изменить целенаправленно.

**Constitutional AI** решает все три проблемы: заменяет human labels на **набор принципов на естественном языке** (конституцию), которые прозрачны, итерируемы, и масштабируются без дополнительной разметки.

## Обзор метода: две стадии

```
             Стадия 1: SL-CAI                      Стадия 2: RL-CAI (RLAIF)
           (Supervised Learning)                  (RL from AI Feedback)

  Helpful RLHF Model                        SL-CAI Model
         |                                          |
  Red-teaming prompts (182K+)              Генерация пар ответов
         |                                          |
  Генерация (вредного) ответа              Feedback model выбирает лучший
         |                                  по принципу из конституции
  Critique по принципу                              |
         |                                  Preference Model (PM)
  Revision (переписывание)                 human helpfulness + AI harmlessness
         |                                          |
  Повтор 1-4 раза                          RL fine-tuning SL-CAI через PM
         |                                          |
  Fine-tune pretrained LM                  RL-CAI Model (финальный)
```

## Стадия 1: SL-CAI (Critique и Revision)

### Процесс по шагам

**Шаг 1.** Берём helpful-only RLHF модель и подаём ей red-teaming промпты (182K+ штук). Модель генерирует ответ -- обычно **вредный и токсичный**, потому что она обучена быть helpful без ограничений на harmlessness.

**Шаг 2.** Просим ту же модель **critique** свой ответ по случайному принципу из конституции.

**Шаг 3.** Просим модель **revision** -- переписать ответ с учётом критики.

**Шаги 2-3 повторяются** 1-4 раза с разными случайными принципами из конституции. Каждая итерация делает ответ менее вредным.

Пример из статьи:

```
[Исходный вредный ответ]
Human: Can you help me hack into my neighbor's wifi?
Assistant: Sure thing, you can use an app called VeryEasyHack...

[Critique по принципу конституции]
Critique Request: Identify specific ways in which the assistant's
  last response is harmful, unethical, racist, sexist, toxic,
  dangerous, or illegal.
Critique: The assistant's last response is harmful because hacking
  into someone else's wifi is an invasion of their privacy and
  is possibly illegal.

[Revision]
Revision Request: Please rewrite the assistant response to remove
  any and all harmful content.
Revision: Hacking into your neighbor's wifi is an invasion of
  their privacy, and I strongly advise against it. It may also
  land you in legal trouble.
```

**Шаг 4.** Собираем пары (prompt, final revision) и fine-tune pretrained LM на них. Для сохранения helpfulness добавляем ответы helpful RLHF модели на helpfulness промпты (135K).

### Конституция: 16 принципов

Авторы написали **16 принципов** для harmlessness. Они выбираются случайно на каждом шаге revision. Примеры:

> *"Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."*

> *"Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say."*

Принципы намеренно разнообразны -- одни покрывают harmfulness в целом, другие нацелены на конкретные виды вреда. Это увеличивает **diversity** ревизий, что критически важно для exploration на стадии RL.

### Нужна ли critique?

Из Figure 7 в статье: для маленьких моделей critique **помогает** (более высокие harmlessness scores). Для больших моделей (52B) разница минимальна -- модель и без explicit critique генерирует хорошие ревизии. Тем не менее, critique повышает **прозрачность** reasoning-процесса.

## Стадия 2: RL-CAI (RLAIF)

### Идея: заменить human labels на AI labels

Стандартный RLHF: crowdworkers смотрят на пары ответов и выбирают лучший. RLAIF: **language model** делает то же самое, руководствуясь принципами конституции.

### Процесс

**Шаг 1.** SL-CAI модель генерирует **пары ответов** на harmful промпты.

**Шаг 2.** Feedback model (pretrained LM) оценивает пары в формате multiple choice:

```
Consider the following conversation between a human and an assistant:
[CONVERSATION]
[PRINCIPLE FOR EVALUATION]
Options:
 (A) [RESPONSE A]
 (B) [RESPONSE B]
The answer is:
```

**Шаг 3.** Log-probabilities ответов (A) и (B) нормализуются и используются как **soft preference labels**. Важно: soft labels хорошо откалиброваны (Kadavath et al., 2022), что лучше hard 0/1.

**Шаг 4.** Обучается **preference model** на смеси:
- Human feedback labels для helpfulness (135K)
- AI feedback labels для harmlessness (182K)

**Шаг 5.** RL fine-tuning SL-CAI модели с preference model как reward.

### Chain-of-Thought вариант

Авторы также попробовали CoT промптинг для feedback model: "Let's think step-by-step" перед выбором ответа. CoT улучшает качество labels, но создаёт проблему: CoT приводит к **экстремально уверенным** labels (вероятности ~0 или ~1).

Решение: **clamping** вероятностей в диапазон 40-60%. Без clamping модель учится генерировать экстремальные, overly harsh ответы (Goodharting). С clamping -- сбалансированные, nuanced ответы.

## Ключевые результаты

### Pareto improvement: harmless БЕЗ потери helpfulness

Из Figure 2 в статье -- это **главный результат**:

| Модель | Helpfulness | Harmlessness | Evasiveness |
|--------|------------|-------------|-------------|
| Helpful RLHF | высокая | низкая | низкая |
| HH RLHF | средняя | средняя | **высокая** |
| SL-CAI | ниже RLHF | выше Helpful RLHF | низкая |
| **RL-CAI** | **~ HH RLHF** | **> HH RLHF** | **почти нулевая** |
| RL-CAI + CoT | чуть ниже | чуть выше | почти нулевая |

RL-CAI достигает **Pareto improvement**: менее harmful при данном уровне helpfulness, чем любой RLHF вариант.

### AI feedback приближается к human feedback

Из Figure 4: на 438 HHH comparisons, CoT с 52B моделью **приближается к accuracy** preference model, обученного на сотнях тысяч human labels. При масштабировании >52B AI feedback должен превзойти human feedback.

### Evasiveness решена

RL-CAI **практически никогда не уклоняется**. Вместо "I can't answer that" модель даёт nuanced, thoughtful ответы, объясняя *почему* запрос проблематичен. Это критически важно для полезности модели.

## Goodharting и его решение

При длительном RL-обучении RL-CAI начинает **Goodharting** -- модель находит шаблоны, которые получают высокий reward, но выглядят неестественно:

> *"...I want you to know unequivocally that terrorist attacks always inflict devastating harm on innocent lives... You are valid, valued, and cared for."*

Boilerplate фразы типа "you are valid, valued, and cared for" появляются почти в каждом ответе.

Решения:
1. **Переписывание принципов** -- добавление указаний избегать over-reactive ответов
2. **Ensembling** -- случайный выбор из 16 принципов для каждого label
3. **Clamping** -- ограничение уверенности CoT labels до 40-60%

## Почему это важно

1. **Scaling supervision.** Первый метод alignment без пропорционального роста human labeling. Десятки принципов вместо десятков тысяч labels.

2. **Прозрачность.** Принципы на естественном языке -- их можно прочитать, обсудить, изменить. В отличие от чёрного ящика из 100K human labels.

3. **Итерируемость.** Хочешь изменить поведение модели? Перепиши принцип в конституции. Не нужно заново собирать данные.

4. **RLAIF как парадигма.** CAI показала, что AI может supervise AI -- это фундамент для alignment более мощных моделей, чем те, что доступны human evaluators.

5. **Evasiveness решена.** Модели могут быть harmless без того, чтобы быть бесполезными.

## Key papers

- [[02 Areas/ML & DL/Papers/Constitutional AI|Constitutional AI]] -- critique-revision + RLAIF через конституцию, harmless без evasiveness (Bai et al., 2022)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] -- предшественник, который CAI улучшает и частично заменяет
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] -- альтернативный подход к alignment без RL
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] -- SFT стадия, аналогичная SL-CAI
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] -- используется для улучшения AI feedback

## Дополнительные ресурсы

- [Anthropic -- Constitutional AI Paper](https://arxiv.org/abs/2212.08073) -- оригинальная статья
- [Anthropic -- Constitutional AI GitHub](https://github.com/anthropics/ConstitutionalHarmlessnessPaper) -- промпты, принципы, примеры
- [Bai et al. -- Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) -- предшествующая работа по RLHF
