---
title: "RLHF"
aliases: [Reinforcement Learning from Human Feedback, InstructGPT alignment]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/InstructGPT]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2]]"
  - "[[02 Areas/ML & DL/Papers/DPO|DPO]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[Hugging Face — Illustrating RLHF](https://huggingface.co/blog/rlhf)"
  - "[Chip Huyen — RLHF](https://huyenchip.com/2023/05/02/rlhf.html)"
  - "[Sebastian Raschka — LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)"
---

# RLHF — Reinforcement Learning from Human Feedback

## Зачем это нужно: разрыв между pre-training и alignment

Pre-trained LLM оптимизирована на **next token prediction** — она предсказывает наиболее вероятное продолжение текста из интернета. Но пользователю нужно не «наиболее вероятное продолжение», а **полезный, честный и безвредный ответ**. Эту разницу называют *alignment gap*.

Конкретный пример: если спросить GPT-3 «Объясни квантовую механику пятилетнему ребёнку», модель может продолжить текст как статья в Википедии, переключиться на другую тему, или повторить вопрос — потому что всё это встречается в обучающих данных. RLHF закрывает этот разрыв: учит модель *следовать инструкциям*, а не просто продолжать текст.

**Ключевой результат InstructGPT** (Ouyang et al., 2022, OpenAI): модель с 1.3B параметров после RLHF предпочтительнее 175B GPT-3 по оценке людей — **в 100 раз меньше**, но значительно лучше следует инструкциям.

## Трёхступенчатый pipeline: SFT → RM → PPO

RLHF pipeline состоит из трёх последовательных этапов. Каждый следующий строится на результатах предыдущего.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/instructgpt/chatgpt-training-pipeline.png]]
*Полный pipeline обучения ChatGPT/InstructGPT: SFT на демонстрациях → обучение Reward Model на сравнениях → оптимизация через PPO (источник: Chip Huyen)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/fig1.png]]
*Figure 1 из DPO (Rafailov et al., 2023): сравнение RLHF pipeline (слева) и DPO (справа). RLHF требует отдельной Reward Model + RL (PPO); DPO оптимизирует предпочтения напрямую*

### Step 1: Supervised Fine-Tuning (SFT)

**Цель:** научить модель формату «вопрос → ответ» вместо «продолжи текст».

**Данные:** ~13,000 демонстраций от labeler'ов — они пишут эталонные ответы на промпты из API. Промпты разнообразные: генерация текста, QA, суммаризация, код, brainstorming.

**Процесс:**
1. Берём pre-trained GPT-3
2. Fine-tune на демонстрациях через стандартный supervised learning (cross-entropy loss)
3. 16 эпох, cosine LR decay, dropout 0.2
4. Обучаем три размера: **1.3B, 6B, 175B**

Почему SFT недостаточен? Демонстраций мало (~13K), и они покрывают только малую часть пространства промптов. Модель выучивает формат, но не «что такое хороший ответ» в общем случае. Нужен способ передать модели *предпочтения* людей на масштабе.

### Step 2: Reward Model (RM) Training

**Цель:** обучить модель-оценщик, которая заменит людей. RM принимает (prompt, response) и выдаёт скалярную оценку качества.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/instructgpt/reward-model-hf.png]]
*Обучение Reward Model: люди ранжируют ответы, модель учится предсказывать эти ранжирования (источник: Hugging Face)*

**Сбор данных:**
1. На каждый промпт генерируется K = 4-9 ответов от SFT модели
2. Labeler'ы **ранжируют** ответы от лучшего к худшему (не ставят абсолютные оценки — сравнивать проще, чем оценивать)
3. Каждое ранжирование K ответов даёт $\binom{K}{2}$ попарных сравнений
4. ~33,000 промптов → 300K-1.8M training examples

**Loss function** (попарный cross-entropy, Bradley-Terry model):

$$\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

где $y_w$ — preferred completion, $y_l$ — dispreferred, $r_\theta(x, y)$ — скалярный reward.

**Почему именно так?** Интуиция: если RM правильно ранжирует пары, разница скоров $r_\theta(x, y_w) - r_\theta(x, y_l)$ должна быть положительной. Sigmoid + log превращают это в дифференцируемый loss.

**Критический трюк:** все $\binom{K}{2}$ сравнений из одного промпта идут в один batch element (не перемешиваются). Это предотвращает overfitting к отдельным парам — модель видит полный ранжинг сразу.

**Размер RM:** используется **6B RM**, не 175B — крупная RM нестабильна как value function в RL.

### Step 3: PPO Fine-Tuning

**Цель:** оптимизировать SFT модель, чтобы она генерировала ответы с высоким reward, не уходя далеко от исходной модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/instructgpt/rlhf-pipeline-hf.png]]
*RL этап: модель генерирует ответы, Reward Model оценивает, PPO обновляет веса политики (источник: Hugging Face)*

**RL формулировка:**
- **Policy (политика)** = языковая модель, генерирующая текст
- **Action space** = словарь токенов (~50K)
- **Observation** = промпт + уже сгенерированные токены
- **Reward** = оценка от RM + штраф за отклонение от SFT модели

**Objective** (уравнение 2 из InstructGPT):

$$\text{objective}(\phi) = \mathbb{E}_{(x, y) \sim \pi_\phi^{RL}} \left[ r_\theta(x, y) - \beta \cdot \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)} \right] + \gamma \cdot \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \pi_\phi^{RL}(x) \right]$$

Три компонента:
1. **$r_\theta(x, y)$** — reward от RM: «насколько хорош ответ»
2. **$\beta \cdot \text{KL}(\pi^{RL} \| \pi^{SFT})$** — KL penalty: не даёт модели уйти далеко от SFT модели. Без этого модель найдёт «хаки» — бессмысленные строки, которые обманывают RM (reward hacking)
3. **$\gamma \cdot \text{pretrain\_loss}$** — pretraining mix: подмешивание оригинального language modeling loss. Предотвращает деградацию на стандартных NLP бенчмарках

**PPO vs PPO-ptx:**
- **PPO** = без pretraining mix → деградация на SQuAD, HellaSwag, DROP, WMT
- **PPO-ptx** = с pretraining mix → деградация устранена без потери preference scores

**Почему PPO, а не другие RL алгоритмы?** PPO — trust region метод: ограничивает размер шага обновления через clipped loss, что предотвращает катастрофические обновления. Для LLM это критически важно — одно плохое обновление может разрушить модель.

## Ключевые результаты InstructGPT

| Метрика | Результат |
|---------|-----------|
| **Helpfulness** | 1.3B PPO-ptx > 175B GPT-3 по human preference |
| **Preference vs GPT-3** | 175B InstructGPT preferred **85 ± 3%** времени |
| **Truthfulness** | ~2x чаще truthful & informative на TruthfulQA |
| **Hallucinations** | 21% (InstructGPT) vs 41% (GPT-3) |
| **Toxicity** | ~25% меньше токсичных outputs |
| **Alignment tax** | PPO-ptx устраняет regression на NLP benchmarks |

**Генерализация:** модель справляется с code summarization и non-English инструкциями, несмотря на их редкость в training data.

## Что не так с RLHF: проблемы и ограничения

**Reward hacking.** Модель может найти способы получать высокий reward без реального улучшения качества. Например, генерировать длинные, уклончивые ответы — labeler'ы часто предпочитают более длинные ответы, и RM это усваивает. KL penalty частично решает проблему, но не полностью.

**Сложность pipeline.** В памяти нужно держать до 4 моделей одновременно: policy, reference policy, reward model, value function. Для 175B это сотни GPU.

**Человеческий bias.** Alignment к предпочтениям конкретной группы из ~40 labeler'ов (Upwork/ScaleAI + OpenAI researchers), а не к «человеческим ценностям» вообще. Inter-rater agreement ~73% — люди сами не согласны друг с другом.

**Парадокс галлюцинаций.** Chip Huyen отмечает, что RLHF может *увеличить* галлюцинации: модель учится генерировать уверенные, «хорошо звучащие» ответы, что иногда конфликтует с фактической точностью.

**Нестабильность обучения.** PPO чувствителен к гиперпараметрам ($\beta$, learning rate, batch size). Маленькие изменения могут привести к collapse или к reward hacking.

## Развитие RLHF: от InstructGPT к ChatGPT и далее

### Iterative RLHF (LLaMA 2)

Meta расширила pipeline до итеративного процесса:
1. Собрать preference data → обучить RM → PPO → получить лучшую модель
2. **Повторить** с новой моделью: собрать новые preference data, обучить новый RM
3. Каждая итерация улучшает качество (5 итераций в LLaMA 2-Chat)

Плюс **Rejection Sampling**: генерируем K ответов, берём лучший по RM — проще PPO, работает как bootstrap.

### Constitutional AI (Anthropic)

Заменяет часть human feedback на **AI feedback**: модель сама оценивает ответы по набору принципов (конституция). Это [[02 Areas/ML & DL/Concepts/Training/Constitutional AI|RLAIF]] — Reinforcement Learning from AI Feedback.

### DPO: RLHF без RL

[[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] (Rafailov et al., 2023) показал, что можно убрать Reward Model и PPO целиком, оптимизируя предпочтения напрямую через classification loss. Математически эквивалентен RLHF, но радикально проще в реализации. Стал стандартом alignment для open-source LLM в 2023-2024.

## Сравнение RLHF pipeline с альтернативами

| Аспект | RLHF (PPO) | DPO | RLAIF |
|--------|-----------|-----|-------|
| **Reward Model** | Отдельная, обучаемая | Имплицитная (в policy) | AI-generated labels |
| **RL loop** | Да (PPO) | Нет | Да или нет |
| **Моделей в памяти** | 4 | 2 | 2-4 |
| **Стабильность** | Чувствителен к гиперпараметрам | Стабильнее | Зависит от метода |
| **Гибкость** | Максимальная (online) | Ограничена offline data | Scalable |
| **Сложность** | Высокая | Низкая | Средняя |

## Почему это важно

RLHF — **ключевой рецепт ChatGPT и всех последующих aligned LLM** (Claude, LLaMA 2-Chat, Gemini, GPT-4):

1. **Alignment gap:** pre-trained LM оптимизирована на next token prediction, не на «follow instructions helpfully». RLHF закрывает этот разрыв.
2. **Scale inversion:** 1.3B aligned > 175B unaligned — alignment важнее размера модели.
3. **PPO-ptx pattern:** стал стандартным «align without forgetting» рецептом.
4. **Эволюция:** от InstructGPT через iterative RLHF (LLaMA 2) к DPO и RLAIF — каждый шаг упрощает pipeline.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — замена PPO на closed-form optimization
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — SFT на инструкциях (Step 1 pipeline)
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — общий принцип адаптации моделей
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — базовый этап перед RLHF
- [[02 Areas/ML & DL/Concepts/Training/Constitutional AI|Constitutional AI]] — RLAIF от Anthropic
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — базовая модель InstructGPT
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA 2|LLaMA 2]] — iterative RLHF

## Дополнительные ресурсы

- [Hugging Face — Illustrating RLHF](https://huggingface.co/blog/rlhf) — лучшие диаграммы pipeline
- [Chip Huyen — RLHF](https://huyenchip.com/2023/05/02/rlhf.html) — глубокий разбор с практическими нюансами
- [Sebastian Raschka — RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — сравнение с DPO и другими методами
