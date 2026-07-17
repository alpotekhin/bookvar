---
title: Reward modeling
type: textbook-chapter
status: canonical
last_updated: 2026-07-17
aliases: [Reward model, Модель вознаграждения]
prerequisites: ["[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data]]"]
next: "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM]]"
primary_sources:
  - https://arxiv.org/abs/2203.02155
  - https://arxiv.org/abs/2403.13787
---

# Reward modeling

> [!abstract] Результат урока
> Вы сможете объяснить архитектуру sequence reward model, вывести и вручную
> вычислить Bradley–Terry loss, обучить её на chosen/rejected pairs, отличить RM
> от value model, outcome/process verifier и LLM judge и диагностировать reward
> hacking до запуска дорогого RL.

## От сравнений к функции, которую можно оптимизировать

Preference dataset знает, что для prompt $x$ ответ $y_w$ предпочли $y_l$. Но
PPO или best-of-N нужны оценки новых, ранее не размеченных ответов. Reward model
$r_\phi(x,y)$ аппроксимирует preference signal одним числом.

Число не является «истинным качеством ответа». Оно полезно только относительно
других ответов из близкого распределения. Если RM дала 8.4, это не означает
84% правильности и не гарантирует сопоставимость между двумя prompts.

## Сквозной пример

Для TCP/UDP у нас есть:

- $y_w$: 76 слов, корректные гарантии и примеры;
- $y_l$: 54 слова, но ложное «UDP всегда быстрее».

Модель должна дать $r_w>r_l$. Она не получает reason code напрямую: если
training data систематически связывает корректность с длиной, RM может выучить
shortcut «длиннее — лучше», а не сетевую семантику.

## 1. Архитектура sequence reward model

![[02 Areas/ML & DL/raw/papers/instructgpt/images/reward-model-hf.png]]

*Как читать схему: language model создаёт candidates для одних и тех же
prompts; люди упорядочивают candidates; reward model учится воспроизводить
относительный порядок скалярными scores. Стрелка к reward model означает
обучение на comparisons, а не передачу «истинной оценки качества». Иллюстрация:
Hugging Face, визуальный разбор RLHF.*

Типичный RM начинает с pretrained/SFT Transformer. Prompt и response соединяют
тем же chat template:

```text
<system>Ты технический ассистент.</system>
<user>Объясни TCP и UDP...</user>
<assistant>TCP устанавливает...</assistant><eos>
```

Transformer выдаёт hidden state $h_t$ на каждой позиции. Линейная head
превращает состояние последнего содержательного token/EOS в scalar:

$$r_\phi(x,y)=w^\top h_{\mathrm{EOS}}+b.$$

Почему EOS? Его representation через causal attention содержит весь предыдущий
prompt и answer. Важно корректно найти последний **не padding** token. Ошибка
`hidden[:, -1]` при right padding обучит head на PAD representations.

```text
prompt + answer → tokenizer → Transformer → h₁ … h_EOS → Linear( d, 1 ) → 0.73
```

*Эта схема пересобрана по объяснениям Lambert, глава Reward Modeling, и pipeline
InstructGPT. Она показывает архитектуру, а не утверждает, что scalar является
абсолютной оценкой.*

## 2. Bradley–Terry model

Пусть вероятность того, что человек предпочитает $y_w$, зависит от разности
latent rewards:

$$
P(y_w\succ y_l\mid x)
=\frac{e^{r_w}}{e^{r_w}+e^{r_l}}
=\sigma(r_w-r_l).
$$

Для observed winner минимизируем negative log-likelihood:

$$
\mathcal L_{\mathrm{RM}}(\phi)
=-\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
$$

В loss входит **разность**. Если прибавить к обоим scores 100, вероятность не
изменится. Поэтому абсолютный offset reward не идентифицируем из парных данных.

### Ручной расчёт: неправильный порядок

Пусть в начале $r_w=-0.4$, $r_l=0.6$. Тогда $\Delta=-1$:

$$P(w\succ l)=\sigma(-1)=0.269,$$

$$\mathcal L=-\ln(0.269)=1.313.$$

Производная по gap:

$$\frac{\partial\mathcal L}{\partial\Delta}=\sigma(\Delta)-1=-0.731.$$

Gradient descent увеличит $\Delta$: поднимет winner относительно loser.

После обучения допустим $r_w=1.4$, $r_l=0.2$, $\Delta=1.2$:

$$P=0.769,\qquad \mathcal L=0.263.$$

Loss стал меньше, но 1.4 не является «оценкой 1.4 из 10». Значимы порядок и
margin в том распределении, где RM проверена.

### Ties и soft labels

Если A выбрали 2 из 3 annotators, target можно хранить как $q=2/3$ и обучать
binary cross-entropy:

$$-q\log\sigma(\Delta)-(1-q)\log(1-\sigma(\Delta)).$$

Это сохраняет неопределённость. Другой вариант — отдельная tie model или
исключение ties; выбор следует документировать.

## 3. Один batch в коде

Chosen и rejected кодируют отдельно, но пропускают через одну RM с общими
весами:

```python
import torch
from torch.nn import functional as F

def last_token_index(attention_mask):
    # Работает и с left padding, и с right padding: берём максимальную
    # фактическую позицию, на которой mask == 1.
    positions = torch.arange(
        attention_mask.size(1), device=attention_mask.device
    ).expand_as(attention_mask)
    return positions.masked_fill(attention_mask == 0, -1).max(dim=1).values

def score(model, reward_head, ids, mask):
    h = model(ids, attention_mask=mask).last_hidden_state
    idx = last_token_index(mask)
    final_h = h[torch.arange(h.size(0), device=h.device), idx]
    return reward_head(final_h).squeeze(-1)

r_w = score(backbone, head, chosen_ids, chosen_mask)
r_l = score(backbone, head, rejected_ids, rejected_mask)
loss = -F.logsigmoid(r_w - r_l).mean()
loss.backward()
```

Для efficiency chosen/rejected можно объединить по batch dimension. Следите,
чтобы prompts и templates совпадали: RM должна сравнивать ответы на один context.

### Тест игрушечного batch

```python
rw = torch.tensor([-0.4, 1.4])
rl = torch.tensor([ 0.6, 0.2])
print((-F.logsigmoid(rw - rl)))  # примерно [1.313, 0.263]
```

Если training implementation не воспроизводит эти числа, до GPU-run дело
доходить не должно.

## 4. От ranking к loss

Для ranking $A\succ B\succ C\succ D$ можно обучаться на всех шести pairs. Тогда
один prompt даст шесть gradient terms и будет весить сильнее prompt с одной
парой. InstructGPT нормировал comparisons внутри prompt, чтобы избежать
переобучения на связанных парах. Альтернативы:

- одна случайная пара на epoch;
- только соседние пары;
- средний loss внутри prompt, затем среднее по prompts;
- listwise Plackett–Luce objective.

Это статистическая часть модели, а не только dataloader detail.

## 5. Что именно может выдавать reward

### Outcome reward model (ORM)

Один score за полный ответ. Удобен для preferences и финальной правильности, но
не сообщает, где рассуждение свернуло не туда.

### Process reward model (PRM)

Оценивает отдельные шаги решения. Для математики может помогать search и credit
assignment, но требует step-level labels и определения границ шага. PRM может
поощрять правдоподобно выглядящие объяснения вместо истинной логики.

### Verifiable reward

Unit tests, exact answer checker, compiler или proof checker дают программный
сигнал. Это уже RLVR: reward не обязан быть learned RM. Верификатор точен только
относительно спецификации; модель может эксплуатировать слабый тест.

### Generative reward model / LLM-as-a-judge

Модель читает rubric, рассуждает и выбирает ответ или генерирует score. Она
гибче scalar head и может объяснить решение, но дороже и подвержена prompt,
position, verbosity и self-preference biases.

### RM и value model — не одно и то же

RM оценивает качество полного $(x,y)$ по preferences. Value model в PPO
предсказывает ожидаемый будущий return из текущего prefix, чтобы уменьшить
variance policy gradient. У них могут быть похожие scalar heads, но разные
targets и роли.

## 6. Evaluation до RL

### Pairwise accuracy

$$\mathrm{Acc}=\frac1N\sum_i
\mathbf 1[r(x_i,y_{w,i})>r(x_i,y_{l,i})].$$

Она понятна, но скрывает easy/hard composition. Разбивайте accuracy по domain,
reason code, длине, source policy и agreement strength.

### Swap test

Перестановка A/B не должна менять смысл решения. Особенно важна для generative
judges. Отдельно измеряйте positional flip rate.

### Challenge sets

Нужны пары:

- краткий правильный против длинного ошибочного;
- вежливый небезопасный против сухого безопасного;
- одинаковый content с разным Markdown;
- correct final answer с неверным reasoning;
- ответ с фальшивыми ссылками против честного признания неопределённости.

RewardBench — пример набора, который оценивает chat, reasoning и safety
preferences и показывает: высокая средняя accuracy не гарантирует устойчивость
во всех категориях.

### Best-of-N smoke test

Сгенерируйте $N$ ответов, выберите максимум RM и отдайте пары человеку. Если при
росте $N$ RM score растёт, а human quality падает, обнаружен overoptimization.
Такой тест дешевле полноценного PPO и часто раньше показывает reward hacking.

## 7. Reward hacking и distribution shift

RM обучена различать candidates из $\pi_{collect}$. Оптимизируемая policy ищет
области, где RM ошибается, и тем самым меняет distribution. Это активный
adversary, даже если никто не планировал атаку.

Для нашего примера RM могла заметить, что хорошие annotator answers чаще
содержат «компромисс» и «гарантия». Policy научится вставлять эти слова без
правильного объяснения. Proxy reward возрастёт, человеческая оценка — нет.

Защита многослойна:

1. разнообразные hard negatives и challenge sets;
2. on-policy пересбор preferences;
3. ensemble/uncertainty и human audits top-scoring samples;
4. KL regularization policy относительно reference;
5. ранняя остановка по gold human eval;
6. разделение verifiable criteria и субъективного preference score.

KL ограничивает уход policy, но не превращает RM в истину. Lilian Weng приводит
примеры, где proxy–gold gap остаётся и при регуляризации: Goodhart нельзя решить
одним коэффициентом.

## 8. Failure cases

| Симптом | Причина | Что проверить |
|---|---|---|
| RM выбирает длинное | verbosity shortcut | length-matched challenge pairs |
| train acc 99%, eval 60% | duplicates или policy shift | group split, source policy |
| scores дрейфуют | offset/scale не закреплены | margins и calibration set |
| разные prompts несопоставимы | pairwise loss только внутри prompt | не трактовать global score буквально |
| красивый wrong answer выигрывает | authority/style bias | adversarial pairs |
| PPO резко эксплуатирует RM | слабое покрытие tails | best-of-N и on-policy relabeling |

## Практикум

1. Для gaps $-2,0,2$ вычислите $P(w\succ l)$, loss и derivative.
2. Из ranking четырёх ответов создайте pairs и сравните вес prompt при sum и
   within-prompt mean loss.
3. Реализуйте reward head с корректным last non-padding index для left и right
   padding.
4. Создайте 12 challenge pairs на verbosity, position и polished-wrong bias.
5. Проведите best-of-4 и best-of-16. Сравните RM score с blind human ranking.
6. Объясните, почему прибавление 50 ко всем rewards не меняет RM loss, но scale
   rewards может изменить последующее RL.

## Курсы и объяснения

- Nathan Lambert, [Reward Modeling](https://rlhfbook.com/c/05-reward-models) —
  Bradley–Terry, architecture, margin/k-wise variants, ORM/PRM/value/judges.
- Stanford CS224R, [The Post-Training Frontier](https://cs224r.stanford.edu/slides/09_cs224r_rlhf_2026.pdf),
  слайды 36–40 — pairwise calibration, RM loss и переход к KL-RLHF.
- Chip Huyen, [Reward model](https://huyenchip.com/2023/05/02/rlhf.html) —
  график $-\log\sigma(\Delta)$ и доступная интерпретация score difference.
- Hugging Face, [Illustrating RLHF](https://huggingface.co/blog/rlhf) —
  визуальная схема rankings → scalar RM.
- Lilian Weng, [Reward Hacking in Reinforcement
  Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) —
  proxy/gold gap и overoptimization.

## Первичные источники

- Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155), Section 3.2.
- Stiennon et al., [Learning to summarize from human
  feedback](https://arxiv.org/abs/2009.01325), 2020.
- Lambert et al., [RewardBench](https://arxiv.org/abs/2403.13787), 2024.
- Bai et al., [Training a Helpful and Harmless Assistant with
  RLHF](https://arxiv.org/abs/2204.05862), 2022.
