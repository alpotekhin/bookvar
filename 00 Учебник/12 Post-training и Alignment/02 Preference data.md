---
title: Preference data
type: textbook-chapter
status: canonical
last_updated: 2026-07-17
aliases: [Данные предпочтений, Human feedback]
prerequisites: ["[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data]]"]
next: "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling]]"
primary_sources:
  - https://arxiv.org/abs/2203.02155
  - https://arxiv.org/abs/2204.05862
---

# Preference data

> [!abstract] Результат урока
> Вы сможете спроектировать comparison task, превратить ranking в обучающие
> пары, оценить согласие разметчиков и объяснить, почему chosen/rejected — это
> результат конкретной rubric и population, а не объективные «хорошо/плохо».

## Почему не попросить человека написать идеальный ответ

Написать хороший технический ответ трудно. Сравнить два готовых ответа обычно
легче: видны фактическая ошибка, лишняя длина и нарушение формата. Preference
data использует эту асимметрию. Модель генерирует несколько candidates, а человек
или judge выбирает лучший.

Но слово *предпочтение* опасно своей простотой. Один человек любит краткость,
другой — объяснения; эксперт замечает сетевую ошибку, неспециалист — нет. Данные
описывают решения определённых annotators по определённой инструкции.

## Сквозной comparison task

Prompt тот же:

> Объясни разницу между TCP и UDP начинающему. Приведи по одному примеру и
> уложись в 90 слов.

**Ответ A** точно описывает гарантии, приводит HTTP и звонок и занимает 76 слов.

**Ответ B** написан проще, но говорит: «UDP всегда быстрее TCP» и занимает 54
слова.

Если rubric требует прежде всего factual correctness, A должен победить. Если
разметчику показать лишь «какой ответ вам больше нравится?», часть людей выберет
B за краткость. Качественная строка поэтому хранит не только пару:

```json
{
  "prompt_id": "net-0042",
  "prompt": "Объясни разницу между TCP и UDP...",
  "responses": [
    {"id": "A", "text": "TCP устанавливает...", "policy": "sft-v3"},
    {"id": "B", "text": "TCP медленный, а UDP всегда быстрее...", "policy": "sft-v3"}
  ],
  "ranking": ["A", "B"],
  "label": {"winner": "A", "loser": "B", "tie": false},
  "rubric": "correctness > instruction_following > clarity > style",
  "annotator_id": "hashed-781",
  "position_order": ["B", "A"],
  "reason_codes": ["B_FACTUAL_ERROR"],
  "timestamp": "2026-07-17"
}
```

`position_order` позволяет измерять positional bias. Версия policy нужна для
анализа on-policy gap. Reason code не обязан входить в loss, но помогает
аудитировать поведение и строить специализированные reward models.

## 1. Откуда берутся candidates

Обычно для prompt $x$ сэмплируют $K$ ответов из текущей или близкой policy:

$$y_i\sim\pi_{\text{collect}}(\cdot\mid x),\qquad i=1,\ldots,K.$$

Temperature и top-p меняют сложность разметки. При слишком низкой temperature
ответы почти одинаковы; при слишком высокой человек выбирает между очевидно
плохими текстами, и данные мало говорят о тонкой границе качества.

**On-policy** candidates получены от версии модели, которую собираются улучшать.
Они показывают её реальные ошибки. Старый публичный preference dataset может
быть off-policy: новая модель почти никогда не породила бы его rejected answers,
а значит обучение тратит сигнал на нерелевантную область.

Полностью on-policy сбор дорог и быстро устаревает. Практический компромисс —
периодически пересэмплировать candidates и хранить policy/version/config.

## 2. Rating, pairwise comparison и ranking

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/post-training/instructgpt-labeling-ui-002.png]]

*Реальный ranking interface InstructGPT: ответы сначала скрыты в верхней
области, затем annotator раскладывает их по рангам. Это не абстрактная метка
`chosen`: результат зависит от набора одновременно показанных candidates,
порядка карточек и инструкции разметчику. Источник: Ouyang et al., 2022,
Figure 12; изображение извлечено из локальной копии paper без уменьшения.*

### Абсолютный rating

Просьба поставить ответу 1–7 кажется информативной, но шкалы людей различаются:
для одного «5» — хороший ответ, для другого — едва приемлемый. Rating полезен,
когда важен threshold или несколько независимых dimensions, но требует
калибровки.

### Pairwise comparison

Annotator отвечает, $A\succ B$, $B\succ A$ или tie. Это простой интерфейс и
естественный формат для Bradley–Terry loss. Однако одна пара не показывает,
насколько сильна разница.

### Ranking K ответов

Полный порядок $A\succ C\succ B\succ D$ содержит больше сигнала за один prompt,
но когнитивно тяжелее. Его часто разворачивают в пары:

$$
(A,C),(A,B),(A,D),(C,B),(C,D),(B,D).
$$

Это шесть строк, но не шесть независимых человеческих решений: все получены из
одного ranking. Если случайно разделить их между train и test, получится leakage.
Split следует делать по prompt/ranking group.

Можно брать только соседние пары $(A,C),(C,B),(B,D)$: меньше данных, зато они
труднее и не дают очевидным крайностям доминировать.

## 3. Как preference превращается в likelihood

Предположим, скрытое качество ответов равно $r_A$ и $r_B$. Bradley–Terry model
задаёт вероятность выбора A:

$$P(A\succ B)=\sigma(r_A-r_B)=\frac{1}{1+e^{-(r_A-r_B)}}.$$

Это пока не reward-model training, а модель наблюдения: annotators чаще выбирают
ответ с более высоким latent utility, но решение остаётся вероятностным.

### Ручной пример

Если $r_A=1.2$, $r_B=0.3$, то разность $0.9$ и

$$P(A\succ B)=\sigma(0.9)\approx0.711.$$

Даже при таком разрыве модель допускает выбор B с вероятностью 29%. Если 100
независимых разметчиков выбрали A 71 раз, это не обязательно «шум»: результат
согласуется с моделью. Preference labels не следует трактовать как безошибочные
ground-truth классы.

При трёх голосах A, A, B эмпирическая вероятность A равна $2/3$. Максимально
правдоподобный score gap удовлетворяет

$$\sigma(\Delta)=2/3\Rightarrow\Delta=\log\frac{2/3}{1/3}=\log2\approx0.693.$$

Так агрегированные голоса могут хранить силу и неопределённость предпочтения,
вместо принудительного hard label.

## 4. Rubric: что означает «лучше»

Rubric должна задавать порядок критериев и конфликтные случаи. Например:

1. безопасность и соблюдение policy;
2. фактическая корректность;
3. выполнение явных ограничений prompt;
4. полнота, релевантность и ясность;
5. стиль.

Без приоритетов annotator не знает, выбрать ли точный длинный ответ или короткий
ответ с небольшой ошибкой. Полезны эталонные примеры, пограничные пары и кнопка
`tie / both bad / cannot judge`. Запрет ties заставляет людей выдумывать сигнал.

Для экспертных областей нужна маршрутизация. Красивый ответ о TCP может победить
в общей разметке и проиграть сетевому инженеру. Annotator population является
частью определения целевой функции.

## 5. Bias и качество разметки

### Position bias

Первый или последний ответ выбирают чаще. Рандомизируйте порядок, храните его и
вставляйте контрольные дубликаты с переставленными сторонами.

### Verbosity bias

Длинный ответ выглядит обстоятельнее даже без дополнительной пользы. Нужны
пары равной корректности разной длины и отдельная проверка instruction limit.

### Authority и style bias

Уверенный тон, Markdown и ссылки могут маскировать ошибку. В audit set должны
быть polished-but-wrong и plain-but-correct examples.

### Sycophancy

Ответ, соглашающийся с ложной предпосылкой пользователя, может казаться
приятнее. Rubric должна предпочитать корректное несогласие.

### Judge-model bias

LLM judges масштабируют сбор, но имеют собственные position/style/self-family
biases. AI feedback — не бесплатная человеческая разметка. Нужна человеческая
калибровочная выборка и периодическое сравнение judge с экспертами.

## 6. Измерение согласия

Простой agreement для двух annotators — доля совпавших labels. Он завышается,
если почти всегда побеждает A. Cohen's $\kappa$ вычитает ожидаемое случайное
совпадение:

$$\kappa=\frac{p_o-p_e}{1-p_e}.$$

Если наблюдаемое agreement $p_o=0.80$, а из marginal frequencies ожидается
$p_e=0.50$, то $\kappa=(0.8-0.5)/(1-0.5)=0.60$.

Низкое agreement не всегда означает плохих annotators. Возможно, candidates
действительно равноценны или rubric неоднозначна. Следует анализировать
disagreement по domain, reason code и annotator, а не выбрасывать всё одной
цифрой.

## 7. Code companion: rankings в пары без leakage

```python
from itertools import combinations

def ranking_to_pairs(row, adjacent_only=False):
    # ranking: response ids от лучшего к худшему
    ids = row["ranking"]
    pairs = zip(ids, ids[1:]) if adjacent_only else combinations(ids, 2)
    text = {r["id"]: r["text"] for r in row["responses"]}
    return [{
        "prompt": row["prompt"],
        "chosen": text[winner],
        "rejected": text[loser],
        "group_id": row["prompt_id"],
        "source_policy": row["responses"][0]["policy"],
    } for winner, loser in pairs]

# Split group_id, а не отдельные пары.
```

Перед экспортом проверьте exact duplicates, swapped contradictions
`(A>B)`/`(B>A)`, пустые ответы, одинаковые chosen/rejected и пересечение prompt
groups между train/eval.

## 8. Failure cases

| Ошибка pipeline | Что выучит следующая стадия |
|---|---|
| rejected всегда очевидно сломан | отличать мусор, но не хорошие ответы |
| A всегда стоит слева | позиционный shortcut |
| один judge без human calibration | вкусы и ошибки judge |
| нет ties | случайные labels на равных парах |
| все пары от старой policy | нерелевантные ошибки |
| ranking pairs разнесены по split | завышенная eval accuracy |
| rubric смешивает safety и style | непонятный scalar compromise |

## Практикум

1. Напишите rubric для prompt про TCP/UDP и разберите конфликт «точнее, но на 15
   слов длиннее».
2. Для ranking $B\succ A\succ D\succ C$ получите все и только соседние пары.
3. Разметьте 20 пар дважды с переставленным порядком. Измерьте flip rate.
4. Посчитайте agreement и $\kappa$ на игрушечной таблице из 30 labels.
5. Спроектируйте sampling policy, которая даёт не только easy negatives, но и
   близкие candidates.
6. Составьте пять polished-but-wrong пар для теста verbosity/authority bias.

## Курсы и объяснения

- Nathan Lambert, [Preference Data](https://rlhfbook.com/c/11-preference-data) —
  interfaces, rankings/ratings, multi-turn, sourcing и biases.
- Chip Huyen, [Reward model и UI comparison
  data](https://huyenchip.com/2023/05/02/rlhf.html) — конкретная строка HH-RLHF
  и наглядный пример несогласия с human label.
- Stanford CS224R, [The Post-Training Frontier](https://cs224r.stanford.edu/slides/09_cs224r_rlhf_2026.pdf),
  слайды 36–37 — почему абсолютные ratings плохо калиброваны и зачем pairwise.
- Nathan Lambert, [лекция 2](https://rlhfbook.com/teach/course/lec2-chap4-5-9/slides.pdf)
  — связь instruction data, comparisons, RM и rejection sampling.

## Первичные источники

- Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155), Sections 3–4
  и Appendix B: selection labelers, demonstrations и rankings.
- Bai et al., [Training a Helpful and Harmless Assistant with
  RLHF](https://arxiv.org/abs/2204.05862), 2022.
- Christiano et al., [Deep Reinforcement Learning from Human
  Preferences](https://arxiv.org/abs/1706.03741), 2017.
