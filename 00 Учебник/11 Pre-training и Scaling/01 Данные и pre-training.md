---
title: Данные и предобучение
type: textbook-chapter
status: canonical
last_updated: 2026-10-26
last_verified: 2026-10-26
source_unit_id:
  - meeting-04-slides-olmo-pretraining
  - lecture-13-section-2-lecture-13-data-i
  - lecture-13-figure-step-12-rendering-1
  - lecture-13-figure-step-33-rendering-1
  - lecture-13-figure-step-35-rendering-1
  - lecture-13-figure-step-37-rendering-1
  - lecture-13-copyright-consent-failure-modes
  - lecture-13-figure-step-71-rendering-1
  - lecture-13-section-92-intellectual-property-law
  - lecture-13-section-111-licenses
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_13.py
  - https://arxiv.org/abs/2407.21783
  - https://arxiv.org/abs/2501.00656
---

# Данные и предобучение языковой модели

<a id="corpus-as-decision-system"></a>
<!-- source_unit_id: lecture-13-section-2-lecture-13-data-i -->

Две языковые модели с одинаковой архитектурой могут по-разному писать код,
решать задачи и отвечать на вопросы, даже если на их обучение потрачено
одинаковое число вычислений. Одно из объяснений — различие в обучающих данных.
Модель, которая часто видела законченные программы с комментариями, обучалась
на других связях между токенами, чем модель, которой преимущественно
показывали новости и рекламные страницы. Число токенов само по себе этого
различия не описывает.

При подготовке корпуса приходится решить, какие документы собирать, как
извлекать из них текст, что удалять и как часто использовать оставшееся.
Эти решения определяют не только тематику данных. Ошибка извлечения может
разорвать формулу или перепутать порядок абзацев; дубликаты — многократно
увеличить частоту одного фрагмента; чрезмерно строгий фильтр — исключить
целую разновидность текстов. Поэтому работу с корпусом нужно рассматривать
как часть обучения, а не только как загрузку файлов перед его началом.

## Как документ влияет на обучение

После извлечения текста и токенизации документ превращается в
последовательность $(x_1,\ldots,x_T)$. Каузальная языковая модель учится
предсказывать каждый следующий токен по предыдущим:

$$
\mathcal L=-\frac{1}{T-1}\sum_{t=1}^{T-1}
\log p_\theta(x_{t+1}\mid x_{\le t}).
$$

Для основного предобучения не требуется, чтобы человек написал отдельный
правильный ответ к каждому примеру: целевой токен уже находится в тексте.
Но функция потерь не различает достоверное объяснение и уверенно написанную
ошибку. Она также не знает, что меню сайта, повторённое на тысячах страниц,
добавляет меньше новой информации, чем тысяча разных статей. Различать такие
случаи приходится при подготовке данных.

Рассмотрим упрощённый корпус из 100 равных по длине документов. Если одна
статья присутствует в нём десять раз, при равномерном выборе документа на неё
приходится 10% примеров. После удаления девяти копий останется 91 документ,
и её доля составит примерно 1,1%. Объём уменьшился лишь на 9%, а частота
именно этой статьи — почти в девять раз. У документов разной длины доля
токенов и доля документов будут различаться; результат также зависит от
правила выборки.

Это пример изменения **веса в обучающих данных**, а не утверждение, что
полученная модель будет ровно в девять раз чаще воспроизводить статью.
Градиенты зависят от текущих параметров, контекста и порядка примеров.
Повторение может помогать усвоению полезного материала, но случайные копии
не должны незаметно определять, чему уделяется большая часть обучения.

## Сохранённый корпус и обучающий поток

Словом «данные» часто называют три разных объекта. Сначала есть исходные
материалы: например, архив HTTP-ответов, выгрузка энциклопедии и репозитории
кода. Из них получают **корпус документов** — извлечённые и отобранные тексты.
Затем из корпуса выбирают примеры, токенизируют их и составляют пакеты,
которые действительно поступают в модель. Это **обучающий поток**.

| Этап | Что происходит с одной веб-страницей | Что можно проверить |
|---|---|---|
| Получение | Сохраняются HTML, адрес и время загрузки | Из какого ответа сервера получен текст |
| Извлечение | Выделяется основная статья, удаляется навигация | Не потеряны ли код, таблицы и порядок абзацев |
| Отбор | Применяются языковые и другие фильтры, ищутся копии | По какой причине документ оставлен или удалён |
| Формирование смеси | Задаётся частота выбора его источника | Сколько токенов каждого типа попадёт в обучение |
| Подготовка пакетов | Текст токенизируется, разбивается или упаковывается | Где границы документов и на каких позициях считается loss |

Пусть после очистки сохранено 900 млрд токенов веб-текста и 100 млрд токенов
кода. Их физическое соотношение — 90:10. Если при обучении выделить коду 30%
из бюджета в 500 млрд токенов, модель увидит около 150 млрд токенов кода.
Это соответствует полутора проходам по кодовому подкорпусу, если внутри него
выборка равномерна. Веб-текста она увидит около 350 млрд токенов — меньше
одного полного прохода. Таким образом, «корпус размером один триллион токенов»
ещё не сообщает ни длительность обучения, ни фактическое соотношение тем.

Состав и частоты определяют содержание обучения; загрузчик должен доставить
выбранные примеры без незапланированных пропусков и повторов. Техническая
сторона этого перехода разобрана в
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|главе о загрузке данных, padding и packing]].
Далее нас интересует предшествующая работа: какие тексты следует передать
этому загрузчику и почему.

<a id="stages-and-data-regimes"></a>
<a id="один-жизненный-цикл-разные-режимы-данных"></a>
## Как меняются данные по ходу обучения

<!-- source_unit_id: lecture-13-figure-step-12-rendering-1 -->
<!-- source_unit_id: lecture-13-figure-step-33-rendering-1 -->
<!-- source_unit_id: lecture-13-figure-step-35-rendering-1 -->
<!-- source_unit_id: lecture-13-figure-step-37-rendering-1 -->

**Предобучение (pre-training)** обычно начинается со случайных весов и
большой разнородной коллекции текста и кода. На этом этапе модель осваивает
язык, фактические связи и способы продолжения самых разных документов.
Однако умение продолжить текст ещё не означает умения вести диалог в нужном
формате. После строки с вопросом базовая модель может написать ответ,
следующий вопрос или продолжение тестового задания — всё это могло встречаться
в её корпусе.

**Промежуточное обучение (mid-training)** продолжает обучение уже полученной
модели на специально подобранной смеси. Например, можно увеличить долю
решённых математических задач и отобранных научных текстов. Во многих рецептах
цель остаётся прежней — предсказывать следующий токен; меняются данные и
расписание скорости обучения. Отдельная стадия позволяет проверить несколько
смесей, начиная с одних и тех же сохранённых весов, без повторения всего
дорогого предобучения. Название не задаёт единого алгоритма или обязательной
длительности этапа.

**Дообучение поведения (post-training)** использует примеры взаимодействия:
запросы и ответы, пары ответов с предпочтением, задачи с проверяемым результатом.
При обучении на демонстрациях текст вопроса служит условием, а потери обычно
считают на ответе ассистента. При обучении по предпочтениям важно уже не просто
воспроизвести один текст, а изменить относительные вероятности разных ответов.
Это иные способы использовать данные, хотя сама модель по-прежнему выдаёт
последовательность токенов.

Границы стадий не строгие. Решённая задача может быть обычным документом в
предобучении, примером для SFT или основой задания с автоматически проверяемым
ответом. Различие определяется тем, **как пример используется в обучении**,
а не только расширением файла или тематикой текста. Такое разделение стадий
принято в [Stanford CS336, Lecture 13](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_13.py#L45-L86).

## Пример: основной корпус OLMo 2 и поздняя смесь Dolmino

В отчёте [OLMo 2, §2.1](https://arxiv.org/html/2501.00656v1#S2.SS1)
опубликован состав основного корпуса. Приведённые ниже величины описывают
сохранённые данные, а не число предъявлений модели. «B» в оригинале означает
миллиард, «T» — триллион токенов.

| Источник | Тип данных | Объём, млрд токенов |
|---|---|---:|
| DCLM baseline | Отобранные веб-страницы | 3 710 |
| StarCoder | Код | 83,0 |
| peS2o | Научные тексты | 58,6 |
| arXiv | Научные статьи | 20,8 |
| OpenWebMath | Математические веб-тексты | 12,2 |
| Algebraic Stack | Математический код | 11,8 |
| Wikipedia и Wikibooks | Энциклопедические и учебные тексты | 3,7 |

Итого — около 3,90 трлн токенов; из-за округления строк сумма может немного
отличаться от указанного в статье итога. Веб-текст составляет более 95%
корпуса. Это даёт широкий охват, но не означает, что такая же пропорция
оптимальна на всех стадиях обучения.

Для позднего этапа авторы собрали Dolmino. Из исходного веб-корпуса они
выбрали более строгим фильтром подмножество в 752 млрд токенов, добавили
научные и энциклопедические тексты, вопросы с ответами и математические
данные. Затем из доступных источников составили несколько обучающих смесей.
В варианте примерно на 50 млрд токенов доли были следующими:

| Источник | Доля в обучающей смеси |
|---|---:|
| Отфильтрованный DCLM | 47,2% |
| FLAN после проверки пересечений с тестами | 16,6% |
| StackExchange Q&A | 2,45% |
| peS2o | 5,85% |
| Wikipedia и Wikibooks | 7,11% |
| Dolmino Math | 20,8% |

*Числа из таблицы 13 [OLMo Team et al., 2 OLMo 2 Furious](https://arxiv.org/html/2501.00656v1#S4.SS3), CC BY 4.0. Округлённые доли дают 100,01%.*

Размер веб-подкорпуса здесь всё ещё намного больше размера математического,
но в обучающем потоке математика занимает уже около пятой части токенов.
Для вариантов на 100 и 300 млрд токенов некоторые небольшие источники
использовались повторно. Это намеренное увеличение их веса, в отличие от
случайного сохранения зеркал одной страницы.

<a id="olmo-two-stage-recipe"></a>
### OLMo 2: the two-stage training recipe

<!-- source_unit_id: meeting-04-slides-olmo-pretraining -->

На схеме из лекции Hanna Hajishirzi скорость обучения после разогрева
уменьшается по косинусному расписанию. При переходе к поздней стадии
меняются смесь данных и расписание: скорость линейно снижается до нуля.
От одного исходного чекпойнта можно запустить несколько таких продолжений
и сравнить смеси, не повторяя основную стадию.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/pretraining/m04-p147-two-stage-training.png|OLMo: warmup, длинная косинусная стадия и короткое линейное снижение до нуля]]

*Hanna Hajishirzi, [Open Training Recipes for Reasoning in Language Models](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf#page=147), Berkeley Advanced LLM Agents, 24 February 2025, physical page 147. Полный исходный слайд, без crop или перерисовки. Это схематическое расписание: длины отрезков не пропорциональны числу токенов, а параметры не одинаковы для всех OLMo 2.*

Полное объяснение двух стадий из
[OLMo Team et al., §2.3](https://arxiv.org/html/2501.00656v1#S2.SS3)
сохранено ниже на английском по
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

<!-- BEGIN SOURCE: OLMo 2 2501.00656v1 recipe -->
<!-- source-olmo:S2.SS3.SSS0.Px1.p1.1 -->
> Departing from previous OLMo versions, OLMo 2 models are randomly initialized from a truncated
> normal distribution with a mean of 0 and a standard deviation of 0.02 (see section
> [3.2](https://arxiv.org/html/2501.00656v1#S3.SS2)). After that, we run a learning rate
> schedule that warms up the learning rate from 0 to the peak learning rate (a hyperparameter)
> over 2000 steps, followed by a cosine decay calibrated to reach 10% of the peak learning
> rate after 5T tokens. For the 7B variant, we truncate the schedule at 4T tokens and then
> begin the second stage. As the 13B variant ran with a higher learning rate from the start,
> we finish the cosine decay at 5T tokens before starting the second stage. In this stage,
> we train on broadly web-based data (see [Table 1](https://arxiv.org/html/2501.00656v1#S2.T1)).

<!-- source-olmo:S2.SS3.SSS0.Px2.p1.1 -->
> In the second stage, we train on the Dolmino Mix 1124
> ([Section §4](https://arxiv.org/html/2501.00656v1#S4)). This mix is smaller, but contains
> higher-quality text, as well as synthetic data to boost key abilities with substantial room
> for improvement after the pretraining stage. For example, early versions of the OLMo 2
> model underperformed on GSM8K relative to peer models, so we curated new data to target
> this. In this stage, we linearly decay the learning rate to zero over the length of the run.

<!-- source-olmo:S2.SS3.SSS0.Px2.p2.1 -->
> To get the most out of this high-quality data, and to find a better local minimum, we
> perform this step multiple times with different random data orders, and then average the
> resulting models ([Wortsman et al., 2022](https://arxiv.org/html/2501.00656v1#bib.bib123)).
> For the 7B variant, we anneal three separate times for 50B tokens each, with different
> randomized data orders; we average the resulting models to produce the final model.
> For the 13B variant, we train three separate times for 100B tokens each (same number of
> update steps as the 7B), and then a fourth time for 300B tokens. The final model is the
> average of all four models. For further details, refer to [Section §4](https://arxiv.org/html/2501.00656v1#S4).
<!-- END SOURCE: OLMo 2 2501.00656v1 recipe -->

The slide's peak of $3\cdot10^{-4}$ corresponds to the 7B configuration.
The 13B peak in the report is $9\cdot10^{-4}$; both use a 2,000-step warmup
([Table 4](https://arxiv.org/html/2501.00656v1#S2.T4)). With 1,024 sequences
of 4,096 tokens per step, the 7B warmup consumes 8,388,608,000 tokens:

$$
\begin{aligned}
1024\times4096&=4{,}194{,}304,\\
2000\times4{,}194{,}304&=8{,}388{,}608{,}000.
\end{aligned}
$$

Doubling the batch
for 13B doubles this count to about 16.78B. Thus the slide's “<10B tokens”
label is not a bound for every model size. Likewise, “99% / 1%” on the next
slide is a teaching approximation: §2.1 describes the main stage as at least
90% of training FLOPs and mid-training as 5–10%.

The report also uses rounded token counts in different places: §2.1 gives
3.90T tokens for the 7B main stage, whereas §2.3 and the evaluation table
describe it as 4T. Neither figure is the 5T horizon used to calibrate the
cosine schedule. A stored corpus size, the number of tokens consumed by a
run, and a schedule horizon need not coincide.

### Independent continuations and checkpoint averaging

Starting from the same pretrained 7B checkpoint, let three runs consume
different permutations of the 50B-token Dolmino mix. Each run has its own
optimizer trajectory and decays its learning rate to zero. The three final
parameter arrays have the same shapes and corresponding coordinates, so
they can be averaged element by element:

$$
\theta_{\mathrm{final}}=
\frac{\theta^{(1)}+\theta^{(2)}+\theta^{(3)}}{3}.
$$

For an illustrative scalar coordinate, final values 0.8, 1.0 and 1.2 give
an averaged value of 1.0. This arithmetic does not prove that the average
model will perform better: the evidence comes from evaluating the resulting
weights. These checkpoints share an initialization and training history;
the recipe does not establish that averaging unrelated pretrained models
will work. In [Table 14](https://arxiv.org/html/2501.00656v1#S4.T14), six
mixtures are each compared using their best single checkpoint and the
three-checkpoint average. Gains are not uniform across every metric: for
mix E, GSM* falls from 60.5 to 43.0 despite gains on the other columns.

The 7B continuation budget is $3\times50\mathrm{B}=150\mathrm{B}$ tokens
across all three branches. No branch is a 150B-token sequential continuation.
For 13B, three 100B branches and one 300B branch consume 600B tokens in total;
the final model averages four parameter arrays with equal weights. The
300B branch does not receive three times the averaging weight.

The comparison below includes new data, additional updates, a new learning-rate
schedule and checkpoint averaging. To isolate the effect of the data mix,
the starting checkpoint, training budget, schedule, averaging procedure
and evaluation protocol would have to be held fixed.

### What changed after mid-training

The following two tables retain every score from
[OLMo 2, Table 9](https://arxiv.org/html/2501.00656v1#S4.T9), also shown on
Berkeley slide 151. The split follows the source's development/held-out
boundary; “Avg” is the source's reported aggregate, not the arithmetic mean
of the six development columns displayed here. These are base models,
before SFT, DPO or RLVR.

| Checkpoint | Avg | MMLU | ARC-C | HSwag | WinoG | NQ | DROP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7B · pretraining | 50.6 | 59.8 | 72.6 | 81.3 | 75.8 | 29.0 | 40.7 |
| 7B · pretraining & mid-training | 61.2 | 63.7 | 79.8 | 83.8 | 77.2 | 36.9 | 60.8 |
| 13B · pretraining | 56.5 | 63.4 | 80.2 | 84.8 | 79.4 | 34.6 | 49.6 |
| 13B · pretraining & mid-training | 66.8 | 67.5 | 83.5 | 86.4 | 81.5 | 46.7 | 70.7 |

| Checkpoint | AGIEval | GSM8K | MMLU-Pro |
|---|---:|---:|---:|
| 7B · pretraining | 44.6 | 24.1 | 27.4 |
| 7B · pretraining & mid-training | 50.4 | 67.5 | 31.0 |
| 13B · pretraining | 48.2 | 37.3 | 31.2 |
| 13B · pretraining & mid-training | 54.2 | 75.1 | 35.1 |

*OLMo Team et al., 2 OLMo 2 Furious, v1, Table 9, CC BY 4.0.
Original numbers and checkpoint order; formatting split into two tables.*

For 7B, the GSM8K difference is 43.4 percentage points, while MMLU changes
by 3.9 points. These task-specific differences would be hidden by reporting
only the aggregate. Natural Questions and DROP use F1, so the numerical
columns are not all accuracy measurements. The authors evaluated full
held-out tasks after development; a separate 200-question GSM* subset
had informed mixture selection (§2.4). The microanneal experiments below
explain how they screened mathematical sources with that development set.

The last slide in this Berkeley sequence compares historical OLMo releases
with Llama 3, but does not plot Qwen 2.5 despite mentioning it in the
headline. For a documented comparison with Qwen 2.5, use
[Table 6 of the report](https://arxiv.org/html/2501.00656v1#S2.T6), with its
named model sizes, metrics and estimated training budgets. These 2024–2025
results do not establish a current ranking.

[[05 Источники/Papers/OLMo 2 Dolmino|OLMo 2: original artifacts, data inventories and recipe]].

Для OLMo 2-Instruct далее использовали рецепт Tülu 3: демонстрации ответов,
предпочтения и задачи с проверяемой наградой. Таблица источников демонстраций
в [отчёте Tülu 3](https://arxiv.org/abs/2411.15124) описывает уже запросы и
ответы, а не просто объёмы веб-корпусов. Подробно этот переход разобран в
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|главе об SFT и данных инструкций]].

Не все отчёты позволяют восстановить такую последовательность решений.
Например, раздел о данных в [The Llama 3 Herd of Models, §3.1](https://arxiv.org/abs/2407.21783)
описывает очистку и дедупликацию, но одного этого описания недостаточно,
чтобы собрать тот же корпус. Для воспроизведения нужны ещё исходные объекты,
версии обработки и фактические частоты выборки.

## Почему очистка данных влияет на устойчивость обучения

Ошибочные или малополезные данные не только занимают место. Иногда они
меняют динамику оптимизации. При исследовании нестабильности OLMo авторы
обнаружили, что в пакетах с резкими скачками нормы градиента часто
встречались длинные повторения коротких последовательностей: например,
обрывки закодированных данных или ряды одинаковых чисел.

В подготовку корпуса добавили удаление документов, содержащих не менее
32 повторений n-граммы длиной от 1 до 13 токенов. Дополнительно загрузчик
распознавал такие участки и исключал соответствующие позиции из расчёта
потерь. Ниже — исходные графики эксперимента с этим дополнительным
маскированием, опубликованные авторами.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/corpus-intro-review-2026-10/olmo2-without-repetition-filter.svg]]

*Без исключения повторяющихся участков из loss: после начального снижения
нормы градиента остаются частые резкие скачки. По горизонтали — шаг обучения,
по вертикали — L2-норма градиента в логарифмическом масштабе. OLMo Team et al.,
[рис. 3, левая панель](https://arxiv.org/html/2501.00656v1#S3.F3), CC BY 4.0; оригинал без изменений.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/corpus-intro-review-2026-10/olmo2-with-repetition-filter.svg]]

*С маскированием повторяющихся участков: скачков заметно меньше, но они
не исчезли полностью. Масштабы осей те же. OLMo Team et al.,
[рис. 3, правая панель](https://arxiv.org/html/2501.00656v1#S3.F3), CC BY 4.0; оригинал без изменений.*

Из этого опыта не следует, что любое повторение вызывает сбой. В отчёте
прямо отмечено: одна и та же последовательность могла сопровождаться скачком
при одном порядке данных и не сопровождаться при другом. Маскирование также
не устранило медленный рост нормы градиента, для которого потребовались
другие изменения рецепта. Практический вывод уже и полезнее: при скачке
нужно сохранять идентификаторы пакета и проверять его содержимое, а не
искать причину только в скорости обучения и численном формате.

## Как проверить, что новый корпус действительно лучше

Фильтр почти всегда уменьшает объём. Само по себе это не говорит, стал ли
корпус лучше: можно удалить испорченные страницы, а можно — редкие, но нужные
примеры. Проверку начинают с просмотра случайных сохранённых и удалённых
документов, отдельно по языкам и типам источников. Общая доля отбраковки
скрывает случаи, когда фильтр хорошо работает на новостях и плохо — на коде.

Затем сравнивают обучение на разных вариантах данных. Если одна модель
видела вдвое больше токенов, результат не позволяет отделить пользу корпуса
от пользы дополнительных вычислений. Для содержательного сравнения
фиксируют модель, токенизатор, бюджет и настройки обучения; меняют только
проверяемую часть подготовки данных. Небольшие предварительные запуски
помогают отсеять неудачные варианты, но их результат не гарантированно
сохраняется при увеличении масштаба.

Наконец, часть задач оставляют для итоговой проверки. Если после каждого
изменения смотреть на один и тот же тест и выбирать лучший результат,
тест постепенно становится средством настройки. В OLMo 2 авторы явно
разделили задачи, использованные при разработке, и отложенные проверки
([§2.4](https://arxiv.org/html/2501.00656v1#S2.SS4)). Даже без прямого попадания
тестовых ответов в корпус многократный подбор смеси по тесту мешает честно
оценить обобщение.

### Короткие продолжения одного чекпойнта

Для проверки нового источника необязательно начинать обучение со случайных
весов. В OLMo 2 использовали *microanneals*: короткие продолжения уже
предобученной модели с уменьшением скорости обучения. В
[§4.2.2 отчёта](https://arxiv.org/html/2501.00656v1#S4.SS2.SSS2)
авторы приводят следующий рецепт; английский текст сохранён по CC BY 4.0:

> 1. identify a source or small collection of math sources that we want to assess the data quality of;
> 2. collect roughly the same quantity of data from the general data mix (e.g., DCLM) as from the math sources to ensure a mixture of high-quality web text alongside domain-specific math;
> 3. train this 50/50 mixture as if it were an annealing run, making sure to linearly drive the learning rate down at the proper rate for this smaller collection of data.

Предположим, нужно сравнить два источника, в каждом по 200 млн токенов.
Для каждого готовят отдельную смесь с 200 млн веб-токенов и обучают
свою копию одного исходного чекпойнта на 400 млн токенов.
Если скорость линейно уменьшается от $\eta_0$ до нуля за весь запуск,
то после 100 млн токенов она равна $0{,}75\eta_0$, после 200 млн —
$0{,}5\eta_0$. Обрезать первые 400 млн токенов длинного запуска
с неизменённым расписанием — другой опыт: модель закончит его при
другой скорости обучения.

Отдельные копии весов нужны и для сравнения данных. Если сначала
обучить на источнике A, а потом на B, влияние B измеряется уже после
обучения на A. Для независимого сравнения обе ветви начинают с одного
чекпойнта; у них совпадают бюджет, веб-компонент, расписание и оценивание.
Перестановки и повторные запуски помогают оценить, не зависит ли
преимущество от одного порядка примеров.

В исследовании OLMo 2 проведены 19 microanneals суммарно на 130 млрд
токенов — меньше трёх полных продолжений по 50 млрд. Это сравнение
объёма обучения на одной модели, не универсальная оценка стоимости
такого поиска. Для выбора математических данных авторы использовали
GSM*: 200 вопросов из GSM8K, выделенных для разработки, и следили
за MMLU. Эти 200 вопросов уже не являлись независимым итоговым тестом.

В таблице 12 меняются не только сами данные, но и их объёмы, доли
и число повторений. Поэтому разницу между её строками нельзя целиком
приписать одному фильтру или повторению. Метод даёт недорогой способ
сравнивать продолжения конкретной предобученной модели; он не заменяет
проверку полного рецепта при целевом бюджете и не показывает, как тот же
источник обучит модель со случайной инициализацией.

<a id="sources-rights-provenance"></a>
## Доступность, разрешение и происхождение — не одно и то же

<!-- source_unit_id: lecture-13-copyright-consent-failure-modes -->
<!-- source_unit_id: lecture-13-figure-step-71-rendering-1 -->
<!-- source_unit_id: lecture-13-section-92-intellectual-property-law -->
<!-- source_unit_id: lecture-13-section-111-licenses -->

Вместе с исходными текстами сохраняют сведения о том, откуда они получены
и на каких условиях доступны. Нужны не только URL и название коллекции,
но и дата загрузки, версия источника, лицензия или иное основание
использования. Условия доступа к сайту, правила автоматической загрузки
и условия повторного распространения текста — разные сведения; их не
стоит заменять одним полем «открытые данные».

| Сведения | Для чего сохранять |
|---|---|
| URL или идентификатор, дата получения, контрольная сумма | Найти именно тот исходный объект, из которого получен пример |
| Лицензия и условия источника на дату получения | Проверять допустимые способы использования и распространения |
| Версии извлечения, нормализации и фильтров | Воспроизвести текст и решение о его включении |
| Причины удаления или маскирования | Проверить ошибки отбора и изменения между версиями |
| Связь документа с токенизированными файлами | Найти затронутые обучающие примеры |

Условия доступа могут измениться уже после формирования корпуса.
[Longpre et al., Consent in Crisis (2024)](https://arxiv.org/abs/2407.14933)
исследовали изменения robots.txt и условий использования на доменах,
представленных в C4, RefinedWeb и Dolma. Это историческое наблюдение о
выбранных сайтах, а не оценка текущих условий каждого источника. В частности,
прогнозную часть их графика нельзя читать как измеренные значения за 2025 год.

Эта документация нужна и для обычной технической работы. Если обнаружена
ошибка извлечения, по идентификатору исходного объекта можно восстановить
документ, исправить обработку и определить затронутые версии корпуса.
Если хранить только перемешанные токены без связи с источниками, такую
проверку провести значительно труднее. Правовое решение о конкретном
источнике требует отдельной проверки применимых условий; запись
метаданных сама по себе не даёт разрешения на использование.

<a id="route-through-corpus"></a>
## Дальнейший маршрут

Следующая глава разбирает
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|сбор, извлечение текста, фильтры качества и выбор смеси]].
За ней следует
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса|поиск дубликатов, защита персональных данных и проверка пересечений с тестами]].
Отдельно рассматриваются
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|синтетические данные]]
и [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|распределение вычислительного бюджета]].
Применить эти решения можно в
[[02 Areas/ML & DL/06 Практика/23 Собрать воспроизводимый pretraining corpus|практике по сборке воспроизводимого корпуса]].

Связанные темы: [[02 Areas/ML & DL/01 Справочник/Текст и токенизация/Токенизация|токенизация]],
[[02 Areas/ML & DL/01 Справочник/Evaluation/Evaluation|оценивание моделей]],
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Llama|семейство Llama]].

**Назад:** [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|Transformer с нуля — формы, параметры и стоимость]].
**Дальше:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Сбор, очистка и смеси данных]].
