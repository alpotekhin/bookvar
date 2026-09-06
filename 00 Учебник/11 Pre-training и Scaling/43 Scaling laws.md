---
title: Законы масштабирования
type: textbook-chapter
status: canonical
last_updated: 2026-09-06
source_unit_id:
  - lecture-09-scaling-history
  - lecture-09-data-scaling
  - lecture-09-mean-estimation-derivation
  - lecture-09-data-scaling-failures
  - lecture-09-engineering-scaling
  - lecture-09-joint-model-data-scaling
  - lecture-09-isoflops-experiment
  - lecture-09-train-inference-tradeoff
  - lecture-09-recap
  - lecture-11-minicpm-scaling
  - lecture-11-deepseek-scaling
  - lecture-11-optimizer-scaling
  - lecture-11-mup-derivation
  - lecture-11-mup-robustness
  - lecture-11-mup-failure-modes
  - lecture-11-recap
primary_sources:
  - https://arxiv.org/abs/2001.08361
  - https://arxiv.org/abs/2203.15556
  - https://arxiv.org/abs/2305.16264
  - https://arxiv.org/abs/2401.00448
  - https://arxiv.org/abs/2401.02954
  - https://arxiv.org/abs/2404.10102
  - https://arxiv.org/abs/2404.05728
  - https://arxiv.org/abs/2404.06395
  - https://arxiv.org/abs/2406.19146
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf
practice: "[[02 Areas/ML & DL/06 Практика/22 Провести scaling-law campaign.md]]"
---

# 43. Законы масштабирования

<a id="scaling-history"></a>

Предположим, что команда собирается обучить языковую модель и располагает
бюджетом в $3\cdot10^{22}$ операций. Этот бюджет можно потратить по-разному:
увеличить модель и показать ей меньше токенов или взять меньшую модель и обучать
её дольше. Проверить каждый вариант полным запуском невозможно, а привычное
«чем больше параметров, тем лучше» не отвечает на вопрос о распределении
фиксированного бюджета.

Законы масштабирования заменяют такой перебор серией контролируемых запусков
меньшего размера. По ним оценивают, как в заданном экспериментальном режиме
меняются потери при росте модели, данных и вычислений, а затем проверяют прогноз
на более крупном удержанном запуске. Это не физический закон и не таблица
вечных коэффициентов. Результат относится к конкретному семейству архитектур,
токенизатору, смеси данных, оптимизатору, расписанию и правилу подсчёта стоимости.

Главный предмет этой главы — не готовое отношение токенов к параметрам, а весь
путь от измерений до решения:

1. что именно измеряется и какие величины считаются масштабом;
2. какие условия необходимо сохранить между пробными запусками;
3. как при фиксированном бюджете найти компромисс между $N$ и $D$;
4. как выбрать форму закона, исследовать остатки и оценить неопределённость;
5. как удешевить серию экспериментов с помощью WSD и переноса по ширине;
6. почему train-optimal модель может не быть оптимальной для продукта.

<a id="data-scaling"></a>
## Степенная зависимость — наблюдение в заданном диапазоне

Для авторегрессионной модели обычно измеряют среднюю отрицательную
логарифмическую вероятность следующего токена на удержанной выборке:

$$
L=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

Сравнивать два значения $L$ можно только при одном токенизаторе и одном
проверочном распределении. Смена словаря меняет единицу измерения, а смена
выборки — сам вопрос, на который отвечает оценка. Плавное уменьшение средней
кросс-энтропии также не означает, что с той же скоростью вырастет точность на
каждом downstream-бенчмарке.

Во многих экспериментах график потерь против объёма данных близок к прямой в
логарифмических координатах. Это означает, что на исследованном интервале
остаток можно приближённо записать как $A D^{-\beta}$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/data-power-law.png]]

*На логарифмических осях степенная зависимость становится прямой. Здесь курс
показывает результат Kaplan et al. как эмпирическое наблюдение, а не как
доказательство неизменного показателя. Источник: [Stanford CS336, lecture 9,
page 15](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf),
по [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361).*

<a id="mean-estimation-derivation"></a>

Степенной темп сам по себе не загадочен. Например, если
$x_1,\ldots,x_n\sim\mathcal N(\mu,\sigma^2)$ и среднее оценивается как
$\hat\mu=n^{-1}\sum_i x_i$, то

$$
\mathbb E(\hat\mu-\mu)^2=\frac{\sigma^2}{n},
\qquad
\log \mathbb E(\hat\mu-\mu)^2=-\log n+2\log\sigma.
$$

Это тоже scaling law, но с известным механизмом и показателем. Для нейронной
языковой модели такой простой вывод отсутствует: наблюдаемые показатели заметно
отличаются и между задачами, и от классического $1/n$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mean-estimation-rate.png]]

*Игрушечный пример отделяет общий математический вид $n^{-\alpha}$ от причин,
по которым конкретное значение $\alpha$ возникает у языковой модели. Источник:
[Stanford CS336, lecture 9, page 17](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

Связь показателя с «внутренней размерностью» данных иногда предлагается как
объяснение, но сама лекция подчёркивает слабое место этого рассуждения: оценки
внутренней размерности нестабильны, а соответствие не является строгим выводом.
Поэтому наклон линии следует сначала воспринимать как оцениваемый параметр
эмпирической модели.

Для совместного влияния размера модели и данных Hoffmann et al. использовали
форму

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

Здесь $N$ — число параметров в заранее определённом соглашении, $D$ — число
предъявленных токенов, а $E$ — асимптотический сдвиг подгонки. Его нельзя без
дополнительных предпосылок отождествлять с истинной энтропией языка или с
неустранимой ошибкой любого другого семейства моделей. Небольшое изменение $E$
может заметно изменить $\alpha$ и $\beta$, особенно при узком диапазоне
пилотных запусков.

<a id="data-scaling-failures"></a>
## Что означает «объём данных»

В простейшей записи $D$ — число токенов, обработанных оптимизатором. Однако
миллиард новых токенов и миллиард повторно предъявленных токенов — не одно и то
же количество информации. Состав корпуса тоже нельзя свести к одному числу.

В показанном в лекции исследовании distribution shift менял прежде всего
вертикальный сдвиг кривой, а не её наклон. Это полезная модель для рассуждения о
ценности более разнообразных данных, но не универсальная теорема: при другом
семействе распределений может измениться и показатель.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/distribution-shift-offset.png]]

*Одинаковый наклон при разных сдвигах означает: объём данных даёт сходный темп
улучшения, но качество их распределения определяет другой уровень потерь.
Источник: [Stanford CS336, lecture 9, page 22](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

При конечном корпусе приходится повторять документы. Muennighoff et al.
показали, что в исследованном ими режиме до четырёх эпох повторение почти не
ухудшало loss относительно такого же числа уникальных токенов, но дальнейшие
проходы приносили всё меньше пользы. Их закон вводит эффективный объём данных,
который растёт медленнее номинального числа предъявленных токенов.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/repeated-data-effective-tokens.png]]

*На слайде номинальный объём разложен на уникальные токены и повторения, после
чего вводится эффективный объём $D'$. Читатель должен различать счётчик работы
оптимизатора и количество новой информации. Источник: [Stanford CS336,
lecture 9, page 24](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf),
по [Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264).*

Наконец, линия, полученная в compute-limited режиме, может перестать работать,
если неограниченно увеличивать только данные или только число шагов. Степенная
функция описывает участок кривой, а не обещает бесконечный запас улучшения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/finite-data-breakdown.png]]

*Прямолинейная экстраполяция нарушается после выхода из режима, в котором были
получены точки. Источник: [Stanford CS336, lecture 9, page 25](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

Таким образом, эксперимент должен фиксировать не только $D$, но и идентичность
корпуса, порядок смешивания источников, дедупликацию, число уникальных токенов и
число эпох. Без этих полей два запуска с одинаковым $D$ могут принадлежать
разным законам.

<a id="engineering-scaling"></a>
## Параметры и FLOPs тоже требуют соглашения

Для плотного Transformer часто используют приближение

$$
C\approx 6ND.
$$

Множитель 6 возникает из грубого подсчёта прямого и обратного проходов через
линейные слои. Обычно $N$ означает non-embedding parameters. Это важно: матрица
эмбеддингов может заметно увеличить число параметров, не создавая той же
зависимости вычислений и качества, что параметры блоков.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/parameter-count-convention.png]]

*Не все параметры одинаково входят в scaling law. Слайд отдельно показывает
эмбеддинги и напоминает о ещё более сложном соглашении для MoE. Источник:
[Stanford CS336, lecture 9, page 35](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

Формула $6ND$ не учитывает стоимость attention при длинном контексте, большой
словарь, activation recomputation, коммуникации, optimizer step, простои и
реальную загрузку ускорителя. У MoE необходимо отдельно указывать общее и
активное число параметров. Поэтому теоретические FLOPs, GPU-hours и деньги — три
разные меры. Переход между ними требует измеренного профиля, а не замены единиц
в таблице.

<a id="joint-model-data-scaling"></a>
## Почему при фиксированном бюджете появляется минимум

Зафиксируем $C$. Тогда рост $N$ вынуждает уменьшить
$D\approx C/(6N)$. Слишком маленькой модели не хватает ёмкости, хотя она видит
много токенов. Слишком крупная модель получает мало обновлений. Между этими
режимами возникает минимум проверочных потерь.

В работе Chinchilla этот компромисс оценивали тремя способами.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/chinchilla-three-methods.png]]

*Три метода отвечают на один вопрос разными статистическими процедурами. Уже
таблица результатов показывает, почему показатель нельзя превращать в
неизменную константу. Источник: [Stanford CS336, lecture 9, page 46](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf),
по [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556).*

### Метод 1: нижняя огибающая всех запусков

Из множества training curves выбирают наименьший loss, достигнутый при каждом
бюджете, и подгоняют степенную зависимость к этой огибающей. Метод использует
весь ход обучения, но чувствителен к тому, какие траектории и точки остановки
вообще попали в эксперимент.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/chinchilla-lower-envelope.png]]

*Нижняя огибающая должна быть составлена из сравнимых запусков. Если у них
разные schedules или качество настройки, линия смешивает масштабирование с
качеством optimization recipe. Источник: [Stanford CS336, lecture 9, page 47](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

<a id="isoflops-experiment"></a>
### Метод 2: IsoFLOP-кривые

Для каждого бюджета $C_i$ выбирают несколько размеров $N_{ij}$ и задают

$$
D_{ij}=\frac{C_i}{6N_{ij}}.
$$

По завершении запусков строят $L(N\mid C_i)$ и находят минимум
$(N_i^*,D_i^*)$. Хорошая сетка охватывает обе стороны минимума. Если лучшая
точка оказалась на границе, минимум не найден: диапазон нужно расширить, а не
проводить кривую через пограничное значение.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/chinchilla-isoflops.png]]

*Слева каждая линия имеет фиксированный бюджет; её минимум задаёт одну
compute-optimal пару. Центральный и правый графики проводят степенные законы
через последовательность минимумов. Источник: [Stanford CS336, lecture 9,
page 48](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf),
Figure 3 из [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556).*

Затем оценивают

$$
N^*(C)=k_NC^a,\qquad D^*(C)=k_DC^b.
$$

Если ограничение действительно имеет вид $C\propto ND$ и оптимум внутренний,
то $a+b\approx1$. Это полезная проверка согласованности, но не доказательство
правильности формы loss или подсчёта FLOPs.

### Метод 3: совместная поверхность

Вместо предварительного выбора минимумов можно подогнать
$E,A,B,\alpha,\beta$ по всем точкам сетки $(N,D)$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/chinchilla-joint-fit.png]]

*Совместная подгонка использует больше наблюдений, но сильнее зависит от формы
функции, весов точек и процедуры оптимизации. Источник: [Stanford CS336,
lecture 9, page 49](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

Подставив $D=C/(6N)$, получаем

$$
L(N\mid C)=E+A N^{-\alpha}+B\left(\frac{6N}{C}\right)^\beta.
$$

Условие нулевой производной даёт

$$
N^*(C)=
\left(\frac{\alpha A}{\beta B6^\beta}\right)^{\frac1{\alpha+\beta}}
C^{\frac{\beta}{\alpha+\beta}},
$$

а из $D^*=C/(6N^*)$ следует

$$
D^*(C)\propto C^{\frac{\alpha}{\alpha+\beta}}.
$$

Поэтому в пределах этой модели
$a=\beta/(\alpha+\beta)$ и $b=\alpha/(\alpha+\beta)$. Вывод верен только при
заданной аддитивной форме loss, точном ограничении $6ND=C$ и внутреннем
минимуме.

### Почему три аккуратных метода могут расходиться

Chinchilla сообщила близкий к симметричному рост $N$ и $D$, тогда как Kaplan et
al. получили более быстрый рост модели: примерно $N^*\propto C^{0.73}$ и
$D^*\propto C^{0.27}$. Разницу нельзя объяснить одной «неправильной цифрой».
На оценку влияют стоимость последнего слоя, диапазон малых моделей, warmup,
перенос learning rate и batch size, правило остановки и математическая форма.

Porian et al. воспроизвели режим Kaplan на OpenWebText2 и RefinedWeb. После
коррекции стоимости последнего слоя, длительности warmup и scale-dependent
настройки оптимизатора оценки приблизились к Chinchilla. Besiroglu et al.
переоценили данные третьего метода Chinchilla и также получили результат,
лучше согласующийся с первыми двумя методами.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/chinchilla-residuals.png]]

*Восстановление исходных точек и анализ остатков меняют вывод о третьем методе.
Гладкая линия без исходных наблюдений не позволяет увидеть такую проблему.
Источник: [Stanford CS336, lecture 9, page 53](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

## Scaling campaign начинается до регрессии

Серия запусков имеет смысл лишь тогда, когда её точки принадлежат одному
экспериментальному режиму. До расходования compute нужно зафиксировать:

- токенизатор, train/validation distributions и порядок смешивания данных;
- семейство архитектур, длину контекста и соглашение о числе параметров;
- optimizer, weight decay, clipping, dtype и initialization;
- правило выбора batch size, learning rate, warmup и decay;
- определение planned и actual compute, оборудование и MFU;
- seed, критерий завершённого запуска и обращение с NaN, timeout и preemption.

Не требуется заранее назначать универсальное число размеров модели или
располагать пилоты на фиксированных долях от целевого бюджета. Требуется другое:
несколько уровней compute, точки по обе стороны каждого наблюдаемого минимума,
резерв на уточнение сетки и хотя бы один масштаб, не участвующий в подгонке.

### Batch size — часть экспериментального режима

Для заданного target loss увеличение batch уменьшает шум градиента и может
сократить число шагов, но после некоторого масштаба требует всё больше примеров
ради небольшого сокращения шагов. В обозначениях лекции

$$
\frac{S}{S_{\min}}-1=
\left(\frac{E_{\mathrm{samples}}}{E_{\min}}-1\right)^{-1},
\qquad
B_{\mathrm{crit}}=\frac{E_{\min}}{S_{\min}}.
$$

$E_{\mathrm{samples}}$ здесь означает число обработанных примеров и не связано
с асимптотой $E$ в модели loss. Формула относится к фиксированным target loss и
training setup.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/critical-batch-tradeoff.png]]

*Critical batch определяется компромиссом между числом шагов и числом
обработанных примеров при достижении заданного loss. Источник: [Stanford CS336,
lecture 9, page 38](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf),
по McCandlish et al.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/critical-batch-vs-loss.png]]

*При более низком target loss оценённый critical batch растёт. Следовательно,
один batch size нельзя объявить оптимальным для всей траектории и всех
масштабов. Источник: [Stanford CS336, lecture 9, page 39](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

### Learning rate нельзя переносить молча

При стандартной параметризации оптимальный learning rate может зависеть от
ширины. Сохранить его постоянным можно только как проверяемую гипотезу или как
следствие заранее выбранной scale-aware parametrization.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/learning-rate-scaling.png]]

*Лекция сопоставляет наблюдаемую зависимость learning rate от масштаба с
попытками сделать перенос предсказуемым. Источник: [Stanford CS336, lecture 9,
page 40](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

Если одна модель завершает cosine decay, а другую останавливают посередине того
же абсолютного расписания, сравнение измеряет и размер, и различную фазу
schedule. Поэтому правило изменения learning rate должно быть привязано к
запланированной длительности каждого запуска либо заменено расписанием,
которое допускает сопоставимые ответвления.

## Подгонка, остатки и неопределённость

IsoFLOP-подход сначала оценивает минимум каждой кривой, затем строит регрессию
для $\log N_i^*$ и $\log D_i^*$. Совместная модель использует все точки сразу.
Ни один метод нельзя оценивать только по красивой линии на log–log plot.

Минимальный анализ включает:

1. исходные точки и обозначение завершённых, неустойчивых и прерванных запусков;
2. проверку, что минимумы не лежат на границе сетки;
3. несколько начальных приближений нелинейного оптимизатора;
4. не менее двух разумных форм подгонки;
5. остатки против $N$, $D$, $C$ и порядка запуска;
6. sensitivity analysis после исключения малого бюджета или влиятельной точки;
7. интервал для целевого $N^*$, $D^*$ и loss;
8. прогноз на заранее удержанном бюджете.

Логарифмирование удобно для визуализации, но меняет структуру ошибки. Если шум
измеряется в единицах loss, линейная регрессия после логарифмирования не
обязательно соответствует правдоподобной likelihood model. Стоит сравнить её с
нелинейной подгонкой в исходном пространстве и явно записать веса точек.

Для uncertainty можно повторить часть конфигураций с разными seed и
бутстрэпировать целые IsoFLOP-группы. Соседние точки одной кривой нельзя без
оглядки считать независимыми: они разделяют данные, код, schedule и часто одну
training trajectory. При экстраполяции интервал должен учитывать не только
случайный seed, но и выбор формы закона.

Полный протокол вынесен в практику
[[02 Areas/ML & DL/06 Практика/22 Провести scaling-law campaign.md]]. В ней
прогноз фиксируется до открытия удержанного результата: это защищает от
незаметной подстройки истории после получения ответа.

<a id="minicpm-scaling"></a>
## Как удешевить кампанию: WSD

Обычный cosine schedule связывает состояние модели с заранее известной точкой
завершения. Если нужны хорошие checkpoints после разных объёмов данных, каждый
из них приходится обучать по собственной траектории от начала. При нескольких
размерах модели и нескольких $D$ стоимость сетки растёт примерно как число
комбинаций двух осей.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/minicpm-cosine-campaign-cost.png]]

*При отдельных cosine-запусках каждый новый endpoint требует новой полной
траектории. Источник: [Stanford CS336, lecture 11, page 13](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf),
по [MiniCPM](https://arxiv.org/abs/2404.06395).*

Warmup–Stable–Decay разделяет обучение на короткий разогрев, длинный участок с
почти постоянным learning rate и завершающий decay. От состояния на stable-фазе
можно запустить несколько независимых decay-ветвей и получить сравнимые
endpoints для разных $D$ без повторения всей ранней траектории.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/minicpm-wsd-branching.png]]

*Разветвление после stable-фазы превращает одну длинную траекторию в несколько
кандидатных точек завершения. Источник: [Stanford CS336, lecture 11, page 14](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf),
по [MiniCPM](https://arxiv.org/abs/2404.06395).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/minicpm-wsd-loss-dynamics.png]]

*В экспериментах MiniCPM loss уменьшался медленнее на stable-фазе и быстро — во
время завершающего decay, занимавшего около 10% соответствующей траектории.
Это наблюдение конкретной работы, а не универсальная доля расписания. Источник:
[Stanford CS336, lecture 11, page 15](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

WSD экономит вычисления кампании, но не делает полученные точки автоматически
сопоставимыми с произвольным cosine run. Нужно проверять, одинаково ли работает
decay на нескольких размерах, и фиксировать точку ветвления, форму и длительность
каждой ветви.

MiniCPM совмещает WSD с переносом гиперпараметров по ширине и затем подгоняет
совместную модель $L(N,D)$. Авторы получают гораздо более data-heavy режим, чем
классический Chinchilla, но именно это различие показывает зависимость результата
от recipe, а не существование нового универсального отношения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/minicpm-joint-scaling-fit.png]]

*Совместная подгонка MiniCPM завершает последовательность «стоимость отдельных
endpoint → WSD-ветвление → наблюдаемая динамика → scaling fit». Источник:
[Stanford CS336, lecture 11, page 18](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf),
по [MiniCPM](https://arxiv.org/abs/2404.06395).*

<a id="deepseek-scaling"></a>
## Другой путь: непосредственно масштабировать гиперпараметры

DeepSeek LLM не использовал μP как исходное допущение. Авторы сначала провели
grid search по batch size и learning rate на малых масштабах, затем подогнали их
зависимость от compute, а после этого выполнили IsoFLOP-анализ model/data
allocation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/deepseek-lr-batch-grid.png]]

*Сетка отвечает на вопрос, где находится область близких к оптимуму
гиперпараметров, а не только какая единственная точка оказалась лучшей.
Источник: [Stanford CS336, lecture 11, page 20](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf),
по [DeepSeek LLM](https://arxiv.org/abs/2401.02954).*

В этой кампании были получены recipe-specific зависимости

$$
\eta_{\mathrm{opt}}=0.3118C^{-0.1250},
\qquad
B_{\mathrm{opt}}=0.2920C^{0.3271}.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/deepseek-hyperparameter-fits.png]]

*Курс прямо отмечает, что fit learning rate выглядит сомнительно. Эту оговорку
нельзя отрезать от графика или заменить формулой без остаточной диагностики.
Источник: [Stanford CS336, lecture 11, page 21](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

В DeepSeek модельным масштабом служат non-embedding FLOPs per token, а не просто
число параметров. Поэтому коэффициенты нельзя переносить на другую архитектуру
подменой $C$ в формуле. Они показывают процедуру: найти область near-optimal
конфигураций, выбрать функциональную форму, подогнать её и сохранить сомнения,
которые видны на исходных точках.

Затем та же работа использует многоступенчатое расписание с быстрым warmup и
двумя этапами уменьшения learning rate и строит IsoFLOP-кривые.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/deepseek-isoflop-curves.png]]

*После настройки гиперпараметров фиксированные compute budgets сравнивают
разные model/data allocations. Источник: [Stanford CS336, lecture 11, page 23](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf),
по [DeepSeek LLM](https://arxiv.org/abs/2401.02954).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/deepseek-loss-prediction.png]]

*Последний шаг — сравнить прогноз scaling law с loss большой модели. Хорошее
совпадение является проверкой данного диапазона и recipe, а не лицензией на
неограниченную экстраполяцию. Источник: [Stanford CS336, lecture 11, page 24](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

<a id="optimizer-scaling"></a>
## Сравнение оптимизаторов: как получить ложного победителя

Оптимизаторы нельзя сравнивать при одном learning rate или при равном числе
шагов без проверки их собственных optima. Даже форма scaling function для
batch и learning rate остаётся эмпирическим выбором: critical-batch law,
степенная функция compute и зависимость от $D$ отвечают на разные вопросы.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/optimizer-search-surface.png]]

*Полная поверхность показывает одновременно область устойчивости и минимум.
Один удачный запуск не характеризует оптимизатор. Источник: [Stanford CS336,
lecture 11, page 34](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

Первый источник ложного вывода — неодинаковый tuning budget. Если AdamW,
Muon или другой optimizer получает сетку, подобранную под конкурента, сравнение
измеряет качество настройки.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/optimizer-unfair-tuning.png]]

*Разным оптимизаторам нужны разные гиперпараметры и, возможно, разные правила
их масштабирования. Источник: [Stanford CS336, lecture 11, page 39](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

Второй источник — compute и Chinchilla ratio. Преимущество, наблюдаемое на
маленькой переобученной модели, может исчезнуть на compute-optimal траектории.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/optimizer-compute-ratio-confound.png]]

*Сравнение следует повторять по масштабу и model/data ratio, иначе эффект
optimizer смешивается с положением точки относительно compute-optimal frontier.
Источник: [Stanford CS336, lecture 11, page 40](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

Наконец, правдоподобная линия на малых масштабах может буквально взорваться при
экстраполяции. Такой запуск нельзя удалять как «неудачный seed»: он опровергает
область применимости процедуры.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/optimizer-extrapolation-failure.png]]

*Неудачная крупная точка важнее красоты малой регрессии: она показывает, что
параметризация или scaling rule требуют пересмотра. Источник: [Stanford CS336,
lecture 11, page 41](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

<a id="mup-derivation"></a>
## μP: перенос по ширине и его границы

Maximum Update Parametrization пытается выбрать initialization и learning-rate
scales так, чтобы при изменении ширины сохранялись масштабы активаций и
обновлений. Интуицию удобно разобрать на глубокой линейной сети

$$
h_l=W_lh_{l-1},\qquad
W_l\in\mathbb R^{n_l\times n_{l-1}}.
$$

Упрощённый вывод в лекции начинается с двух требований:

- отдельные координаты активаций при инициализации имеют порядок $\Theta(1)$,
  поэтому $\lVert h_l\rVert_2=\Theta(\sqrt{n_l})$;
- после одного шага изменение отдельной координаты также остаётся
  $\Theta(1)$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-invariants.png]]

*Два инварианта задают цель параметризации: при росте ширины представления не
должны исчезать или взрываться, а один шаг должен оставаться содержательным.
Источник: [Stanford CS336, lecture 11, page 46](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

Если элементы $W_l$ имеют гауссово распределение со стандартным отклонением
$\sigma_l$, то спектральная норма имеет порядок
$\sigma_l(\sqrt{n_{l-1}}+\sqrt{n_l})$. Чтобы согласовать нормы соседних
активаций, лекция выбирает

$$
\sigma_l=
\frac{\sqrt{n_l}}
{\sqrt{n_{l-1}}(\sqrt{n_l}+\sqrt{n_{l-1}})}
=\Theta\!\left(
\frac{1}{\sqrt{n_{l-1}}}
\min\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)
\right).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-initialization-derivation.png]]

*Знак приблизительного равенства в этом рассуждении опирается на верхнюю оценку
через spectral norm; слайд прямо называет вывод worst-case. Источник: [Stanford
CS336, lecture 11, page 47](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

Для SGD линейный слой получает rank-one update

$$
\Delta W_l=-\eta_l\nabla_{h_l}\ell\,h_{l-1}^{\mathsf T}.
$$

Дальнейший вывод предполагает, что ведущие члены в изменении $h_l$ не
сокращаются и что изменение loss имеет порядок $O(1)$. В этих предпосылках для
SGD получается

$$
\eta_l=\Theta\!\left(\frac{n_l}{n_{l-1}}\right).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-sgd-learning-rate.png]]

*Страница завершает именно упрощённый вывод для линейного слоя и SGD. Нельзя
переносить формулу на Transformer, не указав остальные parameter groups.
Источник: [Stanford CS336, lecture 11, page 49](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

Для Adam масштаб обновления меняется, и в той же упрощённой картине learning
rate получает порядок $\Theta(1/n_{l-1})$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-adam-learning-rate.png]]

*Смена оптимизатора меняет правило. Источник: [Stanford CS336, lecture 11,
page 50](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

Настоящий Transformer содержит embedding, attention, входные и выходные MLP
матрицы, нормализации и softmax output. Для них нужны отдельные правила.
Например, в показанной постановке attention logits масштабируются как $1/d$,
а не привычным $1/\sqrt d$. Здесь $d$ обозначает ширину attention, а не число
обучающих токенов $D$ из scaling law.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-transformer-rules.png]]

*Таблица переводит общую идею в конкретные parameter groups Transformer. Именно
её, а не toy formula, нужно сопоставлять с реализацией. Источник: [Stanford
CS336, lecture 11, page 51](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf),
по [Lingle, 2024](https://arxiv.org/abs/2404.05728).*

<a id="mup-robustness"></a>

В основной серии экспериментов оптимальный learning rate действительно лучше
переносился по ширине с μP, а standard parametrization была менее стабильной.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-transfer-result.png]]

*Положительный результат отвечает на узкий вопрос: сохраняется ли optimum в
проверенной серии ширин при неизменных остальных условиях. Источник: [Stanford
CS336, lecture 11, page 52](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

<a id="mup-failure-modes"></a>

Однако transfer ломался после нескольких, на первый взгляд небольших, изменений.
Learnable gain в RMSNorm нарушал инвариант; sign-based/Lion optimizer требовал
другого scaling rule; сильный weight decay $0.1$ давал наиболее заметный провал.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-rmsnorm-gain-failure.png]]

*Learnable RMSNorm gain нарушает перенос; в исследованной архитектуре gain можно
убрать с малой потерей качества. Источник: [Stanford CS336, lecture 11, page 54](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-lion-failure.png]]

*Правило для Adam не переносится автоматически на optimizer, основанный на
знаке градиента. Источник: [Stanford CS336, lecture 11, page 55](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-weight-decay-failure.png]]

*Сильный weight decay — ещё одно вмешательство, меняющее width transfer.
Источник: [Stanford CS336, lecture 11, page 56](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/mup-summary.png]]

*Итог курса осторожен: μP полезна и часто стабильнее standard parametrization,
но это средство настройки с проверяемыми границами, а не обещание полного
переноса. Источник: [Stanford CS336, lecture 11, page 57](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf).*

<a id="scaling-recap"></a>
## Рецепты современных семейств нельзя сводить к одной дроби

Lecture 11 сопоставляет несколько практических стратегий:

| Семейство | Что масштабировали | Как уменьшали стоимость исследования | Что нельзя переносить как константу |
|---|---|---|---|
| MiniCPM | ширину, learning rate, batch, $N/D$ | μP и WSD-ветви | полученный data-heavy ratio и долю decay |
| DeepSeek LLM | batch, learning rate, model/data allocation | малые grid searches, piecewise schedule, IsoFLOP | коэффициенты fit и выбор FLOPs/token как масштаба |
| Kimi K2 | степень разреженности MoE | отдельная sparsity scaling law | dense-equivalent качество по общему числу параметров |
| Hunyuan | active MoE parameters и данные | IsoFLOP | сообщённое в лекции отношение $96{:}1$ к active parameters |
| Llama 3 | train-optimal model/data allocation и downstream trends | IsoFLOP и малые прогнозные серии | сообщённое train-optimal отношение $39{:}1$ |
| MiniMax-01 | выбор архитектуры и compute frontier | нижняя огибающая запусков | перенос результата на другое архитектурное семейство |

Особенно показательно различие двух чисел для Llama 3. Lecture 11 приводит
$39{:}1$ как результат IsoFLOP-оценки train-optimal режима, а Lecture 9 — около
$215$ токенов на параметр для фактического Llama 3 70B. Это не противоречие:
производитель мог сознательно обучать меньшую модель дольше ради качества,
данных и дешёвого последующего обслуживания.

Правило «около 20 токенов на параметр», возникшее после Chinchilla, столь же
контекстно. Оно описывает порядок величины в определённом исследовании, а не
границу достаточного обучения. Нельзя усреднить $20$, $39$, $96$ и $215$ и
получить рецепт для новой dense- или MoE-модели.

<a id="train-inference-tradeoff"></a>
## Оптимум обучения не равен оптимуму продукта

Chinchilla решает узкую задачу: выбрать $(N,D)$, минимизирующие loss при
фиксированной цене **обучения**. После выпуска параметры модели читаются для
каждого сгенерированного токена. Если ожидается большой поток запросов, меньшую
модель выгодно обучать на большем количестве данных: дополнительная однократная
цена pre-training может окупиться меньшей многократной ценой inference.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/train-vs-inference-ratios.png]]

*Слайд сопоставляет приблизительные training tokens per parameter у нескольких
семейств. Значения для закрытых моделей являются оценками курса, а не
раскрытыми спецификациями. Источник: [Stanford CS336, lecture 9, page 54](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf).*

Sardana et al. добавляют к training cost ожидаемую цену inference demand. В их
экспериментах рост ожидаемого использования смещает optimum к меньшему $N$ и
большему $D$; авторы отдельно обучили 47 моделей и исследовали ratios до 10 000
токенов на параметр. Но и эта модель стоимости неполна без ограничений по
памяти, latency, throughput, batch distribution и параллелизму. Математически
оптимальный нестандартный размер может быть неудобен для реального кластера.

## Разобранный пример: что можно и чего нельзя заключить

Ниже используются вымышленные числа; это не результат Stanford Assignment 3 и
не рецепт для конкретной модели.

Пусть при бюджете $C_1=3\cdot10^{21}$ FLOPs получены три точки:

| $N$ | $D=C_1/(6N)$ | токенов на параметр | validation loss |
|---:|---:|---:|---:|
| 3 млрд | 167 млрд | 55,6 | 2,47 |
| 6 млрд | 83 млрд | 13,9 | **2,42** |
| 12 млрд | 42 млрд | 3,5 | 2,46 |

Средняя конфигурация лучше двух крайних, поэтому наблюдается обе стороны
минимума. Но трёх точек недостаточно, чтобы точно определить его положение:
следующий раунд должен добавить точки между 3B и 12B, а не сразу
экстраполировать.

Предположим, после нескольких бюджетов была получена локальная зависимость

$$
N^*(C)=1{,}8\text{B}
\left(\frac{C}{3\cdot10^{20}}\right)^{0.50}.
$$

Для $C_{\mathrm{target}}=3\cdot10^{22}$ она предсказывает
$N^*_{\mathrm{target}}=18\text{B}$ и

$$
D^*_{\mathrm{target}}=
\frac{3\cdot10^{22}}{6\cdot18\cdot10^9}
\approx2{,}78\cdot10^{11}
$$

токенов. Отношение равно примерно $15{,}4$, но это следствие конкретной
вымышленной подгонки. Если bootstrap даёт для $N^*$ интервал 14–24B, то тому же
compute соответствуют примерно 357–208 млрд токенов. Решение — проверить
несколько конфигураций на удержанном более крупном бюджете, а не округлить 18B и
объявить вопрос закрытым.

## Как читать опубликованный scaling plot

Перед использованием чужой кривой ответьте на вопросы:

1. Что означает $N$: все, non-embedding, active или total parameters?
2. Что означает $D$: уникальные или предъявленные токены, сколько было эпох?
3. Что означает $C$: оценённые FLOPs, measured device time или денежная цена?
4. Сравнимы ли tokenizer, validation distribution, architecture и schedule?
5. Есть ли точки по обе стороны каждого заявленного минимума?
6. Показаны ли исходные наблюдения, неудачные runs, остатки и интервалы?
7. На сколько порядков прогноз выходит за диапазон pilot budgets?
8. Проверялся ли fit на бюджете, который не участвовал в оценке коэффициентов?
9. Как на каждом масштабе настраивались batch, learning rate и optimizer?
10. Совпадает ли целевая функция: train-optimal, inference-aware или
    data-constrained?

Полезный scaling law — это не одна формула в техническом отчёте, а
воспроизводимый экспериментальный объект. Его минимальный комплект состоит из
run ledger, точного cost convention, сырых кривых, процедуры выбора минимумов,
альтернативных fits, residual plots, uncertainty и результата held-out
prediction. Всё остальное — удобная гипотеза, которую ещё предстоит проверить.

## Первоисточники

- Kaplan et al. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020 — ранние зависимости loss от модели, данных и compute.
- Hoffmann et al. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022 — Chinchilla, три метода оценки и IsoFLOP.
- Muennighoff et al. [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264), 2023 — повторение данных и effective data.
- Sardana et al. [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448), 2024 — inference-aware objective.
- DeepSeek-AI. [DeepSeek LLM](https://arxiv.org/abs/2401.02954), 2024 — scaling learning rate, batch и model/data allocation.
- Hu et al. [MiniCPM](https://arxiv.org/abs/2404.06395), 2024 — WSD и экономия scaling campaign.
- Lingle. [An Empirical Study of μP Learning Rate Transfer](https://arxiv.org/abs/2404.05728), 2024 — масштабная проверка переноса μP.
- Besiroglu et al. [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102), 2024 — восстановление данных и перепроверка третьего метода Chinchilla.
- Porian et al. [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/abs/2406.19146), 2024 — влияние warmup, последнего слоя и scale-dependent tuning.
- Tatsunori Hashimoto. [Stanford CS336, lecture 9: Scaling laws](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf), 2026.
- Tatsunori Hashimoto. [Stanford CS336, lecture 11: Scaling laws in the wild](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf), 2026.
- Stanford CS336. [Assignment 3: Scaling laws](https://github.com/stanford-cs336/assignment3-scaling/tree/03e9372992e913061b9e78b5cfcb62ad8a87de35), 2026 — finite-budget campaign and prediction task.
