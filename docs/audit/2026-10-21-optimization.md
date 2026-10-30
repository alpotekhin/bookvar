---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md
route: /textbook/neural-networks/optimizers-schedules/
before_sha256: 45306623c89ae89f89422882b0e7e98ccd126a4dbb0a41fae0ca2115f7787e4c
after_sha256: ffd865468db1e8ea07c64ec84f74a31cd065935f4b8aeb09a0f1f5b65670e43b
---

# Оптимизация: приёмка объяснения

Предыдущий goal turn — progress: переработана и проверена backpropagation.
Её хеш `d329325d295f0fc50ddbf013412c96e0f24f8a31bd6733754e08ae23a5baa198`
подтверждён перед началом этого прохода. Общая цель остаётся активной.

Исходная глава прочитана целиком и сохранена в
[snapshots/2026-10-21-optimization.md](snapshots/2026-10-21-optimization.md).
После содержательных и визуальных исправлений итоговый текст повторно
прочитан целиком четырьмя последовательными диапазонами до EOF.
No-ai-slop проверен отдельно; рассуждения, примеры и исходные рисунки сохранены.
EN, raw/, index Git, коммиты, push и deployment в этом проходе не менялись.

## Объяснения и прочитанные источники

| Механизм | Прочитанный источник | Проверенный конкретный случай |
| --- | --- | --- |
| Исходные потери и маленький набор | Karpathy Recipe §§1–3; CS231n sanity checks | Равномерные 10 классов: ln(10)=2.302585; оговорены противоречивые метки и недостаточная ёмкость |
| Шум и геометрия SGD | D2L Momentum: Basics целиком, включая Leaky Averages, Ill-conditioned Problem, Momentum, Effective Sample Weight; McCandlish §2.1–2.2 | Градиенты 4/−2: среднее 1, дисперсии 9 и 4.5; условие сходимости квадратичной функции |
| Momentum и Nesterov | CS231n Parameter updates/SGD and bells and whistles целиком; D2L Basics | Разная eta исходных графиков обнаружена и подписана; SGD .6 расходится, momentum .6/.5 сходится; lookahead даёт .729 на втором шаге |
| Adagrad/RMSProp/Adam | CS231n Per-parameter adaptive methods целиком; Adam v9 Algorithm 1 и §§2–3 | Накопители на 4/1; три шага каждого метода; коррекция смещения; (10,1) не означает меньший первый шаг первой координаты |
| AdamW | AdamW v3 §2 полностью, включая Algorithms 1/2, Propositions 1–3; CS336 A1 §§4.2–4.5 | Первый шаг с theta=2, g=3: Adam+L2 → 1.90, AdamW → 1.88; различены положения epsilon и коэффициента decay |
| Расписание | CS336 A1 §4.4 полностью; MiniCPM v1 §4.2 полностью с Figures 5/6; CS231n Annealing | Явная нумерация 1..T; первое/пиковое/последнее обновление; изменение T меняет прежнюю cosine-траекторию |
| Критический batch | McCandlish §2.1–2.2, включая Figure 3 и eqs.2.1–2.8 | На той же квадратичной функции Bnoise=9; оптимальные ожидаемые уменьшения .05/.25/.454545 |
| Clipping и точность | CS336 A1 §4.5; Lecture 2 FP32–FP8; PyTorch 2.14 AMP typical/unscale/clipping/accumulation целиком; FP8 v1 §§1–3/Table 1 | Глобальное (6,8) → (3,4), а Adam-обновление другое; FP16 underflow и восстановление порядка 1e−8 после scaling |
| Состояние и продолжение | CS336 A1 §5.2 целиком; CS231n weights:updates; Recipe dependency checks | 2P массивов моментов, 8P байт при FP32; 5→10 шагов с AdamW/RNG/counter точно воспроизводятся; только веса не воспроизводят |
| Матричные методы | SOAP v1 §§2 и 4, Algorithms 1–4/implementation details; Muon Definition, Why orthogonalize, Alternatives/NS proof/Tuning, empirical parameter groups | G=[[3,1],[1,3]], диагональ 4/2; фиксированный поворот меняет поэлементную адаптацию; polar I не равен G/4 |

CS231n также прочитаны целиком подразделы monitoring loss, train/validation,
update ratios и activation distributions. Категоричные старые рекомендации
по оптимизатору и универсальным числам learning rate не перенесены.
Соглашение Nesterov с физической скоростью не выдано за совпадающие числа
с хранимой переменной произвольного библиотечного API.

Adam/AdamW p.2/3 и CS336 A1 p.32 отрендерены и осмотрены:
это подтвердило различие epsilon, а не только символов после извлечения текста.
MiniCPM p.5 и McCandlish p.6 с рисунками осмотрены; SOAP p.6/Algorithm 3
также отрендерена и прочитана. Сравнения качества из этих статей не
выдаются за воспроизведённые нами эксперименты.
SOAP-пример изолирует фиксированный базис, а не имитирует весь алгоритм.
Muon описан через SVD с оговоркой о нулевых сингулярных значениях и конечной NS-итерации.

TinyTorch, ML Visualized, CS230 cheatsheet и SGDR остаются дополнительными
материалами; их полный перенос или повторное чтение в этом проходе не заявляются.

## Исходные изображения

Архивы источников сохранены в `05 Источники/Originals/optimization-2026-10-21/`.

Добавлены три оригинала без изменения пикселей/SVG:

- `d2l-sgd-divergence.svg`: D2L output_momentum_e3683f_15_1.svg,
  SHA `d226ed64352db342b885e5d5c25fa9c34f6b1db88bd03fcfb9359edd8171a0d2`.
- `d2l-momentum-weights.svg`: D2L output_momentum_e3683f_51_0.svg,
  SHA `e9e3fdd12dec59a0372fc68b9589e0bf3ddef55e55950b4325d83a7d17206280`.
- `cs231n-nesterov.jpeg`: cs231n repo, commit
  `8ec57ccfc32379ae4b37c45de949af359673f92f`,
  SHA `af94b879e0913562974b00be9fdb1debcc177fa5595dd995f75d61e6c70e3a4b`.

У D2L сохранены CC BY-SA 4.0, HTML и закреплённый текст курса.
У нового CS231n — MIT notice и закреплённый Markdown.
Registry-тест сравнивает копию/архив, хеши источника и лицензии;
отрицательные мутации источника, хеша, лицензии и provenance отвергаются.

Четыре прежних файла (два D2L SVG, CS231n GIF и learningrates.jpeg)
побайтно совпали со свежими downloads исходников.
FP8 PNG совпал с raster из закреплённого локального CS336 Lecture 2 archive.
Его прежняя registry-запись «User permission record, not a named license»
не превращена в новую подтверждённую лицензию: новых FP8-изображений не добавлено.
Нарисованных заменителей иллюстраций нет.
У GIF уточнены автор Alec Radford и седловая поверхность; у D2L исправлена
ложная подпись о сравнении при одной скорости обучения.

## Исполнение и публикационные проверки

`publishing/tools/check_optimization_examples.py`: PASS на torch 2.14.1 CPU.
Исполнены все четыре опубликованных Python-блока.
Независимые проверки покрывают промежуточные моменты Adam, все D2L координаты,
Nesterov, Adagrad/RMSProp, rescaling/epsilon, расписание и его границы,
critical batch, norm clipping, FP16/BF16/scaling/FP8 bounds,
update ratio/межпримерные зависимости, фиксированный базис/SVD/NS
и точное продолжение состояния. Это не GPU benchmark и не тренировка LLM.

- Unit: 284 PASS / 18 файлов, `--maxWorkers=1 --testTimeout=10000`, exit 0.
- Последняя сборка: 1961 страница, 20.85 s, exit 0.
- На последней сборке: 45 output tests PASS, 0 broken internal links.
- Source coverage: 210 структурных строк PASS, не доказательство полноты курсов.
- `git diff --check` PASS.

Логи: `/tmp/bookvar-optimization-{examples,unit,build,output,links,coverage}-2026-10-21.log`.
Проверка примеров дополнительно запрещает одинарный доллар на отдельной строке:
полное чтение поймало четыре повреждённых display-блока, которые сборка
сама по себе не отвергала. Они исправлены до принятой сборки.

## Визуальная приёмка

Все восемь исходных рисунков осмотрены рядом с текстом на desktop 1365×1000
и mobile 390×844. Nesterov и FP8 дополнительно проверены в просмотрщике
при 2.5× с панорамированием: подписи читаемы; в полном inline-виде
мелкие подписи этих широких оригиналов на телефоне требуют увеличения.

На текущем build: 16 display-формул, 0 KaTeX errors, wide=0 в обеих ширинах.
Cosine-формула была 402/343 и разложена на промежуточные величины.
Две таблицы были 397/343 и 569/343; заголовки сокращены,
пояснения к форматам сохранены прозой.
Теперь все три таблицы 343/343 на mobile и 687/687 на desktop,
осмотрены их значения и строки. Документ 375/375 и 1350/1350.
Duplicate IDs=0. Старые заголовочные якоря и четыре служебных anchor сохранены.

Proof:
`/tmp/bookvar-optimization-desktop-2026-10-21.jpg`,
`/tmp/bookvar-optimization-mobile-schedule-2026-10-21.jpg`,
`/tmp/bookvar-optimization-mobile-fp8-zoom-2026-10-21.jpg`.

Приёмка ограничена этой ревизией главы. Общий учебник, неразобранные
страницы и прежний pixel-match CS231n lecture_2.pdf остаются в общей очереди.
Следующая глава — инициализация, нормализация и residual connections.
