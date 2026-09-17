# Foundations editorial completion — Task 1

Status: implemented, awaiting independent review and primary publication verification.

55/55 assigned RU pages and 10/10 existing authored EN counterparts read completely before and after edits. All 102 original findings have individual outcomes: 96 fixed, 6 already-fixed, 0 rejected, 0 open. No raw course translations or archive rewrites. Existing figures, headings, explicit anchors and source-unit IDs preserved; no substitute images generated.

## Verification and limits

- Every changed page has its exact task-start snapshot under `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/`; all 65 before/after hashes matched at handoff.
- `node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs`: scalar derivations, exact bootstraps, clustering/EM/optimization and neural examples.
- `/private/tmp/bookvar-editorial-venv/bin/python .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-page-code-checks.py`: actual small page snippets with torch 2.8.0 CPU and independent numerical examples. Includes posterior/prior VAE, attention masking forward/backward, RNN autograd, QKV in both locales, independent RoPE identities, label scores and all 165 T5 composition pairs.
- Actual page-code execution is distinguished from independent arithmetic and illustrative pseudocode using stubs. No pretrained-model inference, full model training, TensorFlow execution or benchmark reproduction is claimed.
- Primary-source passages were checked where needed for factual corrections. Reading a whole page does not mean every outgoing source or historical benchmark was re-audited online; evidence per page states the boundary.
- No full site build, commit, push, merge or deployment. Independent content review and full publishing verification remain with the primary agent.

## Individual page and finding outcomes

### 1. 00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md

Before SHA-256: `33c4a883494be2691801d439e63dd3e2331d8aa31e40afe68fb8c929b8ad6de6`  
After SHA-256: `ef16b64cb97a014ec6d06247c20de16cec17dcaf3678782466c41165120200b0`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md`

- **foundations-01-1 · P3 · fixed** — Для двух векторов точнее говорить о свёртке общей координатной оси, а не о двух осях; фраза мешает только что введённому строгому различению ранга и длины оси.

  Disposition: Общая координатная ось i сворачивается; результат скаляр без осей.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 2. 00 Учебник/00 Математические и ML-основания/02 Производная и градиент.md

Before SHA-256: `ef2a01e4d2ad32e58a37534e9e9c383dd3d52ecbe6765ccdfa0bd0d80bc04805`  
After SHA-256: `56f43f60427f6fe476df0d71ccdd54b88500792fa578f28ddce46712b8b868b3`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/00 Математические и ML-основания/02 Производная и градиент.md`

- **foundations-02-1 · P2 · fixed** — Требует уточнения/проверки: одного большинства недостаточно для такого вывода; это иллюстрация скалярной цепочки, а перенос на векторную сеть требует обсуждения произведений якобианов.

  Disposition: Скалярная цепочка отделена от произведения якобианов; контрпример (0.5,0.5,8), log-product criterion и достаточная uniform bound rho<1.

- **foundations-02-2 · P3 · fixed** — Якобиан появляется в подписи без явного определения его элемента и связи с уже введёнными частными производными.

  Disposition: Перед обеими иллюстрациями определён J_ij=∂f_i/∂x_j, m×n, строки/столбцы и Δf≈JΔx.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: [https://cs231n.github.io/optimization-2/](https://cs231n.github.io/optimization-2/) — Backprop chain-rule educational source inspected; added scalar counterexample independently verified.

### 3. 00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм.md

Before SHA-256: `544835308bc6001fa2857b52c8bbaef31b91315ed865f9e16e31c53f2c93c5dc`  
After SHA-256: `9762e698ea59cc4750c04a80689e71798564c6161f655cf68274c3da2cb26f9c`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм.md`

- **foundations-03-1 · P3 · fixed** — Последний раздел переходит от формулы ожидания к шуму minibatch без численного примера; для новичка связь среднего, выборочного среднего и дисперсии остаётся заявленной.

  Disposition: Распределение X=(0,1,3), p=(1/2,1/4,1/4): E=1, Var=1.5; средние двух батчей 1 и1.75; Var(mean)=0.375 при iid.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: [https://d2l.ai/chapter_preliminaries/probability.html](https://d2l.ai/chapter_preliminaries/probability.html) — Probability source inspected; finite distribution moments independently recomputed.

### 4. 00 Учебник/00 Математические и ML-основания/04 Функции потерь.md

Before SHA-256: `7e8d59c3fcebe4e8ff4f8d9cc515e2e21ebeb560ce5c9436acb94aac41229a0b`  
After SHA-256: `0b522f015a4b8bb8193bab79d0ed7d76c8cf53b8a959ab2360481ee1e50a6e7f`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/00 Математические и ML-основания/04 Функции потерь.md`

- **foundations-04-1 · P2 · fixed** — Мотивировка не различает MSE по числовым ID и MSE между вероятностным вектором и one-hot target (Brier score). Официальная документация подтверждает Brier как strictly proper scoring rule; отсутствие расстояния между ID не объясняет выбор CE вместо векторной MSE.

  Disposition: Разделены MSE по ID, multiclass Brier и binary convention; p=.01,y=1: losses .9801/4.6052 и logit-gradients -.019602/-.99.

- **foundations-04-2 · P2 · fixed** — Энтропия и KL до разложения не определены; центральная связь подаётся готовым равенством, поэтому читатель не может восстановить её из текста.

  Disposition: Введены суммы entropy/KL, разложение выведено добавлением/вычитанием qlogq; численный q=(.6,.4),p=(.8,.2).

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) — Brier definition inspected; ID vs probability and binary/multiclass convention separated.

### 5. 00 Учебник/00 Математические и ML-основания/05 Train, validation и test.md

Before SHA-256: `14b59539cf2386b4359757e74cf4546cc2431268562b7d9f37261773f8491f20`  
After SHA-256: `8c7bea69356d52d10f1beed81ee94d31cf88fae5c45eade513f142cb7cfdcccc`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/00 Математические и ML-основания/05 Train, validation и test.md`

- **foundations-05-1 · P2 · fixed** — Таблица называет пользователя единственно «правильной» единицей split, хотя абзац ниже правильно требует имитировать целевой сценарий. Проверка новых сообщений уже известных пользователей и перенос на новых пользователей — разные задачи.

  Disposition: Таблица различает новых пользователей и будущие сообщения известных пользователей.

- **foundations-05-2 · P3 · fixed** — Читатель может принять хеширование даты за temporal holdout; конкретная гарантия такого split в тексте не объяснена.

  Disposition: Hash residues (2,9,1) демонстрируют отсутствие хронологической гарантии; cutoff train1–20/validation21–25/test26–30.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 6. 00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание.md

Before SHA-256: `5e3205320df74dddcadc2500c28d2634e374f92b317be10d70989715ddd9c49d`  
After SHA-256: `1675de08e54bc0ad0f44a1fbe4781352b7bccb0a76ebe8096084dead8f125bad`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание.md`

- **foundations-06-1 · P2 · fixed** — Точное разложение обещано, но остаётся в изображении и краткой подписи: нет заданной случайности по обучающим выборкам, формул bias/variance и разобранного примера.

  Disposition: Условное по x разложение с независимостью D/newY; равновероятные обучающие выборки и прогнозы (1,2,3) против constant1.5.

- **foundations-06-2 · P2 · fixed** — Важнейшее требование неопределённости не сопровождено ни определением доверительного интервала, ни построением paired resampling; читатель получает указание, но не метод.

  Disposition: Определён confidence interval; полный exact paired bootstrap 4^4, квантили [-.5,1], оговорены groups, weighting, fixed predictions versus retraining.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: [https://cs229.stanford.edu/summer2023/BiasVarianceAnalysis.pdf](https://cs229.stanford.edu/summer2023/BiasVarianceAnalysis.pdf) — Conditional bias-variance derivation source inspected; independent exact paired-bootstrap enumeration supports new example.

### 7. 00 Учебник/01 Классическое машинное обучение/00 Карта модуля.md

Before SHA-256: `293f4b56600fcc9cba04b9693f5c48c765bf01635b9ab18ff99ff1fb796f4023`  
After SHA-256: `eef1ea832e61990a71d72b53aee64b0c114c0eeee0b4726dfae7f857ff2ce56c`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/00 Карта модуля.md`

- **foundations-07-1 · P3 · fixed** — Навигационная аннотация смазывает различие: предыдущая глава и глава ансамблей используют для bagging глубокие нестабильные деревья, а для boosting — обычно неглубокие.

  Disposition: Аннотация bagging/boosting различает усреднение и последовательное исправление, не объявляя деревья bagging слабыми.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 8. 00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline.md

Before SHA-256: `d4089e47b9c6f89805c839afed715ce9da8ed4db22d653111dd53c594c0d0b7a`  
After SHA-256: `b188f5bbd8fef8802e2ef18fe3f180e8bc94d2925af2a3138577e8c6f7b229b2`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline.md`

- **foundations-08-1 · P2 · fixed** — Сквозной пример связывает риск ухода с решением, кому звонить, но не различает прогноз риска и эффект звонка: высокий риск сам по себе не показывает, кому вмешательство поможет.

  Disposition: Приоритизация по риску отделена от причинного эффекта звонка; V_TP явно определяется как ценность выявленного риска, не выручка.

- **foundations-08-2 · P3 · fixed** — Стык редактирования оставил «корректный корректный», а определения split повторяют предыдущую главу без нового расчёта.

  Disposition: Дублированное корректный удалено; split сведён к ссылке; конкретная строка (12,100,basic) проходит x=(2,1,1,0), z=.4,p=.5987.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 9. 00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация.md

Before SHA-256: `25834cefdfe76917df00e3dcdf57969776c1021fb39bce941788b7ffc13c960b`  
After SHA-256: `a20ab01f3d353f04f0a922773f10089f85c01bc73bc70d6ec8b6ff17905096c7`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация.md`

- **foundations-09-1 · P2 · fixed** — В начале модель содержит свободный член b, но в нормальном уравнении он исчезает без указания столбца единиц или центрирования; также не введены формы X и y.

  Disposition: Матрица плана n×(d+1) со столбцом единиц и beta=(b,w); нулевой градиент выводит нормальные уравнения, условие полного столбцового ранга явно.

- **foundations-09-2 · P2 · fixed** — Odds/log-odds вводятся эквивалентной формулой, но ни один логистический пример не доведён от w,x,b через sigmoid до odds и решения.

  Disposition: Логистический пример w=(ln2,-ln2),x=(2,1): p=2/3,odds2; +1 даёт p4/5,odds4 и переход через threshold.7.

- **foundations-09-3 · P3 · fixed** — Синтаксис «возвращает непрерывный оценкой вероятности» сломан.

  Disposition: Сломанное согласование исправлено на непрерывную оценку вероятности.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 10. 00 Учебник/01 Классическое машинное обучение/03 Деревья решений.md

Before SHA-256: `f0a420a82578597b8b61870926c37b60133ef87f8533c30a00bfbbb3e0a11c0d`  
After SHA-256: `cb4e77ee2968ec8a95ffaf16e00329ccafe94af76c601befeb29d59d31e5718e`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/03 Деревья решений.md`

- **foundations-10-1 · P3 · fixed** — Стоимость pruning обозначена слишком общо: неизвестно, это misclassification error, взвешенная impurity или SSE, хотя выше обсуждаются разные критерии.

  Disposition: R(T) определён взвешенной impurity по соглашению sklearn; варианты alpha .05/.10 сравнивают одно и два листа.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: [https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning) — Weighted total impurity and alpha complexity pruning definition inspected.

### 11. 00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting.md

Before SHA-256: `b506f9d62f2414911ba7e3a5911d115e4d38ead172e273f569577ba81e35f25b`  
After SHA-256: `103493befe0e502202ef474018a5c22bb820e1e567840099b6130d13d37e0a1e`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting.md`

- **foundations-11-1 · P2 · fixed** — Вторая половина повторно вводит почти весь уже объяснённый материал: формулу дисперсии (39–47/132–144), OOB (61–64/153–157), additive model и псевдоостатки (68–86/161–180), histogram (108–116/199–210). Это следы склейки двух версий, а не углубление.

  Disposition: Четыре повторных введения заменены углубляющими примерами variance/OOB/two-step boosting/binning; все оригинальные рисунки и заголовки сохранены.

- **foundations-11-2 · P2 · fixed** — Нормировка loss не задана; равенство в точности требует 1/2(y−F)², а ранее squared error использовалась без 1/2. Для классификации не определено, F — вероятность или logit.

  Disposition: Зафиксирован loss1/2(y-F)^2; регрессионный цикл MSE5→2→1; binary CE в logit-space с r=y-sigmoid(F).

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 12. 00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога.md

Before SHA-256: `aff1344e398471282885f62b7f00e5ff16932ad03d85ec0230781e73719ea7ee`  
After SHA-256: `24e93c8bacc89066b16aa122c5c1ede94e1c0c540f031847e80b9eb9d93d4d30`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога.md`

- **foundations-12-1 · P2 · fixed** — Повторяются первый CV-раздел, зависимость folds, nested CV, confusion matrix и calibration; скрининговый пример не доведён до чисел TP/FP и сравнения порогов.

  Disposition: Повторы заменены одной OOF-таблицей100объектов, двумя матрицами и стоимостью11vs1, обсуждены внешний цикл и calibration Brier.017.

- **foundations-12-2 · P2 · fixed** — Bootstrap снова назван без метода построения и единицы ресемплинга, хотя страница должна закрывать оценивание; предыдущая фундаментальная глава тоже лишь упоминала его.

  Disposition: Парный bootstrap независимых пациентов: сохранение визитов, macro-vs-micro counterexample, exact27replicates и fixed model versus retraining.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 13. 00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения.md

Before SHA-256: `6b7fedabf4d9cd035e91df1f45c30899605cd0971f3283ea3c5365ae2c0d07aa`  
After SHA-256: `4f5e89090c3bb906b3cd9fded9e9fffbd3cafee3a7597f8924a9f279e995f72e`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения.md`

- **foundations-13-1 · P2 · fixed** — K-means и density-based методы повторно начинаются после уже данных определений; при этом ни одного назначения/обновления центров на конкретных точках нет.

  Disposition: Второй ввод Lloyd заменён двумя итерациями150→39.25→13.9375→4; плотностная механика раскрыта core/border/noise example.

- **foundations-13-2 · P2 · fixed** — Общее требование не объясняет, как это сделать для рассмотренных DBSCAN/HDBSCAN; следующий спектральный раздел прямо предупреждает о неполучении out-of-sample transform автоматически.

  Disposition: Таблица explicit new-object rules KMeans/DBSCAN/HDBSCAN/spectral; отдельно пакет hdbscan approximate_predict и fixed hierarchy.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: [https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html) — approximate_predict uses precomputed hierarchy/prediction data; not refitting clustering on added point.

### 14. 00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian.md

Before SHA-256: `f1c0133c0e7bf1f7b476d05fc1ab445aae464bcc913e9f39b30c46a580d151b6`  
After SHA-256: `4a8537f3b145fb4c71f375c6ccb59438411bfd0e561009dfafb6f790d01b1ea8`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian.md`

- **foundations-14-1 · P2 · fixed** — Для заявленного входного маршрута не определены собственное значение/вектор, PSD, ортогональность как ограничение или Rayleigh quotient. Глава перескакивает от базовых форм матриц к спектральной оптимизации.

  Disposition: Введены Av=lambda v,PSD,orthogonality,Rayleigh quotient; graph4 доведён до eigenvalue.0950124, вектора и partition12/34.

- **foundations-14-2 · P2 · fixed** — Не указано условие положительных степеней и поведение на изолированных вершинах, хотя малый k выше может раздробить граф.

  Disposition: До inverses D оговорены positive degree и выбранное исключение изолятов с отдельной меткой.

Checks: node .superpowers/sdd/2026-09-15-close-editorial-findings/task-1-numerical-checks.cjs passed (executed scalar arithmetic; not ML training)

Source evidence: [https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf](https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf) — Spectral graph/Laplacian treatment inspected; isolated-degree policy explicitly declared locally.

### 15. 00 Учебник/01 Классическое машинное обучение/07 PCA, t-SNE и UMAP.md

Before SHA-256: `00338fdeecd86d1f34829d42af11d7861331416194bc714e8b7a6a2370ee9dba`  
After SHA-256: `c4925068d84c9d173fca5bb918a73efc3630bc7554d2379d30918c42a97aa98b`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/07 PCA, t-SNE и UMAP.md`

- **foundations-15-1 · P3 · fixed** — UMAP-часть обрывается перед самой целевой функцией: параметры a,b не определены, связь min_dist с формой кривой не показана, хотя выше для t-SNE приведён полный KL и градиент.

  Disposition: Выписана binary cross-entropy по парам, посчитаны вклады 0.5004/1.3322; a,b связаны с официальным find_ab_params, min_dist/spread и мягкой целевой кривой. Отличена реальная negative-sampling оптимизация.

Checks: task-1-numerical-checks.cjs: executed scalar checks passed

Source evidence: [https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html#find_ab_params](https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html#find_ab_params) — Official find_ab_params source inspected: smooth curve fitted to target min_dist/spread profile.

### 16. 00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM.md

Before SHA-256: `ff0615a6f8110c1b798f46ee18b1b46cc0f2c79d9ff14299fa28582307b82334`  
After SHA-256: `48638e21a991e6d5879777c1c15f48d115e5f7d84431424c9f4da4d4d35805dd`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM.md`

- **foundations-16-1 · P1 · already-fixed** — Обещанный полный цикл не соответствует заданным параметрам μ=(0,4), π=(0.5,0.5), σ²=(1,1). Для x=(0,1,4,5) E-step по собственной формуле главы даёт γ1≈(0.999665,0.982014,0.000335,0.000006), а таблица содержит (0.99,0.95,0.05,0.01). Поэтому новые средние 0.6/4.4 не являются продолжением этого E-step.

  Disposition: До начала Task 1 полный fixed-variance E→M цикл уже исправлен. Независимый пересчёт gamma, средних .496153/4.468138, весов и likelihood -7.429516→-6.944253 подтвердил согласованность исходной текущей страницы.

- **foundations-16-2 · P2 · fixed** — Многомерная Gaussian и ковариационная матрица не определены формулой; ранее вводились только одномерная плотность и дисперсия. Отсюда «нормировочный множитель», determinant и ellipsoid возникают без опоры.

  Disposition: Добавлены полная Gaussian density, формы и covariance definition, PD условие, determinant normalization и ellipsoid interpretation; diag(4,1) пример даёт quadratic form 2.

Checks: task-1-numerical-checks.cjs: executed scalar checks passed

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 17. 00 Учебник/01 Классическое машинное обучение/09 Kernels, SVM и random Fourier features.md

Before SHA-256: `5cc55f042ea10b5be43b52de14f1beff47f89891fd66a74c8a02506847e3679b`  
After SHA-256: `4c0a0d159c6ee59fb94f233009858507c888f393b068829da3c5daf10da130b9`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Классическое машинное обучение/09 Kernels, SVM и random Fourier features.md`

- **foundations-17-1 · P2 · fixed** — Слишком общее утверждение об утечке противоречит собственному описанию data-independent RFF (323–324). При фиксированном числовом gamma и seed случайные веса не оценивают распределение X; документация отдельно выделяет gamma='scale', который использует X.var().

  Disposition: Разделены fixed-gamma data-independent RFF, gamma=scale, scaler, Nyström landmarks и selection leakage.

- **foundations-17-2 · P2 · fixed** — Для воспроизводимого алгоритма не дано точное распределение частот в принятой параметризации γ, а связь primal→dual дана готовой формулой без объяснения α.

  Disposition: Дан hard-margin Lagrangian и stationarity; frequencies N(0,2 gamma I), paired 2D и random-phase D отражены явно.

Checks: task-1-numerical-checks.cjs: executed scalar checks passed

Source evidence: [https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) — Rahimi/Recht primary RFF paper inspected; frequency distribution and cosine features independently derived.

### 18. 00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md

Before SHA-256: `8bfc9689f5ec39c7d27f9245c9b6d33bcc22815dd6adc1fdd5a9fc5ccc28ffc9`  
After SHA-256: `5611ca4e2e5545be7f5fc89a00e3adb6a58ecc27e8d794f62c172cd278b669dc`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md`

- **foundations-18-1 · P2 · fixed** — В главе явно выбрано соглашение xW и W1 формы d×m, но последняя формула молча переключается на W1x. Без переопределения форм она несовместима с прежними W1/W2.

  Disposition: Все слойные формулы приведены к xW и пояснён переход от столбца одного нейрона к строкам батча.

- **foundations-18-2 · P2 · fixed** — XOR назван минимальным контрпримером, но не показано, какие именно условия/веса дают решение, поэтому представление, созданное скрытым слоем, остаётся обещанием.

  Disposition: Конструктивный ReLU XOR посчитан для 4 входов; указаны все веса и смещения и отличие выразимости от выполненного обучения.

Checks: task-1-numerical-checks.cjs: executed scalar checks passed Actual current page Python code executed in torch2.8.0 CPU via task-1-page-code-checks.py; passed.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 19. 00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md

Before SHA-256: `54eaff3ec4a1f1d9f123c08a7958965d2cf50be5912769ab160901eb81c46c09`  
After SHA-256: `cfc70cfd57de9e045b459bf82d0ae9c3da595c26c71b2de3e5b5afc967294f6e`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md`

- **foundations-19-1 · P3 · fixed** — Цикл с p.data одновременно представлен рядом с PyTorch и micrograd, но не уточняет API и намеренно упрощённую семантику; код не даёт законченного исполняемого примера нейрона.

  Disposition: Старый .data-цикл явно обозначен micrograd-псевдокодом с полной ссылкой. Добавлен самостоятельный PyTorch нейрон со SGD optimizer.step/no_grad; скалярная проверка loss 1.068893→.329935, actual current PyTorch snippet now executed successfully in torch2.8 CPU.

Checks: task-1-numerical-checks.cjs: executed scalar checks passed Actual current page Python code executed in torch2.8.0 CPU via task-1-page-code-checks.py; passed.

Source evidence: [https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — Official zero_grad/backward/step recipe inspected and actual page snippet CPU executed.

### 20. 00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md

Before SHA-256: `7c10582a53c8714887e16f4c741855f20f55f93739a7d5893bdb06e2f79f1ec9`  
After SHA-256: `45306623c89ae89f89422882b0e7e98ccd126a4dbb0a41fae0ca2115f7787e4c`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md`

- **foundations-20-1 · P2 · fixed** — В базовую главу до первого численного шага SGD/Adam вставлены Shampoo, собственный базис, полярное разложение и Newton–Schulz без нужных предпосылок. Устойчивое ядро и продвинутый обзор смешаны.

  Disposition: Muon/SOAP перемещён в дополнительный раздел после устойчивого ядра, с prerequisites. Три последовательных шага SGD/momentum/Adam на одной функции посчитаны с конкретными моментами.

- **foundations-20-2 · P2 · fixed** — При названии schedules отсутствует формула или таблица значений расписания; reader не может восстановить warmup/cosine boundary и шаг конца обучения.

  Disposition: Дана кусочная warmup+cosine и значения на границах и промежуточных шагах; WSD на той же оси, указана конвенция индексации.

Checks: task-1-numerical-checks.cjs: executed scalar checks passed

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 21. 00 Учебник/01 Основы нейронных сетей/02 Функции активации.md

Before SHA-256: `46ded0f6cb4acdfa7d49b95c2006ba006510b85b255f7a6d4c007e4f45aa164f`  
After SHA-256: `0a914295be641aa3056d04ece048828ccb5776b842b059bbb9ae179f83b0a9af`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/02 Функции активации.md`

- **foundations-21-1 · P3 · fixed** — Таблица диапазонов для GELU/SiLU менее точна, чем для остальных: обе функции ограничены снизу, но не сверху; «не ограничен» рядом с интервалами можно прочитать как вся R.

  Disposition: GELU/SiLU table now distinguishes lower boundedness from unbounded positive tail.

- **foundations-21-2 · P3 · fixed** — Второй недостаток назван, но его влияние на обновления не объяснено; следующий аргумент о малых производных относится к насыщению, а не центрированию.

  Disposition: Added numerical nonnegative-input gradient example with batch-sum caveat; separated centering from saturation.

Checks: task-1-numerical-checks.cjs scalar checks passed; not PyTorch runtime execution

Source evidence: [https://cs231n.github.io/neural-networks-1/](https://cs231n.github.io/neural-networks-1/) — Activation tutorial sigmoid positive-output gradient observation inspected; batch-sum limitation supplied explicitly.

### 22. 00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections.md

Before SHA-256: `bd4885d65037ef0dcdfa18399aa4288661c2a91b4dff72ae1f64c1784eef8795`  
After SHA-256: `a7bdd435be16dbd3cab787dcff1c2d5adc41284a551aede3aafd0520ff8e7491`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections.md`

- **foundations-22-1 · P2 · fixed** — Итог противоречит собственным формулам: LayerNorm и RMSNorm для токена используют одну и ту же ось признаков; различаются статистики и вычитание среднего, а не оси.

  Disposition: Summary correctly distinguishes BN/LN axes from LN/RMSNorm statistics on same feature axis.

- **foundations-22-2 · P2 · fixed** — Ни LayerNorm, ни RMSNorm не посчитана хотя бы для одного ненулевого среднего, поэтому ключевое различие двух формул остаётся абстрактным.

  Disposition: x=(1,3) normalized both ways; gamma/beta example shows post-affine moments are not fixed.

Checks: task-1-numerical-checks.cjs scalar checks passed; not PyTorch runtime execution

Source evidence: [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467) — RMSNorm primary abstract inspected; mean-subtraction omission and local moment arithmetic distinguished.

### 23. 00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md

Before SHA-256: `67e4983c07aa89c5270afbd8e9db2eaee40a7f373120c10de0413bb5f3d4b604`  
After SHA-256: `89ba5881b53bbc3025d3c7448785b936f190e4bc7cb0a400a8643a52fe46d721`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md`

- **foundations-23-1 · P3 · fixed** — У учебной функции не задана область 0≤p<1; p=1 даёт деление на ноль, а выходящие за диапазон значения молча проходят.

  Disposition: Explicit 0<=p<1 contract validates before train/eval branching; p=0,.5,1 expected behaviors stated. Added torch import and clarified AdamW update.

Checks: task-1-numerical-checks.cjs scalar checks passed; not PyTorch runtime execution Actual current page Python code executed in torch2.8.0 CPU via task-1-page-code-checks.py; passed.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 24. 00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md

Before SHA-256: `0d6349bf46261f4b3006dd450bee0b00bdab8c29d88bf48c4eb72c054380aa9a`  
After SHA-256: `8b4fc20203c508a72b7452f946e52809f1b28113e9747bb9de2baf7f557a2008`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md`

- **foundations-24-1 · P2 · fixed** — Receptive field объясняется качественно, но формула/послойный расчёт отсутствуют; выходные формы и первый скалярный результат свёртки также оставлены преимущественно рисункам.

  Disposition: One full 3x3/2x2 correlation computed; output-shape and receptive-field recurrence/table derived for three layers.

- **foundations-24-2 · P2 · fixed** — Половина траектории от LeNet к ResNet сводится к историческому перечню; не видно архитектурного отличия block, порядка norm/activation и изменения shortcut.

  Disposition: Existing D2L ResNet-18 figure now accompanied by exact basic-block order, same-shape and projection shortcuts and shape computation; variants distinguished.

Checks: task-1-numerical-checks.cjs scalar checks passed; not PyTorch runtime execution

Source evidence: [https://d2l.ai/chapter_convolutional-modern/resnet.html](https://d2l.ai/chapter_convolutional-modern/resnet.html) — D2L basic residual block implementation inspected: conv/BN/ReLU/conv/BN/add/ReLU, optional projection conv; no inferred projection BN.

### 25. 00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md

Before SHA-256: `06fac3e16d4174e53423fcef1751986f48b8ab32e6a0a276590265f0d72e17d0`  
After SHA-256: `091ab377baf722ca7e5a917c77a614224a73290ef77e531aa981c5ca3cbd4900`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md`

- **foundations-25-1 · P2 · fixed** — ELBO предъявлена без вывода из log p(x)=ELBO+KL(q||p(z|x)), хотя именно отличие prior matching от приближения posterior составляет центральную идею VAE.

  Disposition: Derived log p=ELBO+KL(q||posterior) and distinguished prior KL; connected to EM.

- **foundations-25-2 · P2 · fixed** — Код показывает только sample и decoder, но не задаёт likelihood/reconstruction term, аналитический KL и reduction. Ни одного обучающего шага или численного примера нет.

  Disposition: Defined Bernoulli decoder, analytic Gaussian KL, explicit reductions, numerical loss and complete tiny PyTorch train step plus separate prior generation. Actual current PyTorch train/prior snippet now executed successfully in torch2.8 CPU.

Checks: task-1-numerical-checks.cjs scalar checks passed; not PyTorch runtime execution Actual current page Python code executed in torch2.8.0 CPU via task-1-page-code-checks.py; passed.

Source evidence: [https://github.com/pytorch/examples/blob/main/vae/main.py](https://github.com/pytorch/examples/blob/main/vae/main.py) — Official VAE code inspected BCE-sum plus analytic KL, posterior train and independent prior sampling; page tiny code actually executed.

### 26. 00 Учебник/02 Представление текста и токенизация/00 План источников раздела.md

Before SHA-256: `2349cd1e6e184ee1970f7447a19473026d614bd881e6d4357f373b9e7dbc58ec`  
After SHA-256: `7e9d3ce1facb66ac23e449ffa824ce4d3c13b9ad7961e5819d879c56a8e191b1`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Представление текста и токенизация/00 План источников раздела.md`

- **foundations-26-1 · P2 · fixed** — Редакционное правило противоречит актуальному AGENTS.md, где отсутствие лицензии само по себе не переводит источник в link-only. Будущая работа по этому документу будет следовать устаревшей политике.

  Disposition: Plan refers to current root AGENTS, preserves actual license and separates repository editorial approval from copyright-holder reproduction permission; absence of license is not an open license. Parent's explicit new-object rights clarification followed.

- **foundations-26-2 · P3 · fixed** — Тип editorial отсутствует в перечне допустимых типов AGENTS.md; страница сама помечена непубличной, но контракт исключения не указан.

  Disposition: Current root AGENTS.md lines34–36 explicitly allows type:editorial/status:editorial only for source plans excluded from publication manifest. Page remains clearly unpublished editorial plan.

Checks: not applicable: editorial policy

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 27. 00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md

Before SHA-256: `1c92b64de9e43442e51784121cb56e6a2ae9fefa715b3029f0c3d408dc5c26e6`  
After SHA-256: `08eb673df9123ac39d3aa702505791068bf510d2764e7772e58e455bed018aab`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md`

- **foundations-27-1 · P2 · fixed** — PPMI вводится без формулы PMI/positive clipping, co-occurrence matrix не посчитана на предложениях; GloVe и negative sampling позже тоже не проходят через один численный objective.

  Disposition: Counted four-sentence symmetric context matrix, derived PMI/PPMI, and computed simultaneous positive/negative SGNS SGD update.

- **foundations-27-2 · P2 · fixed** — Для воспроизводимого GloVe не определена f и область ненулевых X_ij при log X_ij, хотя выше сказано, что матрица в основном нулевая.

  Disposition: Nonzero domain, original f(x), x_max=100 and alpha=.75 defined; one weighted contribution .0255519 calculated.

Checks: task-1-numerical-checks.cjs scalar checks passed; not PyTorch runtime execution

Source evidence: [https://aclanthology.org/D14-1162.pdf](https://aclanthology.org/D14-1162.pdf) — Primary GloVe paper opened; task-specific weighted contribution and count matrix independently checked; not a claim that all article results were reproduced.

### 28. 00 Учебник/02 Представление текста и токенизация/01 Представление текста числами.md

Before SHA-256: `d340a5a241481e22f6ef44347674b27f22238c2d3817d9c72f4b37bbf12efa0f`  
After SHA-256: `990e2bf4db0245edc2a4cde974d7462f485bc349d3274c31ddecf96b07e483f2`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Представление текста и токенизация/01 Представление текста числами.md`

- **foundations-28-1 · P2 · fixed** — Пример не демонстрирует одинаковый bag of words для заявленных слов: «собака/собаку», «укусила/укусил», «человека/человек» — разные словоформы. Лемматизация в постановке не введена.

  Disposition: Identical token multisets John loves Mary / Mary loves John replace inflection-changing example.

- **foundations-28-2 · P3 · fixed** — N, df и вариант tf не определены; d уже использована как ширина embedding, а затем как документ.

  Disposition: Raw TF, N, df, document D, natural logarithm and train-only fitting explicit; 2 ln(3/2)=.81093.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 29. 00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md

Before SHA-256: `f6fe241fe2a1e28ce5e1331bff91a56b2cafdb938b5227da9cc98ba692e75213`  
After SHA-256: `e56be63c263f8b13212592aa24c75158d457cc4b00e1d9247ef4a3888536a59c`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md`

- **foundations-29-1 · P1 · already-fixed** — Ручная последовательность для aaabdaaabac не исполнима: после неперекрывающейся замены (a,a)→Z получается Zab dZabac (без пробела: ZabdZabac); пары (Z,b) из следующей строки нет. Следующие заявленные merges поэтому не происходят.

  Disposition: Current pre-task BPE states were already corrected: ZabdZabac → YbdYbac → XdXac. Executed actual page functions to verify every merge.

- **foundations-29-2 · P2 · fixed** — Unigram занимает несколько абзацев без формулы вероятности сегментации, lattice или одного шага Viterbi; название главы обещает сопоставимое объяснение трёх алгоритмов, но полнота есть только у BPE.

  Disposition: Unigram abc complete segmentation products/logs and Viterbi boundary recursion; same-string WordPiece greedy comparison.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: [https://aclanthology.org/P18-1007/](https://aclanthology.org/P18-1007/) — Primary Kudo Unigram paper abstract inspected; exact short-string Viterbi arithmetic independently checked; BPE functions actually executed.

### 30. 00 Учебник/02 Рекомендательные системы/00 Карта модуля.md

Before SHA-256: `a6e71245c17d817eead1bd400d52128da9199488b169c4e1297fbb6a23fdae67`  
After SHA-256: `d02ac6c0124b33956339f3b5cc8a502c7b1eb79d480532b9e44fa64c0baa7768`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Рекомендательные системы/00 Карта модуля.md`

- **foundations-30-1 · P3 · fixed** — Контраст чрезмерно категоричен: классификаторы в эксплуатации тоже могут менять наблюдаемую выборку (например, модерация/кредитование). Это не определяющее отличие рекомендаций.

  Disposition: Contrast limited to iid textbook classification, with moderation feedback counterexample.

- **foundations-30-2 · P3 · fixed** — Фраза относится к retrieval→ranking→reranking, но следует за таблицей из шести шагов, включая logging/response; для всех шести она неверна.

  Disposition: Exactly retrieval/ranking/reranking named as candidate-reduction stages.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 31. 00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline.md

Before SHA-256: `223ba261840c8802935f82d0f23f2ceeb6009430707bf345641e8bb9bac162d3`  
After SHA-256: `0cf15d2a1e945418d4df3ef3b7ff595426138516aca6d41ec09efd062fa884aa`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline.md`

- **foundations-31-1 · P2 · fixed** — Условие feature_timestamp<=observation_time не гарантирует доступность признака: событие может иметь старое event time, но поступить/быть обработано после решения. Термин timestamp не различает эти два времени.

  Disposition: Available_at contract separate from event_time; 10:05 prediction excludes10:08 available aggregate even if event occurred10:00.

- **foundations-31-2 · P2 · fixed** — Один пользовательский пример скрывает межпользовательскую временную утечку: train позднего пользователя может оказаться позже test раннего, а общие item embeddings/popularity уже использовать будущее.

  Disposition: Two-user timeline exposes future shared-embedding leakage; LLO distinguished from global chronological evaluation.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 32. 00 Учебник/02 Рекомендательные системы/02 Collaborative filtering и matrix factorization.md

Before SHA-256: `4906a2915419e93e82969d0697023acba2a05d812ac719a51f63d26250c22650`  
After SHA-256: `9c2f5305233c50b87ba443b10323978ebc5e897c33ff71826b1a4ac4da2dae63`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Рекомендательные системы/02 Collaborative filtering и matrix factorization.md`

- **foundations-32-1 · P2 · fixed** — В цикле loss.backward()/optimizer.step() отсутствует zero_grad; это противоречит отдельно объяснённой ранее семантике накопления и при буквальном исполнении суммирует градиенты разных шагов.

  Disposition: zero_grad before each minibatch; missing construction helpers explicitly pseudocode.

- **foundations-32-2 · P2 · fixed** — Два обновления записаны последовательно без указания старых p/q; ручная реализация может использовать уже обновлённый p для q, получив не одновременный SGD.

  Disposition: Simultaneous old-parameter update gives p(.243,.4086),q(.5172,.1344),dot.18059544; actual tensor arithmetic caught and corrected extra digit in q.

- **foundations-32-3 · P3 · fixed** — Штраф суммируется по событиям и тем самым взвешивает регуляризацию ID его частотой; это допустимый objective, но отличается от одного глобального штрафа по каждому ID и не объяснён.

  Disposition: Per-event frequency-weighted norm penalty distinguished from one global penalty per ID, including bias.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 33. 00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md

Before SHA-256: `fd9632eb3b0f24fcfa094ca6cafe1d473e726abbbbb73c3c8fd449edd6775af6`  
After SHA-256: `5b8e14cc51339c59f8fe31921f713b99abbd2c4c13d70564a28855a686f80b7b`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md`

- **foundations-33-1 · P1 · already-fixed** — В контексте training negative sampler предлагается учитывать будущие positive-события. Это нарушает train-only temporal contract предыдущей главы и той же страницы (287–293), если речь не о явно отдельном oracle benchmark.

  Disposition: Pre-task sampler already rejects future-label use; only positives known before train cutoff excluded, preserving BPR/Ji source explanation.

- **foundations-33-2 · P2 · fixed** — Снова показан цикл backward→step без zero_grad, хотя накопление между шагами здесь не заявлено.

  Disposition: zero_grad, mean over B, BPR score-gradient(-.880797,.880797),updated delta -1.823841.

- **foundations-33-3 · P2 · fixed** — После BPR глава превращается в плотный каталог AutoRec/NeuMF/FM/DeepFM; связи задач есть, но AutoRec/DeepFM не получают сопоставимого разобранного примера.

  Disposition: Objectives separated from architecture family; prerequisite route plus AutoRec loss6.8 and DeepFM logit1.9 worked mechanisms.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 34. 00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки.md

Before SHA-256: `feba3ecb214760ee8532c7f1757e69b403c0431609b357f1890ccffd5fd33300`  
After SHA-256: `bdffc2e21125a8243fff8fb22c2b7fc8d848b3bc98369a8964743d0d34794fa0`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки.md`

- **foundations-34-1 · P2 · fixed** — По карте учебника рекомендации идут до нейросетей/RNN/Transformer, но здесь используются convolution, pooling, embeddings, TransformerBlock и causal/padding mask без локально достаточного введения.

  Disposition: Early route uses windows; advanced Caser/SASRec explicitly deferred after CNN/Transformer prerequisites.

- **foundations-34-2 · P2 · fixed** — HConv/VConv скрывают сам механизм Caser за именами функций; не заданы количество фильтров, выходные формы и оси pooling.

  Disposition: Horizontal/vertical shapes and pooling axes unfolded; actual tensor conv verifies outputs[2,1],[0,-1],rep[0,1],score.7.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: [https://d2l.ai/chapter_recommender-systems/seqrec.html](https://d2l.ai/chapter_recommender-systems/seqrec.html) — Actual D2L Caser horizontal max-pooling and vertical full-height convolution code inspected; independent torch tensor calculation verified.

### 35. 00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация.md

Before SHA-256: `b3e4fdc4a4e740f643ab75b28e62ec14732903f60318096f2095a6ded1c946a8`  
After SHA-256: `c94f8d35036b29e37aa1b157155e75c860c7380149565eba6682dc821e94461d`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация.md`

- **foundations-35-1 · P1 · already-fixed** — Evaluation-код возвращает positives после forbidden/availability filters, а текст требует принудительного присутствия test positive. Это меняет допустимый каталог и маскирует потерю target ранними этапами; противоречит заявленному end-to-end contract и candidate recall ниже.

  Disposition: Pre-task end-to-end replay already preserves eligible missed targets in denominator without reinsertion; sampled benchmark explicitly separate.

- **foundations-35-2 · P2 · already-fixed** — Карточка одновременно задаёт retrieval_top_1000 и evaluation: full_catalog; неясно, какой universe реально scored и что считается промахом retrieval.

  Disposition: Pre-task card already agrees: retrieval_top1000 within eligible universe and evaluation=end_to_end_replay.

- **foundations-35-3 · P2 · fixed** — Модуль обещает научить online experiment, но раздел содержит только решения/логи: нет estimand, доверительного интервала эффекта, мощности/MDE или обработки многократного просмотра.

  Disposition: Overview boundary, source and ITT user-level binary example; effect2pp,CI[1.133,2.867]pp,planning MDE1.188pp and fixed-look requirement.

Checks: task-1-page-code-checks.py: PyTorch2.8 CPU actual page BPE functions; independent MF/BPR/Caser/A-B calculations passed. Other worked arithmetic reviewed directly.

Source evidence: [https://ai.stanford.edu/~ronnyk/GuideControlledExperiments.pdf](https://ai.stanford.edu/~ronnyk/GuideControlledExperiments.pdf) — Primary controlled-experiment guide opened; numerical independent user-level Bernoulli fixed-look Wald CI and planning MDE explicitly derived locally.

### 36. 00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель.md

Before SHA-256: `4bd7364fb3116302eae587d3449ac6048454faec888a7df742154bfe8bfe9ccd`  
After SHA-256: `91357ea247a1316c9b3d68bcabc4d184d9ed6be3d698d3ace620fb0b213a2024`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель.md`

- **foundations-36-1 · P3 · fixed** — Раздел назван идеей Kneser–Ney и действительно остаётся идеей: нормировка continuation, discount D и веса backoff не даны. Читатель не может собрать нормированное распределение метода.

  Disposition: Defined normalized single-discount bigram KN, continuation counts, lambda and unseen-context fallback; five-type transition table gives row(.1,.675,.225), sum1. Full modified algorithm linked to Heafield§3.

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: [https://aclanthology.org/P13-2121.pdf](https://aclanthology.org/P13-2121.pdf) — §3.2 adjusted counts and discount estimates; §3.3 normalization. Current SLP3 no longer has full KN derivation, so added direct primary reference instead.

### 37. 00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction.md

Before SHA-256: `433e794a7bf86c1041144af434f11125e5aea3011dd6387e3aad11b3f3538fc3`  
After SHA-256: `b0205fe2b566cdd98d67b74e3a338a60ec980c40aa70ff82343fb01943b9210b`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction.md`

- **foundations-37-1 · P2 · fixed** — Слово «только» превращает типичный путь создания assistant в необходимое условие. Возможность продолжать диалоговый prompt и надёжное следование инструкциям — разные утверждения; исключительность требует проверки первоисточником.

  Disposition: Base dialog continuation separated from reliable assistant post-training; Brown and Ouyang primary papers consulted.

- **foundations-37-2 · P3 · fixed** — Не оговорён контракт PAD/EOS: при совпадении идентификаторов будут исключены и настоящие EOS-targets.

  Disposition: Position-valid shift to -100 preserves real EOS even with shared PAD/EOS ID; actual snippet executed on [BOS,cat,EOS,PAD].

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) — In-context few-shot transfer without gradient updates. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) — Instruction and human-feedback post-training shapes assistant behavior.

### 38. 00 Учебник/03 Языковое моделирование/02 Sampling и генерация.md

Before SHA-256: `00eaf06d00cd91d942893b9b52d8ff84f978181c85ca3195e89f25277f00c1e0`  
After SHA-256: `2882b2cef731ec974ecaf8f8017b72066e92e92c830109e5971a5686a8b4ab5e`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/03 Языковое моделирование/02 Sampling и генерация.md`

- **foundations-38-1 · P2 · fixed** — Приведённый цикл не реализует собственную оговорку главы о temperature=0: такой режим делит logits на ноль вместо перехода к argmax.

  Disposition: Explicit T=0 argmax, finite nonnegative temperature validation; actual function tested including tiny positive T and invalid values.

- **foundations-38-2 · P2 · fixed** — Главный алгоритм nucleus sampling скрыт в неопределённой функции; ни сортировка, ни правило сохранения токена, пересекающего порог, ни нормировка не разобраны на числах.

  Disposition: Executable nucleus filter includes threshold-crossing token; log(4,3,2,1) examples T1→(4,3,2)/9 and T.5→(16,9)/25 verified in PyTorch.

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751) — Nucleus sampling and unreliable tail truncation; own numerical and implementation checks supplement source.

### 39. 00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT.md

Before SHA-256: `0615bb1b31404689275621bae2f7e463a8b4a8433038915a384ceb7f56c59f02`  
After SHA-256: `7d5a399b7264f41015ef6be3262a30c7611cb1853e832a60bb26e722fd3f9bf5`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT.md`

- **foundations-39-1 · P2 · fixed** — В сумме вкладов по времени не различены полная зависимость h_t от разделяемой W_hh и локальная производная при фиксированном h_{t-1}. При чтении обоих множителей как полных производных получается двойной учёт путей.

  Disposition: Adjoint recursion distinguishes local output gradient and future influence; sum delta_t h_(t-1)^T avoids double-counting. Actual rnn_forward autograd scalar gradient .2533642027283495.

- **foundations-39-2 · P3 · fixed** — Порядок произведения матричных Якоби не определён; для столбцовых состояний нужен J_t J_{t-1} ... J_{k+1}.

  Disposition: Column-vector Jacobian product explicitly J_t J_(t-1)...J_(k+1); norm bound distinguishes sufficient contraction from merely large norms.

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 40. 00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN, LSTM и Seq2Seq.md

Before SHA-256: `221636b6edf025eabefd85004e24a28453f90d99967b5d46ea6dcd3e19a99438`  
After SHA-256: `601ad8c301da506673bef58d360ed171bf1baa95c1c026d79b866e5516a8ff9f`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN, LSTM и Seq2Seq.md`

- **foundations-40-1 · P2 · fixed** — Legacy-глава фактически повторяет три соседние канонические страницы, но не объясняет читателю, является ли она альтернативным коротким маршрутом или устаревшим материалом. Повтор формул LSTM и перехода к bottleneck размывает последовательность учебника.

  Disposition: Legacy chapter explicitly optional visual/historical overview with canonical RNN→LSTM→Seq2Seq links; all strong source explanations preserved.

- **foundations-40-2 · P3 · fixed** — Код даёт формулы вентилей, но не показывает численного прохода состояния, который уже есть в канонической LSTM/GRU.

  Disposition: Declared Linear(H+D,4H), batch shapes, own gate order vs PyTorch and linked existing numeric LSTM walkthrough. Actual snippet shapes executed.

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: [https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) — Variables show two3H biases; note distinguishes reset placement. [https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html) — Variables show two4H biases and i,f,g,o packed order. Runtime checks separately use2.8.

### 41. 00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU.md

Before SHA-256: `c32ff578e78d6a47b90c691682b14abe946579f63655dae6c0a7239d88ee4f15`  
After SHA-256: `f977e07887b104b327e3ea8e60c5a815e0c172767983b6c1e4965671c5c82b74`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU.md`

- **foundations-41-1 · P3 · fixed** — Подсчёт соответствует одной bias на gate; готовые реализации могут иметь отдельные input/recurrent biases, поэтому число не следует непосредственно сверять с API.

  Disposition: Single-bias convention vs separate input/recurrent biases explicit; actual torch2.8 GRU(2,3)=63 and LSTM(2,3)=84 numel verified. API reset-gate variant noted with primary docs.

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: [https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) — Variables show two3H biases; note distinguishes reset placement. [https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html) — Variables show two4H biases and i,f,g,o packed order. Runtime checks separately use2.8.

### 42. 00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора.md

Before SHA-256: `7c90a4aeb04293d2ea4eaa0a4677dcc7ab9b755b2e4d24f0ba8b045b4f774bf1`  
After SHA-256: `d656fe42e038122df03bb8c16c9f2031e761d8d4ab51c53d9b72b34fcd7b8990`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора.md`

- **foundations-42-1 · P3 · fixed** — Псевдокод не определяет судьбу уже завершившихся EOS-гипотез: если расширять каждую, завершённая последовательность может снова продолжиться.

  Disposition: Finished EOS beams carried unchanged; explicit s/length^.6 rule and None on no completed beam at limit; actual pseudocode executed with deterministic expansion stub that rejects EOS expansion.

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 43. 00 Учебник/05 Attention и Transformer/00 Источники и визуальный стандарт.md

Before SHA-256: `f5382a111a9b4a9100a74d1ec12622173198083b4ba1e2fedd21e4caf8e38378`  
After SHA-256: `0471ff58a69773e5be22a9814daf5ac118863c61cba6ff4eab601eec63cd6753`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/05 Attention и Transformer/00 Источники и визуальный стандарт.md`

- **foundations-43-1 · P2 · fixed** — Редакционная инструкция разрешает самостоятельно закрывать визуальный пробел SVG, тогда как текущий AGENTS.md требует искать исходный учебный материал и не создавать свои SVG/Mermaid. Одновременно строка57 объявляет Alammar только reference, хотя соседние канонические главы используют его иллюстрации.

  Disposition: SVG/Mermaid fallback removed, original figures/attribution preserved; Alammar role reconciled and new-object rights distinguished from attribution, following primary clarification.

- **foundations-43-2 · P3 · fixed** — Паспорт описывает старый состав: назначенный корпус содержит также отдельные Masking и Позиционная информация.

  Disposition: Current five-page route includes Masking and Positional information; stale three-chapter claim removed.

Checks: task-1-page-code-checks.py actual CPU snippets plus direct KN scalar normalization

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 44. 00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer.md

Before SHA-256: `0296d53c5fd823835afaee64e448658816dd1d17d25a8b925a3f7074e52e466e`  
After SHA-256: `0332b8584e2d3bded9f082d7a7ebfd3f62d61b8fa095fd4958cc6f318881c476`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer.md`

- **foundations-44-1 · P3 · fixed** — Мини-реализация работает только с равными действительными длинами либо заранее обрезанной памятью: padding keys в батче не маскируются, хотя shapes имеют batch-ось.

  Disposition: Source-length mask applied before softmax; right-padding contract, device requirement, empty-source rejection and left-padding alternative explicit. Actual snippet tested.

Checks: Actual current additive-attention snippet torch2.8 CPU: lengths[3,1], uniform scores, weights[1/3]*3 and[1,0,0]; excluded padded memory90/80 etc; empty source rejected.

Source evidence: Entire current page and its existing source references read; new disposition rests on the explicit mathematical/data-flow counterexample or local contract recorded per finding. No claim that all outgoing sources were independently re-audited online.

### 45. 00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md

Before SHA-256: `819b395e711dc964260bf3f755651b4d03fcf893ad66d3272595b687675a7c20`  
After SHA-256: `7f6fa6fcf5e9cd1b451395712c36849b0488461bfcb0f7222fb6c27f5cc07796`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md`

- **foundations-45-1 · P2 · fixed** — В строке86 q_i=x_iW_Q задаётся как строковый вектор, а q_i^T k_j для строк даёт внешнее произведение, не скаляр. Поздний численный пример q_3 K^T использует именно строковое соглашение.

  Disposition: RU/EN both consistently use row vectors q_i,k_j with s_ij=q_i k_j^T, coordinate sum and explicit shapes. Variance notation also unified; numerical example rounding corrected to alpha(.248255,.248255,.503490),z(1.758725,1).

Checks: Independent exact arithmetic alpha1=.24825507825772308,z1=1.7587246087113846.

Source evidence: [https://arxiv.org/html/1706.03762v7](https://arxiv.org/html/1706.03762v7) — Primary attention projection definition inspected in this task; row-vector algebra and actual RU/EN code independently tested.

EN: `en/00 Textbook/05 Attention and Transformer/02 Self-Attention — Q, K, V.md`; full before/after read. Before `e100e0b09d40b8ee09ee6bd382a6c2e9572707b09b5184b92f3ee5397e4661b3`; after `0abb7aad9c2d76b10cb0894cc69c4f3c321eadaf7045d790e07a7c94b52d1b8a`.

### 46. 00 Учебник/05 Attention и Transformer/03 Полный Transformer.md

Before SHA-256: `caa31cfdff06d1151c51f1a1e2d3482e95b897ffeaf7f558d06394dca83c2ede`  
After SHA-256: `566e8fdbd9b0d8b6ccdeb5dfc48b41e22964707ab571f8caa57571845876c8e6`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/05 Attention и Transformer/03 Полный Transformer.md`

- **foundations-46-1 · P2 · fixed** — Утверждение верно для немаскированного attention, но в главе обсуждается и causal block. Фиксированная causal mask нарушает произвольную перестановочную симметрию; NoPE показывает возможность извлекать позицию без explicit PE.

  Disposition: Full-attention premise explicit; causal mask implicit positional signal explained with Haviv/Kazemnejad NoPE papers in RU/EN.

- **foundations-46-2 · P2 · fixed** — Формула приравнивает Q/K/V исходным состояниям, опуская обучаемые проекции, хотя предыдущая глава специально объясняет их различие.

  Disposition: Q=HdecWQ,K=HencWK,V=HencWV, per-head shapes and final encoder-stack source explicit in both prose and figure caption.

- **foundations-46-3 · P3 · fixed** — Логарифмический путь относится к dilated convolution stack. Первоисточник рядом с таблицей отделяет его от O(T/k) для contiguous kernels; эта оговорка при переносе потеряна.

  Disposition: Dilated convolution label restored; ordinary contiguous O(T/k) vs logarithmic dilated stack path distinguished, per-layer cost vs stack path explained.

Checks: 

Source evidence: [https://arxiv.org/html/1706.03762v7](https://arxiv.org/html/1706.03762v7) — §3.1 final encoder stack output,§3.2 projection roles,§4 contiguous/dilated path lengths. [https://arxiv.org/abs/2305.19466](https://arxiv.org/abs/2305.19466) — NoPE can represent absolute and relative positional encodings. [https://aclanthology.org/2022.findings-emnlp.99/](https://aclanthology.org/2022.findings-emnlp.99/) — Causal LMs acquire positional information without explicit encoding.

EN: `en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md`; full before/after read. Before `b2d96e75c568d57c178374e6e58f2474ba8e51fb878872e5261903fdd1c5fe28`; after `dbb6f8470d6ad92af518013a8ecbbdaacdcc49fdd10655a50309154e727e0223`.

### 47. 00 Учебник/05 Attention и Transformer/04 Позиционная информация.md

Before SHA-256: `97b3006e8e304cf5b0cd00355e27beffe611ce00a0e065e0e3cd8b8cf51616f7`  
After SHA-256: `5f1acc5a26a9769fcbf133bc4302bd52ab0dbdb695c82a8dcc19911f6e221a2a`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/05 Attention и Transformer/04 Позиционная информация.md`

- **foundations-47-1 · P2 · fixed** — В примерах меняются словоформы собака/собаку, человека/человек, укусила/укусил. Это не перестановка тех же токенов и не корректный контроль изолированного влияния порядка.

  Disposition: Both locales use identical-token John loves Mary / Mary loves John control, no inflection or implicit lemmatization.

- **foundations-47-2 · P2 · fixed** — Категоричность неверна для causal Transformer: causal mask уже несёт асимметрию, а NoPE может представлять абсолютную и относительную позицию. Перестановочная эквивариантность первых строк требует full attention и соответствующего преобразования маски.

  Disposition: Full unmasked equivariance F(PX)=PF(X) and joint mask permutation M→PMP^T explicit; fixed causal mask can carry implicit positional signal, primary NoPE references.

- **foundations-47-3 · P2 · fixed** — После строковых проекций x_i W_Q записан столбцовый dot product; ниже RoPE также начинает умножать R_i слева без объявления смены соглашения.

  Disposition: Row shapes retained consistently through baseline/Shaw/T5/ALiBi dot products and RoPE qR_i^T rotation, derivation and numeric example; independent tensor identity and shift tests pass.

Checks: task-1-page-code-checks.py: row-rotation dot identity and common-shift invariance across3 signed index pairs; QKV example executed in both locales.

Source evidence: [https://arxiv.org/abs/2305.19466](https://arxiv.org/abs/2305.19466) — NoPE can represent absolute and relative positional encodings. [https://aclanthology.org/2022.findings-emnlp.99/](https://aclanthology.org/2022.findings-emnlp.99/) — Causal LMs acquire positional information without explicit encoding. [https://arxiv.org/html/2104.09864v5](https://arxiv.org/html/2104.09864v5) — §3.2 equations show R_m^T R_n=R_(n-m); translated column convention into explicitly declared row convention.

EN: `en/00 Textbook/05 Attention and Transformer/04 Positional information.md`; full before/after read. Before `67045dd4ca11aa5c442195576c0aad1b7ffab96b67b7299948f5785cc6bfd640`; after `10de5dfa9cea752b88dfacf9775602a7b1ef7aec5deafac02ce8dfaaed6e2a17`.

### 48. 00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров.md

Before SHA-256: `6d0f42d84033bca4046f0a956fa98f088829324d6caf0c2a436eb6fcdee3b270`  
After SHA-256: `84cb6955535e0282241071216bfd3c7e48023df1ec42b193ab0b610f85817b3e`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров.md`

- **foundations-48-1 · P2 · fixed** — Проверка требует сумму1 от всех строк, хотя выше прямо оговорены полностью masked строки и NaN; для left-padding пример либо падает, либо после корректной zero-output политики всё равно не проходит.

  Disposition: Implemented pre-softmax finite placeholder for empty rows, zero-output/query-padding policy and valid-only normalization assertions. Actual RU/EN snippets pass all/no/right/left padding with finite backward.

- **foundations-48-2 · P3 · fixed** — Q/K/V теперь обозначают входы до проекций, тогда как в предыдущей главе и ниже по странице — уже проекции. Без объявления это выглядит как повторное применение W_Q/W_K/W_V.

  Disposition: Introduced X_Q/X_K/X_V as pre-projection sources; projected Q/K/V preserve previous chapter semantics.

Checks: Actual RU/EN PyTorch 2.8 CPU code: row sums [0,0,1,1] for left padding; all-padding zero weights/output; finite q/k/v gradients; forbidden future zero.

Source evidence: PyTorch 2.14 SDPA documentation inspected: True permits attention, opposite MHA key_padding_mask; runtime tested separately with 2.8.

EN: `en/00 Textbook/05 Attention and Transformer/03 Masking, multi-head attention, and tensor shapes.md`; full before/after read. Before `1d07498b1ba18f1e450aba8390ab5ae04ba8b1c57ae3a82ee556ecad528fb5f7`; after `f3828a4a6b0f43c517f3b3e7c73a8bbc75d56a57a536747cf90786aa43c89a27`.

### 49. 00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна.md

Before SHA-256: `6f9343b2e59cfa283674a9982717e93d08437cf362a5bc0d0735458ed0634029`  
After SHA-256: `df74468c152bba2498a23f446ebe257e62f6bc3f573c8265280bf6971e58d909`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна.md`

- **foundations-49-1 · P2 · fixed** — Категорическое объяснение запрещает BERT видеть target на выбранной позиции, но BERT намеренно оставляет10% выбранных позиций неизменными. Это подтверждено Appendix A.1 первичной статьи и правильно описано в соседней BERT-главе.

  Disposition: MLM conditioned on corrupted input, selected-set 80/10/10 mechanism and unchanged-target loss made consistent with warnings/check; existing EN correction preserved.

- **foundations-49-2 · P2 · fixed** — Упражнение предлагает сравнивать контекстные представления многозначного слова, но в двух русских фразах нет одного и того же токена: «берегу» и «банке» — разные слова. Оно не изолирует изменение контекста при фиксированной лексеме.

  Disposition: Exercise now identical bank token and identical left context, controlled changed suffix, aligned subwords and eval mode; full vs causal prediction distinguishes mechanism from semantic benchmark.

- **foundations-49-3 · P3 · fixed** — Фраза грамматически утверждает, что encoder обучался простому autoregressive процессу, хотя весь раздел объясняет MLM.

  Disposition: Corrected both languages: original BERT trained for MLM, not autoregressive continuation.

Checks: Independent scalar selected-target loss -ln(.2)-ln(.8)=1.83258146374831.

Source evidence: [https://arxiv.org/html/1810.04805v2](https://arxiv.org/html/1810.04805v2) — §3.1/A.1: selected 15%, 80/10/10 replacements, unchanged examples intentionally retained.

EN: `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/01 Three architectural patterns.md`; full before/after read. Before `18beb8222ec5af93f58b26430ca0b51888eccd3b93fed5cbb2d964376e357ce5`; after `a7a5dc42bb01dd54c1ee9835fe1883d3ece57883fdcad22adafae05aa908267a`.

### 50. 00 Учебник/06 Encoder, Decoder и Encoder-Decoder/02 Encoder-Decoder Transformer.md

Before SHA-256: `faf71baf44a0d7310c55b15f0621abff8d70647d03d914817079d142715c311d`  
After SHA-256: `f86b01b82df2c64cde3780b5b7f295b3c42a524ce64dcb9bebcff72f3e4db925`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/02 Encoder-Decoder Transformer.md`

- **foundations-50-1 · P1 · already-fixed** — Подпись описывает другую топологию: исходный Transformer передаёт в cross-attention выход encoder-стека, а не все промежуточные encoder-слои. Это явно установлено §3.1 первичной статьи.

  Disposition: Already present before this task: caption explicitly says last encoder block/whole stack output feeds every decoder cross-attention; intermediate outputs are not separate memory.

- **foundations-50-2 · P2 · fixed** — Сдвиг показан без EOS, поэтому непонятно, как модель учится завершать перевод; в стандартном примере последнее слово является и target, и входом для предсказания EOS.

  Disposition: Added complete teacher-forcing BOS/The/cat vs The/cat/EOS rows with PAD exclusion and shared EOS/PAD positional-mask caveat.

Checks: Manual aligned four-column input/target/loss trace: targets The,cat,EOS included; PAD excluded.

Source evidence: [https://arxiv.org/html/1706.03762v7](https://arxiv.org/html/1706.03762v7) — Previously inspected §3.1: decoder cross-attention reads encoder-stack output; shifted decoder outputs.

### 51. 00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md

Before SHA-256: `9ad6ad3abddb2b18a1e19b316036f0de1da955403a0bd821eb15f2869f6b9d1c`  
After SHA-256: `287efca47e3ce89e0b4c751556ff5c9e011660c16a3b78a47cdd8fd1ff8d249e`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa.md`

- **foundations-51-1 · P2 · fixed** — Вводный пример должен показать необходимость правого контекста, но спорное слово стоит в конце обеих фраз; различающее окружение находится слева. Он не иллюстрирует заявленное преимущество bidirectionality над causal представлением этого токена.

  Disposition: Opening uses identical bank token and prefix with right-only river/office continuation; both languages.

- **foundations-51-2 · P2 · fixed** — В формуле DeBERTa не определены проекции Q^c/K^c/Q^r/K^r, функция δ и её ограничение расстояния. Из трёх качественных пояснений нельзя восстановить вычисление, особенно противоположные направления δ(i,j) и δ(j,i).

  Disposition: Defined H,P,all five content/position projections and row shapes, signed clipped delta(i,j) vs reverse, complete three-term numerical scaled logit and distinction from probability.

Checks: Independent NumPy 2.2.6 CPU dot products 5+2+2=9; scaled logit 3.6742346141747673; indices1,3 for k2 i1 j2.

Source evidence: [https://arxiv.org/html/2006.03654v6](https://arxiv.org/html/2006.03654v6) — §3.1 Eq3 clipped signed offset and Eq4 all five projections, reverse delta in position-content and sqrt(3d) scale inspected.

EN: `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/03 BERT, RoBERTa, and DeBERTa.md`; full before/after read. Before `534756abd309bf9469862e4492d459c639707e44bec7f51495867c9a5621a00c`; after `3103bdee9ad3082b82895151ac86e5d915ba867df9359f0aa64cb6d45a73ed34`.

### 52. 00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md

Before SHA-256: `f6ce0b74d3f005d6603e59e8baffafbc59b84cf770ba20009ccc907f62176507`  
After SHA-256: `f10f0a3e21d7ea7b207420c0ea0a00210af1d6bdc1da9475a656939421ee38f9`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md`

- **foundations-52-1 · P2 · fixed** — Центральное преобразование задачи в последовательность объяснено словами, но ни одна задача не доведена до конкретной полной строки, индексированного состояния Extract, logits и loss. Это оставляет главу историческим обзором вместо воспроизводимого учебного разбора её главного приёма.

  Disposition: Full Start/premise/Delim/hypothesis/Extract example indexed0..8; h_l^8 to class head shape and toy logits/probabilities/class CE plus explicit auxiliary LM calculation, no label input leakage.

- **foundations-52-2 · P3 · fixed** — L1 выше является суммой логарифмов вероятностей, а L2/L1 здесь названы просто правдоподобиями; не указано, максимизируется L3 или минимизируется его отрицание.

  Disposition: L1/L2/L3 consistently log-likelihood sums maximized; minimized -L3=classification CE+lambda LM CE, reduction caveat.

Checks: PyTorch CPU toy logits(1,0,-1) softmax(.6652409558,.2447284711,.0900305732), CE.4076059644; plus .1*8ln2=.9621237089.

Source evidence: [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — §3.1–3.3 equations1–5 inspected: maximize log objectives, task-specific input transformation, final hidden readout.

EN: `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training.md`; full before/after read. Before `ef937dcf4b0af32ed77be3e25bdc44389b594a4ace737f3f96359d19aca657ae`; after `41e5825e5e28d91e836314306623fd2b6abc80774ca3b1b99b9d90849521b204`.

### 53. 00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md

Before SHA-256: `f4537781d085b84de4f8743fe72573cda145505e773e35ef73baac37c712d423`  
After SHA-256: `ba02e4131bb09eb334254eb90a7affa6c761ea4508701c27400c42f8846dfa34`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md`

- **foundations-53-1 · P2 · fixed** — Это историческое употребление GPT-2 не отделено от современного определения по числу демонстраций. Первоисточник §3.7 и §3.8 уже использует example translation/QA pairs; следующая глава определяет zero-shot как отсутствие demonstrations, а переход строки161 создаёт впечатление, что примеры в контексте впервые добавил GPT-3.

  Disposition: Separated historical no-gradient transfer from strict zero-demonstration prompting, cited GPT2 §3.7/3.8 example pairs; complete two-shot translation prefix, generated-vs-input boundary, no fabricated measured output; GPT3 systematic comparison not first demonstrations. Also corrected CoQA comparator attribution to paper's near-human89 F1 wording.

Checks: No model inference claimed; manually checked complete2demonstrations+unfinishedquery layout and exact frozen-weight regime.

Source evidence: [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — §3.5 CoQA55F1 and near-human89 baseline wording; §3.7 translation example pairs+greedy first sentence; §3.8 QA seeded pairs inspected.

EN: `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language.md`; full before/after read. Before `70bf5a2c3b1f2db6501b550252a09901aa71e6ca93e01ce9ce6963ef36a27234`; after `1eafd960b14038c8b001c3f1baf6374a96ffcbcec97be7f2522e20c545571c56`.

### 54. 00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md

Before SHA-256: `0e9e682f4b14cdb74a91d297a749a1df7b3e463b06f342e54f09412815772665`  
After SHA-256: `64b8fc86a86e69fbac77f85eb9c096f0f44a2466ace9e7cca0604cd2e62403ad`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md`

- **foundations-54-1 · P2 · fixed** — Обещанный few-shot пример остаётся перечислением этапов: нет самой инструкции, ни одной пары отзыв→метка, нового отзыва и расчёта вероятности полной метки. Поэтому утверждение об учёте всех токенов метки невозможно проверить на примере главы.

  Disposition: Complete instruction+two reviews/labels+unansweredreview prompt in both locales, explicit toy tokenization and all-label log-prob arithmetic with ranking reversal; restricted normalization, terminator/length caveats, demonstration order and evaluation separation.

Checks: Scalar log scores -1.7147984280919268 and -1.1394342831883648; candidate normalization .36/.64; no GPT3 model inference claimed.

Source evidence: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) — Previously inspected Brown abstract: few-shot demonstrations solely in text, no gradient updates. New prompt/numbers explicitly independently authored teaching example.

EN: `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning.md`; full before/after read. Before `2d10254fcd15aeee1151a341b74a86703e4f9badd7fa0e71324dbaaeaaa726aa`; after `6b2634e62bd0c53a293e541aa1b48f4fd814e39ac7841c10f6fbefc2842ec533`.

### 55. 00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer.md

Before SHA-256: `5dcb1a4d1ca0183225cd9520d25d19ed52eee5e3f422cf4b2a81876d8d4d81ad`  
After SHA-256: `c96dce53be2521aa573181be175e0f979c9cb52677a5ade31ff04c3f96ca0df8`  
Snapshot: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-1/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer.md`

- **foundations-55-1 · P3 · fixed** — Не раскрыт механизм, обеспечивающий среднюю длину span≈3. Простое независимое маскирование15% позиций с объединением соседей такую среднюю длину не задаёт; из текста нельзя воспроизвести corruption sampler.

  Disposition: Exact official random_spans_noise_mask helper parameters and rounding/clipping/count selection, uniform positive partitions, alternating clean/noise spans, reproducible seeds; N40 example 6tokens2spans mean3. Short original sentinel illustration explicitly manually chosen, not15% sample. Default no-roll ending and closing sentinel boundary distinguished from figure.

Checks: Independent exhaustive 5*33=165 positive composition pairs at N40,m6,s2: every mask40tokens,6noisy,2spans,mean3. Official TensorFlow helper inspected but NOT executed; no TensorFlow environment claim.

Source evidence: [https://raw.githubusercontent.com/google-research/text-to-text-transfer-transformer/main/t5/data/preprocessors.py](https://raw.githubusercontent.com/google-research/text-to-text-transfer-transformer/main/t5/data/preprocessors.py) — Inspected random_spans_noise_mask2705–2808, random segmentation2763–2780, nonnoise_span_to_unique_sentinel2971–2980; default no-roll begins clean ends noise; count safeguards and optionalroll explicit.

EN: `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/07 T5 — text-to-text Transformer.md`; full before/after read. Before `003721171a06d5efbf43b86e1037253ec258e6d10c4140b5c9003c789d9d4d86`; after `a1a59d9354dcbed22da3b83a8dfd70c953629a87793f5e1e6404ef3c615a9bb1`.
