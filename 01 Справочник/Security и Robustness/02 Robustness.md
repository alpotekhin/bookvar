---
title: "Robustness ML-систем"
type: concept
status: canonical
last_updated: 2026-07-24
primary_sources:
  - https://mlsysbook.ai/vol2/robust_ai/robust_ai.html
  - https://arxiv.org/abs/1706.06083
---

# Robustness ML-систем

**Robustness** — способность ML-системы сохранять приемлемое поведение при
отклонении от условий, на которых её проверяли. Отклонение может возникнуть в
мире (distribution shift), быть сконструировано противником (adversarial input
или poisoning) или появиться внутри вычислительной системы (bit flip,
неверная предобработка, численная ошибка). Общий симптом — правдоподобный, но
неверный результат без обязательного падения процесса.

> [!info] Полный курс
> [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|Robust AI]]
> — полный оригинальный текст главы Vijay Janapa Reddi et al. с выводами,
> экспериментами, кодом, case studies и иллюстрациями. Источник:
> [mlsysbook.ai](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html), версия
> `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`, лицензия
> [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Три источника нарушения устойчивости

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/f379357456b6019c3804c10b13f2b37a9769547e.svg]]

*Environmental shift, input-level attacks и system-level faults требуют разных
сигналов и защит. Оригинальная Figure 7 из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

1. **Environmental shift.** Мир изменился: новые пользователи, устройства,
   язык, сезон, политика или причинная связь между входом и целью.
2. **Input-level attack.** Противник подбирает запрос или обучающий пример,
   максимизирующий конкретный сбой.
3. **System-level fault.** Код, pipeline, сеть или hardware повреждают вход,
   состояние или вычисление.

Важно различать причину, а не только симптом. Ошибка scaler после deployment
выглядит для drift detector как резкий covariate shift. Corrupted activation
может выглядеть как adversarial example. Полезны три диагностических признака:

- **граница изменения:** естественный shift чаще развивается по сегментам и
  времени, pipeline bug возникает ступенькой после релиза;
- **воспроизводимость:** software/hardware fault коррелирует с версией,
  узлом или path; drift — с population/context;
- **контрольный путь:** повтор на trusted pipeline или другом hardware помогает
  отделить данные от вычисления.

## Distribution shift и concept drift

Пусть обучение описывает \(P_{\text{train}}(X,Y)\), а эксплуатация —
\(P_{\text{prod}}(X,Y)\).

- **Covariate shift:** меняется \(P(X)\), но \(P(Y\mid X)\) считается
  неизменным.
- **Label/prior shift:** меняется \(P(Y)\), а \(P(X\mid Y)\) примерно
  сохраняется.
- **Concept drift:** меняется сама зависимость \(P(Y\mid X)\); старое правило
  больше не описывает мир.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/shift-types.svg]]

*Различие сдвига входов, частот классов и условной зависимости. Оригинальная
иллюстрация из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальный оригинал]].
Лицензия CC BY-NC-SA 4.0.*

Без labels можно наблюдать только входы, embeddings, confidence, entropy и
долю abstention. Эти сигналы обнаруживают изменение, но не доказывают падение
качества. Приходящие с задержкой labels нужны для оценки ошибки и concept
drift. Хороший мониторинг хранит reference window, current window, срезы,
порог, минимальный размер выборки и correction для множества признаков.

Распространённые меры:

- Population Stability Index
  \(\mathrm{PSI}=\sum_i(a_i-e_i)\ln(a_i/e_i)\);
- KL divergence
  \(D_{\mathrm{KL}}(P\|Q)=\sum_xP(x)\ln(P(x)/Q(x))\);
- Jensen–Shannon divergence — симметричная ограниченная версия;
- Kolmogorov–Smirnov statistic для одномерных непрерывных признаков;
- Maximum Mean Discrepancy для многомерных представлений.

Ни один универсальный порог PSI не отделяет «безопасный» мир от опасного. Порог
калибруют на исторических вариациях и связывают с реальной потерей качества.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/robust_ai_psi_drift_knee.svg]]

*PSI становится полезен после привязки статистического сдвига к операционному
действию. Оригинальная иллюстрация из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

Ответ на drift выбирают по причине: recalibration при изменившихся вероятностях,
новый threshold при изменившейся цене ошибок, retraining при накоплении
репрезентативных labels, domain adaptation при новом домене, fallback или
abstention при недостаточной уверенности. Автоматическое переобучение на
непроверенном feedback способно закрепить ошибку или poisoning.

## Adversarial examples

Для классификатора adversarial attack решает внутреннюю задачу

$$
\delta^\star =
\arg\max_{\|\delta\|_p\le\varepsilon}
\mathcal L(f_\theta(x+\delta),y).
$$

FGSM делает один шаг

$$
x_{\text{adv}} =
x+\varepsilon\,\operatorname{sign}
\left(\nabla_x\mathcal L(f_\theta(x),y)\right),
$$

а PGD повторяет шаг и проецирует результат обратно в допустимую
\(\ell_p\)-окрестность. Targeted attack оптимизирует выбранный класс; untargeted
достаточно любого неверного результата. White-box attack использует gradients,
black-box — queries или transferability с surrogate model.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/gradient-attack.svg]]

*Gradient-based атака движет вход в сторону роста loss, сохраняя ограничение на
perturbation. Оригинальная иллюстрация из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальный оригинал]].
Лицензия CC BY-NC-SA 4.0.*

Norm ball — математическая модель противника, а не синоним незаметности.
\(\ell_\infty\)-ограничение на пиксели плохо описывает наклейку на дорожном
знаке, поворот камеры или смысловую замену слова. Robust accuracy публикуют
вместе с attack, norm, \(\varepsilon\), количеством шагов, restart и доступом
противника. Иначе результат невоспроизводим.

## Adversarial training и сертификация

Adversarial training приближённо решает minimax-задачу:

$$
\min_\theta
\mathbb E_{(x,y)}
\left[
\max_{\|\delta\|_p\le\varepsilon}
\mathcal L(f_\theta(x+\delta),y)
\right].
$$

Внутренний PGD делает обучение дороже; устойчивость обычно растёт в пределах
выбранной модели угроз и может уменьшить clean accuracy. Слабая внутренняя атака
создаёт ложное ощущение защиты. Оценивать нужно более сильной или независимой
adaptive attack, которая знает defense.

Input preprocessing, denoising и adversarial detectors иногда повышают цену
атаки, но противник может дифференцировать сквозь обработку, аппроксимировать её
или оптимизировать обход detector. Gradient masking распознают по признакам:
итеративная атака не сильнее одношаговой, black-box неожиданно сильнее
white-box, а увеличение бюджета не ухудшает результат.

**Certified robustness** доказывает постоянство предсказания в формально
заданной области. Randomized smoothing строит сглаженный классификатор по
шумным копиям входа и даёт certified radius в \(\ell_2\). Сертификат строг
только для своей области и не гарантирует устойчивость к semantic shift,
poisoning или hardware corruption.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/robust_ai_robustness_tax.svg]]

*Таксономия различает эмпирические и сертифицированные защиты и не позволяет
сводить их к одному показателю. Оригинальная иллюстрация из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

## Poisoning и backdoors

Poisoning меняет train distribution через небольшое число контролируемых
примеров. Backdoor особенно опасен тем, что clean validation остаётся хорошей:
trigger связывается с целевым ответом и проявляется только после deployment.
В federated learning malicious client может отправлять crafted update вместо
явно плохих samples.

Защита начинается с supply chain:

- identity и репутация источников;
- неизменяемые manifests, hashes и lineage;
- разделение загрузки, разметки и утверждения набора;
- статистика по источникам и версиям;
- ручная ревизия кластеров подозрительных примеров.

Naive z-score и label consistency не ловят образцы, специально сделанные
правдоподобными. Spectral signatures ищут отдельное направление в activation
space; influence methods оценивают, какие train examples сильнее всего меняют
выбранный прогноз; robust aggregation/trimmed mean ограничивают аномальные
client updates. Эти методы вычислительно дороги и зависят от предположений,
поэтому provenance нельзя заменять постфактум-фильтром.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/0d8106a3c7ca65302fa306b4b86e5170bc79e78f.svg]]

*Защита от poisoning начинается на входе данных и продолжается в
представлениях, objective и аудите. Оригинальная иллюстрация из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальный оригинал]].
Лицензия CC BY-NC-SA 4.0.*

Huber loss ограничивает вклад экстремального residual:

$$
L_\delta(a)=
\begin{cases}
\frac12 a^2,& |a|\le\delta,\\
\delta(|a|-\frac12\delta),& |a|>\delta.
\end{cases}
$$

Она снижает влияние gross outliers, но не является универсальной защитой от
clean-label backdoor. Data augmentation и self-supervised pretraining могут
уменьшить зависимость от хрупких shortcuts, но тоже не дают гарантии.

## System-level faults и silent corruption

Устойчивость модели зависит от hardware и распределённого исполнения.
Единичный bit flip в exponent веса способен увеличить его на порядки и
насытить последующие активации, не вызвав exception.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/weight-corruption.svg]]

*Одна ошибка в разряде exponent выводит вес далеко за обычное распределение и
распространяет искажение по forward pass. Оригинальная Figure 5 из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

Если вероятность ошибки одного компонента за интервал равна \(p\), то для
\(N\) независимых компонентов вероятность хотя бы одной ошибки:

$$
\Pr(\ge1)=1-(1-p)^N.
$$

Малый per-device rate становится обычным событием в большой fleet. Sharding
увеличивает поверхность: отказ одного pipeline stage срывает запрос, corrupted
gradient одного worker может повредить глобальное обновление. Quantization,
pruning и более узкие числовые форматы экономят ресурсы, но иногда уменьшают
robustness margin.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/cell-4-output-1.png]]

*Вероятность хотя бы одной silent corruption быстро растёт с числом устройств.
Оригинальный расчёт и график из
[Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/robust_ai|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

Практические контроли: ECC и hardware telemetry, checksum/hash весов,
численные invariants, shadow execution на независимом пути, canary release,
replication критических вычислений, quarantine подозрительного узла и
checkpoint rollback. Нельзя ограничиваться uptime: silent data corruption
сохраняет доступность и разрушает правильность.

## Как оценивать robustness

Один набор «искажённых данных» не доказывает общую устойчивость. Evaluation
matrix должна разделять:

| Ось | Что фиксировать |
|---|---|
| Clean | качество и калибровка на исходном distribution |
| Natural shift | домен, время, устройство, язык, severity |
| Corruption | тип, интенсивность, случайность |
| Adversarial | access, norm/semantic constraint, budget, adaptive attack |
| Poisoning | доля контроля, target, clean-label/backdoor, source access |
| System faults | слой, fault model, injection rate, detection latency |
| Operations | fallback quality, recovery time, coverage мониторинга |

Показатели включают clean и robust accuracy, worst-group performance,
calibration under shift, attack success rate, certified radius, mean time to
detection/recovery и долю запросов, безопасно обработанных fallback. Среднее
нужно дополнять worst-case slice и доверительными интервалами.

## Минимальный production-процесс

1. Записать ожидаемые shifts, adversary и fault model до выбора тестов.
2. Собрать reference distributions и delayed-label pipeline.
3. Проверить natural corruptions и независимые adaptive attacks.
4. Ввести provenance и quarantine до попадания данных в training.
5. Запустить fault injection на preprocessing, weights, workers и network.
6. Определить abstention/fallback, rollback и владельца каждого alert.
7. После релиза сопоставлять drift с outcome metrics и релизами pipeline.
8. После инцидента обновлять threat/fault model, а не только конкретный filter.

## Связанные страницы

- [[02 Areas/ML & DL/01 Справочник/Security и Robustness/01 Security и privacy|Security и privacy]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Responsible systems|Ответственные ML-системы]]
