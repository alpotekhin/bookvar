---
title: "Security и privacy ML-систем"
type: concept
status: canonical
last_updated: 2026-07-24
primary_sources:
  - https://mlsysbook.ai/vol2/security_privacy/security_privacy.html
  - https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
---

# Security и privacy ML-систем

Безопасность и приватность защищают разные свойства. **Security** противостоит
намеренному нарушению конфиденциальности, целостности или доступности данных,
весов, pipeline и сервиса. **Privacy** ограничивает то, что можно узнать о
человеке или организации из данных, обновлений и ответов модели, даже если
система работает штатно и запрос формально разрешён. Шифрование и контроль
доступа могут предотвратить кражу датасета, но не гарантируют, что модель не
выдаст запомненный фрагмент через API.

> [!info] Полный курс
> [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/security_privacy|Security & Privacy]]
> — полный оригинальный текст главы Vijay Janapa Reddi et al. с математикой,
> case studies, таблицами, иллюстрациями и упражнениями. Источник:
> [mlsysbook.ai](https://mlsysbook.ai/vol2/security_privacy/security_privacy.html),
> версия `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`, лицензия
> [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

![[Assets/Sources/Harvard ML Systems/vol2/security_privacy/attack-surface-taxonomy.svg]]

*Поверхность атаки проходит через данные, модель, инфраструктуру и supply
chain. Оригинальная Figure 1 из
[Security & Privacy](https://mlsysbook.ai/vol2/security_privacy/security_privacy.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/security_privacy|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

## Threat model до выбора защиты

Перечень «возможных атак» бесполезен без условий. Полноценная модель угроз
задаёт:

| Поле | Вопрос |
|---|---|
| Актив | Что должно сохранить конфиденциальность, целостность или доступность? |
| Граница доверия | Где меняется владелец, полномочие или среда исполнения? |
| Возможности противника | Есть ли только API, архитектура, веса, данные, доступ к узлу или pipeline? |
| Цель | Кража весов, утечка записи, targeted failure, backdoor, отказ в обслуживании? |
| Момент | Collection, training, distribution, inference или feedback loop? |
| Наблюдаемый сигнал | По чему защита обнаружит или хотя бы ограничит атаку? |

Black-box противник видит только пары «вход–выход»; gray-box знает часть
архитектуры или данных; white-box имеет параметры и градиенты. Уровень доступа
меняет не только силу атаки, но и допустимую защиту. Скрытие confidence scores
может усложнить extraction через публичный API, но бессмысленно против человека
с копией весов.

Риск приоритизируют по вероятности и последствиям в конкретном контексте.
Membership inference для публичной модели на несекретном корпусе и для
медицинской модели на малой выборке — разные риски, хотя техника называется
одинаково.

![[Assets/Sources/Harvard ML Systems/vol2/security_privacy/threat-prioritization-matrix.svg]]

*Матрица приоритетов показывает способ распределять инженерный бюджет, а не
универсальный рейтинг угроз. Оригинальная Figure 3 из
[Security & Privacy](https://mlsysbook.ai/vol2/security_privacy/security_privacy.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/security_privacy|локальный оригинал]].
Лицензия CC BY-NC-SA 4.0.*

## Угрозы по жизненному циклу

| Слой | Основные атаки | Первые средства защиты | Сигнал обнаружения |
|---|---|---|---|
| Данные | poisoning, подмена labels, компрометация источника | provenance, подписи manifests, валидация, least privilege | сдвиг статистик, аномальные labels, новый источник |
| Модель | backdoor, trojan, кража или подмена весов | изолированное обучение, registry ACL, подпись артефакта | изменение hash, weight/behavior anomaly |
| API | extraction, membership inference, adversarial input, prompt injection | authentication, rate limit, минимальные outputs, sandbox tools | повторяющиеся boundary queries, аномальные траектории |
| Инфраструктура | side channel, firmware/dependency compromise, secret theft | secure boot, segmentation, TEE/MIG, dependency scanning | attestation, integrity и hardware telemetry |

### Model extraction и model theft

Прямая кража файла весов — инфраструктурный инцидент. **Model extraction**
восстанавливает функциональную копию через запросы: противник адаптивно
исследует границу решений или собирает ответы для distillation. Точные
вероятности классов обычно несут больше информации, чем top-1 label.

Защита сочетает rate limits на identity, анализ последовательностей запросов,
ограничение разрешения outputs, watermark/fingerprinting и юридические меры.
Ни одна мера не устраняет риск публичного API: слишком сильное округление
ухудшает продукт, а распределённый противник обходит простой лимит по IP.

### Membership inference и извлечение обучающих данных

Membership inference проверяет, участвовала ли конкретная запись в обучении,
часто используя разницу уверенности между train и unseen examples. Извлечение
данных идёт дальше и пытается восстановить содержимое записи. Риск растёт при
переобучении, повторяющихся уникальных строках, богатых outputs и неограниченном
числе запросов.

Нельзя обещать приватность только потому, что имена удалены. Квазиидентификаторы
и корреляции позволяют повторную идентификацию, а параметры могут сохранить
редкий фрагмент. Проверка включает canary exposure, membership-attack audit и
поиск дословных редких последовательностей, но эмпирическая атака не заменяет
формальную гарантию.

![[Assets/Sources/Harvard ML Systems/vol2/security_privacy/security_privacy_output_leakage_ladder.svg]]

*Увеличение подробности ответа повышает полезность API и одновременно объём
утекающей информации. Оригинальная иллюстрация из
[Security & Privacy](https://mlsysbook.ai/vol2/security_privacy/security_privacy.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/security_privacy|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

### Poisoning, backdoor и adversarial input

При poisoning противник влияет на обучение. Availability poisoning ухудшает
общее качество; targeted poisoning меняет отдельное поведение; backdoor
сохраняет нормальное качество на clean set, но активируется выбранным trigger.
Контрмеры начинаются до алгоритма: доверенные источники, manifests, lineage,
разделение полномочий на загрузку и выпуск данных, повторная разметка и анализ
активаций.

Adversarial input действует после обучения и ищет малое изменение \( \delta \),
которое меняет решение:

$$
\max_{\|\delta\|\le \varepsilon}
\mathcal L(f_\theta(x+\delta),y).
$$

Проверка зависит от norm, бюджета, знания модели и цели атаки. «Устойчиво к
FGSM» не означает устойчивость к PGD, adaptive attack или реальному
семантическому изменению. Защиты и их ограничения подробно разобраны в
[[02 Areas/ML & DL/01 Справочник/Security и Robustness/02 Robustness|Robustness]].

Для LLM добавляются prompt injection, утечка system prompt, poisoned retrieval,
небезопасный tool call и exfiltration через содержимое внешнего документа.
Текст из retrieval и браузера остаётся недоверенным вводом, даже если визуально
выглядит как инструкция. Авторизация проверяется кодом у каждого действия;
модель не должна самостоятельно расширять свои полномочия.

## Defense in depth

![[Assets/Sources/Harvard ML Systems/vol2/security_privacy/defense-stack.svg]]

*Многоуровневая защита покрывает данные, модель, deployment, runtime и
аппаратную основу. Оригинальная иллюстрация из
[Security & Privacy](https://mlsysbook.ai/vol2/security_privacy/security_privacy.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/security_privacy|локальная полная глава]].
Лицензия CC BY-NC-SA 4.0.*

Три правила удерживают архитектуру от «одной волшебной защиты».

1. **Defense in depth:** независимые контроли должны не позволять одному
   компрометированному слою открыть всю систему.
2. **Минимальная поверхность:** не публиковать ненужные endpoints и logits,
   отделять training от serving, шифровать артефакты, сокращать retention.
3. **Fail-safe defaults:** при сомнении отклонить подозрительный ввод,
   остановить training на деградации данных, отозвать credential и изолировать
   компонент.

Prevention без detection оставляет команду слепой к обходу. Detection без
containment лишь документирует ущерб. Для каждого контроля заранее нужны
владелец alert, playbook, допустимая задержка реакции и доказательство того, что
изоляция действительно ограничивает blast radius.

## Дифференциальная приватность

Механизм \(M\) удовлетворяет \((\varepsilon,\delta)\)-дифференциальной
приватности, если для любых соседних наборов \(D,D'\), отличающихся записью
одного человека, и любого множества результатов \(S\):

$$
\Pr[M(D)\in S]
\le e^\varepsilon\Pr[M(D')\in S]+\delta.
$$

Гарантия ограничивает, насколько присутствие одного человека способно изменить
распределение результата, и сохраняется при произвольной последующей обработке
уже приватного результата. Меньшее \(\varepsilon\) означает более сильное
ограничение; \(\delta\) допускает малую вероятность нарушения чистой
\(\varepsilon\)-гарантии.

Для запроса с чувствительностью

$$
\Delta f=\max_{D\sim D'}\|f(D)-f(D')\|_1
$$

Laplace mechanism добавляет шум масштаба \(b=\Delta f/\varepsilon\).
Критическая часть — ограничить чувствительность. В DP-SGD индивидуальные
градиенты обрезают до нормы \(C\), агрегируют и добавляют Gaussian noise:

$$
\tilde g =
\frac1B\left(
\sum_{i=1}^{B}
g_i\min\left(1,\frac C{\|g_i\|_2}\right)
+\mathcal N(0,\sigma^2 C^2 I)
\right).
$$

Clipping ограничивает вклад записи, шум скрывает оставшееся различие.
Приватность зависит от sampling rate, числа шагов, noise multiplier и
accountant; одно значение \(\sigma\) без этих параметров ничего не доказывает.

![[Assets/Sources/Harvard ML Systems/vol2/security_privacy/71393fab82f7333b27d3df80508ecbcfeca4b822.svg]]

*DP-SGD: per-example clipping предшествует агрегации и добавлению шума.
Оригинальная иллюстрация из
[Security & Privacy](https://mlsysbook.ai/vol2/security_privacy/security_privacy.html);
[[05 Источники/Courses/Harvard ML Systems/vol2/security_privacy|локальный оригинал]].
Лицензия CC BY-NC-SA 4.0.*

Приватность расходуется при повторных обращениях к тем же людям. При простой
композиции механизмы \((\varepsilon_i,\delta_i)\) дают не более
\((\sum_i\varepsilon_i,\sum_i\delta_i)\). Advanced composition, RDP и privacy
accountants дают более точные границы, но не отменяют глобальный ledger.
Обучение десяти моделей нельзя отчитать как десять независимых гарантий
\(\varepsilon=1\) для одного датасета.

DP не заменяет шифрование, authentication или secure deletion. Она защищает
влияние записи на опубликованный результат, но не спасает исходный датасет на
скомпрометированном сервере. Federated learning также не является автоматической
приватностью: gradients могут раскрывать данные, поэтому используют secure
aggregation, clipping, DP и ограничение telemetry совместно.

## Компромиссы privacy–utility

Шум ухудшает полезность особенно на малых наборах, редких группах и
тонкозернистых задачах. Это может породить новый fairness gap. DP-SGD требует
per-example gradients и часто больших batch, поэтому растут память и
вычисления. Значение \(\varepsilon\) нельзя выбирать как «принятое в индустрии»:
нужны threat model, sweep качества и явное решение о допустимой утечке.

| Если основной риск… | Первая мера |
|---|---|
| кража данных из storage | encryption, KMS, ACL, audit |
| inference о присутствии записи | DP training и ограничение outputs |
| утечка gradients между участниками | secure aggregation + DP |
| повторная идентификация выгрузки | минимизация, aggregation, DP; одной деидентификации мало |
| удаление по запросу | lineage, retention, deletion/unlearning workflow |

## Минимальный security review

- Зафиксированы assets, trust boundaries, adversary capability и attack timing.
- Каждый dataset и artifact имеет provenance, hash, владельца и историю доступа.
- Training, registry и serving разделены; secrets не попадают в логи и prompts.
- API ограничивает лишнюю информацию и обнаруживает автоматизированное
  исследование границы.
- Retrieval и tool outputs считаются недоверенными; действия проходят
  детерминированную авторизацию.
- Privacy claims подкреплены accountant и полным набором параметров.
- Проверены membership/data extraction, poisoning/backdoor и adaptive attacks,
  соответствующие threat model.
- Alert связан с containment, recovery и incident review.

## Связанные страницы

- [[02 Areas/ML & DL/01 Справочник/Security и Robustness/02 Robustness|Robustness]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Responsible systems|Ответственные ML-системы]]
