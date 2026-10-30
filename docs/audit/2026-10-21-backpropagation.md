---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md
route: /textbook/neural-networks/backpropagation/
before_sha256: cfc70cfd57de9e045b459bf82d0ae9c3da595c26c71b2de3e5b5afc967294f6e
after_sha256: d329325d295f0fc50ddbf013412c96e0f24f8a31bd6733754e08ae23a5baa198
---

# Backpropagation: приёмка объяснения

Задача — последовательное ревью оснований для Transformer. Предыдущая
проверенная глава — функции активации. Этот отчёт закрывает только текущую
главу backpropagation, а не весь маршрут и не перенос всех перечисленных курсов.
EN, raw/, Git index, коммиты и публикация в этом проходе не изменялись.

Исходная глава прочитана полностью; сохранена в
[snapshots/2026-10-21-backpropagation.md](snapshots/2026-10-21-backpropagation.md).
Итоговая глава прочитана последовательно целиком; после этого отдельно
перечитаны изменённые формулировки, заголовки и матричная формула.
Редакторская проверка no-ai-slop проведена отдельно от проверки объяснений.

## Изменения и источники

| Механизм | Прочитанный источник | Проверяемый разбор в главе |
| --- | --- | --- |
| Локальная производная и входящая производная | CS231n optimization-2, Simple expressions, Compound expressions, Intuitive understanding | (−2+5)·(−4)=−12; производные (−4,−4,3), а не локальные единицы сложения |
| Общие ветви и порядок | Colah, Computational Graphs, Derivatives, Factoring Paths, Computational Victories; micrograd engine.py полностью | (a+b)(b+1): (2,1) → 6 и (2,5); x*x → 6 при x=3; обратный порядок ждёт обе ветви |
| Умножение, сложение, max | CS231n Patterns in backward flow, включая исходный SVG | 2(xy+max(z,w)): −20 и (−8,6,2,0); отдельно оговорена недифференцируемость равенства |
| Нейрон и группировка операций | CS231n Modularity: Sigmoid example полностью, схема и код; предыдущая глава активаций | tanh: z=−0.5, p=−0.462117, L=1.068893, dz=−1.149879; исходный sigmoid-нейрон явно другой |
| Тензорный слой | CS231n Gradients for vectorized operations полностью | Вывод через индексы XᵀG, GWᵀ и суммы broadcasting; конкретные X/W/b/G, градиенты и формы |
| Реализация | micrograd engine.py, исходные __mul__ и backward | Методы перенесены без перевода/переписывания; checker сравнивает точный текст и исполняет полный оригинальный движок |
| Обновление и накопление | PyTorch backward, Autograd mechanics: history, saved tensors и grad modes | Один SGD-шаг: 1.068893 → 0.329935; backward не меняет вес; накопление 2+3=5, очистка → 3 |
| Обратный режим | Colah Factoring Paths/Computational Victories; контракт PyTorch backward | u=2, v=(u²,3u), seed=(1,2) → 10; VJP не приравнивается к полному Якобиану |
| Ошибки и проверка | PyTorch Autograd mechanics: saved tensors, division by zero, modes, evaluation, in-place/correctness checks; установленная документация gradcheck | detach меняет 4 на 2; замороженный множитель не обрывает вход; NaN после маски и безопасная индексация до деления; in-place выдаёт ошибку; центральная разность и ReLU в нуле |

CS231n optimization-2 прочитан целиком, включая все три SVG, код поэтапного
вычисления, ветвление и раздел о матрицах. В нашей главе используется строковое
соглашение XW; исходник курса показывает WX. Различие с хранением nn.Linear
объяснено, а формы не перенесены механически.

Источник PyTorch 2.14 сохранён как HTML. Сначала была неудачная попытка
получить .rst из Git-тега (404); этот файл не используется как доказательство.
Чтение HTML ограничено указанными полными разделами; complex autograd,
многопоточность и hooks не объявляются разобранными.
Gradcheck проверен по документации установленной torch 2.14.1 и исполнен.
TinyTorch, CS224N, ML Visualized и cheatsheet оставлены дополнительными
ссылками; они не объявляются целиком прочитанными или перенесёнными.

## Исходные рисунки и код

Архив: `05 Источники/Originals/backpropagation-2026-10-21/`.
Никаких новых нарисованных схем нет.

- CS231n: commit `8ec57ccfc32379ae4b37c45de949af359673f92f`,
  optimization-2.md SHA `f11216ba5f9611b039d11d614050c7ad7736df6b9f47ec83e5cfba2f336e366a`.
  LICENSE MIT SHA `85b365f07b4cfc5d678c548bc41965bc3bc782106b8903bd3b876d9d49a4270d`.
  Три встроенных SVG извлечены с нормализацией только xmlns и viewBox;
  элементы рисунка, координаты, подписи и цвета не менялись.
  `prepare_backprop_source_figures.py --check` и registry-тест
  восстанавливают результат из оригинального Markdown и сравнивают текст.
  Положительная проверка и отрицательные мутации provenance/commit/license/hash проходят.
- micrograd: commit `7bc720e951fe422b8f8814aa5aa1b64121d26b4c`.
  engine.py SHA `95f19fa3734458d868e067d770f0773027763a2f654900bfbc1df8c5413acf84`,
  LICENSE SHA `f372f582c99dc5b0e8dcf5ca5542183a735365468857ec6416af628490187cc1`.
  Новый download gout.svg побайтно совпал с существующим curated SVG:
  SHA `c5c5461e9735cd42974fd6b5091eba227e8a25692b59b72ed815661949e189b0`.
  Подпись уточняет, что это ReLU-граф, значения округлены, а seed относится к выходу.
- MIT notices опубликованы локально в `site/public/licenses/cs231n-MIT.txt`
  и `site/public/licenses/micrograd-MIT.txt`, ссылки стоят рядом с материалами.
- HTML PyTorch SHA `53747ac19082c84716209b711b8e403b71280c1868790532b3568753ed76ff02`;
  Colah SHA `776bfab815b8b41d0e91eb8b5161407eccd66fe9e7fa014855a63e10cc6f3a7b`.
  Из этих HTML новые изображения не воспроизводились.

## Исполненные проверки

`publishing/tools/check_backprop_examples.py`: пять самостоятельных
опубликованных Python-блоков исполнены на torch 2.14.1 CPU.
Шестой блок — явно обозначенный фрагмент класса micrograd; оба метода
сравнены с оригиналом, а числовые случаи исполнены полным исходным движком.

Дополнительно проверены общий вес (градиент 40), residual (27 и компенсация
до нуля), broadcasting по B/T, tanh перед матричным backward, неравные
микробатчи, VJP/Якобиан, detach/freeze/no_grad/eval, безопасное и небезопасное
деление, version-check in-place, несколько eps, gradcheck, ReLU в нуле и clipping.
Результат PASS: `/tmp/bookvar-backprop-examples-2026-10-21.log`.

Обязательные проверки:

- 283 unit-теста / 18 файлов: PASS. Использованы
  `--maxWorkers=1 --testTimeout=10000`, без пропуска assertions.
- Последняя сборка: 1961 страница, 23.65 s, exit 0.
- На последней сборке: 45 output-тестов PASS; 0 broken links.
- Source coverage: 210 строк проходят структурную проверку; это не оценка полноты объяснения.
- `git diff --check`: PASS.

Один ранний запуск output/link был сделан до завершения сборки и не принят:
дерево dist ещё не содержало всех страниц. Оба check повторены после exit 0,
а после исправления мобильной формулы и заголовков — на последней сборке.
Итоговые логи:
`/tmp/bookvar-backprop-{unit,build,output,links,coverage}-2026-10-21.log`.

## Локальная визуальная проверка

На `http://127.0.0.1:4330/textbook/neural-networks/backpropagation/`:

- Desktop 1365×1000 и mobile 390×844: все четыре исходных рисунка
  осмотрены рядом с объяснениями. Подписи и числа соответствуют оригиналам.
- Длинные sigmoid/micrograd-графы мелкие при показе целиком, как и оригиналы.
  Просмотрщик проверен с панорамированием: sigmoid при 2.5×,
  micrograd при 4× читаются. Это не утверждение, что их мелкие подписи
  читаемы в уменьшенном inline-виде на телефоне.
- 23 display-формулы, 0 KaTeX errors. У тройной формулы матричного слоя
  обнаружено переполнение 376/343; она разложена по строкам.
  После новой сборки на мобильной ширине wide=0.
- Две таблицы: desktop 687/687, mobile 343/343.
  Документ: desktop 1350/1350, mobile 375/375.
- Duplicate IDs=0; изменённые заголовки сохраняют прежние якоря
  отдельными span. Старый якорь кратких итогов также сохранён.

Снимки локальной проверки:
`/tmp/bookvar-backprop-desktop-2026-10-21.jpg`,
`/tmp/bookvar-backprop-mobile-math-2026-10-21.jpg`,
`/tmp/bookvar-backprop-mobile-zoom-2026-10-21.jpg`.

Отдельных незакрытых замечаний к объяснению этой ревизии не оставлено.
Общая очередь учебника и прежний неподтверждённый pixel-match
CS231n lecture_2.pdf не закрываются этим отчётом.
Следующая глава — оптимизация, затем инициализация/нормализация/residual.
