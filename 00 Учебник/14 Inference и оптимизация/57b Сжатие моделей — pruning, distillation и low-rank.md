---
title: Сжатие моделей — pruning, distillation и low-rank
type: textbook-chapter
status: canonical
last_verified: 2026-08-06
source_language: mixed
---

# Сжатие моделей: pruning, distillation и low-rank

Квантизация меняет способ представления чисел, но оставляет архитектуру и число
параметров прежними. Это только одна ось сжатия. Можно удалить отдельные веса
или целые структурные блоки, заменить большую матрицу произведением двух
маленьких, обучить компактную модель повторять поведение большой либо найти
другую архитектуру под тот же аппаратный бюджет. Эти методы уменьшают разные
виды стоимости и потому не взаимозаменяемы.

Главный вопрос этой главы — не «на сколько процентов модель стала меньше», а
какой физический ресурс перестал быть узким местом после преобразования.
Нулевой вес в плотной матрице всё ещё может участвовать в обычном GEMM;
low-rank-разложение может добавить лишний запуск kernel; студент может хорошо
повторять logits учителя, но потерять редкое поведение. Сжатие считается
системной оптимизацией только после измерения качества, памяти, фактических
операций и задержки на целевой платформе.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/5b8ca5d44a1556b84569cdc9156a94c4b1500e92.svg]]

*Карта методов сжатия и аппаратных эффектов. Сначала выберите ограниченный
ресурс, затем проследите, какое представление и какой kernel должны измениться.
Источник: Vijay Janapa Reddi et al., [Harvard ML Systems Book — Model
Compression](https://mlsysbook.ai/vol1/model_compression/model_compression.html),
CC BY-NC-SA 4.0; локальный оригинальный SVG из опубликованного курса.*

## Четыре разных преобразования

| Метод | Что меняется | Возможная экономия | Что требуется для ускорения |
|---|---|---|---|
| pruning | число ненулевых весов, каналов, голов или слоёв | параметры, FLOP, иногда activations | sparse kernel либо структурно меньшая плотная модель |
| low-rank / tensor factorization | одна большая операция заменяется несколькими меньшими | параметры и умножения | достаточно низкий rank и выгодные shapes |
| distillation | обучается другая, обычно меньшая модель | вся стоимость student | хороший teacher signal и покрывающие данные |
| architecture optimization | меняется topology под бюджет | compute, memory, latency, energy | поиск и измерение на целевом устройстве |

Эти методы можно комбинировать с квантизацией, однако их эффекты нельзя просто
перемножать. После pruning модель может стать compute-bound; после distillation
изменятся подходящий batch и kernel shapes; низкоранговая факторизация может
ухудшить fusion. Каждая стадия требует нового профиля.

## Pruning: параметр равен нулю — но исчезла ли работа?

Для весов $W$ pruning ищет маску $M\in\{0,1\}^{m\times n}$ и использует
$\widehat W=M\odot W$. В идеализированной записи задача выглядит как

$$
\min_{\widehat W}\mathcal L(\widehat W)
\quad\text{subject to}\quad
\|\widehat W\|_0\le k,
$$

где $\|\widehat W\|_0$ — число ненулевых элементов. Точный комбинаторный поиск
слишком дорог, поэтому применяют magnitude pruning, оценки чувствительности,
итеративное удаление с дообучением и структурные ограничения.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/fc4813524b57d9439c3a0b9101f2c1f2a79382c3.svg]]

*Magnitude pruning превращает часть элементов плотной матрицы в нули. Рисунок
показывает статистическую операцию, но не обещает ускорения: справа всё ещё
нужен формат хранения и kernel, умеющий пропускать нули. Источник: [Harvard ML
Systems Book — Model Compression, Figure 2](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-sparse-matrix),
CC BY-NC-SA 4.0.*

### Неструктурированное и структурное разреживание

Неструктурированный метод удаляет отдельные элементы. Он способен дать высокую
долю нулей при малой потере качества, но создаёт нерегулярные индексы и доступ к
памяти. На обычном dense GEMM нули не экономят вычисления. Даже sparse kernel
может проигрывать из-за metadata и плохой загрузки аппаратных блоков.

Структурный pruning удаляет каналы, neurons, attention heads, experts или слои.
Полученные матрицы остаются плотными, но уменьшаются их формы. Поэтому такой
метод чаще даёт реальное ускорение без специальной sparse-библиотеки, хотя
крупная единица удаления сильнее влияет на качество.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/4000264baf2311c2b61ca0ddd3fcedaa9d047ed3.svg]]

*Удаление канала меняет размер следующего слоя, удаление слоя — сам маршрут
вычислений. Источник: [Harvard ML Systems Book — Channel vs. Layer Pruning,
Figure 3](https://mlsysbook.ai/vol1/model_compression/model_compression.html#fig-channel-layer-pruning),
CC BY-NC-SA 4.0.*

Практический протокол состоит из четырёх измерений: sparsity, качество,
фактические shapes/число операций и end-to-end latency. Отчёт «90% weights are
zero» без последних двух измерений ничего не говорит о serving.

## Low-rank factorization: заменить одну матрицу двумя

Для $W\in\mathbb R^{m\times n}$ сингулярное разложение даёт

$$
W=U\Sigma V^\top.
$$

Оставив первые $r$ компонент, получают

$$
W\approx U_r\Sigma_rV_r^\top=AB,
$$

где $A\in\mathbb R^{m\times r}$ и $B\in\mathbb R^{r\times n}$. Число
параметров меняется с $mn$ на $r(m+n)$. Экономия существует, когда
$r(m+n)<mn$; для скорости дополнительно нужно, чтобы два новых GEMM вместе с
записью промежуточной activation были дешевле исходного.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/dcbd6535b09a28528559354f928838baaae7644b.svg]]

*Одна большая матрица заменяется двумя факторами меньшего ранга. Сравнивайте не
только число прямоугольников, а формы обеих операций и промежуточный тензор.
Источник: [Harvard ML Systems Book — Low-Rank Factorization](https://mlsysbook.ai/vol1/model_compression/model_compression.html#sec-model-compression-low-rank-factorization),
CC BY-NC-SA 4.0.*

LoRA использует похожую параметризацию как обучаемую поправку к замороженной
матрице, но это не автоматически сжатая inference-модель. Для deployment
поправку можно слить в $W$ — тогда rank-структура исчезает — либо исполнять
отдельно, платя за дополнительные kernels. Low-rank compression, наоборот,
заменяет саму матрицу и требует проверки approximation error и качества после
fine-tuning.

## Knowledge distillation: передать функцию, а не веса

Teacher вычисляет распределение $p_T(y\mid x)$, student — $p_S(y\mid x)$.
Кроме обычной ошибки по меткам student минимизирует расхождение с мягким
распределением teacher:

$$
\mathcal L=
\alpha\,\mathcal L_{\text{labels}}+
(1-\alpha)\tau^2
D_{KL}\!\left(
\operatorname{softmax}(z_T/\tau),|,
\operatorname{softmax}(z_S/\tau)
\right).
$$

Температура $\tau$ раскрывает относительные вероятности неверных классов: они
несут информацию о сходстве, которой нет в one-hot target. Для языковой модели
можно дистиллировать token logits, скрытые состояния, attention maps,
сгенерированные ответы или целые reasoning traces. Эти сигналы отвечают на
разные вопросы и могут наследовать ошибки teacher.

![[Assets/Sources/Harvard ML Systems/vol1/model_compression/708770dcb60d00ea7f382f39d15b3206ba6ccb1f.svg]]

*Teacher создаёт дополнительный обучающий сигнал для меньшего student.
Источник: [Harvard ML Systems Book — Knowledge Distillation](https://mlsysbook.ai/vol1/model_compression/model_compression.html#sec-model-compression-knowledge-distillation),
CC BY-NC-SA 4.0.*

Главный риск — покрытие. Student хорошо имитирует teacher на распределении
distillation data, но это не гарантирует сохранения редких навыков, safety
границ или calibration за его пределами. Поэтому сравнивают student не только
с teacher на сгенерированных ответах, но и с исходным evaluation contract,
включая worst-case subsets.

## Architecture optimization и graph optimization — разные уровни

Architecture search или ручной redesign меняет число слоёв, ширину, тип блока,
attention pattern и другие параметры модели. Graph/kernel optimization не
меняет математическую функцию: складывает операции, выбирает layout, удаляет
лишние преобразования и планирует память. Первое способно сократить сам объём
работы, второе — уменьшить накладные расходы её исполнения.

Смешивать эти уровни опасно. Operator fusion может ускорить исходную модель без
потери качества, но не уменьшает число её параметров. Pruning может уменьшить
параметры, но без переписывания graph и kernel не изменить latency.

## Как выбирать метод

1. Зафиксировать ограничение: storage, RAM/VRAM, bandwidth, latency, energy или
   стоимость обучения.
2. Снять baseline качества и профиля на целевой платформе.
3. Выбрать преобразование, которое меняет ограничивающий ресурс физически.
4. Проверить корректность и качество сразу после преобразования.
5. Перекомпилировать/перепрофилировать graph и измерить end-to-end workload.
6. Сравнить Pareto-frontier, а не один compression ratio.

Минимальная итоговая таблица содержит параметры, ненулевые параметры, FLOP,
bytes, peak memory, latency distribution, throughput, energy и метрики качества.
Отдельно указываются формат хранения, kernels и hardware: без них результат
сжатия невоспроизводим.

## Практика и первоисточники

- [[05 Источники/Courses/Harvard ML Systems/vol1/model_compression|Harvard ML Systems — полная оригинальная глава Model Compression]];
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/16_compression|TinyTorch Module 16]] — исходные упражнения по magnitude/structured pruning, distillation и SVD;
- [Han et al., Deep Compression](https://arxiv.org/abs/1510.00149);
- [Hinton, Vinyals, Dean — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531);
- [Blalock et al. — What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html).

← [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload|KV-cache compression и offload]] ·
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Спекулятивное декодирование]] →
