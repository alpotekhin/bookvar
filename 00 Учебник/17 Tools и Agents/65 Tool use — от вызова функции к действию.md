---
title: Как вызов функции становится действием
type: textbook-chapter
status: canonical
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля]]"
next: "[[00 Учебник/17 Tools и Agents/66 Agent harness и context engineering]]"
reference_terms:
  - label: Вызов инструмента — схема, права и проверка результата
    label_en: Tool calls — schema, permissions, and result verification
    anchor: tool-call-contract
  - label: Toolformer — отбор вызовов и оценивание
    label_en: Toolformer — call selection and evaluation
    anchor: toolformer-curation-evaluation
  - label: MCP — клиент, сервер и границы протокола
    label_en: MCP — client, server, and protocol boundaries
    anchor: mcp-host-client-server
  - label: MCP — сопоставление ответов и два вида ошибок
    label_en: MCP — response correlation and two kinds of error
    anchor: mcp-response-correlation
  - label: ReAct — prompting, запасная стратегия и обучение
    label_en: ReAct — prompting, fallback, and fine-tuning
    anchor: react-prompting-learning
  - label: Идемпотентность — повтор после потерянного ответа
    label_en: Idempotency — retries after a lost response
    anchor: tool-retry-idempotency
  - label: Вызов инструмента — проверяемое завершение эпизода
    label_en: Tool use — verifying episode completion
    anchor: tool-use-verification
primary_sources:
  - https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure
  - https://react-lm.github.io/
  - https://arxiv.org/abs/2302.04761
  - https://www.anthropic.com/engineering/writing-tools-for-agents
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
  - https://modelcontextprotocol.io/specification/2025-11-25/basic
  - https://modelcontextprotocol.io/specification/2025-11-25/architecture
  - https://modelcontextprotocol.io/specification/2025-11-25/server/tools
  - https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/
source_unit_id:
  - meeting-01-intro-reasoning-agent-progress
  - meeting-03-slides-yu-su-agent-first-vs-llm-first
---

# Как вызов функции становится действием

Пользователь просит прочитать первые две строки `notes.txt` и назвать
значение `timeout`. Модель предлагает `read_file` с начальной строкой 0.
Инструмент нумерует строки с единицы, поэтому приложение отклоняет вызов,
даже не открыв файл. Модель получает ошибку и исправляет аргумент. Только
после удачного чтения ответ «30» можно подтвердить по содержимому файла.

Имя функции и аргументы предлагает модель. Проверяет и исполняет их
управляющая программа, или *harness*. Результат чтения, отказ в доступе
и падение исполнителя должны возвращаться как разные наблюдения: от них
зависит, что можно сделать дальше.

<span id="2-интерфейс-инструмента-является-частью-задачи"></span>

## Что приложение проверяет перед исполнением

<a id="tool-call-contract"></a>

### Один запрос через схему, ошибку и проверку

В учебном файле две строки: `timeout=30` и `retries=2`. Схема принимает
путь и включительный диапазон строк:

<details id="tool-use-code-schema">
<summary>Схема read_file</summary>

```json
{
  "name": "read_file",
  "description": "Read UTF-8 lines from an allowed workspace file; line numbers are 1-based and inclusive.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "minLength": 1},
      "start_line": {"type": "integer", "minimum": 1},
      "end_line": {"type": "integer", "minimum": 1}
    },
    "required": ["path", "start_line", "end_line"],
    "additionalProperties": false
  }
}
```

</details>

Первое предложение модели использует нумерацию с нуля:

<details id="tool-use-code-proposal">
<summary>Предложение модели: строки 0–2</summary>

~~~json
{
  "path": "notes.txt",
  "start_line": 0,
  "end_line": 2
}
~~~

</details>

Валидатор возвращает наблюдение:

<details id="tool-use-code-rejected">
<summary>Наблюдение после отклонения c1</summary>

~~~json
{
  "call_id": "c1",
  "ok": false,
  "error": {
    "type": "schema_error",
    "field": "start_line",
    "minimum": 1
  },
  "executed": false
}
~~~

</details>

Чтения ещё не было. Наблюдение сообщает, какое поле нужно исправить:
следующее предложение использует `start_line=1` при прежних пути и конце
диапазона.

`start_line=2, end_line=1` тоже прошёл бы эту схему: оба числа не меньше
единицы. Управляющая программа отдельно проверяет порядок границ,
размер диапазона, фактически открываемый файл и права на чтение. Исправление
одного поля не отменяет ни одну из остальных проверок. Для допустимого
диапазона и разрешённого файла наблюдение выглядит так:

<details id="tool-use-code-observation">
<summary>Наблюдение после чтения c2</summary>

~~~json
{
  "call_id": "c2",
  "ok": true,
  "executed": true,
  "path": "notes.txt",
  "start_line": 1,
  "end_line": 2,
  "lines": ["timeout=30", "retries=2"],
  "truncated": false,
  "revision": "fixture-v1"
}
~~~

</details>

Ошибка `permission_denied` требовала бы остановки, а не поиска обходного пути.

Тест заранее знает содержимое версии `fixture-v1`. Он проверяет номер
версии, диапазон и признак усечения, затем сравнивает ответ с числом после
`timeout=`. Требование прочитать обе строки не выполняется, если модель
угадала «30» без чтения. В конце главы этот же эпизод разобран
[[#tool-use-verification|в исполняемом примере]].

В реальной файловой системе проверка одного строкового префикса пути
не гарантирует доступ к нужному объекту. Символические ссылки и изменения
файловой системы между проверкой и открытием требуют защиты на стороне
исполнителя. Учебный пример ниже работает со словарём, а не с файловой
системой, поэтому не проверяет такую защиту.

### Что стоит указать в описании инструмента

В описании `read_file` указаны единицы нумерации и то, что обе границы
входят в диапазон. Схема задаёт типы и допустимые значения отдельных полей;
исполнитель проверяет ограничения, которые зависят от нескольких полей или
состояния среды. Для поискового инструмента в описании обычно нужны смысл
`top_k`, допустимые фильтры и правило перехода к следующей странице. Если
поиск видит только названия и аннотации, описание не должно обещать доступ
к полному тексту.

У Anthropic в [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
описание и результат проверяют на задачах с несколькими инструментами:
может ли модель выбрать нужный по имени и назначению, передать аргументы,
а затем взять из ответа значение для следующего вызова? Например,
аргумент `mode` без перечисления значений вынуждает угадывать их даже
при верно выбранном инструменте.

Два инструмента `search` и `lookup` полезно различать по тому, что они
принимают и возвращают: первый ищет совпадения по запросу, второй требует
известный идентификатор и возвращает одну запись. Ограничение объёма
оставляет место для следующего шага. Причина ошибки решает, каким будет
этот шаг: при `not_found` можно уточнить идентификатор, при
`permission_denied` нельзя продолжать без подходящих прав.

<span id="1-что-добавляет-инструмент"></span>

## Как наблюдение попадает в следующий запрос

Обычная языковая модель отображает контекст $c$ в продолжение $y$:

$$y \sim p_\theta(y\mid c).$$

При работе с инструментами часть продолжений интерпретируется не как ответ
пользователю, а как структурированное действие

$$a_t=(\text{name},\text{arguments}).$$

Управляющая программа проверяет действие. Если оно допущено, среда
возвращает наблюдение $o_t$. Если проверка не прошла, приложение может само
сформировать наблюдение об ошибке, не обращаясь к среде. История
$(c,a_1,o_1,\ldots,a_t,o_t)$ позволяет собрать очередной запрос к модели;
передавать всю историю на каждом шаге необязательно.

В [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure)
этот цикл называется Think → Act → Observe. Например, ответ на `c1` объясняет
модели, почему первый вызов не исполнен и какой аргумент нужно изменить.
Наблюдение от `c2` добавляет то, чего не было в предложении модели:
строки, прочитанные исполнителем. В [ReAct](https://react-lm.github.io/)
разница между допущением и наблюдением видна на вопросе об Apple Remote.

<a id="react-hotpotqa-trajectories"></a>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tool-use-2026-10/react-hotpotqa-primary.png|Четыре траектории для вопроса об Apple Remote: обычный запрос, цепочка рассуждений, действия без записанных рассуждений и ReAct.]]

*Shunyu Yao et al., [ReAct v3, Figure 1(1), с. 2](https://arxiv.org/pdf/2210.03629v3#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Первая панель исходного рисунка, с английскими подписями.
[Сводная схема трёх циклов на сайте проекта](https://react-lm.github.io/files/diagram.png).*

В столбце Reason-only модель строит рассуждение без нового внешнего
наблюдения. В Act-only видны действия и ответы среды; промежуточный
текст рассуждения не записан. В панели ReAct модель пишет `Thought`
и `Act`, а среда возвращает `Obs`.
Отсутствие видимого текста между вызовами не доказывает, что у современной
модели нет внутреннего рассуждения; схема сравнивает форматы prompting
исходной работы, а не позволяет наблюдать все вычисления сети.

<span id="4-наблюдение-должно-отвечать-на-следующий-вопрос"></span>

## Что нужно вернуть модели после вызова

В ответе на `c2` есть путь, диапазон и строки. Для поиска нужны другие
данные: заголовок, отрывок, источник и курсор. После создания записи
вместо строк возвращают идентификатор объекта и подтверждённые поля,
после запуска команды — код завершения, вывод и признак усечения.
Состав ответа зависит от того, что системе предстоит сделать с ним.

Вопрос в [траекториях HotpotQA выше](#react-hotpotqa-trajectories):
каким устройством, кроме Apple Remote,
можно управлять программой, для которой этот пульт первоначально предназначался?
Он требует связать два факта: название программы и другое средство управления.

1. `Search[Apple Remote]` возвращает описание пульта и название Front Row.
   Название ещё не отвечает на исходный вопрос, но определяет следующий поиск.
2. `Search[Front Row]` не находит точную страницу. Среда возвращает варианты,
   среди которых есть `Front Row (software)`. Это наблюдение позволяет
   исправить запрос, а не повторять его без изменений.
3. `Search[Front Row (software)]` возвращает сведения о программе.
   В них указаны Apple Remote и функциональные клавиши клавиатуры.
4. `Finish[keyboard function keys]` использует найденное второе средство
   управления.

Act-only на том же рисунке получает похожие результаты поиска, но завершает
словом `yes`. Наличие нужного текста в истории ещё не гарантирует правильного
ответа. В ReAct текстовые шаги между действиями связывают название программы,
уточнение поискового запроса и требуемое устройство. Это исторический пример
из статьи; текущий ответ поисковой системы здесь не воспроизводится.

### Поиск объекта в частично наблюдаемой среде

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tool-use-2026-10/react-alfworld-primary.png|В ALFWorld модель без промежуточного плана повторяет действие у раковины. ReAct находит перечницу на столешнице и кладёт её в ящик.]]

*Shunyu Yao et al., [ReAct v3, Figure 1(2), PDF p. 2](https://arxiv.org/pdf/2210.03629v3#page=2), CC BY 4.0. Нижняя часть исходного рисунка: поиск и перемещение перечницы в ALFWorld. Английские подписи сохранены.*

В ALFWorld агенту нужно найти перечницу и поместить её в ящик. При переходе
к раковине наблюдение перечисляет губку, лопатку и ложку. Перечницы в списке
нет. В траектории Act-only команда «взять перечницу из раковины» получает ответ
`Nothing happens`; повтор той же команды снова ничего не меняет.

В траектории ReAct агент осматривает несколько мест и находит перечницу на
`countertop 3`. После команды взять её среда подтверждает новое состояние:
предмет у агента. Теперь можно идти к ящику. Наблюдение сообщает, что ящик закрыт,
поэтому перед помещением предмета нужен отдельный шаг открытия. Финальное
действие получает подтверждение, что перечница помещена в ящик.

На рисунке часть однотипных поисковых шагов пропущена; это не полный журнал.
Причина успеха показанного эпизода прослеживается по наблюдениям: место предмета
сначала неизвестно, затем найдено, предмет взят, ящик открыт и действие
подтверждено. Повтор неудачной команды без нового наблюдения не приближает
агента к этому состоянию.

<a id="react-prompting-learning"></a>

### ReAct: prompting, fallback and learned trajectories

A thought in the original ReAct formulation changes the text context
available for the next decision. It does not move an object, run a search
or supply an observation. This distinction is visible in the two examples
above: naming Front Row prepares a search, and planning to open a drawer
does not open it. The environment supplies the evidence after the action.

The original prompts do not impose the same cadence on every task.
For HotpotQA and FEVER, demonstrations contain a thought before each
action. For ALFWorld and WebShop, thoughts occur at selected points:
choosing a likely location, recognizing a completed subgoal, or revising
the plan. Several actions can follow without another written thought.
A rigid parser that requires one thought for every action would exclude
some of these original decision-making trajectories.

The HotpotQA prompt contains six annotated demonstrations; FEVER uses
three. In ALFWorld, the authors annotate three demonstrations per task
type and form six ordered two-example prompts. The model in the main
prompting experiments is frozen PaLM-540B. These demonstrations provide
examples of a policy; they are not gradient updates.

#### Switching between retrieval and self-consistency

ReAct can fail because search returns nothing useful or because its
trajectory repeats a previous action. CoT can construct a useful reasoning
sequence while relying on an incorrect remembered fact. Section 3.2 of
[ReAct](https://arxiv.org/abs/2210.03629v3) tests two ways to switch between
the methods.

In ReAct → CoT-SC, the controller first runs the environment-interaction
trajectory. If it has not returned an answer within the chosen budget,
it falls back to sampled CoT answers and voting. The original budgets are
seven steps for HotpotQA and five for FEVER. Exhausting the budget is a
controller event, not a model-generated proof that retrieval is impossible.

In CoT-SC → ReAct, the controller samples several CoT answers first.
If the most frequent answer appears fewer than half as many times as
the number of samples, it invokes ReAct. With five samples and counts
2, 2 and 1, the test triggers retrieval; counts 3 and 2 do not. The rule
uses answer agreement, not the probability that the answer is true.

<details id="tool-use-code-react-vote">
<summary>Python: Когда голосование запускает ReAct</summary>

~~~python
from collections import Counter

def fallback_to_react(extracted_answers):
    if not extracted_answers:
        raise ValueError("at least one sampled answer is required")
    largest_group = max(Counter(extracted_answers).values())
    return largest_group < len(extracted_answers) / 2

assert fallback_to_react(["A", "A", "B", "B", "C"])
assert not fallback_to_react(["A", "A", "A", "B", "B"])
# The literal strict-inequality rule retains an exactly-half group.
assert not fallback_to_react(["A", "A", "B", "B"])
# Agreement alone cannot detect five identical wrong answers.
assert not fallback_to_react(["wrong"] * 5)
~~~

</details>

This example assumes answers have already been extracted and normalized.
It does not infer equivalence between differently worded answers.
The strict comparison is the heuristic stated in the paper; it is not a
calibrated confidence test.

![[00 Учебник/Assets/Figures/curated/agent-reconciliation-2026-10-30/react-cot-sampling.png|Два графика показывают результаты HotpotQA и FEVER при разном числе CoT-SC образцов и переключении на ReAct.]]

*Shunyu Yao et al., [ReAct v3, Figure 2, PDF p. 5](https://arxiv.org/pdf/2210.03629v3#page=5).
CC BY 4.0. Complete original two-panel figure and English caption; surrounding
paper text excluded.*

The horizontal axis counts CoT-SC samples. It does not count all tokens,
search requests or model invocations in a combined method. The two panels
also use different metrics: exact match for HotpotQA and classification
accuracy for FEVER. Their vertical values should not be averaged into one
score. With the 21-sample CoT-SC baseline, Table 1 reports:

| PaLM-540B prompting condition | HotpotQA EM | FEVER accuracy |
|---|---:|---:|
| CoT | 29.4 | 56.3 |
| Act-only | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC, 21 samples | 33.4 | 60.4 |
| ReAct → CoT-SC | 35.1 | 62.0 |
| CoT-SC → ReAct | 34.2 | 64.6 |

ReAct beats Act-only on both tasks, but it does not beat CoT on HotpotQA
in this comparison. The better combined condition differs between tasks.
The curves support the authors' result that combinations can reach the
21-sample CoT-SC score with fewer CoT samples; they do not measure a
corresponding fixed multiple of end-to-end cost savings.

#### Reading the error analysis and the fine-tuning result

Table 2 is a manually labeled analysis of 200 selected trajectories:
successful and unsuccessful cases from both ReAct and CoT. The reported
0% hallucination category belongs to the selected ReAct failures.
It is not a measured zero hallucination rate across every ReAct answer.
The same table assigns 6% of selected ReAct successes to a false-positive
category with hallucinated reasoning or facts: a correct final answer
can coexist with a defective trajectory. Non-informative search, reasoning
errors and answer-label ambiguity remain separate sources of failure.

![[00 Учебник/Assets/Figures/curated/agent-reconciliation-2026-10-30/react-prompt-vs-finetune.png|Слева результаты prompting, справа дообучение PaLM на траекториях. Справа нет результата для 540B.]]

*Shunyu Yao et al., [ReAct v3, Figure 3, PDF p. 7](https://arxiv.org/pdf/2210.03629v3#page=7).
CC BY 4.0. Both original panels, legend, axes and English caption retained.*

The left panel compares prompting formats at three PaLM sizes. ReAct
performs poorly for the smaller prompted models, despite using the same
reasoning-and-action format. The right panel uses 3,000 generated
trajectories selected for correct final answers to fine-tune smaller
models. The ranking changes: fine-tuned ReAct at 8B beats the tested
62B prompting conditions; fine-tuned ReAct at 62B beats the tested
540B prompting conditions. There is no 540B fine-tuning result in the
right panel; its empty position must not be read as a zero score.

This is task-specific evidence on HotpotQA, not a guarantee that a small
fine-tuned agent beats every larger model. Selecting a correct final
answer also does not certify every intermediate thought. At execution
time, observations still come from the environment, even if training
trajectories included observation text.

The ALFWorld numbers use another experimental protocol. The often-quoted
71% is the best of six ReAct prompt choices on 134 unseen games; the mean
over those choices is 57%. It should not replace the mean or be compared
to a mean as if both used the same selection. WebShop evaluates 500
instructions and distinguishes attribute-coverage score from complete
success; its reported ReAct success rate is 40.0%. These are the original
benchmark conditions, not current leaderboard claims.

### Объём и полнота ответа

Для файла инструмент возвращает диапазон строк и признак усечения, для поиска
по коллекции — ограниченную страницу и способ продолжения. Например, ответ
`{"items":[{"id":"p7","title":"ReAct"}],"next_cursor":"page-2"}` сообщает,
что выдача ещё не закончилась. Из него нельзя заключить, что в коллекции
существует только одна работа. Курсор принадлежит той же выборке: при смене
запроса или снимка данных его обычно приходится получать заново.

Если следующий инструмент ожидает идентификатор документа, краткий режим
поиска должен сохранять этот идентификатор. Если нужно прочитать определённый
фрагмент, ответ должен содержать доступную ссылку или адрес фрагмента.
Сокращение вывода, после которого приходится угадывать следующий аргумент,
повышает число ошибок. Эти требования к контексту, краткому и подробному режимам
разбираются в [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents#returning-meaningful-context-from-your-tools).

[[#mcp-response-correlation|Сопоставление по идентификатору]] позволяет
обрабатывать ответы в любом порядке. Два независимых чтения можно выполнять
параллельно. Поиск идентификатора и чтение найденного документа зависимы:
второй вызов нельзя исполнить до результата первого. Даже два read-only вызова
не образуют согласованный снимок автоматически, если данные меняются между ними.

<span id="3-вызов-функции-генерация-кода-и-текстовые-действия"></span>

## Как модель записывает действие

В примере с файлом модель предлагает имя функции и словарь аргументов.
В ReAct действие могло выглядеть как строка `Search[Front Row]`.
Способ записи определяет, что придётся разбирать и проверять приложению.

| Представление | Что меняется |
|---|---|
| Вызов функции в JSON | Схему легко проверить; допустимые типы ещё не гарантируют правильный смысл аргументов. |
| Генерируемый код | Один фрагмент может объединить несколько операций. Исполнять его нужно с ограничениями. |
| Текстовая команда | Для прототипа достаточно небольшой грамматики; неоднозначный текст труднее разбирать. |
| Специальные токены | Модель может учиться предсказывать их напрямую. Формат зависит от словаря и обучающих данных. |

[Toolformer](https://arxiv.org/abs/2302.04761) исследует обучение модели выбору
места вызова API, инструмента и его аргументов. Синтетические примеры
фильтруются по снижению языковой потери, затем используются для дообучения.
Toolformer не решает вопросы разрешений, повторов и побочных
эффектов: они принадлежат исполняющей среде, а не весам модели.

Генерация кода, которую использует семейство `CodeAgent`, даёт модели удобный
язык композиции. Вместо десяти последовательных JSON-вызовов можно написать
цикл обработки результатов. Но такой код необходимо запускать в изолированной
среде с ограниченными файловой системой, сетью, временем, памятью и секретами.

### Как Toolformer получает обучающие примеры

Пусть в исходном корпусе есть предложение «Pittsburgh is also known as the
Steel City». Перед словами «the Steel City» модель предлагает два запроса:
как ещё называют Pittsburgh и в какой стране он находится. Ответы «Steel City»
и «United States» оба могут быть правильными. Но только первый помогает
предсказать ближайшее продолжение именно этого предложения.

В [Toolformer, §2](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf#page=3)
используется три стадии подготовки данных. Сначала несколько демонстраций
подсказывают модели, как вставлять вызовы данного API. Модель оценивает позиции,
в которых вероятно начало вызова, оставляет кандидатов выше порога и сэмплирует
аргументы. Затем программа действительно исполняет эти вызовы. Наконец,
результаты фильтруются по тому, насколько они уменьшают ошибку предсказания
исходного текста. Авторская схема этого процесса находится в
[Figure 2 опубликованной статьи](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf#page=2).

Для вызова $c$ с результатом $r$ обозначим его текстовую запись через
$e(c,r)$. Вводятся три условия для предсказания тех же целевых токенов:
без вызова, с вызовом без ответа и с полным результатом. Ошибка по продолжению
считается как взвешенная сумма отрицательных логарифмов вероятностей:

$$
\begin{aligned}
L_i(z)&=-\sum_{j=i}^{n}w_{j-i}
             \log p_M(x_j\mid z,x_{1:j-1}),\\
L_i^+&=L_i(e(c,r)),\\
L_i^-&=\min\{L_i(\varnothing),L_i(e(c,\varnothing))\},\\
\Delta_i&=L_i^- - L_i^+.
\end{aligned}
$$

Кандидат сохраняют при $\Delta_i\ge\tau_f$. Минимум двух базовых ошибок
проверяет, что полезен результат API, а не только текст вопроса, который
модель сама себе добавила. В реализации отбора запись API служит префиксом
при оценке исходного текста: до дообучения вставка непривычной конструкции
посреди предложения сама могла ухудшить вероятности. После отбора вызов
уже вставляется на выбранное место в обучающем примере.

В эксперименте веса ближайших целевых токенов пропорциональны
$1,0.8,0.6,0.4,0.2$, а затем нулю. Поэтому вызов поощряется там, где ответ
нужен скоро. Ниже заданы учебные вероятности пяти правильных целевых токенов,
а не результаты запуска GPT-J.

<details id="tool-use-code-loss">
<summary>Python: Сравнение четырёх ошибок предсказания</summary>

~~~python
import numpy as np

weights = np.array([1., .8, .6, .4, .2])
weights /= weights.sum()

def continuation_loss(target_probabilities):
    probabilities = np.asarray(target_probabilities, dtype=float)
    assert probabilities.shape == weights.shape
    assert np.all((probabilities > 0) & (probabilities <= 1))
    return -float(weights @ np.log(probabilities))

no_call = continuation_loss([.10, .20, .30, .40, .50])
call_without_result = continuation_loss([.15, .15, .30, .40, .50])
useful_result = continuation_loss([.60, .85, .90, .95, .95])
irrelevant_result = continuation_loss([.05, .05, .05, .05, .05])
baseline = min(no_call, call_without_result)

assert baseline - useful_result >= .5
assert baseline - irrelevant_result < .5
~~~

</details>

Получаются потери 1.6059 без вызова, 1.5474 с одним вопросом,
0.2449 с полезным ответом и 2.9957 с бесполезным. База равна 1.5474:
полезный ответ уменьшил её примерно на 1.3025 и прошёл порог 0.5,
бесполезный увеличил и был отброшен.

Сохранённый фрагмент принимает вид
`Pittsburgh is also known as [QA(...)->Steel City] the Steel City`.
На расширенном корпусе модель дообучается минимизацией обычной языковой
потери. При выводе
генерация останавливается перед ожидаемым ответом API, программа выполняет
вызов, вставляет настоящий результат и продолжает декодирование. Символы
`<API>`, `</API>` и стрелка в формулах статьи обозначают границы;
в эксперименте им соответствовали уже существовавшие последовательности
`[`, `]` и `->`, без расширения словаря модели.

Toolformer учит модель вызывать API, меняя её веса. Основные эксперименты
ReAct с демонстрациями используют замороженный PaLM. Уменьшение языковой
потери ничего не говорит о полномочиях, истинности каждого ответа API или
безопасности побочного эффекта. В частности, правильный ответ
«United States» может быть отброшен как бесполезный для выбранного продолжения,
а ошибочный текст, случайно подсказывающий целевые токены, может пройти фильтр.
В исходном эксперименте использованы чтение, поиск, вычисление, перевод и
получение даты; этот способ подготовки данных нельзя без дополнительного анализа переносить
на операции записи или платежи.

<a id="toolformer-curation-evaluation"></a>

### Toolformer: selecting data and deciding when to call

The loss filter above operates after candidate calls have been proposed.
It cannot retain a useful call that the generator never produced. In
the published experiment, the default sampler first kept positions whose
API-start probability exceeded 0.05, selected at most five such positions
per document, and sampled up to five calls at each position. These are
separate limits: five possible insertion sites can produce up to 25
candidates, not five final training examples.

Calculator and translation examples were harder to obtain. Their recipe
used a zero sampling threshold, up to 20 positions and ten candidates per
position, with a lower loss-improvement threshold of 0.5 instead of 1.0.
This increases the work spent proposing and executing candidates; it does
not make every proposed call useful. The sampling and filtering settings
are specified in [Toolformer, Supplement A](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Supplemental-Conference.pdf#page=1).

For a small example, suppose API-start probabilities at six positions are
0.01, 0.08, 0.07, 0.04, 0.06 and 0.03. A threshold of 0.05 admits positions
1, 2 and 4 using zero-based indexing. A two-position budget retains 1 and 2.
A high score at the third admitted position does not overcome that budget.
The same idea applies to the published five-position limit.

<details id="tool-use-code-sampling">
<summary>Python: Отбор позиций и начало вызова</summary>

~~~python
def insertion_sites(probabilities, threshold, max_positions):
    eligible = [i for i, p in enumerate(probabilities) if p > threshold]
    return sorted(eligible, key=lambda i: (-probabilities[i], i))[:max_positions]

p = [.01, .08, .07, .04, .06, .03]
assert insertion_sites(p, .05, 2) == [1, 2]

# A toy token ranking, not probabilities measured from GPT-J.
def api_start_allowed(ranked_tokens, k, calls_already_made=0):
    return calls_already_made == 0 and "[API]" in ranked_tokens[:k]

ranking = ["ordinary-word", "another-word", "[API]", "other"]
assert not api_start_allowed(ranking, 1)
assert api_start_allowed(ranking, 3)
assert not api_start_allowed(ranking, 3, calls_already_made=1)
~~~

</details>

The stable index tie-break in this illustrative sampler makes its output
reproducible; the paper does not specify it as an implementation requirement.
The API marker in the second function is a toy vocabulary item. The original
experiment uses existing token sequences rather than adding this literal
token to GPT-J.

The minimum in the filtering baseline has a separate purpose. Suppose
loss is 3.0 with no call, 1.0 with the question alone, and 2.4 with the
question and the result. Comparing only against 3.0 would report a gain of
0.6. Comparing against the required minimum, 1.0, gives −1.4, so the call
is rejected. The generated question already helped the model more than
the supplied answer did. This is why filtering against a single no-call
baseline changes the method.

#### The annotator sees more text than the deployed model

The annotation prompt contains the document being annotated. It can use
later text to propose an earlier call; ordinary inference cannot see that
future continuation. Supplement A therefore removes translation examples
whose translation input appears after the call but not before it. Otherwise
the training sequence can teach the model to request a phrase that has
not yet become available.

Other tools also require source-specific preparation. The training calendar
approximates a document's date from its URL and discards documents without
an extractable date. It does not insert today's date into every historical
paragraph. The calculator heuristics favor text containing arithmetic
relations or several numbers. Atlas-large generates QA annotations,
whereas Atlas-xxl answers QA calls at evaluation time. The language model's
6.7B parameters therefore do not describe the full system's inference cost.

After execution and filtering, retained calls are inserted into the text.
Multiple independently sampled calls can appear in an annotated document;
the method does not build examples where a later tool consumes an earlier
tool's result. The experiment also discards documents in which no call
survived. Table 2 reports retained-call example counts under several
thresholds, while Supplement B caps the training examples at 25,000 per API.
Those counts are not all the same quantity.

For instance, the QA, Wikipedia and calendar rows at threshold 1.0 contain
18,526, 60,974 and 20,587 examples before that cap; calculator and translation
use threshold 0.5, with 3,680 and 3,156 examples. A large search row does
not imply that all 60,974 search examples were used in the final capped
recipe. Supplement B describes sequences of at most 1,024 tokens, effective
batch size 128, up to 2,000 updates, and checkpoint selection on a separate
CCNet development set. These are historical training settings, not budgets
reproduced by the small examples here.

#### Decoding changes the frequency of tool use

The published zero-shot evaluation uses greedy decoding with a special
exception: it starts an API call whenever the API-start marker is among
the top ten candidates, even if another token ranks first. The toy ranking
above illustrates the difference between a rank-one and rank-three trigger.
This is an inference rule applied to the same weights.

Supplement E reports that changing this rank limit from 1 to 10 raises
the call rate on T-REx from 40.3% to 98.1%, and on WebQS from 8.5% to 100%.
A high call rate is therefore not solely a learned preference of the model.
The evaluation permits at most one call per input to prevent repeated calls.
That cap is an experimental decoding constraint; it does not mean the
augmented training documents contain at most one inserted call.

When the model reaches the response delimiter, generation pauses. The
executor supplies the real tool response and the closing delimiter, then
the model resumes. A generated claim about what an API returned is not a
substitute for that execution. Original Toolformer cannot use this loop to
refine a search query repeatedly or compose several tools.

#### What the reported measurements establish

The paper evaluates language modeling with API calls disabled. GPT-J +
CCNet and Toolformer-disabled both obtain perplexities 10.3 on WikiText
and 10.5 on the held-out CCNet sample; the original GPT-J has 9.9 and 10.6.
Thus the augmented run matches the unaugmented fine-tuning control in
those two tests, while fine-tuning itself worsens WikiText relative to the
original checkpoint. This does not prove preservation of every language
capability. With calls enabled, evaluating a text's marginal probability
would require accounting for possible calls; the reported comparison
does not perform that calculation.

The gain from a tool also depends on the model and the tool. In
[Supplement G](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Supplemental-Conference.pdf#page=7),
the stronger LLaMA v1 7B model benefits from the targeted Atlas QA tool
on LAMA, but noisy BM25 search can hurt open-domain QA. The following
rows use the appendix's QA evaluation prompt and answer-matching protocol:

Each row compares the base model with the model fine-tuned on
self-generated Toolformer data, first without calls and then with calls.

| Benchmark | Base → calls off → calls on |
|---|---:|
| WebQS | 36.2 → 37.0 → 36.6 |
| Natural Questions | 25.4 → 27.6 → 26.9 |
| TriviaQA | 59.2 → 60.2 → 57.8 |

The last two numbers in each row isolate the effect of enabling calls for
that fine-tuned checkpoint under the reported evaluation. The comparison does not establish
that search is generally harmful: the returned passages, call trigger,
prompt and base knowledge all matter. The paper's scaling observation
around 775M parameters is likewise specific to the tested GPT-2/GPT-J
models and tools, with an exception for search; it is not a universal
minimum model size for tool use.

<a id="mcp-host-client-server"></a>

<span id="где-здесь-mcp"></span>

## Как приложение вызывает инструмент по MCP

Модель предлагает `read_file`, а приложение должно найти и вызвать программу,
которая умеет читать файл. Если это внешний MCP-сервер, приложение (host)
использует свой MCP-клиент: через `tools/list` клиент получает имя, описание и
`inputSchema`, а через `tools/call` передаёт проверенные аргументы.
Сервер выполняет чтение и возвращает результат клиенту. Решение о том, какие
инструменты показать модели и при каких условиях вызвать, остаётся у приложения.

![[00 Учебник/Assets/Figures/curated/tool-use-2026-11-09/mcp-simple-diagram.png|Слева приложения, справа источники данных и инструменты; MCP обозначает протокол обмена между ними]]

*Anthropic и участники проекта Model Context Protocol,
[FAQ, «How does MCP work?»](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/27ab2419191334dd288b05343e3a622ed615913d/docs/faqs.mdx#how-does-mcp-work),
[оригинальная схема](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/27ab2419191334dd288b05343e3a622ed615913d/docs/images/mcp-simple-diagram.png),
[MIT](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/27ab2419191334dd288b05343e3a622ed615913d/LICENSE).*

IDE слева на рисунке может обратиться к файловой системе или к Git справа.
Она открывает отдельное клиентское соединение с каждым сервером.
Центральная плашка `MCP` обозначает общий формат обмена, а не третий
процесс, через который проходят все запросы. Сами данные приложение
передаёт модели отдельно. Роли участников описаны в
[Architecture](https://modelcontextprotocol.io/specification/2025-11-25/architecture#core-components),
а формат `tools/list` и `tools/call` — в
[Tools, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

Например, модель предложила `read_file` с диапазоном строк 1–2.
После проверки клиент может представить вызов серверу так:

<details id="tool-use-code-mcp-request">
<summary>Запрос tools/call с id 17</summary>

~~~json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": {
      "path": "notes.txt",
      "start_line": 1,
      "end_line": 2
    }
  }
}
~~~

</details>

Здесь 17 связывает JSON-RPC запрос с ответом. Это не ключ идемпотентности
операции и не обязательно идентификатор tool call в API модели. Приложение
хранит соответствие между ними. Для успешного чтения ответ сервера может
содержать `content` с текстом и объект `structuredContent`.
Если сервер объявил `outputSchema`, структурированный результат должен
соответствовать этой схеме. Сама структура ответа не доказывает, что
модель выбрала правильные аргументы или что найденные данные истинны.

Подсказка `readOnlyHint` в описании инструмента не заменяет проверку прав.
Спецификация требует считать такие аннотации недоверенными, если они
пришли от сервера, которому приложение не доверяет.

<a id="mcp-response-correlation"></a>

### Если ответы приходят не по порядку

Модель просит одновременно прочитать `notes.txt` и `settings.txt`.
Приложение отправляет два отдельных запроса в одной MCP-сессии. Перед
отправкой оно записывает, какому вызову модели соответствует каждый
JSON-RPC `id`. В этой сессии запрашивающая сторона не использует
один и тот же идентификатор дважды.
[MCP 2025-11-25, Messages](https://modelcontextprotocol.io/specification/2025-11-25/basic#messages).

| Ответ в порядке получения | К какому вызову он относится |
|---|---|
| Сначала `18`: файла нет, `isError: true`. | `c3`, чтение `settings.txt`. |
| Затем `17`: `timeout=30`, `retries=2`. | `c2`, чтение `notes.txt`. |

Если раздать ответы по порядку отправки, модель получит «файла нет»
для `notes.txt` и припишет `timeout=30` файлу `settings.txt`.
Сопоставление по `id` сохраняет обе связи. В случае `18` внешняя
оболочка всё равно содержит `result`: JSON-RPC-запрос обработан,
но чтение не удалось. Если клиент отправит устаревшее имя инструмента,
например `read_old_file`, сервер может вернуть `error` в самой оболочке.
Приложению нужно отличить неисправный вызов от результата чтения.

Функция ниже принимает уже разобранный ответ из одной сессии и возвращает
запись для журнала приложения. Она сохраняет данные обеих ошибок и не
относит неизвестный или уже обработанный `id` ни к одному вызову модели.
Полную проверку входного сообщения выполняет слой MCP-клиента.

<details id="tool-use-code-mcp-correlation">
<summary>Python: Сопоставление ответов MCP</summary>

~~~python
import json

pending = {17: "c2", 18: "c3", 19: "c4"}

def correlate_response(message):
    rpc_id = message.get("id")
    if (message.get("jsonrpc") != "2.0"
            or type(rpc_id) not in (str, int) or rpc_id not in pending):
        return None
    if ("error" in message) == ("result" in message):
        return None
    if "error" in message:
        kind, payload = "rpc_error", message["error"]
    else:
        payload = message["result"]
        if not isinstance(payload, dict):
            return None
        is_error = payload.get("isError", False)
        if type(is_error) is not bool:
            return None
        kind = "tool_error" if is_error else "tool_result"
    return {"call_id": pending.pop(rpc_id), "kind": kind, "payload": payload}

def rpc(rpc_id, **body):
    return {"jsonrpc": "2.0", "id": rpc_id, **body}

missing = rpc(18, result={
    "isError": True,
    "content": [{"type": "text", "text": "settings.txt not found"}],
})
data = {"path": "notes.txt", "lines": ["timeout=30", "retries=2"]}
reading = rpc(17, result={
    "content": [{"type": "text", "text": json.dumps(data)}],
    "structuredContent": data,
})
stale = rpc(19, error={"code": -32602, "message": "Unknown tool: read_old_file"})

assert correlate_response(rpc(999, result={})) is None
assert correlate_response(rpc(17, result={}, error={})) is None
received = [correlate_response(msg) for msg in (missing, reading, stale)]
assert [(item["call_id"], item["kind"]) for item in received] == [
    ("c3", "tool_error"), ("c2", "tool_result"), ("c4", "rpc_error"),
]
assert received[0]["payload"]["content"][0]["text"] == "settings.txt not found"
assert received[1]["payload"]["structuredContent"] == data
assert json.loads(received[1]["payload"]["content"][0]["text"]) == data
assert received[2]["payload"]["code"] == -32602
assert not pending and correlate_response(reading) is None
~~~

</details>

В журнале первым появится `c3`, но прочитанные строки относятся к
`c2`. Приложение ведёт таблицу ожидающих ответов отдельно для каждой
сессии: `id: 17` в двух сессиях может обозначать разные вызовы.
При записи сопоставление не защищает от повторного исполнения. Здесь
три разных идентификатора: `c2` связывает наблюдение с вызовом модели,
`17` связывает ответ с запросом в MCP-сессии, а
[[#tool-retry-idempotency|ключ операции]] связывает несколько попыток выполнить
одно намерение. При потере ответа повтор получает новый JSON-RPC `id`,
но сохраняет ключ операции, если сервер поддерживает такой контракт.

<span id="5-ошибка--часть-протокола"></span>

## Что делать, если вызов не удался

Ошибки следует делить по следующему действию, а не по месту возникновения.

| Что произошло | Следующий шаг |
|---|---|
| Не прошло проверку одно поле. | Вернуть имя поля и ограничение. |
| Сервер временно ограничил частоту запросов. | Проверить, можно ли повторить, и соблюдать задержку и предел попыток. |
| Версия объекта изменилась. | Перечитать версию, затем заново выбрать действие. |
| Операция не разрешена. | Остановить действие; запрашивать подтверждение только там, где политика это допускает. |
| Ответ после записи потерян. | Проверить состояние по ключу операции или через другой авторитетный источник. |
| Нужного объекта нет, а создать его нельзя. | Завершить с объяснением, основанным на наблюдении. |

<a id="tool-retry-idempotency"></a>

### Потерянный ответ после записи

Пусть инструмент `create_note` создаёт заметку. Сервер сохранил её, но ответ
не дошёл до клиента. Timeout означает, что клиент не получил результат вовремя;
он не сообщает, было ли действие выполнено. Новый запрос на создание может
породить дубликат.

Клиент сохраняет идентификатор намерения и использует его при повторах.
Сервер связывает пару «клиент, ключ» с параметрами операции и созданным объектом.
При повторе с теми же параметрами возвращается прежний результат. Тот же ключ
с другими параметрами должен давать конфликт, а не молча создавать другой объект.
Такой контракт разобран в [AWS Builders' Library: Making retries safe with
idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/).

Ниже всё происходит в памяти одного процесса, без файлов, сети и модели.
`lose_reply` имитирует потерю ответа после сохранения. Словарь `operations`
служит журналом уже выполненных намерений.

<details id="tool-use-code-retry">
<summary>Python: Повтор создания заметки</summary>

~~~python
notes = {}
operations = {}

def create_note(caller, key, title, body, *, lose_reply=False):
    identity = (caller, key)
    payload = (title, body)
    if identity in operations:
        old_payload, note_id = operations[identity]
        if old_payload != payload:
            raise ValueError("idempotency_parameter_mismatch")
        return note_id

    note_id = f"n{len(notes) + 1}"
    notes[note_id] = payload
    operations[identity] = (payload, note_id)
    if lose_reply:
        raise TimeoutError("response lost after commit")
    return note_id

try:
    create_note("alice", "task-7/create-1", "ReAct", "Read the paper",
                lose_reply=True)
except TimeoutError:
    pass

assert len(notes) == 1
note_id = create_note("alice", "task-7/create-1", "ReAct", "Read the paper")
assert note_id == "n1" and len(notes) == 1

try:
    create_note("alice", "task-7/create-1", "Toolformer", "A different task")
except ValueError as error:
    assert str(error) == "idempotency_parameter_mismatch"
else:
    raise AssertionError("Changed parameters must not reuse the key")
assert len(notes) == 1
~~~

</details>

Если второй запрос отправить с новым ключом, сервер увидит другое намерение
и создаст вторую заметку. Хеш одинакового текста тоже не заменяет намерение:
пользователь вправе попросить две одинаковые заметки. Идентификатор одного
JSON-RPC запроса из предыдущего раздела связывает сообщения, но сам по себе
не обеспечивает дедупликацию действия.

В этом последовательном примере запись заметки и журнала не прерывается.
В настоящем сервисе их нужно фиксировать атомарно и хранить журнал достаточно
долго для допустимых повторов. Иначе сбой между двумя записями или удаление
ключа вернут возможность дубликата. Клиент должен знать срок действия
контракта; произвольный повтор после его истечения нельзя считать безопасным.
Если сервер не предоставляет такого контракта и фактический исход установить
не удалось, исполнитель сохраняет статус «исход неизвестен» и не выдаёт его
за подтверждённый отказ или успех.

<span id="6-права-нельзя-задать-одной-инструкцией-модели"></span>

## Где проверяются права

Текст «не удаляй файлы» способен повлиять на выбор модели, но не запрещает
системный вызов. Надёжная граница находится после генерации действия:

1. синтаксический анализатор принимает только объявленный формат;
2. валидатор проверяет значения и нормализованные пути;
3. policy engine сопоставляет действие с правами и необходимостью подтверждения;
4. sandbox ограничивает фактические возможности процесса;
5. журнал фиксирует запрос, решение политики и исполненный результат.

Содержимое, полученное из инструмента, остаётся данными. Если библиотечная
страница предлагает сразу оформить бронь, это предложение не меняет просьбу
пользователя «ничего не бронируй». Приложение сохраняет происхождение
сообщения и проверяет предлагаемое действие по действующим разрешениям.

Подтверждение также относится к определённому действию. Разрешение создать
заметку в одном проекте не разрешает отправить её в другой сервис.
Перед исполнением проверяют имя операции, цель и существенные параметры;
изменение цели после подтверждения требует новой проверки. Запрет в prompt
может уменьшить число ошибочных предложений, но фактический отказ должен
возникать до побочного эффекта. Подробные модели угроз и prompt injection
разобраны в [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|главе 72]].

<span id="7-когда-workflow-лучше-свободного-агента"></span>

## Когда порядок действий можно задать заранее

В [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
Anthropic различает заранее заданные процессы и агентов, где модель сама
управляет последовательностью действий. Если порядок действий заранее известен, workflow позволяет явно задать
переходы и условия каждого шага. Это упрощает проверку допустимых траекторий,
но не делает любой workflow автоматически дешевле: стоимость зависит от числа
вызовов модели и инструментов. Свободный цикл полезен, когда результаты среды
меняют следующий шаг и заранее неизвестно число итераций.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-11-10/toolsandbox-roles.png|Пользователь общается с агентом слева; справа агент вызывает инструменты, а среда возвращает результаты.]]

*Jiarui Lu et al., [ToolSandbox v2, Figure 2](https://arxiv.org/html/2408.04682v2#S2.F2),
[оригинал](https://arxiv.org/html/2408.04682v2/architecture_diagram.png),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

В тесте ToolSandbox пользователь просит отправить сообщение. Первый вызов
`send_message` возвращает `ConnectionError`: сотовая связь выключена.
Агент включает её и повторяет отправку. На схеме все три вызова проходят
справа, между агентом и средой, без новой реплики пользователя. Если агенту
не хватает текста сообщения или получателя, он обращается к пользователю
слева. [В статье показана и сама траектория теста](https://arxiv.org/html/2408.04682v2#S1.F1);
[схема «Agents» Anthropic](https://www-cdn.anthropic.com/images/4zrzovbb/website/58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png)
показывает тот же цикл без деталей испытательной среды.

Для задачи «прочитай файл и назови значение» переходы можно задать заранее:
проверка пути, чтение, разбор строки, ответ. Поиск в незнакомом репозитории
обычно требует выбора по ходу работы: ошибка теста определяет следующий файл,
а содержимое файла — следующую проверку. Один процесс может сочетать оба
варианта: свободный поиск причины и фиксированный, проверяемый порядок
применения правки.

Условия остановки и успеха различаются. Если исчерпан лимит шагов, цикл
останавливается даже без ответа. Для чтения файла успехом будет ответ,
подтверждённый чтением; после исправления кода нужна проверка того теста,
который до правки падал.

<span id="где-начинается-агент-модель-память-действие-и-среда"></span>
### Что знает модель до выполнения

<!-- source_unit_id: meeting-01-intro-reasoning-agent-progress -->

Reasoning-модель может долго решать математическую задачу, продолжая одну
последовательность токенов. В инструментальной агентной системе программа
включает модель в замкнутый процесс: передаёт ей наблюдаемое состояние, разрешает
выбрать действие, исполняет его во внешней среде и возвращает результат для
следующего шага. Память сохраняет то, что должно пережить отдельный вызов, а
правило остановки определяет, когда последовательность действий считается
завершённой.

Поиск может вернуть новую статью, компилятор — ошибку, браузер — изменившуюся
страницу, proof assistant — новое состояние доказательства. Эти наблюдения не
гарантируют успеха, но дают возможность обнаружить неверный ход и изменить
следующий. Во вводных слайдах
[Berkeley Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25)
эта конструкция включает LLM, reasoning/planning, retrieval, memory, tools,
action и environment feedback. Схема и её пояснение приведены на физических
страницах 5–6 вводной презентации. Длинный ответ модели
может быть частью решения, но состояние файла или сервиса узнают из внешнего
наблюдения. Наблюдение тоже бывает частичным или устаревшим: чтение первой
страницы поиска не доказывает, что других результатов нет.

<span id="два-способа-проектировать-агента-llm-first-и-agent-first"></span>
### Проектирование от модели и от задачи

<!-- source_unit_id: meeting-03-slides-yu-su-agent-first-vs-llm-first -->

В третьей лекции курса [Berkeley Advanced LLM
Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25) Юй Су противопоставляет
два полезных взгляда на одну систему (физические страницы 5–15). В подходе
**LLM-first** отправной точкой служат возможности готовой модели: к ней
подбирают prompt, инструменты и программную обвязку, чтобы превратить хороший
ответ в последовательность действий. Этот путь удобен для быстрого прототипа,
но легко скрывает устройство задачи за всё более сложным prompt.

В подходе **agent-first** сначала описывают среду: что агент наблюдает, какие
действия ему разрешены, какое состояние нужно хранить, как выглядит успех и
какие переходы необратимы. LLM затем занимает определённое место в этой
конструкции — например, переводит наблюдение на язык плана, выбирает действие
или объясняет результат. Здесь по-прежнему нужны восприятие, модель мира,
планирование и работа с неполным состоянием. Языковая модель расширяет способы
представить эти операции и связать их естественным языком, но не заменяет их
автоматически.

Эти взгляды задают разные вопросы к одному проекту. Для чтения файла
LLM-first вопрос звучит как «понимает ли модель описание `read_file`?»,
agent-first как «что именно доступно для чтения, когда наблюдение считается
актуальным и чем проверяется ответ?». Из этих определений не следует, что
одна школа всегда надёжнее другой. В обоих случаях права и проверка результата
должны существовать в исполняющей программе, а не зависеть от удачного prompt.

На слайде 7 в той же лекции генерация рассуждения относится к действиям
во внутренней среде. Если наблюдение сообщает «ящик закрыт», модель может
записать в плане шаг открытия. Состояние ящика изменится только после
действия в среде, а узнать, удалось ли оно, можно по новому наблюдению.

<span id="8-как-проверять-tool-use"></span>

## Как подтвердить выполнение задачи

Изолированная точность выбора функции полезна, но недостаточна. Проверка должна
последовательно отвечать на вопросы:

- выбран ли правильный инструмент среди похожих;
- валидны ли аргументы и не потеряны ли обязательные ограничения;
- достигнуто ли нужное состояние среды;
- не возникли ли лишние побочные эффекты;
- восстановилась ли система после ошибки;
- сколько вызовов, токенов и подтверждений потребовалось;
- можно ли воспроизвести траекторию из журнала.

Результат теста должен отделять ошибку выбора от ошибки исполнения.
Корректный JSON при недопустимом пути не является успешным вызовом;
успешный вызов с неверным ответом тоже не закрывает задачу.

<a id="tool-use-verification"></a>

### Полный учебный эпизод с проверяемым завершением

В следующем примере предложения модели заданы заранее: сначала неверный
диапазон, затем исправленный вызов, затем ответ. Это позволяет проверить
исполнитель и условие успеха отдельно от способности LLM выбирать действия.
Сообщение об ошибке в этом коде сокращено до её типа; для диалога с моделью
его следует дополнить сведениями о поле и допустимом значении, как выше.
`call_id` хранится вместе с наблюдением, а `seen_ids` запрещает незаметное
повторное использование идентификатора в этом эпизоде.

<details id="tool-use-code-episode">
<summary>Python: Проверка всего эпизода</summary>

~~~python
FILES = {"notes.txt": ("fixture-v1", ("timeout=30", "retries=2"))}
READABLE = {"notes.txt"}

def read_tool(proposal):
    call_id = proposal["call_id"]
    def failure(kind):
        return {"call_id": call_id, "ok": False,
                "error": kind, "executed": False}

    if proposal.get("name") != "read_file":
        return failure("unknown_tool")
    args = proposal.get("arguments")
    fields = {"path", "start_line", "end_line"}
    if not isinstance(args, dict) or set(args) != fields:
        return failure("schema_error")
    path, start, end = (args[k] for k in
                        ("path", "start_line", "end_line"))
    if (not isinstance(path, str) or not path
            or type(start) is not int or type(end) is not int
            or start < 1 or end < 1):
        return failure("schema_error")
    if end < start or end - start + 1 > 100:
        return failure("range_error")
    if path not in READABLE:
        return failure("permission_denied")
    if path not in FILES:
        return failure("not_found")
    revision, lines = FILES[path]
    return {"call_id": call_id, "ok": True, "executed": True,
            "path": path, "start_line": start, "end_line": end,
            "lines": list(lines[start-1:end]),
            "truncated": False, "revision": revision}

def verify(answer, observation):
    revision, lines = FILES["notes.txt"]
    if not observation or not observation.get("ok"):
        return False
    # Compare with the trusted fixture, not text invented by the model.
    expected = {"path": "notes.txt", "start_line": 1, "end_line": 2,
                "lines": list(lines), "truncated": False,
                "revision": revision, "executed": True}
    if any(observation.get(k) != v for k, v in expected.items()):
        return False
    values = [line.split("=", 1)[1] for line in lines
              if line.startswith("timeout=")]
    return len(values) == 1 and answer == values[0]

def run_episode(proposals, max_steps=4):
    history, seen_ids, last_observation = [], set(), None
    for step, proposal in enumerate(proposals):
        if step >= max_steps:
            break
        if proposal.get("kind") == "final":
            return verify(proposal.get("answer"), last_observation), history
        if proposal.get("kind") != "call":
            return False, history
        call_id = proposal.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            return False, history
        seen_ids.add(call_id)
        last_observation = read_tool(proposal)
        history.append((proposal, last_observation))
        if last_observation.get("error") == "permission_denied":
            return False, history
    return False, history

proposals = [
    {"kind": "call", "call_id": "c1", "name": "read_file",
     "arguments": {"path": "notes.txt", "start_line": 0, "end_line": 2}},
    {"kind": "call", "call_id": "c2", "name": "read_file",
     "arguments": {"path": "notes.txt", "start_line": 1, "end_line": 2}},
    {"kind": "final", "answer": "30"},
]
success, history = run_episode(proposals)
assert success
assert [item[1]["executed"] for item in history] == [False, True]
assert [item[1]["call_id"] for item in history] == ["c1", "c2"]
assert not run_episode([{"kind": "final", "answer": "30"}])[0]
assert not run_episode(proposals, max_steps=1)[0]
~~~

</details>

В журнале первая попытка имеет `executed=False`; вторая возвращает две строки
с версией `fixture-v1`. Ответ `30` принимается только после подходящего
наблюдения. Тот же ответ без чтения отвергается, как и эпизод, оборванный после
первой ошибки. Такое правило намеренно соответствует конкретной задаче:
для другого файла, диапазона или формата нужен другой verifier.

При подключении настоящей модели следующий запрос должен получать накопленные
сообщения об инструментах и их результатах. Последовательность `proposals`
здесь заменяет только эту часть, чтобы не требовать API-ключа. Испытание кода
проверяет обработку заданных случаев; оно не измеряет точность модели,
устойчивость песочницы или качество агента на произвольных задачах.

<span id="практика-и-первоисточники"></span>
<span id="реализации-для-разбора"></span>

## Источники

Ноутбуки по вызову функций, MCP и LangGraph собраны отдельно в библиотеке
[[05 Источники/Courses/GenAI Agents|GenAI Agents]].

### Первоисточники и дальнейшее чтение

- [Hugging Face Agents Course, Unit 1](https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure) — цикл Thought–Action–Observation и инструменты.
- [ReAct](https://react-lm.github.io/) — готовые сопоставимые траектории и исходная [статья](https://arxiv.org/abs/2210.03629).
- [Toolformer](https://arxiv.org/abs/2302.04761) — self-supervised обучение вызовам API.
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — проектирование описаний и ответов инструментов.
- Model Context Protocol 2025-11-25: [Architecture](https://modelcontextprotocol.io/specification/2025-11-25/architecture) — роли приложения, клиента и сервера; [Messages](https://modelcontextprotocol.io/specification/2025-11-25/basic#messages) — идентификаторы запросов; [Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — обнаружение, вызовы и два уровня ошибок.
- [AWS Builders' Library: Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) — повторы с устойчивым идентификатором намерения.
- [SWE-agent: Agent-Computer Interfaces](https://arxiv.org/abs/2405.15793) — влияние интерфейса компьютера на качество агента.

Раздел об [[00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers#agent-reward-design|обучении агента по проверяемой награде]] показывает, как проверяют
соответствие вызова заданию и состояние среды после него. В
[[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|главе об оценивании агентов]]
валидность вызова, результат задачи, соблюдение ограничений и повторяемость
оцениваются раздельно.

Предыдущая страница: [[00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля|Инструменты и агенты]].
Далее: [[00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|Управляющая программа и контекст]].
