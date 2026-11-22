---
title: Coding, web, and computer-use agents
description: "How an agent fixes code, chooses an action on a page, and verifies its effect in an application. SWE-agent, state search, OSWorld, and learning from trajectories."
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents.md"
canonical_revision: "31c872de16f13f79b167c0419cd843862ac2f3fc93a63252325d2497ebeb9b4f"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: SWE-agent — ACI commands and feedback
    label_en: SWE-agent — ACI commands and feedback
    anchor: coding-agent-aci
  - label: Set-of-Marks — checking a mark before an action
    label_en: Set-of-Marks — checking a mark before an action
    anchor: web-agent-observation-bound-marks
  - label: OSWorld — environment setup and artifact evaluation
    label_en: OSWorld — environment setup and artifact evaluation
    anchor: osworld-resettable-vm-executable-action-and-final-state-grader
primary_sources:
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
  - https://arxiv.org/abs/2405.15793
  - https://arxiv.org/abs/2407.01489
  - https://arxiv.org/abs/2404.05427
  - https://arxiv.org/abs/2501.07531
  - https://arxiv.org/abs/2403.17134v2
  - https://proceedings.mlr.press/v267/abramovich25a.html
  - https://projectzero.google/2024/10/from-naptime-to-big-sleep.html
  - https://arxiv.org/abs/2306.06070
  - https://arxiv.org/abs/2307.13854
  - https://jykoh.com/vwa
  - https://arxiv.org/abs/2407.01476
  - https://arxiv.org/abs/2502.06776
  - https://arxiv.org/abs/2404.07972
  - https://arxiv.org/abs/2412.09605
  - https://arxiv.org/abs/2412.05479v1
  - https://arxiv.org/abs/2412.04454
source_unit_id:
  - meeting-05-slides-coding-agent-orientation
  - meeting-05-slides-swe-agent-loop
  - meeting-05-slides-coding-agent-design-comparisons
  - meeting-05-slides-ctf-agents
  - meeting-05-slides-vulnerability-discovery-problem
  - meeting-05-slides-charles-sutton-vulnerability-discovery-loop
  - meeting-05-slides-coding-agent-conclusions
  - meeting-05-reading-01-catalogue-record
  - meeting-05-reading-02-catalogue-record
  - meeting-06-slides-visualwebarena
  - meeting-06-slides-vision-language-web-agents
  - meeting-06-slides-search-agent-motivation
  - meeting-06-slides-web-agent-tree-search
  - meeting-06-slides-web-agent-tree-search-results
  - meeting-06-slides-insta-setup
  - meeting-06-slides-insta-verification-scaling
  - meeting-06-slides-web-agent-synthesis
  - meeting-06-reading-01-catalogue-record
  - meeting-06-reading-02-catalogue-record
  - meeting-06-reading-03-catalogue-record
  - meeting-06-reading-04-catalogue-record
  - meeting-07-slides-osworld
  - meeting-07-slides-agenttrek
  - meeting-07-slides-taco
  - meeting-07-slides-aguvis
  - meeting-07-reading-01-catalogue-record
  - meeting-07-reading-02-catalogue-record
---


# Coding, web, and computer-use agents

The parser should turn `token=a=b` into the key `token` and the value
`a=b`. The test gets a `ValueError` instead. The agent has to locate the
cause, edit the function, and check that ordinary assignments still work.

On a website the observation might be an accessibility tree or a
screenshot. A click can advance checkout and change the element numbers;
a successful click tells us nothing about the shipping address on a
completed order. In both settings the model chooses what to do next, an
executor acts, and an independent check inspects the resulting state.
See the [[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|evaluation chapter]].

<span id="агент-как-многошаговая-программа-с-инструментами"></span>
<span id="сквозной-разбор-задача-найди-измени-проверь"></span>

## Repairing a parser from a failing test

<span id="минимальный-parser-repair-с-наблюдаемым-результатом"></span>
<span id="пример-исправление-парсера"></span>

### From the failure to a checked patch

Split at the first `=` and strip surrounding whitespace. Further
equals signs belong in the value. The test report points the agent
to the existing function:

```python
def parse_assignment(line):
    key, value = line.split("=")
    return key.strip(), value.strip()
```

The test `assert parse_assignment("token=a=b") == ("token", "a=b")`
raises `ValueError: too many values to unpack (expected 2)`.
`split("=")` returns `["token", "a", "b"]`, so unpacking fails before
`strip()` runs. Splitting once returns `["token", "a=b"]`.

The executor applies the edit only if the expected source matches:

```diff
-    key, value = line.split("=")
+    key, value = line.split("=", 1)
```

```python
def parse_assignment(line):
    key, value = line.split("=", 1)
    return key.strip(), value.strip()

assert parse_assignment("token=a=b") == ("token", "a=b")
assert parse_assignment(" timeout = 30 ") == ("timeout", "30")
assert parse_assignment("empty=") == ("empty", "")
try:
    parse_assignment("missing_separator")
except ValueError:
    pass
else:
    raise AssertionError("missing separator must fail")
print("parser contract: PASS (4 cases)")
```

The two-sign case fails on the old revision and passes on the
repaired one. The other cases cover whitespace, an empty value,
and a missing separator. Record the tested revision and the actual
test output. The diff should show a change to splitting, not to
the assertions. Quoting and escaping are outside this small contract.

### What the model sees between calls

<!-- source_unit_id: meeting-05-slides-coding-agent-orientation -->

The model first reads the failure and the relevant function. After
the edit, the executor runs the test and returns a new observation.
Write $s_t$ for the full environment state, $o_t$ for what the model
can observe, and $a_t$ for its chosen action:

$$
\begin{aligned}
o_t &= O(s_t),\\
a_t &\sim \pi_\theta(a\mid g,o_{\le t},a_{<t}),\\
s_{t+1} &= T(s_t,a_t).
\end{aligned}
$$

The goal is $g$. The executor carries out the transition $T$.
The history $\tau_t=(o_0,a_0,o_1,\ldots,a_{t-1},o_t)$ records
the file excerpts, edit and test results; the repository contains
the actual files. A successful test in the history does not validate
a later retry that restores a different commit.

<a id="coding-agent-aci"></a>
<span id="swe-agent-модель-управляет-циклом-через-aci"></span>

## The SWE-agent command interface

<!-- source_unit_id: meeting-05-slides-swe-agent-loop -->

[SWE-agent, §3](https://arxiv.org/html/2405.15793v3#S3) lets the model
choose the next operation through an agent–computer interface (ACI).
Search, viewing and editing are commands with bounded results. The
execution environment, rather than the ACI's syntax, determines the
permissions for reading, writing and running programs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/computer-agents-2026-11-09/swe-agent-overview.png|SWE-agent sends commands to the environment and receives feedback through the same interface.]]

*John Yang et al., [SWE-agent v3, Figure 1](https://arxiv.org/html/2405.15793v3#S1.F1),
[original PNG](https://arxiv.org/html/2405.15793v3/swe_agent_overview.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Berkeley, lecture 5, slide 17](https://rdi.berkeley.edu/adv-llm-agents/slides/Code%20Agents%20and%20AI%20for%20Vulnerability%20Detection.pdf#page=17).*

The top arrow carries commands and the lower arrow returns results.
`open` shows up to 100 lines, with line numbers, the file path and
the lines outside the window. `goto` moves that window. The search
interface asks for a narrower query instead of returning more than
50 matches. After `edit 404:407` the viewer shows the changed region;
certain syntax errors cause the edit to be reverted and the error
returned. This prevents some invalid edits, but it is not a test of
whether the parser now preserves the second equals sign.

The harness sends the issue, available commands and relevant
history. The model chooses a read, edit or test. The executor checks
and runs it in the environment, and its output becomes the next
observation. The model can then revise the diagnosis or stop. A
separate evaluator checks the repository. To reproduce that verdict,
save the tested revision, command, exit code and output.

<span id="свободный-loop-aci-и-процедурный-workflow"></span>
<span id="где-находится-управление-свободный-цикл-aci-и-процедурный-workflow"></span>

## Who chooses the next step?

<!-- source_unit_id: meeting-05-slides-coding-agent-design-comparisons -->

After a bad repair, SWE-agent can reread the failing test or search
for a caller. Agentless fixes the sequence in code: locate the defect,
generate candidate patches, and validate them. The model chooses
locations and proposes edits within those stages.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m05-p23-agentless-pipeline.png|Agentless narrows the location, generates candidate repairs and selects a patch with tests.]]

*Charles Sutton, [Berkeley Advanced LLM Agents, lecture 5, slide 23](https://rdi.berkeley.edu/adv-llm-agents/slides/Code%20Agents%20and%20AI%20for%20Vulnerability%20Detection.pdf#page=23);
diagram from Xia et al., [Agentless](https://arxiv.org/abs/2407.01489).*

The upper stage starts with the directory structure, narrows to
files, and uses headers to choose classes or functions before it reads
the editing locations. The lower-right stage proposes several patches.
The lower-left stage uses tests to select one of them.

[Agentless v2, §3.3](https://arxiv.org/html/2407.01489v2#S3.SS3)
first runs generated reproduction tests against the original code and
keeps those that expose the failure. It then tests candidate patches
for regressions, retaining those with the fewest regression failures.
The reproduction test filters that group. In the parser example it
should accept `token=a=b` after the patch while keeping the old
behavior on single-separator assignments. If none of the remaining
patches passes the reproduction test, the method falls back to
regressions alone: the generated test may itself be wrong.
Finally it normalizes the surviving edits through their AST and
selects the most frequent normalized patch.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m05-p28-autocoderover-two-trajectories.png|AutoCodeRover uses separate trajectories for collecting context and producing the patch.]]

*Berkeley Advanced LLM Agents, lecture 5, slide 28; source:
[AutoCodeRover](https://arxiv.org/abs/2404.05427). Full slide.*

[AutoCodeRover, §§4.2–4.5](https://arxiv.org/html/2404.05427v1#S4.SS2)
allows iterations inside two phases with distinct contexts. It
searches by program structure: `search_class` can return signatures,
and `search_method` returns an implementation. In the authors'
`django-13230` example the problem is in the use of `add_item`, so
its definition alone is not enough. When tests exist, suspicious
methods can also be prioritized using coverage in failing tests.
The original setup permits up to three attempts for invalid patch
format or syntax; a well-formed patch still has to fix the bug.

[Passerine, §III](https://arxiv.org/html/2501.07531v1#S3) loops over five
commands: `cat`, `code_search`, `edit`, `bazel` and `finish`. The model
calls Google's internal tools instead of using a direct shell. Search
for historical issues is restricted to an earlier snapshot so that
the agent cannot read the eventual fix.

If a surprise failure requires a new diagnosis, free choice can
take the investigation back to an earlier file. A fixed pipeline
can be simpler to run, but it will not automatically revisit an
incorrect location unless the implementation supports that return.
Compare those choices with the model, budget and evaluator held fixed.

<span id="repairagent-управление-через-конечный-автомат"></span>

### RepairAgent: transitions between investigation and repair

[RepairAgent, §III](https://arxiv.org/html/2403.17134v2#S3) limits the
available tools according to the current state. In `Understand the bug`,
the model gathers evidence. `express_hypothesis` moves it to
`Collect information to fix the bug`. `write_fix` applies a patch,
runs tests and advances to `Try to fix the bug`. On failure the patch
is reverted. The model may use `collect_more_information` to inspect
a caller or `discard_hypothesis` to reconsider the diagnosis.
`Done` finishes the session.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/external-figure-review-2026-11-10/repairagent-figure2.svg|The RepairAgent state machine allows the model to return from a failed fix to more investigation.]]

*Islem Bouzenia, Premkumar Devanbu, Michael Pradel,
[RepairAgent v2, Figure 2](https://arxiv.org/html/2403.17134v2#S3.F2).
[Original SVG](https://arxiv.org/html/2403.17134v2/states_diagram2.drawio.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The middleware presents the current state, tools, accumulated
evidence and latest result. The state graph restricts which
transitions are legal; the model still chooses among them. The
reported configuration permits perfect fault localization by
default, which is a different setup from receiving only an issue
against an entire repository.

<span id="почему-бенчмарки-программирующих-агентов-быстро-устаревают"></span>

## What a passing patch proves

A SWE-bench `resolved` result means the submitted change passed the
tests attached to that task. If our test suite omits `token=a=b`, even
the original parser can pass. The repository snapshot and any
exposure to the eventual fix also affect what an evaluation measures.
The [[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|evaluation chapter]]
explains the `FAIL_TO_PASS` and `PASS_TO_PASS` checks.

<span id="ctf-состояние-специализированные-инструменты-и-обратная-связь"></span>

## Tools that keep their own state

<!-- source_unit_id: meeting-05-slides-ctf-agents -->
<!-- source_unit_id: meeting-05-reading-01-catalogue-record -->

A model call can finish while the debugger still holds a paused
process. Opening a new session loses the stack frame and breakpoint,
even if the conversation history says that execution stopped in
the parser.

<span id="один-сеанс-от-запуска-до-остановки"></span>

### What the debugger session holds

[EnIGMA, §2.1](https://proceedings.mlr.press/v267/abramovich25a.html)
extends the SWE-agent interface with interactive tools. Those tools
keep their own state separately from the main terminal. The authors'
configuration permits one interactive session at a time without
blocking the terminal.

Suppose the toy parser is paused just before the split. The model
can read the source in the editor and inspect `line` in the same
stack frame. Once the session ends, the old frame identifier is
invalid. A timeout does not tell the executor whether the process
is paused, running or gone. Check the session state before sending
another action.

<span id="что-модель-получает-вместо-длинного-вывода"></span>

### An output summary and the original log

Figure 3 of the paper compares three ways to return one command's
output. The interface can show all 170 lines, save them and show
the first 15 with 155 more below, or return a summary and a path to
the full text. In the second case the model can open the omitted
region; in the third it can use the summary to choose where to look.
Read the actual exception and line number in the original log.
The command is the same in all three cases.

In Table 3, EnIGMA with Claude 3.5 Sonnet solves 29.5% of 390 lab
CTF tasks with one attempt per task. Without both interactive tools
it solves 27.4%. Their effect differs by category: `pwn` drops from
20.45% to 11.36%, while `web` goes from 13.79% to 17.24%. The paired
removal does not isolate the debugger's contribution, and these
scores do not measure unknown defects in arbitrary software.

<span id="почему-генерация-патча-и-поиск-уязвимости-разные-задачи"></span>
<span id="почему-генерация-патча-и-поиск-уязвимости--разные-задачи"></span>
<span id="big-sleep-гипотеза-исполнение-sanitizer-воспроизводимый-crash"></span>
<span id="big-sleep-от-гипотезы-к-воспроизводимому-crash-через-исполнение"></span>

## What a program failure establishes

<!-- source_unit_id: meeting-05-slides-vulnerability-discovery-problem -->
<!-- source_unit_id: meeting-05-slides-charles-sutton-vulnerability-discovery-loop -->
<!-- source_unit_id: meeting-05-reading-02-catalogue-record -->

A bug report usually names a symptom or requirement: the parser
must keep the rest of the value. When triaging a suspected new
defect, first establish that the program violated its contract and
that the failure did not arise during setup. Record the program
revision, build configuration and observed failure. A missing
dependency can stop a test before the relevant function runs.

In the [Big Sleep report](https://projectzero.google/2024/10/from-naptime-to-big-sleep.html),
the starting clue was a SQLite change. The authors looked for a
variant of a known defect rather than auditing the repository from
scratch. The first confirming signal was an assertion in an
instrumented build. Behavior with assertions disabled was evaluated
separately. Berkeley lecture 5 discusses a different example
reported by AddressSanitizer. A build-specific crash, a memory error,
and demonstrated exploitability are different findings.

<span id="как-выбирать-степень-свободы-агента"></span>

## How much control to give the model

<!-- source_unit_id: meeting-05-slides-coding-agent-conclusions -->

A failing compiler or test can give fast feedback during a
reversible edit. An action with an irreversible side effect may
require approval before execution. Choose the scope of the agent
with those differences in mind.

| Mode | What the model and the program decide |
|---|---|
| One call | If the function is already known, the model writes a patch and the program applies it and runs tests. |
| Fixed stages | The code orders localization, editing and validation; the model supplies a result within each stage. |
| ACI loop | The model chooses the next repository command. The executor checks its syntax, permissions and limits. |
| Shell or IDE loop | The model can choose from a broader set of commands. A disposable environment limits access and resources, and the program tracks the budget. |

The commands available and the component choosing the next
command are separate design decisions.

<span id="visualwebarena-страница-как-частично-наблюдаемая-среда"></span>

## What a web agent can observe

<!-- source_unit_id: meeting-06-slides-visualwebarena -->
<!-- source_unit_id: meeting-06-reading-02-catalogue-record -->
<!-- source_unit_id: meeting-06-reading-03-catalogue-record -->

The browser and server keep page layout, cookies, tab state,
navigation history and application records. A screenshot misses
elements outside the viewport. An accessibility tree can omit
objects drawn on a canvas. An HTML snapshot can include invisible
elements while omitting a user's saved cart. The agent chooses
actions from the history it can observe, not from the full state.
This is a partially observable Markov decision process (POMDP).

[WebArena](https://arxiv.org/abs/2307.13854) supplies executable sites
and task checks. [VisualWebArena](https://jykoh.com/vwa) adds tasks
where the picture supplies information absent from the request text.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p18-visualwebarena-shopping.png|The shopping task gives the product type in text and the shipping address in the photograph of envelopes.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 18](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=18);
example from [VisualWebArena](https://jykoh.com/vwa).*

In the episode on slides 21–25, the agent opens the printer
category, compares the qualifying color photo printers, and adds
the cheapest to the cart. It then enters the address from the photo
and confirms the order in the test site. Three different failures
can produce the same final message: choose the wrong printer,
read the address incorrectly, or never finish checkout. The
grader checks the order and address, not the presence of a cart item.

Reset the sites and user state before each run. If a previous run
left a printer in the cart, the next agent no longer faces the
same task. Record the allowed actions and budget with the initial
state and final check.

<span id="html-accessibility-tree-screenshot-и-set-of-marks"></span>
<span id="как-представить-страницу-html-accessibility-tree-screenshot-и-set-of-marks"></span>

## Accessibility trees, screenshots and element labels

<!-- source_unit_id: meeting-06-slides-vision-language-web-agents -->
<!-- source_unit_id: meeting-06-reading-01-catalogue-record -->

If an instruction names a button by color or position, its text
label may be insufficient. HTML gives attributes but also includes
scripts, styles and hidden nodes. The accessibility tree represents
roles such as button or textbox, depending on markup quality.
A screenshot preserves visible layout without a symbolic element ID.
Set-of-Marks (SoM) overlays numbers on interactive elements. An action
can then refer to a number, as in `click(mark=17)`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p29-html-vs-som.png|The slide contrasts a textual representation on the left and numbered targets in a screenshot on the right.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 29](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=29).*

A label can cover a word or overlap another target. Page changes
may invalidate the target before the click. A DOM identifier can
also go stale, and an element found in HTML need not be visible
or clickable. Some systems pair the screenshot and accessibility tree
with shared identifiers.

<a id="web-agent-observation-bound-marks"></a>
<span id="метка-относится-к-тому-снимку-на-котором-её-выбрали"></span>

### A mark belongs to one observation

[VisualWebArena, §§3.1–3.2](https://arxiv.org/html/2401.13649v2#S3.SS1)
uses the element ID from the current observation as the argument to
`click [elem]`. Mark 17 may denote the shipping selector before a
redraw and the order confirmation button afterward. The same command
text would click a different target if resolved against the new table.

The example binds the action to the tab, document and observation
revision supplied with the model request. The executor must retain
that key when the model returns; it must not replace it with the key
of the newest frame.

~~~python
frame1 = (("shop-tab", "checkout", 1), {17: "shipping"})
frame2 = (("shop-tab", "checkout", 2),
          {17: "confirm_order", 23: "shipping"})
old_click = (frame1[0], 17)
fresh_click = (frame2[0], 23)
click_log = []

def dispatch_click(request, current, click):
    observed_key, mark = request
    current_key, marks = current
    if observed_key != current_key:
        return "stale_observation"
    if mark not in marks:
        return "unknown_mark"
    click(marks[mark])
    return "dispatched"

# A dispatcher that uses only the latest mark table targets another button.
assert frame2[1][old_click[1]] == "confirm_order"
assert dispatch_click(old_click, frame2, click_log.append) == "stale_observation"
assert dispatch_click((frame2[0], 99), frame2, click_log.append) == "unknown_mark"
assert click_log == []
assert dispatch_click(fresh_click, frame2, click_log.append) == "dispatched"
assert click_log == ["shipping"]
~~~

On `stale_observation`, request a new observation and choose again.
On `unknown_mark` the toy executor makes no click. A live browser
can also change after the key check, so inspect the actual target at
the point of execution or decline a click that cannot be tied to its
observation. Freshness does not authorize order confirmation.
`dispatched` records only a call to the executor. This fixture uses
a list rather than a browser and tests only that handoff.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p35-som-trajectory.png|The illustrated agent matches a user by the image and opens the profile before taking action.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 35](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=35);
VisualWebArena example.*

To compare observation formats, run the same tasks and model
with the accessibility tree, a marked screenshot, and both.
Record wrong targets and invalid commands alongside task success,
action count and context size.

<span id="mind2web-от-множества-элементов-к-одному-действию"></span>

### How Mind2Web reduces the candidates

[Mind2Web, §§2–3](https://arxiv.org/html/2306.06070v1#S2) records
human actions on real sites. A step names an element, an operation
and, when needed, a value. MindAct separates finding candidate
elements from choosing an action.

A small cross-encoder scores DOM candidates from the task,
preceding actions and features such as the candidate's text,
attributes and neighbors. During training the annotated element
is positive and randomly sampled other elements are negative.
The larger model sees the top candidates and short descriptions.
The multiple-choice variant groups five elements plus `None`
and can pass winners to another round.

For a request about clothing size, the small model has to keep
the size selector in the candidate set. The large model then
chooses the operation and value. If the correct element was
dropped, the second step cannot recover it. Measure candidate
recall separately from the correctness of the final action.

In [§4.2](https://arxiv.org/html/2306.06070v1#S4.SS2), offline
step evaluation supplies the recorded history of correct actions.
During an actual browser run the agent's own early errors would
change later observations. Matching steps from a demonstration
is not equivalent to finishing an executable WebArena or OSWorld task.

<span id="почему-реактивный-следующий-клик-ломается-на-длинном-горизонте"></span>
<span id="почему-ошибки-накапливаются-в-длинной-последовательности-действий"></span>

## Errors across a sequence of clicks

<!-- source_unit_id: meeting-06-slides-search-agent-motivation -->

An early click can select the wrong product variant or add a
filter that hides the required item. If all $H$ steps must be
correct and every conditional success probability given a correct
prefix is $p$, the probability of getting through all of them is
$p^H$. At $p=0.9$, ten steps give $0.9^{10}\approx0.35$.
Do not plug Mind2Web offline step accuracy into this formula to
predict end-to-end success. A real agent can recover from some
mistakes, and an early action changes the future observations.
Search lets the agent compare alternatives before continuing.

<span id="best-first-search-в-реальной-web-среде"></span>

## Searching with a return to another branch

<!-- source_unit_id: meeting-06-slides-web-agent-tree-search -->
<!-- source_unit_id: meeting-06-reading-04-catalogue-record -->

[Koh et al., §3.3](https://arxiv.org/html/2407.01476v1#S3.SS3)
store candidate paths in a priority queue. Remove a path, restore
the environment along it, and ask the value model to score the
resulting state. Track the best state already evaluated. Unless
the threshold or budget ends the search, propose more actions and
enqueue their paths.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p50-best-first-search.png|The action tree shows alternative continuations from a single browser observation.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 50](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=50);
[Koh et al.](https://arxiv.org/abs/2407.01476).*

An unvisited child inherits its parent's priority so that the
algorithm need not restore the browser and score every child as
soon as it is proposed. Queue priority is not that child's value.
A poor heuristic can still direct execution down a bad branch.

In [§5.4](https://arxiv.org/html/2407.01476v1#S5.SS4) the
authors reset the test site and replay actions from the beginning.
The back button cannot restore every field, scroll position or
server record. Replay depends on sufficiently repeatable behavior;
the history is not a database snapshot.

Each branch must start from the same parent. Adding a book and
adding a notebook to an empty cart should produce two carts, not
one containing both. In this numerical fixture the root is 10.
Adding 1 and doubling independently should yield 11 and 20.
Mutating one copy twice would give 11 and 22.

```python
from copy import deepcopy

root = {"id": "root", "snapshot": {"x": 10}}
actions = [("add_one", lambda x: x + 1), ("double", lambda x: 2 * x)]
children = []
for name, transition in actions:
    env = deepcopy(root["snapshot"])  # restore перед каждой альтернативой
    env["x"] = transition(env["x"])
    children.append({
        "parent": root["id"], "action": name,
        "observation": {"x": env["x"]}, "snapshot": deepcopy(env)
    })
assert [c["snapshot"]["x"] for c in children] == [11, 20]
assert all(c["parent"] == "root" for c in children)
assert root["snapshot"]["x"] == 10
print("branch isolation: PASS (11, 20)")
```

A VM snapshot or resettable test site can restore hidden state.
Duplicating a tab on an external site cannot undo a server write.
Confine exploratory branches to reads unless the environment
provides isolated copies or an explicit reversible operation.

In [§4.1](https://arxiv.org/html/2407.01476v1#S4.SS1) the value
model sees the task, screenshots, past actions and current URL.
It scores success, progress and failure as 1, 0.5 and 0; an
invalid response also scores 0. Average 20 such responses. Four
success judgments, twelve progress judgments and four failures give
$(4+12\cdot0.5)/20=0.5$. That is a heuristic derived from observations,
not a check of the site's stored state.

The reported setup uses depth 5, up to five continuations and a
budget of 20 evaluated states per search. Twenty sampled outputs
supply the most frequent candidate actions at an expansion. State
evaluations, action proposals and replay all contribute to cost.

<span id="учебный-поиск-с-возвратом-к-другой-ветви"></span>

### Following two branches in the toy environment

The first link from `start` leads to a promising advertisement
and then to a dead end. The second leads to a product and then
the answer. The transition table and scores are fixed; there is
no browser or model in the program.

~~~python
import heapq
from itertools import count

transitions = {
    "start": {"open_first": "advert", "open_second": "product"},
    "advert": {"follow_banner": "dead_end"},
    "product": {"read_details": "answer"},
    "dead_end": {}, "answer": {},
}
values = {"start": 0.5, "advert": 0.8, "dead_end": 0.0,
          "product": 0.6, "answer": 1.0}

def replay(path):
    state = "start"
    for action in path:
        state = transitions[state][action]
    return state

def best_first_fixture(budget=5, depth=2, threshold=0.95):
    if type(budget) is not int or budget < 0 or type(depth) is not int or depth < 0:
        raise ValueError("budget and depth must be nonnegative integers")
    serial = count()
    frontier = [(0.0, next(serial), ())]
    best_path, best_value, visited = None, float("-inf"), []
    reason = "budget"
    while frontier and len(visited) < budget:
        _, _, path = heapq.heappop(frontier)
        state = replay(path)  # reset to start and replay this branch
        value = values[state]
        visited.append((state, value))
        if value > best_value:
            best_path, best_value = path, value
        if value >= threshold:
            reason = "threshold"
            break
        if len(path) < depth:
            for action in transitions[state]:
                heapq.heappush(frontier, (-value, next(serial), path + (action,)))
    else:
        reason = "frontier_empty" if not frontier else "budget"
    return {"path": best_path,
            "state": replay(best_path) if best_path is not None else None,
            "value": best_value if best_path is not None else None,
            "visited": visited, "reason": reason}

result = best_first_fixture()
assert [state for state, _ in result["visited"]] == [
    "start", "advert", "dead_end", "product", "answer"]
assert result["path"] == ("open_second", "read_details")
assert result["state"] == "answer"
assert best_first_fixture(budget=2)["state"] == "advert"
~~~

Both links start with priority 0.5. The first link is visited and
gets 0.8, so its child takes priority over the product link. The
child scores zero. The algorithm returns to the other branch and
finds the answer. A budget of two leaves the advertisement at the
top of the evaluated states. Lowering the threshold to 0.7 would
also stop there. Use a separate outcome check before calling the
best candidate a success; compare search with repeated independent
attempts at the same full cost.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p60-search-backtracking.png|The first trajectory continues a bad branch; the second returns to the decision point and tries another.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 60](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=60).*

<span id="когда-дополнительный-поиск-окупается"></span>

## What search buys and what it costs

<!-- source_unit_id: meeting-06-slides-web-agent-tree-search-results -->

In [Koh et al., Table 2](https://arxiv.org/html/2407.01476v1#S4.SS2),
GPT-4o on VisualWebArena goes from 18.9% to 26.4% with search. The
increase is 7.5 percentage points, or about 39.7% relative. The
paragraph next to the v1 table says 18.6%, which is inconsistent
with both the table and the stated relative gain. The calculation
here uses the table. The lecture slide rounds the two rates.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p70-search-results.png|The lecture compares task success with and without search on WebArena and VisualWebArena.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 70](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=70);
exact values in [Koh et al.](https://arxiv.org/abs/2407.01476).*

Report the extra model calls, actions and resets with task success.
If an action is costly or irreversible, compare predicted outcomes
before execution. A resettable environment can allow direct trials of
the alternatives.

<span id="insta-синтетические-задачи-из-работающего-web"></span>

## How InSTA collects examples from websites

<!-- source_unit_id: meeting-06-slides-insta-setup -->

[InSTA v1](https://arxiv.org/abs/2502.06776v1) starts with a domain.
One model proposes a task, an agent attempts it, and a judge
reviews the recorded attempt. The authors start from the top
million Common Crawl PageRank domains. A model filters out about
85%, leaving roughly 150,000 tasks. That filter sees the instruction
and domain URL, not every page the agent will later visit.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p80-insta-pipeline.png|InSTA proposes a task for a domain, records an attempt, then checks it before turning it into training data.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 80](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=80);
[Trabucco et al.](https://arxiv.org/abs/2502.06776).*

[Appendix A](https://arxiv.org/html/2502.06776v1#A1) describes a
budget of one task per site and at most ten actions. Instructions
exclude purchases, comments and account creation. Those instructions
shape the predominantly information-seeking data, but they are not
execution-time permission checks. These demonstrations cannot establish
that an agent handles checkout or account changes well.

Keep the task, collection time, observations, actions, model versions
and judge decision for each trial. The reasons for rejection help
diagnose which attempts the filter excludes from training.

<span id="верификатор-как-фильтр-синтетической-траектории"></span>

## How InSTA selects successful trajectories

<!-- source_unit_id: meeting-06-slides-insta-verification-scaling -->

In [§5.1](https://arxiv.org/html/2502.06776v1#S5.SS1), the evaluator
sees task $c$, action history $a_{1:T}$ and final text observation
$o_T$. A parser extracts a number from its response:

$$
\widehat r_T=\operatorname{parse}\!\left(
\operatorname{LLM}(c,a_{1:T},o_T)\right).
$$

A score is not calibrated by being written as a number. For
comparison with human ratings the authors treat $\widehat r_T>0.5$
as success. Confidence is $2|\widehat r_T-0.5|$. At 0.9 the
confidence is 0.8, while at either 0 or 1 it is maximal. Selection
can therefore require both success and high confidence.

Slide 91 reports 14.6% confident successes, about 22,000 of 150,000.
[InSTA v1, §5.2](https://arxiv.org/html/2502.06776v1#S5.SS2) reports
16.7%. The two reported numbers need not represent the same run.
Both are estimates from a model judge, not a manual recount of
all tasks.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m06-p90-automatic-verifier.png|The automatic verifier receives the history and final observation rather than the site's hidden state.]]

*Ruslan Salakhutdinov, [Berkeley lecture 6, slide 90](https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf#page=90);
InSTA.*

Human comparison separates the filter, task generator and judge.
On 100 domains Gemini 1.5 Pro gives 97% accuracy for the site
filter. Humans rate 89% of its proposed tasks as feasible. GPT-4o
matches 82.6% of humans' trajectory labels. The large collection
instead uses Llama 3.1 70B; the corresponding reported figures are
85%, 75% and 81.7%. The best result for one component should not be
quoted as the result of the deployed combination.

On the checked subset of Llama's most confident decisions, the
authors report 93.1% accuracy, precision 0.87 and recall 0.82.
False acceptance still puts failed attempts into a dataset of
successes. Inspect that error separately instead of relying on
accuracy alone.

<span id="что-попадает-в-обучение-insta"></span>

### What goes into the training set

InSTA requires confident success and a positive judgment that the
actions contributed to the result. It excludes traces shorter than
three actions and traces with server errors, CAPTCHAs or access
blocks. [Section 6](https://arxiv.org/html/2502.06776v1#S6) holds out
500 of 7,463 selected demonstrations and mixes the remaining
6,963 with human data.

The proportion of human examples is chosen when sampling batches.
It is 0.5 in runs with little human data and 0.8 in the other
configuration. Adding more synthetic trajectories broadens the
available sites and actions but does not itself set the source
sampling rate.

With 32 human actions, the authors report relative increases in
Step Accuracy of 89.5% on Mind2Web and 122.1% on WebLINX. Step
Accuracy checks the element and action, not completion of the
whole task. With more human data the official test results are
similar. The larger gains on held-out InSTA sites show transfer
within that collection process. Neither an offline step score nor
a model-filtered holdout establishes end-to-end success on a live site.

<span id="osworld-resettable-vm-executable-action-and-final-state-grader"></span>
<span id="osworld-восстанавливаемая-vm-исполняемые-действия-и-проверка-результата"></span>

## How OSWorld checks the saved result

<!-- source_unit_id: meeting-07-slides-osworld -->
<!-- source_unit_id: meeting-07-reading-01-catalogue-record -->

[OSWorld](https://arxiv.org/abs/2404.07972) contains 369 tasks across
applications and the operating system. Each task specifies a starting
VM configuration and a result check. The model can observe the screen,
accessibility tree or terminal output, depending on the setup. It
acts through the mouse, keyboard or supported programmatic commands.
The evaluator inspects the environment after the attempt.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/computer-agents-2026-11-09/osworld-task-environment.png|The task configuration initializes the virtual machine; the agent interacts with it and the grader retrieves the resulting artifact.]]

*Tianbao Xie et al., [OSWorld v1, Figure 2](https://arxiv.org/html/2404.07972v1#S2.F2).
[Original PNG](https://arxiv.org/html/2404.07972v1/arch_figure_w_config_new.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Caiming Xiong, Berkeley lecture 7, slide 11](https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf#page=11).*

The task on the left updates an accounting spreadsheet. `Set-up`
opens the workbook. After the agent stops, `Postprocess` activates
the Calc window, `Getter` retrieves a file from the VM, and
`Metrics` compares ranges with a reference workbook. The screenshot
helps the agent edit; the check reads the file. Changes left only in
the application's memory do not count in this comparison.

Different click sequences can produce the same valid file. Preserve
the action log anyway: it shows repetitions or unauthorized actions
that the workbook cannot reveal. [Table 1](https://arxiv.org/html/2404.07972v1#S2.SS2.SSS3)
includes another task in which the grader checks the actual cookies
for `.amazon.com`. A closed settings window cannot confirm deletion.

Reset the VM and apply the task setup before each attempt. Otherwise
the second run could start with an already edited workbook. Keep the
grader's postprocessing and retrieval separate from the agent's
actions. The original scoring setup can award partial credit.
The [[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем#agent-eval-coverage|evaluation chapter]]
separates environment setup failures from runs whose effects
cannot be verified.

<span id="как-наблюдение-и-история-влияют-на-результат-osworld"></span>

### Observation format and history in OSWorld

[Section 4.1](https://arxiv.org/html/2404.07972v1#S4.SS1) compares an
accessibility tree, a screenshot, their combination and SoM. The
tree can include element coordinates. With SoM the model chooses
an ID and the executor maps it to coordinates. This setup uses a
history of three observations and actions. [Appendix C.1](https://arxiv.org/html/2404.07972v1#A3.SS1)
gives the baselines 15 steps and 30 minutes. The general loop in
§2.1 uses 20 steps as an example; match the limit to the experiment
when reproducing a number.

In [§5.2](https://arxiv.org/html/2404.07972v1#S5.SS2), higher screenshot
resolution usually helps the screenshot-only agent on the chosen
subset. SoM improves at 768×432 and then worsens at smaller sizes.
The screenshot baseline still predicts coordinates in the
1920×1080 system. Resizing changes vision and the mapping from
the input image to the executable coordinates.

More history helped SoM, but adding earlier screenshots did not
give the same improvement in the visual baseline. Changing window
positions and adding other applications reduced performance on
28 selected tasks. The agent might switch windows without making
the required controls visible. Test that behavior when transferring
to a different desktop layout.

<span id="agenttrek-tutorials-guided-replay-trajectories"></span>
<span id="agenttrek-от-руководств-к-траекториям-через-управляемое-воспроизведение"></span>

## How AgentTrek turns tutorials into demonstrations

<!-- source_unit_id: meeting-07-slides-agenttrek -->

[AgentTrek](https://arxiv.org/abs/2412.09605) finds a written tutorial,
extracts the goal and steps, and records a multimodal agent's
attempt to follow it. Another model judges the recorded attempt.

[Section 2.1](https://arxiv.org/html/2412.09605v2#S2.SS1) filters
documents using action verbs, interface terms and length. GPT-4o mini
labels some candidates; FastText learns to filter the larger corpus.
The remaining text is standardized into a platform, prerequisites,
instructions and expected outcome. A tutorial still contains no
observation of the current browser.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/external-figure-review-2026-11-10/agenttrek-figure4.png|AgentTrek's text filtering leaves 68.8M entries after prefiltering, 54M after FastText and 18.8M after deduplication.]]

*Yiheng Xu et al., [AgentTrek v2, Figure 4](https://arxiv.org/html/2412.09605v2#S2.F4).
[Original PNG](https://arxiv.org/html/2412.09605v2/images/data_flow_v6.png),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).*

The counts along the top of Figure 4 are text records. The
`Web 2.24M` label on the right still belongs to tutorial selection.
The later browser runs determine whether the instructions can be
executed on the current site.

[Figure 5](https://arxiv.org/html/2412.09605v2#S2.F5) shows actions
sent to the site and observations returned to the agent. In the
[return-policy example, Figure 6](https://arxiv.org/html/2412.09605v2#S2.F6),
the agent must find the rule for the requested product category and
answer from it. Opening the returns page is insufficient. The agent
gets screenshots and an accessibility tree and runs Playwright actions.
The collector also saves DOM, network logs and actions; its record
contains more than the model's input. Keep the tutorial alongside
the trace to see where the agent needed to adapt the instructions.
The authors report 84% agreement with human evaluation, not a
guarantee that every kept trace succeeded.

[Section 2.3](https://arxiv.org/html/2412.09605v2#S2.SS3) produces
training data for text and visual agents. The former get an
accessibility tree and Playwright actions; the latter get an
image and pixel-based commands. Mapping the recorded actions
requires identifying the equivalent target on the screenshot.
A dropdown selection may require a distinct action rather than
a simple renaming of an API call.

<span id="taco-совместно-обучать-reasoning-и-action-tokens"></span>
<span id="taco-совместное-обучение-токенов-рассуждения-и-действий"></span>

## Which tokens TACO learns to predict

<!-- source_unit_id: meeting-07-slides-taco -->

[TACO v1](https://arxiv.org/abs/2412.05479v1) interleaves
generated text and executed actions in a Chain-of-Thought-and-Action
(CoTA). The model can request OCR, object localization, a crop,
depth estimation, retrieval or a calculation. The executor returns
a result before the model continues. A demonstration records when
to call the tool, its arguments and how the result informs the answer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/coding-web-computer-use-2026-10/taco-cota-examples.png|Four CoTA examples show OCR, object search, calculation and retrieval before an answer.]]

*Zixian Ma et al., [TACO v1, Figure 1, p. 2](https://arxiv.org/pdf/2412.05479v1#page=2);
original figure and caption, CC BY 4.0.*

In the top-left example the model calls `OCR(image-0)` to find
the price of tomatoes. OCR returns text from the price labels.
The model matches the product to `8,00` and answers `8.00`.
A fluent answer cannot repair the price if OCR read it incorrectly.

In [§2](https://arxiv.org/html/2412.05479v1#S2), a CoTA step is
text $t_i$, action $a_i$, then observation $o_i$. The loss includes
the generated text and action; the tool output is supplied as
context rather than as the model's target. When the model chooses
OCR it has not yet seen the result. The result becomes available
before it predicts the final answer. A loss mask does not stop
the model from inventing an observation at inference time; the
executor must distinguish its own output from generated text.

[Sections 2.1–2.2](https://arxiv.org/html/2412.05479v1#S2.SS1)
describe two data sources. GPT-4o solves existing visual questions
with available tools; the pipeline checks the answer and parses
the actions as JSON. A wrong chain can be replaced by a direct
ground-truth answer instead of labeled a successful CoTA. Programs
also build template examples from labeled objects, relations and
depth estimates. This increases examples for infrequent tools,
but the resulting distributions need separate evaluation.

The paper's *action-useless* label applies to a dataset, not to
each call in it. A source qualifies if the share of correct CoT
without actions exceeds correct CoTA by more than 10 percentage
points, or if wrong CoTA exceeds correct CoTA by that amount.
For example, 50% correct CoT and 35% correct CoTA meet the first
condition. A dataset count does not identify the reason for an
individual error.

The main recipe uses 293,000 correct GPT-4o CoTA examples across
all 31 sources. The paper compares other formats and mixes rather
than finding that every dataset benefits from filtering or more
programmatic examples. Check the final answer, actions, arguments
and tool outputs separately. Recorded text explaining an action
is a training target, not necessarily evidence of what caused
the model to select the action.

<span id="aguvis-единое-визуальное-observationaction-space"></span>
<span id="aguvis-единое-визуальное-пространство-наблюдений-и-действий"></span>

## How Aguvis learns where to click

<!-- source_unit_id: meeting-07-slides-aguvis -->
<!-- source_unit_id: meeting-07-reading-02-catalogue-record -->

A browser can identify a button by DOM ID; a desktop application
may expose an accessibility node instead. [Aguvis](https://arxiv.org/abs/2412.04454)
uses a screenshot as a common observation and predicts actions
grounded in it. Clicks, text entry and key presses can apply
across platforms. Some platforms need further commands.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m07-p70-aguvis-observation-action.png|Aguvis sees a route form, selects the To field, then proposes a click and the text North Station.]]

*Caiming Xiong, [Berkeley lecture 7, slide 70](https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf#page=70);
diagram from Xu et al., [Aguvis](https://arxiv.org/abs/2412.04454).*

The Boston Logan departure field is filled in. The model identifies
the destination as the next subtask, describes it and proposes
commands for the `To` field. Choosing the right subtask does not
ensure that the coordinates land in the right field. After an
action, a fresh screenshot is needed if the layout might change.

If a 1440×900 screenshot becomes 960×600 without cropping, a button
centered at `(1170, 210)` moves to `(780, 140)` in the resized image.
The executor must invert the scale if it accepts original-image
coordinates. Cropping or a window offset adds a translation.
This example is geometry, not a claim about Aguvis's normalization.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m07-p72-aguvis-two-stage-training.png|Stage 1 learns to locate a specified control. Stage 2 learns to choose what to do given the task and past actions.]]

*Caiming Xiong, [Berkeley lecture 7, slide 72](https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf#page=72);
Xu et al., Aguvis.*

In stage 1 an instruction such as “double-click Best Software”
specifies what to do; the model must find where. In stage 2 the
model must choose the next action from the goal, current screen
and history. Demonstrations can include text before the command,
which the authors call an *inner monologue*.

[Section 2.5](https://arxiv.org/html/2412.04454v1#S2.SS5) packs several
instruction–action pairs for the same screenshot into one training
example. The second stage mixes instructions with fuller descriptions,
rationales and commands. Both stages train on demonstrations, without
a reward update after each live attempted GUI task.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m07-p73-aguvis-data-collection.png|Element annotations yield instructions and coordinates; trajectory annotations supply actions that receive additional rationales.]]

*Caiming Xiong, [Berkeley lecture 7, slide 73](https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf#page=73);
Xu et al., [Aguvis](https://arxiv.org/abs/2412.04454).*

A screenshot and labeled element suffice to generate a grounding
example. A planning example needs a sequence of observations and
actions. GPT-4o sees an already annotated action and supplies a short
instruction and rationale. That text is synthetic annotation rather
than a record of the original operator's reasoning.

Stage 1 uses over a million grounding examples. Stage 2 includes
about 35,000 multi-step trajectories with rationales. The slide
separates grounding data and trajectories by platform.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m07-p75-aguvis-data-stages.png|The slide groups grounding and planning data for Aguvis by stage and platform.]]

*Caiming Xiong, [Berkeley lecture 7, slide 75](https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf#page=75);
Xu et al., Aguvis.*

The ablation varies whether the training includes stage 1, stage 2
and the rationales. Changes differ across ScreenSpot,
Multimodal-Mind2Web and AndroidControl. The reported effect of
rationales applies to the authors' model and data, not to every
GUI agent.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-computer-use/m07-p80-aguvis-ablation.png|Ablation results compare variants with training stages or rationales omitted.]]

*Caiming Xiong, [Berkeley lecture 7, slide 80](https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf#page=80);
Xu et al., Aguvis.*

<span id="что-ограничивает-web-agents-perception-value-state-restoration-and-safety"></span>
<span id="семь-ограничений-web-агента-от-восприятия-до-безопасности"></span>

## Finding the cause of a failed web task

<!-- source_unit_id: meeting-06-slides-web-agent-synthesis -->

Read the first incorrect observation or action in the trace:

- The needed element was absent or misread.
- The agent chose the right operation but mapped it to the wrong target.
- Each action was legal on its own, yet the sequence missed the goal.
- Search expanded a dead end because the evaluator rated it too highly.
- Replay restored the message history but left the application changed.
- The final check accepted a claim without finding the expected effect.
- The executor accepted an action that was outside its permission scope.

More search does not repair a consistent grounding error. Replacing
the grader does not restore a cart. Compare observation formats,
search, and result checks one at a time to establish which
component affected outcomes. Count the relevant failure types as
well as successful tasks.

Page text can include instructions the user did not authorize.
Account changes, payments and message sends require checks by
the executor. See [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|agent-system security]]
for permission scope and threat modeling.

<span id="минимальный-протокол-эксперимента"></span>

## What to record when comparing agents

Record the exact tasks and environment revision, start snapshot and
reset procedure, observation format, actions, model and decoding
settings, and limits. Preserve permission and approval rules,
stopping criteria and the outcome and policy checks. Keep the
trace, resulting files and cost of every run, along with repeat
counts and a task-level uncertainty estimate.

Run both systems under the same protocol. Alongside success, report
invalid actions, resets, wrong targets, false decisions by the grader,
cost and steps. A low step count means little if the work is
declared complete before the artifact is saved.

<span id="источники-и-что-брать-из-каждого"></span>

## Sources

- [Berkeley Advanced LLM Agents, Spring 2025](https://rdi.berkeley.edu/adv-llm-agents/sp25), lectures 5–7. The captions link to the used slides or the original paper figures.
- [SWE-agent](https://arxiv.org/abs/2405.15793), [Agentless](https://arxiv.org/abs/2407.01489), [AutoCodeRover](https://arxiv.org/abs/2404.05427), [Passerine](https://arxiv.org/abs/2501.07531) and [RepairAgent](https://arxiv.org/abs/2403.17134v2) compare interfaces and choices of control flow.
- [EnIGMA](https://proceedings.mlr.press/v267/abramovich25a.html) and [Big Sleep](https://projectzero.google/2024/10/from-naptime-to-big-sleep.html) discuss persistent tool sessions and evaluating reported program failures.
- [Mind2Web](https://arxiv.org/abs/2306.06070), [WebArena](https://arxiv.org/abs/2307.13854), [VisualWebArena](https://jykoh.com/vwa) and [Tree Search for Language Model Agents](https://arxiv.org/abs/2407.01476) cover recorded browser steps, executable sites, visual observations and search.
- [InSTA](https://arxiv.org/abs/2502.06776v1) generates and checks synthetic web tasks.
- [OSWorld](https://arxiv.org/abs/2404.07972), [AgentTrek](https://arxiv.org/abs/2412.09605), [TACO v1](https://arxiv.org/abs/2412.05479v1) and [Aguvis](https://arxiv.org/abs/2412.04454) cover VM-based evaluation and demonstration-based training. Later v4 of the TACO arXiv record is titled LATTE and studies a revised setup.

<span id="связанные-главы"></span>

## Related chapters

Previous: [[en/00 Textbook/17 Tools and Agents/68 Evaluating agent systems|Evaluating agent systems]].

Next: [[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Formal proofs and mathematical agents]].
