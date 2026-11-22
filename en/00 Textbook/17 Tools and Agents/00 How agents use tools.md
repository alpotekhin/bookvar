---
title: How agents use tools
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля.md"
canonical_revision: "ac221121a2c28509b02705d4550ad9d5c8e73bdd699d87e385ff0f55a4df12ef"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/16 Multimodal Models/64h Chameleon - text and images in one sequence]]"
next: "[[00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию]]"
primary_sources:
  - https://www.anthropic.com/research/building-effective-agents
  - https://arxiv.org/html/2405.15793v3
  - https://arxiv.org/html/2408.04682v2
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
---

# How agents use tools

“Find a branch with two copies of this book available. Check North Library
first. Don't reserve anything.” The answer needs information from the
catalogue. The model can choose an action and supply its arguments. The
application queries the catalogue and returns the result to the model. What
the model asks for next depends on that result.

## From a request to an observation

Suppose the application has already identified the edition. It gives the
model the edition ID and a tool for checking one branch. One run could go
like this:

| What the model asks for | What the application does |
|---|---|
| Check North Library. | The application validates the arguments and queries the catalogue. It finds one copy available. |
| Check Central Library. | Given the first result, the model moves on to the second branch. The catalogue shows two copies available as of 14:05. |
| Respond and stop. | The application returns: “Central Library had two copies available at 14:05. I didn't reserve them.” |

The count comes from the catalogue and applies to one edition, one branch,
and the time of the check. One copy at each branch would not meet the
request. If the second check times out, Central Library's count is unknown.
The application can allow a retry within its limit, then report which check
it couldn't complete.

SWE-agent follows a similar exchange with a software repository. The top
arrow carries commands to search, view, or edit files through an interface
to the computer. The bottom arrow carries the resulting observation back.
Opening a file, for example, can return an excerpt with line numbers that
the model uses to choose its next action.

![[00 Учебник/Assets/Figures/curated/computer-agents-2026-11-09/swe-agent-overview.png|SWE-agent actions go through the agent-computer interface to the computer. Feedback goes back through the interface to the model.]]

*John Yang et al., [SWE-agent, Figure 1](https://arxiv.org/html/2405.15793v3#S1.F1),
[original PNG](https://arxiv.org/html/2405.15793v3/swe_agent_overview.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The interface defines commands and formats observations. The runtime also
enforces which actions the agent may execute.*

## When the model chooses the next step

For two known branches, a program could query both counts, check for at
least two copies at either branch, and format the answer. If a language
model is useful for phrasing, it can get the verified counts in one call.

[Anthropic's Building effective agents](https://www.anthropic.com/research/building-effective-agents#what-are-agents)
distinguishes *workflows*, whose control flow is defined in code, from
agents, where the model chooses actions dynamically. “If North has fewer
than two copies, check Central” is still easy to express in code. Model
control is more useful when the system must choose between ambiguous
editions, ask the reader about acceptable branches, or decide whether a
search result justifies another call. The path and number of steps can
then vary between runs.

The application sets permissions in either design. This request needs
read access to the catalogue, so there is no reason to expose a reservation
tool. A well-formed request meets a schema; its arguments don't establish
user authorization. The application also needs stopping rules for a
sufficient answer or an exhausted call budget.

## What to verify after the run

A trace records calls and their observations; the state of the environment
is separate. To pass the library task, name a branch with two copies of the
same edition and leave the reservations unchanged. A test catalogue with
counts of one and two lets you check both the answer “Central Library”
and the reservation records. A second case can make the Central lookup
time out and check that the answer reports an unknown count, not zero.

[ToolSandbox, §2 and Figure 1](https://arxiv.org/html/2408.04682v2#S2)
shows the distinction on a messaging task. The history contains the
requests and observations, separate snapshots track world state, and
milestones test selected events. A record in the message database
answers a different question from the model's claim that it sent the
message.

The same distinction applies to other environments. A browser control
belongs to a [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents#web-agent-observation-bound-marks|specific observation]];
an action requires checking the new page state. In Lean, you check the
[[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты#lean-proof-acceptance|proof of the original statement and the axioms it relies on]].
A natural-language report can't stand in for either check.

<span id="линейный-маршрут"></span>

## Reading order

| Chapter | Main question |
|---|---|
| [[00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|Tool use]] | How does the application receive arguments and return a result? |
| [[00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|Execution and context]] | Which data goes into each model request, and when does the loop stop? |
| [[00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Memory and planning]] | How do you retain results, pick a next step, and resume interrupted work? |
| [[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|Evaluation]] | How do you set the initial state, check the result, and diagnose failures? |
| [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Code, browser, and desktop]] | What can an agent read or change in different environments? |
| [[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Formal mathematics]] | How do you find a proof and check it in Lean? |
| [[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Scientific discovery]] | How do you test a hypothesis with computation or experiment? |
| [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Security]] | How do you restrict actions and keep external data from becoming instructions? |

Set permissions before connecting to real services. The
[[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|security chapter]]
can be read on its own.

<span id="практический-корпус"></span>

## Sources

The [[05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index|Berkeley Advanced LLM Agents archive]]
contains source slides and maps lectures to the chapters on planning,
code, mathematics, and security. Training a model to reason is covered
separately in [[00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR and verifiers]],
with links to Berkeley and
[[05 Источники/Courses/Stanford CS336 Spring 2026/_index|Stanford CS336]].
The [[05 Источники/Courses/GenAI Agents|GenAI Agents catalogue]]
keeps the original notebooks and saved outputs.

Next: [[00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|how a function call becomes an action]].
