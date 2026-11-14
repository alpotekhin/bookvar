---
title: "ToolSandbox: evaluating tool calls against changing state"
type: source-note
status: canonical
last_verified: 2026-11-10
source_url: https://arxiv.org/abs/2408.04682v2
authors: [Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, Ruoming Pang]
year: 2025
source_language: en
---

# ToolSandbox: evaluating tool calls against changing state

[Paper, arXiv v2](https://arxiv.org/html/2408.04682v2) ·
[NAACL 2025 record](https://aclanthology.org/2025.findings-naacl.65/) ·
[Reference implementation](https://github.com/apple-aiml-research/ToolSandbox).

[Figure 2 and §2](https://arxiv.org/html/2408.04682v2#S2) separate
the user, the agent, and the execution environment. A tool can change
the environment, and its result can change what the agent does next.
The evaluator has access to both the message history and state snapshots.

[Figure 1](https://arxiv.org/html/2408.04682v2#S1.F1) follows an agent
trying to send a message with cellular service disabled. The failure
in the transcript is evidence about the attempt, not proof that a message
was sent. The right-hand column shows what the evaluator looks for,
including an entry in the message database.
[Section 2.3](https://arxiv.org/html/2408.04682v2#S2.SS3) explains how
milestones can constrain the order of events without requiring an exact
match to one reference conversation.

Bookvar follows this example in
[[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем#agent-eval-toolsandbox|Evaluating agent systems]].
Both figures are copied unchanged from arXiv v2 under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
