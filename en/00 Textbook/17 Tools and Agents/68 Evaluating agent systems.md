---
title: Evaluating agent systems
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/68 Оценивание агентных систем.md"
canonical_revision: "f3c8c482432746c88910bfb8e6ddac96f90de3e11b23e46dbd1a6d3c30328e62"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/17 Tools and Agents/67 Agent memory, planning, and orchestration]]"
next: "[[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents]]"
reference_terms:
  - label: Agent evaluation — state and side effects
    label_en: Agent evaluation — state and side effects
    anchor: agent-eval-state-side-effects
  - label: AppWorld — checking state changes
    label_en: AppWorld — checking state changes
    anchor: agent-eval-appworld
  - label: Agent evaluation — coverage and unscored runs
    label_en: Agent evaluation — coverage and unscored runs
    anchor: agent-eval-coverage
  - label: pass@k and pass^k — success in any or all trials
    label_en: pass@k and pass^k — success in any or all trials
    anchor: agent-eval-pass-k
  - label: Paired agent evaluation — task-level resampling
    label_en: Paired agent evaluation — task-level resampling
    anchor: agent-eval-paired
  - label: LLM-as-a-judge — rubric and calibration
    label_en: LLM-as-a-judge — rubric and calibration
    anchor: agent-eval-judge
primary_sources:
  - https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
  - https://arxiv.org/abs/2407.18901v1
  - https://arxiv.org/abs/2408.04682v2
  - https://inspect.aisi.org.uk/scoring-policy.html
  - https://arxiv.org/abs/2308.03688
  - https://arxiv.org/abs/2310.06770
  - https://arxiv.org/abs/2406.12045
  - https://arxiv.org/abs/2504.01848
  - https://arxiv.org/abs/2107.03374
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
source_unit_id:
  - meeting-05-slides-coding-agent-evaluation
---

# Evaluating agent systems

In the [[en/00 Textbook/17 Tools and Agents/67 Agent memory, planning, and orchestration#The seminar moves to Friday|previous chapter]],
the agent moved a seminar from Thursday to Friday. Suppose it reports
“Moved the booking and sent the invitation.” The calendar does contain
a Friday entry, but Thursday's booking is still there. The mail log
shows two sends. Grading the answer or checking only the Friday entry
would miss both mistakes.

An evaluation distinguishes the intended change, unexpected actions
and missing evidence. Each trial tests a model, its instructions,
tools and access rules, and the starting environment together. To
measure the effect of context compaction, change that component and
hold the rest fixed.

<span id="1-сначала-определяется-успешное-состояние"></span>

## What counts as completing the task

To pass the seminar task, replace the old booking, send one invitation
and follow the authorization policy. The calendar snapshot can confirm
only the replacement. The other two require a mail log and a permission
record. A task without sending authorization can instead require the
agent to ask and leave the mail unsent.

An outcome grader inspects the result in a database, a file or an
independent service. The agent's statement cannot verify its own action.
A successful create call is not sufficient either: a later call could
delete the new booking. Inspect the final state and, for effects that
can't be undone, the execution log.

Answering a question about the time may require only a text check.
Moving a booking takes actions whose effects change the next tool result.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-11-10/toolsandbox-roles.png|The user and agent exchange messages on the left. On the right, the agent calls tools, changes the environment and receives observations.]]

*Jiarui Lu et al., [ToolSandbox v2, Figure 2](https://arxiv.org/html/2408.04682v2#S2.F2),
[original PNG](https://arxiv.org/html/2408.04682v2/architecture_diagram.png),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
One user turn can lead to several interactions with the environment.*

The [Anthropic evaluation diagram](https://www-cdn.anthropic.com/images/4zrzovbb/website/bd42e7b2f3e9bb5218142796d3ede4816588dec0-4584x2834.png)
makes the distinction for a coding agent: test the resulting server
rather than grading only the final reply.

<a id="agent-eval-state-side-effects"></a>
<span id="перенос-семинара-что-должно-измениться"></span>

### Moving the seminar: what should change

Seminar 42 is at 18:00 on Thursday, October 8. The task allows moving
it to 19:00 on Friday, October 9, in the same room 204 (UTC+3) and
sending one invitation. Attendance doesn't change. Other bookings
must stay as they were. Asking for confirmation without sending
would be a different task with a different success condition.

The grader reads the before and after calendar snapshots, the mail
service's send log and the permission audit. It expects all the
same records except for replacing Thursday with Friday. Row order
doesn't matter, but duplicates do, so the fixture uses a `Counter`.
The mail log must contain one send with the expected fields.
Deleting a calendar record cannot recall a message already sent.

<details id="agent-eval-code-booking">
<summary>Python: Grade the calendar, mail log and permission audit</summary>

~~~python
from collections import Counter

thursday = "2026-10-08T18:00:00+03:00"
friday = "2026-10-09T19:00:00+03:00"
old = ("seminar-42", "204", thursday, "confirmed")
new = ("seminar-42", "204", friday, "confirmed")
other = ("seminar-43", "205", friday, "confirmed")
recipient = "attendee@example.org"
invite = (recipient, "seminar-42", "204", friday)

calendar_write = ("calendar-write", "seminar-42")
mail_send = ("mail-send", recipient)
required_actions = {calendar_write, mail_send}

def grade_booking(before, after, sent, audit):
    # Fixtures and evidence are supplied by the evaluator, not by the agent.
    original = Counter(before)
    if original[old] != 1 or sum(
        count for row, count in original.items() if row[0] == "seminar-42"
    ) != 1:
        raise ValueError("invalid starting snapshot")
    expected = original - Counter([old]) + Counter([new])

    verdicts = {
        "calendar": None if after is None else Counter(after) == expected,
        "mail": None if sent is None else list(sent) == [invite],
        "policy": None,
    }
    if audit is not None:
        observed = {(kind, target) for kind, target, _ in audit}
        denied = any(not allowed for _, _, allowed in audit)
        if denied or observed - required_actions:
            verdicts["policy"] = False
        elif required_actions <= observed:
            verdicts["policy"] = True

    if False in verdicts.values():
        return "incorrect", verdicts
    if None in verdicts.values():
        return "unscored", verdicts
    return "correct", verdicts

before = [old, other]
after = [other, new]
audit = [(*calendar_write, True), (*mail_send, True)]
assert grade_booking(before, after, [invite], audit)[0] == "correct"
assert grade_booking(before, after, [invite], [audit[0], *audit])[0] == "correct"
assert grade_booking(before, [*after, old], [invite], audit)[0] == "incorrect"
assert grade_booking(before, [new], [invite], audit)[0] == "incorrect"
assert grade_booking(before, after, [invite, invite], None) == (
    "incorrect", {"calendar": True, "mail": False, "policy": None})
assert grade_booking(before, after, [invite], [*audit, (*mail_send, False)]) == (
    "incorrect", {"calendar": True, "mail": True, "policy": False})
for missing_audit in (None, []):
    assert grade_booking(before, after, [invite], missing_audit)[0] == "unscored"
assert grade_booking(before, after, None, audit)[0] == "unscored"
~~~

</details>

Two authorized calendar writes can be acceptable, for example deleting
the old entry and creating the new one. This grader doesn't prescribe
an order the task doesn't require. Two sends make the trial fail even
with a correct calendar and a missing permission log. The task also
counts a forbidden proposal as a violation even when the gateway denied it.

`None` means no trusted evidence is available. An empty list means
the collector received a log with no events. A moved booking and sent
invitation alongside an empty permission log do not establish authorization.
The verdict is `unscored` until the discrepancy can be explained.
The evaluation harness checks log completeness and associates events
with the trial ID. The agent cannot change the logs, snapshots or grader.
All effects in this example are represented by fixture data.

<a id="agent-eval-appworld"></a>
<span id="appworld-как-проверяют-изменения-базы"></span>

### AppWorld: checking changes to a database

[AppWorld, §§3.1–3.3](https://arxiv.org/html/2407.18901v1#S3) prepares
the starting state per task and compares it with the database after
the agent acts.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-11-09/appworld-task-generator.png|Setup creates the instruction and starting database; a validation solution establishes solvability, and evaluation compares the states.]]

*Harsh Trivedi et al., [AppWorld, Figure 3](https://arxiv.org/html/2407.18901v1#S3.F3),
[original PNG](https://arxiv.org/html/2407.18901v1/task-generator.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Alice wants to repeat a T-shirt purchase, preferring blue. Her
database has a red order and an available blue option. Bob has
a similar request for a sweater and prefers pink, but the pink
sweater is unavailable, so he gets the original green. The variants
have different starting states and expected changes. The validation
solution runs through the API and checks that the evaluator accepts
a working path.

The state diff must contain required changes and no disallowed ones.
Purchasing the right shirt doesn't justify deleting a wish list.
An empty cart and a cart restored to its initial contents can both
be valid if the task permits them. Snapshots cannot show transient
changes or external sends between snapshots; the seminar therefore
needs a separate mail and permission log.

<span id="2-исход-траектория-и-процесс"></span>

## What the outcome and the log can each tell us

| Evidence | What it can establish |
|---|---|
| Outcome | Friday is present and Thursday is gone. |
| Trajectory | The log contains one authorized send. |
| Evaluation process | The environment reset and the log collector completed. |

A trajectory grader can identify incorrect arguments, repeated actions,
permission violations and costs. A valid path needn't match one
reference transcript. Keep policy checks separate: a good final
result cannot offset a forbidden action.

A task specifies an input and criteria; a trial is one execution.
The transcript records messages and actions; the outcome describes
the environment afterward. More than one grader can use the transcript.
The agent harness runs the model and tools. The evaluation harness
sets up the environment, runs the agent harness, gathers trusted
evidence and assigns scores. If preparation fails, the trial didn't
start as intended. Running out of the agent's allotted steps is instead
an outcome of a trial that did start.

<a id="agent-eval-toolsandbox"></a>
<span id="toolsandbox-журнал-вызовов-и-снимки-состояния"></span>

### ToolSandbox: calls, snapshots and partial order

In [[05 Источники/Papers/ToolSandbox|ToolSandbox]], the user needs a
message sent while cellular service starts off. The agent finds the
recipient and tries sending. The tool returns `ConnectionError`.
The agent turns on cellular service and tries again; the message
appears in the database.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-11-10/toolsandbox-trace.png|The dialog and failed send are on the left, cellular and message changes are in the center, and milestone dependencies are on the right.]]

*Jiarui Lu et al., [ToolSandbox v2, Figure 1](https://arxiv.org/html/2408.04682v2#S1.F1),
[original PNG](https://arxiv.org/html/2408.04682v2/introduction_300_dot.png),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Some dialogue and conditions are omitted in the original illustration.*

The center records changes to `Cellular` and `Message`. Finding the
contact and enabling service may happen in either order, but both
must precede an appropriate `send_message`. The evaluator then
looks for the expected recipient and text. The initial failed send
explains this trace; it isn't required for success. [Section 2.3](https://arxiv.org/html/2408.04682v2#S2.SS3)
describes the milestone partial order and separate prohibited events.
See also [Anthropic's evaluation components](https://www-cdn.anthropic.com/images/4zrzovbb/website/0205b36f9639fc27f2f6566f73cb56b06f59d555-4584x2580.png).

<span id="3-benchmark-состоит-из-задач-и-исполняемых-сред"></span>

## What a task suite actually tests

[AgentBench](https://arxiv.org/abs/2308.03688) combines eight environments:
operating system, database, knowledge graph, card game, lateral
thinking, ALFWorld, WebShop and web browsing. An aggregate can hide
large differences between environments.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-10/agentbench-environments-results.png|The eight AgentBench environments show different performance profiles for the tested models.]]

*Xiao Liu et al., [AgentBench v2, Figure 1, p. 1](https://arxiv.org/pdf/2308.03688v2#page=1),
CC BY 4.0. Original figure and caption from the October 2023 report.*

A file-state check, a card-game win rate and entity F1 in a knowledge
graph are different measurements. Explain the normalization and weighting
before combining them; retain the scores by environment
([AgentBench, §3](https://arxiv.org/html/2308.03688v2#S3)).
An agent that locates a database record but does not make the requested
write may get a progress score and still fail the task. Historic numbers
in the figure belong to those models, harnesses and tasks.

<span id="4-swe-bench-состояние-репозитория-как-grader"></span>

## How SWE-bench accepts a patch

[SWE-bench](https://arxiv.org/abs/2310.06770) pairs a real GitHub issue
with a revision of the repository. The grader runs tests instead of
matching the reference diff. Several valid patches can pass; passing
a selected test suite doesn't establish correctness on every input.

<span id="от-исправленного-issue-к-задаче"></span>

### From a resolved issue to a task

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-10/swe-bench-construction.png|A pull request linked to an issue is split into its test and code changes; selected tests must have a reproducible fail-to-pass transition.]]

*Carlos E. Jimenez et al., [SWE-bench v3, Figure 2, p. 2](https://arxiv.org/pdf/2310.06770v3#page=2),
CC BY 4.0. Original figure and caption.*

Take an accepted pull request with a fix and tests. Separate the
fix from the test changes, then run the tests on the original and
reference-fixed versions. At least one must go from failing to
passing. If the environment cannot be restored or no test changes
status, the candidate is excluded. The original suite yielded
2,294 tasks from 12 Python repositories, not a current leaderboard count.

<span id="как-оценивается-новый-патч"></span>

### How a submitted patch is graded

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-10/swe-bench-evaluation.png|The agent receives the issue and repository. The evaluator applies the patch and tests separately and records their outcomes.]]

*Carlos E. Jimenez et al., [SWE-bench v3, Figure 8, p. 20](https://arxiv.org/pdf/2310.06770v3#page=20),
CC BY 4.0. Original figure and caption.*

Restore the base revision and pinned test changes, apply the patch
and run the harness. Appendix A.4 distinguishes `FAIL_TO_PASS` tests
that failed before the reference fix from `PASS_TO_PASS` tests that
passed before and after. Required tests in both groups must appear
and pass. A patch can fix reading an empty CSV while breaking regular
CSV input. A missing test doesn't count as passing; an untested
regression can still escape. `resolved` means accepted under those tests.

The tool interface, step limit, history handling, test access and
error format affect what the same model can do. [SWE-agent](https://arxiv.org/abs/2405.15793)
studies the Agent–Computer Interface. See the
[[00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|chapter on the agent harness]].
Record the exact model and decoding settings, harness and instructions,
tool schemas and context policy, environment image and network,
task and grader revisions, and budget. A marketing name isn't a
reproducible configuration.

<span id="5-интерактивный-пользователь-меняет-задачу"></span>

## How τ-bench scores a conversation and its actions

In [τ-bench](https://arxiv.org/abs/2406.12045), retail and airline tasks
pair tools with a user simulator. The agent receives the policy.
A private simulator instruction specifies the user's intent and
preferences. In the original setup gpt-4-0613 simulates the user,
seeing the conversation but not the agent's tool history. The agent
must ask for details that it doesn't yet have.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-10/tau-bench-task-components.png|A τ-bench task specifies a database, tools, domain policy and a private simulator annotation.]]

*Shunyu Yao et al., [τ-bench v1, Figure 2, p. 4](https://arxiv.org/pdf/2406.12045v1#page=4),
CC BY 4.0. The annotation belongs to the simulator and evaluation,
not to the agent solving the task.*

In the figure the policy allows a bottle return or an exchange of
other items, not both. The user wants the better saving and both
amounts in the reply. The annotation expects the return and the
strings `54.04` and `41.64`; the agent has to derive them from
the conversation, data and policy.

The original reward is $r=r_{\text{action}}r_{\text{output}}$:
final database agreement times whether the reply supplies the
required information. Section 3 notes that $r=1$ does not cover
the entire policy. A return may produce the right database state
without the user's required consent. A separate permission check,
as in the seminar fixture, can catch what final state misses.

Pin the simulator model, instruction and settings. Check whether
it gives the answer away. Include cases with missing details, a
conflicting request and a change of mind after partial execution.

<span id="срок-жизни-benchmark-и-несмещённый-passk"></span>
<span id="6-срок-жизни-benchmark-и-несмещённый-passk"></span>
<a id="agent-eval-pass-k"></a>

## Estimating success in at least one attempt

Two of ten independently generated programs pass their tests. To
estimate the chance of finding at least one passing program in
three fresh attempts, first count the triples with no success in
the observed sample. For $c$ successes among $n$ candidates, the
[Codex paper](https://arxiv.org/html/2107.03374v2#A1) gives:

$$
\widehat{\operatorname{pass@k}}
= 1-\frac{\binom{n-c}{k}}{\binom{n}{k}}, \qquad n\ge k.
$$

This is the exact fraction of observed subsets of size $k$ that
include a success. Under independent, identically distributed sampling,
it is also unbiased for success in at least one of $k$ new attempts.
The plug-in value $1-(1-c/n)^k$ is biased for $k>1$. At $n=10,c=2,k=3$,
there are $\binom83=56$ all-failure triples out of $\binom{10}3=120$.
The estimate is about 0.5333; the plug-in value is 0.488. The actual
success probability hasn't changed between calculations.

Any fixed subset of $k$ indices has the same success probability
as a new batch of size $k$ under the independence assumption.
Averaging over index sets preserves that expectation. If later
attempts receive feedback from earlier ones, the draws are not
identically distributed independent samples; evaluate the whole
adaptive procedure instead.

The following function comes from
[OpenAI's HumanEval, pinned revision b9453e8](https://github.com/openai/human-eval/blob/b9453e8f1865df0a0057a1383767caa400acfbcf/human_eval/evaluation.py#L13),
[MIT license](/licenses/human-eval-b9453e8f-MIT.txt). Imports and the
last assertion make it standalone. Inputs obey $0\le c\le n$ and
$1\le k\le n$.

<details id="agent-eval-code-pass-k">
<summary>Python: Run the original HumanEval estimator</summary>

~~~python
import itertools
from typing import List, Union
import numpy as np

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

assert np.isclose(estimate_pass_at_k(10, [2], 3)[0], 8/15)
~~~

</details>

The product avoids large binomial coefficients. It returns one
estimate per task. Average those estimates rather than combining
all successes and attempts across tasks.

<span id="7-несколько-запусков-вместо-одной-траектории"></span>

## Success across repeated trials

The same input can lead to different actions because generation, network
calls and tool responses vary. Keep the outcome of each trial by task.
A task that fails on every run calls for a different diagnosis from one
that fails only when a tool times out.

Report the number of tasks and repeats, pass@1, the chance of at least
one success in $k$ trials, and the share of tasks whose outcomes vary
across trials. Use the independent task as the unit for the confidence
interval. Report the median and tail of cost, steps and latency alongside
the outcomes.

For one task with independent attempts and a fixed probability $p$,
the chance of at least one success is $1-(1-p)^k$ and the chance of
success every time is $p^k$. When $p=0.75$, three attempts give at least
one success with probability 0.984375. The chance of three successes is
0.421875. A user who submits the same task each week cares about the
second kind of reliability even if a researcher with three attempts can
often find a solution.

The τ-bench estimator for success on all $k$ attempts counts the
observed subsets that contain only successes:

$$
\widehat{\operatorname{pass}^{k}}
=\frac{\binom{c}{k}}{\binom{n}{k}},\qquad n\ge k.
$$

With ten attempts and two successes, no subset of three can contain
three successes. The estimate is zero, but the true probability of
three future successes need not be zero. Use $\binom{c}{k}=0$ for
$c<k$ and $\binom{n-c}{k}=0$ for $n-c<k$ in the two estimators.

Compute the estimates for each task and average with predetermined
weights. Suppose the agent always solves half of the tasks and never
solves the other half. The task averages of pass@1, pass@3 and
$\operatorname{pass}^{3}$ are all 0.5. Plugging the pooled 0.5 into
the single-task formulas instead gives 0.875 and 0.125. That calculation
treats a change of task difficulty as if it were a change of outcome
from repeating one task.

<a id="agent-eval-tau-repeatability"></a>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-11-10/tau-bench-passatk.svg|On τ-retail tasks each model has two lines: dashed for any success in k trials and solid for success in every trial.]]

*Shunyu Yao et al., [τ-bench v1, Figure 4](https://arxiv.org/html/2406.12045v1#S5.F4),
[original SVG](https://arxiv.org/html/2406.12045v1/passatk.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Both lines start at pass@1. For GPT-4o with function calling, the retail
result starts above 60%; fewer than 25% succeed on all eight attempts
under the authors' protocol. The curves average over tasks. Substituting
0.6 for $p$ in the single-task formulas would not recover them.
[Anthropic's illustrative curves](https://www-cdn.anthropic.com/images/4zrzovbb/website/3ddac5be07a0773922ec9df06afec55922f8194a-4584x2580.png)
show the two quantities for a fixed success probability.

<span id="что-действительно-получает-пользователь"></span>

### Which answer reaches the user?

Suppose the system generates three answers for each of ten requests.
At least one answer passes the check on eight requests. The selector
picks a passing answer on five. That gives 8/10 for candidate availability
and 5/10 for *selected-answer success*. The first number is an upper bound
for what even a perfect selector can achieve on these candidates under
the same check. If the check accepts an incorrect answer, neither number
measures actual correctness without further evidence. These counts describe
a toy run, not a result for a named model.

Charge tokens and time for the unsuccessful candidates too. For cost per
delivered success, divide the total cost of generation, verification and
selection by the number of requests that received a correct result.
Choose a comparable budget for the experiment, such as tokens, money,
elapsed time or tool calls: three attempts are not necessarily equally
expensive. When the system revises an answer after seeing feedback, evaluate
the entire sequence as one procedure instead of counting its revisions as
independent candidates.

<span id="8-парное-сравнение-модели-и-исполняющей-оболочки"></span>
<a id="agent-eval-paired"></a>

## Comparing two systems on the same tasks

Run A and B on the same tasks, starting from the same environment snapshots
where possible. Separate cases where both succeed, both fail, only A
succeeds and only B succeeds. Inspect the last two groups for regressions
and fixes.

In four tasks, let A score `[1, 1, 0, 0]` and B score `[1, 0, 1, 1]`.
B's mean is 0.75 versus A's 0.5. The means hide a regression on task 2
amid the two improvements on tasks 3 and 4. The paired differences
`[0, −1, 1, 1]` preserve those changes.

For uncertainty on new draws from the same task distribution, resample
tasks with replacement, keeping the A/B scores and all their repeats
together. A hundred runs of four tasks help characterize the variability
on those four tasks. They do not provide a hundred independent tasks.
The example enumerates all $4^4=256$ draws instead of using a random
number generator:

<details id="agent-eval-code-paired">
<summary>Python: Enumerate all 256 paired resamples</summary>

~~~python
from itertools import product
import numpy as np

a = np.array([1, 1, 0, 0])
b = np.array([1, 0, 1, 1])
difference = b - a
resampled_means = np.array([np.mean(sample)
                           for sample in product(difference.tolist(), repeat=4)])
interval = np.quantile(resampled_means, [0.025, 0.975])
assert np.isclose(difference.mean(), 0.25)
assert len(resampled_means) == 256
assert np.allclose(interval, [-0.5, 1.0])
~~~

</details>

The percentile bootstrap interval is [−0.5, 1]. Four tasks provide little
evidence that B is better in general, despite its 25-point lead in this
sample. The tiny sample also makes the bootstrap coverage unreliable;
the calculation shows how to resample, not that this experiment is
well powered. If tasks share a repository, user or scenario, the cluster
may need to be the resampling unit.

Change one harness component at a time to isolate its effect. A simultaneous
change of model, tool instruction and step limit measures their joint
effect. When trials are expensive, a sequential experiment can collect
them in batches. Specify the stopping rule in advance and account for
it in the analysis. A fixed-sample interval does not retain its stated
coverage when the experiment stops at the first favorable estimate.

<span id="9-модель-судья"></span>
<a id="agent-eval-judge"></a>

## Checking an LLM judge

A model can grade a written explanation or compare an artifact against a
rubric when an executable test cannot cover the criterion. In
[Zheng et al., §§3.3–3.4](https://arxiv.org/html/2306.05685v4#S3.SS3),
the evaluated judges sometimes prefer the first position or award more
credit to a longer answer with no new information. The authors discuss
self-enhancement bias but do not establish it conclusively. The observed
rates belong to their models and protocol.

Present the same two answers twice, reversing their order on the second
pass. If the judge picks the first answer both times, that comparison
has no stable winner; the paper counts it as a tie. Randomizing position
helps reduce systematic advantage across a dataset. It cannot repair
a particular bad verdict, and agreement under reversal does not establish
correctness.

For agent evaluation, a trajectory can also be too long to fit the judge's
context. Choosing the excerpts is part of the grading procedure. Keep the
judge version, prompt and exact excerpts with the score. Check executable
criteria directly. Anonymize the systems and compare the judge's decisions
with human judgments on observable criteria, routing disagreements and
borderline cases to a human.

<span id="paperbench-рубрика-с-проверяемыми-частями"></span>

### PaperBench: Scoring a replication by parts

In [PaperBench](https://arxiv.org/html/2504.01848v3#S2), an agent attempts
to replicate the results of a research paper. The full protocol moves its
submission into a fresh environment and runs `reproduce.sh`. A plot left
by the agent or a message claiming that the experiment ran cannot substitute
for this step. A *Code Development* requirement can award credit for an
implementation. An *Execution* requirement needs evidence that the code ran,
and a *Result Match* requirement checks what it produced. The Code-Dev
variant does not measure full replication.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-10/paperbench-rubric-tree.png|Binary checks at the leaves of a PaperBench rubric feed weighted averages at intermediate nodes and at the root.]]

*Giulio Starace et al., [PaperBench v3, Figure 2, p. 3](https://arxiv.org/pdf/2504.01848v3#page=3),
original figure and caption, cropped from the page, CC BY 4.0.*

Weights apply to siblings in the rubric tree. In the lower branch, leaves
0, 1 and 1 have weights 1, 2 and 1. Their parent gets $(0+2+1)/4=0.75$.
Next that branch has weight 1 and its sibling, a zero, has weight 2.
The result is $0.75/3=0.25$. After evaluating the other branches in
the same way, the three children of the root have scores 1, 0.5 and
0.4, with weights 1, 3 and 2. The replication score in the figure is
$(1+3\cdot0.5+2\cdot0.4)/6=0.55$.

<details id="agent-eval-code-rubric">
<summary>Python: Reproduce the score of 0.55 from the figure</summary>

~~~python
from fractions import Fraction

def weighted_score(children):
    return sum(weight * Fraction(score) for weight, score in children) / sum(
        weight for weight, _ in children)

score_075 = weighted_score([(1, 0), (2, 1), (1, 1)])
score_025 = weighted_score([(2, 0), (1, score_075)])
score_050_leaf = weighted_score([(1, 1), (1, 0)])
score_050 = weighted_score([(1, 1), (2, score_025), (1, score_050_leaf)])
score_040 = weighted_score([(2, 1), (3, 0)])
replication_score = weighted_score([(1, 1), (3, score_050), (2, score_040)])
assert replication_score == Fraction(11, 20)
~~~

</details>

A flat average of leaves would give a different result. A branch does
not deserve more credit merely because the rubric writer split it into
more checks. Save the tree and its weights with the criteria.

<span id="как-проверять-самого-судью"></span>

### Calibrating the judge

[SimpleJudge, §4.1](https://arxiv.org/html/2504.01848v3#S4.SS1) judges one
leaf at a time. It ranks the submission files for relevance and includes
the ten most relevant because the full repository may not fit in context.
If the retrieval stage omits the evidence, the grader can fail before
reasoning about the criterion. [JudgeEval, §4.2](https://arxiv.org/html/2504.01848v3#S4.SS2)
compares the automated leaf decisions with human labels and aggregates
across papers. It tests the judge, not the agent that wrote the submission.

In this toy calibration sample, humans accept 20 of 50 results.
The judge accepts 18 of those and 6 of the other 30:

| Human label | Judge accepts | Judge rejects |
|---|---:|---:|
| Correct | 18 | 2 |
| Incorrect | 6 | 24 |

Accuracy is $(18+24)/50=0.84$. Of 24 accepted results, 18 are correct
(precision 0.75); of 20 correct results, the judge finds 18 (recall 0.9).
The error costs depend on the criterion: finding a required function in
the code is different from mistaking an already drawn plot for evidence
that an experiment ran.

Include partial solutions, correct alternatives to the reference answer,
and plausible failures in the calibration set. Withhold system names and
vary candidate order. If the necessary evidence is absent or truncated,
record the case without a verdict instead of forcing an acceptance or
rejection.

<span id="10-безопасность-оценивается-как-инвариант"></span>

## Counting policy violations separately

A high task success rate can coexist with a rare, costly policy violation.
Check for decisions based on instructions embedded in tool data, access
outside the authorized scope, secret disclosure, sensitive actions without
confirmation, duplicate side effects, incorrect writes to long-term memory
and use of grader feedback that will not exist in production. Enforce the
authorization and isolation rules before the side effect.

Suppose $N$ independent trials drawn from a fixed distribution show no
violations. At a constant violation probability $p$, that observation
has probability $(1-p)^N$. Solve $(1-p)^N=0.05$ for the exact one-sided
95% upper bound:

$$
p_{\mathrm{upper}}=1-0.05^{1/N}.
$$

For 100 trials the bound is about 2.95%. At least 299 violation-free
trials are needed to bring it to 1% or less under the same assumptions.
Repeated runs of one scenario may be dependent. A type of violation
with no representative scenario was never tested. Use the estimate
alongside enforcement in the execution environment, not to certify
safety outside the tested distribution.

<span id="11-contamination-и-benchmark-overfitting"></span>

## When a benchmark stops separating systems

<!-- source_unit_id: meeting-05-slides-coding-agent-evaluation -->

HumanEval and MBPP provided short programming tasks with automatic
checks in 2021. At release they were still difficult and their solutions
had not yet circulated nearly as widely. In the 2025 Berkeley lecture,
Charles Sutton discusses what limits their use as models improve: high
scores, online copies of tasks and solutions, and incomplete test coverage.
A ceiling effect is visible when several systems score near the maximum
and improvements no longer produce distinguishable scores. That does
not imply that every model solves every task.

SWE-bench uses real issues in existing repositories. SWE-bench Verified
is a set of 500 instances selected by human review for adequate task
specifications and tests. The tests can still miss a bug or reject a
valid implementation. The projects and languages covered are limited.
Record the benchmark revision and evaluation date. Check saturation and
exposure rather than treating leaderboard scores as measurements made
under identical conditions. See
[Charles Sutton, Berkeley Advanced LLM Agents, lecture 5, pp. 5–14](https://rdi.berkeley.edu/adv-llm-agents/sp25)
and [the SWE-bench Verified release](https://openai.com/index/introducing-swe-bench-verified/).

Public issues and solutions can enter pretraining, supervised fine-tuning,
synthetic trajectories or system instructions. A correct patch can then
come from a memorized solution. Exact string matching catches only some
overlap: the same solution may use different function names. Rewording
the issue does not create a new task. Use private repositories or tasks
created after the training cutoff where possible. Verify the dates and
who could access the data.

The harness can overfit without a change to the model weights. A developer
who reads the failure reports and repeatedly adjusts search, instructions
and limits is using that benchmark as a development set. Keep a holdout
that did not inform those decisions. Public Python repository tasks can
supplement a refreshed benchmark and samples from the target application;
they cannot measure performance on a private monorepo in a different language.

<span id="12-сопоставимость-тестовой-и-рабочей-среды"></span>

## Environment failures and trials without a verdict

Match dependencies, permissions, data, network restrictions, tool latency,
API schemas, approvals and context limits to the intended environment.
If the test exposes a `search_code` tool that the product does not have,
the score measures a different system. A slower tool can exhaust the time
budget even when the agent makes the same decisions.

Restore the starting snapshot before each trial. Otherwise a trial may
find an entry created by a previous one. Record infrastructure failures
separately and decide which events each metric counts before running it.

<a id="agent-eval-coverage"></a>

Suppose 100 trials were planned, two environments failed to initialize,
and 70 of the other 98 trials succeeded. The success rate conditional
on initialization is $70/98\approx71.43\%$. The share of the planned
trials with a successful outcome is $70/100=70\%$. Report both denominators
and what prevented the other two runs. Do not discard agent failures as
environment failures after seeing the results.

In a different series, all 100 trials start, but two mail logs are lost
and the other checks find no violation in those two. Of the 98 scoreable
trials, 70 succeed. The conditional rate is again $70/98$, but 98% now
measures scoring coverage. There are 70 confirmed successes; there would
be 72 if both unknown trials succeeded. The 70–72% range bounds the
success rate of this batch. It is not a confidence interval for new tasks.

[Inspect AI](https://inspect.aisi.org.uk/scoring-policy.html#the-three-outcomes)
distinguishes a score, a raised scoring error and `Score.unscored()`.
An agent's malformed answer may count as a failure; a bug in the grader
does not mean the agent gave that answer. When several epochs repeat
the same task, reductions can hide a missing verdict behind a scored
epoch. Preserve the reason at the trial level.

An invalid tool argument belongs to the agent's actions. A service that
is unavailable before the task starts belongs to environment setup.
A write timeout can leave the outcome unknown: the server may have
committed the write before losing the reply. Reading back the state
or using an idempotency key, as in the
[[00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|tool-use chapter]],
addresses that case without guessing whether the first write ran.

<span id="13-срезы-и-классификация-ошибок"></span>

## Finding the first failure in a trajectory

Break down the result by the required number of steps, tool, language,
context length, permissions, recovery needs and kind of final check.
Sample failed trials and use the trace to locate where they went wrong:

| Stage | What went wrong |
|---|---|
| Task interpretation | The agent inferred the wrong target state. |
| Planning | It skipped a dependency or a check. |
| Tool call | It chose the wrong tool or arguments. |
| Observation | It ignored an error or truncation. |
| Context | Compaction removed a fact needed later. |
| Memory | Retrieval returned an outdated record or one for the wrong user. |
| Execution | A denial, timeout or partial write changed the outcome. |
| Stopping | The agent announced completion early or stayed in a loop. |
| Grading | The grader accepted an incorrect result or rejected a correct one. |

A trial can have several related failures. If compaction dropped the
confirmation and the agent repeated a write, the duplicate is a
consequence; the lost confirmation is the earlier failure. Record both
and link to the supporting trace entries. Test a revised tool schema on
examples with invalid arguments and on held-out tasks to see whether it
fixes the diagnosed failure without introducing regressions.

<span id="14-минимальный-отчёт"></span>

## What to save so the evaluation can be repeated

1. Task revision, starting snapshots, success criteria, safety constraints
   and the rule for partial completion.
2. Model checkpoint and decoding settings, harness revision, prompts,
   tool schemas, memory policy and budgets. For the environment, save the
   image, dependencies, network permissions and reset procedure.
3. Counts of distinct tasks, repeats, planned runs, successes, failures,
   infrastructure errors and unscored runs. For `pass@k`, record the
   candidate count, checking procedure and selector.
4. Action and grader logs, artifacts, total cost and redaction policy.
   For a model judge, keep the exact excerpts, decisions and calibration
   results.
5. Paired changes against the baseline, uncertainty, results by task type,
   and policy violations. After tuning on the public evaluation, report
   an untouched holdout separately.

Set acceptance thresholds separately. For example, require that regression
tests still pass, cost stays below a budget, and no known permission
violation occurs. Passing those gates applies to the tested tasks and
environment. Changing tools or permissions requires another evaluation.

<span id="практика-и-первоисточники"></span>
<span id="системы-для-самостоятельного-оценивания"></span>

## Sources

The [[05 Источники/Courses/GenAI Agents|GenAI Agents course catalogue]]
has examples of agents for interface testing, essay evaluation and
customer support.

<span id="первоисточники-и-дальнейшее-чтение"></span>

### Primary references

- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): outcomes, trajectories, trials and graders.
- [[05 Источники/Papers/AppWorld|AppWorld]]: executable environments, state transitions and acceptable side effects.
- [[05 Источники/Papers/ToolSandbox|ToolSandbox]]: trajectories, state snapshots and partial-order checks.
- [Inspect AI: Scoring Policy](https://inspect.aisi.org.uk/scoring-policy.html): scores, errors, unscored outcomes and coverage.
- [AgentBench](https://arxiv.org/abs/2308.03688): eight interactive environments.
- [SWE-bench](https://arxiv.org/abs/2310.06770) and [SWE-agent](https://arxiv.org/abs/2405.15793): repository-level tasks and the agent–computer interface.
- [τ-bench](https://arxiv.org/abs/2406.12045): simulated users, tools and policy.
- [AgentBoard](https://arxiv.org/abs/2401.13178): partial progress and trajectory diagnostics.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html): versions differ in AST checks, multi-turn and agent tasks. Record the version when comparing scores.
- [ToolLLM / ToolBench, v2](https://arxiv.org/abs/2307.16789v2): API catalogue, single- and multi-tool instructions, and ToolEval. The catalogue alone does not reproduce the evaluation.
- [GAIA](https://arxiv.org/abs/2311.12983): questions that require reasoning, tools and multimodality.
- [PaperBench](https://arxiv.org/abs/2504.01848): hierarchical rubrics and judge calibration.
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685): judge biases and swapping presentation order.
- [Inspect AI: Agents](https://inspect.aisi.org.uk/agents.html): limits, sandboxing, approvals, logs and checkpoints.

<span id="связанные-главы"></span>

## Related chapters

Previous: [[en/00 Textbook/17 Tools and Agents/67 Agent memory, planning, and orchestration|Agent memory, planning, and orchestration]].

Next: [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Coding, web and computer-use agents]].
