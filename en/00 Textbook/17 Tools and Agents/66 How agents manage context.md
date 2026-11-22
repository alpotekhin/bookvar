---
title: How agents manage context
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md"
canonical_revision: "4c684a6f8bb23efc739321e3b0056666379c865b6a40aaf0cfae0b3bd57ea1d3"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/17 Tools and Agents/65 How a function call becomes an action]]"
next: "[[00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов]]"
reference_terms:
  - label: Agent loop — model call, execution, and verification
    label_en: Agent loop — model call, execution, and verification
    anchor: harness-loop
  - label: Agent context — token budget
    label_en: Agent context — token budget
    anchor: harness-context-budget
  - label: Prefix caching in the agent loop
    label_en: Prefix caching in the agent loop
    anchor: harness-prompt-prefix
  - label: Event log, active context, and task state
    label_en: Event log, active context, and task state
    anchor: harness-authoritative-state
  - label: Observation masking and history summarization
    label_en: Observation masking and history summarization
    anchor: harness-observation-masking
  - label: Compaction — boundaries around calls and observations
    label_en: Compaction — boundaries around calls and observations
    anchor: harness-compaction-boundaries
  - label: Handoff after an operation with an unknown outcome
    label_en: Handoff after an operation with an unknown outcome
    anchor: harness-handoff-unknown
  - label: Reminder tokens — triggers, reminders, and checks
    label_en: Reminder tokens — triggers, reminders, and checks
    anchor: harness-reminder-tokens
  - label: Agent completion — verifying the current revision
    label_en: Agent completion — verifying the current revision
    anchor: harness-verified-completion
primary_sources:
  - https://openai.com/index/unrolling-the-codex-agent-loop/
  - https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  - https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
  - https://arxiv.org/abs/2405.15793
  - https://docs.openhands.dev/sdk/guides/context-condenser
  - https://arxiv.org/abs/2508.21433v3
  - https://github.com/OpenHands/software-agent-sdk/blob/0996c33916c2a977ea9fe38deee660ff0e1389f2/openhands-sdk/openhands/sdk/context/view/properties/tool_call_matching.py
  - https://huggingface.co/learn/context-course/unit0/introduction
---

# How agents manage context

In the [[00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию#tool-call-contract|previous chapter]],
the model read `timeout=30`. Now the user asks it to write `timeout=45`
and run a test. The executor writes the new value and returns the
revision number. The model proposes saying “done” before running the test.

The agent harness compares the revision under test with the current
revision. Before the next model call, it supplies what the model
needs to continue: what was written and which version still needs a test.

<a id="harness-loop"></a>
<span id="1-из-чего-состоит-один-шаг"></span>

## What happens between model calls

The harness reads task state and builds the next input. It parses the
model's proposal, checks permissions, runs allowed tools, and records
observations. If the model proposes a final answer instead of a tool
call, the application checks the stopping condition.

1. The read returns `timeout=30` at revision 0.
2. The write requests 45 only if the file is still at revision 0.
   It creates revision 1. That revision has no test result yet.
3. The model proposes finishing. The harness rejects the answer and
   says revision 1 still needs a test.
4. The test reports its outcome and revision 1.
5. The harness accepts a final answer if the test passed and the file is
   still at revision 1.

Even if revision 2 also says `timeout=45`, a different part of the file may
have changed. The test of revision 1 does not cover it. The
[[#harness-episode|runnable trace at the end]] covers the early attempt to
finish and a restart after the write.

The January 23, 2026 account of [the Codex CLI agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
describes how that implementation assembles model instructions, sandbox
and permission information, project `AGENTS.md` files, environment
details, tool schemas and a user message. After a tool call, the output
joins the input history for the next request. Other applications can
arrange those inputs differently.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-harness-2026-10/swe-agent-prompt-workflow.png|SWE-agent builds initial messages for the first request, then appends tool output, a no-output acknowledgement or a format error.]]

*John Yang et al., [SWE-agent v3, Appendix C, Figure 26, p. 39](https://arxiv.org/pdf/2405.15793v3#page=39),
CC BY 4.0. Original figure and caption.*

SWE-agent sends system, demonstration and instance messages before the
first iteration. The new message after that depends on what happened:
a tool returned output, it succeeded without output, or the model's
proposal could not be parsed. Empty stdout alone does not establish
success. The shown implementation sends an acknowledgement for a
successful action with no output.

The figure also distinguishes several ways to stop: `submit`, a cost
limit, and repeated format errors. Reaching one of these stops the
episode; it doesn't verify a fix.

<span id="2-сборщик-контекста"></span>

## What the model receives next

Instructions and tool schemas usually persist until the configuration
changes. Task constraints last for the episode. The working directory,
branch and permissions come from the current environment. From the
growing trace, retrieved documents and memory, the harness picks what
the next decision uses. A reminder about a permission change or an
unfinished test remains useful only while the information is current.

After writing `timeout=45`, the model needs the task, confirmation of the
edit and the missing test result. After the test it needs the outcome
instead. The dependency installation log can stay in storage, referenced
by a path. The standing instruction to test an edit hasn't changed.
Anthropic's [prompt and context engineering comparison](https://www-cdn.anthropic.com/images/4zrzovbb/website/faa261102e46c7f090a2402a49000ffae18c5dd6-2292x1290.png)
shows where such instructions and changing data enter the input.

“Check your changes” does not say whether reading the diff is enough.
One standing rule is to run the check specified for the task and report
an inability to run it; the project's configuration supplies the actual
command. This keeps project-specific commands out of the general
instruction. See Anthropic's [discussion of system-prompt detail](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
and its [system prompt illustration](https://www-cdn.anthropic.com/images/4zrzovbb/website/0442fe138158e84ffce92bed1624dd09f37ac46f-2292x1288.png).

<span id="загрузка-сведений-по-мере-необходимости"></span>

### Loading information as needed

An application can put retrieved documents into the initial input or
start with references: a path, function name or report ID. The model
then chooses what to read. A practical combination for the configuration
example is to preload its working directory, write restrictions and test
command, then retrieve the matching file and the code that uses the
setting. Loading the entire repository would add irrelevant text. Supplying
only a path without access to the file would invite the model to guess.

Anthropic calls this arrangement [just-in-time retrieval and progressive disclosure](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
It adds tool calls and depends on good navigation. A reference in the
prompt does not establish that the document was read, and the index can
point to an older revision.

<a id="harness-context-budget"></a>
<span id="бюджет-одного-входа"></span>

### Budgeting one request

Count tokens after the messages and tool schemas are serialized. For
multimodal input, token accounting also depends on the model. Let
$W$ denote the window, $I$ the input, $O$ the output reservation and
$R$ the safety margin:

$$
\begin{aligned}
I &= T_{\mathrm{instructions}}+T_{\mathrm{tools}}\\
  &\quad+T_{\mathrm{task}}+T_{\mathrm{state}}+T_{\mathrm{history}},\\
I+O+R &\le W.
\end{aligned}
$$

Set $W=8192$, reserve 1,024 tokens for generation and leave a margin of
256. Instructions use 1,200, tools 500, the task 280 and environment
state 200. That leaves 4,732 for history. The 4,500-token history fits
with 232 to spare. A 1,700-token observation would put the request at
9,660 including both reservations. The application can shorten the
history or ask for a bounded excerpt if the tool supports it.

<details id="harness-code-budget">
<summary>Python: Budget the original request, compaction and next request</summary>

~~~python
window = 8192
output_reserve, safety = 1024, 256
fixed = 1200 + 500 + 280 + 200
history, incoming = 4500, 1700
history_capacity = window - output_reserve - safety - fixed
assert history_capacity == 4732
assert fixed + history + output_reserve + safety == 7960
assert fixed + history + incoming + output_reserve + safety == 9660

summary, recent_tail = 600, 1200
compaction_instruction = 100
summary_request = fixed + history + compaction_instruction + summary + safety
after_compaction = fixed + summary + recent_tail + incoming + output_reserve + safety
assert summary_request == 7636 and summary_request <= window
assert after_compaction == 6960 and after_compaction <= window
~~~

</details>

The separate summary call fits at 7,636. Replacing old history with a
600-token summary and keeping a 1,200-token tail gives 6,960 including
the expected observation and both reservations. These are hypothetical
token counts, not a guarantee of faithful compression. If even the
summary request is too large, sending the same input again cannot solve it.

<a id="harness-prompt-prefix"></a>
<span id="3-стабильный-префикс-и-динамический-суффикс"></span>

## Which requests share a cacheable prefix

Putting stable instructions, examples and schemas first gives successive
requests a longer common prefix. Prompt caching reuses computed KV
states for that prefix. The new suffix still requires processing and
the model generates a new output. An exact serialized prefix match,
including tools, is a prerequisite rather than a cache-hit guarantee:
routing, thresholds and retention rules also apply. See the
[OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

In the toy token sequence, stable IDs are `[10,11,12,13]` and the first
observation adds `[20,21]`. Appending another result preserves six
matching tokens. Prepending a changing timestamp leaves no shared prefix.

<details id="harness-code-prefix">
<summary>Python: Find the first mismatched token</summary>

~~~python
def common_prefix_length(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))

stable = [10, 11, 12, 13]
first = stable + [20, 21]
next_request = first + [30, 31]
assert common_prefix_length(first, next_request) == 6
assert common_prefix_length([900] + first, [901] + next_request) == 0
assert common_prefix_length(stable, [10, 11, 99, 13]) == 2
~~~

</details>

This tests token equality, not provider cache behavior, latency or billing.
An old permission record can remain in a shared prefix; the execution
decision still uses current permissions. In the January 2026 Codex CLI
account, changes to permission settings can be appended as developer
messages, and working-directory changes as user messages. That avoids
rewriting earlier items without granting those older items authority.

<a id="harness-authoritative-state"></a>
<span id="4-состояние-не-равно-стенограмме"></span>

## What stays outside the model request

The event log records observations and actions for later inspection.
The active context is the bounded projection sent to the model.
Task state tracks goals, completed checks, outstanding work, versions,
budgets and blockers. It is updated by events and checked against the
environment. A summary is a way to show that state to the model, not to
decide whether a test ran.

The log can contain separate events for writing revision 1 and testing
it. Before the test, the model needs “revision 1 written; test pending.”
Afterward, it needs the result for revision 1 rather than a second
install log. Replacing a message in the active context should leave the
event available for inspection. Keeping the event forever doesn't make
it available to a model unless the next input includes it or a way to
retrieve it.

<span id="5-контекст-со-временем-портится"></span>
<span id="5-использование-длинной-истории"></span>

## What gets lost in a long history

In [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/), the
authors vary two things separately: the position of the document with
the answer, and how many distractors surround it. The question about the
first physics Nobel Prize winner stays the same. The input can make
the answer harder to use without removing it.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-harness-2026-10/lost-middle-task-controls.png|For the same Nobel Prize question, the answer-bearing document moves or more distracting documents are added.]]

*Nelson F. Liu et al., [Lost in the Middle, Figures 2–4, PDF p. 4 / printed p. 160](https://aclanthology.org/2024.tacl-1.9.pdf#page=4),
CC BY 4.0. Original prompt and the two experimental controls.*

In experiments with 10, 20 or 30 documents, several tested models
performed better when the relevant passage was near the beginning or
end. The figure below shows the GPT-3.5-Turbo-0613 run with 20
documents. The dashed baseline answers without documents. Adding
documents did not beat that baseline at every position.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-harness-2026-10/lost-middle-position-effect.png|Accuracy changes with the position of the relevant document. The dashed baseline does not use documents.]]

*Nelson F. Liu et al., [Lost in the Middle, Figure 1, PDF p. 2 / printed p. 158](https://aclanthology.org/2024.tacl-1.9.pdf#page=2),
CC BY 4.0. Results for the original models and tests.*

The shape is not universal. Claude-1.3 did well on the paper's synthetic
key-value retrieval task even when multi-document QA remained hard.
Retrieving one exact value and using a retrieved fact in an answer are
different tests.

Agent history can also include true facts that have become stale.
`timeout=30` describes the file before the edit; “the next step is to
edit the file” no longer applies. Repeated test output can add tokens
without adding evidence. Truncation may break the causal trace, masking
keeps actions but hides output, and retrieval depends on locating the
fact again. Persistent notes also need a way to stay current. A handoff
to a fresh session must preserve the live task state.

In the policy described by SWE-agent, the last five observations
keep their full texts; older ones become a line such as
`Old output omitted (101 lines)`. With seven observations, the first
two become placeholders and the action history remains. The line does
not summarize the 101 omitted lines. If their contents are needed, read
the saved log or repeat a permitted read.

<a id="harness-observation-masking"></a>
<span id="сократить-вывод-или-пересказать-ход-работы"></span>

### Masking old output or summarizing old steps

An agent finds a file, reads it and runs a test. If it only needs the
test failure next, the first two results can become placeholders.
The search and read actions remain in context. If the selected line is
needed later, it must be read again or recovered from the log. A
summary might preserve it, but the summary model doesn't know what
every later step will need.

![[00 Учебник/Assets/Figures/curated/harness-context-2026-11-09/management-strategies.png|The same trace with full history on the left, masked observations in the middle, and summarized old steps on the right.]]

*Tobias Lindenbauer et al., [The Complexity Trap, Figure 3](https://arxiv.org/html/2508.21433v3#S3.F3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure.*

In the middle column, `Mask 1` and `Mask 2` replace old observations;
`Reasoning` and `Action` stay. On the right, `Summary 1` replaces both
old steps. Both versions keep the last step intact. The schematic shows
a window of one step; the main runs keep ten. The first summary comes
after 31 steps, summarizing 21 and retaining ten. Masked history still
grows because its reasoning, actions and placeholders accumulate.

In [the Qwen3-Coder 480B SWE-agent runs on 500 SWE-bench Verified tasks](https://arxiv.org/html/2508.21433v3#S4.T1):

| History | Solved | USD per task |
|---|---:|---:|
| Full | 53.4% | 1.29 |
| Masked | 54.8% | 0.61 |
| Summarized | 53.8% | 0.64 |

From the rounded costs, masking saves about 53%:
$(1.29-0.61)/1.29\approx 53\%$. That is the cost of an entire attempt,
not one request. For this model, calls to the summarizer account for
about 7% of the summarization condition's cost. In
[the paired analysis](https://arxiv.org/html/2508.21433v3#A3.T4),
masking's difference in solved rate versus full history spans
−1.6 to +4.4 percentage points. The numbers do not establish a quality
gain or equivalence between the two methods. For Gemini 2.5 Flash
with thinking, shortening the context reduced both cost and success.
Reusing the SWE-agent window unmodified in OpenHands also worsened
the result. See the [[05 Источники/Papers/The Complexity Trap|source note]]
for the separate settings.

<span id="6-compaction-как-преобразование-состояния"></span>
<span id="6-compaction-и-сохранение-состояния-задачи"></span>

## Compacting history without losing the task

Leave room for the compaction request and its output before the window
fills. A useful checkpoint preserves the goal and stopping conditions,
decisions and their basis, artifact versions, work still outstanding
and any failed attempt that can't safely be repeated. It also needs
exact paths, operation IDs and references to the underlying evidence.
A summary of the topic alone is insufficient.

Compaction replaces a representation for the next input. It does
not execute unfinished actions, confirm a result or change application
permissions. After compaction, the application should verify that
constraints, unknown outcomes and references still match task state.

A summarizer can return ordinary text. The [Responses API compaction guide](https://developers.openai.com/api/docs/guides/compaction)
describes a different format with an opaque compaction item. In the
October 20, 2026 documentation snapshot, explicitly calling
`/responses/compact` returns the next window; pass that window forward
rather than extracting only `encrypted_content`. Server-managed
compaction has different input handling. The application's evidence
and state live separately from either representation.

<span id="что-именно-сжимает-openhands"></span>

### Which events OpenHands can replace

In the [OpenHands Context Condenser](https://docs.openhands.dev/sdk/guides/context-condenser),
a `View` is built from events. A condenser produces another view or a
`Condensation` event that identifies what to replace, the summary and
where it belongs. The [OpenHands illustration](https://mintcdn.com/allhandsai/hwL6uO0ZqM-lmlYG/sdk/guides/assets/condenser_overview_light_mode.png?fit=max&auto=format&n=hwL6uO0ZqM-lmlYG&q=85&s=dc85afa9f2821b1f7a305ad1406ab3d9)
shows a shorter view with a summary and a recent tail. Replacing events
in that view does not erase them from the original log.

In the [pinned LLMSummarizingCondenser code](https://github.com/OpenHands/software-agent-sdk/blob/0996c33916c2a977ea9fe38deee660ff0e1389f2/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py),
`max_size` limits the number of events. A separate check handles tokens,
since one observation can be longer than many short messages.
`keep_first` preserves the beginning and a recent tail normally stays
as written. Valid cuts respect groups of linked events.

The documentation's `max_size=10, keep_first=2` example crosses the
limit with 11 independent events. The planned new size is `10//2=5`:
two initial events, a summary and the last two events. With indices
0 through 10, the summary replaces 2 through 8. Group boundaries can
change that result; five is a target, not an exact invariant.

At that revision, exceeding the event-count limit is a soft trigger
and exceeding the token limit is a hard trigger. A `RollingCondenser`
may return the same view after a soft trigger when compaction isn't
available. A hard trigger needs recovery or an error instead. These
behaviors belong to the pinned implementation.

<a id="harness-compaction-boundaries"></a>
<span id="почему-нельзя-сжать-вызов-отдельно-от-ответа"></span>

### Why a call and its response belong together

Recall the [[00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию#mcp-response-correlation|two reads from the preceding chapter]].
OpenHands records two actions from one model response and then the
observations. `c3` returns before `c2`. Initial events 0 and 1 are
protected, and events 2–6 contain no tool calls. With 11 events at
`max_size=10`, keeping the last two would put the cut at index 9.
Both calls would move into the summary and their unpaired results
would remain in the tail.

| Event | Replies still pending |
|---|---|
| 7: call `c2` | `c2` |
| 8: call `c3` | `c2` and `c3` |
| 9: result of `c3` | `c2` |
| 10: result of `c2` | None |

[ToolCallMatchingProperty.manipulation_indices](https://github.com/OpenHands/software-agent-sdk/blob/0996c33916c2a977ea9fe38deee660ff0e1389f2/openhands-sdk/openhands/sdk/context/view/properties/tool_call_matching.py#L61-L95)
allows a cut only when no result is pending. After event 8 there are
two pending calls. After event 9 there is one. The first valid index
at or beyond 9 is 11, just after `c2` returns. The summary replaces
the slice `[2:11]`, leaving no orphan observations in the tail.
In this case the expected two-event tail is empty. The original
events remain in the log; the next input still needs a token count.

<details id="harness-code-cuts">
<summary>Python: Find the cut after both responses</summary>

~~~python
def safe_cuts(events):
    cuts, pending = [0], set()
    for after, (kind, call_id) in enumerate(events, start=1):
        if kind == "call":
            pending.add(call_id)
        elif kind == "result":
            pending.discard(call_id)
        if not pending:
            cuts.append(after)
    return cuts

events = [("message", None)] * 7 + [
    ("call", "c2"), ("call", "c3"),
    ("result", "c3"), ("result", "c2"),
]
keep_first, max_size = 2, 10
tail = max_size // 2 - keep_first - 1
naive_end = len(events) - tail
cuts = safe_cuts(events)
assert (tail, naive_end) == (2, 9)
assert cuts == [0, 1, 2, 3, 4, 5, 6, 7, 11]
end = next(cut for cut in cuts if cut >= naive_end)
assert events[keep_first:end][-4:] == events[7:]
assert end == 11 and events[end:] == []

# Until c2 returns, there is no valid cut at or after boundary 9.
assert all(cut < naive_end for cut in safe_cuts(events[:-1]))
~~~

</details>

The fixture calculates cuts for these toy events rather than running
OpenHands or a summary model. In the pinned version
[`_get_forgotten_events`](https://github.com/OpenHands/software-agent-sdk/blob/0996c33916c2a977ea9fe38deee660ff0e1389f2/openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py#L327-L352)
moves both boundaries to the next valid cut.
[BatchAtomicityProperty](https://github.com/OpenHands/software-agent-sdk/blob/0996c33916c2a977ea9fe38deee660ff0e1389f2/openhands-sdk/openhands/sdk/context/view/properties/batch_atomicity.py#L49-L76)
protects a batch from one model output, and
[ToolLoopAtomicityProperty](https://github.com/OpenHands/software-agent-sdk/blob/0996c33916c2a977ea9fe38deee660ff0e1389f2/openhands-sdk/openhands/sdk/context/view/properties/tool_loop_atomicity.py#L95-L126)
can protect a complete tool loop starting with a thinking block.
Without an ordinary cut, a soft request may keep the current view;
for a hard request the code tries recovery or reports an error.
Neither outcome confirms that a tool finished.

<a id="harness-handoff-unknown"></a>
<span id="handoff-после-записи-с-неизвестным-исходом"></span>

### Handing off a write with an unknown outcome

The user wants exactly one Runbook note in project `P7`. Before calling
`create_note`, the application records the project, title and operation
key `task-42-note-1`. The request goes out with approval for those
arguments, then the connection drops. A timeout does not tell the
client whether the note exists.

“Almost done; retry and report success” loses the operation key and
turns an uncertain outcome into an instruction to act. Preserve the
unknown state and the evidence instead:

<details id="harness-code-handoff">
<summary>Fields that must survive the handoff</summary>

```json
{
  "task_id": "task-42",
  "goal": "Exactly one Runbook note in project P7",
  "status": "in_progress",
  "pending_operation": {
    "name": "create_note",
    "arguments": {"project": "P7", "title": "Runbook"},
    "idempotency_key": "task-42-note-1",
    "outcome": "unknown",
    "event_ref": "events/42#17"
  },
  "next_action": "read_by_idempotency_key",
  "terminal_verified": false,
  "forbidden": ["create_with_new_key", "claim_success_without_readback"]
}
```

</details>

On resume, the harness reads current permissions and checks the
project. If the lookup returns `N9` with project `P7`, title `Runbook`
and the same key, compare those fields against the record. Finding one
matching note does not prove there are no duplicates; use the server's
deduplication contract or a separate check. On an eventually consistent
read, a single `not_found` doesn't prove that the write had no effect.
If the application can't determine the outcome, it leaves it unknown.
For the retry contract, see [[00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию#tool-retry-idempotency|a lost response after a write]].

`events/42#17` points to the original request in the log.
`terminal_verified` stays false until the result is checked. The
example below keeps the protected pieces of `trusted` when validating
a `candidate` checkpoint. A summary can change its wording, but it
can't change these values without new evidence.

<details id="harness-code-checkpoint">
<summary>Python: Reject a checkpoint that changes the operation key</summary>

~~~python
from copy import deepcopy

trusted = {
    "task_id": "task-42",
    "goal": "Exactly one Runbook note in project P7",
    "status": "in_progress",
    "pending_operation": {
        "name": "create_note",
        "arguments": {"project": "P7", "title": "Runbook"},
        "idempotency_key": "task-42-note-1",
        "outcome": "unknown",
        "event_ref": "events/42#17",
    },
    "terminal_verified": False,
    "forbidden": ["create_with_new_key", "claim_success_without_readback"],
}

def valid_checkpoint(candidate):
    return all(candidate.get(key) == value for key, value in trusted.items())

candidate = deepcopy(trusted)
candidate["summary"] = "Read back task-42-note-1 before deciding what to do."
assert valid_checkpoint(candidate)
bad = deepcopy(candidate)
bad["pending_operation"]["idempotency_key"] = "new-key"
assert not valid_checkpoint(bad)
bad = deepcopy(candidate)
bad.update(status="done", terminal_verified=True)
assert not valid_checkpoint(bad)
~~~

</details>

The check catches lost fields or a substituted key. It doesn't
judge the rest of the prose or replace the readback. Once the result
is verified, the application can update the authoritative state for the
next checkpoint. No real note is created by this example.

<span id="7-reminder-tokens--не-особый-вид-токена"></span>
<a id="harness-reminder-tokens"></a>
<span id="7-текстовые-напоминания-reminder-tokens"></span>

## When to add and remove a reminder

“Reminder tokens” here are the ordinary tokens of a reminder added
to the next model request. A literal `<system-reminder>` tag does not
grant a role or permission. The
application and its message protocol determine where the text came from.

The initial SWE-agent instance template repeats interface constraints
and guidance drawn from failed runs. One item begins:

> If you run a command and it doesn't work, try running a different command.

*John Yang et al., [SWE-agent v3, Figure 29, PDF p. 42](https://arxiv.org/pdf/2405.15793v3#page=42),
CC BY 4.0. The full example spans pp. 41–43.*

This advice is about repeating an unsuccessful interface action.
A temporary outage can clear without changing the arguments; whether
a retry is safe depends on the operation's contract. The initial
templates are used once, while the next-step template follows a tool
result. To add a reminder dynamically, define the triggering event,
the text to insert and the state that makes it obsolete.

<span id="напоминание-о-непроверенной-правке"></span>

### A reminder about an untested edit

After writing revision 1, the harness can insert “Run the test for
current revision 1.” A passing test for revision 1 removes it. If
revision 2 appears, the result for revision 1 no longer verifies the
current file.

<details id="harness-code-reminder">
<summary>Python: Replace or clear the verification reminder</summary>

~~~python
def update_reminders(slots, state):
    updated = dict(slots)
    current_checked = (
        state["tested_revision"] == state["revision"]
        and state["test_passed"] is True
    )
    if current_checked:
        updated.pop("verification", None)
    else:
        updated["verification"] = {
            "id": f"verify-revision-{state['revision']}",
            "source": "harness",
            "text": f"Run the test for current revision {state['revision']}.",
        }
    return updated

state = {"revision": 1, "tested_revision": None, "test_passed": False}
slots = update_reminders({}, state)
assert len(update_reminders(slots, state)) == 1
state.update(tested_revision=1, test_passed=True)
assert update_reminders(slots, state) == {}
state["revision"] = 2
assert update_reminders({}, state)["verification"]["id"] == "verify-revision-2"
~~~

</details>

The harness uses a single slot, so rebuilding the input does not
accumulate repeated reminders. A revision change gives the reminder
a new ID. A fresh passing test removes it. Text from a web page cannot
modify `slots` or change its own role by including the tag.

The reminder explains why the test is still needed. The verifier,
not the wording, prevents an early finish. Evaluate reminders on the
same tasks with model, tool set and budgets held fixed. Count early
finish attempts, verified success and extra calls and tokens. A test
that changes the model at the same time cannot isolate the reminder's
effect.

<span id="событийные-обработчики-или-hooks"></span>

### Hooks before and after an event

A harness may run event handlers before a tool call, after a result
or when the model tries to stop. A handler can check permissions,
supply context or both, depending on the event. A notification after
a write cannot prevent the write from having occurred.

In [Claude Code hooks](https://code.claude.com/docs/en/hooks#add-context-for-claude),
`additionalContext` supplies text for the next request. For example,
a `PostToolUse` hook can report a test command after an edit. Claude
Code wraps that text in a system reminder; blocking uses the event's
decision mechanism. A string pasted into a search result isn't a hook
event. When resuming, replayed reminders about a branch, version or
test time may be stale. Check current state rather than accepting every
replayed item as current. The example's `update_reminders` is a local
policy, not a hooks API.

<span id="8-выполнение-права-и-изоляция"></span>

## Where an action is authorized

Parse the call and validate its arguments, including their meaning
in the environment. Then determine whether the action is allowed,
denied or needs confirmation. Execute within the relevant sandbox.
Normalize the observation and update task state afterward.

Permissions are scoped to the operation and resource. A read can
contain secrets, and the executor needs to enforce paths at the
point of access. The January 2026 Codex CLI description distinguishes
local shell isolation from execution on an MCP server. A local sandbox
does not restrict an external service's process.

<span id="9-повторы-и-частичные-побочные-эффекты"></span>

## When an operation can be retried

A temporary 503 can justify a bounded retry if the operation allows it.
A bad argument needs a validation error the model can act on. A
permission denial is a stop for that action. A timeout after a write
requires checking the outcome, as in the checkpoint above.

Set separate limits for model steps, tool calls, time, tokens and cost.
Exhausting them leaves the task `incomplete`. Carry the counters in
the checkpoint so a restart doesn't reset the budget. If time expires
while a process is running, retain its ID and state. Missing output
doesn't mean the process stopped or that it should be started again.

<a id="harness-verified-completion"></a>
<span id="10-наблюдаемость-и-доказательство-завершения"></span>

## How to verify completion against the environment

A trace ties a model input to a proposed call, an authorization decision,
an observation and the resulting state. It can store the context and
compaction policy version, model and sampling settings, authorized and
executed arguments, allowed output references, latency, cost and what
the verifier found.

A context hash helps compare inputs; it cannot reconstruct them.
Replay needs the retained messages, tool versions and an environment
snapshot. Keep secrets out of public logs, and manage access to raw
and redacted traces separately. Even a complete log doesn't make
generation or external services deterministic.

The finish condition for the configuration change combines two
facts: the value is 45 and the required test passed for the current
revision. Specify that condition before the model tries to finish.
Otherwise a successful but unrelated check could be used as evidence.

<span id="работа-в-нескольких-сессиях"></span>

### Continuing in a separate session

In [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents),
the first session prepares the requirements, startup procedure and
handoff. Later sessions pick incomplete features, implement and verify
them, and record progress. The initial and ongoing agents use different
instructions; the design doesn't require different models or runtimes.

“Chat works” can mean several observable events: New Chat opens a
conversation, the message appears, an answer arrives, and the
conversation is still there after reopening. A unit test of the button
handler verifies only part of that sequence. Before accepting an old
`passes=true` mark, identify the revision it refers to and check the
current application. A progress artifact locates work; a version
history locates the change. Neither expands what the user allowed
the agent to commit or publish.

<span id="практика-и-первоисточники"></span>
<a id="harness-episode"></a>
<span id="исполняемый-цикл-и-продолжение-из-checkpoint"></span>

### The trace and a restart from a checkpoint

The fixture replays the five steps from the opening. It uses an
in-memory configuration and fixed proposals, so it tests the executor
without a model, filesystem edits or a real test runner. The input
contains instructions, history and fresh environment information.
The log is separate. The verifier consults the current fixture and its
test result, not the model's statement that the work is finished.
A checkpoint retains the history, call IDs and counters.

<details id="harness-code-episode">
<summary>Python: Reject an early finish, run the test and resume</summary>

~~~python
from copy import deepcopy
import hashlib
import json

def initial_state():
    return {
        "status": "in_progress", "reason": None,
        "model_steps": 0, "tool_calls": 0,
        "tested_revision": None, "test_passed": False,
        "history": [], "events": [], "used_call_ids": [],
    }

def build_context(state, env, can_write):
    return [
        {"role": "system", "content": "Use allowed tools and report verified facts."},
        {"role": "user", "content": "Set timeout to 45 and test the current revision."},
        *deepcopy(state["history"]),
        {"role": "developer", "content": {
            "current_revision": env["revision"], "can_write": can_write,
            "tested_revision": state["tested_revision"],
        }},
    ]

def run_episode(proposals, env, checkpoint=None, *,
                model_limit=8, tool_limit=4, can_write=True):
    state = deepcopy(checkpoint) if checkpoint is not None else initial_state()
    state.update(status="in_progress", reason=None)

    def event(kind, **payload):
        state["events"].append({
            "number": len(state["events"]) + 1, "kind": kind,
            **deepcopy(payload),
        })

    def stop(reason):
        state.update(status="incomplete", reason=reason)
        return state

    for proposal in proposals:
        if state["model_steps"] >= model_limit:
            return stop("model_budget")
        context = build_context(state, env, can_write)
        digest = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()
        event("model_input", sha256=digest)
        state["model_steps"] += 1
        state["history"].append({"role": "assistant", "content": deepcopy(proposal)})

        if proposal.get("kind") == "final":
            verified = (
                env["timeout"] == 45
                and state["tested_revision"] == env["revision"]
                and state["test_passed"] is True
                and proposal.get("timeout") == env["timeout"]
            )
            event("finish_check", revision=env["revision"], passed=verified)
            if verified:
                state.update(status="done", reason="verified")
                return state
            state["history"].append({
                "role": "developer",
                "content": "Goal or test of the current revision is not verified.",
            })
            continue

        if proposal.get("kind") != "tool":
            return stop("invalid_proposal")
        call_id = proposal.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in state["used_call_ids"]:
            return stop("invalid_call_id")
        state["used_call_ids"].append(call_id)
        name = proposal.get("name")
        args = proposal.get("arguments", {})
        if state["tool_calls"] >= tool_limit:
            observation = {"ok": False, "error": "tool_budget", "executed": False}
        elif name == "set_timeout" and not can_write:
            observation = {"ok": False, "error": "permission_denied", "executed": False}
        else:
            state["tool_calls"] += 1
            if name == "read_config":
                observation = {"ok": True, **deepcopy(env)}
            elif name == "set_timeout":
                if (not isinstance(args, dict)
                        or type(args.get("value")) is not int or args.get("value") != 45
                        or type(args.get("expected_revision")) is not int
                        or args.get("expected_revision") != env["revision"]):
                    observation = {"ok": False, "error": "precondition_failed"}
                else:
                    env.update(timeout=45, revision=env["revision"] + 1)
                    state.update(tested_revision=None, test_passed=False)
                    observation = {"ok": True, **deepcopy(env)}
            elif name == "run_test":
                state.update(tested_revision=env["revision"],
                             test_passed=env["timeout"] == 45)
                observation = {"ok": True, "revision": env["revision"],
                               "passed": state["test_passed"]}
            else:
                observation = {"ok": False, "error": "unknown_tool"}
        event("tool_result", call_id=call_id, name=name, result=observation)
        state["history"].append({
            "role": "tool", "call_id": call_id, "content": observation,
        })
        if observation.get("error") in {"tool_budget", "permission_denied"}:
            return stop(observation["error"])
    return stop("no_more_proposals")

proposals = [
    {"kind": "tool", "call_id": "c1", "name": "read_config"},
    {"kind": "tool", "call_id": "c2", "name": "set_timeout",
     "arguments": {"value": 45, "expected_revision": 0}},
    {"kind": "final", "timeout": 45},
    {"kind": "tool", "call_id": "c3", "name": "run_test"},
    {"kind": "final", "timeout": 45},
]
env = {"timeout": 30, "revision": 0}
complete = run_episode(proposals, env)
assert complete["status"] == "done"
assert complete["model_steps"] == 5 and complete["tool_calls"] == 3
checks = [e["passed"] for e in complete["events"] if e["kind"] == "finish_check"]
assert checks == [False, True]

env = {"timeout": 30, "revision": 0}
partial = run_episode(proposals[:2], env)
assert partial["status"] == "incomplete" and env["revision"] == 1
saved = json.loads(json.dumps(partial))
resumed = run_episode(proposals[2:], env, checkpoint=saved)
assert resumed["status"] == "done" and env["revision"] == 1
assert resumed["model_steps"] == 5 and resumed["tool_calls"] == 3
~~~

</details>

The first attempt to finish is recorded but rejected because the
write has no associated test. The same verifier accepts the answer
after `run_test`. In the restart case, the episode ends after the
write. Resuming uses the checkpoint but consults the current `env`
instead of recreating it from the summary. There is no second write;
the revision stays at 1.

The fixture includes two limits, a write permission, artifact
revisions and a completion check. It does not supply a production
schema parser, event store, sandbox, running-tool timeout, tokenizer
or automatic compaction. Its messages aren't an SDK request. The
test says nothing about how often a model would choose the right action.

<span id="реализации-для-разбора"></span>

## Sources

The LangGraph and MCP notebooks are listed in the
[[05 Источники/Courses/GenAI Agents|GenAI Agents catalogue]].

<span id="первоисточники-и-дальнейшее-чтение"></span>

### Primary sources and further reading

- [OpenAI: Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/).
- [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
- [Anthropic: Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents).
- [SWE-agent v3](https://arxiv.org/abs/2405.15793v3), Appendix C, Figures 26–32.
- [OpenHands Context Condenser](https://docs.openhands.dev/sdk/guides/context-condenser).
- [[05 Источники/Papers/The Complexity Trap|Lindenbauer et al., The Complexity Trap]].
- [Hugging Face Context Engineering Course](https://huggingface.co/learn/context-course/unit0/introduction).
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/).
- [OpenAI: Compaction](https://developers.openai.com/api/docs/guides/compaction).
- [OpenAI: Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
- [Claude Code hooks](https://code.claude.com/docs/en/hooks).

Previous: [[en/00 Textbook/17 Tools and Agents/65 How a function call becomes an action|How a function call becomes an action]].
Next: [[00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Memory and planning]].
