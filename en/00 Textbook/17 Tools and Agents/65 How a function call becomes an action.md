---
title: How a function call becomes an action
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию.md"
canonical_revision: "5ff393e9a0d41fad287e6502f3be4432fe5ed729d08044376ac311ae630799f5"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/17 Tools and Agents/00 How agents use tools]]"
next: "[[00 Учебник/17 Tools и Agents/66 Agent harness и context engineering]]"
reference_terms:
  - label: Tool calls — schema, permissions, and result verification
    label_en: Tool calls — schema, permissions, and result verification
    anchor: tool-call-contract
  - label: Toolformer — call selection and evaluation
    label_en: Toolformer — call selection and evaluation
    anchor: toolformer-curation-evaluation
  - label: MCP — client, server, and protocol boundaries
    label_en: MCP — client, server, and protocol boundaries
    anchor: mcp-host-client-server
  - label: MCP — response correlation and two kinds of error
    label_en: MCP — response correlation and two kinds of error
    anchor: mcp-response-correlation
  - label: ReAct — prompting, fallback, and fine-tuning
    label_en: ReAct — prompting, fallback, and fine-tuning
    anchor: react-prompting-learning
  - label: Idempotency — retries after a lost response
    label_en: Idempotency — retries after a lost response
    anchor: tool-retry-idempotency
  - label: Tool use — verifying episode completion
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

# How a function call becomes an action

“Read the first two lines of `notes.txt` and tell me the value of
`timeout`.” The model proposes `read_file` with a starting line of 0.
The tool numbers lines from 1, so the application rejects the call without
opening the file. The model receives the error and corrects the argument.
Only after a successful read can the application verify the answer “30”
against the contents of the file.

The model proposes a function name and arguments. The application code
around the model, often called the *harness*, checks the proposal and
runs the tool if the call is allowed. A successful read, a permission
denial, and a failure in the executor must give the model different
observations: each leaves it with a different set of possible next steps.

<span id="2-интерфейс-инструмента-является-частью-задачи"></span>

## What the application checks before execution

<a id="tool-call-contract"></a>
<span id="один-запрос-через-схему-ошибку-и-проверку"></span>

### A read request, from the schema to the result

The fixture contains `timeout=30` and `retries=2`. The tool takes a path
and an inclusive line range.

<details id="tool-use-code-schema">
<summary>The read_file schema</summary>

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

The first proposal uses zero-based numbering.

<details id="tool-use-code-proposal">
<summary>The proposed line range: 0–2</summary>

~~~json
{
  "path": "notes.txt",
  "start_line": 0,
  "end_line": 2
}
~~~

</details>

The application returns the field error in an observation. This format
belongs to the example application, not to JSON Schema or MCP.

<details id="tool-use-code-rejected">
<summary>Observation after rejecting c1</summary>

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

The next proposal changes `start_line` to 1 and keeps the path and end line.
The order of the bounds is a separate check. `start_line=2, end_line=1`
satisfies the field schema because both values are at least 1. The executor
still has to check the order, the length of the range, which file would be
opened, and whether access to that file is allowed.

<details id="tool-use-code-observation">
<summary>Observation after reading c2</summary>

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

A `permission_denied` observation would stop execution rather than turn
access control into another search problem. The test already knows the
contents of `fixture-v1`. It checks the revision, line range and truncation
flag, then compares the answer with the value after the exact key
`timeout=`. Guessing “30” without reading both lines does not pass this
test. The same trace appears in the [[#tool-use-verification|runnable example]]
at the end of the chapter.

On a real filesystem, checking a path's string prefix is insufficient.
The executor must enforce the boundary when it resolves and opens the
object, including when symbolic links exist or the filesystem changes.
The fixture uses a dictionary; it doesn't exercise a filesystem sandbox.

### What belongs in a tool description

The `read_file` description says that line numbers start at 1 and both
bounds are included. The schema checks types and individual fields;
the executor handles constraints that depend on several fields or on the
state of the environment. A search description should state what
`top_k` means, which filters are accepted and how to request the next
page. A tool that searches titles and abstracts should not promise a
full-text search.

Anthropic's [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
evaluates whether the model can choose the right tool, supply arguments,
and use the result in the next step. An unspecified `mode` makes the
model guess even after it has picked the right tool. Give `search` and
`lookup` distinct jobs: search returns matches to a query; lookup takes
a known ID and returns one record. A `not_found` result may justify
checking the ID. A `permission_denied` result does not grant access.

<span id="1-что-добавляет-инструмент"></span>

## How the observation enters the next request

A language model generates a completion $y$ conditioned on context $c$:

$$y \sim p_\theta(y\mid c).$$

Some completions represent proposed actions rather than answers:

$$a_t=(\text{name},\text{arguments}).$$

If the application admits the action, the environment supplies an
observation $o_t$. If the application rejects the proposal, it can
report the rejection without contacting the environment. The trace
$(c,a_1,o_1,\ldots,a_t,o_t)$ provides material for the next model
request. The application can choose which parts to include rather than
send the entire trace each time.

The [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure)
calls the cycle Think → Act → Observe. Here `c1` tells the model which
argument needs correction; `c2` adds facts that weren't in its
proposal. The [ReAct](https://react-lm.github.io/) traces below show how a
search result can change the next action.

<a id="react-hotpotqa-trajectories"></a>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tool-use-2026-10/react-hotpotqa-primary.png|Four traces for a question about Apple Remote: standard prompting, reasoning only, acting only, and ReAct.]]

*Shunyu Yao et al., [ReAct v3, Figure 1(1), p. 2](https://arxiv.org/pdf/2210.03629v3#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
First panel of the original figure. See also the
[project's comparison of the three loops](https://react-lm.github.io/files/diagram.png).*

In the reason-only column, the model reasons without receiving any new
observation. The act-only column records actions and responses without
written reasoning between them. ReAct records `Thought` and `Act`; the
environment provides `Obs`. These are prompt formats from the paper.
The absence of written reasoning does not reveal which computations a
model performed internally.

<span id="4-наблюдение-должно-отвечать-на-следующий-вопрос"></span>

## What the model needs back from a tool

The response to `c2` includes the path, range and lines. Search needs a
title, passage, source and continuation cursor. A create operation should
return the new object's ID and the fields it can confirm. A command result
needs an exit status, output and a truncation indicator. Which fields
belong in the observation depends on the next decision.

In the [HotpotQA example above](#react-hotpotqa-trajectories), the question
asks which device other than the Apple Remote could control the program
for which the remote was first designed. The answer needs two facts: the
program's name and another way to control it.

1. `Search[Apple Remote]` returns a description of the remote and names
   Front Row. This identifies the next query, but doesn't answer the question.
2. `Search[Front Row]` doesn't find an exact page. The environment suggests
   related titles, including `Front Row (software)`.
3. `Search[Front Row (software)]` returns a description of the program
   that mentions the remote and keyboard function keys.
4. `Finish[keyboard function keys]` answers with the second means of control.

The act-only trace retrieves similar material but finishes with `yes`.
Having evidence in the context does not guarantee a correct answer.
The ReAct trace uses it to choose a new search and connect the program to
the device. The figure shows individual runs from the paper, not a
reproduction against today's search service.

<span id="поиск-объекта-в-частично-наблюдаемой-среде"></span>

### Finding an object in a partially observed environment

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tool-use-2026-10/react-alfworld-primary.png|In ALFWorld, acting only repeats a failed pickup at the sink. ReAct finds the pepper shaker on a countertop and puts it in a drawer.]]

*Shunyu Yao et al., [ReAct v3, Figure 1(2), p. 2](https://arxiv.org/pdf/2210.03629v3#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Lower panel of the original figure, showing the ALFWorld trajectories.*

In ALFWorld, the agent must find a pepper shaker and put it in a drawer.
The sink observation lists a sponge, a spatula and a spoon. The act-only
trace tries to pick up the pepper shaker there anyway. The environment
returns `Nothing happens`, and a repeat yields the same result.

The ReAct trace looks in several places and finds the shaker on
`countertop 3`. The environment confirms that the agent took it.
The drawer is closed when the agent arrives, so it has to open the drawer
before putting the shaker in. The final observation confirms the new
state. The figure omits some repetitive search steps; it shows enough
to distinguish finding the object, taking it, opening the drawer and
putting the object away.

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
<summary>Python: Decide when to fall back to ReAct</summary>

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

![[00 Учебник/Assets/Figures/curated/agent-reconciliation-2026-10-30/react-cot-sampling.png|Two plots compare HotpotQA and FEVER results as the number of CoT-SC samples changes, with and without a switch to ReAct.]]

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

![[00 Учебник/Assets/Figures/curated/agent-reconciliation-2026-10-30/react-prompt-vs-finetune.png|Prompting results on the left and fine-tuning on trajectories on the right. The right panel has no result at 540B.]]

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

<span id="объём-и-полнота-ответа"></span>

### How much of the result to return

The search response `{"items":[{"id":"p7","title":"ReAct"}],"next_cursor":"page-2"}`
doesn't establish that the collection contains only one paper. The cursor
belongs to the same query and result set. When the query or data snapshot
changes, a new cursor may be needed. If a later tool takes a document ID,
a concise search response still has to include that ID. If it takes a
passage address, the response needs to preserve one. Removing a field
that determines the next argument forces the model to guess. Anthropic
covers these trade-offs in [Returning meaningful context from your tools](https://www.anthropic.com/engineering/writing-tools-for-agents#returning-meaningful-context-from-your-tools).

Two independent reads can run in parallel if the application
[[#mcp-response-correlation|matches each response to its request]].
A lookup that needs an ID from a search must wait for the search result.
Even two read-only calls don't guarantee one consistent snapshot if the
underlying data changes between them.

<span id="3-вызов-функции-генерация-кода-и-текстовые-действия"></span>

## How the model represents an action

The file example uses a named function and an argument object. ReAct
can represent an action as text such as `Search[Front Row]`. The
representation affects what the application must parse and check.

| Representation | What changes |
|---|---|
| JSON function call | The schema checks argument types and fields, not their intended meaning. |
| Generated code | A fragment can compose several operations; the executor must restrict its capabilities. |
| Text command | A small grammar is enough for some tasks; ambiguous free text is harder to parse. |
| Token markers | Training can teach the model to emit markers directly; the format depends on the vocabulary and training data. |

A code-based agent can use a loop instead of producing ten consecutive
function calls. Running its output requires isolation from sensitive
files, services and credentials, as well as limits on time and memory.
The protocol for requesting code does not itself provide that isolation.

<span id="как-toolformer-получает-обучающие-примеры"></span>

### How Toolformer selects training examples

Consider the sentence “Pittsburgh is also known as the Steel City.”
Two plausible queries ask what Pittsburgh is called and which country
it's in. Both responses can be correct, but only “Steel City” helps
predict the particular continuation immediately after the call.

[Toolformer, §2](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf#page=3)
uses demonstrations to propose candidate calls at likely positions.
An executor calls the APIs. The data pipeline then retains calls whose
results reduce the prediction loss on the original continuation. The
authors illustrate the process in
[Figure 2](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf#page=2).

Write $e(c,r)$ for call $c$ serialized with response $r$.
The filter evaluates the same target tokens with no call, with a call
but no response, and with the complete call and response. It weights
the negative log probabilities of the target tokens:

$$
\begin{aligned}
L_i(z)&=-\sum_{j=i}^{n}w_{j-i}
             \log p_M(x_j\mid z,x_{1:j-1}),\\
L_i^+&=L_i(e(c,r)),\\
L_i^-&=\min\{L_i(\varnothing),L_i(e(c,\varnothing))\},\\
\Delta_i&=L_i^- - L_i^+.
\end{aligned}
$$

A call is retained when $\Delta_i\ge\tau_f$. The minimum of the two
baseline losses tests whether the response helps beyond the question
the model asked itself. During scoring, the API text is prefixed to the
document; inserting an unfamiliar marker into a sentence could itself
disturb the original model. Only after selection is the call inserted
at its chosen position in the training sequence.

The closest target tokens receive weights proportional to
$1,0.8,0.6,0.4,0.2$; subsequent tokens receive zero. The probabilities
in this example are assigned for illustration. They weren't measured
from GPT-J.

<details id="tool-use-code-loss">
<summary>Python: Compare four continuation losses</summary>

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

The four losses are 1.6059 with no call, 1.5474 with the question only,
0.2449 with the helpful result and 2.9957 with the irrelevant one.
The baseline is 1.5474. The helpful result lowers it by about 1.3025
and clears a threshold of 0.5; the irrelevant result increases the loss
and is rejected.

A retained example can read
`Pittsburgh is also known as [QA(...)->Steel City] the Steel City`.
The augmented corpus uses the standard language modeling objective.
At inference, generation pauses at the response delimiter, the executor
inserts the actual response and the closing delimiter, and decoding resumes.
The paper writes `<API>`, `</API>` and an arrow in the notation; the
experiment uses existing token sequences rather than extending the
vocabulary.

Toolformer changes the model's weights to teach call placement.
The main prompting experiments in ReAct instead use a frozen PaLM model.
A smaller language modeling loss does not validate permissions, factual
accuracy or side effects. “United States” can be filtered out even though
it is true, while an incorrect result that happens to predict the right
tokens can be retained. The original tools do search, retrieval,
arithmetic, translation and calendar lookup. The filter is not a rule
for selecting calls that write data or make payments.

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
<summary>Python: Select positions and trigger tool calls</summary>

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

## How an application calls a tool through MCP

If `read_file` runs on an MCP server, the host application uses its
MCP client to reach it. With `tools/list`, the client learns the name,
description and `inputSchema`. With `tools/call`, it sends the checked
arguments to the server, which runs the tool and responds. The host
decides whether to expose the tool to the model and whether to allow a
particular call.

![[00 Учебник/Assets/Figures/curated/tool-use-2026-11-09/mcp-simple-diagram.png|Applications are on the left and tools and data sources on the right. MCP specifies how they exchange messages.]]

*Anthropic and the Model Context Protocol contributors,
[FAQ: How does MCP work?](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/27ab2419191334dd288b05343e3a622ed615913d/docs/faqs.mdx#how-does-mcp-work),
[original diagram](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/27ab2419191334dd288b05343e3a622ed615913d/docs/images/mcp-simple-diagram.png),
[MIT](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/27ab2419191334dd288b05343e3a622ed615913d/LICENSE).*

An IDE on the left can connect to the filesystem or Git servers on the
right. It has a separate client connection to each server. The central
MCP box is the protocol, not a process that receives every request.
The host chooses what data to pass to the model. The specification
describes the roles in [Architecture](https://modelcontextprotocol.io/specification/2025-11-25/architecture#core-components)
and the requests in [Tools, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

Here the client sends the corrected request for lines 1–2.

<details id="tool-use-code-mcp-request">
<summary>A tools/call request with ID 17</summary>

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

The JSON-RPC ID `17` matches a response to this request. The host may
need to associate it with another ID in the model API. A successful
result can contain textual `content` and an object named
`structuredContent`. If the server advertises an `outputSchema`, its
structured result must conform to it. That says nothing about whether
the model selected the right arguments or the returned facts are true.
Likewise, `readOnlyHint` is an annotation, not enforcement. The
specification instructs clients to treat annotations as untrusted
unless they came from a trusted server.

<a id="mcp-response-correlation"></a>
<span id="если-ответы-приходят-не-по-порядку"></span>

### Responses can arrive out of order

The model asks to read `notes.txt` and `settings.txt`. The host sends
two requests in the same MCP session and records which model call each
JSON-RPC request ID belongs to. A requester must not reuse an ID
within the same session. See [MCP 2025-11-25, Messages](https://modelcontextprotocol.io/specification/2025-11-25/basic#messages).

| Arrival order | Corresponding call |
|---|---|
| `18` first: missing file, `isError: true`. | `c3`, read `settings.txt`. |
| `17` second: `timeout=30` and `retries=2`. | `c2`, read `notes.txt`. |

Assigning the first response to the first request says that `notes.txt`
is missing and that `settings.txt` contains `timeout=30`. Matching by ID
keeps both results attached to the right call. The outer envelope for
`18` has a `result`: JSON-RPC succeeded, but the tool failed. An
unknown tool name such as `read_old_file` can produce an outer
`error` instead. The host needs both distinctions to choose its response.

The example below assumes the MCP client has already parsed the message.
It maps a response from one session to a log entry with the right model
call ID. A duplicated or unknown response doesn't consume a pending call.
Full validation of the wire message belongs to the client.

<details id="tool-use-code-mcp-correlation">
<summary>Python: Correlate out-of-order MCP responses</summary>

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

The log records `c3` first and then `c2`. Separate sessions have separate
pending tables: two sessions can both use `id: 17` without referring to
the same call. There are three kinds of identity here. `c2` attaches a
result to a model call. `17` correlates a request and response on the
connection. An [[#tool-retry-idempotency|operation key]] identifies the
same intended effect across retries. If the result of a write is lost,
the retry uses a new request ID but retains the operation key if the
server supports it.

<span id="5-ошибка--часть-протокола"></span>

## What to do after a failed call

| Outcome | Next step |
|---|---|
| A field fails validation. | Return the field name and constraint. |
| The server rate-limits a request. | Check that retrying is safe and respect the delay and attempt limit. |
| The object's version changed. | Read the new version before choosing an action. |
| The operation isn't authorized. | Stop. Request confirmation only where policy permits it. |
| The reply to a write was lost. | Check the operation's status or another authoritative record. |
| A required object doesn't exist and cannot be created. | End with an explanation based on the observation. |

<a id="tool-retry-idempotency"></a>
<span id="потерянный-ответ-после-записи"></span>

### A lost response after a write

Suppose `create_note` stores a note and the client times out before
the reply arrives. The timeout only tells the client that it didn't
receive a reply in time. Sending a fresh create request may produce a
duplicate.

The client keeps an identifier for the intended operation and sends the
same key on a retry. The server associates the caller and key with the
original parameters and created object. If both match, it returns the
original result; changing the parameters while reusing the key should
produce a conflict. See [AWS Builders' Library: Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/).

This example runs in a single process without files, a network or a
model. `lose_reply` simulates losing the reply after saving the note.

<details id="tool-use-code-retry">
<summary>Python: Retry note creation without duplicating the note</summary>

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

A retry with a new key looks like a new request. Deduplicating by a text
hash alone would also collapse two intentionally identical notes into
one. The operation key represents the user's intent. A JSON-RPC request
ID only correlates messages.

The sample runs sequentially. A real service must record the object and
deduplication entry atomically and retain that entry for the accepted
retry window. A crash between writes or a retry after expiry can
otherwise reintroduce duplicates. If the service cannot guarantee
deduplication and no authoritative record establishes what happened,
the application should preserve the outcome as unknown rather than
report either success or failure as confirmed.

<span id="6-права-нельзя-задать-одной-инструкцией-модели"></span>

## Where permissions are enforced

“Do not delete files” can influence the model's proposal but does not
prevent a system call. The parser accepts only the declared format;
the validator checks values and normalized paths. An authorization
check matches the operation and its target to permissions or an
explicit confirmation. The sandbox limits what the process can do;
the log records the decision and execution result.

Tool output remains data. If a page from a library suggests making a
reservation, its text does not supersede “Don't reserve anything.”
The application keeps the origin of the observation and checks any
proposed action against the original permissions.

Confirmation applies to a concrete operation. Approval to create a
note in one project does not allow the tool to send it to another
service. Before execution, check the operation, target and relevant
parameters. A change to the target needs a new authorization decision.
See the [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|chapter on agent security]]
for the threat model.

<span id="7-когда-workflow-лучше-свободного-агента"></span>

## When the order can be fixed in advance

[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
distinguishes workflows with predefined paths from agents that let the
model choose what to do next. A workflow makes possible transitions
explicit. Its cost still depends on the number of model and tool calls.
A model-led loop helps when the observation changes the next step and
the number of steps isn't known in advance.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-evaluation-2026-11-10/toolsandbox-roles.png|The user talks to the agent on the left; on the right, the agent calls tools and the execution environment sends results back.]]

*Jiarui Lu et al., [ToolSandbox v2, Figure 2](https://arxiv.org/html/2408.04682v2#S2.F2),
[original](https://arxiv.org/html/2408.04682v2/architecture_diagram.png),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

In [one ToolSandbox trace](https://arxiv.org/html/2408.04682v2#S1.F1),
the user asks the agent to send a message. The first `send_message`
call produces a `ConnectionError` because cellular service is off.
The agent enables cellular service and tries again. All three calls
take place on the right side of the diagram without another user turn.
If the recipient or message were missing, a clarifying question would
go to the user on the left. The
[Anthropic agent diagram](https://www-cdn.anthropic.com/images/4zrzovbb/website/58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png)
shows the same feedback loop without the benchmark's roles.

Reading a fixed range and parsing a value have a known order: authorize
the path, read, parse, respond. In an unfamiliar repository, a test
failure can determine which file to examine next. A single system can
use a model-led search to find a cause and a fixed sequence to validate
the resulting change. A step limit can stop either loop without proving
success. A read requires evidence of the value; a code fix needs the
test that failed before the change.

<span id="где-начинается-агент-модель-память-действие-и-среда"></span>

### What the model knows before execution

<!-- source_unit_id: meeting-01-intro-reasoning-agent-progress -->

The [Berkeley Advanced LLM Agents course](https://rdi.berkeley.edu/adv-llm-agents/sp25)
separates the LLM, planning, retrieval, memory, external tools and
feedback on physical slides 5–6 of the introduction. Memory carries
selected state between calls; observations inform subsequent decisions.
A long model response can revise a plan. It does not supply the missing
lines of a file or the state of a remote service. Even a successful read
may be partial or stale: the first page of results cannot prove that
no other pages exist.

<span id="два-способа-проектировать-агента-llm-first-и-agent-first"></span>

### Designing from the model or from the task

<!-- source_unit_id: meeting-03-slides-yu-su-agent-first-vs-llm-first -->

In lecture 3 of the same [Berkeley course](https://rdi.berkeley.edu/adv-llm-agents/sp25),
Yu Su contrasts two design starting points (physical slides 5–15).
An LLM-first design starts with the capabilities of an existing model,
then builds the prompt, tools and surrounding program. This is useful
for prototyping, but unresolved task rules can end up buried in the
prompt. An agent-first design describes the observations, permitted
actions, persistent state, success conditions and irreversible changes
first. It then assigns the model a role, such as interpreting an
observation, choosing an action or explaining a result.

For `read_file`, the first approach asks whether the model can follow
the tool description. The second asks which files may be read, when
a read is fresh enough, and how to verify the answer. Both need the
executor to enforce permissions. Slide 7 treats reasoning as an action
within the internal environment: from “the drawer is closed,” the
model can plan to open it. Only an external action can change the
drawer, and a new observation can confirm the change.

<span id="8-как-проверять-tool-use"></span>

## How to verify task completion

Measure tool selection, argument validity, environment state and side
effects separately. An evaluation can also measure recovery from an
error, the number of calls and confirmations, and whether the log is
sufficient to replay the trace. Well-formed JSON with a disallowed path
isn't a successful call. A successful read followed by a wrong answer
isn't a successful episode.

<a id="tool-use-verification"></a>
<span id="полный-учебный-эпизод-с-проверяемым-завершением"></span>

### A complete local trace

The executor below receives three predetermined proposals: a wrong line
range, the corrected call and the final answer. Error observations here
contain only an error type. An application talking to a model can also
include the field and its constraint, as in the first trace. `seen_ids`
rejects repeated call IDs within the episode.

<details id="tool-use-code-episode">
<summary>Python: Verify the read-and-answer episode</summary>

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

The first log entry has `executed=False`. The second contains the
two lines and their revision, `fixture-v1`. The verifier accepts
`30` only after a matching read. It rejects the same text if there was
no read or if the episode stopped after the first rejected proposal.
Its rule is specific to this task: another file, range or answer
format would require another rule.

In a deployed loop, the next model request would include the relevant
tool calls and observations. The supplied `proposals` replace model
generation so that the example needs no API key. The test checks
the executor and verifier on these inputs, not the accuracy of a model
or the strength of a real filesystem sandbox.

<span id="практика-и-первоисточники"></span>
<span id="реализации-для-разбора"></span>

## Sources

The function-calling, MCP and LangGraph notebooks are available separately
in the [[05 Источники/Courses/GenAI Agents|GenAI Agents source catalogue]].

<span id="первоисточники-и-дальнейшее-чтение"></span>

### Primary sources and further reading

- [Hugging Face Agents Course, Unit 1](https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure) explains the Thought–Action–Observation loop.
- [ReAct](https://react-lm.github.io/) shows the comparison traces from the [paper](https://arxiv.org/abs/2210.03629).
- [Toolformer](https://arxiv.org/abs/2302.04761) trains a model to call APIs from automatically filtered examples.
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) covers tool descriptions and observations.
- Model Context Protocol 2025-11-25: [Architecture](https://modelcontextprotocol.io/specification/2025-11-25/architecture), [Messages](https://modelcontextprotocol.io/specification/2025-11-25/basic#messages) and [Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).
- [AWS Builders' Library: Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) explains how to scope operation identity and handle retries.
- [SWE-agent: Agent-Computer Interfaces](https://arxiv.org/abs/2405.15793) studies how the computer interface affects task success.

The section on [[00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers#agent-reward-design|training with verifiable rewards]]
covers matching a tool call to a request and checking the environment
after the call. The [[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|agent evaluation chapter]]
measures call validity, task outcome, policy compliance and repeatability
separately.

Previous: [[en/00 Textbook/17 Tools and Agents/00 How agents use tools|How agents use tools]].
Next: [[00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|Harness and context engineering]].
