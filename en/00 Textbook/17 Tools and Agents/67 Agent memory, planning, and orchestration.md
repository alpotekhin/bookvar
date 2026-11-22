---
title: Agent memory, planning, and orchestration
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md"
canonical_revision: "d79dc8a9598eb882e7977641633bebdd78917afbff8e2569d00f86caacc112d6"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/17 Tools and Agents/66 How agents manage context]]"
next: "[[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем]]"
reference_terms:
  - label: Agent memory — record revisions and provenance
    label_en: Agent memory — record revisions and provenance
    anchor: agent-memory-records
  - label: Memory retrieval — recency, importance, and relevance
    label_en: Memory retrieval — recency, importance, and relevance
    anchor: agent-memory-ranking
  - label: Planning — preconditions and branching search
    label_en: Planning — preconditions and branching search
    anchor: agent-planning-branches
  - label: Reflexion — feedback between attempts
    label_en: Reflexion — feedback between attempts
    anchor: agent-reflexion-feedback
  - label: Orchestration — checking the inputs a worker result depends on
    label_en: Orchestration — checking the inputs a worker result depends on
    anchor: agent-worker-dependencies
primary_sources:
  - https://arxiv.org/abs/2405.01534v1
  - https://arxiv.org/abs/2310.08560
  - https://arxiv.org/abs/2303.11366
  - https://arxiv.org/abs/2304.11477
  - https://arxiv.org/abs/2305.10601
  - https://arxiv.org/abs/2310.04406
  - https://www.anthropic.com/engineering/building-effective-agents
  - https://cs329a.stanford.edu/
  - https://arxiv.org/abs/2304.05128
  - https://arxiv.org/abs/2310.01798
  - https://arxiv.org/abs/2411.06559
  - https://arxiv.org/abs/2304.03442
  - https://arxiv.org/abs/2405.14831
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
source_unit_id:
  - meeting-06-slides-plan-sequence-learn
  - meeting-01-slides-reflection-and-self-debugging
  - meeting-01-slides-self-correction-failure-modes
  - meeting-01-reading-02-catalogue-record
  - meeting-01-reading-03-catalogue-record
  - meeting-03-slides-memory-takeaway
  - meeting-03-slides-planning-paradigms
  - meeting-03-slides-yu-su-world-model-planning-sequence
  - meeting-03-slides-planning-takeaways
  - meeting-03-slides-parametric-memory-appendix
  - meeting-03-reading-03-catalogue-record
---

# Agent memory, planning, and orchestration

The organizer moves a research seminar from Thursday at 18:00 to
Friday at 19:00. The assistant has a confirmation for room 204 on
Thursday and a reading list for the speaker. It no longer has a room
for the current date. The reading list is still useful if the topic
hasn't changed.

The assistant records Friday and identifies which work needs to be
redone. A draft invitation and a room search can proceed in parallel,
but sending waits for a confirmed room and permission to send. The
seminar shows what memory, planning and orchestration do at different
points in the same task.

<span id="1-память-находится-вне-весов"></span>
<span id="1-внешняя-память-агента"></span>

## Which information survives a break?

A model call sees the tokens in its current input. A month-old
conversation has no effect merely because the application stored it.
The relevant entry must be retrieved into context. Training changes
model weights; building a request changes context. Writes change the
external memory store, and tool actions can change the environment.

“Room 204 was booked” is a record of what the assistant knew at the
time. The calendar entry might have changed. Memory helps locate the
booking; before sending the invitation, read its current room and
time from the calendar. This distinction still applies if both sets
of records happen to live in one database.

<span id="2-иерархия-памяти-memgpt"></span>

## How MemGPT brings information back into context

[MemGPT v2, §§2.1–2.4](https://arxiv.org/html/2310.08560v2#S2)
organizes limited context like a memory hierarchy. The prompt contains
system instructions, an editable working context and a recent message
queue. The archive and full message history sit outside the prompt.
Tool results put retrieved records into the queue. This is the system
in the February 2024 paper, not a description of today's Letta API.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-memory-review-2026-09/memgpt-figure3-memory-hierarchy.png|MemGPT moves information between the context queue, archival storage and external message history.]]

*Charles Packer et al., [MemGPT v2, Figure 3, p. 3](https://arxiv.org/pdf/2310.08560v2#page=3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original diagram without external page and caption text.*

Recall storage retains incoming messages, replies and calls. Archival
storage contains explicitly saved material and ingested documents.
Putting a preference in working context makes it visible in subsequent
requests. Putting a long letter in the archive preserves it for a
later search instead.

When the queue nears its limit, a warning gives the model a chance
to store important details. Section 2.2 illustrates a warning at 70%
of the window, not a universal threshold. Queue eviction can replace
old messages with a recursive summary made from the previous summary
and the evicted messages. Full messages remain in recall storage.
Retrieved messages join the queue, and pagination limits the amount
returned by one search.

Suppose the assistant remembers an offer from a venue but not its
terms. It searches for the venue and rental, gets a page, and requests
another if the first page lacks the letter. The reply containing the
price still needs to be checked against the offer's expiry date.
In the paper, setting `request_heartbeat=true` on a tool call asks
the runtime to continue with another model call after the tool response.
Without that request it yields until another event. The flag does not
increase the context window or start continuous background reasoning.

The seminar goal and date may fit in working context; venue correspondence
belongs in the archive. Retrieval can still pick the wrong letter.
A summary can omit “valid through Friday,” which is why the original
message remains necessary.

<span id="3-какие-виды-памяти-действительно-нужны"></span>

## How facts, examples and procedures differ

Semantic memory records facts, such as room capacity and an evening
preference. Episodic memory retains a particular rejection and its date.
Procedural memory describes how to submit the application. Working
memory holds current candidates and unanswered requests. These roles
do not require four databases. They require different update rules:
one missed deadline doesn't establish that a venue never accepts a
seminar, and a preference for evenings doesn't override approval for
a morning slot on this occasion.

<span id="generative-agents-от-отдельных-событий-к-обобщению"></span>

### Generative Agents: from observations to reflections

[Generative Agents, §4](https://arxiv.org/html/2304.03442v2#S4)
stores observations from a simulated town with creation and access
timestamps. Retrieval considers relevance, recency and importance.
Reflection makes new inferences from retrieved records and stores
links to the observations that support them. Plans also enter memory
and are elaborated from a daily schedule into smaller actions.

A count such as “three conversations today” compresses records.
A reflection such as “the attendees want remote access” infers a
pattern that can change the proposed format. It could also reflect
a joke or a request for an earlier seminar. Keep the inference
separate from what was observed.

When the sum of recent events' importance crosses the paper's
illustrative threshold of 150, the model uses the latest 100 records
to propose three questions. It retrieves supporting memories and
records answers along with their sources. In the
[example in §4.2](https://arxiv.org/html/2304.03442v2#S4.SS2),
Klaus chooses the frequently encountered Wolfgang without reflection,
and Maria after inferring that they share research interests.
[Figure 7](https://arxiv.org/html/2304.03442v2/figures/figure_reflection6.png)
shows how observations support one such inference about Klaus.
If two reflections refer to one conversation, they do not provide two
independent observations. An invalidated source needs a review of its
derived inferences. The simulation measures believable behavior, not
calendar freshness or verified bookings.

<span id="4-конвейер-записи"></span>
<a id="agent-memory-records"></a>

## Recording a new date without losing the previous one

“Thursday at 18:00” and “moved to Friday at 19:00” are both likely matches
for “when is the seminar?” A similarity score need not put the current
value first. Keep both events in the log, but mark which value is current.

<details id="memory-code-record">
<summary>A record that supersedes the old date</summary>

```json
{
  "id": "seminar-42-start-v2",
  "subject": "seminar-42",
  "field": "start_time",
  "value": "2026-10-09T19:00:00+03:00",
  "source": "message-184",
  "recorded_at": "2026-10-01T12:20:00+03:00",
  "supersedes": "seminar-42-start-v1",
  "status": "confirmed"
}
```

</details>

`subject` scopes the update to one seminar. `source` points to the message,
and `supersedes` explains why the old value no longer applies. A newly
received message may quote an old offer, so recording time and effective
time are different. Compare facts in the same scope before creating
a new version; a second mention can corroborate the first, whereas a
conflict may need clarification. An inference cannot silently replace a
confirmed fact. Secrets and instructions from untrusted documents should
not flow automatically into persistent memory.

Changing one entry can leave stale values in a search index, summary and
draft. Derived views must refer to the current record or be refreshed.
Similarly, removing a fact from a table does not remove its old copy in
an index or summary.

<span id="5-извлечение-не-только-nearest-neighbors"></span>
<span id="5-как-выбрать-воспоминания-для-текущего-решения"></span>
<a id="agent-memory-ranking"></a>

## Choosing records for the next action

A room search needs date, attendance and equipment. Writing to the speaker
needs correspondence and unresolved questions. Filter candidates by user,
project, event and time range first. Use an exact search for a booking ID,
text search for a rare name and a vector score to connect “remote access”
with “will it be streamed?” Then rank eligible results. A good match from
another user's records doesn't become eligible.

Generative Agents adds three normalized scores:

$$
\begin{aligned}
S(m,q)={}&\alpha_r\,\widetilde{R}(m)
+\alpha_i\,\widetilde{I}(m)\\
&+\alpha_s\,\widetilde{C}(m,q).
\end{aligned}
$$

$m$ is a memory, $q$ a query, $R$ recency, $I$ importance and $C$ cosine
similarity. The paper uses unit weights. Recency decays since the last
access as $R=0.995^{\Delta t}$, with time measured in simulated hours.
At 0, 24 and 100 hours the scores are about 1, 0.887 and 0.606. Retrieval
can therefore make an old record more recent. A model assigns importance
from 1 to 10 at write time; relevance depends on the current query.
Normalize each column separately:

$$
\widetilde x_i=\frac{x_i-\min_j x_j}{\max_j x_j-\min_j x_j}.
$$

If all values in a column match, this example assigns zeros to that
column. It cannot distinguish candidates. This is a fixture convention,
not a claim about the authors' implementation.

| Record | Hours / importance / similarity → score |
|---|---|
| Confirmed new date | `24 / 9 / 0.9` → 2.712 |
| Weather conversation | `0 / 1 / 0.1` → 1.000 |
| Streaming requirement | `100 / 9 / 0.8` → 1.875 |

The normalized recency values are `[0.712, 1, 0]` to three decimals,
importance is `[1, 0, 1]` and similarity is `[1, 0, 0.875]`.
The sums put the date first and the streaming requirement ahead of
the weather. These are demonstration inputs, not measured scores.

<details id="memory-code-ranking">
<summary>Python: Normalize three columns and rank the records</summary>

~~~python
def minmax(values):
    low, high = min(values), max(values)
    return [(v - low) / (high - low) for v in values] if high > low else [0.0] * len(values)

recency = minmax([0.995 ** hours for hours in [24, 0, 100]])
importance = minmax([9, 1, 9])
relevance = minmax([0.9, 0.1, 0.8])
scores = [sum(parts) for parts in zip(recency, importance, relevance)]
order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
assert order == [0, 2, 1]
assert minmax([4, 4]) == [0.0, 0.0]
~~~

</details>

A new extreme can change every normalized value. The score is neither a
permanent property of a memory nor a probability that it is true. Mark
or filter a superseded date before ranking. After ranking, remove
duplicates and keep source IDs and enough context for each part of the
decision. Preloading information saves calls; requesting it later
avoids using space before it is needed.

<span id="hipporag-найти-факт-через-связи-с-другими-фактами"></span>

### HippoRAG: reaching a fact through other facts

A question asks which Stanford researcher works on Alzheimer's disease.
One passage gives an affiliation; another describes the research.
Independent retrieval for each clue can miss the person who connects
both.

[HippoRAG](https://arxiv.org/abs/2405.14831) extracts entity–relation–entity
triples and links similar names using an encoder. Query entities seed
Personalized PageRank over the graph. The weights of nodes found in
source passages are combined to rank the passages. Answer generation
uses those texts, not the list of edges alone.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-posttraining-memory/m03-p31-hipporag-index-and-retrieval.png|The graph connects Stanford to Thomas and Thomas to research about Alzheimer's disease.]]

*Yu Su, [Berkeley Advanced LLM Agents, Lecture 3, slide 31](https://rdi.berkeley.edu/adv-llm-agents/slides/language_agents_YuSu_Berkeley.pdf#page=31).
Indexing is above, retrieval below. Original method: Gutiérrez et al.,
HippoRAG, NeurIPS 2024.*

$$
\mathbf p^{(k+1)}=(1-d)\mathbf q+dP^\top\mathbf p^{(k)}.
$$

$P$ is the transition matrix, $\mathbf q$ seeds the query entities,
and $d$ is the probability of following a link. The remaining fraction
returns to the query rather than letting the graph's globally popular
nodes dominate. HippoRAG also considers entity specificity at restart.

Take three toy nodes: clues $A$ and $B$ lead to $T$, and $T$ returns
to each clue with probability one half. With $\mathbf q=(0.5,0.5,0)$
and $d=0.85$, the first iteration is $(0.075,0.075,0.85)$.
The second is $(0.43625,0.43625,0.1275)$ because the weight at $T$
flows back. The limit is about $(0.27027,0.27027,0.45946)$.

<details id="memory-code-ppr">
<summary>Python: Trace PageRank past the first transition</summary>

~~~python
from math import isclose

q = [0.5, 0.5, 0.0]
P = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]
def ppr_step(p, restart=q, damping=0.85):
    return [(1 - damping) * restart[j]
            + damping * sum(p[i] * P[i][j] for i in range(3))
            for j in range(3)]

p = ppr_step(q)
assert all(isclose(a, b) for a, b in zip(p, [0.075, 0.075, 0.85]))
for _ in range(200):
    p = ppr_step(p)
assert all(isclose(a, b, abs_tol=1e-12)
           for a, b in zip(p, [10/37, 10/37, 17/37]))
assert isclose(sum(p), 1.0)
~~~

</details>

The fixture omits triplet extraction, entity linking, specificity and
document ranking. A high node weight establishes reachability from the
query, not a fact's truth. A graph that merges two people with the
same name can invent a connection. Keep the link to the passages and
regenerate affected edges when they change. See the
[[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер#HippoRAG: ассоциативное извлечение по графу|RAG chapter]]
for the retrieval pipeline.

<span id="параметрическая-память-не-заменяет-проверяемое-внешнее-состояние"></span>

### What can be inferred from model weights

In [Yu Su's lecture 3, slides 79–80](https://rdi.berkeley.edu/adv-llm-agents/slides/language_agents_YuSu_Berkeley.pdf#page=79),
a Transformer trained for a long time on a synthetic graph starts
generalizing to tasks for which external retrieval struggled to
assemble the chain. This is a controlled example of grokking: delayed
generalization after memorization. It doesn't show that weights can
replace a database of changing facts. A calendar assistant needs to
retrieve a particular booking with its revision, and handle cancellations
without retraining a model.

<span id="6-план-как-проверяемая-гипотеза"></span>
<a id="agent-planning-branches"></a>

## Finding an order in which actions can run

A summary says what happened. A plan identifies the actions that could
reach the goal. The final invitation contains the name of the room, so
sending must wait for a confirmed booking. Break the work into subgoals
with observations that can establish their success: an application ID
confirms that the request was submitted, while a calendar or administrator
response confirms the room.

When a room becomes unavailable, update the room search and the related
text rather than asking the speaker about the unchanged topic again.
The finished work remains in the log. Replanning applies the new
information to steps that are still outstanding.

<span id="от-последовательности-шагов-к-поиску-альтернатив"></span>

### Preconditions, effects and a valid plan

In [LLM+P, §§2–3](https://arxiv.org/html/2304.11477v1#S3), a person supplies
a PDDL domain with actions, preconditions and effects. The model translates
the specific task into a problem file with objects, an initial state and a
goal. A solver plans from both files, and the model explains the result
in natural language. The model doesn't invent the domain rules.

Let `book_room` require `room_available` and add `room_confirmed`.
Let `send_invitation` require both `room_confirmed` and `time_confirmed`
and add `invitation_sent`. If the room is available and the time confirmed,
booking first makes sending possible. Reversing those steps fails the
precondition. Here `book_room` represents a confirmed booking, not an
asynchronous application. A real service can reject the application;
the plan must account for observations. A solver cannot enforce a
streaming requirement omitted from the domain, and legality in the
plan doesn't grant permission to send mail.

<span id="tree-of-thoughts-выбор-между-промежуточными-решениями"></span>

### Tree of Thoughts: exploring partial solutions

The [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#tree-of-thoughts|test-time compute chapter]]
covers arithmetic, crosswords and passage planning. A ToT state here
is an explicit partial solution. The LLM proposes and evaluates
continuations, and the search procedure decides which to visit.
It does not inspect hidden model activations.

*Source: Shunyu Yao et al., [Tree of Thoughts, §3](https://arxiv.org/html/2305.10601v2#S3), CC BY 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-memory-planning-2026-10/tot-game24-search.png|Game of 24 search and separate model prompts for proposing and evaluating candidate steps.]]

*Shunyu Yao et al., [Tree of Thoughts, Figure 2, p. 5](https://arxiv.org/pdf/2305.10601v2#page=5).
Original figure and caption, CC BY 4.0.*

The goal is 24 using 4, 9, 10 and 13 exactly once. After `10 − 4 = 6`
the remaining values are `[6, 9, 13]`. Subtracting 6 from 13 leaves
`[7, 9]`; combining them yields 16, −2, 2, 63, 7/9 or 9/7, never 24.
Subtracting 9 from 13 leaves `[4, 6]`, whose product is 24:
`(10 − 4) × (13 − 9)`. The figure's model evaluator uses `sure`,
`likely` and `impossible`. That heuristic can prune a good branch.
A separate check verifies the arithmetic and use of the original numbers.

Bounded breadth-first search keeps $b$ promising states at a depth;
the authors use $b=5$ and three judgments per candidate in the main
example. With three continuations at each of two levels, nine new
states appear at the second level unless some are pruned. Depth-first
search follows one branch before backtracking. Self-consistency instead
samples completed solutions and votes. Restoring an arithmetic branch
needs saved text; restoring a real service needs a way to restore its
state.

<span id="lats-дерево-с-наблюдениями-среды"></span>

### LATS: a tree whose nodes include observations

[Language Agent Tree Search, §§3.2–4.2](https://arxiv.org/html/2310.04406v3#S4)
expands action histories with observations from the environment.
A completed trajectory supplies a reward that updates the visited nodes;
a failed attempt can contribute a reflection to later trials.

$$
\operatorname{UCT}(s)=V(s)+w\sqrt{\frac{\ln N(p)}{N(s)}}.
$$

$V(s)$ is the node's estimate, $N(s)$ its visits and $N(p)$ its parent's
visits. With $N(p)=10$ and $w=1$, $V=0.7,N=8$ yields about 1.236;
$V=0.5,N=2$ yields 1.573. The second node is visited less often and
gets the larger exploration bonus. Handle zero visits separately.

LATS combines model evaluation and self-consistency before final rewards
are available. Its simulation can include real interactions with a
resettable environment; it isn't a model predicting every screen.
Copying a conversation won't undo a purchase or a message. Where
reset isn't possible, compare alternatives without assuming it is.

<span id="реактивный-агент-поиск-в-реальной-среде-и-model-based-planning"></span>
<span id="реактивное-поведение-реальный-поиск-и-модель-среды"></span>

### Choosing now, trying a path or predicting a transition

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-memory-review-2026-09/webdreamer-planning-paradigms.png|Direct action on the left, backtracking in the environment in the center, predicted alternatives on the right.]]

*Gu et al., [WebDreamer v1, Figure 1](https://arxiv.org/html/2411.06559v1#S1.F1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure.*

A reactive agent chooses from current observations and history without
explicit future states. It can still reason over its options. Search
in the environment tests transitions by execution or in a faithful copy.
A world model instead estimates the result of an action:

$$
\widehat T(s_t,a_t)\rightarrow\widehat s_{t+1}.
$$

$s_t$ is state, $a_t$ is an action and $\widehat s_{t+1}$ is the
prediction. A browser observation $o_t$ does not expose full server
state. A predicted transition is a candidate consequence, not a
screen that was read.

<span id="webdreamer-вообразить-переход-до-необратимого-действия"></span>

### WebDreamer: scoring possible transitions before acting

[WebDreamer, §4](https://arxiv.org/html/2411.06559v1#S4) proposes and prunes
actions, predicts changes after each remaining action and explores short
modelled trajectories. It scores each and executes only the first action
of the chosen trajectory. Then it observes the actual page and repeats.

The paper's task is to buy the cheapest 512 GB drive. The candidates
include Office Products, Electronics and a search for Disk.
The two-step Electronics trajectory anticipates Computers & Accessories
and Data Storage. The scores in the slide are 0.8, 0.4 and 0.1, not
calibrated probabilities that a purchase will succeed.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-posttraining-memory/m03-p70-webdreamer-worked-example.png|Three imagined two-step trajectories are compared; only the first Electronics action is executed.]]

*Yu Su, [Berkeley Advanced LLM Agents, Lecture 3, slide 70](https://rdi.berkeley.edu/adv-llm-agents/slides/language_agents_YuSu_Berkeley.pdf#page=70).
Method and example: Gu et al., [WebDreamer](https://arxiv.org/abs/2411.06559).*

$$
a_t=\arg\max_{a\in\mathcal A_t}
\operatorname{score}\!\left(\operatorname{sim}(o_t,a;H)\right),
$$

$\mathcal A_t$ holds retained candidates, $H$ is the prediction
horizon and $\operatorname{sim}$ returns an imagined trajectory.
The evaluator labels whether the task is complete, the trajectory
makes progress, or it goes in the wrong direction; scores are averaged.
The horizon is recomputed after the real action. A longer trajectory may
compound an early prediction error. Check the product, price, quantity
and user authorization before any purchase; a prediction cannot verify
those facts.

<!-- BEGIN SOURCE: Plan-Seq-Learn 2405.01534v1 -->

<a id="plan-seq-learn"></a>
<span id="plan-seq-learn-от-языковых-подцелей-к-управлению-роботом"></span>

### Plan-Seq-Learn: from language goals to robot control

“Put the nut on the peg” specifies a goal, not motor commands. A robot
needs to locate the nut, approach it, grasp it and align its hole with
the peg. Plan-Seq-Learn separates a language plan, vision and motion
planning, and a learned policy for local interaction. Reinforcement
learning trains the motor policy, not the language model.

<details>
<summary>Source text and attribution</summary>

The expandable source sections preserve the original English explanation
from [Dalal et al., Plan-Seq-Learn, ICLR 2024](https://arxiv.org/html/2405.01534v1):
§`3.2–3.5, `D.3.1 and §B.3, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Berkeley lecture 6, slides 107–116, covers the same method.
[[05 Источники/Papers/Plan-Seq-Learn|Paper and author materials]].

</details>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/planseqlearn-2026-10-26/method.png|Plan-Seq-Learn separates language planning, motion to a target, and learned local interaction.]]

*Dalal et al., [Plan-Seq-Learn, Figure 2](https://arxiv.org/html/2405.01534v1#S3.F2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original PNG.*

The plan supplies the target region and completion condition. The blue
blocks approach the target. The green block interacts locally until a
check reports completion. None of the natural-language labels specifies
a trajectory of joint angles.

<!-- source-psl:S3.SS2 -->

<details id="psl-source-overview">
<summary>§3.2: Bridging the plan and local interaction</summary>

> To solve long-horizon robotics tasks, we need a module capable of bridging the gap between
> zero-shot language model planning and learned low-level control. Observe that many tasks of
> interest can be decomposed into alternating phases of contact-free motion and contact-rich
> interaction. One first approaches a target region and then performs interaction behavior, prior to
> moving to the next sub-task. Contact-free motion generation is exactly the motion planning
> problem. For estimating the position of the target region, we note that state-of-the-art vision
> models are capable of accurate language-conditioned state estimation ([Kirillov et al.,
> 2023](https://arxiv.org/html/2405.01534v1#bib.bib33); [Zhou et al.,
> 2022](https://arxiv.org/html/2405.01534v1#bib.bib77); [Liu et al.,
> 2023b](https://arxiv.org/html/2405.01534v1#bib.bib41); [Bahl et al.,
> 2023](https://arxiv.org/html/2405.01534v1#bib.bib4); [Ye et al.,
> 2023](https://arxiv.org/html/2405.01534v1#bib.bib73); [Labbé et al.,
> 2022](https://arxiv.org/html/2405.01534v1#bib.bib36)). As a result, we propose a Sequencing Module
> which uses off-the-shelf vision models to estimate target robot states from the language plan and
> then achieves these states using a motion planner. From such states, we train interaction policies
> that optimize the task reward using RL. See Alg. [1](https://arxiv.org/html/2405.01534v1#alg1) and
> Fig. [2](https://arxiv.org/html/2405.01534v1#S3.F2) for an overview of our method.

</details>

#### Planning Module: Zero-Shot High-level Planning

The language model sees the task and supported conditions but not the
scene. It proposes pairs such as `gold nut / grasp`. The authors remove
invalid objects or conditions; the LLM is not trained in this experiment.

<!-- source-psl:S3.SS3 -->

<details id="psl-source-plan">
<summary>§3.3: The NutAssembly prompt and plan</summary>

> Long-horizon tasks can be broken into a series of stages to execute. Rather than discovering these
> stages using interaction or using a task planner [Fikes & Nilsson
> (1971)](https://arxiv.org/html/2405.01534v1#bib.bib13) that may require privileged information
> about the environment, we use language models to produce natural language plans zero shot without
> access to the environment. Specifically, given a task description $g_{l}$ by a human, we prompt an
> LLM to produce a plan. Designing the plan granularity and scope are crucial; we need plans that
> can be interpreted by the Sequencing Module, a vision-based system that produces and achieves
> robot poses using motion planning. As a result, the LLM predicts a target region (a natural
> language label of an object/receptacle in the scene, e.g. “silver peg”) which can be translated
> into a target pose to achieve at the beginning of each stage of the plan.

> When the RL policy is executing a step of the plan, we propose to add a stage termination
> condition (e.g. grasp, place, turn, open, push) to know the stage is complete and to move onto the
> next stage. This condition is defined as a function $f_{stage}(O^{global})$ that takes in the
> current observation of the environment and evaluates a binary success criteria as well as a
> natural language descriptor of the condition for prompting the LLM (e.g. ‘grasp’ or ‘place’).
> These stage termination conditions are estimated using visual pose estimates. We describe the
> stage termination conditions in greater detail in Sec.
> [3.5](https://arxiv.org/html/2405.01534v1#S3.SS5) and Appendix
> [D](https://arxiv.org/html/2405.01534v1#A4). The LLM prompt consists of the task description
> $g_{l}$, the list of supported stage termination conditions (which we hold constant across all
> environments) and additional prompting strings for output formatting. We format the language plans
> as follows: (“Region 1”, “Termination Condition 1”), … (“Region N”, “Termination Condition N”),
> assuming the LLM predicts N stages. Given the prompt, the LLM outputs a natural language plan in
> the format listed above. Below, we include an example prompt and plan for the Nut Assembly task.

> Prompt: Stage termination conditions: (grasp, place). Task description: The gold nut goes on the
> gold peg and the silver nut goes on the silver peg. Give me a simple plan to solve the task using
> only the stage termination conditions. Make sure the plan follows the formatting specified below
> and make sure to take into account object geometry. Formatting of output: a list in which each
> element looks like: ($<$object/region$>$, $<$stage termination condition$>$). Don’t output
> anything else.

> Plan: [(“gold nut”,“grasp”), (“gold peg”, “place”), (“silver nut”, “grasp”), (“silver peg”,
> “place”)]

> While any language model can be used to perform this planning process, we found that of a variety
> of publicly available LLMs (via weights or API), only GPT-4 [OpenAI
> (2023)](https://arxiv.org/html/2405.01534v1#bib.bib50) was capable of producing correct plans
> across all the tasks we consider. We sample from the model with temperature $0$ for determinism.
> We also delete components of the plan that contain LLM hallucinations (if present). We provide
> additional details in Appendix [D](https://arxiv.org/html/2405.01534v1#A4) and example prompts in
> Appendix [G](https://arxiv.org/html/2405.01534v1#A7).

</details>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/planseqlearn-2026-10-26/nut-assembly.png|The four stages for placing the gold and silver nuts, separating approach from interaction.]]

*Dalal et al., [Plan-Seq-Learn, Figure 1](https://arxiv.org/html/2405.01534v1#S1.F1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original PNG.*

In the text example the gold nut goes first: grasp the gold nut,
place it on the gold peg, then repeat for silver. Approaching the nut
is distinct from grasping it. A check for `grasp` advances the plan to
the peg. A check for `place` advances it to the next nut. These are
completion conditions, not four pretrained motor skills. The Berkeley
slides illustrate the other valid order, silver first.

Each successful stage leaves the state for the next. If conditional
success at each stage is 0.9, success for four stages is
$0.9^4=0.6561$. This is an example of the product rule, not a measurement
from PSL. Splitting the task into short interactions does not remove
error accumulation.

#### Sequencing Module: Vision-based Plan Tracking

A text label identifies the target region in the global image.
Calibrated depth and the mask estimate its 3D coordinates. Inverse
kinematics gives the target configuration $q_{target}$, and AIT*
looks for a collision-free path from $q_0$. The result is a position
from which to interact, not evidence that the nut has been placed.

<!-- source-psl:S3.SS4 -->

<details id="psl-source-sequence">
<summary>§3.4: Pose estimation and motion planning</summary>

> Given a high-level language plan, we now wish to step through the plan and enable a learned RL
> policy to solve the task, using off-the-shelf vision to produce target poses for a motion planning
> system to achieve. At stage X of the high-level plan, the Sequencing Module takes in the
> corresponding step high-level plan (“Region Y”, “Termination Condition Z”) as well as the current
> global observation of the scene $O^{global}$ (RGB-D view(s) that cover the whole scene), predicts
> a target robot pose $q_{target}$ and then reaches the robot pose using motion planning.

> Vision and Estimation: Using a text label of the target region of interest from the high-level
> plan and observation $O^{global}$, we need to compute a target robot state $q_{target}$ for the
> motion planner to achieve. In principle, we can train an RL policy to solve this task (learn a
> policy $\pi_{v}$ to map $O^{global}$ to $q_{target}$) given the environment reward function.
> However, observe that the 3D position of the target region is a reasonable estimate of the optimal
> policy $\pi_{v}^{*}$ for this task: intuitively, we wish to initialize the robot nearby to the
> region of interest so it can efficiently learn interaction. Thus, we can bypass learning a policy
> for this step by leveraging a vision model to estimate the 3D coordinates of the target region. We
> opt to use Segment Anything [Kirillov et al.
> (2023)](https://arxiv.org/html/2405.01534v1#bib.bib33) to perform segmentation, as it is capable
> of recognizing a wide array of objects, and use calibrated depth images to estimate the
> coordinates of the target region. We convert the estimated region pose into a target robot pose
> $q_{target}$ for motion planning using inverse kinematics.

> Motion Planning: Given a robot start configuration $q_{0}$ and a robot goal configuration
> $q_{target}$ of a robot, the motion planning module aims to find a trajectory of way-points $\tau$
> that form a collision-free path between $q_{0}$ and $q_{target}$. For manipulation tasks, for
> example, $q$ represents the joint angles of a robot arm. We can use motion planning to solve this
> problem directly, such as search-based planning [Cohen et al.
> (2010)](https://arxiv.org/html/2405.01534v1#bib.bib8), sampling-based planning [Kuffner Jr. &
> LaValle (2000)](https://arxiv.org/html/2405.01534v1#bib.bib34) or trajectory optimization
> [Schulman et al. (2013)](https://arxiv.org/html/2405.01534v1#bib.bib54). In our implementation, we
> use AIT* [Strub & Gammell (2020)](https://arxiv.org/html/2405.01534v1#bib.bib57), a sampling-based
> planner, due to its minimal setup requirements (only collision-checking) and favorable performance
> on planning. For implementation details, please see Appendix
> [D](https://arxiv.org/html/2405.01534v1#A4).

> Overall, the Sequencing Module functions as the connective tissue between language and control by
> moving the robot to regions of interest in the plan, enabling the RL agent to quickly learn
> short-horizon interaction behaviors to solve the task.

</details>

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/planseqlearn-2026-10-26/sequencing.png|RGB-D, masks, point clouds and inverse kinematics lead to a collision-free trajectory.]]

*Dalal et al., [Plan-Seq-Learn, Figure D.1](https://arxiv.org/html/2405.01534v1#A4.F1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original PNG.*

#### Learning Module: Efficiently Learning Local Control

One DrQ-v2 policy learns from the task reward across stages. It uses
local images near the target, while the global scene guides the approach.
The stage predicate is checked at each step; it can advance the plan
before the stage budget $H_l$ expires.

<!-- source-psl:S3.SS5 -->

<details id="psl-source-learn">
<summary>§3.5: Shared policy, local observations and stopping</summary>

> Once the agent steps through the plan and achieves states near target regions of interest, it
> needs to train an RL policy $\pi_{\theta}$ to learn low-level control for solving the task. We
> train $\pi_{\theta}$ using DRQ-v2 [Yarats et al.
> (2021)](https://arxiv.org/html/2405.01534v1#bib.bib72), a SOTA visual model-free RL algorithm, to
> produce low-level control actions (joint control or end-effector control) from images.
> Furthermore, we propose three modifications to the learning pipeline in order to further improve
> learning speed and stability.

> First, we train a single RL policy across all stages, stepping through the language plan via the
> Sequencing Module, to optimize the task reward function. The alternative, training a separate
> policy per stage, would require designing stage specific reward functions per task. Instead, our
> design enables the agent to solve the task using a single reward function by sharing the policy
> and value functions across stages. This simplifies the training setup and allowing the agent to
> account for future decisions as well as inaccuracies in the Sequencing Module. For example, if
> $\pi_{\theta}$ is initialized at a sub-optimal position relative to the target region,
> $\pi_{\theta}$ can adapt its behavior according to its value function, which is trained to model
> the full task return $\mathbb{E}\left[\sum_{t}\gamma^{t}\mathcal{R}(s_{t},a_{t},s_{t+1})\right]$.

> Second, instead of executing $\pi_{\theta}$ for a fixed number of steps per stage $H_{l}$, we
> predict a stage termination condition $f_{stage}(O^{global})$ using the language model and
> evaluate the condition at every time-step to test if a stage is complete, otherwise it times out
> after $H_{l}$ steps. For most conditions, $f_{stage}$ is evaluated by computing the pose estimate
> of the relevant object and thresholding. This process functions as a form of curriculum learning:
> only once a stage is completed is the agent allowed to progress to the next stage of the plan. As
> we ablate in Sec. [5](https://arxiv.org/html/2405.01534v1#S5), stage termination conditions enable
> the agent to learn more performant policies by preventing dithering behavior at each stage. As an
> example, in the nut assembly task shown in Fig. [1](https://arxiv.org/html/2405.01534v1#S1.F1),
> once $\pi_{\theta}$ places the silver nut on the silver peg, the placement condition triggers (by
> compare the pose of the nut to the peg pose) and the Sequencing Module moves the arm to near the
> gold peg.

> Finally, as opposed to training the policy using the global view of the scene ($O^{global}$), we
> train using local observations $O^{local}$, which can only observe the scene in a small region
> around the robot (e.g. wrist camera views for robotic manipulation). This design choice affords
> several unique properties that we validate in Appendix
> [C](https://arxiv.org/html/2405.01534v1#A3), namely: 1) improved learning efficiency and speed, 2)
> ease of chaining pre-trained policies. Our policies are capable of leveraging local views because
> of the decomposition in PSL: the RL policy simply has to learn interaction behaviors in a small
> region, it has no need for a global view of the scene, in contrast to an end-to-end RL agent that
> would need to see a global view of the scene to know where to go to solve a task. For additional
> details in regarding the structure and training process of the Learning Module, see Appendix
> [D](https://arxiv.org/html/2405.01534v1#A4).

</details>

After describing placement of the silver nut, `3.5 names `gold peg§
as the next region. That wording conflicts with the plan: grasping
the gold nut must precede placement at the gold peg. The original
excerpt is unchanged. The worked example follows the plan in §3.3.

#### Stage Termination Details

Placement compares the object and target poses. For a grasp, the
object must rise and remain caged by the gripper. Rotation uses the
object's axis, pushing uses displacement, and opening a hinged door
uses the handle's position relative to the hinge. Thresholds come
from the simulated environment's success criteria or reward.

<!-- source-psl:A4.SS3.SSS1 -->

<details id="psl-source-termination">
<summary>§D.3.1: Predicates for grasp, place, turn and open</summary>

> As described in Section [3](https://arxiv.org/html/2405.01534v1#S3), we use stage termination
> conditions to determine when the Learning Module should hand control back to the Sequencing Module
> to continue to the next stage in the plan. For most of the tasks we consider, these stage
> termination conditions amount to checking for a grasp or placement for the target object in the
> stage. For example, for RS-NutRound, the plan for the first stage is (grasp, nut) and the plan for
> the second stage is (place, peg). Placements are straightforward to check: simply evaluate if the
> object being manipulated is within a small region near the target object. This can be computed
> using the estimated pose of the two objects (current and target). Grasps are more challenging to
> estimate and we employ a two stage pipeline to detecting a grasp. First, we estimate the object
> pose and then evaluate if the z value has increased from when the stage began. Second, in order to
> ensure the object is not simply tossed in the air, we check if the robot’s gripper is tightly
> caging the object. We do so by collision checking the object point-cloud against the gripper mesh.
> We use the same collision checking procedure as outlined in Sec
> [3](https://arxiv.org/html/2405.01534v1#S3) for checking collision between the scene point-cloud
> and robot mesh.

> To estimate the turned condition, we compute the mask of the burner knob, evaluate its principal
> axis and measure its angle from vertical. If it is greater than X radians then the stage condition
> triggers. Checking the pushed condition is straightforward, the object needs to have moved by X
> distance forward from the start pose in xy. Finally, for checking the opened condition, we
> estimate the handle pose relative to the hinge and compute the angle of the door. If it is greater
> than X radians we consider the door opened. For the slide cabinet the handle pose itself can be
> used to check opening. Closing is likewise estimated as the inverse of opening. For all
> conditions, we take the threshold X from the environment success condition or reward function.

</details>

#### Limitations

Alternating between approach and interaction does not cover every task:
some require simultaneous manipulation or rapid reactions in a dynamic
scene. The local policy can compensate for a small pose error but has
little chance of correcting a plan that sent it to the wrong object.

<!-- source-psl:A2.SS3 -->

<details id="psl-source-limits">
<summary>§B.3: Limits of the decomposition</summary>

> There are several limitations of PSL which leave scope for future work. 1) We impose a specific
> structure on the language plans and task solution (go to location X, interact there, so on). While
> this assumption covers a broad set of tasks as well illustrate in our experimental evaluation,
> tasks that involve interacting with multiple objects simultaneously or continuous switching
> between interaction and movement in a fluid manner may not be directly applicable. Future work can
> explore integrating a more expressive plan structure with the Sequencing Module. 2) Use of
> motion-planning makes application to dynamic tasks challenging. To that end, research on
> motion-planner distillation, such as Motion Policy Networks [Fishman et al.
> (2022)](https://arxiv.org/html/2405.01534v1#bib.bib14) could enable much faster, reactive
> behavior. 3) Although the RL agent is capable of adapting pose estimation errors, in the current
> formulation, there is not much the Learning Module can do if the high-level plan itself is
> entirely incorrect, or if the Sequencing module misinterprets the language instruction and moves
> the robot to the wrong object. One extension to address this limitation would be to fine-tune the
> Plan and Seq Modules online using RL as well, to adapt the large models to the specific
> environment and reward function.

</details>

The experiments take place in simulators with calibrated cameras,
available rewards and stage completion checks. “No pretrained skill
library” does not mean no environment configuration or plan checking.
WebDreamer estimates hypothetical page transitions; PSL uses a
language decomposition, geometry and actual simulator transitions
to train local control. The [authors' videos and code](https://mihdalal.github.io/planseqlearn/)
show the movement between still frames.

<!-- END SOURCE: Plan-Seq-Learn 2405.01534v1 -->

<span id="когда-более-сильной-модели-нужен-менее-жёсткий-scaffold"></span>
<span id="всегда-ли-дополнительный-планировщик-полезен"></span>

### When branching is worth its cost

A planner adds calls and can introduce errors. If the next step is
already clear, simulating alternatives adds latency. In a task with
deceptive local choices, alternative branches can reveal a dead end
before the agent acts. In lecture 3, slide 72, Yu Su suggests that a
stronger model may need less external scaffolding. Test that hypothesis
by comparing a sequential loop, a fixed workflow and a world-model
planner under a common budget.

<span id="7-рефлексия-полезна-только-через-наблюдаемую-обратную-связь"></span>
<span id="рефлексия-полезна-только-через-наблюдаемую-обратную-связь"></span>
<span id="7-как-использовать-обратную-связь"></span>
<a id="agent-reflexion-feedback"></a>

## What to change after a failed attempt

In [Reflexion](https://arxiv.org/abs/2303.11366), an actor attempts the
task, an evaluator reports the outcome, and a reflection of the attempt
enters the next input. Model weights do not change. The evaluation
may come from the environment or from an internal score, so using
Reflexion does not guarantee an independent check.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-memory-planning-2026-10/reflexion-loop.png|The actor receives an episode trace and reflections accumulated after previous attempts.]]

*Noah Shinn et al., [Reflexion v4, Figure 2, p. 4](https://arxiv.org/pdf/2303.11366v4#page=4).
Original figure, pseudocode and caption, CC BY 4.0.*

A reflection need not keep the entire episode. It should identify
what failed and give the next attempt a constraint or alternative.
The paper's memory usually retains one to three reflections. ALFWorld
keeps the last three and resets between attempts. A calendar does not
reset just because the model starts a new attempt.

The figure's loop condition uses `or`. Read literally, the condition
doesn't enforce both early success and the attempt budget. The toy
code below uses a bounded range and returns as soon as a candidate
passes. The figure itself is unmodified.

“Room 204 seats 40 and we needed 45” makes the next room search
better specified. “I should have been more careful” doesn't tell
the search what to change. A similar case in code is balancing
parentheses. Counting both kinds gives the wrong result for `)(`.
That counterexample tells the next attempt to check every prefix,
not just the final counts.

<details id="memory-code-reflexion">
<summary>Python: Use a concrete counterexample between two attempts</summary>

~~~python
def counts_only(text):
    return text.count("(") == text.count(")")

def prefix_checked(text):
    balance = 0
    for char in text:  # The input contains only '(' and ')'.
        balance += 1 if char == "(" else -1
        if balance < 0:
            return False
    return balance == 0

def try_candidates(candidates, cases, limit=2):
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    memory = []
    for attempt, candidate in enumerate(candidates[:limit], start=1):
        failures = []
        for text, expected in cases:
            actual = candidate(text)
            if actual != expected:
                failures.append((text, expected, actual))
        if not failures:
            return {"status": "passed_given_cases", "attempts": attempt, "memory": memory}
        memory.append({"attempt": attempt, "counterexamples": failures})
    return {"status": "budget_exhausted", "attempts": min(limit, len(candidates)), "memory": memory}

cases = [("()", True), (")(", False), ("(()", False), ("", True)]
result = try_candidates([counts_only, prefix_checked], cases)
assert result["status"] == "passed_given_cases" and result["attempts"] == 2
assert result["memory"][0]["counterexamples"] == [(")(", False, True)]
~~~

</details>

The candidates are predefined, the evaluator deterministic and the
memory stores test failures. This does not reproduce LLM generation
or reflection. Without `)(`, the count-only candidate would pass
these tests. The return value `passed_given_cases` is narrower than
a proof of correctness.

For program repair, a test supplies input, expected output and
actual output. An execution trace can expose intermediate states.
In the slide below, however, the trace means a written simulation
generated by the model. It can suggest an error, but it is not
independent output from running the program.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-posttraining-memory/m01-p64-self-debugging-feedback.png|Four feedback modes: failure signal, unit test, code explanation, and a model-generated execution trace.]]

*Xinyun Chen, [Berkeley Advanced LLM Agents, lecture 1, slide 64](https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf#page=64);
Chen et al., [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128).*

The example translates a C++ function that counts trailing zeros in
a factorial. For $9!$, the answer is 1. C++ integer division in this
example yields $9/5=1$, while Python's `/` yields 1.8. The test
exposes the discrepancy. The explanation suggests integer division
`//` for the counter. Run the test again to check that the repair works.

<span id="почему-самокритика-без-внешнего-сигнала-не-является-проверкой"></span>

### Self-correction and external feedback

“Think again” may improve an answer without giving evidence that the
new answer is correct. In [Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798),
self-correction without external feedback sometimes reduced accuracy.
Their oracle condition tells the model whether an answer is actually
correct. That feedback is not available in ordinary self-critique.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-posttraining-memory/m01-p67-self-correction-no-oracle.png|Self-correction without feedback compared with the condition that gives the model the answer's true correctness.]]

*Xinyun Chen, [Berkeley Advanced LLM Agents, lecture 1, slide 67](https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf#page=67).
Results for the models and tasks of Huang et al., not a current leaderboard.*

Multiple agents can repeat the same misconception. In the GSM8K
comparison below, debate did not beat independent voting with the
same number of answers. More calls can help without communication;
include that control when comparing agent designs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-posttraining-memory/m01-p69-debate-vs-self-consistency.png|GSM8K debate compared with independently sampled answers followed by voting.]]

*Xinyun Chen, [Berkeley Advanced LLM Agents, lecture 1, slide 69](https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf#page=69);
Huang et al., [§3.3, Table 4](https://arxiv.org/html/2310.01798v1#S3.SS3).*

The reported GSM8K test uses gpt-3.5-turbo-0301, three agents and two
rounds. Debate and voting score 83.2% and 85.3% with six answers,
then 83.0% and 88.2% with nine. Equal answer counts do not establish
equal input tokens or latency.

<span id="8-когда-нужен-второй-агент"></span>

## Which tasks can run independently

A single agent can read, write memory and use tools in sequence.
Room search and literature search are independent enough to run
together. The room worker needs the date, attendance and equipment,
not the literature review. A later date change must reach that
worker explicitly even if the application shares a memory store.

Two fixed calls are enough if the same two tasks always run. A
coordinator helps when new observations change the list of tasks:
if no room is free, assign a check for a remote option. It then
combines results for the same date and format. Anthropic's
[orchestrator-workers diagram](https://www-cdn.anthropic.com/images/4zrzovbb/website/8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png)
shows the coordinator choosing the tasks. Workers can be separate
calls to the same model.

<span id="9-основные-схемы-оркестрации"></span>

## How results pass from one step to another

| Pattern | What is passed on |
|---|---|
| Chain | A plan goes to the writer, a draft to the checker. |
| Parallel work | Room candidates and a reading list are collected independently. |
| Routing | A date question goes to the calendar; an article question goes to search. |
| Orchestrator and workers | A gap in the review triggers an extra search. |
| Evaluator and optimizer | A missing time zone goes back to the invitation writer. |
| Discussion | Workers exchange explanations of conflicting results. |

An invitation says 16:00, but the calendar says 16:00 Moscow time.
The evaluator points to the missing zone. The writer changes the
draft; a checker compares the time and zone with the calendar.
Anthropic's [evaluator-optimizer diagram](https://www-cdn.anthropic.com/images/4zrzovbb/website/14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png)
branches between acceptance and feedback. A model can judge clarity.
Code can compare exact fields such as a time, address or recipient list.
Set an iteration limit and retain the last acceptable draft in case a
later revision harms a different part of it.

<span id="10-контракт-между-агентами"></span>
<span id="10-что-передавать-исполнителю-и-получать-обратно"></span>
<a id="agent-worker-dependencies"></a>

## What a worker needs to know

“Handle the room” fails to say whether booking is allowed. A scoped
assignment can be: find three rooms for the agreed date, check capacity
and equipment, return prices and links, and do not book. Include which
actions are allowed even when several workers read the same project data.

A usable result gives each room's ID, capacity, time checked, and source.
It marks anything that wasn't verified. “Use room 204” alone forces the
coordinator to repeat the search or trust an unsupported recommendation.
Attach the input revision to each result and assign separate ownership of
files if workers can write to them.

<span id="результат-зависит-от-конкретных-входов"></span>

### A result depends on specific inputs

Changing attendance from 30 to 45 invalidates room selection, not a reading
list for an unchanged topic. Record each result's dependencies separately.
Room selection depends on attendance and start time; literature depends
on topic. Values and sources below are placeholders, not real lookups.

<details id="memory-code-merge">
<summary>Python: Reject a stale room result and keep the reading list</summary>

~~~python
def accept_result(current, result):
    dependencies = result["depends_on"]
    if not dependencies or any(key not in current or current[key] != expected
                               for key, expected in dependencies.items()):
        return {"status": "stale"}
    return {"status": "accepted_for_review", "value": result["value"],
            "evidence": result["evidence"]}

current = {"people": 30, "start": "2026-10-09T19:00:00+03:00", "topic": "retrieval"}
room = {"depends_on": {"people": 30, "start": current["start"]},
        "value": {"room_id": "204", "capacity": 40}, "evidence": "room-check-7"}
papers = {"depends_on": {"topic": "retrieval"},
          "value": ["paper-1", "paper-2"], "evidence": "search-12"}
assert accept_result(current, room)["status"] == "accepted_for_review"
current["people"] = 45
assert accept_result(current, room)["status"] == "stale"
assert accept_result(current, papers)["status"] == "accepted_for_review"
~~~

</details>

A missing dependency field in the current state makes the result stale.
If the worker omitted a dependency from `depends_on`, this function
cannot infer it. Matching inputs make a result eligible for review,
not verified. The coordinator still checks the capacity, freshness and
sources. A real concurrent commit must compare inputs and save the result
atomically, for example with a conditional database write. The fixture's
dictionary does not provide that transaction.

<span id="11-конфликты-и-отказоустойчивость"></span>

## What to do with stale results

Room 204 seats 40. It was admissible for 30 attendees, but it isn't for
45. Re-run room search; keep the literature search. Comparing versions
at write time is one form of optimistic concurrency control. Comparing
only the inputs a worker used can avoid invalidating unrelated work.
In either case, comparison and update must be atomic.

A timeout while sending an invitation leaves the outcome unknown.
Blindly repeating it can produce two messages. Consult the operation
ID or another authoritative outcome first. Reuse the same key only
under a supported idempotency contract. A checkpoint records which
operations are pending, which work has finished and the versions
used for its results. Persistent memory about a user serves a
different purpose.

<span id="полный-цикл-перенос-семинара"></span>

### The seminar moves to Friday

1. Record the organizer's message and replace the active date with Friday
   at 19:00. The log keeps the old date.
2. Retrieve requirements, attendance, the prior booking and the draft for
   this seminar. Leave unrelated conversations outside the input.
3. Keep the reading list if the topic is unchanged. The old booking
   does not establish a room for Friday.
4. Search again for a room. The writer can update the draft during the
   search, leaving the location pending.
5. Send a booking request for a suitable option if authorized. Record the
   submission; wait for a confirmation before marking the room booked.
6. Combine the confirmed room and draft, check time, address and recipients,
   and send only with authorization. Retain the operation ID.
7. Record the confirmed sending result and retain current arrangements and
   audit events.

If the room search needs 70 seconds, writing needs 100, and merging
needs 10, a sequential schedule takes 180 seconds. Running the two
independent tasks concurrently takes 110 under these assumptions.
Shared queues, API limits and coordination make actual latency
different. The arithmetic illustrates a dependency, not a benchmark.

<span id="12-что-измерять"></span>

## Locating the cause of an error

Measure whether a fact was stored, retrieved and used. If a stored record is absent from the input, check retrieval and context packing.
If the new date was supplied and the model uses the old one, the error arose later. A control with hand-selected
memories separates retrieval from the model's use of context.

Compare no explicit plan, a fixed plan and replanning after observations
on tasks whose inputs can change. For several agents, compare with a
single executor using a similar total budget. Otherwise extra calls
may explain the result. Report latency, cost and actions along with
success. Duplicate messages, superseded facts and lost pending actions
matter even if the final prose sounds convincing.

<span id="практика-и-первоисточники"></span>
<span id="реализации-для-разбора"></span>

## Sources

The memory, LangGraph and orchestration notebooks are listed in the
[[05 Источники/Courses/GenAI Agents|GenAI Agents catalogue]].

<span id="первоисточники-и-дальнейшее-чтение"></span>

### Primary sources and further reading

- [MemGPT](https://arxiv.org/html/2310.08560v2), 2023–2024.
- [Generative Agents](https://arxiv.org/html/2304.03442v2#S4), 2023.
- [HippoRAG](https://arxiv.org/abs/2405.14831), NeurIPS 2024.
- [Reflexion](https://arxiv.org/abs/2303.11366), 2023.
- [LLM+P](https://arxiv.org/abs/2304.11477), 2023.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601), 2023, и [LATS](https://arxiv.org/abs/2310.04406), 2023–2024.
- [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128), 2023–2024.
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/html/2310.01798v1), ICLR 2024.
- [WebDreamer](https://arxiv.org/html/2411.06559v1), 2024.
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), декабрь 2024.
- [Berkeley Advanced LLM Agents, Spring 2025](https://rdi.berkeley.edu/adv-llm-agents/sp25).
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), июнь 2023.

<span id="связанные-главы"></span>

## Related chapters

- Previous: [[en/00 Textbook/17 Tools and Agents/66 How agents manage context|How agents manage context]].
- Next: [[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|Evaluating agents]].
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер|Retrieval and RAG]].
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Agent security]].
