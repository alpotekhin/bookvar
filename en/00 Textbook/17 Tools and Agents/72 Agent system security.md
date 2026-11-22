---
title: Agent system security
description: "Trust boundaries, tool authorization, bound approval and observable side effects."
type: textbook-chapter
status: canonical
locale: en
translation_status: current
translation_of: "00 Учебник/17 Tools и Agents/72 Безопасность агентных систем.md"
canonical_revision: "da2c0bd67c4407f09355f487cf3863681d56f3eea0ffbfc5a4a304706e0128f6"
translation_reviewed: "2026-11-22"
last_updated: 2026-11-22
last_verified: 2026-11-22
reference_terms:
  - label: Action approval — mail version and permissions
    label_en: Action approval — mail version and permissions
    anchor: agent-security-bound-approval
  - label: Transactional outbox — queue entry and delivery
    label_en: Transactional outbox — queue entry and delivery
    anchor: agent-security-transactional-outbox
source_unit_id:
  - meeting-12-slides-agent-security-frame
  - meeting-12-slides-hybrid-agent-attack-surface
  - meeting-12-slides-dawn-song-agentic-threat-model-sequence
  - meeting-12-slides-agent-sql-rce-cases
  - meeting-12-slides-prompt-injection-agentpoison
  - meeting-12-slides-agent-safety-evaluation
  - meeting-12-slides-agent-defense-mechanisms
  - meeting-12-slides-dawn-song-privilege-control-sequence
  - meeting-12-slides-agent-monitoring-and-verification
  - meeting-12-slides-agent-safety-conclusion
  - meeting-12-reading-01-catalogue-record
  - meeting-12-reading-02-catalogue-record
  - meeting-12-reading-03-catalogue-record
  - meeting-12-reading-04-catalogue-record
  - meeting-06-slides-web-agent-adversarial-appendix
primary_sources:
  - https://rdi.berkeley.edu/adv-llm-agents/sp25
  - https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei
  - https://arxiv.org/abs/2407.12784
  - https://arxiv.org/abs/2504.11703
  - https://arxiv.org/html/2504.11703v1
  - https://arxiv.org/html/2504.11703v3
  - https://arxiv.org/abs/2504.11358
  - https://web.mit.edu/Saltzer/www/publications/protection/Basic.html
  - https://www.rfc-editor.org/rfc/rfc9110.html#name-if-match
  - https://www.sqlite.org/lang_transaction.html
  - https://www.sqlite.org/c3ref/bind_blob.html
  - https://learn.microsoft.com/en-us/azure/architecture/databases/guide/transactional-out-box-cosmos#solution
  - https://www.usenix.org/conference/13th-usenix-security-symposium/privtrans-automatically-partitioning-programs-privilege
---

# Agent system security

The user asks the agent to read a quarterly report and save an email
draft. The report claims that sending has already been approved.
That sentence reaches the model alongside the quarterly figures.
The model may take it as permission and propose a send.

The proposal does not call the mail service. Application code
will call it only if it accepts the arguments and finds a valid
authorization. Preparing the draft requires permission to read the
report and write the draft. Sending requires separate approval from
the user's session.

The [[en/00 Textbook/17 Tools and Agents/65 How a function call becomes an action|tool-use chapter]]
explains how a model response becomes an action.

<span id="safety-security-and-the-hybrid-system-boundary"></span>
<span id="два-класса-риска-в-гибридной-системе-safety-и-security"></span>
<span id="ошибка-агента-и-намеренная-атака"></span>
<span id="ошибка-модели-и-полномочия-приложения"></span>

## Model errors and application permissions

A model can mistake “prepare” for “send” even when nobody is
attacking it. The permission check is needed either way. For an
intentional attack, also say what the adversary controls: the report,
the user's request, an application component, or the execution
environment. In the Berkeley lecture, *safety* concerns preventing
harmful behavior, and *security* concerns deliberate interference.

Confidentiality fails if someone receives a report they cannot access.
Integrity fails if the agent changes the report or stores an invented
result as fact. Availability fails if a loop exhausts the request limit
or consumes resources needed for other tasks. A mass mailing can cost
both data confidentiality and the mail quota.

An agent combines an LLM with ordinary software. Storage can check
whether the user may read a document. The executor can compare the
recipient with the confirmed one. The model process need not hold the
mail service credential. These checks can work even if the model
misunderstands the report. See
[Dawn Song, Berkeley Meeting 12, slides 12–15](https://rdi.berkeley.edu/adv-llm-agents/sp25).

<span id="семь-поверхностей-host-user-system-model-tools-response-long-running-process"></span>
<span id="семь-границ-атаки-от-поставки-модели-до-долгоживущего-процесса"></span>
<span id="семь-границ-агентной-системы"></span>
<span id="где-агент-получает-данные-и-меняет-состояние"></span>

## Where the agent reads data and changes state

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-security-2026-10-21/m12-p13.png|Seven transitions cover setup, user input, interaction with the model and tools, the reply, and a task that continues running.]]

*Dawn Song, [Berkeley Advanced LLM Agents, Meeting 12](https://rdi.berkeley.edu/adv-llm-agents/sp25),
slide 13. Full original slide.*

1. **Host → system.** The operator chooses model artifacts,
   dependencies and the runtime. Checking the text of the report
   cannot detect tampering with the code that is meant to check it.
2. **User → system.** Establish the requesting identity and its
   resource permissions. The service account may be able to read
   more reports than the user is allowed to see.
3. **System → model.** The application assembles context from
   the user request, excerpts, search results and history. Source
   labels help the model distinguish them, but cannot enforce what
   the model will do with the words.
4. **Model → system.** The model returns text or structured arguments.
   Application code checks the form and decides whether the call is
   authorized before it reaches a tool.
5. **System ↔ external world.** File, database, network and mail
   permissions limit what the executor can do. Reading a report,
   saving a draft and sending it can use separate grants.
6. **System → user.** The reply itself can disclose data. Blocking the
   send tool does not prevent someone else's report appearing in a
   chat response.
7. **Long-running process.** A resumed task may see old memory or
   retry an earlier request. A timeout may leave the mail outcome
   unknown, and a repeated request may duplicate the action. Track
   task state, permission expiry and retry semantics at this boundary.

See [Meeting 12, slides 20–26](https://rdi.berkeley.edu/adv-llm-agents/sp25).

<span id="threat-model-before-defenses"></span>
<span id="threat-model-строят-до-выбора-защиты"></span>
<span id="модель-угроз-на-примере-работы-с-отчётом"></span>
<span id="кто-может-изменить-отчёт-права-и-действие"></span>

## Who can change the report, permissions, or action

In the report scenario, the user may read the report and asks only
for a draft. The adversary can change the report before the agent
reads it, but cannot change the authenticated request, the executor,
its permissions, or the mail API. Protect the report, the address book
and the ability to send mail. A prohibited transfer requires a real
effect at the receiving boundary; a proposed transfer is a separate
observation.

This is an indirect-injection threat model. If the user request itself
is hostile, its authority and resource scope need checking. If the
executor is compromised, a policy check inside that executor no longer
serves as an independent boundary.

The *trusted computing base* contains the parts that must work for
the guarantee to hold. For the send restriction, it includes parsing
the proposed call, checking permission, selecting credentials and
invoking the correct mail API. Treat the model's suggestion as
untrusted, regardless of how confidently it argues for it.

Record what feedback was available during an evaluation. One attempt
without an observed outcome is different from a long sequence whose
results are visible. Comparing two defenses requires the same limits
and observations. See [Meeting 12, slides 14–15 and 27–28](https://rdi.berkeley.edu/adv-llm-agents/sp25).

<span id="когда-output-модели-становится-query-parameter-or-executable-code"></span>
<span id="когда-вывод-модели-превращается-в-запрос-параметр-или-исполняемый-код"></span>
<span id="как-вывод-модели-используется-программой"></span>
<span id="что-программа-делает-с-ответом-модели"></span>

## What the program does with the model's output

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/security/m12-p27.png|Five uses of model output: display it, pass it to a model, choose a branch, set tool arguments, or execute code.]]

*Dawn Song, [Berkeley Advanced LLM Agents, Meeting 12](https://rdi.berkeley.edu/adv-llm-agents/sp25),
slide 27.*

The lecture labels these uses U1 to U5. Showing text to a user can
already reveal confidential data. The other uses can also affect a
program branch, a tool call or a runtime. The same output string
therefore needs different checks depending on what consumes it.

A schema can check that a proposed mail call has a string recipient
and a body. It cannot show that this recipient was approved, that
the attachment can be shared, or that the user intended to send.
Those facts come from the application session, resource permissions
and the confirmed action.

<span id="параметризация-sql-какую-проблему-она-решает"></span>
<span id="где-заканчивается-sql-запрос-и-начинается-значение"></span>

### Keeping SQL statements separate from values

A report title can contain an apostrophe. Pass the title and owner
as values so the SQL statement stays fixed. The two reports below
have the same title but different owners. The executor gets the owner
from the authenticated session, not from the model.

~~~python
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE reports(id INTEGER, owner TEXT, title TEXT)")
db.executemany(
    "INSERT INTO reports VALUES (?, ?, ?)",
    [(1, "alice", "Quarter's report"), (2, "bob", "Quarter's report")],
)

lookup = "SELECT id FROM reports WHERE owner = ? AND title = ?"
assert db.execute(lookup, ("alice", "Quarter's report")).fetchall() == [(1,)]
assert db.execute(lookup, ("alice", "Unknown report")).fetchall() == []
db.close()
~~~

SQLite [binds values to prepared statements](https://www.sqlite.org/c3ref/bind_blob.html).
The apostrophe stays part of the title, and the owner predicate
stays part of the query. Binding is not an authorization mechanism:
if the tool accepts an arbitrary statement, the model can still ask
for another table. Expose a narrow lookup operation and give the
database identity only the rights it needs.

A syntactically valid program can still read files or use network
resources. An execution environment needs limits on files, network,
secrets, time and memory. A mounted home directory remains readable
from inside a container if the mount and process permissions permit it.

<span id="инструкция-внутри-страницы-данные-а-не-полномочие"></span>
<span id="инструкция-внутри-страницы--данные-а-не-полномочие"></span>
<span id="инструкция-внутри-документа"></span>
<span id="может-ли-документ-разрешить-отправку"></span>

## Approval cannot come from a document

<!-- source_unit_id: meeting-06-slides-web-agent-adversarial-appendix -->

The quarterly figures are evidence about the reporting period.
A claim inside the report that a send was approved is testimony by
the document's author. The application must get approval from the
authenticated user, separately from the report.

In a test, record whether the model proposed a send, whether the
executor denied it and whether the service accepted a call. A mistaken
proposal can be blocked. A safe-looking final answer can also follow
a call that already ran.

An imperative sentence in an external document is not sufficient
evidence of an attack. The user may have requested a translation of
a quoted instruction. Source delimiters can help the model tell these
cases apart, but permission must still be checked after the response.
See [Liu et al., USENIX Security 2024, sections 2–3](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei)
and [Berkeley Meeting 6, adversarial appendix](https://rdi.berkeley.edu/adv-llm-agents/sp25).

<span id="direct-indirect-and-retrieval-memory-poisoning"></span>
<span id="прямые-и-косвенные-инъекции-отравление-retrieval-и-памяти"></span>
<span id="прямые-и-косвенные-инъекции"></span>

## Direct and indirect prompt injection

A direct injection comes from a user prompt. An indirect one comes
from data such as a document, mail message, web page, search result,
or another agent's response. The distinction says who controls the
input and where to enforce its authority. Giving an agent a browser
does not give each website authority over the user's tools.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/security/m12-p48.png|A lecture example shows untrusted text changing the result of a document assessment.]]

*Dawn Song, [Berkeley Meeting 12](https://rdi.berkeley.edu/adv-llm-agents/sp25),
slide 48. The example concerns a changed classification result.
See [Formalizing and Benchmarking Prompt Injection Attacks and Defenses](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei)
for the distinction between replacing a response label and switching
the task. Either can be measured without confusing it with an executed tool call.*

A positive label in this example is an output error. An API call
and the resulting external change would need separate observations.
The diagram is not a recommendation to automate hiring decisions.

<span id="agentpoison-как-запись-оказывается-в-нужном-контексте"></span>
<span id="как-контролировать-записи-в-памяти-агента"></span>

### Controlling records in agent memory

A record saved today can return in another session. Suppose the user
loses access to the original report but a summary and a sending plan
remain in the index. Deleting the source alone does not remove those
records. Track the writer, source resource and revision, enforce read
permission during retrieval, and propagate revocation to derivatives.

In [AgentPoison, section 4.1](https://arxiv.org/html/2407.12784v1#S4.SS1),
ASR-r is the share of trials in which all retrieved demonstrations
were contaminated. ASR-a concerns the agent's action given successful
retrieval. ASR-t counts the final effect in the environment. In the
report case, an untrusted record in context, a proposed action and
an accepted send are distinct observations. Results measured in
modeled environments are not observations about real infrastructure.

The [[en/00 Textbook/17 Tools and Agents/67 Agent memory, planning, and orchestration|memory chapter]]
shows which records persist across sessions. Even an authentic
document cannot expand the set of allowed actions.

<span id="system-evaluation-red-teaming-and-agentxploit"></span>
<span id="проверка-системы-целиком-red-teaming-и-agentxploit"></span>
<span id="проверка-всей-системы-и-адаптивный-противник"></span>
<span id="как-проверить-дошёл-ли-вызов-до-сервиса"></span>

## Checking whether a call reached the service

Set the initial report, permissions, task history and outbox in
an integration test. Use a mail-service double that records calls
without sending email. After the run, compare the model's proposal,
the decision of the policy checker and the call recorded by the
double. “Sent” in the model's reply is not a receipt.

[Berkeley Meeting 12, slides 56–62](https://rdi.berkeley.edu/adv-llm-agents/sp25)
uses a black-box setup to discuss agent evaluation. To compare
defenses, state whether the evaluator saw only the user-visible
result or also the executor's decision, how many trials were allowed,
and which components stayed fixed.

Reset the stored state as well as the conversation between independent
trials. A stale memory or a queued message can change the next trial.
Record model, tool and policy versions, the reset procedure, the
budget and the feedback the evaluation exposes.

Turning off every tool makes some unsafe calls impossible but also
prevents the requested draft from being saved. Check that the
prohibited send did not occur **and** that the authorized draft was
created. Also measure blocked legitimate calls, latency and fidelity
to the source report. See the
[[en/00 Textbook/17 Tools and Agents/68 Evaluating agent systems|chapter on agent evaluation]].


<span id="defense-in-depth-hardening-validation-policy-and-isolation"></span>
<span id="эшелонированная-защита-укрепление-проверка-политика-и-изоляция"></span>
<span id="несколько-уровней-защиты"></span>
<span id="какие-проверки-дополняют-друг-друга"></span>

## Which checks complement one another

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/security/m12-p65.png|Defenses at input, inside the model, at the tool boundary and during execution.]]

*Dawn Song, [Berkeley Advanced LLM Agents, Meeting 12](https://rdi.berkeley.edu/adv-llm-agents/sp25),
slide 65.*

Model training and a detector can reduce unsafe proposals. A schema
detects malformed calls. An access policy decides whether to execute a
well-formed call. Isolation limits the resources a process can reach
if checks fail, and a log records what happened. These controls address
different failures.

If the same model reads the report, proposes the send and decides
whether to allow it, one misunderstanding can affect both decisions.
Have application code compare the recipient and attachment revision
against an approval record the model cannot modify.

Saltzer and Schroeder call this *complete mediation*:

> Every access to every object must be checked for authority.

— [The Protection of Information in Computer Systems, section I.A.3](https://web.mit.edu/Saltzer/www/publications/protection/Basic.html).

Check every use of the tool, including a retry, a restored task, or
another route to the same service. An old permission check must not
survive revocation indefinitely. *Fail-safe defaults* mean denying
actions without permission. *Least privilege* means granting only
the resources and time required: this task needs report reading and
draft creation, not sending.

<span id="progent-least-privilege-as-an-executable-tool-call-policy"></span>
<span id="progent-наименьшие-привилегии-как-исполнимая-политика-вызова"></span>
<span id="progent-проверка-вызова-вне-модели"></span>

## Progent: checking tool calls outside the model

[Progent](https://arxiv.org/html/2504.11703v1#S3) puts a policy check
between the model and its tools. A rule specifies the tool, conditions
on its arguments, an allow or forbid effect, and a response to a denial.
The check runs before the tool. The executor can return a denial to let
the agent revise the plan.

A fallback might stop the run, request review or return an error.
When a send is blocked, the agent can still save the draft if the
policy permits it. A second attempt to send needs the same authorization.
A fallback changes what happens after a denial; it is not itself
permission to perform the action.

<span id="версия-лекции-и-последующее-развитие"></span>
<span id="как-progent-выбирает-правило-в-v1-и-v3"></span>

### How Progent chooses a rule in v1 and v3

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/12-dawn-song-privilege-control-sequence.png|The lecture's Progent rule syntax includes a tool, argument conditions, effect, priority and fallback.]]

*Dawn Song, [Berkeley Meeting 12](https://rdi.berkeley.edu/adv-llm-agents/sp25),
slide 81, alongside [Progent v1, section 3](https://arxiv.org/html/2504.11703v1#S3).
The diagram shows v1, not v3.*

In v1, higher-priority rules are checked first. At equal priority,
forbid comes before allow. The first matching rule decides the result.
A deny rule is therefore not automatically stronger than an allow rule
of higher priority. If nothing matches, the call is denied by default.

[Version 3, dated 14 May 2026](https://arxiv.org/html/2504.11703v3#S4),
removes rule priority within a policy. The executor checks for matching
forbids first. A call may proceed if no forbid matches and an allow
does. This version also checks whether a proposed policy change
adds any allowed calls.

<span id="пример-решения-политики"></span>
<span id="какие-вызовы-разрешены-в-задаче-с-отчётом"></span>

### Which calls the report task allows

The policy allows reading `report-7` and saving a draft, but explicitly
forbids sending. Assume the application has already validated the input
schema and resolved tool names against a fixed registry. The example
models the v3 decision order, without real tools or an SMT solver.

~~~python
def forbidden(call):
    return call["tool"] == "send"

def allowed(call):
    return (
        call["tool"] == "read"
        and call.get("report_id") == "report-7"
    ) or call["tool"] == "save_draft"

def decide(call):
    if forbidden(call):
        return False
    return allowed(call)  # Нет подходящего allow — отказ.

assert decide({"tool": "read", "report_id": "report-7"})
assert not decide({"tool": "read", "report_id": "report-8"})
assert decide({"tool": "save_draft"})
assert not decide({"tool": "send", "to": "outside@example.invalid"})
assert not decide({"tool": "delete"})
~~~

Sending is denied regardless of the model's explanation. Reading
`report-8` is denied because no allow rule applies. The draft can
still contain a bad summary: authorization does not establish the
accuracy of the content.

In a real save operation, also check the workspace and owner. If a
file reference is used, the file actually opened must be the one
the executor checked. File APIs must enforce containment and
access permissions; a string prefix alone is insufficient.

<span id="обновление-политики-и-расширение-полномочий"></span>
<span id="как-проверить-изменение-разрешений"></span>

### Checking a change to permissions

Let $A(P)$ be the set of calls policy $P$ allows, including their
arguments. An update that allows a second report expands that set
even if it introduces no tool name. In v3 the change is a narrowing,
and can be applied without expansion approval, if

$$
A(P')\subseteq A(P).
$$

Encode the policies as predicates and ask whether they admit a
counterexample: a call $v$ allowed by the new policy but denied
by the old one.

$$
\exists v:\quad
\Phi_{P'}(v)\land\neg\Phi_P(v).
$$

A satisfying call requires a decision from the approver. If the
formula is unsatisfiable, there is no added call under the encoding.
An unknown result or an encoding error cannot establish that.

For a finite set we can compare the allowed calls directly:

~~~python
old = frozenset({"read:report-7", "save:draft"})
narrower = frozenset({"read:report-7"})
different = frozenset({"send:outside"})

assert narrower <= old
assert not (different <= old)
assert len(different) < len(old)
assert different - old == {"send:outside"}
~~~

The last policy lists fewer calls, yet it adds a send permission.
Counting rules or tools would miss this. Progent uses an SMT check
over symbolic argument constraints rather than listing all possible
calls. See [Progent v3, sections 4–5](https://arxiv.org/html/2504.11703v3#S4).

Between approved expansions, the allowed set can only shrink. That
does not repair a broad initial policy. Nor does an “approved” update
necessarily mean a person saw it: v3 can automatically allow or deny
expansions, apply rules by tool, or ask for user review. Inspect that
configuration before depending on it for sensitive actions.
See [section 7](https://arxiv.org/html/2504.11703v3#S7).

<span id="порядок-политик-не-равен-пересечению-разрешений"></span>
<span id="как-сочетаются-общая-политика-и-политика-задачи"></span>

### Combining a generic policy and a task policy

In Progent v3's Algorithm 4, an explicit decision by the generic
policy is final. The task policy is consulted only when there is
no explicit decision at the earlier layer.

A generic rule allowing all report reads is not narrowed by a later
task rule about `report-8`. An explicit generic forbid cannot be
reversed by the task policy. This is not set intersection. To require
two independent checks, arrange for both to run. The rule here is the
published composition, not a claim about every implementation.
See [Progent v3, section 7 and Algorithm 4](https://arxiv.org/html/2504.11703v3#S7).

The check covers the calls it sees. For example, an MCP proxy does
not intercept a built-in tool that bypasses MCP. Errors in text
output are another boundary. The guarantee depends on correctly
parsing the call, encoding the policy and invoking the checked
tool. See [Progent v3, section 9](https://arxiv.org/html/2504.11703v3#S9).

<span id="подтверждение-изменение-состояния-и-повтор"></span>
<span id="что-именно-подтверждает-пользователь"></span>

## What the user approves

If the user later approves sending, the approval must name the
sender, recipient, body and attachment revision. “Send the report”
leaves the action ambiguous if the agent can replace an address
or attachment after showing the confirmation.

Suppose the user sees draft `d7` at version 4. Another process moves
it to version 5 before sending. Reading the version before a separate
send catches some cases, but a write can still occur between the
check and the send. A service that supports a conditional operation
can check “send version 4 only” as part of the action. Without that
contract, there is a time-of-check/time-of-use (TOCTOU) gap.
Another separate read does not close it.

A timeout poses a different question. The client has no response,
but the server may have accepted the request. An idempotency key
helps only if the endpoint stores or can recover the result of
the original operation. A random ID in the client's log cannot
deduplicate a request at the server.

Log proposed, authorized, executed and unknown outcomes separately.
Approval does not prove execution. A missing response does not prove
that nothing happened. These are properties of the executor and
service, not atomicity guarantees supplied by Progent.

<a id="agent-security-bound-approval"></a>
<span id="согласие-на-конкретную-версию-письма"></span>

### Approving one draft version

The user sees the sender, recipient, body and attachment revision
of draft `d7`, version 4. The confirmation records that snapshot as
`a4`. The agent can reference the approval but cannot create one
on behalf of the user. The session, not the model's arguments,
supplies the actor.

A changed recipient does not match `a4`. If the submitted arguments
still describe version 4 but the saved draft is now version 5,
they match `a4` but fail the current-state check. Similarly,
[`If-Match` in RFC 9110, section 13.1.1](https://www.rfc-editor.org/rfc/rfc9110.html#name-if-match)
requires a strong comparison of an entity tag before acting on a
resource. It does not grant permission, and the server must apply
the precondition to the resource used by the operation.

The draft, grants and outbox here are in one SQLite database.
[`BEGIN IMMEDIATE`](https://www.sqlite.org/lang_transaction.html)
starts a write transaction before reading. A second writer to the
same database cannot change those rows between the checks and the
enqueue. Starting a write transaction can fail with `SQLITE_BUSY`
when another writer holds it. The example models local states in
memory and never contacts a mail service.

~~~python
import json
import sqlite3
from typing import NamedTuple

class Mail(NamedTuple):
    draft: str
    version: int
    sender: str
    recipient: str
    body: str
    attachment_revision: str

def snapshot(mail):
    return json.dumps(mail, ensure_ascii=False, separators=(",", ":"))

db = sqlite3.connect(":memory:", isolation_level=None)
db.executescript("""
    CREATE TABLE drafts (id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
    CREATE TABLE grants (actor TEXT PRIMARY KEY, epoch INTEGER NOT NULL);
    CREATE TABLE approvals (
        id TEXT PRIMARY KEY, actor TEXT NOT NULL,
        epoch INTEGER NOT NULL, snapshot TEXT NOT NULL
    );
    CREATE TABLE outbox (
        approval TEXT PRIMARY KEY, snapshot TEXT NOT NULL
    );
""")

def one(sql, *params):
    return db.execute(sql, params).fetchone()

def receipt(actor, approval):
    row = one(
        "SELECT o.approval FROM outbox o JOIN approvals a ON a.id = o.approval "
        "WHERE a.actor = ? AND a.id = ?", actor, approval,
    )
    return ("queued", row[0]) if row else None

def enqueue(actor, approval, mail, fault=None):
    # actor comes from the authenticated session, not the tool arguments.
    payload = snapshot(mail)
    db.execute("BEGIN IMMEDIATE")
    with db:
        approved = one(
            "SELECT epoch, snapshot FROM approvals WHERE id = ? AND actor = ?",
            approval, actor,
        )
        if not approved or approved[1] != payload:
            return "approval_mismatch"
        if one("SELECT epoch FROM grants WHERE actor = ?", actor) != (approved[0],):
            return "grant_changed"
        previous = receipt(actor, approval)
        if previous:
            return previous
        if one("SELECT snapshot FROM drafts WHERE id = ?", mail.draft) != (payload,):
            return "draft_changed"
        db.execute("INSERT INTO outbox VALUES (?, ?)", (approval, payload))
        if fault == "before_commit":
            raise RuntimeError("before commit")
    if fault == "after_commit":
        raise TimeoutError("reply lost")
    return ("queued", approval)

v4 = Mail("d7", 4, "me@example.invalid", "team@example.invalid",
          "quarterly summary", "report-7@sha256:fixture-v4")
v5 = v4._replace(version=5, body="corrected quarterly summary")
db.execute("INSERT INTO drafts VALUES (?, ?)", ("d7", snapshot(v4)))
db.execute("INSERT INTO grants VALUES (?, ?)", ("alice", 1))

def confirm_for_test(key, mail):
    # Test fixture: the trusted UI, not the agent, creates approval rows.
    db.execute("INSERT INTO approvals VALUES (?, ?, ?, ?)",
               (key, "alice", 1, snapshot(mail)))

confirm_for_test("a4", v4)
tampered = v4._replace(recipient="other@example.invalid")
assert enqueue("alice", "a4", tampered) == "approval_mismatch"
assert enqueue("alice", "a4", v4._replace(body="changed")) == "approval_mismatch"
assert enqueue("alice", "a4", v4._replace(
    attachment_revision="other-report")) == "approval_mismatch"
assert enqueue("bob", "a4", v4) == "approval_mismatch"
db.execute("UPDATE drafts SET snapshot = ? WHERE id = ?", (snapshot(v5), "d7"))
assert enqueue("alice", "a4", v4) == "draft_changed"
assert one("SELECT COUNT(*) FROM outbox") == (0,)

confirm_for_test("a5", v5)
try:
    enqueue("alice", "a5", v5, fault="before_commit")
except RuntimeError:
    assert receipt("alice", "a5") is None
else:
    raise AssertionError("expected rollback")
try:
    enqueue("alice", "a5", v5, fault="after_commit")
except TimeoutError:
    pass
else:
    raise AssertionError("expected lost response")
assert receipt("alice", "a5") == ("queued", "a5")
assert enqueue("alice", "a5", v5) == ("queued", "a5")
assert one("SELECT snapshot FROM outbox WHERE approval = ?", "a5") == (snapshot(v5),)
assert one("SELECT COUNT(*) FROM outbox") == (1,)

db.execute("DELETE FROM grants WHERE actor = ?", ("alice",))
assert enqueue("alice", "a5", v5) == "grant_changed"
db.execute("INSERT INTO grants VALUES (?, ?)", ("alice", 2))
assert enqueue("alice", "a5", v5) == "grant_changed"
assert receipt("alice", "a5") == ("queued", "a5")
assert receipt("bob", "a5") is None
assert one("SELECT COUNT(*) FROM outbox") == (1,)
~~~

| Event | Result in the outbox |
|---|---|
| Arguments differ from snapshot `a4` | No new row |
| Draft is version 5 but approval was for version 4 | No new row |
| Version 5 approved as `a5`, with a failure before commit | The insert rolls back |
| The response is lost after `a5` commits | A `queued` receipt already exists |
| Retry with the same `a5` | Same receipt, still one row |

The approval identifier is the queue's unique key. An intentional
second send requires another approval. See the
[[en/00 Textbook/17 Tools and Agents/65 How a function call becomes an action#tool-retry-idempotency|lost-response example]]
for the retry contract.

The record for `a5` includes a grant epoch. Revocation prevents a
new enqueue, and a new grant at epoch 2 does not restore an
approval issued at epoch 1. In this policy the owner can still
read the receipt: permission to inspect a past action is separate
from permission to repeat it. Revocation does not delete the
committed row. The fixture's attachment version is illustrative;
a deployed system needs a reference to immutable contents.

<a id="agent-security-transactional-outbox"></a>
<span id="запись-в-очереди-и-отправка-письма"></span>

### Queueing and sending the email

In Microsoft's example, the service stores the order and an
`OrderCreated` event in one transaction, then returns the order ID.
A background worker reads the outbox, publishes the event to the
message bus, and marks it processed.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-approval-2026-11-09/microsoft-transactional-outbox.svg|Transactional outbox: the transaction stores the order and the event, and a separate worker publishes the event after commit.]]

*Microsoft Azure Architecture Center,
[Implement the Transactional Outbox pattern, Solution](https://learn.microsoft.com/en-us/azure/architecture/databases/guide/transactional-out-box-cosmos#solution).
[Unmodified original SVG](https://github.com/MicrosoftDocs/architecture-center/blob/629acf25fc1ab4096cfa102367a569867899b79e/docs/databases/guide/_images/out-box-pattern.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

For the email, one row stores the approved snapshot and proves it
was queued. It does not prove the mail service accepted it. A relay
can hand off a message and then crash before recording that fact.
Preventing a second service-side send requires a deduplication
contract at that boundary and a known retention period. If there
is no such contract and no status lookup, the outcome remains
unknown. The SQLite example checks one queue entry; it makes no
claim of exactly-once delivery.

<span id="разделение-привилегий-опасная-операция-живёт-в-меньшем-компоненте"></span>
<span id="разделение-привилегий"></span>

## Separating privileges

Call restrictions are easier to enforce when the model process
does not hold the credential for the sensitive operation. In
*privilege separation*, the main process runs without privileges.
A smaller monitor performs selected operations and retains
privileged state in its own address space.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/berkeley-agents-2025/security/m12-p90.png|Privtrans partitions the code into an unprivileged process and a monitor that keeps privileged state.]]

*Dawn Song, [Berkeley Meeting 12](https://rdi.berkeley.edu/adv-llm-agents/sp25),
slide 90. See David Brumley and Dawn Song,
[Privtrans, USENIX Security 2004](https://www.usenix.org/conference/13th-usenix-security-symposium/privtrans-automatically-partitioning-programs-privilege).*

In [section 2 of Privtrans](https://www.usenix.org/legacy/publications/library/proceedings/sec04/tech/full_papers/brumley/brumley_html/index.html),
the monitor creates a privileged socket and returns an opaque
handle. A later call supplies the handle. The monitor finds the
descriptor, checks the operation and executes it. Returning the
descriptor itself would allow further use outside that check.
A document-signing service can likewise sign without returning
its private key.

Privtrans partitions C programs using annotations on privileged
functions and data. It propagates those annotations, passes
allowed arguments and results over RPC, and keeps state in the
monitor across calls. The programmer still specifies the intended
application policy.

For the report task, give the mail service a narrow API that
checks the approval. Keep its credential out of the model process.
Two model roles with the same credential are not isolated from
one another at this boundary. The monitor and its interface
remain part of the trusted computing base.


<span id="monitoring-information-flow-datasentinel-and-formal-policy-verification"></span>
<span id="наблюдение-за-выполнением-потоки-данных-и-формальная-проверка-политики"></span>
<span id="происхождение-конфиденциальность-и-потоки-данных"></span>
<span id="куда-можно-передать-данные-из-отчёта"></span>

## Where report data may go

Track confidentiality and the authority of instructions separately.
A private report can contain untrusted instructions. Confidentiality
governs who can receive its contents. Instruction authority governs
whether the text can change the task or permissions.

In *information-flow tracking*, the application carries labels along
with data and its derivatives. A summary of a private report remains
private under a conservative policy. Removing the filename does not
remove the facts. The label also needs to survive a handoff to a model
that never saw the original report.

For example, combining a private report with a public reference
normally produces a private result. Posting the result to a public
channel is denied. A release exception, or *declassification*, needs
a trusted rule, such as an approved anonymized table. A request from
the model to label the result public is not evidence that sensitive
information is gone.

Keep source references so that revocation can reach summaries and
plans derived from a report. Keep these labels in application state
rather than relying on text in the prompt. A conservative rule can
also block a harmless aggregate; define which transformations and
releases are permitted. See
[Meeting 12, slides 92–97](https://rdi.berkeley.edu/adv-llm-agents/sp25).

<span id="datasentinel-детектор-который-должен-поддаваться-инъекции"></span>
<span id="datasentinel-проверка-по-контрольному-маркеру"></span>

## DataSentinel: detecting changes to a control answer

Before reading a document, a separate model gets a control task:
return a random marker. It then reads the text being checked. A
response without the marker flags the document for review. DataSentinel
checks for a substring rather than requiring the entire output to be
equal to the marker.

A model that returns the marker on every document is a poor detector:
its answers do not distinguish clean and suspicious input. The
detector and the model that processes the report have different jobs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/agent-security-2026-10-21/datasentinel-detection.png|The ordinary model returns the control marker in both cases; the trained detector gives different answers.]]

*Yupei Liu et al., [DataSentinel v1, Figure 1](https://arxiv.org/html/2504.11358v1#S1.F1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The figure illustrates why a missing marker triggers review.
It is not a trace of a run for this chapter.*

<span id="зачем-обучать-против-адаптивной-атаки"></span>
<span id="на-чём-обучается-детектор"></span>

### What the detector learns

On clean documents, training favors returning the marker. On documents
with an injection, it favors losing the marker so the application
gets a signal. This training does not change the model that reads
the report. The [paper's section IV](https://arxiv.org/html/2504.11358v1#S4)
describes training against examples that account for the detector's
current behavior. Acceptance still needs separate evaluation data
and measurements of the whole application.

<span id="что-означает-ошибка-детектора"></span>
<span id="сколько-документов-попадёт-на-проверку"></span>

### How many documents are flagged

Suppose 100 of 10,000 documents contain an injection. The detector
flags 90 of those and misses 10. A 1% false positive rate among
9,900 clean documents adds another 99 flags. A reviewer sees 189
flags, and only 90 correspond to injections: a precision of about
47.6%. These made-up counts show how the base rate affects review load.

In one row of Table VI, the authors report a false negative rate of
0.87 for Minimax in an adaptive sentiment-analysis setting, where
the original and injected tasks concern the same type of prediction.
That is a result under a specific protocol, not the overall false
negative rate of DataSentinel. See
[section V-D and the limitations in section VI](https://arxiv.org/html/2504.11358v1#S5.SS4).

Use the signal to route documents for review, quarantine them, or
stop reading. It cannot grant permission to send a report. Training
on a finite sample does not establish correct detection on all inputs.

<span id="почему-одна-guardrail-не-делает-систему-безопасной"></span>
<span id="почему-один-защитный-фильтр-не-делает-систему-безопасной"></span>
<span id="проверки-перед-подключением-инструмента"></span>
<span id="что-проверить-перед-подключением-инструмента"></span>

## Checks before enabling a tool

For the report task, an integration test should show that the allowed
report can be read, another is blocked, the draft exists, and no
unapproved send reaches the mail API. Check the same property after
resuming a task, changing call arguments, or using another application
path to the tool.

When the user approves a send, a changed recipient or attachment
revision must no longer match the approval. Repeating a key should
not duplicate the operation if the endpoint promises that behavior.
After a timeout, keep the outcome unknown until it can be reconciled
with the service. Revocation blocks the next unauthorized call even
if the agent had prepared it earlier.

Link the task, user, policy revision, normalized action, authorization
decision, and observed effect in the audit log. A snapshot ID and
digest usually suffice for correlation; keep secrets and the report
body out of ordinary logs. Keep enough provenance to investigate
without making the log a new disclosure channel.

Document the scenario, assumptions, attempt budget and untested
boundaries. These tests do not prove that the model recognizes
every injection, that authorized emails are factually correct, or
that the runtime handles unknown vulnerabilities.

<span id="источники-и-дальнейшее-чтение"></span>
<span id="источники"></span>

## Sources

- Dawn Song, [Towards Building Safe & Secure Agentic AI](https://rdi.berkeley.edu/adv-llm-agents/sp25), Berkeley Advanced LLM Agents, Meeting 12, 2025.
- Yupei Liu et al., [Formalizing and Benchmarking Prompt Injection Attacks and Defenses](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei), USENIX Security 2024, sections 2–3.
- Zhaorun Chen et al., [AgentPoison v1](https://arxiv.org/html/2407.12784v1), NeurIPS 2024, sections 3–4. Evaluation distinguishes retrieved records, actions, and their effects.
- Tianneng Shi et al., [Progent v1](https://arxiv.org/html/2504.11703v1), sections 3–4, and [Progent v3](https://arxiv.org/html/2504.11703v3), sections 4–7 and 9. See also the [pinned implementation](https://github.com/sunblaze-ucb/progent/tree/5be7b63fa96f70bc19b72fbcee81f1c0bcc1a565).
- Yupei Liu et al., [DataSentinel v1](https://arxiv.org/html/2504.11358v1), IEEE S&P 2025, sections III–IV, V-D and VI.
- Jerome H. Saltzer and Michael D. Schroeder, [The Protection of Information in Computer Systems](https://web.mit.edu/Saltzer/www/publications/protection/Basic.html), 1975, section I.A.3.
- David Brumley and Dawn Song, [Privtrans](https://www.usenix.org/legacy/publications/library/proceedings/sec04/tech/full_papers/brumley/brumley_html/index.html), USENIX Security 2004, section 2.
- [[05 Источники/Implementation studies/Conditional actions and transactional outbox|Conditional actions and the transactional outbox]]: RFC 9110, SQLite transactions and the Microsoft diagram.

Previous: [[en/00 Textbook/17 Tools and Agents/71 Scientific discovery agents|Scientific discovery agents]].
This is the final chapter of the core course. Return to the [[en/00 Textbook/_index|textbook contents]].
