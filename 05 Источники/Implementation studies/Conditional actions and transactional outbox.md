---
title: "Conditional actions and transactional outbox"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://learn.microsoft.com/en-us/azure/architecture/databases/guide/transactional-out-box-cosmos#solution
---

# Conditional actions and transactional outbox

[RFC 9110, §13.1.1](https://www.rfc-editor.org/rfc/rfc9110.html#name-if-match)
describes a precondition on the selected representation. Strong entity
tags let the origin server detect a changed representation before applying
the method. This is distinct from authorizing the caller or deduplicating
the operation.

[SQLite, Transactions](https://www.sqlite.org/lang_transaction.html)
describes the write transaction acquired by `BEGIN IMMEDIATE`.
The textbook example checks authorization, the approved snapshot, and
the current draft inside that transaction. It does not issue a network
request from inside the database transaction.

[Microsoft Azure Architecture Center, Solution](https://learn.microsoft.com/en-us/azure/architecture/databases/guide/transactional-out-box-cosmos#solution)
shows a database commit followed by publication from a separate worker.
The textbook reproduces the [original sequence diagram](https://github.com/MicrosoftDocs/architecture-center/blob/629acf25fc1ab4096cfa102367a569867899b79e/docs/databases/guide/_images/out-box-pattern.svg)
without changes under the repository's
[CC BY 4.0 licence](https://github.com/MicrosoftDocs/architecture-center/blob/629acf25fc1ab4096cfa102367a569867899b79e/LICENSE).
The documentation's order event is an illustration of the pattern,
not the mail tool implemented in Bookvar.

[AWS Builders’ Library](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
discusses the separate contract for repeated requests: a stable identifier
for the same intent, matching parameters, a recorded effect, and a limit
on how long the service can deduplicate retries.

[[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем#agent-security-bound-approval|Textbook: version-bound approval and a single queue entry]].
