---
title: "AppWorld: state-based evaluation of interactive agents"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2407.18901v1
authors: [Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, Niranjan Balasubramanian]
year: 2024
source_language: en
---

# AppWorld: state-based evaluation of interactive agents

[Paper, arXiv v1](https://arxiv.org/html/2407.18901v1) ·
[ACL 2024 record](https://aclanthology.org/2024.acl-long.850/) ·
[Project repository](https://github.com/StonyBrookNLP/appworld).

[Figure 3](https://arxiv.org/html/2407.18901v1#S3.F3) follows two task
instances from one shopping scenario. Setup changes the starting database
along with the instruction and expected values. A validation solution uses
the app APIs; its resulting database must pass the same checks used to
evaluate agents. The validation solution establishes that the task is
solvable without prescribing one API sequence for every correct agent.

[Section 3.2](https://arxiv.org/html/2407.18901v1#S3.SS2) checks the database
diff for required changes and rejects changes outside the permitted set.
The paper reports task goal completion (passing every check in one task)
and scenario goal completion (passing every task in one scenario).
[Appendix D](https://arxiv.org/html/2407.18901v1#A4) explains the hash-based
state comparison.

The Bookvar chapter on
[[00 Учебник/17 Tools и Agents/68 Оценивание агентных систем#agent-eval-appworld|evaluating agent systems]]
reads Figure 3 and compares state-diff checks with a calendar example that
also needs an independent mail log. The figure is reproduced unchanged
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
