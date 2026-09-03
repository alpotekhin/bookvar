# Berkeley Advanced LLM Agents Textbook Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate all twelve Berkeley Advanced LLM Agents Spring 2025 lectures into a coherent advanced agent route covering reasoning, memory, planning, coding and GUI agents, formal reasoning, discovery, and security.

**Architecture:** The existing Bookvar chapters 65–68 remain the prerequisite route for tool calls, harness state, memory/orchestration, and evaluation. Seven new advanced chapters deepen the mechanisms without mirroring the lecture calendar. Each chapter combines complete course arguments, original slide sequences, primary papers, and explicit system interfaces. A verifiable coding-agent practice and an AgentX-style project capstone turn the theory into reproducible work without misrepresenting unofficial lab material as Berkeley coursework.

**Tech Stack:** Obsidian Markdown, original Berkeley PDF slide assets, YAML provenance ledgers, Lean 4 interface examples, agent evaluation schemas, Astro/Starlight, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-04-stanford-berkeley-ingestion-design.md`

## Global Constraints

- Complete `2026-09-04-course-source-ledgers.md` Tasks 1 and 3 before this plan.
- Keep all twelve lecture bundles visible in the source hub but organize the textbook by concepts.
- Preserve strong original English explanations and slide labels; add short editorial bridges only where needed for continuity.
- Treat lecture slides as explanations and their linked papers/projects as primary support for technical claims.
- Keep multi-slide arguments intact. Do not replace them with generated summaries or diagrams.
- Separate task, environment, observation, action space, tool executor, policy, memory, verifier, grader, and safety policy.
- Do not claim that a student mirror is an official Berkeley lab. Label Bookvar exercises as adaptations.
- Before every batch commit, read each rendered destination end to end and verify the six-part sequence `problem → full source argument → mechanism/example → evidence boundary → primary source → transition/exercise`. Inspect every changed visual at desktop and narrow widths and reject summary-like, promotional, or machine-written prose.
- Focused commands in individual tasks supplement the mandatory full gate: `pnpm --dir publishing test`, `pnpm --dir publishing build`, `pnpm --dir publishing check:links`, `pnpm --dir site check`, `pnpm --dir site build`, and `pnpm --dir publishing test:output`.
- Every integrated unit uses a destination anchor and reciprocal `source_unit_id`; pointing many lecture sections at one generic page is not coverage.
- Before each commit, stage exact files individually and inspect `git diff --cached --name-only`. Never stage an entire module directory in this dirty worktree.
- Keep `en/` untouched. Do not push, deploy, or merge.
- Stage only named files; preserve unrelated user changes.

---

### Task 1: Establish the advanced-agent transition and reasoning route

**Files:**
- Modify: `00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md`
- Modify: `00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию.md`
- Modify: `00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md`
- Create: `00 Учебник/17 Tools и Agents/69 Reasoning, search и planning.md`
- Create: `00 Учебник/Assets/Figures/curated/berkeley-agents-2025/reasoning-planning/`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Jan 27, Feb 3, and the reasoning/planning portions of Feb 10.
- Builds from model-level test-time compute to agent-level action/search under an environment state.

- [ ] **Step 1: Reconcile inference-time reasoning with the existing reasoning chapter**

Explain sampling/search, critique, self-correction, external feedback, verifier-guided search, and the evidence that unaided self-correction can fail. Keep optimization and inference interventions distinct.

- [ ] **Step 2: Integrate learning-to-reason methods**

Connect preference optimization, iterative reasoning improvement, chain-of-verification, open training recipes, and their data-generation loops. State what signal changes the policy and where evaluation leakage can enter.

Include the OPRO/prompt-optimization loop from `Large Language Models as
Optimizers`: candidate instruction generation, evaluation examples, score
feedback, optimization trajectory, and the distinction between optimizing a
prompt and updating model parameters.

- [ ] **Step 3: Write the advanced planning chapter**

Develop LLM-first versus agent-first views, internal reasoning as an action, formal versus open-world planning, tree search, irreversible actions, replanning, model-based planning, and the boundary between a textual plan and an executable policy.

Route `Tree Search for Language Model Agents` from the Mar 10 reading list to
this chapter and distinguish its environment search from ordinary token-level
beam search.

- [ ] **Step 4: Preserve the course's stepwise reasoning visuals**

Use the full sequences that compare direct generation, search, feedback, and planning. Captions must state the state transition or evaluation signal shown in each frame.

- [ ] **Step 5: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate Berkeley agent reasoning and planning"
```

---

### Task 2: Separate memory, context, retrieval, and world models

**Files:**
- Modify: `00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md`
- Modify: `00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md`
- Create: `00 Учебник/17 Tools и Agents/70 Память и world models агентов.md`
- Modify: `00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер.md`
- Create: `00 Учебник/Assets/Figures/curated/berkeley-agents-2025/memory-world-models/`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers the memory and world-model portions of Feb 10.
- Memory writes and retrievals are observable harness operations; parametric memory is not represented as a database.

- [ ] **Step 1: Establish the memory taxonomy**

Distinguish parameters, current context, scratch/work state, episodic records, semantic knowledge, procedural policy, and external artifacts. For each, state its write path, retrieval key, lifetime, authority, and failure mode.

- [ ] **Step 2: Integrate associative and structured retrieval**

Explain HippoRAG's graph/associative motivation, structured memory, temporal links, knowledge conflict, and continual update. Connect to RAG without implying that all long-term memory is retrieval over chunks.

- [ ] **Step 3: Explain world models as transition models**

Show how an agent can predict action consequences, score candidate plans, and reduce unsafe exploration. Separate a learned transition model from a natural-language chain of thought.

- [ ] **Step 4: Preserve Yu Su's visual argument**

Retain the sequences for agent capability decomposition, memory types, HippoRAG, implicit reasoning, and model-based web planning rather than selecting one decorative overview slide.

- [ ] **Step 5: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate Berkeley agent memory and world models"
```

---

### Task 3: Integrate open post-training recipes for agentic reasoning

**Files:**
- Modify: `00 Учебник/12 Post-training и Alignment/01 SFT и instruction data.md`
- Modify: `00 Учебник/12 Post-training и Alignment/02 Preference data.md`
- Modify: `00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM.md`
- Modify: `00 Учебник/12 Post-training и Alignment/05 DPO.md`
- Modify: `00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers.md`
- Modify: `00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md`
- Modify: `00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер.md`
- Modify: `00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md`
- Create: `00 Учебник/Assets/Figures/curated/berkeley-agents-2025/open-recipes/`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`

**Interfaces:**
- Covers Feb 24 and its Tulu 3, DPO/PPO-practices, and OpenScholar reading cluster.
- Extends the Stanford post-training route with open-recipe design, ablation, and reproducibility rather than duplicating objectives.

- [ ] **Step 1: Map the open recipe end to end**

Trace data curation, SFT, preference data, DPO/PPO variants, evaluation suites, checkpoint selection, and reproducibility artifacts. Mark which results are recipe-specific.

- [ ] **Step 2: Integrate DPO/PPO practice evidence**

Explain the experimental controls needed to compare algorithms fairly: data, initialization, sampling, regularization, reward normalization, compute, and evaluation protocol.

- [ ] **Step 3: Connect OpenScholar to agentic research workflows**

Use it as a case study in retrieval, synthesis, citation, and evaluation—not as a generic RAG claim. Link the relevant primary paper and source artifact.

Place the retrieval/synthesis/citation mechanism in the RAG chapter and the
iterative research workflow, tool state, and stopping/evaluation contract in
the harness chapter. The post-training pages may reference the recipe but must
not absorb OpenScholar as an unrelated paragraph.

- [ ] **Step 4: Import and caption recipe/evaluation visuals**

Preserve pipeline and ablation figures that change how the recipe should be interpreted. Exclude leaderboard decoration only with a ledger reason.

- [ ] **Step 5: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate Berkeley open reasoning recipes"
```

---

### Task 4: Build a systems view of coding, web, and GUI agents

**Files:**
- Create: `00 Учебник/17 Tools и Agents/71 Coding, web и GUI agents.md`
- Modify: `00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию.md`
- Modify: `00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md`
- Modify: `00 Учебник/17 Tools и Agents/68 Оценивание агентных систем.md`
- Modify: `00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding.md`
- Modify: `00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM.md`
- Create: `00 Учебник/Assets/Figures/curated/berkeley-agents-2025/coding-web-gui/`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Mar 3, Mar 10, and Mar 17.
- Uses a shared system schema: task, environment state, observation, action space, executor, feedback channel, stopping rule, and grader.

- [ ] **Step 1: Explain interactive coding agents**

Develop repository navigation, hypothesis formation, code modification, compiler/test/static-analysis feedback, vulnerability discovery, and the difference between completion and an interactive investigation. Use Project Naptime/Big Sleep as attributed case studies.

- [ ] **Step 2: Explain web-agent environments**

Compare Mind2Web, WebArena, and VisualWebArena by observation modality, action representation, state reset, task realism, leakage, and grader. Do not collapse benchmark score into general web competence.

- [ ] **Step 3: Explain GUI agents from perception to action**

Cover screenshots, accessibility trees, grounding, coordinate actions, pure-vision policies, OSWorld, AGUVIS, temporal state, irreversible actions, and recovery.

- [ ] **Step 4: Preserve environment and trajectory visuals**

Use original benchmark/interface figures and complete example trajectories. Captions must identify what the model sees, what it may do, and how success is judged.

- [ ] **Step 5: Strengthen agent evaluation**

Add outcome versus trajectory metrics, environment determinism, hidden state, reset reliability, cost, wall time, attempts, and failure taxonomy.

- [ ] **Step 6: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate Berkeley coding web and GUI agents"
```

---

### Task 5: Build the formal reasoning and theorem-proving route

**Files:**
- Create: `00 Учебник/17 Tools и Agents/72 Формальное рассуждение и theorem proving.md`
- Create: `00 Учебник/17 Tools и Agents/72a Search, RL и theorem-proving agents.md`
- Modify: `00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers.md`
- Modify: `00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md`
- Create: `06 Практика/25 Построить verifiable coding agent.md`
- Create: `06 Практика/Contracts/bookvar-verifiable-coding-agent.yml`
- Create: `00 Учебник/Assets/Figures/curated/berkeley-agents-2025/formal-reasoning/`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Chapter 72 covers Lean foundations, LeanDojo, premise retrieval, and autoformalization from Apr 7. Chapter 72a covers AlphaProof/self-play on Mar 31 and advanced prover agents on Apr 14.
- Practice interface separates visible task/signature/specification skeleton from hidden tests and verifier outcomes.

- [ ] **Step 1: Introduce formal systems before agent algorithms**

Explain formal statements, proof terms/tactics, kernel checking, theorem libraries, premises, search state, and the boundary between a plausible natural-language proof and a machine-checked proof.

- [ ] **Step 2: Integrate mathematical data and RL**

Cover data curation, continual pretraining/finetuning, verified rewards, AlphaProof-style search and learning, and the difference between game self-play and theorem-proving feedback.

Give explicit dispositions to the AlphaZero/self-play reading and the two
broader Mar 31 readings on mathematical practice and library construction; do
not force background readings into the main chapter when they belong in the
source hub.

- [ ] **Step 3: Explain LeanDojo and premise retrieval**

Trace theorem state, retrieved premises, tactic proposal, execution, diagnostics, backtracking, and proof completion. Then add autoformalization and its specification risks.

Use `Autoformalizing Euclidean Geometry` to explain representation choices,
diagram assumptions, theorem-library alignment, and why syntactic acceptance
does not establish faithfulness to the informal problem.

- [ ] **Step 4: Integrate advanced theorem-proving agents**

Cover draft-sketch-prove, long-context theorem proving, interleaving thinking and proving, and proof optimization. Compare where each method places the search, retrieval, and verifier loop.

- [ ] **Step 5: Preserve the full formal-reasoning visual sequence**

Use Kaiyu Yang's Lean/theorem-proving pipeline and supporting AlphaProof/advanced proving visuals as multi-frame explanations, not isolated title slides.

- [ ] **Step 6: Write the verifiable coding-agent practice**

Specify a Bookvar-original visible task/signature and hidden tests, Lean
implementation/specification skeleton, and bounded
generate-compile-diagnose-repair-verify loop. Reject `sorry`, `admit`, new
`axiom` declarations, unsound options/imports, and trivial-explosion routes.
Run Lean in a resource-limited sandbox with per-attempt timeout.

The semantic-specification audit uses a reference implementation,
positive/negative tests, mutations, and generated counterexamples. State
explicitly that kernel acceptance proves the supplied proposition, not the
informal task intent.

Define disjoint parse, compile, functional-test, specification, kernel-proof,
full-success, timeout, and harness/infra outcomes; infra errors are not model
failures. Compare direct generation, compiler-feedback repair,
verifier-feedback repair, and retrieval/example-assisted repair under fixed
paired-task budgets. Freeze task set, model/provider/revision, prompt, decoding
parameters/seeds, token/tool/attempt budgets, and Lean/mathlib/container
versions. Report raw n/N with confidence intervals and cost/latency
distributions. The evidence bundle contains full trajectories, commands,
verifier logs, per-task JSON, environment lock/image digest, and aggregation
script. Hidden tests and specification-audit artifacts never enter prompts or
repair traces. Label the exercise `Bookvar adaptation inspired by Berkeley
Advanced LLM Agents Sp25`, never `Berkeley Lab 1/2`.

- [ ] **Step 7: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate Berkeley formal reasoning and verification"
```

---

### Task 6: Explain abstraction and scientific discovery

**Files:**
- Create: `00 Учебник/17 Tools и Agents/73 Абстракция и научные открытия.md`
- Create: `00 Учебник/Assets/Figures/curated/berkeley-agents-2025/discovery/`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Apr 21 and the symbolic-regression/formal-theorem-proving reading cluster.
- Separates hypothesis search, neural guidance, abstraction induction, concept-library update, and evaluation.

- [ ] **Step 1: Develop the search-and-abstraction mechanism**

Explain candidate generation, symbolic execution/evaluation, neural scoring, discovery of reusable abstractions, and how learned concepts change the future search space.

Route `An In-Context Learning Agent for Formal Theorem-Proving` from the Apr
21 reading list back to chapter 72a with its own source-unit anchor. Apr 21 is
not fully covered by the LaSR/concept-library material alone.

- [ ] **Step 2: Integrate LaSR and concept libraries**

Use the original slide sequence to show hypothesis evolution and concept evolution as different state updates. Explain why a shorter expression is not automatically a scientific explanation.

- [ ] **Step 3: Establish evidence and failure modes**

Cover identifiability, extrapolation, spurious fits, rediscovery versus novelty, benchmark leakage, and the need for independent validation.

- [ ] **Step 4: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate Berkeley abstraction and discovery"
```

---

### Task 7: Make agent security a first-class system chapter

**Files:**
- Create: `00 Учебник/17 Tools и Agents/74 Безопасность и privilege control агентов.md`
- Modify: `00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию.md`
- Modify: `00 Учебник/17 Tools и Agents/66 Agent harness и context engineering.md`
- Modify: `00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md`
- Modify: `00 Учебник/17 Tools и Agents/68 Оценивание агентных систем.md`
- Create: `00 Учебник/Assets/Figures/curated/berkeley-agents-2025/security/`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Apr 28 and Privtrans, DataSentinel, AgentPoison, and Progent.
- Security policy mediates every external action independently of model text.

- [ ] **Step 1: Define the agent threat model**

Map principals, trusted and untrusted inputs, secrets, memory, tools, external services, side effects, and policy boundaries. Explain direct/indirect prompt injection, tool/supply-chain compromise, and memory/knowledge-base poisoning.

- [ ] **Step 2: Separate model safety from system safety**

Show why refusal benchmarks do not establish end-to-end safety when the harness can retrieve hostile content or execute privileged tools.

- [ ] **Step 3: Integrate defenses by enforcement point**

Cover input isolation/detection, provenance, taint tracking, tool allowlists, argument validation, least privilege, privilege separation, programmable privilege control, sandboxing, approvals, audit logs, and recovery.

Develop Privtrans as program partitioning for privilege separation and
DataSentinel as game-theoretic prompt-injection detection before comparing them
with AgentPoison and Progent. Keep their threat assumptions and enforcement
points explicit.

- [ ] **Step 4: Preserve Dawn Song's threat and defense sequences**

Use the compound-agent, attack-surface, AgentPoison, and privilege-control sequences. Captions must identify attacker capability, violated boundary, defense location, and residual risk.

- [ ] **Step 5: Add end-to-end security evaluation**

Require adversarial tasks, benign utility, attack success, policy violations, false positives, memory persistence, multi-step trajectories, and tool-side-effect audit.

- [ ] **Step 6: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate Berkeley agent security"
```

---

### Task 8: Add the advanced agent capstone and close Berkeley coverage

**Files:**
- Create: `06 Практика/26 Спроектировать и оценить агентную систему.md`
- Modify: `06 Практика/_index.md`
- Modify: `00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля.md`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index.md`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Modify: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Capstone lifecycle: proposal, milestone, evaluation plan, reproducible artifact or demonstration, evidence bundle, and final report.
- Supports application and research variants without copying Berkeley grading policy as Bookvar requirements.

- [ ] **Step 1: Write the Bookvar AgentX-style capstone contract**

Require a task/environment/grader definition, baselines, safety model,
evaluation set, trajectory logging, cost/latency measurement, failure taxonomy,
ablations, reproducibility instructions, and evidence bundle. Attribute the
proposal/milestone/public-artifact lifecycle to AgentX separately; label
Bookvar-added metrics and interfaces as editorial adaptations.

- [ ] **Step 2: Connect prerequisites and advanced chapters**

Make the module map linear: 65 tool use → 66 harness/context → 67 memory/orchestration → 68 evaluation → 69–74 advanced route → Bookvar verifiable-coding practice → Bookvar AgentX-style capstone.

- [ ] **Step 3: Close all Berkeley ledger rows**

Every lecture section, reading cluster, and meaningful visual must have a final disposition. Any source-only or excluded row must state why it does not belong in the main narrative.

- [ ] **Step 4: Read the complete agent route after rendering**

Look specifically for repeated definitions, unexplained jumps from model reasoning to environment action, lecture-summary prose, and visuals that do not advance the argument.

- [ ] **Step 5: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: complete Berkeley advanced agents route"
```
