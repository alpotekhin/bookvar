---
title: Courses and reading paths
type: source-note
status: active
locale: en
translation_of: "05 Источники/Курсы.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
---

# Courses and reading paths

Choose a topic below to find both a Bookvar chapter and its original course
material. The textbook follows a thematic sequence; the source collection
preserves each course's own lectures, figures, and assignments in their
original language. English source archives are available even where a Bookvar
chapter still has only Russian text. Such chapters display a fallback notice.

## Read by topic

| Topic | Start in Bookvar | Original course material |
|---|---|---|
| Classical ML | [[00 Учебник/01 Классическое машинное обучение/00 Карта модуля|Classical ML module]] | [[05 Источники/Courses/HSE ML course|HSE ML course, in Russian]] |
| Tokenization and Transformer | [[00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|Tokenization]], [[00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention]] | [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-01: Overview, tokenization|CS336 lecture 1]], [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-03: Architectures, hyperparameters|lecture 3]], Assignment 1 |
| GPUs, memory, profiling, and kernels | [[00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU and memory]], [[00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|Triton]] | [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-05: GPUs, TPUs|CS336 lectures 5–6]], Assignment 2; [[05 Источники/Courses/Efficient DL Systems|Efficient DL Systems]] |
| Distributed training | [[00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision|Training parallelism]] | [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-07: Parallelism|CS336 lectures 7–8]], [[05 Источники/Courses/Harvard ML Systems|Harvard Volume II]], [[05 Источники/Courses/Efficient DL Systems|Efficient DL Systems]] |
| Data and scaling laws | [[00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Training data]], [[00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Scaling laws]] | [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-09: Scaling laws|CS336 lectures 9 and 11]], [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-13: Data (sources, datasets)|13–14]], Assignments 3–4 |
| SFT, preferences, and RLVR | [[00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники|Post-training module]] | [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-15: Mid/post-training (SFT/RLHF)|CS336 lectures 15–16]], Assignment 5; [[05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index#4. 24 февраля — open post-training recipes|Berkeley lecture 4]] |
| Inference and serving | [[00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|Prefill and decode]], [[00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Scheduling]] | [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-10: Inference|CS336 lecture 10]], [[05 Источники/Courses/Efficient DL Systems|Efficient DL Systems]], [[05 Источники/Courses/Harvard ML Systems|Harvard ML Systems]] |
| Multimodal models | [[00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Vision–language models]] | [[05 Источники/Courses/Stanford CS336 Spring 2026/_index#lecture-17: Alignment - multimodality|CS336 lecture 17]] |
| Reasoning, memory, and planning | [[00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Test-time compute]], [[00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Memory and planning]] | [[05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index#1. 27 января — inference-time reasoning|Berkeley lectures 1–3]] |
| Coding, web, and GUI agents | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Coding and computer-use agents]] | [[05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index#5. 3 марта — coding agents and vulnerability detection|Berkeley lectures 5–7]], [[05 Источники/Courses/GenAI Agents|GenAI Agents notebooks]] |
| Proofs, discovery, and safety | [[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Formal mathematics]], [[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Scientific discovery]], [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Agent safety]] | [[05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index#8. 31 марта — AlphaProof|Berkeley lectures 8–12]] |

## What is available

### Stanford CS336 — Spring 2026

[[05 Источники/Courses/Stanford CS336 Spring 2026/_index|Open the archived course]] ·
[Official course](https://cs336.stanford.edu/)

The archive contains material for 17 lectures and five assignments, with
the optional safety/RLHF supplement inside Assignment 5. The schedule also records two guest
sessions without published official artifacts. The material is already linked
to textbook sections, not merely stored for future use.

The coverage ledger contains 807 semantic units: 557 integrated, 24 already
covered, 21 assignment-contract-only, 151 source-only, and 54 excluded.
Of the 557 integrated units, 373 come from assignment material and 184 from
lectures. The separate visual ledger has 159 records: 73 integrated,
9 already covered, 45 source-only, and 32 excluded. These are bookkeeping
categories, not a count of fully reviewed chapters or individual figures.

Multi-token prediction is now integrated into the
[[00 Учебник/11 Pre-training и Scaling/42 Next-token prediction#multi-token-training-objective|training-objective chapter]],
while cross-layer KV sharing and causal sliding windows are explained in the
[[00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание#cross-layer-kv-sharing|long-context chapter]].
FP4 is explained in [[00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|quantization]],
ARC-AGI-3 in [[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|evaluation]],
and AIR-Bench/GCG in [[00 Учебник/18 Evaluation и методология/59a Responsible systems|responsible systems]].
Text coverage does not imply that every source slide was reproduced; the ledger retains the exact visual status.
Exact submission requirements,
repeated explanations, and course administration can remain in the archive.

Bookvar practice: [[06 Практика/20 Собрать языковую модель с нуля|build a language model]],
[[06 Практика/21 Профилировать и ускорить Transformer kernel|profile a kernel]],
[[06 Практика/22 Провести scaling-law campaign|fit scaling laws]],
[[06 Практика/23 Собрать воспроизводимый pretraining corpus|build a corpus]], and
[[06 Практика/24 Post-training и RLVR для математического reasoning|post-training and RLVR]].

### Berkeley Advanced LLM Agents — Spring 2025

[[05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index|Open lectures and readings]] ·
[Official course](https://rdi.berkeley.edu/adv-llm-agents/sp25)

The collection covers 12 meetings, 13 slide PDFs, recording metadata, and
37 reading records. The 140 integrated units include 105 slide-derived units
and 35 reading catalogue entries; the latter are links and descriptions, not
mirrored papers. Another 6 units are already covered, 38 remain source-only,
and 28 are excluded. Of 161 visual records, 47 are integrated, 87 source-only,
and 27 excluded. A visual record may describe several slides.

xGen and GenS are explained in [[00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|video models]],
and ESCHER in [[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|scientific discovery]].
Remaining source-only material includes recording metadata, course
administration, and Plan–Sequence–Learn: robotics needs motion-planning and
low-level-control foundations beyond the current web/GUI agent chapters. A page
explicitly marked confidential in lecture 6 is excluded from reuse.

Bookvar practice: [[06 Практика/25 Воспроизводимо оценить и red-team компьютерного агента|evaluate a computer-use agent]],
[[06 Практика/26 Поиск доказательства в Lean с verifier|search for a Lean proof]], and
[[06 Практика/27 Проверяемый научный поиск на символьной регрессии|symbolic regression]].
These are Bookvar adaptations, not official Berkeley labs: the archived public
syllabus does not expose a verified official lab handout or starter repository.

### Efficient DL Systems and Harvard ML Systems

[[05 Источники/Courses/Efficient DL Systems|Efficient DL Systems]] preserves
the original nine-week course: ten slide decks, ten notebooks, and twenty
README/homework documents. Its theory and experiments complement the textbook's
systems, distributed-training, and inference sections.

[[05 Источники/Courses/Harvard ML Systems|Harvard ML Systems]] contains 29 full
English chapters, alongside [[05 Источники/Courses/Harvard ML Systems/Labs and slides|34 labs and 35 slide decks]]
and [[05 Источники/Courses/Harvard ML Systems/tinytorch/README|20 TinyTorch modules]].
For the labs, start with [[06 Практика/18 Harvard ML Systems — интерактивные design labs|the Bookvar practice guide]].

Both courses have source-to-chapter maps. Their automated checks establish
files, destinations, and source links; they do not compare every source section
with the textbook's explanation. Archive completeness and full thematic
integration are separate claims.

### HSE and legacy SHAD notes

The [[05 Источники/Courses/HSE ML course|HSE archive]] contains 67 Russian
materials from two Spring 2026 tracks. Its [[05 Источники/Source maps/HSE ML course — link map|topic map]]
is useful for classical ML, but full chapter-by-chapter transfer has not been
established. Older notes for a separate SHAD LLM course are legacy material;
they do not yet have a restored complete archive or a verified destination map.

### Other source collections

- [[05 Источники/Courses/Stanford CS230 Cheatsheets|Stanford CS230 visual references]]
- [[05 Источники/Courses/Machine Learning Visualized|Machine Learning Visualized]]
- [[05 Источники/Courses/GenAI Agents|GenAI Agents notebooks]]
- [Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Lena Voita's NLP course](https://lena-voita.github.io/nlp_course.html)
- [Jurafsky and Martin: Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) and [CME295](https://cme295.stanford.edu/syllabus/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1) and [Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)

These links are reading options, not claims that each course has been imported
in full. The Russian catalogue contains additional source suggestions.

## How new course material enters the textbook

Keep the original with its author, URL, version, and reuse terms. Compare its
sections with existing chapters. Add a new mechanism, example, derivation, or
figure to the chapter that explains that topic; link to existing coverage when
it already suffices. Create a chapter only when the subject needs a separate
sequence of prerequisites and explanation, not simply because a new lecture
has appeared.

Text and figures are tracked separately. After integration, read the whole
chapter and inspect the figures on the rendered page. A passing link check
does not establish explanatory quality, and a source-only item is not
automatically a missing chapter: exact assignment contracts and duplicate or
administrative material can properly remain in the original.
