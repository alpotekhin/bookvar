import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import YAML from 'yaml';
import { beforeTopicConsolidation } from './topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = path => readFileSync(resolve(root, path), 'utf8');
const sha = path => createHash('sha256').update(readFileSync(resolve(root, path))).digest('hex');
const manifest = YAML.parse(read('publishing/navigation.yml'));
const pages = manifest.sections.flatMap(section => section.pages);
const route = value => {
  const page = pages.find(page => page.route === value);
  if (!page) throw new Error('Unknown canonical destination: ' + value);
  // The residual decisions refer to their reviewed chapter revision, before
  // the subsequent PEFT insertion. Only its exact journaled delta is reversed.
  const reviewed = beforeTopicConsolidation(read(page.source), page.source, root,
    '2026-10-29-scheduler-lab-additions.json');
  return { source: page.source, route: page.route,
    sha256: createHash('sha256').update(reviewed).digest('hex') };
};
const byFile = part => {
  const found = pages.filter(page => page.source.startsWith('00 Учебник/') && page.source.includes(part));
  if (found.length !== 1) throw new Error('Ambiguous chapter: ' + part);
  return route(found[0].route);
};
const owners = {
  ttc: route('textbook/reasoning/test-time-compute'),
  scaling: route('textbook/pretraining/scaling-laws'),
  arithmetic: route('textbook/ml-systems/transformer-moe-arithmetic'),
  gpu: route('textbook/ml-systems/gpu-cuda-memory-hierarchy'),
  kernels: route('textbook/ml-systems/gpu-kernels-triton'),
  distributed: route('textbook/pretraining/distributed-systems'),
  physics: route('textbook/inference/llm-inference-physics'),
  schedule: route('textbook/inference/scheduling'),
  speculative: route('textbook/inference/speculative-decoding'),
  quantization: route('textbook/inference/quantization'),
  compression: route('textbook/inference/model-compression-pruning-distillation-low-rank'),
  diffusion: route('textbook/pretraining/masked-diffusion-language-models'),
  attention: route('textbook/transformer/self-attention'),
  compressed: byFile('/03 Длинный контекст'),
  stateful: byFile('Mamba, RWKV, RetNet'),
  data: byFile('/41 Сбор, очистка'),
  quality: route('textbook/pretraining/dedup-pii-corpus-quality'),
  evaluation: byFile('/59 Оценивание моделей'),
  judge: route('textbook/post-training/reward-modeling'),
  responsible: route('textbook/evaluation/responsible-systems'),
  security: route('textbook/agents/security'),
  bert: route('textbook/models/bert-roberta-deberta'),
  gpt2: route('textbook/models/gpt-2'),
  gpt3: route('textbook/models/gpt-3'),
  t5: route('textbook/models/t5'),
  ngram: route('textbook/language-modeling/ngram'),
  seq2seq: route('textbook/recurrent-networks/seq2seq-bottleneck'),
  backprop: route('textbook/neural-networks/backpropagation'),
  optimizer: route('textbook/neural-networks/optimizers-schedules'),
  sft: route('textbook/post-training/sft-instruction-data'),
  preference: route('textbook/post-training/preference-data'),
  grpo: route('textbook/post-training/grpo-deepseek-r1'),
  rlvr: route('textbook/post-training/rlvr-verifiers'),
  harness: route('textbook/agents/harness-context-engineering'),
  memory: route('textbook/agents/memory-planning-orchestration'),
  computer: route('textbook/agents/coding-web-computer-use'),
  formal: route('textbook/agents/formal-math'),
  science: route('textbook/agents/scientific-discovery'),
};
const result = (decision, keys = [], reason = '') => ({
  decision, reason, canonicalOwners: keys.map(key => owners[key])
});
// These are explicit editorial decisions, not a title-matching classifier.
const cs = {
  'lecture-01-section-4-cs336-language-models-from-scratch-spring-2026': result('retain-course-context', [], 'Course identity and staff, not a technical explanation.'),
  'lecture-01-section-15-why-did-we-make-this-course': result('retain-course-context', [], 'Teaching philosophy and course motivation remain with the original syllabus.'),
  'lecture-01-section-26-the-industrialization-of-language-models': result('retain-historical-framing', ['arithmetic', 'scaling'], 'Dated frontier-cost anecdotes and illustrative emergence plots are not universal scaling laws. Resource mechanisms have canonical chapters.'),
  'lecture-01-section-40-what-can-we-learn-in-this-class-that-transfers-to-frontier-models': result('retain-course-context', ['scaling'], 'Distinguishes mechanics from scale-specific intuitions; no new model algorithm is introduced.'),
  'lecture-01-section-47-intuitions': result('retain-course-context', ['optimizer'], 'A caution about empirical architectural choices, not an additional algorithm.'),
  'lecture-01-section-51-the-bitter-lesson': result('covered-overview', ['scaling', 'ttc'], 'Scalable algorithms and finite compute budgets are explained in the budget chapters; keep the original orientation.'),
  'lecture-01-section-54-accuracy-efficiency-x-resources': result('covered-overview', ['arithmetic', 'scaling'], 'Compute/data/model allocation and hardware accounting are already detailed. The slogan is not promoted into an accuracy equation.'),
  'lecture-01-section-62-pre-neural-before-2010s': result('covered-overview', ['ngram'], 'Historical orientation points to the full n-gram explanation.'),
  'lecture-01-section-65-neural-ingredients-2010s': result('covered-overview', ['seq2seq', 'optimizer', 'attention', 'distributed'], 'Named ingredients each have full explanations; do not add another name-only chronology.'),
  'lecture-01-section-74-early-foundation-models-late-2010s': result('covered-overview', ['bert', 't5'], 'Encoder and text-to-text mechanisms are in their canonical architecture chapters.'),
  'lecture-01-section-78-embracing-scaling': result('covered-overview', ['gpt2', 'gpt3', 'scaling'], 'Early GPT and compute-optimal scaling are explained with their source-specific limitations.'),
  'lecture-01-section-84-open-models': result('retain-historical-framing', ['data'], 'A dated catalogue of releases/openness is retained as source context, not treated as current rankings or additional architecture specifications.'),
  'lecture-01-section-221-kernels': result('covered-overview', ['kernels', 'gpu'], 'Fusion, tiling and memory movement are developed beyond this introductory list.'),
  'lecture-01-section-231-parallelism': result('covered-overview', ['distributed'], 'The distributed module covers collectives and DP/TP/PP/SP/EP; this is the syllabus overview.'),
  'lecture-01-section-237-inference': result('covered-overview', ['physics', 'schedule', 'speculative'], 'Prefill/decode, scheduling and exact speculative sampling have detailed canonical explanations.'),
  'lecture-01-section-294-evaluation': result('covered-overview', ['evaluation'], 'Internal development evaluation and held-out, protocol-specific external evaluation are already distinguished.'),
  'lecture-01-section-302-data-curation': result('covered-overview', ['data', 'quality'], 'Sources, rights/provenance and extraction/filtering are explained in the data module.'),
  'lecture-01-section-309-data-processing': result('covered-overview', ['data', 'quality'], 'Transformation, filtering, deduplication and mixtures are developed in the canonical pipeline.'),
  'lecture-10-section-223-taking-shortcuts-lossy': result('covered-overview', ['compression', 'quantization'], 'A category heading for approximation, not a separate method to duplicate.'),
  'lecture-10-section-274-deepseek-v4-attention': result('mixed-range-component-map', ['compressed', 'stateful', 'diffusion', 'quantization'], 'Steps 274–279 cover CSA/HCA; 280–287 summarize cache/architecture choices; 288–310 begin quantization. Preserve the parent status and distinguish the components.'),
  'lecture-13-section-143-lawsuits': result('mixed-range-component-map', ['data', 'quality', 'responsible'], 'The title overextends: steps 143–159 are legal framing; steps 162–419 cover corpus sources through DCLM. Dated legal claims remain source-only; corpus mechanisms have separate owners.'),
  'lecture-13-section-484-summary': result('covered-overview', ['data', 'quality'], 'Recap of the source and filtering discussion, not an additional unimplemented method.'),
  'lecture-15-recap': result('covered-overview', ['sft', 'preference'], 'SFT/RLHF recap remains with the course; full mechanisms precede the recap in the textbook.'),
  'lecture-16-recap': result('covered-overview', ['grpo', 'rlvr'], 'RLVR recap, not an additional training algorithm.'),
};
const berkeley = {
  'official-syllabus-coursework-and-schedule': result('retain-course-context', [], 'Original course schedule, grading and coursework contract.'),
  'meeting-01-intro-course-identity-and-staff': result('retain-course-context', [], 'Title and teaching staff.'),
  'meeting-01-intro-prior-course-and-agent-frame': result('covered-overview', ['harness'], 'Prior MOOC context and the environment/action/feedback frame.'),
  'meeting-01-intro-course-topic-map': result('retain-course-context', [], 'Course topic list; it does not define another textbook reading order.'),
  'meeting-01-intro-official-coursework-contract': result('retain-course-context', [], 'Original grading, lab and project requirements.'),
  'meeting-01-slides-inference-time-reasoning-orientation': result('retain-course-context', ['ttc'], 'Single title slide; inference-time mechanisms are explained in the chapter.'),
  'meeting-02-slides-learning-to-reason-framing': result('covered-overview', ['ttc', 'judge'], 'Self-improvement motivation and System 1/System 2 framing; not a separate architecture claim.'),
  'meeting-02-slides-reasoning-training-foundations': result('covered-overview', ['ngram', 'bert', 'gpt2', 'gpt3', 'scaling'], 'Historical slides revisit language modeling, neural methods, attention and scaling.'),
  'meeting-02-slides-learning-to-reason-synthesis': { ...result('gap-filled-with-component-map', ['judge', 'ttc', 'harness'], 'Page 103 summarizes judging; page 104 introduces Coconut, now explained at coconut-continuous-thought; page 105 lists open research directions.'), newAnchor: 'coconut-continuous-thought' },
  'meeting-03-slides-memory-planning-orientation': result('covered-overview', ['harness', 'memory'], 'Orientation and agent definition; the mechanisms are in the existing memory/planning module.'),
  'meeting-04-slides-open-posttraining-overview': result('retain-historical-framing', ['data', 'grpo', 'ttc'], 'Open ecosystem and dated comparisons are introductory context; pretraining/post-training/test-time mechanisms have their owners.'),
  'meeting-04-slides-open-recipe-synthesis': result('retain-course-context', [], 'Research agenda, not an additional established method.'),
  'meeting-06-slides-web-agent-overview': result('covered-overview', ['computer'], 'Examples and the grounding/HTML/vision frame introduce later mechanisms, already covered in computer-use agents.'),
  'meeting-06-slides-proprietary-adjacent-demo': result('retain-demonstration', [], 'Robot manipulation demonstrations do not supply a reproducible algorithm or training recipe. Keep them in their original source context.'),
  'meeting-07-slides-gui-agent-landscape': result('covered-overview', ['computer'], 'Landscape of computer/web/mobile agents, with demonstrations rather than new mechanism definitions.'),
  'meeting-07-slides-gui-agent-summary': result('covered-overview', ['computer', 'memory'], 'Summary names OSWorld, data synthesis, grounding and memory; detailed explanations remain in the corresponding chapters.'),
  'meeting-08-slides-alphaproof-orientation': result('covered-overview', ['formal'], 'Motivation for mathematics and a title slide; no missing proving algorithm.'),
  'meeting-08-slides-formal-reasoning-future': result('retain-historical-framing', ['formal'], 'Future ambitions are not reported as implemented capabilities.'),
  'meeting-08-reading-03-catalogue-record': result('retain-catalogue', ['formal'], 'Only recording metadata is archived; no claim of a new viewing or source transcription.'),
  'meeting-08-reading-04-catalogue-record': result('retain-catalogue', ['formal'], 'Publisher-article metadata is a reading link, not an unmirrored textbook chapter.'),
  'meeting-09-slides-autoformalization-capability-race': result('covered-overview', ['formal', 'rlvr'], 'Dated capability framing and why math/code have checkable outcomes. Testing limits remain explicit in the verifier chapter.'),
  'meeting-10-slides-advanced-proving-frame': result('retain-course-context', ['formal'], 'Expert-domain examples introduce the lecture; they do not specify further proving methods.'),
  'meeting-11-slides-abstraction-discovery-frame': result('covered-overview', ['science'], 'Hypothesis generation, search and experiment feedback are the opening frame for the existing scientific-discovery explanation.'),
};
const figureOwners = {
  'lecture-01-figure-step-37-rendering-1': 'arithmetic',
  'lecture-01-figure-step-39-rendering-1': 'scaling',
  'lecture-01-figure-step-220-rendering-1': 'gpu',
  'lecture-01-figure-step-241-rendering-1': 'physics',
  'lecture-01-figure-step-276-rendering-1': 'scaling',
  'lecture-01-figure-step-305-rendering-1': 'data',
  'lecture-02-figure-step-125-rendering-1': 'gpu',
  'lecture-02-figure-step-369-rendering-1': 'backprop',
  'lecture-02-figure-step-427-rendering-1': 'backprop',
  'lecture-02-figure-step-926-rendering-1': 'backprop',
  'lecture-06-figure-step-82-rendering-1': 'gpu',
  'lecture-07-figure-step-5-rendering-1': 'distributed',
  'lecture-07-figure-step-118-rendering-1': 'distributed',
  'lecture-10-figure-step-375-rendering-1': 'speculative',
  ...Object.fromEntries([45, 58, 60, 79, 87, 208, 210, 215].map(step => ['lecture-12-figure-step-' + step + '-rendering-1', 'evaluation'])),
  'lecture-12-figure-step-144-rendering-1': 'judge',
  'lecture-12-figure-step-150-rendering-1': 'judge',
  'lecture-12-figure-step-233-rendering-1': 'responsible',
  'lecture-12-figure-step-239-rendering-1': 'security',
  'lecture-13-figure-step-73-rendering-1': 'data',
  'lecture-14-figure-step-289-rendering-1': 'quality',
  'lecture-14-figure-step-291-rendering-1': 'data',
};
const courses = ['Stanford CS336 Spring 2026', 'Berkeley Advanced LLM Agents Spring 2025'];
const records = [];
for (const course of courses) {
  const base = '05 Источники/Courses/' + course + '/';
  const path = base + 'coverage.yml';
  const rows = YAML.parse(read(path)).rows.filter(row => ['source-only', 'contract-only'].includes(row.disposition));
  for (const row of rows) {
    let decision = (course.startsWith('Stanford') ? cs : berkeley)[row.source_unit];
    if (['deliverable', 'test-interface'].includes(row.kind)) {
      const assignment = Number(row.source_unit.match(/^assignment-0([1-5])-/)?.[1]);
      if (!assignment) throw new Error('Unknown assignment: ' + row.source_unit);
      const practice = pages.find(page => page.source.startsWith('06 Практика/' + (19 + assignment) + ' '));
      if (!practice) throw new Error('Missing assignment practice');
      decision = {
        decision: 'retain-assignment-interface',
        reason: 'Original API/deliverable/test contract remains in the course, linked from its practice. This does not claim a completed student solution or GPU experiment.',
        canonicalOwners: [route(practice.route)],
      };
    } else if (row.kind === 'figure') {
      const owner = figureOwners[row.source_unit];
      if (!owner) throw new Error('Unreviewed residual figure: ' + row.source_unit);
      decision = result('retain-source-figure', [owner], 'Retain the original course figure and source context. A topic owner is not a claim that this exact image was newly reused or visually re-reviewed.');
    } else if (row.kind === 'segment') {
      decision = result('retain-recording', [], 'Preserved original recording link; no new full viewing or transcript is claimed. Slide-based topic integration is recorded separately.');
    } else if (row.kind === 'administrative') {
      decision = result('retain-bibliography', [], 'Original assignment bibliography is retained rather than copied into another chapter.');
    }
    if (!decision) throw new Error('Residual needs an explicit editorial decision: ' + row.source_unit);
    const location = row.source_location.split('#')[0];
    const sourceFile = /^https?:/.test(location) ? undefined : base + location;
    if (sourceFile && !existsSync(resolve(root, sourceFile))) throw new Error('Missing retained source: ' + sourceFile);
    records.push({
      course, unit: row.source_unit, title: row.title, kind: row.kind,
      sourceDisposition: row.disposition, sourceLocation: row.source_location,
      ...(sourceFile ? { sourceFile, sourceSha256: sha(sourceFile) } : {}),
      ...decision,
    });
  }
}
const legacy = JSON.parse(read('docs/audit/2026-10-29-training-consolidation.json')).entries
  .filter(row => row.publicationStatus === 'unpublished-legacy').map(row => {
    if (pages.some(page => page.source === row.source)) throw new Error('Archive accidentally published: ' + row.source);
    if (sha(row.source) !== row.sourceSha256) throw new Error('Unreviewed archived-note change: ' + row.source);
    return { source: row.source, sha256: sha(row.source), decision: 'retain-unpublished',
      scope: 'Unchanged and outside the reader manifest. This check is not a full semantic/factual review or an equivalence claim.' };
  });
const counts = key => Object.fromEntries([...new Set(records.map(row => row[key]))].sort()
  .map(value => [value, records.filter(row => row[key] === value).length]));
const report = {
  date: '2026-10-29',
  scope: 'Disposition of the existing residual course units and preservation of unpublished notes. Counts are not counts of completed chapter transfers.',
  readScope: 'Remaining text frames were read; referenced earlier canonical reviews retain their own scope. Assignment interfaces, recordings, catalogue-only records and old figure selections are preserved, not reclassified as new complete semantic or visual reviews.',
  sourceLedgers: courses.map(course => ({ path: '05 Источники/Courses/' + course + '/coverage.yml', sha256: sha('05 Источники/Courses/' + course + '/coverage.yml') })),
  counts: { sourceDisposition: counts('sourceDisposition'), decision: counts('decision'), unpublishedNotes: legacy.length },
  newContentGapClosed: { method: 'Coconut', source: 'Berkeley meeting 2 slide 104', destination: owners.ttc, anchor: 'coconut-continuous-thought' },
  records, unpublishedNotes: legacy,
};
const output = resolve(root, 'docs/audit/2026-10-29-final-closure-course-residuals.json');
const serialized = JSON.stringify(report, null, 2) + '\n';
if (process.argv.includes('--write')) writeFileSync(output, serialized);
else if (readFileSync(output, 'utf8') !== serialized) throw new Error('Residual decision evidence changed; review the affected source or destination before regenerating.');
console.log(JSON.stringify({ mode: process.argv.includes('--write') ? 'write' : 'check', records: records.length, counts: report.counts }));
