import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const lean = process.argv[2] ?? 'lean';
const chapter = new URL('../../00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md', import.meta.url);
const source = readFileSync(chapter, 'utf8');
const blocks = [...source.matchAll(/^\x60\x60\x60lean\n([\s\S]*?)\n\x60\x60\x60$/gm)].map(match => match[1]);
assert.ok(blocks.length > 0, 'no Lean examples found');

function run(args, input) {
  const result = spawnSync(lean, args, { input, encoding: 'utf8', timeout: 30_000 });
  if (result.error) throw result.error;
  return { status: result.status, output: result.stdout + result.stderr };
}

const version = run(['--version']);
assert.equal(version.status, 0, version.output);
const audit = blocks.find(block => block.includes('namespace Given'));
assert.ok(audit, 'missing proof-acceptance example');
for (const [i, block] of blocks.entries()) {
  const { status, output } = run(['--stdin'], block + '\n');
  assert.equal(status, 0, `Lean example ${i + 1}:\n${output}`);
  if (block !== audit) {
    assert.doesNotMatch(output, /declaration uses 'sorry'/);
    continue;
  }
  for (const line of [
    "warning: declaration uses 'sorry'",
    "'valid' does not depend on any axioms",
    "'unfinished' depends on axioms: [sorryAx]",
    "'forged' depends on axioms: [invented]",
    "'Candidate.easy' does not depend on any axioms",
    'Candidate.easy : Candidate.Goal',
  ]) assert.ok(output.includes(line), output);
}

const wrong = run(['--stdin'], audit + '\nexample : Given.Goal := Candidate.easy\n');
assert.equal(wrong.status, 1, wrong.output);
assert.match(wrong.output, /error: type mismatch/);
assert.match(wrong.output, /Candidate\.Goal : Prop[\s\S]*Given\.Goal : Prop/);
console.log(`${version.output.trim()}: ${blocks.length} compiled examples; substituted statement rejected.`);
