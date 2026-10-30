import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const chapter = read('00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md');
const addition = chapter.split('<!-- BEGIN REVIEW: Berkeley finite products 2026-10-26 -->')[1]
  .split('<!-- END REVIEW: Berkeley finite products 2026-10-26 -->')[0];
const normalize = (text: string) => text.replace(/\s+/g, ' ').trim();

describe('Berkeley finite products and the original Mathlib explanation', () => {
  it('retains the entire original module introduction in English', () => {
    const source = read('05 Источники/Originals/formal-zeta-2026-10-26/EulerProduct-Basic.lean');
    expect(createHash('sha256').update(source).digest('hex')).toBe(
      '18ae67ac1cf4af8b95293b338da3339554c3f6ea470564516982b2eba4af81b3');
    const intro = source.split('# Euler Products\n\n')[1].split('\n## Tags')[0];
    const quote = addition.split('<!-- BEGIN SOURCE: Mathlib Euler Products introduction -->')[1]
      .split('<!-- END SOURCE: Mathlib Euler Products introduction -->')[0].replace(/^> ?/gm, '');
    expect(normalize(quote)).toBe(normalize(intro));
    expect(read('site/public/licenses/mathlib-c44e0c8-Apache-2.0.txt')).toBe(
      read('05 Источники/Originals/formal-zeta-2026-10-26/mathlib-LICENSE.txt'));
  });

  it('keeps the finite examples and analytical assumptions distinct from an executed prover trace', () => {
    expect(addition.match(/^```python$/gm)).toHaveLength(2);
    expect(addition).not.toMatch(/^```(?:lean|coq)$/m);
    for (const value of ['remainders == [17, 17, 100]', 'sum(answers) == 239',
      'F(91, 36)', 'nonprime_denominators', 'f(1)=1', '\\operatorname{Re}s>1',
      'non-integer $x>1$', 'symmetric limit', 'not establish those theorems',
      'infinitely many\nprimes']) expect(addition).toContain(value);
    expect(read('publishing/navigation.yml')).toContain('route: sources/papers/mathlib-euler-products');
  });
});
