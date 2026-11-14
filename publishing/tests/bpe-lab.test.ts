import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { bpeDefaults, decodeBpeLab, encodeBpeLab, showBpeBytes, trainBpeLab } from '../../site/src/lib/bpe-lab';
import { loadManifest } from '../adapter/manifest.js';

describe('BPE lab', () => {
  it('trains the three worked merges and encodes a separate input at every step', () => {
    const { training, probe, limit } = bpeDefaults;
    const trace = trainBpeLab(training, limit);
    expect(trace.frames.map(frame => frame.ids.length)).toEqual([11, 9, 7, 5]);
    expect(trace.merges).toEqual([
      { left: 97, right: 97, count: 4, id: 256, replacements: 2 },
      { left: 256, right: 97, count: 2, id: 257, replacements: 2 },
      { left: 257, right: 98, count: 2, id: 258, replacements: 2 },
    ]);
    const expected = [
      [97, 97, 97, 98, 97, 99], [256, 97, 98, 97, 99], [257, 98, 97, 99], [258, 97, 99],
    ];
    for (const [step, ids] of expected.entries()) {
      const actual = encodeBpeLab(probe, trace.merges.slice(0, step));
      expect(actual).toEqual(ids);
      expect(decodeBpeLab(actual, trace.vocab)).toBe(probe);
    }
  });

  it('does not count replacements twice, break ties by ID, or merge an empty frame', () => {
    const aaa = trainBpeLab('aaa', 12);
    expect(aaa.frames[1].ids).toEqual([256, 97]);
    expect(aaa.merges[0]).toMatchObject({ count: 2, replacements: 1 });
    expect(aaa.merges).toHaveLength(2);
    expect(trainBpeLab('zab', 1).merges[0]).toMatchObject({ left: 122, right: 97, count: 1 });
    expect(trainBpeLab('', 12).merges).toEqual([]);
    expect(trainBpeLab('zab', 0).frames).toHaveLength(1);
    expect(() => trainBpeLab('a', 13)).toThrow(RangeError);
    expect(() => decodeBpeLab([999], aaa.vocab)).toThrow(RangeError);
  });

  it('reassembles UTF-8 before decoding without removing a leading BOM or normalizing', () => {
    const trace = trainBpeLab(bpeDefaults.training, 3);
    const utf8 = encodeBpeLab('я🦊', trace.merges);
    expect(utf8).toEqual([209, 143, 240, 159, 166, 138]);
    expect(showBpeBytes(trace.vocab[209])).toBe('⟦D1⟧');
    expect(showBpeBytes(trace.vocab[32])).toBe('␠');
    for (const text of ['я🦊', '', '\uFEFFa', 'é', 'e\u0301', 'Ａ', 'a\n  b', '<|endoftext|>']) {
      for (let step = 0; step <= trace.merges.length; step++) {
        expect(decodeBpeLab(encodeBpeLab(text, trace.merges.slice(0, step)), trace.vocab)).toBe(text);
      }
    }
  });

  it('applies the paper’s given merge order rather than guessing a new order from the probe', () => {
    const vocab = Array.from({ length: 256 }, (_, id) => [id]);
    const given = [[97, 98], [99, 98], [256, 97], [258, 257]].map(([left, right], rank) => {
      vocab.push([...vocab[left], ...vocab[right]]);
      return { left, right, id: 256 + rank, count: 0, replacements: 0 };
    });
    expect(encodeBpeLab('abaabacbcb', given)).toEqual([258, 259, 257]);
    expect([258, 259, 257].map(id => showBpeBytes(vocab[id]))).toEqual(['aba', 'abacb', 'cb']);
    expect(decodeBpeLab([258, 259, 257], vocab)).toBe('abaabacbcb');
  });

  it('registers and connects both editions to the canonical chapter and practice index', () => {
    const root = resolve(import.meta.dirname, '../..');
    const read = (p: string) => readFileSync(resolve(root, p), 'utf8');
    const source = '06 Практика/01a Обучить BPE на двух строках.md';
    const sourceEn = 'en/06 Practice/01a Train BPE on two strings.md';
    const chapter = '00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram';
    const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
    expect(manifest.sections.find(s => s.id === 'practice')?.pages.find(p => p.route === 'practice/byte-pair-encoding'))
      .toMatchObject({ source, sourceEn });
    expect(read(sourceEn)).toContain(createHash('sha256').update(read(source)).digest('hex'));
    for (const file of [source, sourceEn]) expect(read(file)).toContain(chapter + '#bpe-merge-trace');
    for (const file of [chapter + '.md', '06 Практика/_index.md']) expect(read(file)).toContain(source.slice(0, -3));
  });
});
