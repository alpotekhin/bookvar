import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { readingSlides, scikitMoocReaderBody } from '../adapter/scikit-mooc-reader.js';

const root = new URL('../../', import.meta.url);
const source = '05 Источники/Courses/Scikit-learn MOOC/';
const route = 'sources/courses/scikit-learn-mooc/';
const read = (name: string) => readFileSync(new URL(name, root), 'utf8');
const code = (text: string) => [...text.matchAll(/^ {0,3}\x60{3}[^\n]*\n[\s\S]*?^ {0,3}\x60{3}[ \t]*$/gm)].map(m => m[0]);
const figures = (text: string) => [...text.matchAll(/!\[\[Assets\/Sources\/Scikit-learn MOOC\/figures\/[^\n]+?\]\]/g)].map(m => m[0]);

describe('scikit-learn MOOC reader', () => {
  it('retains every original slide figure occurrence and code block in all twelve decks', () => {
    const decks = readdirSync(new URL(source + 'slides/', root))
      .filter(file => !['README.md.md', 'intro_words.md.md'].includes(file));
    expect(decks).toHaveLength(12);
    for (const deck of decks) {
      const raw = read(source + 'slides/' + deck).split('---').slice(2).join('---');
      const rendered = readingSlides(raw);
      expect(code(rendered), deck).toEqual(code(raw));
      expect(figures(rendered), deck).toEqual(figures(raw));
      expect(rendered).not.toMatch(/^\s*(?:class:|\.column\d?\[|\.pull-(?:left|right)\[|\?\?\?$|\]$)/m);
    }
  });

  it('handles nested wrappers, inline source links, and speaker notes between fragments', () => {
    const fig = '![[Assets/Sources/Scikit-learn MOOC/figures/bagging.svg|Points]]';
    const raw = ['class: split-50', '.pull-left[.center[' + fig + ']]', '.column2[Model',
      '[Read more](https://example.com/a)', ']', '???', 'First notes', '--', 'Reveal', '???',
      'Second notes', '---', '???', '---', ''].join('\n');
    const htmlSource = readingSlides(raw);
    expect(htmlSource).toContain(fig + '\nModel\n[Read more](https://example.com/a)');
    expect(htmlSource).toMatch(/<summary>Speaker notes<\/summary>\s+First notes\s+<\/details>\s+Reveal/);
    expect(htmlSource).toMatch(/<summary>Speaker notes<\/summary>\s+Second notes/);
    expect(htmlSource.match(/<summary>Speaker notes<\/summary>/g)).toHaveLength(2);
  });

  it('restricts publication rewrites to the archive and leaves iframe examples in code alone', () => {
    const tag = '<iframe class="video" src="https://www.youtube.com/embed/SnvdnIlOsHQ?rel=0"></iframe>';
    const fenced = '\x60\x60\x60html\n' + tag + '\n\x60\x60\x60\n';
    const body = tag + '\n\n' + fenced;
    expect(scikitMoocReaderBody('textbook/example', source + 'jupyter-book/x.md.md', body)).toBe(body);
    expect(scikitMoocReaderBody(route + 'jupyter-book/x', 'not-an-archive.md', body)).toBe(body);
    const converted = scikitMoocReaderBody(route + 'jupyter-book/x', source + 'jupyter-book/x.md.md', body);
    expect(converted).toBe('[Watch the course video on YouTube](https://www.youtube.com/watch?v=SnvdnIlOsHQ)\n\n' + fenced);
  });

  it('pins both unmodified teaching figures to the separately credited upstream blobs', () => {
    const policy = JSON.parse(read('publishing/adapter/scikit-mooc-palmerpenguins.json'));
    const file = 'python_scripts/trees_dataset.py.md';
    const body = scikitMoocReaderBody(route + 'python-scripts/trees-dataset', source + file, read(source + file));
    for (const [name, row] of Object.entries(policy.figures) as [string, {sha256: string; gitBlob: string; size: number[]}][]) {
      const bytes = readFileSync(new URL('Assets/Sources/Scikit-learn MOOC/palmerpenguins/' + name, root));
      expect(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex')).toBe(row.gitBlob);
      expect(createHash('sha256').update(bytes).digest('hex')).toBe(row.sha256);
      expect([bytes.readUInt32BE(16), bytes.readUInt32BE(20)]).toEqual(row.size);
      expect(body).toContain('![[' + 'Assets/Sources/Scikit-learn MOOC/palmerpenguins/' + name + '|');
      expect(body).toContain('/' + policy.revision + '/man/figures/' + name + ')');
    }
    expect(body.match(/Teaching-use permission/g)).toHaveLength(2);
    expect(code(body)).toEqual(code(read(source + file)));
  });
});
