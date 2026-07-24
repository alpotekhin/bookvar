import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const requireFromSite = createRequire(new URL('../../site/package.json', import.meta.url));
const katex = requireFromSite('katex');

const expressions = JSON.parse(readFileSync(0, 'utf8'));
const rendered = expressions.map(({ value, displayMode }) =>
  katex.renderToString(value, {
    displayMode,
    throwOnError: false,
    strict: 'ignore',
    output: 'html'
  }).replaceAll('$', '&#36;')
);
process.stdout.write(JSON.stringify(rendered));
