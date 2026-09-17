import { createHash } from 'node:crypto';

// SHA256 of original files from the frozen Harvard labs-slides manifest.
// Each retained payload + one terminal LF was independently checked against it.
const HARVARD_SOURCE_SHA256: Readonly<Record<string, string>> = {
  "harvard-ml-systems/labs/vol1/lab-10-model-compress": "91b9f786295921dba651f13733668965734c838aa4fbc10c37208b36d939c097",
  "harvard-ml-systems/slides/vol1/00-course-overview": "583f8798a589656effda66e48a2400a1b7e70eb340e7ed2d8adf578e3c56c474",
  "harvard-ml-systems/slides/vol1/01-introduction": "016bb92092f8175f97400ff87b657dbc98701c1245e866a99684fc2874effae0",
  "harvard-ml-systems/slides/vol1/02-ml-systems": "fb45faf49385bcea94528fb058302bcd510138fd70d4043ebe5e485a245a18de",
  "harvard-ml-systems/slides/vol1/03-ml-workflow": "a5680e46c681dca965a8a5c72863841589991e92b02c8f17346e16b0414096b4",
  "harvard-ml-systems/slides/vol1/04-data-engineering": "d05632f87480fd7f1296c85359a5caa6a44ae598e4ec7c1d5f9b42f12679a162",
  "harvard-ml-systems/slides/vol1/05-nn-computation": "fd319fbb81fc61de108c46064ce1310a7a0b0bdf51f4b0da70198755ae61e807",
  "harvard-ml-systems/slides/vol1/06-nn-architectures": "84dbcefc2446075b7cd0090247d4e4c6cefa1192f39a7c5511893f73cfab7d5a",
  "harvard-ml-systems/slides/vol1/07-frameworks": "b5d314d3947bcd73d3a17d67374b6c6986e36c80833c8c341c5a878ef4a7ecbe",
  "harvard-ml-systems/slides/vol1/08-training": "dd4882835e47b4e73efa5204d2de0d64340a4dc6ab322b1fa57083ae5dc8efe3",
  "harvard-ml-systems/slides/vol1/09-data-selection": "c0ce3b87c3cee9cc6ec8b8eeac3076952467c93aef32581562c89c99974c163b",
  "harvard-ml-systems/slides/vol1/10-model-compression": "0a775e4d2a9f7aa3a3b64f2895a016d5cec6e07c7b8aa856fbc4581e1d6cf588",
  "harvard-ml-systems/slides/vol1/11-hw-acceleration": "880eab49bc79eed75a07658a95adb92497c016de2fbb669dcb0757b117aa3a8c",
  "harvard-ml-systems/slides/vol1/12-benchmarking": "65266a65b87b9f44b3c56408dd926d3abf98e9647f48dbfdad77a0d3cf1b4418",
  "harvard-ml-systems/slides/vol1/13-model-serving": "17d0eb900d696263d0f85e58c971c20fd52abcf09265dbaaab5c4e4ee18801ce",
  "harvard-ml-systems/slides/vol1/14-ml-ops": "c47cdf8edb1d91ab7d7ed9c298b9e7f6bcdc7e30a9045a45b8d9134af155e32d",
  "harvard-ml-systems/slides/vol1/15-responsible-engr": "f9a86c9a21f1147389edfaed37ec9071fa9d117e031e3e5d0bd81a0dd9f2a8fc",
  "harvard-ml-systems/slides/vol1/16-conclusion": "ee04e24731b33d41a1a7f5163401dbd884ac003b7fcebffe150233328588d130",
  "harvard-ml-systems/slides/vol2/00-course-overview": "70efde76c5b65e9291f7bb29fb60cc2aa51e1ac5f3814160146396e5a567b63f",
  "harvard-ml-systems/slides/vol2/01-introduction": "60a1d8371324e9af84d6be7433a01d6490ec7351a3d838746d70367bfebee2ea",
  "harvard-ml-systems/slides/vol2/02-compute-infrastructure": "9f7632eea594e34731a2e575a772afaea2d93b09c599e9abddaa6ff076c17883",
  "harvard-ml-systems/slides/vol2/03-network-fabrics": "3405a3ee7e1aaeaa24cff2bfd21f5de3ec7efb678790f5fc26cd018208461f8d",
  "harvard-ml-systems/slides/vol2/04-data-storage": "f660d0820fa2f233bc0d4322056ece7607c968f7ea821a472365aefd65a85566",
  "harvard-ml-systems/slides/vol2/05-distributed-training": "2b875221b249fc55aae6bd9ca4af80fd9dc5fc11f249a112e9dd801518358070",
  "harvard-ml-systems/slides/vol2/06-collective-communication": "9d99967212e900bc48eac665bc37de97eb6f7b68a953230fb94cb70859dbe30f",
  "harvard-ml-systems/slides/vol2/07-fault-tolerance": "585b17779fbeff5511c46f3d956dccbefc22902ecb64caa0d3f3c119f3b1272e",
  "harvard-ml-systems/slides/vol2/08-fleet-orchestration": "58f823fa2ae4e1a0225216d6214fee580adc1346e5a2ec7a2528bb0a3d87337f",
  "harvard-ml-systems/slides/vol2/09-performance-engineering": "f264acdbf7c487a55e9389ec2cb64972025780d279af0139d69906637f5b168e",
  "harvard-ml-systems/slides/vol2/10-inference": "c4bd3440691c4e5e73ad5629364a1b6e11e05c096036f2d06d92e86637f88653",
  "harvard-ml-systems/slides/vol2/11-edge-intelligence": "c94a214ce667d351652530ac846ffe74354540f027a1ee213b399d33cc1c4ed8",
  "harvard-ml-systems/slides/vol2/12-ops-scale": "ef63aa3bd042f2cc84218bd5c12032417fadfb980ad737a21cae4e396c1e04a1",
  "harvard-ml-systems/slides/vol2/13-security-privacy": "5251f3239fbbe7a0399e8de0543105b1f561cb0939bffbbce68f4a534edb1684",
  "harvard-ml-systems/slides/vol2/14-robust-ai": "f799908a24df63cac820ba406d700b03c9394d13242befa16efd81c8332fbdb1",
  "harvard-ml-systems/slides/vol2/15-sustainable-ai": "9d75de9f77e360f6d8f37e9943c2578dc712ad4ba3c757e6fe0a63c13a136ad0",
  "harvard-ml-systems/slides/vol2/16-responsible-ai": "e32fee18c7740c8e0b55de95ee3d428ef53914d1b6e5277f9b297c3e5266af86",
  "harvard-ml-systems/slides/vol2/17-conclusion": "d62780cfa0fcd11974cac062a8362b95ec6f7c139d3827c731a73b3fe3cd4cd4"
};

function isFrozenOriginal(route: string, source: string): boolean {
  return createHash('sha256').update(source + '\n').digest('hex') === HARVARD_SOURCE_SHA256[route];
}


/**
 * Render-only archive compatibility, never canonical prose. Harvard math is
 * recovered from retained original source under an exact old-projection guard;
 * five notebook pages receive lexical repairs. See source-math-review.md.
 * Unknown or mismatched input is unchanged; errors are never hidden or suppressed.
 */
export function normalizeArchiveMath(markdown: string, fileURL?: URL): string {
  const route = fileURL?.pathname.match(/\/generated\/(?:en\/)?sources\/courses\/(.+)\.md$/)?.[1];
  if (!route) return markdown;
  if (/^harvard-ml-systems\/slides\/vol[12]\/[\w-]+$/.test(route)) return restoreHarvardSlides(markdown, route);
  if (route === 'harvard-ml-systems/labs/vol1/lab-10-model-compress') return restoreHarvardCompressionLab(markdown, route);
  let normalize: ((text: string) => string) | undefined;
  if (route === 'lf-recommenders/diversity-metrics') {
    normalize = text => text.replace(/\\textrm\{(reco|train)_df\}/g,
      (_match, name: string) => String.raw`\textrm{${name}\_df}`);
  } else if (route === 'hse-ml-course/ml1-2026-spring/seminars/sem09-trees-ipynb') {
    // MathJax-style dollars inside \text collide with Markdown dollar tokenization.
    // KaTeX supports \(...\) for nested math in text; preserve all mathematical tokens.
    normalize = text => text.replaceAll(
      String.raw`\text{$|R_m|$ не зависит от $j$ и $s$}`,
      String.raw`\text{\(|R_m|\) не зависит от \(j\) и \(s\)}`
    );
  } else if (route === 'hse-ml-course/ml1-2026-spring/homework-practice/homework-practice-05-trees/homework-practice-05-trees-ipynb') {
    // CommonMark display fences must close on their own line.
    normalize = text => text.replaceAll(
      String.raw`$$ \mathbb{E}_{x} f(x) \approx \frac 1 N \sum_{i=1}^N f(x_i), \quad
\mathbb{E}_{x, y} f(x, y) \approx \frac 1 N \sum_{i=1}^N f(x_i, y_i),$$`,
      () => String.raw`$$
\mathbb{E}_{x} f(x) \approx \frac 1 N \sum_{i=1}^N f(x_i), \quad
\mathbb{E}_{x, y} f(x, y) \approx \frac 1 N \sum_{i=1}^N f(x_i, y_i),
$$`
    );
  } else if (route === 'hse-ml-course/ml2-2026-spring/homeworks-practice/homework-practice-11-random-features/homework-practice-11-random-features-ipynb') {
    normalize = text => text.replaceAll(
      String.raw`$$\tilde \varphi(x) = (
\cos (w_1^T x + b_1),
\dots,
\cos (w_n^T x + b_n)
),$$`,
      () => String.raw`$$
\tilde \varphi(x) = (
\cos (w_1^T x + b_1),
\dots,
\cos (w_n^T x + b_n)
),
$$`
    );
  } else if (route === 'scikit-learn-mooc/python-scripts/dev-features-importance') {
    // \textdollar avoids an escaped dollar adjacent to the closing math delimiter.
    normalize = text => text.replaceAll(String.raw`$100k\$$`, () => String.raw`$100k\textdollar$`);
  }
  if (!normalize) return markdown;

  // Do not rewrite literal code. Only transform gaps between code spans/fences;
  // a longer outer fence can contain shorter fences, and an unclosed fence
  // protects everything through EOF. No placeholder strings enter the source.
  const prose = (text: string): string => {
    const spans = /(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)/g;
    let end = 0;
    let result = '';
    for (const span of text.matchAll(spans)) {
      result += normalize!(text.slice(end, span.index)) + span[0];
      end = span.index + span[0].length;
    }
    return result + normalize!(text.slice(end));
  };
  const openings = /^ {0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)/gm;
  let end = 0;
  let result = '';
  for (let open = openings.exec(markdown); open; open = openings.exec(markdown)) {
    const closings = new RegExp(`^ {0,3}${open[1][0]}{${open[1].length},}[ \\t]*$`, 'gm');
    closings.lastIndex = openings.lastIndex;
    const close = closings.exec(markdown);
    const fenceEnd = close ? close.index + close[0].length : markdown.length;
    result += prose(markdown.slice(end, open.index)) + markdown.slice(open.index, fenceEnd);
    end = fenceEnd;
    openings.lastIndex = fenceEnd;
  }
  return result + prose(markdown.slice(end));
}

/** Decode only the plain, doubled-backslash literal form observed in lab 10. */
function restoreHarvardCompressionLab(markdown: string, route: string): string {
  const startMarker = '## Readable lab narrative\n\n';
  const endMarker = '\n\n## Complete original Marimo source\n\n';
  const start = markdown.indexOf(startMarker);
  const end = markdown.indexOf(endMarker);
  if (start < 0 || end <= start) return markdown;
  const original = markdown.slice(end + endMarker.length).match(/^```python\n([\s\S]*?)\n```(?:\n|$)/)?.[1];
  if (!original || !isFrozenOriginal(route, original)) return markdown;
  const cells = [...original.matchAll(/mo\.md\(\s*(r|f|rf|fr)?('''|""")([\s\S]*?)\2\s*\)/g)];
  // Do not evaluate f-strings, raw strings or arbitrary Python escape syntax.
  if (!cells.length || cells.some(cell => cell[1] || /(?<!\\)\\(?:[^\\]|$)/.test(cell[3]))) return markdown;
  const oldProjection = cells.map(cell => cell[3].trim()
    .replace(/\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?\}/g, '`…`'))
    .filter(Boolean).join('\n\n---\n\n');
  if (markdown.slice(start + startMarker.length, end) !== oldProjection) return markdown;
  const restored = cells.map(cell => cell[3].trim()
    .replaceAll('\\\\', '\\')
    // Literal identifier underscores in TeX text, not mathematical subscripts.
    .replace(/\\text\{([^{}]*)\}/g, (_match, text: string) => String.raw`\text{${text.replace(/(?<!\\)_/g, String.raw`\_`)}}`))
    .filter(Boolean).join('\n\n---\n\n');
  return markdown.slice(0, start + startMarker.length) + restored + markdown.slice(end);
}


/** Byte-for-byte equivalent of the frozen importer clean_tex, for a match guard. */
function cleanImportedTex(text: string): string {
  for (const [from, to] of [
    [String.raw`\&`, '&'], [String.raw`\%`, '%'],
    [String.raw`\_`, '_'], [String.raw`\times`, '×'],
    [String.raw`\rightarrow`, '→'], ['~', ' ']
  ]) text = text.replaceAll(from, to);
  for (const command of ['textbf', 'alert'])
    text = text.replace(new RegExp(String.raw`\\${command}\{([^{}]*)\}`, 'g'), '**$1**');
  for (const command of ['textit', 'emph'])
    text = text.replace(new RegExp(String.raw`\\${command}\{([^{}]*)\}`, 'g'), '*$1*');
  return text
    .replace(/\\(?:small|footnotesize|scriptsize|tiny|normalsize)\b/g, '')
    .replace(/\\(?:pause|centering|vfill|medskip|smallskip|bigskip)\b/g, '')
    .replace(/\\href\{([^{}]*)\}\{([^{}]*)\}/g, '[$2]($1)')
    .replace(/\\url\{([^{}]*)\}/g, '<$1>')
    .replace(/\\includegraphics(?:\[[^\]]*\])?\{([^{}]*)\}/g, '*Figure: `$1`*')
    .replace(/\\begin\{(?:itemize|enumerate|columns?|block|exampleblock|alertblock)\}(?:\{[^{}]*\})?/g, '')
    .replace(/\\end\{(?:itemize|enumerate|columns?|block|exampleblock|alertblock)\}/g, '')
    .replace(/^\s*\\item(?:<[^>]*>)?\s*/gm, '- ')
    .replace(/\\[A-Za-z@]+\*?(?:\[[^\]]*\])?/g, '')
    .replace(/\n{3,}/g, '\n\n').trim();
}

function cleanTexPreservingMath(source: string): string {
  const expressions: string[] = [];
  let prefix = 'BOOKVARMATH';
  while (source.includes(prefix)) prefix += 'X';
  const protectedText = source.replace(/(?<!\\)(\${1,2})(?!\$)([\s\S]*?)(?<!\\)\1(?!\$)/g,
    (_match, delimiter: string, expression: string) => {
      const index = expressions.length;
      // remark-math does not interpret backslash-escaped dollars inside math;
      // the equivalent TeX command avoids prematurely ending the Markdown span.
      expression = expression.replaceAll(String.raw`\$`, String.raw`\textdollar{}`);
      // Display fences on separate lines avoid swallowing later prose in remark-math.
      expressions.push(delimiter === '$$' ? `\n\n$$\n${expression.trim()}\n$$\n\n` : `$${expression}$`);
      return `${prefix}${index}END`;
    });
  return cleanImportedTex(protectedText).replace(new RegExp(`${prefix}(\\d+)END`, 'g'),
    (_match, index: string) => expressions[Number(index)]);
}

/**
 * Recover the math directly from the complete retained TeX, only when the old
 * projection exactly reproduces the current readable block. Never infer an
 * operator from a damaged expression. A changed source/projection fails closed.
 */
function restoreHarvardSlides(markdown: string, route: string): string {
  const startMarker = '## Readable slide sequence\n\n';
  const endMarker = '\n\n## Complete original Beamer source\n\n';
  const start = markdown.indexOf(startMarker);
  const end = markdown.indexOf(endMarker);
  if (start < 0 || end <= start) return markdown;
  const original = markdown.slice(end + endMarker.length).match(/^```tex\n([\s\S]*?)\n```(?:\n|$)/)?.[1];
  if (!original || !isFrozenOriginal(route, original)) return markdown;
  const frames = [...original.matchAll(/\\begin\{frame\}(?:\[[^\]]*\])?(?:\{([^{}]*)\})?([\s\S]*?)\\end\{frame\}/g)];
  const project = (preserveMath: boolean): string => frames.map((frame, index) => {
    const title = cleanImportedTex(frame[1] || `Frame ${index + 1}`);
    const body = preserveMath ? cleanTexPreservingMath(frame[2]) : cleanImportedTex(frame[2]);
    return body ? `## ${title}\n\n${body}` : '';
  }).filter(Boolean).join('\n\n');
  const oldProjection = project(false);
  if (!oldProjection || markdown.slice(start + startMarker.length, end) !== oldProjection) return markdown;
  return markdown.slice(0, start + startMarker.length) + project(true) + markdown.slice(end);
}
