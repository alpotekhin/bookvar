import { mapMarkdownProse } from './markdown-prose.js';

const courses = '05 Источники/Courses/';
const recommenderRoute = 'sources/courses/lf-recommenders/';
const recommenderRoot = courses + 'Linux Foundation Recommenders/';
const recommenderAssets = 'https://raw.githubusercontent.com/recommenders-team/resources/e76bb39727103e45dd8011b6d0eca2f1aad3c0f7/images/';
const recommenderLicense = '/licenses/recommenders-resources-e76bb397-MIT.txt';

const mlvRoute = 'sources/courses/machine-learning-visualized/';
const mlvRoot = courses + 'Machine Learning Visualized/';
const mlvRevision = '950f8ec4671c0790cdd867514cc114fea01f5ab0';
const mlvRepo = 'https://github.com/gavinkhung/machine-learning-visualized/';
const mlvLicense = '/licenses/machine-learning-visualized-950f8ec4-MIT.txt';

const csRoute = 'sources/courses/stanford-cs230-cheatsheets/';
const csRoot = courses + 'Stanford CS230 Cheatsheets/';
const csRevision = '4653bc01297b269edb19e844b01127ba13de59df';
const csRepo = 'https://github.com/afshinea/stanford-cs-230-deep-learning/';
const csLicense = '/licenses/stanford-cs230-cheatsheets-4653bc01-MIT.txt';
const csSheets = [
  ['cnn', 'cheatsheet-convolutional-neural-networks.pdf', 'Convolutional Neural Networks'],
  ['rnn', 'cheatsheet-recurrent-neural-networks.pdf', 'Recurrent Neural Networks'],
  ['tips-and-tricks', 'cheatsheet-deep-learning-tips-tricks.pdf', 'Deep Learning Tips and Tricks'],
  ['complete', 'super-cheatsheet-deep-learning.pdf', 'Complete Deep Learning Cheatsheet'],
] as const;

function originalPdf(body: string, upstream: string, local: string, license: string, previewPage = 1): string {
  const oldIntro = 'The embedded PDF is the primary visual document. The page-separated text below was extracted mechanically for search and has not been rewritten.';
  return body.replace(oldIntro, 'The PDF retains the original figures, equations, and page layout. The text below is a mechanical extraction for search.')
    .replace(/^<iframe class="source-pdf" src="([^"]+)" title="([^"]+)" loading="lazy"><\/iframe>$/m,
      (tag, src: string, title: string) => {
        if (src !== upstream + '?raw=1') return tag;
        const preview = local.replace(/\.pdf$/, `-page-${previewPage}.png`);
        return `[Open the original PDF at full size](${local}) · [Source revision](${upstream}) · [MIT license](${license})\n\n` +
          `[![${title}, page ${previewPage}](${preview})](${local}#page=${previewPage})`;
      });
}

function recommenderFigure(prose: string, markup: string, file: string, asset: string, alt: string, credit = ''): string {
  return prose.replace(markup, `![[${asset}|${alt}]]\n\n` +
    `*Recommenders contributors. [Original figure](${recommenderAssets}${file}) · [MIT license](${recommenderLicense}).${credit}*`);
}

export function courseLibraryReaderMedia(route: string, source: string, body: string): string {
  if (source.startsWith(recommenderRoot)) {
    if (route === recommenderRoute + 'als-operationalization') return mapMarkdownProse(body, prose =>
      recommenderFigure(prose, '![architecture](https://raw.githubusercontent.com/recommenders-team/resources/main/images/reco-arch.png)',
        'reco-arch.png', '00 Учебник/Assets/Figures/curated/linux-foundation-recommenders-architecture.png',
        'Offline training produces per-user recommendations stored in Cosmos DB and served by an API.'));
    if (route === recommenderRoute + 'sar-deep-dive') return mapMarkdownProse(body, prose =>
      recommenderFigure(prose, '<img src="https://raw.githubusercontent.com/recommenders-team/resources/main/images/sar_schema.svg?sanitize=true">',
        'sar_schema.svg', '00 Учебник/Assets/Figures/curated/recommender-mf-2026-10-25/sar-schema.svg',
        'SAR multiplies the user-item affinity matrix by the item-item similarity matrix to score items.'));
    if (route === recommenderRoute + 'ncf-deep-dive') return mapMarkdownProse(body, prose =>
      recommenderFigure(prose, '<img src="https://raw.githubusercontent.com/recommenders-team/resources/main/images/NCF.svg?sanitize=true">',
        'NCF.svg', 'Assets/Sources/Linux Foundation Recommenders/NCF.svg',
        'NeuMF concatenates the GMF element-wise product and the MLP output before predicting an interaction score.',
        ' Architecture from Xiangnan He et al., [Neural Collaborative Filtering, Figure 3](https://arxiv.org/html/1708.05031v2#S3.F3) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).'));
  }
  if (source.startsWith(mlvRoot) && route.startsWith(mlvRoute)) {
    if (source === mlvRoot + 'README.md' && route === mlvRoute + 'overview') {
      const descriptions: Record<string, string> = {
        home: 'Preview of the Machine Learning Visualized website.',
        marimo: 'The Marimo notebook plots changes as the model parameters move.',
        latex: 'The book places equations and their derivations beside the code.',
      };
      return mapMarkdownProse(body, prose => prose.replace(
        /!\[[^\]\n]+\]\(https:\/\/github\.com\/gavinkhung\/machine-learning-visualized\/raw\/950f8ec4671c0790cdd867514cc114fea01f5ab0\/gifs\/(home|marimo|latex)\.gif\)/g,
        (_match, name: string) => `![[Assets/Sources/Machine Learning Visualized/${name}.gif|${descriptions[name]}]]\n\n` +
          `*Gavin Hung, Machine Learning Visualized. [Original animation](${mlvRepo}blob/${mlvRevision}/gifs/${name}.gif) · [MIT license](${mlvLicense}).*`,
      ));
    }
    if (source === mlvRoot + 'book/main.pdf.md' && route === mlvRoute + 'complete-book') {
      return originalPdf(body, `${mlvRepo}blob/${mlvRevision}/book/main.pdf`,
        '/source-documents/machine-learning-visualized/main.pdf', mlvLicense, 5);
    }
    const demos = [
      ['chapter1/interactive_linear_regression.md', 'interactive-linear-regression', '9hsfob'],
      ['chapter3/interactive_logistic_regression.md', 'interactive-logistic-regression', 'cs19cb'],
      ['chapter3/interactive_perceptron.md', 'interactive-perceptron', '50ccg0'],
    ] as const;
    const demo = demos.find(([file, slug]) => source === mlvRoot + file && route === mlvRoute + slug);
    if (demo) {
      const url = 'https://marimo.app/l/' + demo[2];
      return mapMarkdownProse(body, prose => prose.replace(/<iframe\b[^>]*><\/iframe>/g, tag =>
        tag.includes(`src="${url}?show-code=false&embed=true"`)
          ? `[Open Gavin Hung’s interactive notebook in Marimo](${url})` : tag));
    }
  }
  if (source.startsWith(csRoot) && route.startsWith(csRoute)) {
    if (source === csRoot + 'README.md' && route === csRoute + 'overview') {
      return body.replace(/^\|<a href="https:\/\/github\.com\/afshinea\/stanford-cs-230-deep-learning\/blob\/master\/en\/[^\n]+\n^\|:--:[^\n]+\n^\|[^\n]+/gm, table => {
        if (!table.includes('illustrations/cover/en-00')) return table;
        return csSheets.filter(([, file]) => table.includes('/en/' + file + '"'))
          .map(([slug, , title]) => `- [${title}](/${csRoute}${slug}/)`).join('\n');
      });
    }
    const sheet = csSheets.find(([slug, file]) => route === csRoute + slug && source === csRoot + 'en/' + file + '.md');
    if (sheet) {
      return originalPdf(body, `${csRepo}blob/${csRevision}/en/${sheet[1]}`,
        '/source-documents/stanford-cs230-cheatsheets/' + sheet[1], csLicense);
    }
  }
  return body;
}
