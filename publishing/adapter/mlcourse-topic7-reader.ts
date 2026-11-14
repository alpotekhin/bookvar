import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const evidence = JSON.parse(readFileSync(new URL('./mlcourse-topic7-evidence.json', import.meta.url), 'utf8')) as {
  revision: string;
  source: { path: string };
  plots: Record<string, { asset: string; alt: string; section: string }>;
  codeCells: Array<{ block: number; archivedSha256: string; source: string }>;
};
const original = 'https://mlcourse.ai/book/topic07/topic7_pca_clustering.html';
const upstream = 'https://github.com/Yorko/mlcourse.ai/blob/' + evidence.revision + '/';
const assets = 'Assets/Sources/mlcourse.ai/topic-7/';
const route = 'sources/courses/advanced-unsupervised-classical/mlcourse-topic7';
const source = '05 Источники/Courses/Advanced Unsupervised and Classical ML/mlcourse-topic7.html.md';
const rawPlot = 'https://mlcourse.ai/_images/';
const license = 'https://creativecommons.org/licenses/by-nc-sa/4.0/';
const restored = new Map(evidence.codeCells.map(cell => [cell.block, cell]));
const metricNames = ['ARI', 'AMI', 'Homogeneity', 'Completeness', 'V-measure', 'Silhouette'];
const metricRows = ['K-means', 'Affinity', 'Spectral', 'Agglomerative'];
const metricOutput = new RegExp('^' + metricNames.join('\\n') + '\\n\\n' +
  metricRows.map(name => name + '\\n(?:\\d\\.\\d{6}\\n){6}').join('\\n'), 'm');

function metricsTable(raw: string): string {
  const row = (cells: string[]) => '| ' + cells.join(' | ') + ' |';
  const results = raw.trimEnd().split('\n\n').slice(1).map(group => row(group.split('\n')));
  return [row(['Algorithm', ...metricNames]), row(Array(metricNames.length + 1).fill('---')), ...results].join('\n') + '\n\n' +
    '> Results from the original course run. Here agglomerative clustering scores highest on ARI against the known digit labels, while spectral clustering scores highest on silhouette, which uses distances rather than those labels.\n';
}

function figure(asset: string, alt: string, credit: string) {
  return '\n\n![[' + asset + '|' + alt + ']]\n\n*' + credit + '*\n\n';
}

function plot(hash: string): string {
  const row = evidence.plots[hash];
  if (!row) throw new Error('Unreviewed mlcourse.ai output: ' + hash);
  return figure(assets + row.asset, row.alt,
    'Sergey Korolev and the mlcourse.ai contributors, [Topic 7](' + original + '#' + row.section +
    '), [original PNG](' + rawPlot + hash + '.png), [CC BY-NC-SA 4.0](' + license + '). Unchanged.');
}

const notes = {
  iris: '> The scatter plot describes the full Iris dataset. The later accuracy comparison also fits PCA on the full dataset before splitting, which lets the test rows influence the projection. For a held-out test, split first and fit a pipeline with PCA and the tree on the training rows only. With scikit-learn 1.7.2 and the split below, the train-only pipeline also scores 41/45 (91.1%), versus 40/45 (88.9%) without PCA. One extra correct prediction on this split does not establish an improvement.',
  digits: '> This example uses scikit-learn’s 8×8 Digits dataset, not the 28×28 MNIST dataset named in the plot title. Colors show the known digit labels, which PCA and t-SNE do not receive as input.',
  variance: '> The original plot starts at array index 0, where one component is already included. The first 20 components explain 89.43% of the variance; the first 21 explain 90.32%. The first index above 0.90 is 20, so the count is 21. The blue line at index 21 in the original marks the 22nd entry.',
  elbow: '> The plotted values are the square roots of inertia. In scikit-learn, inertia sums squared Euclidean distances to the assigned centroids. It is not the sum of unsquared distances. At k = 2, 3 and 4 the values are approximately 12.523, 6.005 and 5.275. The bend at 3 is a heuristic for these points, not a general proof of the best number of clusters.',
};

function restoreCode(markdown: string): string {
  let index = 0;
  let matched = 0;
  const converted = markdown.replace(/^\x60{3}\n([\s\S]*?)^\x60{3}$/gm, (block, body: string) => {
    const cell = restored.get(index++);
    if (!cell) return block;
    if (createHash('sha256').update(body).digest('hex') !== cell.archivedSha256) {
      throw new Error('The mlcourse.ai archive no longer matches the reviewed code block: ' + cell.block);
    }
    matched++;
    return '\x60\x60\x60python\n' + cell.source.replace(/^\n+/, '').trimEnd() + '\n\x60\x60\x60';
  });
  if (matched !== restored.size) throw new Error('Missing reviewed mlcourse.ai code blocks');
  return converted;
}

export function mlcourseTopic7ReaderBody(page: string, owner: string, body: string): string {
  if (page !== route || owner !== source) return body;
  const blocks = restoreCode(body).split(/(^\x60{3}(?:python)?\n[\s\S]*?^\x60{3}$)/m);
  return blocks.map((prose, index) => index % 2 ? prose : prose
    .replace('Bookvar removes browser chrome and executable scripts from saved HTML while\n> preserving the article body. Code and RST are byte-for-byte inside the fence.',
      'Bookvar removes browser chrome and restores code indentation from the\n> [pinned lesson source](' + upstream + evidence.source.path + ').')
    // The old HTML importer split the anchor around each output into four lines.
    .replace(/^\[\n!\[\.\.\/\.\.\/_images\/[a-f0-9]{64}\.png\]\(https:\/\/mlcourse\.ai\/_images\/([a-f0-9]{64})\.png\)\n\n\]\(https:\/\/mlcourse\.ai\/_images\/\1\.png\)$/gm,
      (_block, hash: string) => {
        const result = plot(hash);
        const name = evidence.plots[hash].asset;
        const note = name === 'iris-two-components.png' ? notes.iris
          : name === 'digits-pca.png' ? notes.digits
          : name === 'digits-explained-variance.png' ? notes.variance
          : name === 'kmeans-elbow.png' ? notes.elbow : '';
        return result + (note ? note + '\n\n' : '');
      })
    .replace('![Source figure](https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg)',
      '[Original course banner](' + original + '#id1)')
    .replace('![Source figure](https://habrastorage.org/getpro/habr/post_images/bb6/fe7/f06/bb6fe7f06e114bcc9c354a1cb025b966.png)',
      figure(assets + 'GaussianScatterPCA.svg', 'Two perpendicular arrows start at the center of the point cloud. The longer one follows its main direction of spread.',
        'Nicoguaro, [GaussianScatterPCA.svg](https://commons.wikimedia.org/wiki/File:GaussianScatterPCA.svg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). ' +
        'Unchanged figure illustrating the same PCA geometry; the image used by the course remains [on its original page](' + original + '#id5).'))
    .replace(/!\[Source figure\]\(https:\/\/habrastorage\.org\/getpro\/habr\/post_images\/8b9\/ae5\/586\/8b9ae55861f22a2809e8b3a00ef815ad\.png\)\nThe examples of the outcomes from different algorithms from scikit-learn/,
      figure('00 Учебник/Assets/Figures/curated/advanced-unsupervised-classical/spectral-clustering-comparison.png',
        'Each column runs a different clustering algorithm on the same shapes. Compare how the top row’s concentric circles are split.',
        'The scikit-learn developers, [official clustering comparison](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), ' +
        '[BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/9a210f70301a76c7ff3ea68cdec3748ec58834be/COPYING). ' +
        'This gallery version is not the historical screenshot [in the original lesson](' + original + '#id9).'))
    .replace('You can run the algorithm several times and then average all the centroid results.',
      'You can run the algorithm several times and then average all the centroid results.\n\n' +
      '> For standard K-means, the mean update minimizes squared Euclidean distances. With absolute distance, the minimizing representative in one dimension is a median, not a mean. ' +
      'For multiple random starts, keep the run with the lowest inertia; averaging centroids across runs can mix up their labels. See [K-means in Bookvar](/textbook/classical-ml/clustering/#k-means).')
    .replace('Values close to zero mean the splits are independent, and those close to 1 mean they are similar (with complete match at \\(\\text{AMI} = 1\\)).',
      'Values close to zero mean the splits are independent, and those close to 1 mean they are similar (with complete match at \\(\\text{AMI} = 1\\)).\n\n' +
      '> Both ARI and AMI can be negative. For labels [0, 0, 1, 1] versus [0, 1, 0, 1], each score is −0.5: the agreement is below the chance baseline. ' +
      'A value around zero is the baseline, not proof of independence.')
    .replace(metricOutput, metricsTable)
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, tex: string) => '\n\n$$\n' + tex.trim() + '\n$$\n\n')
    .replace(/\\\((.+?)\\\)/g, (_match, tex: string) => '$' + tex + '$')).join('');
}
